from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(os.getenv("ATHAR_ROOT", "/workspace/data/athar"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-14B")

ADVISOR_PROMPTS_DIR = Path(
    os.getenv("ADVISOR_PROMPTS_DIR", str(ROOT / "prompts" / "advisors"))
)
META_PROMPT_PATH = Path(
    os.getenv("META_PROMPT_PATH", str(ROOT / "prompts" / "meta" / "AOS-META-00.md"))
)
ADVISOR_REGISTRY_PATH = Path(
    os.getenv("ADVISOR_REGISTRY_PATH", str(ROOT / "advisors" / "advisors_registry_35.json"))
)

MAX_MODEL_INPUT_TOKENS = int(os.getenv("MAX_MODEL_INPUT_TOKENS", "30000"))
ADVISOR_MAX_NEW_TOKENS = int(os.getenv("ADVISOR_MAX_NEW_TOKENS", "1400"))
META_MAX_NEW_TOKENS = max(int(os.getenv("META_MAX_NEW_TOKENS", "5200")), 5200)
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.20"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.90"))
COUNCIL_META_MODE = os.getenv("COUNCIL_META_MODE", "adapter").strip().lower()

CONSULTATION_SCHEMA_VERSION = "athar_consultation_v2"
SPRINT_COUNT = 12
OPEN_STATUS = "OPEN"
EVIDENCE_CODES = {"E1", "E2", "E3", "I1", "I2", "A1", "U"}
INTERACTION_TYPES = {
    "CONSENSUS",
    "COMPLEMENTARY",
    "TRADE-OFF",
    "CONFLICT",
    "EVIDENCE GAP",
    "SCOPE CONFLICT",
}
CONFIDENCE_LEVELS = {"High", "Medium", "Low"}



class AtharCouncilEngine:
    """Runs selected advisors independently, then synthesizes with AOS-META-00.

    The engine can either load its own Qwen model (standalone mode), or reuse an
    already-loaded Qwen model/tokenizer supplied by the unified RunPod handler.
    Reusing the model is the production path because it avoids a second 14B copy
    on the same GPU.
    """

    def __init__(
        self,
        *,
        base_model=None,
        tokenizer=None,
        specialist_adapter_path: Optional[str | Path] = None,
        meta_adapter_path: Optional[str | Path] = None,
        model_lock: Optional[threading.RLock] = None,
    ) -> None:
        registry_doc = self._load_json(ADVISOR_REGISTRY_PATH)
        advisors = registry_doc.get("advisors", registry_doc)
        if not isinstance(advisors, list) or len(advisors) != 35:
            raise ValueError("advisors_registry_35.json must contain exactly 35 advisors.")

        self.registry_by_model_id = {x["advisor_id"]: x for x in advisors}
        self.registry_by_number = {int(x["advisor_number"]): x for x in advisors}
        self.registry_by_name = {
            self._norm(x["advisor_name_ar"]): x for x in advisors
        }
        self.meta_prompt = self._load_prompt(META_PROMPT_PATH)
        self.model_lock = model_lock or threading.RLock()

        specialist_adapter_path = self._require_adapter(
            specialist_adapter_path,
            "specialist",
        )
        meta_adapter_path = self._require_adapter(
            meta_adapter_path,
            "meta",
        )

        if base_model is None:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                torch_dtype=compute_dtype,
                device_map={"": 0} if torch.cuda.is_available() else "auto",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            )

        if tokenizer is None:
            raise ValueError("A tokenizer is required when reusing a shared base model.")

        self.tokenizer = tokenizer

        # The unified handler passes a plain AutoModelForCausalLM here on the
        # first council request. We wrap that same object with PEFT so no second
        # copy of Qwen3-14B is loaded into GPU memory.
        if isinstance(base_model, PeftModel):
            self.model = base_model
            existing = set(getattr(self.model, "peft_config", {}).keys())
            if "specialist" not in existing:
                self.model.load_adapter(
                    str(specialist_adapter_path),
                    adapter_name="specialist",
                    is_trainable=False,
                )
            if "meta" not in existing:
                self.model.load_adapter(
                    str(meta_adapter_path),
                    adapter_name="meta",
                    is_trainable=False,
                )
        else:
            self.model = PeftModel.from_pretrained(
                base_model,
                str(specialist_adapter_path),
                adapter_name="specialist",
                is_trainable=False,
            )
            self.model.load_adapter(
                str(meta_adapter_path),
                adapter_name="meta",
                is_trainable=False,
            )

        self.model.eval()

    @staticmethod
    def _norm(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Required prompt file not found: {path}")
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"Council prompts must be Markdown/TXT: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        return text

    @staticmethod
    def _require_adapter(path: Optional[str | Path], label: str) -> Path:
        if not path:
            raise FileNotFoundError(f"{label} adapter path was not provided.")
        p = Path(path)
        adapter_file = p / "adapter_model.safetensors"
        config_file = p / "adapter_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Missing {label} adapter_config.json: {config_file}")
        if not adapter_file.exists():
            raise FileNotFoundError(f"Missing {label} adapter_model.safetensors: {adapter_file}")
        if adapter_file.stat().st_size < 10_000_000:
            raise RuntimeError(
                f"{label} adapter looks like a Git LFS pointer, not real weights: {adapter_file}"
            )
        return p

    @staticmethod
    def clean_model_text(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I)
        return text.strip()

    @classmethod
    def extract_json_object(cls, text: str) -> Dict[str, Any]:
        cleaned = cls.clean_model_text(text)
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                return value
        raise ValueError("Meta Advisor did not return a valid JSON object.")

    def _render_chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate(
        self,
        adapter: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        *,
        deterministic: bool = False,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> str:
        rendered = self._render_chat(system_prompt, user_prompt)
        ids = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if len(ids) > MAX_MODEL_INPUT_TOKENS:
            raise ValueError(
                f"Input too large: {len(ids)} tokens > {MAX_MODEL_INPUT_TOKENS}. "
                "The authoritative prompt was NOT truncated."
            )

        with self.model_lock:
            use_base = adapter == "base"
            adapter_ctx = (
                self.model.disable_adapter()
                if use_base and hasattr(self.model, "disable_adapter")
                else nullcontext()
            )
            with adapter_ctx:
                if not use_base:
                    self.model.set_adapter(adapter)

                inputs = self.tokenizer(rendered, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                    use_cache=True,
                )
                if no_repeat_ngram_size:
                    gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
                if deterministic:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs.update(
                        do_sample=True,
                        temperature=GEN_TEMPERATURE,
                        top_p=GEN_TOP_P,
                    )

                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)

                generated = output_ids[0, inputs["input_ids"].shape[1] :]
                return self.clean_model_text(
                    self.tokenizer.decode(generated, skip_special_tokens=True)
                )

    def _incoming_advisors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        value = (request.get("input") or {}).get("advisors") or []
        return value if isinstance(value, list) else []

    def _resolve_incoming(self, incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the backend advisor to the authoritative Athar advisor registry.

        Screen 3 currently sends an integer ``id`` for each council member.  The
        backend was given the same 1..35 catalogue ordering, so when a canonical
        AOS-* code is not present we accept that integer as the advisor number.
        Canonical codes remain preferred whenever they are supplied.
        """
        model_id = (
            incoming.get("model_advisor_id")
            or incoming.get("advisor_id")
            or incoming.get("system_code")
            or incoming.get("code")
        )

        registry_entry = None
        if isinstance(model_id, str):
            registry_entry = self.registry_by_model_id.get(model_id.strip())

        backend_id = incoming.get("id")
        if registry_entry is None and isinstance(backend_id, int):
            registry_entry = self.registry_by_number.get(backend_id)

        if registry_entry is None:
            name = self._norm(incoming.get("name") or incoming.get("advisor_name_ar"))
            registry_entry = self.registry_by_name.get(name)

        if registry_entry is None:
            raise ValueError(
                "Could not map Backend advisor to an Athar advisor. "
                f"Incoming advisor: {incoming}. Send model_advisor_id such as AOS-SP-08 "
                "or a Screen-3 advisor id matching the 1..35 Athar registry order."
            )

        result = dict(registry_entry)
        result["backend_id"] = backend_id if isinstance(backend_id, int) else None
        # Preserve the current backend payload for audit/debug without letting it
        # alter the authoritative Expert DNA identity.
        result["backend_payload"] = {
            k: incoming.get(k)
            for k in ("id", "name", "title", "capabilities")
            if k in incoming
        }
        return result

    def _resolve_selected_advisors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_obj = request.get("input") or {}
        incoming = self._incoming_advisors(request)
        incoming_by_backend_id = {
            x.get("id"): x
            for x in incoming
            if isinstance(x, dict) and x.get("id") is not None
        }
        incoming_by_model_id: Dict[str, Dict[str, Any]] = {}
        for x in incoming:
            if not isinstance(x, dict):
                continue
            mid = (
                x.get("model_advisor_id")
                or x.get("advisor_id")
                or x.get("system_code")
                or x.get("code")
            )
            if isinstance(mid, str):
                incoming_by_model_id[mid.strip()] = x

        selected = input_obj.get("selected_advisor_ids")
        if selected is None:
            selected = input_obj.get("selected_advisors")

        resolved: List[Dict[str, Any]] = []

        if isinstance(selected, list) and selected:
            for raw in selected:
                if isinstance(raw, dict):
                    resolved.append(self._resolve_incoming(raw))
                    continue

                if isinstance(raw, str):
                    canonical = raw.strip()
                    reg = self.registry_by_model_id.get(canonical)
                    if reg is None:
                        # Backward-compatible numeric string support.
                        if canonical.isdigit():
                            reg = self.registry_by_number.get(int(canonical))
                        if reg is None:
                            raise ValueError(f"Unknown Athar advisor ID: {raw}")
                    item = dict(reg)
                    mapped = incoming_by_model_id.get(item["advisor_id"])
                    item["backend_id"] = mapped.get("id") if mapped and isinstance(mapped.get("id"), int) else None
                    resolved.append(item)
                    continue

                if isinstance(raw, int):
                    # Numeric values remain accepted only for legacy callers with
                    # an explicit input.advisors mapping, or as advisor numbers 1..35.
                    mapped = incoming_by_backend_id.get(raw)
                    if mapped is not None:
                        resolved.append(self._resolve_incoming(mapped))
                        continue
                    reg = self.registry_by_number.get(raw)
                    if reg is None:
                        raise ValueError(f"Unknown Athar advisor number: {raw}")
                    item = dict(reg)
                    item["backend_id"] = raw if isinstance(raw, int) else None
                    resolved.append(item)
                    continue

                raise ValueError(f"Unsupported selected advisor value: {raw!r}")
        else:
            for item in incoming:
                if isinstance(item, dict):
                    resolved.append(self._resolve_incoming(item))

        if not resolved:
            raise ValueError(
                "No selected advisors received. Send selected_advisor_ids using canonical "
                "IDs such as AOS-SP-13."
            )

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in resolved:
            model_id = item["advisor_id"]
            if model_id not in seen:
                seen.add(model_id)
                unique.append(item)
        if len(unique) > 16:
            raise ValueError("A consultation may include at most 16 selected advisors.")
        return unique

    def _shared_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        input_obj = request.get("input") or {}
        return {
            "run_id": request.get("run_id"),
            "consultation_id": request.get("consultation_id"),
            "topic": request.get("topic"),
            "organization": input_obj.get("organization"),
            "programs": input_obj.get("programs"),
            "track": input_obj.get("track"),
            "goal": input_obj.get("goal"),
            "impact_map": input_obj.get("impact_map"),
        }

    @staticmethod
    def _extract_scope_contract(prompt: str, max_chars: int = 2600) -> str:
        """Extract a compact Owned Outcome / boundaries card from full Expert DNA."""
        text = re.sub(r"<PARSED TEXT FOR PAGE:[^>]+>", "", str(prompt), flags=re.I)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        start_markers = (
            "OWNED OUTCOME",
            "النتيجة التي تملكها",
            "SCOPE OF AUTHORITY",
            "نطاق الاختصاص",
        )
        starts = [text.upper().find(m.upper()) for m in start_markers]
        starts = [x for x in starts if x >= 0]
        start = min(starts) if starts else 0

        # Include enough material to cover owned outcome, "تملك/لا تملك", and
        # boundaries with adjacent advisors without injecting the whole DNA twice.
        excerpt = text[start : start + max_chars]
        return excerpt.strip()

    @staticmethod
    def _advisor_output_is_bad(text: str) -> bool:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(clean) < 220:
            return True
        if len(clean) > 7600:
            return True

        forbidden_echo = (
            "OWNED OUTCOME",
            "POSITION WITHIN ATHAR OS",
            "SYSTEM PROMPT",
            "IDENTITY",
            "MISSION",
        )
        upper = clean.upper()
        if any(x in upper for x in forbidden_echo):
            return True

        lines = [
            re.sub(r"\s+", " ", x).strip().lower()
            for x in str(text or "").splitlines()
            if len(re.sub(r"\s+", " ", x).strip()) >= 24
        ]
        if len(lines) >= 10 and (len(set(lines)) / len(lines)) < 0.74:
            return True

        # Excessive sectioning is a common symptom of the old repetitive output.
        headings = len(re.findall(r"(?m)^\s*#{2,6}\s+", str(text or "")))
        if headings > 8:
            return True

        return False

    def _run_advisor(
        self,
        advisor: Dict[str, Any],
        request: Dict[str, Any],
        selected_council: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self._load_prompt(ADVISOR_PROMPTS_DIR / advisor["prompt_file"])
        scope_contract = self._extract_scope_contract(prompt)
        other_selected = [
            {
                "advisor_id": x["advisor_id"],
                "advisor_name_ar": x["advisor_name_ar"],
            }
            for x in selected_council
            if x["advisor_id"] != advisor["advisor_id"]
        ]

        task = {
            "instruction": (
                "قدّم First Pass مستقلًا ومحددًا للحالة، مستخدمًا Expert DNA الأصلي كمرجع ملزم. "
                "ابدأ بتحليل الحالة مباشرة ولا تعِد تعريف المستشار أو Mission أو Owned Outcome أو Tools. "
                "التزم حرفيًا بما يملكه هذا المستشار وما لا يملكه: لا تحل موضوعًا يملكه مستشار آخر، "
                "حتى لو ظهر في الحالة؛ ضعه فقط تحت إحالات خارج النطاق. لا تطلع على آراء الآخرين. "
                "كل فكرة تُذكر مرة واحدة فقط: لا تكرر نفس التوصية تحت التدخلات والأولويات والمخاطر والخلاصة. "
                "لا تخترع حقائق أو نسبًا أو مستهدفات أو تواريخ أو مددًا أو افتراضات غير معلنة. "
                "إذا ذكرت عدد البرامج أو حالاتها، فاقرأ القائمة حرفيًا ولا تعد البرنامج نفسه مرتين. "
                "ميّز بين حقيقة من case_context، واستنتاج مهني، وافتراض يحتاج تحققًا. "
                "استخدم بحد أقصى أربع توصيات مملوكة فعلًا لاختصاصك، وثلاث فجوات بيانات، وإحالتين خارج النطاق. "
                "اجعل الرد مركزًا وغير مكرر."
            ),
            "required_structure": [
                "تشخيص داخل النطاق",
                "توصيات مملوكة للاختصاص",
                "الأدلة وفجوات البيانات",
                "المخاطر والشروط",
                "إحالات خارج النطاق عند الحاجة",
            ],
            "hard_rules": [
                "لا تقدم حلًا تنفيذيًا في مجال مستشار آخر؛ الإحالة فقط.",
                "لا تستخدم رقم مستشار منفردًا مثل 15؛ استخدم AOS-* فقط إذا كنت متأكدًا من الكود، وإلا اذكر المجال دون اختلاق ID.",
                "لا تنشئ مستهدفًا رقميًا أو نسبة تحسن أو مدة تنفيذ من عندك.",
                "الأرقام التاريخية في الحالة لا تتحول إلى أهداف مستقبلية.",
                "لا تكرر الفكرة نفسها بصياغات متعددة.",
                "إذا كانت الأدلة غير كافية، اذكر ما يلزم التحقق منه بدل افتراض النتيجة.",
            ],
            "advisor_id": advisor["advisor_id"],
            "advisor_name_ar": advisor["advisor_name_ar"],
            "scope_contract": scope_contract,
            "other_selected_advisors_without_opinions": other_selected,
            "case_context": self._shared_context(request),
        }

        user_prompt = json.dumps(task, ensure_ascii=False, indent=2)
        opinion = self._generate(
            "specialist",
            prompt,
            user_prompt,
            min(ADVISOR_MAX_NEW_TOKENS, 1000),
            deterministic=True,
            repetition_penalty=1.10,
            no_repeat_ngram_size=8,
        )

        if self._advisor_output_is_bad(opinion):
            retry_task = dict(task)
            retry_task["instruction"] = (
                task["instruction"]
                + " المحاولة السابقة كانت طويلة أو متكررة أو خرجت عن الشكل المطلوب. "
                  "أعد الصياغة من الصفر بشكل أقصر. احذف كل تكرار وكل توصية خارج نطاقك، "
                  "ولا تعرض تعريفات Expert DNA. ركّز فقط على ما تملكه أنت في هذه الحالة."
            )
            opinion = self._generate(
                "specialist",
                prompt,
                json.dumps(retry_task, ensure_ascii=False, indent=2),
                min(ADVISOR_MAX_NEW_TOKENS, 850),
                deterministic=True,
                repetition_penalty=1.14,
                no_repeat_ngram_size=8,
            )

        return {
            "advisor_id": advisor["advisor_id"],
            "backend_id": advisor.get("backend_id"),
            "advisor_name_ar": advisor["advisor_name_ar"],
            "opinion": opinion,
        }

    @staticmethod
    def _normalize_digits(text: Any) -> str:
        return str(text or "").translate(
            str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        )

    @classmethod
    def _strip_structural_identifiers(cls, text: Any) -> str:
        normalized = cls._normalize_digits(text)
        normalized = re.sub(
            r"\b(?:AOS-(?:LD|SP|FG|SE)-\d{1,2}|AOS-META-\d{1,2}|ATHAR-ADV-\d{1,2})\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\b(?:INT|SPRINT|SPR)-0?\d{1,2}\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        # The 12-week horizon is a fixed product constraint, not a model-created target.
        normalized = re.sub(
            r"\b12\s*(?:week|weeks|أسبوع|اسبوع|أسبوعًا|اسبوعا|أسابيع)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\b(?:sprint|week|الأسبوع|الاسبوع|السبرنت)\s*(?:#\s*)?(?:[1-9]|1[0-2])\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized

    @classmethod
    def _extract_number_tokens(cls, text: Any) -> set[str]:
        normalized = cls._strip_structural_identifiers(text)
        return set(re.findall(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])", normalized))

    @classmethod
    def _collect_strings(cls, value: Any) -> List[str]:
        out: List[str] = []
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                out.extend(cls._collect_strings(item))
        elif isinstance(value, list):
            for item in value:
                out.extend(cls._collect_strings(item))
        return out

    def _canonical_advisor_id(self, value: Any) -> Optional[str]:
        if isinstance(value, dict):
            value = value.get("advisor_id") or value.get("model_advisor_id") or value.get("id")
        if isinstance(value, int):
            row = self.registry_by_number.get(value)
            return row["advisor_id"] if row else None
        raw = str(value or "").strip()
        if raw in self.registry_by_model_id:
            return raw
        if raw.isdigit():
            row = self.registry_by_number.get(int(raw))
            return row["advisor_id"] if row else None
        return None

    def _normalize_attribution(self, value: Any, selected_ids: List[str]) -> List[str]:
        if not isinstance(value, list):
            value = [value] if value is not None else []
        allowed = set(selected_ids)
        out: List[str] = []
        for raw in value:
            canonical = self._canonical_advisor_id(raw)
            if canonical and canonical in allowed and canonical not in out:
                out.append(canonical)
        return out

    @staticmethod
    def _normalize_confidence(value: Any) -> str:
        raw = str(value or "").strip().lower()
        high = {"high", "مرتفع", "مرتفعة", "عالي", "عالية", "مؤكد", "مرجح جدا", "مرجح جدًا"}
        med = {"medium", "متوسط", "متوسطة", "محتمل"}
        low = {"low", "منخفض", "منخفضة", "إشارة ضعيفة", "اشارة ضعيفة", "غير معلوم"}
        if raw in {x.lower() for x in high}:
            return "High"
        if raw in {x.lower() for x in med}:
            return "Medium"
        if raw in {x.lower() for x in low}:
            return "Low"
        return ""

    @staticmethod
    def _normalize_interaction(value: Any) -> str:
        raw = re.sub(r"\s+", " ", str(value or "").strip().upper())
        raw = raw.replace("OFF-TRADE", "TRADE-OFF").replace("TRADE OFF", "TRADE-OFF")
        raw = raw.replace("EVIDENCE_GAP", "EVIDENCE GAP").replace("SCOPE_CONFLICT", "SCOPE CONFLICT")
        return raw if raw in INTERACTION_TYPES else ""

    @staticmethod
    def _normalize_evidence_code(value: Any) -> str:
        raw = str(value or "").strip().upper()
        return raw if raw in EVIDENCE_CODES else ""

    @staticmethod
    def _more_conservative_evidence(codes: List[str]) -> str:
        clean = [x for x in codes if x in EVIDENCE_CODES]
        if not clean:
            return ""
        for code in ("U", "A1", "I2"):
            if code in clean:
                return code
        if len(set(clean)) == 1:
            return clean[0]
        # A sprint combining several supported interventions is a synthesis.
        return "I1"

    @staticmethod
    def _minimum_confidence(values: List[str]) -> str:
        score = {"Low": 1, "Medium": 2, "High": 3}
        clean = [x for x in values if x in score]
        if not clean:
            return ""
        return min(clean, key=lambda x: score[x])

    @staticmethod
    def _derive_council_interaction(interactions: List[str]) -> str:
        clean = [x for x in interactions if x in INTERACTION_TYPES]
        for priority in (
            "CONFLICT",
            "SCOPE CONFLICT",
            "EVIDENCE GAP",
            "TRADE-OFF",
            "COMPLEMENTARY",
            "CONSENSUS",
        ):
            if priority in clean:
                return priority
        return "CONSENSUS"

    @staticmethod
    def _derive_overall_confidence(interventions: List[Dict[str, Any]]) -> str:
        if not interventions:
            return "Low"
        levels = [str(x.get("confidence_level") or "") for x in interventions]
        interactions = [str(x.get("interaction_type") or "") for x in interventions]
        evidences = [str(x.get("evidence_classification") or "") for x in interventions]

        low_count = sum(1 for x in levels if x == "Low")
        if low_count >= max(1, (len(interventions) + 1) // 2):
            return "Low"
        if any(x in {"U", "A1"} for x in evidences):
            return "Medium" if low_count == 0 else "Low"
        if any(x in {"CONFLICT", "EVIDENCE GAP", "SCOPE CONFLICT"} for x in interactions):
            return "Medium" if low_count == 0 else "Low"
        if all(x == "High" for x in levels):
            return "High"
        return "Medium"

    def _normalize_meta_result(
        self,
        result: Dict[str, Any],
        selected_ids: List[str],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise ValueError("Meta result must be an object.")
        # JSON round-trip gives us a detached plain object.
        result = json.loads(json.dumps(result, ensure_ascii=False))

        result["schema_version"] = CONSULTATION_SCHEMA_VERSION
        result["consultation_id"] = (
            request.get("consultation_id")
            or (request.get("input") or {}).get("consultation_id")
        )
        result["status"] = OPEN_STATUS
        result["involved_advisor_ids"] = list(selected_ids)

        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict):
            suggestion = {}
            result["suggestion"] = suggestion

        interventions = suggestion.get("interventions")
        if not isinstance(interventions, list):
            interventions = []
            suggestion["interventions"] = interventions

        by_id: Dict[str, Dict[str, Any]] = {}
        for idx, intervention in enumerate(interventions, start=1):
            if not isinstance(intervention, dict):
                continue
            iid = f"INT-{idx:02d}"
            intervention["intervention_id"] = iid
            intervention["status"] = OPEN_STATUS
            intervention["attribution"] = self._normalize_attribution(
                intervention.get("attribution") or intervention.get("advisor_ids"),
                selected_ids,
            )
            intervention["evidence_classification"] = self._normalize_evidence_code(
                intervention.get("evidence_classification") or intervention.get("evidence_class")
            )
            intervention["interaction_type"] = self._normalize_interaction(
                intervention.get("interaction_type") or intervention.get("discussion_type")
            )
            confidence = self._normalize_confidence(
                intervention.get("confidence_level") or intervention.get("confidence")
            )
            evidence = intervention["evidence_classification"]
            interaction = intervention["interaction_type"]
            if evidence in {"U", "A1"}:
                confidence = "Low"
            elif evidence == "I2" and confidence == "High":
                confidence = "Medium"
            if interaction in {"CONFLICT", "EVIDENCE GAP", "SCOPE CONFLICT"} and confidence == "High":
                confidence = "Medium"
            intervention["confidence_level"] = confidence

            results = intervention.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        # Result rows inherit provenance/classification from their
                        # parent intervention so the UI never displays synthetic
                        # metadata disconnected from the council reasoning.
                        item["status"] = OPEN_STATUS
                        item["attribution"] = list(intervention["attribution"])
                        item["evidence_classification"] = intervention["evidence_classification"]
                        item["interaction_type"] = intervention["interaction_type"]
                        item["confidence_level"] = intervention["confidence_level"]
            by_id[iid] = intervention

        sprints = suggestion.get("sprints")
        if not isinstance(sprints, list):
            for alias in ("sprint_plan", "weekly_plan", "weeks"):
                candidate = suggestion.get(alias)
                if isinstance(candidate, list):
                    sprints = candidate
                    break
        if not isinstance(sprints, list):
            sprints = []
        suggestion["sprints"] = sprints
        suggestion["sprint_count"] = SPRINT_COUNT

        for idx, sprint in enumerate(sprints, start=1):
            if not isinstance(sprint, dict):
                continue
            sprint["sprint_number"] = idx
            sprint["week_number"] = idx
            sprint["status"] = OPEN_STATUS

            source_ids = sprint.get("source_intervention_ids") or sprint.get("intervention_ids") or []
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            source_ids = [str(x).strip().upper() for x in source_ids if str(x).strip()]
            source_ids = [x for x in source_ids if x in by_id]
            sprint["source_intervention_ids"] = list(dict.fromkeys(source_ids))

            sprint["attribution"] = self._normalize_attribution(
                sprint.get("attribution") or sprint.get("advisor_ids"),
                selected_ids,
            )
            sources = [by_id[x] for x in sprint["source_intervention_ids"] if x in by_id]
            if not sprint["attribution"] and sources:
                merged: List[str] = []
                for source in sources:
                    for aid in source.get("attribution", []):
                        if aid not in merged:
                            merged.append(aid)
                sprint["attribution"] = merged

            evidence = self._normalize_evidence_code(
                sprint.get("evidence_classification") or sprint.get("evidence_class")
            )
            if not evidence and sources:
                evidence = self._more_conservative_evidence(
                    [str(x.get("evidence_classification") or "") for x in sources]
                )
            sprint["evidence_classification"] = evidence

            interaction = self._normalize_interaction(
                sprint.get("interaction_type") or sprint.get("discussion_type")
            )
            if not interaction and sources:
                interactions = [str(x.get("interaction_type") or "") for x in sources]
                interaction = (
                    interactions[0]
                    if len(set(interactions)) == 1 and interactions[0] in INTERACTION_TYPES
                    else "COMPLEMENTARY"
                )
            sprint["interaction_type"] = interaction

            confidence = self._normalize_confidence(
                sprint.get("confidence_level") or sprint.get("confidence")
            )
            if not confidence and sources:
                confidence = self._minimum_confidence(
                    [str(x.get("confidence_level") or "") for x in sources]
                )
            if evidence in {"U", "A1"}:
                confidence = "Low"
            elif evidence == "I2" and confidence == "High":
                confidence = "Medium"
            if interaction in {"CONFLICT", "EVIDENCE GAP", "SCOPE CONFLICT"} and confidence == "High":
                confidence = "Medium"
            sprint["confidence_level"] = confidence

            if not str(sprint.get("evidence_basis") or "").strip() and sources:
                basis = [str(x.get("evidence_basis") or "").strip() for x in sources]
                basis = [x for x in basis if x]
                if basis:
                    sprint["evidence_basis"] = "؛ ".join(basis)[:900]

        item_interactions = [str(x.get("interaction_type") or "") for x in interventions]
        derived_interaction = self._derive_council_interaction(item_interactions)
        requested_interaction = self._normalize_interaction(result.get("council_interaction_type"))
        result["council_interaction_type"] = (
            requested_interaction
            if requested_interaction and requested_interaction in item_interactions
            else derived_interaction
        )

        derived_confidence = self._derive_overall_confidence(interventions)
        requested_confidence = self._normalize_confidence(result.get("overall_confidence"))
        confidence_score = {"Low": 1, "Medium": 2, "High": 3}
        if requested_confidence:
            # The Meta Advisor applies the Confidence Engine; code only prevents
            # the public API from being more optimistic than item-level evidence.
            result["overall_confidence"] = min(
                (requested_confidence, derived_confidence),
                key=lambda x: confidence_score[x],
            )
        else:
            result["overall_confidence"] = derived_confidence

        # Derive a stable top-level attribution map from item-level provenance.
        attribution_summary = []
        for aid in selected_ids:
            titles = [
                str(x.get("title") or "").strip()
                for x in interventions
                if aid in x.get("attribution", []) and str(x.get("title") or "").strip()
            ]
            if not titles:
                titles = [
                    str(x.get("title") or "").strip()
                    for x in sprints
                    if aid in x.get("attribution", []) and str(x.get("title") or "").strip()
                ]
            titles = list(dict.fromkeys(titles))[:4]
            if titles:
                attribution_summary.append({
                    "advisor_id": aid,
                    "contribution": "؛ ".join(titles),
                })
        result["attribution"] = attribution_summary

        return result

    def _validate_backend_result(self, result: Dict[str, Any], selected_ids: List[str]) -> None:
        if result.get("schema_version") != CONSULTATION_SCHEMA_VERSION:
            raise ValueError("schema_version is invalid.")
        if result.get("status") != OPEN_STATUS:
            raise ValueError("status must be OPEN.")
        if not isinstance(result.get("recommendation_text"), str) or len(result["recommendation_text"].strip()) < 40:
            raise ValueError("recommendation_text must be a substantive string.")
        if result.get("overall_confidence") not in CONFIDENCE_LEVELS:
            raise ValueError("overall_confidence must be High, Medium, or Low.")
        if result.get("council_interaction_type") not in INTERACTION_TYPES:
            raise ValueError("council_interaction_type is invalid.")

        involved = result.get("involved_advisor_ids")
        if involved != selected_ids:
            raise ValueError("involved_advisor_ids must use the selected canonical AOS-* IDs in order.")

        attribution_summary = result.get("attribution")
        if not isinstance(attribution_summary, list) or not attribution_summary:
            raise ValueError("attribution must be a non-empty array.")
        for item in attribution_summary:
            if not isinstance(item, dict):
                raise ValueError("attribution items must be objects.")
            if item.get("advisor_id") not in selected_ids:
                raise ValueError("attribution contains an advisor that was not selected.")
            if not isinstance(item.get("contribution"), str) or not item["contribution"].strip():
                raise ValueError("attribution.contribution is required.")

        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict):
            raise ValueError("suggestion must be an object.")
        interventions = suggestion.get("interventions")
        if not isinstance(interventions, list) or not (1 <= len(interventions) <= 4):
            raise ValueError("suggestion.interventions must contain 1 to 4 interventions.")

        valid_intervention_ids = set()
        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, dict):
                raise ValueError(f"interventions[{i}] must be an object.")
            required = {
                "intervention_id", "title", "impact_description", "reportable_value",
                "attribution", "evidence_classification", "evidence_basis",
                "interaction_type", "confidence_level", "status", "results",
            }
            missing = required - intervention.keys()
            if missing:
                raise ValueError(f"interventions[{i}] missing: {sorted(missing)}")
            iid = intervention["intervention_id"]
            if iid != f"INT-{i + 1:02d}":
                raise ValueError(f"interventions[{i}].intervention_id is invalid.")
            valid_intervention_ids.add(iid)
            if intervention.get("status") != OPEN_STATUS:
                raise ValueError(f"interventions[{i}].status must be OPEN.")
            if not isinstance(intervention.get("title"), str) or not intervention["title"].strip():
                raise ValueError(f"interventions[{i}].title is required.")
            if not isinstance(intervention.get("impact_description"), str) or not intervention["impact_description"].strip():
                raise ValueError(f"interventions[{i}].impact_description is required.")
            if not isinstance(intervention.get("reportable_value"), str) or not intervention["reportable_value"].strip():
                raise ValueError(f"interventions[{i}].reportable_value is required.")
            attrs = intervention.get("attribution")
            if not isinstance(attrs, list) or not attrs or any(x not in selected_ids for x in attrs):
                raise ValueError(f"interventions[{i}].attribution must use selected canonical IDs.")
            if intervention.get("evidence_classification") not in EVIDENCE_CODES:
                raise ValueError(f"interventions[{i}].evidence_classification is invalid.")
            if not isinstance(intervention.get("evidence_basis"), str) or not intervention["evidence_basis"].strip():
                raise ValueError(f"interventions[{i}].evidence_basis is required.")
            if intervention.get("interaction_type") not in INTERACTION_TYPES:
                raise ValueError(f"interventions[{i}].interaction_type is invalid.")
            if intervention.get("confidence_level") not in CONFIDENCE_LEVELS:
                raise ValueError(f"interventions[{i}].confidence_level is invalid.")

            results = intervention.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(f"interventions[{i}].results must be non-empty.")
            for j, result_item in enumerate(results):
                if not isinstance(result_item, dict) or not isinstance(result_item.get("text"), str) or not result_item["text"].strip():
                    raise ValueError(f"interventions[{i}].results[{j}].text is required.")
                if result_item.get("status") != OPEN_STATUS:
                    raise ValueError(f"interventions[{i}].results[{j}].status must be OPEN.")
                if result_item.get("attribution") != intervention.get("attribution"):
                    raise ValueError(f"interventions[{i}].results[{j}].attribution must inherit from intervention.")
                if result_item.get("evidence_classification") != intervention.get("evidence_classification"):
                    raise ValueError(f"interventions[{i}].results[{j}].evidence_classification must inherit from intervention.")
                if result_item.get("interaction_type") != intervention.get("interaction_type"):
                    raise ValueError(f"interventions[{i}].results[{j}].interaction_type must inherit from intervention.")
                if result_item.get("confidence_level") != intervention.get("confidence_level"):
                    raise ValueError(f"interventions[{i}].results[{j}].confidence_level must inherit from intervention.")
                outputs = result_item.get("outputs")
                if not isinstance(outputs, list) or not outputs:
                    raise ValueError(f"interventions[{i}].results[{j}].outputs must be non-empty.")
                for k, output in enumerate(outputs):
                    if not isinstance(output, dict) or not isinstance(output.get("text"), str) or not output["text"].strip():
                        raise ValueError(f"interventions[{i}].results[{j}].outputs[{k}].text is required.")

        sprints = suggestion.get("sprints")
        if not isinstance(sprints, list) or len(sprints) != SPRINT_COUNT:
            raise ValueError(f"suggestion.sprints must contain exactly {SPRINT_COUNT} weekly sprints.")
        if suggestion.get("sprint_count") != SPRINT_COUNT:
            raise ValueError(f"suggestion.sprint_count must be {SPRINT_COUNT}.")

        covered_interventions = set()
        for i, sprint in enumerate(sprints):
            if not isinstance(sprint, dict):
                raise ValueError(f"sprints[{i}] must be an object.")
            required = {
                "sprint_number", "week_number", "title", "objective", "actions", "outputs",
                "source_intervention_ids", "attribution", "evidence_classification",
                "evidence_basis", "interaction_type", "confidence_level", "status",
            }
            missing = required - sprint.keys()
            if missing:
                raise ValueError(f"sprints[{i}] missing: {sorted(missing)}")
            expected = i + 1
            if sprint.get("sprint_number") != expected or sprint.get("week_number") != expected:
                raise ValueError(f"sprints[{i}] must map to sprint/week {expected}.")
            if sprint.get("status") != OPEN_STATUS:
                raise ValueError(f"sprints[{i}].status must be OPEN.")
            if not isinstance(sprint.get("title"), str) or not sprint["title"].strip():
                raise ValueError(f"sprints[{i}].title is required.")
            if not isinstance(sprint.get("objective"), str) or not sprint["objective"].strip():
                raise ValueError(f"sprints[{i}].objective is required.")
            actions = sprint.get("actions")
            outputs = sprint.get("outputs")
            if not isinstance(actions, list) or not (1 <= len(actions) <= 4) or not all(isinstance(x, str) and x.strip() for x in actions):
                raise ValueError(f"sprints[{i}].actions must contain 1 to 4 strings.")
            if not isinstance(outputs, list) or not (1 <= len(outputs) <= 3) or not all(isinstance(x, str) and x.strip() for x in outputs):
                raise ValueError(f"sprints[{i}].outputs must contain 1 to 3 strings.")
            source_ids = sprint.get("source_intervention_ids")
            if not isinstance(source_ids, list) or not source_ids or any(x not in valid_intervention_ids for x in source_ids):
                raise ValueError(f"sprints[{i}].source_intervention_ids are invalid.")
            covered_interventions.update(source_ids)
            attrs = sprint.get("attribution")
            if not isinstance(attrs, list) or not attrs or any(x not in selected_ids for x in attrs):
                raise ValueError(f"sprints[{i}].attribution must use selected canonical IDs.")
            if sprint.get("evidence_classification") not in EVIDENCE_CODES:
                raise ValueError(f"sprints[{i}].evidence_classification is invalid.")
            if not isinstance(sprint.get("evidence_basis"), str) or not sprint["evidence_basis"].strip():
                raise ValueError(f"sprints[{i}].evidence_basis is required.")
            if sprint.get("interaction_type") not in INTERACTION_TYPES:
                raise ValueError(f"sprints[{i}].interaction_type is invalid.")
            if sprint.get("confidence_level") not in CONFIDENCE_LEVELS:
                raise ValueError(f"sprints[{i}].confidence_level is invalid.")

        if covered_interventions != valid_intervention_ids:
            missing = sorted(valid_intervention_ids - covered_interventions)
            raise ValueError(f"Every intervention must be scheduled in the 12 sprints. Missing: {missing}")

    def _grounding_violations(self, result: Dict[str, Any], request: Dict[str, Any]) -> List[str]:
        """Reject fabricated quantitative claims while allowing structural IDs/week numbering."""
        violations: List[str] = []
        source_obj = self._shared_context(request)
        source_text = self._normalize_digits(json.dumps(source_obj, ensure_ascii=False, sort_keys=True))
        source_numbers = self._extract_number_tokens(source_text)

        def scan(path: str, value: Any) -> None:
            for s in self._collect_strings(value):
                norm = self._normalize_digits(s)
                for number in self._extract_number_tokens(norm):
                    if number not in source_numbers:
                        violations.append(f"{path} contains unsupported number: {number}")
                percentage_claims = re.findall(
                    r"(?:زيادة|رفع|خفض|تقليل|تحسين|الوصول|تحقيق|مستهدف|بنسبة)"
                    r".{0,50}?(\d+(?:[.,]\d+)?)\s*(?:%|٪)",
                    self._strip_structural_identifiers(norm),
                )
                for pct_number in percentage_claims:
                    # Historical percentages already present in case_context may be
                    # repeated. New planning percentages remain forbidden.
                    if pct_number not in source_numbers:
                        violations.append(f"{path} contains an unsupported planning percentage.")
                duration_matches = re.findall(
                    r"(?:خلال|في غضون|مدة)\s+\d+(?:[.,]\d+)?\s*"
                    r"(?:يوم|أيام|أسبوع|أسابيع|شهر|أشهر|سنة|سنوات|عام|أعوام)",
                    self._strip_structural_identifiers(norm),
                )
                for phrase in duration_matches:
                    if phrase not in source_text:
                        violations.append(f"{path} contains unsupported duration: {phrase}")

        scan("recommendation_text", result.get("recommendation_text"))
        suggestion = result.get("suggestion") or {}
        for idx, intervention in enumerate(suggestion.get("interventions") or []):
            if not isinstance(intervention, dict):
                continue
            scan(
                f"interventions[{idx}]",
                {
                    "title": intervention.get("title"),
                    "impact_description": intervention.get("impact_description"),
                    "reportable_value": intervention.get("reportable_value"),
                    "evidence_basis": intervention.get("evidence_basis"),
                    "results": intervention.get("results"),
                },
            )
        for idx, sprint in enumerate(suggestion.get("sprints") or []):
            if not isinstance(sprint, dict):
                continue
            scan(
                f"sprints[{idx}]",
                {
                    "title": sprint.get("title"),
                    "objective": sprint.get("objective"),
                    "actions": sprint.get("actions"),
                    "outputs": sprint.get("outputs"),
                    "evidence_basis": sprint.get("evidence_basis"),
                },
            )

        return list(dict.fromkeys(violations))

    def _build_scope_cards(self, advisor_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cards = []
        for output in advisor_outputs:
            aid = output.get("advisor_id")
            reg = self.registry_by_model_id.get(str(aid))
            if not reg:
                continue
            prompt = self._load_prompt(ADVISOR_PROMPTS_DIR / reg["prompt_file"])
            cards.append({
                "advisor_id": reg["advisor_id"],
                "advisor_name_ar": reg["advisor_name_ar"],
                "scope_contract": self._extract_scope_contract(prompt, max_chars=1800),
            })
        return cards

    def _run_meta(self, request: Dict[str, Any], advisor_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._last_meta_debug = {
            "schema_repair_used": False,
            "repair_used": False,
            "initial_schema_error": None,
            "initial_grounding_violations": [],
            "final_grounding_violations": [],
        }
        selected_ids = [str(x["advisor_id"]) for x in advisor_outputs]
        scope_cards = self._build_scope_cards(advisor_outputs)

        meta_task = {
            "instruction": (
                "هذه مرحلة التركيب النهائي للمجلس. طبّق AOS-META-00 الأصلي بالكامل، لا تلخص الآراء فقط. "
                "قارن المساهمات، اكشف الاتفاق والتكامل والمفاضلة والتعارض ونقص الدليل وتجاوز النطاق، "
                "ثم أصدر توصية موحدة قابلة للتنفيذ. أي رأي يتجاوز Scope Contract لمستشاره لا يتحول إلى "
                "توصية نهائية إلا إذا كان مستشار مختار آخر يملك هذا المجال ويدعمه. "
                "استخدم IDs الرسمية AOS-* فقط في Attribution. لا تستخدم أرقامًا مثل 13 أو 15 كهوية مستشار. "
                "ابنِ خطة تنفيذ من 12 Sprint بالضبط، كل Sprint = أسبوع واحد، تغطي التدخلات النهائية فقط ولا "
                "تخلق نطاقًا جديدًا. هيكل الأسابيع الاثني عشر قيد منتج ثابت ومسموح؛ ممنوع اختراع أي مدة أخرى. "
                "لا تخترع أرقامًا أو نسبًا أو خطوط أساس أو مستهدفات أو تواريخ أو ميزانيات. يجوز فقط إعادة استخدام رقم/نسبة واردة صراحة في case_context وفي نفس الدلالة. "
                "أعد JSON صالحًا فقط وفق العقد المطلوب."
            ),
            "evidence_protocol": {
                "E1": "VERIFIED ORGANIZATION DATA — بيانات مثبتة من الجهة.",
                "E2": "AUTHORITATIVE REFERENCE — مرجع رسمي أو نظام أو معيار موثوق موجود فعلاً في الأدلة.",
                "E3": "CORROBORATED EVIDENCE — معلومة مدعومة من أكثر من مصدر موثوق.",
                "I1": "STRONG INFERENCE — استنتاج قوي من الأدلة.",
                "I2": "WORKING INFERENCE — استنتاج يحتاج تحققًا إضافيًا.",
                "A1": "EXPLICIT ASSUMPTION — افتراض معلن يحتاج اختبارًا.",
                "U": "UNKNOWN — غير معروف.",
            },
            "evidence_rules": [
                "صنّف أساس كل تدخل وكل Sprint بكود واحد فقط.",
                "لا تستخدم E2 أو E3 ما لم يوجد المرجع/التأييد فعلاً في المدخلات.",
                "التوصية المشتقة من بيانات الجهة تكون عادة I1 أو I2؛ لا تسمِّ الاستنتاج E1 لمجرد أن البيانات الأصلية E1.",
                "A1 وU لا يتحولان إلى حقيقة، ويجب أن يخفضا الثقة.",
            ],
            "interaction_protocol": {
                "CONSENSUS": "اتفاق قوي.",
                "COMPLEMENTARY": "آراء مختلفة لكنها متكاملة.",
                "TRADE-OFF": "خيارات صحيحة بينها مفاضلة.",
                "CONFLICT": "تعارض مباشر.",
                "EVIDENCE GAP": "الخلاف أو القرار متأثر بنقص الأدلة.",
                "SCOPE CONFLICT": "أحد الآراء تجاوز نطاق المستشار أو اصطدم بملكية تخصص آخر.",
            },
            "confidence_mapping": {
                "High": "مؤكد أو مرجح جدًا: أدلة كافية/قوية وعدم يقين محدود.",
                "Medium": "محتمل: توجد أدلة لكن بدائل أو فجوات معقولة.",
                "Low": "إشارة ضعيفة أو غير معلوم: البيانات غير كافية أو يعتمد على افتراضات.",
            },
            "case_context": self._shared_context(request),
            "selected_advisor_ids": selected_ids,
            "advisor_scope_cards": scope_cards,
            "selected_advisor_outputs": advisor_outputs,
            "required_schema": {
                "recommendation_text": "string — النص النهائي الموحد",
                "overall_confidence": "High|Medium|Low according to Confidence Engine",
                "council_interaction_type": "CONSENSUS|COMPLEMENTARY|TRADE-OFF|CONFLICT|EVIDENCE GAP|SCOPE CONFLICT",
                "suggestion": {
                    "interventions": [
                        {
                            "title": "string",
                            "impact_description": "string",
                            "reportable_value": "string; قيمة/مؤشر قابل للتقرير. يجوز استخدام هدف رقمي فقط إذا ورد صراحة في case_context لنفس المعنى",
                            "attribution": ["AOS-*-NN from selected_advisor_ids only"],
                            "evidence_classification": "E1|E2|E3|I1|I2|A1|U",
                            "evidence_basis": "string explaining the actual basis",
                            "interaction_type": "CONSENSUS|COMPLEMENTARY|TRADE-OFF|CONFLICT|EVIDENCE GAP|SCOPE CONFLICT",
                            "confidence_level": "High|Medium|Low",
                            "results": [
                                {"text": "string", "outputs": [{"text": "string"}]}
                            ],
                        }
                    ],
                    "sprints": [
                        {
                            "title": "string",
                            "objective": "string",
                            "actions": ["1-4 concise actions"],
                            "outputs": ["1-3 deliverables"],
                            "source_intervention_ids": ["INT-01 etc., based on intervention order"],
                            "attribution": ["selected AOS-* IDs only"],
                            "evidence_classification": "E1|E2|E3|I1|I2|A1|U",
                            "evidence_basis": "string",
                            "interaction_type": "allowed interaction enum",
                            "confidence_level": "High|Medium|Low",
                        }
                    ],
                },
            },
            "hard_rules": [
                "Return valid JSON only. No Markdown fences.",
                "Return 1 to 4 interventions; omit weak or unsupported interventions.",
                "Return exactly 12 sprints in chronological order; one sprint equals one week.",
                "INT-01 means the first intervention in the returned interventions array, INT-02 the second, and so on.",
                "Every intervention must be scheduled in at least one sprint.",
                "Every intervention and sprint must have non-empty attribution using selected canonical AOS-* IDs only.",
                "Do not create an intervention from case_context alone; it must be supported by at least one selected advisor opinion within that advisor's scope.",
                "When an advisor opinion crosses its scope, classify the issue as SCOPE CONFLICT and exclude that out-of-scope part unless the owning selected advisor supports it.",
                "Do not generate beneficiaries_count, evidence, or is_selected.",
                "Do not invent numeric targets, percentages, baselines, dates, deadlines, budgets, or durations other than the fixed 12 weekly sprint structure.",
                "reportable_value يلتزم بعقد Screen 3: إن وُجد هدف رقمي معتمد وصريح في case_context لنفس المعنى يجوز تكراره كما هو؛ وإلا استخدم مؤشرًا قابلًا للقياس دون اختراع قيمة رقمية.",
                "Do not write advisor codes inside recommendation prose; IDs belong only in attribution fields.",
                "Keep sprint actions case-specific and non-repetitive; later sprints may continue earlier work but must add a distinct next step or deliverable.",
            ],
        }

        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"

        def generate_parse_normalize(task: Dict[str, Any]) -> Dict[str, Any]:
            raw_text = self._generate(
                meta_adapter,
                self.meta_prompt,
                json.dumps(task, ensure_ascii=False, indent=2),
                META_MAX_NEW_TOKENS,
                deterministic=True,
                repetition_penalty=1.08,
                no_repeat_ngram_size=8,
            )
            parsed = self.extract_json_object(raw_text)
            normalized = self._normalize_meta_result(parsed, selected_ids, request)
            self._validate_backend_result(normalized, selected_ids)
            return normalized

        try:
            result = generate_parse_normalize(meta_task)
        except Exception as exc:
            self._last_meta_debug["schema_repair_used"] = True
            self._last_meta_debug["initial_schema_error"] = str(exc)[:1200]
            schema_repair = dict(meta_task)
            schema_repair["instruction"] = (
                meta_task["instruction"]
                + " المحاولة السابقة لم تلتزم بعقد الـAPI. أعد بناء الإجابة كاملة من الصفر. "
                  "يجب وجود recommendation_text، من 1 إلى 4 interventions، و12 sprints بالضبط. "
                  "لا تحذف attribution/evidence_classification/evidence_basis/interaction_type/confidence_level."
            )
            schema_repair["previous_validation_error"] = str(exc)[:1200]
            result = generate_parse_normalize(schema_repair)

        violations = self._grounding_violations(result, request)
        self._last_meta_debug["initial_grounding_violations"] = list(violations)

        if violations:
            self._last_meta_debug["repair_used"] = True
            repair_task = dict(meta_task)
            repair_task["instruction"] = (
                meta_task["instruction"]
                + " المحاولة السابقة خالفت Grounding. أعد JSON كاملًا مع الحفاظ على 12 Sprint، "
                  "واحذف أي رقم/نسبة/مدة/مستهدف غير موجود في case_context. لا تدافع عن النص السابق."
            )
            repair_task["validation_errors"] = violations[:20]
            repair_task["previous_invalid_output"] = result
            result = generate_parse_normalize(repair_task)
            violations = self._grounding_violations(result, request)

        self._last_meta_debug["final_grounding_violations"] = list(violations)
        if violations:
            raise ValueError(
                "Council grounding validation failed after repair: "
                + " | ".join(violations[:8])
            )

        return result

    @staticmethod
    def _screen3_confidence(value: Any) -> str:
        mapping = {
            "High": "مرتفعة",
            "Medium": "متوسطة",
            "Low": "منخفضة",
            "مرتفعة": "مرتفعة",
            "متوسطة": "متوسطة",
            "منخفضة": "منخفضة",
        }
        return mapping.get(str(value or "").strip(), "متوسطة")

    @staticmethod
    def _backend_advisor_ids(selected: List[Dict[str, Any]]) -> List[int]:
        """Return the exact integer IDs Screen 3 expects.

        Prefer the incoming backend ``id``.  For the currently agreed Athar
        catalogue, an omitted backend id falls back to the authoritative
        advisor_number (1..35) so the contract stays usable during integration.
        """
        out: List[int] = []
        for advisor in selected:
            raw = advisor.get("backend_id")
            if not isinstance(raw, int):
                raw = advisor.get("advisor_number")
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value not in out:
                out.append(value)
        return out

    def _screen3_public_response(
        self,
        rich_result: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Project the rich council result onto the *exact* Screen-3 contract.

        Attribution/evidence/interaction/12-sprint metadata remains in the
        private run log for the later product contracts.  It is deliberately
        not leaked into Screen 3 before the backend has fields for it.
        """
        suggestion = rich_result.get("suggestion") or {}
        public_interventions: List[Dict[str, Any]] = []

        for intervention in suggestion.get("interventions") or []:
            if not isinstance(intervention, dict):
                continue
            public_results: List[Dict[str, Any]] = []
            for result_item in intervention.get("results") or []:
                if not isinstance(result_item, dict):
                    continue
                public_outputs = [
                    {"text": str(output.get("text") or "").strip()}
                    for output in (result_item.get("outputs") or [])
                    if isinstance(output, dict) and str(output.get("text") or "").strip()
                ]
                if public_outputs and str(result_item.get("text") or "").strip():
                    public_results.append({
                        "text": str(result_item.get("text") or "").strip(),
                        "outputs": public_outputs,
                    })

            if not public_results:
                continue

            public_interventions.append({
                "title": str(intervention.get("title") or "").strip(),
                "confidence_level": self._screen3_confidence(
                    intervention.get("confidence_level")
                ),
                "impact_description": str(
                    intervention.get("impact_description") or ""
                ).strip(),
                "reportable_value": str(
                    intervention.get("reportable_value") or ""
                ).strip(),
                "results": public_results,
            })

        public = {
            "involved_advisor_ids": self._backend_advisor_ids(selected),
            "suggestion": {"interventions": public_interventions},
        }
        self._validate_screen3_public_response(public)
        return public

    @staticmethod
    def _validate_screen3_public_response(result: Dict[str, Any]) -> None:
        if set(result.keys()) != {"involved_advisor_ids", "suggestion"}:
            raise ValueError("Screen 3 response has unexpected top-level fields.")
        ids = result.get("involved_advisor_ids")
        if not isinstance(ids, list) or any(not isinstance(x, int) for x in ids):
            raise ValueError("Screen 3 involved_advisor_ids must be list<int>.")
        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict) or set(suggestion.keys()) != {"interventions"}:
            raise ValueError("Screen 3 suggestion must contain interventions only.")
        interventions = suggestion.get("interventions")
        if not isinstance(interventions, list) or not (1 <= len(interventions) <= 4):
            raise ValueError("Screen 3 requires 1 to 4 interventions.")

        required = {
            "title", "confidence_level", "impact_description",
            "reportable_value", "results",
        }
        allowed_confidence = {
            "مرتفعة", "متوسطة", "منخفضة", "high", "medium", "low"
        }
        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, dict) or set(intervention.keys()) != required:
                raise ValueError(f"Screen 3 interventions[{i}] fields are invalid.")
            for key in ("title", "impact_description", "reportable_value"):
                if not isinstance(intervention.get(key), str) or not intervention[key].strip():
                    raise ValueError(f"Screen 3 interventions[{i}].{key} is required.")
            if str(intervention.get("confidence_level") or "").lower() not in {
                x.lower() for x in allowed_confidence
            }:
                raise ValueError(f"Screen 3 interventions[{i}].confidence_level is invalid.")
            results = intervention.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(f"Screen 3 interventions[{i}].results must be non-empty.")
            for j, result_item in enumerate(results):
                if not isinstance(result_item, dict) or set(result_item.keys()) != {"text", "outputs"}:
                    raise ValueError(f"Screen 3 results[{j}] fields are invalid.")
                if not isinstance(result_item.get("text"), str) or not result_item["text"].strip():
                    raise ValueError(f"Screen 3 results[{j}].text is required.")
                outputs = result_item.get("outputs")
                if not isinstance(outputs, list) or not outputs:
                    raise ValueError(f"Screen 3 results[{j}].outputs must be non-empty.")
                for k, output in enumerate(outputs):
                    if (
                        not isinstance(output, dict)
                        or set(output.keys()) != {"text"}
                        or not isinstance(output.get("text"), str)
                        or not output["text"].strip()
                    ):
                        raise ValueError(f"Screen 3 outputs[{k}] must contain text only.")

    def _regenerate_single_output(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Screen-3 single-output regeneration sub-flow.

        This is a rewrite of one output block, not a new council decision.  The
        Meta adapter is used to preserve the unified council style.  No new
        quantitative claim may be introduced beyond the supplied context.
        """
        input_obj = request.get("input") or {}
        existing_text = str(input_obj.get("existing_text") or "").strip()
        if not existing_text:
            raise ValueError("is_output_regeneration requires input.existing_text.")

        source_context = {
            "existing_text": existing_text,
            "impact_description": input_obj.get("impact_description"),
            "social_problem": input_obj.get("social_problem"),
            "organization": input_obj.get("organization"),
            "goal": input_obj.get("goal"),
            "impact_map": input_obj.get("impact_map"),
        }
        task = {
            "instruction": (
                "أعد صياغة مخرج واحد فقط لشاشة خريطة الأثر. لا تنشئ تدخلًا جديدًا، "
                "ولا تغير المقصود الاستراتيجي. أعد JSON صالحًا فقط بالمفتاح text. "
                "اجعل النص أوضح وأكثر تنفيذية، ولا تخترع رقمًا أو نسبة أو تاريخًا أو مدة. "
                "يجوز فقط الاحتفاظ برقم موجود أصلًا في السياق وبنفس الدلالة."
            ),
            "context": source_context,
            "required_schema": {"text": "string"},
        }

        def generate(task_obj: Dict[str, Any]) -> str:
            raw = self._generate(
                "meta" if COUNCIL_META_MODE == "adapter" else "base",
                self.meta_prompt,
                json.dumps(task_obj, ensure_ascii=False, indent=2),
                500,
                deterministic=True,
                repetition_penalty=1.08,
                no_repeat_ngram_size=8,
            )
            parsed = self.extract_json_object(raw)
            value = str(parsed.get("text") or "").strip()
            if not value:
                raise ValueError("Regenerated Screen 3 output is empty.")
            return value

        generated_text = generate(task)
        source_text = self._normalize_digits(
            json.dumps(source_context, ensure_ascii=False, sort_keys=True)
        )
        source_numbers = self._extract_number_tokens(source_text)
        unsupported = [
            n for n in self._extract_number_tokens(generated_text)
            if n not in source_numbers
        ]
        if unsupported:
            repair = dict(task)
            repair["instruction"] += (
                " المحاولة السابقة أضافت أرقامًا غير موجودة. أعد الصياغة بدون أي رقم جديد."
            )
            repair["unsupported_numbers"] = unsupported
            generated_text = generate(repair)
            unsupported = [
                n for n in self._extract_number_tokens(generated_text)
                if n not in source_numbers
            ]
            if unsupported:
                raise ValueError(
                    "Regeneration grounding failed; unsupported numbers: "
                    + ", ".join(unsupported)
                )

        raw_advisors = self._incoming_advisors(request)
        involved = []
        for advisor in raw_advisors:
            if isinstance(advisor, dict) and isinstance(advisor.get("id"), int):
                if advisor["id"] not in involved:
                    involved.append(advisor["id"])

        public = {
            "involved_advisor_ids": involved,
            "suggestion": {"text": generated_text},
        }
        if set(public.keys()) != {"involved_advisor_ids", "suggestion"}:
            raise ValueError("Invalid regeneration envelope.")
        return public

    def consult(self, request: Dict[str, Any]) -> Dict[str, Any]:
        input_obj = request.get("input") or {}
        debug = bool(request.get("debug") or input_obj.get("debug"))

        if bool(input_obj.get("is_output_regeneration")):
            total_start = time.perf_counter()
            public = self._regenerate_single_output(request)
            elapsed = round(time.perf_counter() - total_start, 3)
            if not debug:
                public["_athar_internal"] = {
                    "mode": "single_output_regeneration",
                    "timings_seconds": {"total": elapsed},
                }
                return public
            return {
                "debug": True,
                "mode": "single_output_regeneration",
                "timings_seconds": {"total": elapsed},
                "final_result": public,
            }

        total_start = time.perf_counter()
        selected = self._resolve_selected_advisors(request)

        opinions: List[Dict[str, Any]] = []
        advisor_timings: Dict[str, float] = {}
        for item in selected:
            advisor_id = item["advisor_id"]
            print(f"[council] Starting advisor {advisor_id}...", flush=True)
            started = time.perf_counter()
            opinion = self._run_advisor(item, request, selected)
            elapsed = time.perf_counter() - started
            advisor_timings[advisor_id] = round(elapsed, 3)
            opinions.append(opinion)
            print(f"[council] Finished advisor {advisor_id} in {elapsed:.2f}s", flush=True)

        print(
            f"[council] Starting AOS-META-00 synthesis for {len(opinions)} advisor(s)...",
            flush=True,
        )
        meta_started = time.perf_counter()
        rich_result = self._run_meta(request, opinions)
        meta_elapsed = time.perf_counter() - meta_started
        total_elapsed = time.perf_counter() - total_start
        print(
            f"[council] Finished AOS-META-00 in {meta_elapsed:.2f}s; total council time {total_elapsed:.2f}s",
            flush=True,
        )

        public_result = self._screen3_public_response(rich_result, selected)
        timings = {
            "advisors": advisor_timings,
            "meta": round(meta_elapsed, 3),
            "total": round(total_elapsed, 3),
        }

        if not debug:
            # Screen 3 receives exactly the contract it already materializes.
            # Rich Meta metadata (Attribution/Evidence/Interaction/12 sprints)
            # stays private until the backend contracts for those later screens arrive.
            public_result["_athar_internal"] = {
                "advisor_outputs": opinions,
                "rich_meta_result": rich_result,
                "meta_debug": getattr(self, "_last_meta_debug", {}),
                "timings_seconds": timings,
                "canonical_advisor_ids": [x["advisor_id"] for x in selected],
                "backend_advisor_ids": self._backend_advisor_ids(selected),
            }
            return public_result

        return {
            "debug": True,
            "selected_advisor_ids": [x["advisor_id"] for x in selected],
            "backend_advisor_ids": self._backend_advisor_ids(selected),
            "advisor_outputs": opinions,
            "meta_debug": getattr(self, "_last_meta_debug", {}),
            "timings_seconds": timings,
            "final_result": public_result,
            "rich_meta_result": rich_result,
        }

