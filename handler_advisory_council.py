from __future__ import annotations

import ast
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
META_MAX_NEW_TOKENS = max(int(os.getenv("META_MAX_NEW_TOKENS", "2200")), 1600)
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

    @staticmethod
    def _balanced_object_candidate(text: str) -> str:
        """Return the first balanced JSON-like object, ignoring braces inside strings."""
        start = text.find("{")
        if start < 0:
            return text
        depth = 0
        in_string = False
        quote = ""
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote:
                    in_string = False
                    quote = ""
                continue
            if ch in ('"', "'"):
                in_string = True
                quote = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]

    @staticmethod
    def _repair_common_json_syntax(candidate: str) -> str:
        """Conservatively repair common LLM JSON syntax mistakes.

        This does not alter semantic content. It only normalizes property quoting
        and trailing commas that strict JSON rejects.
        """
        repaired = str(candidate).strip()
        # Curly quotes are frequently emitted around property names. Normalize
        # them before quoting bare keys.
        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        # Quote single-quoted property names (not arbitrary values).
        repaired = re.sub(
            r"([\{,]\s*)'([^'\n]+)'\s*:",
            lambda m: m.group(1) + json.dumps(m.group(2), ensure_ascii=False) + ":",
            repaired,
        )
        # Quote bare ASCII property names such as title: or results:.
        repaired = re.sub(
            r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:',
            r'\1"\2":',
            repaired,
        )
        # Remove trailing commas before a closing object/array.
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    @classmethod
    def extract_json_object(cls, text: str) -> Dict[str, Any]:
        cleaned = cls.clean_model_text(text)
        cleaned = re.sub(r"^\s*```(?:json|javascript|js)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        candidate = cls._balanced_object_candidate(cleaned)

        errors: List[str] = []
        for label, payload in (
            ("strict", cleaned),
            ("balanced", candidate),
            ("common-repair", cls._repair_common_json_syntax(candidate)),
        ):
            try:
                value = json.loads(payload)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                errors.append(f"{label}: {exc}")

        # Optional JSON5 fallback when the runtime already provides it. This is
        # deliberately optional so deployment does not gain a new dependency.
        try:
            import json5  # type: ignore
            value = json5.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception as exc:
            errors.append(f"json5: {exc}")

        # Python-literal fallback covers single-quoted dict/list output. Only
        # literal structures are accepted; no code execution is possible.
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except Exception as exc:
            errors.append(f"literal: {exc}")

        excerpt = candidate[:1600].replace("\n", " ")
        raise ValueError(
            "Meta Advisor returned malformed JSON after parser repair attempts. "
            + " | ".join(errors[-4:])
            + f" | excerpt={excerpt!r}"
        )

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
        """Resolve one backend advisor to the authoritative Athar registry.

        The agreed backend identifier is now the canonical Athar ID itself
        (for example ``AOS-SP-13``).  Legacy integer IDs are still accepted only
        as an integration fallback, but every public response is canonical AOS-*.
        """
        incoming_id = incoming.get("id")
        model_id = (
            incoming.get("model_advisor_id")
            or incoming.get("advisor_id")
            or incoming.get("system_code")
            or incoming.get("code")
            or (incoming_id if isinstance(incoming_id, str) else None)
        )

        registry_entry = None
        if isinstance(model_id, str):
            registry_entry = self.registry_by_model_id.get(model_id.strip())

        # Transitional compatibility only. Do not expose these numeric IDs back
        # to the caller; the canonical AOS-* identifier is authoritative.
        if registry_entry is None and isinstance(incoming_id, int):
            registry_entry = self.registry_by_number.get(incoming_id)

        if registry_entry is None:
            name = self._norm(incoming.get("name") or incoming.get("advisor_name_ar"))
            registry_entry = self.registry_by_name.get(name)

        if registry_entry is None:
            raise ValueError(
                "Could not map Backend advisor to an Athar advisor. "
                f"Incoming advisor: {incoming}. Send id/model_advisor_id such as AOS-SP-08."
            )

        result = dict(registry_entry)
        result["backend_id"] = registry_entry["advisor_id"]
        # Preserve the incoming payload for audit/debug without letting it alter
        # the authoritative Expert DNA identity.
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
                or (x.get("id") if isinstance(x.get("id"), str) else None)
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
                    item["backend_id"] = item["advisor_id"]
                    if mapped is not None:
                        item["backend_payload"] = dict(mapped)
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
                    item["backend_id"] = item["advisor_id"]
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

        # Reject obvious multilingual corruption before it reaches the backend.
        # Normal English acronyms/terms are allowed, but Cyrillic/CJK leakage or
        # a single token mixing Arabic and Latin characters is not.
        raw = str(text or "")
        if re.search(r"[\u0400-\u052F\u4E00-\u9FFF\u3040-\u30FF]", raw):
            return True
        for token in re.findall(r"\S+", raw):
            if re.search(r"[\u0600-\u06FF]", token) and re.search(r"[A-Za-z]", token):
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
                "إذا كان topic هو interventions أو كانت هناك impact_map، فاجعل توصياتك تخدم قرار التدخل مباشرة: "
                "اربط رأيك بالمشكلة الاجتماعية ومحركات الأثر والبرامج القائمة. إذا كان اختصاصك تمكينيًا مثل التمويل أو MEAL أو القياس، "
                "فلا تجعل أداة التمكين بديلًا عن التدخل المستفيد-محور؛ وضّح كيف تدعم أو تتحقق من تدخل قائم/مقترح ضمن نطاقك. "
                "إذا كنت تملك المحافظ/البرامج، يجوز لك ترجيح أو تقوية أو ترتيب أولوية برنامج قائم، لكن لا تخترع تصميم خدمة تفصيليًا خارج الأدلة. "
                "اجعل الرد مركزًا وغير مكرر. "
                "اكتب العربية سليمة وواضحة، ولا تخلط أحرفًا لاتينية داخل كلمة عربية. "
                "يجوز استخدام المصطلحات/الاختصارات المهنية المعروفة مثل MEAL وKPI وSROI وContribution Analysis ككلمات مستقلة فقط."
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

    def _run_advisor_screen3_intervention(
        self,
        advisor: Dict[str, Any],
        request: Dict[str, Any],
        selected_council: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the Specialist normally, then project its opinion to Screen-3 JSON.

        The Specialist V2 adapter was trained to produce advisory prose, not a
        strict JSON object. Forcing the adapter itself to emit the backend schema
        caused valid advisory content to be rejected for formatting reasons.

        Current-stage contract therefore does two things only:
        1) obtain the advisor's independent first-pass opinion with Specialist V2;
        2) deterministically project that opinion into the existing Screen-3
           Interventions -> Results -> Outputs envelope.

        No Meta Advisor synthesis or second-model semantic rewrite is used here.
        """
        base_output = self._run_advisor(advisor, request, selected_council)
        opinion = str(base_output.get("opinion") or "").strip()
        if not opinion:
            raise ValueError(f"Specialist {advisor['advisor_id']} returned an empty opinion.")

        def clean_line(value: Any) -> str:
            line = str(value or "").strip()
            line = re.sub(r"^\s{0,4}#{1,6}\s*", "", line)
            line = re.sub(r"^\s*(?:[-*•]+|\d+[\.)]|[أ-ي][\.)])\s*", "", line)
            line = line.replace("**", "").replace("__", "").replace("`", "")
            line = re.sub(r"\s+", " ", line).strip(" :-–—\t")
            return line

        raw_lines = [x.rstrip() for x in opinion.splitlines()]
        lines = [clean_line(x) for x in raw_lines]

        # Collect recommendation/intervention lines first. The advisor prose may
        # use several equivalent Arabic section labels, so parsing is tolerant.
        section_active = False
        recommendation_lines: List[str] = []
        diagnosis_lines: List[str] = []
        for raw, line in zip(raw_lines, lines):
            if not line:
                continue
            lower = line.lower()
            is_heading = bool(re.match(r"^\s*#{1,6}\s+", raw))
            if any(key in lower for key in (
                "التوصيات", "توصيات", "التدخلات", "التدخلات المطلوبة",
                "الأولويات", "الخطوات المقترحة", "الإجراءات المقترحة",
            )) and (is_heading or len(line) <= 90):
                section_active = True
                continue
            if section_active and any(key in lower for key in (
                "المخاطر", "الافتراضات", "الافتراض", "نقاط التحقق",
                "فجوات البيانات", "الإحالات", "الخلاصة", "الأدلة",
            )) and (is_heading or len(line) <= 90):
                section_active = False
                continue

            # Prefer numbered/top-level bullets as discrete recommendations.
            is_top_item = bool(re.match(r"^\s*(?:\d+[\.)]|[-*•])\s+", raw))
            if section_active and is_top_item and 12 <= len(line) <= 420:
                recommendation_lines.append(line)
            elif not section_active and len(diagnosis_lines) < 5 and 30 <= len(line) <= 520:
                diagnosis_lines.append(line)

        # Fallback: use substantive numbered/bulleted lines from anywhere.
        if not recommendation_lines:
            for raw, line in zip(raw_lines, lines):
                if (
                    bool(re.match(r"^\s*(?:\d+[\.)]|[-*•])\s+", raw))
                    and 12 <= len(line) <= 420
                ):
                    recommendation_lines.append(line)
                if len(recommendation_lines) >= 3:
                    break

        # Last fallback: use the strongest substantive sentences from the prose.
        if not recommendation_lines:
            compact = re.sub(r"\s+", " ", opinion)
            for sentence in re.split(r"(?<=[.!؟])\s+", compact):
                sentence = clean_line(sentence)
                if 25 <= len(sentence) <= 420:
                    recommendation_lines.append(sentence)
                if len(recommendation_lines) >= 2:
                    break

        recommendation_lines = list(dict.fromkeys(recommendation_lines))

        # Lightweight scope guard for the currently used Screen-3 specialists.
        # The full Expert DNA remains authoritative; this filter only prevents a
        # neighboring domain recommendation from being projected publicly.
        scope_keywords = {
            "AOS-SP-13": ("برنامج", "مشروع", "محفظ", "أولو", "جدوى", "اعتماد", "موارد", "توسع", "منافع"),
            "AOS-SP-15": ("أثر", "تقييم", "متابعة", "تعلم", "meal", "مؤشر", "نظرية", "مساهمة", "نتائج"),
            "AOS-FG-18": ("تمويل", "دخل", "مانح", "موارد", "استدام", "إيراد", "تبرع", "شراك"),
        }
        keys = scope_keywords.get(advisor["advisor_id"])
        if keys:
            scoped = [
                x for x in recommendation_lines
                if any(k in x.lower() for k in keys)
            ]
            if scoped:
                recommendation_lines = scoped
        recommendation_lines = recommendation_lines[:2]
        if not recommendation_lines:
            recommendation_lines = [
                "تطبيق التوصية الأساسية للمستشار ضمن نطاق اختصاصه وبالاستناد إلى بيانات الحالة المتاحة"
            ]

        # Build a concise title from the first owned recommendation. This avoids
        # asking the Specialist to learn a second output grammar just for the API.
        first_rec = recommendation_lines[0]
        title = first_rec.split(":", 1)[0].strip()
        if len(title) < 10 or len(title) > 150:
            title = f"توصية {advisor['advisor_name_ar']} للحالة الحالية"

        # Build the public impact description from authoritative case context,
        # not from free-form diagnosis prose. This prevents an otherwise useful
        # specialist answer from leaking a typo or an unsupported contextual claim
        # into the backend contract. The actual recommendations below still come
        # from the independent Specialist opinion.
        input_obj = request.get("input") or {}
        track = input_obj.get("track") or {}
        goal_obj = input_obj.get("goal") or {}
        org_obj = input_obj.get("organization") or {}
        goal_statement = str(goal_obj.get("statement") or "").strip() if isinstance(goal_obj, dict) else str(goal_obj or "").strip()
        important_notes = str(org_obj.get("important_notes") or "").strip() if isinstance(org_obj, dict) else ""

        impact_templates = {
            "AOS-SP-13": "يركز رأي المستشار على ترتيب أولوية البرامج والمشاريع القائمة وربط قرارات التوسع بالقيمة الاستراتيجية والجدوى والموارد المتاحة.",
            "AOS-SP-15": "يركز رأي المستشار على بناء منظومة متابعة وتقييم وتعلم وقياس أثر تربط البرامج القائمة بنتائج المستفيدين والهدف الاستراتيجي المعتمد.",
            "AOS-FG-18": "يركز رأي المستشار على تنويع مصادر الدخل وتقليل الاعتماد على التمويل الموسمي بما يدعم استدامة البرامج والخدمات.",
        }
        impact_description = impact_templates.get(
            advisor["advisor_id"],
            f"يركز رأي {advisor['advisor_name_ar']} على تطبيق توصية داخل نطاق اختصاصه بما يخدم الهدف المعتمد وبيانات الحالة المتاحة.",
        )
        if advisor["advisor_id"] in {"AOS-SP-13", "AOS-SP-15"} and goal_statement:
            impact_description += f" الهدف المعتمد: {goal_statement}"
        elif advisor["advisor_id"] == "AOS-FG-18" and important_notes:
            impact_description += f" ويستند إلى الملاحظة المؤسسية: {important_notes}"
        impact_description = impact_description[:900].strip()

        primary_indicator = ""
        if isinstance(track, dict):
            primary_indicator = str(track.get("primary_indicator") or "").strip()

        indicator_by_advisor = {
            "AOS-SP-13": "مؤشر أولوية وجدوى البرامج والمشاريع",
            "AOS-SP-15": "مؤشر نتائج وأثر البرامج المستهدفة",
            "AOS-FG-18": "مؤشر تنوع واستدامة مصادر التمويل",
            "AOS-LD-04": "مؤشر قيمة وجودة الشراكات",
            "AOS-SP-12": "مؤشر تقدم التنفيذ التشغيلي",
            "AOS-SP-14": "مؤشر أداء مرتبط بالهدف المعتمد",
        }
        reportable_value = indicator_by_advisor.get(
            advisor["advisor_id"],
            primary_indicator or "مؤشر متابعة مرتبط بنطاق التوصية",
        )

        # Confidence is conservative by default in this no-Meta stage. If the
        # Specialist explicitly states a level, preserve it.
        confidence = "متوسطة"
        normalized_opinion = re.sub(r"\s+", " ", opinion)
        if re.search(r"(?:ثقة|الثقة)\s*(?:مرتفعة|عالية|مرتفع|عالي)", normalized_opinion):
            confidence = "مرتفعة"
        elif re.search(r"(?:ثقة|الثقة)\s*(?:منخفضة|ضعيفة|منخفض|ضعيف)", normalized_opinion):
            confidence = "منخفضة"

        results: List[Dict[str, Any]] = []
        for rec in recommendation_lines[:2]:
            rec = rec[:420].strip()
            if not rec:
                continue
            # Use the advisor-owned recommendation as both the expected result
            # and its directly traceable output. This is intentionally simple and
            # lossless; later backend contracts can split richer fields.
            output_text = rec
            if ":" in rec:
                left, right = [x.strip() for x in rec.split(":", 1)]
                if len(right) >= 12:
                    rec = left if len(left) >= 10 else rec
                    output_text = right
            results.append({
                "text": rec,
                "outputs": [{"text": output_text}],
            })

        if not results:
            results = [{
                "text": "اعتماد رأي المستشار ضمن نطاق اختصاصه",
                "outputs": [{"text": first_rec[:420]}],
            }]

        intervention = {
            "title": title,
            "confidence_level": confidence,
            "impact_description": impact_description,
            "reportable_value": reportable_value,
            "results": results,
        }

        # Sanitize only unsupported numerical tokens in the projected public
        # fields. The raw independent opinion is preserved internally unchanged.
        source_text = self._normalize_digits(
            json.dumps(self._shared_context(request), ensure_ascii=False, sort_keys=True)
        )
        source_numbers = self._extract_number_tokens(source_text)

        COMMON_TEXT_FIXES = {
            "التناوي": "التنموي",
            "التفاعلي": "التنموي",
            "التقيم": "التقييم",
            "مستدمة": "مستدامة",
            "كبرية": "كبرى",
            "التمويل الموسمية": "التمويل الموسمي",
            "الخطوط النقلية": "خطوط النقل",
            "المدرسية": "المدرسية",
            "الت_dropout": "التسرب",
            "تموilen": "تمويل",
        }

        def scrub_string(value: str) -> str:
            value = str(value or "")
            for bad, good in COMMON_TEXT_FIXES.items():
                value = value.replace(bad, good)
            value = re.sub(r"[\u0400-\u052F\u4E00-\u9FFF\u3040-\u30FF]", "", value)

            # Remove mixed Arabic/Latin corruption token-by-token while allowing
            # professional Latin acronyms/terms as separate tokens.
            kept_tokens = []
            for token in value.split():
                if re.search(r"[\u0600-\u06FF]", token) and re.search(r"[A-Za-z]", token):
                    arabic = re.sub(r"[A-Za-z]+", "", token)
                    token = arabic if len(re.sub(r"[^\u0600-\u06FF]", "", arabic)) >= 3 else ""
                if token:
                    kept_tokens.append(token)
            value = " ".join(kept_tokens)

            def repl(match: re.Match) -> str:
                token = self._normalize_digits(match.group(0))
                return match.group(0) if token in source_numbers else ""

            value = re.sub(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])", repl, value)
            # Preserve a percentage sign only when it still follows a grounded number.
            value = re.sub(r"(?<!\d)\s*(?:%|٪)\s*", " ", value)
            value = re.sub(r"\s*([%٪])\s*", r"\1 ", value)
            value = re.sub(r"\s+", " ", value).strip(" -–—,:؛")
            return value

        intervention["title"] = scrub_string(intervention["title"]) or f"توصية {advisor['advisor_name_ar']}"
        intervention["impact_description"] = scrub_string(intervention["impact_description"]) or "تطبيق رأي المستشار على الحالة ضمن حدود البيانات المتاحة."
        intervention["reportable_value"] = scrub_string(intervention["reportable_value"]) or "مؤشر متابعة مرتبط بنطاق التوصية"
        for result in intervention["results"]:
            result["text"] = scrub_string(result["text"]) or "نتيجة متوقعة من تطبيق التوصية"
            for output in result["outputs"]:
                output["text"] = scrub_string(output["text"]) or "مخرج تنفيذي مرتبط بالتوصية"

        # Final structural validation. At this stage formatting can no longer
        # fail because the JSON object is constructed by Python, not generated by
        # the model.
        self._validate_screen3_public_response({
            "involved_advisor_ids": [advisor["advisor_id"]],
            "suggestion": {"interventions": [intervention]},
        })

        return {
            "advisor_id": advisor["advisor_id"],
            "backend_id": advisor.get("backend_id") or advisor["advisor_id"],
            "advisor_name_ar": advisor["advisor_name_ar"],
            "opinion": opinion,
            "intervention": intervention,
            "projection_mode": "deterministic_from_specialist_opinion",
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

    def _advisor_refs_to_ids(self, value: Any, selected_ids: List[str]) -> List[str]:
        """Map compact 1-based advisor references to canonical selected AOS IDs.

        The Meta model never needs to reproduce long advisor IDs. It only emits
        refs like [1, 3], which are mapped deterministically to the council that
        the backend already selected. This removes a common hallucination source.
        """
        if not isinstance(value, list):
            value = [value] if value is not None else []
        out: List[str] = []
        for raw in value:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(selected_ids):
                aid = selected_ids[idx - 1]
                if aid not in out:
                    out.append(aid)
        return out

    @staticmethod
    def _has_foreign_script(text: Any) -> bool:
        raw = str(text or "")
        # Arabic + ordinary Latin acronyms are fine. Cyrillic/CJK leakage is not.
        return bool(re.search(r"[\u0400-\u052F\u4E00-\u9FFF\u3040-\u30FF]", raw))

    def _build_private_sprints(
        self,
        interventions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a private 12-week structural allocation without another LLM call.

        Screen 3 does not expose sprints yet. We therefore keep a deterministic,
        grounded weekly allocation in private run metadata and avoid forcing the
        Meta model to generate a second long JSON structure that can truncate.
        """
        if not interventions:
            return []

        phase_labels = (
            "تهيئة التنفيذ",
            "تجهيز التنفيذ",
            "بدء التنفيذ",
            "استكمال التنفيذ",
            "متابعة التنفيذ",
            "متابعة النتائج",
            "تحقق مرحلي",
            "تحسين التنفيذ",
            "استكمال التحسين",
            "مراجعة النتائج",
            "تثبيت التعلم",
            "إقفال الدورة وتوثيق الخطوة التالية",
        )
        sprints: List[Dict[str, Any]] = []
        n = len(interventions)
        for week in range(1, SPRINT_COUNT + 1):
            source = interventions[(week - 1) % n]
            iid = str(source.get("intervention_id") or f"INT-{((week - 1) % n) + 1:02d}")
            title = str(source.get("title") or "التدخل").strip()
            result_texts: List[str] = []
            output_texts: List[str] = []
            for item in source.get("results") or []:
                if not isinstance(item, dict):
                    continue
                txt = str(item.get("text") or "").strip()
                if txt:
                    result_texts.append(txt)
                for output in item.get("outputs") or []:
                    if isinstance(output, dict):
                        ot = str(output.get("text") or "").strip()
                        if ot:
                            output_texts.append(ot)

            base_action = result_texts[0] if result_texts else f"متابعة تنفيذ {title}"
            base_output = output_texts[0] if output_texts else title
            sprints.append({
                "sprint_number": week,
                "week_number": week,
                "title": f"{phase_labels[week - 1]} — {title}",
                "objective": f"تقدم مرحلي في «{title}» ضمن حدود التوصية المعتمدة.",
                "actions": [base_action],
                "outputs": [base_output],
                "source_intervention_ids": [iid],
                "attribution": list(source.get("attribution") or []),
                "evidence_classification": str(source.get("evidence_classification") or "I2"),
                "evidence_basis": str(source.get("evidence_basis") or "بيانات الحالة ورأي المجلس المختار."),
                "interaction_type": str(source.get("interaction_type") or "COMPLEMENTARY"),
                "confidence_level": str(source.get("confidence_level") or "Medium"),
                "status": OPEN_STATUS,
            })
        return sprints

    def _normalize_meta_result(
        self,
        result: Dict[str, Any],
        selected_ids: List[str],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise ValueError("Meta result must be an object.")
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

        raw_interventions = suggestion.get("interventions")
        if not isinstance(raw_interventions, list):
            raw_interventions = []

        interventions: List[Dict[str, Any]] = []
        for raw in raw_interventions[:4]:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            impact = str(raw.get("impact_description") or "").strip()
            reportable = str(raw.get("reportable_value") or "").strip()
            if not title or not impact or not reportable:
                continue

            attrs = self._advisor_refs_to_ids(
                raw.get("advisor_refs") or raw.get("advisor_ref"), selected_ids
            )
            # Backward compatibility if the model still emits canonical IDs.
            if not attrs:
                attrs = self._normalize_attribution(
                    raw.get("attribution") or raw.get("advisor_ids"), selected_ids
                )
            # Fail-safe provenance: never hallucinate a non-selected advisor.
            if not attrs:
                attrs = list(selected_ids)

            evidence = self._normalize_evidence_code(
                raw.get("evidence_classification") or raw.get("evidence_class")
            ) or "I2"
            interaction = self._normalize_interaction(
                raw.get("interaction_type") or raw.get("discussion_type")
            ) or ("CONSENSUS" if len(attrs) == 1 else "COMPLEMENTARY")
            confidence = self._normalize_confidence(
                raw.get("confidence_level") or raw.get("confidence")
            ) or "Medium"
            if evidence in {"U", "A1"}:
                confidence = "Low"
            elif evidence == "I2" and confidence == "High":
                confidence = "Medium"
            if interaction in {"CONFLICT", "EVIDENCE GAP", "SCOPE CONFLICT"} and confidence == "High":
                confidence = "Medium"

            results: List[Dict[str, Any]] = []
            for item in raw.get("results") or []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                outputs: List[Dict[str, str]] = []
                for output in item.get("outputs") or []:
                    if isinstance(output, dict):
                        ot = str(output.get("text") or "").strip()
                    else:
                        ot = str(output or "").strip()
                    if ot:
                        outputs.append({"text": ot})
                if text and outputs:
                    results.append({
                        "text": text,
                        "outputs": outputs[:3],
                        "status": OPEN_STATUS,
                        "attribution": list(attrs),
                        "evidence_classification": evidence,
                        "interaction_type": interaction,
                        "confidence_level": confidence,
                    })
            if not results:
                continue

            interventions.append({
                "intervention_id": f"INT-{len(interventions) + 1:02d}",
                "title": title,
                "impact_description": impact,
                "reportable_value": reportable,
                "attribution": attrs,
                "evidence_classification": evidence,
                "evidence_basis": str(raw.get("evidence_basis") or "بيانات الحالة ورأي المجلس المختار ضمن نطاق المستشارين.").strip(),
                "interaction_type": interaction,
                "confidence_level": confidence,
                "status": OPEN_STATUS,
                "results": results,
            })

        suggestion["interventions"] = interventions
        suggestion["sprints"] = self._build_private_sprints(interventions)
        suggestion["sprint_count"] = SPRINT_COUNT

        if not str(result.get("recommendation_text") or "").strip():
            titles = [x["title"] for x in interventions]
            result["recommendation_text"] = (
                "يوصي المجلس بالتركيز على: " + "؛ ".join(titles)
                if titles else "لا توجد توصية مكتملة بسبب نقص مخرجات قابلة للاعتماد."
            )

        item_interactions = [str(x.get("interaction_type") or "") for x in interventions]
        result["council_interaction_type"] = self._derive_council_interaction(item_interactions)
        result["overall_confidence"] = self._derive_overall_confidence(interventions)

        attribution_summary = []
        for aid in selected_ids:
            titles = [
                str(x.get("title") or "").strip()
                for x in interventions
                if aid in x.get("attribution", []) and str(x.get("title") or "").strip()
            ]
            if titles:
                attribution_summary.append({
                    "advisor_id": aid,
                    "contribution": "؛ ".join(list(dict.fromkeys(titles))[:4]),
                })
        # If the compact model was conservative and attributed everything to a
        # subset, still preserve the complete selected council in involved IDs;
        # attribution itself remains item-based.
        result["attribution"] = attribution_summary or [
            {"advisor_id": aid, "contribution": "مساهمة ضمن المجلس المختار."}
            for aid in selected_ids
        ]
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

    @staticmethod
    def _impact_tokens(value: Any) -> set[str]:
        text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF]+", " ", str(value or "").lower())
        stop = {
            "الجمعية", "جمعية", "البرنامج", "برنامج", "المشروع", "مشروع", "البرامج", "المشاريع",
            "الحالي", "الحالية", "الحالية", "المستهدف", "المستهدفة", "الهدف", "الأهداف", "اهداف",
            "تحليل", "تحديد", "تصميم", "تطوير", "تحسين", "تعزيز", "خطة", "استراتيجية", "استراتيجي",
            "استراتيجية", "قياس", "مؤشر", "مؤشرات", "نتائج", "مخرجات", "الأثر", "اثر", "التدخل",
            "التدخلات", "على", "من", "في", "إلى", "الى", "عن", "مع", "لدى", "ضمن", "بين", "كل",
            "هذه", "هذا", "ذلك", "التي", "الذي", "و", "أو", "او", "ثم", "بما", "بناء", "مدى",
        }
        return {tok for tok in text.split() if len(tok) >= 3 and tok not in stop}

    def _screen3_impact_quality_violations(
        self,
        result: Dict[str, Any],
        request: Dict[str, Any],
    ) -> List[str]:
        """Fail closed when Screen-3 drifts into support work instead of interventions.

        Direct-impact proposals must visibly anchor to the social problem, an impact
        driver, target group, or an existing program. Analysis/MEAL/finance/resource
        management may support those proposals, but normally cannot dominate them.
        """
        input_obj = request.get("input") or {}
        topic = str(request.get("topic") or input_obj.get("topic") or "").strip().lower()
        impact_map = input_obj.get("impact_map") or {}
        if topic != "interventions" and not isinstance(impact_map, dict):
            return []
        if not isinstance(impact_map, dict):
            impact_map = {}

        goal = input_obj.get("goal") or {}
        if not isinstance(goal, dict):
            goal = {"statement": goal}
        programs = input_obj.get("programs") or []
        if not isinstance(programs, list):
            programs = []

        social_problem = str(impact_map.get("social_problem") or goal.get("social_problem") or "")
        impact_drivers = str(impact_map.get("impact_drivers") or "")
        target_group = str(goal.get("target_group") or "")
        goal_statement = str(goal.get("statement") or "")

        program_names: List[str] = []
        program_blobs: List[str] = []
        for program in programs:
            if not isinstance(program, dict):
                continue
            name = str(program.get("name") or "").strip()
            if name:
                program_names.append(name)
            program_blobs.extend([
                name,
                str(program.get("description") or ""),
                str(program.get("target_audience") or ""),
                str(program.get("beneficiary_value") or ""),
            ])

        anchor_blob = " ".join([
            social_problem,
            impact_drivers,
            target_group,
            goal_statement,
            str(impact_map.get("association_scope") or ""),
            *program_blobs,
        ])
        anchors = self._impact_tokens(anchor_blob)
        driver_tokens = self._impact_tokens(impact_drivers)
        program_tokens = self._impact_tokens(" ".join(program_names))
        if len(anchors) < 2:
            return []

        suggestion = result.get("suggestion") or {}
        interventions = suggestion.get("interventions") or []
        if not isinstance(interventions, list) or not interventions:
            return []

        # A title led by these concepts is a support/enabler proposal, not a
        # beneficiary/program-facing intervention for a social-impact Screen 3.
        support_terms = (
            "تحليل", "تقييم", "مراجعة", "قياس", "منهجية", "إطار", "مؤشر", "مؤشرات",
            "تمويل", "مالي", "مالية", "موارد", "مانح", "مانحين", "استدامة مالية", "حوكمة",
            "محفظة", "أولويات", "جدوى", "لوحة", "بيانات", "متابعة", "تعلم", "استراتيجية مانحين",
            "استراتيجية موارد", "إدارة الموارد", "الشراكات المؤسسية",
        )
        direct_action_terms = (
            "توسيع", "تعزيز", "تطوير", "تحسين", "تشغيل", "تقديم", "تمكين", "دعم", "حماية",
            "نقل", "مساندة", "خدمة", "برنامج", "مشروع", "مبادرة", "حافلات", "حقيبة", "تعليمي",
            "مدرسي", "الطلاب", "الطالب", "الأسر", "المستفيد", "المستفيدين",
        )
        finance_terms = (
            "تمويل", "مالي", "مالية", "موارد", "دخل", "مانح", "مانحين", "تبرعات",
            "استدامة مالية", "شراكات مؤسسية", "استراتيجية مانحين", "استراتيجية موارد",
        )
        measurement_terms = (
            "قياس", "تقييم", "مؤشرات", "متابعة", "تعلم", "meal", "منهجية قياس", "منظومة قياس",
        )

        direct_count = 0
        support_count = 0
        finance_count = 0
        measurement_count = 0
        program_linked_direct = 0
        driver_linked_direct = 0

        for intervention in interventions:
            if not isinstance(intervention, dict):
                continue
            title = str(intervention.get("title") or "").strip().lower()
            blob = " ".join(self._collect_strings({
                "title": intervention.get("title"),
                "impact_description": intervention.get("impact_description"),
                "results": intervention.get("results"),
            })).lower()
            blob_tokens = self._impact_tokens(blob)
            overlap = blob_tokens & anchors
            program_overlap = blob_tokens & program_tokens
            driver_overlap = blob_tokens & driver_tokens
            support_title = any(term in title for term in support_terms)
            action_signal = any(term in blob for term in direct_action_terms)

            # Direct means: case-anchored + action/program facing + not titled as
            # a support function. Generic references to "impact" alone are not enough.
            direct = (
                not support_title
                and action_signal
                and len(overlap) >= 2
                and (bool(program_overlap) or bool(driver_overlap) or bool(target_group and self._impact_tokens(target_group) & blob_tokens))
            )
            if direct:
                direct_count += 1
                if program_overlap:
                    program_linked_direct += 1
                if driver_overlap:
                    driver_linked_direct += 1
            else:
                support_count += 1

            if any(term in title for term in finance_terms):
                finance_count += 1
            if any(term in title for term in measurement_terms):
                measurement_count += 1

        violations: List[str] = []
        n = len(interventions)
        if n >= 3 and direct_count < 2:
            violations.append(
                "Screen 3 requires at least two direct beneficiary/program-facing interventions when returning 3-4 proposals; support/enabler proposals currently dominate."
            )
        elif n == 2 and direct_count < 1:
            violations.append(
                "Screen 3 requires at least one direct beneficiary/program-facing intervention when returning two proposals."
            )
        elif n == 1 and direct_count < 1:
            violations.append(
                "The single Screen-3 proposal must be a direct beneficiary/program-facing intervention, not only analysis/measurement/finance/resource support."
            )

        track_blob = json.dumps(input_obj.get("track") or {}, ensure_ascii=False).lower()
        goal_blob = json.dumps(goal, ensure_ascii=False).lower()
        enabling_track = any(term in (track_blob + " " + goal_blob) for term in (
            "مالي", "تمويل", "استدامة مالية", "self-funding", "funding", "حوكمة", "تحول رقمي"
        ))
        if n >= 3 and support_count > 1 and not enabling_track:
            violations.append(
                "At most one standalone support/enabler intervention is allowed in a 3-4 proposal social-impact Screen 3; embed the other support work under direct interventions."
            )
        if finance_count > 1 and not enabling_track:
            violations.append(
                "Duplicate financial/resource enablers detected; consolidate them into one supporting intervention or embed them under direct interventions."
            )
        if measurement_count > 1:
            violations.append(
                "Duplicate MEAL/measurement enablers detected; consolidate them unless they represent materially different evidence decisions."
            )

        # When the case provides concrete programs and impact drivers, a social-
        # impact result should visibly use at least one of those anchors.
        if program_names and program_linked_direct == 0 and not enabling_track:
            violations.append(
                "No direct intervention is visibly anchored to an existing program even though programs are provided; strengthen/prioritize a supported existing program instead of returning only internal analysis."
            )
        if driver_tokens and driver_linked_direct == 0 and not enabling_track:
            violations.append(
                "No direct intervention visibly addresses a named impact driver; use the impact drivers as intervention anchors."
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
        """Compact two-attempt Meta synthesis for Screen 3.

        The public Screen-3 contract only needs interventions/results/outputs.
        Asking the model to also emit 12 full sprints and long canonical IDs made
        outputs unnecessarily large and was the main source of truncation and
        malformed JSON.  v4.2 asks for a compact semantic draft; code then adds
        canonical IDs, private metadata and the 12-week structural allocation.
        """
        self._last_meta_debug = {
            "compact_meta_contract": True,
            "schema_retry_used": False,
            "quality_retry_used": False,
            "initial_error": None,
            "initial_grounding_violations": [],
            "initial_impact_quality_violations": [],
            "initial_language_violations": [],
            "final_grounding_violations": [],
            "final_impact_quality_violations": [],
            "final_language_violations": [],
        }
        selected_ids = [str(x["advisor_id"]) for x in advisor_outputs]
        scope_cards = self._build_scope_cards(advisor_outputs)
        advisor_ref_map = {
            str(i + 1): {
                "advisor_id": aid,
                "advisor_name_ar": str(advisor_outputs[i].get("advisor_name_ar") or ""),
            }
            for i, aid in enumerate(selected_ids)
        }

        input_obj = request.get("input") or {}
        imap = input_obj.get("impact_map") or {}
        goal = input_obj.get("goal") or {}
        programs = input_obj.get("programs") or []

        meta_task = {
            "instruction": (
                "طبّق AOS-META-00 كمرحلة تركيب نهائي للمجلس. أعد JSON صغيرًا وصالحًا فقط. "
                "لا تكتب 12 Sprint في إجابتك؛ النظام سيقسم التدخلات إلى 12 أسبوعًا بعد التركيب. "
                "لا تكتب أكواد AOS داخل JSON. استخدم advisor_refs فقط كأرقام 1..N وفق advisor_ref_map. "
                "لا تستخدم أي مستشار غير موجود في advisor_ref_map. "
                "هذه شاشة Impact Map وtopic=interventions: المطلوب تدخلات أثر/برامج فعلية، لا قائمة أعمال دعم داخلية. "
                "إذا أعدت 3 أو 4 تدخلات فيجب أن يكون اثنان منها على الأقل مباشرين للمستفيد/البرنامج ويرتبطان صراحة "
                "بالمشكلة الاجتماعية أو impact_drivers أو برنامج قائم. اسمح بحد أقصى بتدخل تمكيني مستقل واحد (MEAL/تمويل/موارد/حوكمة). "
                "يمكن دمج القياس والتمويل كـresults/outputs أو شروط دعم تحت تدخل مباشر. "
                "لا تبدأ عناوين التدخلات المباشرة بتحليل/تقييم/قياس/إطار/منهجية/تمويل/موارد. "
                "استخدم البرامج القائمة ومحركات الأثر كمرساة عندما تدعمها آراء المستشارين ضمن نطاقهم. "
                "لا تخترع خدمة أو شراكة أو موردًا غير مدعوم. لا تخترع أي رقم أو نسبة أو مبلغ أو مدة أو تاريخ. "
                "يجوز إعادة استخدام رقم موجود صراحة في case_context وبنفس الدلالة فقط. "
                "اكتب النصوص بالعربية السليمة؛ يسمح فقط بالمصطلحات/الاختصارات الإنجليزية المعتادة مثل MEAL عند الحاجة. "
                "لا تستخدم أي أحرف صينية أو يابانية أو كيريلية."
            ),
            "case_context": self._shared_context(request),
            "advisor_ref_map": advisor_ref_map,
            "advisor_scope_cards": scope_cards,
            "selected_advisor_outputs": advisor_outputs,
            "direct_anchors": {
                "social_problem": str(imap.get("social_problem") or (goal.get("social_problem") if isinstance(goal, dict) else "") or ""),
                "impact_drivers": str(imap.get("impact_drivers") or ""),
                "target_group": str(goal.get("target_group") if isinstance(goal, dict) else ""),
                "existing_program_names": [
                    str(x.get("name") or "") for x in programs
                    if isinstance(x, dict) and str(x.get("name") or "").strip()
                ],
            },
            "evidence_codes": ["E1", "E2", "E3", "I1", "I2", "A1", "U"],
            "interaction_types": [
                "CONSENSUS", "COMPLEMENTARY", "TRADE-OFF", "CONFLICT",
                "EVIDENCE GAP", "SCOPE CONFLICT",
            ],
            "confidence_levels": ["High", "Medium", "Low"],
            "required_schema": {
                "recommendation_text": "string",
                "suggestion": {
                    "interventions": [
                        {
                            "title": "string",
                            "impact_description": "string",
                            "reportable_value": "string",
                            "advisor_refs": ["1-based integers from advisor_ref_map only"],
                            "evidence_classification": "one evidence code",
                            "evidence_basis": "short string",
                            "interaction_type": "one interaction type",
                            "confidence_level": "High|Medium|Low",
                            "results": [
                                {"text": "string", "outputs": [{"text": "string"}]}
                            ],
                        }
                    ]
                },
            },
            "hard_rules": [
                "Valid JSON only; double quotes only; no Markdown.",
                "Return 2 to 4 interventions for this Screen-3 case when evidence supports them.",
                "With 3-4 interventions, at least 2 are direct beneficiary/program interventions and at most 1 is a standalone enabler.",
                "Each intervention has 1-2 results; each result has 1-2 outputs.",
                "advisor_refs may contain only integers shown in advisor_ref_map.",
                "No AOS/ATHAR advisor codes in prose or outputs.",
                "No invented quantitative claims, budgets, durations, deadlines, or targets.",
            ],
        }

        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"

        def generate_once(task: Dict[str, Any]) -> Dict[str, Any]:
            raw = self._generate(
                meta_adapter,
                self.meta_prompt,
                json.dumps(task, ensure_ascii=False, indent=2),
                META_MAX_NEW_TOKENS,
                deterministic=True,
                repetition_penalty=1.10,
                no_repeat_ngram_size=8,
            )
            parsed = self.extract_json_object(raw)
            normalized = self._normalize_meta_result(parsed, selected_ids, request)
            self._validate_backend_result(normalized, selected_ids)
            return normalized

        def language_violations(result: Dict[str, Any]) -> List[str]:
            violations: List[str] = []
            suggestion = result.get("suggestion") or {}
            for i, intervention in enumerate(suggestion.get("interventions") or []):
                if not isinstance(intervention, dict):
                    continue
                scan_values = {
                    "title": intervention.get("title"),
                    "impact_description": intervention.get("impact_description"),
                    "reportable_value": intervention.get("reportable_value"),
                    "evidence_basis": intervention.get("evidence_basis"),
                    "results": intervention.get("results"),
                }
                for text in self._collect_strings(scan_values):
                    if self._has_foreign_script(text):
                        violations.append(f"interventions[{i}] contains foreign-script leakage")
                        break
                    if re.search(r"\b(?:ATHAR|AOS)-(?:LD|SP|FG|SE)?-?\d+\b", text, flags=re.I):
                        violations.append(f"interventions[{i}] exposes advisor code in user-facing text")
                        break
            return list(dict.fromkeys(violations))

        initial_error: Optional[Exception] = None
        result: Optional[Dict[str, Any]] = None
        try:
            result = generate_once(meta_task)
        except Exception as exc:
            initial_error = exc
            self._last_meta_debug["schema_retry_used"] = True
            self._last_meta_debug["initial_error"] = str(exc)[:1200]

        if result is not None:
            grounding = self._grounding_violations(result, request)
            impact = self._screen3_impact_quality_violations(result, request)
            language = language_violations(result)
            self._last_meta_debug["initial_grounding_violations"] = list(grounding)
            self._last_meta_debug["initial_impact_quality_violations"] = list(impact)
            self._last_meta_debug["initial_language_violations"] = list(language)
        else:
            grounding, impact, language = [], [], []

        needs_retry = result is None or bool(grounding or impact or language)
        if needs_retry:
            self._last_meta_debug["quality_retry_used"] = True
            retry = dict(meta_task)
            retry["instruction"] = (
                meta_task["instruction"]
                + " هذه محاولة القبول النهائية. اكتب JSON أقصر وبنية أبسط. "
                  "اجعل التدخلين الأولين على الأقل مباشرين للمستفيد/البرنامج ومربوطين باسم برنامج قائم أو impact driver عندما تدعم الآراء ذلك. "
                  "ادمج القياس والتمويل كدعم ولا تسمح لهما بالسيطرة على القائمة. "
                  "راجع كل رقم واحذف أي رقم غير موجود حرفيًا في case_context. "
                  "استخدم advisor_refs فقط ولا تكتب أي ID نصي للمستشار."
            )
            retry["previous_errors"] = {
                "schema": str(initial_error)[:900] if initial_error else None,
                "grounding": grounding[:8],
                "impact_quality": impact[:8],
                "language": language[:8],
            }
            # Do not include the malformed/raw previous JSON; that tends to
            # anchor the model on the exact syntax/ID mistakes we are repairing.
            result = generate_once(retry)

        grounding = self._grounding_violations(result, request)
        impact = self._screen3_impact_quality_violations(result, request)
        language = language_violations(result)
        self._last_meta_debug["final_grounding_violations"] = list(grounding)
        self._last_meta_debug["final_impact_quality_violations"] = list(impact)
        self._last_meta_debug["final_language_violations"] = list(language)

        if grounding:
            raise ValueError("Council grounding validation failed: " + " | ".join(grounding[:8]))
        if impact:
            raise ValueError("Council Screen-3 impact quality validation failed: " + " | ".join(impact[:6]))
        if language:
            raise ValueError("Council language validation failed: " + " | ".join(language[:6]))
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
    def _backend_advisor_ids(selected: List[Dict[str, Any]]) -> List[str]:
        """Return canonical Athar advisor IDs for the backend/public API.

        Backend and AI now share the same authoritative identity, e.g.
        ``AOS-SP-13``. Numeric catalogue positions are never emitted publicly.
        """
        out: List[str] = []
        for advisor in selected:
            value = str(advisor.get("advisor_id") or "").strip()
            if re.fullmatch(r"AOS-(?:LD|SP|FG|SE)-\d{2}", value) and value not in out:
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

        expected_ids = self._backend_advisor_ids(selected)
        if not expected_ids:
            raise ValueError("No canonical AOS advisor IDs resolved for Screen 3 public response.")
        rich_ids = self._normalize_attribution(rich_result.get("involved_advisor_ids"), expected_ids)
        if set(rich_ids) != set(expected_ids):
            # Model output never controls identity. The selected council resolved
            # from the authoritative registry is the API source of truth.
            rich_result["involved_advisor_ids"] = list(expected_ids)

        public = {
            "involved_advisor_ids": list(expected_ids),
            "suggestion": {"interventions": public_interventions},
        }
        self._validate_screen3_public_response(public)
        if public["involved_advisor_ids"] != expected_ids:
            raise ValueError("Screen 3 advisor ID projection mismatch.")
        return public

    @staticmethod
    def _validate_screen3_public_response(result: Dict[str, Any]) -> None:
        if set(result.keys()) != {"involved_advisor_ids", "suggestion"}:
            raise ValueError("Screen 3 response has unexpected top-level fields.")
        ids = result.get("involved_advisor_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(x, str)
                or re.fullmatch(r"AOS-(?:LD|SP|FG|SE)-\d{2}", x) is None
                for x in ids
            )
            or len(ids) != len(set(ids))
        ):
            raise ValueError(
                "Screen 3 involved_advisor_ids must be unique canonical AOS-* IDs."
            )
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
        involved: List[str] = []
        for advisor in raw_advisors:
            if not isinstance(advisor, dict):
                continue
            canonical = self._canonical_advisor_id(advisor)
            if canonical and canonical not in involved:
                involved.append(canonical)

        public = {
            "involved_advisor_ids": involved,
            "suggestion": {"text": generated_text},
        }
        if set(public.keys()) != {"involved_advisor_ids", "suggestion"}:
            raise ValueError("Invalid regeneration envelope.")
        if any(
            re.fullmatch(r"AOS-(?:LD|SP|FG|SE)-\d{2}", x) is None
            for x in involved
        ):
            raise ValueError("Regeneration involved_advisor_ids must use canonical AOS-* IDs.")
        return public

    def consult(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Current Screen-3 stage: selected Specialist opinions only, no Meta synthesis.

        Each selected advisor independently emits exactly one Screen-3 intervention
        in the backend's existing Interventions -> Results -> Outputs schema.
        AOS-META-00 is intentionally not called in this product stage.
        """
        input_obj = request.get("input") or {}
        debug = bool(request.get("debug") or input_obj.get("debug"))

        if bool(input_obj.get("is_output_regeneration")):
            total_start = time.perf_counter()
            public = self._regenerate_single_output(request)
            elapsed = round(time.perf_counter() - total_start, 3)
            if not debug:
                return public
            return {
                "debug": True,
                "mode": "single_output_regeneration",
                "timings_seconds": {"total": elapsed},
                "final_result": public,
            }

        total_start = time.perf_counter()
        selected = self._resolve_selected_advisors(request)

        # The backend Screen-3 contract currently allows at most four proposals.
        # With Meta deliberately disabled, one proposal maps to one selected
        # Specialist, so accepting more than four would silently drop opinions.
        if len(selected) > 4:
            raise ValueError(
                "Current Screen-3 specialist-opinions stage supports at most 4 selected advisors "
                "because the backend contract accepts at most 4 interventions."
            )

        advisor_outputs: List[Dict[str, Any]] = []
        interventions: List[Dict[str, Any]] = []
        advisor_timings: Dict[str, float] = {}

        for item in selected:
            advisor_id = item["advisor_id"]
            print(f"[specialist-only] Starting advisor {advisor_id}...", flush=True)
            started = time.perf_counter()
            output = self._run_advisor_screen3_intervention(item, request, selected)
            elapsed = time.perf_counter() - started
            advisor_timings[advisor_id] = round(elapsed, 3)
            advisor_outputs.append(output)
            interventions.append(output["intervention"])
            print(
                f"[specialist-only] Finished advisor {advisor_id} in {elapsed:.2f}s",
                flush=True,
            )

        involved_ids = self._backend_advisor_ids(selected)
        public_result = {
            "involved_advisor_ids": involved_ids,
            "suggestion": {"interventions": interventions},
        }
        self._validate_screen3_public_response(public_result)

        total_elapsed = round(time.perf_counter() - total_start, 3)
        timings = {
            "advisors": advisor_timings,
            "meta": 0.0,
            "total": total_elapsed,
        }

        internal = {
            "mode": "specialist_opinions_only",
            "meta_called": False,
            "advisor_outputs": advisor_outputs,
            "timings_seconds": timings,
            "canonical_advisor_ids": involved_ids,
            # Position i in interventions corresponds to position i in these IDs.
            "intervention_advisor_map": [
                {
                    "index": i,
                    "advisor_id": row["advisor_id"],
                }
                for i, row in enumerate(advisor_outputs)
            ],
        }

        if not debug:
            # Return the exact backend Screen-3 envelope only. Internal advisor
            # prose/timings stay available through debug mode and server logs.
            return public_result

        return {
            "debug": True,
            "mode": "specialist_opinions_only",
            "selected_advisor_ids": involved_ids,
            "advisor_outputs": advisor_outputs,
            "timings_seconds": timings,
            "final_result": public_result,
        }

