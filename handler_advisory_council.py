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
ADVISOR_MAX_NEW_TOKENS = int(os.getenv("ADVISOR_MAX_NEW_TOKENS", "1800"))
META_MAX_NEW_TOKENS = int(os.getenv("META_MAX_NEW_TOKENS", "2600"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.20"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.90"))
COUNCIL_META_MODE = os.getenv("COUNCIL_META_MODE", "adapter").strip().lower()


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
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                    use_cache=True,
                )
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
        model_id = (
            incoming.get("model_advisor_id")
            or incoming.get("advisor_id")
            or incoming.get("system_code")
            or incoming.get("code")
        )

        registry_entry = None
        if isinstance(model_id, str):
            registry_entry = self.registry_by_model_id.get(model_id.strip())

        if registry_entry is None:
            name = self._norm(incoming.get("name") or incoming.get("advisor_name_ar"))
            registry_entry = self.registry_by_name.get(name)

        if registry_entry is None:
            raise ValueError(
                "Could not map Backend advisor to an Athar advisor. "
                f"Incoming advisor: {incoming}. Send model_advisor_id such as AOS-SP-08."
            )

        result = dict(registry_entry)
        result["backend_id"] = incoming.get("id")
        return result

    def _resolve_selected_advisors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_obj = request.get("input") or {}
        incoming = self._incoming_advisors(request)
        incoming_by_backend_id = {
            x.get("id"): x for x in incoming if isinstance(x, dict) and x.get("id") is not None
        }
        incoming_by_model_id = {}
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
                    reg = self.registry_by_model_id.get(raw.strip())
                    if reg is None:
                        raise ValueError(f"Unknown Athar advisor ID: {raw}")
                    item = dict(reg)
                    mapped = incoming_by_model_id.get(raw.strip())
                    item["backend_id"] = mapped.get("id") if mapped else None
                    resolved.append(item)
                    continue

                if isinstance(raw, int):
                    # Numeric values are treated as Backend DB IDs only when the
                    # request also supplies input.advisors with an explicit mapping.
                    mapped = incoming_by_backend_id.get(raw)
                    if mapped is None:
                        raise ValueError(
                            f"Numeric selected advisor ID {raw} has no mapping in input.advisors. "
                            "Do not assume Backend DB IDs equal Athar advisor numbers."
                        )
                    resolved.append(self._resolve_incoming(mapped))
                    continue

                raise ValueError(f"Unsupported selected advisor value: {raw!r}")
        else:
            for item in incoming:
                if isinstance(item, dict):
                    resolved.append(self._resolve_incoming(item))

        if not resolved:
            raise ValueError(
                "No selected advisors received. Send selected_advisor_ids/selected_advisors "
                "or provide only the selected advisors in input.advisors."
            )

        unique = []
        seen = set()
        for item in resolved:
            model_id = item["advisor_id"]
            if model_id not in seen:
                seen.add(model_id)
                unique.append(item)
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

    def _run_advisor(self, advisor: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._load_prompt(ADVISOR_PROMPTS_DIR / advisor["prompt_file"])
        task = {
            "instruction": (
                "طبّق System Prompt الأصلي بالكامل على الحالة التالية. "
                "قدّم رأيك المستقل فقط ضمن اختصاصك ولا تطلع على آراء أي مستشار آخر. "
                "حلّل الحالة وقدّم فقط التدخلات والأولويات والمخاطر والافتراضات ونقاط التحقق "
                "التي يدعمها السياق صراحة. لا تخترع بيانات أو أرقامًا أو نسبًا أو مددًا أو "
                "خطوط أساس أو مستهدفات غير موجودة في case_context. "
                "لا تحوّل موضوعًا خارج اختصاصك إلى توصية منك؛ إذا ظهر احتياج خارج نطاقك "
                "فاذكره كإحالة لمستشار مختص فقط. "
                "استند إلى حقائق الحالة الفعلية، وافصل بوضوح بين الحقيقة والاستنتاج."
            ),
            "strict_scope_rules": [
                "كل توصية يجب أن تقع داخل نطاق هذا المستشار كما يحدده System Prompt الأصلي.",
                "ممنوع إنشاء مستهدف رقمي أو نسبة تحسن أو مدة تنفيذ من عندك.",
                "الأرقام التاريخية الموجودة في الحالة لا تتحول تلقائيًا إلى مستهدفات مستقبلية.",
                "إذا لم توجد أدلة كافية لتوصية ضمن الاختصاص، صرّح بذلك بدل ملء الفراغ.",
            ],
            "advisor_id": advisor["advisor_id"],
            "advisor_name_ar": advisor["advisor_name_ar"],
            "case_context": self._shared_context(request),
        }
        opinion = self._generate(
            "specialist",
            prompt,
            json.dumps(task, ensure_ascii=False, indent=2),
            ADVISOR_MAX_NEW_TOKENS,
        )
        return {
            "advisor_id": advisor["advisor_id"],
            "backend_id": advisor.get("backend_id"),
            "advisor_name_ar": advisor["advisor_name_ar"],
            "opinion": opinion,
        }

    @staticmethod
    def _validate_backend_result(result: Dict[str, Any]) -> None:
        if not isinstance(result.get("involved_advisor_ids"), list):
            raise ValueError("involved_advisor_ids must be an array.")
        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict):
            raise ValueError("suggestion must be an object.")
        interventions = suggestion.get("interventions")
        if not isinstance(interventions, list) or not interventions:
            raise ValueError("suggestion.interventions must be a non-empty array.")

        required = {
            "title", "confidence_level", "impact_description",
            "reportable_value", "results",
        }
        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, dict):
                raise ValueError(f"interventions[{i}] must be an object.")
            missing = required - intervention.keys()
            if missing:
                raise ValueError(f"interventions[{i}] missing: {sorted(missing)}")
            results = intervention.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(f"interventions[{i}].results must be non-empty.")
            for j, result_item in enumerate(results):
                if not isinstance(result_item, dict) or not isinstance(result_item.get("text"), str):
                    raise ValueError(f"interventions[{i}].results[{j}].text is required.")
                outputs = result_item.get("outputs")
                if not isinstance(outputs, list) or not outputs:
                    raise ValueError(
                        f"interventions[{i}].results[{j}].outputs must be non-empty."
                    )
                for k, output in enumerate(outputs):
                    if not isinstance(output, dict) or not isinstance(output.get("text"), str):
                        raise ValueError(
                            f"interventions[{i}].results[{j}].outputs[{k}].text is required."
                        )

    @staticmethod
    def _normalize_digits(text: Any) -> str:
        return str(text or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))

    @classmethod
    def _extract_number_tokens(cls, text: Any) -> set[str]:
        normalized = cls._normalize_digits(text)

        # Advisor/system identifiers contain numbers that are identifiers, not
        # quantitative claims. Example: AOS-SP-13 must not be interpreted as an
        # unsupported numeric target of "13" by the grounding validator.
        normalized = re.sub(
            r"\b(?:AOS-(?:LD|SP|FG|SE)-\d{1,2}|AOS-META-\d{1,2}|ATHAR-ADV-\d{1,2})\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )

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

    def _grounding_violations(
        self,
        result: Dict[str, Any],
        request: Dict[str, Any],
    ) -> List[str]:
        """Detect common fabricated planning numbers before a result leaves RunPod.

        Historical numbers may be repeated only when they already occur in the
        authoritative case context. reportable_value is intentionally treated as
        a metric/evidence label, not a model-invented target.
        """
        violations: List[str] = []
        source_obj = self._shared_context(request)
        source_text = self._normalize_digits(
            json.dumps(source_obj, ensure_ascii=False, sort_keys=True)
        )
        source_numbers = self._extract_number_tokens(source_text)

        suggestion = result.get("suggestion") or {}
        interventions = suggestion.get("interventions") or []

        for idx, intervention in enumerate(interventions):
            if not isinstance(intervention, dict):
                continue

            reportable = str(intervention.get("reportable_value") or "")
            reportable_norm = self._normalize_digits(reportable)
            if self._extract_number_tokens(reportable_norm) or "%" in reportable_norm or "٪" in reportable_norm:
                violations.append(
                    f"interventions[{idx}].reportable_value must be a measurable metric/evidence label "
                    "without a model-created numeric target."
                )

            for field_name, field_value in intervention.items():
                for s in self._collect_strings(field_value):
                    norm = self._normalize_digits(s)

                    # Any number absent from the original case is unsupported.
                    for number in self._extract_number_tokens(norm):
                        if number not in source_numbers:
                            violations.append(
                                f"interventions[{idx}].{field_name} contains unsupported number: {number}"
                            )

                    # Planning percentages are not allowed unless the exact claim
                    # itself already exists in the authoritative case context.
                    if re.search(
                        r"(?:زيادة|رفع|خفض|تقليل|تحسين|الوصول|تحقيق|مستهدف|بنسبة)"
                        r".{0,50}\d+(?:[.,]\d+)?\s*(?:%|٪)",
                        norm,
                    ):
                        if norm not in source_text:
                            violations.append(
                                f"interventions[{idx}].{field_name} contains an unsupported planning percentage."
                            )

                    # Same protection for invented execution horizons such as
                    # "خلال 12 شهرًا".
                    duration_matches = re.findall(
                        r"(?:خلال|في غضون|مدة)\s+"
                        r"\d+(?:[.,]\d+)?\s*"
                        r"(?:يوم|أيام|أسبوع|أسابيع|شهر|أشهر|سنة|سنوات|عام|أعوام)",
                        norm,
                    )
                    for phrase in duration_matches:
                        if phrase not in source_text:
                            violations.append(
                                f"interventions[{idx}].{field_name} contains unsupported duration: {phrase}"
                            )

        # De-duplicate while preserving order.
        return list(dict.fromkeys(violations))

    def _run_meta(
        self,
        request: Dict[str, Any],
        advisor_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Debug metadata is kept outside the production response unless debug=true.
        self._last_meta_debug = {
            "repair_used": False,
            "initial_grounding_violations": [],
            "final_grounding_violations": [],
        }
        selected_model_ids = [x["advisor_id"] for x in advisor_outputs]

        meta_task = {
            "instruction": (
                "هذه مرحلة التركيب النهائي للمجلس. طبّق AOS-META-00 الأصلي بالكامل. "
                "قارن آراء المستشارين، اكشف الاتفاق والتعارض والتكرار والتكامل، "
                "ثم كوّن أفضل توصية موحدة قابلة للتنفيذ. لا تعمل مجرد تلخيص. "
                "التدخل النهائي يجب أن يكون له أصل واضح في رأي مستشار مختار واحد على الأقل؛ "
                "لا تستخدم case_context وحده لابتكار تدخل جديد خارج آراء المجلس. "
                "استخدم case_context للتحقق من الحقائق فقط. "
                "إذا كان هناك مستشار واحد فقط، فكل التدخلات يجب أن تبقى داخل نطاق اختصاصه. "
                "ممنوع اختلاق أي رقم أو نسبة أو خط أساس أو مستهدف أو مدة تنفيذ. "
                "الأرقام التاريخية لا تتحول إلى أهداف مستقبلية. "
                "أعد JSON صالحًا فقط وفق المخطط المطلوب."
            ),
            "case_context": self._shared_context(request),
            "selected_advisor_ids": selected_model_ids,
            "selected_advisor_outputs": advisor_outputs,
            "required_schema": {
                "involved_advisor_ids": ["integer Backend IDs"],
                "suggestion": {
                    "interventions": [
                        {
                            "title": "string",
                            "confidence_level": "مرتفعة | متوسطة | منخفضة",
                            "impact_description": "string",
                            "reportable_value": (
                                "اسم مؤشر أو دليل قابل للرصد فقط، بدون اختراع رقم أو نسبة أو مستهدف"
                            ),
                            "results": [
                                {
                                    "text": "string",
                                    "outputs": [{"text": "string"}],
                                }
                            ],
                        }
                    ]
                },
            },
            "rules": [
                "Return valid JSON only.",
                "No Markdown fences.",
                "Return 1 to 4 interventions, and omit weak/unsupported interventions.",
                "Every intervention must be supported by at least one selected advisor opinion.",
                "Do not create an intervention from case_context alone.",
                "Stay inside the expertise of the selected advisors.",
                "Do not generate beneficiaries_count, evidence, or is_selected.",
                "Do not invent numbers, percentages, baselines, targets, dates, deadlines, or execution durations.",
                "reportable_value is a measurable metric/evidence label, NOT a target value.",
                "If the case does not provide a target, write the metric to track, not a made-up target.",
                "Do not write advisor codes or advisor IDs inside user-facing intervention/result/output text; advisor attribution is internal.",
            ],
        }

        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"

        def generate_and_parse(task: Dict[str, Any]) -> Dict[str, Any]:
            raw_text = self._generate(
                meta_adapter,
                self.meta_prompt,
                json.dumps(task, ensure_ascii=False, indent=2),
                META_MAX_NEW_TOKENS,
                deterministic=True,
            )
            parsed = self.extract_json_object(raw_text)
            backend_ids = [
                x["backend_id"]
                for x in advisor_outputs
                if isinstance(x.get("backend_id"), int)
            ]
            parsed["involved_advisor_ids"] = backend_ids
            self._validate_backend_result(parsed)
            return parsed

        result = generate_and_parse(meta_task)
        violations = self._grounding_violations(result, request)
        self._last_meta_debug["initial_grounding_violations"] = list(violations)

        if violations:
            self._last_meta_debug["repair_used"] = True
            repair_task = dict(meta_task)
            repair_task["instruction"] = (
                meta_task["instruction"]
                + " المحاولة السابقة خالفت قواعد الـGrounding. أصلحها بالكامل، "
                  "ولا تدافع عن النص السابق. احذف أي تدخل غير مدعوم، واستبدل أي "
                  "مستهدف رقمي مختلق باسم مؤشر قابل للقياس بدون قيمة مستهدفة."
            )
            repair_task["validation_errors"] = violations
            repair_task["previous_invalid_output"] = result
            result = generate_and_parse(repair_task)
            violations = self._grounding_violations(result, request)

        self._last_meta_debug["final_grounding_violations"] = list(violations)

        if violations:
            raise ValueError(
                "Council grounding validation failed after repair: "
                + " | ".join(violations[:8])
            )

        return result

    def consult(self, request: Dict[str, Any]) -> Dict[str, Any]:
        input_obj = request.get("input") or {}
        debug = bool(request.get("debug") or input_obj.get("debug"))

        total_start = time.perf_counter()
        selected = self._resolve_selected_advisors(request)

        opinions: List[Dict[str, Any]] = []
        advisor_timings: Dict[str, float] = {}

        for item in selected:
            advisor_id = item["advisor_id"]
            print(f"[council] Starting advisor {advisor_id}...", flush=True)
            started = time.perf_counter()
            opinion = self._run_advisor(item, request)
            elapsed = time.perf_counter() - started
            advisor_timings[advisor_id] = round(elapsed, 3)
            opinions.append(opinion)
            print(
                f"[council] Finished advisor {advisor_id} in {elapsed:.2f}s",
                flush=True,
            )

        print(
            f"[council] Starting AOS-META-00 synthesis for {len(opinions)} advisor(s)...",
            flush=True,
        )
        meta_started = time.perf_counter()
        final_result = self._run_meta(request, opinions)
        meta_elapsed = time.perf_counter() - meta_started
        total_elapsed = time.perf_counter() - total_start
        print(
            f"[council] Finished AOS-META-00 in {meta_elapsed:.2f}s; "
            f"total council time {total_elapsed:.2f}s",
            flush=True,
        )

        # Production behavior remains unchanged unless debug=true is explicitly sent.
        if not debug:
            return final_result

        return {
            "debug": True,
            "selected_advisor_ids": [x["advisor_id"] for x in selected],
            "advisor_outputs": opinions,
            "meta_debug": getattr(self, "_last_meta_debug", {}),
            "timings_seconds": {
                "advisors": advisor_timings,
                "meta": round(meta_elapsed, 3),
                "total": round(total_elapsed, 3),
            },
            "final_result": final_result,
        }
