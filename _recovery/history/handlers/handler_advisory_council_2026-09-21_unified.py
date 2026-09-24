from __future__ import annotations

import json
import os
import re
import threading
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
                "حلّل الحالة وقدّم التدخلات والأولويات والمخاطر والافتراضات ونقاط التحقق "
                "التي يدعمها السياق فقط، ولا تخترع بيانات غير موجودة."
            ),
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

    def _run_meta(
        self,
        request: Dict[str, Any],
        advisor_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        meta_task = {
            "instruction": (
                "هذه مرحلة التركيب النهائي للمجلس. طبّق AOS-META-00 الأصلي بالكامل. "
                "قارن آراء المستشارين، اكشف الاتفاق والتعارض والتكرار والتكامل، "
                "ثم كوّن أفضل توصية موحدة قابلة للتنفيذ. لا تعمل مجرد تلخيص. "
                "أعد JSON صالحًا فقط وفق المخطط المطلوب."
            ),
            "case_context": self._shared_context(request),
            "selected_advisor_outputs": advisor_outputs,
            "required_schema": {
                "involved_advisor_ids": ["integer Backend IDs"],
                "suggestion": {
                    "interventions": [
                        {
                            "title": "string",
                            "confidence_level": "مرتفعة | متوسطة | منخفضة",
                            "impact_description": "string",
                            "reportable_value": "string",
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
                "Return 2 to 4 interventions when supported by the case.",
                "Do not generate beneficiaries_count, evidence, or is_selected.",
            ],
        }

        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"
        raw = self._generate(
            meta_adapter,
            self.meta_prompt,
            json.dumps(meta_task, ensure_ascii=False, indent=2),
            META_MAX_NEW_TOKENS,
            deterministic=True,
        )
        result = self.extract_json_object(raw)

        backend_ids = [
            x["backend_id"]
            for x in advisor_outputs
            if isinstance(x.get("backend_id"), int)
        ]
        result["involved_advisor_ids"] = backend_ids
        self._validate_backend_result(result)
        return result

    def consult(self, request: Dict[str, Any]) -> Dict[str, Any]:
        selected = self._resolve_selected_advisors(request)
        opinions = [self._run_advisor(item, request) for item in selected]
        return self._run_meta(request, opinions)
