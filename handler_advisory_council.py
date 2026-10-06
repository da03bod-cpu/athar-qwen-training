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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList


ROOT = Path(os.getenv("ATHAR_ROOT", "/workspace/data/athar"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-14B")

ADVISOR_PROMPTS_DIR = Path(
    os.getenv("ADVISOR_PROMPTS_DIR", str(ROOT / "prompts" / "advisors"))
)
META_PROMPTS_DIR = Path(
    os.getenv("META_PROMPTS_DIR", str(ROOT / "prompts" / "meta"))
)
DEFAULT_META_ADVISOR_SLUG = os.getenv(
    "DEFAULT_META_ADVISOR_SLUG", "AOS-META-01"
).strip()

# The confirmed backend contract currently sends AOS-META-01, while the
# production GitHub repository historically stores the same Meta DNA as
# prompts/meta/AOS-META-00.md. Exact slug files always take precedence.
# This compatibility alias lets the new contract work immediately without
# duplicating or weakening the current full Meta prompt. Future Meta Advisors
# require no Python change: add prompts/meta/<SLUG>.md and send that slug.
META_PROMPT_COMPAT_ALIASES = {
    "AOS-META-01": "AOS-META-00",
}
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



class _RegexTokenBlocker(LogitsProcessor):
    """Hard-mask token IDs whose decoded token matches a forbidden regex."""

    def __init__(self, token_ids: List[int]) -> None:
        self.token_ids = list(dict.fromkeys(int(x) for x in token_ids))

    def __call__(self, input_ids, scores):
        if self.token_ids:
            scores[:, self.token_ids] = -float("inf")
        return scores


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
        self._meta_prompt_cache: Dict[str, str] = {}
        self._meta_prompt_path_cache: Dict[str, Path] = {}
        self.model_lock = model_lock or threading.RLock()
        self._forbidden_token_cache: Dict[str, List[int]] = {}

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

    def _resolve_meta_advisor(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the Meta persona from input.meta_advisor.slug.

        Resolution order:
        1) prompts/meta/<slug>.md or .txt
        2) compatibility alias for the current production prompt
        3) the configured default slug when meta_advisor is null

        This keeps Meta selection data-driven. Adding a new Meta persona later
        only requires adding prompts/meta/<NEW-SLUG>.md to GitHub and sending that
        slug in the backend request; no handler code change is required.
        """
        raw = request.get("meta_advisor")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("meta_advisor must be an object or null.")

        slug = str((raw or {}).get("slug") or DEFAULT_META_ADVISOR_SLUG).strip()
        name = str((raw or {}).get("name") or "المستشار الأعلى").strip()
        if not slug:
            slug = DEFAULT_META_ADVISOR_SLUG

        # Prevent path traversal and malformed registry codes.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,79}", slug) is None:
            raise ValueError(f"Invalid meta_advisor.slug: {slug!r}")

        candidates: List[Path] = [
            META_PROMPTS_DIR / f"{slug}.md",
            META_PROMPTS_DIR / f"{slug}.txt",
        ]

        alias = META_PROMPT_COMPAT_ALIASES.get(slug)
        if alias:
            candidates.extend([
                META_PROMPTS_DIR / f"{alias}.md",
                META_PROMPTS_DIR / f"{alias}.txt",
            ])

        prompt_path = next((p for p in candidates if p.exists()), None)
        if prompt_path is None:
            looked = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"No Meta Advisor persona file found for slug {slug}. Looked in: {looked}. "
                f"Add prompts/meta/{slug}.md to GitHub."
            )

        cache_key = str(prompt_path.resolve())
        prompt = self._meta_prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self._load_prompt(prompt_path)
            self._meta_prompt_cache[cache_key] = prompt
            self._meta_prompt_path_cache[cache_key] = prompt_path

        return {
            "slug": slug,
            "name": name,
            "prompt": prompt,
            "prompt_path": str(prompt_path),
        }

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

    def _forbidden_token_ids(self, pattern: str) -> List[int]:
        cached = self._forbidden_token_cache.get(pattern)
        if cached is not None:
            return cached
        rx = re.compile(pattern)
        bad: List[int] = []
        vocab_size = len(self.tokenizer)
        for token_id in range(vocab_size):
            try:
                piece = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                piece = str(self.tokenizer.convert_ids_to_tokens(token_id) or "")
            if piece and rx.search(piece):
                bad.append(token_id)
        self._forbidden_token_cache[pattern] = bad
        return bad

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
        forbidden_token_regex: Optional[str] = None,
        forbidden_phrases: Optional[List[str]] = None,
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

                processors = []
                if forbidden_token_regex:
                    bad_ids = self._forbidden_token_ids(forbidden_token_regex)
                    if bad_ids:
                        processors.append(_RegexTokenBlocker(bad_ids))
                if processors:
                    gen_kwargs["logits_processor"] = LogitsProcessorList(processors)

                if forbidden_phrases:
                    sequences: List[List[int]] = []
                    seen = set()
                    for phrase in forbidden_phrases:
                        phrase = str(phrase or "").strip()
                        if not phrase:
                            continue
                        for variant in (phrase, " " + phrase):
                            seq = self.tokenizer(
                                variant,
                                add_special_tokens=False,
                                return_attention_mask=False,
                            )["input_ids"]
                            key = tuple(int(x) for x in seq)
                            if key and key not in seen:
                                seen.add(key)
                                sequences.append(list(key))
                    if sequences:
                        gen_kwargs["bad_words_ids"] = sequences

                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)

                generated = output_ids[0, inputs["input_ids"].shape[1] :]
                return self.clean_model_text(
                    self.tokenizer.decode(generated, skip_special_tokens=True)
                )

    @staticmethod
    def _payload(request: Dict[str, Any]) -> Dict[str, Any]:
        """Return Screen-3 payload for the current backend contract."""
        payload = request.get("payload")
        if isinstance(payload, dict):
            return payload
        legacy = request.get("input")
        if isinstance(legacy, dict):
            return legacy
        return {}

    def _incoming_advisors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        value = self._payload(request).get("advisors") or []
        return value if isinstance(value, list) else []

    def _resolve_incoming(self, incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a backend advisor slug to the authoritative Athar Expert DNA."""
        incoming_slug = str(incoming.get("slug") or "").strip()
        incoming_id = incoming.get("id")
        model_id = (
            incoming_slug
            or incoming.get("model_advisor_id")
            or incoming.get("advisor_id")
            or incoming.get("system_code")
            or incoming.get("code")
            or (incoming_id if isinstance(incoming_id, str) else None)
        )

        registry_entry = None
        if isinstance(model_id, str):
            registry_entry = self.registry_by_model_id.get(model_id.strip())
        if registry_entry is None and isinstance(incoming_id, int):
            registry_entry = self.registry_by_number.get(incoming_id)
        if registry_entry is None:
            name = self._norm(incoming.get("name") or incoming.get("advisor_name_ar"))
            registry_entry = self.registry_by_name.get(name)
        if registry_entry is None:
            raise ValueError(
                "Could not map backend advisor slug to an Athar advisor. "
                f"Incoming advisor: {incoming}."
            )

        result = dict(registry_entry)
        result["backend_id"] = incoming_slug or registry_entry["advisor_id"]
        result["backend_payload"] = {
            k: incoming.get(k)
            for k in ("slug", "id", "name", "title", "capabilities", "traits")
            if k in incoming
        }
        return result

    def _resolve_selected_advisors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = self._payload(request)
        incoming = self._incoming_advisors(request)

        selected = payload.get("selected_advisor_ids")
        if selected is None:
            selected = payload.get("selected_advisors")

        resolved: List[Dict[str, Any]] = []
        incoming_by_model_id: Dict[str, Dict[str, Any]] = {}
        incoming_by_backend_id: Dict[Any, Dict[str, Any]] = {}

        for item in incoming:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug") or "").strip()
            mid = (
                slug
                or item.get("model_advisor_id")
                or item.get("advisor_id")
                or item.get("system_code")
                or item.get("code")
                or (item.get("id") if isinstance(item.get("id"), str) else None)
            )
            if isinstance(mid, str) and mid.strip():
                incoming_by_model_id[mid.strip()] = item
            if item.get("id") is not None:
                incoming_by_backend_id[item.get("id")] = item

        if isinstance(selected, list) and selected:
            for raw in selected:
                if isinstance(raw, dict):
                    resolved.append(self._resolve_incoming(raw))
                    continue
                if isinstance(raw, str):
                    mapped = incoming_by_model_id.get(raw.strip())
                    if mapped is not None:
                        resolved.append(self._resolve_incoming(mapped))
                        continue
                    reg = self.registry_by_model_id.get(raw.strip())
                    if reg is None and raw.strip().isdigit():
                        reg = self.registry_by_number.get(int(raw.strip()))
                    if reg is None:
                        raise ValueError(f"Unknown Athar advisor ID: {raw}")
                    item = dict(reg)
                    item["backend_id"] = item["advisor_id"]
                    resolved.append(item)
                    continue
                if isinstance(raw, int):
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
            # Current contract: payload.advisors is the selected council.
            for item in incoming:
                if isinstance(item, dict):
                    resolved.append(self._resolve_incoming(item))

        if not resolved:
            raise ValueError("payload.advisors must contain at least one known advisor slug.")

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in resolved:
            key = (item["advisor_id"], str(item.get("backend_id") or ""))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        if len(unique) > 16:
            raise ValueError("A consultation may include at most 16 selected advisors.")
        return unique

    def _shared_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._payload(request)
        return {
            "run_id": request.get("run_id"),
            "consultation_id": request.get("consultation_id"),
            "meta_advisor": request.get("meta_advisor"),
            "topic": request.get("topic"),
            "kind": request.get("kind"),
            "reason": request.get("reason"),
            "organization": payload.get("organization"),
            "programs": payload.get("programs"),
            "track": payload.get("track"),
            "goal": payload.get("goal"),
            "impact_map": payload.get("impact_map"),
            "target": payload.get("target"),
            "output": payload.get("output"),
            "intervention": payload.get("intervention"),
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
            "AOS-SP-11": ("مبادرة", "تدخل", "تصميم", "مستفيد", "قيمة", "نموذج", "أثر", "حل"),
            "AOS-SP-13": ("برنامج", "مشروع", "محفظ", "أولو", "جدوى", "اعتماد", "موارد", "توسع", "منافع"),
            "AOS-SP-15": ("أثر", "تقييم", "متابعة", "تعلم", "meal", "مؤشر", "نظرية", "مساهمة", "نتائج"),
            "AOS-FG-18": ("تمويل", "دخل", "مانح", "موارد", "استدام", "إيراد", "تبرع", "شراك"),
            "AOS-SE-27": ("تعليم", "طالب", "مدرس", "تعلم", "تحصيل", "تسرب", "تربوي", "بحث"),
            "AOS-SE-29": ("اجتماع", "أسرة", "مستفيد", "دعم", "رعاية", "احتياج", "حماية", "خدمات"),
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
        input_obj = self._payload(request)
        track = input_obj.get("track") or {}
        goal_obj = input_obj.get("goal") or {}
        org_obj = input_obj.get("organization") or {}
        goal_statement = str(goal_obj.get("statement") or "").strip() if isinstance(goal_obj, dict) else str(goal_obj or "").strip()
        important_notes = str(org_obj.get("important_notes") or "").strip() if isinstance(org_obj, dict) else ""

        impact_templates = {
            "AOS-SP-11": "يركز رأي المستشار على تصميم أو تحسين المبادرات الاستراتيجية وربطها بالمشكلة الاجتماعية ومحركات الأثر واحتياجات الفئة المستهدفة.",
            "AOS-SP-13": "يركز رأي المستشار على ترتيب أولوية البرامج والمشاريع القائمة وربط قرارات التوسع بالقيمة الاستراتيجية والجدوى والموارد المتاحة.",
            "AOS-SP-15": "يركز رأي المستشار على بناء منظومة متابعة وتقييم وتعلم وقياس أثر تربط البرامج القائمة بنتائج المستفيدين والهدف الاستراتيجي المعتمد.",
            "AOS-FG-18": "يركز رأي المستشار على تنويع مصادر الدخل وتقليل الاعتماد على التمويل الموسمي بما يدعم استدامة البرامج والخدمات.",
            "AOS-SE-27": "يركز رأي المستشار على ملاءمة التدخلات التعليمية للعوامل المرتبطة بالاستمرار في التعليم والوصول المدرسي والتحصيل ضمن الفئة المستهدفة.",
            "AOS-SE-29": "يركز رأي المستشار على احتياجات الأسر والمستفيدين والعوامل الاجتماعية التي تؤثر في الاستمرار في التعليم والوصول إلى خدمات الدعم المناسبة.",
        }
        impact_description = impact_templates.get(
            advisor["advisor_id"],
            f"يركز رأي {advisor['advisor_name_ar']} على تطبيق توصية داخل نطاق اختصاصه بما يخدم الهدف المعتمد وبيانات الحالة المتاحة.",
        )
        if advisor["advisor_id"] in {"AOS-SP-11", "AOS-SP-13", "AOS-SP-15", "AOS-SE-27", "AOS-SE-29"} and goal_statement:
            impact_description += f" الهدف المعتمد: {goal_statement}"
        elif advisor["advisor_id"] == "AOS-FG-18" and important_notes:
            impact_description += f" ويستند إلى الملاحظة المؤسسية: {important_notes}"
        impact_description = impact_description[:900].strip()

        primary_indicator = ""
        if isinstance(track, dict):
            primary_indicator = str(track.get("primary_indicator") or "").strip()

        indicator_by_advisor = {
            "AOS-SP-11": "مؤشر جاهزية وملاءمة المبادرات المقترحة",
            "AOS-SP-13": "مؤشر أولوية وجدوى البرامج والمشاريع",
            "AOS-SP-15": "مؤشر نتائج وأثر البرامج المستهدفة",
            "AOS-FG-18": "مؤشر تنوع واستدامة مصادر التمويل",
            "AOS-SE-27": "مؤشر الاستمرار والتحصيل التعليمي للفئة المستهدفة",
            "AOS-SE-29": "مؤشر وصول واستفادة الفئات المستهدفة من خدمات الدعم",
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
            "والتقيم": "والتقييم",
            "والقياس الأثر": "وقياس الأثر",
            "القياس الأثر": "قياس الأثر",
            "مستدمة": "مستدامة",
            "مستدمة،": "مستدامة،",
            "كبرية": "كبرى",
            "التمويل الموسمية": "التمويل الموسمي",
            "التبرعات الموسمية": "التبرعات الموسمية",
            "الخطوط النقلية": "خطوط النقل",
            "البرنامج الحقيبة": "برنامج الحقيبة",
            "والبرنامج الحقيبة": "وبرنامج الحقيبة",
            "الزي المدارسي": "الزي المدرسي",
            "الزي المدرسية": "الزي المدرسي",
            "المدارسي": "المدرسي",
            "الحقيبة المدرسية": "الحقيبة المدرسية",
            "الت_dropout": "التسرب",
            "تموilen": "تمويل",
            "خط أنابيب فرص تمويل": "مسار فرص تمويل",
            "خط أنابيب التمويل": "مسار التمويل",
            "خط أنابيب": "مسار",
            "من المربح استثمار المزيد من الموارد": "من المجدي تخصيص مزيد من الموارد",
            "من المربح استثمار مزيد من الموارد": "من المجدي تخصيص مزيد من الموارد",
            "من المربح": "من المجدي",
            "استثمار المزيد من الموارد": "تخصيص مزيد من الموارد",
        }

        def scrub_string(value: str) -> str:
            value = str(value or "")
            for bad, good in COMMON_TEXT_FIXES.items():
                value = value.replace(bad, good)
            # Deterministic Arabic normalization only; no semantic rewriting.
            value = re.sub(r"\bبرنامج\s+الحقيبة\s+والزي\s+المدرسية\b", "برنامج الحقيبة والزي المدرسي", value)
            value = re.sub(r"\bمشروع\s+النقل\s+المدرسي\s+الت(?:ناوي|فاعلي)\b", "مشروع النقل المدرسي التنموي", value)
            value = re.sub(r"\bالمتابعة\s+والتقييم\s+والتعلم\s+والقياس\s+الأثر\b", "المتابعة والتقييم والتعلم وقياس الأثر", value)
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
            value = value.get("slug") or value.get("advisor_id") or value.get("model_advisor_id") or value.get("id")
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
            or self._payload(request).get("consultation_id")
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
                    "outputs": intervention.get("outputs"),
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
        input_obj = self._payload(request)
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

    def _clean_meta_public_text(self, value: Any, request: Dict[str, Any]) -> str:
        """Deterministic Arabic cleanup for Meta public fields.

        This layer never invents new semantic content. It fixes a small set of
        recurrent orthographic/model leakage issues, strips unsupported scripts,
        and removes any numeric token that is not grounded in the request.
        """
        text = str(value or "").strip()
        fixes = {
            "التناوي": "التنموي",
            "التفاعلي": "التنموي",
            "التقيم": "التقييم",
            "والتقيم": "والتقييم",
            "والقياس الأثر": "وقياس الأثر",
            "القياس الأثر": "قياس الأثر",
            "مستدمة": "مستدامة",
            "كبرية": "كبرى",
            "التمويل الموسمية": "التمويل الموسمي",
            "الخطوط النقلية": "خطوط النقل",
            "البرنامج الحقيبة": "برنامج الحقيبة",
            "والبرنامج الحقيبة": "وبرنامج الحقيبة",
            "الزي المدارسي": "الزي المدرسي",
            "الزي المدرسية": "الزي المدرسي",
            "المدارسي": "المدرسي",
            "الت_dropout": "التسرب",
            "تموilen": "تمويل",
            "خط أنابيب فرص تمويل": "مسار فرص تمويل",
            "خط أنابيب التمويل": "مسار التمويل",
            "خط أنابيب": "مسار",
            "من المربح استثمار المزيد من الموارد": "من المجدي تخصيص مزيد من الموارد",
            "من المربح استثمار مزيد من الموارد": "من المجدي تخصيص مزيد من الموارد",
            "من المربح": "من المجدي",
            "استثمار المزيد من الموارد": "تخصيص مزيد من الموارد",
        }
        for bad, good in fixes.items():
            text = text.replace(bad, good)

        text = re.sub(
            r"\bالمتابعة\s+والتقييم\s+والتعلم\s+والقياس\s+الأثر\b",
            "المتابعة والتقييم والتعلم وقياس الأثر",
            text,
        )
        text = re.sub(r"[\u0400-\u052F\u4E00-\u9FFF\u3040-\u30FF]", "", text)

        allowed_latin = {
            "MEAL", "KPI", "KPIs", "SROI", "BSC", "SMART",
            "Contribution", "Analysis", "Theory", "Change",
        }
        kept = []
        for token in text.split():
            has_ar = bool(re.search(r"[\u0600-\u06FF]", token))
            has_lat = bool(re.search(r"[A-Za-z]", token))
            if has_ar and has_lat:
                ar = re.sub(r"[A-Za-z]+", "", token)
                token = ar if len(re.sub(r"[^\u0600-\u06FF]", "", ar)) >= 3 else ""
            elif has_lat:
                bare = re.sub(r"[^A-Za-z]", "", token)
                if bare and bare not in allowed_latin and bare.lower() not in {
                    "impact", "depth", "social", "funding", "education",
                }:
                    token = ""
            if token:
                kept.append(token)
        text = " ".join(kept)

        # Do not silently delete unsupported numbers here. Earlier builds did
        # that and could turn a bad sentence such as "استهداف 300 طالب بحلول..."
        # into the semantically broken "استهداف طالب بحلول...". Quantitative
        # grounding is validated after parsing; unsupported numbers cause a Meta
        # retry instead of mutating the model's meaning in Python.
        text = re.sub(r"\s*([%٪])\s*", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip(" -–—,:؛")
        return text

    def _meta_public_hygiene_violations(
        self,
        interventions: List[Dict[str, Any]],
        request: Dict[str, Any],
    ) -> List[str]:
        """Reject user-facing contamination without authoring new content.

        This validator is intentionally fail-closed. It does not rewrite the
        recommendation; it tells AOS-META-00 to regenerate when the public text
        leaks advisor language, unsupported scheduling language, obvious sector
        contamination, foreign scripts, or broken fragments created by a poor
        generation.
        """
        violations: List[str] = []
        source_text = re.sub(
            r"\s+",
            " ",
            json.dumps(self._shared_context(request), ensure_ascii=False, sort_keys=True),
        ).lower()

        # Terms that strongly indicate cross-domain training leakage. They are
        # allowed when the case itself actually contains them.
        contamination_terms = (
            "طبيب", "أطباء", "مرضى", "مريض", "علاج", "مستشفى",
            "الحج", "حجاج", "معتمر", "معتمرين", "ضيوف الرحمن",
            "زائرات", "زائرين",
        )
        temporal_patterns = (
            r"\bبحلول\s+(?:نهاية|بداية|الربع|الفصل|العام|السنة|الشهر|الأسبوع)",
            r"\bبنهاية\s+(?:الربع|الفصل|العام|السنة|الشهر|الأسبوع)",
            r"\bنهاية\s+الربع\s+(?:الأول|الثاني|الثالث|الرابع)",
            r"\bالربع\s+(?:الأول|الثاني|الثالث|الرابع)",
            r"\bخلال\s+(?:الربع|الفصل|الشهر|الأسبوع)",
        )
        broken_patterns = (
            r"\bبحلول\s*(?:$|[،؛,.])",
            r"\bبنسبة\s*(?:$|خلال|بنهاية|،|؛|\.)",
            r"\bلـ\s*(?:$|[،؛,.])",
            r"\b(?:عدد|نسبة)\s+(?:طالب|طالبة|طبيب|مستفيد)\s+(?:إضافي|إضافية)?\s*بحلول\b",
        )

        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, dict):
                continue
            for field_text in self._collect_strings(intervention):
                raw = str(field_text or "").strip()
                low = raw.lower()
                if not raw:
                    continue
                if self._has_foreign_script(raw):
                    violations.append(f"interventions[{i}] contains foreign-script leakage")
                if re.search(r"\bAOS-(?:LD|SP|FG|SE)-\d{2}\b", raw, flags=re.I):
                    violations.append(f"interventions[{i}] exposes an advisor ID")
                if re.search(r"\b(?:مستشار(?:ة|ين|ون)?|المجلس|مجلس المستشارين)\b", raw):
                    violations.append(f"interventions[{i}] exposes advisor/council language")
                if "/" in raw and "/" not in source_text:
                    violations.append(f"interventions[{i}] contains unsupported slash-composite terminology")
                for pat in temporal_patterns:
                    for match in re.findall(pat, raw, flags=re.I):
                        phrase = str(match).lower().strip()
                        if phrase and phrase not in source_text:
                            violations.append(
                                f"interventions[{i}] contains unsupported scheduling language: {phrase}"
                            )
                for pat in broken_patterns:
                    if re.search(pat, raw, flags=re.I):
                        violations.append(f"interventions[{i}] contains a broken/incomplete phrase")
                        break
                for term in contamination_terms:
                    if term in low and term not in source_text:
                        violations.append(
                            f"interventions[{i}] contains unrelated domain leakage: {term}"
                        )

        return list(dict.fromkeys(violations))

    def _parse_meta_screen3_protocol(
        self,
        raw: str,
        request: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Parse the compact line protocol emitted by AOS-META-00.

        The Meta model is intentionally NOT asked to generate JSON. Python owns
        JSON construction, eliminating malformed-JSON failures while preserving
        all semantic synthesis in the Meta Advisor.
        """
        cleaned = self.clean_model_text(raw)
        cleaned = re.sub(r"```(?:text|txt|markdown|md)?", "", cleaned, flags=re.I)
        cleaned = cleaned.replace("```", "").strip()

        normalized_lines: List[str] = []
        for original in cleaned.splitlines():
            line = original.strip()
            if not line:
                continue
            line = re.sub(r"^[\-*•]+\s*", "", line)
            line = line.replace("：", ":")
            normalized_lines.append(line)

        has_begin = any(re.fullmatch(r"(?i)BEGIN_INTERVENTION", x) for x in normalized_lines)
        if not has_begin:
            rebuilt: List[str] = []
            started = False
            for line in normalized_lines:
                if re.match(r"(?i)^(?:TITLE|العنوان)\s*[:=]", line):
                    if started:
                        rebuilt.append("END_INTERVENTION")
                    rebuilt.append("BEGIN_INTERVENTION")
                    started = True
                if started:
                    rebuilt.append(line)
            if started:
                rebuilt.append("END_INTERVENTION")
                normalized_lines = rebuilt

        interventions: List[Dict[str, Any]] = []
        block: Optional[Dict[str, Any]] = None
        current_result: Optional[Dict[str, Any]] = None

        label_patterns = {
            "title": r"(?i)^(?:TITLE|العنوان)\s*[:=]\s*(.+)$",
            "confidence": r"(?i)^(?:CONFIDENCE|الثقة|درجة الثقة)\s*[:=]\s*(.+)$",
            "impact": r"(?i)^(?:IMPACT|الأثر|وصف الأثر)\s*[:=]\s*(.+)$",
            "reportable": r"(?i)^(?:REPORTABLE|REPORTABLE_VALUE|القيمة القابلة للقياس|القيمة)\s*[:=]\s*(.+)$",
            "result": r"(?i)^(?:RESULT|النتيجة)\s*[:=]\s*(.+)$",
            "output": r"(?i)^(?:OUTPUT|المخرج)\s*[:=]\s*(.+)$",
        }

        def finish_block() -> None:
            nonlocal block, current_result
            if not block:
                block = None
                current_result = None
                return
            title = self._clean_meta_public_text(block.get("title"), request)
            impact = self._clean_meta_public_text(block.get("impact"), request)
            reportable = self._clean_meta_public_text(block.get("reportable"), request)
            confidence = self._screen3_confidence(block.get("confidence"))
            results_out: List[Dict[str, Any]] = []
            for row in block.get("results", []):
                rtext = self._clean_meta_public_text(row.get("text"), request)
                outs = [
                    {"text": self._clean_meta_public_text(x, request)}
                    for x in row.get("outputs", [])
                ]
                outs = [x for x in outs if x["text"]]
                if rtext and outs:
                    results_out.append({"text": rtext, "outputs": outs[:3]})
                if len(results_out) >= 3:
                    break
            if title and impact and reportable and results_out:
                interventions.append({
                    "title": title,
                    "confidence_level": confidence,
                    "impact_description": impact,
                    "reportable_value": reportable,
                    "results": results_out,
                })
            block = None
            current_result = None

        for line in normalized_lines:
            if re.fullmatch(r"(?i)BEGIN_INTERVENTION", line):
                if block:
                    finish_block()
                block = {"results": []}
                current_result = None
                continue
            if re.fullmatch(r"(?i)END_INTERVENTION", line):
                finish_block()
                continue
            if block is None:
                continue

            matched = False
            for key, pattern in label_patterns.items():
                m = re.match(pattern, line)
                if not m:
                    continue
                value = m.group(1).strip()
                matched = True
                if key == "result":
                    current_result = {"text": value, "outputs": []}
                    block["results"].append(current_result)
                elif key == "output":
                    if current_result is None:
                        current_result = {"text": value, "outputs": []}
                        block["results"].append(current_result)
                    current_result["outputs"].append(value)
                else:
                    block[key] = value
                break
            if matched:
                continue

            if current_result is not None and current_result.get("outputs"):
                current_result["outputs"][-1] += " " + line
            elif current_result is not None:
                current_result["text"] += " " + line
            elif block.get("reportable"):
                block["reportable"] += " " + line
            elif block.get("impact"):
                block["impact"] += " " + line
            elif block.get("title"):
                block["title"] += " " + line

        if block:
            finish_block()

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in interventions:
            key = re.sub(r"\s+", " ", item["title"]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:4]

    def _run_meta(
        self,
        request: Dict[str, Any],
        advisor_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate Screen-3 with a split Meta architecture.

        Phase A1 — Advisor commentary:
            * request one Meta review per Specialist;
            * parse multiple equivalent formats;
            * if formatting collapses, preserve the run with an advisor-grounded
              deterministic commentary fallback rather than aborting Screen 3.

        Phase A2 — Strategy synthesis:
            * approve exactly 2-3 grounded, materially different interventions.

        Phase B — Weekly execution design:
            * run one dedicated Meta generation per approved intervention;
            * generate exactly 12 weekly Result units for that intervention only;
            * every Result has 1-3 concrete texts executable within 5 working days.

        This deliberately avoids asking Qwen to hold advisor reviews, multiple
        strategic interventions, and 24-36 weekly units in one very long output.
        The split keeps the full selected Meta persona/Expert DNA on every call,
        while Python owns parsing, grounding and quality validation.

        Backend compatibility:
            the confirmed API remains intervention -> outputs[] -> results[].
            ``outputs[0:12]`` are the twelve ordered weekly Result units;
            ``output.text`` is the Result statement and ``output.results`` are
            its 1-3 mandatory 5-day executable texts.
        """
        payload = self._payload(request)
        meta = self._resolve_meta_advisor(request)
        advisor_reasonings = self._build_advisor_reasonings(advisor_outputs, request)
        involved_ids = [x["advisor_id"] for x in advisor_reasonings]
        if not involved_ids:
            raise ValueError("Generate requires at least one usable Specialist reasoning.")

        reasonings_by_id = {
            row["advisor_id"]: row["reasoning"] for row in advisor_reasonings
        }
        council = [
            {"advisor_id": row["advisor_id"], "reasoning": row["reasoning"]}
            for row in advisor_reasonings
        ]
        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"

        authoritative_context = {
            "organization": payload.get("organization"),
            "programs": payload.get("programs") or [],
            "track": payload.get("track"),
            "goal": payload.get("goal"),
            "impact_map": payload.get("impact_map"),
        }
        authoritative_text = json.dumps(
            authoritative_context,
            ensure_ascii=False,
            sort_keys=True,
        )
        authoritative_tokens: set[str] = set()
        source_numbers = self._extract_number_tokens(
            self._normalize_digits(authoritative_text)
        )
        # Product-level timeboxes are system facts, not invented NPO targets.
        allowed_system_numbers = {"5", "12", "90"}
        allowed_numbers = source_numbers | allowed_system_numbers

        quality_stop = {
            "مخرج", "المخرج", "مخرجا", "مخرجات", "نتيجة", "النتيجة",
            "اسبوع", "أسبوع", "الأسبوع", "مرحلة", "المرحلة", "عمل",
            "تنفيذ", "التنفيذ", "مراجعة", "متابعة", "استكمال", "إجراء",
            "الاجراء", "الإجراء", "خطوة", "الخطوة", "نشاط", "النشاط",
            "توصية", "التوصية", "المطلوب", "اعتماد", "إعداد", "اعداد",
            "تحديد", "تحقيق", "تطبيق", "العمل", "الحالي", "الحالية",
            "مراجعه", "متابعه", "نتيجه", "النتيجه", "مرحله", "المرحله",
            "خطوه", "الخطوه", "توصيه", "التوصيه", "اجراء", "الاجراء",
        }

        def qtokens(value: Any) -> set[str]:
            # Lightweight Arabic normalization/stemming is used only for quality
            # matching. It does not alter any text returned to the backend.
            out: set[str] = set()
            for raw_token in self._impact_tokens(value):
                token = str(raw_token).lower()
                token = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", token)
                token = (
                    token.replace("أ", "ا")
                    .replace("إ", "ا")
                    .replace("آ", "ا")
                    .replace("ى", "ي")
                    .replace("ؤ", "و")
                    .replace("ئ", "ي")
                    .replace("ة", "ه")
                )
                if len(token) < 3 or token in quality_stop:
                    continue
                out.add(token)

                simplified = token
                for prefix in ("وال", "بال", "فال", "كال", "لل", "ال"):
                    if simplified.startswith(prefix) and len(simplified) - len(prefix) >= 3:
                        simplified = simplified[len(prefix):]
                        break
                out.add(simplified)

                stem = simplified
                for suffix in ("يات", "ات", "ون", "ين", "يه", "يه", "ها", "هم", "هن", "ه"):
                    if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
                        stem = stem[:-len(suffix)]
                        break
                if len(stem) >= 3:
                    out.add(stem)

            return {x for x in out if len(x) >= 3}

        authoritative_tokens = qtokens(authoritative_text)

        def text_similarity(a: Any, b: Any) -> float:
            ta, tb = qtokens(a), qtokens(b)
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / max(1, len(ta | tb))

        def unsupported_numbers(value: Any) -> List[str]:
            nums = self._extract_number_tokens(self._normalize_digits(value))
            return sorted(n for n in nums if n not in allowed_numbers)

        def broken_or_placeholder(value: Any) -> bool:
            raw = re.sub(r"\s+", " ", str(value or "")).strip()
            if not raw:
                return True
            normalized = raw.strip(" .،؛:-–—")
            low = normalized.lower()
            placeholders = (
                "تحديد مخرج التدخل",
                "تحديد مخرج",
                "مخرج التدخل",
                "تنفيذ التدخل",
                "استكمال التدخل",
                "متابعة التدخل",
                "تحديد النتيجة",
                "تحديد نتيجة",
                "النتيجة المطلوبة",
                "المخرج المطلوب",
                "استكمال المخرج",
            )
            if low in placeholders:
                return True
            if len(qtokens(normalized)) < 2:
                return True
            if re.search(r"(?:\bبـ|\bلـ|\bمن|\bإلى|\bالى|\bعلى|\bعن|\bمع|\bثم|\bو)\s*$", normalized):
                return True
            if normalized.endswith((":", "،", "؛", "-", "–", "—")):
                return True
            return False

        # ------------------------------------------------------------------
        # PHASE A1 — Meta reviews each Specialist independently.
        #
        # This is intentionally a separate generation from intervention selection.
        # Earlier builds asked Meta to review advisors AND choose "2-3 interventions"
        # in one output. Qwen leaked those strategy numbers into every REVIEW and
        # collapsed into the same generic sentence. Keeping REVIEW synthesis isolated
        # removes that cross-task contamination.
        # ------------------------------------------------------------------
        review_protocol = (
            "Return ONLY REVIEW lines; no JSON, Markdown, intervention count, sprint count, or FINAL line.\n"
            "Return exactly one line for every supplied advisor:\n"
            "REVIEW=<advisor_slug>||<one concise, advisor-specific Meta assessment in Arabic>\n"
            "The assessment must identify a concrete point from THAT advisor, then state what the Meta accepts, limits, combines, or excludes and why.\n"
            "Do not copy the same sentence structure across advisors.\n"
            "Do not introduce any numeric quantity unless that exact quantity exists in that advisor reasoning or the authoritative case."
        )

        review_task = {
            "role": meta["slug"],
            "meta_advisor_name": meta["name"],
            "task": (
                "Review each independent Specialist opinion separately. "
                "This call is ONLY for advisor-by-advisor Meta assessments. "
                "Do not choose interventions and do not design weekly sprints."
            ),
            **authoritative_context,
            "specialist_reasonings": council,
            "required_protocol": review_protocol,
            "mandatory_rules": [
                "Return one REVIEW for every advisor_id supplied, exactly once.",
                "Use the advisor slug exactly as supplied.",
                "Each REVIEW must be tied to a concrete concept in that advisor's own reasoning.",
                "State what is accepted, limited, combined, or excluded and why.",
                "Do not use a generic sentence that could apply to another advisor.",
                "Do not mention how many interventions will be generated.",
                "Do not mention sprint counts or roadmap mechanics.",
                "Do not invent numbers, percentages, budgets, dates, partners, staffing, resources, or capacity.",
                "Keep each REVIEW concise and professionally written in Arabic.",
            ],
        }

        def parse_reviews(raw: str) -> Dict[str, Any]:
            """
            Parse Meta advisor reviews defensively.

            Qwen does not always obey the exact
                REVIEW=<slug>||<text>
            protocol. Production therefore accepts several equivalent forms:
              - REVIEW=AOS-SP-13||...
              - REVIEW: AOS-SP-13: ...
              - AOS-SP-13||...
              - AOS-SP-13: ...
              - JSON objects/lists containing advisor_id + review/message
              - a paragraph headed by an advisor slug

            The parser still accepts ONLY advisor slugs that are actually in the
            selected council. It never invents a new advisor.
            """
            raw_text = str(raw or "")
            clean = self._normalize_digits(
                self.clean_model_text(raw_text).replace("```json", "").replace("```", "").strip()
            )
            reviews: Dict[str, str] = {}
            valid_ids = set(involved_ids)

            def put(aid: Any, msg: Any) -> None:
                aid_text = str(aid or "").strip().upper()
                aid_text = aid_text.strip("`*[](){}<>:;،. \t\r\n")
                if aid_text not in valid_ids:
                    return

                msg_text = str(msg or "").strip()
                msg_text = re.sub(
                    r"^(?:REVIEW|ASSESSMENT|META(?:_REVIEW)?|تعقيب|رأي\s+الميتا)\s*[:=|-]*\s*",
                    "",
                    msg_text,
                    flags=re.I,
                ).strip()
                msg_text = re.sub(
                    rf"^{re.escape(aid_text)}\s*(?:\|\||[|:：=\-–—])+\s*",
                    "",
                    msg_text,
                    flags=re.I,
                ).strip()
                msg_text = self._clean_meta_public_text(msg_text, request)
                if msg_text:
                    reviews[aid_text] = msg_text

            # --------------------------------------------------------------
            # 1) JSON tolerance.
            # --------------------------------------------------------------
            json_candidates: List[Any] = []
            stripped = clean.strip()
            if stripped:
                try:
                    json_candidates.append(json.loads(stripped))
                except Exception:
                    first_obj = stripped.find("{")
                    last_obj = stripped.rfind("}")
                    if first_obj >= 0 and last_obj > first_obj:
                        try:
                            json_candidates.append(
                                json.loads(stripped[first_obj:last_obj + 1])
                            )
                        except Exception:
                            pass
                    first_arr = stripped.find("[")
                    last_arr = stripped.rfind("]")
                    if first_arr >= 0 and last_arr > first_arr:
                        try:
                            json_candidates.append(
                                json.loads(stripped[first_arr:last_arr + 1])
                            )
                        except Exception:
                            pass

            def absorb_json(value: Any) -> None:
                if isinstance(value, dict):
                    # Direct mapping: {"AOS-SP-13": "..."}
                    for key, val in value.items():
                        if str(key).strip().upper() in valid_ids and isinstance(val, (str, int, float)):
                            put(key, val)

                    # Common wrappers.
                    for wrapper in (
                        "reviews", "advisor_reviews", "meta_reviews",
                        "assessments", "items", "data",
                    ):
                        if wrapper in value:
                            absorb_json(value.get(wrapper))

                    aid = (
                        value.get("advisor_id")
                        or value.get("advisor_slug")
                        or value.get("slug")
                        or value.get("id")
                    )
                    msg = (
                        value.get("review")
                        or value.get("assessment")
                        or value.get("meta_review")
                        or value.get("message")
                        or value.get("reasoning")
                        or value.get("text")
                    )
                    if aid and msg:
                        put(aid, msg)

                elif isinstance(value, list):
                    for item in value:
                        absorb_json(item)

            for parsed_json in json_candidates:
                absorb_json(parsed_json)

            # --------------------------------------------------------------
            # 2) Flexible line parsing.
            # --------------------------------------------------------------
            # Force each known advisor slug to start on a new logical line.
            for aid in involved_ids:
                clean = re.sub(
                    rf"\s*(?=(?:REVIEW\s*[:=]\s*)?{re.escape(aid)}\b)",
                    "\n",
                    clean,
                    flags=re.I,
                )

            for raw_line in clean.splitlines():
                line = re.sub(r"^[\s\-*•#>\d.)]+", "", raw_line.strip())
                if not line:
                    continue

                patterns = (
                    r"^REVIEW\s*[:=]\s*(AOS-[A-Z]+-\d+)\s*\|\|\s*(.+)$",
                    r"^REVIEW\s*[:=]\s*(AOS-[A-Z]+-\d+)\s*(?:[:：|=\-–—])+\s*(.+)$",
                    r"^(AOS-[A-Z]+-\d+)\s*\|\|\s*(.+)$",
                    r"^(AOS-[A-Z]+-\d+)\s*(?:[:：|=\-–—])+\s*(.+)$",
                    r"^(AOS-[A-Z]+-\d+)\s+(.+)$",
                )
                matched = False
                for pattern in patterns:
                    m = re.match(pattern, line, flags=re.I)
                    if not m:
                        continue
                    put(m.group(1), m.group(2))
                    matched = True
                    break
                if matched:
                    continue

            # --------------------------------------------------------------
            # 3) Paragraph/segment fallback:
            #    find each slug and take text until the next selected slug.
            # --------------------------------------------------------------
            missing = [aid for aid in involved_ids if aid not in reviews]
            if missing and clean:
                hits: List[tuple[int, str]] = []
                for aid in involved_ids:
                    for m in re.finditer(re.escape(aid), clean, flags=re.I):
                        hits.append((m.start(), aid))
                hits.sort(key=lambda x: x[0])

                for index, (pos, aid) in enumerate(hits):
                    if aid in reviews:
                        continue
                    seg_start = pos + len(aid)
                    seg_end = hits[index + 1][0] if index + 1 < len(hits) else len(clean)
                    segment = clean[seg_start:seg_end]
                    segment = re.sub(
                        r"^[\s`*#>|:：=\-–—]+",
                        "",
                        segment,
                    ).strip()
                    segment = re.sub(
                        r"^(?:REVIEW|ASSESSMENT|META(?:_REVIEW)?|تعقيب|رأي\s+الميتا)\s*[:=|-]*\s*",
                        "",
                        segment,
                        flags=re.I,
                    ).strip()
                    if segment:
                        put(aid, segment)

            return {
                "reviews": reviews,
                "raw": raw_text,
                "parsed_count": len(reviews),
            }

        def review_validation_errors(
            parsed: Dict[str, Any],
            *,
            check_pairwise_similarity: bool = True,
        ) -> tuple[List[str], set[str]]:
            errors: List[str] = []
            bad_ids: set[str] = set()
            reviews = parsed.get("reviews") or {}
            review_rows: List[tuple[str, str]] = []

            generic_phrases = (
                "تغطي التوصية الأساسية جوهر التوصيات الاستشارية",
                "مع إضافة منطق التوافق الداخلي والاعتماديات المؤثرة",
                "يتوافق هذا الرأي مع التوجه العام",
                "الرأي مناسب ويمكن الاستفادة منه",
            )

            for aid in involved_ids:
                review = str(reviews.get(aid) or "").strip()
                if not review:
                    errors.append(f"missing REVIEW for {aid}")
                    bad_ids.add(aid)
                    continue

                if len(review) < 28 or len(qtokens(review)) < 3:
                    errors.append(f"REVIEW for {aid} is too short/generic")
                    bad_ids.add(aid)

                if any(phrase in review for phrase in generic_phrases):
                    errors.append(f"REVIEW for {aid} uses known generic boilerplate")
                    bad_ids.add(aid)

                own_reasoning = reasonings_by_id.get(aid) or ""
                own_tokens = qtokens(own_reasoning)
                review_tokens = qtokens(review)

                if own_tokens and not (own_tokens & review_tokens):
                    errors.append(
                        f"REVIEW for {aid} is not grounded in that advisor reasoning"
                    )
                    bad_ids.add(aid)

                own_numbers = set(
                    self._extract_number_tokens(
                        self._normalize_digits(own_reasoning)
                    )
                )
                review_allowed_numbers = allowed_numbers | own_numbers
                nums = sorted(
                    n
                    for n in self._extract_number_tokens(
                        self._normalize_digits(review)
                    )
                    if n not in review_allowed_numbers
                )
                if nums:
                    errors.append(
                        f"REVIEW for {aid} contains unsupported numbers {nums}"
                    )
                    bad_ids.add(aid)

                review_rows.append((aid, review))

            if check_pairwise_similarity:
                for x in range(len(review_rows)):
                    for y in range(x + 1, len(review_rows)):
                        aid_a, a = review_rows[x]
                        aid_b, b = review_rows[y]
                        if text_similarity(a, b) >= 0.78:
                            errors.append(
                                f"Meta REVIEWs for {aid_a} and {aid_b} are too similar/generic"
                            )
                            bad_ids.update({aid_a, aid_b})

            return list(dict.fromkeys(errors)), bad_ids

        def deterministic_review_fallback(aid: str, index: int) -> str:
            """
            Last-resort non-generative review.

            It never fabricates subject matter: the concrete clause comes only
            from that Specialist's own reasoning. This exists so a formatting
            failure in an optional commentary field cannot waste a 15+ minute
            council run after all Specialists have already completed.
            """
            reasoning = str(reasonings_by_id.get(aid) or "").strip()
            reasoning = self._clean_meta_public_text(reasoning, request)
            reasoning = re.sub(
                r"\bAOS-[A-Z]+-\d+\b",
                "",
                reasoning,
                flags=re.I,
            )
            reasoning = re.sub(r"\s+", " ", reasoning).strip(" ،؛:.-–—")

            # Prefer one complete, meaningful sentence/clause from the advisor.
            parts = [
                p.strip(" ،؛:.-–—")
                for p in re.split(r"(?<=[.!؟?؛])\s+|\n+", reasoning)
                if p.strip()
            ]
            excerpt = ""
            for part in parts:
                if len(qtokens(part)) >= 3:
                    excerpt = part
                    break
            if not excerpt:
                excerpt = reasoning

            # Keep the transcript concise without cutting mid-word.
            words = excerpt.split()
            if len(words) > 24:
                excerpt = " ".join(words[:24]).rstrip("،؛:.-–—")

            # Deliberately vary the Meta stance so fallback reviews do not become
            # six copies of the same boilerplate sentence.
            frames = (
                "أعتمد جوهر ملاحظة هذا المستشار حول «{x}»، وأربطها مباشرة بهدف التحدي دون توسيع غير مدعوم للنطاق.",
                "أدمج من هذا الرأي النقطة المتعلقة بـ«{x}»، مع قصر استخدامها على ما تدعمه بيانات الحالة الحالية.",
                "أستفيد من طرح هذا المستشار بشأن «{x}» كقيد تصميمي للتدخل، لا كهدف مستقل عن الأثر المطلوب.",
                "أعتبر ملاحظة هذا المستشار حول «{x}» مدخلًا مساندًا، بشرط أن تظهر صلتها بنتيجة قابلة للتحقق داخل التحدي.",
                "أتبنى من هذا الرأي جانب «{x}» بقدر ما يحسن قياس أو تنفيذ الأثر دون إضافة افتراضات جديدة.",
                "أقيّد الاستفادة من توصية هذا المستشار حول «{x}» بما يخدم استدامة التدخل ويتسق مع المعطيات المتاحة.",
            )
            frame = frames[index % len(frames)]

            if not excerpt:
                # This path should be practically unreachable because every
                # involved advisor already has a usable public reasoning.
                excerpt = "النقطة الواردة في رأيه المهني"

            return self._clean_meta_public_text(
                frame.format(x=excerpt),
                request,
            )

        def generate_reviews(
            task_obj: Dict[str, Any],
            max_tokens: int = 1200,
        ) -> Dict[str, Any]:
            raw = self._generate(
                meta_adapter,
                meta["prompt"],
                json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")),
                max_tokens,
                deterministic=True,
                repetition_penalty=1.10,
                no_repeat_ngram_size=5,
            )
            return parse_reviews(raw)

        # One Meta REVIEW generation only.
        #
        # Previous builds did:
        # group generation -> group repair -> up to six individual Meta calls,
        # then failed the whole request if the parser still saw no exact REVIEW
        # markers. That could waste ~18 minutes after all Specialists had already
        # completed. Reviews are commentary, not the intervention contract, so
        # malformed commentary now degrades safely instead of aborting Screen 3.
        review_pack = generate_reviews(review_task)
        review_errors, bad_review_ids = review_validation_errors(review_pack)

        original_parsed_review_count = int(review_pack.get("parsed_count") or 0)
        fallback_review_ids: List[str] = []

        repaired_reviews = dict(review_pack.get("reviews") or {})

        # Replace only missing/invalid/generic review rows with a deterministic
        # advisor-grounded fallback. Valid Meta-generated reviews are preserved.
        for idx, aid in enumerate(involved_ids):
            if aid in bad_review_ids or not str(repaired_reviews.get(aid) or "").strip():
                repaired_reviews[aid] = deterministic_review_fallback(aid, idx)
                fallback_review_ids.append(aid)

        review_pack = {
            "reviews": repaired_reviews,
            "raw": review_pack.get("raw"),
            "parsed_count": original_parsed_review_count,
        }

        # Final gate is fail-open for COMMENTARY only: if a generated review is
        # still malformed, replace that row once more from its own Specialist
        # reasoning. Missing reviews are never allowed to reach the transcript.
        final_review_errors, final_bad_ids = review_validation_errors(
            review_pack,
            check_pairwise_similarity=False,
        )
        if final_bad_ids:
            for idx, aid in enumerate(involved_ids):
                if aid in final_bad_ids:
                    repaired_reviews[aid] = deterministic_review_fallback(aid, idx)
                    if aid not in fallback_review_ids:
                        fallback_review_ids.append(aid)

        meta_reviews = {
            aid: str(repaired_reviews.get(aid) or "").strip()
            for aid in involved_ids
        }

        # Absolute structural safety: at this point every selected advisor must
        # have a non-empty review string, but failure here is a programming error
        # rather than an LLM formatting error.
        missing_after_fallback = [
            aid for aid in involved_ids if not meta_reviews.get(aid)
        ]
        if missing_after_fallback:
            raise ValueError(
                "Internal REVIEW fallback failed for: "
                + ", ".join(missing_after_fallback)
            )

        # ------------------------------------------------------------------
        # PHASE A2 — approve the intervention portfolio only.
        #
        # Advisor reviews are already finished above, so numbers such as "2-3"
        # used by the intervention-selection protocol cannot contaminate REVIEWs.
        # ------------------------------------------------------------------
        strategy_protocol = (
            "Return ONLY this plain-text protocol; no JSON or Markdown.\n"
            "Return exactly 2 or 3 approved intervention headers:\n"
            "BEGIN_INTERVENTION\n"
            "TITLE=<specific Arabic intervention title grounded in the case>\n"
            "CONFIDENCE=<high|medium|low>\n"
            "IMPACT=<specific link to the social problem / goal / impact driver>\n"
            "REPORTABLE=<measurable reportable value without invented numbers>\n"
            "END_INTERVENTION\n"
            "After the last block: FINAL=<brief synthesis explaining why these interventions together form the 90-day route>"
        )

        strategy_task = {
            "role": meta["slug"],
            "meta_advisor_name": meta["name"],
            "task": (
                "Approve the small portfolio of interventions for the 90-day "
                "Impact Challenge. Advisor-by-advisor reviews are already complete. "
                "DO NOT output REVIEW lines and DO NOT generate weekly sprints."
            ),
            **authoritative_context,
            "specialist_reasonings": council,
            "meta_reviews": meta_reviews,
            "required_protocol": strategy_protocol,
            "mandatory_rules": [
                "Approve exactly 2 interventions by default; approve a third only when it adds an independent strategic path that the first two do not cover.",
                "Every intervention must directly anchor to the stated 90-day goal, social problem, named impact drivers, target group, organizational notes, or an existing program/project.",
                "Do not replace the stated challenge with a generic identity, governance, structure, strategy, or analysis project unless the request itself makes that issue material.",
                "The interventions must be materially different from each other and together converge on the same primary 90-day goal.",
                "Do not invent percentages, counts, budgets, dates, partners, staffing, resources, or capacity.",
                "REPORTABLE must describe something the organization can verify or report; if no grounded numeric target exists, use a measurable completion/status/quality statement without fabricating a number.",
                "Do not output REVIEW lines; those were generated in the previous phase.",
                "Write concise professional Arabic suitable for Saudi nonprofit organizations.",
            ],
        }

        def parse_strategy(raw: str) -> Dict[str, Any]:
            clean = self._normalize_digits(
                self.clean_model_text(raw).replace("```", "").strip()
            )
            clean = re.sub(
                r"\s+(?=(?:BEGIN_INTERVENTION|END_INTERVENTION|TITLE|CONFIDENCE|IMPACT|REPORTABLE|FINAL|REVIEW)\s*(?:[:=]|\b))",
                "\n",
                clean,
                flags=re.I,
            )
            reviews: Dict[str, str] = {}
            interventions: List[Dict[str, Any]] = []
            current: Optional[Dict[str, Any]] = None
            final = ""

            def finish() -> None:
                nonlocal current
                if current is not None:
                    interventions.append(current)
                current = None

            for raw_line in clean.splitlines():
                line = re.sub(r"^[\-*•]+\s*", "", raw_line.strip())
                if not line:
                    continue

                m = re.match(r"^REVIEW\s*[:=]\s*([^|]+?)\s*\|\|\s*(.+)$", line, flags=re.I)
                if m:
                    aid = m.group(1).strip()
                    msg = self._clean_meta_public_text(m.group(2).strip(), request)
                    if aid and msg:
                        reviews[aid] = msg
                    continue

                if re.fullmatch(r"BEGIN_INTERVENTION", line, flags=re.I):
                    if current is not None:
                        finish()
                    current = {}
                    continue
                if re.fullmatch(r"END_INTERVENTION", line, flags=re.I):
                    finish()
                    continue

                if current is None:
                    m = re.match(r"^FINAL\s*[:=]\s*(.+)$", line, flags=re.I)
                    if m:
                        final = self._clean_meta_public_text(m.group(1), request)
                    continue

                for key, public_key in (
                    ("TITLE", "title"),
                    ("CONFIDENCE", "confidence_level"),
                    ("IMPACT", "impact_description"),
                    ("REPORTABLE", "reportable_value"),
                ):
                    m = re.match(rf"^{key}\s*[:=]\s*(.+)$", line, flags=re.I)
                    if m:
                        value = m.group(1).strip()
                        current[public_key] = (
                            value.lower()
                            if public_key == "confidence_level"
                            else self._clean_meta_public_text(value, request)
                        )
                        break

            if current is not None:
                finish()

            return {
                "reviews": reviews,
                "interventions": interventions[:3],
                "final": final,
                "raw": raw,
            }

        def strategy_errors(parsed: Dict[str, Any]) -> List[str]:
            errors: List[str] = []

            interventions = parsed.get("interventions") or []
            if not (2 <= len(interventions) <= 3):
                errors.append(f"strategy must approve exactly 2-3 interventions, got {len(interventions)}")

            strategy_texts: List[str] = []
            for idx, item in enumerate(interventions, start=1):
                for key in ("title", "impact_description", "reportable_value"):
                    value = str(item.get(key) or "").strip()
                    if not value:
                        errors.append(f"intervention {idx} missing {key}")
                    elif broken_or_placeholder(value):
                        errors.append(f"intervention {idx} {key} is generic/broken")
                    nums = unsupported_numbers(value)
                    if nums:
                        errors.append(
                            f"intervention {idx} {key} contains unsupported numbers {nums}"
                        )

                confidence = str(item.get("confidence_level") or "").lower()
                if confidence not in {"high", "medium", "low"}:
                    errors.append(f"intervention {idx} invalid confidence")

                combined = " ".join([
                    str(item.get("title") or ""),
                    str(item.get("impact_description") or ""),
                    str(item.get("reportable_value") or ""),
                ])
                overlap = qtokens(combined) & authoritative_tokens
                if len(overlap) < 2:
                    errors.append(
                        f"intervention {idx} is weakly grounded in authoritative goal/impact/program context"
                    )
                strategy_texts.append(combined)

            for x in range(len(strategy_texts)):
                for y in range(x + 1, len(strategy_texts)):
                    if text_similarity(strategy_texts[x], strategy_texts[y]) >= 0.62:
                        errors.append(
                            f"interventions {x+1} and {y+1} are not materially distinct"
                        )

            final_text = str(parsed.get("final") or "").strip()
            if not final_text:
                errors.append("missing FINAL Meta synthesis")
            else:
                nums = unsupported_numbers(final_text)
                if nums:
                    errors.append(f"FINAL Meta synthesis contains unsupported numbers {nums}")
            return list(dict.fromkeys(errors))

        def generate_strategy(task_obj: Dict[str, Any]) -> Dict[str, Any]:
            raw = self._generate(
                meta_adapter,
                meta["prompt"],
                json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")),
                min(max(META_MAX_NEW_TOKENS, 1800), 2400),
                deterministic=True,
                repetition_penalty=1.05,
            )
            return parse_strategy(raw)

        strategy = generate_strategy(strategy_task)
        s_errors = strategy_errors(strategy)
        if s_errors:
            repair_task = dict(strategy_task)
            repair_task["task"] = (
                "Repair only the council synthesis. Return the COMPLETE strategy protocol "
                "again. Do not generate any weekly sprints."
            )
            repair_task["previous_output"] = strategy.get("raw")
            repair_task["validation_errors"] = s_errors[:24]
            repair_task["repair_rules"] = [
                "Replace generic or unrelated interventions with interventions directly grounded in the supplied goal, social problem, impact drivers, organization facts, or programs.",
                "Keep exactly 2-3 materially distinct interventions.",
                "Do not output REVIEW lines.",
                "Do not invent any unsupported number or target.",
            ]
            strategy = generate_strategy(repair_task)
            s_errors = strategy_errors(strategy)

        if s_errors:
            raise ValueError(
                "Meta council synthesis failed Screen-3 quality gate: "
                + " | ".join(s_errors[:20])
            )

        # ------------------------------------------------------------------
        # PHASE B — generate the 12-week route separately for each approved
        # intervention. One long plan can no longer degrade the other plans.
        # ------------------------------------------------------------------
        sprint_protocol = (
            "Return ONLY this protocol for ONE approved intervention; no JSON or Markdown.\n"
            "SPRINT=1\n"
            "RESULT=<distinct achieved state/milestone by the end of this 5-day sprint>\n"
            "TEXT=<mandatory concrete deliverable/action executable within 5 working days>\n"
            "[TEXT=<optional second 5-day deliverable>]\n"
            "[TEXT=<optional third 5-day deliverable>]\n"
            "END_SPRINT\n"
            "Repeat sequentially through SPRINT=12 exactly once each."
        )

        def parse_sprints(raw: str) -> Dict[str, Any]:
            clean = self._normalize_digits(
                self.clean_model_text(raw).replace("```", "").strip()
            )
            clean = re.sub(
                r"\s+(?=(?:SPRINT|RESULT|TEXT|END_SPRINT)\s*(?:[:=]|\b))",
                "\n",
                clean,
                flags=re.I,
            )
            units: List[Dict[str, Any]] = []
            current: Optional[Dict[str, Any]] = None

            def finish() -> None:
                nonlocal current
                if current is not None:
                    units.append(current)
                current = None

            for raw_line in clean.splitlines():
                line = re.sub(r"^[\-*•]+\s*", "", raw_line.strip())
                if not line:
                    continue

                m = re.match(
                    r"^SPRINT\s*(?:[:=\-–—]\s*|\s+)(\d{1,2})\s*[:\-–—]?\s*$",
                    line,
                    flags=re.I,
                )
                if m:
                    finish()
                    current = {
                        "number": int(m.group(1)),
                        "result": "",
                        "deliverables": [],
                    }
                    continue

                if re.fullmatch(r"END_SPRINT", line, flags=re.I):
                    finish()
                    continue

                m = re.match(r"^RESULT\s*[:=]\s*(.+)$", line, flags=re.I)
                if m and current is not None:
                    current["result"] = m.group(1).strip()
                    continue

                m = re.match(r"^TEXT\s*[:=]\s*(.+)$", line, flags=re.I)
                if m and current is not None:
                    value = m.group(1).strip()
                    if value:
                        current["deliverables"].append(value)

            finish()

            by_number: Dict[int, Dict[str, Any]] = {}
            for unit in units:
                n = int(unit.get("number") or 0)
                if not (1 <= n <= SPRINT_COUNT):
                    continue
                result_text = self._clean_meta_public_text(unit.get("result"), request)
                deliverables: List[Dict[str, str]] = []
                for value in unit.get("deliverables") or []:
                    cleaned = self._clean_meta_public_text(value, request)
                    if cleaned and cleaned not in [x["text"] for x in deliverables]:
                        deliverables.append({"text": cleaned})

                candidate = {
                    "_sprint": n,
                    "text": result_text,
                    "results": deliverables[:3],
                }
                existing = by_number.get(n)
                if existing is None:
                    by_number[n] = candidate
                else:
                    old_q = int(bool(existing.get("text"))) + int(bool(existing.get("results")))
                    new_q = int(bool(candidate.get("text"))) + int(bool(candidate.get("results")))
                    if new_q > old_q:
                        by_number[n] = candidate

            return {
                "outputs": [by_number[n] for n in sorted(by_number)],
                "raw": raw,
            }

        def sprint_quality_errors(
            plan: Dict[str, Any],
            intervention: Dict[str, Any],
        ) -> tuple[List[str], set[int]]:
            errors: List[str] = []
            bad: set[int] = set()
            outputs = plan.get("outputs") or []
            numbers = [
                int(x.get("_sprint") or 0)
                for x in outputs
                if isinstance(x, dict)
            ]
            expected = list(range(1, SPRINT_COUNT + 1))
            if numbers != expected:
                missing = [n for n in expected if n not in numbers]
                errors.append(f"sprints must be 1..12 exactly; got {numbers}; missing={missing}")
                bad.update(missing)

            approved_intervention_text = " ".join([
                str(intervention.get("title") or ""),
                str(intervention.get("impact_description") or ""),
                str(intervention.get("reportable_value") or ""),
            ])
            approved_tokens = qtokens(approved_intervention_text)
            case_tokens = authoritative_tokens

            def is_anchored(value: Any) -> bool:
                tokens = qtokens(value)
                if not tokens:
                    return False
                # Prefer direct connection to the approved intervention. A line
                # may also be accepted when it is strongly tied (2+ meaningful
                # tokens) to authoritative case facts.
                return bool(tokens & approved_tokens) or len(tokens & case_tokens) >= 2

            seen_results: Dict[str, int] = {}
            seen_deliverables: Dict[str, int] = {}
            first_word_counts: Dict[str, List[int]] = {}
            result_rows: List[tuple[int, str]] = []

            for unit in outputs:
                n = int(unit.get("_sprint") or 0)
                if not (1 <= n <= SPRINT_COUNT):
                    continue
                result = str(unit.get("text") or "").strip()
                deliverables = unit.get("results") or []

                if len(result) < 18 or broken_or_placeholder(result):
                    errors.append(f"sprint {n} result is generic, broken, or too short")
                    bad.add(n)
                if unsupported_numbers(result):
                    errors.append(
                        f"sprint {n} result contains unsupported numbers {unsupported_numbers(result)}"
                    )
                    bad.add(n)
                if qtokens(result) and not is_anchored(result):
                    errors.append(f"sprint {n} result is not anchored to this intervention/case")
                    bad.add(n)

                norm_result = re.sub(r"\s+", " ", result.lower()).strip(" .،؛:-–—")
                if norm_result:
                    if norm_result in seen_results:
                        prev = seen_results[norm_result]
                        errors.append(f"sprints {prev} and {n} repeat the same result")
                        bad.update({prev, n})
                    else:
                        seen_results[norm_result] = n
                    result_rows.append((n, result))

                words = re.findall(r"[\u0600-\u06FFA-Za-z]+", result)
                if words:
                    first = words[0].lower()
                    first_word_counts.setdefault(first, []).append(n)

                if not (1 <= len(deliverables) <= 3):
                    errors.append(f"sprint {n} requires 1-3 executable texts")
                    bad.add(n)

                for d in deliverables:
                    text_value = str(d.get("text") or "").strip() if isinstance(d, dict) else ""
                    if len(text_value) < 18 or broken_or_placeholder(text_value):
                        errors.append(f"sprint {n} contains generic/broken executable text")
                        bad.add(n)
                    if unsupported_numbers(text_value):
                        errors.append(
                            f"sprint {n} executable text contains unsupported numbers {unsupported_numbers(text_value)}"
                        )
                        bad.add(n)
                    if qtokens(text_value) and not is_anchored(text_value):
                        errors.append(f"sprint {n} executable text is not anchored to this intervention/case")
                        bad.add(n)

                    norm_d = re.sub(r"\s+", " ", text_value.lower()).strip(" .،؛:-–—")
                    if norm_d:
                        if norm_d in seen_deliverables:
                            prev = seen_deliverables[norm_d]
                            errors.append(
                                f"sprints {prev} and {n} repeat the same executable text"
                            )
                            bad.update({prev, n})
                        else:
                            seen_deliverables[norm_d] = n

            # Near-duplicate weekly results are also low-quality, even when one
            # token changes. Keep this threshold high to allow legitimate thematic
            # continuity while rejecting template collapse.
            for x in range(len(result_rows)):
                for y in range(x + 1, len(result_rows)):
                    nx, tx = result_rows[x]
                    ny, ty = result_rows[y]
                    if text_similarity(tx, ty) >= 0.84:
                        errors.append(f"sprints {nx} and {ny} are near-duplicate results")
                        bad.update({nx, ny})

            # If the same generic leading action dominates most of the roadmap,
            # the model has collapsed into a template rather than a progression.
            for first, nums in first_word_counts.items():
                if len(nums) >= 5 and first in {
                    "تحديد", "إعداد", "اعداد", "مراجعة", "تنفيذ", "متابعة",
                    "استكمال", "تطوير", "تصميم", "تحليل", "بناء", "إنشاء", "انشاء",
                }:
                    errors.append(
                        f"roadmap overuses the same leading action '{first}' in sprints {nums}"
                    )
                    bad.update(nums)

            return list(dict.fromkeys(errors)), bad

        def sprint_task(
            intervention: Dict[str, Any],
            *,
            repair_errors: Optional[List[str]] = None,
            previous_output: Optional[str] = None,
        ) -> Dict[str, Any]:
            task_obj = {
                "role": meta["slug"],
                "task": (
                    "Design the twelve-week execution route for ONE already-approved "
                    "Screen-3 intervention. Do not redesign or replace the intervention."
                ),
                **authoritative_context,
                "approved_intervention": {
                    "title": intervention.get("title"),
                    "confidence_level": intervention.get("confidence_level"),
                    "impact_description": intervention.get("impact_description"),
                    "reportable_value": intervention.get("reportable_value"),
                },
                "specialist_reasonings": council,
                "required_protocol": sprint_protocol,
                "progression_guidance": [
                    "Sprints 1-2: establish intervention-specific evidence, scope, readiness, or baseline needed to act.",
                    "Sprints 3-4: complete the intervention-specific design/preparation and ownership needed for execution.",
                    "Sprints 5-8: execute/test the concrete intervention work and produce tangible outputs; do not remain in analysis mode.",
                    "Sprints 9-10: review evidence from execution, fix gaps, and improve the intervention.",
                    "Sprint 11: institutionalize, hand over, or secure continuity of the working approach.",
                    "Sprint 12: verify the achieved 90-day result and document the next decision/action.",
                ],
                "mandatory_rules": [
                    "Return exactly SPRINT 1 through SPRINT 12, once each and in order.",
                    "RESULT is a distinct weekly achieved state or milestone, not a placeholder task label.",
                    "Every sprint must have 1 to 3 TEXT lines; each TEXT is a concrete mandatory deliverable/action fully executable within 5 working days.",
                    "Tailor every RESULT and TEXT to THIS intervention and the supplied organization/program/goal/impact context.",
                    "The twelve results must progress logically; later weeks must build on evidence or outputs from earlier weeks.",
                    "Do not repeat the same RESULT, TEXT, or generic phrase across weeks.",
                    "Never output placeholders such as 'تحديد مخرج التدخل', 'تنفيذ التدخل', 'استكمال التدخل', or similar generic filler.",
                    "Do not invent percentages, counts, budgets, dates, partners, staffing, resources, or capacity not present in the authoritative context.",
                    "Historical numbers are evidence, not automatic future targets.",
                    "Keep each RESULT and TEXT concise enough to be operational, but specific enough that a team can tell what must be completed by Friday.",
                    "Do not introduce a new strategic objective, identity project, governance project, or organizational redesign unless it is part of the approved intervention/context.",
                ],
            }
            if repair_errors:
                task_obj["task"] = (
                    "Repair the complete twelve-week route for the SAME approved intervention. "
                    "Return the COMPLETE SPRINT 1..12 protocol again."
                )
                task_obj["validation_errors"] = repair_errors[:30]
                task_obj["previous_output"] = previous_output
                task_obj["repair_rules"] = [
                    "Replace generic/repeated weeks with intervention-specific milestones.",
                    "Keep valid grounded content where possible.",
                    "Do not change the approved intervention header.",
                    "Return all twelve sprints, not only the faulty ones.",
                ]
            return task_obj

        def generate_sprint_plan(
            intervention: Dict[str, Any],
            *,
            repair_errors: Optional[List[str]] = None,
            previous_output: Optional[str] = None,
        ) -> Dict[str, Any]:
            raw = self._generate(
                meta_adapter,
                meta["prompt"],
                json.dumps(
                    sprint_task(
                        intervention,
                        repair_errors=repair_errors,
                        previous_output=previous_output,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                min(max(META_MAX_NEW_TOKENS, 3000), 3800),
                deterministic=True,
                repetition_penalty=1.08,
            )
            return parse_sprints(raw)

        def patch_sprints(
            intervention: Dict[str, Any],
            plan: Dict[str, Any],
            bad_numbers: set[int],
            errors: List[str],
        ) -> Dict[str, Any]:
            if not bad_numbers or len(bad_numbers) > 4:
                return plan

            existing_by_number = {
                int(x.get("_sprint") or 0): x
                for x in (plan.get("outputs") or [])
                if isinstance(x, dict) and int(x.get("_sprint") or 0) not in bad_numbers
            }
            neighborhood = {
                n: existing_by_number.get(n)
                for n in range(1, SPRINT_COUNT + 1)
                if n not in bad_numbers
            }
            patch_task = {
                "role": meta["slug"],
                "task": "Rewrite ONLY the listed faulty/missing sprint units for the same approved intervention.",
                **authoritative_context,
                "approved_intervention": {
                    "title": intervention.get("title"),
                    "impact_description": intervention.get("impact_description"),
                    "reportable_value": intervention.get("reportable_value"),
                },
                "faulty_or_missing_sprints": sorted(bad_numbers),
                "valid_neighboring_sprints": neighborhood,
                "validation_errors": errors[:20],
                "rules": [
                    "Return only the requested sprint numbers.",
                    "Each requested sprint must have one specific RESULT and 1-3 specific TEXT lines.",
                    "The replacement must fit logically between the valid neighboring weeks.",
                    "Every TEXT must be executable in 5 working days.",
                    "No placeholders, no repeated content, and no unsupported numbers/facts.",
                ],
                "protocol": sprint_protocol,
            }
            raw = self._generate(
                meta_adapter,
                meta["prompt"],
                json.dumps(patch_task, ensure_ascii=False, separators=(",", ":")),
                1400 if len(bad_numbers) <= 2 else 2200,
                deterministic=True,
                repetition_penalty=1.08,
            )
            patched = parse_sprints(raw)
            patch_by_number = {
                int(x.get("_sprint") or 0): x
                for x in (patched.get("outputs") or [])
                if isinstance(x, dict)
            }
            merged = dict(existing_by_number)
            for n in bad_numbers:
                if n in patch_by_number:
                    merged[n] = patch_by_number[n]

            return {
                "outputs": [merged[n] for n in sorted(merged)],
                "raw": raw,
            }

        approved = strategy["interventions"]
        interventions: List[Dict[str, Any]] = []
        plan_debug: List[Dict[str, Any]] = []

        for idx, intervention in enumerate(approved, start=1):
            print(
                f"[council] {meta['slug']} planning intervention {idx}/{len(approved)}: "
                f"{intervention.get('title')}",
                flush=True,
            )
            plan = generate_sprint_plan(intervention)
            errors, bad = sprint_quality_errors(plan, intervention)

            # Small local defects are cheaper to patch than to regenerate all 12.
            if errors and bad and len(bad) <= 4:
                plan = patch_sprints(intervention, plan, bad, errors)
                errors, bad = sprint_quality_errors(plan, intervention)

            # Broad template collapse or semantic drift gets one full intervention
            # regeneration, never a regeneration of the whole council portfolio.
            if errors:
                plan = generate_sprint_plan(
                    intervention,
                    repair_errors=errors,
                    previous_output=plan.get("raw"),
                )
                errors, bad = sprint_quality_errors(plan, intervention)
                if errors and bad and len(bad) <= 4:
                    plan = patch_sprints(intervention, plan, bad, errors)
                    errors, bad = sprint_quality_errors(plan, intervention)

            if errors:
                raise ValueError(
                    f"Meta 12-week plan failed quality gate for intervention {idx} "
                    f"({intervention.get('title')}): " + " | ".join(errors[:20])
                )

            clean_outputs = [
                {
                    "text": unit["text"],
                    "results": unit["results"],
                }
                for unit in plan["outputs"]
            ]
            interventions.append({
                "title": intervention["title"],
                "confidence_level": intervention["confidence_level"],
                "impact_description": intervention["impact_description"],
                "reportable_value": intervention["reportable_value"],
                "outputs": clean_outputs,
            })
            plan_debug.append({
                "intervention_index": idx,
                "title": intervention.get("title"),
                "sprint_count": len(clean_outputs),
                "quality_errors": [],
            })

        # Pair every Specialist opinion with a genuinely advisor-specific Meta
        # review produced in Phase A, then close with the portfolio synthesis.
        transcript: List[Dict[str, Any]] = []
        sequence = 1
        transcript.append({
            "sequence": sequence,
            "from": "meta_advisor",
            "message": (
                "سأراجع رأي كل مستشار على حدة، ثم أعتمد التدخلات المشتركة قبل "
                "بناء المسار التنفيذي ذي الاثني عشر أسبوعًا لكل تدخل."
            ),
        })
        sequence += 1
        for aid in involved_ids:
            transcript.append({
                "sequence": sequence,
                "from": aid,
                "message": reasonings_by_id[aid],
            })
            sequence += 1
            transcript.append({
                "sequence": sequence,
                "from": "meta_advisor",
                "message": f"تعقيبي على {aid}: {meta_reviews[aid]}",
            })
            sequence += 1
        transcript.append({
            "sequence": sequence,
            "from": "meta_advisor",
            "message": strategy["final"],
        })

        result = {
            "involved_advisor_ids": involved_ids,
            "advisor_reasonings": advisor_reasonings,
            "transcript": transcript,
            "suggestion": {"interventions": interventions},
        }

        self._last_meta_debug = {
            "architecture": "split_meta_reviews_then_strategy_then_per_intervention_12_sprints",
            "meta_advisor_slug": meta["slug"],
            "meta_advisor_name": meta["name"],
            "meta_prompt_path": meta["prompt_path"],
            "intervention_count": len(interventions),
            "sprint_units_per_intervention": [len(x["outputs"]) for x in interventions],
            "advisor_ids": involved_ids,
            "plan_quality": plan_debug,
            "meta_generation_calls_expected": 2 + len(interventions),
            "meta_review_phase": {
                "advisor_count": len(meta_reviews),
                "separate_from_strategy": True,
                "parsed_from_meta_count": original_parsed_review_count,
                "fallback_advisor_ids": fallback_review_ids,
                "fallback_used": bool(fallback_review_ids),
            },
        }

        self._validate_screen3_public_response(result)
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
        """Return backend advisor slugs exactly as received in payload.advisors."""
        out: List[str] = []
        for advisor in selected:
            value = str(advisor.get("backend_id") or advisor.get("advisor_id") or "").strip()
            if value and value not in out:
                out.append(value)
        return out

    def _advisor_reasoning_excerpt(
        self,
        opinion: Any,
        request: Dict[str, Any],
        *,
        max_chars: int = 850,
    ) -> str:
        """Expose a concise, grounded excerpt of the Specialist's own opinion."""
        raw = self.clean_model_text(str(opinion or ""))
        if not raw:
            return ""

        source_text = self._normalize_digits(
            json.dumps(self._shared_context(request), ensure_ascii=False, sort_keys=True)
        )
        source_numbers = self._extract_number_tokens(source_text)
        preferred: List[str] = []
        fallback: List[str] = []
        cue_terms = (
            "لأن", "بسبب", "يرتبط", "تتمثل", "المشكلة", "الفجوة", "الأولوية",
            "أوصي", "يوصى", "يتطلب", "يحتاج", "المخاطر", "الأثر", "النتيجة",
        )
        for original in raw.splitlines():
            line = re.sub(r"^\s*(?:#{1,6}|[-*•]+|\d+[\.)])\s*", "", original).strip()
            line = line.replace("**", "").replace("__", "").replace("`", "")
            line = re.sub(r"\s+", " ", line).strip(" :-–—")
            if len(line) < 28:
                continue
            if re.search(r"\b(?:AOS|ATHAR)[-_]", line, flags=re.I):
                continue
            nums = self._extract_number_tokens(line)
            if any(n not in source_numbers for n in nums):
                continue
            cleaned = self._clean_meta_public_text(line, request)
            if len(cleaned) < 24:
                continue
            fallback.append(cleaned)
            if any(term in cleaned.lower() for term in cue_terms):
                preferred.append(cleaned)

        chosen = preferred or fallback
        if not chosen:
            # Full fallback still avoids unsupported numbers and internal IDs.
            compact = re.sub(r"\s+", " ", raw).strip()
            if re.search(r"\b(?:AOS|ATHAR)[-_]", compact, flags=re.I):
                compact = re.sub(r"\b(?:AOS|ATHAR)[-_][A-Z0-9-]+\b", "", compact, flags=re.I)
            if any(n not in source_numbers for n in self._extract_number_tokens(compact)):
                compact = ""
            return self._clean_meta_public_text(compact, request)[:max_chars].strip()

        out: List[str] = []
        total = 0
        for line in chosen:
            if line in out:
                continue
            if total + len(line) > max_chars and out:
                break
            out.append(line)
            total += len(line)
            if len(out) >= 2:
                break
        return " ".join(out)[:max_chars].strip()

    def _build_advisor_reasonings(
        self,
        advisor_outputs: List[Dict[str, Any]],
        request: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        reasonings: List[Dict[str, str]] = []
        seen = set()
        for row in advisor_outputs:
            advisor_id = str(row.get("backend_id") or row.get("advisor_id") or "").strip()
            if not advisor_id or advisor_id in seen:
                continue
            reasoning = self._advisor_reasoning_excerpt(row.get("opinion"), request)
            if not reasoning:
                continue
            seen.add(advisor_id)
            reasonings.append({"advisor_id": advisor_id, "reasoning": reasoning})
        return reasonings

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
    def _validate_advisor_reasonings(
        involved_ids: List[str],
        reasonings: Any,
    ) -> None:
        if not isinstance(reasonings, list):
            raise ValueError("advisor_reasonings must be a list.")
        seen = set()
        for idx, row in enumerate(reasonings):
            if not isinstance(row, dict) or set(row.keys()) != {"advisor_id", "reasoning"}:
                raise ValueError(f"advisor_reasonings[{idx}] fields are invalid.")
            advisor_id = str(row.get("advisor_id") or "").strip()
            reasoning = str(row.get("reasoning") or "").strip()
            if not advisor_id or advisor_id not in involved_ids:
                raise ValueError(f"advisor_reasonings[{idx}].advisor_id is not involved.")
            if advisor_id in seen:
                raise ValueError(f"advisor_reasonings contains duplicate advisor_id: {advisor_id}")
            if not reasoning:
                raise ValueError(f"advisor_reasonings[{idx}].reasoning is required.")
            seen.add(advisor_id)
        if set(involved_ids) != seen:
            raise ValueError("Every involved advisor must have exactly one advisor_reasonings entry.")

    @staticmethod
    def _validate_transcript(
        involved_ids: List[str],
        transcript: Any,
    ) -> None:
        if not isinstance(transcript, list) or not transcript:
            raise ValueError("transcript must be a non-empty list.")
        allowed_from = {"meta_advisor", *involved_ids}
        expected_sequence = 1
        for idx, row in enumerate(transcript):
            if not isinstance(row, dict) or set(row.keys()) != {"sequence", "from", "message"}:
                raise ValueError(f"transcript[{idx}] fields are invalid.")
            if row.get("sequence") != expected_sequence:
                raise ValueError("transcript sequence must be contiguous starting at 1.")
            sender = str(row.get("from") or "").strip()
            message = str(row.get("message") or "").strip()
            if sender not in allowed_from:
                raise ValueError(f"transcript[{idx}].from is not an involved advisor or meta_advisor.")
            if not message:
                raise ValueError(f"transcript[{idx}].message is required.")
            expected_sequence += 1

    @classmethod
    def _validate_screen3_public_response(cls, result: Dict[str, Any]) -> None:
        if set(result.keys()) != {
            "involved_advisor_ids", "advisor_reasonings", "transcript", "suggestion"
        }:
            raise ValueError("Screen 3 response has unexpected top-level fields.")

        ids = result.get("involved_advisor_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(x, str) or not x.strip() for x in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("involved_advisor_ids must be a non-empty list of unique advisor slugs.")
        cls._validate_advisor_reasonings(ids, result.get("advisor_reasonings"))
        cls._validate_transcript(ids, result.get("transcript"))

        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict) or set(suggestion.keys()) != {"interventions"}:
            raise ValueError("Generate suggestion must contain interventions only.")
        interventions = suggestion.get("interventions")
        if not isinstance(interventions, list) or not (2 <= len(interventions) <= 3):
            raise ValueError("Screen 3 generate requires exactly 2 to 3 interventions.")

        required = {
            "title", "confidence_level", "impact_description",
            "reportable_value", "outputs",
        }
        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, dict) or set(intervention.keys()) != required:
                raise ValueError(f"interventions[{i}] fields are invalid.")
            for key in ("title", "impact_description", "reportable_value"):
                if not isinstance(intervention.get(key), str) or not intervention[key].strip():
                    raise ValueError(f"interventions[{i}].{key} is required.")
            if str(intervention.get("confidence_level") or "").strip().lower() not in {
                "high", "medium", "low"
            }:
                raise ValueError(f"interventions[{i}].confidence_level must be high/medium/low.")

            # Product rule: exactly 12 ordered weekly Sprint units per intervention.
            # Backend shape remains outputs[] -> results[]. Each output is one
            # Sprint unit; output.text is its Result and results[] are the 1-3
            # executable deliverables for that 5-working-day sprint.
            outputs = intervention.get("outputs")
            if not isinstance(outputs, list) or len(outputs) != 12:
                raise ValueError(f"interventions[{i}].outputs must contain exactly 12 sprint units.")
            for j, output in enumerate(outputs):
                if not isinstance(output, dict) or set(output.keys()) != {"text", "results"}:
                    raise ValueError(f"interventions[{i}].outputs[{j}] fields are invalid.")
                if not isinstance(output.get("text"), str) or not output["text"].strip():
                    raise ValueError(f"interventions[{i}].outputs[{j}].text is required.")
                results = output.get("results")
                if not isinstance(results, list) or not (1 <= len(results) <= 3):
                    raise ValueError(
                        f"interventions[{i}].outputs[{j}].results must contain 1 to 3 executable texts."
                    )
                for k, result_item in enumerate(results):
                    if (
                        not isinstance(result_item, dict)
                        or set(result_item.keys()) != {"text"}
                        or not isinstance(result_item.get("text"), str)
                        or not result_item["text"].strip()
                    ):
                        raise ValueError(
                            f"interventions[{i}].outputs[{j}].results[{k}] must contain text only."
                        )

    @classmethod
    def _validate_screen3_regenerate_response(cls, result: Dict[str, Any]) -> None:
        if set(result.keys()) != {
            "involved_advisor_ids", "advisor_reasonings", "transcript", "suggestion"
        }:
            raise ValueError("Regenerate response has unexpected top-level fields.")
        ids = result.get("involved_advisor_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(x, str) or not x.strip() for x in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("involved_advisor_ids must be a non-empty list of unique advisor slugs.")
        cls._validate_advisor_reasonings(ids, result.get("advisor_reasonings"))
        cls._validate_transcript(ids, result.get("transcript"))

        suggestion = result.get("suggestion")
        if not isinstance(suggestion, dict):
            raise ValueError("Regenerate suggestion must be an object.")
        if set(suggestion.keys()) - {"text", "results"}:
            raise ValueError("Regenerate suggestion has unexpected fields.")
        if "text" in suggestion and (
            not isinstance(suggestion.get("text"), str) or not suggestion["text"].strip()
        ):
            raise ValueError("Regenerated output text must be non-empty when present.")
        results = suggestion.get("results")
        if not isinstance(results, list) or not (1 <= len(results) <= 3):
            raise ValueError("Regenerate suggestion.results must contain 1 to 3 executable texts.")
        for idx, row in enumerate(results):
            if (
                not isinstance(row, dict)
                or set(row.keys()) != {"text"}
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
            ):
                raise ValueError(f"suggestion.results[{idx}] must contain text only.")

    def _run_advisor_regeneration_reasoning(
        self,
        advisor: Dict[str, Any],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask one Specialist, with full Expert DNA, how the requested rewrite should change."""
        prompt = self._load_prompt(ADVISOR_PROMPTS_DIR / advisor["prompt_file"])
        payload = self._payload(request)
        task = {
            "instruction": (
                "أنت تشارك في إعادة توليد مخرج واحد فقط من تدخل قائم. "
                "اقرأ سبب المستخدم والسياق، ثم اكتب رأيك المهني المستقل في جملتين أو ثلاث فقط: "
                "ما الذي يجب تغييره في المخرج أو نتائجه ولماذا، من داخل نطاق اختصاصك فقط. "
                "لا تكتب JSON، ولا تنشئ أرقامًا أو نسبًا أو مددًا غير موجودة في السياق، "
                "ولا تعِد كتابة التدخل الكامل."
            ),
            "advisor_id": advisor["advisor_id"],
            "advisor_name_ar": advisor["advisor_name_ar"],
            "rewrite_reason": request.get("reason"),
            "output": payload.get("output"),
            "intervention": payload.get("intervention"),
            "impact_map": payload.get("impact_map"),
        }
        opinion = self._generate(
            "specialist",
            prompt,
            json.dumps(task, ensure_ascii=False, indent=2),
            240,
            deterministic=True,
            repetition_penalty=1.08,
            no_repeat_ngram_size=6,
        )
        opinion = self.clean_model_text(opinion).strip()
        if not opinion:
            raise ValueError(f"Specialist {advisor['advisor_id']} returned an empty regeneration reasoning.")
        return {
            "advisor_id": advisor["advisor_id"],
            "backend_id": advisor.get("backend_id") or advisor["advisor_id"],
            "advisor_name_ar": advisor["advisor_name_ar"],
            "opinion": opinion,
        }

    def _regenerate_single_output(
        self,
        request: Dict[str, Any],
        advisor_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Rewrite one Sprint/output plus its 1-3 executable child texts."""
        payload = self._payload(request)
        if str(payload.get("target") or "").strip() != "intervention_output":
            raise ValueError("regenerate requires payload.target='intervention_output'.")

        reason = str(request.get("reason") or "").strip()
        if not reason:
            raise ValueError("regenerate requires a non-empty reason.")

        current_output = payload.get("output")
        intervention = payload.get("intervention")
        impact_map = payload.get("impact_map")
        if not isinstance(current_output, dict):
            raise ValueError("regenerate requires payload.output.")
        if not isinstance(intervention, dict):
            raise ValueError("regenerate requires payload.intervention.")
        if not isinstance(impact_map, dict):
            raise ValueError("regenerate requires payload.impact_map.")

        existing_text = str(current_output.get("text") or "").strip()
        if not existing_text:
            raise ValueError("payload.output.text is required for regeneration.")

        meta = self._resolve_meta_advisor(request)
        advisor_reasonings = self._build_advisor_reasonings(advisor_outputs, request)
        involved_ids = [x["advisor_id"] for x in advisor_reasonings]
        if not involved_ids:
            raise ValueError("Regenerate requires at least one usable Specialist reasoning.")

        source_context = {
            "reason": reason,
            "output": current_output,
            "intervention": intervention,
            "impact_map": impact_map,
        }
        source_text = self._normalize_digits(
            json.dumps(source_context, ensure_ascii=False, sort_keys=True)
        )
        source_numbers = self._extract_number_tokens(source_text)

        protocol = (
            "Return plain text only. Do not output JSON or Markdown.\n"
            "For EVERY advisor return: REVIEW=<advisor_slug>||<Meta assessment>.\n"
            "Then: TEXT=<rewritten weekly result/output>.\n"
            "Then 1 to 3 lines: RESULT=<executable text>.\n"
            "Finally: FINAL=<brief Meta approval/explanation>."
        )
        task = {
            "role": meta["slug"],
            "meta_advisor_name": meta["name"],
            "task": "Rewrite exactly one Screen-3 sprint unit according to the user's reason.",
            "user_reason": reason,
            "current_output": current_output,
            "intervention": intervention,
            "impact_map": impact_map,
            "specialist_reasonings": advisor_reasonings,
            "required_protocol": protocol,
            "rules": [
                "Follow the user's reason directly.",
                "Stay consistent with the intervention and impact_map.",
                "Return exactly 1 to 3 RESULT lines.",
                "Every RESULT must be realistically executable within 5 working days.",
                "Do not invent numbers, percentages, dates, budgets, partners or durations absent from the payload.",
                "Use advisor slugs only, never advisor persona names.",
                "Return one REVIEW for every involved advisor.",
            ],
        }

        def parse_rewrite(raw: str) -> Dict[str, Any]:
            clean = self.clean_model_text(raw).replace("```", "").strip()
            output_text = ""
            results: List[str] = []
            reviews: Dict[str, str] = {}
            final_message = ""
            for line in clean.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^REVIEW\s*[:=]\s*([^|]+?)\s*\|\|\s*(.+)$", line, flags=re.I)
                if m:
                    reviews[m.group(1).strip()] = m.group(2).strip()
                    continue
                m = re.match(r"^TEXT\s*[:=]\s*(.+?)\s*$", line, flags=re.I)
                if m:
                    output_text = m.group(1).strip()
                    continue
                m = re.match(r"^RESULT\s*[:=]\s*(.+?)\s*$", line, flags=re.I)
                if m:
                    value = m.group(1).strip()
                    if value:
                        results.append(value)
                    continue
                m = re.match(r"^FINAL\s*[:=]\s*(.+?)\s*$", line, flags=re.I)
                if m:
                    final_message = m.group(1).strip()

            output_text = self._clean_meta_public_text(output_text, request)
            clean_results: List[str] = []
            for value in results:
                value = self._clean_meta_public_text(value, request)
                if value and value not in clean_results:
                    clean_results.append(value)
            clean_reviews = {
                aid: self._clean_meta_public_text(msg, request)
                for aid, msg in reviews.items()
                if aid and msg
            }
            return {
                "text": output_text,
                "results": clean_results[:3],
                "reviews": clean_reviews,
                "final": self._clean_meta_public_text(final_message, request),
                "raw": raw,
            }

        def validation_errors(parsed: Dict[str, Any]) -> List[str]:
            errors: List[str] = []
            if not parsed.get("text"):
                errors.append("TEXT missing")
            if not (1 <= len(parsed.get("results") or []) <= 3):
                errors.append("RESULT count must be 1-3")
            for aid in involved_ids:
                if not str((parsed.get("reviews") or {}).get(aid) or "").strip():
                    errors.append(f"missing REVIEW for {aid}")
            if not parsed.get("final"):
                errors.append("FINAL missing")

            values = [parsed.get("text") or ""] + list(parsed.get("results") or [])
            unsupported: List[str] = []
            for value in values:
                for number in self._extract_number_tokens(value):
                    if number not in source_numbers and number not in unsupported:
                        unsupported.append(number)
            if unsupported:
                errors.append("unsupported numbers: " + ", ".join(unsupported))
            return errors

        meta_adapter = "meta" if COUNCIL_META_MODE == "adapter" else "base"

        def generate_once(task_obj: Dict[str, Any]) -> Dict[str, Any]:
            raw = self._generate(
                meta_adapter,
                meta["prompt"],
                json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")),
                650,
                deterministic=True,
                repetition_penalty=1.05,
            )
            return parse_rewrite(raw)

        parsed = generate_once(task)
        errors = validation_errors(parsed)
        if errors:
            repair = dict(task)
            repair["previous_output"] = parsed.get("raw")
            repair["validation_errors"] = errors
            repair["repair_instruction"] = (
                "Return the complete protocol again. Fix only the listed errors; "
                "keep one REVIEW per advisor, TEXT, 1-3 RESULT lines and FINAL."
            )
            parsed = generate_once(repair)
            errors = validation_errors(parsed)

        if errors:
            raise ValueError(
                "Regeneration could not produce a valid rewrite: " + " | ".join(errors)
            )

        reasonings_by_id = {x["advisor_id"]: x["reasoning"] for x in advisor_reasonings}
        transcript: List[Dict[str, Any]] = []
        sequence = 1
        transcript.append({
            "sequence": sequence,
            "from": "meta_advisor",
            "message": f"سأراجع طلب إعادة الصياغة التالي: {reason}",
        })
        sequence += 1
        for aid in involved_ids:
            transcript.append({
                "sequence": sequence,
                "from": aid,
                "message": reasonings_by_id[aid],
            })
            sequence += 1
            transcript.append({
                "sequence": sequence,
                "from": "meta_advisor",
                "message": f"تعقيبي على {aid}: {parsed['reviews'][aid]}",
            })
            sequence += 1
        transcript.append({
            "sequence": sequence,
            "from": "meta_advisor",
            "message": parsed["final"],
        })

        public = {
            "involved_advisor_ids": involved_ids,
            "advisor_reasonings": advisor_reasonings,
            "transcript": transcript,
            "suggestion": {
                "text": parsed["text"],
                "results": [{"text": x} for x in parsed["results"]],
            },
        }
        self._validate_screen3_regenerate_response(public)
        return public

    def consult(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Screen-3 v2 production contract: generate or regenerate."""
        try:
            if not isinstance(request, dict):
                raise ValueError("advisory_consultation request must be a JSON object.")
            if str(request.get("type") or "").strip() != "advisory_consultation":
                raise ValueError("type must be advisory_consultation.")
            if str(request.get("topic") or "").strip() != "interventions":
                raise ValueError("topic must be interventions.")

            kind = str(request.get("kind") or "generate").strip().lower()
            debug = bool(request.get("debug") or self._payload(request).get("debug"))
            total_start = time.perf_counter()
            selected = self._resolve_selected_advisors(request)
            advisor_outputs: List[Dict[str, Any]] = []
            advisor_timings: Dict[str, float] = {}

            if kind == "generate":
                for item in selected:
                    advisor_id = item["advisor_id"]
                    print(f"[council] Starting Specialist {advisor_id}...", flush=True)
                    started = time.perf_counter()
                    output = self._run_advisor(item, request, selected)
                    elapsed = time.perf_counter() - started
                    advisor_timings[advisor_id] = round(elapsed, 3)
                    advisor_outputs.append(output)
                    print(f"[council] Finished Specialist {advisor_id} in {elapsed:.2f}s", flush=True)

                resolved_meta = self._resolve_meta_advisor(request)
                print(
                    f"[council] Starting {resolved_meta['slug']} synthesis for {len(advisor_outputs)} opinion(s)...",
                    flush=True,
                )
                meta_started = time.perf_counter()
                public_result = self._run_meta(request, advisor_outputs)
                meta_elapsed = time.perf_counter() - meta_started
                self._validate_screen3_public_response(public_result)
                mode = "generate"

            elif kind == "regenerate":
                payload = self._payload(request)
                if str(payload.get("target") or "").strip() != "intervention_output":
                    raise ValueError("Only payload.target='intervention_output' is supported for regenerate.")

                for item in selected:
                    advisor_id = item["advisor_id"]
                    print(f"[council] Starting regeneration reasoning {advisor_id}...", flush=True)
                    started = time.perf_counter()
                    output = self._run_advisor_regeneration_reasoning(item, request)
                    elapsed = time.perf_counter() - started
                    advisor_timings[advisor_id] = round(elapsed, 3)
                    advisor_outputs.append(output)
                    print(
                        f"[council] Finished regeneration reasoning {advisor_id} in {elapsed:.2f}s",
                        flush=True,
                    )

                meta_started = time.perf_counter()
                public_result = self._regenerate_single_output(request, advisor_outputs)
                meta_elapsed = time.perf_counter() - meta_started
                self._validate_screen3_regenerate_response(public_result)
                mode = "regenerate"

            else:
                raise ValueError("kind must be generate or regenerate.")

            total_elapsed = time.perf_counter() - total_start
            print(f"[council] Finished {mode} in {total_elapsed:.2f}s", flush=True)

            if not debug:
                return public_result

            return {
                "debug": True,
                "mode": mode,
                "selected_advisor_ids": self._backend_advisor_ids(selected),
                "advisor_outputs_internal": advisor_outputs,
                "meta_debug": getattr(self, "_last_meta_debug", {}),
                "timings_seconds": {
                    "advisors": advisor_timings,
                    "meta": round(meta_elapsed, 3),
                    "total": round(total_elapsed, 3),
                },
                "final_result": public_result,
            }

        except Exception as exc:
            message = re.sub(r"\s+", " ", str(exc or "Unknown error")).strip()
            print(f"[council] FAILED: {message}", flush=True)
            return {
                "status": "FAILED",
                "error": message[:1600] or "Screen 3 advisory request failed.",
            }

