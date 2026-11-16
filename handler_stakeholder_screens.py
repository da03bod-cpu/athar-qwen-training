"""Athar stakeholder screens P1/P3/P4 on the *existing* RunPod endpoint.

The Backend owns every complete task prompt and all business rules. This worker
only (a) resolves the one approved P1 library marker when supplied, (b) invokes
Qwen once, and (c) validates the JSON contract before returning it. No external
services, prompt persistence, cross-tenant retrieval, or Backend changes.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
import re

from knowledge.errors import KnowledgeError
from knowledge.security import require_id

MARKER = "[[ATHAR_KNOWLEDGE_CONTEXT]]"
ROUTES = {
    "stakeholder_category_extraction": ("category-extraction-v1", 9000),
    "stakeholder_survey_questions": ("survey-questions-v1", 4500),
    "stakeholder_council_analysis": ("council-analysis-v1", 4800),
}
ALLOWED_SOURCES = frozenset(("استبيان", "اجتماع", "مكالمة", "ورشة عمل"))
ALLOWED_STATES = frozenset(("متحقق", "جزئي", "غير متحقق", "لا توجد بيانات كافية"))
QUESTION_TYPES = frozenset(("rating_1_5", "multiple_choice", "yes_partial_no", "open_text"))
QUESTION_BASIS = frozenset(("الرضا العام", "الوعود", "الإجراءات", "كيف يتأثرون"))


def reject(message: str):
    raise KnowledgeError("INVALID_MODEL_OUTPUT", message)


def _valid_string(value, *, max_len=None):
    return isinstance(value, str) and bool(value.strip()) and (max_len is None or len(value.strip()) <= max_len)


def _text_list(value, min_count, max_count, max_len):
    return (isinstance(value, list) and min_count <= len(value) <= max_count
            and all(_valid_string(v, max_len=max_len) for v in value))


def _standalone_json(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        reject("Model response must be a JSON object")
    value = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", value, re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    try:
        result = json.loads(value)
    except (ValueError, TypeError):
        reject("Model did not return standalone JSON")
    if not isinstance(result, dict):
        reject("Model response must be a JSON object")
    return result


def _extract_categories(out):
    if set(out) - {"categories", "existing_completions"} or "categories" not in out:
        reject("Incorrect extraction keys")
    categories = out["categories"]
    if not isinstance(categories, list) or not categories:
        reject("categories must be a nonempty array")
    # Report only schema field/index (never the generated category content).
    # The Backend P1 schema has NO per-point maxLength for actions/promises.
    fields = {"name", "type", "affected", "interest", "power", "actions", "promises", "source_reason"}
    for index, cat in enumerate(categories):
        if not isinstance(cat, dict):
            reject(f"Malformed category fields: categories[{index}]")
        if set(cat) != fields:
            reject(f"Malformed category fields: categories[{index}]")
        rules = (
            ("name", _valid_string(cat["name"], max_len=80)),
            ("type", cat["type"] in ("داخلي", "خارجي")),
            ("affected", _valid_string(cat["affected"])),
            ("interest", cat["interest"] in ("High", "Low")),
            ("power", cat["power"] in ("High", "Low")),
            ("actions", _text_list(cat["actions"], 2, 5, None)),
            ("promises", _text_list(cat["promises"], 2, 5, None)),
            ("source_reason", _valid_string(cat["source_reason"], max_len=200)),
        )
        for field, valid in rules:
            if not valid:
                reject(f"Invalid category data: categories[{index}].{field}")
    extra = out.get("existing_completions", [])
    if not isinstance(extra, list):
        reject("existing_completions must be an array")
    for cat in extra:
        if not isinstance(cat, dict) or not _valid_string(cat.get("name")) or set(cat) - {"name", "affected", "interest", "power", "actions", "promises"}:
            reject("Invalid existing completion")
        if "affected" in cat and not isinstance(cat["affected"], str):
            reject("Invalid affected completion")
        for field in ("interest", "power"):
            if field in cat and cat[field] not in ("High", "Low", ""):
                reject("Invalid completion interest/power")
        for field in ("actions", "promises"):
            if field in cat and (not isinstance(cat[field], list) or len(cat[field]) > 5
                                 or any(not _valid_string(v) for v in cat[field])):
                reject("Invalid completion point list")
    return out


def _is_regeneration(prompt):
    # The Backend's P3 prompt includes this literal label on a regeneration.
    return "هذه إعادة توليد" in prompt


def _survey_source_items(prompt):
    """Read the Backend-owned P3 data block only when it is unambiguous.

    The Backend remains authoritative: this is a format guard, never a second
    source of business data. The fallback case has no promises/actions.
    """
    data_blocks = re.findall(r"<data>\s*(.*?)\s*</data>", prompt, flags=re.DOTALL)
    for block in data_blocks:
        if not re.search(r"^\s*كيف يتأثرون\s*:", block, re.MULTILINE):
            continue
        found = {}
        for label in ("الإجراءات", "الوعود"):
            match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*([^\n]*)(.*?)(?=^\s*(?:الإجراءات|الوعود)\s*:|\Z)",
                              block, re.MULTILINE | re.DOTALL)
            if not match:
                found[label] = None
                continue
            # One item may start on the label line: «الوعود: - ...».
            lines = (match.group(1) + "\n" + match.group(2)).splitlines()
            items = [line.strip()[2:].strip() for line in lines if line.strip().startswith("- ")]
            found[label] = items
        return found
    return None


def _survey_questions(out, prompt):
    if set(out) != {"questions"} or not isinstance(out["questions"], list) or not out["questions"]:
        reject("Expected nonempty questions array")
    anchor = 0
    source_items = _survey_source_items(prompt)
    for item in out["questions"]:
        if not isinstance(item, dict) or not {"text", "type", "basis"}.issubset(item) or set(item) - {"text", "type", "basis", "options", "source_item"}:
            reject("Malformed survey question")
        if not _valid_string(item["text"], max_len=300) or item["type"] not in QUESTION_TYPES or item["basis"] not in QUESTION_BASIS:
            reject("Invalid survey question text/type/basis")
        if item["type"] == "multiple_choice":
            if not _text_list(item.get("options"), 2, 8, 80):
                reject("multiple_choice requires 2-8 options")
        elif "options" in item:
            reject("options only allowed for multiple_choice")
        if "source_item" in item and item["source_item"] is not None and not isinstance(item["source_item"], str):
            reject("Invalid source_item")
        if item["basis"] in ("الوعود", "الإجراءات"):
            if not _valid_string(item.get("source_item")):
                reject("Promise/action question requires source_item")
            if source_items is not None and source_items.get(item["basis"]) is not None:
                allowed_items = {_normalize_label(s) for s in source_items[item["basis"]]}
                if _normalize_label(item["source_item"]) not in allowed_items:
                    reject("Survey source_item must match a Backend-provided action/promise")
        if item["basis"] == "الرضا العام":
            anchor += 1
            if item["type"] != "rating_1_5":
                reject("General satisfaction must be rating_1_5")
    # On regeneration the anchor may already be approved and is not replaced.
    if (anchor != 1 if not _is_regeneration(prompt) else anchor > 1):
        reject("Incorrect number of satisfaction anchor questions")
    return out


def _prompt_source_types(prompt):
    # Contract P4 includes the list after «أنواع الأنشطة الموجودة فقط:».
    match = re.search(r"أنواع الأنشطة الموجودة فقط\s*:\s*([^\n\r]+)", prompt)
    if not match:
        return None
    return {name for name in ALLOWED_SOURCES if name in match.group(1)}


def _prompt_promises(prompt):
    # The first P4 <data> segment lists promises before computed metrics.
    match = re.search(r"الوعود\s*\(بالترتيب\)\s*:\s*\n(.*?)(?=\n\s*المؤشرات المحسوبة\s*:|\n\s*الاستبيانات\s*:|</data>)", prompt, re.DOTALL)
    if not match:
        return None  # Backend remains authoritative about exact promise matching.
    text = match.group(1).strip()
    if not text or text == "لا يوجد":
        return []
    return [line.lstrip()[2:].strip() for line in text.splitlines() if line.lstrip().startswith("- ")]


def _normalize_label(text):
    return " ".join(str(text).strip().strip("\"'«»“”").split())


def _council_analysis(out, prompt):
    if set(out) != {"observations", "recommendations", "promise_fulfilment"}:
        reject("Incorrect council analysis keys")
    obs, recs, ful = out["observations"], out["recommendations"], out["promise_fulfilment"]
    if not isinstance(obs, list) or not 3 <= len(obs) <= 6:
        reject("observations must have 3-6 items")
    allowed = _prompt_source_types(prompt)
    for item in obs:
        if not isinstance(item, dict) or set(item) != {"text", "sources"} or not _valid_string(item["text"], max_len=1000):
            reject("Invalid observation")
        sources = item["sources"]
        if not isinstance(sources, list) or not sources or any(not isinstance(x, str) or x not in ALLOWED_SOURCES for x in sources):
            reject("Invalid observation sources")
        if allowed is not None and any(x not in allowed for x in sources):
            reject("Observation cites an unavailable source")
    if not isinstance(recs, list) or not 3 <= len(recs) <= 5 or any(
        not isinstance(x, dict) or set(x) != {"text"} or not _valid_string(x["text"], max_len=1000) for x in recs
    ):
        reject("Invalid recommendations")
    if not isinstance(ful, list):
        reject("promise_fulfilment must be array")
    seen = set()
    for item in ful:
        if not isinstance(item, dict) or set(item) != {"promise", "state", "evidence"} or not _valid_string(item["promise"]) or item["state"] not in ALLOWED_STATES or not _valid_string(item["evidence"], max_len=1000):
            reject("Invalid promise fulfilment")
        norm = _normalize_label(item["promise"])
        if norm in seen:
            reject("Duplicate promise fulfilment")
        seen.add(norm)
    promises = _prompt_promises(prompt)
    if promises is not None and seen != {_normalize_label(x) for x in promises}:
        reject("Promise fulfilment must match every supplied promise exactly once")
    return out


def parse_reply(raw, route, prompt):
    out = _standalone_json(raw)
    if route == "stakeholder_category_extraction":
        return _extract_categories(out)
    if route == "stakeholder_survey_questions":
        return _survey_questions(out, prompt)
    if route == "stakeholder_council_analysis":
        return _council_analysis(out, prompt)
    raise KnowledgeError("INVALID_REQUEST", "Unknown stakeholder screen route")


def _inject_extraction_knowledge(req, prompt):
    knowledge = req.get("knowledge")
    if knowledge is None:
        if MARKER in prompt:
            raise KnowledgeError("INVALID_REQUEST", "Unresolved knowledge marker")
        return prompt
    if not isinstance(knowledge, dict) or knowledge.get("mode") != "worker_retrieve_v1":
        raise KnowledgeError("INVALID_REQUEST", "Unsupported knowledge retrieval mode")
    if knowledge.get("context_marker") != MARKER or prompt.count(MARKER) != 1:
        raise KnowledgeError("INVALID_REQUEST", "Knowledge marker must occur exactly once")
    org = require_id(knowledge.get("organization_id"), "organization_id")
    proj = require_id(knowledge.get("project_id"), "project_id")
    query = knowledge.get("query")
    if not isinstance(query, str) or not query.strip():
        raise KnowledgeError("INVALID_REQUEST", "Knowledge query required")
    max_chunks = knowledge.get("max_chunks", 12)
    max_evidence_tokens = knowledge.get("max_evidence_tokens", 5000)
    if type(max_chunks) is not int or not 1 <= max_chunks <= 12 or type(max_evidence_tokens) is not int or not 1 <= max_evidence_tokens <= 5000:
        raise KnowledgeError("INVALID_REQUEST", "Knowledge passage limits exceed P1 contract")
    index_info = knowledge.get("organization_index")
    if not isinstance(index_info, dict) or type(index_info.get("generation")) is not int or index_info["generation"] < 1:
        raise KnowledgeError("INDEX_NOT_READY", "Published organization index required")
    from knowledge.embedding import embedder
    from knowledge.retrieval import load_index, search_pack, evidence_context
    encoding = embedder()
    pack, _ = load_index(index_info.get("index_ref"), scope="organization_library",
                         organization_id=org, project_id=proj,
                         generation=index_info["generation"], embedding=encoding)
    hits = search_pack(pack, query, max_chunks=max_chunks, max_evidence_tokens=max_evidence_tokens, encoding=encoding)
    return prompt.replace(MARKER, evidence_context(hits))


def prepare_prompt(req):
    route = req.get("type")
    if route not in ROUTES:
        raise KnowledgeError("INVALID_REQUEST", "Unknown stakeholder screen type")
    if req.get("prompt_id") != ROUTES[route][0]:
        raise KnowledgeError("INVALID_REQUEST", "Unsupported prompt_id")
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 160000:
        raise KnowledgeError("INVALID_REQUEST", "A complete Backend prompt is required")
    if route == "stakeholder_category_extraction":
        return _inject_extraction_knowledge(req, prompt)
    if "knowledge" in req or MARKER in prompt:
        raise KnowledgeError("INVALID_REQUEST", "P3/P4 do not use library retrieval")
    return prompt


def _env_int(name, default, low, high):
    try:
        value = int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        raise KnowledgeError("INVALID_CONFIGURATION", "Invalid screen inference configuration") from None
    if not low <= value <= high:
        raise KnowledgeError("INVALID_CONFIGURATION", "Screen inference limit outside supported range")
    return value


def _infer_qwen(prompt, route, *, load_model, get_model, get_tokenizer, model_lock):
    """One deterministic base-Qwen call; the complete Backend prompt is the user message."""
    import torch
    max_new_tokens = _env_int("ATHAR_SCREEN_" + route.upper() + "_MAX_NEW_TOKENS", ROUTES[route][1], 256, 12000)
    max_input_tokens = _env_int("ATHAR_SCREENS_MAX_INPUT_TOKENS", 24000, 1000, 30000)
    with model_lock:
        load_model()
        model, tokenizer = get_model(), get_tokenizer()
        if model is None or tokenizer is None:
            raise KnowledgeError("MODEL_UNAVAILABLE", "Qwen model is not ready")
        # No extra domain prompt: the Backend already supplies all the rules.
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        encoded = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        input_size = int(encoded["attention_mask"].sum().item())
        if input_size > max_input_tokens:
            raise KnowledgeError("CONTEXT_LIMIT_EXCEEDED", "Backend prompt exceeds model input limit")
        model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if isinstance(model_max, int) and input_size + max_new_tokens > model_max:
            raise KnowledgeError("CONTEXT_LIMIT_EXCEEDED", "Prompt and output exceed model context")
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        disable_adapter = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        generation_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.04, use_cache=True,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        # No artificial 180-second cap: backend owns the 1800/300-second deadlines.
        with disable_adapter, torch.inference_mode():
            generated = model.generate(**encoded, **generation_kw)
        return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def dispatch(req, *, load_model, get_model, get_tokenizer, model_lock, inference_override=None):
    route = req.get("type")
    try:
        if route not in ROUTES:
            raise KnowledgeError("INVALID_REQUEST", "Unknown stakeholder screen type")
        if req.get("preflight") is True:
            return {"status": "OK", "preflight": True, "type": route,
                    "prompt_id": ROUTES[route][0], "model": "Qwen/Qwen3-14B",
                    "worker_retrieval": route == "stakeholder_category_extraction",
                    "same_endpoint": True, "backend_changes_required": False}
        prompt = prepare_prompt(req)
        raw = (inference_override(prompt) if inference_override is not None else
               _infer_qwen(prompt, route, load_model=load_model, get_model=get_model,
                           get_tokenizer=get_tokenizer, model_lock=model_lock))
        return parse_reply(raw, route, prompt)
    except KnowledgeError as exc:
        return {"status": "FAILED", "type": route, "run_id": req.get("run_id"),
                "error_code": exc.code, "error": str(exc)}
    except Exception:
        return {"status": "FAILED", "type": route, "run_id": req.get("run_id"),
                "error_code": "INTERNAL_ERROR", "error": "Stakeholder screen inference failed"}
