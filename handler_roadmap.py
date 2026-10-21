"""Athar OS Screen 5 (خارطة الطريق) — backend-contract router.

Integration: consult_roadmap(existing_council_engine, job_input).
Uses already-loaded Specialist/Meta PEFT adapters via AtharCouncilEngine._generate.
No additional model instance, LoRA, dataset or third-party library required.

Contract: Screen 5 (خارطة الطريق) — AI Contract, 2026-10.
V11 core preserved: deliverable completion checks, focused repairs and transcript checks.
V12 contract bridge: planned_percent, Screen 3 result shape, and optional
Meta decision/synthesis/evidence metadata.
V12.1: isolate Arabic corruption in publishable fields; repair meta note
separately without discarding a valid sprint. Restore V11 TASK|text prompts.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROADMAP_CODE_VERSION = "screen5-v12.1-v11-preserved-field-isolation"
VALID_UNITS = frozenset(("percent", "number", "multiplier"))
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_INVALID_SCRIPTS = re.compile(r"[\u0400-\u052F\u3040-\u30FF\u4E00-\u9FFF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF]")
_ADVISOR_SLUG = re.compile(r"^AOS-[A-Z]{2,5}-\d{2}$")
_META_SLUG = re.compile(r"^AOS-META-\d{2}$")
# Only council transcript messages may reference selected advisor slugs.
# Tasks/titles/indicators are organization-facing work, never instructions
# to invoke experts. This directly guards an issue observed on the GPU smoke.
_ANY_ADVISOR_ID = re.compile(r"(?<![A-Za-z0-9])AOS-[A-Z]{2,5}-\d{2}(?![A-Za-z0-9])")
_ADVISOR_AS_TASK = re.compile(
    r"(?:المستشار(?:ين)?|مستشار(?:ين)?|المستشارة|مستشارة|الخبير(?:ين)?|خبير(?:ين)?)"
)
_WEAK_INDICATOR = re.compile(r"(?:التوصيات\s+المؤجلة|التوصيات\s+المؤثرة|التغطية\s+المؤقتة\s+للتمويل)")
# A structurally valid unit can still contradict the Arabic indicator label.
# Refuse ambiguous/incorrect model output rather than altering its semantics.
_PERCENT_LABEL = re.compile(r"^(?:نسبة|النسبة|مؤشر نسبة|معدل النسبة)\s")
_COUNT_LABEL = re.compile(r"^(?:عدد|العدد|إجمالي عدد)\s")
_MULTIPLIER_LABEL = re.compile(r"^(?:مضاعف|معامل المضاعفة)\s")
_WEEKLY_COMPLETION_REASON = re.compile(r"(?:إنجاز|الإنجاز|اكتمال|إتمام|عمل ملموس|مهام السبرينت|التنفيذ خلال)")
_COMPLETION_WORDS = re.compile(
    r"(?:اكتمال|إتمام|إنجاز|المنجز|المنفذ|المصنف|المصنفة|الموثق|الموثقة|"
    r"المعتمد|المعتمدة|المراجع|المراجعة|المحلل|المحللة|المحدد|المحددة|"
    r"المسجل|المسجلة|المجموع|المجمعة|المدخل|المدخلة|المستكمل|المستكملة|"
    r"المعد|المعدة|المستلم|المستلمة|المسلم|المسلمة|الجاهز|الجاهزة|"
    r"توثيق|تصنيف|تسليم|إعداد|إنهاء)"
)

# Historical lookbacks are facts about availability/period, not target proposals.
# Validate them against the authoritative sprint context; numeric KPI targets
# remain suggestions, as explicitly allowed by the Screen 5 contract.
_HISTORICAL_LOOKBACK = re.compile(
    r"(?:(?:السنوات?|للسنوات?)\s+(?:الثلاث|الثلاثة|الأربع|الخمس|ثلاث|أربع|خمس|[0-9٠-٩]+)\s+(?:الأخيرة|الماضية)"
    r"|(?:آخر|خلال)\s+(?:ثلاث|ثلاثة|أربع|خمس|[0-9٠-٩]+)\s+سنوات?)"
)


class RoadmapError(ValueError):
    """Validation/generation failure that must NOT result in partial success."""


# Defensive checks for documented Arabic corruption observed in live RunPod
# results. These cannot replace full linguistic review, but ensure known
# failures are never treated as a successful customer-visible result.
_SUSPECT_ARABIC = re.compile(
    r"\b(?:المؤثمة|التضاريب)\b|الجمعية\s+(?:لن\s+)?يبدأ\b"
)


def _require_generated_arabic_quality(raw: str) -> None:
    """Reject the entire generation and retry, rather than dropping one task.

    A mixed-language output may omit the crucial first step if filtered down
    to 3-5 tasks. Rejecting the batch keeps execution order and coverage.
    """
    if _INVALID_SCRIPTS.search(raw):
        raise RoadmapError(
            "Generated text mixes non-Arabic writing systems (e.g. Korean). "
            "Rewrite every task and META_MESSAGE in natural Arabic only."
        )
    if _SUSPECT_ARABIC.search(raw):
        raise RoadmapError(
            "Generated Arabic contains corrupted or ungrammatical wording. "
            "Rewrite all tasks and META_MESSAGE in clear Modern Standard Arabic."
        )


def _arabic_field_error(value: str) -> str | None:
    """Check only the field that may enter the published result.

    V11/V12's whole-raw guard rejected a *valid* sprint when an unrelated
    META_MESSAGE contained a known typo. Do not silently autocorrect model
    content: repair the exact field with the existing Meta adapter instead.
    """
    if _INVALID_SCRIPTS.search(value):
        return "contains non-Arabic-script text"
    match = _SUSPECT_ARABIC.search(value)
    if match:
        return f"contains suspect Arabic expression {match.group(0)!r}"
    return None


def _sprint_arabic_issues(sprint: dict) -> dict[str, str]:
    issues: dict[str, str] = {}
    for key in ("title", "description"):
        problem = _arabic_field_error(str(sprint.get(key, "")))
        if problem:
            issues["card"] = f"Sprint {sprint.get('number')} {key} {problem}"
            break
    for row in sprint.get("tasks", []):
        problem = _arabic_field_error(str(row.get("text", "")))
        if problem:
            issues["tasks"] = f"Sprint {sprint.get('number')} task {problem}"
            break
    for row in sprint.get("indicators", []):
        problem = _arabic_field_error(str(row.get("name", "")))
        if problem:
            issues["indicators"] = f"Sprint {sprint.get('number')} KPI {problem}"
            break
    return issues


def _text(value: Any, *, limit: int = 1000, arabic: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    s = re.sub(r"\s+", " ", value).strip().strip("` ")
    if not s or len(s) > limit or _INVALID_SCRIPTS.search(s):
        return ""
    if arabic and not _ARABIC.search(s):
        return ""
    return s


def _clean_model(value: Any) -> str:
    s = str(value or "")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    return re.sub(r"```(?:text|txt|json|markdown|md)?\s*|```", "", s, flags=re.I).strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _required_number(value: Any) -> int:
    if isinstance(value, bool):
        raise RoadmapError("Sprint number cannot be a boolean")
    try:
        n = int(str(value))
        if str(value).strip() not in (str(n),):
            raise ValueError()
    except (TypeError, ValueError):
        raise RoadmapError("Sprint number must be an integer") from None
    if not 1 <= n <= 12:
        raise RoadmapError("Sprint number outside range 1..12")
    return n


def _number(value: Any, unit: str) -> int | float:
    if isinstance(value, bool):
        raise RoadmapError("Boolean indicator target")
    try:
        num = float(str(value).strip().translate(_ARABIC_DIGITS))
    except (ValueError, TypeError):
        raise RoadmapError(f"Invalid numeric indicator target: {value!r}") from None
    if not math.isfinite(num) or num < 0 or (unit == "percent" and num > 100):
        raise RoadmapError("Invalid numeric indicator target range")
    if unit == "multiplier" and num <= 0:
        raise RoadmapError("Multiplier target must be positive")
    return int(num) if num.is_integer() else round(num, 4)


def _indicator(data: Any) -> dict:
    if not isinstance(data, dict):
        raise RoadmapError("Indicator must be an object")
    name = _text(data.get("name"), limit=180, arabic=True)
    unit = data.get("unit")
    if not name or unit not in VALID_UNITS:
        raise RoadmapError("Missing indicator name or invalid unit")
    if _ANY_ADVISOR_ID.search(name) or _ADVISOR_AS_TASK.search(name):
        raise RoadmapError("Indicator must measure sprint work, not an advisor")
    if _WEAK_INDICATOR.search(name):
        raise RoadmapError("Indicator does not measure a concrete sprint deliverable")
    if _PERCENT_LABEL.search(name) and unit != "percent":
        raise RoadmapError("Indicator name begins with نسبة, so unit must be percent (not number)")
    if _COUNT_LABEL.search(name) and unit != "number":
        raise RoadmapError("Indicator name begins with عدد, so unit must be number")
    if _MULTIPLIER_LABEL.search(name) and unit != "multiplier":
        raise RoadmapError("Indicator name begins with مضاعف, so unit must be multiplier")
    return {"name": name, "unit": unit, "target": _number(data.get("target"), unit)}


def _planned_percent(value: Any) -> int:
    """Weekly task completion proposal; never an achieved percentage.

    Legacy V11 model replies wrote TASK|text, which means the model planned an
    entire executable five-day task. Preserve their meaning with a default of
    100. New TASK|text|percent replies may expressly choose a partial stage.
    """
    if value is None or value == "":
        return 100
    if isinstance(value, bool):
        raise RoadmapError("Task planned_percent must be a whole number 0..100")
    try:
        numeric = float(str(value).strip().translate(_ARABIC_DIGITS))
    except (TypeError, ValueError):
        raise RoadmapError("Task planned_percent is not numeric") from None
    if not math.isfinite(numeric) or not 0 <= numeric <= 100:
        raise RoadmapError("Task planned_percent outside 0..100")
    return int(math.floor(numeric + 0.5))


def _task_line(value: str) -> dict:
    """Both original V11 TASK|Arabic and new TASK|Arabic|40 are accepted."""
    if "|" in value:
        phrase, sep, maybe_percent = value.rpartition("|")
        if sep and re.fullmatch(r"[0-9٠-٩۰-۹]{1,3}(?:[.٫][0-9٠-٩۰-۹]+)?", maybe_percent.strip()):
            return {"text": phrase, "planned_percent": _planned_percent(maybe_percent)}
    return {"text": value, "planned_percent": 100}


def _validate_tasks(tasks: Any, *, minimum: int = 1) -> list[dict]:
    if not isinstance(tasks, list) or not minimum <= len(tasks) <= 8:
        raise RoadmapError("Tasks list length invalid")
    out = []
    for row in tasks:
        value = row if isinstance(row, str) else row.get("text") if isinstance(row, dict) else None
        clean = _text(value, limit=360, arabic=True)
        if not clean:
            raise RoadmapError("Empty/invalid task text")
        if _ANY_ADVISOR_ID.search(clean) or _ADVISOR_AS_TASK.search(clean):
            raise RoadmapError("Task is about calling an advisor, not an executable NGO action")
        percent = _planned_percent(row.get("planned_percent") if isinstance(row, dict) else None)
        out.append({"text": clean, "planned_percent": percent})
    if len({_norm(x["text"]) for x in out}) != len(out):
        raise RoadmapError("Duplicate tasks in sprint")
    return out


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _validate_sprint(row: Any, number: int) -> dict:
    if not isinstance(row, dict) or _required_number(row.get("number")) != number:
        raise RoadmapError(f"Missing or wrong sprint number {number}")
    title = _text(row.get("title"), limit=130, arabic=True)
    desc = _text(row.get("description"), limit=650, arabic=True)
    if not title or not desc:
        raise RoadmapError(f"Sprint {number} missing Arabic title or description")
    if _ANY_ADVISOR_ID.search(title + " " + desc):
        raise RoadmapError(f"Sprint {number} title/description contains advisor ID")
    # Contract design target: approximately 3-5 tasks and indicators per sprint.
    tasks = _validate_tasks(row.get("tasks"), minimum=3)
    if len(tasks) > 5:
        raise RoadmapError(f"Sprint {number} should have 3-5 tasks")
    inds = row.get("indicators")
    if not isinstance(inds, list) or not 3 <= len(inds) <= 5:
        raise RoadmapError(f"Sprint {number} should have 3-5 indicators")
    indicators = [_indicator(x) for x in inds]
    if len({_norm(x["name"]) for x in indicators}) != len(indicators):
        raise RoadmapError(f"Sprint {number} has duplicate indicators")
    return {"number": number, "title": title, "description": desc,
            "tasks": tasks, "indicators": indicators}


def _advisor_roster(payload: dict, council: Any) -> list[dict]:
    raw = payload.get("advisors", [])
    if not isinstance(raw, list):
        raise RoadmapError("payload.advisors must be a list")
    if len(raw) > 16:
        raise RoadmapError("Too many advisors (max 16)")
    result, used = [], set()
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            continue
        slug = row["slug"].strip()
        # Never infer a canonical ID from a display name or a catalog integer.
        if not _ADVISOR_SLUG.fullmatch(slug) or slug in used:
            continue
        registry = getattr(council, "registry_by_model_id", {}).get(slug)
        if not isinstance(registry, dict) or not registry.get("prompt_file"):
            continue  # Unknown IDs must not be presented as involved advisors.
        used.add(slug)
        result.append({"slug": slug, "entry": registry,
                       "title": _text(row.get("title"), limit=160)})
    return result


def _meta_prompt(council: Any, meta: Any) -> str:
    """Resolve the exact Meta DNA through Screen-3 V8 when available.

    V8 no longer exposes ``self.meta_prompt`` and already supports the
    AOS-META-01 -> AOS-META-00 fallback when no per-slug file exists.
    Delegating to the council's authoritative resolver avoids a duplicate or
    inconsistent identity mapping between Screen 3 and Screen 5.
    """
    if meta is not None:
        if not isinstance(meta, dict) or not _META_SLUG.fullmatch(str(meta.get("slug", ""))):
            raise RoadmapError("Invalid meta_advisor slug")

    resolver = getattr(council, "_resolve_meta_advisor", None)
    if callable(resolver):
        resolved = resolver({"meta_advisor": meta})
        if not isinstance(resolved, dict):
            raise RoadmapError("Meta resolver returned no valid persona object")
        prompt = resolved.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RoadmapError("Meta resolver returned an empty persona")
        return prompt.strip()

    # Compatibility with council engines predating Screen-3 V8.
    if meta is None or meta.get("slug") == "AOS-META-00":
        legacy = getattr(council, "meta_prompt", None)
        if not isinstance(legacy, str) or not legacy.strip():
            raise RoadmapError("No Meta persona available in legacy council")
        return legacy.strip()
    directory = Path(os.environ.get("META_PROMPTS_DIR", str(Path(os.environ.get(
        "ATHAR_ROOT", "/workspace/data/athar")) / "prompts" / "meta")))
    for target in (meta["slug"], "AOS-META-00" if meta["slug"] == "AOS-META-01" else ""):
        if not target:
            continue
        for suffix in (".md", ".txt"):
            path = directory / f"{target}{suffix}"
            if path.is_file():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
    raise RoadmapError(f"Meta persona prompt is not installed: {meta['slug']}")


def _input(request: dict) -> dict:
    if not isinstance(request, dict):
        raise RoadmapError("Request must be a JSON object")
    # Accept flat RunPod job.input, and also the contract envelope for unit calls.
    candidate = request.get("input", request)
    data = candidate if isinstance(candidate, dict) else None
    if not data or data.get("type") != "advisory_consultation" or data.get("topic") != "roadmap":
        raise RoadmapError("Expected advisory_consultation / roadmap")
    if not isinstance(data.get("payload"), dict):
        raise RoadmapError("Missing payload object")
    if data.get("kind") not in ("generate", "regenerate"):
        raise RoadmapError("Unknown roadmap kind")
    return data


def _normalize_output(value: Any) -> dict | None:
    """Bridge Screen-3 result objects to Screen-5 string results.

    The backend owns output assignment; this copy is for model context only.
    Do not rewrite the backend's output or return it in suggestion.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RoadmapError("Sprint output must be an object or null")
    results = value.get("results")
    if results is None:
        results = []
    if not isinstance(results, list):
        raise RoadmapError("Sprint output.results must be a list")
    normalized = []
    for result in results:
        candidate = result.get("text") if isinstance(result, dict) else result
        if not isinstance(candidate, str) or not candidate.strip():
            raise RoadmapError("Sprint output result must contain nonempty text")
        normalized.append(candidate.strip())
    return {**value, "results": normalized}


def _output_key(value: Any) -> str:
    return _json(_normalize_output(value)) if value is not None else "null"


def _validate_generate_request(payload: dict) -> list[dict]:
    rows = payload.get("sprints")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise RoadmapError("payload.sprints must contain between 1 and 12 entries")
    numbers = set()
    output = []
    for row in rows:
        if not isinstance(row, dict):
            raise RoadmapError("Each payload.sprints entry must be an object")
        number = _required_number(row.get("number"))
        if number in numbers:
            raise RoadmapError(f"Duplicate sprint {number}")
        numbers.add(number)
        item_output = _normalize_output(row.get("output"))
        output.append({"number": number, "output": item_output})
    return sorted(output, key=lambda x: x["number"])


def _context(data: dict, rows: list[dict] | None = None) -> dict:
    p = data["payload"]
    context = {k: p.get(k) for k in (
        "organization", "programs", "track", "goal", "impact_map", "intervention"
    )}
    intervention = context.get("intervention")
    if isinstance(intervention, dict) and isinstance(intervention.get("outputs"), list):
        context["intervention"] = {**intervention, "outputs": [
            _normalize_output(output) for output in intervention["outputs"]
        ]}
    return context | ({"sprints": rows} if rows is not None else {})


def _specialist_messages(council: Any, data: dict, advisors: list[dict], context: dict) -> tuple[list[dict], list[dict]]:
    messages, opinions = [], []
    for item in advisors:
        prompt = council._load_prompt(Path(os.environ.get("ADVISOR_PROMPTS_DIR",
            str(Path(os.environ.get("ATHAR_ROOT", "/workspace/data/athar")) / "prompts" / "advisors")))
            / item["entry"]["prompt_file"])
        task = {
            "task": "أعطِ رأيًا مستقلًا موجزًا لبناء خارطة الطريق. التزم بحدود Expert DNA الأصلي وبالمخرج المعتمد، ولا تخترع حقائق أو بيانات سابقة. لا تضع أرقام مستهدفات؛ يقترح المجلس المستهدفات التنفيذية لاحقًا. اكتب سطرين: REASONING|تبرير مهني مختصر، ثم RECOMMENDATION|أولوية عملية. اكتب بالعربية. لا تستخدم أسماء مستشارين أو تحاكِ محادثة.",
            "advisor_id": item["slug"],
            "specialization": item["title"],
            "context": context,
        }
        raw = _clean_model(council._generate("specialist", prompt, _json(task),
                                            int(os.getenv("ROADMAP_SPECIALIST_TOKENS", "420")),
                                            deterministic=True))
        reason, rec = "", ""
        for line in raw.splitlines():
            if line.startswith("REASONING|"):
                reason = _text(line.partition("|")[2], limit=750, arabic=True)
            elif line.startswith("RECOMMENDATION|"):
                rec = _text(line.partition("|")[2], limit=750, arabic=True)
        if not reason or not rec:
            # Specialist V2 was trained on advisory prose, not a line grammar.
            # When it answers in normal Markdown, quote substantive lines from
            # the actual independent opinion rather than inventing an answer.
            prose = []
            for line in raw.splitlines():
                candidate = re.sub(r"^\s*(?:#{1,6}\s*|[-*•]\s*|\d+[.)]\s*)", "", line).strip()
                candidate = re.sub(r"^(?:REASONING|RECOMMENDATION)\|", "", candidate)
                candidate = _text(candidate, limit=750, arabic=True)
                if candidate and len(candidate) >= 24 and candidate not in prose:
                    prose.append(candidate)
            if not reason and prose:
                reason = prose[0]
            if not rec:
                rec = next((x for x in prose if x != reason), "")
        if not reason or not rec:
            raise RoadmapError(f"Specialist opinion invalid: {item['slug']}")
        opinions.append({"advisor_id": item["slug"], "reasoning": reason, "recommendation": rec})
        messages.append({"from": item["slug"], "message": rec})
    return opinions, messages


def _meta_generate(
    council: Any,
    system: str,
    task: dict,
    tokens: int,
    *,
    sample: bool = False,
    block_non_arabic_scripts: bool = False,
) -> str:
    """Use the existing Meta adapter; avoid repeated identical greedy failures.

    The production AtharCouncilEngine._generate supports forbidden_token_regex:
    a logits processor excludes tokens whose decoded form contains Hangul,
    Han, Kana, or Cyrillic. The downstream text validator remains mandatory.
    Only retries sample: first attempt stays reproducible.
    """
    kwargs = {"deterministic": not sample, "repetition_penalty": 1.08}
    if block_non_arabic_scripts:
        kwargs["forbidden_token_regex"] = _INVALID_SCRIPTS.pattern
    return _clean_model(council._generate("meta", system, _json(task), tokens, **kwargs))


def _lines(raw: str) -> list[str]:
    return [re.sub(r"^[*•-]\s+", "", s.strip()).replace("｜", "|")
            for s in raw.splitlines() if s.strip()]


def _parse_indicator_line(line: str) -> dict:
    segments = line.split("|")
    if len(segments) < 4 or segments[0].strip() != "INDICATOR":
        raise RoadmapError("Indicator line must be INDICATOR|name|unit|target")
    # Permit a vertical bar in the label, but not in the unit/target.
    return _indicator({"name": "|".join(segments[1:-2]),
                       "unit": segments[-2].strip(), "target": segments[-1].strip()})


def _parse_batch(raw: str, numbers: list[int]) -> tuple[list[dict], str]:
    sprints, current, meta_message = [], None, ""
    for line in _lines(raw):
        if line.startswith("BEGIN_SPRINT|"):
            if current is not None:
                raise RoadmapError("Nested sprint block")
            current = {"number": _required_number(line.partition("|")[2]),
                       "tasks": [], "indicators": []}
            continue
        if line == "END_SPRINT":
            if current is None:
                raise RoadmapError("END_SPRINT without BEGIN_SPRINT")
            sprints.append(_validate_sprint(current, current["number"]))
            current = None
            continue
        if line.startswith("META_MESSAGE|"):
            message = _text(line.partition("|")[2], limit=1100, arabic=True)
            if message:
                meta_message = message
            continue
        if current is None:
            continue
        key, delim, val = line.partition("|")
        if not delim:
            raise RoadmapError("Unrecognized line in sprint block")
        if key == "TITLE":
            current["title"] = val
        elif key == "DESCRIPTION":
            current["description"] = val
        elif key == "TASK":
            current["tasks"].append(_task_line(val))
        elif key == "INDICATOR":
            try:
                current["indicators"].append(_parse_indicator_line(line))
            except RoadmapError as exc:
                # Name the exact sprint and offending line in the *repair*
                # instruction. V6 only forwarded a generic validation error,
                # which caused the Meta model to repeat the same weak KPI.
                indicator_name = _text("|".join(line.split("|")[1:-2]), limit=160)
                raise RoadmapError(
                    f"Sprint {current['number']} indicator {indicator_name!r}: {exc}"
                ) from exc
        else:
            raise RoadmapError("Unknown sprint output field")
    if current is not None:
        raise RoadmapError("Unclosed sprint block")
    if [x["number"] for x in sprints] != numbers:
        raise RoadmapError(f"Sprint block numbers do not match: expected {numbers}")
    if not meta_message:
        raise RoadmapError("Meta Advisor message missing")
    return sprints, meta_message


def _parse_regeneration(raw: str, kind: str, existing: Any) -> tuple[dict, str]:
    tasks, indicator, message = [], None, ""
    for line in _lines(raw):
        if line.startswith("META_MESSAGE|"):
            message = _text(line.partition("|")[2], limit=1100, arabic=True)
        elif kind == "sprint" and line.startswith("TASK|"):
            tasks.append(_task_line(line.partition("|")[2]))
        elif kind == "sprint_indicator" and line.startswith("INDICATOR|"):
            if indicator is not None:
                raise RoadmapError("Regeneration returned multiple indicators")
            indicator = _parse_indicator_line(line)
    if not message:
        raise RoadmapError("Meta Advisor message missing")
    if kind == "sprint":
        out = _validate_tasks(tasks)
        previous = [_norm(x.get("text") if isinstance(x, dict) else x) for x in existing]
        if previous == [_norm(x["text"]) for x in out]:
            raise RoadmapError("Regeneration returned unchanged task list")
        return {"tasks": out}, message
    if indicator is None:
        raise RoadmapError("Regeneration returned no indicator")
    if _norm(indicator["name"]) in {_norm(x) for x in existing}:
        raise RoadmapError("Regeneration repeated an existing indicator name")
    return indicator, message


def _task_advisory_insights(opinions: list[dict]) -> list[dict]:
    """Pass independent specialist advice to Meta without advisor handles.

    The transcript and advisor_reasonings still preserve the actual selected
    advisor IDs.  Those IDs and advisor titles have no operational reason to
    appear inside the task-list generation prompt, and encourage the model to
    generate steps like 'call AOS-FG-18' instead of NGO work.
    """
    result = []
    for opinion in opinions:
        item = {}
        for field in ("reasoning", "recommendation"):
            value = _text(opinion.get(field), limit=750, arabic=True)
            if value and not _ANY_ADVISOR_ID.search(value) and not _ADVISOR_AS_TASK.search(value):
                item[field] = value
        if item:
            result.append(item)
    return result


def _recover_valid_task_subset(raw: str, existing: list[Any]) -> tuple[dict, str]:
    """Keep only Meta's valid executable tasks, never author new model output.

    If at least 3 different new tasks remain, the response still matches the
    contract's single tasks[] rewrite. If fewer remain, reject and retry; no
    misleading success or invented deterministic fallback is allowed.
    """
    lines = _lines(raw)
    notes = [_text(line.partition("|")[2], limit=1100, arabic=True)
             for line in lines if line.startswith("META_MESSAGE|")]
    notes = [x for x in notes if x]
    if not notes:
        raise RoadmapError("Meta Advisor message missing")
    task_lines = [line.partition("|")[2] for line in lines if line.startswith("TASK|")]
    if not task_lines:
        raise RoadmapError("No tasks to recover")
    valid, seen = [], set()
    for candidate in task_lines:
        try:
            task = _validate_tasks([_task_line(candidate)])[0]
        except RoadmapError:
            continue
        key = _norm(task["text"])
        if key not in seen:
            seen.add(key)
            valid.append(task)
    # The contract allows at least one, but 3-5 ensures a useful weekly list.
    if not 3 <= len(valid) <= 5:
        raise RoadmapError("Too few valid executable tasks after filtering")
    old_tasks = [_norm(x.get("text") if isinstance(x, dict) else x)
                 for x in existing]
    if old_tasks == [_norm(x["text"]) for x in valid]:
        raise RoadmapError("Regeneration returned unchanged task list")
    return {"tasks": valid}, notes[-1]


def _check_indicator_follows_reason(indicator: dict, reason: str) -> None:
    """On an explicit weekly-completion rewrite, block baseline/outcome ratios.

    The contract allows proposed numeric targets without benchmarks. This guard
    *only* checks that a completion-oriented user request receives a progress
    metric, not another measurement of the association's current finances.
    """
    if not _WEEKLY_COMPLETION_REASON.search(reason or ""):
        return
    if not _COMPLETION_WORDS.search(indicator["name"]):
        raise RoadmapError(
            "User requested a measurable sprint completion indicator, but the "
            "suggested name describes a baseline/outcome rather than completed work. "
            "Use a documented/approved/classified/completed deliverable."
        )


def _meta_context(advisors: list[dict], opinions: list[dict]) -> list[dict]:
    titles = {x["slug"]: x["title"] for x in advisors}
    return [{"slug": o["advisor_id"], "role": titles[o["advisor_id"]],
             "reasoning": o["reasoning"], "recommendation": o["recommendation"]} for o in opinions]


def _meta_system(persona: str) -> str:
    return persona + "\n\n" + (
        "أنت تقود مجلس خارطة الطريق طبقًا لعقد Screen 5. مخرج السبرينت المعين من Backend هو المرجع الملزم. "
        "لا تُغيّر output أو phase أو results. إذا تكرر المخرج عبر أسابيع متجاورة، ابنِ عليه بالتدرج بدل إعادة نفس المهام. "
        "المهام موجهة لفريق الجمعية وليست أوامر لتشغيل المستشارين أو التواصل معهم: "
        "ممنوع أن يظهر أي رمز مستشار AOS- أو لفظ مستشار أو خبير في TITLE وDESCRIPTION وTASK وINDICATOR. "
        "حوّل الرأي الاستشاري إلى فعل قابل للتسليم مثل 'تصنيف الإيرادات حسب المصدر في جدول موحد'. "
        "كل مهمة إجراء عملي واحد محدد يمكن إتمامه في خمسة أيام عمل، بترتيب الجمع ثم الإعداد ثم المراجعة/الاعتماد المناسب للمخرج. "
        "اختر مؤشرات تقيس إنجاز المخرج خلال هذا السبرينت نفسه، مثل اكتمال البيانات ذات الصلة أو عدد المصادر المصنفة "
        "أو عدد التقارير التي سُلّمت، ولا تستخدم مؤشرات غامضة مثل 'التوصيات المؤثرة' أو 'التوصيات المؤجلة'. "
        "في سبرينت التقييم أو التقرير الختامي: اختر مقياسًا لعمل موثق، مثل عدد تقارير التقييم المُعدّة، "
        "نسبة اكتمال التقرير الختامي، عدد الدروس المستفادة الموثقة، أو عدد إجراءات المتابعة المحددة؛ "
        "لا تكتب عدد التوصيات المؤثرة أو المؤجلة أو تقييمًا انطباعيًا لا يمكن التحقق منه. "
        "يُسمح لك باقتراح مستهدفات target رقمية معقولة حتى بلا Benchmarks، لكنها مقترحات تنفيذية وليست حقائق تاريخية. "
        "لا تفترض توفر عدد سنوات أو سجلات أو نسب فعلية لم يذكرها سياق المخرج أو المنظمة؛ استخدم 'البيانات المتاحة' عند غيابها. "
        "الوحدات المسموحة فقط percent أو number أو multiplier، وpercent من 0 إلى 100 لا 0 إلى 1. "
        "كل مؤشر يبدأ بكلمة نسبة يجب أن تكون وحدته percent، ويبدأ بعدد يجب أن تكون وحدته number. "
        "لا تستخدم عددًا لتمثيل نسبة حتى لو ذكر المستهدف 10%. لا تغيّر معنى المؤشر أو وحدته دون تطابق دقيق. "
        "في إعادة التوليد إذا طلبت الجمعية قياس إنجاز العمل هذا الأسبوع، لا ترجع نسبة توزيع التمويل "
        "(مثل نسبة الإيرادات غير الموسمية من إجمالي الدخل) لأنها تصف وضعًا ماليًا لا إنجاز مهمة. "
        "اختر بدلاً منها نسبة اكتمال توثيق البيانات أو عدد التقارير المعتمدة أو عدد المصادر المصنفة. "
        "اجعل رسالة META_MESSAGE متفقة مع اسم المؤشر ووحدته ومستهدفه النهائي في الإخراج. "
        "لا تذكر أي مستشار داخل رسائل المجلس إلا برمز slug صحيح لمستشار وارد في advisors الذين ساهموا، "
        "واكتب صوت الميتا بصيغة meta_advisor؛ لا تختلق مستشارين جدد. "
        "اكتب عربية فصحى سليمة وخالية من الكلمات الكورية والصينية والألفاظ المحرّفة أو الركيكة. "
        "لا تكتب الحقول الداخلية output.text أو output.results أو outputs/indicators أو رموز track داخل النص العربي. "
        "ممنوع المؤشرات العامة المنسوخة مثل عدد مخرجات العمل الموثقة أو عدد العناصر المنجزة؛ "
        "اكتب مؤشرات تحمل اسم الشيء المطلوب في المخرج، ومهمة تنتهي بصياغته أو تسليمه. "
        "لا تخرج عن مخرج السبرينت المعين من النظام أو توصي بأعمال تتبع مخرجات مختلفة. "
        "تأكد من سلامة المهام وعناوينها وMETA_MESSAGE؛ لا تُخرج JSON. التزم ببروتوكول السطور حرفيًا."
    )


def _indicator_examples_for_output(output: Any) -> list[str]:
    """Output-specific examples, never generic placeholder KPIs."""
    profile = _profile_for_output(output)
    if profile:
        return profile["examples"]
    title = _text((output or {}).get("text"), limit=200) if isinstance(output, dict) else ""
    return [f"عدد بنود «{title}» الموثقة|number|3"] if title else []


def _execution_focus_for_output(output: Any) -> str:
    """Offer grounded, output-type-specific execution guidance."""
    profile = _profile_for_output(output)
    if profile:
        return profile["focus"]
    name = _text(output.get("text"), limit=180) if isinstance(output, dict) else ""
    return (f"أنتج المخرج المحدد «{name}» من بيانات الجمعية، واعرض خطوة تسليم واضحة."
            if name else "نفذ العمل المناسب لهدف الجمعية من دون افتراض مخرجات مفقودة.")


# Screen 5 V10: classify only from the Backend-assigned output, never invent
# a new output, phase or result. The examples are guidance for the Meta model,
# not hard-coded response content.
_INTERNAL_LEAK = re.compile(
    r"(?i)(?:\boutputs?\s*[./]\s*(?:text|results|indicators)\b|"
    r"\boutput\s*(?:[0-9]+|\.|/)|\bfinancial[-_]sustainability\b|"
    r"\b(?:payload|indicators|results)\s*[./]\s*(?:text|name|unit|target)\b|"
    r"\b(?:output|outputs|results|indicators|payload|track)\b)"
)
_GENERIC_KPI = frozenset({
    "عدد مخرجات العمل الموثقة", "نسبة اكتمال توثيق المخرج", "عدد العناصر المنجزة",
    "نسبة اكتمال إعداد المخرج", "عدد عناصر المخرج المكتملة", "عدد العناصر المراجعة والموثقة",
    "عدد التقارير المعدة", "عدد النتائج المسجلة في التقرير",
})

# Priority matters: 'تقرير ختامي' must be classified before 'تقرير'.
_OUTPUT_PROFILES = (
    ("final", ("ختامي",), ("تقرير", "ختامي", "متابعة", "خطة"),
     ("تقرير", "الخطة", "المتابعة", "الختامي"),
     ("عدد التقارير الختامية المعدة|number|1", "نسبة اكتمال خطة المتابعة|percent|100", "عدد إجراءات المتابعة المجدولة|number|3"),
     "أنجز التقرير الختامي اعتمادًا على تقييم سابق، وأعد خطة متابعة مستقلة تتضمن إجراءات ومواعيد وآلية مراجعة؛ لا تكتف بكتابة التقرير أو جدوله الزمني."),
    ("evaluating", ("تقييم",), ("تقييم", "نتائج", "تقرير", "دروس"),
     ("التقييم", "نتائج", "الأدلة", "الدروس"),
     ("عدد تقارير تقييم التنفيذ المعدة|number|1", "نسبة توثيق نتائج الإجراءات|percent|100", "عدد الدروس المستفادة المسجلة|number|3"),
     "وثق الأدلة ونتائج الإجراءات، ثم أعد تقرير تقييم وخلاصة دروس مستفادة."),
    ("offers", ("عروض", "شراكة"), ("عروض", "عرض", "شراكة", "شركاء"),
     ("العروض", "عرض", "الشركاء", "الشراكة"),
     ("عدد عروض الشراكة الجاهزة|number|2", "نسبة اكتمال محتوى عروض الشراكة|percent|100", "عدد عروض الشراكة المراجعة|number|2"),
     "اجمع عناصر القيمة المقترحة للشريك، وصغ عرضًا قابلًا للتخصيص، ثم راجع حزمة العروض النهائية."),
    ("dashboard", ("لوحة", "مؤشر"), ("لوحة", "مؤشرات", "بيانات"),
     ("اللوحة", "المؤشرات", "البيانات"),
     ("عدد مؤشرات الاستدامة المعرّفة|number|4", "نسبة اكتمال لوحة المتابعة|percent|100", "عدد مصادر بيانات المؤشرات الموثقة|number|3"),
     "حدّد تعريف كل مؤشر ومصدره وطريقة تحديثه، ثم أنشئ أو صمّم لوحة مؤشرات قابلة للاستخدام مع طريقة تحديثها؛ تحديد المؤشرات وحده لا ينجز المخرج."),
    ("pilot", ("تجربة",), ("تجربة", "نتائج", "قناة", "اختبار"),
     ("التجربة", "القناة", "نتائج", "اختبار"),
     ("عدد التجارب التجريبية المنفذة|number|1", "عدد نتائج اختبار قناة التمويل الموثقة|number|3", "نسبة اكتمال توثيق التجربة|percent|100"),
     "حدد نطاق تجربة صغيرة لقناة التمويل، نفذ الخطوات الممكنة، وقس النتائج ووثّقها؛ لا تكتف بإعادة كتابة التوصية."),
    ("guide", ("دليل", "إجراءات"), ("دليل", "إجراءات", "خطوات", "موافقات"),
     ("الدليل", "الإجراءات", "الخطوات", "الموافقات"),
     ("عدد إجراءات تنمية الموارد الموثقة|number|5", "نسبة اكتمال مسودة دليل الإجراءات|percent|100", "عدد خطوات الموافقة المراجعة|number|3"),
     "ارسم تسلسل الإجراءات وحدد النماذج والموافقات والضوابط ثم أعد دليلًا قابلًا للتطبيق."),
    ("campaign", ("حملة",), ("حملة", "قنوات", "رسائل", "محتوى"),
     ("الحملة", "القنوات", "الرسائل", "المحتوى"),
     ("عدد رسائل الحملة التمويلية المُعدّة|number|3", "عدد قنوات الحملة المحددة|number|2", "نسبة اكتمال خطة الحملة الرقمية|percent|100"),
     "اختر القنوات، وصغ الرسائل والمحتوى، وحدّد جدول الحملة ومؤشرات متابعتها؛ جهّز خطة قابلة للتنفيذ."),
    ("cashflow", ("تدفقات",), ("نموذج", "تدفقات", "النقدية", "توقع"),
     ("التدفقات", "النقدية", "النموذج", "التوقع"),
     ("عدد نماذج توقع التدفقات النقدية المعدة|number|1", "نسبة اكتمال بيانات التدفقات النقدية|percent|100", "عدد فترات التوقع المدرجة|number|3"),
     "صنّف التدفقات الداخلة والخارجة المتاحة، وابنِ نموذج توقع وراجع الفجوات التمويلية."),
    ("matrix", ("مصفوفة",), ("مصفوفة", "جدوى", "معايير", "تقييم"),
     ("المصفوفة", "الجدوى", "المعايير", "التقييم"),
     ("عدد معايير تقييم الجدوى المحددة|number|4", "عدد مصادر الإيرادات المقيّمة|number|3", "نسبة اكتمال مصفوفة الجدوى|percent|100"),
     "حدّد معايير الجدوى، وقارن فرص الإيراد، ووثّق ترتيبها في مصفوفة قرار."),
    ("partnerships", ("فرص", "شراكات"), ("قائمة", "فرص", "شراكات", "تواصل"),
     ("القائمة", "الفرص", "الشراكات", "الجهات"),
     ("عدد فرص الشراكات التمويلية المحصورة|number|5", "عدد جهات الشراكة المصنفة|number|3", "نسبة اكتمال قائمة فرص الشراكات|percent|100"),
     "احصر فرص الشراكات الملائمة وصنّف الجهات، ثم أعد قائمة بالفرص ورسائل تواصل أولية."),
    ("income_plan", ("خطة", "تنويع"), ("خطة", "تنويع", "دخل", "اعتماد"),
     ("الخطة", "التنويع", "الدخل", "الاعتماد"),
     ("عدد مصادر الدخل البديلة المقترحة|number|3", "نسبة اكتمال صياغة خطة التنويع|percent|100", "عدد عناصر خطة الدخل المعتمدة|number|3"),
     "حوّل التحليل إلى خطة بدائل وأولويات وجدول تنفيذ؛ إذا اشترط المخرج الاعتماد فأضف مهمة عرض الخطة على الجهة المعتمدة للاعتماد، لا تفترض اعتمادًا تم مسبقًا."),
    ("income_report", ("تقرير", "تحليل"), ("تقرير", "تحليل", "تمويل", "دخل"),
     ("التقرير", "المصادر", "الدخل", "التمويل"),
     ("عدد مصادر الدخل المصنفة|number|5", "نسبة اكتمال تحليل مصادر التمويل|percent|100", "عدد تقارير تحليل التمويل المعدة|number|1"),
     "اجمع السجلات المتاحة وصنّف الإيرادات، ثم أعد تقرير تحليل مصادر التمويل بنفسه."),
)
_DELIVERABLE_VERBS = re.compile(
    r"(?:أعد|اعد|إعداد|انشئ|أنشئ|صمم|صمّم|ابن|ابنِ|بناء|جهز|جهّز|"
    r"صياغة|صغ|وثق|وثّق|سجّل|سجل|حدّد|حدد|أنجز|أنجز|راجع|اكتب|رتب|رتّب|اعرض|عرض|استكمل)"
)


def _profile_for_output(output: Any) -> dict | None:
    if not isinstance(output, dict):
        return None
    text = _text(output.get("text"), limit=500)
    if not text:
        return None
    ordered = sorted(_OUTPUT_PROFILES, key=lambda p: (p[0] != "matrix", p[0] != "offers"))
    for name, triggers, artifact, indicator_terms, examples, focus in ordered:
        if all(t in text for t in triggers):
            return {"kind": name, "artifact_terms": artifact,
                    "indicator_terms": indicator_terms, "examples": list(examples),
                    "focus": focus}
    return None  # unknown outputs: don't impose arbitrary hard-coded rules


def _claims_deliverable_action(task: str, artifact: tuple[str, ...]) -> bool:
    """A negated mention ('دون تجهيز عرض') is not a deliverable action."""
    if not _DELIVERABLE_VERBS.search(task) or not any(w in task for w in artifact):
        return False
    # Local negation before a production verb invalidates the evidence when
    # it is the only mention of the deliverable in the task.
    if re.search(r"(?:دون|بدون|بلا|عدم)\s+(?:\S+\s+){0,3}(?:عرض|عروض|حزمة|تقرير|لوحة|مصفوفة|نموذج|قائمة|دليل|خطة|تجربة)", task):
        return False
    if re.search(r"(?:دون|بدون|بلا|عدم)\s+(?:\S+\s+){0,2}(?:تجهيز|إعداد|صياغة|بناء|إنشاء)\s+", task):
        return False
    return True


# V11: these checks assert actual production of the promised artifact. A task
# that merely mentions a dashboard or a plan as context is not sufficient.
_ARTIFACT_PRODUCTION_VERBS = re.compile(
    r"(?:أنشئ|انشئ|أنشاء|إنشاء|اعد|أعد|إعداد|جهز|جهّز|تجهيز|"
    r"صمم|صمّم|تصميم|ابن|ابنِ|بناء|اطور|طوّر|تطوير|اكتب|كتابة|"
    r"حرر|حرّر|صياغة|صغ|طور|أنجز|انتج|أنتج)"
)


def _builds_named_artifact(tasks: list[str], artifact: str) -> bool:
    """Require a build/write verb near the artifact itself, in the same task."""
    artifact_re = {
        "dashboard": r"(?:لوح(?:ة|ه)\s+(?:مؤشرات|متابعة)|لوحة\s+(?:قياس|تتبع))",
        "followup": r"(?:خطة\s+(?:المتابعة|متابعة|الرصد)|برنامج\s+متابعة)",
        "report": r"(?:التقرير\s+الختامي|تقرير\s+ختامي)",
    }[artifact]
    pattern = re.compile(r"(?:" + _ARTIFACT_PRODUCTION_VERBS.pattern + r")(?:.{0,90}?)(?:" + artifact_re + r")")
    for task in tasks:
        if not _claims_deliverable_action(task, ("لوحة", "خطة", "تقرير")):
            continue
        for match in pattern.finditer(task):
            # 'إعداد جدول زمني للتقرير' writes a timetable, not the report.
            # Likewise a list *about* a dashboard is not a built dashboard.
            middle = task[match.start():match.end()]
            if re.search(r"(?:جدول(?:ا|ًا)?|قائمة|مصفوفة|ملخص).{2,70}(?:ل(?:ل)?|عن)", middle):
                continue
            return True
    return False


def _includes_approval_submission(tasks: list[str]) -> bool:
    """Approval must be requested from a human; don't claim it already happened."""
    return any(re.search(
        r"(?:اعرض|عرض|ارفع|رفع|قدم|قدّم|تقديم|احل|أحل|أرسل|إرسال)"
        r".{0,110}(?:خطة|الخطة|اعتماد|المجلس|الإدارة)"
        r".{0,110}(?:للاعتماد|للمراجعة|مجلس|الإدارة|الموافقة|الاعتماد)", task
    ) for task in tasks)


def _content_quality_issues(sprint: dict, output: Any, earlier: list[dict]) -> dict[str, str]:
    """Conservative content checks; repair with Meta instead of changing data.

    Refuse template placeholders, generic recycled KPIs, and absent tangible
    deliverables for recognized output types. Preserve flexibility for unknown
    user-authored outputs (including null outputs).
    """
    issues = {}
    heading = sprint["title"] + " " + sprint["description"]
    if _INTERNAL_LEAK.search(heading):
        issues["card"] = "Title/description exposes internal JSON field names"
    task_texts = [x["text"] for x in sprint["tasks"]]
    if any(_INTERNAL_LEAK.search(x) for x in task_texts):
        issues["tasks"] = "Task exposes internal JSON field names instead of NGO work"
    indicators = sprint["indicators"]
    names = [_norm(x["name"]) for x in indicators]
    if any(_INTERNAL_LEAK.search(x["name"]) for x in indicators):
        issues["indicators"] = "Indicator exposes internal JSON field names"
    if any(name in _GENERIC_KPI for name in names):
        issues["indicators"] = "Generic KPI repeated across unrelated deliverables"

    profile = _profile_for_output(output)
    if profile:
        # One explicit action must result in or directly prepare the actual
        # named deliverable; generic data gathering alone is not sufficient.
        artifact = profile["artifact_terms"]
        if not any(_claims_deliverable_action(task, artifact)
                   for task in task_texts):
            issues["tasks"] = f"Sprint tasks never produce the assigned {profile['kind']} deliverable"
        specific = sum(any(term in name or (term.startswith("ال") and term[2:] in name)
                           for term in profile["indicator_terms"])
                       for name in names)
        if specific < 2:
            issues["indicators"] = f"At least 2 KPIs must measure the assigned {profile['kind']} deliverable"
        # Multi-part deliverables need all their parts. A dashboard KPI does
        # not prove somebody built the dashboard; a final report alone does
        # not satisfy a separate follow-up plan.
        kind = profile["kind"]
        if kind == "dashboard" and not _builds_named_artifact(task_texts, "dashboard"):
            issues["tasks"] = (
                "Sprint tasks identify KPIs but never actually create/build the assigned dashboard"
            )
        if kind == "final":
            if not _builds_named_artifact(task_texts, "report"):
                issues["tasks"] = "Sprint tasks never write the final report"
            if not _builds_named_artifact(task_texts, "followup"):
                issues["tasks"] = "Sprint tasks never create the required follow-up plan"
        if kind == "income_plan" and isinstance(output, dict):
            # Only demand a submission when Backend explicitly requested an
            # approved plan. Approval itself is outside AI's control.
            output_text = _text(output.get("text"), limit=500)
            if "معتمد" in output_text and not _includes_approval_submission(task_texts):
                issues["tasks"] = (
                    "Backend output requires an approved plan: include a task to submit the plan for review/approval"
                )
    # Two completely shared KPIs means the cards are substantially templated.
    for prior in earlier:
        shared = len(set(names) & {_norm(x["name"]) for x in prior["indicators"]})
        if shared >= 2:
            issues["indicators"] = (
                f"{shared} identical KPIs reused from sprint {prior['number']}; "
                "indicators must measure this output"
            )
            break
    return issues


def _parse_quality_repair(raw: str, fields: set[str], old: dict) -> tuple[dict, str]:
    if fields == {"card"}:
        # A full sprint rewrite is necessary to fix a corrupted headline.
        rows, note = _parse_batch(raw, [old["number"]])
        return rows[0], note
    tasks, indicators, notes = [], [], []
    for line in _lines(raw):
        if line.startswith("TASK|"):
            tasks.append(_task_line(line.partition("|")[2]))
        elif line.startswith("INDICATOR|"):
            indicators.append(_parse_indicator_line(line))
        elif line.startswith("META_MESSAGE|"):
            notes.append(_text(line.partition("|")[2], limit=900, arabic=True))
    if not notes or not notes[-1]:
        raise RoadmapError("Quality repair is missing META_MESSAGE")
    repaired = dict(old)
    if "tasks" in fields:
        repaired["tasks"] = _validate_tasks(tasks, minimum=3)
        if len(repaired["tasks"]) > 5:
            raise RoadmapError("Quality repair must return 3-5 tasks")
    if "indicators" in fields:
        if not 3 <= len(indicators) <= 5:
            raise RoadmapError("Quality repair must return 3-5 indicators")
        if len({_norm(x["name"]) for x in indicators}) != len(indicators):
            raise RoadmapError("Quality repair repeated the same KPI")
        repaired["indicators"] = indicators
    return _validate_sprint(repaired, old["number"]), notes[-1]


def _repair_sprint_quality(*, council: Any, system: str, sprint: dict, source: dict,
                           opinions: list[dict], previous: list[dict],
                           errors: dict[str, str]) -> tuple[dict, str]:
    """Targeted same-job Meta repair: preserve valid fields; no template output."""
    output = source.get("output")
    profile = _profile_for_output(output)
    fields = set(errors)
    if "card" in fields:
        fields = {"card"}
    task = {
        "quality_repair": True,
        "number": sprint["number"],
        "target_fields": sorted(fields),
        "source_output": output,
        "original_title": sprint["title"] if not _INTERNAL_LEAK.search(sprint["title"]) else None,
        "original_description": sprint["description"] if not _INTERNAL_LEAK.search(sprint["description"]) else None,
        # Do not include erroneous prior tasks/KPI labels: the model used to
        # copy such examples in V8/V9. Preserve them only in Python validation.
        "why_rejected": errors,
        "deliverable_execution_focus": profile["focus"] if profile else
            "أنجز المخرج المحدد في سياق السبرينت، لا مجرد وصف التوصيات.",
        "specific_indicator_examples": profile["examples"] if profile else [],
        "previous_stage_headings": [{"number": row["number"], "title": row["title"]}
                                    for row in previous[-4:]],
        "advisory_insights": _task_advisory_insights(opinions),
        "task": ("صحح الحقول المطلوبة فقط. أفعال ملموسة لفريق الجمعية، 3-5 مهام تنتج "
                 "المخرج المخصص من Backend، و3-5 مؤشرات تقيس جوانبه بدقة. "
                 "إذا المطلوب لوحة مؤشرات، أضف مهمة إنشاء/تصميم اللوحة نفسها، لا تعريف مؤشرات فقط. "
                 "إذا المطلوب تقرير ختامي وخطة متابعة، أضف مهمتين واضحتين لإنجاز كليهما. "
                 "إذا المخرج خطة معتمدة، أضف مهمة عرض الخطة للاعتماد دون الادعاء أنها معتمدة فعلًا. "
                 "ممنوع مؤشرات القوالب العامة أو حقول برمجية. "
                 "لا تغير أي حقل آخر ولا تفترض حقائق غير مذكورة. "),
        "required_output_format": (
            f"BEGIN_SPRINT|{sprint['number']}\nTITLE|عنوان خاص بالمخرج\nDESCRIPTION|وصف مرتبط بالمخرج\nTASK|مهمة محددة\nINDICATOR|اسم مؤشر خاص بالمخرج|number|1\nEND_SPRINT\nMETA_MESSAGE|ملخص"
            if fields == {"card"} else
            ("TASK|مهمة تنفيذية أولى\nTASK|مهمة تنفيذية ثانية\nTASK|مهمة تنفيذية ثالثة\n" if "tasks" in fields else "")
            + ("INDICATOR|مؤشر أول متعلق بالمخرج|number|1\nINDICATOR|مؤشر ثان متخصص|percent|100\nINDICATOR|مؤشر ثالث متخصص|number|3\n"
               if "indicators" in fields else "")
            + "META_MESSAGE|سبب الإصلاح بإيجاز"
        ),
    }
    attempts = max(1, min(3, int(os.getenv("ROADMAP_V10_QUALITY_ATTEMPTS", "2"))))
    last = errors
    for attempt in range(attempts):
        payload = dict(task)
        if attempt:
            payload["last_validation_error"] = last
            payload["task"] += "تأكد أن كل مؤشر خاص بهذا المخرج، وأن مهمة على الأقل تنتج العمل المطلوب نصًا."
        raw = _meta_generate(council, system, payload,
                             int(os.getenv("ROADMAP_V10_REPAIR_TOKENS", "1150")),
                             sample=(attempt > 0), block_non_arabic_scripts=True)
        try:
            repaired, note = _parse_quality_repair(raw, fields, sprint)
            field_issues = _sprint_arabic_issues(repaired)
            if field_issues:
                raise RoadmapError("; ".join(field_issues.values()))
            if _INTERNAL_LEAK.search(note):
                raise RoadmapError("Meta transcript contains internal JSON identifiers")
            _check_council_message(note, {o["advisor_id"] for o in opinions})
            _check_historical_lookback(repaired, {"output": output})
            remaining = _content_quality_issues(repaired, output, previous)
            if remaining:
                raise RoadmapError("; ".join(remaining.values()))
            _check_recent_history(repaired, previous)
            print(f"[roadmap] quality repair succeeded sprint={sprint['number']} "
                  f"fields={','.join(sorted(fields))} attempt={attempt+1}", flush=True)
            return repaired, note
        except RoadmapError as exc:
            last = str(exc)
            print(f"[roadmap] quality repair rejected sprint={sprint['number']} "
                  f"attempt={attempt+1}/{attempts}: {last}", flush=True)
    raise RoadmapError(f"Sprint {sprint['number']} quality repair failed: {last}")


_SUSPECT_META_NOTE = re.compile(
    r"(?:المجالدين|التضاريد|التضاريب|صنّاف|المستفدة|تبدأين|جوهرة|الجوهرة|استدعِ)"
    r"|إال|\bSprint\b", re.I
)


def _ensure_readable_meta_note(council: Any, system: str, note: str,
                               sprints: list[dict], opinions: list[dict]) -> str:
    """Repair only an unreadable internal Meta message, never invent a transcript.

    Do not re-run the whole 12-sprint generation because its admin-only note
    contained an English template variable or a known corrupted expression.
    """
    selected = {x["advisor_id"] for x in opinions}
    if not _INTERNAL_LEAK.search(note) and not _SUSPECT_META_NOTE.search(note) and not _arabic_field_error(note):
        _check_council_message(note, selected)
        return note
    for attempt in range(2):
        context = {
            "task": "اكتب رسالة واحدة موجزة بصوت المستشار الأعلى تصف القرارات الحقيقية المبنية على عناوين السبرينتات.",
            "sprints": [{"number": x["number"], "title": x["title"]} for x in sprints],
            "reasons": _task_advisory_insights(opinions),
            "constraints": "عربية فصحى فقط، دون رموز حقول برمجية أو مصطلح Sprint أو كلمات مشوهة أو مستشار غير مختار.",
            "required_output_format": "META_MESSAGE|شرح عربي موجز لخارطة الطريق",
        }
        raw = _meta_generate(council, system, context, 320,
                             sample=(attempt > 0), block_non_arabic_scripts=True)
        try:
            _require_generated_arabic_quality(raw)
            lines = [line.partition("|")[2] for line in _lines(raw)
                     if line.startswith("META_MESSAGE|")]
            if len(lines) != 1:
                raise RoadmapError("Meta message rewrite must return exactly one line")
            new = _text(lines[0], limit=900, arabic=True)
            if not new or _INTERNAL_LEAK.search(new) or _SUSPECT_META_NOTE.search(new) or _arabic_field_error(new):
                raise RoadmapError("Meta message still contains internal jargon or corrupted wording")
            _check_council_message(new, selected)
            print("[roadmap] isolated meta transcript rewrite succeeded", flush=True)
            return new
        except RoadmapError as exc:
            print(f"[roadmap] isolated meta transcript rewrite attempt={attempt+1} rejected: {exc}",flush=True)
    raise RoadmapError("Cannot produce a readable meta transcript message")


def _repair_colliding_tasks(
    *,
    council: Any,
    system: str,
    data: dict,
    sprint: dict,
    source: dict,
    prior_rows: list[dict],
    opinions: list[dict],
    collision: str,
    repair_budget: dict[int, int],
) -> tuple[dict, str]:
    """Ask the *actual Meta model* for only the offending task list.

    Do not expose the full earlier task wording to the model: feeding rejected
    tasks back as examples made it copy them in earlier builds.  We provide
    authoritative output/results and previous stage titles, not task text.
    This operates inside the same job, using the already loaded Meta adapter.
    No synthetic tasks are inserted, and failed repair cannot be published.
    """
    number = sprint["number"]
    request = {
        "task": (
            "أعد صياغة مهام السبرينت المحدد فقط؛ الموديل كرر مهام سبرينت أقدم. "
            "اكتب 3 إلى 5 مهام جديدة ملموسة، مميزة لهذا السبرينت ومخرجه المحدد من النظام، "
            "يمكن لفريق الجمعية إنجازها خلال خمسة أيام عمل. "
            "لا تكتب عنوانًا أو وصفًا أو مؤشرات أو تغيّر المخرج. "
            "إذا كان المخرج متعلقًا بالتقييم، أنجز تحليل الأدلة والتقييم نفسه. "
            "إذا كان المخرج تقريرًا ختاميًا، اجمع نتائج التقييم واكتب التقرير وخطة المتابعة. "
            "اجعل القائمة تنتج artifact ملموسًا مثل مستند أو مصفوفة أو خطة، لا مجرد إعادة جمع عامة. "
            "تجنب تكرار أعمال السبرينتات السابقة ولا تخترع سجلات أو نتائج غير مذكورة. "
            "لا تُعيّن مستشارًا للتنفيذ، ولا تستخدم أي أكواد مستشارين في المهام. "
            "الكتابة بالعربية الفصحى وببروتوكول السطور فقط."
        ),
        "collision_diagnostic": collision,
        "sprint_to_rewrite": {
            "number": number,
            "output": source.get("output"),
            "title": sprint["title"],
            "description": sprint["description"],
        },
        "output_execution_focus": _execution_focus_for_output(source.get("output")),
        "goal": data["payload"].get("goal"),
        "intervention": {"title": (data["payload"].get("intervention") or {}).get("title")}
        if isinstance(data["payload"].get("intervention"), dict) else None,
        "independent_advisor_insights": _task_advisory_insights(opinions),
        "prior_stage_headings": [
            {"number": x["number"], "title": x["title"]}
            for x in prior_rows
        ],
        "required_output_format": (
            "TASK|مهمة عملية خاصة بمخرج السبرينت\n"
            "TASK|مهمة لاحقة تكمل الأولى\n"
            "TASK|مهمة تنتج جزءًا قابلًا للتحقق من المخرج\n"
            "META_MESSAGE|تفسير عربي موجز للتعديل"
        ),
    }
    max_per_sprint = max(1, min(3, int(os.getenv("ROADMAP_COLLISION_REPAIR_ATTEMPTS", "2"))))
    remaining = repair_budget.get(number, max_per_sprint)
    if remaining <= 0:
        raise RoadmapError(f"Sprint {number} focused repair budget exhausted within this job")
    attempts = min(2, remaining)
    last_error = collision
    for attempt in range(attempts):
        repair_budget[number] = repair_budget.get(number, max_per_sprint) - 1
        payload = dict(request)
        if attempt:
            payload["repair_instruction"] = (
                "الصياغة السابقة لم تحقق شرط التميز: " + last_error + ". "
                "غيّر نوع الأعمال وأفعال التنفيذ واربط كل مهمة مباشرة بإنتاج المخرج، "
                "ولا تكرر مراحل جمع البيانات العامة من سبرينتات سابقة."
            )
        raw = _meta_generate(
            council, system, payload,
            int(os.getenv("ROADMAP_REPAIR_TASK_TOKENS", "850")),
            sample=True,
            block_non_arabic_scripts=True,
        )
        try:
            # Preserve valid tasks if only the admin-only note has a typo.
            suggestion, note = _parse_regeneration(raw, "sprint", sprint["tasks"])
            tasks = suggestion["tasks"]
            for task_row in tasks:
                _require_generated_arabic_quality(task_row["text"])
            if not 3 <= len(tasks) <= 5:
                raise RoadmapError("Focused repair must return 3-5 tasks")
            _check_council_message(note, {o["advisor_id"] for o in opinions})
            repaired = {**sprint, "tasks": tasks}
            _check_historical_lookback(repaired, {
                "output": source.get("output"),
                "organization": data["payload"].get("organization"),
                "goal": data["payload"].get("goal"),
            })
            _check_recent_history(repaired, prior_rows)
            print(
                f"[roadmap] focused task repair succeeded for sprint {number} "
                f"(attempt {attempt + 1}/{attempts})", flush=True,
            )
            return repaired, note
        except RoadmapError as exc:
            last_error = str(exc)
            print(
                f"[roadmap] focused task repair sprint={number} "
                f"attempt={attempt + 1}/{attempts} rejected: {last_error}",
                flush=True,
            )
    raise RoadmapError(
        f"Sprint {number} task collision could not be repaired: {last_error}"
    )


def _write_sprint_batch(
    *,
    data: dict,
    council: Any,
    advisors: list[dict],
    system: str,
    opinions: list[dict],
    rows: list[dict],
    completed: list[dict],
    output_by_number: dict[int, Any],
    repair_budget: dict[int, int] | None = None,
) -> tuple[list[dict], list[str]]:
    """Generate a validated batch; isolate failures without returning partials.

    Multi-sprint batches get two attempts. If they still fail, split them and
    re-generate each failed portion independently. One-sprint batches receive
    up to three attempts with precise repair guidance. Earlier accepted
    sprints remain in-memory only until the *entire* request succeeds.
    """
    if repair_budget is None:
        repair_budget = {}
    numbers = [x["number"] for x in rows]
    payload = data["payload"]
    task = {
        "task": (
            "أنشئ سبرينت لكل رقم معتمد أدناه فقط، اعتمادًا على output.text وoutput.results. "
            "لكل سبرينت عنوان ووصف قصير و3-5 مهام متسلسلة ملموسة لفريق الجمعية و3-5 مؤشرات تقيس إنجاز المخرج. "
            "اعرض محتوى الإنتاج المطلوب فعليًا، وليس ما ينبغي على المستشارين فعله أو مجرد تلخيص آرائهم. "
            "لا تنشئ أسماء مستشارين ولا مهام للتواصل معهم. "
            "كل مؤشر لازم يقيس عنصرًا قابلًا للتوثيق خلال خمسة أيام العمل مثل تقرير مُعد أو نتيجة مُوثّقة. "
            "لا تكتب مؤشرات غامضة مثل التوصيات المؤثرة أو التوصيات المؤجلة أو التغطية المؤقتة للتمويل. "
            "المستهدفات أرقام تنفيذية مقترحة لهذا الأسبوع وليست نتائج مسجلة. "
            "حتى عندما تكون مخرجات السبرينتات مختلفة، ممنوع نسخ قائمة المهام نفسها أو تكرار ثلاث مهام منها. "
            "مثال: سبرينت تقييم النتائج يحلل أدلة التنفيذ ويصدر تقرير تقييم؛ "
            "أما سبرينت التقرير الختامي والمتابعة فيحوّل تقييمًا منجزًا إلى تقرير ختامي وخطة إجراءات؛ "
            "لا يبدأ السبرينت الختامي بجمع نفس البيانات ولا تكرر مهام سبرينت التقييم. "
            "إذا تكرر المخرج في أكثر من سبرينت، طوّر العمل تدريجيًا حسب output_occurrence. "
            "اقرأ execution_focus_by_sprint لتوضيح التقدم المميز بكل سبرينت. "
            "لا ترجع حقول output/results/phase ولا تضف أو تحذف أرقام السبرينت."
        ),
        "context": _context(data),
        "advisors": _meta_context(advisors, opinions),
        # Prior tasks are used by _check_recent_history, but should NOT be
        # included verbatim in the prompt: Meta had copied their wording.
        "previous_sprints_summary": [
            {"number": x["number"], "title": x["title"],
             "output_text": _text(
                 (output_by_number.get(x["number"]) or {}).get("text"),
                 limit=260,
             )}
            for x in completed[-4:]
        ],
        "sprints_to_write": rows,
        "execution_focus_by_sprint": {
            str(row["number"]): _execution_focus_for_output(row["output"])
            for row in rows
        },
        "indicator_examples_by_sprint": {
            str(x["number"]): _indicator_examples_for_output(x["output"])
            for x in rows
        },
        "required_output_format": (
            f"BEGIN_SPRINT|{rows[0]['number']}\nTITLE|عنوان عربي\nDESCRIPTION|وصف عربي\n"
            "TASK|مهمة تنفيذية\nTASK|مهمة تنفيذية أخرى\nTASK|مهمة تنفيذية ثالثة\n"
            "INDICATOR|مؤشر خاص بالمخرج الأول|number|1\n"
            "INDICATOR|مؤشر مخصص ثان|percent|100\n"
            "INDICATOR|مؤشر مخصص ثالث|number|3\n"
            "END_SPRINT\nكرر بنفس الترتيب لكل سبرينت مطلوب، ثم META_MESSAGE|خلاصة قرار المجلس بالعربية\n"
            "SYNTHESIS|COMPLEMENTARY أو CONSENSUS أو TRADE_OFF أو CONFLICT أو EVIDENCE_GAP أو SCOPE_CONFLICT؛ "
            "اختر رمزًا واحدًا يعكس حكمك الفعلي على آراء المستشارين، وليس اتفاقًا مفترضًا"
        ),
    }
    attempts = (2 if len(rows) > 1 else max(
        1, min(4, int(os.getenv("ROADMAP_SINGLE_SPRINT_ATTEMPTS", "3")))
    ))
    error = None
    for attempt in range(attempts):
        candidate = dict(task)
        if error:
            candidate["repair_instruction"] = (
                f"المحاولة السابقة رُفضت: {error}. أعد السبرينتات المطلوبة كاملة بالأرقام نفسها، "
                "واستبدل المؤشر الضعيف بمؤشر يقيس مستندًا أو عنصرًا منجزًا من المخرج المخصص للسبرينت. "
                "indicator_examples_by_sprint اقتراحات مستهدفات وليست حقائق عن الجمعية. "
                "إذا سبب الرفض هو تكرار المهام مع سبرينت آخر، فأعد كتابة 3-5 مهام مختلفة مرتبطة "
                "بمخرج كل سبرينت وبخطوة تقدّم جديدة؛ لا تنسخ مهام previous_sprints_summary أو سبرينت آخر في الدفعة. "
                "أعد كتابة 3-5 مؤشرات سليمة لكل سبرينت ولا تكرر الأسماء داخل السبرينت."
            )
        raw = _meta_generate(
            council, system, candidate,
            int(os.getenv("ROADMAP_META_TOKENS", "1900")),
            sample=(attempt > 0),
            block_non_arabic_scripts=True,
        )
        try:
            # Check publishable fields, not the entire raw model response:
            # an admin-only META_MESSAGE must not invalidate valid sprint work.
            generated, note = _parse_batch(raw, numbers)
            metadata = _contract_metadata(raw)
            _check_council_message(note, {a["slug"] for a in advisors})
            prior_rows = [
                {**p, "output_key": _output_key(output_by_number[p["number"]])}
                for p in completed
            ]
            repair_notes = []
            for index, (sprint, source) in enumerate(zip(generated, rows)):
                _check_historical_lookback(sprint, {
                    "output": source["output"],
                    "organization": payload.get("organization"),
                    "goal": payload.get("goal"),
                })
                current = {**sprint, "output_key": _output_key(source["output"])}
                try:
                    _check_recent_history(current, prior_rows)
                except RoadmapError as collision:
                    # Repair only task duplication. Invalid indicators, missing
                    # sprints etc. still follow the existing retry/isolation.
                    if "repeats" not in str(collision) or "sprint tasks" not in str(collision):
                        raise
                    repaired, repair_note = _repair_colliding_tasks(
                        council=council, system=system, data=data,
                        sprint=sprint, source=source, prior_rows=prior_rows,
                        opinions=opinions, collision=str(collision),
                        repair_budget=repair_budget,
                    )
                    _check_council_message(repair_note, {a["slug"] for a in advisors})
                    generated[index] = repaired
                    current = {**repaired, "output_key": _output_key(source["output"])}
                    repair_notes.append(repair_note)
                # V10: the model may satisfy the line protocol but still
                # return unrelated actions or recycled generic indicators.
                # Repair that one card, not the entire 12-week roadmap.
                quality_issues = _content_quality_issues(generated[index], source.get("output"), prior_rows)
                quality_issues.update(_sprint_arabic_issues(generated[index]))
                if quality_issues:
                    repaired, quality_note = _repair_sprint_quality(
                        council=council, system=system, sprint=generated[index],
                        source=source, opinions=opinions, previous=prior_rows,
                        errors=quality_issues,
                    )
                    generated[index] = repaired
                    current = {**repaired, "output_key": _output_key(source["output"])}
                    repair_notes.append(quality_note)
                prior_rows.append(current)
            all_notes = [note] + repair_notes
            all_notes = [_ensure_readable_meta_note(council, system, text,
                         generated, opinions) for text in all_notes]
            if metadata:
                all_notes[0] = {"message": all_notes[0], **metadata}
            return generated, all_notes
        except RoadmapError as exc:
            # The assigned cards already passed validation and the isolated
            # admin-only transcript note exhausted its own correction tries.
            # Regenerating the entire sprint here wastes GPU and can damage
            # otherwise valid work. Fail explicitly, never return partials.
            if str(exc) == "Cannot produce a readable meta transcript message":
                raise
            error = str(exc)
            print(
                f"[roadmap] generate batch={numbers} attempt={attempt + 1}/{attempts} "
                f"rejected: {error}", flush=True,
            )

    if len(rows) > 1:
        print(
            f"[roadmap] batch={numbers} exhausted retries; isolating sprints "
            "without accepting partial results", flush=True,
        )
        mid = len(rows) // 2
        left, left_notes = _write_sprint_batch(
            data=data, council=council, advisors=advisors, system=system,
            opinions=opinions, rows=rows[:mid], completed=completed,
            output_by_number=output_by_number, repair_budget=repair_budget,
        )
        right, right_notes = _write_sprint_batch(
            data=data, council=council, advisors=advisors, system=system,
            opinions=opinions, rows=rows[mid:], completed=completed + left,
            output_by_number=output_by_number, repair_budget=repair_budget,
        )
        return left + right, left_notes + right_notes
    raise RoadmapError(f"Failed sprint batch {numbers} after isolated retries: {error}")


def _complete(data: dict, council: Any, advisors: list[dict], persona: str) -> dict:
    payload = data["payload"]
    requested = _validate_generate_request(payload)
    if all(x["output"] is None for x in requested) and not payload.get("goal") and not payload.get("intervention"):
        raise RoadmapError("No outputs, intervention, or goal to ground roadmap")
    opinions, messages = _specialist_messages(council, data, advisors,
                                               _context(data, requested))
    system = _meta_system(persona)
    batch_size = max(1, min(3, int(os.getenv("ROADMAP_BATCH_SIZE", "2"))))
    completed, meta_messages = [], []
    encountered = Counter()
    output_by_number = {x["number"]: x["output"] for x in requested}
    all_keys = Counter(_output_key(x["output"]) for x in requested)
    repair_budget: dict[int, int] = {}
    for first in range(0, len(requested), batch_size):
        rows = requested[first:first + batch_size]
        payload_rows = []
        for row in rows:
            key = _output_key(row["output"])
            encountered[key] += 1
            payload_rows.append({**row,
                                 "output_occurrence": encountered[key],
                                 "output_total_sprints": all_keys[key]})
        new_sprints, notes = _write_sprint_batch(
            data=data, council=council, advisors=advisors, system=system,
            opinions=opinions, rows=payload_rows, completed=completed,
            output_by_number=output_by_number, repair_budget=repair_budget,
        )
        completed.extend(new_sprints)
        meta_messages.extend(notes)
    if [x["number"] for x in completed] != [x["number"] for x in requested]:
        raise RoadmapError("Incomplete roadmap: no partial successes allowed")
    for model_message in messages:
        _require_generated_arabic_quality(model_message["message"])
        _check_council_message(model_message["message"], {a["slug"] for a in advisors})
    extras = {}
    for note in meta_messages:
        if isinstance(note, dict):
            for key in ("decision_memory", "human_escalation"):
                if key in note:
                    extras[key] = note[key]
    return _envelope(opinions, messages, meta_messages, {"sprints": completed}, extras)


def _regenerate(data: dict, council: Any, advisors: list[dict], persona: str) -> dict:
    payload = data["payload"]
    kind = payload.get("target")
    if kind not in ("sprint", "sprint_indicator"):
        raise RoadmapError("Unknown regeneration target")
    sprint = payload.get("sprint")
    if not isinstance(sprint, dict):
        raise RoadmapError("Regeneration requires sprint context")
    _required_number(sprint.get("number"))
    sprint = {**sprint, "output": _normalize_output(sprint.get("output"))}
    if not _text(sprint.get("title"), limit=130, arabic=True) or not _text(sprint.get("description"), limit=650, arabic=True):
        raise RoadmapError("Regeneration requires sprint title and description")
    reason = _text(data.get("reason"), limit=1500)
    if kind == "sprint":
        old = payload.get("tasks", [])
        if not isinstance(old, list):
            raise RoadmapError("tasks must be a list")
        old_names = old
        output_rule = "TASK|مهمة عربية، سطر لكل مهمة؛ 3 إلى 5 مهام جديدة مختلفة ومحددة بالترتيب."
    else:
        old = payload.get("indicator")
        other = payload.get("other_indicators", [])
        if not isinstance(old, dict) or not isinstance(other, list):
            raise RoadmapError("Regeneration requires current indicator and other_indicators")
        previous_name = _text(old.get("name"))
        if not previous_name:
            raise RoadmapError("Current indicator name is required")
        old_names = [previous_name] + [x for x in other if isinstance(x, str)]
        output_rule = (
            "INDICATOR|اسم مؤشر جديد|unit|target؛ سطر واحد فقط. "
            "مثال صحيح: INDICATOR|عدد التقارير المعتمدة|number|1 أو "
            "INDICATOR|نسبة اكتمال توثيق المصادر|percent|100. "
            "الاسم الذي يبدأ بـ(نسبة) يلزمه percent، والذي يبدأ بـ(عدد) يلزمه number. "
            "إن طلبت الجمعية مؤشر إنجاز خلال السبرينت، اكتب مؤشر اكتمال أو عمل منجز "
            "وليس نسبة تكوين الإيرادات أو الوضع الحالي."
        )
    # Keep the old malformed tasks for semantic comparison, but do not feed
    # them into Specialist/Meta prompts where foreign text can be echoed.
    advisor_context = {"sprint": sprint, "reason": reason,
                       "other_indicators": payload.get("other_indicators")}
    if kind == "sprint_indicator":
        advisor_context["existing"] = old
    opinions, messages = _specialist_messages(council, data, advisors, advisor_context)
    for opinion in opinions:
        _require_generated_arabic_quality(opinion["reasoning"])
        _require_generated_arabic_quality(opinion["recommendation"])
    task = {
        "task": ("أعد توليد " + ("مهام هذا السبرينت فقط" if kind == "sprint" else "مؤشر واحد فقط")
                 + ". التزم بسبب المستخدم، والمخرج والعنوان والوصف. لا تعِد النص السابق ولا تغيّر أي حقل آخر. "
                 "يجب أن يكون البديل عملًا/مقياسًا تنفيذيًا لفريق الجمعية، لا توجيهًا للمستشارين ولا مؤشرًا غامضًا."),
        "reason": reason, "sprint": sprint, "previous": old,
        "other_indicators": payload.get("other_indicators", []),
        "advisors": _meta_context(advisors, opinions),
        "required_output_format": output_rule + "\nMETA_MESSAGE|سبب اختيار البديل بصوت الميتا بالعربية\nSYNTHESIS|رمز الحكم الصحيح على آراء المستشارين",
    }
    if kind == "sprint":
        # Specialist expertise still contributes, but Meta gets only the
        # substantive advice, not identity slugs or titles. This prevents
        # advisor orchestration directives from contaminating the task list.
        task.pop("advisors", None)
        # Do not feed the old bad task text back to Meta. Its prior references
        # to advisor slugs and unsupported lookbacks caused the model to echo
        # the very work items the user is asking to replace. We still compare
        # against the original old list after generation.
        task.pop("previous", None)
        task["previous_tasks_issue"] = (
            "لا تكرر المهام الحالية؛ المطلوب مهام مختلفة ومباشرة لفريق الجمعية."
        )
        task["advisory_insights"] = _task_advisory_insights(opinions)
        task["task"] = (
            "أعد كتابة قائمة مهام تنفيذية لفريق الجمعية فقط، وفق reason ومخرج "
            "السبرينت الحالي. اكتب 3 إلى 5 أفعال واضحة، قابلة للإنجاز خلال "
            "خمسة أيام، مختلفة عن النص القديم. لا تضف أسماء أدوار استشارية أو "
            "دعوات لاجتماعات مع خبراء. استعمل أفعالًا مثل جمع وتصنيف وتحليل "
            "وتوثيق ومراجعة، كلما كانت مناسبة للمخرج. لا تؤلف حقائق."
        )
    attempts = 2 if kind == "sprint_indicator" else max(
        1, min(4, int(os.getenv("ROADMAP_REGEN_TASK_ATTEMPTS", "3")))
    )
    error = None
    for attempt in range(attempts):
        candidate = dict(task)
        if error:
            candidate["repair_instruction"] = (
                "المحاولة السابقة لم تُقبل: " + error + ". "
                "ابدأ من جديد تمامًا؛ اكتب 4 مهام جديدة ملموسة يقوم بها موظفو "
                "الجمعية، كل واحدة فعل يمكن توثيق إنجازه قبل نهاية السبرينت، "
                "بلا توجيهات للخبراء أو رموز أو مسميات استشارية. "
                "اكتب بالعربية الفصحى السليمة فقط، بلا أحرف كورية أو صينية أو ألفاظ مشوهة، "
                "وراجع صياغة رسالة META_MESSAGE أيضًا. "
                "اتبع البروتوكول TASK|المهمة لكل مهمة وMETA_MESSAGE|... في النهاية."
                if kind == "sprint" else
                f"ردك السابق رُفض: {error}. أعِد الرد بالبروتوكول المطلوب."
            )
        raw = _meta_generate(
            council, _meta_system(persona), candidate,
            int(os.getenv("ROADMAP_REGEN_TOKENS", "850")),
            sample=(attempt > 0),
            block_non_arabic_scripts=True,
        )
        try:
            metadata = _contract_metadata(raw)
            try:
                suggestion, note = _parse_regeneration(raw, kind, old_names)
            except RoadmapError as first_error:
                if kind != "sprint":
                    raise
                # Preserve only genuine executable tasks from Meta's response;
                # do not fabricate a replacement for failed model generations.
                try:
                    suggestion, note = _recover_valid_task_subset(raw, old_names)
                except RoadmapError:
                    raise first_error
            if kind == "sprint":
                _check_historical_lookback({"tasks": suggestion["tasks"]}, sprint)
                for row in suggestion["tasks"]:
                    _require_generated_arabic_quality(row["text"])
            else:
                _check_indicator_follows_reason(suggestion, reason)
                _require_generated_arabic_quality(suggestion["name"])
            _check_council_message(note, {a["slug"] for a in advisors})
            if _INTERNAL_LEAK.search(note) or _SUSPECT_META_NOTE.search(note) or _arabic_field_error(note):
                note = _ensure_readable_meta_note(council, _meta_system(persona), note,
                           [{"number": sprint.get("number", 1),
                             "title": sprint.get("title", "العمل المعتمد")}], opinions)
            if kind == "sprint":
                if any(_INTERNAL_LEAK.search(x["text"]) for x in suggestion["tasks"]):
                    raise RoadmapError("Regenerated task contains internal JSON field references")
                # The request supplies only sprint/output/title/description;
                # use that authoritative context and do not invent new output.
                profile = _profile_for_output(sprint.get("output"))
                if profile and not any(_claims_deliverable_action(t["text"], profile["artifact_terms"])
                     for t in suggestion["tasks"]):
                    raise RoadmapError("Regenerated tasks do not produce the assigned sprint output")
            elif _INTERNAL_LEAK.search(suggestion["name"]):
                raise RoadmapError("Regenerated indicator contains internal JSON field references")
            break
        except RoadmapError as exc:
            error = str(exc)
            print(f"[roadmap] regenerate target={kind} attempt={attempt + 1}/{attempts} rejected: {error}", flush=True)
    else:
        raise RoadmapError(f"Failed regeneration after {attempts} attempts: {error}")
    for model_message in messages:
        _require_generated_arabic_quality(model_message["message"])
        _check_council_message(model_message["message"], {a["slug"] for a in advisors})
    return _envelope(opinions, messages,
                     [{"message": note, **metadata}] if metadata else [note],
                     suggestion, metadata)


def _check_council_message(message: str, selected: set[str]) -> None:
    unselected = set(_ANY_ADVISOR_ID.findall(message)) - selected
    if unselected:
        raise RoadmapError("Council message mentions advisor not selected for this request: "
                           + ", ".join(sorted(unselected)))


def _check_historical_lookback(sprint: dict, context: Any) -> None:
    """Avoid inventing years of past financial records absent in backend data."""
    delivered = " ".join([
        sprint.get("title", ""), sprint.get("description", ""),
        *(task.get("text", "") for task in sprint.get("tasks", [])),
        *(indicator.get("name", "") for indicator in sprint.get("indicators", [])),
    ])
    if _HISTORICAL_LOOKBACK.search(delivered) and not _HISTORICAL_LOOKBACK.search(_json(context)):
        raise RoadmapError("Unsupported historical lookback period; use available records")


def _check_recent_history(sprint: dict, previous_sprints: list[dict]) -> None:
    """Prevent copied task lists across ANY sprint, regardless of output.

    Contract: each five-day sprint must advance its assigned output. V7 only
    compared sprints with the same output_key, so two distinct outputs could
    receive the same three tasks and falsely pass as COMPLETED. We reject exact
    copied task sets throughout the job. A high 3+-task overlap with a recent
    sprint is likewise not acceptable. A single shared housekeeping action is
    allowed, because that can legitimately recur across work packages.
    """
    if not previous_sprints:
        return
    current = {_norm(task["text"]) for task in sprint["tasks"]}
    for prior in previous_sprints:
        before = {_norm(task["text"]) for task in prior["tasks"]}
        shared = current & before
        same = current == before
        near_identical = len(shared) >= 3 and (
            len(shared) / max(1, min(len(current), len(before))) >= 0.80
        )
        if same or near_identical:
            sid = sprint.get("number", "?")
            pid = prior.get("number", "?")
            kind = "identical sprint tasks" if same else "near-duplicate sprint tasks"
            raise RoadmapError(
                f"Sprint {sid} repeats {kind} from sprint {pid} "
                f"({len(shared)} common tasks). Write distinct executable tasks "
                "for this sprint's own output rather than recycling another sprint."
            )


_SYNTHESIS_CODES = frozenset((
    "CONSENSUS", "COMPLEMENTARY", "TRADE_OFF", "CONFLICT", "EVIDENCE_GAP", "SCOPE_CONFLICT"
))
_DECISION_STATUSES = frozenset((
    "APPROVED", "APPROVED WITH CONDITIONS", "PILOT FIRST", "NEEDS MORE EVIDENCE",
    "ESCALATE", "DO NOT PROCEED"
))
_ESCALATION_TYPES = frozenset((
    "BINDING_LEGAL_OPINION", "MEDICAL_DECISION", "REGULATED_INVESTMENT",
    "FORMAL_ACCOUNTING_OPINION", "SENSITIVE_HR_DECISION", "HIGH_RISK_CYBERSECURITY",
    "CONTRACTUAL_COMMITMENT", "SENSITIVE_PERSONAL_DATA", "BOARD_OR_ASSEMBLY_EXCLUSIVE"
))


def _contract_metadata(raw: str) -> dict:
    """Read *optional* Meta metadata from a reply, without assuming agreement.

    Does not participate in sprint parsing. If the old V11 model omits these
    lines, the original content and council execution remain unchanged.
    """
    entries: dict[str, str] = {}
    for line in _lines(raw):
        key, sep, value = line.partition("|")
        if sep and key in {"SYNTHESIS", "DECISION_STATUS", "DECISION_ISSUE",
                           "DECISION", "DECISION_OWNER", "DECISION_OUTCOME",
                           "ESCALATION_TYPE", "ESCALATION_STATUS", "ESCALATION_REASON"}:
            entries[key] = value.strip()
    result: dict[str, Any] = {}
    synthesis = entries.get("SYNTHESIS", "")
    if synthesis in _SYNTHESIS_CODES:
        result["synthesis"] = synthesis
    status = entries.get("DECISION_STATUS")
    issue = _text(entries.get("DECISION_ISSUE"), limit=700, arabic=True)
    decision = _text(entries.get("DECISION"), limit=1200, arabic=True)
    if status in _DECISION_STATUSES and issue and decision:
        memory = {"status": status, "issue": issue, "decision": decision}
        owner = _text(entries.get("DECISION_OWNER"), limit=200, arabic=True)
        outcome = _text(entries.get("DECISION_OUTCOME"), limit=600, arabic=True)
        if owner:
            memory["owner"] = owner
        if outcome:
            memory["outcome"] = outcome
        result["decision_memory"] = memory
    esc_type = entries.get("ESCALATION_TYPE")
    esc_status = entries.get("ESCALATION_STATUS")
    esc_reason = _text(entries.get("ESCALATION_REASON"), limit=1200, arabic=True)
    if esc_type in _ESCALATION_TYPES and esc_status in _DECISION_STATUSES and esc_reason:
        result["human_escalation"] = {"type": esc_type, "status": esc_status,
                                      "reason": esc_reason}
    return result


def _envelope(opinions: list[dict], messages: list[dict], meta_messages: list[Any],
              suggestion: dict, contract_extras: dict | None = None) -> dict:
    transcript = []
    # These are actual model-produced independent opinions and meta decisions,
    # represented as a chronological council group chat for the admin UI.
    selected = {x["advisor_id"] for x in opinions}
    for message in messages:
        row = {"sequence": len(transcript) + 1, **message}
        if row.get("from") in selected:
            # Independent opinions are professional interpretation, not proof
            # of verified project outcomes or secondary-source evidence.
            row["evidence"] = {
                "class": "I2",
                "description": "تقدير مهني من معطيات الطلب يحتاج إلى التحقق قبل إثبات النتائج."
            }
        transcript.append(row)
    for note in meta_messages:
        row = {"sequence": len(transcript) + 1, "from": "meta_advisor",
               "message": note["message"] if isinstance(note, dict) else note}
        if isinstance(note, dict) and note.get("synthesis") in _SYNTHESIS_CODES:
            row["synthesis"] = note["synthesis"]
        transcript.append(row)
    for opinion in opinions:
        _require_generated_arabic_quality(opinion["reasoning"])
    envelope = {
        "involved_advisor_ids": [x["advisor_id"] for x in opinions],
        "advisor_reasonings": [{"advisor_id": x["advisor_id"], "reasoning": x["reasoning"]} for x in opinions],
        "transcript": transcript,
        "suggestion": suggestion
    }
    if isinstance(contract_extras, dict):
        for key in ("decision_memory", "human_escalation"):
            if isinstance(contract_extras.get(key), dict):
                envelope[key] = contract_extras[key]
    return envelope


def consult_roadmap(council: Any, request: dict) -> dict:
    """Public API. Always returns full contract JSON or FAILED; never partial data."""
    print(f"[roadmap] code_version={ROADMAP_CODE_VERSION}", flush=True)
    try:
        data = _input(request)
        advisors = _advisor_roster(data["payload"], council)
        persona = _meta_prompt(council, data.get("meta_advisor"))
        if data["kind"] == "generate":
            return _complete(data, council, advisors, persona)
        return _regenerate(data, council, advisors, persona)
    except RoadmapError as exc:
        return {"status": "FAILED", "error": str(exc)}
    except Exception as exc:
        # Do not accidentally tell backend a completed roadmap has been saved.
        print(f"[roadmap] ERROR: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "FAILED", "error": "Roadmap inference failed; inspect RunPod logs."}
