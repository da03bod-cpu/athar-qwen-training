"""Athar OS Screen 5 (خارطة الطريق) — backend-contract router.

Integration: consult_roadmap(existing_council_engine, job_input).
Uses already-loaded Specialist/Meta PEFT adapters via AtharCouncilEngine._generate.
No additional model instance, LoRA, dataset or third-party library required.

Contract: Screen 5 (خارطة الطريق) — AI Contract, 2026-10.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

VALID_UNITS = frozenset(("percent", "number", "multiplier"))
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_INVALID_SCRIPTS = re.compile(r"[\u0400-\u052F\u3040-\u30FF\u4E00-\u9FFF]")
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
        out.append({"text": clean})
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


def _output_key(value: Any) -> str:
    return _json(value) if value is not None else "null"


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
        item_output = row.get("output")
        if item_output is not None and not isinstance(item_output, dict):
            raise RoadmapError("Sprint output must be an object or null")
        output.append({"number": number, "output": item_output})
    return sorted(output, key=lambda x: x["number"])


def _context(data: dict, rows: list[dict] | None = None) -> dict:
    p = data["payload"]
    return {k: p.get(k) for k in (
        "organization", "programs", "track", "goal", "impact_map", "intervention"
    )} | ({"sprints": rows} if rows is not None else {})


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


def _meta_generate(council: Any, system: str, task: dict, tokens: int) -> str:
    return _clean_model(council._generate("meta", system, _json(task), tokens,
                                         deterministic=True, repetition_penalty=1.08))


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
            current["tasks"].append({"text": val})
        elif key == "INDICATOR":
            current["indicators"].append(_parse_indicator_line(line))
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
            tasks.append({"text": line.partition("|")[2]})
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
            task = _validate_tasks([candidate])[0]
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
        "اكتب العربية الفصحى، ولا تُخرج JSON. التزم ببروتوكول السطور حرفيًا."
    )


def _complete(data: dict, council: Any, advisors: list[dict], persona: str) -> dict:
    payload = data["payload"]
    requested = _validate_generate_request(payload)
    # An entirely empty case cannot be completed credibly.
    if all(x["output"] is None for x in requested) and not payload.get("goal") and not payload.get("intervention"):
        raise RoadmapError("No outputs, intervention, or goal to ground roadmap")
    opinions, messages = _specialist_messages(council, data, advisors,
                                               _context(data, requested))
    system = _meta_system(persona)
    batch_size = max(1, min(3, int(os.getenv("ROADMAP_BATCH_SIZE", "2"))))
    completed, meta_messages = [], []
    encountered = Counter()
    all_keys = Counter(_output_key(x["output"]) for x in requested)
    for first in range(0, len(requested), batch_size):
        batch = requested[first:first + batch_size]
        payload_rows = []
        for row in batch:
            key = _output_key(row["output"])
            encountered[key] += 1
            payload_rows.append({**row,
                                 "output_occurrence": encountered[key],
                                 "output_total_sprints": all_keys[key]})
        task = {
            "task": (
                "أنشئ سبرينت لكل رقم معتمد أدناه فقط، اعتمادًا على output.text وoutput.results. "
                "لكل سبرينت عنوان ووصف قصير و3-5 مهام متسلسلة ملموسة لفريق الجمعية و3-5 مؤشرات تقيس إنجاز المخرج. "
                "اعرض محتوى الإنتاج المطلوب فعليًا، وليس ما ينبغي على المستشارين فعله أو مجرد تلخيص آرائهم. "
                "لا تنشئ أسماء مستشارين ولا مهام للتواصل معهم. "
                "إذا كان المخرج تقرير تحليل مصادر التمويل، فتتعلق المهام بجمع الإيرادات وتصنيف مصادرها وحساب نسبها وتوثيق التقرير، "
                "وتقيس المؤشرات البيانات المصنفة والتقرير، لا عدد التوصيات المؤجلة. "
                "المستهدفات أرقام تنفيذية مقترحة لهذا الأسبوع وليست نتائج مسجلة. "
                "إذا تكرر المخرج في أكثر من سبرينت، طوّر العمل تدريجيًا حسب output_occurrence. "
                "لا ترجع حقول output/results/phase ولا تضف أو تحذف أرقام السبرينت."
            ),
            "context": _context(data), "advisors": _meta_context(advisors, opinions),
            "previous_sprints_summary": [{"number": x["number"], "title": x["title"],
                                           "tasks": [y["text"] for y in x["tasks"]]}
                                          for x in completed[-3:]],
            "sprints_to_write": payload_rows,
            "required_output_format": (
                f"BEGIN_SPRINT|{batch[0]['number']}\nTITLE|عنوان عربي\nDESCRIPTION|وصف عربي\nTASK|مهمة 1\nTASK|مهمة 2\nTASK|مهمة 3\n"
                "INDICATOR|اسم مقياس لإنجاز المخرج|number|1\nINDICATOR|اسم مقياس ثان لإنجاز المخرج|percent|100\nINDICATOR|اسم مقياس ثالث لإنجاز المخرج|number|3\n"
                "END_SPRINT\nكرر بنفس الترتيب لكل سبرينت مطلوب، ثم META_MESSAGE|خلاصة قرار المجلس بالعربية"
            ),
        }
        numbers = [x["number"] for x in batch]
        error = None
        for attempt in range(2):
            candidate = dict(task)
            if error:
                candidate["repair_instruction"] = f"الاستجابة السابقة رُفضت: {error}. أعد كل كتل السبرينت لهذه المجموعة كاملة؛ لا تحذف أي سطر."
            raw = _meta_generate(council, system, candidate,
                                 int(os.getenv("ROADMAP_META_TOKENS", "1900")))
            try:
                new_sprints, note = _parse_batch(raw, numbers)
                _check_council_message(note, {a["slug"] for a in advisors})
                previous_rows = [
                    {**p, "output_key": _output_key(next(
                        (r["output"] for r in requested if r["number"] == p["number"]), None
                    ))} for p in completed
                ]
                for sprint, source in zip(new_sprints, batch):
                    _check_historical_lookback(sprint, {
                        "output": source["output"],
                        "organization": payload.get("organization"),
                        "goal": payload.get("goal"),
                    })
                    current = {**sprint, "output_key": _output_key(source["output"])}
                    _check_recent_history(current, previous_rows)
                    previous_rows.append(current)
                break
            except RoadmapError as exc:
                error = str(exc)
        else:
            raise RoadmapError(f"Failed sprint batch {numbers}: {error}")
        completed.extend(new_sprints)
        meta_messages.append(note)
    if [x["number"] for x in completed] != [x["number"] for x in requested]:
        raise RoadmapError("Incomplete roadmap: no partial successes allowed")
    for model_message in messages:
        _check_council_message(model_message["message"], {a["slug"] for a in advisors})
    return _envelope(opinions, messages, meta_messages, {"sprints": completed})


def _regenerate(data: dict, council: Any, advisors: list[dict], persona: str) -> dict:
    payload = data["payload"]
    kind = payload.get("target")
    if kind not in ("sprint", "sprint_indicator"):
        raise RoadmapError("Unknown regeneration target")
    sprint = payload.get("sprint")
    if not isinstance(sprint, dict):
        raise RoadmapError("Regeneration requires sprint context")
    _required_number(sprint.get("number"))
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
    opinions, messages = _specialist_messages(council, data, advisors,
                    {"sprint": sprint, "reason": reason, "existing": old,
                     "other_indicators": payload.get("other_indicators")})
    task = {
        "task": ("أعد توليد " + ("مهام هذا السبرينت فقط" if kind == "sprint" else "مؤشر واحد فقط")
                 + ". التزم بسبب المستخدم، والمخرج والعنوان والوصف. لا تعِد النص السابق ولا تغيّر أي حقل آخر. "
                 "يجب أن يكون البديل عملًا/مقياسًا تنفيذيًا لفريق الجمعية، لا توجيهًا للمستشارين ولا مؤشرًا غامضًا."),
        "reason": reason, "sprint": sprint, "previous": old,
        "other_indicators": payload.get("other_indicators", []),
        "advisors": _meta_context(advisors, opinions),
        "required_output_format": output_rule + "\nMETA_MESSAGE|سبب اختيار البديل بصوت الميتا بالعربية",
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
                "اتبع البروتوكول TASK|... لكل مهمة وMETA_MESSAGE|... في النهاية."
                if kind == "sprint" else
                f"ردك السابق رُفض: {error}. أعِد الرد بالبروتوكول المطلوب."
            )
        raw = _meta_generate(council, _meta_system(persona), candidate,
                             int(os.getenv("ROADMAP_REGEN_TOKENS", "850")))
        try:
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
            else:
                _check_indicator_follows_reason(suggestion, reason)
            _check_council_message(note, {a["slug"] for a in advisors})
            break
        except RoadmapError as exc:
            error = str(exc)
            print(f"[roadmap] regenerate target={kind} attempt={attempt + 1}/{attempts} rejected: {error}", flush=True)
    else:
        raise RoadmapError(f"Failed regeneration after {attempts} attempts: {error}")
    for model_message in messages:
        _check_council_message(model_message["message"], {a["slug"] for a in advisors})
    return _envelope(opinions, messages, [note], suggestion)


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
    """Block obvious duplicate task sets in adjacent sprints on the same output."""
    if not previous_sprints:
        return
    current_set = {_norm(task["text"]) for task in sprint["tasks"]}
    for prior in previous_sprints:
        if prior["output_key"] != sprint["output_key"]:
            continue
        prior_set = {_norm(task["text"]) for task in prior["tasks"]}
        if current_set == prior_set:
            raise RoadmapError("Repeated output has identical sprint tasks instead of progression")


def _envelope(opinions: list[dict], messages: list[dict], meta_messages: list[str], suggestion: dict) -> dict:
    transcript = []
    # These are actual model-produced independent opinions and meta decisions,
    # represented as a chronological council group chat for the admin UI.
    for message in messages:
        transcript.append({"sequence": len(transcript) + 1, **message})
    for note in meta_messages:
        transcript.append({"sequence": len(transcript) + 1,
                           "from": "meta_advisor", "message": note})
    return {"involved_advisor_ids": [x["advisor_id"] for x in opinions],
            "advisor_reasonings": [{"advisor_id": x["advisor_id"], "reasoning": x["reasoning"]} for x in opinions],
            "transcript": transcript,
            "suggestion": suggestion}


def consult_roadmap(council: Any, request: dict) -> dict:
    """Public API. Always returns full contract JSON or FAILED; never partial data."""
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
