import os
import glob
import json
import re
import shutil
import subprocess
import time
import threading
import base64
import gc
import hashlib
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from collections import Counter, deque

import runpod


ROOT = "/workspace/data/athar"

GITHUB_REPO = os.environ.get(
    "GITHUB_REPO",
    ""
).strip()

GITHUB_BRANCH = os.environ.get(
    "GITHUB_BRANCH",
    "main"
)

GIT_USER_NAME = os.environ.get(
    "GIT_USER_NAME",
    "athar-runpod"
)

GIT_USER_EMAIL = os.environ.get(
    "GIT_USER_EMAIL",
    "athar-runpod@users.noreply.github.com"
)

MATCHER_REGISTRY_REL = "advisors/advisors_registry_v2.json"
MATCHER_PROMPT_REL = "prompts/matcher_system_prompt_v2.md"

# This is the exact per-candidate instruction used to build train_matcher_v1.jsonl.
MATCHER_CANDIDATE_SYSTEM_PROMPT = """أنت Athar OS Advisor Candidate Matcher.
مهمتك تقييم مدى ملاءمة مستشار واحد فقط لهذه المنظمة في وضعها الحالي، بالاعتماد على بيانات المنظمة والبرامج وملف المستشار.

لا تشترط أن تكتب المنظمة كلمة "تحتاج" أو تصف مشكلة صراحة. اعتبر ثلاثة أنواع من الأدلة:
1) حاجة أو فجوة أو مخاطرة مذكورة صراحة.
2) حاجة يمكن استنتاجها مباشرة من وقائع المدخل دون اختراع معلومات جديدة.
3) واقع تشغيلي أو برامجي يجعل خبرة المستشار ذات قيمة مادية واضحة الآن.

لا تعتبر مجرد إنجاز سابق سببًا كافيًا وحده لترشيح المستشار إذا لم توجد حاجة حالية أو فرصة أو مخاطرة أو تعقيد ذو صلة.
activation_conditions دليل إيجابي قوي، وscope_boundaries حد ملزم.
not_primary_when لا يعني الاستبعاد التلقائي، لكنه يمنع تضخيم الملاءمة إذا كان الدور الحقيقي يخص مستشارًا آخر.
ميّز بدقة بين المجالات المتجاورة، خصوصًا:
- KPI ولوحات القيادة مقابل MEAL وقياس الأثر.
- تصميم المبادرات مقابل التخطيط التشغيلي مقابل إدارة المحافظ والمشاريع.
- الحوكمة والامتثال مقابل القيادة التنفيذية.
- التشخيص المؤسسي مقابل التخطيط الاستراتيجي.
- التحول والتغيير مقابل الجودة.

أعد JSON صالحًا فقط، واجعل reason جملة واحدة لا تتجاوز 18 كلمة ولا تسرد أسماء المجالات:
{"relevant": true/false, "score": 0.0, "reason": "سبب عربي قصير ومحدد مستند إلى واقعة من المدخل"}

التقدير:
0.85-1.00 ملاءمة مباشرة ومحورية.
0.70-0.84 ملاءمة قوية وواضحة.
0.50-0.69 دور مساند مادي.
0.35-0.49 دور محدود فقط إذا كان له سبب محدد.
أقل من 0.35 يكون relevant=false عادة.
"""

MATCHER_BASE_MODEL = "Qwen/Qwen3-14B"
MATCHER_BATCH_SIZE = int(os.environ.get("MATCHER_BATCH_SIZE", "2"))
MATCHER_MAX_INPUT_TOKENS = int(os.environ.get("MATCHER_MAX_INPUT_TOKENS", "8192"))
MATCHER_MAX_NEW_TOKENS = int(os.environ.get("MATCHER_MAX_NEW_TOKENS", "96"))
MATCHER_MIN_RELEVANT_SCORE = float(os.environ.get("MATCHER_MIN_RELEVANT_SCORE", "0.35"))

_MATCHER_MODEL = None
_MATCHER_TOKENIZER = None
_MATCHER_REGISTRY = None
_MATCHER_DEVICE = None


# ---------------------------------------------------------------------
# Grounded routing engine v5
# Production advisory_match no longer depends on the binary Matcher LoRA.
# It uses the base Qwen3-14B as a two-stage grounded router:
#   1) extract advisory needs/opportunities with evidence IDs
#   2) compare ALL advisors together against those grounded needs
# ---------------------------------------------------------------------

ROUTER_MAX_INPUT_TOKENS = int(
    os.environ.get("ROUTER_MAX_INPUT_TOKENS", "20000")
)
ROUTER_NEEDS_MAX_NEW_TOKENS = int(
    os.environ.get("ROUTER_NEEDS_MAX_NEW_TOKENS", "900")
)
ROUTER_RANK_MAX_NEW_TOKENS = int(
    os.environ.get("ROUTER_RANK_MAX_NEW_TOKENS", "1400")
)
ROUTER_MIN_SCORE = float(
    os.environ.get("ROUTER_MIN_SCORE", "0.35")
)

_ROUTER_MODEL = None
_ROUTER_TOKENIZER = None
_ROUTER_REGISTRY = None
_ROUTER_DEVICE = None

NEEDS_SYSTEM_PROMPT = """أنت محلل احتياجات استشارية لمنظومة Athar OS.

ستستلم قائمة FACTS مرقمة مأخوذة من بيانات منظمة وبرامجها.
استخرج فقط الاحتياجات أو المخاطر أو الفرص الاستشارية التي يمكن دعمها مباشرة بهذه الوقائع.

قواعد ملزمة:
- لا تشترط وجود كلمة "تحتاج" أو "مشكلة".
- لا تخترع فجوة غير مدعومة.
- الإنجاز السابق وحده لا يعني وجود مشكلة حالية.
- يمكن اعتبار تعقيد حقيقي في المحفظة أو البرامج فرصة استشارية إذا كانت له قيمة مادية واضحة.
- كل حاجة يجب أن تشير إلى evidence_ids صحيحة من FACTS.
- اجمع الوقائع المتشابهة في حاجة واحدة بدل التكرار.
- أخرج من 0 إلى 8 احتياجات فقط.
- لا ترشح مستشارين في هذه المرحلة.

kind يجب أن يكون واحدًا من:
explicit_gap
direct_inference
advisory_opportunity

priority يجب أن يكون:
high
medium
low

أعد JSON فقط بهذا الشكل:
{
  "needs": [
    {
      "need_id": "N1",
      "need": "وصف عربي موجز للحاجة",
      "kind": "direct_inference",
      "priority": "high",
      "evidence_ids": ["F2", "P3"]
    }
  ]
}
"""

ROUTING_SYSTEM_PROMPT = """أنت محرك توجيه المستشارين في Athar OS.

ستستلم:
1) GROUNDED_NEEDS: احتياجات أو فرص أو تعقيدات موثقة، وكل واحدة مرتبطة بأدلة.
2) ADVISORS: ملفات التوجيه للمستشارين.

قيّم جميع المستشارين معًا، وليس كل مستشار بمعزل عن الآخرين.

المطلوب:
- رشح كل مستشار له قيمة مادية حقيقية لإحدى الاحتياجات الحالية.
- لا يوجد عدد ثابت للترشيحات.
- لا تضف مستشارًا لمجرد أن مجاله مهم عمومًا.
- لا تخترع احتياجًا جديدًا غير موجود في GROUNDED_NEEDS.
- كل مستشار مرشح يجب أن يحتوي matched_need_ids غير فارغة.
- activation_conditions دليل إيجابي قوي.
- scope_boundaries حد ملزم.
- not_primary_when لا يعني الاستبعاد التلقائي؛ قد يكون الدور supporting إذا أضاف قيمة مادية.
- قارن المجالات المتجاورة حتى لا تكرر نفس الحاجة بلا داعٍ.

تمييزات مهمة:
- Advisor 14: KPI وتعريف المؤشرات والمصادر وخطوط الأساس والمستهدفات ولوحات القيادة.
- Advisor 15: MEAL والنتائج والتقييم والتعلم ونظرية التغيير وقوة دليل الأثر والسببية.
- Advisor 11: تصميم التدخل أو المبادرة قبل اعتمادها واختبار الفرضيات والقيمة.
- Advisor 12: تحويل أعمال معتمدة إلى خطة تشغيلية وملاك وجدول وموارد واعتماديات.
- Advisor 13: المحافظ والبرامج والمشاريع وPMO والبوابات والمنافع وتعارض الموارد.
- Advisor 16: الحوكمة والامتثال والصلاحيات والسياسات والضوابط وأدلة التطبيق.
- Advisor 1: القيادة التنفيذية وحسم القرار وترتيب الأولويات والتنفيذ.
- Advisor 5: التحول والتبني والمقاومة وموجات التغيير؛ مجرد وجود ERP لا يكفي.
- Advisor 6: الجودة وعدم المطابقة والمعايير والإجراءات التصحيحية والتحسين المستمر.

الدرجات:
0.85-1.00 = مباشر ومحوري
0.70-0.84 = قوي وواضح
0.50-0.69 = مساند مادي
0.35-0.49 = محدود لكن مبرر
أقل من 0.35 لا ترشحه عادة.

role يجب أن يكون primary أو supporting.

أعد JSON فقط:
{
  "ranked": [
    {
      "advisor_id": 14,
      "score": 0.91,
      "role": "primary",
      "matched_need_ids": ["N2"],
      "reason": "سبب عربي قصير ومحدد"
    }
  ]
}
"""



# ---------------------------------------------------------------------
# Rich AI Router v8
# The model evaluates ALL 16 advisors together using rich Expert DNA.
# There is NO deterministic need gate and NO hard-coded advisor mapping.
# ---------------------------------------------------------------------

RICH_REGISTRY_PATH = os.environ.get(
    "RICH_REGISTRY_PATH",
    f"{ROOT}/data/advisors_registry_rich_v4_35.json"
)

RICH_MAX_INPUT_TOKENS = int(
    os.environ.get("RICH_MAX_INPUT_TOKENS", "20000")
)

RICH_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_MAX_NEW_TOKENS", "1400")
)

RICH_MIN_RELEVANT_SCORE = float(
    os.environ.get("RICH_MIN_RELEVANT_SCORE", "0.40")
)

_RICH_MODEL = None
_RICH_TOKENIZER = None
_RICH_REGISTRY = None
_RICH_DEVICE = None

# Shared by advisory_match and advisory_consultation so adapter switching can
# never race with a base-model routing generation inside the same worker.
INFERENCE_MODEL_LOCK = threading.RLock()
_COUNCIL_ENGINE = None
_COUNCIL_ADAPTER_PATHS = None

RICH_ROUTER_SYSTEM_PROMPT = """أنت Athar OS Rich Advisor Router.

هذه مرحلة اكتشاف الملاءمة قبل أن تختار الجمعية ستة مستشارين.
مهمتك تقييم الـ16 مستشارًا جميعًا باستخدام ملفات Expert DNA الغنية، ثم إرجاع كل المستشارين المناسبين فعلاً فقط.

قواعد حاسمة:
1) فكّر في الـ35 جميعًا قبل الإخراج.
2) لا يوجد عدد ثابت؛ قد يكون المناسب 2 أو 5 أو 8 أو أكثر.
3) لا تشترط وجود كلمة "تحتاج" أو "مشكلة". الملاءمة قد تأتي من:
   - فجوة أو مخاطرة صريحة.
   - حاجة مستنتجة مباشرة من الوقائع.
   - تعقيد تشغيلي/برامجي حقيقي.
   - فرصة تحسين مادية واضحة من واقع عمل المنظمة.
4) لا تعتبر الإنجاز السابق وحده دليلاً على وجود فجوة حالية.
5) Supporting مقبول فقط إذا كانت له قيمة مادية مستقلة، وليس ارتباطًا هامشيًا.
6) لا تطبق Minimum Expert Principle هنا؛ الاختيار النهائي لستة مستشارين يتم لاحقًا.
7) كل مستشار مختار يجب أن يستند إلى evidence_ids صحيحة من FACTS.
8) activation_when دليل إيجابي، وnot_primary_when وboundaries تمنع تضخيم الدور.
9) فرّق بدقة بين:
   - 14 KPI/Dashboard و15 MEAL/Impact.
   - 11 Initiative Design و12 Operational Planning و13 Portfolio/Program/Project.
   - 1 Executive Leadership و16 Governance/Compliance.
   - 2 Institutional Diagnosis و8 Strategic Planning.
   - 5 Change/Adoption و6 Quality/Continuous Improvement.
10) لا تُخرج المستشارين غير المناسبين.

معايرة score:
0.85-1.00 = ملاءمة محورية وواضحة
0.70-0.84 = ملاءمة قوية
0.50-0.69 = دور مساند مادي
0.40-0.49 = قيمة محدودة لكن حقيقية
أقل من 0.40 = لا تخرجه

role:
core = يعالج بعدًا رئيسيًا ظاهرًا في الحالة
supporting = يضيف بعدًا مساندًا ماديًا

أعد JSON صالحًا فقط، بدون Markdown، وبدون شرح خارج JSON.
استخدم المفتاح advisor_id حرفيًا كما هو.
اجعل reason جملة واحدة قصيرة جدًا، بحد أقصى 14 كلمة.
اجعل evidence_ids من 1 إلى 3 فقط.

الشكل المطلوب:
{
  "matches": [
    {
      "advisor_id": 13,
      "score": 0.91,
      "role": "core",
      "evidence_ids": ["F6", "P3"],
      "reason": "سبب عربي قصير ومحدد"
    }
  ]
}
"""


def build_rich_facts(organization, programs):

    facts = []

    def add_fact(fid, source, value):
        if value is None:
            return

        if isinstance(value, (list, dict)):
            text = json.dumps(
                value,
                ensure_ascii=False
            )
        else:
            text = str(value).strip()

        if text:
            facts.append({
                "fact_id": fid,
                "source": source,
                "text": text,
            })

    add_fact("F1", "organization.name", organization.get("name"))
    add_fact("F2", "organization.type", organization.get("type"))
    add_fact("F3", "organization.sector", organization.get("sector"))
    add_fact("F4", "organization.activity_fields", organization.get("activity_fields"))
    add_fact("F5", "organization.short_description", organization.get("short_description"))
    add_fact("F6", "organization.detailed_description", organization.get("detailed_description"))
    add_fact("F7", "organization.competitive_advantage", organization.get("competitive_advantage"))
    add_fact("F8", "organization.important_notes", organization.get("important_notes"))

    for index, program in enumerate(programs, start=1):
        if not isinstance(program, dict):
            continue

        parts = []

        for key in [
            "name",
            "type",
            "description",
            "target_audience",
            "beneficiary_value",
            "delivery_method",
        ]:
            value = program.get(key)

            if value is not None and str(value).strip():
                parts.append(
                    f"{key}={str(value).strip()}"
                )

        if parts:
            facts.append({
                "fact_id": f"P{index}",
                "source": f"programs[{index - 1}]",
                "text": " | ".join(parts),
            })

    return facts


def load_rich_registry():

    if not os.path.isfile(RICH_REGISTRY_PATH):
        raise RuntimeError(
            f"Rich advisor registry not found: {RICH_REGISTRY_PATH}"
        )

    with open(
        RICH_REGISTRY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    advisors = registry.get("advisors", [])

    if len(advisors) != 35:
        raise RuntimeError(
            f"Rich registry must contain exactly 35 advisors; got {len(advisors)}"
        )

    advisor_ids = [
        int(advisor.get("advisor_id"))
        for advisor in advisors
    ]

    if advisor_ids != list(range(1, 36)):
        raise RuntimeError(
            f"Rich registry advisor IDs must be 1..35; got {advisor_ids}"
        )

    system_codes = [
        str(advisor.get("system_code", "")).strip()
        for advisor in advisors
    ]

    if any(not code for code in system_codes):
        raise RuntimeError(
            "Every rich advisor must have a non-empty system_code."
        )

    if len(system_codes) != len(set(system_codes)):
        raise RuntimeError(
            "Rich advisor system_code values must be unique."
        )

    return registry


def ensure_rich_router_model():

    global _RICH_MODEL
    global _RICH_TOKENIZER
    global _RICH_REGISTRY
    global _RICH_DEVICE

    if (
        _RICH_MODEL is not None
        and _RICH_TOKENIZER is not None
        and _RICH_REGISTRY is not None
    ):
        return

    print(
        "Loading Qwen3-14B for rich AI advisor routing...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    _RICH_REGISTRY = load_rich_registry()

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model.eval()

    _RICH_MODEL = model
    _RICH_TOKENIZER = tokenizer
    _RICH_DEVICE = next(model.parameters()).device

    print(
        f"Rich AI router ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def _extract_first_json_object(text):

    cleaned = (
        str(text)
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(
                cleaned[start:end + 1]
            )
        except Exception:
            pass

    raise ValueError(
        f"Rich router did not return valid JSON. Raw: {cleaned[:1200]}"
    )


def generate_rich_evaluations(
    organization,
    programs,
):

    import torch

    facts = build_rich_facts(
        organization,
        programs,
    )

    payload = {
        "organization_name": organization.get("name"),
        "facts": facts,
        "advisors": _RICH_REGISTRY["advisors"],
    }

    messages = [
        {
            "role": "system",
            "content": RICH_ROUTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich router input too long: "
            f"{input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {
        key: value.to(_RICH_DEVICE)
        for key, value in encoded.items()
    }

    print(
        f"Rich routing input tokens: {input_tokens}",
        flush=True,
    )

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=8,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    raw_text = _RICH_TOKENIZER.decode(
        generated[0],
        skip_special_tokens=True,
    )

    parsed = _extract_first_json_object(
        raw_text
    )

    return {
        "parsed": parsed,
        "facts": facts,
        "input_tokens": input_tokens,
        "raw_text": raw_text,
    }



def normalize_rich_matches(parsed, facts):

    valid_ids = set(range(1, 17))
    fact_ids = {fact["fact_id"] for fact in facts}

    raw_matches = parsed.get("matches", [])

    if not isinstance(raw_matches, list):
        raise ValueError("Rich router output missing matches list.")

    normalized = []
    seen = set()

    for row in raw_matches:

        if not isinstance(row, dict):
            continue

        advisor_value = None
        for key in ("advisor_id", " advisor_id", "adviser_id", " adviser_id"):
            if key in row:
                advisor_value = row.get(key)
                break

        try:
            advisor_id = int(advisor_value)
        except (TypeError, ValueError):
            continue

        if advisor_id not in valid_ids or advisor_id in seen:
            continue

        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(1.0, score))

        if score < RICH_MIN_RELEVANT_SCORE:
            continue

        role = str(row.get("role", "supporting")).strip().lower()

        if role not in {"core", "supporting"}:
            role = "supporting"

        evidence_ids = row.get("evidence_ids", [])

        if not isinstance(evidence_ids, list):
            evidence_ids = []

        evidence_ids = [
            str(fid)
            for fid in evidence_ids
            if str(fid) in fact_ids
        ][:3]

        if not evidence_ids:
            continue

        reason = re.sub(
            r"\s+",
            " ",
            str(row.get("reason", "")).strip(),
        )

        if not reason:
            reason = "ملاءمة مادية مدعومة بوقائع من بيانات المنظمة."

        if len(reason.split()) > 18:
            reason = " ".join(reason.split()[:18]).rstrip("،,.") + "."

        normalized.append({
            "advisor_id": advisor_id,
            "score": round(score, 4),
            "role": role,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })

        seen.add(advisor_id)

    normalized.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return normalized


def advisory_match_rich_v8(job_input):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    print(
        "Evaluating all 16 advisors with rich Expert DNA...",
        flush=True,
    )

    result = generate_rich_evaluations(
        organization,
        programs,
    )

    ranked = normalize_rich_matches(
        result["parsed"],
        result["facts"],
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v8_1",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "matched_advisors": len(ranked),
        "input_tokens": result["input_tokens"],
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["raw_output"] = result["raw_text"]

    return response



RICH_V9_GROUPS = [
    [1, 2, 3, 4],
    [5, 6, 7, 16],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
]

RICH_V9_MAX_NEW_TOKENS = int(os.environ.get("RICH_V9_MAX_NEW_TOKENS", "420"))

RICH_V9_SYSTEM_PROMPT = """أنت Athar OS Rich Advisor Router v9.

ستستلم FACTS موثقة عن منظمة وبرامجها، وأربعة مستشارين بملفات Expert DNA غنية.
قيّم الأربعة جميعًا، ثم أخرج فقط المستشارين الذين لديهم قيمة مادية حقيقية الآن.

قواعد:
- لا يوجد عدد ثابت.
- لا تشترط كلمة "تحتاج" أو "مشكلة".
- يجوز الترشيح بسبب فجوة صريحة، حاجة مستنتجة مباشرة، تعقيد تشغيلي/برامجي حقيقي، أو فرصة تحسين مادية واضحة.
- لا تخترع مشكلة غير موجودة.
- الإنجاز السابق وحده لا يعني وجود حاجة حالية.
- وجود ERP أو إعادة هيكلة أو درجة حوكمة مرتفعة لا يعني تلقائيًا الحاجة لمستشار تغيير أو حوكمة.
- وجود برنامج قائم لا يعني تلقائيًا الحاجة لإعادة تصميمه.
- كثرة البرامج وتنوعها قد تدعم 12 أو 13 إذا كان التعقيد واضحًا.
- 15 يحتاج دليل نتائج/تقييم/أثر/تعلم، وليس مجرد وجود برامج.
- 14 يحتاج KPI/مصادر بيانات/خط أساس/مستهدفات/لوحات.
- 8 يحتاج قرارًا أو مراجعة أو مفاضلة استراتيجية حقيقية.
- 2 يحتاج تشخيص/نضج/جاهزية أو فجوة قدرة فعلية.
- 16 يحتاج فجوة/مخاطرة/قرار حوكمي أو امتثال فعلي.
- 5 يحتاج تحول/تبنٍ/مقاومة/انتقال فعلي.
- Supporting مقبول فقط إذا كانت له قيمة مادية مستقلة.
- كل ترشيح يجب أن يستند إلى 1-3 evidence_ids صحيحة.
- السبب يجب أن يوضح لماذا المستشار مناسب الآن.

الدرجات أعداد صحيحة:
85-100 محوري
70-84 قوي
50-69 مساند مادي
40-49 محدود لكنه حقيقي
أقل من 40 لا تخرجه

أخرج سطرًا واحدًا لكل مستشار مناسب فقط:
ADVISOR_ID|SCORE|ROLE|EVIDENCE_IDS|REASON

ROLE = core أو supporting
EVIDENCE_IDS مثال F6,P3
REASON جملة عربية قصيرة.

مثال:
13|92|core|F6,P3|تعدد البرامج وتنوعها يخلق حاجة فعلية لإدارة المحفظة والأولويات.

إذا لم يكن أي منهم مناسبًا اكتب:
NONE

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _rich_v9_advisor_map():
    return {int(a["advisor_id"]): a for a in _RICH_REGISTRY["advisors"]}


def _parse_rich_v9_lines(text, allowed_advisor_ids, valid_fact_ids):
    cleaned = str(text).replace("```text", "").replace("```", "").strip()

    if not cleaned or cleaned.upper() == "NONE":
        return []

    results = []
    seen = set()

    for raw_line in cleaned.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line.strip())
        if not line or line.upper() == "NONE":
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        a_raw, s_raw, role_raw, ev_raw, reason_raw = [x.strip() for x in parts]
        a_m = re.search(r"\d+", a_raw)
        s_m = re.search(r"\d+(?:\.\d+)?", s_raw)

        if not a_m or not s_m:
            continue

        advisor_id = int(a_m.group())
        if advisor_id not in allowed_advisor_ids or advisor_id in seen:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))
        if score < 40:
            continue

        role = role_raw.lower()
        if role not in {"core", "supporting"}:
            role = "supporting"

        evidence_ids = []
        for token in re.split(r"[,،;\s]+", ev_raw):
            token = token.strip().upper()
            if token in valid_fact_ids and token not in evidence_ids:
                evidence_ids.append(token)
        evidence_ids = evidence_ids[:3]

        if not evidence_ids:
            continue

        reason = re.sub(r"\s+", " ", reason_raw).strip()
        if not reason:
            continue
        if len(reason.split()) > 18:
            reason = " ".join(reason.split()[:18]).rstrip("،,.") + "."

        results.append({
            "advisor_id": advisor_id,
            "score": round(score / 100.0, 4),
            "role": role,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })
        seen.add(advisor_id)

    return results


def generate_rich_v9_group(facts, advisors):
    import torch

    messages = [
        {"role": "system", "content": RICH_V9_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(
            {"facts": facts, "advisors": advisors},
            ensure_ascii=False
        )},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["attention_mask"].sum().item())

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(f"Rich v9 group input too long: {input_tokens}")

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V9_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    text = _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True)
    return text, input_tokens


def advisory_match_rich_v9(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()

    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    advisor_map = _rich_v9_advisor_map()

    all_matches = []
    debug_groups = []
    total_tokens = 0

    print("Rich v9: evaluating all 16 advisors in four groups...", flush=True)

    for idx, group_ids in enumerate(RICH_V9_GROUPS, start=1):
        print(f"Rich v9 group {idx}/4: {group_ids}", flush=True)

        raw_text, input_tokens = generate_rich_v9_group(
            facts,
            [advisor_map[i] for i in group_ids],
        )
        total_tokens += input_tokens

        parsed = _parse_rich_v9_lines(
            raw_text,
            set(group_ids),
            valid_fact_ids,
        )
        all_matches.extend(parsed)

        if job_input.get("debug", False):
            debug_groups.append({
                "group": group_ids,
                "input_tokens": input_tokens,
                "raw_output": raw_text,
                "parsed_matches": parsed,
            })

    all_matches.sort(key=lambda x: x["score"], reverse=True)

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v16_relation_class",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "matched_advisors": len(all_matches),
        "total_input_tokens": total_tokens,
        "ranked": all_matches,
    }

    if job_input.get("debug", False):
        response["group_debug"] = debug_groups

    return response


RICH_V10_MAX_NEW_TOKENS = int(os.environ.get("RICH_V10_MAX_NEW_TOKENS", "1300"))

RICH_V10_SYSTEM_PROMPT = """أنت Athar OS Global Rich Advisor Router v10.
قارن الـ16 مستشارًا جميعًا معًا ثم أخرج فقط من لديهم قيمة استشارية مادية حقيقية الآن.

السؤال الحاكم: هل وجود هذا المستشار الآن سيضيف قيمة مستقلة ومادية تدعمها الوقائع الحالية؟
لا ترشح مستشارًا لأن تخصصه مهم عمومًا.

مصادر الملاءمة المقبولة:
- فجوة/مشكلة/مخاطرة صريحة.
- حاجة مستنتجة مباشرة من الوقائع.
- تعقيد تشغيلي/برامجي قائم يفعّل خبرة المستشار.
- فرصة تحسين مادية واضحة ومسنودة.

قواعد منع التضخيم:
- لا يوجد عدد ثابت ولا تملأ القائمة.
- الإنجاز السابق ليس فجوة حالية.
- 16 لا يُرشح لمجرد درجة حوكمة مرتفعة أو وجود سياسات.
- 5 لا يُرشح لمجرد ERP أو إعادة هيكلة دون تبنٍ/مقاومة/انتقال.
- 11 لا يُرشح لمجرد وجود برامج قائمة؛ يلزم تصميم/إعادة تصميم/Pilot/فرضية تدخل.
- 13 قد يُرشح عند كثرة البرامج وتداخلها وأولوياتها ومواردها.
- 12 قد يُرشح عند وجود تعقيد تشغيلي أو موسمية أو جداول وموارد واعتماديات.
- 15 يحتاج دليل نتائج/تقييم/أثر/تعلم؛ لا يكفي وجود برامج.
- 14 يحتاج KPI/بيانات أداء/خط أساس/مستهدفات/لوحات؛ لا يكفي ERP أو نمو الإيرادات.
- 6 يحتاج قضية جودة/اتساق/معايير/شكاوى/تحسين؛ لا تكفي الخدمات وحدها.
- 7 يحتاج خطر/اعتمادية/استمرارية/تعطل؛ لا تكفي خدمة حرجة وحدها.
- 3 يحتاج تحليل بيئة/اتجاهات/مقارنة/قرار توسع أو عدم يقين.
- 4 يحتاج شركاء/أصحاب مصلحة/مانحين/اعتماد خارجي مدعوم.
- 8 يحتاج قرارًا أو مراجعة أو مفاضلة استراتيجية حقيقية.
- 10 يحتاج قضية أو أهداف استراتيجية تحتاج صياغة/ترابط.
- 2 يحتاج تشخيص نضج/جاهزية/قدرات أو فجوة مؤسسية.
- 1 يحتاج قرارًا تنفيذيًا متعدد الأبعاد أو ترتيب أولويات/موارد/ملكية.
- 9 يحتاج سؤال هوية/غرض/رؤية/رسالة/قيم فعلي.
- إذا كان مستشاران متجاوران يعالجان نفس الواقعة، احتفظ بكليهما فقط إذا كانت القيمة المستقلة واضحة.

فرّق خصوصًا بين 14 و15، وبين 11 و12 و13، وبين 1 و16، وبين 2 و8، وبين 5 و6.

SCORE:
90-100 محوري جدًا
80-89 قوي
70-79 واضح
55-69 مساند مادي
40-54 محدود لكنه حقيقي
أقل من 40 لا تخرجه

ROLE = core أو supporting

أخرج فقط:
ADVISOR_ID|SCORE|ROLE|EVIDENCE_IDS|REASON

EVIDENCE_IDS من 1 إلى 3 ويجب أن تكون موجودة في FACTS.
REASON جملة عربية قصيرة لا تتجاوز 16 كلمة.
إذا لم يوجد أحد اكتب NONE.
ممنوع JSON وممنوع Markdown وأي شرح إضافي.
"""

def generate_rich_v10_global(facts, advisors):
    import torch
    messages = [
        {"role": "system", "content": RICH_V10_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"facts": facts, "advisors": advisors}, ensure_ascii=False)},
    ]
    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = _RICH_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["attention_mask"].sum().item())
    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(f"Rich v10 global input too long: {input_tokens}")
    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}
    print(f"Rich v10 global input tokens: {input_tokens}", flush=True)
    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V10_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )
    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    return _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True), input_tokens

def advisory_match_rich_v10(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()
    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    print("Rich v10: globally comparing all 16 advisors...", flush=True)
    raw_text, input_tokens = generate_rich_v10_global(facts, _RICH_REGISTRY["advisors"])
    ranked_internal = _parse_rich_v9_lines(
        raw_text,
        set(range(1, 17)),
        valid_fact_ids,
    )
    ranked_internal.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    advisor_by_number = {
        int(advisor["advisor_id"]): advisor
        for advisor in _RICH_REGISTRY["advisors"]
    }

    ranked = []

    for item in ranked_internal:
        advisor_number = int(item["advisor_id"])
        advisor = advisor_by_number[advisor_number]

        ranked.append({
            "advisor_id": advisor.get(
                "system_code",
                str(advisor_number),
            ),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "score": item["score"],
            "role": item["role"],
            "evidence_ids": item["evidence_ids"],
            "reason": item["reason"],
        })

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v10_1_global",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "matched_advisors": len(ranked),
        "input_tokens": input_tokens,
        "ranked": ranked,
    }
    if job_input.get("debug", False):
        response["raw_output"] = raw_text
    return response


# ---------------------------------------------------------------------
# Rich AI Router v11
# Pass 1: global proposal across all 16 rich profiles.
# Pass 2: adversarial AI adjudication that removes speculative matches.
# Python only parses, validates IDs/evidence, and formats registered IDs.
# ---------------------------------------------------------------------

RICH_V11_REVIEW_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V11_REVIEW_MAX_NEW_TOKENS", "1200")
)

RICH_V11_REVIEW_PROMPT = """أنت Athar OS Adversarial Routing Adjudicator.

لديك:
1) FACTS موثقة عن المنظمة وبرامجها.
2) ملفات Expert DNA للـ35 مستشارًا.
3) PROPOSED_MATCHES من مرحلة AI أولى.

مهمتك مراجعة كل ترشيح بصرامة ثم الاحتفاظ فقط بالمستشارين الذين توجد لهم حاجة أو فرصة تحسين مادية حقيقية الآن.
أنت مرحلة منع الـOvermatching. الافتراضي هو DROP ما لم تثبت الوقائع Trigger حقيقيًا.

السؤال الحاكم لكل ترشيح:
"لو لم يكن هذا المستشار موجودًا الآن، هل هناك قرار/مشكلة/تعقيد/تحسين مادي ظاهر في الوقائع سيبقى دون مالك مناسب؟"

قواعد إلزامية:
- لا تقبل سببًا من نوع "وجود البرامج يعني الحاجة..." إلا إذا كانت طبيعة البرامج نفسها تخلق تعقيدًا يطابق نطاق المستشار مباشرة.
- لا تحول الإنجاز إلى فجوة.
- لا تحول وجود نظام أو سياسة أو برنامج إلى مشكلة غير مذكورة.
- لا تستخدم استنتاجات افتراضية مثل "قد تحتاج" أو "من الأفضل" أو "يمكن أن يفيد".
- يجب أن يرتبط كل KEEP بـ activation_when حقيقي ومستقل في DNA المستشار.
- إذا كان نفس الاحتياج مملوكًا بشكل أوضح لمستشار آخر، أسقط المستشار الأضعف ما لم يضيف قيمة مستقلة مختلفة.
- لا يوجد عدد ثابت. احتفظ بأي عدد تبرره الأدلة فعلًا.
- لا تستخدم ترتيب أو درجة المرحلة الأولى كدليل؛ راجع من الصفر.

اختبارات منع الاستنتاج الزائد:
- ERP / الأرشفة / إعادة الهيكلة إنجازات؛ لا تثبت تلقائيًا Change Management أو KPI أو Maturity.
- ارتفاع درجة الحوكمة لا يثبت Governance Gap.
- وجود برامج كثيرة لا يثبت الحاجة إلى MEAL أو KPI أو Initiative Redesign.
- وجود خدمات صحية/اجتماعية لا يثبت مشكلة Quality.
- وجود خدمات موسمية لا يثبت Business Continuity إلا مع خطر/تعطل/اعتمادية حرجة.
- زيادة الإيرادات لا تثبت الحاجة إلى Partnerships أو External Analysis أو KPI.
- وجود برامج تدريبية لا يثبت Change Adoption.
- وجود مبادرات قائمة لا يعني أنها تحتاج إعادة تصميم.
- الاستراتيجية لا تُفترض لمجرد كبر المنظمة.
- التشخيص المؤسسي لا يُفترض لمجرد أن المنظمة نفذت تطويرًا سابقًا.
- المستشار التنفيذي لا يُرشح لمجرد وجود برامج كثيرة؛ يلزم قرار تنفيذي متعدد الأبعاد أو مفاضلة/ملكية/موارد واضحة.
- Portfolio/Program/Project Advisor يمكن أن يكون مناسبًا عندما يظهر تعدد وتنوع وتداخل كبير للبرامج والمحافظ والأولويات.
- Operational Planning Advisor يمكن أن يكون مناسبًا عندما تظهر موسمية/جداول/موارد/تنسيق تشغيلي بين برامج متعددة.

لكل مستشار مقترح أخرج سطرًا واحدًا فقط:
ADVISOR_ID|KEEP_OR_DROP|FINAL_SCORE|ROLE|EVIDENCE_IDS|REASON

KEEP_OR_DROP = KEEP أو DROP
FINAL_SCORE عدد صحيح 0-100.
ROLE = core أو supporting أو none.
EVIDENCE_IDS من FACTS فقط، 1-3 أدلة عند KEEP، ويمكن أن تكون - عند DROP.
REASON جملة عربية قصيرة تشرح سبب القرار.

إذا KEEP:
- FINAL_SCORE يجب أن يكون 40 أو أكثر.
إذا DROP:
- FINAL_SCORE أقل من 40 وROLE=none.

أخرج فقط السطور، بدون JSON وبدون Markdown وبدون شرح إضافي.
"""


def _parse_v11_review_lines(text, proposed_ids, valid_fact_ids):
    cleaned = str(text).replace("```text", "").replace("```", "").strip()
    kept = []
    decisions = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 5)
        if len(parts) != 6:
            continue

        a_raw, decision_raw, score_raw, role_raw, ev_raw, reason_raw = [p.strip() for p in parts]
        a_match = re.search(r"\d+", a_raw)
        s_match = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not a_match or not s_match:
            continue

        advisor_id = int(a_match.group())
        if advisor_id not in proposed_ids:
            continue

        decision = decision_raw.upper()
        if decision not in {"KEEP", "DROP"}:
            continue

        score = float(s_match.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        role = role_raw.lower()
        if role not in {"core", "supporting", "none"}:
            role = "none" if decision == "DROP" else "supporting"

        evidence_ids = []
        if ev_raw != "-":
            for token in re.split(r"[,،;\s]+", ev_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)
        evidence_ids = evidence_ids[:3]

        reason = re.sub(r"\s+", " ", reason_raw).strip()
        if len(reason.split()) > 20:
            reason = " ".join(reason.split()[:20]).rstrip("،,.") + "."

        final_decision = decision
        if decision == "KEEP" and (score < 40 or not evidence_ids):
            final_decision = "DROP"
            role = "none"

        decisions[advisor_id] = {
            "decision": final_decision,
            "score": round(score / 100.0, 4),
            "role": role,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

        if final_decision == "KEEP":
            if role == "none":
                role = "supporting"
            kept.append({
                "advisor_id": advisor_id,
                "score": round(score / 100.0, 4),
                "role": role,
                "evidence_ids": evidence_ids,
                "reason": reason,
            })

    # Any proposed advisor not explicitly reviewed is not silently kept.
    return kept, decisions


def generate_rich_v11_review(facts, advisors, proposed_matches):
    import torch

    messages = [
        {"role": "system", "content": RICH_V11_REVIEW_PROMPT},
        {"role": "user", "content": json.dumps({
            "facts": facts,
            "advisors": advisors,
            "proposed_matches": proposed_matches,
        }, ensure_ascii=False)},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(encoded["attention_mask"].sum().item())
    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v11 review input too long: {input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}
    print(f"Rich v11 review input tokens: {input_tokens}", flush=True)

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V11_REVIEW_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    raw_text = _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True)
    return raw_text, input_tokens


def advisory_match_rich_v11(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()

    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    advisors = _RICH_REGISTRY["advisors"]

    print("Rich v11 pass 1/2: global discovery across all 16 advisors...", flush=True)
    proposal_raw, proposal_tokens = generate_rich_v10_global(facts, advisors)

    proposed = _parse_rich_v9_lines(
        proposal_raw,
        set(range(1, 17)),
        valid_fact_ids,
    )
    proposed.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"Rich v11 pass 2/2: adversarial review of {len(proposed)} proposed advisors...",
        flush=True,
    )

    if proposed:
        review_raw, review_tokens = generate_rich_v11_review(facts, advisors, proposed)
        kept_internal, review_decisions = _parse_v11_review_lines(
            review_raw,
            {int(item["advisor_id"]) for item in proposed},
            valid_fact_ids,
        )
    else:
        review_raw = "NONE"
        review_tokens = 0
        kept_internal = []
        review_decisions = {}

    kept_internal.sort(key=lambda x: x["score"], reverse=True)

    advisor_by_number = {
        int(advisor["advisor_id"]): advisor
        for advisor in advisors
    }

    ranked = []
    for item in kept_internal:
        number = int(item["advisor_id"])
        advisor = advisor_by_number[number]
        ranked.append({
            "advisor_id": advisor.get("system_code", str(number)),
            "advisor_name": advisor.get("name_ar", advisor.get("name_en")),
            "score": item["score"],
            "role": item["role"],
            "evidence_ids": item["evidence_ids"],
            "reason": item["reason"],
        })

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v11_adjudicated",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "proposed_advisors": len(proposed),
        "matched_advisors": len(ranked),
        "proposal_input_tokens": proposal_tokens,
        "review_input_tokens": review_tokens,
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["proposal_raw_output"] = proposal_raw
        response["review_raw_output"] = review_raw
        response["review_decisions"] = review_decisions

    return response


# ---------------------------------------------------------------------
# Rich AI Router v12
# Pass 1: AI discovers material needs/opportunities WITHOUT seeing advisors.
# Pass 2: AI globally maps all 16 rich advisor profiles ONLY to those needs.
# This prevents reverse-rationalization and still lets AI select all matches.
# ---------------------------------------------------------------------

RICH_V12_NEEDS_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V12_NEEDS_MAX_NEW_TOKENS", "700")
)

RICH_V12_MATCH_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V12_MATCH_MAX_NEW_TOKENS", "1100")
)

RICH_V12_NEEDS_PROMPT = """أنت Athar OS Advisory Need Discovery Engine.

ستستلم FACTS فقط عن المنظمة وبرامجها. لا يوجد أمامك أي مستشارين في هذه المرحلة.
استخرج كل الاحتياجات أو فرص التحسين الاستشارية الحالية والمادية التي تدعمها الوقائع فعلًا.

المقصود بالاحتياج المادي:
- مشكلة أو فجوة أو مخاطرة صريحة.
- قرار أو مفاضلة مهمة تحتاج معالجة.
- تعقيد حقيقي ناتج عن حجم/تنوع/تداخل البرامج أو المواسم أو الموارد.
- فرصة تحسين واضحة ومباشرة يمكن استنتاجها من الوقائع دون اختراع مشكلة.

قواعد صارمة:
- لا تستخرج احتياجًا لأن مجالًا ما مهم عمومًا.
- الإنجاز السابق ليس فجوة حالية.
- تطبيق ERP، إعادة الهيكلة، وجود سياسات، أو ارتفاع الحوكمة تُعامل كإنجازات ما لم يظهر تحدٍ حالي مرتبط بها.
- لا تفترض KPI أو Dashboard أو Baseline أو Targets إن لم توجد إشارة فعلية للقياس والأداء.
- لا تفترض MEAL أو Impact Measurement لمجرد وجود برامج.
- لا تفترض Change Management لمجرد وجود نظام جديد أو برامج تدريبية.
- لا تفترض Governance Gap لمجرد أن الجهة جمعية أهلية أو لديها درجة حوكمة.
- لا تفترض Strategy Review لمجرد تنوع البرامج أو كبر المنظمة.
- لا تفترض Partnerships لمجرد وجود إيرادات أو برامج.
- لا تفترض Quality Problem لمجرد تقديم خدمات.
- لا تفترض Business Continuity Risk لمجرد وجود خدمات موسمية.
- تعدد وتنوع البرامج يمكن أن يولد احتياجًا ماديًا لإدارة المحفظة والأولويات إذا كان واضحًا.
- اختلاف المواسم والجداول وطرق التقديم يمكن أن يولد احتياجًا ماديًا للتنسيق والتخطيط التشغيلي.
- يمكن وجود أكثر من احتياج، لكن لا تكرر نفس الفكرة بصيغ مختلفة.
- لا يوجد عدد ثابت. استخرج كل ما تدعمه الوقائع، ولا تملأ القائمة.

PRIORITY:
high = يؤثر مباشرة في القرار/التنفيذ/النتائج
medium = مهم لكنه مساند
low = قيمة محدودة؛ استخدمه فقط إذا كان ماديًا فعلًا

أخرج سطرًا واحدًا لكل احتياج:
NEED_ID|PRIORITY|EVIDENCE_IDS|NEED

مثال:
N1|high|F6,P1,P10|تعدد البرامج وتنوعها يخلق حاجة لإدارة المحفظة وترتيب الأولويات والاعتماديات.

EVIDENCE_IDS من 1 إلى 4 فقط ويجب أن تكون موجودة في FACTS.
NEED جملة عربية محددة تصف الاحتياج الحالي لا اسم تخصص.
إذا لم توجد احتياجات مادية اكتب:
NONE

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""

RICH_V12_MATCH_PROMPT = """أنت Athar OS Global Advisor Matching Engine.

ستستلم:
1) FACTS موثقة.
2) NEEDS تم استخراجها مستقلًا قبل رؤية المستشارين.
3) ملفات Expert DNA الغنية للـ35 مستشارًا.

مهمتك مقارنة الـ35 جميعًا معًا وإرجاع كل مستشار مناسب ماديًا لاحتياج واحد أو أكثر من NEEDS.

قواعد حاسمة:
- ممنوع اختراع احتياج جديد في هذه المرحلة.
- لا يمكن اختيار مستشار إلا إذا كان مرتبطًا مباشرة بـ matched_need_ids موجودة في NEEDS.
- activation_when وowned_outcome وscope تحدد الملاءمة.
- not_primary_when وboundaries تمنع تضخيم الدور.
- لا تختَر مستشارًا بسبب علاقة عامة أو لأن تخصصه "مفيد عادة".
- إذا عالج مستشاران نفس الاحتياج، احتفظ بكليهما فقط إذا كان لكل منهما دور مستقل ومادي مختلف.
- لا يوجد عدد ثابت. أخرج كل المناسبين فقط.
- المستشار الأساسي core يملك نتيجة رئيسية للاحتياج.
- supporting يضيف قيمة مستقلة مادية للاحتياج لكنه لا يملكه أساسًا.
- لا تستخدم FACTS لتكوين احتياج جديد؛ استخدمها فقط للتحقق من NEEDS وأسباب المطابقة.

فرّق بدقة بين:
14 KPI/Dashboard و15 MEAL/Impact
11 Initiative Design و12 Operational Planning و13 Portfolio/Program/Project
1 Executive Leadership و16 Governance/Compliance
2 Institutional Diagnosis و8 Strategic Planning
5 Change/Adoption و6 Quality/Continuous Improvement

SCORE كعدد صحيح:
90-100 = تطابق مباشر ومحوري
80-89 = قوي
70-79 = واضح
55-69 = مساند مادي
40-54 = محدود لكنه حقيقي
أقل من 40 = لا تخرجه

أخرج سطرًا واحدًا لكل مستشار مناسب:
ADVISOR_ID|SCORE|ROLE|MATCHED_NEED_IDS|EVIDENCE_IDS|REASON

مثال:
13|94|core|N1|F6,P1,P10|يمتلك تنظيم المحفظة والأولويات والاعتماديات التي يطلبها N1.

MATCHED_NEED_IDS يجب أن تكون من NEEDS فقط.
EVIDENCE_IDS من 1 إلى 3 فقط من FACTS.
REASON جملة عربية قصيرة ومحددة.
إذا لم يوجد مستشار مناسب اكتب:
NONE

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _parse_v12_needs(text, valid_fact_ids):
    cleaned = str(text).replace("```text", "").replace("```", "").strip()

    if not cleaned or cleaned.upper() == "NONE":
        return []

    needs = []
    seen = set()

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.upper() == "NONE":
            continue

        parts = line.split("|", 3)
        if len(parts) != 4:
            continue

        need_id_raw, priority_raw, evidence_raw, need_text_raw = [p.strip() for p in parts]

        m = re.search(r"N\s*(\d+)", need_id_raw, flags=re.IGNORECASE)
        if not m:
            continue

        need_id = f"N{int(m.group(1))}"
        if need_id in seen:
            continue

        priority = priority_raw.lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"

        evidence_ids = []
        for token in re.split(r"[,،;\s]+", evidence_raw):
            token = token.strip().upper()
            if token in valid_fact_ids and token not in evidence_ids:
                evidence_ids.append(token)
        evidence_ids = evidence_ids[:4]

        need_text = re.sub(r"\s+", " ", need_text_raw).strip()

        if not evidence_ids or not need_text:
            continue

        if len(need_text.split()) > 30:
            need_text = " ".join(need_text.split()[:30]).rstrip("،,.") + "."

        needs.append({
            "need_id": need_id,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "need": need_text,
        })
        seen.add(need_id)

    return needs


def generate_rich_v12_needs(facts):
    import torch

    messages = [
        {"role": "system", "content": RICH_V12_NEEDS_PROMPT},
        {"role": "user", "content": json.dumps({"facts": facts}, ensure_ascii=False)},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(encoded["attention_mask"].sum().item())

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v12 needs input too long: {input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    print(f"Rich v12 needs input tokens: {input_tokens}", flush=True)

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V12_NEEDS_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    raw_text = _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True)

    return raw_text, input_tokens


def _parse_v12_matches(text, valid_advisor_ids, valid_need_ids, valid_fact_ids):
    cleaned = str(text).replace("```text", "").replace("```", "").strip()

    if not cleaned or cleaned.upper() == "NONE":
        return []

    matches = []
    seen = set()

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.upper() == "NONE":
            continue

        parts = line.split("|", 5)
        if len(parts) != 6:
            continue

        advisor_raw, score_raw, role_raw, needs_raw, evidence_raw, reason_raw = [p.strip() for p in parts]

        a_m = re.search(r"\d+", advisor_raw)
        s_m = re.search(r"\d+(?:\.\d+)?", score_raw)

        if not a_m or not s_m:
            continue

        advisor_id = int(a_m.group())
        if advisor_id not in valid_advisor_ids or advisor_id in seen:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        if score < 40:
            continue

        role = role_raw.lower()
        if role not in {"core", "supporting"}:
            role = "supporting"

        matched_need_ids = []
        for token in re.split(r"[,،;\s]+", needs_raw):
            token = token.strip().upper()
            if token in valid_need_ids and token not in matched_need_ids:
                matched_need_ids.append(token)

        if not matched_need_ids:
            continue

        evidence_ids = []
        for token in re.split(r"[,،;\s]+", evidence_raw):
            token = token.strip().upper()
            if token in valid_fact_ids and token not in evidence_ids:
                evidence_ids.append(token)
        evidence_ids = evidence_ids[:3]

        if not evidence_ids:
            continue

        reason = re.sub(r"\s+", " ", reason_raw).strip()
        if not reason:
            continue

        if len(reason.split()) > 20:
            reason = " ".join(reason.split()[:20]).rstrip("،,.") + "."

        matches.append({
            "advisor_id": advisor_id,
            "score": round(score / 100.0, 4),
            "role": role,
            "matched_need_ids": matched_need_ids,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })
        seen.add(advisor_id)

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


def generate_rich_v12_matches(facts, needs, advisors):
    import torch

    messages = [
        {"role": "system", "content": RICH_V12_MATCH_PROMPT},
        {"role": "user", "content": json.dumps({
            "facts": facts,
            "needs": needs,
            "advisors": advisors,
        }, ensure_ascii=False)},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(encoded["attention_mask"].sum().item())

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v12 match input too long: {input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    print(f"Rich v12 match input tokens: {input_tokens}", flush=True)

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V12_MATCH_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    raw_text = _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True)

    return raw_text, input_tokens


def advisory_match_rich_v12(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()

    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v12 pass 1/2: discovering advisory needs independently from advisor profiles...",
        flush=True,
    )

    needs_raw, needs_tokens = generate_rich_v12_needs(facts)
    needs = _parse_v12_needs(needs_raw, valid_fact_ids)

    print(
        f"Rich v12 discovered {len(needs)} grounded needs/opportunities.",
        flush=True,
    )

    if needs:
        print(
            "Rich v12 pass 2/2: globally matching all 16 rich advisor profiles to discovered needs...",
            flush=True,
        )

        match_raw, match_tokens = generate_rich_v12_matches(
            facts,
            needs,
            advisors,
        )

        internal_matches = _parse_v12_matches(
            match_raw,
            set(range(1, 17)),
            {n["need_id"] for n in needs},
            valid_fact_ids,
        )
    else:
        match_raw = "NONE"
        match_tokens = 0
        internal_matches = []

    advisor_by_number = {
        int(advisor["advisor_id"]): advisor
        for advisor in advisors
    }

    ranked = []
    for item in internal_matches:
        number = int(item["advisor_id"])
        advisor = advisor_by_number[number]

        ranked.append({
            "advisor_id": advisor.get("system_code", str(number)),
            "advisor_name": advisor.get("name_ar", advisor.get("name_en")),
            "score": item["score"],
            "role": item["role"],
            "matched_need_ids": item["matched_need_ids"],
            "evidence_ids": item["evidence_ids"],
            "reason": item["reason"],
        })

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v12_need_first",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "needs_count": len(needs),
        "needs": needs,
        "matched_advisors": len(ranked),
        "needs_input_tokens": needs_tokens,
        "matching_input_tokens": match_tokens,
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["needs_raw_output"] = needs_raw
        response["matching_raw_output"] = match_raw

    return response


# ---------------------------------------------------------------------
# Rich AI Router v13
# Pass 1: discover needs without advisors.
# Pass 2: validate needs without advisors.
# Pass 3: globally match all 16 advisors only to validated needs.
# ---------------------------------------------------------------------

RICH_V13_NEED_REVIEW_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V13_NEED_REVIEW_MAX_NEW_TOKENS", "650")
)

RICH_V13_NEED_REVIEW_PROMPT = """أنت Athar OS Need Validation Engine.

ستستلم FACTS موثقة وCANDIDATE_NEEDS تم استخراجها دون رؤية المستشارين.
راجع كل احتياج بصرامة، ومهمتك منع اختراع احتياجات غير موجودة.

KEEP فقط إذا كان الاحتياج:
1) مذكورًا صراحة كمشكلة/فجوة/مخاطرة/قرار/هدف تحسين.
2) أو نتيجة مباشرة وواضحة لتعقيد ظاهر، مثل:
   - كثرة وتنوع وتداخل البرامج => إدارة محفظة/أولويات/اعتماديات.
   - اختلاف المواسم والجداول وطرق التنفيذ => تخطيط وتشغيل وتنسيق موارد.
3) أو فرصة تحسين مادية واضحة جدًا لا تحتاج افتراض مشكلة جديدة.

DROP إذا احتاج افتراضًا إضافيًا غير موجود في FACTS، أو استُخدمت فيه صياغات مثل:
"قد يحتاج"، "ربما"، "يمكن أن يحتاج"، "يفضل"، "من المحتمل".

قواعد خاصة:
- ERP أو الأرشفة أو إعادة الهيكلة لا تثبت مشكلة تكامل/تبني/أداء.
- تنوع البرامج لا يثبت تلقائيًا الحاجة إلى قياس أثر أو KPI أو إعادة تصميم مبادرات.
- برامج ضيوف الرحمن لا تثبت الحاجة إلى منصة موحدة أو حوكمة جديدة.
- ارتفاع الحوكمة أو وجود سياسات لا يثبت فجوة حوكمة.
- نمو الإيرادات لا يثبت فجوة شراكات أو تحليل خارجي.
- لا تنشئ احتياجًا جديدًا في المراجعة؛ فقط KEEP أو DROP.

أخرج:
NEED_ID|KEEP_OR_DROP|PRIORITY|EVIDENCE_IDS|REASON

KEEP_OR_DROP = KEEP أو DROP
PRIORITY = high أو medium أو low أو none
EVIDENCE_IDS من FACTS فقط، 1-4 عند KEEP، ويمكن - عند DROP.
REASON سبب عربي مختصر.

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""

RICH_V13_MATCH_PROMPT = """أنت Athar OS Global Advisor Matching Engine v13.

ستستلم:
1) FACTS موثقة.
2) VALIDATED_NEEDS تم اكتشافها ومراجعتها قبل رؤية المستشارين.
3) ملفات Expert DNA الغنية للـ35 مستشارًا.

قارن الـ35 جميعًا معًا وأخرج كل مستشار يملك قيمة مستقلة ومادية مرتبطة مباشرة بـ VALIDATED_NEEDS.

قاعدة الملكية:
لا يكفي أن "يساعد" المستشار. يجب أن يكون الاحتياج داخل owned_outcome أو core scope أو activation_when له بوضوح.
إذا كان الاحتياج مملوكًا بوضوح لمستشار متخصص، لا تُضف مستشارًا أعم أو مجاورًا إلا إذا كان له مخرج مستقل مطلوب صراحة.

أمثلة منع التوسّع:
- "إدارة المحفظة/الأولويات/الاعتماديات بين البرامج" يطابق 13 مباشرة.
  لا تضف 1 أو 8 أو 10 إلا إذا كان الاحتياج نفسه يتضمن قرارًا تنفيذيًا أو مفاضلة استراتيجية أو معمار أهداف.
- "التخطيط التشغيلي/المواسم/الجداول/الموارد" يطابق 12 مباشرة.
  لا تضف 5 إلا إذا كان هناك تغيير/مقاومة/انتقال فعلي.
- 14 يحتاج احتياجًا صريحًا للـKPI/القياس/الخط الأساس/المستهدفات/اللوحات.
- 15 يحتاج احتياجًا صريحًا للتقييم/الأثر/النتائج/التعلم.
- 11 يحتاج احتياجًا صريحًا لتصميم/إعادة تصميم مبادرة أو Pilot.
- 16 يحتاج احتياجًا صريحًا للحوكمة/الامتثال/الصلاحيات/السياسات أو فجوة تطبيق.
- 7 يحتاج احتياجًا صريحًا للمخاطر/الاستمرارية/التعطل.
- 6 يحتاج احتياجًا صريحًا للجودة/المعايير/الشكاوى/التحسين.
- 3 يحتاج احتياجًا صريحًا للتحليل البيئي/الاتجاهات/المقارنة/عدم اليقين.
- 4 يحتاج احتياجًا صريحًا لأصحاب المصلحة/الشراكات.
- 8 يحتاج احتياجًا صريحًا لاستراتيجية/مراجعة/خيارات استراتيجية.
- 2 يحتاج احتياجًا صريحًا للتشخيص/النضج/الجاهزية.
- 1 يحتاج احتياجًا صريحًا لقرار تنفيذي متعدد الأبعاد أو نموذج تشغيل/ملكية/مفاضلة تنفيذية.
- 9 يحتاج احتياجًا صريحًا للهوية/الرؤية/الرسالة/القيم.

لا يوجد عدد ثابت.

SCORE:
90-100 = مالك مباشر ومحوري
80-89 = قوي جدًا
70-79 = واضح
55-69 = supporting مستقل ومادي
40-54 = محدود لكنه حقيقي
أقل من 40 = لا تخرجه

أخرج:
ADVISOR_ID|SCORE|ROLE|MATCHED_NEED_IDS|EVIDENCE_IDS|REASON

ROLE = core أو supporting
MATCHED_NEED_IDS من VALIDATED_NEEDS فقط
EVIDENCE_IDS من FACTS فقط، 1-3
REASON جملة عربية قصيرة ومحددة

إذا لم يوجد أحد اكتب NONE.
ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def generate_rich_v13_need_review(facts, candidate_needs):
    import torch

    messages = [
        {"role": "system", "content": RICH_V13_NEED_REVIEW_PROMPT},
        {"role": "user", "content": json.dumps(
            {"facts": facts, "candidate_needs": candidate_needs},
            ensure_ascii=False
        )},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(encoded["attention_mask"].sum().item())
    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(f"Rich v13 need review input too long: {input_tokens}")

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V13_NEED_REVIEW_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    return _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True), input_tokens


def _parse_v13_need_review(text, candidate_needs, valid_fact_ids):
    candidate_by_id = {n["need_id"]: n for n in candidate_needs}
    cleaned = str(text).replace("```text", "").replace("```", "").strip()

    validated = []
    decisions = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        need_raw, decision_raw, priority_raw, evidence_raw, reason_raw = [p.strip() for p in parts]
        m = re.search(r"N\s*(\d+)", need_raw, flags=re.IGNORECASE)
        if not m:
            continue

        need_id = f"N{int(m.group(1))}"
        if need_id not in candidate_by_id:
            continue

        decision = decision_raw.upper()
        if decision not in {"KEEP", "DROP"}:
            continue

        priority = priority_raw.lower()
        if priority not in {"high", "medium", "low", "none"}:
            priority = "none" if decision == "DROP" else candidate_by_id[need_id]["priority"]

        evidence_ids = []
        if evidence_raw != "-":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)
        evidence_ids = evidence_ids[:4]

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        decisions[need_id] = {
            "decision": decision,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

        if decision == "KEEP" and evidence_ids:
            original = candidate_by_id[need_id]
            validated.append({
                "need_id": need_id,
                "priority": priority if priority != "none" else original["priority"],
                "evidence_ids": evidence_ids,
                "need": original["need"],
            })

    return validated, decisions


def generate_rich_v13_matches(facts, needs, advisors):
    import torch

    messages = [
        {"role": "system", "content": RICH_V13_MATCH_PROMPT},
        {"role": "user", "content": json.dumps(
            {"facts": facts, "validated_needs": needs, "advisors": advisors},
            ensure_ascii=False
        )},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(encoded["attention_mask"].sum().item())
    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(f"Rich v13 matching input too long: {input_tokens}")

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V12_MATCH_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    return _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True), input_tokens


def advisory_match_rich_v13(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()

    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    advisors = _RICH_REGISTRY["advisors"]

    print("Rich v13 pass 1/3: discovering needs without advisors...", flush=True)
    needs_raw, needs_tokens = generate_rich_v12_needs(facts)
    candidate_needs = _parse_v12_needs(needs_raw, valid_fact_ids)

    print(
        f"Rich v13 pass 2/3: validating {len(candidate_needs)} candidate needs...",
        flush=True,
    )

    if candidate_needs:
        review_raw, review_tokens = generate_rich_v13_need_review(
            facts, candidate_needs
        )
        validated_needs, need_decisions = _parse_v13_need_review(
            review_raw, candidate_needs, valid_fact_ids
        )
    else:
        review_raw = "NONE"
        review_tokens = 0
        validated_needs = []
        need_decisions = {}

    print(f"Rich v13 validated {len(validated_needs)} needs.", flush=True)

    if validated_needs:
        print("Rich v13 pass 3/3: globally matching all 16 advisors...", flush=True)
        match_raw, match_tokens = generate_rich_v13_matches(
            facts, validated_needs, advisors
        )
        internal_matches = _parse_v12_matches(
            match_raw,
            set(range(1, 17)),
            {n["need_id"] for n in validated_needs},
            valid_fact_ids,
        )
    else:
        match_raw = "NONE"
        match_tokens = 0
        internal_matches = []

    advisor_by_number = {
        int(advisor["advisor_id"]): advisor
        for advisor in advisors
    }

    ranked = []
    for item in internal_matches:
        number = int(item["advisor_id"])
        advisor = advisor_by_number[number]
        ranked.append({
            "advisor_id": advisor.get("system_code", str(number)),
            "advisor_name": advisor.get("name_ar", advisor.get("name_en")),
            "score": item["score"],
            "role": item["role"],
            "matched_need_ids": item["matched_need_ids"],
            "evidence_ids": item["evidence_ids"],
            "reason": item["reason"],
        })

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v13_validated_needs",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "candidate_needs_count": len(candidate_needs),
        "candidate_needs": candidate_needs,
        "validated_needs_count": len(validated_needs),
        "validated_needs": validated_needs,
        "matched_advisors": len(ranked),
        "needs_input_tokens": needs_tokens,
        "need_review_input_tokens": review_tokens,
        "matching_input_tokens": match_tokens,
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["needs_raw_output"] = needs_raw
        response["need_review_raw_output"] = review_raw
        response["need_review_decisions"] = need_decisions
        response["matching_raw_output"] = match_raw

    return response


# ---------------------------------------------------------------------
# Rich AI Router v15
# Pass 1: discover needs without advisors.
# Pass 2: validate needs without advisors.
# Pass 3: globally propose advisors for validated needs.
# Pass 4: adversarial ownership review of proposed advisors.
# ---------------------------------------------------------------------

RICH_V15_REVIEW_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V15_REVIEW_MAX_NEW_TOKENS", "850")
)

RICH_V16_REVIEW_PROMPT = """أنت Athar OS Advisor Relevance Judge v16.

ستستلم FACTS وVALIDATED_NEEDS وPROPOSED_MATCHES وملفات Expert DNA للمستشارين المقترحين.

هدفك الوحيد هو تحديد درجة الصلة الحقيقية لكل مستشار بالاحتياجات المؤكدة.
لا تقلل العدد ولا تكبره. العدد غير مهم.

صنّف كل مستشار في واحدة فقط من:
DIRECT = الاحتياج يقع مباشرة داخل owned_outcome / core_scope للمستشار.
MATERIAL_SUPPORT = للمستشار مساهمة مستقلة ومادية واضحة في نفس الاحتياج، وليست مجرد مساعدة عامة.
ADJACENT = المجال قريب أو قد يساعد، لكن لا يوجد مخرج مستقل مطلوب من الاحتياج الحالي.
UNRELATED = لا توجد صلة حقيقية بالاحتياج المؤكد.

النتيجة النهائية:
- DIRECT => KEEP
- MATERIAL_SUPPORT => KEEP
- ADJACENT => DROP
- UNRELATED => DROP

اختبار المساهمة المستقلة:
لا يكفي أن يستطيع المستشار "المساعدة".
يجب أن تستطيع تسمية مخرج مستقل سيقدمه لمعالجة VALIDATED_NEED نفسه.
إذا كان السبب يعيد صياغة الاحتياج فقط دون مخرج مختلف، صنّفه ADJACENT.

أمثلة حاسمة:
- إدارة المحفظة والأولويات والاعتماديات بين البرامج:
  13 DIRECT.
  1 لا يصبح MATERIAL_SUPPORT إلا إذا كان الاحتياج يتطلب فعلًا قرارًا تنفيذيًا/تخصيص موارد/ملكية تنفيذية مستقلة، وليس لمجرد كلمة "أولويات".
  8 لا يصبح MATERIAL_SUPPORT إلا إذا كان هناك اختيار/مفاضلة استراتيجية فعلية، لا مجرد ترتيب برامج.
  3 لا يصبح MATERIAL_SUPPORT إلا إذا كان هناك تحليل بيئي/اتجاهات/مقارنة مطلوب فعلًا.
  10 لا يصبح MATERIAL_SUPPORT إلا إذا كان هناك صياغة قضايا/أهداف استراتيجية مطلوبة.
- التخطيط التشغيلي والمواسم والجداول والموارد:
  12 DIRECT.
  13 يمكن أن يكون MATERIAL_SUPPORT إذا كان هناك اعتماديات أو موارد على مستوى المحفظة.
  5 ليس related إلا مع تبنٍ/مقاومة/تحول.
  7 ليس related إلا مع خطر/استمرارية/تعطل.
- 14 يحتاج KPI/قياس/مستهدفات/لوحات.
- 15 يحتاج تقييم/أثر/نتائج/تعلم.
- 16 يحتاج حوكمة/امتثال/صلاحيات/سياسات.
- 6 يحتاج جودة/معايير/تحسين.
- 11 يحتاج تصميم/إعادة تصميم مبادرة.
- 4 يحتاج أصحاب مصلحة/شراكات.
- 2 يحتاج تشخيص/نضج/جاهزية.
- 9 يحتاج هوية/رؤية/رسالة/قيم.

ممنوع اختراع احتياج جديد.
ممنوع استخدام FACTS لتبرير مجال غير موجود في VALIDATED_NEEDS.
ممنوع الاحتفاظ بمستشار لمجرد أنه senior أو عام أو "مفيد".

أخرج سطرًا لكل مستشار مقترح:
ADVISOR_ID|RELATION|FINAL_SCORE|ROLE|MATCHED_NEED_IDS|EVIDENCE_IDS|REASON

RELATION = DIRECT أو MATERIAL_SUPPORT أو ADJACENT أو UNRELATED
ROLE = core أو supporting أو none
DIRECT/MATERIAL_SUPPORT يجب أن يكون له matched_need_ids وevidence_ids صحيحة.
ADJACENT/UNRELATED => ROLE=none وFINAL_SCORE أقل من 40.
REASON يجب أن يذكر الصلة الفعلية أو سبب عدم كفايتها بجملة عربية قصيرة.

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def generate_rich_v15_review(
    facts,
    validated_needs,
    proposed_matches,
    proposed_advisors,
):
    import torch

    messages = [
        {
            "role": "system",
            "content": RICH_V16_REVIEW_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "facts": facts,
                    "validated_needs": validated_needs,
                    "proposed_matches": proposed_matches,
                    "advisor_profiles": proposed_advisors,
                },
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v16 review input too long: {input_tokens}"
        )

    encoded = {
        k: v.to(_RICH_DEVICE)
        for k, v in encoded.items()
    }

    print(
        f"Rich v16 ownership review input tokens: {input_tokens}",
        flush=True,
    )

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V15_REVIEW_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    raw_text = _RICH_TOKENIZER.decode(
        generated[0],
        skip_special_tokens=True,
    )

    return raw_text, input_tokens



def _parse_v16_review(
    text,
    proposed_ids,
    valid_need_ids,
    valid_fact_ids,
):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    kept = []
    decisions = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 6)
        if len(parts) != 7:
            continue

        (
            advisor_raw,
            relation_raw,
            score_raw,
            role_raw,
            needs_raw,
            evidence_raw,
            reason_raw,
        ) = [p.strip() for p in parts]

        a_m = re.search(r"\d+", advisor_raw)
        s_m = re.search(r"\d+(?:\.\d+)?", score_raw)

        if not a_m or not s_m:
            continue

        advisor_id = int(a_m.group())
        if advisor_id not in proposed_ids:
            continue

        relation = relation_raw.upper()
        if relation not in {
            "DIRECT",
            "MATERIAL_SUPPORT",
            "ADJACENT",
            "UNRELATED",
        }:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        role = role_raw.lower()
        if role not in {"core", "supporting", "none"}:
            role = "none"

        matched_need_ids = []
        if needs_raw != "-":
            for token in re.split(r"[,،;\s]+", needs_raw):
                token = token.strip().upper()
                if token in valid_need_ids and token not in matched_need_ids:
                    matched_need_ids.append(token)

        evidence_ids = []
        if evidence_raw != "-":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)
        evidence_ids = evidence_ids[:3]

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        should_keep = relation in {"DIRECT", "MATERIAL_SUPPORT"}

        # Schema/evidence validation only; semantic decision remains AI-made.
        if should_keep:
            if not matched_need_ids or not evidence_ids:
                should_keep = False
            else:
                if score < 40:
                    score = 40.0
                if role == "none":
                    role = "core" if relation == "DIRECT" else "supporting"
        else:
            score = min(score, 39.0)
            role = "none"

        decisions[advisor_id] = {
            "relation": relation,
            "kept": should_keep,
            "score": round(score / 100.0, 4),
            "role": role,
            "matched_need_ids": matched_need_ids,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

        if should_keep:
            kept.append({
                "advisor_id": advisor_id,
                "score": round(score / 100.0, 4),
                "role": role,
                "matched_need_ids": matched_need_ids,
                "evidence_ids": evidence_ids,
                "reason": reason,
            })

    kept.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return kept, decisions


def advisory_match_rich_v16(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        f["fact_id"]
        for f in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    # Pass 1: need discovery
    print(
        "Rich v16 pass 1/4: discovering needs without advisors...",
        flush=True,
    )

    needs_raw, needs_tokens = generate_rich_v12_needs(
        facts
    )

    candidate_needs = _parse_v12_needs(
        needs_raw,
        valid_fact_ids,
    )

    # Pass 2: need validation
    print(
        f"Rich v16 pass 2/4: validating {len(candidate_needs)} candidate needs...",
        flush=True,
    )

    if candidate_needs:
        need_review_raw, need_review_tokens = generate_rich_v13_need_review(
            facts,
            candidate_needs,
        )

        validated_needs, need_review_decisions = _parse_v13_need_review(
            need_review_raw,
            candidate_needs,
            valid_fact_ids,
        )
    else:
        need_review_raw = "NONE"
        need_review_tokens = 0
        validated_needs = []
        need_review_decisions = {}

    # Pass 3: broad global matching proposal
    if validated_needs:
        print(
            "Rich v16 pass 3/4: proposing advisors globally...",
            flush=True,
        )

        proposal_raw, proposal_tokens = generate_rich_v13_matches(
            facts,
            validated_needs,
            advisors,
        )

        proposed_matches = _parse_v12_matches(
            proposal_raw,
            set(range(1, 17)),
            {n["need_id"] for n in validated_needs},
            valid_fact_ids,
        )
    else:
        proposal_raw = "NONE"
        proposal_tokens = 0
        proposed_matches = []

    # Pass 4: ownership adjudication
    if proposed_matches:
        proposed_ids = {
            int(item["advisor_id"])
            for item in proposed_matches
        }

        advisor_by_number = {
            int(advisor["advisor_id"]): advisor
            for advisor in advisors
        }

        proposed_profiles = [
            advisor_by_number[i]
            for i in sorted(proposed_ids)
        ]

        print(
            f"Rich v16 pass 4/4: ownership review of {len(proposed_matches)} proposed advisors...",
            flush=True,
        )

        ownership_raw, ownership_tokens = generate_rich_v15_review(
            facts,
            validated_needs,
            proposed_matches,
            proposed_profiles,
        )

        kept_internal, ownership_decisions = _parse_v16_review(
            ownership_raw,
            proposed_ids,
            {n["need_id"] for n in validated_needs},
            valid_fact_ids,
        )
    else:
        advisor_by_number = {
            int(advisor["advisor_id"]): advisor
            for advisor in advisors
        }
        ownership_raw = "NONE"
        ownership_tokens = 0
        kept_internal = []
        ownership_decisions = {}

    # Registered IDs in final output
    ranked = []

    for item in kept_internal:
        number = int(item["advisor_id"])
        advisor = advisor_by_number[number]

        ranked.append({
            "advisor_id": advisor.get(
                "system_code",
                str(number),
            ),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "score": item["score"],
            "role": item["role"],
            "matched_need_ids": item["matched_need_ids"],
            "evidence_ids": item["evidence_ids"],
            "reason": item["reason"],
        })

    # External API contract: keep the response payload minimal and stable.
    # Internal routing still uses needs, roles, evidence, and review stages,
    # but clients receive only advisor_id, score, and reason.
    public_ranked = [
        {
            "advisor_id": item["advisor_id"],
            "score": item["score"],
            "reason": item["reason"],
        }
        for item in ranked
    ]

    return {
        "ranked": public_ranked
    }


# ---------------------------------------------------------------------
# Rich AI Router v17
# Goal: build a CHOICE POOL for the association (it will later choose 6).
# We do NOT minimize the advisor count.
# We keep every genuinely related advisor with a distinct material contribution.
# Target pool: 8-10 when evidence supports it; minimum desired pool is 7.
# ---------------------------------------------------------------------

RICH_V17_POOL_MIN = int(
    os.environ.get("RICH_V17_POOL_MIN", "7")
)

RICH_V17_POOL_TARGET = int(
    os.environ.get("RICH_V17_POOL_TARGET", "9")
)

RICH_V17_POOL_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V17_POOL_MAX_NEW_TOKENS", "1500")
)

RICH_V17_EXPAND_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V17_EXPAND_MAX_NEW_TOKENS", "900")
)

RICH_V17_POOL_PROMPT = """أنت Athar OS Advisor Candidate Pool Engine v17.

هذه ليست مرحلة اختيار الستة النهائيين.
الجمعية ستختار لاحقًا 6 مستشارين من القائمة التي ترجعها أنت.
مهمتك بناء Candidate Pool أوسع من 6، لكن كل مستشار فيه يجب أن يكون مرتبطًا فعلًا باحتياجات الجمعية.

ستستلم:
1) FACTS موثقة عن الجمعية وبرامجها.
2) VALIDATED_NEEDS تم اكتشافها ومراجعتها قبل رؤية المستشارين.
3) ملفات Expert DNA الغنية للـ35 مستشارًا.

المعيار ليس "هل هذا المستشار ضروري وحده؟"
المعيار هو:
"هل لهذا المستشار مساهمة مستقلة، واضحة، ومسنودة بالأدلة في واحد أو أكثر من الاحتياجات المؤكدة، بما يجعله خيارًا حقيقيًا للجمعية عند اختيار الستة؟"

صنّف كل مستشار في واحدة فقط:

PRIMARY_FIT
= يملك الاحتياج مباشرة أو يعالج جزءًا محوريًا منه.

COMPLEMENTARY_FIT
= لا يملك الاحتياج بالكامل، لكنه يضيف مساهمة مستقلة ومادية ومختلفة عن المالك الرئيسي.

RELEVANT_OPTION
= مرتبط فعليًا بالاحتياج الحالي وله قيمة استشارية واضحة، لكن مساهمته أقل مركزية من الفئتين السابقتين.

ADJACENT
= قريب من الموضوع أو قد يساعد عمومًا، لكن لا توجد مساهمة مستقلة واضحة مطلوبة من الاحتياج الحالي.

UNRELATED
= لا توجد صلة حقيقية.

القائمة النهائية يجب أن تحتوي:
PRIMARY_FIT + COMPLEMENTARY_FIT + RELEVANT_OPTION فقط.

قواعد مهمة:
- لا تطبق Minimum Expert Principle.
- لا تحاول تقليل العدد إلى 2 أو 3.
- الجمعية تحتاج قائمة اختيار أوسع من 6.
- استهدف عادةً 8 إلى 10 مستشارين إذا كانت الوقائع تسمح.
- لا تضف ADJACENT أو UNRELATED فقط للوصول إلى العدد.
- يمكن أن يرتبط عدة مستشارين بنفس الاحتياج إذا كانت مساهمة كل منهم مختلفة فعلًا.
- لا تخترع احتياجًا جديدًا خارج VALIDATED_NEEDS.
- يمكنك استخدام FACTS لتفسير لماذا مساهمة المستشار مادية الآن، لكن لا تستخدمها لإنشاء مشكلة جديدة.

أمثلة تمييز:
- إدارة المحفظة والأولويات والاعتماديات:
  13 غالبًا PRIMARY_FIT.
  12 قد يكون COMPLEMENTARY_FIT إذا كانت الأولويات مرتبطة بالتنفيذ والجداول والموارد.
  1 قد يكون COMPLEMENTARY_FIT إذا كان هناك قرار تنفيذي متعدد البرامج أو تخصيص موارد أو حسم ملكيات.
  8 قد يكون RELEVANT_OPTION إذا كانت الأولويات تتطلب مفاضلة استراتيجية فعلية بين مسارات/برامج.
  10 قد يكون RELEVANT_OPTION إذا كانت الأولويات تحتاج ربطًا واضحًا بالقضايا والأهداف الاستراتيجية.
  3 يكون RELEVANT_OPTION فقط إذا كان تحديد الأولويات يتطلب قراءة داخلية/خارجية أو اتجاهات تدعم القرار.
- التخطيط التشغيلي والمواسم والجداول والموارد:
  12 غالبًا PRIMARY_FIT.
  13 قد يكون COMPLEMENTARY_FIT عند وجود اعتماديات ومحفظة متعددة البرامج.
  6 قد يكون RELEVANT_OPTION إذا كان اتساق التنفيذ والخدمة عبر البرامج المتعددة قضية مادية ظاهرة.
  5 يحتاج فعلًا تحول/تبنٍ/انتقال، وليس مجرد تنفيذ.
  7 يحتاج خطر/استمرارية/تعطل، وليس مجرد موسمية.
- 14 لا يدخل إلا إذا كان هناك احتياج فعلي للقياس/KPI/المستهدفات/لوحات الأداء.
- 15 لا يدخل إلا إذا كان هناك احتياج فعلي للتقييم/الأثر/النتائج/التعلم.
- 16 لا يدخل إلا إذا كان هناك احتياج حوكمة/امتثال/صلاحيات/سياسات.
- 11 لا يدخل إلا إذا كان هناك تصميم/إعادة تصميم مبادرة.
- 4 لا يدخل إلا إذا كان هناك أصحاب مصلحة/شراكات ذات صلة بالاحتياج.
- 2 لا يدخل إلا إذا كان هناك تشخيص/نضج/جاهزية.
- 9 لا يدخل إلا إذا كان هناك هوية/رؤية/رسالة/قيم.

تقييم SCORE:
90-100 = PRIMARY_FIT قوي جدًا
80-89 = PRIMARY_FIT / COMPLEMENTARY_FIT قوي
70-79 = COMPLEMENTARY_FIT واضح
55-69 = RELEVANT_OPTION مادي
40-54 = RELEVANT_OPTION أضعف لكنه ما زال حقيقيًا
أقل من 40 = ADJACENT أو UNRELATED ولا يظهر في القائمة

السبب REASON مهم جدًا:
- لا تكتب سببًا مختصرًا مثل "يملك إدارة المحفظة".
- اكتب 25 إلى 45 كلمة عربية واضحة.
- يجب أن يشرح:
  1) ما الواقعة/الاحتياج الذي يربطه بالجمعية الآن.
  2) ما المساهمة المحددة التي سيقدمها.
  3) لماذا هذه المساهمة مختلفة أو مفيدة عند اختيار المستشارين الستة.
- لا تكرر اسم التخصص فقط.
- لا تستخدم أسبابًا عامة أو تسويقية.

أخرج سطرًا لكل واحد من الـ16 مستشارًا:
ADVISOR_ID|RELATION|SCORE|ROLE|MATCHED_NEED_IDS|EVIDENCE_IDS|REASON

RELATION = PRIMARY_FIT أو COMPLEMENTARY_FIT أو RELEVANT_OPTION أو ADJACENT أو UNRELATED
ROLE = core أو supporting أو none
MATCHED_NEED_IDS من VALIDATED_NEEDS فقط
EVIDENCE_IDS من FACTS فقط، 1-4
REASON كما هو موضح أعلاه

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


RICH_V17_EXPAND_PROMPT = """أنت Athar OS Candidate Pool Expansion Reviewer.

الجولة الأولى أعادت أقل من 7 مستشارين، بينما الجمعية ستختار 6 وتحتاج Candidate Pool أوسع.
راجع فقط المستشارين المستبعدين من الجولة الأولى.

مهمتك ليست ملء العدد بأي ثمن.
احتفظ بمستشار إضافي فقط إذا كان يمكن تصنيفه بصدق كـ RELEVANT_OPTION أو أعلى:
أي لديه مساهمة مستقلة ومادية ومسنودة في أحد VALIDATED_NEEDS، وليس مجرد علاقة عامة أو مجاورة.

ممنوع:
- اختراع احتياج جديد.
- تحويل إنجاز إلى مشكلة.
- إدخال مستشار لأن تخصصه مفيد عمومًا.
- إدخال ADJACENT فقط للوصول إلى 7.

اكتب فقط المستشارين الإضافيين الحقيقيين:
ADVISOR_ID|RELATION|SCORE|ROLE|MATCHED_NEED_IDS|EVIDENCE_IDS|REASON

RELATION يجب أن تكون PRIMARY_FIT أو COMPLEMENTARY_FIT أو RELEVANT_OPTION.
REASON من 25 إلى 45 كلمة عربية، ويشرح الوقائع + المساهمة + سبب الصلة الفعلية.

إذا لا يوجد مستشار إضافي حقيقي اكتب:
NONE
"""


def _parse_v17_pool_lines(
    text,
    valid_advisor_ids,
    valid_need_ids,
    valid_fact_ids,
    allow_only_kept=False,
):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    if not cleaned or cleaned.upper() == "NONE":
        return [], {}

    kept = []
    decisions = {}

    keep_relations = {
        "PRIMARY_FIT",
        "COMPLEMENTARY_FIT",
        "RELEVANT_OPTION",
    }

    valid_relations = keep_relations | {
        "ADJACENT",
        "UNRELATED",
    }

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 6)
        if len(parts) != 7:
            continue

        (
            advisor_raw,
            relation_raw,
            score_raw,
            role_raw,
            needs_raw,
            evidence_raw,
            reason_raw,
        ) = [p.strip() for p in parts]

        a_m = re.search(r"\d+", advisor_raw)
        s_m = re.search(r"\d+(?:\.\d+)?", score_raw)

        if not a_m or not s_m:
            continue

        advisor_id = int(a_m.group())
        if advisor_id not in valid_advisor_ids:
            continue

        relation = relation_raw.upper()
        if relation not in valid_relations:
            continue

        if allow_only_kept and relation not in keep_relations:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        role = role_raw.lower()
        if role not in {"core", "supporting", "none"}:
            role = "none"

        matched_need_ids = []
        if needs_raw != "-":
            for token in re.split(r"[,،;\s]+", needs_raw):
                token = token.strip().upper()
                if token in valid_need_ids and token not in matched_need_ids:
                    matched_need_ids.append(token)

        evidence_ids = []
        if evidence_raw != "-":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)

        evidence_ids = evidence_ids[:4]

        reason = re.sub(
            r"\s+",
            " ",
            reason_raw,
        ).strip()

        should_keep = relation in keep_relations

        # Schema grounding only; AI owns the semantic selection.
        if should_keep:
            if not matched_need_ids or not evidence_ids or not reason:
                should_keep = False
            else:
                if score < 40:
                    score = 40.0

                if role == "none":
                    role = (
                        "core"
                        if relation == "PRIMARY_FIT"
                        else "supporting"
                    )
        else:
            score = min(score, 39.0)
            role = "none"

        decisions[advisor_id] = {
            "relation": relation,
            "kept": should_keep,
            "score": round(score / 100.0, 4),
            "role": role,
            "matched_need_ids": matched_need_ids,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

        if should_keep:
            kept.append({
                "advisor_id": advisor_id,
                "relation": relation,
                "score": round(score / 100.0, 4),
                "role": role,
                "matched_need_ids": matched_need_ids,
                "evidence_ids": evidence_ids,
                "reason": reason,
            })

    # Deduplicate by advisor, keeping highest score.
    dedup = {}
    for item in kept:
        aid = item["advisor_id"]
        if aid not in dedup or item["score"] > dedup[aid]["score"]:
            dedup[aid] = item

    kept = list(dedup.values())
    kept.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return kept, decisions


def _generate_v17_pool(
    facts,
    validated_needs,
    advisors,
):
    import torch

    messages = [
        {
            "role": "system",
            "content": RICH_V17_POOL_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "facts": facts,
                    "validated_needs": validated_needs,
                    "advisors": advisors,
                    "candidate_pool_requirement": {
                        "association_final_selection_count": 6,
                        "desired_candidate_pool_minimum": RICH_V17_POOL_MIN,
                        "target_candidate_pool_size": RICH_V17_POOL_TARGET,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v17 pool input too long: {input_tokens}"
        )

    encoded = {
        k: v.to(_RICH_DEVICE)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V17_POOL_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.06,
            no_repeat_ngram_size=8,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    return (
        _RICH_TOKENIZER.decode(
            generated[0],
            skip_special_tokens=True,
        ),
        input_tokens,
    )


def _generate_v17_expansion(
    facts,
    validated_needs,
    excluded_advisors,
    current_pool,
):
    import torch

    messages = [
        {
            "role": "system",
            "content": RICH_V17_EXPAND_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "facts": facts,
                    "validated_needs": validated_needs,
                    "current_pool": current_pool,
                    "excluded_advisors": excluded_advisors,
                    "desired_candidate_pool_minimum": RICH_V17_POOL_MIN,
                },
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v17 expansion input too long: {input_tokens}"
        )

    encoded = {
        k: v.to(_RICH_DEVICE)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V17_EXPAND_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.06,
            no_repeat_ngram_size=8,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    return (
        _RICH_TOKENIZER.decode(
            generated[0],
            skip_special_tokens=True,
        ),
        input_tokens,
    )


def advisory_match_rich_v17(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        f["fact_id"]
        for f in facts
    }

    advisors = _RICH_REGISTRY["advisors"]
    advisor_by_number = {
        int(a["advisor_id"]): a
        for a in advisors
    }

    # 1) Need discovery
    print(
        "Rich v17 pass 1/4: discovering grounded needs...",
        flush=True,
    )

    needs_raw, needs_tokens = generate_rich_v12_needs(
        facts
    )

    candidate_needs = _parse_v12_needs(
        needs_raw,
        valid_fact_ids,
    )

    # 2) Need validation
    print(
        f"Rich v17 pass 2/4: validating {len(candidate_needs)} needs...",
        flush=True,
    )

    if candidate_needs:
        need_review_raw, need_review_tokens = generate_rich_v13_need_review(
            facts,
            candidate_needs,
        )

        validated_needs, need_review_decisions = _parse_v13_need_review(
            need_review_raw,
            candidate_needs,
            valid_fact_ids,
        )
    else:
        need_review_raw = "NONE"
        need_review_tokens = 0
        validated_needs = []
        need_review_decisions = {}

    # 3) Build candidate pool directly from all 16.
    if validated_needs:
        print(
            "Rich v17 pass 3/4: building broad but relevant advisor candidate pool...",
            flush=True,
        )

        pool_raw, pool_tokens = _generate_v17_pool(
            facts,
            validated_needs,
            advisors,
        )

        pool_internal, pool_decisions = _parse_v17_pool_lines(
            pool_raw,
            set(range(1, 17)),
            {n["need_id"] for n in validated_needs},
            valid_fact_ids,
        )
    else:
        pool_raw = "NONE"
        pool_tokens = 0
        pool_internal = []
        pool_decisions = {}

    # 4) If the genuine pool is still too narrow, one bounded AI expansion pass.
    expansion_raw = "NONE"
    expansion_tokens = 0

    if (
        validated_needs
        and len(pool_internal) < RICH_V17_POOL_MIN
    ):
        kept_ids = {
            item["advisor_id"]
            for item in pool_internal
        }

        excluded_advisors = [
            advisor
            for advisor in advisors
            if int(advisor["advisor_id"]) not in kept_ids
        ]

        print(
            f"Rich v17 pass 4/4: pool has {len(pool_internal)} advisors; "
            f"reviewing excluded profiles for additional genuine relevance...",
            flush=True,
        )

        expansion_raw, expansion_tokens = _generate_v17_expansion(
            facts,
            validated_needs,
            excluded_advisors,
            pool_internal,
        )

        additions, _ = _parse_v17_pool_lines(
            expansion_raw,
            {
                int(a["advisor_id"])
                for a in excluded_advisors
            },
            {n["need_id"] for n in validated_needs},
            valid_fact_ids,
            allow_only_kept=True,
        )

        combined = {
            item["advisor_id"]: item
            for item in pool_internal
        }

        for item in additions:
            if item["advisor_id"] not in combined:
                combined[item["advisor_id"]] = item

        pool_internal = list(
            combined.values()
        )

        pool_internal.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

    # Final public payload: exactly advisor_id + score + detailed reason.
    public_ranked = []

    for item in pool_internal:
        advisor = advisor_by_number[
            int(item["advisor_id"])
        ]

        public_ranked.append({
            "advisor_id": advisor.get(
                "system_code",
                str(item["advisor_id"]),
            ),
            "score": item["score"],
            "reason": item["reason"],
        })

    return {
        "ranked": public_ranked
    }


# ---------------------------------------------------------------------
# Rich AI Router v19
# Key changes:
# 1) Never tell the advisor matcher to "fill" a target count.
# 2) Discover grounded NEEDS + OPPORTUNITIES before seeing advisors.
# 3) Use advisor SYSTEM_CODE directly inside the AI prompt/output to avoid
#    numeric-ID identity drift.
# 4) If the pool is too narrow, expand GROUNDED THEMES (not advisors),
#    validate them, then rematch.
# 5) Public payload stays exactly: {"ranked":[advisor_id, score, reason]}.
# ---------------------------------------------------------------------

RICH_V18_THEME_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V18_THEME_MAX_NEW_TOKENS", "950")
)

RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS", "850")
)

RICH_V18_MATCH_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V18_MATCH_MAX_NEW_TOKENS", "2800")
)

RICH_V18_THEME_EXPAND_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V18_THEME_EXPAND_MAX_NEW_TOKENS", "850")
)

RICH_V18_DESIRED_CHOICE_POOL_MIN = int(
    os.environ.get("RICH_V18_DESIRED_CHOICE_POOL_MIN", "7")
)

RICH_V18_THEME_PROMPT = """أنت Athar OS Advisory Theme Discovery Engine v20.

ستستلم FACTS فقط عن الجمعية وبرامجها. لا ترى أي مستشارين في هذه المرحلة.

استخرج كل Advisory Theme مادي ومثبت يمكن أن يستخدم لاحقًا في ترشيح المستشارين.
الثيمات خمسة أنواع فقط:

EXPLICIT_NEED
= فجوة أو مشكلة أو مخاطرة أو هدف تحسين مذكور بوضوح.

OPERATIONAL_COMPLEXITY
= تعقيد حقيقي ظاهر مثل كثرة البرامج، المواسم، الاعتماديات، الموارد أو تعدد مسارات التنفيذ.

STRATEGIC_OPPORTUNITY
= قرار أو مفاضلة أو توسع أو تغيير اتجاه يحتاج قرارًا استراتيجيًا فعليًا.

SUSTAINMENT_OPPORTUNITY
= إنجاز أو قدرة قائمة تمر بتوسع/تحول/تكامل ملموس يحتاج تثبيتًا أو تطويرًا محددًا. مجرد الحفاظ على شيء جيد لا يكفي.

SECTOR_PORTFOLIO
= مجال قطاعي جوهري ومتكرر في أعمال الجمعية أو أهدافها، بحيث تكون الخبرة القطاعية المتخصصة ذات قيمة مباشرة في مراجعة أو تطوير البرامج الحالية حتى دون وجود مشكلة.

قواعد شديدة الأهمية:
- لا تحول الإنجاز إلى مشكلة.
- وجود رؤية ورسالة واضحتين ليس احتياج هوية.
- وجود أهداف استراتيجية واضحة ليس احتياجًا لإعادة صياغة الأهداف.
- وجود درجة حوكمة مرتفعة ليس احتياج حوكمة.
- وجود مؤشرات وأرقام ليس وحده احتياجًا لبناء KPI أو Dashboard.
- وجود برامج قائمة ليس وحده احتياجًا لإعادة تصميم المبادرات.
- لا تعتبر كل نشاط موسمي مخاطرة أو كل شراكة مشكلة.
- SECTOR_PORTFOLIO يستخدم فقط عندما يكون المجال جزءًا ماديًا من رسالة الجمعية أو محفظة برامجها أو أهدافها، وليس فعالية عابرة واحدة.
- استخرج جميع الثيمات الحقيقية، ولا تبحث عن حد أدنى أو أقصى للعدد.

أمثلة:
- 61 برنامجًا عبر عدة مسارات = OPERATIONAL_COMPLEXITY لإدارة المحفظة والتنسيق التشغيلي.
- 9 شراكات يعتمد عليها تنفيذ خدمات متعددة = SUSTAINMENT_OPPORTUNITY لإدارة محفظة الشراكات والقيمة المتبادلة، إذا كان الاعتماد على الشركاء ظاهرًا.
- برامج صحية متكررة لكبار السن = SECTOR_PORTFOLIO للصحة.
- خدمات اجتماعية مستمرة لكبار السن = SECTOR_PORTFOLIO للخدمات الاجتماعية.
- هدف صريح لإجراء بحوث ودراسات + برامج تعليمية = SECTOR_PORTFOLIO للتعليم والبحث.
- هدف صريح لدعم حقوق كبار السن والتوعية بها = SECTOR_PORTFOLIO للحقوق والمناصرة.
- قاعدة تطوع كبيرة وفرص تطوعية متعددة = SECTOR_PORTFOLIO للعمل التطوعي.
- برامج ثقافية وترفيهية متكررة = SECTOR_PORTFOLIO للثقافة والترفيه إذا كانت مادية ضمن المحفظة.

أخرج:
THEME_ID|TYPE|PRIORITY|EVIDENCE_IDS|THEME

TYPE = EXPLICIT_NEED أو OPERATIONAL_COMPLEXITY أو STRATEGIC_OPPORTUNITY أو SUSTAINMENT_OPPORTUNITY أو SECTOR_PORTFOLIO
PRIORITY = high أو medium أو low
EVIDENCE_IDS من FACTS فقط، 1-4
THEME جملة عربية واضحة ومحددة

إذا لا يوجد شيء مادي اكتب NONE.
ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""

RICH_V18_THEME_REVIEW_PROMPT = """أنت Athar OS Advisory Theme Validator v20.

راجع CANDIDATE_THEMES مقابل FACTS فقط. لا ترى المستشارين.

KEEP إذا كان الثيم:
- مدعومًا مباشرة بالوقائع، أو
- استنتاجًا قريبًا وماديًا من تعقيد ظاهر، أو
- فرصة تطوير مرتبطة بتوسع/تحول/قرار ملموس، أو
- SECTOR_PORTFOLIO جوهريًا ومتكررًا في رسالة الجمعية أو برامجها أو أهدافها.

DROP إذا:
- يحتاج افتراضًا إضافيًا غير موجود.
- يحول إنجازًا إلى مشكلة.
- مجرد مجال عام مفيد.
- يكرر ثيمًا آخر دون إضافة مادية.
- يبني احتياجًا وظيفيًا فقط لأن الجمعية لديها إنجاز قائم.
- يعتبر الهوية الواضحة سببًا لمستشار هوية، أو الأهداف الواضحة سببًا لمستشار أهداف، أو درجة الحوكمة العالية سببًا لمستشار حوكمة، أو وجود أرقام سببًا تلقائيًا لمستشار KPI.

بالنسبة لـ SECTOR_PORTFOLIO:
KEEP فقط إذا كان القطاع جزءًا ماديًا من العمل المتكرر أو الهدف المؤسسي، وليس ذكرًا جانبيًا أو فعالية عابرة.

أخرج لكل ثيم:
THEME_ID|KEEP_OR_DROP|PRIORITY|EVIDENCE_IDS|REASON

KEEP_OR_DROP = KEEP أو DROP
PRIORITY = high أو medium أو low أو none
EVIDENCE_IDS من FACTS فقط، 1-4 عند KEEP ويمكن - عند DROP
REASON سبب عربي واضح للمراجعة

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""

RICH_V18_THEME_EXPAND_PROMPT = """أنت Athar OS Advisory Theme Coverage Reviewer v20.

لديك FACTS وVALIDATED_THEMES الحالية، والقائمة الحالية لم تنتج Candidate Pool واسعًا كفاية لاختيار 6 مستشارين.

لا تبحث عن أسماء مستشارين.
ابحث فقط عن ثيمات حقيقية فاتت الجولة الأولى، وخاصة:
- تعقيد تشغيلي أو محفظي مثبت.
- شراكات جوهرية يعتمد عليها تنفيذ الخدمات.
- قطاعات مادية ومتكررة ضمن البرامج والأهداف مثل الصحة، الخدمات الاجتماعية، التعليم والبحث، الثقافة، الحقوق، التطوع، البيئة، الإسكان أو غيرها.
- فرص توسع أو تغيير حقيقية.

ممنوع:
- اختراع فجوة للوصول إلى عدد أكبر.
- تحويل الإنجاز إلى مشكلة.
- اعتبار الهوية/الحوكمة/الأهداف/المؤشرات الحالية احتياجًا لمجرد وجودها.
- تكرار VALIDATED_THEMES.

أخرج الثيمات الإضافية فقط:
THEME_ID|TYPE|PRIORITY|EVIDENCE_IDS|THEME

TYPE = EXPLICIT_NEED أو OPERATIONAL_COMPLEXITY أو STRATEGIC_OPPORTUNITY أو SUSTAINMENT_OPPORTUNITY أو SECTOR_PORTFOLIO
إذا لا توجد ثيمات إضافية حقيقية اكتب NONE.
ممنوع JSON وممنوع Markdown وممنوع شرح إضافي.
"""

RICH_V18_MATCH_PROMPT = """أنت Athar OS Global Advisor Relevance Engine v20.

ستستلم:
1) FACTS موثقة.
2) VALIDATED_THEMES تم اكتشافها والتحقق منها قبل رؤية المستشارين.
3) ROUTING_CARDS للـ35 مستشارًا.

كل بطاقة تحتوي advisor_class:
FUNCTIONAL = المستشارون 1-25.
SECTOR = المستشارون 26-35.

مهمتك تقييم الـ35 جميعًا عالميًا. لا يوجد Target Count. العدد نتيجة الأدلة فقط.

قاعدة الترشيح تختلف حسب الفئة:

أولًا — FUNCTIONAL
لا ترشح مستشارًا وظيفيًا إلا إذا كان هناك احتياج/قرار/تعقيد/فرصة تطوير حالية تتطلب Owned Outcome الخاص به.
مجرد وجود قدرة أو إنجاز في مجاله لا يكفي.

أمثلة منع ملزمة:
- رؤية ورسالة واضحتان لا تفعّلان AOS-SP-09.
- أهداف استراتيجية واضحة لا تفعّل AOS-SP-10 إلا إذا كانت تحتاج مراجعة/إعادة بناء فعلية.
- وجود استراتيجية وأهداف لا يفعّل AOS-SP-08 دون مفاضلة أو تحديث أو قرار استراتيجي حقيقي.
- درجة حوكمة مرتفعة لا تفعّل AOS-FG-16 دون فجوة امتثال/صلاحيات/سياسات/حوكمة حالية.
- وجود أرقام أو نتائج لا يفعّل AOS-SP-14 دون احتياج حقيقي لنظام KPI/مستهدفات/لوحة/تعريفات أداء.
- وجود برامج لا يفعّل AOS-SP-11 دون تصميم مبادرة جديدة أو إعادة تصميم قائمة.
- وجود برامج ناجحة لا يفعّل AOS-SP-15 دون احتياج فعلي للتقييم/الأثر/التعلم أو قرار توسع مبني على الدليل.

أمثلة إيجابية:
- محفظة كبيرة من البرامج وتنافس الموارد/الأولويات يمكن أن تفعّل AOS-SP-13.
- برامج كثيرة ومتكررة وموسمية تحتاج تنسيقًا للمخرجات والموارد والمواعيد يمكن أن تفعّل AOS-SP-12.
- شبكة شراكات كبيرة يعتمد عليها تنفيذ خدمات متعددة يمكن أن تفعّل AOS-LD-04 إذا كان المطلوب إدارة قيمة الشراكات ومحفظتها، وليس مجرد التواصل معها.

ثانيًا — SECTOR
يمكن ترشيح المستشار القطاعي عندما يوجد SECTOR_PORTFOLIO جوهري أو هدف قطاعي صريح، حتى لو لم توجد مشكلة.
لكن يجب أن يكون القطاع ماديًا ومتكررًا، وأن يضيف المستشار ذكاءً قطاعيًا محددًا للبرامج الحالية.

أمثلة:
- برامج صحية متكررة لكبار السن => AOS-SE-28.
- خدمات اجتماعية ورعاية كبار السن => AOS-SE-29.
- هدف بحثي/تعليمي أو برامج تعليمية جوهرية => AOS-SE-27.
- برامج ثقافية وترفيهية متكررة => AOS-SE-26 إذا كانت مادية.
- دعم حقوق الفئة والتوعية بالحقوق => AOS-SE-32.
- منظومة تطوع وفرص ومتطوعون بأعداد مادية => AOS-SE-33.

قاعدة حدود التخصص:
- AOS-SE-33 إذا اختير بسبب التطوع، يجب أن يكون السبب عن منظومة التطوع/رحلة المتطوع/قيمة التطوع، وليس تصميم الشراكات؛ تصميم الشراكات يخص AOS-LD-04.
- AOS-FG-16 لا يملك تحسين الشراكات.
- AOS-SP-14 يقيس الأداء ولا يملك قياس الأثر السببي؛ الأثر لـ AOS-SP-15.
- لا تنسب أي مخرج لمستشار لا يملكه في ROUTING_CARD.

تصنيفات الصلة:
DIRECT = يملك المخرج المحوري المطلوب.
COMPLEMENTARY = يضيف مخرجًا مستقلًا وماديًا مختلفًا.
RELEVANT_OPTION = قيمة حقيقية لكنها أقل مركزية.
ADJACENT = قريب فقط، لا تخرجه.
UNRELATED = لا تخرجه.

أخرج فقط DIRECT + COMPLEMENTARY + RELEVANT_OPTION.

SCORE:
90-100 = DIRECT محوري جدًا
80-89 = DIRECT/COMPLEMENTARY قوي
70-79 = COMPLEMENTARY واضح
55-69 = RELEVANT_OPTION مادي
50-54 = RELEVANT_OPTION حقيقي لكنه أقل مركزية
أقل من 50 لا تخرجه

قاعدة إخراج إلزامية:
- قيّم كل ROUTING_CARD أمام جميع VALIDATED_THEMES.
- أخرج كل مستشار حصل على SCORE يساوي 50 أو أكثر.
- لا تسقط مستشارًا مناسبًا لمجرد وجود مستشار أعلى منه.
- لا يوجد حد أقصى لعدد المستشارين.
- لا تخرج أي مستشار أقل من 50.

REASON:
- 30 إلى 55 كلمة عربية.
- اذكر الوقائع/الثيم الذي يربطه بالجمعية.
- اشرح المخرج المحدد الذي يقدمه هذا المستشار تحديدًا.
- لا تقل فقط "لديها برامج إذن تحتاج المستشار".
- لا تختلق فجوة غير موجودة.

أخرج:
SYSTEM_CODE|RELATION|SCORE|THEME_IDS|EVIDENCE_IDS|REASON

RELATION = DIRECT أو COMPLEMENTARY أو RELEVANT_OPTION
THEME_IDS من VALIDATED_THEMES فقط
EVIDENCE_IDS من FACTS فقط، 1-4

إذا لا يوجد مستشار مناسب اكتب NONE.
ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _v18_generate_text(system_prompt, payload, max_new_tokens, max_time_seconds=None):
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich v18 input too long: {input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {
        k: v.to(_RICH_DEVICE)
        for k, v in encoded.items()
    }

    with INFERENCE_MODEL_LOCK:
        adapter_context = (
            _RICH_MODEL.disable_adapter()
            if hasattr(_RICH_MODEL, "disable_adapter")
            else None
        )

        if adapter_context is None:
            with torch.inference_mode():
                output_ids = _RICH_MODEL.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    max_time=max_time_seconds,
                    do_sample=False,
                    repetition_penalty=1.06,
                    no_repeat_ngram_size=8,
                    eos_token_id=_RICH_TOKENIZER.eos_token_id,
                    pad_token_id=_RICH_TOKENIZER.pad_token_id,
                    use_cache=True,
                )
        else:
            # Once the council has attached LoRA adapters to the shared Qwen
            # model, advisor matching must still run on the exact base model.
            with adapter_context:
                with torch.inference_mode():
                    output_ids = _RICH_MODEL.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        max_time=max_time_seconds,
                        do_sample=False,
                        repetition_penalty=1.06,
                        no_repeat_ngram_size=8,
                        eos_token_id=_RICH_TOKENIZER.eos_token_id,
                        pad_token_id=_RICH_TOKENIZER.pad_token_id,
                        use_cache=True,
                    )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    raw_text = _RICH_TOKENIZER.decode(
        generated[0],
        skip_special_tokens=True,
    )

    return raw_text, input_tokens


def _parse_v18_themes(text, valid_fact_ids):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    if not cleaned or cleaned.upper() == "NONE":
        return []

    valid_types = {
        "EXPLICIT_NEED",
        "OPERATIONAL_COMPLEXITY",
        "STRATEGIC_OPPORTUNITY",
        "SUSTAINMENT_OPPORTUNITY",
        "SECTOR_PORTFOLIO",
    }

    themes = []
    seen = set()

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        theme_raw, type_raw, priority_raw, evidence_raw, text_raw = [
            p.strip()
            for p in parts
        ]

        m = re.search(
            r"(?:T|N)\s*(\d+)",
            theme_raw,
            flags=re.IGNORECASE,
        )
        if not m:
            continue

        theme_id = f"T{int(m.group(1))}"
        if theme_id in seen:
            continue

        theme_type = type_raw.upper()
        if theme_type not in valid_types:
            continue

        priority = priority_raw.lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"

        evidence_ids = []
        for token in re.split(
            r"[,،;\s]+",
            evidence_raw,
        ):
            token = token.strip().upper()
            if (
                token in valid_fact_ids
                and token not in evidence_ids
            ):
                evidence_ids.append(token)

        evidence_ids = evidence_ids[:4]

        theme_text = re.sub(
            r"\s+",
            " ",
            text_raw,
        ).strip()

        if not evidence_ids or not theme_text:
            continue

        themes.append({
            "theme_id": theme_id,
            "type": theme_type,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "theme": theme_text,
        })

        seen.add(theme_id)

    return themes


def _parse_v18_theme_review(
    text,
    candidate_themes,
    valid_fact_ids,
):
    by_id = {
        item["theme_id"]: item
        for item in candidate_themes
    }

    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    validated = []
    decisions = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        theme_raw, decision_raw, priority_raw, evidence_raw, reason_raw = [
            p.strip()
            for p in parts
        ]

        m = re.search(
            r"(?:T|N)\s*(\d+)",
            theme_raw,
            flags=re.IGNORECASE,
        )
        if not m:
            continue

        theme_id = f"T{int(m.group(1))}"
        if theme_id not in by_id:
            continue

        decision = decision_raw.upper()
        if decision not in {"KEEP", "DROP"}:
            continue

        priority = priority_raw.lower()
        if priority not in {
            "high",
            "medium",
            "low",
            "none",
        }:
            priority = (
                "none"
                if decision == "DROP"
                else by_id[theme_id]["priority"]
            )

        evidence_ids = []
        if evidence_raw != "-":
            for token in re.split(
                r"[,،;\s]+",
                evidence_raw,
            ):
                token = token.strip().upper()
                if (
                    token in valid_fact_ids
                    and token not in evidence_ids
                ):
                    evidence_ids.append(token)

        evidence_ids = evidence_ids[:4]

        reason = re.sub(
            r"\s+",
            " ",
            reason_raw,
        ).strip()

        decisions[theme_id] = {
            "decision": decision,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

        if decision == "KEEP" and evidence_ids:
            original = by_id[theme_id]
            validated.append({
                "theme_id": theme_id,
                "type": original["type"],
                "priority": (
                    original["priority"]
                    if priority == "none"
                    else priority
                ),
                "evidence_ids": evidence_ids,
                "theme": original["theme"],
            })

    return validated, decisions


def _v18_routing_cards(advisors):
    """Build compact but complete routing cards for all 35 advisors."""

    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))
        advisor_class = (
            "SECTOR"
            if advisor_num >= 26
            else "FUNCTIONAL"
        )

        cards.append({
            "system_code": advisor.get("system_code"),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "advisor_class": advisor_class,
            "group": advisor.get("group"),
            "mission": advisor.get("mission"),
            "owned_outcome": advisor.get("owned_outcome"),
            "owns": (advisor.get("owns") or [])[:10],
            "activation_when": (advisor.get("activation_when") or [])[:10],
            "not_primary_when": (advisor.get("not_primary_when") or [])[:6],
            "boundaries": (advisor.get("boundaries") or [])[:5],
        })

    return cards


def _parse_v18_matches(
    text,
    valid_system_codes,
    valid_theme_ids,
    valid_fact_ids,
):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    if not cleaned or cleaned.upper() == "NONE":
        return []

    matches = []
    seen = set()

    valid_relations = {
        "DIRECT",
        "COMPLEMENTARY",
        "RELEVANT_OPTION",
    }

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("|", 5)
        if len(parts) != 6:
            continue

        (
            code_raw,
            relation_raw,
            score_raw,
            themes_raw,
            evidence_raw,
            reason_raw,
        ) = [p.strip() for p in parts]

        system_code = code_raw.strip()
        if (
            system_code not in valid_system_codes
            or system_code in seen
        ):
            continue

        relation = relation_raw.upper()
        if relation not in valid_relations:
            continue

        s_m = re.search(
            r"\d+(?:\.\d+)?",
            score_raw,
        )
        if not s_m:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100

        score = max(
            0.0,
            min(100.0, score),
        )

        if score < 50:
            continue

        theme_ids = []
        for token in re.split(
            r"[,،;\s]+",
            themes_raw,
        ):
            token = token.strip().upper()
            if (
                token in valid_theme_ids
                and token not in theme_ids
            ):
                theme_ids.append(token)

        evidence_ids = []
        for token in re.split(
            r"[,،;\s]+",
            evidence_raw,
        ):
            token = token.strip().upper()
            if (
                token in valid_fact_ids
                and token not in evidence_ids
            ):
                evidence_ids.append(token)

        evidence_ids = evidence_ids[:4]

        reason = re.sub(
            r"\s+",
            " ",
            reason_raw,
        ).strip()

        if (
            not theme_ids
            or not evidence_ids
            or not reason
        ):
            continue

        matches.append({
            "advisor_id": system_code,
            "relation": relation,
            "score": round(
                score / 100.0,
                4,
            ),
            "theme_ids": theme_ids,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })

        seen.add(system_code)

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return matches


def advisory_match_rich_v20(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        f["fact_id"]
        for f in facts
    }

    advisors = _RICH_REGISTRY["advisors"]
    routing_cards = _v18_routing_cards(
        advisors
    )

    # Pass 1: grounded needs + opportunities, without advisors.
    print(
        "Rich v20 pass 1: discovering grounded advisory themes...",
        flush=True,
    )

    themes_raw, themes_tokens = _v18_generate_text(
        RICH_V18_THEME_PROMPT,
        {"facts": facts},
        RICH_V18_THEME_MAX_NEW_TOKENS,
    )

    candidate_themes = _parse_v18_themes(
        themes_raw,
        valid_fact_ids,
    )

    # Pass 2: validate themes, still without advisors.
    print(
        f"Rich v20 pass 2: validating {len(candidate_themes)} themes...",
        flush=True,
    )

    if candidate_themes:
        review_raw, review_tokens = _v18_generate_text(
            RICH_V18_THEME_REVIEW_PROMPT,
            {
                "facts": facts,
                "candidate_themes": candidate_themes,
            },
            RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS,
        )

        validated_themes, theme_decisions = _parse_v18_theme_review(
            review_raw,
            candidate_themes,
            valid_fact_ids,
        )
    else:
        review_raw = "NONE"
        review_tokens = 0
        validated_themes = []
        theme_decisions = {}

    # Pass 3: global match using system_code identities.
    if validated_themes:
        print(
            f"Rich v20 pass 3: matching 35 advisors to {len(validated_themes)} validated themes...",
            flush=True,
        )

        match_raw, match_tokens = _v18_generate_text(
            RICH_V18_MATCH_PROMPT,
            {
                "facts": facts,
                "validated_themes": validated_themes,
                "routing_cards": routing_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

        matches = _parse_v18_matches(
            match_raw,
            {
                card["system_code"]
                for card in routing_cards
                if card.get("system_code")
            },
            {
                theme["theme_id"]
                for theme in validated_themes
            },
            valid_fact_ids,
        )
    else:
        match_raw = "NONE"
        match_tokens = 0
        matches = []

    # Pass 4: if the evidence-backed choice pool is too narrow, expand THEMES,
    # not advisors. This avoids inventing advisor relevance merely to hit a count.
    expansion_raw = "NONE"
    expansion_tokens = 0

    if (
        len(matches) < RICH_V18_DESIRED_CHOICE_POOL_MIN
        and validated_themes
    ):
        print(
            f"Rich v20 pass 4: pool has {len(matches)} advisors; "
            "searching for overlooked grounded themes, not forcing advisors...",
            flush=True,
        )

        expansion_raw, expansion_tokens = _v18_generate_text(
            RICH_V18_THEME_EXPAND_PROMPT,
            {
                "facts": facts,
                "validated_themes": validated_themes,
            },
            RICH_V18_THEME_EXPAND_MAX_NEW_TOKENS,
        )

        extra_candidates = _parse_v18_themes(
            expansion_raw,
            valid_fact_ids,
        )

        # Remove theme IDs already used and renumber extras safely.
        existing_texts = {
            t["theme"].strip().lower()
            for t in validated_themes
        }

        filtered_extras = []
        next_theme_number = (
            max(
                [
                    int(
                        re.search(r"\d+", t["theme_id"]).group()
                    )
                    for t in validated_themes
                ],
                default=0,
            )
            + 1
        )

        for item in extra_candidates:
            if item["theme"].strip().lower() in existing_texts:
                continue

            item = dict(item)
            item["theme_id"] = f"T{next_theme_number}"
            next_theme_number += 1
            filtered_extras.append(item)

        if filtered_extras:
            extra_review_raw, extra_review_tokens = _v18_generate_text(
                RICH_V18_THEME_REVIEW_PROMPT,
                {
                    "facts": facts,
                    "candidate_themes": filtered_extras,
                },
                RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS,
            )

            extra_validated, _ = _parse_v18_theme_review(
                extra_review_raw,
                filtered_extras,
                valid_fact_ids,
            )

            if extra_validated:
                validated_themes = (
                    validated_themes
                    + extra_validated
                )

                rematch_raw, rematch_tokens = _v18_generate_text(
                    RICH_V18_MATCH_PROMPT,
                    {
                        "facts": facts,
                        "validated_themes": validated_themes,
                        "routing_cards": routing_cards,
                    },
                    RICH_V18_MATCH_MAX_NEW_TOKENS,
                )

                rematches = _parse_v18_matches(
                    rematch_raw,
                    {
                        card["system_code"]
                        for card in routing_cards
                        if card.get("system_code")
                    },
                    {
                        theme["theme_id"]
                        for theme in validated_themes
                    },
                    valid_fact_ids,
                )

                if len(rematches) >= len(matches):
                    matches = rematches

    if len(matches) < RICH_V18_DESIRED_CHOICE_POOL_MIN:
        print(
            "Rich v20 warning: grounded data supports fewer than the desired "
            f"{RICH_V18_DESIRED_CHOICE_POOL_MIN} advisor choices. "
            "Returning only genuinely related advisors instead of fabricating relevance.",
            flush=True,
        )

    # Exact public contract requested by the application.
    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v21 — Arabic-only public language
#
# Guarantees:
# - advisor_id remains the registered system code (e.g. AOS-SE-28).
# - Public "reason" text must contain Arabic letters only (plus numbers
#   and punctuation).
# - Latin/Cyrillic/CJK/Greek/etc. letters are detected.
# - If contamination appears, ONE short repair pass rewrites reasons only.
# - A strict sanitizer is still applied as a final safety net.
# ---------------------------------------------------------------------

import unicodedata

RICH_V21_LANGUAGE_REPAIR_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V21_LANGUAGE_REPAIR_MAX_NEW_TOKENS", "1100")
)

RICH_V21_ARABIC_RULES = """
قواعد اللغة الإلزامية:
- كل النصوص الوصفية التي تولدها يجب أن تكون باللغة العربية فقط.
- ممنوع استخدام أي حروف إنجليزية أو روسية أو صينية أو يابانية أو كورية أو يونانية أو أي أبجدية غير عربية داخل النص الوصفي.
- استخدم المقابل العربي للمصطلحات الأجنبية بدل كتابتها بحروف لاتينية.
- يجوز فقط للحقول البروتوكولية الثابتة مثل SYSTEM_CODE وTHEME_ID وEVIDENCE_IDS وRELATION وTYPE أن تبقى بالصيغة المحددة في التعليمات.
- حقل REASON وحقل THEME وأسباب المراجعة يجب أن تكون حروفها عربية بالكامل.
- إذا احتجت إلى ذكر نظام أو تقنية أجنبية، اكتب وصفها العربي فقط، مثل: نظام تخطيط موارد المؤسسة، ولا تكتب الاسم الأجنبي.
"""

# Strengthen every generation stage, while keeping protocol tokens intact.
RICH_V18_THEME_PROMPT = RICH_V18_THEME_PROMPT + "\n" + RICH_V21_ARABIC_RULES
RICH_V18_THEME_REVIEW_PROMPT = RICH_V18_THEME_REVIEW_PROMPT + "\n" + RICH_V21_ARABIC_RULES
RICH_V18_THEME_EXPAND_PROMPT = RICH_V18_THEME_EXPAND_PROMPT + "\n" + RICH_V21_ARABIC_RULES
RICH_V18_MATCH_PROMPT = RICH_V18_MATCH_PROMPT + """
\nقواعد إضافية خاصة بحقل REASON:
- REASON يجب أن يكون عربيًا خالصًا من 30 إلى 55 كلمة.
- لا تستخدم اختصارات أو كلمات أجنبية داخل REASON.
- لا تكتب أسماء النماذج أو الأطر الإنجليزية داخل REASON.
- استخدم صياغة عربية طبيعية ومهنية وخالية من الحروف الدخيلة.
""" + "\n" + RICH_V21_ARABIC_RULES


RICH_V21_REASON_REPAIR_PROMPT = """أنت مدقق لغوي عربي لمنظومة أثر.

ستستلم قائمة أسباب ترشيح جاهزة ومثبتة لمستشارين.
مهمتك لغوية فقط، ولا يجوز تغيير:
- المستشار المختار.
- درجة الملاءمة.
- معنى السبب.
- الوقائع أو الادعاءات.
- قوة العلاقة.

أعد صياغة كل سبب بلغة عربية سليمة وواضحة فقط.

قواعد إلزامية:
- ممنوع وجود أي حرف غير عربي داخل السبب.
- لا تستخدم الإنجليزية أو الروسية أو الصينية أو اليابانية أو الكورية أو اليونانية أو أي أبجدية أخرى.
- استخدم مقابلات عربية للمصطلحات الأجنبية.
- أصلح أي أحرف دخيلة ظهرت داخل كلمة عربية.
- حافظ على السبب بين 30 و55 كلمة قدر الإمكان.
- لا تضف معلومة جديدة.
- لا تغير الترتيب.

أخرج سطرًا لكل مستشار فقط:
SYSTEM_CODE|REASON

SYSTEM_CODE يبقى كما هو لأنه رمز تقني مسجل.
REASON عربي فقط.
ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _v21_contains_non_arabic_letters(text):
    """
    True if text contains a Unicode letter that is not Arabic-script.
    Digits, whitespace and punctuation are allowed.
    """
    for ch in str(text):
        if not ch.isalpha():
            continue

        try:
            name = unicodedata.name(ch)
        except ValueError:
            return True

        if "ARABIC" not in name:
            return True

    return False


def _v21_arabic_only_sanitize(text):
    """
    Final hard safety net:
    preserve Arabic letters, combining marks, digits, spaces and punctuation;
    remove letters from every other script.
    """
    out = []

    for ch in str(text):
        category = unicodedata.category(ch)

        if category.startswith("L"):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = ""

            if "ARABIC" in name:
                out.append(ch)
            else:
                out.append(" ")
            continue

        # Keep Arabic diacritics/combining marks, digits, spaces and punctuation.
        if category.startswith("M"):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = ""
            if "ARABIC" in name:
                out.append(ch)
            continue

        if (
            category.startswith("N")
            or category.startswith("P")
            or category.startswith("Z")
        ):
            out.append(ch)
            continue

        # Common harmless symbols used in Arabic prose.
        if ch in {"٪", "﷼"}:
            out.append(ch)
        else:
            out.append(" ")

    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    cleaned = re.sub(r"\s+([،؛:,.!?؟])", r"\1", cleaned)

    return cleaned


def _v21_repair_reasons(matches):
    if not matches:
        return matches

    contaminated = [
        item
        for item in matches
        if _v21_contains_non_arabic_letters(item.get("reason", ""))
    ]

    if not contaminated:
        # Still apply the final safety net.
        for item in matches:
            item["reason"] = _v21_arabic_only_sanitize(
                item.get("reason", "")
            )
        return matches

    print(
        f"Rich v21 language guard: repairing {len(contaminated)} "
        "reason(s) containing non-Arabic letters...",
        flush=True,
    )

    payload = {
        "items": [
            {
                "advisor_id": item["advisor_id"],
                "reason": item["reason"],
            }
            for item in contaminated
        ]
    }

    try:
        repaired_raw, _ = _v18_generate_text(
            RICH_V21_REASON_REPAIR_PROMPT,
            payload,
            RICH_V21_LANGUAGE_REPAIR_MAX_NEW_TOKENS,
        )

        repaired_by_code = {}

        for raw_line in str(repaired_raw).splitlines():
            line = raw_line.strip()

            if not line or "|" not in line:
                continue

            code, reason = line.split("|", 1)
            code = code.strip()
            reason = re.sub(r"\s+", " ", reason).strip()

            if not code or not reason:
                continue

            # Accept only repair text that is itself Arabic-only.
            if _v21_contains_non_arabic_letters(reason):
                continue

            repaired_by_code[code] = reason

        for item in matches:
            code = item["advisor_id"]

            if code in repaired_by_code:
                item["reason"] = repaired_by_code[code]

    except Exception as exc:
        print(
            f"Rich v21 language repair pass failed; "
            f"using strict sanitizer fallback: {exc}",
            flush=True,
        )

    # Absolute output guarantee even if the repair model failed.
    for item in matches:
        cleaned = _v21_arabic_only_sanitize(
            item.get("reason", "")
        )

        # Avoid an empty public reason after sanitisation.
        if not cleaned:
            cleaned = (
                "ترتبط خبرة هذا المستشار مباشرة باحتياج موثق في بيانات الجمعية، "
                "ويقدم مساهمة تخصصية واضحة تدعم تحسين القرار والتنفيذ ضمن نطاق اختصاصه."
            )

        item["reason"] = cleaned

    return matches


def advisory_match_rich_v21(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        f["fact_id"]
        for f in facts
    }

    advisors = _RICH_REGISTRY["advisors"]
    routing_cards = _v18_routing_cards(
        advisors
    )

    print(
        "Rich v21 pass 1: discovering grounded advisory themes...",
        flush=True,
    )

    themes_raw, themes_tokens = _v18_generate_text(
        RICH_V18_THEME_PROMPT,
        {"facts": facts},
        RICH_V18_THEME_MAX_NEW_TOKENS,
    )

    candidate_themes = _parse_v18_themes(
        themes_raw,
        valid_fact_ids,
    )

    print(
        f"Rich v21 pass 2: validating {len(candidate_themes)} themes...",
        flush=True,
    )

    if candidate_themes:
        review_raw, review_tokens = _v18_generate_text(
            RICH_V18_THEME_REVIEW_PROMPT,
            {
                "facts": facts,
                "candidate_themes": candidate_themes,
            },
            RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS,
        )

        validated_themes, theme_decisions = _parse_v18_theme_review(
            review_raw,
            candidate_themes,
            valid_fact_ids,
        )
    else:
        review_raw = "NONE"
        review_tokens = 0
        validated_themes = []
        theme_decisions = {}

    if validated_themes:
        print(
            f"Rich v21 pass 3: matching 35 advisors to "
            f"{len(validated_themes)} validated themes...",
            flush=True,
        )

        match_raw, match_tokens = _v18_generate_text(
            RICH_V18_MATCH_PROMPT,
            {
                "facts": facts,
                "validated_themes": validated_themes,
                "routing_cards": routing_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

        matches = _parse_v18_matches(
            match_raw,
            {
                card["system_code"]
                for card in routing_cards
                if card.get("system_code")
            },
            {
                theme["theme_id"]
                for theme in validated_themes
            },
            valid_fact_ids,
        )
    else:
        match_raw = "NONE"
        match_tokens = 0
        matches = []

    expansion_raw = "NONE"
    expansion_tokens = 0

    if (
        len(matches) < RICH_V18_DESIRED_CHOICE_POOL_MIN
        and validated_themes
    ):
        print(
            f"Rich v21 pass 4: pool has {len(matches)} advisors; "
            "searching for overlooked grounded themes...",
            flush=True,
        )

        expansion_raw, expansion_tokens = _v18_generate_text(
            RICH_V18_THEME_EXPAND_PROMPT,
            {
                "facts": facts,
                "validated_themes": validated_themes,
            },
            RICH_V18_THEME_EXPAND_MAX_NEW_TOKENS,
        )

        extra_candidates = _parse_v18_themes(
            expansion_raw,
            valid_fact_ids,
        )

        existing_texts = {
            t["theme"].strip().lower()
            for t in validated_themes
        }

        filtered_extras = []

        next_theme_number = (
            max(
                [
                    int(
                        re.search(
                            r"\d+",
                            t["theme_id"],
                        ).group()
                    )
                    for t in validated_themes
                ],
                default=0,
            )
            + 1
        )

        for item in extra_candidates:
            if item["theme"].strip().lower() in existing_texts:
                continue

            item = dict(item)
            item["theme_id"] = f"T{next_theme_number}"
            next_theme_number += 1
            filtered_extras.append(item)

        if filtered_extras:
            extra_review_raw, extra_review_tokens = _v18_generate_text(
                RICH_V18_THEME_REVIEW_PROMPT,
                {
                    "facts": facts,
                    "candidate_themes": filtered_extras,
                },
                RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS,
            )

            extra_validated, _ = _parse_v18_theme_review(
                extra_review_raw,
                filtered_extras,
                valid_fact_ids,
            )

            if extra_validated:
                validated_themes = (
                    validated_themes
                    + extra_validated
                )

                rematch_raw, rematch_tokens = _v18_generate_text(
                    RICH_V18_MATCH_PROMPT,
                    {
                        "facts": facts,
                        "validated_themes": validated_themes,
                        "routing_cards": routing_cards,
                    },
                    RICH_V18_MATCH_MAX_NEW_TOKENS,
                )

                rematches = _parse_v18_matches(
                    rematch_raw,
                    {
                        card["system_code"]
                        for card in routing_cards
                        if card.get("system_code")
                    },
                    {
                        theme["theme_id"]
                        for theme in validated_themes
                    },
                    valid_fact_ids,
                )

                if len(rematches) >= len(matches):
                    matches = rematches

    if len(matches) < RICH_V18_DESIRED_CHOICE_POOL_MIN:
        print(
            "Rich v21 warning: grounded data supports fewer than the desired "
            f"{RICH_V18_DESIRED_CHOICE_POOL_MIN} advisor choices. "
            "Returning only genuinely related advisors.",
            flush=True,
        )

    # NEW: repair any script contamination and enforce Arabic-only reasons.
    matches = _v21_repair_reasons(matches)

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v22 — compact 35-advisor routing context
#
# Fixes:
# - v21 could exceed RICH_MAX_INPUT_TOKENS because all 35 routing cards
#   were too verbose (example: 20,680 > 20,000 tokens).
# - Keep the 20k safety limit.
# - Send only the fields required for advisor routing.
# - Arabic-only output guard from v21 remains active.
# ---------------------------------------------------------------------

def _v18_routing_cards(advisors):
    """
    Compact routing cards for 35 advisors.

    We intentionally keep only:
    - registered system code
    - Arabic advisor name
    - functional/sector class
    - owned outcome
    - core owned areas
    - strongest activation conditions
    - strongest non-primary conditions

    Full DNA remains in the registry; it is not necessary to inject the
    entire DNA into every global routing call.
    """

    cards = []

    for advisor in advisors:
        advisor_num = int(
            advisor.get("advisor_id")
        )

        advisor_class = (
            "SECTOR"
            if advisor_num >= 26
            else "FUNCTIONAL"
        )

        cards.append({
            "system_code": advisor.get(
                "system_code"
            ),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "advisor_class": advisor_class,
            "owned_outcome": advisor.get(
                "owned_outcome"
            ),
            "owns": (
                advisor.get("owns") or []
            )[:6],
            "activation_when": (
                advisor.get("activation_when") or []
            )[:6],
            "not_primary_when": (
                advisor.get("not_primary_when") or []
            )[:3],
        })

    return cards


def advisory_match_rich_v22(job_input):
    """
    v22 reuses the complete v21 pipeline and Arabic-only language guard,
    but with the compact 35-advisor routing cards defined above.
    """
    return advisory_match_rich_v21(
        job_input
    )


# ---------------------------------------------------------------------
# Rich AI Router v23 — split functional/sector adjudication
#
# Why:
# - With 35 advisors in one global matching pass, sector advisors can
#   dominate attention and suppress genuine functional advisors.
# - v23 discovers/validates themes once, then runs TWO independent
#   matching passes:
#       1) FUNCTIONAL advisors 1-25
#       2) SECTOR advisors 26-35
# - No minimum-count expansion is used. The pool emerges from evidence.
# - Arabic-only public reasons from v21 remain enforced.
# ---------------------------------------------------------------------

RICH_V23_THEME_PROMPT = RICH_V18_THEME_PROMPT + """

فحص تغطية إلزامي قبل إخراج الثيمات:
راجع FACTS كلها ثم افحص بصورة مستقلة ما إذا كان يوجد دليل مادي على كل بُعد من الأبعاد التالية:
1) تعقيد المحفظة: كثرة البرامج/المبادرات وتعدد المسارات والحاجة إلى ترتيب أو تنسيق المحفظة.
2) التعقيد التشغيلي: برامج دورية أو موسمية أو متعددة الجهات والموارد والمواعيد.
3) الشراكات: اعتماد مادي على شبكة شركاء لتنفيذ خدمات متعددة أو توسيع القيمة المتبادلة.
4) التطوع: قاعدة متطوعين وفرص تطوعية جوهرية ومتكررة.
5) القطاعات الجوهرية: صحة، خدمات اجتماعية، تعليم/بحث، ثقافة/ترفيه، حقوق، بيئة، إسكان، دعوة/ضيوف الرحمن، أو جمعية مهنية.
6) أي قرار توسع أو تغيير أو استدامة مثبت في الوقائع.

لا تُخرج بُعدًا بلا دليل، لكن لا تسقط بُعدًا ماديًا فقط لأن ثيمًا قطاعيًا آخر يبدو أوضح.
المطلوب جميع الثيمات المستقلة الحقيقية، وليس أهم عدد محدود منها.
"""


def _v23_routing_cards(advisors, advisor_class, detailed=True):
    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))
        current_class = "SECTOR" if advisor_num >= 26 else "FUNCTIONAL"

        if current_class != advisor_class:
            continue

        if detailed:
            card = {
                "system_code": advisor.get("system_code"),
                "advisor_name": advisor.get(
                    "name_ar",
                    advisor.get("name_en"),
                ),
                "advisor_class": current_class,
                "mission": advisor.get("mission"),
                "owned_outcome": advisor.get("owned_outcome"),
                "owns": (advisor.get("owns") or [])[:8],
                "activation_when": (advisor.get("activation_when") or [])[:8],
                "not_primary_when": (advisor.get("not_primary_when") or [])[:4],
                "boundaries": (advisor.get("boundaries") or [])[:3],
            }
        else:
            card = {
                "system_code": advisor.get("system_code"),
                "advisor_name": advisor.get(
                    "name_ar",
                    advisor.get("name_en"),
                ),
                "advisor_class": current_class,
                "owned_outcome": advisor.get("owned_outcome"),
                "owns": (advisor.get("owns") or [])[:5],
                "activation_when": (advisor.get("activation_when") or [])[:5],
                "not_primary_when": (advisor.get("not_primary_when") or [])[:2],
            }

        cards.append(card)

    return cards


RICH_V23_FUNCTIONAL_MATCH_PROMPT = RICH_V18_MATCH_PROMPT + """

هذه الجولة مخصصة للمستشارين الوظيفيين FUNCTIONAL فقط.
لا تتوقع وجود المستشارين القطاعيين في ROUTING_CARDS ولا تعاقب المستشار الوظيفي بسبب غيابهم.
قيّم كل بطاقة وظيفية أمام جميع VALIDATED_THEMES ذات الصلة.

تذكير مهم:
- تعقيد محفظة كبيرة قد يفعّل مستشار المحافظ والبرامج والمشاريع.
- التشغيل المتكرر والمتعدد المسارات قد يفعّل مستشار التخطيط التشغيلي.
- شبكة شراكات يعتمد عليها التنفيذ قد تفعّل مستشار أصحاب المصلحة والشراكات.
- لا تفعّل الهوية أو الحوكمة أو المؤشرات أو الاستراتيجية لمجرد أن هذه الأشياء موجودة بالفعل.
"""

RICH_V23_SECTOR_MATCH_PROMPT = RICH_V18_MATCH_PROMPT + """

هذه الجولة مخصصة للمستشارين القطاعيين SECTOR فقط.
لا تتوقع وجود المستشارين الوظيفيين في ROUTING_CARDS.
قيّم كل مستشار قطاعي فقط إذا كان القطاع جوهريًا ومتكررًا في رسالة الجمعية أو أهدافها أو محفظة برامجها.

لا تُضعف مستشارًا قطاعيًا فقط لأن هناك مستشارًا وظيفيًا قد يغطي جانب التنفيذ؛ المستشار القطاعي يضيف الذكاء الفني للقطاع نفسه.
"""


def _v23_match_pass(
    system_prompt,
    facts,
    validated_themes,
    advisors,
    advisor_class,
    valid_fact_ids,
):
    detailed_cards = _v23_routing_cards(
        advisors,
        advisor_class,
        detailed=True,
    )

    valid_codes = {
        card["system_code"]
        for card in detailed_cards
        if card.get("system_code")
    }

    valid_theme_ids = {
        theme["theme_id"]
        for theme in validated_themes
    }

    payload = {
        "facts": facts,
        "validated_themes": validated_themes,
        "routing_cards": detailed_cards,
    }

    try:
        raw, tokens = _v18_generate_text(
            system_prompt,
            payload,
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        if "input too long" not in str(exc).lower():
            raise

        print(
            f"Rich v23 {advisor_class} pass exceeded token budget; "
            "retrying with compact routing cards...",
            flush=True,
        )

        compact_cards = _v23_routing_cards(
            advisors,
            advisor_class,
            detailed=False,
        )

        valid_codes = {
            card["system_code"]
            for card in compact_cards
            if card.get("system_code")
        }

        raw, tokens = _v18_generate_text(
            system_prompt,
            {
                "facts": facts,
                "validated_themes": validated_themes,
                "routing_cards": compact_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

    matches = _parse_v18_matches(
        raw,
        valid_codes,
        valid_theme_ids,
        valid_fact_ids,
    )

    return matches, raw, tokens


RICH_V24_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V24_MIN_PUBLIC_SCORE", "0.50")
)

def advisory_match_rich_v24(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        f["fact_id"]
        for f in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v24 pass 1: discovering all grounded advisory themes...",
        flush=True,
    )

    themes_raw, themes_tokens = _v18_generate_text(
        RICH_V23_THEME_PROMPT,
        {"facts": facts},
        RICH_V18_THEME_MAX_NEW_TOKENS,
    )

    candidate_themes = _parse_v18_themes(
        themes_raw,
        valid_fact_ids,
    )

    print(
        f"Rich v24 pass 2: validating {len(candidate_themes)} themes...",
        flush=True,
    )

    if candidate_themes:
        review_raw, review_tokens = _v18_generate_text(
            RICH_V18_THEME_REVIEW_PROMPT,
            {
                "facts": facts,
                "candidate_themes": candidate_themes,
            },
            RICH_V18_THEME_REVIEW_MAX_NEW_TOKENS,
        )

        validated_themes, theme_decisions = _parse_v18_theme_review(
            review_raw,
            candidate_themes,
            valid_fact_ids,
        )
    else:
        validated_themes = []

    if not validated_themes:
        return {"ranked": []}

    print(
        f"Rich v24 pass 3A: matching 25 FUNCTIONAL advisors to "
        f"{len(validated_themes)} validated themes...",
        flush=True,
    )

    functional_matches, _, _ = _v23_match_pass(
        RICH_V23_FUNCTIONAL_MATCH_PROMPT,
        facts,
        validated_themes,
        advisors,
        "FUNCTIONAL",
        valid_fact_ids,
    )

    print(
        f"Rich v24 pass 3B: matching 10 SECTOR advisors to "
        f"{len(validated_themes)} validated themes...",
        flush=True,
    )

    sector_matches, _, _ = _v23_match_pass(
        RICH_V23_SECTOR_MATCH_PROMPT,
        facts,
        validated_themes,
        advisors,
        "SECTOR",
        valid_fact_ids,
    )

    merged = {}
    for item in functional_matches + sector_matches:
        code = item["advisor_id"]

        if (
            code not in merged
            or item["score"] > merged[code]["score"]
        ):
            merged[code] = item

    matches = sorted(
        [
            item
            for item in merged.values()
            if item["score"] >= RICH_V24_MIN_PUBLIC_SCORE
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    # Enforce Arabic-only public reasons.
    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v24 final grounded pool >= {RICH_V24_MIN_PUBLIC_SCORE:.2f}: {len(matches)} advisors "
        f"({len(functional_matches)} functional + "
        f"{len(sector_matches)} sector).",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v25 — score every advisor, then filter >= 0.50
#
# Core guarantee:
# - Every one of the 35 advisors receives an AI relevance score.
# - Two independent passes:
#     FUNCTIONAL: advisors 1-25
#     SECTOR: advisors 26-35
# - The model is NOT allowed to omit low-scoring advisors.
# - Only after all scores are parsed do we filter score >= 0.50.
# - No theme gate can cause an empty result before scoring.
# - Arabic-only public reasons remain enforced.
# ---------------------------------------------------------------------

RICH_V25_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V25_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V25_FUNCTIONAL_PROMPT = """
أنت محرك تقييم ملاءمة المستشارين الوظيفيين في منظومة أثر.

لديك:
FACTS = وقائع موثقة عن الجمعية وبرامجها.
ROUTING_CARDS = بطاقات المستشارين الوظيفيين.

مهمتك:
قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء.
لا يجوز حذف أي مستشار من التقييم.
أعط كل مستشار درجة من 0 إلى 100 حسب ملاءمته الحالية الفعلية للجمعية.

قاعدة التقييم:
- 90-100: احتياج مباشر ومحوري يملكه هذا المستشار.
- 80-89: احتياج مباشر وقوي.
- 70-79: مساهمة تكميلية واضحة ومادية.
- 60-69: خيار ذو قيمة حقيقية ومسنودة بالأدلة.
- 50-59: ملاءمة حقيقية لكنها أقل مركزية.
- 0-49: غير كافٍ للترشيح الحالي.

قواعد صارمة للمستشارين الوظيفيين:
- وجود شيء ناجح في مجال المستشار لا يعني أن الجمعية تحتاجه.
- وجود رؤية ورسالة واضحة لا يفعّل مستشار الهوية.
- وجود أهداف استراتيجية واضحة لا يفعّل مستشار القضايا والأهداف إلا إذا وُجد احتياج لإعادة بناء أو مراجعة فعلية.
- وجود استراتيجية لا يفعّل مستشار التخطيط الاستراتيجي دون قرار استراتيجي أو مفاضلة أو تحديث حقيقي.
- ارتفاع الحوكمة لا يفعّل مستشار الحوكمة دون فجوة أو قرار حوكمي فعلي.
- وجود أرقام لا يفعّل مستشار المؤشرات دون احتياج حقيقي لبناء أو تطوير نظام أداء.
- وجود برامج لا يفعّل مستشار تصميم المبادرات إلا عند تصميم أو إعادة تصميم تدخل.
- وجود برامج كثيرة ومتنوعة قد يفعّل مستشار المحافظ والبرامج والمشاريع إذا ظهر تعقيد محفظة حقيقي.
- التشغيل المتكرر أو الموسمي أو متعدد الجهات والموارد قد يفعّل مستشار التخطيط التشغيلي.
- شبكة شركاء يعتمد عليها تنفيذ الخدمات قد تفعّل مستشار أصحاب المصلحة والشراكات.
- وجود أموال أو إيرادات لا يفعّل المستشار المالي أو تنمية الموارد تلقائيًا؛ يجب أن توجد حاجة مالية أو تمويلية فعلية موثقة.
- لا تخترع فجوة أو مشكلة غير موجودة في FACTS.

حدود التخصص:
- لا تنسب للمستشار مخرجًا لا يملكه في ROUTING_CARD.
- إذا كان السبب أقرب لتخصص مستشار آخر، خفّض الدرجة.
- لا تستخدم عدد المستشارين المطلوب كعامل في الدرجة.

قواعد اللغة:
- حقل REASON عربي فقط.
- ممنوع أي حروف إنجليزية أو روسية أو صينية أو يابانية أو كورية أو يونانية داخل REASON.
- SYSTEM_CODE يبقى كما هو لأنه رمز تقني.
- استخدم المصطلحات العربية بدل الكلمات الأجنبية.

صيغة الإخراج الإلزامية:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|REASON

تعليمات الإخراج:
- أخرج سطرًا واحدًا لكل ROUTING_CARD، بنفس عدد البطاقات تمامًا.
- SCORE رقم من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط، من 1 إلى 4 معرفات عند وجود دليل.
- إذا كانت الدرجة أقل من 50 ومافيش دليل مباشر، يمكن كتابة NONE في EVIDENCE_IDS.
- REASON يشرح سبب الدرجة باختصار وبدون اختلاق.
- للمستشارين بدرجة 50 أو أكثر: السبب من 25 إلى 50 كلمة عربية ويذكر الوقائع والمساهمة المحددة.
- للمستشارين أقل من 50: يكفي سبب عربي قصير يوضح لماذا لا توجد ملاءمة كافية.
- ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""

RICH_V25_SECTOR_PROMPT = """
أنت محرك تقييم ملاءمة المستشارين القطاعيين في منظومة أثر.

لديك:
FACTS = وقائع موثقة عن الجمعية وبرامجها.
ROUTING_CARDS = بطاقات المستشارين القطاعيين.

مهمتك:
قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء.
لا يجوز حذف أي مستشار من التقييم.
أعط كل مستشار درجة من 0 إلى 100 حسب مدى جوهرية قطاعه في رسالة الجمعية وأهدافها ومحفظة برامجها الحالية.

قاعدة التقييم:
- 90-100: القطاع جوهري ومحوري ومتكرر جدًا في عمل الجمعية.
- 80-89: القطاع جوهري وله برامج وخدمات واضحة ومتكررة.
- 70-79: القطاع مهم وله حضور مادي واضح.
- 60-69: القطاع ذو صلة حقيقية لكنه ليس المحور الأول.
- 50-59: صلة قطاعية حقيقية ومحدودة نسبيًا.
- 0-49: القطاع غير مادي أو مجرد نشاط عابر.

قواعد صارمة للمستشارين القطاعيين:
- المستشار القطاعي يمكن أن يكون مناسبًا بسبب وجود محفظة برامج جوهرية في قطاعه حتى دون وجود مشكلة.
- لا يكفي نشاط واحد عابر لرفع الدرجة إلى 50.
- الصحة المتكررة والرعاية الصحية لكبار السن قد تفعّل مستشار الصحة.
- الرعاية والخدمات الاجتماعية وإدارة احتياجات كبار السن قد تفعّل مستشار الخدمات الاجتماعية.
- البرامج التعليمية أو البحثية المادية أو هدف صريح للبحث والتعليم قد تفعّل مستشار التعليم والبحث.
- البرامج الثقافية والترفيهية المتكررة قد تفعّل مستشار الثقافة والترفيه.
- دعم الحقوق أو التوعية بها أو المناصرة قد يفعّل مستشار الحقوق والمناصرة.
- منظومة تطوع مادية ومتكررة تفعّل مستشار دعم العمل الخيري والتطوعي من زاوية التطوع وبناء القدرة، وليس من زاوية تصميم الشراكات.
- لا ترفع مستشارًا قطاعيًا بسبب كلمة عابرة فقط.
- لا تستخدم عدد المستشارين المطلوب كعامل في الدرجة.

حدود التخصص:
- لا تنسب للمستشار مخرجًا لا يملكه في ROUTING_CARD.
- إذا كان السبب وظيفيًا بحتًا وليس قطاعيًا، خفّض الدرجة.

قواعد اللغة:
- حقل REASON عربي فقط.
- ممنوع أي حروف إنجليزية أو روسية أو صينية أو يابانية أو كورية أو يونانية داخل REASON.
- SYSTEM_CODE يبقى كما هو لأنه رمز تقني.
- استخدم المصطلحات العربية بدل الكلمات الأجنبية.

صيغة الإخراج الإلزامية:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|REASON

تعليمات الإخراج:
- أخرج سطرًا واحدًا لكل ROUTING_CARD، بنفس عدد البطاقات تمامًا.
- SCORE رقم من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط، من 1 إلى 4 معرفات عند وجود دليل.
- إذا كانت الدرجة أقل من 50 ومافيش دليل مباشر، يمكن كتابة NONE في EVIDENCE_IDS.
- REASON يشرح سبب الدرجة باختصار وبدون اختلاق.
- للمستشارين بدرجة 50 أو أكثر: السبب من 25 إلى 50 كلمة عربية ويذكر الوقائع والمساهمة المحددة.
- للمستشارين أقل من 50: يكفي سبب عربي قصير.
- ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _v25_cards(advisors, advisor_class, compact=False):
    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))
        current_class = "SECTOR" if advisor_num >= 26 else "FUNCTIONAL"

        if current_class != advisor_class:
            continue

        if compact:
            card = {
                "system_code": advisor.get("system_code"),
                "advisor_name": advisor.get(
                    "name_ar",
                    advisor.get("name_en"),
                ),
                "owned_outcome": advisor.get("owned_outcome"),
                "owns": (advisor.get("owns") or [])[:5],
                "activation_when": (advisor.get("activation_when") or [])[:5],
                "not_primary_when": (advisor.get("not_primary_when") or [])[:2],
            }
        else:
            card = {
                "system_code": advisor.get("system_code"),
                "advisor_name": advisor.get(
                    "name_ar",
                    advisor.get("name_en"),
                ),
                "mission": advisor.get("mission"),
                "owned_outcome": advisor.get("owned_outcome"),
                "owns": (advisor.get("owns") or [])[:7],
                "activation_when": (advisor.get("activation_when") or [])[:7],
                "not_primary_when": (advisor.get("not_primary_when") or [])[:4],
                "boundaries": (advisor.get("boundaries") or [])[:3],
            }

        cards.append(card)

    return cards


def _v25_parse_scores(text, expected_codes, valid_fact_ids):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    rows = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 3)
        if len(parts) != 4:
            continue

        code_raw, score_raw, evidence_raw, reason_raw = [
            p.strip() for p in parts
        ]

        code = code_raw.strip()
        if code not in expected_codes or code in rows:
            continue

        m = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100

        score = max(0.0, min(100.0, score))

        evidence_ids = []
        if evidence_raw.upper() != "NONE":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if (
                    token in valid_fact_ids
                    and token not in evidence_ids
                ):
                    evidence_ids.append(token)

        evidence_ids = evidence_ids[:4]

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        if not reason:
            reason = "لا توجد ملاءمة كافية مدعومة بالوقائع الحالية."

        rows[code] = {
            "advisor_id": code,
            "score": round(score / 100.0, 4),
            "evidence_ids": evidence_ids,
            "reason": reason,
        }

    return rows


def _v25_score_pass(
    prompt,
    facts,
    advisors,
    advisor_class,
    valid_fact_ids,
):
    cards = _v25_cards(
        advisors,
        advisor_class,
        compact=False,
    )

    expected_codes = {
        card["system_code"]
        for card in cards
        if card.get("system_code")
    }

    try:
        raw, tokens = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        if "input too long" not in str(exc).lower():
            raise

        print(
            f"Rich v25 {advisor_class}: retrying with compact cards...",
            flush=True,
        )

        cards = _v25_cards(
            advisors,
            advisor_class,
            compact=True,
        )

        expected_codes = {
            card["system_code"]
            for card in cards
            if card.get("system_code")
        }

        raw, tokens = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

    parsed = _v25_parse_scores(
        raw,
        expected_codes,
        valid_fact_ids,
    )

    missing_codes = expected_codes - set(parsed.keys())

    if missing_codes:
        print(
            f"Rich v25 warning: {advisor_class} model omitted "
            f"{len(missing_codes)} advisor score rows: "
            f"{sorted(missing_codes)}",
            flush=True,
        )

    return parsed, raw, tokens, expected_codes


def advisory_match_rich_v25(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v25 pass A: scoring every FUNCTIONAL advisor 1-25...",
        flush=True,
    )

    functional_scores, _, _, functional_codes = _v25_score_pass(
        RICH_V25_FUNCTIONAL_PROMPT,
        facts,
        advisors,
        "FUNCTIONAL",
        valid_fact_ids,
    )

    print(
        "Rich v25 pass B: scoring every SECTOR advisor 26-35...",
        flush=True,
    )

    sector_scores, _, _, sector_codes = _v25_score_pass(
        RICH_V25_SECTOR_PROMPT,
        facts,
        advisors,
        "SECTOR",
        valid_fact_ids,
    )

    all_scores = {}
    all_scores.update(functional_scores)
    all_scores.update(sector_scores)

    expected_all = functional_codes | sector_codes
    missing_all = expected_all - set(all_scores.keys())

    # Missing model rows are treated as unscored/0, never as eligible.
    for code in missing_all:
        all_scores[code] = {
            "advisor_id": code,
            "score": 0.0,
            "evidence_ids": [],
            "reason": "لم يقدم النموذج تقييمًا صالحًا لهذا المستشار في هذه الجولة.",
        }

    matches = [
        item
        for item in all_scores.values()
        if (
            item["score"] >= RICH_V25_MIN_PUBLIC_SCORE
            and item.get("evidence_ids")
        )
    ]

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Enforce Arabic-only public reasons.
    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v25 scored {len(expected_all)} advisors; "
        f"returning {len(matches)} with score >= "
        f"{RICH_V25_MIN_PUBLIC_SCORE:.2f} and grounded evidence.",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v26 — calibrated 0.50 threshold
#
# Problem fixed:
# v25 correctly scored all 35 advisors, but the model used 0.50 as a
# "maybe / possible" floor, which produced many false positives.
#
# v26 keeps the user's exact business rule:
#   return every advisor whose FINAL score >= 0.50
#
# But score semantics are calibrated:
#   >= 0.50 means there is grounded, current, material relevance.
#   "possible / maybe / no clear evidence" MUST remain below 0.50.
#
# Functional advisors get a strict second-pass adjudication because they
# require an actual need/decision/gap, not merely the existence of a domain.
# Sector advisors may qualify from a material recurring sector portfolio.
# Arabic-only reasons remain enforced.
# ---------------------------------------------------------------------

RICH_V26_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V26_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V26_FUNCTIONAL_PROMPT = """
أنت محرك تقييم صارم لملاءمة المستشارين الوظيفيين في منظومة أثر.

لديك:
FACTS = وقائع موثقة عن الجمعية وبرامجها.
ROUTING_CARDS = بطاقات المستشارين الوظيفيين.

قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء.
لا يجوز حذف أي مستشار من التقييم.

المعنى الدقيق للدرجة:
- 90-100: احتياج أو قرار محوري ومثبت يملكه المستشار مباشرة.
- 80-89: احتياج مباشر قوي ومدعوم بأكثر من واقعة.
- 70-79: مساهمة وظيفية مادية وواضحة ومطلوبة الآن.
- 60-69: ملاءمة حقيقية ومثبتة لكن أقل مركزية.
- 50-59: توجد واقعة محددة تثبت حاجة فعلية لخدمة هذا المستشار الآن.
- 40-49: احتمال منطقي أو فرصة محتملة لكن الدليل غير كافٍ للترشيح.
- 20-39: صلة عامة بالمجال فقط.
- 0-19: غير مرتبط بالحالة الحالية.

قاعدة حاسمة:
الدرجة 50 ليست "ربما".
لكي يحصل المستشار الوظيفي على 50 أو أكثر يجب أن توجد واقعة محددة في FACTS تحقق شرط تفعيل حقيقي لهذا المستشار.

إذا كان تقييمك يتضمن أي معنى من الآتي:
- "حاجة محتملة"
- "قد يحتاج"
- "قد يشير"
- "ربما"
- "لا توجد أدلة واضحة"
- "مع أن الجمعية لديها..."
- "لمجرد وجود برامج أو موظفين أو أموال أو بيانات"
فلا يجوز أن تكون الدرجة 50 أو أكثر، ويجب أن تكون 49 أو أقل.

قواعد منع الإيجابيات الكاذبة:
- وجود حوكمة مرتفعة لا يفعّل مستشار الحوكمة، بل يخفض الحاجة إليه ما لم توجد فجوة حوكمة موثقة أو قرار حوكمي جديد.
- وجود رؤية ورسالة واضحة لا يفعّل مستشار الهوية.
- وجود أهداف استراتيجية واضحة لا يفعّل مستشار القضايا والأهداف إلا إذا توجد مشكلة أو إعادة صياغة أو تعارض أو مراجعة موثقة.
- وجود استراتيجية لا يفعّل مستشار التخطيط الاستراتيجي دون قرار استراتيجي أو مفاضلة أو تحديث حقيقي.
- وجود مؤشرات وأرقام لا يفعّل مستشار المؤشرات دون حاجة موثقة لبناء أو إصلاح منظومة قياس أداء.
- وجود بيانات لا يفعّل مستشار المتابعة والتقييم دون حاجة موثقة لقياس نتائج أو أثر أو تقييم.
- وجود برامج كثيرة لا يفعّل مستشار تصميم المبادرات إلا إذا كان هناك تصميم أو إعادة تصميم تدخل.
- وجود أموال لا يفعّل المستشار المالي دون موازنة أو سيولة أو تكلفة أو انحراف أو قرار مالي.
- وجود تمويل لا يفعّل تنمية الموارد دون فجوة تمويل أو تنويع موارد أو مانحين أو استدامة مالية موثقة.
- وجود موظفين لا يفعّل الموارد البشرية دون هيكل أو عبء أو أدوار أو مهارات أو أداء أو قوى عاملة تحتاج معالجة.
- وجود عمليات لا يفعّل مستشار العمليات دون تأخير أو هدر أو أخطاء أو إعادة تصميم أو أتمتة عملية موثقة.
- وجود قنوات تواصل لا يفعّل الاتصال المؤسسي دون مشكلة أو هدف اتصالي موثق.
- وجود حضور رقمي لا يفعّل التسويق الرقمي دون حملة أو تحويل أو اكتساب مطلوب.
- وجود تقنية لا يفعّل التحول الرقمي دون مشكلة رقمية أو أتمتة أو نظام أو تكامل أو حالة استخدام موثقة.

قواعد التفعيل الإيجابي:
- تعدد البرامج والمسارات بشكل مادي قد يفعّل مستشار المحافظ والبرامج والمشاريع عندما توجد حاجة فعلية للتنسيق أو الأولويات أو الترابط أو إدارة المحفظة.
- التشغيل الدوري أو الموسمي أو متعدد الجهات والموارد قد يفعّل مستشار التخطيط التشغيلي.
- شبكة شراكات جوهرية يعتمد عليها التنفيذ قد تفعّل مستشار أصحاب المصلحة والشراكات.
- قرار توسع أو تغيير أو استثمار أو إعادة تنظيم موثق يمكن أن يفعّل المستشار المختص به حسب ملكيته الفعلية.

حدود التخصص:
- لا تنسب للمستشار عملاً يخص مستشارًا آخر.
- إذا كان السبب الحقيقي يصف تخصص مستشار آخر، خفض الدرجة.
- لا تستخدم عدد المستشارين المطلوب كعامل في الدرجة.
- لا تخترع فجوة غير موجودة.

اللغة:
- REASON عربي فقط.
- SYSTEM_CODE يبقى كما هو.
- لا تستخدم أي حروف أجنبية داخل REASON.

الإخراج الإلزامي:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

تعليمات:
- سطر واحد لكل ROUTING_CARD بلا استثناء.
- SCORE من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط، من 1 إلى 4 معرفات، أو NONE.
- ACTIVATION جملة عربية قصيرة تحدد الواقعة التي فعّلت المستشار، أو "لا يوجد تفعيل كافٍ".
- REASON يشرح الدرجة دون اختلاق.
- إذا SCORE >= 50 فيجب أن تكون ACTIVATION واقعة محددة وليست احتمالًا.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V26_SECTOR_PROMPT = """
أنت محرك تقييم صارم لملاءمة المستشارين القطاعيين في منظومة أثر.

قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء.

المعنى الدقيق للدرجة:
- 90-100: القطاع جوهري جدًا ومتكرر في رسالة الجمعية ومحفظة خدماتها.
- 80-89: قطاع رئيسي وله عدة برامج أو خدمات واضحة.
- 70-79: قطاع مادي ومهم في عمل الجمعية.
- 60-69: صلة قطاعية حقيقية وواضحة لكن أقل مركزية.
- 50-59: صلة قطاعية فعلية ومثبتة وليست مجرد ذكر عابر.
- 40-49: نشاط محدود أو عابر أو غير كافٍ للترشيح.
- أقل من 40: غير مادي للحالة الحالية.

قاعدة حاسمة:
50 أو أكثر يعني أن القطاع حاضر فعليًا في رسالة الجمعية أو أهدافها أو عدة برامج أو خدمة جوهرية.
مجرد فعالية واحدة أو كلمة واحدة لا تكفي.

أمثلة:
- برامج صحية متكررة ورعاية صحية فعلية قد تفعّل مستشار الصحة.
- خدمات رعاية اجتماعية مستمرة قد تفعّل مستشار الخدمات الاجتماعية.
- برامج ثقافية وترفيهية متكررة قد تفعّل مستشار الثقافة والترفيه.
- هدف بحثي صريح أو برامج تعليمية مادية قد تفعّل مستشار التعليم والبحث.
- منظومة تطوع جوهرية ومتكررة قد تفعّل مستشار دعم العمل الخيري والتطوعي من زاوية التطوع وبناء القدرة.
- حقوق الفئة أو التوعية بالحقوق أو المناصرة كهدف فعلي قد تفعّل مستشار الحقوق والمناصرة.
- لا تستخدم الشراكات وحدها سببًا لتفعيل مستشار التطوع إذا لم يوجد تطوع مادي.

لا تستخدم عدد المستشارين المطلوب كعامل في الدرجة.
لا تخترع نشاطًا أو قطاعًا غير موجود.

اللغة:
- REASON عربي فقط.
- SYSTEM_CODE يبقى كما هو.
- لا تستخدم أي حروف أجنبية داخل REASON.

الإخراج الإلزامي:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

تعليمات:
- سطر واحد لكل ROUTING_CARD بلا استثناء.
- SCORE من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط، أو NONE.
- ACTIVATION جملة عربية قصيرة تحدد النشاط أو الهدف القطاعي المثبت، أو "لا يوجد تفعيل كافٍ".
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V26_FUNCTIONAL_REVIEW_PROMPT = """
أنت مراجع نهائي صارم لدرجات المستشارين الوظيفيين.

ستستلم:
FACTS
CANDIDATES
ROUTING_CARDS

كل مرشح في CANDIDATES حصل مبدئيًا على 50 أو أكثر.
راجعهم من جديد بصورة مستقلة.

الغرض من هذه الجولة هو منع استخدام 50 كدرجة "ربما".

اختبار الاحتفاظ:
احتفظ بدرجة 50 أو أكثر فقط إذا:
1) توجد واقعة موثقة ومحددة في FACTS.
2) الواقعة تحقق شرط تفعيل حقيقي للمستشار.
3) المساهمة المطلوبة تقع داخل ملكية المستشار.
4) السبب لا يعتمد على افتراض فجوة غير مذكورة.
5) مجرد وجود مجال ناجح أو قائم لا يعتبر احتياجًا استشاريًا.

قواعد خفض إلزامية:
- إذا كان السبب يقول أو يعني "حاجة محتملة"، "قد يحتاج"، "لا توجد أدلة واضحة"، "ربما"، أو يستنتج فجوة من مجرد وجود النشاط: اجعل الدرجة 49 أو أقل.
- إذا كانت الحوكمة مرتفعة ولا توجد فجوة موثقة: الحوكمة أقل من 50.
- إذا كانت الهوية واضحة ولا توجد مراجعة هوية: الهوية أقل من 50.
- إذا لا توجد مراجعة استراتيجية أو مفاضلة استراتيجية موثقة: التخطيط الاستراتيجي أقل من 50.
- إذا لا توجد حاجة موثقة لموازنة أو سيولة أو تكلفة أو انحراف: المالي أقل من 50.
- إذا لا توجد حاجة موثقة لتنمية الموارد أو تنويع الدخل: الاستدامة المالية وتنمية الموارد أقل من 50.
- إذا لا توجد حاجة موثقة للهيكل أو القوى العاملة أو الأدوار أو المهارات: الموارد البشرية أقل من 50.
- إذا لا توجد مشكلة عملية أو إعادة تصميم أو تحسين موثقة: العمليات أقل من 50.
- إذا لا توجد مشكلة أو هدف رقمي موثق: التحول الرقمي أقل من 50.
- إذا لا توجد حملة أو تحويل أو اكتساب موثق: التسويق الرقمي أقل من 50.

لا تخفض مستشارًا صحيحًا فقط لتقليل العدد.
لا يوجد حد أقصى لعدد النتائج.
الحد الوحيد هو الدليل والملاءمة.

أخرج لكل مرشح:
SYSTEM_CODE|FINAL_SCORE|EVIDENCE_IDS|REASON

- FINAL_SCORE من 0 إلى 100.
- REASON عربي فقط.
- لا تضف أي مستشار غير موجود في CANDIDATES.
- ممنوع JSON وMarkdown.
"""


def _v26_parse_scores(text, expected_codes, valid_fact_ids):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    rows = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        code_raw, score_raw, evidence_raw, activation_raw, reason_raw = [
            p.strip() for p in parts
        ]

        code = code_raw.strip()
        if code not in expected_codes or code in rows:
            continue

        m = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        evidence_ids = []
        if evidence_raw.upper() != "NONE":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)

        activation = re.sub(r"\s+", " ", activation_raw).strip()
        reason = re.sub(r"\s+", " ", reason_raw).strip()

        rows[code] = {
            "advisor_id": code,
            "score": round(score / 100.0, 4),
            "evidence_ids": evidence_ids[:4],
            "activation": activation,
            "reason": reason or "لا توجد ملاءمة كافية مدعومة بالوقائع الحالية.",
        }

    return rows


def _v26_score_pass(prompt, facts, advisors, advisor_class, valid_fact_ids):
    cards = _v25_cards(advisors, advisor_class, compact=False)

    expected_codes = {
        card["system_code"]
        for card in cards
        if card.get("system_code")
    }

    try:
        raw, tokens = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        if "input too long" not in str(exc).lower():
            raise

        cards = _v25_cards(advisors, advisor_class, compact=True)
        expected_codes = {
            card["system_code"]
            for card in cards
            if card.get("system_code")
        }

        raw, tokens = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

    parsed = _v26_parse_scores(
        raw,
        expected_codes,
        valid_fact_ids,
    )

    missing = expected_codes - set(parsed.keys())
    for code in missing:
        parsed[code] = {
            "advisor_id": code,
            "score": 0.0,
            "evidence_ids": [],
            "activation": "لا يوجد تقييم صالح.",
            "reason": "لم يقدم النموذج تقييمًا صالحًا لهذا المستشار في هذه الجولة.",
        }

    return parsed, cards


def _v26_parse_review(text, expected_codes, valid_fact_ids):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    reviewed = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 3)
        if len(parts) != 4:
            continue

        code_raw, score_raw, evidence_raw, reason_raw = [
            p.strip() for p in parts
        ]

        code = code_raw.strip()
        if code not in expected_codes or code in reviewed:
            continue

        m = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        evidence_ids = []
        if evidence_raw.upper() != "NONE":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        reviewed[code] = {
            "advisor_id": code,
            "score": round(score / 100.0, 4),
            "evidence_ids": evidence_ids[:4],
            "reason": reason or "لا توجد ملاءمة كافية مدعومة بالوقائع الحالية.",
        }

    return reviewed


def _v26_review_functional_candidates(
    facts,
    initial_scores,
    functional_cards,
    valid_fact_ids,
):
    candidates = [
        {
            "advisor_id": item["advisor_id"],
            "initial_score": item["score"],
            "evidence_ids": item.get("evidence_ids", []),
            "activation": item.get("activation", ""),
            "reason": item.get("reason", ""),
        }
        for item in initial_scores.values()
        if item["score"] >= RICH_V26_MIN_PUBLIC_SCORE
    ]

    if not candidates:
        return {}

    candidate_codes = {
        item["advisor_id"]
        for item in candidates
    }

    relevant_cards = [
        card
        for card in functional_cards
        if card.get("system_code") in candidate_codes
    ]

    raw, _ = _v18_generate_text(
        RICH_V26_FUNCTIONAL_REVIEW_PROMPT,
        {
            "facts": facts,
            "candidates": candidates,
            "routing_cards": relevant_cards,
        },
        RICH_V18_MATCH_MAX_NEW_TOKENS,
    )

    reviewed = _v26_parse_review(
        raw,
        candidate_codes,
        valid_fact_ids,
    )

    # If reviewer omitted a candidate, keep initial score only if the
    # initial activation/evidence is usable; otherwise demote safely.
    for code in candidate_codes:
        if code in reviewed:
            continue

        initial = initial_scores[code]
        activation = str(initial.get("activation", "")).strip()

        weak_markers = (
            "لا توجد",
            "محتمل",
            "قد ",
            "ربما",
            "لا يوجد",
        )

        if (
            not initial.get("evidence_ids")
            or any(marker in activation for marker in weak_markers)
        ):
            fallback_score = 0.49
        else:
            fallback_score = initial["score"]

        reviewed[code] = {
            "advisor_id": code,
            "score": fallback_score,
            "evidence_ids": initial.get("evidence_ids", []),
            "reason": initial.get("reason", ""),
        }

    return reviewed


def advisory_match_rich_v26(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v26 pass A: calibrated scoring for all 25 FUNCTIONAL advisors...",
        flush=True,
    )

    functional_scores, functional_cards = _v26_score_pass(
        RICH_V26_FUNCTIONAL_PROMPT,
        facts,
        advisors,
        "FUNCTIONAL",
        valid_fact_ids,
    )

    print(
        "Rich v26 pass B: calibrated scoring for all 10 SECTOR advisors...",
        flush=True,
    )

    sector_scores, _ = _v26_score_pass(
        RICH_V26_SECTOR_PROMPT,
        facts,
        advisors,
        "SECTOR",
        valid_fact_ids,
    )

    print(
        "Rich v26 pass C: strict review of functional candidates >= 0.50...",
        flush=True,
    )

    reviewed_functional = _v26_review_functional_candidates(
        facts,
        functional_scores,
        functional_cards,
        valid_fact_ids,
    )

    # Functional: use reviewed score for initial candidates; all other
    # functional advisors stay below threshold according to pass A.
    final_scores = {}

    for code, item in functional_scores.items():
        if code in reviewed_functional:
            final_scores[code] = reviewed_functional[code]
        else:
            final_scores[code] = {
                "advisor_id": code,
                "score": item["score"],
                "evidence_ids": item.get("evidence_ids", []),
                "reason": item.get("reason", ""),
            }

    # Sector: initial calibrated score is final.
    for code, item in sector_scores.items():
        final_scores[code] = {
            "advisor_id": code,
            "score": item["score"],
            "evidence_ids": item.get("evidence_ids", []),
            "reason": item.get("reason", ""),
        }

    matches = [
        item
        for item in final_scores.values()
        if (
            item["score"] >= RICH_V26_MIN_PUBLIC_SCORE
            and item.get("evidence_ids")
        )
    ]

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v26 returning {len(matches)} advisors with "
        f"final calibrated score >= {RICH_V26_MIN_PUBLIC_SCORE:.2f}.",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v27 — final grounded adjudication for all 35 advisors
#
# Goal:
# 1) Score all 35 advisors independently.
# 2) Run one final AI adjudication over ALL 35, not just initial >= 0.50.
# 3) Final KEEP requires a concrete activation fact.
# 4) Mere "could help / maybe / no clear evidence" is DROP.
# 5) Public output contains every FINAL KEEP advisor with score >= 0.50.
# 6) Final reasons are rewritten in clean Arabic.
# ---------------------------------------------------------------------

RICH_V27_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V27_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V27_FINAL_PROMPT = """
أنت الحكم النهائي لاختيار مستشاري منظومة أثر.

ستستلم:
FACTS = الوقائع الموثقة عن الجمعية وبرامجها.
INITIAL_SCORES = التقييم الأولي لكل المستشارين الخمسة والثلاثين.
ROUTING_CARDS = بطاقات مختصرة لكل المستشارين الخمسة والثلاثين.

مهمتك:
أعد تقييم جميع المستشارين الخمسة والثلاثين بصورة مستقلة، ثم اتخذ قرارًا نهائيًا لكل مستشار:
KEEP أو DROP.

لا تعتمد على الدرجة الأولية بوصفها حقيقة.
هي مجرد رأي أولي قابل للرفع أو الخفض.
القرار النهائي يجب أن يعتمد على FACTS ونطاق المستشار فقط.

المعيار النهائي للدرجة:
- 90-100: ارتباط مباشر ومحوري ومثبت بقوة.
- 80-89: ارتباط مباشر وقوي ومادي.
- 70-79: ارتباط واضح ومهم.
- 60-69: ارتباط حقيقي ومثبت لكنه أقل مركزية.
- 50-59: ارتباط حقيقي ومادي وله واقعة تفعيل محددة.
- 40-49: احتمال أو فائدة ممكنة لكن الدليل غير كافٍ للترشيح.
- أقل من 40: غير مرتبط بالحاجة الحالية.

قاعدة حاسمة:
الدرجة 50 أو أكثر لا تعني "قد يكون مفيدًا".
الدرجة 50 أو أكثر تعني أن هناك واقعة حالية محددة في FACTS تبرر إدخال هذا المستشار الآن.

========================
المستشارون الوظيفيون
========================
للمستشار الوظيفي:
لا يكفي أن يكون مجاله موجودًا في الجمعية.
يجب أن توجد حاجة أو قرار أو فجوة أو تعقيد حالي يقع داخل ملكيته.

أمثلة إلزامية:
- وجود حوكمة مرتفعة ليس سببًا لإدخال مستشار الحوكمة، بل العكس ما لم توجد فجوة أو متطلبات حوكمة جديدة.
- وجود رؤية ورسالة واضحة ليس سببًا لإدخال مستشار الهوية.
- وجود أهداف استراتيجية واضحة ليس سببًا لإدخال مستشار القضايا والأهداف.
- وجود استراتيجية ليس سببًا لإدخال مستشار التخطيط الاستراتيجي ما لم توجد مراجعة أو مفاضلة أو تحديث أو قرار استراتيجي حقيقي.
- وجود أرقام ليس سببًا لإدخال مستشار مؤشرات الأداء.
- وجود بيانات ليس سببًا لإدخال مستشار المتابعة والتقييم.
- وجود أموال أو تمويل ليس سببًا لإدخال المستشار المالي أو تنمية الموارد.
- وجود موظفين ليس سببًا لإدخال مستشار الموارد البشرية.
- وجود عمليات ليس سببًا لإدخال مستشار العمليات.
- وجود موقع أو قنوات رقمية ليس سببًا لإدخال مستشار التسويق أو التحول الرقمي.
- تعدد البرامج والمسارات بصورة كبيرة ومادية يمكن أن يبرر مستشار المحافظ والبرامج والمشاريع.
- التشغيل المتكرر أو الموسمي أو متعدد الموارد والشركاء يمكن أن يبرر مستشار التخطيط التشغيلي.
- الاعتماد الفعلي على شبكة شراكات متعددة لتنفيذ البرامج يمكن أن يبرر مستشار أصحاب المصلحة والشراكات.

إذا كان السبب الوحيد هو:
"الجمعية لديها برامج متعددة ولذلك تحتاج..."
فهذا غير كافٍ وحده لمعظم المستشارين الوظيفيين.

========================
المستشارون القطاعيون
========================
للمستشار القطاعي:
يمكن أن يكون KEEP إذا كان القطاع نفسه جوهريًا ومتكررًا في رسالة الجمعية أو أهدافها أو محفظة برامجها، حتى لو لم توجد "مشكلة" داخل القطاع.

لكن:
- نشاط عابر واحد لا يكفي.
- ذكر كلمة في الوصف لا يكفي.
- يجب أن يكون هناك حضور مادي ومتكرر أو هدف صريح وجوهري.

========================
منع الاختلاق
========================
ممنوع تمامًا اختراع:
- فجوة.
- مشكلة.
- ضعف.
- تحدٍ.
- حاجة تمويلية.
- حاجة تنظيمية.
- حاجة للتحول.
- حاجة للجودة.
- حاجة للبيانات.
إذا لم تظهر في FACTS.

إذا كتبت في السبب أي معنى مثل:
"لا توجد أدلة"
"لا توجد فجوة"
"قد يحتاج"
"حاجة محتملة"
"ربما"
"قد يشير"
فقرار المستشار يجب أن يكون DROP ودرجته أقل من 50.

========================
حدود التخصص
========================
كل مستشار يجب أن يبرر من خلال ملكيته الفعلية.
لا تنسب إدارة الشراكات لمستشار التطوع.
لا تنسب الحقوق لمستشار الخدمات الاجتماعية إذا كان السبب قانونيًا أو مناصرة.
لا تنسب إدارة المحفظة للمستشار التنفيذي إذا كان السبب يخص المحافظ والبرامج.
إذا كان السبب الحقيقي يخص مستشارًا آخر، خفض الدرجة أو اختر DROP.

========================
اللغة
========================
- السبب النهائي عربي سليم وواضح فقط.
- أصلح الأخطاء الإملائية والنحوية.
- ممنوع الحروف الإنجليزية أو الروسية أو الصينية أو أي أبجدية غير عربية داخل REASON.
- SYSTEM_CODE يبقى كما هو لأنه رمز تقني.
- لا تكتب اختصارات أجنبية داخل السبب.

========================
الإخراج
========================
أخرج سطرًا واحدًا لكل مستشار من الخمسة والثلاثين بلا استثناء:

SYSTEM_CODE|DECISION|FINAL_SCORE|EVIDENCE_IDS|REASON

DECISION:
KEEP
أو
DROP

FINAL_SCORE:
رقم من 0 إلى 100.

EVIDENCE_IDS:
من FACTS فقط، من 1 إلى 4 معرفات.
إذا لا يوجد دليل كافٍ اكتب NONE.

REASON:
- إذا KEEP: سبب عربي من 25 إلى 50 كلمة يذكر الواقعة الفعلية والقيمة المحددة التي يضيفها المستشار.
- إذا DROP: سبب عربي قصير يوضح لماذا لا يوجد تفعيل كافٍ.
- لا تخترع أي معلومة.
- ممنوع JSON.
- ممنوع Markdown.
- ممنوع أي شرح إضافي.

لا يوجد عدد مطلوب للمستشارين.
قد يكون العدد 4 أو 8 أو 15.
المعيار الوحيد هو الدليل والملاءمة الحقيقية.
"""


def _v27_compact_cards(advisors):
    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))
        advisor_class = "SECTOR" if advisor_num >= 26 else "FUNCTIONAL"

        cards.append({
            "system_code": advisor.get("system_code"),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "advisor_class": advisor_class,
            "owned_outcome": advisor.get("owned_outcome"),
            "owns": (advisor.get("owns") or [])[:5],
            "activation_when": (advisor.get("activation_when") or [])[:5],
            "not_primary_when": (advisor.get("not_primary_when") or [])[:2],
        })

    return cards


def _v27_parse_final(text, expected_codes, valid_fact_ids):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    rows = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        code_raw, decision_raw, score_raw, evidence_raw, reason_raw = [
            p.strip() for p in parts
        ]

        code = code_raw.strip()
        if code not in expected_codes or code in rows:
            continue

        decision = decision_raw.upper()
        if decision not in {"KEEP", "DROP"}:
            continue

        m = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        evidence_ids = []
        if evidence_raw.upper() != "NONE":
            for token in re.split(r"[,،;\s]+", evidence_raw):
                token = token.strip().upper()
                if token in valid_fact_ids and token not in evidence_ids:
                    evidence_ids.append(token)

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        rows[code] = {
            "advisor_id": code,
            "decision": decision,
            "score": round(score / 100.0, 4),
            "evidence_ids": evidence_ids[:4],
            "reason": reason,
        }

    return rows


def _v27_contradiction_guard(item):
    """
    Generic logical guard:
    A public KEEP >= 0.50 cannot simultaneously say there is no evidence,
    no gap, or only a possible need.
    """
    reason = str(item.get("reason", "")).strip()

    contradiction_markers = (
        "لا توجد أدلة",
        "لا يوجد دليل",
        "لا توجد فجوة",
        "لا توجد حاجة",
        "لا يوجد احتياج",
        "حاجة محتملة",
        "احتياج محتمل",
        "قد يحتاج",
        "قد تحتاج",
        "ربما",
        "قد يشير",
        "قد تشير",
    )

    return any(marker in reason for marker in contradiction_markers)


def advisory_match_rich_v27(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v27 pass A: initial scoring for all FUNCTIONAL advisors...",
        flush=True,
    )

    functional_scores, _ = _v26_score_pass(
        RICH_V26_FUNCTIONAL_PROMPT,
        facts,
        advisors,
        "FUNCTIONAL",
        valid_fact_ids,
    )

    print(
        "Rich v27 pass B: initial scoring for all SECTOR advisors...",
        flush=True,
    )

    sector_scores, _ = _v26_score_pass(
        RICH_V26_SECTOR_PROMPT,
        facts,
        advisors,
        "SECTOR",
        valid_fact_ids,
    )

    initial_scores = {}
    initial_scores.update(functional_scores)
    initial_scores.update(sector_scores)

    expected_codes = {
        advisor.get("system_code")
        for advisor in advisors
        if advisor.get("system_code")
    }

    initial_payload = [
        {
            "advisor_id": code,
            "initial_score": item.get("score", 0.0),
            "evidence_ids": item.get("evidence_ids", []),
            "activation": item.get("activation", ""),
            "reason": item.get("reason", ""),
        }
        for code, item in initial_scores.items()
    ]

    routing_cards = _v27_compact_cards(advisors)

    print(
        "Rich v27 pass C: final grounded adjudication for all 35 advisors...",
        flush=True,
    )

    try:
        final_raw, _ = _v18_generate_text(
            RICH_V27_FINAL_PROMPT,
            {
                "facts": facts,
                "initial_scores": initial_payload,
                "routing_cards": routing_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        if "input too long" not in str(exc).lower():
            raise

        # Retry with an even smaller card set while preserving every advisor.
        smaller_cards = [
            {
                "system_code": card["system_code"],
                "advisor_name": card["advisor_name"],
                "advisor_class": card["advisor_class"],
                "owned_outcome": card["owned_outcome"],
                "activation_when": card["activation_when"][:3],
            }
            for card in routing_cards
        ]

        final_raw, _ = _v18_generate_text(
            RICH_V27_FINAL_PROMPT,
            {
                "facts": facts,
                "initial_scores": initial_payload,
                "routing_cards": smaller_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

    final_rows = _v27_parse_final(
        final_raw,
        expected_codes,
        valid_fact_ids,
    )

    missing = expected_codes - set(final_rows.keys())
    if missing:
        print(
            f"Rich v27 warning: final adjudicator omitted "
            f"{len(missing)} advisor(s): {sorted(missing)}",
            flush=True,
        )

    matches = []

    for code, item in final_rows.items():
        if item["decision"] != "KEEP":
            continue

        if item["score"] < RICH_V27_MIN_PUBLIC_SCORE:
            continue

        if not item.get("evidence_ids"):
            continue

        if _v27_contradiction_guard(item):
            print(
                f"Rich v27 contradiction guard dropped {code}.",
                flush=True,
            )
            continue

        matches.append(item)

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Final foreign-script protection from v21.
    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v27 final pool: {len(matches)} advisors with "
        f"KEEP + score >= {RICH_V27_MIN_PUBLIC_SCORE:.2f}.",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v28 — three focused scoring panels
#
# Design:
# - Every advisor is still scored.
# - No global 35-advisor adjudication pass.
# - Advisors are scored in 3 focused panels so the model does not lose
#   specialist ownership boundaries:
#     Panel A: advisors 1-15  (leadership + strategy)
#     Panel B: advisors 16-25 (institutional functions + growth)
#     Panel C: advisors 26-35 (sector advisors)
# - Final public rule is unchanged: score >= 0.50 + grounded evidence.
# - 0.50 means CURRENT MATERIAL RELEVANCE, never "maybe useful".
# ---------------------------------------------------------------------

RICH_V28_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V28_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V28_PANEL_A_PROMPT = """
أنت محرك تقييم مستشاري القيادة والاستراتيجية في منظومة أثر.
قيّم كل مستشار في ROUTING_CARDS بلا استثناء.

تعريف الدرجة:
90-100 ارتباط مباشر ومحوري ومثبت.
80-89 ارتباط مباشر قوي.
70-79 ارتباط واضح ومادي.
60-69 ارتباط حقيقي ومثبت لكنه أقل مركزية.
50-59 احتياج حالي حقيقي له واقعة تفعيل محددة.
40-49 احتمال أو فائدة ممكنة فقط.
أقل من 40 لا توجد ملاءمة حالية كافية.

قاعدة حاسمة:
50 أو أكثر لا تعني "قد يكون مفيدًا".
يجب أن توجد واقعة حالية محددة في FACTS تحقق شرط تفعيل حقيقي للمستشار.

قواعد الملكية:
- المستشار التنفيذي لا يأخذ دور إدارة المحفظة أو التشغيل لمجرد كثرة البرامج.
- مستشار التحليل الداخلي والخارجي لا يُفعّل لمجرد وجود برامج أو شركاء؛ يجب وجود قرار أو حاجة لتحليل البيئة أو الفرص أو التهديدات أو العوامل الداخلية والخارجية.
- مستشار أصحاب المصلحة والشراكات يُفعّل إذا كان التنفيذ يعتمد فعليًا على شبكة شركاء متعددة أو توجد حاجة لإدارة القيمة والعلاقات والأدوار بين الشركاء.
- مستشار التغيير لا يُفعّل لمجرد إطلاق برامج؛ يجب وجود تحول أو انتقال أو تبنٍ أو مقاومة أو تغيير مؤسسي حقيقي.
- مستشار الجودة لا يُفعّل لمجرد وجود خدمات؛ يجب وجود حاجة جودة أو معايير أو عدم مطابقة أو تحسين خدمة موثق.
- مستشار المخاطر لا يُفعّل لمجرد وجود نشاط؛ يجب وجود مخاطر أو استمرارية أو تعرض موثق.
- مستشار التخطيط الاستراتيجي لا يُفعّل لمجرد وجود أهداف أو برامج؛ يجب وجود مراجعة أو مفاضلة أو تحديث أو قرار استراتيجي.
- مستشار الهوية لا يُفعّل إذا كانت الرؤية والرسالة والقيم واضحة ولا توجد حاجة لإعادة بنائها.
- مستشار القضايا والأهداف لا يُفعّل لمجرد وجود أهداف؛ يجب وجود مشكلة في القضايا أو الأولويات أو صياغة النتائج.
- مستشار تصميم المبادرات لا يُفعّل لمجرد وجود مبادرات قائمة؛ يجب وجود تصميم أو إعادة تصميم تدخل.
- مستشار التخطيط التشغيلي يمكن أن يُفعّل عند وجود تشغيل متكرر أو موسمي أو يومي/أسبوعي متعدد الموارد والشركاء ويحتاج تنسيقًا تشغيليًا.
- مستشار المحافظ والبرامج والمشاريع يمكن أن يُفعّل عندما توجد محفظة كبيرة ومتعددة المسارات تحتاج تنظيم الأولويات والترابط والموارد والمنافع.
- مستشار المؤشرات لا يُفعّل لمجرد وجود أرقام.
- مستشار المتابعة والتقييم والأثر لا يُفعّل لمجرد وجود بيانات؛ يجب وجود هدف قياس نتائج أو أثر أو تقييم فعلي.

إذا كان السبب يحتوي معنى:
"قد يحتاج" أو "حاجة محتملة" أو "ربما" أو "لا توجد أدلة واضحة"
فالدرجة يجب أن تكون أقل من 50.

ممنوع اختراع فجوة أو مشكلة غير موجودة في FACTS.

الإخراج:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

- سطر واحد لكل بطاقة.
- SCORE من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط أو NONE.
- ACTIVATION يجب أن تكون الواقعة الفعلية التي فعّلت المستشار، أو "لا يوجد تفعيل كافٍ".
- REASON عربي فقط.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V28_PANEL_B_PROMPT = """
أنت محرك تقييم مستشاري الوظائف المؤسسية والنمو في منظومة أثر.
قيّم كل مستشار في ROUTING_CARDS بلا استثناء.

تعريف الدرجة:
90-100 ارتباط مباشر ومحوري ومثبت.
80-89 ارتباط مباشر قوي.
70-79 ارتباط واضح ومادي.
60-69 ارتباط حقيقي ومثبت لكنه أقل مركزية.
50-59 احتياج حالي حقيقي له واقعة تفعيل محددة.
40-49 احتمال أو فائدة ممكنة فقط.
أقل من 40 لا توجد ملاءمة حالية كافية.

قاعدة حاسمة:
مجرد وجود المجال داخل أي جمعية لا يساوي احتياجًا استشاريًا.
50 أو أكثر يحتاج واقعة تفعيل محددة داخل FACTS.

قواعد صارمة:
- الحوكمة: ارتفاع درجة الحوكمة إنجاز وليس فجوة. لا تُفعّل المستشار إلا بمتطلب حوكمي أو امتثال أو صلاحيات أو سياسة أو فجوة موثقة.
- المالية والموازنات: وجود برامج أو مشروع ممول لا يكفي. يلزم موازنة أو تكلفة أو سيولة أو انحراف أو تدفق أو قرار مالي أو إعادة تخصيص موثق.
- الاستدامة المالية وتنمية الموارد: وجود تمويل لا يكفي. يلزم تنويع إيرادات أو فجوة تمويل أو اعتماد على ممول أو منح أو مانحين أو استراتيجية موارد موثقة.
- الأوقاف والاستثمار الاجتماعي: لا يُفعّل بلا وقف أو أصل استثماري أو استثمار اجتماعي أو قرار واضح في هذا المجال.
- الموارد البشرية والتصميم التنظيمي: المتطوعون ليسوا موظفين. كثرة المتطوعين لا تُفعّل الموارد البشرية وحدها. يلزم هيكل أو موظفون أو أدوار أو عبء عمل أو قوى عاملة أو جدارات أو أداء أو إعادة تنظيم موثق.
- العمليات: وجود برامج لا يكفي. يلزم إجراء أو تدفق خدمة أو تأخير أو اختناق أو هدر أو إعادة عمل أو تحسين عملية أو أتمتة عملية موثقة.
- الاتصال المؤسسي: وجود برامج لا يكفي. يلزم هدف أو مشكلة اتصال أو سمعة أو رسائل أو جمهور أو أزمة اتصال موثقة.
- التسويق الرقمي: لا يُفعّل بلا حملة أو اكتساب أو تحويل أو إعلان أو قمع رقمي أو هدف تسويقي محدد.
- التحول الرقمي والذكاء الاصطناعي: لا يُفعّل بلا مشكلة رقمية أو نظام أو أتمتة أو تكامل أو حالة استخدام أو قرار تقني موثق.
- البيانات والمعرفة والتقارير: وجود أرقام أو تقارير لا يكفي. يلزم مشكلة جودة بيانات أو مصدر حقيقة أو حوكمة بيانات أو معرفة حرجة أو تقارير تحتاج إعادة تصميم موثقة.

إذا كان السبب يحتوي معنى:
"لأن لديها برامج متعددة"
"قد تحتاج"
"حاجة محتملة"
"ربما"
"لا توجد أدلة"
من دون واقعة تفعيل تخصصية، فالدرجة يجب أن تكون أقل من 50.

ممنوع اختراع فجوة أو حاجة.

الإخراج:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

- سطر واحد لكل بطاقة.
- SCORE من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط أو NONE.
- ACTIVATION واقعة تفعيل تخصصية محددة أو "لا يوجد تفعيل كافٍ".
- REASON عربي فقط.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V28_PANEL_C_PROMPT = """
أنت محرك تقييم المستشارين القطاعيين في منظومة أثر.
قيّم كل مستشار في ROUTING_CARDS بلا استثناء.

تعريف الدرجة:
90-100 القطاع جوهري جدًا ومتكرر في رسالة الجمعية وبرامجها.
80-89 القطاع رئيسي وله عدة برامج أو خدمات واضحة.
70-79 القطاع مهم وله حضور مادي متكرر.
60-69 صلة قطاعية حقيقية وواضحة لكنها أقل مركزية.
50-59 صلة قطاعية فعلية ومثبتة وليست عابرة.
40-49 نشاط محدود أو جانبي لا يكفي للترشيح.
أقل من 40 غير مادي للحالة.

بالنسبة للمستشار القطاعي، لا يلزم وجود "مشكلة":
يكفي أن يكون القطاع نفسه جوهريًا ومتكررًا في رسالة الجمعية أو أهدافها أو محفظة برامجها.

لكن:
- نشاط واحد عابر لا يكفي.
- كلمة واحدة في وصف الجمعية لا تكفي.
- يجب أن توجد برامج متكررة أو هدف صريح أو خدمة جوهرية.

حدود مهمة:
- مستشار التطوع يُفعّل بسبب منظومة تطوع وبناء قدرات، لا بسبب الشراكات وحدها.
- مستشار الحقوق يُفعّل عند وجود حقوق أو مناصرة أو دعم قانوني أو توعية حقوقية جوهرية.
- مستشار التعليم والبحث يُفعّل عند وجود برامج تعليمية/بحثية مادية أو هدف بحثي وتعليمي صريح.
- مستشار البيئة لا يُفعّل بسبب "تحسين المشهد" إذا لم يكن هناك تدخل بيئي حقيقي.
- مستشار الإسكان لا يُفعّل بسبب جودة الحياة عامة دون تدخل سكني أو تنموي مكاني.
- مستشار الدعوة وخدمة ضيوف الرحمن لا يُفعّل بسبب التوعية العامة دون سياق ديني أو ضيوف الرحمن.
- مستشار الجمعيات المهنية لا يُفعّل إلا إذا كانت الجهة جمعية/رابطة مهنية أو لديها نموذج عضوية مهنية جوهري.

إذا كان السبب يقول "لا توجد برامج محددة" أو "غير واضح" فالدرجة يجب أن تكون أقل من 50.

الإخراج:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

- سطر واحد لكل بطاقة.
- SCORE من 0 إلى 100.
- EVIDENCE_IDS من FACTS فقط أو NONE.
- ACTIVATION يذكر البرنامج أو الهدف القطاعي المثبت.
- REASON عربي فقط.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""


def _v28_cards_for_range(advisors, start_id, end_id):
    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))
        if not (start_id <= advisor_num <= end_id):
            continue

        cards.append({
            "system_code": advisor.get("system_code"),
            "advisor_name": advisor.get(
                "name_ar",
                advisor.get("name_en"),
            ),
            "mission": advisor.get("mission"),
            "owned_outcome": advisor.get("owned_outcome"),
            "owns": (advisor.get("owns") or [])[:7],
            "activation_when": (advisor.get("activation_when") or [])[:7],
            "not_primary_when": (advisor.get("not_primary_when") or [])[:4],
            "boundaries": (advisor.get("boundaries") or [])[:3],
        })

    return cards


def _v28_score_panel(
    prompt,
    facts,
    advisors,
    start_id,
    end_id,
    valid_fact_ids,
):
    cards = _v28_cards_for_range(
        advisors,
        start_id,
        end_id,
    )

    expected_codes = {
        card["system_code"]
        for card in cards
        if card.get("system_code")
    }

    try:
        raw, _ = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        if "input too long" not in str(exc).lower():
            raise

        compact_cards = []
        for card in cards:
            compact_cards.append({
                "system_code": card["system_code"],
                "advisor_name": card["advisor_name"],
                "owned_outcome": card["owned_outcome"],
                "owns": card["owns"][:5],
                "activation_when": card["activation_when"][:5],
                "not_primary_when": card["not_primary_when"][:2],
            })

        raw, _ = _v18_generate_text(
            prompt,
            {
                "facts": facts,
                "routing_cards": compact_cards,
            },
            RICH_V18_MATCH_MAX_NEW_TOKENS,
        )

    parsed = _v26_parse_scores(
        raw,
        expected_codes,
        valid_fact_ids,
    )

    for code in expected_codes - set(parsed.keys()):
        parsed[code] = {
            "advisor_id": code,
            "score": 0.0,
            "evidence_ids": [],
            "activation": "لا يوجد تقييم صالح.",
            "reason": "لم يقدم النموذج تقييمًا صالحًا لهذا المستشار.",
        }

    return parsed


def _v28_has_contradiction(item):
    text = (
        str(item.get("activation", ""))
        + " "
        + str(item.get("reason", ""))
    )

    markers = (
        "لا توجد أدلة",
        "لا يوجد دليل",
        "لا توجد فجوة",
        "لا توجد حاجة",
        "لا يوجد احتياج",
        "حاجة محتملة",
        "احتياج محتمل",
        "قد يحتاج",
        "قد تحتاج",
        "ربما",
        "لا توجد برامج",
        "لا يوجد برنامج",
        "غير واضح",
    )

    return any(marker in text for marker in markers)


def advisory_match_rich_v28(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v28 panel A: scoring advisors 1-15...",
        flush=True,
    )
    panel_a = _v28_score_panel(
        RICH_V28_PANEL_A_PROMPT,
        facts,
        advisors,
        1,
        15,
        valid_fact_ids,
    )

    print(
        "Rich v28 panel B: scoring advisors 16-25...",
        flush=True,
    )
    panel_b = _v28_score_panel(
        RICH_V28_PANEL_B_PROMPT,
        facts,
        advisors,
        16,
        25,
        valid_fact_ids,
    )

    print(
        "Rich v28 panel C: scoring advisors 26-35...",
        flush=True,
    )
    panel_c = _v28_score_panel(
        RICH_V28_PANEL_C_PROMPT,
        facts,
        advisors,
        26,
        35,
        valid_fact_ids,
    )

    all_scores = {}
    all_scores.update(panel_a)
    all_scores.update(panel_b)
    all_scores.update(panel_c)

    matches = []

    for item in all_scores.values():
        if item["score"] < RICH_V28_MIN_PUBLIC_SCORE:
            continue

        if not item.get("evidence_ids"):
            continue

        if _v28_has_contradiction(item):
            print(
                f"Rich v28 contradiction guard dropped "
                f"{item['advisor_id']}.",
                flush=True,
            )
            continue

        matches.append(item)

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v28 final pool: {len(matches)} advisors with "
        f"score >= {RICH_V28_MIN_PUBLIC_SCORE:.2f}.",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v29 — evidence-entailment verification
#
# Core design:
# 1) Score every advisor first.
# 2) Recover any advisor row omitted by the model.
# 3) For every candidate >= 0.50, verify the ACTUAL cited fact text
#    against that advisor's activation conditions and ownership.
# 4) Functional and sector candidates are verified separately.
# 5) Only VERIFIED KEEP + final score >= 0.50 is public.
#
# This prevents "absence = need" errors such as:
# - high governance score -> governance advisor
# - no documented budget issue -> finance advisor
# - volunteers -> HR advisor
# - programs exist -> operations/data/communications advisor
# ---------------------------------------------------------------------

RICH_V29_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V29_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V29_RECOVERY_PROMPT = """
أنت محرك تقييم مستشار واحد أو عدة مستشارين لم يتم إخراج تقييمهم في الجولة السابقة.

قيّم كل ROUTING_CARD بلا استثناء.
استخدم FACTS فقط.
لا تخترع أي فجوة أو مشكلة أو حاجة.

أخرج:
SYSTEM_CODE|SCORE|EVIDENCE_IDS|ACTIVATION|REASON

SCORE من 0 إلى 100.
EVIDENCE_IDS من FACTS فقط أو NONE.
ACTIVATION واقعة تفعيل حقيقية أو "لا يوجد تفعيل كافٍ".
REASON عربي فقط.

قاعدة:
50 أو أكثر يعني ملاءمة حالية حقيقية ومثبتة.
"قد يحتاج" أو "ربما" أو "لا توجد أدلة" أو مجرد وجود المجال = أقل من 50.

ممنوع JSON وMarkdown.
"""

RICH_V29_FUNCTIONAL_VERIFY_PROMPT = """
أنت مراجع أدلة صارم للمستشارين الوظيفيين في منظومة أثر.

ستستلم CANDIDATES.
كل مرشح يحتوي على:
- بطاقة المستشار.
- الدرجة الأولية.
- سبب التقييم الأولي.
- واقعة التفعيل الأولية.
- الأدلة النصية الفعلية المقتبسة من FACTS.

مهمتك ليست البحث عن سبب جديد لإبقاء المستشار.
مهمتك اختبار:
هل الأدلة النصية الفعلية تثبت احتياجًا أو قرارًا أو تعقيدًا حاليًا يقع داخل ملكية هذا المستشار؟

اختبار KEEP:
KEEP فقط إذا تحققت الشروط الأربعة:
1) الدليل إيجابي وموجود فعليًا، وليس غياب معلومة.
2) الدليل يحقق شرط تفعيل للمستشار، وليس مجرد أن مجال المستشار موجود في الجمعية.
3) المساهمة تقع داخل OWNERSHIP الحقيقي للمستشار.
4) الملاءمة مادية بما يكفي للحصول على 50 أو أكثر الآن.

اختبار DROP:
DROP إذا كان الاستدلال من نوع:
- لم يذكر وجود شيء، إذن نحتاج مستشارًا له.
- لا توجد خطة/سياسة/تحليل، إذن نحتاج المستشار.
- الجمعية لديها برامج، إذن تحتاج مالية/عمليات/بيانات/اتصال/موارد بشرية.
- الجمعية لديها متطوعون، إذن تحتاج موارد بشرية.
- الجمعية حققت حوكمة مرتفعة، إذن تحتاج حوكمة.
- الجمعية لديها رسالة وهوية واضحة، إذن تحتاج هوية.
- الجمعية لديها أهداف، إذن تحتاج قضايا وأهداف.
- الجمعية لديها أرقام، إذن تحتاج مؤشرات أو بيانات.
- الجمعية لديها تمويل، إذن تحتاج تنمية موارد.
- مجرد "قد يكون مفيدًا" أو "حاجة محتملة".

استثناءات مشروعة عندما يكون التعقيد نفسه واقعة تفعيل:
- محفظة كبيرة ومتعددة المسارات يمكن أن تفعل مستشار المحافظ والبرامج والمشاريع.
- تشغيل دوري/يومي/أسبوعي/موسمي متعدد الموارد أو الشركاء يمكن أن يفعل التخطيط التشغيلي.
- اعتماد التنفيذ على شبكة شركاء متعددة يمكن أن يفعل أصحاب المصلحة والشراكات.
- قرار توسع أو تحول أو إعادة تنظيم موثق يمكن أن يفعل المستشار المتخصص به.

قواعد تخصصية صارمة:
- المالية تحتاج دليلًا على موازنة/تكلفة/سيولة/تدفقات/انحراف/قرار مالي، لا مجرد مشروع أو برنامج.
- تنمية الموارد تحتاج فجوة تمويل/تنويع دخل/مانحين/منح/اعتماد تمويلي، لا مجرد وجود شراكات.
- الموارد البشرية تحتاج موظفين/هيكل/أدوار/عبء/قوى عاملة/جدارات/أداء، لا المتطوعين وحدهم.
- العمليات تحتاج إجراء/تدفق/اختناق/تأخير/هدر/إعادة عمل/تحسين خدمة أو أتمتة عملية، لا مجرد وجود برامج.
- البيانات تحتاج مشكلة جودة/ملكية/مصدر حقيقة/وصول/معرفة حرجة/تقارير، لا مجرد أرقام.
- الاتصال يحتاج هدف أو مشكلة اتصال/سمعة/رسائل/جمهور، لا مجرد برامج.
- الحوكمة تحتاج فجوة أو التزام أو صلاحيات أو سياسة أو قرار حوكمي. الإنجاز المرتفع ليس احتياجًا.
- التحليل الداخلي والخارجي يحتاج قرارًا أو حاجة لتحليل البيئة، لا مجرد وجود شركاء وبرامج.
- القيادة التنفيذية تحتاج قرارًا تنفيذيًا مؤسسيًا فعليًا يتجاوز ملكية المستشارين المتخصصين.

أعد تقييم الدرجة:
- KEEP: FINAL_SCORE من 50 إلى 100.
- DROP: FINAL_SCORE من 0 إلى 49.

السبب النهائي:
- عربي مهني واضح.
- لا تستخدم حروفًا أجنبية.
- لا تخترع أي معلومة.
- إذا KEEP، اذكر الواقعة التي تثبت التفعيل وما الذي سيضيفه المستشار تحديدًا.
- إذا DROP، اذكر باختصار لماذا الأدلة لا تثبت احتياجًا حاليًا.

الإخراج:
SYSTEM_CODE|DECISION|FINAL_SCORE|REASON

سطر لكل مرشح بلا استثناء.
ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V29_SECTOR_VERIFY_PROMPT = """
أنت مراجع أدلة صارم للمستشارين القطاعيين في منظومة أثر.

ستستلم مرشحين قطاعيين مع:
- بطاقة المستشار.
- الدرجة الأولية.
- الأدلة النصية الفعلية.

المستشار القطاعي لا يحتاج وجود "مشكلة".
يمكن KEEP إذا كان القطاع نفسه جوهريًا ومتكررًا في:
- رسالة الجمعية،
- أهدافها الصريحة،
- أو محفظة برامجها وخدماتها.

KEEP إذا:
1) توجد أكثر من إشارة مادية أو برنامج/هدف جوهري للقطاع، أو خدمة أساسية واضحة.
2) الخبرة القطاعية للمستشار تضيف قيمة مباشرة للبرامج الحالية.
3) السبب يقع فعلًا داخل تخصص المستشار.

DROP إذا:
- الارتباط مجرد كلمة عابرة أو فعالية وحيدة غير جوهرية.
- السبب الحقيقي يخص مستشارًا قطاعيًا آخر.
- لا توجد برامج أو أهداف مادية في القطاع.
- الاستدلال يعتمد على احتمالات غير موجودة.

حدود مهمة:
- التطوع: يحتاج منظومة تطوع أو فرص ومتطوعين ماديين، وليس الشراكات وحدها.
- الحقوق: هدف صريح بحقوق الفئة أو مناصرة/توعية حقوقية يمكن أن يكون تفعيلًا ماديًا حتى دون خدمة قانونية كاملة.
- التعليم والبحث: هدف بحثي صريح أو برامج تعليمية/توعوية مادية يمكن أن يفعّله.
- الثقافة والترفيه: برامج متكررة ثقافية/ترفيهية أو اجتماعية ذات مضمون ثقافي واضح.
- البيئة: لا يكفي تحسين المشهد العام إذا لم يوجد تدخل بيئي حقيقي.
- الإسكان: لا تكفي جودة الحياة العامة دون تدخل سكني/تنموي مكاني.
- الدعوة وضيوف الرحمن: لا تكفي التوعية العامة دون سياق ديني أو خدمة ضيوف الرحمن.
- الجمعيات المهنية: يجب أن تكون الجهة جمعية/رابطة مهنية أو نموذج عضوية مهنية جوهري.

أعد تقييم الدرجة:
KEEP = 50 إلى 100.
DROP = أقل من 50.

السبب النهائي:
عربي مهني واضح فقط، دون حروف أجنبية أو معلومات مخترعة.

الإخراج:
SYSTEM_CODE|DECISION|FINAL_SCORE|REASON

سطر لكل مرشح بلا استثناء.
ممنوع JSON وMarkdown وأي شرح إضافي.
"""


def _v29_card(advisor):
    return {
        "system_code": advisor.get("system_code"),
        "advisor_name": advisor.get(
            "name_ar",
            advisor.get("name_en"),
        ),
        "advisor_class": (
            "SECTOR"
            if int(advisor.get("advisor_id")) >= 26
            else "FUNCTIONAL"
        ),
        "mission": advisor.get("mission"),
        "owned_outcome": advisor.get("owned_outcome"),
        "owns": (advisor.get("owns") or [])[:7],
        "activation_when": (advisor.get("activation_when") or [])[:7],
        "not_primary_when": (advisor.get("not_primary_when") or [])[:4],
        "boundaries": (advisor.get("boundaries") or [])[:3],
    }


def _v29_fact_map(facts):
    return {
        fact["fact_id"]: {
            "source": fact.get("source"),
            "text": fact.get("text"),
        }
        for fact in facts
    }


def _v29_recover_missing(
    prompt,
    facts,
    parsed,
    expected_codes,
    cards_by_code,
    valid_fact_ids,
):
    missing = expected_codes - set(parsed.keys())

    if not missing:
        return parsed

    recovery_cards = [
        cards_by_code[code]
        for code in sorted(missing)
        if code in cards_by_code
    ]

    if not recovery_cards:
        return parsed

    print(
        f"Rich v29 recovering {len(recovery_cards)} omitted advisor score rows...",
        flush=True,
    )

    raw, _ = _v18_generate_text(
        RICH_V29_RECOVERY_PROMPT,
        {
            "facts": facts,
            "routing_cards": recovery_cards,
        },
        min(RICH_V18_MATCH_MAX_NEW_TOKENS, 1600),
    )

    recovered = _v26_parse_scores(
        raw,
        set(missing),
        valid_fact_ids,
    )

    parsed.update(recovered)

    for code in missing - set(recovered.keys()):
        parsed[code] = {
            "advisor_id": code,
            "score": 0.0,
            "evidence_ids": [],
            "activation": "لا يوجد تقييم صالح.",
            "reason": "لم يقدم النموذج تقييمًا صالحًا لهذا المستشار.",
        }

    return parsed


def _v29_initial_score_all(
    facts,
    advisors,
    valid_fact_ids,
):
    # Use calibrated v26 scoring semantics for the initial broad pass.
    functional_cards = _v25_cards(
        advisors,
        "FUNCTIONAL",
        compact=False,
    )
    sector_cards = _v25_cards(
        advisors,
        "SECTOR",
        compact=False,
    )

    functional_codes = {
        c["system_code"] for c in functional_cards
    }
    sector_codes = {
        c["system_code"] for c in sector_cards
    }

    functional_raw, _ = _v18_generate_text(
        RICH_V26_FUNCTIONAL_PROMPT,
        {
            "facts": facts,
            "routing_cards": functional_cards,
        },
        RICH_V18_MATCH_MAX_NEW_TOKENS,
    )

    sector_raw, _ = _v18_generate_text(
        RICH_V26_SECTOR_PROMPT,
        {
            "facts": facts,
            "routing_cards": sector_cards,
        },
        RICH_V18_MATCH_MAX_NEW_TOKENS,
    )

    functional_scores = _v26_parse_scores(
        functional_raw,
        functional_codes,
        valid_fact_ids,
    )
    sector_scores = _v26_parse_scores(
        sector_raw,
        sector_codes,
        valid_fact_ids,
    )

    cards_by_code = {}
    for advisor in advisors:
        card = _v29_card(advisor)
        if card.get("system_code"):
            cards_by_code[card["system_code"]] = card

    functional_scores = _v29_recover_missing(
        RICH_V26_FUNCTIONAL_PROMPT,
        facts,
        functional_scores,
        functional_codes,
        cards_by_code,
        valid_fact_ids,
    )

    sector_scores = _v29_recover_missing(
        RICH_V26_SECTOR_PROMPT,
        facts,
        sector_scores,
        sector_codes,
        cards_by_code,
        valid_fact_ids,
    )

    return functional_scores, sector_scores, cards_by_code


def _v29_candidate_packages(
    score_rows,
    cards_by_code,
    fact_map,
):
    packages = []

    for code, item in score_rows.items():
        if item.get("score", 0.0) < RICH_V29_MIN_PUBLIC_SCORE:
            continue

        evidence = []
        for fid in item.get("evidence_ids", []):
            if fid in fact_map:
                evidence.append({
                    "fact_id": fid,
                    "source": fact_map[fid].get("source"),
                    "text": fact_map[fid].get("text"),
                })

        packages.append({
            "system_code": code,
            "initial_score": item.get("score"),
            "initial_activation": item.get("activation"),
            "initial_reason": item.get("reason"),
            "evidence": evidence,
            "routing_card": cards_by_code.get(code, {}),
        })

    return packages


def _v29_parse_verification(text, expected_codes):
    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    rows = {}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 3)
        if len(parts) != 4:
            continue

        code_raw, decision_raw, score_raw, reason_raw = [
            p.strip() for p in parts
        ]

        code = code_raw.strip()
        if code not in expected_codes or code in rows:
            continue

        decision = decision_raw.upper()
        if decision not in {"KEEP", "DROP"}:
            continue

        m = re.search(r"\d+(?:\.\d+)?", score_raw)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        reason = re.sub(r"\s+", " ", reason_raw).strip()

        rows[code] = {
            "advisor_id": code,
            "decision": decision,
            "score": round(score / 100.0, 4),
            "reason": reason,
        }

    return rows


def _v29_verify_candidates(prompt, packages):
    if not packages:
        return {}

    expected_codes = {
        item["system_code"]
        for item in packages
    }

    raw, _ = _v18_generate_text(
        prompt,
        {
            "candidates": packages,
        },
        min(RICH_V18_MATCH_MAX_NEW_TOKENS, 2200),
    )

    verified = _v29_parse_verification(
        raw,
        expected_codes,
    )

    # Omitted verification rows are rejected, never silently kept.
    for code in expected_codes - set(verified.keys()):
        verified[code] = {
            "advisor_id": code,
            "decision": "DROP",
            "score": 0.0,
            "reason": "لم يثبت التحقق النهائي وجود تفعيل حالي كافٍ لهذا المستشار.",
        }

    return verified


def _v29_reason_contradiction(reason):
    text = str(reason or "")

    markers = (
        "لا توجد أدلة",
        "لا يوجد دليل",
        "لا توجد فجوة",
        "لا توجد حاجة",
        "لا يوجد احتياج",
        "حاجة محتملة",
        "احتياج محتمل",
        "قد يحتاج",
        "قد تحتاج",
        "ربما",
        "لم يذكر",
        "لم يُذكر",
        "لم يتم توثيق",
        "لم يتضح",
        "إمكانية",
    )

    return any(marker in text for marker in markers)


def advisory_match_rich_v29(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = build_rich_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    fact_map = _v29_fact_map(facts)
    advisors = _RICH_REGISTRY["advisors"]

    print(
        "Rich v29 pass A/B: scoring all 35 advisors...",
        flush=True,
    )

    functional_scores, sector_scores, cards_by_code = _v29_initial_score_all(
        facts,
        advisors,
        valid_fact_ids,
    )

    functional_packages = _v29_candidate_packages(
        functional_scores,
        cards_by_code,
        fact_map,
    )

    sector_packages = _v29_candidate_packages(
        sector_scores,
        cards_by_code,
        fact_map,
    )

    print(
        f"Rich v29 pass C: verifying {len(functional_packages)} "
        "functional candidates against their exact evidence...",
        flush=True,
    )

    functional_verified = _v29_verify_candidates(
        RICH_V29_FUNCTIONAL_VERIFY_PROMPT,
        functional_packages,
    )

    print(
        f"Rich v29 pass D: verifying {len(sector_packages)} "
        "sector candidates against their exact evidence...",
        flush=True,
    )

    sector_verified = _v29_verify_candidates(
        RICH_V29_SECTOR_VERIFY_PROMPT,
        sector_packages,
    )

    final_rows = {}
    final_rows.update(functional_verified)
    final_rows.update(sector_verified)

    matches = []

    for item in final_rows.values():
        if item.get("decision") != "KEEP":
            continue

        if item.get("score", 0.0) < RICH_V29_MIN_PUBLIC_SCORE:
            continue

        if _v29_reason_contradiction(item.get("reason")):
            print(
                f"Rich v29 contradiction guard dropped {item['advisor_id']}.",
                flush=True,
            )
            continue

        matches.append(item)

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Foreign-script safety guard.
    matches = _v21_repair_reasons(matches)

    print(
        f"Rich v29 final pool: {len(matches)} verified advisors "
        f"with final score >= {RICH_V29_MIN_PUBLIC_SCORE:.2f}.",
        flush=True,
    )

    return {
        "ranked": [
            {
                "advisor_id": item["advisor_id"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in matches
        ]
    }


# ---------------------------------------------------------------------
# Rich AI Router v30 — production fast + grounded + clean Arabic
#
# Production architecture:
#   1) Build compact, granular Arabic facts.
#   2) TWO model calls only:
#        - advisors 1-25
#        - advisors 26-35
#   3) Model outputs ONLY: SYSTEM_CODE|SCORE|EVIDENCE_IDS
#      (no generated public prose).
#   4) Python validates the cited evidence against advisor-specific
#      activation signals and boundaries.
#   5) Python writes the public Arabic reason deterministically from
#      the source facts. This eliminates garbled generated Arabic such
#      as "براغم", broken words, Cyrillic/CJK contamination, etc.
#   6) Return every genuinely grounded advisor with score >= 0.50.
#
# Normal path = 2 generations total.
# A tiny recovery generation runs ONLY if the model omits advisor rows.
# ---------------------------------------------------------------------

RICH_V30_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V30_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V30_FUNCTIONAL_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V30_FUNCTIONAL_MAX_NEW_TOKENS", "700")
)

RICH_V30_SECTOR_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V30_SECTOR_MAX_NEW_TOKENS", "380")
)

RICH_V30_RECOVERY_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V30_RECOVERY_MAX_NEW_TOKENS", "280")
)


def _v30_normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _v30_trim(text, limit):
    text = _v30_normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + "…"


def _v30_arabic_public_text(text):
    """
    Public prose safety:
    - preserve Arabic-script letters, numbers, whitespace and punctuation.
    - remove all Latin/Cyrillic/CJK/etc. letters.
    - normalize spacing.
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    out = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith("L"):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = ""

            if "ARABIC" in name:
                out.append(ch)
            else:
                out.append(" ")
            continue

        if category.startswith("M"):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = ""
            if "ARABIC" in name:
                out.append(ch)
            continue

        if (
            category.startswith("N")
            or category.startswith("P")
            or category.startswith("Z")
        ):
            out.append(ch)
            continue

        if ch in {"٪", "﷼"}:
            out.append(ch)
        else:
            out.append(" ")

    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    cleaned = re.sub(r"\s+([،؛:,.!?؟])", r"\1", cleaned)
    return cleaned


def _v30_program_type_ar(value):
    value = str(value or "").strip().lower()
    if value == "project":
        return "مشروع"
    return "برنامج"


def _v30_build_facts(organization, programs):
    """
    Granular facts make evidence citation easier and reduce hallucinated
    "missing information = need" reasoning.
    """
    facts = []

    def add(fid, source, text):
        text = _v30_normalize_text(text)
        if text:
            facts.append({
                "fact_id": fid,
                "source": source,
                "text": text,
            })

    add("O1", "اسم الجمعية", organization.get("name"))
    add("O2", "نوع الجمعية", organization.get("type"))
    add("O3", "القطاع", organization.get("sector"))

    activity_fields = organization.get("activity_fields") or []
    if isinstance(activity_fields, list) and activity_fields:
        add(
            "O4",
            "مجالات النشاط",
            "مجالات النشاط: " + "، ".join(
                _v30_normalize_text(x)
                for x in activity_fields
                if _v30_normalize_text(x)
            ),
        )

    add(
        "O5",
        "الوصف المختصر",
        _v30_trim(organization.get("short_description"), 700),
    )

    add(
        "O6",
        "الوصف التفصيلي",
        _v30_trim(organization.get("detailed_description"), 1400),
    )

    add(
        "O7",
        "الميزة المؤسسية",
        _v30_trim(organization.get("competitive_advantage"), 650),
    )

    # Rich v34 recovery hardening:
    # Keep decision-relevant structured context from the backend payload.
    # Earlier versions ignored these fields, which could hide explicit needs
    # such as impact measurement or funding concentration from the router.
    context_fields = (
        ("O8", "السياق الاستراتيجي", "strategic_context", 900),
        ("O9", "سياق الحوكمة", "governance_context", 900),
        ("O10", "السياق المالي", "financial_context", 900),
        ("O11", "سياق الأداء والأثر", "performance_context", 1000),
    )

    for fid, source, key, limit in context_fields:
        add(
            fid,
            source,
            _v30_trim(organization.get(key), limit),
        )

    current_challenges = organization.get("current_challenges") or []
    if isinstance(current_challenges, list):
        for idx, challenge in enumerate(current_challenges, start=1):
            add(
                f"H{idx}",
                f"تحدٍ حالي {idx}",
                _v30_trim(challenge, 500),
            )

    important_notes = organization.get("important_notes") or []
    if isinstance(important_notes, list):
        for idx, note in enumerate(important_notes, start=1):
            add(
                f"N{idx}",
                f"ملاحظة مؤسسية {idx}",
                _v30_trim(note, 420),
            )

    for idx, program in enumerate(programs, start=1):
        if not isinstance(program, dict):
            continue

        name = _v30_normalize_text(program.get("name"))
        ptype = _v30_program_type_ar(program.get("type"))
        description = _v30_trim(program.get("description"), 520)
        audience = _v30_trim(program.get("target_audience"), 220)
        value = _v30_trim(program.get("beneficiary_value"), 260)
        delivery = _v30_trim(program.get("delivery_method"), 260)
        status = _v30_trim(program.get("status"), 120)
        notes = _v30_trim(program.get("notes"), 520)

        pieces = []
        if name:
            pieces.append(f'{ptype} "{name}"')
        if description:
            pieces.append(f"الوصف: {description}")
        if audience:
            pieces.append(f"الفئة المستهدفة: {audience}")
        if value:
            pieces.append(f"القيمة للمستفيد: {value}")
        if delivery:
            pieces.append(f"طريقة التنفيذ: {delivery}")
        if status:
            pieces.append(f"الحالة: {status}")
        if notes:
            pieces.append(f"ملاحظات: {notes}")

        if pieces:
            add(
                f"P{idx}",
                f"البرنامج أو المشروع {idx}",
                "؛ ".join(pieces),
            )

    return facts


def _v30_cards(advisors, start_id, end_id):
    cards = []

    for advisor in advisors:
        advisor_num = int(advisor.get("advisor_id"))

        if not (start_id <= advisor_num <= end_id):
            continue

        cards.append({
            "system_code": advisor.get("system_code"),
            "name": advisor.get("name_ar"),
            "mission": _v30_trim(advisor.get("mission"), 250),
            "owned_outcome": _v30_trim(
                advisor.get("owned_outcome"),
                280,
            ),
            "activation_when": [
                _v30_trim(x, 140)
                for x in (advisor.get("activation_when") or [])[:6]
            ],
            "not_primary_when": [
                _v30_trim(x, 140)
                for x in (advisor.get("not_primary_when") or [])[:3]
            ],
        })

    return cards


RICH_V30_FUNCTIONAL_PROMPT = """
أنت محرك تقييم ملاءمة المستشارين الوظيفيين في منظومة أثر.

المطلوب:
قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء، اعتمادًا على FACTS فقط.

الدرجة:
90-100 = ارتباط مباشر ومحوري مثبت.
80-89 = ارتباط مباشر قوي.
70-79 = ارتباط واضح ومادي.
60-69 = ارتباط حقيقي ومثبت لكنه أقل مركزية.
50-59 = احتياج حالي حقيقي تدعمه واقعة تفعيل محددة.
40-49 = احتمال أو فائدة ممكنة فقط، غير كافٍ للترشيح.
أقل من 40 = غير مرتبط حاليًا.

قاعدة 50:
لا تمنح 50 أو أكثر لمجرد أن مجال المستشار موجود في الجمعية.
يجب أن توجد واقعة تفعيل إيجابية فعلية.

ممنوع الاستدلال بالغياب:
- "لم يذكر وجود خطة" لا يعني الحاجة إلى مستشار الخطة.
- "لا توجد معلومات مالية" لا يعني الحاجة إلى المستشار المالي.
- "لا يوجد تحليل" لا يعني الحاجة إلى مستشار التحليل.
- الإنجاز المرتفع لا يعني وجود فجوة؛ درجة حوكمة مرتفعة مثلًا ليست سببًا وحدها لاختيار مستشار الحوكمة.

قواعد حدود الملكية:
- كثرة البرامج قد تبرر مستشار المحافظ والبرامج والمشاريع إذا كان التعقيد فعليًا.
- التشغيل الدوري أو اليومي أو الأسبوعي أو الموسمي متعدد الموارد والشركاء قد يبرر التخطيط التشغيلي.
- الاعتماد على شبكة شراكات متعددة في التنفيذ قد يبرر أصحاب المصلحة والشراكات.
- المتطوعون وحدهم لا يبررون مستشار الموارد البشرية.
- وجود برامج وحده لا يبرر المالية أو العمليات أو البيانات أو الاتصال أو تنمية الموارد.
- وجود رؤية ورسالة واضحة لا يبرر مستشار الهوية.
- وجود أهداف واضحة لا يبرر مستشار القضايا والأهداف.
- لا تخترع فجوة أو مشكلة أو قرارًا غير موجود في FACTS.

الإخراج فقط:
SYSTEM_CODE|SCORE|EVIDENCE_IDS

مثال شكلي:
AOS-XX-00|72|N3,P2

قواعد الإخراج:
- سطر واحد لكل ROUTING_CARD بنفس عدد البطاقات.
- SCORE من 0 إلى 100.
- إذا SCORE >= 50 يجب ذكر من 1 إلى 3 EVIDENCE_IDS حقيقية من FACTS.
- إذا SCORE < 50 يمكن كتابة NONE.
- ممنوع كتابة الأسباب.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""

RICH_V30_SECTOR_PROMPT = """
أنت محرك تقييم المستشارين القطاعيين في منظومة أثر.

قيّم كل مستشار موجود في ROUTING_CARDS بلا استثناء اعتمادًا على FACTS فقط.

للمستشار القطاعي:
لا يشترط وجود مشكلة أو فجوة.
يمكن أن يكون مناسبًا إذا كان القطاع نفسه جوهريًا ومتكررًا في رسالة الجمعية أو أهدافها أو برامجها وخدماتها.

الدرجة:
90-100 = القطاع محوري جدًا ومتكرر.
80-89 = قطاع رئيسي وله عدة برامج أو خدمات واضحة.
70-79 = قطاع مهم وله حضور مادي.
60-69 = صلة قطاعية حقيقية وواضحة.
50-59 = صلة فعلية مثبتة لكنها أقل مركزية.
40-49 = نشاط جانبي أو عابر.
أقل من 40 = غير مادي.

قواعد:
- فعالية عابرة واحدة لا تكفي وحدها.
- لا تخترع قطاعًا غير موجود.
- التطوع يحتاج منظومة تطوع أو متطوعين وفرصًا حقيقية.
- الحقوق تحتاج هدفًا أو نشاطًا فعليًا متعلقًا بالحقوق أو المناصرة أو الدعم القانوني.
- البيئة تحتاج تدخلًا بيئيًا حقيقيًا.
- الإسكان يحتاج تدخلًا سكنيًا أو تنمويًا مكانيًا.
- الدعوة وضيوف الرحمن تحتاج سياقًا دينيًا أو خدمة حجاج/معتمرين/زوار.
- الجمعيات المهنية تحتاج نموذج جمعية أو رابطة مهنية وعضوية مهنية.

الإخراج فقط:
SYSTEM_CODE|SCORE|EVIDENCE_IDS

- سطر واحد لكل ROUTING_CARD.
- SCORE من 0 إلى 100.
- إذا SCORE >= 50 اذكر من 1 إلى 3 EVIDENCE_IDS حقيقية.
- إذا SCORE < 50 يمكن كتابة NONE.
- ممنوع الأسباب.
- ممنوع JSON وMarkdown وأي شرح إضافي.
"""


def _v30_digits_to_ascii(text):
    table = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    )
    return str(text).translate(table)


def _v30_parse_scores(text, expected_codes, valid_fact_ids):
    rows = {}

    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 2)
        if len(parts) != 3:
            continue

        code = parts[0].strip()
        if code not in expected_codes or code in rows:
            continue

        score_text = _v30_digits_to_ascii(parts[1])
        m = re.search(r"\d+(?:\.\d+)?", score_text)
        if not m:
            continue

        score = float(m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))

        evidence_ids = []
        evidence_text = _v30_digits_to_ascii(parts[2]).upper()

        if evidence_text != "NONE":
            for token in re.findall(
                r"\b(?:O|N|P)\d+\b",
                evidence_text,
            ):
                if (
                    token in valid_fact_ids
                    and token not in evidence_ids
                ):
                    evidence_ids.append(token)

        rows[code] = {
            "advisor_id": code,
            "score": round(score / 100.0, 4),
            "evidence_ids": evidence_ids[:3],
        }

    return rows


def _v30_recover_missing(
    facts,
    missing_cards,
    valid_fact_ids,
):
    if not missing_cards:
        return {}

    expected_codes = {
        c["system_code"]
        for c in missing_cards
    }

    raw, _ = _v18_generate_text(
        """قيّم كل بطاقة مستشار بلا استثناء اعتمادًا على FACTS فقط.
أخرج فقط:
SYSTEM_CODE|SCORE|EVIDENCE_IDS
إذا الدرجة 50 أو أكثر يجب ذكر دليل حقيقي من FACTS.
50 أو أكثر يعني ملاءمة حالية مثبتة، وليس مجرد احتمال.
ممنوع الأسباب وJSON وMarkdown.""",
        {
            "facts": facts,
            "routing_cards": missing_cards,
        },
        RICH_V30_RECOVERY_MAX_NEW_TOKENS,
    )

    return _v30_parse_scores(
        raw,
        expected_codes,
        valid_fact_ids,
    )


def _v30_score_group(
    prompt,
    facts,
    cards,
    valid_fact_ids,
    max_new_tokens,
):
    expected_codes = {
        c["system_code"]
        for c in cards
    }

    raw, input_tokens = _v18_generate_text(
        prompt,
        {
            "facts": facts,
            "routing_cards": cards,
        },
        max_new_tokens,
    )

    rows = _v30_parse_scores(
        raw,
        expected_codes,
        valid_fact_ids,
    )

    missing_codes = expected_codes - set(rows.keys())

    if missing_codes:
        card_map = {
            c["system_code"]: c
            for c in cards
        }

        missing_cards = [
            card_map[code]
            for code in sorted(missing_codes)
        ]

        print(
            f"Rich v30 recovery: {len(missing_cards)} omitted advisor rows.",
            flush=True,
        )

        recovered = _v30_recover_missing(
            facts,
            missing_cards,
            valid_fact_ids,
        )
        rows.update(recovered)

    for code in expected_codes - set(rows.keys()):
        rows[code] = {
            "advisor_id": code,
            "score": 0.0,
            "evidence_ids": [],
        }

    return rows, input_tokens


# ------------------------------------------------------------------
# Deterministic evidence guards
# ------------------------------------------------------------------

_V30_NEED_MARKERS = (
    "حاجة",
    "احتياج",
    "فجوة",
    "تحد",
    "مشكلة",
    "ضعف",
    "تحسين",
    "تطوير",
    "مراجعة",
    "إعادة",
    "تحديث",
    "توسع",
    "توسيع",
    "مستقبل",
    "طموح",
    "خطة",
    "قرار",
    "أولو",
    "تنظيم",
    "إدارة",
    "استدام",
)


def _v30_contains_any(text, terms):
    text = _v30_normalize_text(text).lower()
    return any(term.lower() in text for term in terms)


def _v30_evidence_blob(item, fact_map):
    texts = []

    for fid in item.get("evidence_ids", []):
        fact = fact_map.get(fid)
        if fact:
            texts.append(fact["text"])

    return " ".join(texts)


def _v30_org_blob(organization, programs):
    pieces = []

    for key in [
        "activity_fields",
        "short_description",
        "detailed_description",
        "competitive_advantage",
        "important_notes",
    ]:
        value = organization.get(key)

        if isinstance(value, list):
            pieces.extend(str(x) for x in value)
        elif value:
            pieces.append(str(value))

    for p in programs:
        if not isinstance(p, dict):
            continue
        for key in [
            "name",
            "description",
            "target_audience",
            "beneficiary_value",
            "delivery_method",
        ]:
            value = p.get(key)
            if value:
                pieces.append(str(value))

    return _v30_normalize_text(" ".join(pieces)).lower()


def _v30_positive_need(text):
    return _v30_contains_any(text, _V30_NEED_MARKERS)


def _v30_gate(advisor_num, evidence_blob, org_blob, program_count):
    """
    This is not the scoring engine.
    AI still assigns the relevance score.
    The gate only rejects impossible/unsupported activations.
    """

    e = _v30_normalize_text(evidence_blob).lower()
    o = org_blob

    if not e:
        return False

    # Leadership / diagnosis / strategy
    if advisor_num == 1:
        return (
            _v30_contains_any(
                e,
                (
                    "قرار تنفيذي",
                    "أولويات",
                    "قيادة",
                    "مجلس",
                    "إعادة تنظيم",
                    "إعادة هيكلة",
                    "توسع",
                    "توسيع",
                    "موارد",
                ),
            )
            and _v30_positive_need(e)
        )

    if advisor_num == 2:
        return _v30_contains_any(
            e,
            (
                "نضج",
                "تشخيص",
                "جاهزية",
                "فجوة",
                "قدرات",
                "خط أساس",
                "تقييم مؤسسي",
            ),
        )

    if advisor_num == 3:
        return _v30_contains_any(
            e,
            (
                "فرص",
                "تهديد",
                "بيئة خارج",
                "بيئة داخل",
                "اتجاهات",
                "منافس",
                "تحليل داخلي",
                "تحليل خارجي",
            ),
        )

    if advisor_num == 4:
        return _v30_contains_any(
            e,
            (
                "شراكة",
                "شراكات",
                "شريك",
                "شركاء",
                "أصحاب المصلحة",
                "جهات",
            ),
        )

    if advisor_num == 5:
        return _v30_contains_any(
            e,
            (
                "تحول",
                "تغيير",
                "إعادة هيكلة",
                "إعادة تنظيم",
                "تبني",
                "مقاومة",
                "انتقال",
            ),
        )

    if advisor_num == 6:
        return _v30_contains_any(
            e,
            (
                "جودة",
                "رضا",
                "شكوى",
                "شكاوى",
                "معيار",
                "معايير",
                "تميز",
                "عدم مطابقة",
            ),
        ) and _v30_positive_need(e)

    if advisor_num == 7:
        return _v30_contains_any(
            e,
            (
                "مخاطر",
                "خطر",
                "استمرارية",
                "أزمة",
                "أزمات",
                "طوارئ",
                "تعطل",
            ),
        )

    if advisor_num == 8:
        return (
            _v30_contains_any(
                e,
                (
                    "استراتيجية",
                    "استراتيجي",
                    "أولويات",
                    "توجهات",
                    "خطة استراتيجية",
                ),
            )
            and _v30_positive_need(e)
        )

    if advisor_num == 9:
        return (
            _v30_contains_any(
                e,
                ("رؤية", "رسالة", "قيم", "هوية"),
            )
            and _v30_contains_any(
                e,
                ("تطوير", "مراجعة", "إعادة", "صياغة", "تحديث", "غير واضحة"),
            )
        )

    if advisor_num == 10:
        return (
            _v30_contains_any(
                e,
                ("قضية استراتيجية", "قضايا استراتيجية", "أهداف استراتيجية", "أولويات"),
            )
            and _v30_contains_any(
                e,
                ("تعارض", "ترتيب", "إعادة صياغة", "تحديد", "تطوير", "مراجعة"),
            )
        )

    if advisor_num == 11:
        return _v30_contains_any(
            e,
            (
                "مبادرة جديدة",
                "برنامج جديد",
                "مشروع جديد",
                "تصميم مبادرة",
                "تطوير مبادرة",
                "إطلاق",
                "مستقبل",
                "مستقبلي",
                "طموح",
            ),
        )

    if advisor_num == 12:
        cadence_terms = (
            "أسبوع",
            "أسبوعي",
            "يومي",
            "يومياً",
            "يوميًا",
            "عدة أيام",
            "ثلاثة أيام",
            "موسم",
            "موسمي",
            "رمضان",
            "تشغيل",
            "جدول",
            "مواعيد",
        )
        return _v30_contains_any(e, cadence_terms)

    if advisor_num == 13:
        return (
            _v30_contains_any(
                e,
                (
                    "محفظة",
                    "برامج متعددة",
                    "مبادرات متعددة",
                    "مسارات",
                    "61 برنامج",
                    "عشرات البرامج",
                ),
            )
            or program_count >= 4
        )

    if advisor_num == 14:
        return _v30_contains_any(
            e,
            (
                "مؤشر",
                "مؤشرات",
                "مستهدف",
                "لوحة",
                "أداء",
                "مقياس الرضا",
            ),
        )

    if advisor_num == 15:
        return _v30_contains_any(
            e,
            (
                "قياس الأثر",
                "الأثر الاجتماعي",
                "تقييم",
                "نتائج",
                "بحث",
                "دراسة",
                "مقياس الرضا",
            ),
        )

    # Governance / finance / growth / organisation / operations / tech / data
    if advisor_num == 16:
        governance = _v30_contains_any(
            e,
            (
                "حوكمة",
                "امتثال",
                "صلاحيات",
                "سياسات",
                "لجان",
                "إفصاح",
                "تعارض مصالح",
                "مساءلة",
            ),
        )
        governance_need = _v30_contains_any(
            e,
            (
                "فجوة",
                "ضعف",
                "تحسين",
                "مراجعة",
                "تعارض",
                "غير واضح",
                "التزام",
                "مخالفة",
            ),
        )
        high_score_only = (
            _v30_contains_any(e, ("99.", "99٪", "99%"))
            and not governance_need
        )
        return governance and governance_need and not high_score_only

    if advisor_num == 17:
        finance_terms = (
            "موازنة",
            "ميزانية",
            "تكلفة",
            "سيولة",
            "تدفق نقد",
            "انحراف مالي",
            "مصروف",
            "مصروفات",
            "إيراد",
            "إيرادات",
            "التزامات مالية",
            "إعادة تخصيص",
        )
        return _v30_contains_any(e, finance_terms) and _v30_positive_need(e)

    if advisor_num == 18:
        return _v30_contains_any(
            e,
            (
                "تنمية الموارد",
                "استدامة مالية",
                "تبرعات",
                "تبرع",
                "منح",
                "مانحين",
                "مانح",
                "تنويع الإيرادات",
                "اعتماد على ممول",
                "فجوة تمويل",
                "جمع التبرعات",
                "مصادر دخل",
            ),
        )

    if advisor_num == 19:
        return _v30_contains_any(
            e,
            (
                "وقف",
                "أوقاف",
                "استثمار اجتماعي",
                "استثمار مؤثر",
                "أصل استثماري",
                "سياسة استثمار",
            ),
        )

    if advisor_num == 20:
        hr_terms = (
            "موظف",
            "موظفين",
            "هيكل",
            "وظيف",
            "قوى عاملة",
            "عبء العمل",
            "جدارات",
            "أداء الموظفين",
            "مهارات الموظفين",
            "توظيف",
            "أدوار وظيفية",
        )
        return _v30_contains_any(e, hr_terms)

    if advisor_num == 21:
        return _v30_contains_any(
            e,
            (
                "عملية",
                "عمليات",
                "إجراء",
                "إجراءات",
                "تدفق",
                "اختناق",
                "تأخير",
                "هدر",
                "إعادة عمل",
                "رحلة خدمة",
                "زمن انتظار",
            ),
        )

    if advisor_num == 22:
        return _v30_contains_any(
            e,
            (
                "اتصال",
                "سمعة",
                "رسائل",
                "إعلام",
                "صورة ذهنية",
                "علاقات عامة",
                "محتوى",
                "اتصال داخلي",
                "أزمة اتصال",
            ),
        )

    if advisor_num == 23:
        return _v30_contains_any(
            e,
            (
                "تسويق رقمي",
                "حملة",
                "حملات",
                "إعلانات",
                "تحويل",
                "اكتساب",
                "صفحة هبوط",
                "إعادة استهداف",
            ),
        )

    if advisor_num == 24:
        return _v30_contains_any(
            e,
            (
                "تحول رقمي",
                "ذكاء اصطناعي",
                "أتمتة",
                "نظام",
                "منصة",
                "تكامل",
                "تقنية",
                "رقمنة",
                "أودو",
                "أرشيف إلكتروني",
            ),
        )

    if advisor_num == 25:
        data_terms = (
            "حوكمة البيانات",
            "جودة البيانات",
            "قاموس بيانات",
            "مصدر الحقيقة",
            "تقارير",
            "سجلات",
            "أرشيف",
            "إدارة معرفة",
            "توثيق",
            "صلاحيات وصول",
        )
        return _v30_contains_any(e, data_terms) and _v30_positive_need(e)

    # Sector advisors
    sector_terms = {
        26: (
            "ثقاف",
            "ترفيه",
            "تراث",
            "فنون",
            "مسرح",
            "موسيقى",
            "مهرجان",
            "ديوانية",
        ),
        27: (
            "تعليم",
            "تدريب",
            "بحث",
            "دراسة",
            "طلاب",
            "مدارس",
            "قدوات",
            "نقل المعرفة",
        ),
        28: (
            "صح",
            "مستشفى",
            "علاج",
            "فحص",
            "فحوصات",
            "وقاية",
            "تغذية",
            "العلاج الطبيعي",
        ),
        29: (
            "رعاية اجتماعية",
            "خدمات اجتماعية",
            "كبار السن",
            "أسرة",
            "أسر",
            "تمكين اجتماعي",
            "جودة الحياة",
            "دعم اجتماعي",
        ),
        30: (
            "بيئة",
            "بيئي",
            "تشجير",
            "نفايات",
            "إعادة تدوير",
            "مياه",
            "طاقة",
            "مناخ",
            "تنوع حيوي",
        ),
        31: (
            "سكن",
            "إسكان",
            "ترميم",
            "إيجار",
            "منازل",
            "أحياء",
            "تنمية محلية",
            "سبل العيش",
        ),
        32: (
            "حقوق",
            "مناصرة",
            "قانون",
            "عدالة",
            "شكوى",
            "دعم قانوني",
            "حماية حق",
        ),
        33: (
            "تطوع",
            "متطوع",
            "متطوعين",
            "فرص تطوعية",
            "بناء قدرات",
            "حاضنة",
            "مسرعة",
        ),
        34: (
            "دعوة",
            "ديني",
            "دينية",
            "حجاج",
            "معتمرين",
            "ضيوف الرحمن",
            "عمرة",
            "قرآن",
            "جاليات",
        ),
        35: (
            "جمعية مهنية",
            "رابطة مهنية",
            "عضوية",
            "أعضاء مهنيين",
            "تطوير مهني",
            "لجان مهنية",
        ),
    }

    if advisor_num in sector_terms:
        return _v30_contains_any(e, sector_terms[advisor_num])

    return False


def _v30_reason(advisor, item, fact_map):
    evidence_texts = []

    for fid in item.get("evidence_ids", []):
        fact = fact_map.get(fid)
        if not fact:
            continue

        text = _v30_arabic_public_text(
            _v30_trim(fact["text"], 230)
        )
        if text:
            evidence_texts.append(text)

    name = _v30_arabic_public_text(
        advisor.get("name_ar") or "هذا المستشار"
    )

    if not evidence_texts:
        return (
            f"يرتبط {name} بالجمعية وفق نطاق اختصاصه، "
            "وتوجد في بياناتها وقائع مباشرة تبرر إشراكه في التقييم الاستشاري الحالي."
        )

    if len(evidence_texts) == 1:
        return (
            f"يرتبط {name} بالجمعية لأن البيانات توضح أن "
            f"{evidence_texts[0]}. "
            "وهذه واقعة مباشرة تقع ضمن نطاق اختصاصه وتبرر إشراكه "
            "دون افتراض احتياجات غير مذكورة في بيانات الجمعية."
        )

    return (
        f"يرتبط {name} بالجمعية لأن البيانات توضح أن "
        f"{evidence_texts[0]}، كما توضح أن {evidence_texts[1]}. "
        "وهاتان الواقعتان تقعان مباشرة ضمن نطاق اختصاصه وتبرران إشراكه "
        "في التقييم الاستشاري الحالي."
    )


def advisory_match_rich_v30(job_input):
    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = _v30_build_facts(
        organization,
        programs,
    )

    valid_fact_ids = {
        fact["fact_id"]
        for fact in facts
    }

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]
    advisor_by_code = {
        advisor["system_code"]: advisor
        for advisor in advisors
    }

    functional_cards = _v30_cards(
        advisors,
        1,
        25,
    )

    sector_cards = _v30_cards(
        advisors,
        26,
        35,
    )

    print(
        "Rich v30: scoring 25 functional advisors (compact, no prose generation)...",
        flush=True,
    )

    functional_scores, functional_input_tokens = _v30_score_group(
        RICH_V30_FUNCTIONAL_PROMPT,
        facts,
        functional_cards,
        valid_fact_ids,
        RICH_V30_FUNCTIONAL_MAX_NEW_TOKENS,
    )

    print(
        "Rich v30: scoring 10 sector advisors (compact, no prose generation)...",
        flush=True,
    )

    sector_scores, sector_input_tokens = _v30_score_group(
        RICH_V30_SECTOR_PROMPT,
        facts,
        sector_cards,
        valid_fact_ids,
        RICH_V30_SECTOR_MAX_NEW_TOKENS,
    )

    all_scores = {}
    all_scores.update(functional_scores)
    all_scores.update(sector_scores)

    org_blob = _v30_org_blob(
        organization,
        programs,
    )

    matches = []

    for code, item in all_scores.items():
        if item["score"] < RICH_V30_MIN_PUBLIC_SCORE:
            continue

        if not item.get("evidence_ids"):
            continue

        advisor = advisor_by_code.get(code)
        if not advisor:
            continue

        advisor_num = int(advisor["advisor_id"])

        evidence_blob = _v30_evidence_blob(
            item,
            fact_map,
        )

        if not _v30_gate(
            advisor_num,
            evidence_blob,
            org_blob,
            len(programs),
        ):
            print(
                f"Rich v30 evidence guard dropped {code} "
                f"(score={item['score']:.2f}).",
                flush=True,
            )
            continue

        matches.append({
            "advisor_id": code,
            "score": item["score"],
            "reason": _v30_reason(
                advisor,
                item,
                fact_map,
            ),
        })

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print(
        "Rich v30 complete: "
        f"{len(matches)} advisors >= {RICH_V30_MIN_PUBLIC_SCORE:.2f}; "
        f"input tokens functional={functional_input_tokens}, "
        f"sector={sector_input_tokens}. "
        "Public reasons were generated deterministically in Python.",
        flush=True,
    )

    return {
        "ranked": matches
    }


# ---------------------------------------------------------------------
# Rich AI Router v31 — zero-generation relevance classifier
#
# This replaces autoregressive routing with direct next-token scoring.
#
# Why this is different:
# - The already-loaded Qwen3-14B is still the AI decision maker.
# - It DOES NOT generate routing prose or 35 output lines.
# - For each advisor it sees only its most relevant evidence snippets.
# - One batched forward pass scores "relevant" vs "not relevant".
# - Score is the model probability of true current relevance.
# - Threshold remains exactly 0.50.
# - Public Arabic reasons are deterministic, concise, and source-grounded.
#
# No extra model download. No new embedding model. No generated Arabic.
# ---------------------------------------------------------------------

RICH_V31_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V31_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V31_BATCH_SIZE = int(
    os.environ.get("RICH_V31_BATCH_SIZE", "4")
)

RICH_V31_TOP_FACTS = int(
    os.environ.get("RICH_V31_TOP_FACTS", "6")
)

RICH_V31_MAX_PROMPT_TOKENS = int(
    os.environ.get("RICH_V31_MAX_PROMPT_TOKENS", "2600")
)

RICH_V31_LOGIT_TEMPERATURE = float(
    os.environ.get("RICH_V31_LOGIT_TEMPERATURE", "1.0")
)

_RICH_V31_STOPWORDS = {
    "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "على", "إلى", "الى",
    "عن", "من", "في", "مع", "أو", "او", "ثم", "كما", "كل", "عند",
    "وجود", "يوجد", "توجد", "الحاجة", "احتياج", "المستشار", "مستشار",
    "الجمعية", "المنظمة", "المؤسسة", "مجال", "مجالات", "دعم", "تحسين",
    "تطوير", "إدارة", "ادارة", "بشكل", "ضمن", "حسب", "قبل", "بعد",
    "العمل", "الأعمال", "خلال", "ذات", "ذو", "وهو", "وهي", "يكون",
    "تكون", "يمكن", "قابل", "قابلة", "حالي", "حالية", "فعلي", "فعلية",
}


def _v31_norm_ar(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("ـ", "")
    # Remove Arabic diacritics.
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = (
        text
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _v31_tokens(text):
    norm = _v31_norm_ar(text)
    words = re.findall(r"[\u0600-\u06FF]{2,}", norm)

    result = []
    for word in words:
        if word in _RICH_V31_STOPWORDS:
            continue
        if len(word) < 3:
            continue
        result.append(word)

    return result


def _v31_safe_cut(text, limit):
    text = _v30_arabic_public_text(text)
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= limit:
        return text.rstrip(" .،؛:")

    piece = text[:limit]
    if " " in piece:
        piece = piece.rsplit(" ", 1)[0]

    return piece.rstrip(" .،؛:")


def _v31_build_facts(organization, programs):
    facts = _v30_build_facts(
        organization,
        programs,
    )

    # Structural facts derived only from the supplied payload.
    program_count = len([
        p for p in programs
        if isinstance(p, dict)
    ])

    if program_count:
        facts.append({
            "fact_id": "C1",
            "source": "بنية المحفظة",
            "text": (
                f"تحتوي الحمولة الحالية على {program_count} "
                "برنامجًا ومشروعًا موثقًا موزعة على مجالات متعددة."
            ),
        })

    program_blob = " ".join(
        str(p.get("name", "")) + " "
        + str(p.get("description", "")) + " "
        + str(p.get("delivery_method", ""))
        for p in programs
        if isinstance(p, dict)
    )

    cadence_terms = (
        "أسبوع", "اسبوع", "يومي", "دوري", "موسمي",
        "رمضان", "شهري", "سنوي", "متكرر",
    )

    cadence_hits = sum(
        1
        for p in programs
        if isinstance(p, dict)
        and _v30_contains_any(
            " ".join(
                str(p.get(k, ""))
                for k in ("name", "description", "delivery_method")
            ),
            cadence_terms,
        )
    )

    if cadence_hits >= 2:
        facts.append({
            "fact_id": "C2",
            "source": "تعقيد التشغيل",
            "text": (
                f"يوجد في الحمولة الحالية {cadence_hits} برامج أو مشاريع "
                "على الأقل ذات طبيعة دورية أو موسمية أو متكررة، "
                "ما يخلق احتياجًا فعليًا للتنسيق التشغيلي بين المواعيد والموارد."
            ),
        })

    partnership_terms = (
        "شراكة", "شراكات", "شريك", "شركاء",
        "بالتعاون", "تعاون مع", "مستشفى", "جهة",
    )

    partnership_hits = sum(
        1
        for p in programs
        if isinstance(p, dict)
        and _v30_contains_any(
            " ".join(
                str(p.get(k, ""))
                for k in ("name", "description", "delivery_method")
            ),
            partnership_terms,
        )
    )

    org_text = " ".join(
        str(organization.get(k, ""))
        for k in (
            "short_description",
            "detailed_description",
            "important_notes",
            "competitive_advantage",
        )
    )

    if (
        partnership_hits >= 2
        or _v30_contains_any(
            org_text,
            ("شراكات", "شركاء", "9 شراكات", "تسع شراكات"),
        )
    ):
        facts.append({
            "fact_id": "C3",
            "source": "الشراكات في التنفيذ",
            "text": (
                f"يظهر التنفيذ المشترك أو الشراكات في {partnership_hits} "
                "برامج أو مشاريع على الأقل في الحمولة الحالية، "
                "إضافة إلى ما ورد في بيانات الجمعية عن شبكة الشراكات."
            ),
        })

    if _v30_contains_any(
        org_text + " " + program_blob,
        ("متطوع", "متطوعين", "تطوع", "فرص تطوعية"),
    ):
        facts.append({
            "fact_id": "C4",
            "source": "منظومة التطوع",
            "text": (
                "تتضمن بيانات الجمعية منظومة تطوع فعلية تشمل متطوعين "
                "وفرصًا تطوعية وأنشطة تطوعية مرتبطة بتنفيذ البرامج."
            ),
        })

    return facts


def _v31_advisor_query(advisor):
    pieces = [
        advisor.get("name_ar"),
        advisor.get("mission"),
        advisor.get("owned_outcome"),
    ]

    pieces.extend((advisor.get("owns") or [])[:8])
    pieces.extend((advisor.get("activation_when") or [])[:8])

    return " ".join(
        str(x)
        for x in pieces
        if x
    )


def _v31_fact_relevance(advisor, fact):
    query_text = _v31_advisor_query(advisor)
    query_tokens = _v31_tokens(query_text)
    fact_tokens = _v31_tokens(fact.get("text"))

    if not query_tokens or not fact_tokens:
        return 0.0

    qset = set(query_tokens)
    fset = set(fact_tokens)

    overlap = qset & fset

    # Weighted lexical overlap.
    score = sum(
        1.0 + min(len(token), 8) / 8.0
        for token in overlap
    )

    score /= max(3.0, len(fset) ** 0.5)

    # Activation phrase coverage bonus.
    for phrase in (advisor.get("activation_when") or [])[:8]:
        ptokens = set(_v31_tokens(phrase))
        if not ptokens:
            continue

        coverage = len(ptokens & fset) / len(ptokens)

        if coverage >= 0.55:
            score += 2.0 * coverage
        elif coverage >= 0.35:
            score += 0.8 * coverage

    # Ownership phrase bonus.
    for phrase in (advisor.get("owns") or [])[:8]:
        ptokens = set(_v31_tokens(phrase))
        if not ptokens:
            continue

        coverage = len(ptokens & fset) / len(ptokens)
        if coverage >= 0.5:
            score += 1.0 * coverage

    # Generic org-name/type facts are not useful routing evidence.
    if fact.get("fact_id") in {"O1", "O2"}:
        score *= 0.05

    # Computed structural facts are intentionally strong for the
    # specialist domains they describe.
    fid = fact.get("fact_id")
    advisor_num = int(advisor.get("advisor_id"))

    if fid == "C1" and advisor_num == 13:
        score += 5.0
    elif fid == "C2" and advisor_num == 12:
        score += 5.0
    elif fid == "C3" and advisor_num == 4:
        score += 5.0
    elif fid == "C4" and advisor_num == 33:
        score += 5.0

    return float(score)


def _v31_top_facts(advisor, facts):
    ranked = sorted(
        (
            (_v31_fact_relevance(advisor, fact), fact)
            for fact in facts
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    selected = []

    for score, fact in ranked:
        if len(selected) >= RICH_V31_TOP_FACTS:
            break

        # Always allow strongest structural facts; ordinary facts need
        # at least some lexical/activation relationship.
        if score <= 0 and not str(fact.get("fact_id", "")).startswith("C"):
            continue

        if fact.get("fact_id") in {"O1", "O2"}:
            continue

        selected.append({
            "fact_id": fact["fact_id"],
            "source": fact.get("source"),
            "text": _v31_safe_cut(
                fact.get("text"),
                430,
            ),
            "_retrieval_score": round(score, 4),
        })

    return selected


def _v31_card(advisor):
    advisor_num = int(advisor.get("advisor_id"))

    return {
        "system_code": advisor.get("system_code"),
        "name": advisor.get("name_ar"),
        "class": (
            "SECTOR"
            if advisor_num >= 26
            else "FUNCTIONAL"
        ),
        "mission": _v31_safe_cut(
            advisor.get("mission"),
            230,
        ),
        "owned_outcome": _v31_safe_cut(
            advisor.get("owned_outcome"),
            260,
        ),
        "activation_when": [
            _v31_safe_cut(x, 140)
            for x in (advisor.get("activation_when") or [])[:6]
        ],
        "not_primary_when": [
            _v31_safe_cut(x, 140)
            for x in (advisor.get("not_primary_when") or [])[:4]
        ],
        "boundaries": [
            _v31_safe_cut(x, 160)
            for x in (advisor.get("boundaries") or [])[:3]
        ],
    }


RICH_V31_CLASSIFIER_SYSTEM = """
أنت مصنف ملاءمة لمستشار واحد في منظومة أثر.

المطلوب قرار واحد فقط:
1 = المستشار مرتبط حاليًا ارتباطًا حقيقيًا وماديًا بالجمعية.
0 = لا توجد ملاءمة حالية كافية.

للمستشار FUNCTIONAL:
اختر 1 فقط إذا أظهرت الأدلة حاجة أو قرارًا أو فجوة أو تعقيدًا حاليًا
يقع مباشرة داخل ملكية المستشار.
مجرد وجود المجال في الجمعية لا يكفي.
غياب معلومة ليس احتياجًا.
إنجاز مرتفع ليس فجوة.
المتطوعون وحدهم ليسوا حاجة موارد بشرية.
وجود برامج وحده لا يعني حاجة مالية أو تشغيلية أو بيانات أو اتصال.
لكن تعقيد محفظة كبير، تشغيل متكرر متعدد الموارد، أو اعتماد فعلي على شبكة
شراكات يمكن أن يكون تفعيلًا حقيقيًا للمستشار المختص.

للمستشار SECTOR:
اختر 1 إذا كان القطاع نفسه جوهريًا ومتكررًا في رسالة الجمعية أو أهدافها
أو محفظة برامجها، حتى دون وجود مشكلة.
نشاط واحد عابر أو كلمة عابرة لا يكفي.

التزم بحدود بطاقة المستشار.
لا تخترع معلومات غير موجودة.
إذا كانت الأدلة ملتبسة أو مجرد احتمال فاختر 0.

اكتب رقمًا واحدًا فقط: 1 أو 0.
"""


def _v31_prompt_for_advisor(advisor, selected_facts):
    payload = {
        "advisor": _v31_card(advisor),
        "evidence": [
            {
                "fact_id": f["fact_id"],
                "source": f.get("source"),
                "text": f.get("text"),
            }
            for f in selected_facts
        ],
    }

    messages = [
        {
            "role": "system",
            "content": RICH_V31_CLASSIFIER_SYSTEM,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]

    return _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _v31_label_ids():
    positive = _RICH_TOKENIZER.encode(
        "1",
        add_special_tokens=False,
    )
    negative = _RICH_TOKENIZER.encode(
        "0",
        add_special_tokens=False,
    )

    if len(positive) != 1 or len(negative) != 1:
        raise ValueError(
            "Rich v31 requires single-token labels 1 and 0."
        )

    return positive[0], negative[0]


def _v31_forward_scores(prompts):
    import inspect
    import torch

    pos_id, neg_id = _v31_label_ids()

    results = []
    batch_size = max(1, RICH_V31_BATCH_SIZE)

    start = 0

    while start < len(prompts):
        current = prompts[start:start + batch_size]

        encoded = _RICH_TOKENIZER(
            current,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=RICH_V31_MAX_PROMPT_TOKENS,
            add_special_tokens=False,
        )

        encoded = {
            k: v.to(_RICH_DEVICE)
            for k, v in encoded.items()
        }

        forward_kwargs = dict(encoded)

        try:
            signature = inspect.signature(
                _RICH_MODEL.forward
            )
            if "logits_to_keep" in signature.parameters:
                forward_kwargs["logits_to_keep"] = 1
        except Exception:
            pass

        try:
            with torch.inference_mode():
                outputs = _RICH_MODEL(
                    **forward_kwargs
                )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise

            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)

            print(
                f"Rich v31 reduced classifier batch size to {batch_size} after CUDA OOM.",
                flush=True,
            )
            continue

        logits = outputs.logits

        # With logits_to_keep=1 -> [B, 1, V].
        # Otherwise left padding ensures the final sequence position is
        # the assistant generation point for every row.
        last_logits = logits[:, -1, :].float()

        pair = torch.stack(
            [
                last_logits[:, neg_id],
                last_logits[:, pos_id],
            ],
            dim=-1,
        )

        temperature = max(
            0.05,
            float(RICH_V31_LOGIT_TEMPERATURE),
        )

        probs = torch.softmax(
            pair / temperature,
            dim=-1,
        )[:, 1]

        results.extend(
            float(x)
            for x in probs.detach().cpu().tolist()
        )

        del outputs
        del logits
        del last_logits
        del pair
        del encoded

        start += len(current)

    return results


def _v31_fact_summary(fact):
    fid = str(fact.get("fact_id", ""))
    text = _v31_safe_cut(
        fact.get("text"),
        150,
    )

    if fid.startswith("P"):
        m = re.search(
            r'(?:برنامج|مشروع)\s+"([^"]+)"',
            text,
        )
        if m:
            return f'«{_v31_safe_cut(m.group(1), 90)}»'

    if fid.startswith("C"):
        return _v31_safe_cut(text, 135)

    # Prefer the first meaningful clause for notes/descriptions.
    pieces = re.split(r"[؛.!؟]", text)
    for piece in pieces:
        piece = _v31_safe_cut(piece, 125)
        if len(piece) >= 18:
            return piece

    return _v31_safe_cut(text, 125)


def _v31_public_reason(advisor, selected_facts):
    name = _v31_safe_cut(
        advisor.get("name_ar") or "هذا المستشار",
        100,
    )

    summaries = []

    for fact in selected_facts[:3]:
        summary = _v31_fact_summary(fact)

        if (
            summary
            and summary not in summaries
        ):
            summaries.append(summary)

        if len(summaries) == 2:
            break

    advisor_num = int(advisor.get("advisor_id"))
    is_sector = advisor_num >= 26

    if not summaries:
        return (
            f"يرتبط {name} بالحالة الحالية لأن الأدلة الموثقة في بيانات الجمعية "
            "تقع مباشرة ضمن نطاق اختصاصه وتدعم إشراكه في التقييم الاستشاري."
        )

    if len(summaries) == 1:
        evidence_phrase = summaries[0]
    else:
        evidence_phrase = (
            summaries[0]
            + "، وكذلك "
            + summaries[1]
        )

    if is_sector:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الوقائع حضورًا فعليًا ومتكررًا لهذا القطاع في عمل الجمعية، "
            "لذلك تقع الملاءمة مباشرة ضمن نطاق اختصاصه."
        )
    else:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتوضح هذه الوقائع حاجة أو تعقيدًا حاليًا يقع مباشرة ضمن نطاق اختصاصه، "
            "دون افتراض فجوات غير مذكورة في بيانات الجمعية."
        )

    return _v30_arabic_public_text(
        reason
    )


def advisory_match_rich_v31(job_input):
    started = time.time()

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = _v31_build_facts(
        organization,
        programs,
    )

    advisors = _RICH_REGISTRY["advisors"]

    prompts = []
    evidence_by_code = {}
    advisor_by_code = {}

    for advisor in advisors:
        code = advisor.get("system_code")
        if not code:
            continue

        selected = _v31_top_facts(
            advisor,
            facts,
        )

        evidence_by_code[code] = selected
        advisor_by_code[code] = advisor

        prompts.append({
            "advisor_id": code,
            "prompt": _v31_prompt_for_advisor(
                advisor,
                selected,
            ),
        })

    print(
        f"Rich v31: scoring {len(prompts)} advisors with zero generated tokens...",
        flush=True,
    )

    probabilities = _v31_forward_scores(
        [
            item["prompt"]
            for item in prompts
        ]
    )

    matches = []

    for item, probability in zip(
        prompts,
        probabilities,
    ):
        code = item["advisor_id"]

        if probability < RICH_V31_MIN_PUBLIC_SCORE:
            continue

        advisor = advisor_by_code[code]
        selected = evidence_by_code[code]

        # Do not expose a positive result with no usable evidence.
        if not selected:
            continue

        matches.append({
            "advisor_id": code,
            "score": round(
                float(probability),
                4,
            ),
            "reason": _v31_public_reason(
                advisor,
                selected,
            ),
        })

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    elapsed = round(
        time.time() - started,
        2,
    )

    print(
        f"Rich v31 complete in {elapsed}s. "
        f"Returned {len(matches)} advisors >= "
        f"{RICH_V31_MIN_PUBLIC_SCORE:.2f}. "
        "Autoregressive routing generations: 0.",
        flush=True,
    )

    return {
        "ranked": matches
    }


# ---------------------------------------------------------------------
# Rich AI Router v32 — ONE-PASS production router
#
# Root fix:
# - Do NOT ask Qwen to compare all 35 verbose advisor DNAs.
# - Do NOT do 35 forward passes.
# - Do NOT generate Arabic reasons.
#
# Pipeline:
#   A) Python creates HIGH-RECALL candidates from explicit evidence only.
#   B) ONE Qwen generation adjudicates only those candidates.
#   C) Python validates returned evidence IDs.
#   D) Python writes concise clean Arabic reasons deterministically.
#
# Normal path = ONE model generation.
# Final threshold remains score >= 0.50.
# ---------------------------------------------------------------------

RICH_V32_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V32_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V32_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V32_MAX_NEW_TOKENS", "320")
)

# Functional activation vocabulary is intentionally narrow.
# Mere existence of a domain is not enough.
_V32_FUNCTIONAL_TERMS = {
    1: (
        "قرار تنفيذي", "أولوية تنفيذية", "أولويات تنفيذية",
        "إعادة تنظيم", "إعادة هيكلة", "توسع مؤسسي", "نمو مؤسسي",
        "قيادة تنفيذية",
    ),
    2: (
        "نضج مؤسسي", "تشخيص مؤسسي", "تقييم مؤسسي",
        "جاهزية مؤسسية", "خط أساس مؤسسي", "فجوة مؤسسية",
        "قدرات مؤسسية",
    ),
    3: (
        "تحليل داخلي", "تحليل خارجي", "فرص وتهديدات",
        "فرص", "تهديدات", "اتجاهات السوق", "اتجاهات المجتمع",
        "عوامل خارجية", "عوامل داخلية",
    ),
    4: (
        "شراكة", "شراكات", "شريك", "شركاء",
        "أصحاب المصلحة", "بالتعاون مع",
    ),
    5: (
        "تحول مؤسسي", "تغيير مؤسسي", "إعادة هيكلة",
        "إعادة تنظيم", "إدارة التغيير", "مقاومة التغيير",
        "تبني التغيير",
    ),
    6: (
        "جودة الخدمة", "جودة الخدمات", "معايير الجودة",
        "رضا المستفيد", "رضا المستفيدين", "شكاوى",
        "عدم مطابقة", "تميز مؤسسي",
    ),
    7: (
        "مخاطر", "خطر", "استمرارية الأعمال", "استمرارية",
        "طوارئ", "أزمة", "أزمات", "تعطل",
    ),
    8: (
        "مراجعة استراتيجية", "تحديث الاستراتيجية",
        "تحديث الخطة الاستراتيجية", "مفاضلة استراتيجية",
        "خيارات استراتيجية", "توجه استراتيجي جديد",
    ),
    9: (
        "تطوير الرؤية", "تحديث الرؤية", "تطوير الرسالة",
        "تحديث الرسالة", "إعادة صياغة الرؤية",
        "إعادة صياغة الرسالة", "تطوير الهوية",
    ),
    10: (
        "قضايا استراتيجية", "إعادة صياغة الأهداف",
        "تطوير الأهداف الاستراتيجية", "ترتيب الأولويات الاستراتيجية",
        "تعارض الأهداف",
    ),
    11: (
        "مبادرة جديدة", "برنامج جديد", "مشروع جديد",
        "تصميم مبادرة", "تصميم برنامج", "إعادة تصميم",
        "مبادرات مستقبلية", "مشروعات مستقبلية",
    ),
    12: (
        "أسبوعي", "أسبوعية", "يومي", "يومية", "دوري", "دورية",
        "موسمي", "موسمية", "رمضان", "تشغيل", "جدول",
        "مواعيد", "خطة تشغيلية",
    ),
    13: (
        "محفظة", "برامج متعددة", "مبادرات متعددة",
        "مشاريع متعددة", "مسارات", "برامج ومشاريع",
    ),
    14: (
        "مؤشرات أداء", "مؤشر أداء", "لوحة قيادة",
        "لوحة مؤشرات", "مستهدفات أداء", "منظومة الأداء",
        "قياس الأداء",
    ),
    15: (
        "قياس الأثر", "إدارة الأثر", "تقييم الأثر",
        "متابعة وتقييم", "تقييم النتائج", "نظرية التغيير",
        "التعلم المؤسسي",
    ),
    16: (
        "فجوة حوكمة", "تحسين الحوكمة", "امتثال",
        "مخالفة", "صلاحيات", "تعارض مصالح",
        "سياسة حوكمة", "لائحة حوكمة",
    ),
    17: (
        "موازنة", "ميزانية", "تكلفة", "تكاليف",
        "سيولة", "تدفق نقدي", "انحراف مالي",
        "إعادة تخصيص مالي", "مصروفات", "إيرادات",
    ),
    18: (
        "استدامة مالية", "تنمية الموارد", "تنويع الإيرادات",
        "فجوة تمويل", "جمع التبرعات", "تبرعات",
        "مانحين", "مانح", "منح", "مصادر دخل",
    ),
    19: (
        "وقف", "أوقاف", "استثمار اجتماعي",
        "استثمار مؤثر", "سياسة استثمار", "أصل استثماري",
    ),
    20: (
        "هيكل تنظيمي", "إعادة هيكلة", "موظفين", "موظف",
        "قوى عاملة", "عبء العمل", "جدارات",
        "توظيف", "أدوار وظيفية", "أداء الموظفين",
    ),
    21: (
        "إجراءات", "إجراء", "عملية", "عمليات",
        "اختناق", "تأخير", "هدر", "إعادة عمل",
        "تحسين العمليات", "رحلة الخدمة",
    ),
    22: (
        "استراتيجية اتصال", "اتصال مؤسسي", "سمعة",
        "علاقات عامة", "صورة ذهنية", "رسائل",
        "أزمة اتصال", "إعلام",
    ),
    23: (
        "تسويق رقمي", "حملة رقمية", "حملات رقمية",
        "إعلانات", "اكتساب", "تحويل", "إعادة استهداف",
    ),
    24: (
        "تحول رقمي", "ذكاء اصطناعي", "أتمتة",
        "تكامل الأنظمة", "رقمنة", "منصة رقمية",
        "نظام رقمي", "حالة استخدام",
    ),
    25: (
        "حوكمة البيانات", "جودة البيانات", "قاموس بيانات",
        "مصدر الحقيقة", "إدارة المعرفة", "أرشيف",
        "سجلات", "تقارير", "ملكية البيانات",
    ),
}

_V32_SECTOR_TERMS = {
    26: (
        "ثقافي", "ثقافية", "ثقافة", "ترفيهي", "ترفيهية",
        "تراث", "تراثي", "فنون", "ديوانية",
    ),
    27: (
        "تعليم", "تعليمي", "تعليمية", "تدريب",
        "بحث", "بحوث", "دراسة", "دراسات",
        "قدوات", "نقل المعرفة", "تبادل الخبرات",
    ),
    28: (
        "صحة", "صحي", "صحية", "مستشفى", "علاج",
        "علاجي", "فحوصات", "وقاية", "تغذية",
        "العلاج الطبيعي",
    ),
    29: (
        "رعاية اجتماعية", "خدمات اجتماعية", "اجتماعي",
        "اجتماعية", "كبار السن", "جودة الحياة",
        "تمكين اجتماعي", "دعم اجتماعي",
    ),
    30: (
        "بيئة", "بيئي", "بيئية", "تشجير", "نفايات",
        "إعادة تدوير", "مناخ", "تنوع حيوي",
    ),
    31: (
        "إسكان", "سكن", "سكني", "ترميم",
        "إيجار", "منازل", "تنمية محلية", "سبل العيش",
    ),
    32: (
        "حقوق", "حق كبار السن", "مناصرة",
        "دعم قانوني", "قانوني", "توعية بالحقوق",
    ),
    33: (
        "تطوع", "تطوعي", "تطوعية", "متطوع",
        "متطوعين", "فرص تطوعية", "بناء قدرات",
    ),
    34: (
        "دعوة", "دعوي", "ديني", "دينية",
        "ضيوف الرحمن", "حجاج", "معتمرين",
        "عمرة", "تعليم ديني",
    ),
    35: (
        "جمعية مهنية", "رابطة مهنية", "عضوية مهنية",
        "أعضاء مهنيين", "تطوير مهني", "لجان مهنية",
    ),
}

# Advisors whose activation is directly supported by computed structural facts.
_V32_STRUCTURAL_FACTS = {
    4: ("C3",),
    12: ("C2",),
    13: ("C1",),
    33: ("C4",),
}


def _v32_fact_text(fact):
    return _v31_norm_ar(fact.get("text", ""))


def _v32_match_fact_ids(facts, terms):
    matched = []

    normalized_terms = [
        _v31_norm_ar(term)
        for term in terms
        if term
    ]

    for fact in facts:
        fid = fact.get("fact_id")

        # Name/type facts are never sufficient routing evidence.
        if fid in {"O1", "O2"}:
            continue

        text = _v32_fact_text(fact)

        if any(term in text for term in normalized_terms):
            matched.append(fid)

    return matched


def _v32_candidate_evidence(advisor_num, facts):
    evidence = []

    for fid in _V32_STRUCTURAL_FACTS.get(advisor_num, ()):
        if any(f.get("fact_id") == fid for f in facts):
            evidence.append(fid)

    if advisor_num <= 25:
        terms = _V32_FUNCTIONAL_TERMS.get(advisor_num, ())
    else:
        terms = _V32_SECTOR_TERMS.get(advisor_num, ())

    for fid in _v32_match_fact_ids(facts, terms):
        if fid not in evidence:
            evidence.append(fid)

    return evidence


def _v32_candidate_allowed(advisor_num, evidence_ids, fact_map, program_count):
    """
    Hard activation boundary.
    This does NOT choose the advisor; it only prevents impossible candidates.
    """

    if not evidence_ids:
        return False

    texts = " ".join(
        fact_map[fid]["text"]
        for fid in evidence_ids
        if fid in fact_map
    )

    norm = _v31_norm_ar(texts)

    # Structural advisors.
    if advisor_num == 4:
        return "C3" in evidence_ids or _v30_contains_any(
            norm, ("شراكة", "شراكات", "شريك", "شركاء")
        )

    if advisor_num == 12:
        return "C2" in evidence_ids

    if advisor_num == 13:
        return "C1" in evidence_ids and program_count >= 4

    if advisor_num == 33:
        return "C4" in evidence_ids

    # Functional advisors need explicit specialist activation.
    if advisor_num <= 25:
        # Governance: high governance achievement alone is not a need.
        if advisor_num == 16:
            need_terms = (
                "فجوة", "تحسين", "امتثال", "مخالفة",
                "صلاحيات", "تعارض", "سياسة", "لائحة",
            )
            return _v30_contains_any(norm, need_terms)

        # Finance requires a real finance concept, not "program funding".
        if advisor_num == 17:
            return _v30_contains_any(
                norm,
                (
                    "موازنة", "ميزانية", "تكلفة", "سيولة",
                    "تدفق نقدي", "انحراف مالي",
                    "مصروفات", "ايرادات",
                ),
            )

        # HR cannot be activated by volunteers alone.
        if advisor_num == 20:
            return _v30_contains_any(
                norm,
                (
                    "هيكل تنظيمي", "موظف", "موظفين",
                    "قوى عاملة", "عبء العمل", "جدارات",
                    "توظيف", "ادوار وظيفية",
                ),
            )

        # Operations requires process language; cadence belongs to SP-12.
        if advisor_num == 21:
            return _v30_contains_any(
                norm,
                (
                    "اجراء", "اجراءات", "عملية", "عمليات",
                    "اختناق", "تاخير", "هدر", "اعادة عمل",
                    "تحسين العمليات", "رحلة الخدمة",
                ),
            )

        # KPI/data/impact require explicit system/measurement intent.
        if advisor_num == 14:
            return _v30_contains_any(
                norm,
                (
                    "مؤشرات اداء", "لوحة مؤشرات",
                    "لوحة قيادة", "قياس الاداء",
                    "مستهدفات اداء", "منظومة الاداء",
                ),
            )

        if advisor_num == 15:
            return _v30_contains_any(
                norm,
                (
                    "قياس الاثر", "تقييم الاثر",
                    "متابعة وتقييم", "تقييم النتائج",
                    "نظرية التغيير", "التعلم المؤسسي",
                ),
            )

        if advisor_num == 25:
            return _v30_contains_any(
                norm,
                (
                    "حوكمة البيانات", "جودة البيانات",
                    "قاموس بيانات", "مصدر الحقيقة",
                    "ادارة المعرفة", "ملكية البيانات",
                ),
            )

        # For the remaining functional advisors, at least one explicit
        # activation phrase from the narrow vocabulary is already required.
        return True

    # Sector boundaries.
    if advisor_num == 34:
        # Ramadan/Quranic social activities alone do not make the NGO
        # a religious-da'wah portfolio.
        return _v30_contains_any(
            norm,
            (
                "دعوة", "دعوي", "تعليم ديني",
                "ضيوف الرحمن", "حجاج", "معتمرين", "عمرة",
            ),
        )

    if advisor_num == 35:
        return _v30_contains_any(
            norm,
            ("جمعية مهنية", "رابطة مهنية", "عضوية مهنية"),
        )

    # Sector advisors should have material evidence:
    # one explicit organizational/goal fact OR at least two evidence facts.
    org_goal_evidence = any(
        fid.startswith(("O", "N"))
        for fid in evidence_ids
    )

    return org_goal_evidence or len(evidence_ids) >= 2


def _v32_compact_fact(fact):
    fid = fact["fact_id"]
    text = _v30_arabic_public_text(
        _v30_trim(fact.get("text"), 260)
    )
    return {
        "fact_id": fid,
        "text": text,
    }


def _v32_build_candidates(advisors, facts):
    fact_map = {
        f["fact_id"]: f
        for f in facts
    }

    program_count = sum(
        1
        for f in facts
        if str(f.get("fact_id", "")).startswith("P")
    )

    candidates = []

    for advisor in advisors:
        num = int(advisor["advisor_id"])
        evidence_ids = _v32_candidate_evidence(
            num,
            facts,
        )

        if not _v32_candidate_allowed(
            num,
            evidence_ids,
            fact_map,
            program_count,
        ):
            continue

        # Keep a small evidence set to minimize prompt length.
        structural = [
            fid for fid in evidence_ids
            if fid.startswith("C")
        ]
        ordinary = [
            fid for fid in evidence_ids
            if not fid.startswith("C")
        ]

        selected_ids = (
            structural[:2]
            + ordinary[:4]
        )[:5]

        candidate_facts = [
            _v32_compact_fact(fact_map[fid])
            for fid in selected_ids
            if fid in fact_map
        ]

        candidates.append({
            "advisor_id": advisor["system_code"],
            "advisor_name": advisor.get("name_ar"),
            "advisor_class": (
                "SECTOR"
                if num >= 26
                else "FUNCTIONAL"
            ),
            "owned_outcome": _v30_arabic_public_text(
                _v30_trim(
                    advisor.get("owned_outcome"),
                    220,
                )
            ),
            "activation_when": [
                _v30_arabic_public_text(
                    _v30_trim(x, 115)
                )
                for x in (
                    advisor.get("activation_when") or []
                )[:4]
            ],
            "evidence": candidate_facts,
        })

    return candidates


RICH_V32_ADJUDICATOR_PROMPT = """
أنت الحكم النهائي السريع لملاءمة مجموعة مرشحين استشاريين في منظومة أثر.

مهم:
Python أجرى قبل هذه الجولة فحصًا أوليًا للأدلة وحدود التخصص.
أمامك الآن فقط مرشحون لديهم دليل محتمل.
مهمتك أن تمنح كل مرشح درجة نهائية دقيقة.

معنى الدرجات:
90-100 = ارتباط مباشر ومحوري جدًا.
80-89 = ارتباط مباشر وقوي.
70-79 = ارتباط واضح ومادي.
60-69 = ارتباط حقيقي لكنه أقل مركزية.
50-59 = ارتباط حقيقي مثبت لكنه محدود نسبيًا.
أقل من 50 = الدليل لا يكفي للترشيح النهائي.

للمستشار FUNCTIONAL:
لا تمنح 50 أو أكثر إلا إذا كانت الأدلة تثبت حاجة أو قرارًا أو تعقيدًا حاليًا يقع داخل OWNED_OUTCOME.
مجرد وجود المجال أو وجود برامج لا يكفي.
غياب معلومة لا يعتبر حاجة.
لا تخترع فجوة غير مذكورة.

للمستشار SECTOR:
يمكن أن يحصل على 50 أو أكثر إذا كانت الأدلة تثبت أن القطاع حاضر بصورة جوهرية أو متكررة في أهداف الجمعية أو برامجها، حتى دون وجود مشكلة.

قواعد:
- قيّم كل CANDIDATE بلا استثناء.
- لا تستخدم عددًا مستهدفًا للمستشارين.
- لا ترفع مستشارًا لأن تخصصه مفيد عمومًا.
- استخدم فقط EVIDENCE المرفقة بكل مرشح.
- إذا كان الدليل يخص مستشارًا آخر بصورة أوضح، اجعل الدرجة أقل من 50.

الإخراج فقط:
ADVISOR_ID|SCORE|EVIDENCE_IDS

مثال شكلي:
AOS-SP-13|82|C1,P3

- SCORE من 0 إلى 100.
- إذا SCORE >= 50 يجب ذكر من 1 إلى 3 معرفات من EVIDENCE الخاصة بنفس المرشح.
- إذا SCORE < 50 اكتب NONE.
- ممنوع الأسباب.
- ممنوع JSON.
- ممنوع Markdown.
- ممنوع أي شرح إضافي.
"""


def _v32_parse_results(text, candidates):
    candidate_map = {
        c["advisor_id"]: c
        for c in candidates
    }

    rows = {}

    cleaned = (
        str(text)
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()

        if not line or "|" not in line:
            continue

        parts = line.split("|", 2)

        if len(parts) != 3:
            continue

        code = parts[0].strip()

        if (
            code not in candidate_map
            or code in rows
        ):
            continue

        score_text = _v30_digits_to_ascii(
            parts[1]
        )

        m = re.search(
            r"\d+(?:\.\d+)?",
            score_text,
        )

        if not m:
            continue

        score = float(m.group())

        if score <= 1:
            score *= 100

        score = max(
            0.0,
            min(100.0, score),
        )

        allowed_ids = {
            e["fact_id"]
            for e in candidate_map[code]["evidence"]
        }

        returned_ids = []

        for token in re.findall(
            r"\b(?:O|N|P|C)\d+\b",
            _v30_digits_to_ascii(
                parts[2]
            ).upper(),
        ):
            if (
                token in allowed_ids
                and token not in returned_ids
            ):
                returned_ids.append(token)

        rows[code] = {
            "advisor_id": code,
            "score": round(
                score / 100.0,
                4,
            ),
            "evidence_ids": returned_ids[:3],
        }

    return rows


def _v32_program_name(text):
    m = re.search(
        r'(?:برنامج|مشروع)\s+"([^"]+)"',
        str(text),
    )

    if m:
        return _v30_arabic_public_text(
            m.group(1)
        )

    return ""


def _v32_evidence_label(fid, fact_map):
    fact = fact_map.get(fid)

    if not fact:
        return ""

    if fid == "C1":
        return "تعدد البرامج والمشاريع وتنوع مجالاتها"

    if fid == "C2":
        return "وجود برامج دورية وموسمية ومتكررة تحتاج إلى تنسيق تشغيلي"

    if fid == "C3":
        return "اعتماد تنفيذ عدد من البرامج على الشراكات والتعاون مع جهات متعددة"

    if fid == "C4":
        return "وجود منظومة تطوع فعلية تشمل متطوعين وفرصًا تطوعية"

    if fid.startswith("P"):
        name = _v32_program_name(
            fact.get("text")
        )

        if name:
            return f"برنامج «{name}»"

    source = _v30_arabic_public_text(
        fact.get("source") or ""
    )

    if source:
        return source

    return _v31_safe_cut(
        fact.get("text"),
        100,
    )


def _v32_reason(advisor, evidence_ids, fact_map):
    name = _v30_arabic_public_text(
        advisor.get("name_ar")
        or "المستشار"
    )

    labels = []

    for fid in evidence_ids:
        label = _v32_evidence_label(
            fid,
            fact_map,
        )

        if (
            label
            and label not in labels
        ):
            labels.append(label)

    labels = labels[:2]

    num = int(
        advisor["advisor_id"]
    )

    if not labels:
        return (
            f"ترتبط خبرة {name} باحتياج موثق في بيانات الجمعية "
            "ويقع هذا الاحتياج مباشرة ضمن نطاق اختصاصه."
        )

    if len(labels) == 1:
        evidence_phrase = labels[0]
    else:
        evidence_phrase = (
            labels[0]
            + "، إلى جانب "
            + labels[1]
        )

    if num >= 26:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الأدلة حضور هذا القطاع بصورة فعلية في أنشطة الجمعية، "
            "مما يجعل خبرة المستشار مرتبطة مباشرة بالمحفظة الحالية."
        )
    else:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الأدلة وجود حاجة أو تعقيد حالي يقع مباشرة ضمن نطاق اختصاصه، "
            "دون افتراض فجوات غير مذكورة في البيانات."
        )

    return _v30_arabic_public_text(
        reason
    )


def advisory_match_rich_v32(job_input):
    started = time.time()

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = _v31_build_facts(
        organization,
        programs,
    )

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    advisor_by_code = {
        advisor["system_code"]: advisor
        for advisor in advisors
    }

    candidates = _v32_build_candidates(
        advisors,
        facts,
    )

    print(
        f"Rich v32: {len(candidates)} evidence-backed candidates "
        "from 35 advisors; running ONE adjudication generation...",
        flush=True,
    )

    if not candidates:
        return {"ranked": []}

    raw, input_tokens = _v18_generate_text(
        RICH_V32_ADJUDICATOR_PROMPT,
        {
            "candidates": candidates,
        },
        RICH_V32_MAX_NEW_TOKENS,
    )

    rows = _v32_parse_results(
        raw,
        candidates,
    )

    matches = []

    for candidate in candidates:
        code = candidate["advisor_id"]
        item = rows.get(code)

        if not item:
            continue

        if item["score"] < RICH_V32_MIN_PUBLIC_SCORE:
            continue

        if not item.get("evidence_ids"):
            continue

        advisor = advisor_by_code.get(code)

        if not advisor:
            continue

        matches.append({
            "advisor_id": code,
            "score": item["score"],
            "reason": _v32_reason(
                advisor,
                item["evidence_ids"],
                fact_map,
            ),
        })

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    elapsed = round(
        time.time() - started,
        2,
    )

    print(
        f"Rich v32 complete in {elapsed}s. "
        f"Input tokens={input_tokens}. "
        f"Candidates={len(candidates)}. "
        f"Returned={len(matches)}. "
        f"Minimum={RICH_V34_MIN_ADVISORS}. "
        "Model generations=1.",
        flush=True,
    )

    return {
        "ranked": matches
    }


# ---------------------------------------------------------------------
# Rich AI Router v33 — calibrated one-pass production router
#
# Fixes v32 without adding more model calls:
# - removes ambiguous lexical triggers that created false candidates
# - ranks evidence by specificity (program facts > generic descriptions)
# - hardens functional activation boundaries
# - protects strong structural/sector evidence from accidental under-scoring
# - keeps ONE Qwen generation only
# - public Arabic reasons remain deterministic
# ---------------------------------------------------------------------

RICH_V33_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V33_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V33_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V33_MAX_NEW_TOKENS", "240")
)

# Refined activation vocabulary.
# Ambiguous single words such as "فرص" and generic "بيئة" are intentionally removed.
_V33_FUNCTIONAL_TERMS = dict(_V32_FUNCTIONAL_TERMS)
_V33_FUNCTIONAL_TERMS.update({
    11: (
        "مبادرة جديدة", "برنامج جديد", "مشروع جديد",
        "تصميم مبادرة", "تصميم المبادرات", "تصميم برنامج",
        "إعادة تصميم", "إعادة تصميم المبادرات",
        "تحسين تصميم", "تطوير تصميم المبادرات",
        "مبادرات مستقبلية", "مشروعات مستقبلية",
    ),
    15: (
        "قياس الأثر", "إدارة الأثر", "تقييم الأثر",
        "متابعة وتقييم", "تقييم النتائج", "نظرية التغيير",
        "التعلم المؤسسي", "أثر البرامج", "قياس النتائج",
        "نتائج طويلة الأجل", "استدامة النتائج",
        "استدامة دخل المستفيدين", "قرار التوسع", "جدوى التوسع",
        "ربط التدريب بالتوظيف",
    ),
    18: (
        "استدامة مالية", "تنمية الموارد", "تنويع الإيرادات",
        "تنويع مصادر الدخل", "فجوة تمويل", "جمع التبرعات",
        "تبرعات", "مانحين", "مانح", "منح", "مصادر دخل",
        "مصدر تمويل", "مصدر تمويل رئيسي", "اعتماد على مانح",
        "تركيز التمويل",
    ),
    20: (
        "هيكل تنظيمي", "إعادة هيكلة", "موظفين", "موظف",
        "قوى عاملة", "عبء العمل", "جدارات",
        "أدوار وظيفية", "أداء الموظفين", "موارد بشرية",
        "فريق العمل", "كادر", "كوادر",
    ),
    3: (
        "تحليل داخلي", "تحليل خارجي", "الفرص والتهديدات",
        "فرص استراتيجية", "تهديدات استراتيجية",
        "عوامل خارجية", "عوامل داخلية",
        "اتجاهات السوق", "اتجاهات المجتمع",
    ),
    21: (
        "تحسين العمليات", "إعادة تصميم العمليات",
        "إجراءات", "إجراء تشغيلي", "تدفق الخدمة",
        "اختناق", "تأخير", "هدر", "إعادة عمل",
        "رحلة الخدمة", "زمن الانتظار",
    ),
    25: (
        "حوكمة البيانات", "جودة البيانات", "قاموس بيانات",
        "مصدر الحقيقة", "إدارة المعرفة", "ملكية البيانات",
        "صلاحيات البيانات", "مشكلة التقارير",
    ),
})

_V33_SECTOR_TERMS = dict(_V32_SECTOR_TERMS)
_V33_SECTOR_TERMS.update({
    30: (
        "بيئي", "بيئية", "استدامة بيئية", "تشجير",
        "نفايات", "إعادة تدوير", "تلوث", "مناخ",
        "تنوع حيوي", "اقتصاد دائري",
    ),
    32: (
        "حقوق كبار السن", "حقوق المستفيدين", "توعية بالحقوق",
        "مناصرة", "دعم قانوني", "خدمة قانونية",
        "حماية الحقوق",
    ),
})


def _v33_term_hits(text, terms):
    norm = _v31_norm_ar(text)

    hits = []
    for term in terms:
        nterm = _v31_norm_ar(term)
        if nterm and nterm in norm:
            hits.append(term)

    return hits


def _v33_fact_priority(advisor_num, fact, terms):
    """
    Rank evidence so specific program/goal facts beat broad generic descriptions.
    """
    fid = str(fact.get("fact_id", ""))
    text = fact.get("text", "")
    hits = _v33_term_hits(text, terms)

    score = len(hits) * 4.0

    if fid.startswith("C"):
        score += 12.0
    elif fid.startswith("P"):
        score += 7.0
    elif fid.startswith("N"):
        score += 5.0
    elif fid in {"O4", "O5", "O6"}:
        score += 2.0
    elif fid.startswith("O"):
        score += 1.0

    # Sector advisors should strongly prefer actual programs/projects.
    if advisor_num >= 26 and fid.startswith("P"):
        score += 4.0

    # Functional structural evidence is intentionally strong.
    structural = {
        4: "C3",
        12: "C2",
        13: "C1",
        33: "C4",
    }
    if structural.get(advisor_num) == fid:
        score += 20.0

    return score


def _v33_candidate_evidence(advisor_num, facts):
    if advisor_num <= 25:
        terms = _V33_FUNCTIONAL_TERMS.get(advisor_num, ())
    else:
        terms = _V33_SECTOR_TERMS.get(advisor_num, ())

    fact_map = {
        f["fact_id"]: f
        for f in facts
    }

    evidence_ids = []

    for fid in _V32_STRUCTURAL_FACTS.get(advisor_num, ()):
        if fid in fact_map:
            evidence_ids.append(fid)

    ranked = []

    for fact in facts:
        fid = fact.get("fact_id")

        if fid in {"O1", "O2"}:
            continue

        hits = _v33_term_hits(
            fact.get("text", ""),
            terms,
        )

        if not hits:
            continue

        ranked.append((
            _v33_fact_priority(
                advisor_num,
                fact,
                terms,
            ),
            fid,
        ))

    ranked.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for _, fid in ranked:
        if fid not in evidence_ids:
            evidence_ids.append(fid)

    return evidence_ids


def _v33_candidate_allowed(advisor_num, evidence_ids, fact_map, program_count):
    if not evidence_ids:
        return False

    evidence_text = " ".join(
        fact_map[fid]["text"]
        for fid in evidence_ids
        if fid in fact_map
    )

    norm = _v31_norm_ar(evidence_text)

    # Strong structural activations.
    if advisor_num == 4:
        return (
            "C3" in evidence_ids
            or _v30_contains_any(
                norm,
                ("شراكة", "شراكات", "شريك", "شركاء"),
            )
        )

    if advisor_num == 12:
        return "C2" in evidence_ids

    if advisor_num == 13:
        return (
            "C1" in evidence_ids
            and program_count >= 4
        )

    if advisor_num == 33:
        return "C4" in evidence_ids

    # False-positive prone functional advisors.
    if advisor_num == 1:
        return _v30_contains_any(
            norm,
            (
                "قرار تنفيذي", "أولويات تنفيذية",
                "إعادة تنظيم", "إعادة هيكلة",
                "توسع مؤسسي", "قيادة تنفيذية",
            ),
        )

    if advisor_num == 2:
        return _v30_contains_any(
            norm,
            (
                "نضج مؤسسي", "تشخيص مؤسسي",
                "تقييم مؤسسي", "جاهزية مؤسسية",
                "خط أساس مؤسسي", "فجوة مؤسسية",
            ),
        )

    if advisor_num == 3:
        return _v30_contains_any(
            norm,
            (
                "تحليل داخلي", "تحليل خارجي",
                "الفرص والتهديدات",
                "فرص استراتيجية", "تهديدات استراتيجية",
                "عوامل خارجية", "عوامل داخلية",
            ),
        )

    if advisor_num == 5:
        return _v30_contains_any(
            norm,
            (
                "تحول مؤسسي", "تغيير مؤسسي",
                "إعادة هيكلة", "إعادة تنظيم",
                "إدارة التغيير", "مقاومة التغيير",
            ),
        )

    if advisor_num == 6:
        return (
            _v30_contains_any(
                norm,
                (
                    "جودة الخدمة", "جودة الخدمات",
                    "معايير الجودة", "رضا المستفيد",
                    "شكاوى", "عدم مطابقة",
                ),
            )
            and _v30_contains_any(
                norm,
                (
                    "تحسين", "تطوير", "مشكلة",
                    "ضعف", "فجوة", "شكوى", "شكاوى",
                ),
            )
        )

    if advisor_num == 7:
        return _v30_contains_any(
            norm,
            (
                "مخاطر", "خطر", "استمرارية الأعمال",
                "طوارئ", "أزمة", "تعطل",
            ),
        )

    if advisor_num == 8:
        return _v30_contains_any(
            norm,
            (
                "مراجعة استراتيجية", "تحديث الاستراتيجية",
                "مفاضلة استراتيجية", "خيارات استراتيجية",
                "توجه استراتيجي جديد",
            ),
        )

    if advisor_num == 9:
        return _v30_contains_any(
            norm,
            (
                "تطوير الرؤية", "تحديث الرؤية",
                "تطوير الرسالة", "تحديث الرسالة",
                "إعادة صياغة الرؤية", "تطوير الهوية",
            ),
        )

    if advisor_num == 10:
        return _v30_contains_any(
            norm,
            (
                "قضايا استراتيجية", "إعادة صياغة الأهداف",
                "ترتيب الأولويات الاستراتيجية",
                "تعارض الأهداف",
            ),
        )

    if advisor_num == 11:
        return _v30_contains_any(
            norm,
            (
                "مبادرة جديدة", "برنامج جديد", "مشروع جديد",
                "تصميم مبادرة", "تصميم المبادرات", "تصميم برنامج",
                "إعادة تصميم", "إعادة تصميم المبادرات",
                "تحسين تصميم", "تطوير تصميم المبادرات",
                "مبادرات مستقبلية",
            ),
        )

    if advisor_num == 14:
        return _v30_contains_any(
            norm,
            (
                "مؤشرات أداء", "لوحة مؤشرات",
                "لوحة قيادة", "قياس الأداء",
                "منظومة الأداء", "مستهدفات أداء",
            ),
        )

    if advisor_num == 15:
        return _v30_contains_any(
            norm,
            (
                "قياس الأثر", "إدارة الأثر",
                "تقييم الأثر", "متابعة وتقييم",
                "تقييم النتائج", "نظرية التغيير",
                "أثر البرامج", "قياس النتائج",
                "نتائج طويلة الأجل", "استدامة النتائج",
                "استدامة دخل المستفيدين",
                "قرار التوسع", "جدوى التوسع",
                "ربط التدريب بالتوظيف",
            ),
        )

    if advisor_num == 16:
        # Explicit statements that governance/authority/compliance problems do
        # NOT exist must not activate the governance advisor.
        strong_positive = (
            "فجوة حوكمة", "تحسين الحوكمة", "مخالفة",
            "تعارض مصالح", "ضعف الحوكمة", "مشكلة حوكمة",
            "قصور الحوكمة",
        )
        if _v30_contains_any(norm, strong_positive):
            return True

        for fid in evidence_ids:
            fact_text = _v31_norm_ar(
                (fact_map.get(fid) or {}).get("text", "")
            )
            if not _v30_contains_any(
                fact_text,
                ("امتثال", "صلاحيات", "حوكمة"),
            ):
                continue
            if _v30_contains_any(
                fact_text,
                (
                    "لا توجد", "لا يوجد", "لا تظهر", "لا يظهر",
                    "دون مشكلة", "بدون مشكلة", "لا توجد مشكلة",
                    "لا توجد فجوة", "لا يوجد قصور",
                ),
            ):
                continue
            return True

        return False

    if advisor_num == 17:
        return _v30_contains_any(
            norm,
            (
                "موازنة", "ميزانية", "تكلفة",
                "سيولة", "تدفق نقدي",
                "انحراف مالي", "إعادة تخصيص مالي",
            ),
        )

    if advisor_num == 18:
        return _v30_contains_any(
            norm,
            (
                "استدامة مالية", "تنمية الموارد",
                "تنويع الإيرادات", "تنويع مصادر الدخل",
                "فجوة تمويل", "جمع التبرعات",
                "مانحين", "مانح", "منح",
                "مصادر دخل", "مصدر تمويل",
                "اعتماد على مانح", "تركيز التمويل",
            ),
        )

    if advisor_num == 19:
        return _v30_contains_any(
            norm,
            (
                "وقف", "أوقاف", "استثمار اجتماعي",
                "استثمار مؤثر", "سياسة استثمار",
            ),
        )

    if advisor_num == 20:
        # Do NOT treat beneficiary training/employment as an internal HR need.
        # AOS-FG-20 requires evidence about the organization's own workforce,
        # structure, roles, capacity, or employee performance.
        return _v30_contains_any(
            norm,
            (
                "هيكل تنظيمي", "إعادة هيكلة",
                "موظف", "موظفين", "قوى عاملة",
                "عبء العمل", "جدارات", "أدوار وظيفية",
                "أداء الموظفين", "موارد بشرية",
                "فريق العمل", "كادر", "كوادر",
            ),
        )

    if advisor_num == 21:
        process = _v30_contains_any(
            norm,
            (
                "إجراءات", "إجراء تشغيلي",
                "تدفق الخدمة", "اختناق",
                "تأخير", "هدر", "إعادة عمل",
                "تحسين العمليات", "رحلة الخدمة",
            ),
        )
        actual_need = _v30_contains_any(
            norm,
            (
                "تحسين", "إعادة تصميم", "اختناق",
                "تأخير", "هدر", "إعادة عمل",
                "مشكلة", "ضعف",
            ),
        )
        return process and actual_need

    if advisor_num == 22:
        return _v30_contains_any(
            norm,
            (
                "استراتيجية اتصال", "اتصال مؤسسي",
                "سمعة", "علاقات عامة",
                "صورة ذهنية", "أزمة اتصال",
            ),
        )

    if advisor_num == 23:
        return _v30_contains_any(
            norm,
            (
                "تسويق رقمي", "حملة رقمية",
                "إعلانات", "اكتساب",
                "تحويل", "إعادة استهداف",
            ),
        )

    if advisor_num == 24:
        return _v30_contains_any(
            norm,
            (
                "تحول رقمي", "ذكاء اصطناعي",
                "أتمتة", "تكامل الأنظمة",
                "رقمنة", "منصة رقمية",
            ),
        )

    if advisor_num == 25:
        return _v30_contains_any(
            norm,
            (
                "حوكمة البيانات", "جودة البيانات",
                "قاموس بيانات", "مصدر الحقيقة",
                "إدارة المعرفة", "ملكية البيانات",
            ),
        )

    # Sector-specific hard boundaries.
    if advisor_num == 30:
        return _v30_contains_any(
            norm,
            (
                "بيئي", "بيئية", "استدامة بيئية",
                "تشجير", "نفايات", "إعادة تدوير",
                "تلوث", "مناخ", "تنوع حيوي",
            ),
        )

    if advisor_num == 31:
        return _v30_contains_any(
            norm,
            (
                "إسكان", "سكن", "سكني",
                "ترميم", "إيجار", "منازل",
                "تنمية محلية",
            ),
        )

    if advisor_num == 34:
        return _v30_contains_any(
            norm,
            (
                "دعوة", "دعوي", "تعليم ديني",
                "ضيوف الرحمن", "حجاج",
                "معتمرين", "عمرة",
            ),
        )

    if advisor_num == 35:
        return _v30_contains_any(
            norm,
            (
                "جمعية مهنية", "رابطة مهنية",
                "عضوية مهنية",
            ),
        )

    # Other sector advisors:
    # one explicit organization/goal fact or at least two concrete facts.
    org_goal = any(
        fid.startswith(("O", "N"))
        for fid in evidence_ids
    )

    program_facts = sum(
        1
        for fid in evidence_ids
        if fid.startswith("P")
    )

    return org_goal or program_facts >= 2


def _v33_strength(advisor_num, evidence_ids):
    """
    Generic evidence calibration, independent from advisor count.
    """
    if advisor_num in {4, 12, 13, 33}:
        structural = {
            4: "C3",
            12: "C2",
            13: "C1",
            33: "C4",
        }
        if structural[advisor_num] in evidence_ids:
            return "STRONG"

    program_count = sum(
        1
        for fid in evidence_ids
        if fid.startswith("P")
    )

    org_goal = any(
        fid.startswith(("O", "N"))
        for fid in evidence_ids
    )

    if advisor_num >= 26:
        if program_count >= 2:
            return "STRONG"
        if program_count >= 1 and org_goal:
            return "STRONG"
        if org_goal:
            return "MEDIUM"

    return "MEDIUM"


def _v33_build_candidates(advisors, facts):
    fact_map = {
        f["fact_id"]: f
        for f in facts
    }

    program_count = sum(
        1
        for f in facts
        if str(f.get("fact_id", "")).startswith("P")
    )

    candidates = []

    for advisor in advisors:
        num = int(advisor["advisor_id"])

        evidence_ids = _v33_candidate_evidence(
            num,
            facts,
        )

        if not _v33_candidate_allowed(
            num,
            evidence_ids,
            fact_map,
            program_count,
        ):
            continue

        if num <= 25:
            terms = _V33_FUNCTIONAL_TERMS.get(num, ())
        else:
            terms = _V33_SECTOR_TERMS.get(num, ())

        ranked_ids = sorted(
            evidence_ids,
            key=lambda fid: _v33_fact_priority(
                num,
                fact_map[fid],
                terms,
            ) if fid in fact_map else 0.0,
            reverse=True,
        )

        selected_ids = ranked_ids[:5]

        candidates.append({
            "advisor_id": advisor["system_code"],
            "advisor_name": advisor.get("name_ar"),
            "advisor_class": (
                "SECTOR"
                if num >= 26
                else "FUNCTIONAL"
            ),
            "owned_outcome": _v30_arabic_public_text(
                _v30_trim(
                    advisor.get("owned_outcome"),
                    220,
                )
            ),
            "evidence_strength": _v33_strength(
                num,
                selected_ids,
            ),
            "evidence": [
                _v32_compact_fact(
                    fact_map[fid]
                )
                for fid in selected_ids
                if fid in fact_map
            ],
        })

    return candidates


RICH_V33_ADJUDICATOR_PROMPT = """
أنت الحكم النهائي لملاءمة مرشحين استشاريين في منظومة أثر.

كل مرشح اجتاز مسبقًا فحصًا صارمًا لحدود تخصصه، وسترى:
- نوع المستشار.
- المخرج الذي يملكه.
- قوة الدليل الأولية.
- أدلة حقيقية فقط.

امنح كل مرشح درجة نهائية.

الدرجات:
90-100 = ارتباط محوري جدًا.
80-89 = ارتباط مباشر وقوي.
70-79 = ارتباط واضح ومادي.
60-69 = ارتباط حقيقي مثبت لكنه أقل مركزية.
50-59 = ارتباط مثبت لكنه محدود نسبيًا.
أقل من 50 = لا يكفي للترشيح النهائي.

FUNCTIONAL:
يجب أن تثبت الأدلة حاجة أو قرارًا أو تعقيدًا حاليًا داخل ملكية المستشار.
لا تحوّل وجود المجال إلى فجوة.

SECTOR:
يكفي أن تثبت الأدلة حضور القطاع بصورة جوهرية أو متكررة في أهداف الجمعية أو برامجها.

قوة الدليل:
- STRONG تعني أن النظام وجد دليلًا هيكليًا أو عدة وقائع مستقلة مباشرة.
  لا تخفضه تحت 50 إلا إذا كانت الأدلة نفسها لا تخص المستشار بوضوح.
- MEDIUM يحتاج حكمك المعتاد.

لا يوجد عدد مستهدف للمستشارين.
قيّم كل مرشح بلا استثناء.

الإخراج فقط:
ADVISOR_ID|SCORE|EVIDENCE_IDS

- إذا SCORE >= 50 اذكر من 1 إلى 3 معرفات من أدلة نفس المرشح.
- إذا SCORE < 50 اكتب NONE.
- ممنوع الأسباب وJSON وMarkdown وأي شرح إضافي.
"""


def _v33_reason(advisor, evidence_ids, fact_map):
    """
    Concise Arabic, using specific evidence rather than labels such as
    'detailed description' whenever possible.
    """
    name = _v30_arabic_public_text(
        advisor.get("name_ar") or "المستشار"
    )

    labels = []

    for fid in evidence_ids:
        fact = fact_map.get(fid)
        if not fact:
            continue

        if fid in {"C1", "C2", "C3", "C4"}:
            label = _v32_evidence_label(
                fid,
                fact_map,
            )
        elif fid.startswith("P"):
            pname = _v32_program_name(
                fact.get("text")
            )
            label = (
                f"برنامج «{pname}»"
                if pname
                else _v31_safe_cut(
                    fact.get("text"),
                    105,
                )
            )
        else:
            # Use an actual source-grounded clause, not a generic source label.
            clean = _v30_arabic_public_text(
                fact.get("text")
            )
            clauses = re.split(
                r"[؛.!؟]",
                clean,
            )
            label = ""
            for clause in clauses:
                clause = _v31_safe_cut(
                    clause,
                    120,
                )
                if len(clause) >= 20:
                    label = clause
                    break

            if not label:
                label = _v31_safe_cut(
                    clean,
                    120,
                )

        if (
            label
            and label not in labels
        ):
            labels.append(label)

        if len(labels) == 2:
            break

    if not labels:
        return (
            f"يرتبط {name} باحتياج موثق في بيانات الجمعية "
            "ويقع هذا الاحتياج مباشرة ضمن نطاق اختصاصه."
        )

    evidence_phrase = labels[0]

    if len(labels) == 2:
        evidence_phrase += "، إلى جانب " + labels[1]

    num = int(advisor["advisor_id"])

    if num >= 26:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الوقائع حضور هذا القطاع بصورة فعلية في برامج الجمعية وأهدافها، "
            "مما يجعل خبرة المستشار مرتبطة مباشرة بالمحفظة الحالية."
        )
    else:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الوقائع وجود حاجة أو تعقيد حالي يقع مباشرة ضمن نطاق اختصاصه، "
            "دون افتراض فجوات غير مذكورة في البيانات."
        )

    return _v30_arabic_public_text(
        reason
    )


def advisory_match_rich_v33(job_input):
    started = time.time()

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = _v31_build_facts(
        organization,
        programs,
    )

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    advisor_by_code = {
        advisor["system_code"]: advisor
        for advisor in advisors
    }

    candidates = _v33_build_candidates(
        advisors,
        facts,
    )

    print(
        f"Rich v33: {len(candidates)} strictly evidence-backed candidates "
        "from 35 advisors; running ONE adjudication generation...",
        flush=True,
    )

    if not candidates:
        return {"ranked": []}

    raw, input_tokens = _v18_generate_text(
        RICH_V33_ADJUDICATOR_PROMPT,
        {
            "candidates": candidates,
        },
        RICH_V33_MAX_NEW_TOKENS,
    )

    rows = _v32_parse_results(
        raw,
        candidates,
    )

    candidate_by_code = {
        c["advisor_id"]: c
        for c in candidates
    }

    matches = []

    for candidate in candidates:
        code = candidate["advisor_id"]
        item = rows.get(code)

        ai_score = (
            item["score"]
            if item
            else 0.0
        )

        strength = candidate.get(
            "evidence_strength",
            "MEDIUM",
        )

        # Strong evidence receives a conservative floor so an obvious,
        # directly-grounded advisor is not lost due generation variance.
        evidence_floor = (
            0.60
            if strength == "STRONG"
            else 0.0
        )

        final_score = max(
            ai_score,
            evidence_floor,
        )

        if final_score < RICH_V33_MIN_PUBLIC_SCORE:
            continue

        if (
            item
            and item.get("evidence_ids")
        ):
            evidence_ids = item["evidence_ids"]
        elif strength == "STRONG":
            evidence_ids = [
                e["fact_id"]
                for e in candidate["evidence"][:2]
            ]
        else:
            continue

        advisor = advisor_by_code.get(code)

        if not advisor:
            continue

        matches.append({
            "advisor_id": code,
            "score": round(
                final_score,
                4,
            ),
            "reason": _v33_reason(
                advisor,
                evidence_ids,
                fact_map,
            ),
        })

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    elapsed = round(
        time.time() - started,
        2,
    )

    print(
        f"Rich v33 complete in {elapsed}s. "
        f"Input tokens={input_tokens}. "
        f"Candidates={len(candidates)}. "
        f"Returned={len(matches)}. "
        "Model generations=1.",
        flush=True,
    )

    return {
        "ranked": matches
    }


# ---------------------------------------------------------------------
# Rich AI Router v34 — final one-pass production hardening
#
# Keeps the successful v33 architecture:
#   - ONE Qwen generation
#   - score threshold >= 0.50
#   - deterministic Arabic reasons
#
# Final hardening:
#   1) Re-validates the exact evidence for every returned advisor.
#   2) Uses the strongest advisor-specific evidence for public reasons,
#      instead of trusting whichever evidence IDs the LLM happened to cite.
#   3) Tightens false-positive-prone sector boundaries (environment,
#      development/housing, rights, religion, professional associations).
#   4) Extracts the exact relevant Arabic clause from broad organization
#      facts, so reasons do not show truncated generic text.
# ---------------------------------------------------------------------

RICH_V34_MIN_PUBLIC_SCORE = float(
    os.environ.get("RICH_V34_MIN_PUBLIC_SCORE", "0.50")
)

RICH_V34_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_V34_MAX_NEW_TOKENS", "240")
)

# Product requirement: never return fewer than this number when the payload
# contains enough evidence-backed advisors. The final fallback below also
# fills from the closest fact-backed advisors so the API contract stays stable.
RICH_V34_MIN_ADVISORS = int(
    os.environ.get("RICH_V34_MIN_ADVISORS", "6")
)

# Backend/UI contract caps the candidate list while preserving the user's
# explicit minimum of six evidence-backed advisors.
RICH_V34_MAX_ADVISORS = int(
    os.environ.get("RICH_V34_MAX_ADVISORS", "16")
)


def _v34_terms_for_advisor(advisor_num):
    if advisor_num <= 25:
        return _V33_FUNCTIONAL_TERMS.get(advisor_num, ())
    return _V33_SECTOR_TERMS.get(advisor_num, ())


def _v34_direct_fact_match(advisor_num, fact):
    fid = str(fact.get("fact_id", ""))
    text = _v31_norm_ar(fact.get("text", ""))

    # Structural facts are intentionally advisor-specific.
    structural = {
        4: "C3",
        12: "C2",
        13: "C1",
        33: "C4",
    }

    if structural.get(advisor_num) == fid:
        return True

    # Functional hard boundaries added after recovery testing.
    if advisor_num == 15:
        return _v30_contains_any(
            text,
            (
                "قياس الأثر", "إدارة الأثر", "تقييم الأثر",
                "متابعة وتقييم", "تقييم النتائج", "نظرية التغيير",
                "أثر البرامج", "قياس النتائج", "نتائج طويلة الأجل",
                "استدامة النتائج", "استدامة دخل المستفيدين",
                "قرار التوسع", "جدوى التوسع", "ربط التدريب بالتوظيف",
            ),
        )

    if advisor_num == 18:
        return _v30_contains_any(
            text,
            (
                "استدامة مالية", "تنمية الموارد", "تنويع الإيرادات",
                "تنويع مصادر الدخل", "فجوة تمويل", "جمع التبرعات",
                "مانحين", "مانح", "منح", "مصادر دخل",
                "مصدر تمويل", "اعتماد على مانح", "تركيز التمويل",
            ),
        )

    if advisor_num == 16:
        if _v30_contains_any(
            text,
            (
                "لا توجد", "لا يوجد", "لا تظهر", "لا يظهر",
                "دون مشكلة", "بدون مشكلة", "لا توجد مشكلة",
                "لا توجد فجوة", "لا يوجد قصور",
            ),
        ) and _v30_contains_any(
            text,
            ("امتثال", "صلاحيات", "حوكمة"),
        ):
            return False
        return _v30_contains_any(
            text,
            (
                "فجوة حوكمة", "تحسين الحوكمة", "امتثال",
                "مخالفة", "صلاحيات", "تعارض مصالح",
                "ضعف الحوكمة", "مشكلة حوكمة", "قصور الحوكمة",
            ),
        )

    if advisor_num == 20:
        return _v30_contains_any(
            text,
            (
                "هيكل تنظيمي", "إعادة هيكلة", "موظف", "موظفين",
                "قوى عاملة", "عبء العمل", "جدارات", "أدوار وظيفية",
                "أداء الموظفين", "موارد بشرية", "فريق العمل",
                "كادر", "كوادر",
            ),
        )

    # Hard boundaries for the most false-positive-prone sectors.
    if advisor_num == 30:
        return _v30_contains_any(
            text,
            (
                "بيئي", "بيئية", "استدامة بيئية",
                "تشجير", "نفايات", "اعادة تدوير",
                "تلوث", "مناخ", "تنوع حيوي",
                "اقتصاد دائري",
            ),
        )

    if advisor_num == 31:
        return _v30_contains_any(
            text,
            (
                "اسكان", "سكن", "سكني", "سكنية",
                "ترميم منزل", "ترميم المنازل",
                "ايجار سكن", "منازل",
                "تنمية محلية", "تنمية مجتمعية",
                "سبل العيش", "تمكين اقتصادي",
            ),
        )

    if advisor_num == 32:
        return _v30_contains_any(
            text,
            (
                "حقوق كبار السن", "حقوق المستفيدين",
                "توعية بالحقوق", "دعم الحقوق",
                "مناصرة", "دعم قانوني",
                "خدمة قانونية", "حماية الحقوق",
            ),
        )

    if advisor_num == 34:
        return _v30_contains_any(
            text,
            (
                "دعوة", "دعوي", "تعليم ديني",
                "ضيوف الرحمن", "حجاج",
                "معتمرين", "عمرة",
            ),
        )

    if advisor_num == 35:
        return _v30_contains_any(
            text,
            (
                "جمعية مهنية", "رابطة مهنية",
                "عضوية مهنية", "اعضاء مهنيين",
                "تطوير مهني", "لجان مهنية",
            ),
        )

    terms = _v34_terms_for_advisor(advisor_num)
    return bool(_v33_term_hits(text, terms))


def _v34_verified_evidence(advisor, candidate, fact_map):
    advisor_num = int(advisor["advisor_id"])
    terms = _v34_terms_for_advisor(advisor_num)

    valid = []

    for evidence in candidate.get("evidence", []):
        fid = evidence.get("fact_id")
        fact = fact_map.get(fid)

        if not fact:
            continue

        if not _v34_direct_fact_match(
            advisor_num,
            fact,
        ):
            continue

        priority = _v33_fact_priority(
            advisor_num,
            fact,
            terms,
        )

        valid.append(
            (priority, fid)
        )

    valid.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        fid
        for _, fid in valid[:3]
    ]


def _v34_matching_clause(advisor_num, text):
    clean = _v30_arabic_public_text(text)
    if not clean:
        return ""

    terms = [
        _v31_norm_ar(term)
        for term in _v34_terms_for_advisor(advisor_num)
        if term
    ]

    # Add strict sector phrases that may not be in the broad vocabulary.
    if advisor_num == 31:
        terms.extend([
            "اسكان", "سكن", "ترميم", "ايجار",
            "تنمية محلية", "تنمية مجتمعية",
            "سبل العيش", "تمكين اقتصادي",
        ])
    elif advisor_num == 32:
        terms.extend([
            "حقوق كبار السن", "توعية بالحقوق",
            "دعم الحقوق", "مناصرة", "قانوني",
        ])

    # Split broad organization facts into meaningful clauses and return
    # the clause that actually contains the advisor-specific evidence.
    clauses = re.split(
        r"[؛.!؟\n]|،(?=\s)",
        clean,
    )

    for clause in clauses:
        clause = re.sub(r"\s+", " ", clause).strip()
        if len(clause) < 8:
            continue

        norm_clause = _v31_norm_ar(clause)

        if any(term in norm_clause for term in terms):
            return _v31_safe_cut(
                clause,
                145,
            )

    return _v31_safe_cut(
        clean,
        135,
    )


def _v34_evidence_label(advisor_num, fid, fact_map):
    fact = fact_map.get(fid)
    if not fact:
        return ""

    if fid == "C1":
        return "تعدد البرامج والمشاريع وتنوع مجالاتها"

    if fid == "C2":
        return "وجود برامج دورية وموسمية ومتكررة تحتاج إلى تنسيق تشغيلي"

    if fid == "C3":
        return "اعتماد تنفيذ عدد من البرامج على الشراكات والتعاون مع جهات متعددة"

    if fid == "C4":
        return "وجود منظومة تطوع فعلية تشمل متطوعين وفرصًا تطوعية"

    if fid.startswith("P"):
        pname = _v32_program_name(
            fact.get("text")
        )
        if pname:
            return f"برنامج «{pname}»"

    return _v34_matching_clause(
        advisor_num,
        fact.get("text", ""),
    )


def _v34_reason(advisor, evidence_ids, fact_map):
    advisor_num = int(advisor["advisor_id"])

    name = _v30_arabic_public_text(
        advisor.get("name_ar")
        or "المستشار"
    )

    labels = []

    for fid in evidence_ids:
        label = _v34_evidence_label(
            advisor_num,
            fid,
            fact_map,
        )

        if (
            label
            and label not in labels
        ):
            labels.append(label)

        if len(labels) == 2:
            break

    if not labels:
        return (
            f"يرتبط {name} باحتياج موثق في بيانات الجمعية "
            "ويقع هذا الاحتياج مباشرة ضمن نطاق اختصاصه."
        )

    evidence_phrase = labels[0]

    if len(labels) == 2:
        evidence_phrase += "، إلى جانب " + labels[1]

    if advisor_num >= 26:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الوقائع حضور هذا القطاع بصورة فعلية ومادية في عمل الجمعية، "
            "مما يجعل خبرة المستشار مرتبطة مباشرة بالمحفظة الحالية."
        )
    else:
        reason = (
            f"يرتبط {name} بالجمعية استنادًا إلى {evidence_phrase}. "
            "وتثبت هذه الوقائع وجود حاجة أو تعقيد حالي يقع مباشرة ضمن نطاق اختصاصه، "
            "دون افتراض فجوات غير مذكورة في البيانات."
        )

    return _v30_arabic_public_text(
        reason
    )


def _v34_explicit_need_floor(advisor_num, evidence_ids, fact_map, strength):
    """Deterministic floor for explicit high-signal needs.

    This prevents the one-pass adjudicator from accidentally suppressing a
    clearly stated impact-measurement or funding-diversification need.
    """
    floor = 0.60 if strength == "STRONG" else 0.0

    evidence_text = " ".join(
        str((fact_map.get(fid) or {}).get("text", ""))
        for fid in evidence_ids
    )
    norm = _v31_norm_ar(evidence_text)

    if advisor_num == 15 and _v30_contains_any(
        norm,
        (
            "قياس الأثر", "تقييم الأثر", "متابعة وتقييم",
            "أثر البرامج", "قياس النتائج", "نتائج طويلة الأجل",
            "استدامة دخل المستفيدين", "قرار التوسع", "جدوى التوسع",
        ),
    ):
        floor = max(floor, 0.70)

    if advisor_num == 18 and _v30_contains_any(
        norm,
        (
            "تنويع مصادر الدخل", "تنويع الإيرادات", "استدامة مالية",
            "مصدر تمويل رئيسي", "اعتماد على مانح", "مانح رئيسي",
            "تركيز التمويل",
        ),
    ):
        floor = max(floor, 0.70)

    return floor


def _v34_minimum_fill_candidates(advisors, facts, fact_map, excluded_codes):
    """Build conservative fact-backed fillers when strict candidates are < minimum.

    No advisor is added merely because a minimum count exists: every filler must
    still have at least one advisor-specific direct fact match.
    """
    rows = []

    for advisor in advisors:
        code = advisor.get("system_code")
        if not code or code in excluded_codes:
            continue

        num = int(advisor["advisor_id"])
        terms = _v34_terms_for_advisor(num)
        matched = []

        for fact in facts:
            if not _v34_direct_fact_match(num, fact):
                continue
            priority = _v33_fact_priority(num, fact, terms)
            matched.append((priority, fact.get("fact_id")))

        matched.sort(key=lambda x: x[0], reverse=True)
        evidence_ids = [fid for _, fid in matched[:3] if fid]

        if not evidence_ids:
            continue

        rows.append({
            "advisor_id": code,
            "advisor": advisor,
            "verified_ids": evidence_ids,
            "priority": matched[0][0] if matched else 0.0,
        })

    rows.sort(
        key=lambda x: (x["priority"], len(x["verified_ids"])),
        reverse=True,
    )
    return rows


def advisory_match_rich_v34(job_input):
    started = time.time()

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    facts = _v31_build_facts(
        organization,
        programs,
    )

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    advisors = _RICH_REGISTRY["advisors"]

    advisor_by_code = {
        advisor["system_code"]: advisor
        for advisor in advisors
    }

    candidates = _v33_build_candidates(
        advisors,
        facts,
    )

    print(
        f"Rich v34: {len(candidates)} strictly evidence-backed candidates "
        "from 35 advisors; running ONE adjudication generation...",
        flush=True,
    )

    if not candidates:
        return {"ranked": []}

    raw, input_tokens = _v18_generate_text(
        RICH_V33_ADJUDICATOR_PROMPT,
        {
            "candidates": candidates,
        },
        RICH_V34_MAX_NEW_TOKENS,
    )

    rows = _v32_parse_results(
        raw,
        candidates,
    )

    matches = []
    scored_candidates = []

    for candidate in candidates:
        code = candidate["advisor_id"]
        item = rows.get(code)

        ai_score = (
            item["score"]
            if item
            else 0.0
        )

        advisor = advisor_by_code.get(code)
        if not advisor:
            continue

        # Public evidence is re-validated deterministically against the
        # exact advisor domain. We do not trust an unrelated evidence ID
        # merely because the LLM returned it.
        verified_ids = _v34_verified_evidence(
            advisor,
            candidate,
            fact_map,
        )

        if not verified_ids:
            print(
                f"Rich v34 evidence verification dropped {code}.",
                flush=True,
            )
            continue

        advisor_num = int(advisor["advisor_id"])
        strength = candidate.get(
            "evidence_strength",
            "MEDIUM",
        )

        evidence_floor = _v34_explicit_need_floor(
            advisor_num,
            verified_ids,
            fact_map,
            strength,
        )

        final_score = max(
            ai_score,
            evidence_floor,
        )

        scored_candidates.append({
            "advisor_id": code,
            "advisor": advisor,
            "verified_ids": verified_ids,
            "score": final_score,
            "strength": strength,
        })

        if final_score < RICH_V34_MIN_PUBLIC_SCORE:
            continue

        matches.append({
            "advisor_id": code,
            "score": round(final_score, 4),
            "reason": _v34_reason(
                advisor,
                verified_ids,
                fact_map,
            ),
        })

    # Enforce the product-level minimum of 6 advisors without changing the
    # one-generation architecture. First fill from strict evidence-backed
    # candidates that the adjudicator under-scored.
    if len(matches) < RICH_V34_MIN_ADVISORS:
        existing = {row["advisor_id"] for row in matches}
        strength_rank = {"STRONG": 2, "MEDIUM": 1}

        fallback = sorted(
            (x for x in scored_candidates if x["advisor_id"] not in existing),
            key=lambda x: (
                strength_rank.get(x["strength"], 0),
                x["score"],
                len(x["verified_ids"]),
            ),
            reverse=True,
        )

        for item in fallback:
            if len(matches) >= RICH_V34_MIN_ADVISORS:
                break

            matches.append({
                "advisor_id": item["advisor_id"],
                "score": round(
                    max(item["score"], RICH_V34_MIN_PUBLIC_SCORE),
                    4,
                ),
                "reason": _v34_reason(
                    item["advisor"],
                    item["verified_ids"],
                    fact_map,
                ),
            })
            existing.add(item["advisor_id"])

        # If the strict candidate gate itself produced fewer than six, relax
        # only that gate. Every added advisor must STILL have a direct
        # advisor-specific fact match; completely unrelated padding is never used.
        if len(matches) < RICH_V34_MIN_ADVISORS:
            relaxed = _v34_minimum_fill_candidates(
                advisors,
                facts,
                fact_map,
                existing,
            )

            for item in relaxed:
                if len(matches) >= RICH_V34_MIN_ADVISORS:
                    break

                matches.append({
                    "advisor_id": item["advisor_id"],
                    "score": round(RICH_V34_MIN_PUBLIC_SCORE, 4),
                    "reason": _v34_reason(
                        item["advisor"],
                        item["verified_ids"],
                        fact_map,
                    ),
                })
                existing.add(item["advisor_id"])

    matches.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Stable product contract: no more than 16 candidates are returned.
    # The minimum-fill logic above still guarantees >= 6 when enough
    # evidence-backed advisors exist.
    matches = matches[:max(RICH_V34_MIN_ADVISORS, RICH_V34_MAX_ADVISORS)]

    elapsed = round(
        time.time() - started,
        2,
    )

    print(
        f"Rich v34 complete in {elapsed}s. "
        f"Input tokens={input_tokens}. "
        f"Candidates={len(candidates)}. "
        f"Returned={len(matches)}. "
        "Model generations=1.",
        flush=True,
    )

    return {
        "ranked": matches
    }


RUNS = {
    "base": {
        "config": f"{ROOT}/configs/base_config.yaml",
        "source_checkpoint_rel": "checkpoints/base",
        "target_checkpoint_rel": "checkpoints/base",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-qlora",
        "resume": True,
    },

    "meta": {
        "config": f"{ROOT}/configs/meta_config.yaml",
        "source_checkpoint_rel": "checkpoints/meta",
        "target_checkpoint_rel": "checkpoints/meta",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-meta-qlora",
        "resume": True,
    },

    "specialist": {
        "config": f"{ROOT}/configs/specialist_config.yaml",
        "source_checkpoint_rel": "checkpoints/specialist",
        "target_checkpoint_rel": "checkpoints/specialist",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-specialist-qlora",
        "resume": True,
    },

    # New task:
    # Start from Meta adapter weights, but train as a NEW task/run.
    "matcher": {
        "config": f"{ROOT}/configs/matcher_config.yaml",
        "source_checkpoint_rel": "checkpoints/matcher",
        "target_checkpoint_rel": "checkpoints/matcher",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-matcher-v2-qlora",
        "resume": False,
    },
}


def run_command(cmd, cwd=None, env=None, stream=False):

    if stream:

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        tail = deque(maxlen=200)

        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "Command failed:\n" +
                "".join(tail)[-8000:]
            )

        return "".join(tail)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stdout[-8000:]
        )

    return result.stdout


def clone_repo_without_lfs(token, repo_dir="/tmp/athar_training_repo"):

    if not GITHUB_REPO:
        raise RuntimeError(
            "GITHUB_REPO environment variable is required. "
            "Set it to the new recovered repository, for example "
            "YOUR_USERNAME/athar-qwen-training-recovered."
        )

    shutil.rmtree(
        repo_dir,
        ignore_errors=True
    )

    clone_url = (
        f"https://x-access-token:{token}"
        f"@github.com/{GITHUB_REPO}.git"
    )

    env = os.environ.copy()

    # Clone normal files but skip all heavy LFS files initially.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            GITHUB_BRANCH,
            clone_url,
            repo_dir,
        ],
        env=env,
    )

    return repo_dir, env


def clone_source_checkpoint(training_type, token):

    info = RUNS[training_type]

    repo_dir, env = clone_repo_without_lfs(token)

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    source_checkpoint_rel = info["source_checkpoint_rel"]

    # Download only the source adapter needed for this run.
    run_command(
        [
            "git",
            "lfs",
            "pull",
            f"--include={source_checkpoint_rel}/**",
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_path = os.path.join(
        repo_dir,
        source_checkpoint_rel
    )

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors"
    )

    if not os.path.exists(adapter_file):
        raise RuntimeError(
            "adapter_model.safetensors was not downloaded."
        )

    # Detect an LFS pointer accidentally being used as model weights.
    if os.path.getsize(adapter_file) < 10_000_000:
        raise RuntimeError(
            "Checkpoint appears to be a Git LFS pointer, "
            "not the real adapter file."
        )

    return repo_dir, checkpoint_path, env


def load_matcher_assets(repo_dir):

    registry_path = os.path.join(
        repo_dir,
        MATCHER_REGISTRY_REL
    )

    prompt_path = os.path.join(
        repo_dir,
        MATCHER_PROMPT_REL
    )

    if not os.path.exists(registry_path):
        raise RuntimeError(
            f"Matcher registry not found: {MATCHER_REGISTRY_REL}"
        )

    if not os.path.exists(prompt_path):
        raise RuntimeError(
            f"Matcher prompt not found: {MATCHER_PROMPT_REL}"
        )

    with open(
        registry_path,
        "r",
        encoding="utf-8"
    ) as f:
        registry = json.load(f)

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as f:
        matcher_prompt = f.read()

    advisors = registry.get("advisors")

    if not isinstance(advisors, list):
        raise RuntimeError(
            "Matcher registry field 'advisors' must be a list."
        )

    if len(advisors) != 16:
        raise RuntimeError(
            f"Expected 16 advisors, found {len(advisors)}."
        )

    advisor_ids = [
        advisor.get("advisor_id")
        for advisor in advisors
    ]

    if advisor_ids != list(range(1, 17)):
        raise RuntimeError(
            f"Advisor IDs must be 1..16. Found: {advisor_ids}"
        )

    return registry, matcher_prompt



def clone_matcher_checkpoint(token):

    repo_dir, env = clone_repo_without_lfs(
        token,
        repo_dir="/tmp/athar_inference_repo",
    )

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_rel = RUNS["matcher"]["target_checkpoint_rel"]

    run_command(
        [
            "git",
            "lfs",
            "pull",
            f"--include={checkpoint_rel}/**",
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_path = os.path.join(
        repo_dir,
        checkpoint_rel,
    )

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors",
    )

    adapter_config = os.path.join(
        checkpoint_path,
        "adapter_config.json",
    )

    if not os.path.exists(adapter_file):
        raise RuntimeError(
            "Matcher adapter_model.safetensors was not downloaded."
        )

    if os.path.getsize(adapter_file) < 10_000_000:
        raise RuntimeError(
            "Matcher checkpoint appears to be a Git LFS pointer."
        )

    if not os.path.exists(adapter_config):
        raise RuntimeError(
            "Matcher adapter_config.json was not found."
        )

    return repo_dir, checkpoint_path


def compact_candidate(advisor):

    return {
        "advisor_id": advisor.get("advisor_id"),
        "name_ar": advisor.get("name_ar"),
        "mission_summary": advisor.get("mission_summary", ""),
        "core_scope": advisor.get("core_scope", []),
        "activation_conditions": advisor.get("activation_conditions", []),
        "not_primary_when": advisor.get("not_primary_when", []),
        "scope_boundaries": advisor.get("scope_boundaries", ""),
        "match_signals": advisor.get("match_signals", []),
    }


def ensure_matcher_model(token):

    global _MATCHER_MODEL
    global _MATCHER_TOKENIZER
    global _MATCHER_REGISTRY
    global _MATCHER_DEVICE

    if (
        _MATCHER_MODEL is not None
        and _MATCHER_TOKENIZER is not None
        and _MATCHER_REGISTRY is not None
    ):
        return

    print(
        "Loading matcher model and adapter...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    repo_dir, checkpoint_path = clone_matcher_checkpoint(
        token
    )

    registry, _ = load_matcher_assets(
        repo_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False,
    )

    model.eval()

    _MATCHER_MODEL = model
    _MATCHER_TOKENIZER = tokenizer
    _MATCHER_REGISTRY = registry
    _MATCHER_DEVICE = next(model.parameters()).device

    print(
        f"Matcher ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def build_candidate_prompt(
    organization,
    programs,
    advisor,
):

    user_payload = {
        "organization": organization,
        "programs": programs,
        "candidate_advisor": compact_candidate(advisor),
    }

    messages = [
        {
            "role": "system",
            "content": MATCHER_CANDIDATE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
            ),
        },
    ]

    return _MATCHER_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_json_object(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "", 1).strip()

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[index:]
            )

            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Model did not return valid JSON. Raw output: {text[:500]}"
    )



def parse_candidate_output(text):

    # First try strict JSON parsing.
    try:
        return extract_json_object(text)
    except ValueError:
        pass

    # Robust fallback for truncated/repetitive generations:
    # recover the classification and score even if "reason" was not closed.
    relevant_match = re.search(
        r'"relevant"\s*:\s*(true|false)',
        text,
        flags=re.IGNORECASE,
    )

    score_match = re.search(
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
    )

    reason_match = re.search(
        r'"reason"\s*:\s*"([^"]*)',
        text,
        flags=re.DOTALL,
    )

    if relevant_match is None or score_match is None:
        raise ValueError(
            f"Could not recover matcher classification. Raw output: {text[:500]}"
        )

    relevant = (
        relevant_match.group(1).lower() == "true"
    )

    score = float(
        score_match.group(1)
    )

    reason = ""

    if reason_match is not None:
        reason = re.sub(
            r"\s+",
            " ",
            reason_match.group(1),
        ).strip()

        # Keep a runaway unfinished reason from polluting the API.
        words = reason.split()

        if len(words) > 24:
            reason = " ".join(words[:24]).rstrip("،,.") + "."

    if not reason:
        reason = (
            "تم استرجاع التصنيف والدرجة من استجابة غير مكتملة."
        )

    return {
        "relevant": relevant,
        "score": score,
        "reason": reason,
    }


def normalize_candidate_result(
    advisor_id,
    raw_result,
):

    relevant = raw_result.get("relevant", False)

    if isinstance(relevant, str):
        relevant = relevant.strip().lower() == "true"

    relevant = bool(relevant)

    try:
        score = float(
            raw_result.get("score", 0.0)
        )
    except (TypeError, ValueError):
        score = 0.0

    score = max(
        0.0,
        min(1.0, score),
    )

    reason = str(
        raw_result.get("reason", "")
    ).strip()

    return {
        "advisor_id": advisor_id,
        "relevant": relevant,
        "score": round(score, 4),
        "reason": reason,
    }


def evaluate_advisor_batch(
    prompts,
    advisors,
):

    import torch

    encoded = _MATCHER_TOKENIZER(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    prompt_token_counts = encoded[
        "attention_mask"
    ].sum(dim=1).tolist()

    too_long = [
        {
            "advisor_id": advisors[index]["advisor_id"],
            "tokens": int(token_count),
        }
        for index, token_count in enumerate(
            prompt_token_counts
        )
        if token_count > MATCHER_MAX_INPUT_TOKENS
    ]

    if too_long:
        raise ValueError(
            "Matcher input exceeds safe token limit: "
            + json.dumps(
                too_long,
                ensure_ascii=False,
            )
        )

    encoded = {
        key: value.to(_MATCHER_DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _MATCHER_MODEL.generate(
            **encoded,
            max_new_tokens=MATCHER_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=8,
            eos_token_id=_MATCHER_TOKENIZER.eos_token_id,
            pad_token_id=_MATCHER_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    texts = _MATCHER_TOKENIZER.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    results = []

    for advisor, text in zip(
        advisors,
        texts,
    ):
        parsed = parse_candidate_output(
            text
        )

        normalized = normalize_candidate_result(
            advisor["advisor_id"],
            parsed,
        )

        normalized["input_tokens"] = int(
            prompt_token_counts[len(results)]
        )

        results.append(
            normalized
        )

    return results


def advisory_match_inference(
    job_input,
    token,
):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_matcher_model(
        token
    )

    advisors = _MATCHER_REGISTRY["advisors"]

    evaluations = []

    for start in range(
        0,
        len(advisors),
        MATCHER_BATCH_SIZE,
    ):

        batch_advisors = advisors[
            start:start + MATCHER_BATCH_SIZE
        ]

        prompts = [
            build_candidate_prompt(
                organization,
                programs,
                advisor,
            )
            for advisor in batch_advisors
        ]

        print(
            "Evaluating advisors: "
            + ", ".join(
                str(a["advisor_id"])
                for a in batch_advisors
            ),
            flush=True,
        )

        batch_results = evaluate_advisor_batch(
            prompts,
            batch_advisors,
        )

        evaluations.extend(
            batch_results
        )

    ranked = [
        {
            "advisor_id": row["advisor_id"],
            "score": row["score"],
            "reason": row["reason"],
        }
        for row in evaluations
        if (
            row["relevant"]
            and row["score"] >= MATCHER_MIN_RELEVANT_SCORE
        )
    ]

    ranked.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": len(evaluations),
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["evaluations"] = evaluations

    return response



def ensure_grounded_router_model(token):

    global _ROUTER_MODEL
    global _ROUTER_TOKENIZER
    global _ROUTER_REGISTRY
    global _ROUTER_DEVICE

    if (
        _ROUTER_MODEL is not None
        and _ROUTER_TOKENIZER is not None
        and _ROUTER_REGISTRY is not None
    ):
        return

    print(
        "Loading grounded routing model...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    # We need only text assets from GitHub for production routing.
    # No Matcher LoRA is loaded here.
    repo_dir, _ = clone_repo_without_lfs(
        token,
        repo_dir="/tmp/athar_grounded_router_repo",
    )

    registry, _ = load_matcher_assets(
        repo_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model.eval()

    _ROUTER_MODEL = model
    _ROUTER_TOKENIZER = tokenizer
    _ROUTER_REGISTRY = registry
    _ROUTER_DEVICE = next(model.parameters()).device

    print(
        f"Grounded router ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def build_grounded_facts(
    organization,
    programs,
):

    facts = []

    org_fields = [
        ("F1", "short_description"),
        ("F2", "detailed_description"),
        ("F3", "competitive_advantage"),
        ("F4", "important_notes"),
    ]

    for fact_id, field in org_fields:
        value = organization.get(field)

        if value is None:
            continue

        if isinstance(value, (list, dict)):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )
        else:
            value = str(value).strip()

        if value:
            facts.append({
                "fact_id": fact_id,
                "source": f"organization.{field}",
                "text": value,
            })

    activity_fields = organization.get(
        "activity_fields",
        []
    )

    if activity_fields:
        facts.append({
            "fact_id": "F5",
            "source": "organization.activity_fields",
            "text": json.dumps(
                activity_fields,
                ensure_ascii=False,
            ),
        })

    for index, program in enumerate(
        programs,
        start=1,
    ):

        if not isinstance(program, dict):
            continue

        parts = []

        for field in [
            "name",
            "type",
            "description",
            "target_audience",
            "beneficiary_value",
            "delivery_method",
        ]:
            value = program.get(field)

            if value is None:
                continue

            value = str(value).strip()

            if value:
                parts.append(
                    f"{field}={value}"
                )

        if parts:
            facts.append({
                "fact_id": f"P{index}",
                "source": f"programs[{index - 1}]",
                "text": " | ".join(parts),
            })

    return facts


def generate_json_with_router(
    system_prompt,
    payload,
    max_new_tokens,
):

    import torch

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _ROUTER_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _ROUTER_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > ROUTER_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Grounded router input is too long: "
            f"{input_tokens} tokens > {ROUTER_MAX_INPUT_TOKENS}"
        )

    encoded = {
        key: value.to(_ROUTER_DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _ROUTER_MODEL.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.06,
            no_repeat_ngram_size=10,
            eos_token_id=_ROUTER_TOKENIZER.eos_token_id,
            pad_token_id=_ROUTER_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    text = _ROUTER_TOKENIZER.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    parsed = extract_json_object(
        text
    )

    return parsed, input_tokens, text


def normalize_grounded_needs(
    raw_needs,
    facts,
):

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    allowed_kinds = {
        "explicit_gap",
        "direct_inference",
        "advisory_opportunity",
    }

    allowed_priorities = {
        "high",
        "medium",
        "low",
    }

    normalized = []

    if not isinstance(raw_needs, list):
        return normalized

    for index, raw_need in enumerate(
        raw_needs[:8],
        start=1,
    ):

        if not isinstance(raw_need, dict):
            continue

        need_text = str(
            raw_need.get("need", "")
        ).strip()

        if not need_text:
            continue

        evidence_ids = raw_need.get(
            "evidence_ids",
            []
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            evidence_ids = []

        evidence_ids = [
            str(evidence_id)
            for evidence_id in evidence_ids
            if str(evidence_id) in fact_map
        ]

        # A need without a valid evidence pointer is rejected.
        if not evidence_ids:
            continue

        need_id = f"N{len(normalized) + 1}"

        kind = raw_need.get(
            "kind",
            "direct_inference",
        )

        if kind not in allowed_kinds:
            kind = "direct_inference"

        priority = raw_need.get(
            "priority",
            "medium",
        )

        if priority not in allowed_priorities:
            priority = "medium"

        normalized.append({
            "need_id": need_id,
            "need": need_text,
            "kind": kind,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "evidence": [
                {
                    "fact_id": evidence_id,
                    "source": fact_map[evidence_id]["source"],
                    "text": fact_map[evidence_id]["text"],
                }
                for evidence_id in evidence_ids
            ],
        })

    return normalized



def _text_blob(organization, programs):
    parts = []
    for field in [
        "name", "type", "sector", "short_description",
        "detailed_description", "competitive_advantage", "important_notes",
    ]:
        value = organization.get(field)
        if value:
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(str(value))

    activity_fields = organization.get("activity_fields", [])
    if activity_fields:
        parts.append(json.dumps(activity_fields, ensure_ascii=False))

    for program in programs:
        if not isinstance(program, dict):
            continue
        for field in [
            "name", "type", "description",
            "target_audience", "beneficiary_value", "delivery_method",
        ]:
            value = program.get(field)
            if value:
                parts.append(str(value))

    return " ".join(parts)


def _contains_any(text, terms):
    return any(term in text for term in terms)


def extract_grounded_advisory_needs(organization, programs):
    """
    v6: deterministic, evidence-first signal extraction.
    No LLM is allowed to decide whether the organization has "a need".
    This prevents annual-report language from collapsing to needs=[].
    """

    facts = build_grounded_facts(organization, programs)
    fact_map = {f["fact_id"]: f for f in facts}
    blob = _text_blob(organization, programs)
    blob_lower = blob.lower()
    signals = []

    def add_signal(signal_type, statement, evidence_ids, priority):
        valid = [eid for eid in evidence_ids if eid in fact_map]
        if not valid:
            return
        if any(x.get("signal_type") == signal_type for x in signals):
            return

        signals.append({
            "need_id": f"N{len(signals) + 1}",
            "need": statement,
            "kind": "advisory_opportunity",
            "priority": priority,
            "signal_type": signal_type,
            "evidence_ids": valid,
            "evidence": [
                {
                    "fact_id": eid,
                    "source": fact_map[eid]["source"],
                    "text": fact_map[eid]["text"],
                }
                for eid in valid
            ],
        })

    # 1) Portfolio/program complexity is directly observable.
    program_count = len(programs)
    if program_count >= 6:
        ids = [f"P{i}" for i in range(1, min(program_count, 6) + 1)]
        add_signal(
            "portfolio_complexity",
            f"وجود {program_count} برنامجًا/مبادرة متنوعة يخلق تعقيدًا ماديًا في إدارة المحفظة والبرامج والأولويات والمنافع والتنسيق بينها.",
            ids,
            "high" if program_count >= 10 else "medium",
        )

    # 2) Seasonal / time-bound operational complexity.
    seasonal_terms = [
        "موسم", "موسمية", "رمضان", "الحج", "حاج", "الحجاج",
        "ضيف الرحمن", "ضيوف الرحمن", "بداية العام الدراسي", "صيفي",
    ]
    seasonal_ids = []
    for i, program in enumerate(programs, start=1):
        ptext = " ".join(
            str(program.get(k, ""))
            for k in ["name", "description", "delivery_method"]
        )
        if _contains_any(ptext, seasonal_terms):
            seasonal_ids.append(f"P{i}")

    if len(seasonal_ids) >= 2:
        add_signal(
            "operational_coordination",
            "وجود عدة برامج موسمية أو مقيدة بتوقيتات تنفيذية مختلفة يخلق حاجة مادية للتخطيط التشغيلي والتنسيق بين الجداول والملاك والموارد.",
            seasonal_ids[:6],
            "medium",
        )

    # 3) MEAL / impact only when explicit language exists.
    meal_terms = [
        "قياس الأثر", "إدارة الأثر", "الأثر الاجتماعي",
        "تقييم الأثر", "نتائج البرامج", "نظرية التغيير",
        "متابعة وتقييم", "المتابعة والتقييم", "meal",
    ]
    if _contains_any(blob_lower, [x.lower() for x in meal_terms]):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"].lower(), [x.lower() for x in meal_terms])
        ][:6]
        add_signal(
            "impact_measurement",
            "توجد إشارة صريحة إلى قياس الأثر أو توجيه البرامج لدعمه، ما يبرر مراجعة إطار النتائج والتقييم والتعلم وقوة دليل الأثر.",
            ids,
            "high",
        )

    # 4) KPI only when explicit KPI/dashboard language exists.
    kpi_terms = [
        "مؤشرات الأداء", "مؤشر أداء", "kpi", "لوحة قيادة",
        "dashboard", "خط الأساس", "المستهدفات", "مصدر بيانات",
    ]
    if _contains_any(blob_lower, [x.lower() for x in kpi_terms]):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"].lower(), [x.lower() for x in kpi_terms])
        ][:6]
        add_signal(
            "kpi_management",
            "توجد إشارات صريحة إلى مؤشرات الأداء أو مصادرها أو خطوط الأساس أو لوحات القيادة، ما يبرر دعم منظومة KPI واتخاذ القرار.",
            ids,
            "high",
        )

    # 5) Governance: require an actual current gap/risk, not an achievement.
    governance_terms = ["حوكمة", "امتثال", "صلاحيات", "سياسات", "إجراءات"]
    gap_terms = [
        "ضعف", "غياب", "غير واضح", "تعارض", "قصور",
        "مخالفة", "عدم امتثال", "تحتاج", "بحاجة", "مطلوب",
    ]
    for fact in facts:
        text = fact["text"]
        if _contains_any(text, governance_terms) and _contains_any(text, gap_terms):
            add_signal(
                "governance_gap",
                "توجد فجوة أو مخاطرة حوكمة/امتثال مذكورة صراحة وتحتاج ضبط الصلاحيات أو السياسات أو أدلة التطبيق.",
                [fact["fact_id"]],
                "high",
            )
            break

    # 6) Change management: only on adoption/resistance evidence.
    change_terms = [
        "مقاومة التغيير", "ضعف التبني", "عدم التبني", "رفض النظام",
        "صعوبة التغيير", "إدارة التغيير", "تحديات التبني",
    ]
    if _contains_any(blob, change_terms):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"], change_terms)
        ][:6]
        add_signal(
            "change_adoption",
            "توجد إشارات صريحة إلى تحديات تبني أو مقاومة تغيير تتطلب إدارة تغيير منظمة.",
            ids,
            "high",
        )

    # 7) Strategy: require explicit strategic review/development language.
    strategy_terms = [
        "خطة استراتيجية", "استراتيجية", "أهداف استراتيجية",
        "أولويات استراتيجية", "قضايا استراتيجية",
    ]
    strategy_action_terms = [
        "تحديث", "مراجعة", "إعادة", "غير واضحة",
        "تحتاج", "بحاجة", "مطلوب", "تطوير",
    ]
    for fact in facts:
        text = fact["text"]
        if _contains_any(text, strategy_terms) and _contains_any(text, strategy_action_terms):
            add_signal(
                "strategy",
                "توجد حاجة أو فرصة استراتيجية صريحة تتعلق بمراجعة أو تطوير الاتجاه والأهداف والأولويات.",
                [fact["fact_id"]],
                "high",
            )
            break

    # 8) Stakeholders / partnerships when explicitly present.
    partnership_terms = [
        "شراكات", "شركاء", "أصحاب المصلحة",
        "الجهات المانحة", "مانحين",
    ]
    if _contains_any(blob, partnership_terms):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"], partnership_terms)
        ][:6]
        add_signal(
            "stakeholders_partnerships",
            "توجد شراكات أو أطراف مصلحة متعددة بما يجعل إدارة العلاقة والقيمة المتبادلة مجالًا استشاريًا ماديًا.",
            ids,
            "medium",
        )

    return {
        "needs": signals[:8],
        "facts": facts,
        "input_tokens": 0,
        "raw_text": None,
    }


def compact_routing_advisor(advisor):

    return {
        "advisor_id": advisor.get("advisor_id"),
        "name_ar": advisor.get("name_ar"),
        "mission_summary": advisor.get(
            "mission_summary",
            "",
        ),
        "owned_outcome": advisor.get(
            "owned_outcome",
            "",
        ),
        "core_scope": advisor.get(
            "core_scope",
            [],
        ),
        "activation_conditions": advisor.get(
            "activation_conditions",
            [],
        ),
        "not_primary_when": advisor.get(
            "not_primary_when",
            [],
        ),
        "scope_boundaries": advisor.get(
            "scope_boundaries",
            "",
        ),
        "match_signals": advisor.get(
            "match_signals",
            [],
        ),
    }


def route_all_advisors(
    needs,
):

    advisors = [
        compact_routing_advisor(
            advisor
        )
        for advisor in _ROUTER_REGISTRY["advisors"]
    ]

    routing_needs = [
        {
            "need_id": need["need_id"],
            "need": need["need"],
            "kind": need["kind"],
            "priority": need["priority"],
            "evidence": [
                evidence["text"]
                for evidence in need["evidence"]
            ],
        }
        for need in needs
    ]

    raw, input_tokens, raw_text = generate_json_with_router(
        ROUTING_SYSTEM_PROMPT,
        {
            "grounded_needs": routing_needs,
            "advisors": advisors,
        },
        ROUTER_RANK_MAX_NEW_TOKENS,
    )

    return {
        "raw_ranked": raw.get(
            "ranked",
            []
        ),
        "input_tokens": input_tokens,
        "raw_text": raw_text,
    }


def validate_global_ranking(
    raw_ranked,
    needs,
):

    valid_advisor_ids = {
        advisor["advisor_id"]
        for advisor in _ROUTER_REGISTRY["advisors"]
    }

    valid_need_ids = {
        need["need_id"]
        for need in needs
    }

    seen_advisors = set()
    ranked = []

    if not isinstance(
        raw_ranked,
        list,
    ):
        return ranked

    for row in raw_ranked:

        if not isinstance(row, dict):
            continue

        try:
            advisor_id = int(
                row.get("advisor_id")
            )
        except (TypeError, ValueError):
            continue

        if (
            advisor_id not in valid_advisor_ids
            or advisor_id in seen_advisors
        ):
            continue

        matched_need_ids = row.get(
            "matched_need_ids",
            []
        )

        if not isinstance(
            matched_need_ids,
            list,
        ):
            matched_need_ids = []

        matched_need_ids = [
            str(need_id)
            for need_id in matched_need_ids
            if str(need_id) in valid_need_ids
        ]

        # This is the key guardrail:
        # no advisor can be returned without a grounded need.
        if not matched_need_ids:
            continue

        try:
            score = float(
                row.get("score", 0.0)
            )
        except (TypeError, ValueError):
            continue

        score = max(
            0.0,
            min(1.0, score),
        )

        if score < ROUTER_MIN_SCORE:
            continue

        role = str(
            row.get("role", "supporting")
        ).strip().lower()

        if role not in {
            "primary",
            "supporting",
        }:
            role = "supporting"

        reason = str(
            row.get("reason", "")
        ).strip()

        if not reason:
            reason = (
                "ملاءمة مرتبطة بحاجة موثقة في بيانات المنظمة."
            )

        words = reason.split()

        if len(words) > 30:
            reason = (
                " ".join(words[:30]).rstrip(
                    "،,."
                )
                + "."
            )

        ranked.append({
            "advisor_id": advisor_id,
            "score": round(score, 4),
            "role": role,
            "matched_need_ids": matched_need_ids,
            "reason": reason,
        })

        seen_advisors.add(
            advisor_id
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked


def advisory_match_grounded_v5(
    job_input,
    token,
):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_grounded_router_model(
        token
    )

    print(
        "Stage 1/2: building deterministic grounded advisory signals...",
        flush=True,
    )

    extraction = extract_grounded_advisory_needs(
        organization,
        programs,
    )

    needs = extraction["needs"]

    print(
        f"Grounded advisory signals built: {len(needs)}",
        flush=True,
    )

    if not needs:
        response = {
            "status": "completed",
            "type": "advisory_match",
            "routing_engine": "grounded_v6",
            "run_id": job_input.get("run_id"),
            "organization_name": organization.get("name"),
            "needs_count": 0,
            "ranked": [],
        }

        if job_input.get("debug", False):
            response["needs"] = []
            response["need_extraction_input_tokens"] = extraction[
                "input_tokens"
            ]

        return response

    print(
        "Stage 2/2: comparing all advisors together...",
        flush=True,
    )

    routing = route_all_advisors(
        needs
    )

    ranked = validate_global_ranking(
        routing["raw_ranked"],
        needs,
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "grounded_v6",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "needs_count": len(needs),
        "ranked": [
            {
                "advisor_id": row["advisor_id"],
                "score": row["score"],
                "reason": row["reason"],
            }
            for row in ranked
        ],
    }

    if job_input.get("debug", False):
        response["needs"] = needs
        response["routing_details"] = ranked
        response["need_extraction_input_tokens"] = extraction[
            "input_tokens"
        ]
        response["routing_input_tokens"] = routing[
            "input_tokens"
        ]

    return response


def normalize_advisory_input(raw_input):

    # Prefer a real JSON object, but temporarily accept
    # a JSON string for backend compatibility.
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"advisory_match input is not valid JSON: {exc}"
            )

    if not isinstance(raw_input, dict):
        raise ValueError(
            "advisory_match input must be a JSON object."
        )

    organization = raw_input.get("organization")
    programs = raw_input.get("programs", [])

    if not isinstance(organization, dict):
        raise ValueError(
            "input.organization must be a JSON object."
        )

    if not isinstance(programs, list):
        raise ValueError(
            "input.programs must be a JSON array."
        )

    return organization, programs


def advisory_match_preflight(job_input, token):

    repo_dir, _ = clone_repo_without_lfs(token)

    registry, matcher_prompt = load_matcher_assets(
        repo_dir
    )

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    advisors = registry["advisors"]

    return {
        "status": "matcher_preflight_ok",
        "type": "advisory_match",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "programs_count": len(programs),
        "advisors_count": len(advisors),
        "advisor_ids": [
            advisor["advisor_id"]
            for advisor in advisors
        ],
        "registry_version": registry.get("version"),
        "matcher_prompt_chars": len(matcher_prompt),
        "note": (
            "Matcher assets and request schema are valid. "
            "Matcher assets and request schema are valid. "
            "The trained matcher adapter is available for inference."
        ),
    }


def validate_training_files(training_type):

    info = RUNS[training_type]

    if not os.path.exists(info["config"]):
        raise RuntimeError(
            f"Config not found: {info['config']}"
        )

    extra = {}

    if training_type == "matcher":

        train_path = f"{ROOT}/data/train_matcher_v2.jsonl"
        validation_path = f"{ROOT}/data/validation_matcher_v2.jsonl"

        if not os.path.exists(train_path):
            raise RuntimeError(
                f"Matcher train dataset not found: {train_path}"
            )

        if not os.path.exists(validation_path):
            raise RuntimeError(
                f"Matcher validation dataset not found: {validation_path}"
            )

        def count_jsonl(path):
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Invalid JSONL in {path} at line "
                            f"{line_number}: {exc}"
                        )

                    if not isinstance(row.get("messages"), list):
                        raise RuntimeError(
                            f"Missing messages list in {path} "
                            f"at line {line_number}"
                        )

                    count += 1

            return count

        extra = {
            "train_samples": count_jsonl(train_path),
            "validation_samples": count_jsonl(validation_path),
            "train_dataset": train_path,
            "validation_dataset": validation_path,
        }

    return extra


def latest_checkpoint(output_dir):

    paths = glob.glob(
        os.path.join(
            output_dir,
            "checkpoint-*"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No checkpoint found in {output_dir}"
        )

    return max(
        paths,
        key=lambda path: int(
            os.path.basename(path).split("-")[-1]
        )
    )


def push_checkpoint(
    training_type,
    repo_dir,
    new_checkpoint,
    env,
):

    target_checkpoint_rel = RUNS[training_type][
        "target_checkpoint_rel"
    ]

    destination = os.path.join(
        repo_dir,
        target_checkpoint_rel
    )

    shutil.rmtree(
        destination,
        ignore_errors=True
    )

    shutil.copytree(
        new_checkpoint,
        destination
    )

    run_command(
        [
            "git",
            "config",
            "user.name",
            GIT_USER_NAME,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "config",
            "user.email",
            GIT_USER_EMAIL,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "add",
            target_checkpoint_rel,
        ],
        cwd=repo_dir,
        env=env,
    )

    status = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--quiet",
        ],
        cwd=repo_dir,
        env=env,
    )

    if status.returncode == 0:
        return "No checkpoint changes detected"

    run_command(
        [
            "git",
            "commit",
            "-m",
            f"Update {training_type} checkpoint from RunPod Serverless",
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "push",
            "origin",
            GITHUB_BRANCH,
        ],
        cwd=repo_dir,
        env=env,
        stream=True,
    )

    commit_sha = run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_dir,
        env=env,
    ).strip()

    return commit_sha


def training_preflight(
    training_type,
    checkpoint_path,
):

    info = RUNS[training_type]

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors"
    )

    extra = validate_training_files(
        training_type
    )

    response = {
        "status": "training_preflight_ok",
        "training_type": training_type,
        "config": info["config"],
        "resume_mode": info["resume"],
        "source_checkpoint": checkpoint_path,
        "target_checkpoint_rel": info["target_checkpoint_rel"],
        "adapter_size_mb": round(
            os.path.getsize(adapter_file) / 1024 / 1024,
            2
        ),
    }

    response.update(extra)

    if training_type == "matcher":
        response["note"] = (
            "Matcher v2 will refine the existing matcher adapter "
            "using lora_model_dir and will start a NEW "
            "optimizer/scheduler state. It will NOT use "
            "--resume-from-checkpoint."
        )

    return response



def _real_adapter_checkpoint(path):
    adapter_file = os.path.join(
        path,
        "adapter_model.safetensors",
    )
    config_file = os.path.join(
        path,
        "adapter_config.json",
    )

    return (
        os.path.isfile(adapter_file)
        and os.path.getsize(adapter_file) >= 10_000_000
        and os.path.isfile(config_file)
    )


def ensure_council_adapter_paths():
    """Resolve the real specialist/meta LoRA files.

    Preference order for the Specialist adapter:
      1) checkpoints/specialist-v2  (new continual-learning adapter)
      2) checkpoints/specialist     (stable fallback)

    If Specialist V2 is not baked into the current image but GITHUB_TOKEN is
    available, the worker checks the current GitHub branch before falling back
    to a local Specialist V1. This is what allows a freshly trained V2 adapter
    to become active without requiring a separate Serverless endpoint.
    """

    global _COUNCIL_ADAPTER_PATHS

    if _COUNCIL_ADAPTER_PATHS is not None:
        return _COUNCIL_ADAPTER_PATHS

    local_specialist_v2 = os.path.join(
        ROOT,
        "checkpoints",
        "specialist-v2",
    )
    local_specialist_v1 = os.path.join(
        ROOT,
        "checkpoints",
        "specialist",
    )
    local_meta = os.path.join(
        ROOT,
        "checkpoints",
        "meta",
    )

    # Best case: both active adapters are already real files in this worker.
    if (
        _real_adapter_checkpoint(local_specialist_v2)
        and _real_adapter_checkpoint(local_meta)
    ):
        _COUNCIL_ADAPTER_PATHS = (
            local_specialist_v2,
            local_meta,
        )

        print(
            "Council adapters: local Specialist V2 + local Meta.",
            flush=True,
        )

        return _COUNCIL_ADAPTER_PATHS

    token = os.environ.get("GITHUB_TOKEN")

    # If GitHub is available, always check the current branch for Specialist V2
    # before accepting a baked/local V1 fallback.
    if token:
        repo_dir, env = clone_repo_without_lfs(
            token,
            repo_dir="/tmp/athar_council_repo",
        )

        run_command(
            [
                "git",
                "lfs",
                "install",
                "--local",
            ],
            cwd=repo_dir,
            env=env,
        )

        run_command(
            [
                "git",
                "lfs",
                "pull",
                (
                    "--include="
                    "checkpoints/specialist-v2/**,"
                    "checkpoints/specialist/**,"
                    "checkpoints/meta/**"
                ),
                "--exclude=",
            ],
            cwd=repo_dir,
            env=env,
        )

        remote_specialist_v2 = os.path.join(
            repo_dir,
            "checkpoints",
            "specialist-v2",
        )

        remote_specialist_v1 = os.path.join(
            repo_dir,
            "checkpoints",
            "specialist",
        )

        remote_meta = os.path.join(
            repo_dir,
            "checkpoints",
            "meta",
        )

        specialist_path = (
            remote_specialist_v2
            if _real_adapter_checkpoint(
                remote_specialist_v2
            )
            else remote_specialist_v1
        )

        if not _real_adapter_checkpoint(
            specialist_path
        ):
            raise RuntimeError(
                "Neither Specialist V2 nor Specialist V1 LoRA "
                "was downloaded correctly from Git LFS."
            )

        if not _real_adapter_checkpoint(
            remote_meta
        ):
            raise RuntimeError(
                "Meta LoRA was not downloaded correctly from Git LFS."
            )

        _COUNCIL_ADAPTER_PATHS = (
            specialist_path,
            remote_meta,
        )

        print(
            "Council adapters downloaded from current GitHub branch: "
            f"specialist={specialist_path} meta={remote_meta}",
            flush=True,
        )

        return _COUNCIL_ADAPTER_PATHS

    # No GitHub token: production can still use real adapters already baked
    # into the image, but cannot discover a newer remote V2.
    if (
        _real_adapter_checkpoint(local_specialist_v1)
        and _real_adapter_checkpoint(local_meta)
    ):
        _COUNCIL_ADAPTER_PATHS = (
            local_specialist_v1,
            local_meta,
        )

        print(
            "Council adapters: local Specialist V1 fallback + local Meta.",
            flush=True,
        )

        return _COUNCIL_ADAPTER_PATHS

    raise RuntimeError(
        "No usable Specialist/Meta adapters are available locally, "
        "and GITHUB_TOKEN is not configured to download them."
    )


def ensure_council_engine():
    """Create the council on top of the SAME Qwen3-14B used by matching."""

    global _COUNCIL_ENGINE
    global _RICH_MODEL
    global _RICH_DEVICE

    if _COUNCIL_ENGINE is not None:
        return _COUNCIL_ENGINE

    # Load the exact production routing base model once.
    ensure_rich_router_model()

    specialist_path, meta_path = ensure_council_adapter_paths()

    from handler_advisory_council import AtharCouncilEngine

    engine = AtharCouncilEngine(
        base_model=_RICH_MODEL,
        tokenizer=_RICH_TOKENIZER,
        specialist_adapter_path=specialist_path,
        meta_adapter_path=meta_path,
        model_lock=INFERENCE_MODEL_LOCK,
    )

    # PeftModel wraps the same base object. Point the matcher global at the
    # wrapper too, then _v18_generate_text disables adapters for advisory_match.
    _RICH_MODEL = engine.model
    _RICH_DEVICE = next(_RICH_MODEL.parameters()).device
    _COUNCIL_ENGINE = engine

    print(
        "Athar Advisory Council ready on shared Qwen3-14B base model.",
        flush=True,
    )

    return _COUNCIL_ENGINE


def advisory_consultation_inference(job_input):
    engine = ensure_council_engine()
    return engine.consult(job_input)


# =====================================================================
# Continual Learning + Specialist V2
# =====================================================================
#
# This stays inside the SAME RunPod Serverless handler:
#
#   advisory_match
#   advisory_consultation
#   advisory_feedback
#   continual_status
#   train_specialist_v2
#
# Production consultations are logged to the private GitHub repository
# (when GITHUB_TOKEN is configured). Human-approved/corrected advisor
# answers become eligible SFT examples. Rejected answers are retained for
# audit/future preference training but are never used as positive SFT data.
#
# Fine-tuning is NEVER triggered after every request. It is explicitly
# triggered with type=train_specialist_v2 so bad generations cannot teach
# themselves back into the model automatically.
# =====================================================================

CONTINUAL_ROOT_REL = os.environ.get(
    "CONTINUAL_ROOT_REL",
    "continual",
).strip("/")

CONTINUAL_LEARNING_ENABLED = (
    os.environ.get("CONTINUAL_LEARNING_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

CONTINUAL_AUTO_LOG_RUNS = (
    os.environ.get("CONTINUAL_AUTO_LOG_RUNS", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

CONTINUAL_STORAGE_BACKEND = os.environ.get(
    "CONTINUAL_STORAGE_BACKEND",
    "github",
).strip().lower()

ATHAR_S3_BUCKET = os.environ.get(
    "ATHAR_S3_BUCKET",
    os.environ.get("S3_BUCKET", ""),
).strip()

ATHAR_S3_PREFIX = os.environ.get(
    "ATHAR_S3_PREFIX",
    "athar-os",
).strip("/")

ATHAR_S3_ENDPOINT_URL = os.environ.get(
    "ATHAR_S3_ENDPOINT_URL",
    os.environ.get("AWS_ENDPOINT_URL", ""),
).strip()

SPECIALIST_V2_TARGET_REL = os.environ.get(
    "SPECIALIST_V2_TARGET_REL",
    "checkpoints/specialist-v2",
)

SPECIALIST_V2_SEQUENCE_LEN = int(
    os.environ.get("SPECIALIST_V2_SEQUENCE_LEN", "10240")
)

SPECIALIST_V2_BASE_TRAIN_REL = os.environ.get(
    "SPECIALIST_V2_BASE_TRAIN_REL",
    "data/train_specialist.jsonl",
)

SPECIALIST_V2_BASE_VALIDATION_REL = os.environ.get(
    "SPECIALIST_V2_BASE_VALIDATION_REL",
    "data/validation_specialist.jsonl",
)

# Balanced all-35 bootstrap datasets. The first V2 pass intentionally uses a
# modest, equal number of examples per advisor. The Full Expert DNA remains in
# every sample, while production feedback grows the dataset over time.
SPECIALIST_V2_GENERATED_TRAIN_REL = os.environ.get(
    "SPECIALIST_V2_GENERATED_TRAIN_REL",
    "data/train_specialist_v2_generated.jsonl",
)

SPECIALIST_V2_GENERATED_VALIDATION_REL = os.environ.get(
    "SPECIALIST_V2_GENERATED_VALIDATION_REL",
    "data/validation_specialist_v2_generated.jsonl",
)

SPECIALIST_V2_TARGET_TRAIN_PER_ADVISOR = int(
    os.environ.get("SPECIALIST_V2_TARGET_TRAIN_PER_ADVISOR", "10")
)

SPECIALIST_V2_TARGET_VALIDATION_PER_ADVISOR = int(
    os.environ.get("SPECIALIST_V2_TARGET_VALIDATION_PER_ADVISOR", "2")
)

_SPECIALIST_V2_TRAINING_LOCK = threading.Lock()
_SPECIALIST_DATASET_BUILD_LOCK = threading.Lock()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value, fallback="item"):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value[:120] or fallback


def _github_contents_url(path):
    if not GITHUB_REPO:
        raise RuntimeError(
            "GITHUB_REPO environment variable is required for GitHub storage."
        )
    encoded_path = urllib.parse.quote(
        str(path).lstrip("/"),
        safe="/",
    )
    return (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/contents/{encoded_path}"
    )


def _github_api_json(
    method,
    url,
    token,
    payload=None,
    timeout=30,
):
    body = None

    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "athar-os-runpod-serverless",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode("utf-8")
            return (
                json.loads(raw)
                if raw.strip()
                else {}
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        if exc.code == 404:
            return None

        raise RuntimeError(
            f"GitHub API {method} failed "
            f"with HTTP {exc.code}: {raw[:1500]}"
        ) from exc


def _github_get_file(path, token):
    data = _github_api_json(
        "GET",
        _github_contents_url(path),
        token,
    )

    if data is None:
        return None, None

    encoded = data.get("content", "")
    encoding = data.get("encoding")

    if encoding != "base64":
        raise RuntimeError(
            f"Unexpected GitHub content encoding "
            f"for {path}: {encoding}"
        )

    raw = base64.b64decode(
        encoded.replace("\n", "")
    )

    return raw, data.get("sha")


def _github_get_json(path, token):
    raw, sha = _github_get_file(
        path,
        token,
    )

    if raw is None:
        return None, None

    try:
        return (
            json.loads(raw.decode("utf-8")),
            sha,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Stored continual-learning file is "
            f"not valid JSON: {path}"
        ) from exc


def _github_upsert_json(
    path,
    payload,
    token,
    message,
):
    _, existing_sha = _github_get_file(
        path,
        token,
    )

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    ).encode("utf-8")

    body = {
        "message": message,
        "content": base64.b64encode(
            raw
        ).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }

    if existing_sha:
        body["sha"] = existing_sha

    result = _github_api_json(
        "PUT",
        _github_contents_url(path),
        token,
        payload=body,
        timeout=60,
    )

    return {
        "path": path,
        "commit_sha": (
            (result or {})
            .get("commit", {})
            .get("sha")
        ),
    }



def _s3_client():
    if not ATHAR_S3_BUCKET:
        raise RuntimeError(
            "ATHAR_S3_BUCKET (or S3_BUCKET) is required "
            "for continual-learning storage."
        )

    try:
        import boto3
    except Exception as exc:
        raise RuntimeError(
            "boto3 is required for S3 continual-learning storage. "
            "The RunPod Python environment normally installs it with runpod."
        ) from exc

    kwargs = {}

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )

    if region:
        kwargs["region_name"] = region

    if ATHAR_S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = (
            ATHAR_S3_ENDPOINT_URL
        )

    return boto3.client(
        "s3",
        **kwargs,
    )


def _s3_key(path):
    path = str(path).lstrip("/")

    if ATHAR_S3_PREFIX:
        return (
            f"{ATHAR_S3_PREFIX}/"
            f"{path}"
        )

    return path


def _s3_put_json(
    path,
    payload,
):
    client = _s3_client()

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    ).encode("utf-8")

    key = _s3_key(path)

    client.put_object(
        Bucket=ATHAR_S3_BUCKET,
        Key=key,
        Body=raw,
        ContentType="application/json; charset=utf-8",
    )

    return {
        "path": path,
        "s3_key": key,
        "bucket": ATHAR_S3_BUCKET,
    }


def _s3_get_json(path):
    client = _s3_client()
    key = _s3_key(path)

    try:
        response = client.get_object(
            Bucket=ATHAR_S3_BUCKET,
            Key=key,
        )
    except Exception as exc:
        # Keep this generic because S3-compatible providers do not all expose
        # exactly the same exception class hierarchy.
        code = (
            getattr(exc, "response", {})
            .get("Error", {})
            .get("Code")
        )

        if str(code) in {
            "NoSuchKey",
            "404",
            "NotFound",
        }:
            return None

        raise

    raw = response["Body"].read()

    return json.loads(
        raw.decode("utf-8")
    )


def _s3_list_paths(prefix):
    client = _s3_client()
    full_prefix = _s3_key(
        prefix
    )

    paths = []
    continuation = None

    while True:
        kwargs = {
            "Bucket": ATHAR_S3_BUCKET,
            "Prefix": full_prefix,
            "MaxKeys": 1000,
        }

        if continuation:
            kwargs[
                "ContinuationToken"
            ] = continuation

        response = client.list_objects_v2(
            **kwargs
        )

        for item in response.get(
            "Contents",
            [],
        ):
            key = item.get("Key")

            if not key:
                continue

            if (
                ATHAR_S3_PREFIX
                and key.startswith(
                    ATHAR_S3_PREFIX + "/"
                )
            ):
                relative = key[
                    len(ATHAR_S3_PREFIX) + 1:
                ]
            else:
                relative = key

            paths.append(relative)

        if not response.get(
            "IsTruncated",
            False,
        ):
            break

        continuation = response.get(
            "NextContinuationToken"
        )

        if not continuation:
            break

    return paths


def _continual_storage_description():
    if CONTINUAL_STORAGE_BACKEND == "s3":
        return {
            "backend": "s3",
            "bucket": ATHAR_S3_BUCKET or None,
            "prefix": ATHAR_S3_PREFIX or None,
            "endpoint_url": (
                ATHAR_S3_ENDPOINT_URL
                or None
            ),
        }

    if CONTINUAL_STORAGE_BACKEND == "github":
        return {
            "backend": "github",
            "repo": GITHUB_REPO,
            "branch": GITHUB_BRANCH,
        }

    return {
        "backend": CONTINUAL_STORAGE_BACKEND,
    }


def _continual_put_json(
    path,
    payload,
    message,
):
    if CONTINUAL_STORAGE_BACKEND == "s3":
        return _s3_put_json(
            path,
            payload,
        )

    if CONTINUAL_STORAGE_BACKEND == "github":
        token = os.environ.get(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is required when "
                "CONTINUAL_STORAGE_BACKEND=github."
            )

        return _github_upsert_json(
            path,
            payload,
            token,
            message,
        )

    raise RuntimeError(
        "CONTINUAL_STORAGE_BACKEND must be 's3' or 'github'."
    )


def _continual_get_json(path):
    if CONTINUAL_STORAGE_BACKEND == "s3":
        return _s3_get_json(
            path
        )

    if CONTINUAL_STORAGE_BACKEND == "github":
        token = os.environ.get(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is required when "
                "CONTINUAL_STORAGE_BACKEND=github."
            )

        payload, _ = _github_get_json(
            path,
            token,
        )

        return payload

    raise RuntimeError(
        "CONTINUAL_STORAGE_BACKEND must be 's3' or 'github'."
    )


def _continual_list_paths(prefix):
    if CONTINUAL_STORAGE_BACKEND == "s3":
        return _s3_list_paths(
            prefix
        )

    if CONTINUAL_STORAGE_BACKEND == "github":
        # GitHub mode is intended only as a compatibility fallback.
        # Training/status reads GitHub data from a fresh clone instead.
        return []

    raise RuntimeError(
        "CONTINUAL_STORAGE_BACKEND must be 's3' or 'github'."
    )


def _load_continual_feedback_records(
    repo_dir=None,
):
    records = []

    if CONTINUAL_STORAGE_BACKEND == "s3":
        # Initial Specialist V2 training must not be blocked just because
        # continual-learning storage has not been configured yet. In that
        # case there simply cannot be any stored feedback to merge.
        if not ATHAR_S3_BUCKET:
            print(
                "Continual-learning S3 bucket is not configured; "
                "continuing with base Specialist V2 data only.",
                flush=True,
            )
            return records

        prefix = (
            f"{CONTINUAL_ROOT_REL}/"
            "feedback/"
        )

        for path in _continual_list_paths(
            prefix
        ):
            if not path.endswith(
                ".json"
            ):
                continue

            try:
                payload = (
                    _continual_get_json(
                        path
                    )
                )

                if isinstance(
                    payload,
                    dict,
                ):
                    records.append(
                        (path, payload)
                    )

            except Exception as exc:
                print(
                    "Skipping unreadable S3 feedback "
                    f"{path}: {exc}",
                    flush=True,
                )

        return records

    if CONTINUAL_STORAGE_BACKEND == "github":
        if not repo_dir:
            return records

        feedback_glob = os.path.join(
            repo_dir,
            CONTINUAL_ROOT_REL,
            "feedback",
            "**",
            "*.json",
        )

        for path in sorted(
            glob.glob(
                feedback_glob,
                recursive=True,
            )
        ):
            try:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    payload = json.load(file)

                records.append(
                    (path, payload)
                )
            except Exception as exc:
                print(
                    f"Skipping invalid feedback file "
                    f"{path}: {exc}",
                    flush=True,
                )

        return records

    return records


def _advisor_number_from_id(advisor_id):
    text = str(advisor_id or "").strip()

    match = re.search(
        r"(\d{1,2})(?!.*\d)",
        text,
    )

    if not match:
        return None

    number = int(match.group(1))

    if 1 <= number <= 35:
        return number

    return None


def _quality_flags_for_text(
    advisor_id,
    text,
):
    text = str(text or "")
    flags = []

    # Known corruption patterns seen in production tests.
    suspicious_literals = [
        "EXIST",
        "_EXIST",
        "ت_EXIST",
        "لاEXIST",
        "لا_EXIST",
    ]

    hits = [
        item
        for item in suspicious_literals
        if item in text
    ]

    if hits:
        flags.append({
            "type": "foreign_placeholder_noise",
            "values": hits,
        })

    # Cyrillic / CJK / Hangul corruption should never appear in the Arabic
    # advisory prose. Latin technical terms alone are not treated as corruption.
    if re.search(
        r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]",
        text,
    ):
        flags.append({
            "type": "unexpected_script",
        })

    invalid_refs = []

    for match in re.finditer(
        r"(?:المستشار|advisor)"
        r"\s*(?:الـ|ال)?\s*"
        r"(\d{1,3})",
        text,
        flags=re.IGNORECASE,
    ):
        number = int(match.group(1))

        if number > 35:
            invalid_refs.append(number)

    if invalid_refs:
        flags.append({
            "type": "invalid_advisor_reference",
            "values": sorted(set(invalid_refs)),
        })

    normalized_lines = [
        re.sub(
            r"\s+",
            " ",
            line,
        ).strip().lower()
        for line in text.splitlines()
        if len(
            re.sub(r"\s+", " ", line).strip()
        ) >= 25
    ]

    if len(normalized_lines) >= 10:
        unique_ratio = (
            len(set(normalized_lines))
            / len(normalized_lines)
        )

        if unique_ratio < 0.65:
            flags.append({
                "type": "strong_repetition",
                "unique_ratio": round(
                    unique_ratio,
                    4,
                ),
            })

    if len(text.strip()) < 160:
        flags.append({
            "type": "too_short",
        })

    return flags


def _consultation_quality_report(response):
    advisor_reports = []

    for item in (
        response.get("advisor_outputs", [])
        if isinstance(response, dict)
        else []
    ):
        if not isinstance(item, dict):
            continue

        advisor_id = item.get("advisor_id")
        opinion = item.get("opinion", "")

        flags = _quality_flags_for_text(
            advisor_id,
            opinion,
        )

        advisor_reports.append({
            "advisor_id": advisor_id,
            "flags": flags,
        })

    meta_debug = (
        response.get("meta_debug", {})
        if isinstance(response, dict)
        else {}
    )

    grounding_violations = (
        meta_debug.get(
            "final_grounding_violations",
            [],
        )
        if isinstance(meta_debug, dict)
        else []
    )

    return {
        "advisor_reports": advisor_reports,
        "meta_grounding_violations": grounding_violations,
        "has_quality_warning": bool(
            grounding_violations
            or any(
                row["flags"]
                for row in advisor_reports
            )
        ),
    }


def _consultation_run_path(run_id):
    return (
        f"{CONTINUAL_ROOT_REL}/runs/"
        f"{_safe_component(run_id, 'run')}.json"
    )


def _log_consultation_for_learning(
    job_input,
    response,
    internal_context=None,
):
    if (
        not CONTINUAL_LEARNING_ENABLED
        or not CONTINUAL_AUTO_LOG_RUNS
    ):
        return None

    run_id = (
        job_input.get("run_id")
        or str(uuid.uuid4())
    )

    # Public responses intentionally omit raw specialist opinions. Preserve them
    # only in the private continual-learning log so advisory_feedback can still
    # find the exact advisor answer without leaking internal council detail to UI.
    log_response = dict(response) if isinstance(response, dict) else response
    if isinstance(log_response, dict) and isinstance(internal_context, dict):
        if isinstance(internal_context.get("advisor_outputs"), list):
            log_response["advisor_outputs"] = internal_context["advisor_outputs"]
        if isinstance(internal_context.get("meta_debug"), dict):
            log_response["meta_debug"] = internal_context["meta_debug"]
        if isinstance(internal_context.get("timings_seconds"), dict):
            log_response["timings_seconds"] = internal_context["timings_seconds"]

    record = {
        "schema_version": "athar_continual_run_v2",
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "request": job_input,
        "response": log_response,
        "quality": _consultation_quality_report(
            log_response
        ),
    }

    try:
        result = _continual_put_json(
            _consultation_run_path(
                run_id
            ),
            record,
            message=(
                "Log Athar advisory consultation "
                f"{run_id}"
            ),
        )

        print(
            "Continual learning: consultation "
            f"{run_id} logged to "
            f"{CONTINUAL_STORAGE_BACKEND}.",
            flush=True,
        )

        return result

    except Exception as exc:
        # Logging must never break a production consultation.
        print(
            "Continual learning warning: "
            f"failed to persist consultation log: {exc}",
            flush=True,
        )
        return None


def _runtime_case_context_from_request(
    request,
):
    raw = request.get("input", {})

    if not isinstance(raw, dict):
        raw = {}

    # The selected advisor list is routing metadata rather than part of the
    # organization's case evidence, so do not teach it as case content.
    case_context = {
        key: value
        for key, value in raw.items()
        if key != "advisors"
    }

    for key in (
        "topic",
        "consultation_id",
        "track",
        "goal",
        "impact_map",
    ):
        if (
            key in request
            and key not in case_context
        ):
            case_context[key] = request[key]

    return case_context


def _build_runtime_training_user(
    request,
    advisor_id,
):
    payload = {
        "instruction": (
            "استخدم System Prompt الأصلي كمنهج داخلي للتحليل، "
            "ولا تعِد كتابته أو تلخيصه. ابدأ مباشرة بتحليل الحالة "
            "المقدمة، وافصل الحقائق عن الاستنتاجات والافتراضات. "
            "طبّق أدوات المستشار على الحالة، وقدّم الأولويات "
            "والتوصيات والمخاطر والمؤشرات وفجوات البيانات دون "
            "اختراع أرقام أو مستهدفات غير موجودة."
        ),
        "advisor_id": advisor_id,
        "case_context": (
            _runtime_case_context_from_request(
                request
            )
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _find_advisor_output(
    run_record,
    advisor_id,
):
    response = run_record.get(
        "response",
        {},
    )

    for item in response.get(
        "advisor_outputs",
        [],
    ):
        if not isinstance(item, dict):
            continue

        if str(
            item.get("advisor_id")
        ).strip() == str(advisor_id).strip():
            return item

    return None


def advisory_feedback(job_input):
    if not CONTINUAL_LEARNING_ENABLED:
        return {
            "status": "disabled",
            "type": "advisory_feedback",
            "message": (
                "CONTINUAL_LEARNING_ENABLED is disabled."
            ),
        }

    run_id = str(
        job_input.get("run_id")
        or ""
    ).strip()

    advisor_id = str(
        job_input.get("advisor_id")
        or ""
    ).strip()

    rating = str(
        job_input.get("rating")
        or ""
    ).strip().lower()

    corrected_output = str(
        job_input.get("corrected_output")
        or ""
    ).strip()

    notes = str(
        job_input.get("notes")
        or ""
    ).strip()

    force_include = bool(
        job_input.get(
            "force_include",
            False,
        )
    )

    if not run_id:
        raise ValueError(
            "advisory_feedback requires run_id."
        )

    if not advisor_id:
        raise ValueError(
            "advisory_feedback requires advisor_id."
        )

    if _advisor_number_from_id(
        advisor_id
    ) is None:
        raise ValueError(
            "advisor_id must resolve to an Athar advisor 1..35."
        )

    valid_ratings = {
        "accepted",
        "rejected",
        "corrected",
    }

    if rating not in valid_ratings:
        raise ValueError(
            "rating must be accepted, rejected, or corrected."
        )

    run_record = job_input.get(
        "run_snapshot"
    )

    if not isinstance(
        run_record,
        dict,
    ):
        run_record = _continual_get_json(
            _consultation_run_path(
                run_id
            )
        )

    if not isinstance(
        run_record,
        dict,
    ):
        raise FileNotFoundError(
            f"No consultation log found for run_id={run_id}."
        )

    advisor_output = _find_advisor_output(
        run_record,
        advisor_id,
    )

    if advisor_output is None:
        raise ValueError(
            f"Advisor {advisor_id} was not found "
            f"in consultation run {run_id}."
        )

    original_output = str(
        advisor_output.get("opinion")
        or ""
    ).strip()

    if rating == "corrected":
        if len(
            corrected_output
        ) < 160:
            raise ValueError(
                "corrected_output is required for rating='corrected' "
                "and must contain a substantive advisor answer."
            )

        chosen_output = corrected_output

    elif rating == "accepted":
        chosen_output = original_output

    else:
        chosen_output = ""

    quality_flags = (
        _quality_flags_for_text(
            advisor_id,
            chosen_output,
        )
        if chosen_output
        else []
    )

    severe_types = {
        "foreign_placeholder_noise",
        "unexpected_script",
        "invalid_advisor_reference",
        "strong_repetition",
        "too_short",
    }

    severe_quality_issue = any(
        flag.get("type")
        in severe_types
        for flag in quality_flags
    )

    training_eligible = bool(
        chosen_output
        and rating in {
            "accepted",
            "corrected",
        }
        and (
            force_include
            or not severe_quality_issue
        )
    )

    request_snapshot = (
        run_record.get(
            "request",
            {},
        )
    )

    feedback_id = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:10]
    )

    feedback_record = {
        "schema_version": "athar_advisory_feedback_v1",
        "feedback_id": feedback_id,
        "created_at": _utc_now_iso(),
        "run_id": run_id,
        "advisor_id": advisor_id,
        "rating": rating,
        "notes": notes or None,
        "quality_flags": quality_flags,
        "force_include": force_include,
        "training_eligible": training_eligible,
        "original_output": original_output,
        "corrected_output": (
            corrected_output
            if rating == "corrected"
            else None
        ),
        "training_example": (
            {
                "advisor_id": advisor_id,
                "user": _build_runtime_training_user(
                    request_snapshot,
                    advisor_id,
                ),
                "assistant": chosen_output,
            }
            if training_eligible
            else None
        ),
    }

    path = (
        f"{CONTINUAL_ROOT_REL}/feedback/"
        f"{feedback_id}__"
        f"{_safe_component(run_id, 'run')}__"
        f"{_safe_component(advisor_id, 'advisor')}.json"
    )

    saved = _continual_put_json(
        path,
        feedback_record,
        message=(
            "Add Athar advisory feedback "
            f"{feedback_id}"
        ),
    )

    return {
        "status": "saved",
        "type": "advisory_feedback",
        "feedback_id": feedback_id,
        "run_id": run_id,
        "advisor_id": advisor_id,
        "rating": rating,
        "training_eligible": training_eligible,
        "quality_flags": quality_flags,
        "storage": (
            _continual_storage_description()
        ),
        "saved_path": saved.get(
            "path"
        ),
        "s3_key": saved.get(
            "s3_key"
        ),
        "github_commit": saved.get(
            "commit_sha"
        ),
    }


def _messages_by_role(row):
    result = {}

    for message in row.get(
        "messages",
        [],
    ):
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role")
            or ""
        ).strip()

        content = str(
            message.get("content")
            or ""
        )

        if role:
            result.setdefault(
                role,
                [],
            ).append(content)

    return result


def _extract_training_advisor_number(row):
    metadata = row.get("metadata") if isinstance(row, dict) else None
    if isinstance(metadata, dict):
        try:
            metadata_number = int(metadata.get("advisor_number"))
        except Exception:
            metadata_number = None
        if metadata_number is not None and 1 <= metadata_number <= 35:
            return metadata_number

    messages = _messages_by_role(
        row
    )

    combined = "\n".join(
        text
        for values in messages.values()
        for text in values
    )

    patterns = [
        r"ADVISOR_ID\s*:\s*([A-Z0-9-]+)",
        r"(?:ATHAR-ADV-|AOS-[A-Z]+-)(\d{1,2})",
        r"ADVISOR\s+(\d{1,2})",
    ]

    for index, pattern in enumerate(patterns):
        match = re.search(
            pattern,
            combined,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = match.group(1)

        if index == 0:
            number = _advisor_number_from_id(
                value
            )
        else:
            try:
                number = int(value)
            except ValueError:
                number = None

        if (
            number is not None
            and 1 <= number <= 35
        ):
            return number

    return None


def _full_advisor_prompt(
    repo_dir,
    advisor_number,
):
    path = os.path.join(
        repo_dir,
        "prompts",
        "advisors",
        f"advisor_{advisor_number:02d}.md",
    )

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Full advisor prompt missing: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        prompt = file.read().strip()

    if len(prompt) < 1000:
        raise RuntimeError(
            f"Advisor prompt looks unexpectedly short: {path}"
        )

    return prompt


def _clean_old_specialist_user(text):
    text = re.sub(
        r"(?i)^\s*ADVISOR_ID\s*:\s*"
        r"[A-Z0-9-]+\s*",
        "",
        str(text or ""),
        count=1,
    )

    return text.strip()


def _example_hash(messages):
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _write_jsonl(path, rows):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL {path} line {line_number}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def _load_jsonl_if_exists(path):
    if not os.path.isfile(path):
        return []
    return _load_jsonl(path)


def _generated_dataset_path(repo_dir, split):
    if split == "train":
        rel = SPECIALIST_V2_GENERATED_TRAIN_REL
    elif split == "validation":
        rel = SPECIALIST_V2_GENERATED_VALIDATION_REL
    else:
        raise ValueError("split must be train or validation")
    return os.path.join(repo_dir, rel)


def _rows_by_advisor(rows):
    grouped = {number: [] for number in range(1, 36)}
    for row in rows:
        number = _extract_training_advisor_number(row)
        if number is not None and 1 <= number <= 35:
            grouped[number].append(row)
    return grouped


def _specialist_generated_example_is_valid(case_text, answer_text):
    case_text = str(case_text or "").strip()
    answer_text = str(answer_text or "").strip()

    if len(case_text) < 100 or len(case_text) > 2400:
        return False, "case_length"
    if len(answer_text) < 450 or len(answer_text) > 9000:
        return False, "answer_length"

    # Reject obvious prompt echo / identity dump.
    normalized = re.sub(r"\s+", " ", answer_text).lower()
    forbidden_markers = [
        "system prompt",
        "owned outcome",
        "scope boundaries",
        "advisor identity",
        "الرمز النظامي",
        "نوع المكوّن",
        "حالة البرومبت",
        "هوية المستشار",
    ]
    marker_hits = sum(
        1 for marker in forbidden_markers
        if marker.lower() in normalized
    )
    if marker_hits >= 2:
        return False, "prompt_echo"

    # No corrupted CJK leakage in Arabic training answers.
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", answer_text):
        return False, "cjk_leakage"

    # Never teach references to advisors outside the authoritative 35.
    for match in re.finditer(
        r"(?:AOS-[A-Z]+-|ATHAR-ADV-)(\d{1,2})",
        answer_text,
        flags=re.IGNORECASE,
    ):
        if int(match.group(1)) > 35:
            return False, "invalid_advisor_id"

    # Multi-digit numerical claims in the answer must be grounded in the case.
    # Single digits are allowed for list numbering / framework steps.
    case_numbers = set(re.findall(r"(?<!\d)\d{2,4}(?!\d)", case_text))
    answer_numbers = set(re.findall(r"(?<!\d)\d{2,4}(?!\d)", answer_text))
    unsupported = sorted(answer_numbers - case_numbers)
    if unsupported:
        return False, "unsupported_numbers:" + ",".join(unsupported[:8])

    lines = [
        re.sub(r"\s+", " ", line).strip().lower()
        for line in answer_text.splitlines()
        if len(re.sub(r"\s+", " ", line).strip()) >= 30
    ]
    if len(lines) >= 10:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.72:
            return False, "repetition"

    return True, "ok"


def _specialist_dataset_author_prompt(full_prompt):
    instructions = """# ATHAR DATASET AUTHORING MODE
أنت الآن تبني مثال تدريب واحد عالي الجودة لهذا المستشار، وليس مطلوبًا منك شرح البرومبت أو تعريف المستشار.
أنشئ حالة واقعية لمنظمة غير ربحية تقع بوضوح داخل اختصاص هذا المستشار، ثم اكتب الإجابة المثالية التي نريد أن يتعلمها النموذج.

قواعد إلزامية:
- لا تنسخ أو تلخص System Prompt ولا تعرض Identity/Mission/Owned Outcome/Scope/Tools.
- الحالة يجب أن تحتوي وقائع كافية ليبني المستشار عليها رأيًا مهنيًا.
- افصل في الإجابة بين الحقائق، الاستنتاجات، والافتراضات عند الحاجة.
- طبّق أدوات ومنهجيات المستشار على الحالة بدل شرحها نظريًا.
- لا تخترع أرقامًا أو نسبًا أو مددًا أو مستهدفات غير موجودة في نص الحالة.
- إذا غابت معلومة، استخدم صياغة مثل: لا يظهر في البيانات المقدمة.
- لا تحل محل مستشار آخر؛ الإحالة مسموحة فقط إذا احتاجت الحالة فعلًا اختصاصًا آخر.
- لا تذكر أي مستشار برقم خارج 01..35.
- العربية يجب أن تكون سليمة وواضحة بلا كلمات صينية أو نص مشوّه.
- اجعل المثال مختلفًا ماديًا عن الأمثلة السابقة في السياق والقرار المطلوب.
- اجعل الإجابة مفصلة ومهنية ولكن مركزة، من دون حشو أو إعادة صياغة الحالة عدة مرات.

مهم جدًا: لا تستخدم JSON ولا Markdown code fences.
استخدم هذا القالب النصي حرفيًا:
CASE:
نص الحالة هنا
ANSWER:
نص الإجابة هنا
END_EXAMPLE
"""
    return full_prompt + "\n\n" + instructions


def _specialist_dataset_critic_prompt(full_prompt):
    instructions = """# ATHAR TRAINING DATA QUALITY GATE
راجع الأمثلة المرشحة قبل إدخالها للتدريب.
مرّر المثال فقط إذا كان داخل اختصاص المستشار، خاصًا بالحالة، لا يكرر البرومبت، لا يخترع أرقامًا غير موجودة في الحالة، ولا يخرج إلى اختصاصات أخرى، وإجابته مكتملة وليست مقطوعة.
لا تعِد كتابة الإجابة.

مهم: لا تستخدم JSON.
لكل مثال أخرج سطرًا واحدًا فقط بهذا الشكل:
0|PASS
أو:
0|REJECT|سبب مختصر
واستخدم رقم index الموجود في المدخل.
"""
    return full_prompt + "\n\n" + instructions


def _specialist_salvage_complete_examples(raw_text):
    """Parse one bootstrap example robustly.

    New authoring calls use simple CASE/ANSWER delimiters to avoid spending
    minutes producing valid escaped JSON. Legacy JSON is still accepted so old
    jobs/retries remain compatible.
    """
    cleaned = (
        str(raw_text or "")
        .strip()
        .replace("```json", "")
        .replace("```text", "")
        .replace("```", "")
        .strip()
    )
    if not cleaned:
        return []

    # Backward-compatible JSON fast path.
    try:
        parsed = json.loads(cleaned)
        examples = parsed.get("examples", []) if isinstance(parsed, dict) else []
        valid = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            case_text = str(item.get("case") or "").strip()
            answer_text = str(item.get("answer") or "").strip()
            if case_text and answer_text:
                valid.append({"case": case_text, "answer": answer_text})
        if valid:
            return valid
    except Exception:
        pass

    # Robust plain-text format. Accept English or Arabic marker variants.
    case_match = re.search(
        r"(?:^|\n)\s*(?:CASE|الحالة)\s*:\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    answer_match = re.search(
        r"(?:^|\n)\s*(?:ANSWER|الإجابة)\s*:\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not case_match or not answer_match or answer_match.start() <= case_match.end():
        return []

    case_text = cleaned[case_match.end():answer_match.start()].strip()
    answer_text = cleaned[answer_match.end():].strip()
    answer_text = re.split(
        r"(?:^|\n)\s*END_EXAMPLE\s*$",
        answer_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if not case_text or not answer_text:
        return []

    return [{"case": case_text, "answer": answer_text}]


def _specialist_generate_candidate_call(
    full_prompt,
    advisor_number,
    split,
    count,
    variation_offset,
    angles,
    max_new_tokens,
    max_time_seconds=150,
):
    # One candidate per call is deliberate. It makes truncation recoverable and
    # keeps progress visible in RunPod logs.
    payload = {
        "advisor_number": advisor_number,
        "split": split,
        "examples_required": 1,
        "variation_offset": variation_offset,
        "scenario_angle": (angles or [""])[0],
        "instruction": (
            "أنشئ مثالًا واحدًا فقط. لا تستخدم اسم جمعية حقيقية. "
            "لا تجعل الإجابة قصيرة أو عامة. التزم بقالب CASE ثم ANSWER ثم END_EXAMPLE."
        ),
    }

    print(
        f"[all35] advisor={advisor_number:02d} split={split} "
        f"generate offset={variation_offset} start",
        flush=True,
    )
    started = time.time()
    raw_text, _ = _v18_generate_text(
        _specialist_dataset_author_prompt(full_prompt),
        payload,
        max_new_tokens=max_new_tokens,
        max_time_seconds=max_time_seconds,
    )
    examples = _specialist_salvage_complete_examples(raw_text)

    # If generation consumed virtually the whole token allowance and never
    # emitted END_EXAMPLE, consider it truncated even if CASE/ANSWER parsed.
    try:
        output_tokens = len(
            _RICH_TOKENIZER(
                raw_text,
                add_special_tokens=False,
            )["input_ids"]
        )
    except Exception:
        output_tokens = None

    ended_cleanly = bool(
        re.search(r"END_EXAMPLE\s*$", str(raw_text or "").strip(), flags=re.IGNORECASE)
    )
    token_truncated = (
        output_tokens is not None
        and output_tokens >= max(1, int(max_new_tokens) - 8)
        and not ended_cleanly
    )
    if token_truncated:
        examples = []

    print(
        f"[all35] advisor={advisor_number:02d} split={split} "
        f"generate done seconds={time.time() - started:.1f} "
        f"parsed={len(examples)} raw_chars={len(raw_text)} "
        f"output_tokens={output_tokens} ended={ended_cleanly} truncated={token_truncated}",
        flush=True,
    )
    return examples, raw_text


def _generate_specialist_dataset_batch(
    repo_dir,
    advisor_number,
    split,
    batch_size,
    variation_offset,
):
    ensure_rich_router_model()

    full_prompt = _full_advisor_prompt(
        repo_dir,
        advisor_number,
    )

    scenario_angles = [
        "نمو أو توسع يحتاج قرارًا",
        "تعارض أولويات وموارد محدودة",
        "بيانات ناقصة مع قرار مطلوب",
        "تعقيد تشغيلي بين عدة برامج",
        "مخاطر تنفيذ أو اعتماديات",
        "جودة خدمة وتجربة مستفيد",
        "تحول أو تغيير في طريقة العمل",
        "شراكات وأطراف متعددة",
        "ضغط موسمي أو زيادة طلب",
        "حاجة إلى قياس ومتابعة",
        "استدامة قدرة قائمة بعد نجاح سابق",
        "حالة فيها بدائل ومفاضلة واضحة",
    ]

    angles = [
        scenario_angles[(variation_offset + i) % len(scenario_angles)]
        for i in range(batch_size)
    ]

    candidates = []
    seen_cases = set()
    max_single_attempts = max(3, batch_size * 3)
    single_attempt = 0

    while len(candidates) < batch_size and single_attempt < max_single_attempts:
        target_index = len(candidates)
        single_angle = [angles[target_index % len(angles)]]
        single_offset = variation_offset + single_attempt

        generated, _ = _specialist_generate_candidate_call(
            full_prompt=full_prompt,
            advisor_number=advisor_number,
            split=split,
            count=1,
            variation_offset=single_offset,
            angles=single_angle,
            max_new_tokens=2800,
            max_time_seconds=150,
        )

        if generated:
            item = generated[0]
            case_key = re.sub(
                r"\s+",
                " ",
                str(item.get("case") or ""),
            ).strip().lower()
            if case_key and case_key not in seen_cases:
                seen_cases.add(case_key)
                candidates.append(item)
                print(
                    f"[all35] advisor={advisor_number:02d} split={split} "
                    f"candidate ready={len(candidates)}/{batch_size}",
                    flush=True,
                )
            else:
                print(
                    f"[all35] advisor={advisor_number:02d} split={split} "
                    "candidate duplicate/empty; retrying",
                    flush=True,
                )
        else:
            print(
                f"[all35] advisor={advisor_number:02d} split={split} "
                "candidate could not be parsed or was truncated; retrying",
                flush=True,
            )

        single_attempt += 1

    candidates = candidates[:batch_size]

    # Critic output is line based, not JSON, so a tiny formatting mistake can no
    # longer discard a good candidate.
    critic_payload = {
        "advisor_number": advisor_number,
        "split": split,
        "examples": [
            {
                "index": index,
                "case": str(item.get("case") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
            }
            for index, item in enumerate(candidates)
            if isinstance(item, dict)
        ],
    }

    critic_map = {}
    if critic_payload["examples"]:
        print(
            f"[all35] advisor={advisor_number:02d} split={split} "
            f"critic start examples={len(critic_payload['examples'])}",
            flush=True,
        )
        critic_started = time.time()
        critic_raw, _ = _v18_generate_text(
            _specialist_dataset_critic_prompt(full_prompt),
            critic_payload,
            max_new_tokens=320,
            max_time_seconds=60,
        )
        print(
            f"[all35] advisor={advisor_number:02d} split={split} "
            f"critic done seconds={time.time() - critic_started:.1f} raw={critic_raw[:300]!r}",
            flush=True,
        )
        for line in str(critic_raw or "").splitlines():
            m = re.match(
                r"\s*(\d+)\s*\|\s*(PASS|REJECT)(?:\s*\|\s*(.*))?\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if not m:
                continue
            idx = int(m.group(1))
            critic_map[idx] = {
                "pass": m.group(2).upper() == "PASS",
                "issues": [m.group(3).strip()] if m.group(3) else [],
            }

    accepted = []
    rejected = []

    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue

        case_text = str(item.get("case") or "").strip()
        answer_text = str(item.get("answer") or "").strip()

        review = critic_map.get(index)
        if isinstance(review, dict) and review.get("pass") is False:
            rejected.append({
                "index": index,
                "reason": "critic_rejected",
                "issues": review.get("issues", []),
            })
            continue

        valid, reason = _specialist_generated_example_is_valid(
            case_text,
            answer_text,
        )
        if not valid:
            rejected.append({
                "index": index,
                "reason": reason,
            })
            continue

        accepted.append({
            "messages": [
                {
                    "role": "user",
                    "content": case_text,
                },
                {
                    "role": "assistant",
                    "content": answer_text,
                },
            ],
            "metadata": {
                "advisor_number": advisor_number,
                "source": "all35_bootstrap_plaintext_plus_critic",
                "split": split,
                "created_at": _utc_now_iso(),
            },
        })

    return accepted, rejected


def _commit_specialist_generated_data(repo_dir, env, advisor_number):
    run_command(
        ["git", "config", "user.name", GIT_USER_NAME],
        cwd=repo_dir,
        env=env,
    )
    run_command(
        ["git", "config", "user.email", GIT_USER_EMAIL],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "add",
            SPECIALIST_V2_GENERATED_TRAIN_REL,
            SPECIALIST_V2_GENERATED_VALIDATION_REL,
        ],
        cwd=repo_dir,
        env=env,
    )

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_dir,
        env=env,
    )
    if status.returncode == 0:
        return "No generated dataset changes detected"

    run_command(
        [
            "git",
            "commit",
            "-m",
            f"Add Specialist V2 bootstrap data for advisor {advisor_number:02d}",
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        ["git", "pull", "--rebase", "origin", GITHUB_BRANCH],
        cwd=repo_dir,
        env=env,
        stream=True,
    )
    run_command(
        ["git", "push", "origin", GITHUB_BRANCH],
        cwd=repo_dir,
        env=env,
        stream=True,
    )

    return run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        env=env,
    ).strip()


def generate_specialist_all35_data(job_input):
    if not _SPECIALIST_DATASET_BUILD_LOCK.acquire(blocking=False):
        return {
            "status": "busy",
            "type": "build_specialist_all35_data",
            "message": "A dataset generation job is already running in this worker.",
        }

    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is required for build_specialist_all35_data."
            )

        advisor_start = int(job_input.get("advisor_start", 11))
        advisor_end = int(job_input.get("advisor_end", 35))
        if not (1 <= advisor_start <= advisor_end <= 35):
            raise ValueError("advisor_start/advisor_end must be within 1..35")

        target_train = int(
            job_input.get(
                "target_train_per_advisor",
                SPECIALIST_V2_TARGET_TRAIN_PER_ADVISOR,
            )
        )
        target_validation = int(
            job_input.get(
                "target_validation_per_advisor",
                SPECIALIST_V2_TARGET_VALIDATION_PER_ADVISOR,
            )
        )
        if not (1 <= target_train <= 58):
            raise ValueError("target_train_per_advisor must be 1..58")
        if not (1 <= target_validation <= 19):
            raise ValueError("target_validation_per_advisor must be 1..19")

        batch_size = int(job_input.get("batch_size", 3))
        batch_size = max(1, min(batch_size, 3))

        repo_dir, env = clone_repo_without_lfs(
            token,
            repo_dir="/tmp/athar_all35_dataset_repo",
        )

        # Confirm prompt availability before spending GPU time.
        for number in range(advisor_start, advisor_end + 1):
            _full_advisor_prompt(repo_dir, number)

        train_path = _generated_dataset_path(repo_dir, "train")
        validation_path = _generated_dataset_path(repo_dir, "validation")
        train_rows = _load_jsonl_if_exists(train_path)
        validation_rows = _load_jsonl_if_exists(validation_path)

        def counts_for(rows):
            grouped = _rows_by_advisor(rows)
            return {number: len(grouped[number]) for number in range(1, 36)}

        if job_input.get("preflight", False):
            return {
                "status": "all35_dataset_preflight_ok",
                "type": "build_specialist_all35_data",
                "advisor_range": [advisor_start, advisor_end],
                "target_train_per_advisor": target_train,
                "target_validation_per_advisor": target_validation,
                "generated_train_counts": counts_for(train_rows),
                "generated_validation_counts": counts_for(validation_rows),
                "note": (
                    "Existing advisors 01-10 already have legacy Specialist data; "
                    "generated files are mainly required for 11-35."
                ),
            }

        ensure_rich_router_model()

        progress = []
        rejected_total = 0
        latest_commit = None

        for advisor_number in range(advisor_start, advisor_end + 1):
            print(
                f"[all35] ===== advisor {advisor_number:02d} started =====",
                flush=True,
            )
            train_grouped = _rows_by_advisor(train_rows)
            validation_grouped = _rows_by_advisor(validation_rows)

            existing_train = len(train_grouped[advisor_number])
            existing_validation = len(validation_grouped[advisor_number])

            # Generate only the missing quantity so rerunning the job resumes.
            for split, target, rows, existing in [
                ("train", target_train, train_rows, existing_train),
                ("validation", target_validation, validation_rows, existing_validation),
            ]:
                needed = max(0, target - existing)
                print(
                    f"[all35] advisor={advisor_number:02d} split={split} "
                    f"existing={existing} target={target} needed={needed}",
                    flush=True,
                )
                attempts = 0
                max_attempts = max(4, needed * 3)

                while needed > 0 and attempts < max_attempts:
                    current_batch = min(batch_size, needed)
                    print(
                        f"[all35] advisor={advisor_number:02d} split={split} "
                        f"batch attempt={attempts + 1}/{max_attempts} size={current_batch}",
                        flush=True,
                    )
                    accepted, rejected = _generate_specialist_dataset_batch(
                        repo_dir,
                        advisor_number,
                        split,
                        current_batch,
                        variation_offset=(existing + attempts * batch_size),
                    )
                    rejected_total += len(rejected)
                    print(
                        f"[all35] advisor={advisor_number:02d} split={split} "
                        f"batch result accepted={len(accepted)} rejected={len(rejected)}",
                        flush=True,
                    )

                    # De-duplicate by user case, even if the model ignores a
                    # variation seed on a retry.
                    existing_cases = {
                        re.sub(r"\s+", " ", _messages_by_role(r).get("user", [""])[-1]).strip().lower()
                        for r in rows
                        if _messages_by_role(r).get("user")
                    }
                    for row in accepted:
                        user_case = _messages_by_role(row).get("user", [""])[-1]
                        key = re.sub(r"\s+", " ", user_case).strip().lower()
                        if not key or key in existing_cases:
                            continue
                        rows.append(row)
                        existing_cases.add(key)
                        existing += 1
                        needed -= 1
                        if needed <= 0:
                            break

                    attempts += 1

                if needed > 0:
                    raise RuntimeError(
                        f"Could not generate enough accepted {split} examples "
                        f"for advisor {advisor_number:02d}. Missing {needed}. "
                        "Rerun the same request to resume after reviewing logs."
                    )

            _write_jsonl(train_path, train_rows)
            _write_jsonl(validation_path, validation_rows)

            latest_commit = _commit_specialist_generated_data(
                repo_dir,
                env,
                advisor_number,
            )
            print(
                f"[all35] ===== advisor {advisor_number:02d} saved commit={latest_commit} =====",
                flush=True,
            )

            train_count = len(_rows_by_advisor(train_rows)[advisor_number])
            validation_count = len(_rows_by_advisor(validation_rows)[advisor_number])
            progress.append({
                "advisor_number": advisor_number,
                "train": train_count,
                "validation": validation_count,
                "github_commit": latest_commit,
            })

        return {
            "status": "completed",
            "type": "build_specialist_all35_data",
            "advisor_range": [advisor_start, advisor_end],
            "target_train_per_advisor": target_train,
            "target_validation_per_advisor": target_validation,
            "progress": progress,
            "rejected_candidates": rejected_total,
            "generated_train_counts": counts_for(train_rows),
            "generated_validation_counts": counts_for(validation_rows),
            "github_commit": latest_commit,
            "next_step": (
                "Run train_specialist_v2 with preflight=true. "
                "Full training will be blocked until all 35 advisors meet "
                "the balanced train/validation targets."
            ),
        }

    finally:
        _SPECIALIST_DATASET_BUILD_LOCK.release()


def build_specialist_v2_datasets(
    repo_dir,
    work_dir,
    include_feedback=True,
):
    train_source = os.path.join(
        repo_dir,
        SPECIALIST_V2_BASE_TRAIN_REL,
    )
    validation_source = os.path.join(
        repo_dir,
        SPECIALIST_V2_BASE_VALIDATION_REL,
    )

    if not os.path.isfile(train_source):
        raise RuntimeError(f"Base Specialist dataset missing: {train_source}")
    if not os.path.isfile(validation_source):
        raise RuntimeError(
            f"Base Specialist validation dataset missing: {validation_source}"
        )

    for advisor_number in range(1, 36):
        _full_advisor_prompt(repo_dir, advisor_number)

    def convert_rows(rows, source_name):
        converted = []
        for row in rows:
            roles = _messages_by_role(row)
            advisor_number = _extract_training_advisor_number(row)
            if advisor_number is None:
                raise RuntimeError(
                    f"Could not determine advisor ID from a Specialist row in {source_name}."
                )
            users = roles.get("user", [])
            assistants = roles.get("assistant", [])
            if not users or not assistants:
                raise RuntimeError(
                    f"Specialist row must contain user and assistant messages: {source_name}"
                )
            messages = [
                {
                    "role": "system",
                    "content": _full_advisor_prompt(repo_dir, advisor_number),
                },
                {
                    "role": "user",
                    "content": _clean_old_specialist_user(users[-1]),
                },
                {
                    "role": "assistant",
                    "content": assistants[-1].strip(),
                },
            ]
            converted.append({"messages": messages})
        return converted

    base_train_rows = convert_rows(_load_jsonl(train_source), train_source)
    base_validation_rows = convert_rows(
        _load_jsonl(validation_source), validation_source
    )

    generated_train_path = _generated_dataset_path(repo_dir, "train")
    generated_validation_path = _generated_dataset_path(repo_dir, "validation")

    generated_train_rows = convert_rows(
        _load_jsonl_if_exists(generated_train_path), generated_train_path
    )
    generated_validation_rows = convert_rows(
        _load_jsonl_if_exists(generated_validation_path), generated_validation_path
    )

    train_pool = base_train_rows + generated_train_rows
    validation_pool = base_validation_rows + generated_validation_rows

    train_grouped = _rows_by_advisor(train_pool)
    validation_grouped = _rows_by_advisor(validation_pool)

    target_train = SPECIALIST_V2_TARGET_TRAIN_PER_ADVISOR
    target_validation = SPECIALIST_V2_TARGET_VALIDATION_PER_ADVISOR

    balanced_train = []
    balanced_validation = []
    train_counts = {}
    validation_counts = {}
    train_shortages = {}
    validation_shortages = {}

    for number in range(1, 36):
        selected_train = train_grouped[number][:target_train]
        selected_validation = validation_grouped[number][:target_validation]

        balanced_train.extend(selected_train)
        balanced_validation.extend(selected_validation)
        train_counts[f"{number:02d}"] = len(selected_train)
        validation_counts[f"{number:02d}"] = len(selected_validation)

        if len(selected_train) < target_train:
            train_shortages[f"{number:02d}"] = target_train - len(selected_train)
        if len(selected_validation) < target_validation:
            validation_shortages[f"{number:02d}"] = (
                target_validation - len(selected_validation)
            )

    # Deduplicate the balanced baseline.
    seen_train = set()
    deduped_train = []
    for row in balanced_train:
        key = _example_hash(row["messages"])
        if key in seen_train:
            continue
        seen_train.add(key)
        deduped_train.append(row)

    continual_added = 0
    feedback_records = []
    if include_feedback:
        feedback_records = _load_continual_feedback_records(repo_dir=repo_dir)
        for path, feedback in feedback_records:
            if not feedback.get("training_eligible", False):
                continue
            example = feedback.get("training_example")
            if not isinstance(example, dict):
                continue
            advisor_number = _advisor_number_from_id(example.get("advisor_id"))
            if advisor_number is None:
                continue
            user_text = str(example.get("user") or "").strip()
            assistant_text = str(example.get("assistant") or "").strip()
            if len(user_text) < 40 or len(assistant_text) < 160:
                continue
            messages = [
                {
                    "role": "system",
                    "content": _full_advisor_prompt(repo_dir, advisor_number),
                },
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            key = _example_hash(messages)
            if key in seen_train:
                continue
            seen_train.add(key)
            deduped_train.append({"messages": messages})
            continual_added += 1

    os.makedirs(work_dir, exist_ok=True)
    train_path = os.path.join(work_dir, "train_specialist_v2.jsonl")
    validation_path = os.path.join(work_dir, "validation_specialist_v2.jsonl")
    _write_jsonl(train_path, deduped_train)
    _write_jsonl(validation_path, balanced_validation)

    covered_advisors = [
        number for number in range(1, 36)
        if train_counts.get(f"{number:02d}", 0) > 0
    ]

    all35_ready = not train_shortages and not validation_shortages

    return {
        "train_path": train_path,
        "validation_path": validation_path,
        "base_train_examples": len(base_train_rows),
        "base_validation_examples": len(base_validation_rows),
        "generated_train_examples": len(generated_train_rows),
        "generated_validation_examples": len(generated_validation_rows),
        "train_examples": len(deduped_train),
        "validation_examples": len(balanced_validation),
        "feedback_files_seen": len(feedback_records),
        "continual_examples_added": continual_added,
        "target_train_per_advisor": target_train,
        "target_validation_per_advisor": target_validation,
        "advisor_example_counts": train_counts,
        "validation_advisor_example_counts": validation_counts,
        "covered_advisors": covered_advisors,
        "uncovered_advisors": [
            number for number in range(1, 36)
            if number not in covered_advisors
        ],
        "train_shortages": train_shortages,
        "validation_shortages": validation_shortages,
        "all35_ready": all35_ready,
    }


def _clone_specialist_learning_source(
    token,
):
    repo_dir, env = clone_repo_without_lfs(
        token,
        repo_dir="/tmp/athar_specialist_v2_repo",
    )

    run_command(
        [
            "git",
            "lfs",
            "install",
            "--local",
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "lfs",
            "pull",
            (
                "--include="
                "checkpoints/specialist-v2/**,"
                "checkpoints/specialist/**"
            ),
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )

    v2_path = os.path.join(
        repo_dir,
        "checkpoints",
        "specialist-v2",
    )

    v1_path = os.path.join(
        repo_dir,
        "checkpoints",
        "specialist",
    )

    source_path = (
        v2_path
        if _real_adapter_checkpoint(v2_path)
        else v1_path
    )

    if not _real_adapter_checkpoint(
        source_path
    ):
        # Disaster-recovery mode: the deleted GitHub organization took the
        # historical LFS objects with it. We still have the complete balanced
        # all-35 training data, so build a fresh Specialist V2 LoRA directly
        # from Qwen3-14B instead of blocking on the missing V1/V2 adapter.
        source_path = None
        source_version = "base-qwen3-recovery"
    else:
        source_version = (
            "specialist-v2"
            if source_path == v2_path
            else "specialist"
        )

    return (
        repo_dir,
        source_path,
        source_version,
        env,
    )


def _training_gpu_profile():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Specialist V2 training requires a CUDA GPU."
        )

    properties = torch.cuda.get_device_properties(
        0
    )

    total_gb = (
        properties.total_memory
        / 1024
        / 1024
        / 1024
    )

    bf16_supported = bool(
        getattr(
            torch.cuda,
            "is_bf16_supported",
            lambda: False,
        )()
    )

    if total_gb < 22:
        raise RuntimeError(
            "Specialist V2 keeps the full 10,240-token DNA context. "
            f"The current GPU has only {total_gb:.1f} GB VRAM. "
            "Use a Serverless worker with at least about 24 GB VRAM."
        )

    constrained = total_gb < 40

    return {
        "name": properties.name,
        "total_vram_gb": round(
            total_gb,
            2,
        ),
        "bf16_supported": bf16_supported,
        "constrained_memory_mode": constrained,
    }


def _axolotl_binary(name):
    candidates = [
        f"/workspace/axolotl-venv/bin/{name}",
        shutil.which(name),
    ]

    for candidate in candidates:
        if (
            candidate
            and os.path.isfile(candidate)
        ):
            return candidate

    raise RuntimeError(
        f"Required Axolotl executable not found: {name}"
    )


def _check_training_runtime():
    python_bin = _axolotl_binary(
        "python"
    )

    result = run_command(
        [
            python_bin,
            "-c",
            (
                "import axolotl, torch; "
                "import liger_kernel; "
                "print('training-runtime-ok')"
            ),
        ]
    )

    return result.strip()


def _write_specialist_v2_config(
    source_adapter,
    dataset_info,
    work_dir,
    mode,
    gpu_profile,
):
    output_dir = os.path.join(
        work_dir,
        f"output_{mode}",
    )

    prepared_dir = os.path.join(
        work_dir,
        f"prepared_{mode}",
    )

    config_path = os.path.join(
        work_dir,
        f"specialist_v2_{mode}.yaml",
    )

    bf16 = bool(
        gpu_profile[
            "bf16_supported"
        ]
    )

    constrained = bool(
        gpu_profile[
            "constrained_memory_mode"
        ]
    )

    lines = [
        "base_model: Qwen/Qwen3-14B",
        "strict: false",
        "chat_template: qwen3",
        "",
    ]

    # When the original Git LFS adapters are gone, source_adapter is None and
    # Axolotl creates a brand-new QLoRA adapter on Qwen3-14B. Otherwise we keep
    # the normal continual-learning behavior and refine the recovered adapter.
    if source_adapter:
        lines.extend([
            f"lora_model_dir: {source_adapter}",
            "",
        ])

    lines.extend([
        "datasets:",
        f"  - path: {dataset_info['train_path']}",
        "    ds_type: json",
        "    split: train",
        "    type: chat_template",
        "    field_messages: messages",
        "    roles_to_train:",
        "      - assistant",
        "    train_on_eos: turn",
    ])

    if mode == "full":
        lines.extend([
            "",
            "test_datasets:",
            f"  - path: {dataset_info['validation_path']}",
            "    ds_type: json",
            "    split: train",
            "    type: chat_template",
            "    field_messages: messages",
            "    roles_to_train:",
            "      - assistant",
            "    train_on_eos: turn",
        ])

    lines.extend([
        "",
        "dataset_exact_deduplication: true",
        f"dataset_prepared_path: {prepared_dir}",
        f"output_dir: {output_dir}",
        "",
        f"sequence_len: {SPECIALIST_V2_SEQUENCE_LEN}",
        "excess_length_strategy: raise",
        "sample_packing: false",
        "eval_sample_packing: false",
        "",
        "load_in_4bit: true",
        "adapter: qlora",
        "lora_r: 16",
        "lora_alpha: 32",
        "lora_dropout: 0.0",
        "lora_target_modules:",
        "  - q_proj",
        "  - k_proj",
        "  - v_proj",
        "  - o_proj",
        "  - down_proj",
        "  - up_proj",
        "",
        "lora_qkv_kernel: true",
        "lora_o_kernel: true",
        "lora_mlp_kernel: true",
        "embeddings_skip_upcast: true",
        "",
        f"bf16: {'true' if bf16 else 'false'}",
        f"fp16: {'false' if bf16 else 'true'}",
        "tf32: true",
        "attn_implementation: flash_attention_2",
        "",
        "gradient_checkpointing: true",
        "gradient_checkpointing_kwargs:",
        "  use_reentrant: false",
    ])

    if constrained:
        lines.extend([
            "activation_offloading: hidden_states",
            "selective_checkpointing:",
            "  save:",
            "    - attention",
            "  offload: true",
        ])
    else:
        lines.append(
            "activation_offloading: false"
        )

    lines.extend([
        "",
        "plugins:",
        "  - axolotl.integrations.liger.LigerPlugin",
        "liger_fused_linear_cross_entropy: true",
        "liger_rope: false",
        "liger_rms_norm: false",
        "liger_glu_activation: false",
        "liger_layer_norm: false",
        "",
        "micro_batch_size: 1",
        "eval_batch_size: 1",
    ])

    if mode == "smoke":
        lines.extend([
            "gradient_accumulation_steps: 1",
            "max_steps: 1",
            'eval_strategy: "no"',
            'save_strategy: "no"',
        ])
    else:
        lines.extend([
            "gradient_accumulation_steps: 8",
            "num_epochs: 1",
        ])

    lines.extend([
        "",
        "optimizer: paged_adamw_8bit",
        "learning_rate: 0.00002",
        "lr_scheduler: cosine",
        "warmup_ratio: 0.05",
        "weight_decay: 0.0",
        "max_grad_norm: 0.1",
        "",
        "logging_steps: 1",
    ])

    if mode == "full":
        lines.extend([
            "evals_per_epoch: 1",
            "saves_per_epoch: 1",
            "save_total_limit: 2",
        ])

    lines.extend([
        "seed: 42",
        "",
        "dataloader_num_workers: 2",
        "dataloader_prefetch_factor: 4",
        "dataloader_pin_memory: true",
        "",
    ])

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )

    return {
        "config_path": config_path,
        "output_dir": output_dir,
        "prepared_dir": prepared_dir,
    }


def _release_inference_models_for_training():
    global _MATCHER_MODEL
    global _MATCHER_TOKENIZER
    global _ROUTER_MODEL
    global _ROUTER_TOKENIZER
    global _RICH_MODEL
    global _RICH_TOKENIZER
    global _RICH_DEVICE
    global _COUNCIL_ENGINE
    global _COUNCIL_ADAPTER_PATHS

    with INFERENCE_MODEL_LOCK:
        _COUNCIL_ENGINE = None
        _COUNCIL_ADAPTER_PATHS = None

        _MATCHER_MODEL = None
        _MATCHER_TOKENIZER = None

        _ROUTER_MODEL = None
        _ROUTER_TOKENIZER = None

        _RICH_MODEL = None
        _RICH_TOKENIZER = None
        _RICH_DEVICE = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        except Exception as exc:
            print(
                "CUDA cleanup warning: "
                f"{exc}",
                flush=True,
            )


def _find_adapter_output(output_dir):
    candidates = []

    if _real_adapter_checkpoint(
        output_dir
    ):
        candidates.append(
            output_dir
        )

    for path in glob.glob(
        os.path.join(
            output_dir,
            "checkpoint-*",
        )
    ):
        if _real_adapter_checkpoint(
            path
        ):
            candidates.append(path)

    if not candidates:
        raise RuntimeError(
            f"No real adapter_model.safetensors "
            f"was produced under {output_dir}."
        )

    def step_value(path):
        match = re.search(
            r"checkpoint-(\d+)$",
            path,
        )

        if match:
            return int(
                match.group(1)
            )

        return 10**12

    return max(
        candidates,
        key=step_value,
    )



def _check_github_publish_access(
    repo_dir,
    env,
):
    """Fail before expensive training if RunPod cannot publish to GitHub.

    The Specialist adapter is ~190 MB, so Git LFS must be available and
    *.safetensors must be tracked. A dry-run push also verifies that the
    configured GITHUB_TOKEN has write access to the target branch.
    """

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is missing. Configure a GitHub token with "
            "Contents: Read and write for the Athar training repository."
        )

    lfs_version = run_command(
        ["git", "lfs", "version"],
        cwd=repo_dir,
        env=env,
    ).strip()

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    # Ensure large adapter weights are always committed through Git LFS.
    attrs = run_command(
        [
            "git",
            "check-attr",
            "filter",
            "--",
            "checkpoints/specialist-v2/adapter_model.safetensors",
        ],
        cwd=repo_dir,
        env=env,
    )

    if "lfs" not in attrs.lower():
        run_command(
            ["git", "lfs", "track", "*.safetensors"],
            cwd=repo_dir,
            env=env,
        )

        run_command(
            ["git", "add", ".gitattributes"],
            cwd=repo_dir,
            env=env,
        )

        # Do not commit this here. The final publish commit will include it.

    # This authenticates as a writer without changing the repository.
    # It catches a read-only token / branch protection before GPU training.
    dry_run = run_command(
        [
            "git",
            "push",
            "--dry-run",
            "origin",
            f"HEAD:{GITHUB_BRANCH}",
        ],
        cwd=repo_dir,
        env=env,
        stream=True,
    )

    return {
        "status": "github_publish_ready",
        "repo": GITHUB_REPO,
        "branch": GITHUB_BRANCH,
        "lfs": lfs_version,
        "dry_run": dry_run[-1000:],
    }


def _push_specialist_v2_adapter(
    repo_dir,
    adapter_dir,
    env,
):
    # Re-check immediately before publishing. This is cheap and prevents a
    # successful training run from failing because of a read-only token.
    _check_github_publish_access(
        repo_dir,
        env,
    )

    target_rel = (
        SPECIALIST_V2_TARGET_REL
    )

    destination = os.path.join(
        repo_dir,
        target_rel,
    )

    shutil.rmtree(
        destination,
        ignore_errors=True,
    )

    os.makedirs(
        destination,
        exist_ok=True,
    )

    required = [
        "adapter_model.safetensors",
        "adapter_config.json",
    ]

    optional = [
        "README.md",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]

    for name in required:
        source = os.path.join(
            adapter_dir,
            name,
        )

        if not os.path.isfile(source):
            raise RuntimeError(
                f"Training output missing {name}: {source}"
            )

        shutil.copy2(
            source,
            os.path.join(
                destination,
                name,
            ),
        )

    for name in optional:
        source = os.path.join(
            adapter_dir,
            name,
        )

        if os.path.isfile(source):
            shutil.copy2(
                source,
                os.path.join(
                    destination,
                    name,
                ),
            )

    run_command(
        [
            "git",
            "config",
            "user.name",
            GIT_USER_NAME,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "config",
            "user.email",
            GIT_USER_EMAIL,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "add",
            target_rel,
        ],
        cwd=repo_dir,
        env=env,
    )

    status = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--quiet",
        ],
        cwd=repo_dir,
        env=env,
    )

    if status.returncode == 0:
        return "No Specialist V2 adapter changes detected"

    run_command(
        [
            "git",
            "commit",
            "-m",
            (
                "Update Specialist V2 adapter "
                "from RunPod Serverless"
            ),
        ],
        cwd=repo_dir,
        env=env,
    )

    # Feedback can arrive while a long training job is running. Rebase the
    # checkpoint commit onto the newest main branch before pushing.
    run_command(
        [
            "git",
            "pull",
            "--rebase",
            "origin",
            GITHUB_BRANCH,
        ],
        cwd=repo_dir,
        env=env,
        stream=True,
    )

    last_error = None

    for push_attempt in range(1, 4):
        try:
            run_command(
                [
                    "git",
                    "push",
                    "origin",
                    GITHUB_BRANCH,
                ],
                cwd=repo_dir,
                env=env,
                stream=True,
            )

            last_error = None
            break

        except Exception as exc:
            last_error = exc

            print(
                "GitHub publish attempt "
                f"{push_attempt}/3 failed: {exc}",
                flush=True,
            )

            if push_attempt < 3:
                time.sleep(5 * push_attempt)

                run_command(
                    [
                        "git",
                        "pull",
                        "--rebase",
                        "origin",
                        GITHUB_BRANCH,
                    ],
                    cwd=repo_dir,
                    env=env,
                    stream=True,
                )

    if last_error is not None:
        raise RuntimeError(
            "Training completed, but publishing Specialist V2 to GitHub "
            f"failed after 3 attempts: {last_error}"
        )

    return run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_dir,
        env=env,
    ).strip()


def train_specialist_v2(job_input):
    if not _SPECIALIST_V2_TRAINING_LOCK.acquire(
        blocking=False
    ):
        return {
            "status": "busy",
            "type": "train_specialist_v2",
            "message": (
                "A Specialist V2 training job is already "
                "running in this worker."
            ),
        }

    try:
        token = os.environ.get(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is required for "
                "train_specialist_v2."
            )

        mode = str(
            job_input.get("mode")
            or "smoke"
        ).strip().lower()

        if mode not in {
            "smoke",
            "full",
        }:
            raise ValueError(
                "mode must be 'smoke' or 'full'."
            )

        include_feedback_requested = bool(
            job_input.get(
                "include_feedback",
                True,
            )
        )

        continual_storage_ready = not (
            CONTINUAL_STORAGE_BACKEND == "s3"
            and not ATHAR_S3_BUCKET
        )

        include_feedback = (
            include_feedback_requested
            and continual_storage_ready
        )

        feedback_warning = None
        if (
            include_feedback_requested
            and not continual_storage_ready
        ):
            feedback_warning = (
                "S3 continual-learning storage is not configured yet, "
                "so this run will use only the existing Specialist "
                "training data. Configure ATHAR_S3_BUCKET before "
                "collecting/using production feedback."
            )

        gpu_profile = (
            _training_gpu_profile()
        )

        runtime_check = (
            _check_training_runtime()
        )

        (
            repo_dir,
            source_adapter,
            source_version,
            git_env,
        ) = _clone_specialist_learning_source(
            token
        )

        github_publish = None

        if mode == "full" or bool(
            job_input.get(
                "check_github_write",
                False,
            )
        ):
            github_publish = (
                _check_github_publish_access(
                    repo_dir,
                    git_env,
                )
            )

        work_dir = (
            "/tmp/athar_specialist_v2_work"
        )

        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )

        os.makedirs(
            work_dir,
            exist_ok=True,
        )

        dataset_info = (
            build_specialist_v2_datasets(
                repo_dir,
                work_dir,
                include_feedback=(
                    include_feedback
                ),
            )
        )

        config_info = (
            _write_specialist_v2_config(
                source_adapter,
                dataset_info,
                work_dir,
                mode,
                gpu_profile,
            )
        )

        preflight = {
            "status": (
                "specialist_v2_preflight_ok"
            ),
            "type": "train_specialist_v2",
            "mode": mode,
            "source_adapter": source_version,
            "source_adapter_path": source_adapter,
            "recovery_from_base": source_adapter is None,
            "target_adapter": (
                SPECIALIST_V2_TARGET_REL
            ),
            "sequence_len": (
                SPECIALIST_V2_SEQUENCE_LEN
            ),
            "gpu": gpu_profile,
            "training_runtime": runtime_check,
            "dataset": {
                key: value
                for key, value in dataset_info.items()
                if key not in {
                    "train_path",
                    "validation_path",
                }
            },
            "config_path": (
                config_info["config_path"]
            ),
            "continual_storage": (
                _continual_storage_description()
            ),
            "feedback_requested": (
                include_feedback_requested
            ),
            "feedback_included": (
                include_feedback
            ),
            "warning": feedback_warning,
            "github_publish": github_publish,
        }

        if job_input.get(
            "preflight",
            False,
        ):
            return preflight

        if (
            mode == "full"
            and bool(job_input.get("require_all_35", True))
            and not dataset_info.get("all35_ready", False)
        ):
            raise RuntimeError(
                "Full Specialist V2 training is blocked because the balanced "
                "all-35 dataset is not ready. Run type='build_specialist_all35_data' "
                "for advisors 11..35, then rerun the preflight."
            )

        # The parent worker may already hold Qwen3-14B for inference.
        # Free it before launching Axolotl in a child process.
        _release_inference_models_for_training()

        shutil.rmtree(
            config_info["output_dir"],
            ignore_errors=True,
        )

        shutil.rmtree(
            config_info["prepared_dir"],
            ignore_errors=True,
        )

        accelerate = _axolotl_binary(
            "accelerate"
        )

        cmd = [
            accelerate,
            "launch",
            "-m",
            "axolotl.cli.train",
            config_info["config_path"],
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

        print(
            "Starting Specialist V2 "
            f"{mode} training from "
            f"{source_version}...",
            flush=True,
        )

        started = time.time()

        log_tail = run_command(
            cmd,
            cwd=repo_dir,
            env=env,
            stream=True,
        )

        elapsed = round(
            time.time() - started,
            2,
        )

        adapter_dir = (
            _find_adapter_output(
                config_info["output_dir"]
            )
        )

        adapter_file = os.path.join(
            adapter_dir,
            "adapter_model.safetensors",
        )

        result = {
            "status": "completed",
            "type": "train_specialist_v2",
            "mode": mode,
            "source_adapter": source_version,
            "sequence_len": (
                SPECIALIST_V2_SEQUENCE_LEN
            ),
            "gpu": gpu_profile,
            "dataset": {
                key: value
                for key, value in dataset_info.items()
                if key not in {
                    "train_path",
                    "validation_path",
                }
            },
            "training_seconds": elapsed,
            "adapter_dir": adapter_dir,
            "adapter_size_mb": round(
                os.path.getsize(
                    adapter_file
                )
                / 1024
                / 1024,
                2,
            ),
            "log_tail": log_tail[-8000:],
        }

        if mode == "smoke":
            result[
                "activated"
            ] = False
            result[
                "next_step"
            ] = (
                "Smoke succeeded. Run again with "
                "mode='full' to train and publish "
                "checkpoints/specialist-v2."
            )

            return result

        commit_sha = (
            _push_specialist_v2_adapter(
                repo_dir,
                adapter_dir,
                git_env,
            )
        )

        global _COUNCIL_ADAPTER_PATHS
        global _COUNCIL_ENGINE

        _COUNCIL_ADAPTER_PATHS = None
        _COUNCIL_ENGINE = None

        result.update({
            "activated": True,
            "saved_to": (
                SPECIALIST_V2_TARGET_REL
            ),
            "github_commit": commit_sha,
            "next_step": (
                "New consultations will prefer "
                "Specialist V2. Run the same "
                "AOS-LD-04 + AOS-SP-12 regression "
                "case to compare quality."
            ),
        })

        return result

    finally:
        _SPECIALIST_V2_TRAINING_LOCK.release()




def _write_meta_recovery_config(repo_dir, work_dir, mode, gpu_profile):
    """Create a fresh Meta QLoRA config from the recovered gold dataset."""
    output_dir = os.path.join(work_dir, f"output_{mode}")
    prepared_dir = os.path.join(work_dir, f"prepared_{mode}")
    config_path = os.path.join(work_dir, f"meta_recovery_{mode}.yaml")
    bf16 = bool(gpu_profile["bf16_supported"])
    constrained = bool(gpu_profile["constrained_memory_mode"])

    train_path = os.path.join(repo_dir, "data", "train_meta_balanced.jsonl")
    validation_path = os.path.join(repo_dir, "data", "validation_meta.jsonl")
    if not os.path.isfile(train_path):
        raise RuntimeError(f"Recovered Meta train data not found: {train_path}")
    if mode == "full" and not os.path.isfile(validation_path):
        raise RuntimeError(f"Recovered Meta validation data not found: {validation_path}")

    lines = [
        "base_model: Qwen/Qwen3-14B",
        "strict: false",
        "chat_template: qwen3",
        "",
        "datasets:",
        f"  - path: {train_path}",
        "    ds_type: json",
        "    split: train",
        "    type: chat_template",
        "    field_messages: messages",
        "    roles_to_train:",
        "      - assistant",
        "    train_on_eos: turn",
    ]
    if mode == "full":
        lines.extend([
            "",
            "test_datasets:",
            f"  - path: {validation_path}",
            "    ds_type: json",
            "    split: train",
            "    type: chat_template",
            "    field_messages: messages",
            "    roles_to_train:",
            "      - assistant",
            "    train_on_eos: turn",
        ])

    lines.extend([
        "",
        "dataset_exact_deduplication: true",
        f"dataset_prepared_path: {prepared_dir}",
        f"output_dir: {output_dir}",
        "sequence_len: 2048",
        "sample_packing: false",
        "eval_sample_packing: false",
        "load_in_4bit: true",
        "adapter: qlora",
        "lora_r: 16",
        "lora_alpha: 32",
        "lora_dropout: 0.0",
        "lora_target_modules:",
        "  - q_proj",
        "  - k_proj",
        "  - v_proj",
        "  - o_proj",
        "  - down_proj",
        "  - up_proj",
        "lora_qkv_kernel: true",
        "lora_o_kernel: true",
        "lora_mlp_kernel: true",
        "embeddings_skip_upcast: true",
        f"bf16: {'true' if bf16 else 'false'}",
        f"fp16: {'false' if bf16 else 'true'}",
        "tf32: true",
        "attn_implementation: flash_attention_2",
        "gradient_checkpointing: true",
        "gradient_checkpointing_kwargs:",
        "  use_reentrant: false",
    ])
    if constrained:
        lines.extend([
            "activation_offloading: hidden_states",
            "selective_checkpointing:",
            "  save:",
            "    - attention",
            "  offload: true",
        ])
    else:
        lines.append("activation_offloading: false")

    lines.extend([
        "plugins:",
        "  - axolotl.integrations.liger.LigerPlugin",
        "liger_fused_linear_cross_entropy: true",
        "liger_rope: false",
        "liger_rms_norm: false",
        "liger_glu_activation: false",
        "liger_layer_norm: false",
        "micro_batch_size: 1",
        "eval_batch_size: 1",
    ])
    if mode == "smoke":
        lines.extend([
            "gradient_accumulation_steps: 1",
            "max_steps: 1",
            'eval_strategy: "no"',
            'save_strategy: "no"',
        ])
    else:
        lines.extend([
            "gradient_accumulation_steps: 8",
            "num_epochs: 4",
        ])
    lines.extend([
        "optimizer: paged_adamw_8bit",
        "learning_rate: 0.0001",
        "lr_scheduler: cosine",
        "warmup_ratio: 0.1",
        "weight_decay: 0.0",
        "max_grad_norm: 0.1",
        "logging_steps: 1",
    ])
    if mode == "full":
        lines.extend([
            "evals_per_epoch: 2",
            "saves_per_epoch: 1",
            "save_total_limit: 2",
        ])
    lines.extend([
        "seed: 42",
        "dataloader_num_workers: 2",
        "dataloader_prefetch_factor: 4",
        "dataloader_pin_memory: true",
        "",
    ])

    with open(config_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return {
        "config_path": config_path,
        "output_dir": output_dir,
        "prepared_dir": prepared_dir,
        "train_path": train_path,
        "validation_path": validation_path,
    }


def _push_recovered_adapter(repo_dir, adapter_dir, target_rel, label, env):
    """Publish a recovered LoRA through Git LFS to the new repository."""
    _check_github_publish_access(repo_dir, env)
    destination = os.path.join(repo_dir, target_rel)
    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(destination, exist_ok=True)

    for name in ("adapter_model.safetensors", "adapter_config.json"):
        source = os.path.join(adapter_dir, name)
        if not os.path.isfile(source):
            raise RuntimeError(f"Recovery output missing {name}: {source}")
        shutil.copy2(source, os.path.join(destination, name))

    for name in (
        "README.md", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja",
    ):
        source = os.path.join(adapter_dir, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(destination, name))

    run_command(["git", "lfs", "track", "*.safetensors"], cwd=repo_dir, env=env)
    run_command(["git", "config", "user.name", GIT_USER_NAME], cwd=repo_dir, env=env)
    run_command(["git", "config", "user.email", GIT_USER_EMAIL], cwd=repo_dir, env=env)
    run_command(["git", "add", ".gitattributes", target_rel], cwd=repo_dir, env=env)

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_dir, env=env
    )
    if status.returncode == 0:
        return "No recovered adapter changes detected"

    run_command(
        ["git", "commit", "-m", f"Recover {label} adapter on RunPod Serverless"],
        cwd=repo_dir, env=env,
    )
    run_command(
        ["git", "pull", "--rebase", "origin", GITHUB_BRANCH],
        cwd=repo_dir, env=env, stream=True,
    )
    last_error = None
    for attempt in range(1, 4):
        try:
            run_command(
                ["git", "push", "origin", GITHUB_BRANCH],
                cwd=repo_dir, env=env, stream=True,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * attempt)
                run_command(
                    ["git", "pull", "--rebase", "origin", GITHUB_BRANCH],
                    cwd=repo_dir, env=env, stream=True,
                )
    if last_error is not None:
        raise RuntimeError(f"{label} recovery trained but GitHub publish failed: {last_error}")
    return run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir, env=env).strip()


def recover_meta_adapter(job_input):
    """Rebuild the deleted Meta LoRA from the recovered Meta training dataset."""
    if not _SPECIALIST_V2_TRAINING_LOCK.acquire(blocking=False):
        return {
            "status": "busy",
            "type": "recover_meta_adapter",
            "message": "Another GPU training job is already running in this worker.",
        }
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is required for recover_meta_adapter.")
        mode = str(job_input.get("mode") or "smoke").strip().lower()
        if mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'.")

        gpu_profile = _training_gpu_profile()
        runtime_check = _check_training_runtime()
        repo_dir, git_env = clone_repo_without_lfs(
            token, repo_dir="/tmp/athar_meta_recovery_repo"
        )
        run_command(["git", "lfs", "install", "--local"], cwd=repo_dir, env=git_env)
        github_publish = None
        if mode == "full" or bool(job_input.get("check_github_write", False)):
            github_publish = _check_github_publish_access(repo_dir, git_env)

        work_dir = "/tmp/athar_meta_recovery_work"
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        config_info = _write_meta_recovery_config(
            repo_dir, work_dir, mode, gpu_profile
        )

        preflight = {
            "status": "meta_recovery_preflight_ok",
            "type": "recover_meta_adapter",
            "mode": mode,
            "source": "Qwen/Qwen3-14B",
            "target_adapter": "checkpoints/meta",
            "gpu": gpu_profile,
            "training_runtime": runtime_check,
            "config_path": config_info["config_path"],
            "github_publish": github_publish,
        }
        if job_input.get("preflight", False):
            return preflight

        _release_inference_models_for_training()
        shutil.rmtree(config_info["output_dir"], ignore_errors=True)
        shutil.rmtree(config_info["prepared_dir"], ignore_errors=True)
        accelerate = _axolotl_binary("accelerate")
        cmd = [accelerate, "launch", "-m", "axolotl.cli.train", config_info["config_path"]]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        started = time.time()
        log_tail = run_command(cmd, cwd=repo_dir, env=env, stream=True)
        elapsed = round(time.time() - started, 2)
        adapter_dir = _find_adapter_output(config_info["output_dir"])
        adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
        result = {
            "status": "completed",
            "type": "recover_meta_adapter",
            "mode": mode,
            "source": "Qwen/Qwen3-14B",
            "training_seconds": elapsed,
            "adapter_dir": adapter_dir,
            "adapter_size_mb": round(os.path.getsize(adapter_file) / 1024 / 1024, 2),
            "log_tail": log_tail[-8000:],
        }
        if mode == "smoke":
            result["activated"] = False
            result["next_step"] = "Smoke succeeded. Run mode='full' to publish checkpoints/meta."
            return result

        commit_sha = _push_recovered_adapter(
            repo_dir, adapter_dir, "checkpoints/meta", "Meta", git_env
        )
        global _COUNCIL_ADAPTER_PATHS
        global _COUNCIL_ENGINE
        _COUNCIL_ADAPTER_PATHS = None
        _COUNCIL_ENGINE = None
        result.update({
            "activated": True,
            "saved_to": "checkpoints/meta",
            "github_commit": commit_sha,
            "next_step": "Meta recovered. Recover/publish Specialist V2, then rerun advisory_consultation.",
        })
        return result
    finally:
        _SPECIALIST_V2_TRAINING_LOCK.release()


def continual_status(job_input):
    ratings = Counter()
    eligible = 0

    if CONTINUAL_STORAGE_BACKEND == "s3":
        run_paths = [
            path
            for path in _continual_list_paths(
                f"{CONTINUAL_ROOT_REL}/runs/"
            )
            if path.endswith(".json")
        ]

        feedback_records = (
            _load_continual_feedback_records()
        )

        for _, row in feedback_records:
            ratings[
                str(row.get("rating"))
            ] += 1

            if row.get(
                "training_eligible",
                False,
            ):
                eligible += 1

        feedback_count = len(
            feedback_records
        )

    elif CONTINUAL_STORAGE_BACKEND == "github":
        token = os.environ.get(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is required for continual_status "
                "when CONTINUAL_STORAGE_BACKEND=github."
            )

        repo_dir, _ = clone_repo_without_lfs(
            token,
            repo_dir="/tmp/athar_continual_status",
        )

        feedback_records = (
            _load_continual_feedback_records(
                repo_dir=repo_dir
            )
        )

        run_paths = glob.glob(
            os.path.join(
                repo_dir,
                CONTINUAL_ROOT_REL,
                "runs",
                "*.json",
            )
        )

        for _, row in feedback_records:
            ratings[
                str(row.get("rating"))
            ] += 1

            if row.get(
                "training_eligible",
                False,
            ):
                eligible += 1

        feedback_count = len(
            feedback_records
        )

    else:
        raise RuntimeError(
            "CONTINUAL_STORAGE_BACKEND must be 's3' or 'github'."
        )

    # Checking GitHub for the published V2 adapter is intentionally separate
    # from the S3 feedback store.
    v2_exists = None
    token = os.environ.get(
        "GITHUB_TOKEN"
    )

    if token:
        try:
            repo_dir, _ = clone_repo_without_lfs(
                token,
                repo_dir="/tmp/athar_continual_v2_status",
            )

            v2_exists = os.path.isfile(
                os.path.join(
                    repo_dir,
                    SPECIALIST_V2_TARGET_REL,
                    "adapter_config.json",
                )
            )
        except Exception:
            v2_exists = None

    return {
        "status": "ok",
        "type": "continual_status",
        "continual_learning_enabled": (
            CONTINUAL_LEARNING_ENABLED
        ),
        "storage": (
            _continual_storage_description()
        ),
        "logged_runs": len(
            run_paths
        ),
        "feedback_records": (
            feedback_count
        ),
        "ratings": dict(
            ratings
        ),
        "training_eligible_feedback": eligible,
        "specialist_v2_exists_in_repo": (
            v2_exists
        ),
    }


def handler(job):
    job_input = job.get(
        "input",
        {},
    )

    if not isinstance(
        job_input,
        dict,
    ):
        return {
            "error": (
                "RunPod input must be a JSON object."
            )
        }

    request_type = job_input.get(
        "type"
    )

    # ---------------------------------------------------------------
    # Production advisor discovery / matching
    # ---------------------------------------------------------------
    if request_type == "advisory_match":
        if job_input.get(
            "preflight",
            False,
        ):
            registry = load_rich_registry()

            return {
                "status": "advisory_match_preflight_ok",
                "routing_engine": "rich_ai_v34_final_one_pass",
                "model": MATCHER_BASE_MODEL,
                "registry_path": RICH_REGISTRY_PATH,
                "advisor_count": len(
                    registry.get(
                        "advisors",
                        [],
                    )
                ),
                "advisor_id_format": "AOS-(LD|SP|FG|SE)-NN",
                "minimum_advisors": RICH_V34_MIN_ADVISORS,
                "maximum_advisors": RICH_V34_MAX_ADVISORS,
            }

        return advisory_match_rich_v34(
            job_input
        )

    # ---------------------------------------------------------------
    # Production advisory council:
    # selected advisors -> independent Full DNA opinions only (current stage)
    # ---------------------------------------------------------------
    if request_type == "advisory_consultation":
        if job_input.get(
            "preflight",
            False,
        ):
            registry_path = os.path.join(
                ROOT,
                "advisors",
                "advisors_registry_35.json",
            )

            with open(
                registry_path,
                "r",
                encoding="utf-8",
            ) as file:
                registry = json.load(file)

            return {
                "status": "advisory_consultation_preflight_ok",
                "model": MATCHER_BASE_MODEL,
                "advisor_count": len(
                    registry.get(
                        "advisors",
                        [],
                    )
                ),
                "shares_base_model_with_advisory_match": True,
                "adapter_preference": [
                    "checkpoints/specialist-v2",
                    "checkpoints/specialist",
                ],
                "meta_mode": os.environ.get(
                    "COUNCIL_META_MODE",
                    "adapter",
                ),
                "continual_learning_enabled": (
                    CONTINUAL_LEARNING_ENABLED
                ),
                "response_contract_version": "impact_challenge_screen_3_specialist_opinions_v1_aos_ids",
                "build_version": "athar-screen3-specialist-opinions-v1-2026-09-26",
                "topic": "interventions",
                "backend_advisor_id_type": "canonical_aos_string",
                "canonical_advisor_id_format": "AOS-(LD|SP|FG|SE)-NN",
                "screen3_public_response": {
                    "involved_advisor_ids": "list<string canonical AOS-*>",
                    "suggestion.interventions": [
                        "title",
                        "confidence_level",
                        "impact_description",
                        "reportable_value",
                        "results[].text",
                        "results[].outputs[].text",
                    ],
                },
                "single_output_regeneration_supported": True,
                "grounded_numbers_only": True,
                "consultation_stage": "specialist_opinions_only",
                "meta_synthesis_enabled": False,
                "one_intervention_per_selected_advisor": True,
                "max_selected_advisors_for_current_screen3_contract": 4,
                "intervention_order_matches_involved_advisor_ids": True,
                "deferred_meta_stage": {
                    "meta_synthesis": True,
                    "attribution": True,
                    "evidence_classification": ["E1", "E2", "E3", "I1", "I2", "A1", "U"],
                    "interaction_types": [
                        "CONSENSUS",
                        "COMPLEMENTARY",
                        "TRADE-OFF",
                        "CONFLICT",
                        "EVIDENCE GAP",
                        "SCOPE CONFLICT",
                    ],
                    "confidence_levels": ["High", "Medium", "Low"],
                    "item_status": "OPEN",
                    "sprint_count": 12,
                    "sprint_duration": "1 week",
                    "publicly_exposed": False,
                    "note": "Not executed in the current stage. These fields are deferred until their backend contracts arrive.",
                },
            }

        try:
            response = advisory_consultation_inference(
                job_input
            )
        except Exception as exc:
            # Screen-3 integration contract expects a structured failure payload
            # rather than a non-JSON traceback as the application response.
            failed = {
                "status": "failed",
                "error": str(exc),
            }
            if bool(job_input.get("debug") or (job_input.get("input") or {}).get("debug")):
                failed["error_type"] = type(exc).__name__
            return failed

        internal_context = None
        if isinstance(response, dict):
            internal_context = response.pop("_athar_internal", None)

        # Single-output regeneration is a UI rewrite action, not a new council
        # training example. Do not feed it into Specialist continual learning.
        is_regeneration = bool(
            (job_input.get("input") or {}).get("is_output_regeneration")
        )
        if not is_regeneration:
            _log_consultation_for_learning(
                job_input,
                response,
                internal_context=internal_context,
            )

        return response

    # ---------------------------------------------------------------
    # Human feedback for continual learning.
    # accepted  -> positive SFT example if quality gates pass
    # corrected -> corrected positive SFT example
    # rejected  -> stored only, never positive SFT
    # ---------------------------------------------------------------
    if request_type == "advisory_feedback":
        return advisory_feedback(
            job_input
        )

    if request_type == "recover_meta_adapter":
        return recover_meta_adapter(
            job_input
        )

    if request_type == "continual_status":
        return continual_status(
            job_input
        )

    # ---------------------------------------------------------------
    # Build balanced Full-DNA bootstrap data for advisors 11..35.
    # The job is resumable because each completed advisor is committed to GitHub.
    # ---------------------------------------------------------------
    if request_type == "build_specialist_all35_data":
        return generate_specialist_all35_data(
            job_input
        )

    # ---------------------------------------------------------------
    # Specialist V2 training on the SAME Serverless endpoint.
    #
    # Start with:
    #   {"type":"train_specialist_v2","mode":"smoke"}
    #
    # Then:
    #   {"type":"train_specialist_v2","mode":"full"}
    # ---------------------------------------------------------------
    if request_type == "train_specialist_v2":
        return train_specialist_v2(
            job_input
        )

    # ---------------------------------------------------------------
    # Legacy training routes retained for compatibility.
    # ---------------------------------------------------------------
    token = os.environ.get(
        "GITHUB_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is missing."
        )

    training_type = job_input.get(
        "training_type"
    )

    if training_type not in RUNS:
        return {
            "error": (
                "Supported type values are "
                "'advisory_match', "
                "'advisory_consultation', "
                "'advisory_feedback', "
                "'continual_status', "
                "'build_specialist_all35_data', and "
                "'train_specialist_v2'. "
                "Legacy training_type may be "
                "base, meta, specialist, or matcher."
            )
        }

    info = RUNS[
        training_type
    ]

    print(
        f"Preparing legacy {training_type} training...",
        flush=True,
    )

    (
        repo_dir,
        checkpoint_path,
        git_env,
    ) = clone_source_checkpoint(
        training_type,
        token,
    )

    print(
        f"Source checkpoint: {checkpoint_path}",
        flush=True,
    )

    if job_input.get(
        "preflight",
        False,
    ):
        return training_preflight(
            training_type,
            checkpoint_path,
        )

    _release_inference_models_for_training()

    shutil.rmtree(
        info["output_dir"],
        ignore_errors=True,
    )

    os.makedirs(
        info["output_dir"],
        exist_ok=True,
    )

    cmd = [
        _axolotl_binary(
            "accelerate"
        ),
        "launch",
        "-m",
        "axolotl.cli.train",
        info["config"],
    ]

    if info["resume"]:
        cmd.extend([
            "--resume-from-checkpoint",
            checkpoint_path,
        ])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    print(
        f"Starting legacy {training_type} training...",
        flush=True,
    )

    log_tail = run_command(
        cmd,
        cwd=ROOT,
        env=env,
        stream=True,
    )

    new_checkpoint = latest_checkpoint(
        info["output_dir"]
    )

    print(
        f"Latest checkpoint: {new_checkpoint}",
        flush=True,
    )

    commit_sha = push_checkpoint(
        training_type,
        repo_dir,
        new_checkpoint,
        git_env,
    )

    return {
        "status": "completed",
        "training_type": training_type,
        "checkpoint": os.path.basename(
            new_checkpoint
        ),
        "saved_to": info[
            "target_checkpoint_rel"
        ],
        "github_commit": commit_sha,
        "log_tail": log_tail,
    }


runpod.serverless.start({
    "handler": handler
})
