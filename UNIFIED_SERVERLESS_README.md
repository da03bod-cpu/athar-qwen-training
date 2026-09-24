# Athar OS — Unified RunPod Serverless

نفس الـEndpoint يدعم الآن مسارين Production مع الإبقاء على Training routes الحالية.

## 1) اختيار المستشارين — بدون تغيير في المنطق الحالي

```json
{
  "input": {
    "type": "advisory_match",
    "input": {
      "organization": {},
      "programs": []
    }
  }
}
```

النتيجة تظل من `advisory_match_rich_v34` كما كانت.

## 2) تشغيل المجلس الاستشاري بعد اختيار المستشارين

```json
{
  "input": {
    "type": "advisory_consultation",
    "run_id": "RUN-001",
    "consultation_id": 42,
    "topic": "interventions",
    "input": {
      "selected_advisor_ids": [
        "AOS-SP-08",
        "AOS-SP-13"
      ],
      "organization": {},
      "programs": [],
      "advisors": [
        {
          "id": 101,
          "model_advisor_id": "AOS-SP-08",
          "name": "مستشار التخطيط الاستراتيجي"
        },
        {
          "id": 106,
          "model_advisor_id": "AOS-SP-13",
          "name": "مستشار المحافظ والبرامج والمشاريع"
        }
      ],
      "track": {},
      "goal": {},
      "impact_map": {}
    }
  }
}
```

التنفيذ:

1. يستخدم نفس Qwen3-14B الموجود في `advisory_match`، ولا يحمل نسخة ثانية من الموديل.
2. ينزّل `checkpoints/specialist` و`checkpoints/meta` من Git LFS عند أول طلب Council فقط.
3. كل مستشار مختار يحصل على ملفه الأصلي `prompts/advisors/advisor_XX.md` بشكل مستقل.
4. لا يرى أي مستشار مخرجات مستشار آخر.
5. AOS-META-00 يستقبل Context الحالة + الآراء المستقلة ويخرج النتيجة النهائية فقط.
6. `involved_advisor_ids` ترجع Backend integer IDs الموجودة في `input.advisors[].id`؛ لا يتم اختراع IDs.

## Environment variable مهم

لأن الريبو Private والـLoRA داخل Git LFS، يجب أن يظل في Endpoint:

```text
GITHUB_TOKEN=<token with read access to da02bod-art/athar-qwen-training>
```

`advisory_match` لا يحتاج التوكن. `advisory_consultation` يحتاجه في أول مرة لكل Worker لتحميل الـSpecialist والـMeta adapters.

## Optional

```text
COUNCIL_META_MODE=adapter
```

القيمة الافتراضية `adapter` تستخدم Meta LoRA. يمكن ضبطها إلى `base` للاختبار إذا أردنا مقارنة جودة الـsynthesis لأن الـMeta adapter الحالي تم تدريبه أساسًا على routing/selection data.

## Preflight

Selection:

```json
{"input":{"type":"advisory_match","preflight":true}}
```

Council:

```json
{"input":{"type":"advisory_consultation","preflight":true}}
```

Council preflight لا يحمل الموديل ولا ينزل الـLoRA؛ فقط يتحقق من Registry/route metadata.

## مهم بخصوص DOCX

هذا الريبو نفسه لا يحتوي Document Extraction pipeline لملف DOCX الخام. `advisory_match` هنا يستقبل `organization` و`programs` كـJSON بعد مرحلة الاستخراج. إذا كان الـDOCX عندك يدخل Endpoint آخر للاستخراج حاليًا، يظل هذا الجزء Upstream كما هو، ثم يُرسل الناتج إلى `advisory_match`.
