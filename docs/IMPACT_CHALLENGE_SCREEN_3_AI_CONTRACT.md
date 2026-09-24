# تحدي الأثر 10X — Screen 3 (خريطة الأثر / Impact Map)
## AI Integration Contract & Schema Specification

This document defines the interface between the **Backend (`Advisory` / `ImpactChallenge` domains)** and the **AI Team (LLM Provider / n8n / RunPod)** for generating intervention proposals on Screen 3 (**خريطة الأثر — Impact Map**).

---

## 1. System Overview & AI Run Lifecycle

When an organization clicks **"طلب اقتراحات الذكاء الاصطناعي"** (Request AI Analysis) on Screen 3, the backend opens an asynchronous `AdvisoryConsultation` and initiates an `AiRun`.

### Lifecycle Diagram:
```
Frontend (Screen 3)
   │
   │ POST /api/v1/impact-challenge/impact-map/analysis
   ▼
Backend (ImpactMapService)
   │
   │ 1. Saves impact_drivers & association_scope
   │ 2. AdvisoryCouncilInterface::consult()
   ▼
AiRun Dispatch (Async Job)
   │
   │ POST Webhook / RunPod / n8n Worker
   ▼
AI Model / Workflow
   │
   │ Process context (Track, Goal, Problem, Drivers, Scope)
   │ Generates structured JSON adhering to this contract
   ▼
AiRun Completion Callback
   │
   │ Backend MaterializeInterventionsFromConsultation listener:
   │ Inserts Interventions → Results → Outputs into the database
   ▼
Frontend polls GET /api/v1/impact-challenge/impact-map
   └── Receives populated interventions with results and outputs
```

---

## 2. Generic Envelope & Metadata

Every AI run dispatched by the system includes standard metadata:

| Key | Type | Description |
|---|---|---|
| `run_id` | `string` (UUID) | Unique identifier of the execution run (e.g. `9f1c2b3a-4e5d-6c7b-8a90-123456789abc`) |
| `type` | `string` | Always `"advisory_consultation"` for council advisory requests |
| `consultation_id` | `integer` | Database ID of the parent `advisory_consultations` record |
| `topic` | `string` | Always `"interventions"` for Screen 3 |

---

## 3. Input Specification (What the AI Receives)

The AI receives the organization's complete institutional profile, all existing programs/projects, council advisors, and the full challenge context accumulated from Screens 1, 2, and 3 inside the standard **`input`** object:

```json
{
  "run_id": "9f1c2b3a-4e5d-6c7b-8a90-123456789abc",
  "type": "advisory_consultation",
  "consultation_id": 42,
  "topic": "interventions",
  "input": {
    "organization": {
      "name": "جمعية تمكين الأجيال الأهلية",
      "type": "جمعية أهلية",
      "sector": "غير ربحي",
      "activity_fields": ["تعليم", "تمكين اجتماعي", "تنمية مجتمعية"],
      "short_description": "جمعية متخصصة في دعم التعليم الأساسي ورعاية الطلاب في المناطق الطرفية والقرى النائية.",
      "detailed_description": "نعمل منذ 7 سنوات على تقديم برامج دعم مدرسي، توفير حقائب مدرسية، وتسيير خطوط نقل آمنة للطلبة المعوزين، بهدف خفض التسرب الدراسي ورفع جودة التحصيل.",
      "competitive_advantage": "شراكات استراتيجية مع 15 مدرسة حكومية في القرى، وشبكة علاقات موثوقة مع أعيان المجتمع المحلي وممثلي وزارة التعليم.",
      "important_notes": "الاعتماد المالي الحالي بنسبة 60% على تبرعات مواسم، وتسعى الجمعية لشراكات مؤسسية مستدامة مع مانحين رئيسيين ومؤسسات مانحة كبرى."
    },
    "programs": [
      {
        "name": "مشروع النقل المدرسي التنموي",
        "type": "project",
        "description": "تأمين حافلات مستأجرة لنقل 120 طالب وطالبة من القرى إلى مدارس المراكز.",
        "target_audience": "طلبة المرحلة الابتدائية والمتوسطة في القرى غير المخدومة.",
        "beneficiary_value": "حماية الأطفال من مخاطر الطرق وخفض الغياب المدرسي.",
        "delivery_method": "حافلات يومية عبر خطين رئيسيين."
      },
      {
        "name": "برنامج الحقيبة والزي المدرسي",
        "type": "program",
        "description": "توزيع مستلزمات دراسية متكاملة مع بداية كل عام دراسي للأسر الضمانية.",
        "target_audience": "أبناء الأسر المسجلة في قوائم الدعم الاجتماعي.",
        "beneficiary_value": "تخفيف العبء المالي عن أولياء الأمور.",
        "delivery_method": "توزيع ميداني بالتعاون مع لجان الأحياء."
      }
    ],
    "advisors": [
      {
        "id": 1,
        "name": "مستشار الأثر الاجتماعي",
        "title": "قياس وتصميم التدخلات الاجتماعية",
        "capabilities": ["impact_design", "beneficiary_reach", "social_outcomes"]
      },
      {
        "id": 3,
        "name": "مستشار الكفاءة التشغيلية",
        "title": "إدارة العمليات وسلاسل الإمداد",
        "capabilities": ["operations_management", "cycle_time_reduction", "resource_allocation"]
      }
    ],
    "track": {
      "id": 1,
      "name": "الأثر الاجتماعي",
      "code": "social_impact",
      "primary_indicator": "Impact Depth"
    },
    "goal": {
      "statement": "زيادة معدل إتمام التعليم الأساسي للأطفال في المناطق النائية بنسبة 20% خلال عامين.",
      "bsc_axis": "العملاء والمستفيدين",
      "clarity_rating": "مكتمل",
      "social_problem": "ارتفاع معدلات التسرب من التعليم الأساسي في القرى النائية بسبب صعوبة المواصلات ونقص المدارس.",
      "target_group": "الأطفال في سن التعليم الأساسي (6-15 سنة) في القرى غير المخدومة.",
      "vision_alignment": "مستهدفات رؤية 2030 لرفع جودة التعليم ومكافحة التسرب."
    },
    "impact_map": {
      "social_problem": "ارتفاع معدلات التسرب من التعليم الأساسي في القرى النائية بسبب صعوبة المواصلات ونقص المدارس.",
      "impact_drivers": "النقل المدرسي الآمن، توفير بيئة تعليمية محفزة، دعم الأسر المتعففة لمصاريف الدراسة.",
      "association_scope": "المناطق الريفية والنائية في منطقة المدينة المنورة وجازان."
    }
  }
}
```

### Input Field Definitions:

#### A. Institutional & Organizational Context (`input.organization`)
| Field | Type | Description |
|---|---|---|
| `name` | `string` | الاسم الرسمي للجمعية الأهلية أو المؤسسة غير الربحية |
| `type` | `string` | نوع المنظمة (`"جمعية أهلية"`, `"مؤسسة أهلية"`, etc.) |
| `sector` | `string` | القطاع (`"غير ربحي"`, etc.) |
| `activity_fields` | `list<string>` | مجالات النشاط الرئيسية (e.g. تعليم، رعاية اجتماعية، صحة، تمكين) |
| `short_description` | `string\|null` | نبذة تعريفية مختصرة عن المنظمة |
| `detailed_description` | `string\|null` | الوصف التفصيلي لعمل الجمعية وأنشطتها ومجتمعاتها |
| `competitive_advantage` | `string\|null` | الميزة التنافسية للجمعية والشراكات الميدانية |
| `important_notes` | `string\|null` | ملاحظات هامة حول الوضع المالي أو التشغيلي أو أهداف المرحلة |

#### B. Existing Programs & Projects (`input.programs`)
| Field | Type | Description |
|---|---|---|
| `name` | `string` | اسم البرنامج أو المشروع القائم في الجمعية |
| `type` | `string` | نوع الكيان: `"program"` (برنامج مستمر) أو `"project"` (مشروع محدد بزمن) |
| `description` | `string\|null` | وصف تفصيلي للبرنامج أو المشروع |
| `target_audience` | `string\|null` | الفئة المستهدفة من البرنامج |
| `beneficiary_value` | `string\|null` | القيمة المضافة أو الأثر المباشر المقدم للمستفيد |
| `delivery_method` | `string\|null` | آلية تقديم الخدمة (حضوري، ميداني، منصة إلكترونية، إلخ) |

#### C. Advisory Council Context (`input.advisors`)
| Field | Type | Description |
|---|---|---|
| `id` | `integer` | معرّف المستشار في المجلس الذكي |
| `name` | `string` | اسم المستشار (e.g. مستشار قياس الأثر) |
| `title` | `string` | الاختصاص الاستشاري |
| `capabilities` | `list<string>` | قدرات المستشار التخصصية |

#### D. Selected Track & Strategic Goal (`input.track` & `input.goal`)
| Field | Type | Description |
|---|---|---|
| `track.name` | `string` | اسم المسار المختار من المسارات الخمسة (e.g. الأثر الاجتماعي) |
| `track.primary_indicator` | `string` | المؤشر الأساسي المعتمد للمسار (e.g. Impact Depth, Self-Funding Ratio) |
| `goal.statement` | `string` | نص الهدف الاستراتيجي المعتمد لدورة التحدي (ثابت طوال الرحلة) |
| `goal.bsc_axis` | `string\|null` | المحور الاستراتيجي لبطاقة الأداء المتوازن (BSC) |
| `goal.clarity_rating` | `string\|null` | درجة نضج / وضوح الهدف |
| `goal.social_problem` | `string\|null` | المشكلة الاجتماعية المعالجة بالهدف |
| `goal.target_group` | `string\|null` | الفئة المستهدفة بالهدف |
| `goal.vision_alignment` | `string\|null` | ارتباط الهدف برؤية المملكة 2030 |

#### E. Screen 3 Inputs (`input.impact_map`)
| Field | Type | Description |
|---|---|---|
| `social_problem` | `string` | نص المشكلة الاجتماعية المنسوخة آلياً من الهدف المعتمد |
| `impact_drivers` | `string` | محركات الأثر (مُدخلة يدوياً من الجمعية في الشاشة الثالثة) |
| `association_scope` | `string` | نطاق عمل الجمعية أو الارتباط (مُدخل يدوياً من الجمعية) |

---

## 4. Output Specification (What the AI Must Return)

The AI service must respond with HTTP `200` returning the standard `AdvisoryConsultation` result envelope.

### Outer Response Envelope:

```json
{
  "involved_advisor_ids": [1, 3],
  "suggestion": {
    "interventions": [ ... ]
  }
}
```

* `involved_advisor_ids` (`list<int>`): The IDs of council advisors who contributed to this proposal (can be `[]` if using unified model).
* `suggestion.interventions` (`list<object>`): Array of recommended interventions (typically 2 to 4 proposals).

---

### Detailed Structure of `suggestion.interventions`:

In accordance with the PRD:
1. **Each intervention** has a distinct **`title`**, an **`impact_description`**, a **`confidence_level`**, and a **`reportable_value`**.
2. **Every intervention contains multiple `results`** (النتائج المتوقعة).
3. **Every result contains its own `outputs`** (المخرجات المباشرة الخاصة بتلك النتيجة).
4. `beneficiaries_count`, `evidence`, and `is_selected` are **NOT** generated by AI; they are entered/managed by the non-profit organization.

#### Complete Example JSON:

```json
{
  "involved_advisor_ids": [1, 2],
  "suggestion": {
    "interventions": [
      {
        "title": "برنامج حافلات الأمل للنقل المدرسي المنتظم",
        "confidence_level": "مرتفعة",
        "impact_description": "تمكين الطلاب والطالبات في القرى المعزولة من الوصول اليومي الآمن إلى مدارسهم بما يقضي على عائق المسافة الجغرافية.",
        "reportable_value": "خفض نسبة التسرب المدرسي للطلاب المستهدفين بنسبة لا تقل عن 18% بنهاية العام الدراسي.",
        "results": [
          {
            "text": "انتظام الحضور الدراسي للطلاب المنقولين بمعدل لا يقل عن 92% طوال الفصل الدراسي الأول.",
            "outputs": [
              {
                "text": "تجهيز وتشغيل 4 حافلات مدرسية مخصصة ومطابقة لمعايير السلامة."
              },
              {
                "text": "تحديد وتدشين 6 مسارات نقل يومية تغطي 8 قرى نائية."
              },
              {
                "text": "التعاقد مع سائقين محليين مؤهلين وتدريبهم على بروتوكولات الأمان وحماية الأطفال."
              }
            ]
          },
          {
            "text": "تحسن ملحوظ في التحصيل الدراسي بنسبة 15% بين الطلاب الملتزمين بالحضور.",
            "outputs": [
              {
                "text": "نظام متابعة إلكتروني أسبوعي لغياب وحضور الطلاب مع إدارات المدارس."
              },
              {
                "text": "عقد ورشتي عمل توعوية لأولياء الأمور حول أهمية الالتزام المدرسي."
              }
            ]
          }
        ]
      },
      {
        "title": "مبادرة الحقيبة والمساندة التعليمية للأسر النائية",
        "confidence_level": "متوسطة",
        "impact_description": "تخفيف العبء المالي المباشر عن الأسر الأشد حاجة لضمان عدم إخراج الأبناء من التعليم لأسباب مادية.",
        "reportable_value": "تغطية 100% من الاحتياجات المدرسية للأطفال المستهدفين مع انعدام حالات الانقطاع المالي.",
        "results": [
          {
            "text": "تأمين المستلزمات الأساسية بالكامل للطلبة قبل انطلاق العام الدراسي.",
            "outputs": [
              {
                "text": "توزيع 350 حقيبة مدرسية متكاملة بالزي المدرسي."
              },
              {
                "text": "صرف قسائم دعم للمصاريف المدرسية لـ 120 أسرة مستحقة."
              }
            ]
          }
        ]
      }
    ]
  }
}
```

---

## 5. Schema Reference Table for `suggestion.interventions`

| Field | Type | Required | Description & Constraints |
|---|---|---|---|
| `title` | `string` | **Yes** | اسم/عنوان المبادرة أو التدخل (e.g. "برنامج النقل المدرسي") |
| `confidence_level` | `string` | **Yes** | درجة الثقة المقترحة: `"مرتفعة"` \| `"متوسطة"` \| `"منخفضة"` (أو `"high"` \| `"medium"` \| `"low"`) |
| `impact_description` | `string` | **Yes** | وصف الأثر المتوقع للتدخل وكيف يحل المشكلة |
| `reportable_value` | `string` | **Yes** | القيمة الكمية القابلة للقياس والتقرير للمانحين ومجلس الإدارة |
| `results` | `array<object>` | **Yes** | قائمة النتائج المتوقعة تحت هذا التدخل (Results / Outcomes) |
| `results[].text` | `string` | **Yes** | نص النتيجة |
| `results[].outputs` | `array<object>` | **Yes** | قائمة المخرجات التنفيذية المباشرة المفرعة من هذه النتيجة |
| `results[].outputs[].text` | `string` | **Yes** | نص المخرج |

---

## 6. Single-Output Regeneration Sub-Flow (`is_output_regeneration`)

If the organization requests **"إعادة توليد"** for a single specific output block on Screen 3:

### Request Input to AI:
```json
{
  "run_id": "...",
  "type": "advisory_consultation",
  "consultation_id": 43,
  "topic": "interventions",
  "input": {
    "is_output_regeneration": true,
    "existing_text": "تجهيز وتشغيل 4 حافلات مدرسية مخصصة ومطابقة لمعايير السلامة.",
    "impact_description": "تمكين الطلاب والطالبات في القرى المعزولة من الوصول اليومي...",
    "social_problem": "ارتفاع معدلات التسرب من التعليم الأساسي..."
  }
}
```

### Expected Response:
```json
{
  "involved_advisor_ids": [1],
  "suggestion": {
    "text": "التعاقد مع شركة نقل معتمدة لتوفير أسطول حافلات حديث ومطابق لمعايير السلامة المدرسية."
  }
}
```

---

## 7. Error Handling & Validation Rules

1. **JSON Validity:** The response body must be strictly valid JSON without Markdown code blocks (` ```json `) when sent over HTTP webhooks.
2. **Hierarchy Enforcement:** Outputs **must** be nested inside Results (`intervention.results[].outputs[]`), **not** vice versa.
3. **Empty Arrays:** An intervention should have at least 1 result, and each result should have at least 1 output.
4. **Failure Callback:** If the AI model fails or encounters a content-filter error, return:
   ```json
   {
     "status": "failed",
     "error": "Reason for failure..."
   }
   ```
