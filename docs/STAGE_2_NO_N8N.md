# Athar OS — Stage 2/3 Council Execution Without n8n

This implementation removes n8n from the advisory path.

## Runtime flow

Backend
→ RunPod Serverless `handler_advisory_council.py`
→ resolve only the advisors selected by the organization
→ run each selected advisor independently with that advisor's **original full Expert DNA/System Prompt**
→ collect the raw independent opinions internally
→ switch to the **Meta adapter**
→ run the **original AOS-META-00 prompt**
→ synthesize the council into one unified answer
→ validate the final JSON
→ return the current Backend contract unchanged.

The individual advisor opinions are intentionally not returned to the Backend yet.

## Required repository layout

```text
athar-qwen-training/
├─ handler_advisory_council.py
├─ advisors/
│  └─ advisors_registry_35.json
├─ prompts/
│  ├─ advisors/
│  │  ├─ advisor_01.md
│  │  ├─ advisor_02.md
│  │  ├─ ...
│  │  └─ advisor_35.md
│  └─ meta/
│     └─ AOS-META-00.md
└─ checkpoints/
   ├─ specialist/
   │  └─ checkpoint-44/
   └─ meta/
      └─ checkpoint-18/
```

Do not summarize or shorten the advisor prompts when placing them under
`prompts/advisors/`. Store the authoritative full prompt text.

## Registry format

`advisors/advisors_registry_35.json`:

```json
[
  {
    "backend_id": 1,
    "advisor_code": "THE-EXACT-MODEL-ADVISOR-CODE",
    "name": "Exact advisor name",
    "prompt_file": "advisor_01.md"
  }
]
```

The registry must use the real Backend IDs and the exact advisor codes already
used by Athar OS. Do not infer or renumber them.

## Backend input

Preferred:

```json
{
  "run_id": "uuid",
  "type": "advisory_consultation",
  "consultation_id": 42,
  "topic": "interventions",
  "input": {
    "selected_advisor_ids": [1, 3, 7, 8, 10, 13],
    "organization": {},
    "programs": [],
    "advisors": [],
    "track": {},
    "goal": {},
    "impact_map": {}
  }
}
```

If `selected_advisor_ids` is not present, the worker treats the advisors inside
`input.advisors` as the selected council.

## Important behavior

- Advisor calls are isolated from each other.
- Each advisor sees only the shared case context plus its own original prompt.
- No advisor sees another advisor's opinion.
- AOS-META-00 sees all completed opinions only in the final synthesis stage.
- The original long prompts are never truncated silently.
- If the configured token limit is exceeded, the job fails explicitly rather
  than silently deleting part of an Expert DNA prompt.
- Final JSON is validated before being returned.
- The public output remains:
  `involved_advisor_ids + suggestion.interventions`.

## Environment variables

```text
MODEL_ID=Qwen/Qwen3-14B
SPECIALIST_ADAPTER_PATH=/path/to/checkpoint-44
META_ADAPTER_PATH=/path/to/checkpoint-18
ADVISOR_PROMPTS_DIR=/app/prompts/advisors
META_PROMPT_PATH=/app/prompts/meta/AOS-META-00.md
ADVISOR_REGISTRY_PATH=/app/advisors/advisors_registry_35.json

MAX_MODEL_INPUT_TOKENS=30000
ADVISOR_MAX_NEW_TOKENS=1600
META_MAX_NEW_TOKENS=2200
GEN_TEMPERATURE=0.25
GEN_TOP_P=0.90
```

## Next repository task

1. Export the 35 authoritative advisor prompts to UTF-8 Markdown.
2. Export the authoritative AOS-META-00 prompt to Markdown.
3. Build `advisors_registry_35.json` from the real IDs/codes.
4. Add this handler to the existing Serverless image.
5. Test with one advisor, then two, then the normal selected council.
