# Dataset Inventory

| File | Recovered rows | Purpose |
|---|---:|---|
| `train.jsonl` | 670 | original Base/SFT train |
| `validation.jsonl` | 220 | original Base/SFT validation |
| `train_meta_balanced.jsonl` | 131 | Meta advisor train |
| `validation_meta.jsonl` | 30 | Meta validation |
| `train_specialist.jsonl` | 580 | legacy Specialist, advisors 01–10 |
| `validation_specialist.jsonl` | 190 | legacy Specialist validation |
| `train_matcher_v1.jsonl` | 252 | Matcher v1 train |
| `validation_matcher_v1.jsonl` | 67 | Matcher v1 validation |
| `train_matcher_v2.jsonl` | 256 | Matcher v2 train |
| `validation_matcher_v2.jsonl` | 128 | Matcher v2 validation |
| `train_specialist_v2_generated.jsonl` | 250 | bootstrap additions for advisors 11–35 |
| `validation_specialist_v2_generated.jsonl` | 50 | bootstrap validation additions for advisors 11–35 |
| `train_specialist_v2.jsonl` | 580 | reconstructed legacy Specialist rows with full advisor system prompt |
| `validation_specialist_v2.jsonl` | 190 | reconstructed legacy validation with full prompts |
| `compiled/train_specialist_v2_all35_350.jsonl` | 350 | exact balanced shape used by latest all-35 pipeline, no feedback |
| `compiled/validation_specialist_v2_all35_70.jsonl` | 70 | exact balanced validation shape |

The 250/50 bootstrap files are synthetic bootstrap examples constructed from the advisors' authoritative Expert DNA. They are not human-reviewed gold labels. Production accepted/corrected feedback was intended to improve the training set over time.
