# Athar OS — Qwen3-14B RunPod Repository (Recovered)

This repository is a recovery of the deleted Athar OS training/serverless repository, reconstructed from the exact repository snapshot, project files, generated files, and the latest successful RunPod results available before deletion.

## Latest operational state reached

The latest Specialist V2 pipeline passed both preflight and a real one-step smoke update on 2026-09-24:

- Base model: `Qwen/Qwen3-14B`
- Method: QLoRA 4-bit
- Advisors covered: **35 / 35**
- Balanced Specialist V2 training set: **350 examples** (10 per advisor)
- Balanced validation set: **70 examples** (2 per advisor)
- Sequence length: **10,240**
- GPU used in the final smoke: **NVIDIA RTX A5000, 23.55 GB VRAM**
- Smoke loss: **0.967**
- Smoke grad norm: **0.9088**
- Smoke perplexity: **2.63**
- Peak active GPU memory: **12.61 GiB**
- Smoke adapter size: **190.06 MB**
- Result: `Training completed` and adapter saved successfully.

See `docs/LATEST_SPECIALIST_V2_STATUS.md` and `_recovery/logs/`.

## Important recovery limitation

The old GitHub source archive contained the checkpoint binaries as Git LFS pointers, not the real LFS objects. The exact pointer OIDs are preserved under:

`_recovery/lfs_pointers/`

and indexed in:

`docs/LFS_POINTER_OIDS.json`

Therefore the following real adapter weights are **not inside this ZIP**:

- `checkpoints/base/adapter_model.safetensors`
- `checkpoints/meta/adapter_model.safetensors`
- `checkpoints/specialist/adapter_model.safetensors`
- `checkpoints/matcher/adapter_model.safetensors`
- the smoke-only `checkpoints/specialist-v2` adapter

The last verified old adapters were about **190.06 MB each**. Their metadata/config/tokenizer files are preserved.

Do not run full Specialist V2 training until a real `specialist` or `specialist-v2` adapter has been restored, or until we deliberately switch to a from-base recovery training path.

## Repository structure

- `handler.py` — latest unified Serverless handler, including all-35 Specialist V2, continual-learning hooks, GitHub publish preflight, and GitHub-direct publishing.
- `handler_advisory_council.py` — latest recovered council engine.
- `advisors/` — 35-advisor registry plus schema.
- `prompts/advisors/` — all 35 full original advisor prompts.
- `prompts/meta/AOS-META-00.md` — Meta Advisor prompt.
- `data/` — original SFT, Meta, Specialist, Matcher datasets plus the newest all-35 V2 bootstrap data.
- `data/compiled/` — deterministic 350/70 all-35 datasets prepared without using RunPod generation tokens.
- `configs/` — original training configs plus the latest Specialist V2 reference config.
- `checkpoints/` — recovered non-LFS checkpoint metadata; real LFS weights are currently missing.
- `test_payloads/` — Serverless sample/preflight payloads from the previous repo.
- `docs/` — contracts, Expert DNA, recovery status, and LFS records.
- `_recovery/` — historical handlers, logs, pointer records, and reference patch files.

## Create the replacement GitHub repository

Create a new **private** repository under an owner you control, for example:

`YOUR_GITHUB_USERNAME/athar-qwen-training-recovered`

Upload/push the contents of this directory. Do not reuse the deleted organization name as a hard-coded dependency.

Then set these RunPod environment variables:

```text
GITHUB_REPO=YOUR_GITHUB_USERNAME/athar-qwen-training-recovered
GITHUB_BRANCH=main
GITHUB_TOKEN=<fine-grained token with Contents: Read and write on this repo>
CONTINUAL_STORAGE_BACKEND=github
```

`handler.py` in this recovered package no longer hard-codes the deleted repository; `GITHUB_REPO` is required when a GitHub operation is used.

## Verify the recovered repository locally

```bash
python scripts/verify_repo.py
```

This verifies the prompt count, registries, dataset counts, compiled all-35 data, Python syntax, and checkpoint recovery status.

## Current Specialist V2 data

Original legacy Specialist data:

- `data/train_specialist.jsonl` — 580 examples, advisors 01–10 only, 58 each.
- `data/validation_specialist.jsonl` — 190 examples.

Newest all-35 bootstrap additions:

- `data/train_specialist_v2_generated.jsonl` — 250 examples for advisors 11–35, 10 each.
- `data/validation_specialist_v2_generated.jsonl` — 50 examples for advisors 11–35, 2 each.

Deterministic final balanced data used by the latest preflight/smoke:

- `data/compiled/train_specialist_v2_all35_350.jsonl`
- `data/compiled/validation_specialist_v2_all35_70.jsonl`

The full Expert DNA prompt is embedded as the `system` message in the compiled files.

## What not to do yet

Do **not** send `{"type":"train_specialist_v2","mode":"full"}` against this recovered repo until the real source Specialist adapter is restored/rebuilt. The latest handler intentionally rejects Git LFS pointer stubs.
