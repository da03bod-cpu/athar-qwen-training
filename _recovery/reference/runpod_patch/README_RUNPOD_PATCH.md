# Athar OS — RunPod patch + Specialist V2 training

Files in this patch:

- `handler_advisory_council.py` — full council handler with anti-prompt-echo / anti-repetition retry logic.
- `build_specialist_v2.py` — rebuilds the V2 Specialist dataset using the full original Expert DNA prompts.
- `specialist_v2_runpod.yaml` — single-GPU QLoRA continuation config for Qwen3-14B at sequence length 10,240.
- `train_specialist_v2_runpod.sh` — validates GPU/LFS/data and launches the training.

## Recommended RunPod GPU

Use one GPU with at least 40GB VRAM. Prefer A100 80GB or H100 80GB. L40S 48GB / A100 40GB can be tried with this config.

## Put the files in the repo

From the repository root:

```bash
cp handler_advisory_council.py ./handler_advisory_council.py
cp build_specialist_v2.py ./build_specialist_v2.py
cp specialist_v2_runpod.yaml ./specialist_v2_runpod.yaml
cp train_specialist_v2_runpod.sh ./train_specialist_v2_runpod.sh
chmod +x train_specialist_v2_runpod.sh
```

Commit and push before starting the pod if GitHub is the source of truth.

## Training

On the RunPod training pod, from the repo root:

```bash
git lfs pull
./train_specialist_v2_runpod.sh
```

The expected final adapter path is:

```text
outputs/qwen3-14b-athar-specialist-v2/adapter_model.safetensors
```

## Important coverage note

The current source Specialist dataset contains 580 training rows covering advisors 01–10 only (58 rows each). This training fixes the full-DNA training/inference mismatch for those examples and improves the shared Specialist behavior, but it is not direct supervised coverage for advisors 11–35. Do not claim 35-advisor direct training until high-quality cases for 11–35 are added.
