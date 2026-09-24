# Recovery Status

## Recovered exactly or from exact project artifacts

- Full repository snapshot from 2026-09-21.
- All 35 advisor prompts.
- AOS-META-00 prompt.
- `advisors_registry_35.json` and rich/matcher registries.
- Original training/validation datasets for base, Meta, Specialist, Matcher v1 and Matcher v2.
- Original base/meta/specialist/matcher YAML configs.
- Latest unified advisory council implementation.
- Latest unified Serverless `handler.py` chain through the 2026-09-24 GitHub-direct patch.
- All-35 Specialist V2 bootstrap dataset generated outside RunPod: 250 train + 50 validation for advisors 11–35.
- Deterministic final balanced all-35 dataset: 350 train + 70 validation.
- Latest Specialist V2 preflight/smoke logs and metrics.
- Original Git LFS pointer OIDs for checkpoint binary files.

## Not recoverable from the project files currently available

The actual Git LFS binary objects for the old adapters were not embedded in the source ZIP. The source ZIP only contained pointer stubs. The missing real weights are the main remaining gap.

Known old adapter history:

- Base/SFT: latest known original checkpoint lineage reached `checkpoint-54`.
- Meta: latest known original checkpoint lineage reached `checkpoint-18`.
- Specialist: latest known original checkpoint lineage reached `checkpoint-44`.
- Matcher: a trained matcher adapter existed, but its real binary is also absent from the recovered archive.
- Specialist V2: only the final one-step smoke adapter was produced; it was stored in the RunPod worker `/tmp` and was never published before repository deletion.

## Current production architecture preserved in code

1. `advisory_match` evaluates advisor relevance across all 35 advisors.
2. The organization selects advisors.
3. `advisory_consultation` runs only selected advisors independently using their full Expert DNA prompts.
4. Advisors do not see one another's opinions.
5. AOS-META-00 synthesizes the independent opinions into the Screen 3 JSON contract.
6. The same Qwen3-14B base model is shared in memory and LoRA adapters are switched as needed.
7. Specialist outputs are guarded against prompt echo, repetition, unsupported numbers, unsupported durations, and out-of-scope content.
8. Specialist V2 uses the 10,240-token full-DNA context and constrained-memory training path for ~24 GB GPUs.

## Recommended recovery order

1. Push this recovered repository to a new private GitHub repository.
2. Set `GITHUB_REPO` and a write-capable `GITHUB_TOKEN` in RunPod.
3. Try to recover the actual `specialist` and `meta` adapter binaries from any surviving warm RunPod worker, Kaggle session, local clone, or other cache.
4. Only after real adapters exist, rerun Specialist V2 preflight and then the full training job.
5. If the real adapters cannot be recovered, rebuild them from the preserved datasets instead of using the pointer stubs.
