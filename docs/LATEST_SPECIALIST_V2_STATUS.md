# Latest Specialist V2 Status — 2026-09-24

## Final preflight

The last all-35 preflight completed successfully with:

- `all35_ready: true`
- `covered_advisors: 1..35`
- `uncovered_advisors: []`
- `train_examples: 350`
- `validation_examples: 70`
- `train_shortages: {}`
- `validation_shortages: {}`
- `generated_train_examples: 250`
- `generated_validation_examples: 50`
- `sequence_len: 10240`
- `source_adapter: specialist`
- GPU: NVIDIA RTX A5000, 23.55 GB VRAM
- BF16 supported: true
- constrained memory mode: true

## Final smoke training

A real optimizer step completed successfully:

- `status: completed`
- `activated: false`
- adapter output: `/tmp/athar_specialist_v2_work/output_smoke`
- adapter size: `190.06 MB`
- loss: `0.967`
- grad norm: `0.9088`
- learning rate: `2e-05`
- perplexity: `2.63`
- max active GPU memory: `12.61 GiB`
- max allocated GPU memory: `12.53 GiB`
- reserved GPU memory: `13.5 GiB`
- trainable tokens: `567`
- total tokens in the step: `8384`
- actual trainer runtime for the step: about `16.3 s`
- total reported training pipeline time: `97.89 s`
- Axolotl message: `Training completed!` and model successfully saved.

This proved the 24 GB constrained-memory path works for the all-35 data at sequence length 10,240.

## What happened after the smoke

Before full training, GitHub write access was intentionally preflighted. A `403` exposed that the token had no repository access. While fixing that, the GitHub organization containing the original repository was deleted, which deleted the repository. No full Specialist V2 checkpoint was published after the smoke.
