---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3-14B
tags:
- axolotl
- base_model:adapter:Qwen/Qwen3-14B
- lora
- transformers
datasets:
- /tmp/athar_meta_recovery_repo/data/train_meta_balanced.jsonl
pipeline_tag: text-generation
model-index:
- name: tmp/athar_meta_recovery_work/output_full
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.20.0.dev0`
```yaml
base_model: Qwen/Qwen3-14B
strict: false
chat_template: qwen3

datasets:
  - path: /tmp/athar_meta_recovery_repo/data/train_meta_balanced.jsonl
    ds_type: json
    split: train
    type: chat_template
    field_messages: messages
    roles_to_train:
      - assistant
    train_on_eos: turn

test_datasets:
  - path: /tmp/athar_meta_recovery_repo/data/validation_meta.jsonl
    ds_type: json
    split: train
    type: chat_template
    field_messages: messages
    roles_to_train:
      - assistant
    train_on_eos: turn

dataset_exact_deduplication: true
dataset_prepared_path: /tmp/athar_meta_recovery_work/prepared_full
output_dir: /tmp/athar_meta_recovery_work/output_full
sequence_len: 2048
sample_packing: false
eval_sample_packing: false
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - down_proj
  - up_proj
lora_qkv_kernel: true
lora_o_kernel: true
lora_mlp_kernel: true
embeddings_skip_upcast: true
bf16: true
fp16: false
tf32: true
attn_implementation: flash_attention_2
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
activation_offloading: hidden_states
selective_checkpointing:
  save:
    - attention
  offload: true
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_fused_linear_cross_entropy: true
liger_rope: false
liger_rms_norm: false
liger_glu_activation: false
liger_layer_norm: false
micro_batch_size: 1
eval_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 4
optimizer: paged_adamw_8bit
learning_rate: 0.0001
lr_scheduler: cosine
warmup_ratio: 0.1
weight_decay: 0.0
max_grad_norm: 0.1
logging_steps: 1
evals_per_epoch: 2
saves_per_epoch: 1
save_total_limit: 2
seed: 42
dataloader_num_workers: 2
dataloader_prefetch_factor: 4
dataloader_pin_memory: true

```

</details><br>

# tmp/athar_meta_recovery_work/output_full

This model is a fine-tuned version of [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) on the /tmp/athar_meta_recovery_repo/data/train_meta_balanced.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6204
- Ppl: 5.0553
- Memory/max Active (gib): 11.12
- Memory/max Allocated (gib): 11.12
- Memory/device Reserved (gib): 11.49

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.PAGED_ADAMW_8BIT with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 6
- training_steps: 66

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Ppl    | Active (gib) | Allocated (gib) | Reserved (gib) |
|:-------------:|:------:|:----:|:---------------:|:------:|:------------:|:---------------:|:--------------:|
| No log        | 0      | 0    | 2.2901          | 9.8761 | 11.04        | 11.04           | 11.16          |
| 1.5869        | 0.5496 | 9    | 1.7206          | 5.5880 | 11.12        | 11.12           | 11.52          |
| 0.7462        | 1.0611 | 18   | 1.3617          | 3.9027 | 11.12        | 11.12           | 11.51          |
| 0.2570        | 1.6107 | 27   | 1.4720          | 4.3581 | 11.12        | 11.12           | 11.47          |
| 0.0846        | 2.1221 | 36   | 1.5973          | 4.9398 | 11.12        | 11.12           | 11.49          |
| 0.1111        | 2.6718 | 45   | 1.6745          | 5.3364 | 11.12        | 11.12           | 11.52          |
| 0.0731        | 3.1832 | 54   | 1.6251          | 5.0790 | 11.12        | 11.12           | 11.49          |
| 0.0419        | 3.7328 | 63   | 1.6202          | 5.0539 | 11.12        | 11.12           | 11.5           |
| 0.0773        | 3.9160 | 66   | 1.6204          | 5.0553 | 11.12        | 11.12           | 11.49          |


### Framework versions

- PEFT 0.21.0
- Transformers 5.17.0
- Pytorch 2.12.1+cu130
- Datasets 4.8.4
- Tokenizers 0.23.1