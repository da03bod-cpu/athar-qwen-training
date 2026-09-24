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
- /tmp/athar_specialist_v2_work/train_specialist_v2.jsonl
pipeline_tag: text-generation
model-index:
- name: tmp/athar_specialist_v2_work/output_full
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
  - path: /tmp/athar_specialist_v2_work/train_specialist_v2.jsonl
    ds_type: json
    split: train
    type: chat_template
    field_messages: messages
    roles_to_train:
      - assistant
    train_on_eos: turn

test_datasets:
  - path: /tmp/athar_specialist_v2_work/validation_specialist_v2.jsonl
    ds_type: json
    split: train
    type: chat_template
    field_messages: messages
    roles_to_train:
      - assistant
    train_on_eos: turn

dataset_exact_deduplication: true
dataset_prepared_path: /tmp/athar_specialist_v2_work/prepared_full
output_dir: /tmp/athar_specialist_v2_work/output_full

sequence_len: 10240
excess_length_strategy: raise
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
num_epochs: 1

optimizer: paged_adamw_8bit
learning_rate: 0.00002
lr_scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.0
max_grad_norm: 0.1

logging_steps: 1
evals_per_epoch: 1
saves_per_epoch: 1
save_total_limit: 2
seed: 42

dataloader_num_workers: 2
dataloader_prefetch_factor: 4
dataloader_pin_memory: true

```

</details><br>

# tmp/athar_specialist_v2_work/output_full

This model is a fine-tuned version of [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) on the /tmp/athar_specialist_v2_work/train_specialist_v2.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.PAGED_ADAMW_8BIT with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 2
- training_steps: 44

### Training results



### Framework versions

- PEFT 0.21.0
- Transformers 5.17.0
- Pytorch 2.12.1+cu130
- Datasets 4.8.4
- Tokenizers 0.23.1