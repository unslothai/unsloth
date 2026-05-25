# Unsloth Config Reference

All fields from `cli/config.py` Pydantic models. CLI flags use kebab-case (`--lora-r`), config YAML uses snake_case (`lora_r`).

## Top-Level

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | str \| null | null | HuggingFace model ID or local path. Required for training. Use non-GGUF variant. |

## Data Section (`data:`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `dataset` | str \| null | null | HuggingFace dataset ID (e.g. `tatsu-lab/alpaca`) |
| `local_dataset` | list[str] \| null | null | Local dataset file paths. **YAML only** — no CLI flag (List type skipped). |
| `format_type` | `auto` \| `alpaca` \| `chatml` \| `sharegpt` | `auto` | Dataset format auto-detection or explicit. |

## Training Section (`training:`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `training_type` | `lora` \| `full` | `lora` | LoRA (adapter) or full fine-tuning. Full needs more VRAM. |
| `max_seq_length` | int | 2048 | Maximum sequence length for tokenization. |
| `load_in_4bit` | bool | true | 4-bit quantization. Reduces VRAM significantly. |
| `output_dir` | Path | `./outputs` | Directory for checkpoints and logs. |
| `num_epochs` | int | 3 | Number of training epochs. Ignored if `max_steps > 0`. |
| `learning_rate` | float | 2e-4 | Learning rate. |
| `batch_size` | int | 2 | Per-device batch size. |
| `gradient_accumulation_steps` | int | 4 | Effective batch = batch_size * gradient_accumulation_steps. |
| `warmup_steps` | int | 5 | Linear warmup steps. |
| `max_steps` | int | 0 | Max training steps. 0 = use `num_epochs` instead. |
| `save_steps` | int | 0 | Save checkpoint every N steps. 0 = only save at end. |
| `weight_decay` | float | 0.01 | Weight decay for AdamW. |
| `random_seed` | int | 3407 | Random seed for reproducibility. |
| `packing` | bool | false | Pack multiple sequences into `max_seq_length` windows. |
| `train_on_completions` | bool | false | Only compute loss on assistant completions. |
| `gradient_checkpointing` | `unsloth` \| `true` \| `none` | `unsloth` | `unsloth` is optimized and uses less VRAM than `true`. |

## LoRA Section (`lora:`)

Only used when `training_type: lora`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `lora_r` | int | 64 | LoRA rank. Higher = more parameters. |
| `lora_alpha` | int | 16 | LoRA alpha scaling factor. |
| `lora_dropout` | float | 0.0 | Dropout on LoRA layers. |
| `target_modules` | str | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Comma-separated module names. Leave empty for vision models using `vision_all_linear`. |
| `vision_all_linear` | bool | false | Apply LoRA to all linear layers in vision models. |
| `use_rslora` | bool | false | Rank-Stabilized LoRA. |
| `use_loftq` | bool | false | LoFTQ initialization. |
| `finetune_vision_layers` | bool | true | Fine-tune vision encoder layers (vision-language models). |
| `finetune_language_layers` | bool | true | Fine-tune language layers (vision-language models). |
| `finetune_attention_modules` | bool | true | Fine-tune attention modules. |
| `finetune_mlp_modules` | bool | true | Fine-tune MLP modules. |

## Logging Section (`logging:`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enable_wandb` | bool | false | Enable Weights & Biases logging. |
| `wandb_project` | str | `unsloth-training` | W&B project name. |
| `wandb_token` | str \| null | null | W&B API token. Also reads `WANDB_API_KEY` env var. |
| `enable_tensorboard` | bool | false | Enable TensorBoard logging. |
| `tensorboard_dir` | str | `runs` | TensorBoard log directory. |
| `hf_token` | str \| null | null | HuggingFace token. Also reads `HF_TOKEN` env var. |
