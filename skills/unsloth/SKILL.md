---
name: unsloth
description: >
  Fine-tune, run inference on, and export LLMs and vision models using the
  Unsloth Studio CLI. Use when the user wants to train/fine-tune a model,
  run inference, export checkpoints to GGUF or other formats, manage the
  Unsloth Studio server, or anything involving unsloth. Always use this skill
  when the user mentions unsloth, fine-tuning, LoRA, model training, or
  model export — even if they don't explicitly say "unsloth".
license: Apache-2.0
metadata:
  author: unslothai
  version: "1.0"
---

## Prerequisites

```bash
pip install unsloth                # install the CLI entrypoint
unsloth studio setup               # one-time runtime setup (~/.unsloth/studio/.venv)
```

- Python 3.11-3.13 is required inside the studio venv (created by `unsloth studio setup`). The system Python can differ — the CLI re-executes in the studio venv automatically.
- After setup, commands like `train`, `inference`, `export`, `list-checkpoints` re-launch inside `~/.unsloth/studio/.venv` (you'll see "Launching with studio venv..." in output). Dependencies are resolved from the studio venv, not the user's environment.
- Training requires a supported GPU. For chat-only inference on macOS / CPU-only systems, use GGUF models so the CLI routes to the llama.cpp backend.
- Environment variables: `HF_TOKEN` (HuggingFace), `WANDB_API_KEY` (Weights & Biases)

## All CLI Commands

```bash
# === TRAINING ===
# Config file approach (RECOMMENDED — handles lists, avoids shell quoting)
unsloth train --config config.yaml
# --dry-run prints resolved config as YAML and exits (does NOT validate required fields)
unsloth train --config config.yaml --dry-run
# CLI flags (override config values). Use non-GGUF model IDs for training.
unsloth train --model "unsloth/Qwen3-0.6B" --dataset "tatsu-lab/alpaca" --training-type lora --num-epochs 3 --learning-rate 2e-4 --batch-size 2 --gradient-accumulation-steps 4 --max-seq-length 2048 --load-in-4bit --lora-r 64 --lora-alpha 16 --output-dir ./outputs --hf-token $HF_TOKEN --wandb-token $WANDB_API_KEY
# --local-dataset has NO CLI flag — use YAML config with local_dataset as a list
# More flags: --format-type auto|alpaca|chatml|sharegpt  --warmup-steps 5  --max-steps 0
#   --save-steps 0  --weight-decay 0.01  --random-seed 3407  --packing  --train-on-completions
#   --gradient-checkpointing unsloth|true|none  --lora-dropout 0.0  --use-rslora  --use-loftq
#   --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   --vision-all-linear  --finetune-vision-layers  --finetune-language-layers
#   --finetune-attention-modules  --finetune-mlp-modules
#   --enable-wandb  --wandb-project "unsloth-training"  --enable-tensorboard  --tensorboard-dir "runs"

# === INFERENCE ===
# Two positional args: MODEL PROMPT. Use GGUF models for macOS/CPU inference.
unsloth inference "unsloth/Qwen3-0.6B" "What is AI?"
unsloth inference "unsloth/Qwen3.5-0.8B-GGUF" "What is AI?"
# Optional flags: --temperature 0.7  --top-p 0.9  --top-k 40  --max-new-tokens 256
#   --repetition-penalty 1.1  --system-prompt "..."  --max-seq-length 2048  --load-in-4bit
unsloth inference "unsloth/Qwen3-0.6B" "Explain LoRA" --temperature 0.7 --max-new-tokens 512

# === EXPORT ===
unsloth list-checkpoints --outputs-dir ./outputs
# Formats: merged-16bit | merged-4bit | gguf | lora. GGUF quantizations: q4_k_m | q5_k_m | q8_0 | f16
unsloth export ./outputs/checkpoint-100 ./exported --format gguf --quantization q4_k_m
# Push to HuggingFace Hub (--push-to-hub requires --repo-id)
unsloth export ./outputs/checkpoint-100 ./exported --format lora --push-to-hub --repo-id user/model-name --hf-token $HF_TOKEN --private
# Additional flags: --max-seq-length 2048  --load-in-4bit / --no-load-in-4bit

# === STUDIO SERVER ===
unsloth studio setup                                   # one-time environment setup
unsloth studio -H 0.0.0.0 -p 8000                     # start web UI (default port: 8000)
unsloth studio --silent                                # suppress startup messages
unsloth studio reset-password                          # reset admin password (deletes auth DB)
unsloth ui -p 8000                                     # alias for "unsloth studio"
```

## Config File Format

A YAML config has 5 sections: `model`, `data`, `training`, `lora`, `logging`.

```yaml
model: unsloth/Qwen3-0.6B                # HuggingFace model ID (use non-GGUF for training)

data:
  dataset: tatsu-lab/alpaca               # HuggingFace dataset ID
  local_dataset:                          # local paths (YAML only — no CLI flag for list fields)
    - ./data/train.jsonl
  format_type: auto                       # auto | alpaca | chatml | sharegpt

training:
  training_type: lora                     # lora | full
  max_seq_length: 2048
  load_in_4bit: true
  output_dir: outputs
  num_epochs: 3
  learning_rate: 0.0002
  batch_size: 2
  gradient_accumulation_steps: 4

lora:
  lora_r: 64
  lora_alpha: 16
  target_modules: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

logging:
  enable_wandb: false
  wandb_project: unsloth-training
```

See `references/config-reference.md` for every field with type, default, and description.
See `assets/lora-text-train.yaml` for a copy-paste template.

## Key Gotchas

- `local_dataset` is a list field — there is **no CLI flag** for it. Always set it in the YAML config.
- `--dry-run` prints the resolved config but does **not** validate required fields (model, dataset can be null). Check the printed YAML yourself.
- Config file approach is preferred for agents (handles lists, avoids shell quoting issues)
- Use **non-GGUF** model IDs for training (e.g. `unsloth/Qwen3-0.6B`, not `-GGUF`)
- For inference, **GGUF** model IDs/files use `llama-server` while non-GGUF model IDs use the standard Unsloth backend; on macOS or other chat-only environments, prefer GGUF models.
- CLI commands auto-relaunch inside `~/.unsloth/studio/.venv` ("Launching with studio venv..." in output). Dependencies come from the studio venv, not the user's environment.
- Studio home directory: `~/.unsloth/studio/`, default training output: `./outputs`
- CLI flags override config file values (precedence: CLI > config > defaults)
- Boolean flags use `--flag / --no-flag` pattern (e.g. `--load-in-4bit / --no-load-in-4bit`)
- `--push-to-hub` requires `--repo-id`
