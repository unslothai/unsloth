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
# ═══ TRAINING ═══

# Config file approach (RECOMMENDED — handles lists, avoids shell quoting)
unsloth train --config config.yaml
# --dry-run prints resolved config as YAML and exits (does NOT validate required fields)
unsloth train --config config.yaml --dry-run

# Pure CLI flags (CLI flags override config file values)
unsloth train \
  --model "unsloth/Qwen3-0.6B" \
  --dataset "tatsu-lab/alpaca" \
  --training-type lora \
  --num-epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 2048 \
  --load-in-4bit \
  --lora-r 64 --lora-alpha 16 \
  --output-dir ./outputs \
  --hf-token $HF_TOKEN \
  --wandb-token $WANDB_API_KEY

# IMPORTANT: --local-dataset has no CLI flag — use the YAML config to set local_dataset as a list.
# IMPORTANT: --dry-run prints the resolved config but does NOT check required fields (model, dataset).
#   Verify the printed YAML yourself before running without --dry-run.
# IMPORTANT: Use non-GGUF model IDs for training (e.g. unsloth/Qwen3-0.6B, NOT the -GGUF variant)

# Additional training flags:
#   --format-type auto|alpaca|chatml|sharegpt   (default: auto)
#   --warmup-steps 5                            (linear warmup)
#   --max-steps 0                               (0 = use num-epochs)
#   --save-steps 0                              (0 = don't save intermediate checkpoints)
#   --weight-decay 0.01
#   --random-seed 3407
#   --packing / --no-packing                    (pack sequences to max_seq_length)
#   --train-on-completions / --no-train-on-completions
#   --gradient-checkpointing unsloth|true|none  (default: unsloth)
#   --lora-dropout 0.0
#   --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   --vision-all-linear / --no-vision-all-linear  (for vision models)
#   --use-rslora / --no-use-rslora
#   --use-loftq / --no-use-loftq
#   --finetune-vision-layers / --no-finetune-vision-layers
#   --finetune-language-layers / --no-finetune-language-layers
#   --finetune-attention-modules / --no-finetune-attention-modules
#   --finetune-mlp-modules / --no-finetune-mlp-modules
#   --enable-wandb / --no-enable-wandb
#   --wandb-project "unsloth-training"
#   --enable-tensorboard / --no-enable-tensorboard
#   --tensorboard-dir "runs"

# ═══ INFERENCE ═══

# Standard inference
unsloth inference "unsloth/Qwen3-0.6B" "What is AI?"

# GGUF inference (use on macOS / CPU-only systems)
unsloth inference "unsloth/Qwen3.5-0.8B-GGUF" "What is AI?"

# Inference options:
#   --temperature (float, default: 0.7) — sampling temperature
#   --top-p (float, default: 0.9) — nucleus sampling
#   --top-k (int, default: 40) — top-k sampling
#   --max-new-tokens (int, default: 256) — max tokens to generate
#   --repetition-penalty (float, default: 1.1) — penalty for repeated tokens
#   --system-prompt (str) — optional system prompt to prepend
#   --max-seq-length (int, default: 2048) — max sequence length
#   --load-in-4bit (default: on) — 4-bit quantization

# ═══ EXPORT ═══

# List available checkpoints
unsloth list-checkpoints --outputs-dir ./outputs

# Export to GGUF (most common for deployment)
# --format: merged-16bit | merged-4bit | gguf | lora
# --quantization: q4_k_m | q5_k_m | q8_0 | f16 (gguf only)
unsloth export ./outputs/checkpoint-100 ./exported \
  --format gguf \
  --quantization q4_k_m

# Export and push to HuggingFace Hub
unsloth export ./outputs/checkpoint-100 ./exported \
  --format lora \
  --push-to-hub --repo-id user/model-name \
  --hf-token $HF_TOKEN \
  --private

# Additional export flags:
#   --max-seq-length 2048
#   --load-in-4bit / --no-load-in-4bit

# ═══ STUDIO SERVER ═══

# one-time environment setup
unsloth studio setup
# start web UI (default port: 8000)
unsloth studio -H 0.0.0.0 -p 8000
# suppress startup messages
unsloth studio --silent
# reset admin password (deletes auth DB)
unsloth studio reset-password
# alias for "unsloth studio"
unsloth ui -p 8000
```

## Config File Format

A YAML config has 5 sections: `model`, `data`, `training`, `lora`, `logging`. Copy-paste templates:

- `assets/lora-text-train.yaml` — LoRA text fine-tuning (most common)
- `assets/full-finetune.yaml` — full fine-tuning (no LoRA, more VRAM)
- `assets/vision-lora-train.yaml` — vision model LoRA

See `references/config-reference.md` for every field with type, default, and description.

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
