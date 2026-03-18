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
pip install unsloth
unsloth studio setup
```

- Python 3.11-3.13 is required inside the studio venv (created by `unsloth studio setup`). The system Python can differ — the CLI re-executes in the studio venv automatically.
- After setup, commands like `train`, `inference`, `export`, `list-checkpoints` re-launch inside `~/.unsloth/studio/.venv` (you'll see "Launching with studio venv..." in output). Dependencies are resolved from the studio venv, not the user's environment.
- Training requires a supported GPU. For chat-only inference on macOS / CPU-only systems, use GGUF models so the CLI routes to the llama.cpp backend.
- Environment variables: `HF_TOKEN` (HuggingFace), `WANDB_API_KEY` (Weights & Biases)

## Training

Config file approach is recommended — handles lists, avoids shell quoting.

```bash
unsloth train --config config.yaml
unsloth train --config config.yaml --dry-run
```

CLI flags (override config values). Use non-GGUF model IDs for training.

```bash
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
```

- `--dry-run` prints resolved config as YAML and exits. Does NOT validate required fields.
- `--local-dataset` has no CLI flag — set `local_dataset` as a list in the YAML config.
- `--format-type` auto | alpaca | chatml | sharegpt (default: auto)
- `--warmup-steps` (default: 5), `--max-steps` (default: 0, uses num-epochs), `--save-steps` (default: 0)
- `--weight-decay` (default: 0.01), `--random-seed` (default: 3407)
- `--packing`, `--train-on-completions`, `--gradient-checkpointing` unsloth | true | none
- `--lora-dropout` (default: 0.0), `--target-modules` (default: q/k/v/o/gate/up/down_proj)
- `--vision-all-linear`, `--finetune-vision-layers`, `--finetune-language-layers`
- `--enable-wandb`, `--wandb-project`, `--enable-tensorboard`, `--tensorboard-dir`

## Inference

Two positional args: `MODEL` and `PROMPT`. Use GGUF models on macOS / CPU-only systems.

```bash
unsloth inference "unsloth/Qwen3.5-4B-GGUF" "What is AI?"
```

- `--temperature` (float, default: 0.7) — sampling temperature
- `--top-p` (float, default: 0.9) — nucleus sampling
- `--top-k` (int, default: 40) — top-k sampling
- `--max-new-tokens` (int, default: 256) — max tokens to generate
- `--repetition-penalty` (float, default: 1.1) — penalty for repeated tokens
- `--system-prompt` (str) — optional system prompt to prepend
- `--max-seq-length` (int, default: 2048) — max sequence length
- `--load-in-4bit` (default: on) — 4-bit quantization

## Export

```bash
unsloth list-checkpoints --outputs-dir ./outputs
unsloth export ./outputs/checkpoint-100 ./exported \
  --format gguf \
  --quantization q4_k_m
unsloth export ./outputs/checkpoint-100 ./exported \
  --format lora \
  --push-to-hub --repo-id user/model-name \
  --hf-token $HF_TOKEN \
  --private
```

- `--format` merged-16bit | merged-4bit | gguf | lora
- `--quantization` q4_k_m | q5_k_m | q8_0 | f16 (gguf only)
- `--push-to-hub` requires `--repo-id`
- `--max-seq-length` (default: 2048), `--load-in-4bit` / `--no-load-in-4bit`

## Studio Server

```bash
unsloth studio setup
unsloth studio -H 0.0.0.0 -p 8000
unsloth studio --silent
unsloth studio reset-password
unsloth ui -p 8000
```

## References

- `references/config-reference.md` — every config field with type, default, and description
- `assets/lora-text-train.yaml` — copy-paste config template for LoRA text fine-tuning (most common)
- `assets/full-finetune.yaml` — copy-paste config template for full fine-tuning (no LoRA, more VRAM)
- `assets/vision-lora-train.yaml` — copy-paste config template for vision model LoRA

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
