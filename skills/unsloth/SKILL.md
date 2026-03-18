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

## Training

Use a YAML config file (recommended) or CLI flags. Always use non-GGUF model IDs for training.

```bash
unsloth train --config config.yaml
unsloth train --config config.yaml --dry-run
```

`--dry-run` prints the resolved config as YAML and exits. It does not validate required fields.
`--local-dataset` has no CLI flag — set `local_dataset` as a list in the YAML config.
CLI flags override config values. See `references/config-reference.md` for all fields.

## Inference

Two positional args: `MODEL` and `PROMPT`. Use GGUF models on macOS / CPU-only systems.

```bash
unsloth inference "unsloth/Qwen3-0.6B" "What is AI?"
unsloth inference "unsloth/Qwen3.5-0.8B-GGUF" "What is AI?"
unsloth inference "unsloth/Qwen3-0.6B" "Explain LoRA" --temperature 0.7 --max-new-tokens 512
```

Flags: `--temperature` `--top-p` `--top-k` `--max-new-tokens` `--repetition-penalty` `--system-prompt` `--max-seq-length` `--load-in-4bit`

## Export

```bash
unsloth list-checkpoints --outputs-dir ./outputs
unsloth export ./outputs/checkpoint-100 ./exported --format gguf --quantization q4_k_m
unsloth export ./outputs/checkpoint-100 ./exported --format lora --push-to-hub --repo-id user/model-name --hf-token $HF_TOKEN --private
```

Formats: `merged-16bit` `merged-4bit` `gguf` `lora`. GGUF quantizations: `q4_k_m` `q5_k_m` `q8_0` `f16`. `--push-to-hub` requires `--repo-id`.

## Studio Server

```bash
unsloth studio setup
unsloth studio -H 0.0.0.0 -p 8000
unsloth studio reset-password
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
