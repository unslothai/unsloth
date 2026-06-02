# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject

logger = logging.getLogger(__name__)
from utils.upload_limits import (
    MAX_UPLOAD_LIMIT_MB,
    MIN_UPLOAD_LIMIT_MB,
    default_upload_limit_mb,
    get_upload_limit_mb,
    set_upload_limit_mb,
    upload_limit_bytes,
    upload_limit_label,
)

router = APIRouter()


class UploadLimitPayload(BaseModel):
    max_upload_size_mb: int = Field(..., ge = MIN_UPLOAD_LIMIT_MB, le = MAX_UPLOAD_LIMIT_MB)


class UploadLimitResponse(BaseModel):
    max_upload_size_mb: int
    max_upload_size_bytes: int
    max_upload_size_label: str
    default_upload_size_mb: int
    min_upload_size_mb: int = MIN_UPLOAD_LIMIT_MB
    max_allowed_upload_size_mb: int = MAX_UPLOAD_LIMIT_MB


def _upload_limit_response(limit_mb: int) -> UploadLimitResponse:
    return UploadLimitResponse(
        max_upload_size_mb = limit_mb,
        max_upload_size_bytes = upload_limit_bytes(limit_mb),
        max_upload_size_label = upload_limit_label(limit_mb),
        default_upload_size_mb = default_upload_limit_mb(),
    )


@router.get("/upload-limit", response_model = UploadLimitResponse)
def get_upload_limit(
    current_subject: str = Depends(get_current_subject),
) -> UploadLimitResponse:
    return _upload_limit_response(get_upload_limit_mb())


@router.put("/upload-limit", response_model = UploadLimitResponse)
def update_upload_limit(
    payload: UploadLimitPayload,
    current_subject: str = Depends(get_current_subject),
) -> UploadLimitResponse:
    try:
        limit_mb = set_upload_limit_mb(payload.max_upload_size_mb)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    return _upload_limit_response(limit_mb)


# Documentation export endpoint for RAG and offline use
_DOCS_CACHE: str | None = None


def _get_bundled_docs() -> str:
    """
    Return bundled documentation as markdown.
    Loads from the assets/docs directory if available, otherwise returns a
    comprehensive quick-reference guide with links to online documentation.
    """
    global _DOCS_CACHE
    if _DOCS_CACHE is not None:
        return _DOCS_CACHE

    # Try to load bundled docs from assets
    assets_dir = Path(__file__).parent.parent / "assets" / "docs"
    bundled_docs_path = assets_dir / "unsloth-docs.md"

    if bundled_docs_path.exists():
        try:
            _DOCS_CACHE = bundled_docs_path.read_text(encoding = "utf-8")
            return _DOCS_CACHE
        except Exception as e:
            logger.warning(f"Failed to read bundled docs: {e}")

    # Return comprehensive quick-reference guide
    _DOCS_CACHE = _generate_quick_reference_docs()
    return _DOCS_CACHE


def _generate_quick_reference_docs() -> str:
    """Generate a comprehensive quick-reference documentation file."""
    return """# Unsloth Documentation

> **Quick Reference Guide for Local RAG & Offline Use**
> For the most up-to-date documentation, visit: https://unsloth.ai/docs

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Supported Models](#supported-models)
4. [Fine-tuning Guide](#fine-tuning-guide)
5. [LoRA & QLoRA](#lora--qlora)
6. [Dataset Preparation](#dataset-preparation)
7. [Training Parameters](#training-parameters)
8. [Exporting Models](#exporting-models)
9. [GGUF Quantization](#gguf-quantization)
10. [Inference](#inference)
11. [Troubleshooting](#troubleshooting)

---

## Getting Started

Unsloth is a library for fast and memory-efficient LLM fine-tuning. It provides:

- **2x faster training** with 50% less memory usage
- **QLoRA and LoRA support** for efficient fine-tuning
- **GGUF export** for deployment with llama.cpp
- **Unsloth Studio** - a visual interface for training and inference

### Quick Start

```python
from unsloth import FastLanguageModel

# Load a model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

---

## Installation

### pip Installation

```bash
pip install unsloth
```

### Unsloth Studio

```bash
pip install unsloth[studio]
unsloth-studio
```

### From Source

```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
pip install -e .
```

---

## Supported Models

Unsloth supports a wide range of models including:

### Text Models
- Llama 3.2, 3.1, 3, 2 (all sizes)
- Mistral, Mixtral
- Qwen 2.5, 3
- Phi-3, Phi-4
- Gemma 2, 3
- DeepSeek V2, V3

### Vision Models
- Llama 3.2 Vision
- Qwen 2.5 VL
- Pixtral

### Audio Models
- Qwen 2 Audio

For a complete list, see: https://unsloth.ai/docs/models

---

## Fine-tuning Guide

### Basic Training Loop

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)

trainer.train()
```

---

## LoRA & QLoRA

### LoRA Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `r` | LoRA rank | 8-64 |
| `lora_alpha` | Scaling factor | Same as r |
| `lora_dropout` | Dropout rate | 0-0.1 |
| `target_modules` | Layers to adapt | All attention + MLP |

### Memory Comparison

| Method | VRAM (7B model) |
|--------|-----------------|
| Full Fine-tuning | ~60GB |
| LoRA | ~16GB |
| QLoRA (4-bit) | ~6GB |

---

## Dataset Preparation

### Alpaca Format

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

### ShareGPT Format

```json
{
  "conversations": [
    {"from": "human", "value": "Hello!"},
    {"from": "gpt", "value": "Hi there!"}
  ]
}
```

### ChatML Format

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

---

## Training Parameters

### Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `learning_rate` | Step size | 1e-5 to 2e-4 |
| `batch_size` | Samples per step | 1-8 |
| `gradient_accumulation` | Virtual batch size | 4-16 |
| `epochs` | Full dataset passes | 1-3 |
| `warmup_steps` | LR warmup | 5-100 |
| `weight_decay` | Regularization | 0-0.1 |

### Gradient Checkpointing

- `"none"` - Fastest, most memory
- `"true"` - Standard checkpointing
- `"unsloth"` - Optimized, recommended

---

## Exporting Models

### Save LoRA Adapters

```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### Merge and Save

```python
model.save_pretrained_merged("merged_model", tokenizer)
```

### Push to Hub

```python
model.push_to_hub("username/model-name", token="hf_token")
```

---

## GGUF Quantization

### Export to GGUF

```python
model.save_pretrained_gguf(
    "model",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Quantization Methods

| Method | Size | Quality | Speed |
|--------|------|---------|-------|
| `f16` | 100% | Best | Slow |
| `q8_0` | 50% | Excellent | Fast |
| `q5_k_m` | 35% | Very Good | Fast |
| `q4_k_m` | 25% | Good | Fastest |
| `q2_k` | 15% | Acceptable | Fastest |

---

## Inference

### With Unsloth

```python
FastLanguageModel.for_inference(model)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### With llama.cpp (GGUF)

```bash
./llama-server -m model.gguf -c 2048
```

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Enable gradient checkpointing
- Use 4-bit quantization

**Slow Training**
- Enable Flash Attention 2
- Use `unsloth` gradient checkpointing
- Increase batch size if memory allows

**Model Not Loading**
- Check model name spelling
- Verify HuggingFace token
- Ensure sufficient disk space

### Getting Help

- GitHub Issues: https://github.com/unslothai/unsloth/issues
- Documentation: https://unsloth.ai/docs
- Discord: https://discord.gg/unsloth

---

## Links

- **Documentation**: https://unsloth.ai/docs
- **GitHub**: https://github.com/unslothai/unsloth
- **Models**: https://huggingface.co/unsloth
- **Changelog**: https://unsloth.ai/docs/new/changelog

---

*Generated by Unsloth Studio - For offline RAG and local reference*
"""


@router.get(
    "/docs/download",
    response_class = PlainTextResponse,
    summary = "Download documentation for offline use",
    description = "Returns Unsloth documentation as a markdown file for RAG and offline use.",
)
def download_docs(
    current_subject: str = Depends(get_current_subject),
) -> PlainTextResponse:
    """
    Download Unsloth documentation as a markdown file.
    Useful for local RAG systems and offline reference.
    """
    docs_content = _get_bundled_docs()
    return PlainTextResponse(
        content = docs_content,
        media_type = "text/markdown",
        headers = {
            "Content-Disposition": "attachment; filename=unsloth-docs.md",
        },
    )
