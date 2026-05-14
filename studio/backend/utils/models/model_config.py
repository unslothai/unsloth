# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Model and LoRA configuration handling
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from utils.paths import (
    normalize_path,
    is_local_path,
    is_model_cached,
    get_cache_path,
    resolve_cached_repo_id_case,
    outputs_root,
    exports_root,
    resolve_output_dir,
    resolve_export_dir,
)
from utils.utils import without_hf_auth
import structlog
from loggers import get_logger
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import hashlib
import json
import threading
import yaml


from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)

# ── Model size extraction ────────────────────────────────────
import re as _re

_MODEL_SIZE_RE = _re.compile(
    r"(?:^|[-_/])(\d+\.?\d*)\s*([bm])(?:$|[-_/])", _re.IGNORECASE
)
# MoE active-parameter pattern: matches "A3B", "A3.5B", etc.
_ACTIVE_SIZE_RE = _re.compile(
    r"(?:^|[-_/])a(\d+\.?\d*)\s*([bm])(?:$|[-_/])", _re.IGNORECASE
)


def extract_model_size_b(model_id: str) -> float | None:
    """Extract model size in billions from a model identifier.

    Prefers MoE active-parameter notation (e.g. ``A3B`` in
    ``Qwen3.5-35B-A3B``) over the total parameter count.
    Handles both ``B`` (billions) and ``M`` (millions) suffixes.
    """
    mid = (model_id or "").lower()
    active = _ACTIVE_SIZE_RE.search(mid)
    if active:
        val = float(active.group(1))
        return val / 1000.0 if active.group(2).lower() == "m" else val
    size = _MODEL_SIZE_RE.search(mid)
    if not size:
        return None
    val = float(size.group(1))
    return val / 1000.0 if size.group(2).lower() == "m" else val


# Model name mapping: maps all equivalent model names to their canonical YAML config file
# Format: "canonical_model_name.yaml": [list of all equivalent model names]
# Based on the model mapper provided - canonical filename is based on the first model name in the mapper
MODEL_NAME_MAPPING = {
    # ── Embedding models ──
    "unsloth_all-MiniLM-L6-v2.yaml": [
        "unsloth/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    "unsloth_bge-m3.yaml": [
        "unsloth/bge-m3",
        "BAAI/bge-m3",
    ],
    "unsloth_embeddinggemma-300m.yaml": [
        "unsloth/embeddinggemma-300m",
        "google/embeddinggemma-300m",
    ],
    "unsloth_gte-modernbert-base.yaml": [
        "unsloth/gte-modernbert-base",
        "Alibaba-NLP/gte-modernbert-base",
    ],
    "unsloth_Qwen3-Embedding-0.6B.yaml": [
        "unsloth/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-0.6B",
        "unsloth/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-4B",
    ],
    # ── Other models ──
    "unsloth_answerdotai_ModernBERT-large.yaml": [
        "answerdotai/ModernBERT-large",
    ],
    "unsloth_Qwen2.5-Coder-7B-Instruct-bnb-4bit.yaml": [
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ],
    "unsloth_codegemma-7b-bnb-4bit.yaml": [
        "unsloth/codegemma-7b-bnb-4bit",
        "unsloth/codegemma-7b",
        "google/codegemma-7b",
    ],
    "unsloth_ERNIE-4.5-21B-A3B-PT.yaml": [
        "unsloth/ERNIE-4.5-21B-A3B-PT",
    ],
    "unsloth_ERNIE-4.5-VL-28B-A3B-PT.yaml": [
        "unsloth/ERNIE-4.5-VL-28B-A3B-PT",
    ],
    "tiiuae_Falcon-H1-0.5B-Instruct.yaml": [
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "unsloth/Falcon-H1-0.5B-Instruct",
    ],
    "unsloth_functiongemma-270m-it.yaml": [
        "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
        "google/functiongemma-270m-it",
        "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
    ],
    "unsloth_gemma-2-2b.yaml": [
        "unsloth/gemma-2-2b-bnb-4bit",
        "google/gemma-2-2b",
    ],
    "unsloth_gemma-2-27b-bnb-4bit.yaml": [
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-9b",
        "google/gemma-2-9b",
        "unsloth/gemma-2-27b",
        "google/gemma-2-27b",
    ],
    "unsloth_gemma-3-4b-pt.yaml": [
        "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
        "google/gemma-3-4b-pt",
        "unsloth/gemma-3-4b-pt-bnb-4bit",
    ],
    "unsloth_gemma-3-4b-it.yaml": [
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
    ],
    "unsloth_gemma-3-27b-it.yaml": [
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
    ],
    "unsloth_gemma-3-270m-it.yaml": [
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "google/gemma-3-270m-it",
        "unsloth/gemma-3-270m-it-bnb-4bit",
    ],
    "unsloth_gemma-3n-E4B-it.yaml": [
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    ],
    "unsloth_gemma-3n-E4B.yaml": [
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "google/gemma-3n-E4B",
    ],
    "unsloth_gemma-4-31B-it.yaml": [
        "unsloth/gemma-4-31B-it",
        "google/gemma-4-31B-it",
    ],
    "unsloth_gemma-4-26B-A4B-it.yaml": [
        "unsloth/gemma-4-26B-A4B-it",
        "google/gemma-4-26B-A4B-it",
    ],
    "unsloth_gemma-4-E2B-it.yaml": [
        "unsloth/gemma-4-E2B-it",
        "google/gemma-4-E2B-it",
    ],
    "unsloth_gemma-4-E4B-it.yaml": [
        "unsloth/gemma-4-E4B-it",
        "google/gemma-4-E4B-it",
    ],
    "unsloth_gemma-4-31B.yaml": [
        "unsloth/gemma-4-31B",
        "google/gemma-4-31B",
    ],
    "unsloth_gemma-4-26B-A4B.yaml": [
        "unsloth/gemma-4-26B-A4B",
        "google/gemma-4-26B-A4B",
    ],
    "unsloth_gemma-4-E2B.yaml": [
        "unsloth/gemma-4-E2B",
        "google/gemma-4-E2B",
    ],
    "unsloth_gemma-4-E4B.yaml": [
        "unsloth/gemma-4-E4B",
        "google/gemma-4-E4B",
    ],
    "unsloth_gpt-oss-20b.yaml": [
        "openai/gpt-oss-20b",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b-BF16",
    ],
    "unsloth_gpt-oss-120b.yaml": [
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    ],
    "unsloth_granite-4.0-350m-unsloth-bnb-4bit.yaml": [
        "unsloth/granite-4.0-350m",
        "ibm-granite/granite-4.0-350m",
        "unsloth/granite-4.0-350m-bnb-4bit",
    ],
    "unsloth_granite-4.0-h-micro.yaml": [
        "ibm-granite/granite-4.0-h-micro",
        "unsloth/granite-4.0-h-micro-bnb-4bit",
        "unsloth/granite-4.0-h-micro-unsloth-bnb-4bit",
    ],
    "unsloth_LFM2-1.2B.yaml": [
        "unsloth/LFM2-1.2B",
    ],
    "unsloth_llama-3-8b-bnb-4bit.yaml": [
        "unsloth/llama-3-8b",
        "meta-llama/Meta-Llama-3-8B",
    ],
    "unsloth_llama-3-8b-Instruct-bnb-4bit.yaml": [
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    "unsloth_Meta-Llama-3.1-70B-bnb-4bit.yaml": [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-unsloth-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B",
    ],
    "unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit.yaml": [
        "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "RedHatAI/Llama-3.1-8B-Instruct-FP8",
        "unsloth/Llama-3.1-8B-Instruct-FP8-Block",
        "unsloth/Llama-3.1-8B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-3B-Instruct.yaml": [
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.2-3B-Instruct-FP8",
        "unsloth/Llama-3.2-3B-Instruct-FP8-Block",
        "unsloth/Llama-3.2-3B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-1B-Instruct.yaml": [
        "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.2-1B-Instruct-FP8",
        "unsloth/Llama-3.2-1B-Instruct-FP8-Block",
        "unsloth/Llama-3.2-1B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-11B-Vision-Instruct.yaml": [
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    ],
    "unsloth_Llama-3.3-70B-Instruct.yaml": [
        "unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.3-70B-Instruct",
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.3-70B-Instruct-FP8",
        "unsloth/Llama-3.3-70B-Instruct-FP8-Block",
        "unsloth/Llama-3.3-70B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llasa-3B.yaml": [
        "HKUSTAudio/Llasa-1B",
        "unsloth/Llasa-3B",
    ],
    "unsloth_Magistral-Small-2509-unsloth-bnb-4bit.yaml": [
        "unsloth/Magistral-Small-2509",
        "mistralai/Magistral-Small-2509",
        "unsloth/Magistral-Small-2509-bnb-4bit",
    ],
    "unsloth_Ministral-3-3B-Instruct-2512.yaml": [
        "unsloth/Ministral-3-3B-Instruct-2512",
    ],
    "unsloth_mistral-7b-v0.3-bnb-4bit.yaml": [
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-v0.3",
        "mistralai/Mistral-7B-v0.3",
    ],
    "unsloth_Mistral-Nemo-Base-2407-bnb-4bit.yaml": [
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Base-2407",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ],
    "unsloth_Mistral-Small-Instruct-2409.yaml": [
        "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "mistralai/Mistral-Small-Instruct-2409",
    ],
    "unsloth_mistral-7b-instruct-v0.3-bnb-4bit.yaml": [
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "unsloth_Qwen2.5-1.5B-Instruct.yaml": [
        "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    ],
    "unsloth_Nemotron-3-Nano-30B-A3B.yaml": [
        "unsloth/Nemotron-3-Nano-30B-A3B",
    ],
    "unsloth_orpheus-3b-0.1-ft.yaml": [
        "unsloth/orpheus-3b-0.1-ft",
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
        "canopylabs/orpheus-3b-0.1-ft",
        "unsloth/orpheus-3b-0.1-ft-bnb-4bit",
    ],
    "OuteAI_Llama-OuteTTS-1.0-1B.yaml": [
        "OuteAI/Llama-OuteTTS-1.0-1B",
        "unsloth/Llama-OuteTTS-1.0-1B",
        "unsloth/llama-outetts-1.0-1b",
        "OuteAI/OuteTTS-1.0-0.6B",
        "unsloth/OuteTTS-1.0-0.6B",
        "unsloth/outetts-1.0-0.6b",
    ],
    "unsloth_PaddleOCR-VL.yaml": [
        "unsloth/PaddleOCR-VL",
    ],
    "unsloth_Phi-3-medium-4k-instruct.yaml": [
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "microsoft/Phi-3-medium-4k-instruct",
    ],
    "unsloth_Phi-3.5-mini-instruct.yaml": [
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    "unsloth_Phi-4.yaml": [
        "unsloth/phi-4-unsloth-bnb-4bit",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
    ],
    "unsloth_Pixtral-12B-2409.yaml": [
        "unsloth/Pixtral-12B-2409-unsloth-bnb-4bit",
        "mistralai/Pixtral-12B-2409",
        "unsloth/Pixtral-12B-2409-bnb-4bit",
    ],
    "unsloth_Qwen2-7B.yaml": [
        "unsloth/Qwen2-7B-bnb-4bit",
        "Qwen/Qwen2-7B",
    ],
    "unsloth_Qwen2-VL-7B-Instruct.yaml": [
        "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen2-VL-7B-Instruct",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    ],
    "unsloth_Qwen2.5-7B.yaml": [
        "unsloth/Qwen2.5-7B-unsloth-bnb-4bit",
        "Qwen/Qwen2.5-7B",
        "unsloth/Qwen2.5-7B-bnb-4bit",
    ],
    "unsloth_Qwen2.5-Coder-1.5B-Instruct.yaml": [
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ],
    "unsloth_Qwen2.5-Coder-14B-Instruct.yaml": [
        "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ],
    "unsloth_Qwen2.5-VL-7B-Instruct-bnb-4bit.yaml": [
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    ],
    "unsloth_Qwen3-0.6B.yaml": [
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
        "Qwen/Qwen3-0.6B-FP8",
        "unsloth/Qwen3-0.6B-FP8",
    ],
    "unsloth_Qwen3-4B-Instruct-2507.yaml": [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "unsloth/Qwen3-4B-Instruct-2507-FP8",
    ],
    "unsloth_Qwen3-4B-Thinking-2507.yaml": [
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "unsloth/Qwen3-4B-Thinking-2507-FP8",
    ],
    "unsloth_Qwen3-14B-Base-unsloth-bnb-4bit.yaml": [
        "unsloth/Qwen3-14B-Base",
        "Qwen/Qwen3-14B-Base",
        "unsloth/Qwen3-14B-Base-bnb-4bit",
    ],
    "unsloth_Qwen3-14B.yaml": [
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
        "Qwen/Qwen3-14B-FP8",
        "unsloth/Qwen3-14B-FP8",
    ],
    "unsloth_Qwen3-32B.yaml": [
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "Qwen/Qwen3-32B-FP8",
        "unsloth/Qwen3-32B-FP8",
    ],
    "unsloth_Qwen3-VL-8B-Instruct-unsloth-bnb-4bit.yaml": [
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "unsloth/Qwen3-VL-8B-Instruct-FP8",
        "unsloth/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    ],
    "sesame_csm-1b.yaml": [
        "sesame/csm-1b",
        "unsloth/csm-1b",
    ],
    "Spark-TTS-0.5B_LLM.yaml": [
        "Spark-TTS-0.5B/LLM",
        "unsloth/Spark-TTS-0.5B",
    ],
    "unsloth_tinyllama-bnb-4bit.yaml": [
        "unsloth/tinyllama",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ],
    "unsloth_whisper-large-v3.yaml": [
        "unsloth/whisper-large-v3",
        "openai/whisper-large-v3",
    ],
}

# Reverse mapping for quick lookup: model_name -> canonical_filename
_REVERSE_MODEL_MAPPING = {}
for canonical_file, model_names in MODEL_NAME_MAPPING.items():
    for model_name in model_names:
        _REVERSE_MODEL_MAPPING[model_name.lower()] = canonical_file


def load_model_config(
    model_name: str,
    use_auth: bool = False,
    token: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Load model config with optional authentication control.
    """
    from transformers import AutoConfig

    if token:
        # Explicit token provided - use it
        return AutoConfig.from_pretrained(
            model_name, trust_remote_code = trust_remote_code, token = token
        )

    if not use_auth:
        # Load without any authentication (for public model checks)
        with without_hf_auth():
            return AutoConfig.from_pretrained(
                model_name,
                trust_remote_code = trust_remote_code,
                token = None,
            )

    # Use default authentication (cached tokens)
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code = trust_remote_code,
    )


# VLM architecture suffixes and known VLM model_type values.
_VLM_ARCH_SUFFIXES = ("ForConditionalGeneration", "ForVisionText2Text")
_VLM_MODEL_TYPES = {
    "phi3_v",
    "llava",
    "llava_next",
    "llava_onevision",
    "internvl_chat",
    "cogvlm2",
    "minicpmv",
}

# Pre-computed .venv_t5 paths and backend dir for subprocess version switching.
# Vision check uses 5.5.0 (newest, recognizes all architectures).
from utils.paths.storage_roots import studio_root as _studio_root  # noqa: E402

_VENV_T5_DIR = str(_studio_root() / ".venv_t5_550")
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent.parent)

# Inline script executed in a subprocess with transformers 5.x activated.
# Receives model_name and token via argv, prints JSON result to stdout.
_VISION_CHECK_SCRIPT = r"""
import sys, os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Activate transformers 5.x
venv_t5 = sys.argv[1]
backend_dir = sys.argv[2]
model_name = sys.argv[3]
token = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "" else None

sys.path.insert(0, venv_t5)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from transformers import AutoConfig
    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    config = AutoConfig.from_pretrained(model_name, **kwargs)

    is_vlm = False
    if hasattr(config, "architectures"):
        is_vlm = any(
            x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
            for x in config.architectures
        )
    if not is_vlm and hasattr(config, "vision_config"):
        is_vlm = True
    if not is_vlm and hasattr(config, "img_processor"):
        is_vlm = True
    if not is_vlm and hasattr(config, "image_token_index"):
        is_vlm = True
    if not is_vlm and hasattr(config, "model_type"):
        vlm_types = {"phi3_v","llava","llava_next","llava_onevision",
                      "internvl_chat","cogvlm2","minicpmv"}
        if config.model_type in vlm_types:
            is_vlm = True

    model_type = getattr(config, "model_type", "unknown")
    archs = getattr(config, "architectures", [])
    print(json.dumps({"is_vision": is_vlm, "model_type": model_type,
                       "architectures": archs}))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(1)
"""


def _is_vision_model_subprocess(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[bool]:
    """Run is_vision_model check in a subprocess with transformers 5.x.

    Same pattern as training/inference workers: spawn a clean subprocess
    with .venv_t5/ prepended to sys.path so AutoConfig recognizes newer
    architectures (glm4_moe_lite, etc.).

    Returns True/False for definitive results, or None for transient failures
    (timeouts, subprocess errors) so callers can decide whether to cache
    the result. Subprocess failures are treated as transient because they
    can be caused by temporary HF/auth/network issues.
    """
    token_arg = hf_token or ""

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _VISION_CHECK_SCRIPT,
                _VENV_T5_DIR,
                _BACKEND_DIR,
                model_name,
                token_arg,
            ],
            capture_output = True,
            text = True,
            timeout = 60,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            logger.warning(
                "Vision check subprocess failed for '%s': %s",
                model_name,
                stderr or result.stdout.strip(),
            )
            return None

        data = json.loads(result.stdout.strip())
        if "error" in data:
            logger.warning(
                "Vision check subprocess error for '%s': %s",
                model_name,
                data["error"],
            )
            return None

        is_vlm = data["is_vision"]
        logger.info(
            "Vision check (subprocess, transformers 5.x) for '%s': "
            "model_type=%s, architectures=%s, is_vision=%s",
            model_name,
            data.get("model_type"),
            data.get("architectures"),
            is_vlm,
        )
        return is_vlm

    except subprocess.TimeoutExpired:
        logger.warning("Vision check subprocess timed out for '%s'", model_name)
        return None
    except Exception as exc:
        logger.warning("Vision check subprocess failed for '%s': %s", model_name, exc)
        return None


def _token_fingerprint(token: Optional[str]) -> Optional[str]:
    """Return a SHA256 digest of the token for use as a cache key.

    Avoids storing the raw bearer token in process memory as a dict key.
    """
    if token is None:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Cache vision detection results per session to avoid repeated subprocess spawns.
# Keyed by (normalized_model_name, token_fingerprint) to handle gated models correctly.
# Only definitive results (True/False from successful detection) are cached;
# transient failures (network errors, timeouts) are NOT cached so they can be retried.
_vision_detection_cache: Dict[Tuple[str, Optional[str]], bool] = {}
_vision_cache_lock = threading.Lock()


def is_vision_model(model_name: str, hf_token: Optional[str] = None) -> bool:
    """
    Detect vision-language models (VLMs) by checking architecture in config.
    Works for fine-tuned models since they inherit the base architecture.

    For models that require transformers 5.x (e.g. GLM-4.7-Flash), the check
    runs in a subprocess with .venv_t5/ activated -- same pattern as the
    training and inference workers.

    Results are cached per (model_name, token_fingerprint) for the lifetime of
    the process to avoid repeated subprocess spawns and HuggingFace API calls.
    Transient failures are not cached so they can be retried on the next call.

    Args:
        model_name: Model identifier (HF repo or local path)
        hf_token: Optional HF token for accessing gated/private models
    """
    # Normalize model name for cache key to avoid duplicate entries for
    # different casings of the same HF repo (e.g. "Org/Model" vs "org/model").
    try:
        if is_local_path(model_name):
            resolved_name = normalize_path(model_name)
        else:
            resolved_name = resolve_cached_repo_id_case(model_name)
    except Exception as exc:
        logger.debug(
            "Could not normalize model name '%s' for cache key: %s",
            model_name,
            exc,
        )
        resolved_name = model_name
    cache_key = (resolved_name, _token_fingerprint(hf_token))

    # Lock-free fast path for cache hits. Uses a sentinel to distinguish
    # "key not found" from "value is False" in a single atomic dict.get() call.
    _MISS = object()
    cached = _vision_detection_cache.get(cache_key, _MISS)
    if cached is not _MISS:
        return cached

    # Compute outside the lock to avoid serializing long-running detection
    # (subprocess spawns with 60s timeout, HF API calls) across all models.
    # The tradeoff: two concurrent calls for the same uncached model may
    # both run detection, but they produce the same result and the second
    # write is a benign no-op.
    result = _is_vision_model_uncached(resolved_name, hf_token)
    # Only cache definitive results; None means a transient failure occurred
    # and we should retry on the next call instead of locking in a wrong answer.
    if result is not None:
        with _vision_cache_lock:
            _vision_detection_cache[cache_key] = result
        return result
    return False


def _is_vision_model_uncached(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[bool]:
    """Uncached vision model detection -- called by is_vision_model().

    Returns True/False for definitive results, or None when detection failed
    due to a transient error (network, timeout, subprocess failure) so the
    caller knows not to cache the result.

    Do not call directly; use is_vision_model() instead.
    """
    # Models that need transformers 5.x must be checked in a subprocess
    # because AutoConfig in the main process (transformers 4.57.x) doesn't
    # recognize their architectures.
    from utils.transformers_version import needs_transformers_5

    if needs_transformers_5(model_name):
        logger.info(
            "Model '%s' needs transformers 5.x -- checking vision via subprocess",
            model_name,
        )
        return _is_vision_model_subprocess(model_name, hf_token = hf_token)

    try:
        config = load_model_config(model_name, use_auth = True, token = hf_token)

        # Exclude audio-only models that share ForConditionalGeneration suffix
        # (e.g. CsmForConditionalGeneration, WhisperForConditionalGeneration)
        _audio_only_model_types = {"csm", "whisper"}
        model_type = getattr(config, "model_type", None)
        if model_type in _audio_only_model_types:
            return False

        # Check 1: Architecture class name patterns
        if hasattr(config, "architectures"):
            is_vlm = any(x.endswith(_VLM_ARCH_SUFFIXES) for x in config.architectures)
            if is_vlm:
                logger.info(
                    f"Model {model_name} detected as VLM: architecture {config.architectures}"
                )
                return True

        # Check 2: Has vision_config (most VLMs: LLaVA, Gemma-3, Qwen2-VL, etc.)
        if hasattr(config, "vision_config"):
            logger.info(f"Model {model_name} detected as VLM: has vision_config")
            return True

        # Check 3: Has img_processor (Phi-3.5 Vision uses this instead of vision_config)
        if hasattr(config, "img_processor"):
            logger.info(f"Model {model_name} detected as VLM: has img_processor")
            return True

        # Check 4: Has image_token_index (common in VLMs for image placeholder tokens)
        if hasattr(config, "image_token_index"):
            logger.info(f"Model {model_name} detected as VLM: has image_token_index")
            return True

        # Check 5: Known VLM model_type values that may not match above checks
        if hasattr(config, "model_type"):
            if config.model_type in _VLM_MODEL_TYPES:
                logger.info(
                    f"Model {model_name} detected as VLM: model_type={config.model_type}"
                )
                return True

        return False

    except Exception as e:
        logger.warning(f"Could not determine if {model_name} is vision model: {e}")
        # Permanent failures (model not found, gated, bad config) should be
        # cached as False. Transient failures (network, timeout) should not.
        try:
            from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
        except ImportError:
            try:
                from huggingface_hub.utils import (
                    RepositoryNotFoundError,
                    GatedRepoError,
                )
            except ImportError:
                RepositoryNotFoundError = GatedRepoError = None
        if RepositoryNotFoundError is not None and isinstance(
            e, (RepositoryNotFoundError, GatedRepoError)
        ):
            return False
        if isinstance(e, (ValueError, json.JSONDecodeError)):
            return False
        return None


VALID_AUDIO_TYPES = ("snac", "csm", "bicodec", "dac", "whisper", "audio_vlm")

# Cache detection results per session to avoid repeated API calls
_audio_detection_cache: Dict[str, Optional[str]] = {}

# Tokenizer token patterns → audio_type (all 6 types detected from tokenizer_config.json)
_AUDIO_TOKEN_PATTERNS = {
    "csm": lambda tokens: "<|AUDIO|>" in tokens and "<|audio_eos|>" in tokens,
    "whisper": lambda tokens: "<|startoftranscript|>" in tokens,
    "audio_vlm": lambda tokens: "<audio_soft_token>" in tokens,
    "bicodec": lambda tokens: any(t.startswith("<|bicodec_") for t in tokens),
    "dac": lambda tokens: "<|audio_start|>" in tokens
    and "<|audio_end|>" in tokens
    and "<|text_start|>" in tokens
    and "<|text_end|>" in tokens,
    "snac": lambda tokens: sum(1 for t in tokens if t.startswith("<custom_token_"))
    > 10000,
}


def detect_audio_type(model_name: str, hf_token: Optional[str] = None) -> Optional[str]:
    """
    Dynamically detect if a model is an audio model and return its type.

    Fully dynamic — works for any model, not just known ones.
    Uses tokenizer_config.json special tokens to detect all 6 audio types.

    Returns: audio_type string ('snac', 'csm', 'bicodec', 'dac', 'whisper', 'audio_vlm') or None.
    """
    if model_name in _audio_detection_cache:
        return _audio_detection_cache[model_name]

    result = _detect_audio_from_tokenizer(model_name, hf_token)

    _audio_detection_cache[model_name] = result
    if result:
        logger.info(f"Model {model_name} detected as audio model: audio_type={result}")
    return result


def _detect_audio_from_tokenizer(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[str]:
    """Detect audio type from tokenizer special tokens (for LLM-based audio models).

    First checks local HF cache, then fetches tokenizer_config.json from HuggingFace.
    Checks added_tokens_decoder for distinctive patterns.
    """

    def _check_token_patterns(tok_config: dict) -> Optional[str]:
        added = tok_config.get("added_tokens_decoder", {})
        if not added:
            return None
        token_contents = [v.get("content", "") for v in added.values()]
        for audio_type, check_fn in _AUDIO_TOKEN_PATTERNS.items():
            if check_fn(token_contents):
                return audio_type
        return None

    # 1) Check local HF cache first (works for gated/offline models)
    try:
        repo_dir = get_cache_path(model_name)
        if repo_dir is not None and repo_dir.exists():
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    for tok_path in [
                        "tokenizer_config.json",
                        "LLM/tokenizer_config.json",
                    ]:
                        tok_file = snapshot / tok_path
                        if tok_file.exists():
                            tok_config = json.loads(tok_file.read_text())
                            result = _check_token_patterns(tok_config)
                            if result:
                                return result
    except Exception as e:
        logger.debug(f"Could not check local cache for {model_name}: {e}")

    # 2) Fall back to HuggingFace API
    try:
        import requests
        import os

        paths_to_try = ["tokenizer_config.json", "LLM/tokenizer_config.json"]
        # Use provided token, or fall back to env
        token = hf_token or os.environ.get("HF_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        for tok_path in paths_to_try:
            url = f"https://huggingface.co/{model_name}/resolve/main/{tok_path}"
            resp = requests.get(url, headers = headers, timeout = 15)
            if not resp.ok:
                continue

            tok_config = resp.json()
            result = _check_token_patterns(tok_config)
            if result:
                return result

        return None
    except Exception as e:
        logger.debug(
            f"Could not detect audio type from tokenizer for {model_name}: {e}"
        )
        return None


def is_audio_input_type(audio_type: Optional[str]) -> bool:
    """Check if an audio_type accepts audio input (ASR/speech understanding).

    Whisper (ASR) and audio_vlm (Gemma3n) accept audio input.
    """
    return audio_type in ("whisper", "audio_vlm")


def _is_mmproj(filename: str) -> bool:
    """Check if a GGUF filename is a vision projection (mmproj) file."""
    return "mmproj" in filename.lower()


def _is_gguf_filename(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


def _iter_gguf_files(directory: Path, recursive: bool = False):
    if not directory.is_dir():
        return
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for f in iterator:
        if f.is_file() and _is_gguf_filename(f.name):
            yield f


def detect_mmproj_file(path: str, search_root: Optional[str] = None) -> Optional[str]:
    """
    Find the mmproj (vision projection) GGUF file for a given model.

    Args:
        path: Directory to search — or a .gguf file (uses its parent dir
            as the starting point).
        search_root: Optional outer directory that should also be scanned
            (and any directory between it and ``path``). This handles
            local layouts where the model weights live in a quant-named
            subdir (``snapshot/BF16/foo.gguf``) but the mmproj sits at
            the snapshot root (``snapshot/mmproj-BF16.gguf``). When
            ``None``, only the immediate parent dir is scanned, matching
            the historical behavior.

    Returns:
        Full path to the mmproj .gguf file, or None if not found.
    """
    p = Path(path)
    start_dir = p.parent if p.is_file() else p
    if not start_dir.is_dir():
        return None

    # Build the list of dirs to scan: immediate dir first, then walk up
    # to (and including) ``search_root`` if it is an ancestor. We walk
    # incrementally rather than recursing into ``search_root`` so we
    # don't accidentally pick up an mmproj from a sibling subdir
    # belonging to a different model variant.
    seen: set[Path] = set()
    scan_order: list[Path] = []

    def _add(d: Path) -> None:
        try:
            resolved = d.resolve()
        except OSError:
            return
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        scan_order.append(resolved)

    _add(start_dir)

    # When ``path`` is a symlink (e.g. Ollama's ``.studio_links/...gguf``
    # -> ``blobs/sha256-...``), the symlink's parent directory rarely
    # contains the mmproj sibling; the real mmproj file lives next to
    # the symlink target. Add the target's parent to the scan so vision
    # GGUFs that are surfaced via symlinks are still recognised as
    # vision models.
    try:
        if p.is_symlink() and p.is_file():
            target_parent = p.resolve().parent
            if target_parent.is_dir():
                _add(target_parent)
    except OSError:
        pass
    if search_root is not None:
        try:
            root_resolved = Path(search_root).resolve()
            start_resolved = start_dir.resolve()
            # Only walk if start_dir is inside (or equal to) search_root.
            if root_resolved == start_resolved or (
                start_resolved.is_relative_to(root_resolved)
                if hasattr(start_resolved, "is_relative_to")
                else str(start_resolved).startswith(str(root_resolved) + "/")
            ):
                cur = start_resolved
                # Walk up from start_dir to (and including) root_resolved.
                while cur != root_resolved and cur.parent != cur:
                    cur = cur.parent
                    _add(cur)
                    if cur == root_resolved:
                        break
        except OSError:
            pass

    for d in scan_order:
        for f in _iter_gguf_files(d):
            if _is_mmproj(f.name):
                return str(f.resolve())
    return None


def detect_gguf_model(path: str) -> Optional[str]:
    """
    Check if the given local path is or contains a GGUF model file.

    Handles two cases:
    1. path is a direct .gguf file path
    2. path is a directory containing .gguf files

    Skips mmproj (vision projection) files — those must be passed via
    ``--mmproj``, not ``-m``.  Use :func:`detect_mmproj_file` instead.

    Returns the full path to the .gguf file if found, None otherwise.
    For HuggingFace repo detection, use detect_gguf_model_remote() instead.
    """
    p = Path(path)

    # Case 1: direct .gguf file
    if p.suffix.lower() == ".gguf" and p.is_file():
        if _is_mmproj(p.name):
            return None
        # Use absolute (not resolve) to preserve symlink names -- e.g.
        # Ollama .studio_links/model.gguf -> blobs/sha256-... should
        # keep the readable symlink name, not the opaque blob hash.
        return str(p.absolute())

    # Case 2: directory containing .gguf files (skip mmproj)
    if p.is_dir():
        gguf_files = sorted(
            (f for f in _iter_gguf_files(p) if not _is_mmproj(f.name)),
            key = lambda f: f.stat().st_size,
            reverse = True,
        )
        if gguf_files:
            return str(gguf_files[0].resolve())

    return None


# Preferred GGUF quantization levels, in descending priority.
# Q4_K_M is a good default: small, fast, acceptable quality.
# UD (Unsloth Dynamic) variants are always preferred over standard quants
# because they provide better quality per bit. If the repo has no UD variants
# (e.g., bartowski repos), the standard quants are used as fallback.
# Ordered by best size/quality tradeoff, not raw quality.
_GGUF_QUANT_PREFERENCE = [
    # UD variants (best quality per bit) -- Q4 is the sweet spot
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "UD-Q5_K_XL",
    "UD-Q3_K_XL",
    "UD-Q6_K_XL",
    "UD-Q6_K_S",
    "UD-Q8_K_XL",
    "UD-Q2_K_XL",
    "UD-IQ4_NL",
    "UD-IQ4_XS",
    "UD-IQ3_S",
    "UD-IQ3_XXS",
    "UD-IQ2_M",
    "UD-IQ2_XXS",
    "UD-IQ1_M",
    "UD-IQ1_S",
    # Standard quants (fallback for non-Unsloth repos)
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_S",
    "Q2_K",
    "Q2_K_L",
    "IQ4_NL",
    "IQ4_XS",
    "IQ3_M",
    "IQ3_XXS",
    "IQ2_M",
    "IQ1_M",
    "F16",
    "BF16",
    "F32",
]


def _pick_best_gguf(filenames: list[str]) -> Optional[str]:
    """
    Pick the best GGUF file from a list of filenames.

    Prefers quantization levels in _GGUF_QUANT_PREFERENCE order.
    Falls back to the first .gguf file found.
    """
    gguf_files = [f for f in filenames if f.lower().endswith(".gguf")]
    if not gguf_files:
        return None

    # Try preferred quantization levels
    for quant in _GGUF_QUANT_PREFERENCE:
        for f in gguf_files:
            if quant in f:
                return f

    # Fallback: first GGUF file
    return gguf_files[0]


@dataclass
class GgufVariantInfo:
    """A single GGUF quantization variant from a HuggingFace repo."""

    filename: str  # e.g., "gemma-3-4b-it-Q4_K_M.gguf"
    quant: str  # e.g., "Q4_K_M" (extracted from filename)
    size_bytes: int  # file size


def _extract_quant_label(filename: str) -> str:
    """
    Extract quantization label like Q4_K_M, IQ4_XS, BF16 from a GGUF filename.

    Examples:
        "gemma-3-4b-it-Q4_K_M.gguf"          → "Q4_K_M"
        "model-IQ4_NL.gguf"                   → "IQ4_NL"
        "model-BF16.gguf"                     → "BF16"
        "model-UD-IQ1_S.gguf"                 → "UD-IQ1_S"
        "model-UD-TQ1_0.gguf"                 → "UD-TQ1_0"
        "MXFP4_MOE/model-MXFP4_MOE-0001.gguf"→ "MXFP4_MOE"
    """
    import re

    # Use only the basename (rfilename may include directory)
    basename = filename.rsplit("/", 1)[-1]
    # Strip .gguf and any shard suffix (-00001-of-00010)
    stem = re.sub(r"-\d{3,}-of-\d{3,}", "", basename.rsplit(".", 1)[0])
    # Match known quantization patterns
    match = re.search(
        r"(UD-)?"  # Optional UD- prefix (Ultra Discrete)
        r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"  # MXFP variants: MXFP4, MXFP4_MOE
        r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"  # IQ variants: IQ4_XS, IQ4_NL, IQ1_S
        r"|TQ[0-9]+_[0-9]+"  # Ternary quant: TQ1_0, TQ2_0
        r"|Q[0-9]+_K_[A-Z]+"  # K-quant: Q4_K_M, Q3_K_S
        r"|Q[0-9]+_[0-9]+"  # Standard: Q8_0, Q5_1
        r"|Q[0-9]+_K"  # Short K-quant: Q6_K
        r"|BF16|F16|F32)",  # Full precision
        stem,
        re.IGNORECASE,
    )
    if match:
        prefix = match.group(1) or ""
        return f"{prefix}{match.group(2)}"
    # Fallback: last segment after hyphen
    return stem.split("-")[-1]


def list_gguf_variants(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> tuple[list[GgufVariantInfo], bool]:
    """
    List all GGUF quantization variants in a HuggingFace repo.

    Separates main model files from mmproj (vision projection) files.
    The presence of mmproj files indicates a vision-capable model.

    Returns:
        (variants, has_vision): list of non-mmproj GGUF variants + vision flag.
    """
    from huggingface_hub import model_info as hf_model_info

    info = hf_model_info(repo_id, token = hf_token, files_metadata = True)
    variants: list[GgufVariantInfo] = []
    has_vision = False

    quant_totals: dict[str, int] = {}  # quant -> total bytes
    quant_first_file: dict[str, str] = {}  # quant -> first filename (for display)

    for sibling in info.siblings:
        fname = sibling.rfilename
        if not fname.lower().endswith(".gguf"):
            continue
        size = sibling.size or 0

        # mmproj files are vision projection models, not main model files
        if "mmproj" in fname.lower():
            has_vision = True
            continue

        quant = _extract_quant_label(fname)
        quant_totals[quant] = quant_totals.get(quant, 0) + size
        if quant not in quant_first_file:
            quant_first_file[quant] = fname

    for quant, total_size in quant_totals.items():
        variants.append(
            GgufVariantInfo(
                filename = quant_first_file[quant],
                quant = quant,
                size_bytes = total_size,
            )
        )

    # Sort by size descending (largest = best quality first).
    # Recommended pinning and OOM demotion are handled client-side
    # where GPU VRAM info is available.
    variants.sort(key = lambda v: -v.size_bytes)

    return variants, has_vision


def _resolve_gguf_dir(p: Path) -> Optional[Path]:
    """Resolve a path to the directory containing GGUF variants.

    If *p* is already a directory, returns it directly.  If *p* is a ``.gguf``
    file whose parent directory has model metadata (``config.json`` or
    ``adapter_config.json``), returns the parent -- all GGUFs in that
    directory belong to the same model.  Returns ``None`` for loose standalone
    GGUFs (no config) to avoid cross-wiring unrelated models.
    """
    if p.is_dir():
        return p
    if p.is_file() and p.suffix.lower() == ".gguf":
        parent = p.parent
        if (
            (parent / "config.json").exists()
            or (parent / "adapter_config.json").exists()
            or (parent / "export_metadata.json").exists()
        ):
            return parent
    return None


def list_local_gguf_variants(
    directory: str,
) -> tuple[list[GgufVariantInfo], bool]:
    """List GGUF quantization variants in a local directory.

    Mirrors :func:`list_gguf_variants` but reads from the filesystem
    instead of the HuggingFace API.  Aggregates shard sizes by quant
    label so that split GGUFs appear as a single variant.

    Returns:
        (variants, has_vision): list of non-mmproj GGUF variants + vision flag.
    """
    p = _resolve_gguf_dir(Path(directory))
    if p is None:
        return [], False

    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}
    has_vision = False

    # Recurse so variant-specific subdirectories (e.g. ``BF16/...gguf``
    # used by some HF GGUF repos for the largest quants) are picked up.
    # Filenames in the result preserve the relative subpath so that
    # ``_find_local_gguf_by_variant`` can locate the file again.
    for f in sorted(_iter_gguf_files(p, recursive = True)):
        if _is_mmproj(f.name):
            has_vision = True
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        quant = _extract_quant_label(f.name)
        quant_totals[quant] = quant_totals.get(quant, 0) + size
        # Only compute the (potentially expensive) relative path when this
        # is the first file we've seen for this quant -- after that we'd
        # discard the result anyway. Use posix-style separators so the
        # filename matches what ``list_gguf_variants`` (the remote HF
        # API path) returns on every platform; otherwise Windows would
        # emit ``BF16\foo.gguf`` here.
        if quant not in quant_first_file:
            quant_first_file[quant] = f.relative_to(p).as_posix()

    variants = [
        GgufVariantInfo(
            filename = quant_first_file[q],
            quant = q,
            size_bytes = s,
        )
        for q, s in quant_totals.items()
    ]
    variants.sort(key = lambda v: -v.size_bytes)
    return variants, has_vision


def _find_local_gguf_by_variant(directory: str, variant: str) -> Optional[str]:
    """Find the GGUF file in *directory* matching a quantization *variant*.

    For sharded GGUFs (multiple files with the same quant label), returns
    the first shard (sorted by name) which is what ``llama-server -m`` expects.

    Returns the resolved absolute path, or ``None`` if no match.
    """
    p = _resolve_gguf_dir(Path(directory))
    if p is None:
        return None

    # Recurse into subdirectories so variants stored under a quant-named
    # subdir (e.g. ``BF16/foo-BF16-00001-of-00002.gguf``) are found.
    matches = sorted(
        f
        for f in _iter_gguf_files(p, recursive = True)
        if not _is_mmproj(f.name) and _extract_quant_label(f.name) == variant
    )
    if matches:
        return str(matches[0].resolve())
    return None


def detect_gguf_model_remote(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """
    Check if a HuggingFace repo contains GGUF files.

    Returns the filename of the best GGUF file in the repo, or None.

    Retries on transient HF Hub failures (network hiccups, 5xx, slow
    cold-start of the API). Without retry, a single transient failure
    here returns None silently and the caller treats the repo as
    non-GGUF -- which on Apple Silicon (Mac UI route) means falling
    through to the MLX backend, which then fails opening a non-existent
    config.json on the GGUF-only repo. Three attempts with 1s/2s/4s
    backoff covers the typical free-runner HF Hub flakiness.
    """
    import time
    from huggingface_hub import model_info as hf_model_info

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            info = hf_model_info(repo_id, token = hf_token)
            repo_files = [s.rfilename for s in info.siblings]
            return _pick_best_gguf(repo_files)
        except Exception as e:
            last_err = e
            # 404 / RepoNotFound is permanent -- don't waste attempts.
            err_name = type(e).__name__
            if err_name in (
                "RepositoryNotFoundError",
                "GatedRepoError",
                "RevisionNotFoundError",
                "EntryNotFoundError",
            ):
                logger.debug(f"Could not check GGUF files for '{repo_id}': {e}")
                return None
            if attempt < 2:
                time.sleep(2**attempt)
    logger.warning(
        f"Could not check GGUF files for '{repo_id}' after 3 attempts: " f"{last_err}"
    )
    return None


def download_gguf_file(
    repo_id: str,
    filename: str,
    hf_token: Optional[str] = None,
) -> str:
    """
    Download a specific GGUF file from a HuggingFace repo.

    Returns the local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id = repo_id,
        filename = filename,
        token = hf_token,
    )
    return local_path


# Cache embedding detection results per session to avoid repeated HF API calls
_embedding_detection_cache: Dict[tuple, bool] = {}


def is_embedding_model(model_name: str, hf_token: Optional[str] = None) -> bool:
    """
    Detect embedding/sentence-transformer models using HuggingFace model metadata.

    Uses a belt-and-suspenders approach combining three signals:
      1. "sentence-transformers" in model tags
      2. "feature-extraction" in model tags
      3. pipeline_tag is "sentence-similarity" or "feature-extraction"

    This catches all known embedding models including those like gte-modernbert
    whose library_name is "transformers" rather than "sentence-transformers".

    Args:
        model_name: Model identifier (HF repo or local path)
        hf_token: Optional HF token for accessing gated/private models

    Returns:
        True if the model is an embedding model, False otherwise.
        Defaults to False for local paths or on errors.
    """
    cache_key = (model_name, hf_token)
    if cache_key in _embedding_detection_cache:
        return _embedding_detection_cache[cache_key]

    # Local paths: check for sentence-transformer marker file (modules.json)
    if is_local_path(model_name):
        local_dir = normalize_path(model_name)
        is_emb = os.path.isfile(os.path.join(local_dir, "modules.json"))
        _embedding_detection_cache[cache_key] = is_emb
        return is_emb

    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(model_name, token = hf_token)
        tags = set(info.tags or [])
        pipeline_tag = info.pipeline_tag or ""

        is_emb = (
            "sentence-transformers" in tags
            or "feature-extraction" in tags
            or pipeline_tag in ("sentence-similarity", "feature-extraction")
        )

        _embedding_detection_cache[cache_key] = is_emb
        if is_emb:
            logger.info(
                f"Model {model_name} detected as embedding model: "
                f"pipeline_tag={pipeline_tag}, "
                f"sentence-transformers in tags={('sentence-transformers' in tags)}, "
                f"feature-extraction in tags={('feature-extraction' in tags)}"
            )
        return is_emb

    except Exception as e:
        logger.warning(f"Could not determine if {model_name} is embedding model: {e}")
        _embedding_detection_cache[cache_key] = False
        return False


def _has_model_weight_files(model_dir: Path) -> bool:
    """Return True when a directory contains loadable model weights."""
    for item in model_dir.iterdir():
        if not item.is_file():
            continue

        suffix = item.suffix.lower()
        if suffix == ".safetensors":
            return True
        if suffix == ".gguf":
            return "mmproj" not in item.name.lower()
        if suffix == ".bin":
            name = item.name.lower()
            if (
                name.startswith("pytorch_model")
                or name.startswith("model")
                or name.startswith("adapter_model")
                or name.startswith("consolidated")
            ):
                return True
    return False


def _detect_training_output_type(model_dir: Path) -> Optional[str]:
    """Classify a Studio training output as LoRA or full finetune."""
    adapter_config = model_dir / "adapter_config.json"
    adapter_model = model_dir / "adapter_model.safetensors"
    if adapter_config.exists() or adapter_model.exists():
        return "lora"

    config_file = model_dir / "config.json"
    if config_file.exists() and _has_model_weight_files(model_dir):
        return "merged"

    return None


def _looks_like_lora_adapter(model_dir: Path) -> bool:
    return model_dir.is_dir() and (
        (model_dir / "adapter_config.json").exists()
        or any(model_dir.glob("adapter_model*.safetensors"))
        or any(model_dir.glob("adapter_model*.bin"))
    )


def scan_trained_models(
    outputs_dir: str = str(outputs_root()),
) -> List[Tuple[str, str, str]]:
    """
    Scan outputs folder for trained Studio models.

    Returns:
        List of tuples: [(display_name, model_path, model_type), ...]
        model_type is "lora" for adapter runs and "merged" for full finetunes.
    """
    trained_models = []
    outputs_path = resolve_output_dir(outputs_dir)

    if not outputs_path.exists():
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return trained_models

    try:
        for item in outputs_path.iterdir():
            if item.is_dir():
                model_type = _detect_training_output_type(item)
                if model_type is None:
                    continue

                display_name = item.name
                model_path = str(item)
                trained_models.append((display_name, model_path, model_type))
                logger.debug("Found trained model: %s (%s)", display_name, model_type)

        # Sort by modification time (newest first)
        trained_models.sort(key = lambda x: Path(x[1]).stat().st_mtime, reverse = True)

        logger.info(
            "Found %s trained models in %s",
            len(trained_models),
            outputs_dir,
        )
        return trained_models

    except Exception as e:
        logger.error(f"Error scanning outputs folder: {e}")
        return []


def scan_exported_models(
    exports_dir: str = str(exports_root()),
) -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Scan exports folder for exported models (merged, LoRA, GGUF).

    Supports two directory layouts:
      - Two-level: {run}/{checkpoint}/  (merged & LoRA exports)
      - Flat:      {name}-finetune-gguf/  (GGUF exports)

    Returns:
        List of tuples: [(display_name, model_path, export_type, base_model), ...]
        export_type: "lora" | "merged" | "gguf"
    """
    results = []
    exports_path = resolve_export_dir(exports_dir)

    if not exports_path.exists():
        return results

    try:
        for run_dir in exports_path.iterdir():
            if not run_dir.is_dir():
                continue

            # Check for flat GGUF export (e.g. exports/gemma-3-4b-it-finetune-gguf/)
            # Filter out mmproj (vision projection) files — they aren't loadable as main models
            gguf_files = [
                f for f in _iter_gguf_files(run_dir) if not _is_mmproj(f.name)
            ]
            if gguf_files:
                base_model = None
                export_meta = run_dir / "export_metadata.json"
                try:
                    if export_meta.exists():
                        meta = json.loads(export_meta.read_text())
                        base_model = meta.get("base_model")
                except Exception:
                    pass

                display_name = run_dir.name
                model_path = str(gguf_files[0])  # path to the .gguf file
                results.append((display_name, model_path, "gguf", base_model))
                logger.debug(f"Found GGUF export: {display_name}")
                continue

            # Two-level: {run}/{checkpoint}/
            for checkpoint_dir in run_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                adapter_config = checkpoint_dir / "adapter_config.json"
                config_file = checkpoint_dir / "config.json"
                has_weights = any(checkpoint_dir.glob("*.safetensors")) or any(
                    checkpoint_dir.glob("*.bin")
                )
                has_gguf = any(_iter_gguf_files(checkpoint_dir))

                base_model = None
                export_type = None

                if adapter_config.exists():
                    export_type = "lora"
                    try:
                        cfg = json.loads(adapter_config.read_text())
                        base_model = cfg.get("base_model_name_or_path")
                    except Exception:
                        pass
                elif config_file.exists() and has_weights:
                    export_type = "merged"
                    export_meta = checkpoint_dir / "export_metadata.json"
                    try:
                        if export_meta.exists():
                            meta = json.loads(export_meta.read_text())
                            base_model = meta.get("base_model")
                    except Exception:
                        pass
                elif has_gguf:
                    export_type = "gguf"
                    gguf_list = list(_iter_gguf_files(checkpoint_dir))
                    # Check checkpoint_dir first, then fall back to parent run_dir
                    # (export.py writes metadata to the top-level export directory)
                    for meta_dir in (checkpoint_dir, run_dir):
                        export_meta = meta_dir / "export_metadata.json"
                        try:
                            if export_meta.exists():
                                meta = json.loads(export_meta.read_text())
                                base_model = meta.get("base_model")
                                if base_model:
                                    break
                        except Exception:
                            pass

                    display_name = f"{run_dir.name} / {checkpoint_dir.name}"
                    model_path = str(gguf_list[0]) if gguf_list else str(checkpoint_dir)
                    results.append((display_name, model_path, export_type, base_model))
                    logger.debug(f"Found GGUF export: {display_name}")
                    continue
                else:
                    continue

                # Fallback: read base model from the original training run's
                # adapter_config.json in ./outputs/{run_name}/
                if not base_model:
                    outputs_adapter_cfg = (
                        resolve_output_dir(run_dir.name) / "adapter_config.json"
                    )
                    try:
                        if outputs_adapter_cfg.exists():
                            cfg = json.loads(outputs_adapter_cfg.read_text())
                            base_model = cfg.get("base_model_name_or_path")
                    except Exception:
                        pass

                display_name = f"{run_dir.name} / {checkpoint_dir.name}"
                model_path = str(checkpoint_dir)
                results.append((display_name, model_path, export_type, base_model))
                logger.debug(f"Found exported model: {display_name} ({export_type})")

        results.sort(key = lambda x: Path(x[1]).stat().st_mtime, reverse = True)
        logger.info(f"Found {len(results)} exported models in {exports_dir}")
        return results

    except Exception as e:
        logger.error(f"Error scanning exports folder: {e}")
        return []


def get_base_model_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Read the base model name from a local training or checkpoint directory."""
    try:
        checkpoint_path_obj = Path(checkpoint_path)

        adapter_config_path = checkpoint_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    logger.info(
                        "Detected base model from adapter_config.json: %s", base_model
                    )
                    return base_model

        config_path = checkpoint_path_obj / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                for key in ("model_name", "_name_or_path"):
                    base_model = config.get(key)
                    if base_model and str(base_model) != str(checkpoint_path_obj):
                        logger.info(
                            "Detected base model from config.json (%s): %s",
                            key,
                            base_model,
                        )
                        return base_model

        training_args_path = checkpoint_path_obj / "training_args.bin"
        if training_args_path.exists():
            try:
                import torch

                training_args = torch.load(training_args_path)
                if hasattr(training_args, "model_name_or_path"):
                    base_model = training_args.model_name_or_path
                    logger.info(
                        "Detected base model from training_args.bin: %s", base_model
                    )
                    return base_model
            except Exception as e:
                logger.warning(f"Could not load training_args.bin: {e}")

        dir_name = checkpoint_path_obj.name
        if dir_name.startswith("unsloth_"):
            parts = dir_name.split("_")
            if len(parts) >= 2:
                model_parts = parts[1:-1]
                base_model = "unsloth/" + "_".join(model_parts)
                logger.info("Detected base model from directory name: %s", base_model)
                return base_model

        logger.warning(f"Could not detect base model for checkpoint: {checkpoint_path}")
        return None

    except Exception as e:
        logger.error(f"Error reading base model from checkpoint config: {e}")
        return None


def get_base_model_from_lora(lora_path: str) -> Optional[str]:
    """
    Read the base model name from a LoRA adapter's config.

    Args:
        lora_path: Path to the LoRA adapter directory

    Returns:
        Base model identifier or None if not found
    """
    try:
        lora_path_obj = Path(lora_path)

        if not _looks_like_lora_adapter(lora_path_obj):
            return None

        # Try adapter_config.json first
        adapter_config_path = lora_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    logger.info(
                        f"Detected base model from adapter_config.json: {base_model}"
                    )
                    return base_model

        # Fallback: try training_args.bin (requires torch)
        training_args_path = lora_path_obj / "training_args.bin"
        if training_args_path.exists():
            try:
                import torch

                training_args = torch.load(training_args_path)
                if hasattr(training_args, "model_name_or_path"):
                    base_model = training_args.model_name_or_path
                    logger.info(
                        f"Detected base model from training_args.bin: {base_model}"
                    )
                    return base_model
            except Exception as e:
                logger.warning(f"Could not load training_args.bin: {e}")

        # Last resort: parse from directory name
        # Format: unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit_timestamp
        dir_name = lora_path_obj.name
        if dir_name.startswith("unsloth_"):
            # Remove timestamp suffix (usually _1234567890)
            parts = dir_name.split("_")
            # Reconstruct model name
            if len(parts) >= 2:
                model_parts = parts[1:-1]  # Skip "unsloth" and timestamp
                base_model = "unsloth/" + "_".join(model_parts)
                logger.info(f"Detected base model from directory name: {base_model}")
                return base_model

        logger.warning(f"Could not detect base model for LoRA: {lora_path}")
        return None

    except Exception as e:
        logger.error(f"Error reading base model from LoRA config: {e}")
        return None


# Status indicators that appear in UI dropdowns
UI_STATUS_INDICATORS = [" (Ready)", " (Loading...)", " (Active)", "↓ "]


def load_model_defaults(model_name: str) -> Dict[str, Any]:
    """
    Load default training parameters for a model from YAML file.

    Args:
        model_name: Model identifier (e.g., "unsloth/Meta-Llama-3.1-8B-bnb-4bit")

    Returns:
        Dictionary with default parameters from YAML file, or empty dict if not found

    The function looks for a YAML file in configs/model_defaults/ (including subfolders)
    based on the model name or its aliases from MODEL_NAME_MAPPING.
    If no specific file exists, it falls back to default.yaml.
    """
    try:
        # Get the script directory to locate configs
        script_dir = Path(__file__).parent.parent.parent
        defaults_dir = script_dir / "assets" / "configs" / "model_defaults"

        # First, check if model is in the mapping
        if model_name.lower() in _REVERSE_MODEL_MAPPING:
            canonical_file = _REVERSE_MODEL_MAPPING[model_name.lower()]
            # Search in subfolders and root
            for config_path in defaults_dir.rglob(canonical_file):
                if config_path.is_file():
                    with open(config_path, "r", encoding = "utf-8") as f:
                        config = yaml.safe_load(f) or {}
                        logger.info(
                            f"Loaded model defaults from {config_path} (via mapping)"
                        )
                        return config

        # If model_name is a local path (e.g. /home/.../Spark-TTS-0.5B/LLM from
        # adapter_config.json, or C:\Users\...\model on Windows), try matching
        # the last 1-2 path components against the registry
        # (e.g. "Spark-TTS-0.5B/LLM").
        _is_local_path = is_local_path(model_name)
        # Normalize Windows backslash paths so Path().parts splits correctly
        # on POSIX/WSL hosts (pathlib treats backslashes as literals on Linux).
        _normalized = normalize_path(model_name) if _is_local_path else model_name
        if model_name.lower() not in _REVERSE_MODEL_MAPPING and _is_local_path:
            parts = Path(_normalized).parts
            for depth in [2, 1]:
                if len(parts) >= depth:
                    suffix = "/".join(parts[-depth:])
                    if suffix.lower() in _REVERSE_MODEL_MAPPING:
                        canonical_file = _REVERSE_MODEL_MAPPING[suffix.lower()]
                        for config_path in defaults_dir.rglob(canonical_file):
                            if config_path.is_file():
                                with open(config_path, "r", encoding = "utf-8") as f:
                                    config = yaml.safe_load(f) or {}
                                    logger.info(
                                        f"Loaded model defaults from {config_path} (via path suffix '{suffix}')"
                                    )
                                    return config

        # Try exact model name match (for backward compatibility).
        # For local filesystem paths, use only the directory basename to
        # avoid passing absolute paths (e.g. C:\...) into rglob which
        # raises "Non-relative patterns are unsupported" on Windows.
        _lookup_name = Path(_normalized).name if _is_local_path else model_name
        model_filename = _lookup_name.replace("/", "_") + ".yaml"
        # Search in subfolders and root
        for config_path in defaults_dir.rglob(model_filename):
            if config_path.is_file():
                with open(config_path, "r", encoding = "utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded model defaults from {config_path}")
                    return config

        # Fall back to default.yaml
        default_config_path = defaults_dir / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, "r", encoding = "utf-8") as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Loaded default model defaults from {default_config_path}")
                return config

        logger.warning(f"No default config found for model {model_name}")
        return {}

    except Exception as e:
        logger.error(f"Error loading model defaults for {model_name}: {e}")
        return {}


@dataclass
class ModelConfig:
    """Configuration for a model to load"""

    identifier: str  # Clean model identifier (org/name or path)
    display_name: str  # Original UI display name
    path: str  # Normalized filesystem path
    is_local: bool  # Is this a local file vs HF model?
    is_cached: bool  # Is this already in HF cache?
    is_vision: bool  # Is this a vision model?
    is_lora: bool  # Is this a lora adapter?
    is_gguf: bool = False  # Is this a GGUF model?
    is_audio: bool = False  # Is this a TTS audio model?
    audio_type: Optional[str] = (
        None  # Audio codec type: 'snac', 'csm', 'bicodec', 'dac'
    )
    has_audio_input: bool = False  # Accepts audio input (ASR/speech understanding)
    gguf_file: Optional[str] = None  # Full path to the .gguf file (local mode)
    gguf_mmproj_file: Optional[str] = (
        None  # Full path to the mmproj .gguf file (vision projection)
    )
    gguf_hf_repo: Optional[str] = (
        None  # HF repo ID for -hf mode (e.g. "unsloth/gemma-3-4b-it-GGUF")
    )
    gguf_variant: Optional[str] = None  # Quantization variant (e.g. "Q4_K_M")
    base_model: Optional[str] = None  # Base model (for LoRAs)

    @classmethod
    def from_lora_path(
        cls, lora_path: str, hf_token: Optional[str] = None
    ) -> Optional["ModelConfig"]:
        """
        Create ModelConfig from a local LoRA adapter path.

        Automatically detects the base model from adapter config.

        Args:
            lora_path: Path to LoRA adapter (e.g., "./outputs/unsloth_Meta-Llama-3.1_.../")
            hf_token: HF token for vision detection

        Returns:
            ModelConfig for the LoRA adapter
        """
        try:
            lora_path_obj = Path(lora_path)

            if not lora_path_obj.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return None

            # Get base model
            base_model = get_base_model_from_lora(lora_path)
            if not base_model:
                logger.error(f"Could not determine base model for LoRA: {lora_path}")
                return None

            # Check if base model is vision
            is_vision = is_vision_model(base_model, hf_token = hf_token)

            # Check if base model is audio
            audio_type = detect_audio_type(base_model, hf_token = hf_token)

            display_name = lora_path_obj.name
            identifier = lora_path  # Use path as identifier for local LoRAs

            return cls(
                identifier = identifier,
                display_name = display_name,
                path = lora_path,
                is_local = True,
                is_cached = True,  # Local LoRAs are always "cached"
                is_vision = is_vision,
                is_lora = True,
                is_audio = audio_type is not None and audio_type != "audio_vlm",
                audio_type = audio_type,
                has_audio_input = is_audio_input_type(audio_type),
                base_model = base_model,
            )

        except Exception as e:
            logger.error(f"Error creating ModelConfig from LoRA path: {e}")
            return None

    @classmethod
    def from_identifier(
        cls,
        model_id: str,
        hf_token: Optional[str] = None,
        is_lora: bool = False,
        gguf_variant: Optional[str] = None,
    ) -> Optional["ModelConfig"]:
        """
        Create ModelConfig from a clean model identifier.

        For FastAPI routes where the frontend sends sanitized model paths.
        No Gradio dropdown parsing - expects clean identifiers like:
        - "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        - "./outputs/my_lora_adapter"
        - "/absolute/path/to/model"

        Args:
            model_id: Clean model identifier (HF repo name or local path)
            hf_token: Optional HF token for vision detection on gated models
            is_lora: Whether this is a LoRA adapter
            gguf_variant: Optional GGUF quantization variant (e.g. "Q4_K_M").
                For remote GGUF repos, specifies which quant to load via -hf.
                If None, auto-selects using _pick_best_gguf().

        Returns:
            ModelConfig or None if configuration cannot be created
        """
        if not model_id or not model_id.strip():
            return None

        identifier = model_id.strip()
        is_local = is_local_path(identifier)
        path = normalize_path(identifier) if is_local else identifier

        # Add unsloth/ prefix for shorthand HF models
        if not is_local and "/" not in identifier:
            identifier = f"unsloth/{identifier}"
            path = identifier

        # Preserve requested casing, but if a case-variant already exists in local HF cache,
        # reuse that exact repo_id spelling to avoid one-time re-downloads after #2592.
        if not is_local:
            resolved_identifier = resolve_cached_repo_id_case(identifier)
            if resolved_identifier != identifier:
                logger.info(
                    "Using cached repo_id casing '%s' for requested '%s'",
                    resolved_identifier,
                    identifier,
                )
                identifier = resolved_identifier
                path = resolved_identifier

        # Auto-detect GGUF models (check before LoRA/vision detection)
        if is_local:
            if gguf_variant:
                gguf_file = _find_local_gguf_by_variant(path, gguf_variant)
            else:
                gguf_file = detect_gguf_model(path)
            if gguf_file:
                display_name = Path(gguf_file).stem
                logger.info(f"Detected local GGUF model: {gguf_file}")

                # Detect vision: check if base model is vision, then look for mmproj
                mmproj_file = None
                gguf_is_vision = False
                gguf_dir = Path(gguf_file).parent

                # Determine if this is a vision model from export metadata
                base_is_vision = False
                meta_path = gguf_dir / "export_metadata.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        base = meta.get("base_model")
                        if base and is_vision_model(base, hf_token = hf_token):
                            base_is_vision = True
                            logger.info(f"GGUF base model '{base}' is a vision model")
                    except Exception as e:
                        logger.debug(f"Could not read export metadata: {e}")

                # If vision (or mmproj happens to exist), find the mmproj
                # file. The recursive variant scan in
                # ``_find_local_gguf_by_variant`` may have returned a
                # weight file inside a quant-named subdir (e.g.
                # ``.../BF16/foo.gguf``) while ``mmproj-*.gguf`` lives
                # at the snapshot root. Pass ``search_root=path`` so
                # ``detect_mmproj_file`` walks up to the snapshot root
                # instead of seeing only the weight file's immediate
                # parent.
                mmproj_file = detect_mmproj_file(gguf_file, search_root = path)
                if mmproj_file:
                    gguf_is_vision = True
                    logger.info(f"Detected mmproj for vision: {mmproj_file}")
                elif base_is_vision:
                    logger.warning(
                        f"Base model is vision but no mmproj file found in {gguf_dir}"
                    )

                return cls(
                    identifier = identifier,
                    display_name = display_name,
                    path = path,
                    is_local = True,
                    is_cached = True,
                    is_vision = gguf_is_vision,
                    is_lora = False,
                    is_gguf = True,
                    gguf_file = gguf_file,
                    gguf_mmproj_file = mmproj_file,
                )
        else:
            # Check if the HF repo contains GGUF files
            gguf_filename = detect_gguf_model_remote(identifier, hf_token = hf_token)
            if gguf_filename:
                # Preflight: verify llama-server binary exists BEFORE user waits
                # for a multi-GB download that llama-server handles natively
                from core.inference.llama_cpp import LlamaCppBackend

                if not LlamaCppBackend._find_llama_server_binary():
                    raise RuntimeError(
                        "llama-server binary not found — cannot load GGUF models. "
                        "Run setup.sh to build it, or set LLAMA_SERVER_PATH."
                    )

                # Use list_gguf_variants() to detect vision & resolve variant
                variants, has_vision = list_gguf_variants(identifier, hf_token = hf_token)
                variant = gguf_variant
                if not variant:
                    # Auto-select best quantization
                    variant_filenames = [v.filename for v in variants]
                    best = _pick_best_gguf(variant_filenames)
                    if best:
                        variant = _extract_quant_label(best)
                    else:
                        variant = "Q4_K_M"  # Fallback — llama-server's own default

                display_name = f"{identifier.split('/')[-1]} ({variant})"
                logger.info(
                    f"Detected remote GGUF repo '{identifier}', "
                    f"variant={variant}, vision={has_vision}"
                )
                return cls(
                    identifier = identifier,
                    display_name = display_name,
                    path = identifier,
                    is_local = False,
                    is_cached = False,
                    is_vision = has_vision,
                    is_lora = False,
                    is_gguf = True,
                    gguf_file = None,
                    gguf_hf_repo = identifier,
                    gguf_variant = variant,
                )

        # Auto-detect LoRA for local paths (check adapter_config.json on disk)
        if not is_lora and is_local:
            detected_base = (
                get_base_model_from_lora(path)
                if _looks_like_lora_adapter(Path(path))
                else None
            )
            if detected_base:
                is_lora = True
                logger.info(
                    f"Auto-detected local LoRA adapter at '{path}' (base: {detected_base})"
                )

        # Auto-detect LoRA for remote HF models (check repo file listing)
        if not is_lora and not is_local:
            try:
                from huggingface_hub import model_info as hf_model_info

                info = hf_model_info(identifier, token = hf_token)
                repo_files = [s.rfilename for s in info.siblings]
                if "adapter_config.json" in repo_files:
                    is_lora = True
                    logger.info(f"Auto-detected remote LoRA adapter: '{identifier}'")
            except Exception as e:
                logger.debug(
                    f"Could not check remote LoRA status for '{identifier}': {e}"
                )

        # Handle LoRA adapters
        base_model = None
        if is_lora:
            if is_local:
                # Local LoRA: read adapter_config.json from disk
                base_model = get_base_model_from_lora(path)
            else:
                # Remote LoRA: download adapter_config.json from HF
                try:
                    from huggingface_hub import hf_hub_download

                    config_path = hf_hub_download(
                        identifier, "adapter_config.json", token = hf_token
                    )
                    with open(config_path, "r") as f:
                        adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
                    if base_model:
                        logger.info(f"Resolved remote LoRA base model: '{base_model}'")
                except Exception as e:
                    logger.warning(
                        f"Could not download adapter_config.json for '{identifier}': {e}"
                    )

            if not base_model:
                logger.warning(f"Could not determine base model for LoRA '{path}'")
                return None
            check_model = base_model
        else:
            check_model = identifier

        vision = is_vision_model(check_model, hf_token = hf_token)
        audio_type_val = detect_audio_type(check_model, hf_token = hf_token)
        has_audio_in = is_audio_input_type(audio_type_val)

        display_name = Path(path).name if is_local else identifier.split("/")[-1]

        return cls(
            identifier = identifier,
            display_name = display_name,
            path = path,
            is_local = is_local,
            is_cached = is_model_cached(identifier) if not is_local else True,
            is_vision = vision,
            is_lora = is_lora,
            is_audio = audio_type_val is not None and audio_type_val != "audio_vlm",
            audio_type = audio_type_val,
            has_audio_input = has_audio_in,
            base_model = base_model,
        )

    @classmethod
    def from_ui_selection(
        cls,
        dropdown_value: Optional[str],
        search_value: Optional[str],
        local_models: list = None,
        hf_token: Optional[str] = None,
        is_lora: bool = False,
    ) -> Optional["ModelConfig"]:
        """
        Create a universal ModelConfig from UI dropdown/search selections.
        Handles base models and LoRA adapters.
        """
        selected = None
        if search_value and search_value.strip():
            selected = search_value.strip()
        elif dropdown_value:
            selected = dropdown_value

        if not selected:
            return None

        display_name = selected

        #  Use the correct 'local_models' parameter to resolve display names
        if " (Active)" in selected or " (Ready)" in selected:
            clean_display_name = selected.replace(" (Active)", "").replace(
                " (Ready)", ""
            )
            if local_models:
                for local_display, local_path in local_models:
                    if local_display == clean_display_name:
                        selected = local_path
                        break

        # Clean all UI status indicators to get the final identifier
        identifier = selected
        for status in UI_STATUS_INDICATORS:
            identifier = identifier.replace(status, "")
        identifier = identifier.strip()

        is_local = is_local_path(identifier)
        path = normalize_path(identifier) if is_local else identifier

        # Add unsloth/ prefix for shorthand HF models
        if not is_local and "/" not in identifier:
            identifier = f"unsloth/{identifier}"
            path = identifier

        if not is_local:
            resolved_identifier = resolve_cached_repo_id_case(identifier)
            if resolved_identifier != identifier:
                identifier = resolved_identifier
                path = resolved_identifier

        # --- Logic for Base Model and Vision Detection ---
        base_model = None
        is_vision = False

        if is_lora:
            # For a LoRA, we MUST find its base model.
            base_model = get_base_model_from_lora(path)
            if not base_model:
                logger.warning(
                    f"Could not determine base model for LoRA '{path}'. Cannot create config."
                )
                return None  # Cannot proceed without a base model

            # A LoRA's vision capability is determined by its base model.
            is_vision = is_vision_model(base_model, hf_token = hf_token)
        else:
            # For a base model, just check its own vision status.
            is_vision = is_vision_model(identifier, hf_token = hf_token)

        from utils.paths import is_model_cached

        is_cached = is_model_cached(identifier) if not is_local else True

        return cls(
            identifier = identifier,
            display_name = display_name,
            path = path,
            is_local = is_local,
            is_cached = is_cached,
            is_vision = is_vision,
            is_lora = is_lora,
            base_model = base_model,  # This will be None for base models, and populated for LoRAs
        )
