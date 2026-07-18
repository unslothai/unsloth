# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model and LoRA configuration handling."""

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
from utils.models.gguf_metadata import (
    is_mmproj_by_metadata,
    pairing_score,
    read_gguf_general_metadata,
)
import structlog
from loggers import get_logger
import os
import re
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


_OFFLINE_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_offline() -> bool:
    """True if an HF offline env var is truthy (canonical strip+lower parse, on/true/yes/1)."""
    return (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_TRUE_VALUES
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE_VALUES
    )


# ── Model size extraction ────────────────────────────────────
import re as _re

_MODEL_SIZE_RE = _re.compile(r"(?:^|[-_/])(\d+\.?\d*)\s*([bm])(?:$|[-_/])", _re.IGNORECASE)
# MoE active-parameter pattern: "A3B", "A3.5B", etc.
_ACTIVE_SIZE_RE = _re.compile(r"(?:^|[-_/])a(\d+\.?\d*)\s*([bm])(?:$|[-_/])", _re.IGNORECASE)
# Gemma 3n/4 effective-parameter pattern: "E2B", "E4B" -- the runtime
# footprint (MatFormer + per-layer embeddings), which is the size that
# matters for size-gated policies like sub-3B speculative-decoding fallback.
_EFFECTIVE_SIZE_RE = _re.compile(r"(?:^|[-_/])e(\d+\.?\d*)\s*([bm])(?:$|[-_/])", _re.IGNORECASE)


def extract_model_size_b(model_id: str) -> float | None:
    """Extract model size in billions from a model identifier.

    Prefers MoE active-parameter notation (e.g. ``A3B`` in
    ``Qwen3.5-35B-A3B``), then Gemma effective-parameter notation
    (e.g. ``E2B``), over total params. Handles ``B`` (billions) and
    ``M`` (millions) suffixes.
    """
    mid = (model_id or "").lower()
    # First match wins, in priority order: active > effective > total.
    for pattern in (_ACTIVE_SIZE_RE, _EFFECTIVE_SIZE_RE, _MODEL_SIZE_RE):
        m = pattern.search(mid)
        if m:
            val = float(m.group(1))
            return val / 1000.0 if m.group(2).lower() == "m" else val
    return None


# Maps equivalent model names to their canonical YAML config file.
# Format: "canonical_model_name.yaml": [equivalent model names].
# Canonical filename derives from the first model name in each list.
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

# Reverse lookup: model_name -> canonical_filename
_REVERSE_MODEL_MAPPING = {}
for canonical_file, model_names in MODEL_NAME_MAPPING.items():
    for model_name in model_names:
        _REVERSE_MODEL_MAPPING[model_name.lower()] = canonical_file


def load_model_config(
    model_name: str,
    use_auth: bool = False,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    """Load model config with optional authentication control.

    ``trust_remote_code`` defaults to ``False``: capability detection and
    metadata lookups must never execute a model repo's ``auto_map`` Python.
    Deliberate remote-code loads pass the flag explicitly through
    ``FastLanguageModel.from_pretrained`` with the user's own consent.

    ``local_files_only`` keeps the config read on the local HF cache (offline
    export), so an offline probe never blocks on the network.
    """
    from transformers import AutoConfig

    if token:
        return AutoConfig.from_pretrained(
            model_name,
            trust_remote_code = trust_remote_code,
            token = token,
            local_files_only = local_files_only,
        )

    if not use_auth:
        # No auth, for public model checks
        with without_hf_auth():
            return AutoConfig.from_pretrained(
                model_name,
                trust_remote_code = trust_remote_code,
                token = None,
                local_files_only = local_files_only,
            )

    # Default auth (cached tokens)
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code = trust_remote_code,
        local_files_only = local_files_only,
    )


# Detection sets come from the installed transformers registry, unioned with a
# small curated set of auto_map VLMs (DeepSeek-OCR, Kimi, phi3_v) whose arch is
# repo-defined and absent from the registry. ForConditionalGeneration is NOT a
# vision signal (overloaded across text/audio/vision); ForVisionText2Text is.
_VLM_ARCH_SUFFIXES = ("ForVisionText2Text",)

_CURATED_REMOTE_VLM_TYPES = frozenset(
    {
        "phi3_v",
        "llava",
        "llava_next",
        "llava_onevision",
        "internvl_chat",
        "cogvlm2",
        "minicpmv",
        "gemma4",
        "deepseek_vl_v2",
        "kimi_k25",
    }
)

# Fallbacks used only if the transformers registry import fails.
_FALLBACK_AUDIO_MODEL_TYPES = frozenset({"csm", "whisper"})


def _build_detection_sets():
    """Return (vlm_model_types, vlm_class_names, audio_model_types) from the
    installed transformers registry, unioned with the curated repo-code VLM
    set. Reads only static name dicts -- no model is loaded, no code runs.
    Falls back to curated/hardcoded values if transformers is unavailable.
    """
    try:
        from transformers.models.auto import modeling_auto as _ma

        def _names(attr):
            d = getattr(_ma, attr, None)
            return dict(d) if d else {}

        itt = _names("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES")
        v2s = _names("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES")
        vlm_types = set(itt) | set(v2s) | set(_CURATED_REMOTE_VLM_TYPES)
        vlm_classes = set(itt.values()) | set(v2s.values())

        audio_types: set = set()
        for attr in (
            "MODEL_FOR_CTC_MAPPING_NAMES",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES",
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES",
        ):
            audio_types |= set(_names(attr))
        audio_types |= set(_FALLBACK_AUDIO_MODEL_TYPES)

        return frozenset(vlm_types), frozenset(vlm_classes), frozenset(audio_types)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not build detection sets from transformers: %s", exc)
        return (
            frozenset(_CURATED_REMOTE_VLM_TYPES),
            frozenset(),
            frozenset(_FALLBACK_AUDIO_MODEL_TYPES),
        )


_VLM_MODEL_TYPES, _VLM_CLASS_NAMES, _AUDIO_ONLY_MODEL_TYPES = _build_detection_sets()

# Pre-computed .venv_t5 paths and backend dir for subprocess version switching.
# Vision check uses the Gemma 4 5.5 sidecar for existing Gemma 4 architectures.
from utils.paths.storage_roots import studio_root as _studio_root  # noqa: E402

_VENV_T5_DIR = str(_studio_root() / ".venv_t5_550")
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _is_vlm(config) -> bool:
    architectures = getattr(config, "architectures", None) or []
    model_type = getattr(config, "model_type", None)
    explicit_vision = (
        hasattr(config, "vision_config")
        or hasattr(config, "img_processor")
        or hasattr(config, "image_token_index")
        or hasattr(config, "projector_config")
    )
    # Audio-only models are vision only if they carry an explicit vision sub-config.
    if model_type in _AUDIO_ONLY_MODEL_TYPES and not explicit_vision:
        return False
    return (
        explicit_vision
        or any(x in _VLM_CLASS_NAMES for x in architectures)
        or any(isinstance(x, str) and x.endswith(_VLM_ARCH_SUFFIXES) for x in architectures)
        or model_type in _VLM_MODEL_TYPES
    )


def _raw_config_has_vision_config(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[bool]:
    try:
        if is_local_path(model_name):
            config_path = Path(normalize_path(model_name)).expanduser() / "config.json"
        else:
            from huggingface_hub import hf_hub_download
            config_path = Path(
                hf_hub_download(
                    repo_id = model_name,
                    filename = "config.json",
                    token = hf_token,
                    local_files_only = local_files_only,
                )
            )
        config = json.loads(config_path.read_text())
        architectures = config.get("architectures") or []
        model_type = config.get("model_type")
        explicit_vision = (
            "vision_config" in config
            or "img_processor" in config
            or "image_token_index" in config
            or "projector_config" in config
        )
        # Audio-only models are vision only if they carry an explicit vision sub-config.
        if model_type in _AUDIO_ONLY_MODEL_TYPES and not explicit_vision:
            return False
        return (
            explicit_vision
            or any(isinstance(x, str) and x in _VLM_CLASS_NAMES for x in architectures)
            or any(isinstance(x, str) and x.endswith(_VLM_ARCH_SUFFIXES) for x in architectures)
            or model_type in _VLM_MODEL_TYPES
        )
    except Exception as exc:
        logger.warning("Could not read config.json for '%s': %s", model_name, exc)
        return None


# why: inline _is_vlm and constants are prepended so the subprocess stays
# self-contained and does not import the parent backend module graph.
_VISION_CHECK_INLINE_HELPERS = (
    "_VLM_ARCH_SUFFIXES = " + repr(tuple(_VLM_ARCH_SUFFIXES)) + "\n"
    "_VLM_MODEL_TYPES = " + repr(set(_VLM_MODEL_TYPES)) + "\n"
    "_VLM_CLASS_NAMES = " + repr(set(_VLM_CLASS_NAMES)) + "\n"
    "_AUDIO_ONLY_MODEL_TYPES = " + repr(set(_AUDIO_ONLY_MODEL_TYPES)) + "\n"
    "def _is_vlm(config):\n"
    "    architectures = getattr(config, 'architectures', None) or []\n"
    "    model_type = getattr(config, 'model_type', None)\n"
    "    explicit_vision = (\n"
    "        hasattr(config, 'vision_config')\n"
    "        or hasattr(config, 'img_processor')\n"
    "        or hasattr(config, 'image_token_index')\n"
    "        or hasattr(config, 'projector_config')\n"
    "    )\n"
    "    if model_type in _AUDIO_ONLY_MODEL_TYPES and not explicit_vision:\n"
    "        return False\n"
    "    return (\n"
    "        explicit_vision\n"
    "        or any(x in _VLM_CLASS_NAMES for x in architectures)\n"
    "        or any(isinstance(x, str) and x.endswith(_VLM_ARCH_SUFFIXES) for x in architectures)\n"
    "        or model_type in _VLM_MODEL_TYPES\n"
    "    )\n"
)

# Subprocess script run with transformers 5.x active. Takes model_name and
# token via argv, prints JSON result to stdout.
_VISION_CHECK_SCRIPT = (
    r"""
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

"""
    + _VISION_CHECK_INLINE_HELPERS
    + r"""
try:
    from transformers import AutoConfig

    # Union the ACTIVE sidecar's registry into the inlined parent-process sets
    # so architectures only the sidecar knows still classify correctly.
    try:
        from transformers.models.auto import modeling_auto as _ma
        for _attr in ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
                      "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
            _d = dict(getattr(_ma, _attr, None) or {})
            _VLM_MODEL_TYPES |= set(_d)
            _VLM_CLASS_NAMES |= set(_d.values())
        for _attr in ("MODEL_FOR_CTC_MAPPING_NAMES",
                      "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES",
                      "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
                      "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES",
                      "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES",
                      "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES"):
            _AUDIO_ONLY_MODEL_TYPES |= set(dict(getattr(_ma, _attr, None) or {}))
    except Exception:
        pass

    # Capability detection never executes model repo code.
    kwargs = {"trust_remote_code": False}
    if token:
        kwargs["token"] = token
    config = AutoConfig.from_pretrained(model_name, **kwargs)

    is_vlm = _is_vlm(config)

    model_type = getattr(config, "model_type", None)
    archs = getattr(config, "architectures", [])
    print(json.dumps({"is_vision": is_vlm, "model_type": model_type,
                       "architectures": archs}))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(1)
"""
)


def _is_vision_model_subprocess(model_name: str, hf_token: Optional[str] = None) -> Optional[bool]:
    """Run is_vision_model in a subprocess with transformers 5.x.

    Spawns a clean subprocess with .venv_t5/ on sys.path so AutoConfig
    recognizes newer architectures. Returns True/False for definitive results,
    or None for transient failures (timeouts, subprocess errors), which are not
    cached so they can be retried.
    """
    token_arg = hf_token or ""

    # Latest-only architectures need the latest sidecar for AutoConfig;
    # other tiers keep the 5.5 sidecar.
    sidecar_dir = _VENV_T5_DIR
    try:
        from utils.transformers_version import _VENV_T5_LATEST_DIR, get_transformers_tier
        if get_transformers_tier(model_name, hf_token, probe = False) == "latest":
            sidecar_dir = _VENV_T5_LATEST_DIR
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _VISION_CHECK_SCRIPT,
                sidecar_dir,
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
    """SHA256 digest of the token for use as a cache key (avoids storing the
    raw bearer token in process memory)."""
    if token is None:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Vision detection cache keyed by (name, token, local_files_only); only definitive results cached.
_vision_detection_cache: Dict[Tuple[str, Optional[str], bool], bool] = {}
_vision_cache_lock = threading.Lock()


def is_vision_model(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> bool:
    """Detect VLMs via the config architecture (works for fine-tunes); transformers-5.x
    models are checked in a .venv_t5/ subprocess. Cached per (model_name, token,
    local_files_only) minus transient failures; local_files_only is in the key so an
    offline probe never shares an online entry."""
    # Local GGUF models are served by llama-server. Their multimodal
    # capability comes from a companion mmproj, not a Transformers config.
    # Do not cache this lookup: a projector may be added beside an existing
    # weight file after it was first inspected.
    if is_local_path(model_name):
        local_path = normalize_path(model_name)
        gguf_file = detect_gguf_model(local_path)
        if gguf_file:
            companion_root = _local_gguf_companion_search_root(local_path, gguf_file)
            mmproj_file = detect_mmproj_file(gguf_file, search_root = companion_root)
            is_vision = mmproj_file is not None
            logger.debug(
                "Local GGUF vision check for '%s': mmproj=%s, is_vision=%s",
                gguf_file,
                mmproj_file,
                is_vision,
            )
            return is_vision

    # Normalize model name so different casings of the same repo share a key
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
    # Key on effective offline (kwarg OR env) so an offline probe can't poison a later
    # online lookup once the env var is cleared.
    effective_offline = bool(local_files_only or _env_offline())
    cache_key = (resolved_name, _token_fingerprint(hf_token), effective_offline)

    # Lock-free fast path for cache hits. Sentinel distinguishes "key not found"
    # from "value is False" in a single atomic dict.get() call.
    _MISS = object()
    cached = _vision_detection_cache.get(cache_key, _MISS)
    if cached is not _MISS:
        return cached

    # Compute outside the lock so long-running detection isn't serialized across
    # models. Two concurrent calls may both run, but produce the same result.
    result = _is_vision_model_uncached(resolved_name, hf_token, local_files_only = effective_offline)
    # Only cache definitive results; None is a transient failure, retry later.
    if result is not None:
        with _vision_cache_lock:
            _vision_detection_cache[cache_key] = result
        return result
    return False


def _is_vision_model_uncached(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[bool]:
    """Uncached vision detection; use is_vision_model() instead.

    Returns True/False for definitive results, or None on transient errors
    (network, timeout, subprocess failure) so the caller knows not to cache.
    """
    # Try the raw-config reader FIRST (code-free, version-independent): it classifies
    # repo-code VLMs like DeepSeek-OCR via declarative vision_config with no remote-code
    # execution or transformers-5.x subprocess.
    raw = _raw_config_has_vision_config(
        model_name, hf_token = hf_token, local_files_only = local_files_only
    )
    if raw is not None:
        if raw is False and not local_files_only:
            # Raw heuristics predate latest-only architectures; on the latest tier,
            # trust that sidecar's AutoConfig probe over the heuristic False. An
            # inconclusive probe (sidecar mid-repair, timeout) is transient: return
            # None so the heuristic False is not cached and the model is re-probed.
            try:
                from utils.transformers_version import get_transformers_tier
                if get_transformers_tier(model_name, hf_token, probe = False) == "latest":
                    return _is_vision_model_subprocess(model_name, hf_token = hf_token)
            except Exception:
                pass
        return raw

    # Raw read failed transiently: fall back to AutoConfig (remote code DISABLED), via a
    # transformers-5.x subprocess if needed. Skip that subprocess offline (it probes the network).
    from utils.transformers_version import needs_transformers_5

    if not local_files_only and needs_transformers_5(model_name):
        logger.info(
            "Model '%s' needs transformers 5.x -- checking vision via subprocess",
            model_name,
        )
        return _is_vision_model_subprocess(model_name, hf_token = hf_token)

    try:
        config = load_model_config(
            model_name,
            use_auth = True,
            token = hf_token,
            local_files_only = local_files_only,
        )

        if _is_vlm(config):
            model_type = getattr(config, "model_type", None)
            archs = getattr(config, "architectures", None) or []
            logger.info(
                "Model %s detected as VLM (model_type=%s, architectures=%s)",
                model_name,
                model_type,
                archs,
            )
            return True

        return False

    except Exception as e:
        logger.warning(f"Could not determine if {model_name} is vision model: {e}")
        # Permanent failures (not found, gated, bad config) cache as False;
        # transient ones (network, timeout) should not.
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

# Keyed like the vision cache by (name, token, local_files_only) so an unauthenticated
# or offline miss cannot poison a later authenticated / online lookup.
_audio_detection_cache: Dict[Tuple[str, Optional[str], bool], Optional[str]] = {}

# Tokenizer token patterns → audio_type (all 6 types from tokenizer_config.json)
_AUDIO_TOKEN_PATTERNS = {
    "csm": lambda tokens: "<|AUDIO|>" in tokens and "<|audio_eos|>" in tokens,
    "whisper": lambda tokens: "<|startoftranscript|>" in tokens,
    # Gemma 3n: <audio_soft_token>; Gemma 4: <|audio|> (not csm's <|AUDIO|>).
    "audio_vlm": lambda tokens: "<audio_soft_token>" in tokens or "<|audio|>" in tokens,
    "bicodec": lambda tokens: any(t.startswith("<|bicodec_") for t in tokens),
    "dac": lambda tokens: (
        "<|audio_start|>" in tokens
        and "<|audio_end|>" in tokens
        and "<|text_start|>" in tokens
        and "<|text_end|>" in tokens
    ),
    "snac": lambda tokens: (sum(1 for t in tokens if t.startswith("<custom_token_")) > 10000),
}


def detect_audio_type(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[str]:
    """Detect if a model is an audio model and return its type.

    Works for any model via tokenizer_config.json special tokens.
    Returns an audio_type string ('snac', 'csm', 'bicodec', 'dac', 'whisper',
    'audio_vlm') or None.

    When local_files_only is True (offline export) the remote HuggingFace fetch
    is skipped so detection never blocks on a network read; only the local HF
    cache is consulted.
    """
    # Normalize casing + include the token fingerprint (mirrors is_vision_model).
    try:
        if is_local_path(model_name):
            resolved_name = normalize_path(model_name)
        else:
            resolved_name = resolve_cached_repo_id_case(model_name)
    except Exception:
        resolved_name = model_name
    # Key on effective offline (kwarg OR env), matching where the remote fetch is skipped,
    # so an offline negative can't poison a later online probe.
    effective_offline = bool(local_files_only or _env_offline())
    cache_key = (resolved_name, _token_fingerprint(hf_token), effective_offline)
    if cache_key in _audio_detection_cache:
        return _audio_detection_cache[cache_key]

    result, definitive = _detect_audio_from_tokenizer(
        model_name, hf_token, local_files_only = effective_offline
    )
    # Cache only definitive results; a transient read failure stays None and retries.
    if definitive:
        _audio_detection_cache[cache_key] = result
    if result:
        logger.info(f"Model {model_name} detected as audio model: audio_type={result}")
    return result


def _detect_audio_from_tokenizer(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Tuple[Optional[str], bool]:
    """Detect audio type from tokenizer special tokens.

    Checks local HF cache first, then (unless local_files_only) fetches
    tokenizer_config.json from HF; examines added_tokens_decoder for distinctive
    patterns.

    Returns (audio_type_or_None, definitive). definitive is False only on a
    transient read failure (network/timeout/5xx) so the caller skips caching and
    retries; a successful read with no audio tokens is a definitive None.
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

    read_any = False  # parsed at least one tokenizer_config -> a None is definitive

    # 1) Local HF cache first (works for gated/offline models)
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
                            read_any = True
                            result = _check_token_patterns(tok_config)
                            if result:
                                return result, True
    except Exception as e:
        logger.debug(f"Could not check local cache for {model_name}: {e}")

    # 2) Fall back to the HuggingFace API. This raw requests.get ignores the HF offline
    #    flag, so gate it on local_files_only OR the env vars to skip the network offline.
    if local_files_only or _env_offline():
        return None, read_any

    try:
        import requests
        import os
    except Exception:
        return None, read_any

    paths_to_try = ["tokenizer_config.json", "LLM/tokenizer_config.json"]
    token = hf_token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    transient = False  # a fetch failed for a non-404 reason (network/5xx)
    for tok_path in paths_to_try:
        url = f"https://huggingface.co/{model_name}/resolve/main/{tok_path}"
        try:
            resp = requests.get(url, headers = headers, timeout = 15)
        except Exception as e:
            logger.debug(f"Could not fetch {tok_path} for {model_name}: {e}")
            transient = True
            continue
        if resp.status_code == 404:
            continue  # genuinely absent on this path
        if not resp.ok:
            transient = True  # 5xx/403/etc -- can't tell, don't cache
            continue
        try:
            tok_config = resp.json()
        except Exception as e:
            logger.debug(f"Bad tokenizer_config for {model_name}/{tok_path}: {e}")
            transient = True
            continue
        read_any = True
        result = _check_token_patterns(tok_config)
        if result:
            return result, True

    # No audio tokens: definitive unless every attempt failed transiently.
    return None, (read_any or not transient)


def is_audio_input_type(audio_type: Optional[str]) -> bool:
    """True if an audio_type accepts audio input: whisper (ASR), audio_vlm (Gemma3n)."""
    return audio_type in ("whisper", "audio_vlm")


def _is_mmproj(filename: str) -> bool:
    """Check if a GGUF filename is a vision projection (mmproj) file."""
    return "mmproj" in filename.lower()


def _is_mtp_drafter(path: str) -> bool:
    """True for a separate-file MTP drafter (speculative head), a companion
    to the main model rather than a selectable quant: the repo-root
    ``mtp-*.gguf`` or the ``MTP/`` subdir copies (Gemma 4).

    Mirrors hub.utils.gguf.is_mtp_drafter_path (utils cannot import hub).
    Must be excluded everywhere mmproj is, or the drafter leaks into variant
    menus (a phantom quant) and quant-matched file lookups -- e.g. a ``Q8_0``
    request must not resolve to ``MTP/...-Q8_0-MTP.gguf``, which sorts ahead
    of the real weight.
    """
    p = path.lower()
    if not p.endswith(".gguf"):
        return False
    name = p.rsplit("/", 1)[-1]
    return name.startswith("mtp-") or "/mtp/" in f"/{p}"


# Family tokens for #5347's filename fallback. Lowercase; order irrelevant.
_MODEL_FAMILY_TOKENS: tuple[str, ...] = (
    "qwen",
    "gemma",
    "llama",
    "mistral",
    "ministral",
    "magistral",
    "devstral",
    "phi",
    "deepseek",
    "internvl",
    "minicpm",
    "llava",
    "glm",
    "yi",
    "command-r",
    "molmo",
    "pixtral",
    "smolvlm",
    "moondream",
    "granite",
    "ovis",
    "nemotron",
    "kimi",
    "nanonets",
    "cosmos",
    "mimo",
    "apriel",
    "lfm",
)


# Word-bounded match: a letter on either side disqualifies (stops ``phi``
# matching ``sapphire``, ``yi`` matching ``tiny``).
_FAMILY_TOKEN_RE_CACHE: Dict[str, "_re.Pattern[str]"] = {}


def _family_token_re(token: str) -> "_re.Pattern[str]":
    pat = _FAMILY_TOKEN_RE_CACHE.get(token)
    if pat is None:
        pat = _re.compile(rf"(?:^|[^a-z])({_re.escape(token)})(?:[^a-z]|$)")
        _FAMILY_TOKEN_RE_CACHE[token] = pat
    return pat


def _detect_family_token(filename: str) -> Optional[str]:
    """Leftmost-position match; ties prefer the longer token."""
    name = filename.lower()
    best: Optional[tuple[int, int, str]] = None  # (start, -len, token)
    for token in _MODEL_FAMILY_TOKENS:
        m = _family_token_re(token).search(name)
        if m is None:
            continue
        key = (m.start(1), -len(token), token)
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


def mmproj_matches_model_family(model_path: str, mmproj_path: str) -> bool:
    """Launcher guard: True unless both filenames carry recognised family
    tokens that disagree."""
    model_fam = _detect_family_token(Path(model_path).name)
    mmproj_fam = _detect_family_token(Path(mmproj_path).name)
    if model_fam is None or mmproj_fam is None:
        return True
    return model_fam == mmproj_fam


def _shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


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
    """Find the mmproj GGUF for a model.

    ``path``: directory or a .gguf file. ``search_root``: optional ancestor
    to also walk (snapshot layouts where the weight is in ``snapshot/BF16/``
    but the projector sits at ``snapshot/``). Returns the projector path or
    ``None``."""
    p = Path(path)
    start_dir = p.parent if p.is_file() else p
    if not start_dir.is_dir():
        return None

    # Walk incrementally so a sibling subdir's mmproj cannot leak in.
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

    # Ollama's .studio_links/foo.gguf -> blobs/sha256-...: also scan target dir.
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
            if root_resolved == start_resolved or (
                start_resolved.is_relative_to(root_resolved)
                if hasattr(start_resolved, "is_relative_to")
                else str(start_resolved).startswith(str(root_resolved) + "/")
            ):
                cur = start_resolved
                while cur != root_resolved and cur.parent != cur:
                    cur = cur.parent
                    _add(cur)
                    if cur == root_resolved:
                        break
        except OSError:
            pass

    candidates: list[Path] = []
    seen_resolved: set[Path] = set()
    for d in scan_order:
        for f in _iter_gguf_files(d):
            try:
                resolved = f.resolve()
            except OSError:
                continue
            if resolved in seen_resolved:
                continue
            # Prefer ``general.type=='mmproj'``, else filename.
            meta = read_gguf_general_metadata(str(resolved))
            by_meta = is_mmproj_by_metadata(meta)
            if by_meta is True or (by_meta is None and _is_mmproj(f.name)):
                seen_resolved.add(resolved)
                candidates.append(resolved)

    if not candidates:
        return None

    # Directory path: no model name to compare against; legacy behaviour.
    if not p.is_file():
        return str(candidates[0])

    # Stage 1: GGUF metadata. Stage 2: filename family token (#5347).
    model_stem = p.stem.lower()
    model_family = _detect_family_token(p.name)
    weight_meta = read_gguf_general_metadata(str(p))

    scored: list[tuple[int, Path]] = []
    for c in candidates:
        cand_meta = read_gguf_general_metadata(str(c))
        meta_score = pairing_score(weight_meta, cand_meta)
        if meta_score == -1:
            logger.info(f"detect_mmproj_file: dropped {c.name} (metadata mismatch)")
            continue
        if meta_score == 0 and model_family is not None:
            # Unrecognised candidate family is a wildcard (``mmproj-F16.gguf``).
            cand_family = _detect_family_token(c.name)
            if cand_family is not None and cand_family != model_family:
                logger.info(
                    f"detect_mmproj_file: dropped {c.name} "
                    f"(filename family {cand_family!r} vs model {model_family!r})"
                )
                continue
        scored.append((meta_score, c))

    if not scored:
        return None

    # Score first, then longest shared prefix, then shorter stem.
    best = max(
        scored,
        key = lambda sc: (
            sc[0],
            _shared_prefix_len(model_stem, sc[1].stem.lower()),
            -len(sc[1].stem),
        ),
    )
    return str(best[1])


def detect_mtp_file(path: str, search_root: Optional[str] = None) -> Optional[str]:
    """Find the separate MTP drafter (``mtp-*.gguf``) for a local GGUF model.

    The drafter that pairs with the main weights sits at the repo/snapshot
    root (Gemma 4); the weight itself may be at the root or in a quant subdir,
    so scan the weight's directory and ``search_root``. Matches by the
    ``mtp-`` filename prefix unsloth uses for ``-hf`` auto-discovery -- the
    same signal as the HF download path. Repos that bake the head into the
    main GGUF (Qwen) have no such sibling, so this returns None.

    Pairs by name so a multi-model folder can't attach a foreign drafter:
    unsloth names the drafter ``mtp-<model>.gguf`` where ``<model>`` prefixes
    the weight filename across all Gemma 4 repos (e.g.
    ``mtp-gemma-4-12B-it.gguf`` next to ``gemma-4-12B-it-qat-Q4_0.gguf``).
    An unmatched drafter is skipped (fail-safe: no MTP).
    """
    p = Path(path)
    weight_name = p.name.lower() if p.suffix.lower() == ".gguf" else None
    start_dir = p.parent if p.is_file() else p
    dirs = [start_dir]
    if search_root is not None:
        dirs.append(Path(search_root))
    for d in dirs:
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for f in entries:
            name = f.name.lower()
            if not (name.startswith("mtp-") and name.endswith(".gguf")):
                continue
            stem = name[len("mtp-") : -len(".gguf")]
            if not stem or (weight_name is not None and not weight_name.startswith(stem)):
                continue
            try:
                if f.is_file():
                    return str(f.resolve())
            except OSError:
                continue
    return None


def detect_gguf_model(path: str) -> Optional[str]:
    """Check if a local path is or contains a GGUF model file.

    Handles a direct .gguf path or a directory of .gguf files. Skips mmproj
    files (pass those via ``--mmproj``; see :func:`detect_mmproj_file`). Returns
    the .gguf path or None. For HF repos, use detect_gguf_model_remote().
    """
    p = Path(path)

    # Case 1: direct .gguf file
    if p.suffix.lower() == ".gguf":
        # Companions are not models: rejecting a drafter here also keeps
        # detect_mtp_file from pairing the same file with itself
        # (-m drafter --model-draft drafter). Include the immediate parent
        # dir so the MTP/ subdir copies are caught -- the basename alone
        # (...-MTP.gguf) doesn't match the predicate's mtp- prefix.
        rel = f"{p.parent.name}/{p.name}"
        quant = _extract_quant_label(rel)
        if _is_mmproj(p.name) or _is_mtp_drafter(rel) or _is_big_endian_gguf_path(rel, quant):
            return None
        # Extension is authoritative: don't gate on is_file()/exists(), which
        # can fail in the Windows lock window after llama-server is killed.
        try:
            is_dir = p.is_dir()
        except OSError:
            is_dir = False  # stat() unavailable in the lock window
        if not is_dir:
            return str(p.absolute())  # absolute() keeps symlink names readable
        # Directory named "*.gguf": fall through to the dir scan below.

    # Case 2: directory containing .gguf files (skip mmproj / MTP drafter)
    if p.is_dir():
        gguf_files = []
        for f in _iter_gguf_files(p):
            context_rel = f"{f.parent.name}/{f.name}"
            quant = _extract_quant_label(context_rel)
            if (
                _is_mmproj(f.name)
                or _is_mtp_drafter(context_rel)
                or _is_big_endian_gguf_path(context_rel, quant)
            ):
                continue
            gguf_files.append(f)
        gguf_files.sort(key = lambda f: f.stat().st_size, reverse = True)
        if gguf_files:
            return str(gguf_files[0].resolve())

    return None


# Preferred GGUF quant levels, descending priority. UD (Unsloth Dynamic)
# variants beat standard quants on quality per bit; repos without UD fall back
# to standard quants. Ordered by size/quality tradeoff, not raw quality.
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
    """Pick the best GGUF file: quant levels in _GGUF_QUANT_PREFERENCE order, else first .gguf."""
    gguf_files = [f for f in filenames if f.lower().endswith(".gguf")]
    if not gguf_files:
        return None

    for quant in _GGUF_QUANT_PREFERENCE:
        for f in gguf_files:
            if quant in f:
                return f

    return gguf_files[0]


@dataclass
class GgufVariantInfo:
    """A single GGUF quantization variant from a HuggingFace repo."""

    filename: str  # e.g., "gemma-3-4b-it-Q4_K_M.gguf"
    quant: str  # e.g., "Q4_K_M" (extracted from filename)
    size_bytes: int  # file size


def _extract_quant_label(filename: str) -> str:
    """
    Extract quant label like Q4_K_M, IQ4_XS, BF16 from a GGUF filename.

    Examples:
        "gemma-3-4b-it-Q4_K_M.gguf"          → "Q4_K_M"
        "model-IQ4_NL.gguf"                   → "IQ4_NL"
        "model-BF16.gguf"                     → "BF16"
        "model-UD-IQ1_S.gguf"                 → "UD-IQ1_S"
        "model-UD-TQ1_0.gguf"                 → "UD-TQ1_0"
        "MXFP4_MOE/model-MXFP4_MOE-0001.gguf"→ "MXFP4_MOE"
        "Qwen3.6-IQ4_XS-3.53bpw.gguf"         → "IQ4_XS-3.53bpw"
    """
    import re

    basename = filename.rsplit("/", 1)[-1]
    # Strip .gguf and any shard suffix (-00001-of-00010)
    stem = re.sub(r"-\d{3,}-of-\d{3,}", "", basename.rsplit(".", 1)[0])
    quant_re = (
        r"(UD-)?"  # Optional UD- prefix (Ultra Discrete)
        r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"  # MXFP variants: MXFP4, MXFP4_MOE
        r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"  # IQ variants: IQ4_XS, IQ4_NL, IQ1_S
        r"|TQ[0-9]+_[0-9]+"  # Ternary quant: TQ1_0, TQ2_0
        r"|Q[0-9]+_K_[A-Z]+"  # K-quant: Q4_K_M, Q3_K_S
        r"|Q[0-9]+_[0-9]+"  # Standard: Q8_0, Q5_1
        r"|Q[0-9]+_K"  # Short K-quant: Q6_K
        r"|BF16|F16|F32)"  # Full precision
        # Optional bits-per-weight modifier so repos that ship multiple
        # files at the same base quant (e.g. byteshape's IQ4_XS at 3.53,
        # 3.97, 4.19 bpw) don't collapse into a single merged variant.
        r"(-[0-9]+(?:\.[0-9]+)?bpw)?"
    )
    match = re.search(quant_re, stem, re.IGNORECASE)
    # Subdir layouts like ``BF16/foo.gguf`` keep the quant in the directory,
    # not the basename. Check parent dirs too so the label matches the
    # snapshot-relative path produced elsewhere.
    if not match and "/" in filename:
        parents = filename.rsplit("/", 1)[0]
        for segment in reversed(parents.split("/")):
            m = re.search(quant_re, segment, re.IGNORECASE)
            if m:
                match = m
                break
    if match:
        prefix = match.group(1) or ""
        bpw = match.group(3) or ""
        return f"{prefix}{match.group(2)}{bpw}"
    # Fallback: last hyphen-separated segment
    return stem.split("-")[-1]


_BIG_ENDIAN_GGUF_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)
_GGUF_KNOWN_QUANT_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|Q[0-9]+_[0-9]+"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)


def _is_big_endian_gguf_path(path: str, quant: str = "") -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0].lower()
    quant_key = quant.strip().lower()
    quant_index = stem.find(quant_key) if quant_key else -1
    parent = normalized.rsplit("/", 1)[0].lower() if "/" in normalized else ""
    quant_in_parent_only = (
        bool(parent)
        and quant_index < 0
        and (
            (quant_key and quant_key in parent)
            or (not quant_key and _GGUF_KNOWN_QUANT_RE.search(parent) is not None)
        )
    )
    for match in _BIG_ENDIAN_GGUF_FILENAME_RE.finditer(stem):
        if quant_index >= 0 and quant_index < match.start():
            return True
        tail = stem[match.end() :].lstrip("._-")
        if not tail or _GGUF_KNOWN_QUANT_RE.search(tail) is None:
            return not quant_in_parent_only
    return False


def _local_gguf_companion_search_root(selected_path: str, gguf_file: str) -> str:
    """Directory to scan upward from for local GGUF companion files."""
    import re

    selected = Path(selected_path)
    gguf_path = Path(gguf_file)
    if selected.suffix.lower() != ".gguf":
        return selected_path

    gguf_dir = gguf_path.parent
    if not gguf_dir.name:
        return str(gguf_dir)

    quant_dir_re = (
        r"(UD-)?("
        r"MXFP[0-9]+(?:_[A-Z0-9]+)*"
        r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
        r"|TQ[0-9]+_[0-9]+"
        r"|Q[0-9]+_K_[A-Z]+"
        r"|Q[0-9]+_[0-9]+"
        r"|Q[0-9]+_K"
        r"|BF16|F16|F32"
        r")"
    )
    if re.fullmatch(quant_dir_re, gguf_dir.name, re.IGNORECASE):
        return str(gguf_dir.parent)
    return str(gguf_dir)


def _iter_hf_cache_snapshots(repo_id: str):
    """Yield HF cache snapshot dirs for *repo_id*, newest first.

    Empty if HF_HUB_CACHE is missing, the repo isn't cached, or has no
    snapshots. Repo name match is case-insensitive to handle casing drift
    between download time and lookup.
    """
    try:
        from huggingface_hub import constants as hf_constants
    except Exception:
        return

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    target = f"models--{repo_id.replace('/', '--')}".lower()
    repo_dirs: list[Path] = []
    try:
        if not cache_dir.is_dir():
            return
        for entry in cache_dir.iterdir():
            if entry.is_dir() and entry.name.lower() == target:
                repo_dirs.append(entry)
    except OSError:
        return
    if not repo_dirs:
        return

    snap_dirs: list[Path] = []
    for repo_dir in repo_dirs:
        snapshots = repo_dir / "snapshots"
        try:
            if snapshots.is_dir():
                for snap_dir in snapshots.iterdir():
                    try:
                        if snap_dir.is_dir():
                            snap_dirs.append(snap_dir)
                    except OSError:
                        continue
        except OSError:
            continue
    if not snap_dirs:
        return
    snap_dirs_with_mtime = []
    for snap_dir in snap_dirs:
        try:
            snap_dirs_with_mtime.append((snap_dir.stat().st_mtime, snap_dir))
        except OSError:
            continue
    snap_dirs_with_mtime.sort(key = lambda item: item[0], reverse = True)
    yield from (snap_dir for _, snap_dir in snap_dirs_with_mtime)


def _list_gguf_variants_from_hf_cache(repo_id: str) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    """Variants from the local HF cache snapshot, or None if not cached.

    A newer snapshot can hold only a companion file (for example a vision
    projector fetched on demand) while the quant files live in an older
    snapshot. Returning the first snapshot that merely reports a vision flag
    would shadow those real variants, so keep scanning older snapshots for
    actual variants and carry the vision flag across snapshots.
    """
    any_vision = False
    for snap in _iter_hf_cache_snapshots(repo_id):
        variants, has_vision = list_local_gguf_variants(str(snap))
        any_vision = any_vision or has_vision
        if variants:
            return variants, any_vision
    if any_vision:
        return [], True
    return None


def list_gguf_variants(
    repo_id: str, hf_token: Optional[str] = None
) -> tuple[list[GgufVariantInfo], bool]:
    """List all GGUF quant variants in a HF repo.

    Separates main model files from mmproj (vision projection) files; mmproj
    presence flags a vision-capable model.

    Returns:
        (variants, has_vision): non-mmproj GGUF variants + vision flag.
    """
    from huggingface_hub import model_info as hf_model_info

    # Offline: skip the API and serve from cache
    if _env_offline():
        cached = _list_gguf_variants_from_hf_cache(repo_id)
        if cached is not None:
            return cached

    try:
        info = hf_model_info(repo_id, token = hf_token, files_metadata = True)
    except Exception as e:
        # Permanent errors (deleted/gated/bad revision) must surface to the
        # caller; serving stale cache would mask the real cause. Matches the
        # early-return in ``detect_gguf_model_remote``.
        if type(e).__name__ in (
            "RepositoryNotFoundError",
            "GatedRepoError",
            "RevisionNotFoundError",
            "EntryNotFoundError",
        ):
            raise
        # API failed transiently; fall back to local snapshot if fully downloaded.
        cached = _list_gguf_variants_from_hf_cache(repo_id)
        if cached is not None:
            logger.warning(
                "HF API unreachable for %s (%s); using local cache snapshot.",
                repo_id,
                e.__class__.__name__,
            )
            return cached
        raise
    variants: list[GgufVariantInfo] = []
    has_vision = False

    quant_totals: dict[str, int] = {}  # quant -> total bytes
    quant_first_file: dict[str, str] = {}  # quant -> first filename (display)

    for sibling in info.siblings:
        fname = sibling.rfilename
        if not fname.lower().endswith(".gguf"):
            continue
        size = sibling.size or 0

        # mmproj files are vision projections, not main model files
        if "mmproj" in fname.lower():
            has_vision = True
            continue
        # MTP drafters are speculative-decoding companions, not quants.
        if _is_mtp_drafter(fname):
            continue

        quant = _extract_quant_label(fname)
        if _is_big_endian_gguf_path(fname, quant):
            continue
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

    # Sort by size descending (largest = best quality first); pinning and OOM
    # demotion happen client-side where GPU VRAM info exists.
    variants.sort(key = lambda v: -v.size_bytes)

    return variants, has_vision


def _resolve_gguf_dir(p: Path) -> Optional[Path]:
    """Resolve a path to the directory containing GGUF variants.

    Directory *p* returns directly. A ``.gguf`` file whose parent dir has
    model metadata (``config.json`` or ``adapter_config.json``) returns the
    parent -- all GGUFs there belong to the same model. Returns ``None`` for
    loose standalone GGUFs (no config) to avoid cross-wiring unrelated models.
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


def list_local_gguf_variants(directory: str) -> tuple[list[GgufVariantInfo], bool]:
    """List GGUF quant variants in a local directory.

    Like :func:`list_gguf_variants` but reads the filesystem. Aggregates shard
    sizes by quant label so split GGUFs appear as one variant.

    Returns:
        (variants, has_vision): non-mmproj GGUF variants + vision flag.
    """
    p = _resolve_gguf_dir(Path(directory))
    if p is None:
        return [], False

    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}
    has_vision = False

    # Recurse so variant-specific subdirs (e.g. ``BF16/...gguf`` used by
    # some HF GGUF repos for the largest quants) are picked up. Result
    # filenames keep the relative subpath so ``_find_local_gguf_by_variant``
    # can locate the file again.
    for f in sorted(_iter_gguf_files(p, recursive = True)):
        if _is_mmproj(f.name):
            has_vision = True
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        # Use the relative path so ``BF16/foo.gguf`` and ``Q4_K_M/foo.gguf``
        # get distinct quant labels instead of collapsing on basename.
        rel = f.relative_to(p).as_posix()
        if _is_mtp_drafter(rel):
            continue
        quant = _extract_quant_label(rel)
        if _is_big_endian_gguf_path(rel, quant):
            continue
        quant_totals[quant] = quant_totals.get(quant, 0) + size
        if quant not in quant_first_file:
            quant_first_file[quant] = rel

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

    For sharded GGUFs (multiple files sharing a quant label), returns the
    first shard (sorted by name), which is what ``llama-server -m`` expects.

    Returns the resolved absolute path, or ``None`` if no match.
    """
    p = _resolve_gguf_dir(Path(directory))
    if p is None:
        return None

    # Recurse so variants under a quant-named subdir (e.g.
    # ``BF16/foo-BF16-00001-of-00002.gguf``) are found. Match the relative
    # path so the quant label can come from the dir name when the basename
    # omits it.
    matches = []
    for f in _iter_gguf_files(p, recursive = True):
        rel = f.relative_to(p).as_posix()
        if _is_mmproj(f.name) or _is_mtp_drafter(rel):
            continue
        quant = _extract_quant_label(rel)
        if quant != variant or _is_big_endian_gguf_path(rel, quant):
            continue
        matches.append(f)
    matches.sort()
    if matches:
        return str(matches[0].resolve())
    return None


def _detect_gguf_from_hf_cache(repo_id: str) -> Optional[str]:
    """Best GGUF filename for *repo_id* from the local HF cache, or None.

    Excludes mmproj (vision projector) files so a partial cache holding only
    the projector cannot route it as the main model.
    """
    for snap in _iter_hf_cache_snapshots(repo_id):
        rel_files = []
        for f in _iter_gguf_files(snap, recursive = True):
            rel = f.relative_to(snap).as_posix()
            quant = _extract_quant_label(rel)
            if _is_mmproj(f.name) or _is_mtp_drafter(rel) or _is_big_endian_gguf_path(rel, quant):
                continue
            rel_files.append(rel)
        if rel_files:
            return _pick_best_gguf(rel_files)
    return None


def detect_gguf_model_remote(repo_id: str, hf_token: Optional[str] = None) -> Optional[str]:
    """Return the best GGUF filename in a HF repo, or None.

    Retries (3 attempts, 1s/2s/4s backoff) on transient HF Hub failures: a
    silent None would make the caller treat a GGUF-only repo as non-GGUF and
    fall through to MLX on Apple Silicon. Offline falls back to the local cache.
    """
    import time
    from huggingface_hub import model_info as hf_model_info

    if _env_offline():
        cached = _detect_gguf_from_hf_cache(repo_id)
        if cached is not None:
            return cached

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            info = hf_model_info(repo_id, token = hf_token)
            repo_files = []
            for sibling in info.siblings:
                fname = sibling.rfilename
                if not fname.lower().endswith(".gguf"):
                    continue
                quant = _extract_quant_label(fname)
                if (
                    _is_mmproj(fname)
                    or _is_mtp_drafter(fname)
                    or _is_big_endian_gguf_path(fname, quant)
                ):
                    continue
                repo_files.append(fname)
            return _pick_best_gguf(repo_files)
        except Exception as e:
            last_err = e
            # 404 / RepoNotFound is permanent -- don't retry
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

    # All attempts failed; fall back to local cache for offline users.
    cached = _detect_gguf_from_hf_cache(repo_id)
    if cached is not None:
        logger.warning(
            "HF API unreachable for '%s' (%s); using local cache to detect GGUF.",
            repo_id,
            type(last_err).__name__ if last_err else "unknown",
        )
        return cached

    logger.warning(f"Could not check GGUF files for '{repo_id}' after 3 attempts: {last_err}")
    return None


def download_gguf_file(
    repo_id: str,
    filename: str,
    hf_token: Optional[str] = None,
) -> str:
    """Download a specific GGUF file from a HF repo; returns the local path."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id = repo_id,
        filename = filename,
        token = hf_token,
    )
    return local_path


# Cache embedding detection per session to avoid repeated HF API calls
_embedding_detection_cache: Dict[tuple, bool] = {}


def _embedding_marker_in_hf_cache(repo_id: str) -> Optional[bool]:
    """Sentence-transformers detection from the local HF cache, no network call.

    True/False when the ACTIVE cached revision carries / lacks the
    ``modules.json`` marker (the same signal ``is_embedding_model`` uses for
    local paths), None when the repo is not in the cache (or the cache is
    unreadable). The revision ``refs/main`` resolves to is authoritative when
    recorded: the cache keeps snapshots of older revisions, and a repo that
    later stopped (or started) being a sentence-transformers model must be
    judged by its current revision, not any historical one. Snapshots are only
    scanned newest-first when no ref exists. Never raises -- a cache mutating
    underneath (concurrent model deletion) reads as not-cached so callers keep
    their normal fallback."""
    try:
        snapshots = list(_iter_hf_cache_snapshots(repo_id))
        if not snapshots:
            return None
        # Prefer the snapshot refs/main points at (the active revision).
        snapshots_dir = snapshots[0].parent
        try:
            commit = (snapshots_dir.parent / "refs" / "main").read_text(encoding = "utf-8").strip()
        except OSError:
            commit = ""  # no ref recorded: fall back to the newest-first scan
        if commit:
            # A ref is recorded, so it is authoritative. If its snapshot is not
            # materialized (partial download / pruning) treat the repo as not
            # cached (None) rather than scanning stale history -- the exact
            # stale-cache class this helper avoids.
            preferred = snapshots_dir / commit
            if not preferred.is_dir():
                return None
            return (preferred / "modules.json").is_file()
        for snap in snapshots:
            try:
                if (snap / "modules.json").is_file():
                    return True
            except OSError:
                continue
        return False
    except Exception:
        return None


def is_embedding_model(model_name: str, hf_token: Optional[str] = None) -> bool:
    """Detect embedding/sentence-transformer models via HF metadata.

    Combines three signals: "sentence-transformers" or "feature-extraction" in
    tags, or pipeline_tag in {"sentence-similarity", "feature-extraction"}.
    Catches models like gte-modernbert whose library_name is "transformers".

    Args:
        model_name: Model identifier (HF repo or local path)
        hf_token: Optional HF token for gated/private models

    Returns:
        True if embedding model, else False (default for local paths or errors).
    """
    cache_key = (model_name, hf_token)
    if cache_key in _embedding_detection_cache:
        return _embedding_detection_cache[cache_key]

    # Local paths: check for sentence-transformer marker (modules.json)
    if is_local_path(model_name):
        local_dir = normalize_path(model_name)
        is_emb = os.path.isfile(os.path.join(local_dir, "modules.json"))
        _embedding_detection_cache[cache_key] = is_emb
        return is_emb

    # Prefer the local HF cache: a sentence-transformers repo carries
    # modules.json in its snapshot, so an already-downloaded model needs no
    # network call. This also lets an offline / HF_HUB_OFFLINE session classify a
    # cached model instead of hanging on a model_info() request that fails with a
    # DNS error and only ever gets retried (#6817).
    cache_hit = _embedding_marker_in_hf_cache(model_name)
    if cache_hit is True:
        _embedding_detection_cache[cache_key] = True
        logger.info(f"Model {model_name} detected as embedding model via HF cache (modules.json)")
        return True
    if _env_offline():
        # Offline: the cache is the only source; anything not positively a
        # sentence-transformers model is treated as non-embedding rather than
        # making a network call that cannot succeed. Do NOT cache this negative:
        # it is offline-conditional. A tag-only (feature-extraction) embedder can
        # only be confirmed online, and the same (model_name, hf_token) key is
        # reused for online lookups after the env var clears in this process.
        return False

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


def scan_trained_models(outputs_dir: str = str(outputs_root())) -> List[Tuple[str, str, str]]:
    """Scan outputs folder for trained Studio models.

    Returns:
        List of (display_name, model_path, model_type), where model_type is
        "lora" for adapter runs or "merged" for full finetunes.
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

        # Sort by mtime, newest first
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
    """Scan exports folder for exported models (merged, LoRA, GGUF).

    Supports two layouts: two-level {run}/{checkpoint}/ (merged & LoRA) and
    flat {name}-finetune-gguf/ (GGUF).

    Returns:
        List of (display_name, model_path, export_type, base_model), where
        export_type is "lora" | "merged" | "gguf".
    """
    results = []
    exports_path = resolve_export_dir(exports_dir)

    if not exports_path.exists():
        return results

    try:
        for run_dir in exports_path.iterdir():
            if not run_dir.is_dir():
                continue

            # Flat GGUF export (e.g. exports/gemma-3-4b-it-finetune-gguf/).
            # Skip mmproj (vision projection) files — not loadable as main models.
            gguf_files = [f for f in _iter_gguf_files(run_dir) if not _is_mmproj(f.name)]
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
                model_path = str(gguf_files[0])
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
                    # checkpoint_dir first, then run_dir (export.py writes
                    # metadata to the top-level export dir)
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

                # Fallback: base model from ./outputs/{run_name}/adapter_config.json
                if not base_model:
                    outputs_adapter_cfg = resolve_output_dir(run_dir.name) / "adapter_config.json"
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
                    logger.info("Detected base model from adapter_config.json: %s", base_model)
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

        # TODO: torch.load default weights_only=True (torch >= 2.6) rejects pickled TrainingArguments; re-enable via safe_globals or weights_only=False once threat model allows.
        # training_args_path = checkpoint_path_obj / "training_args.bin"
        # if training_args_path.exists():
        #     try:
        #         import torch
        #
        #         training_args = torch.load(training_args_path)
        #         if hasattr(training_args, "model_name_or_path"):
        #             base_model = training_args.model_name_or_path
        #             logger.info(
        #                 "Detected base model from training_args.bin: %s", base_model
        #             )
        #             return base_model
        #     except Exception as e:
        #         logger.warning(f"Could not load training_args.bin: {e}")

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
    """Read the base model name from a LoRA adapter's config, or None."""
    try:
        lora_path_obj = Path(lora_path)

        if not _looks_like_lora_adapter(lora_path_obj):
            return None

        # adapter_config.json first
        adapter_config_path = lora_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    logger.info(f"Detected base model from adapter_config.json: {base_model}")
                    return base_model

        # Fallback: try training_args.bin (requires torch)
        # TODO: torch.load default weights_only=True (torch >= 2.6) rejects pickled TrainingArguments; also an remote code execution sink for third-party LoRAs via this route, re-enable behind a trust check if needed.
        # training_args_path = lora_path_obj / "training_args.bin"
        # if training_args_path.exists():
        #     try:
        #         import torch
        #
        #         training_args = torch.load(training_args_path)
        #         if hasattr(training_args, "model_name_or_path"):
        #             base_model = training_args.model_name_or_path
        #             logger.info(
        #                 f"Detected base model from training_args.bin: {base_model}"
        #             )
        #             return base_model
        #     except Exception as e:
        #         logger.warning(f"Could not load training_args.bin: {e}")

        # Last resort: parse from dir name (unsloth_<model>_<timestamp>)
        dir_name = lora_path_obj.name
        if dir_name.startswith("unsloth_"):
            parts = dir_name.split("_")
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


def get_base_model_from_lora_identifier(
    identifier: str, hf_token: Optional[str] = None
) -> Optional[str]:
    """Resolve a LoRA adapter's base model for a LOCAL dir OR a REMOTE HF repo.

    ``get_base_model_from_lora`` only reads a local adapter directory (it requires
    ``is_dir()``). The SECURITY gates must also follow a *remote* adapter's base,
    because the base model's code / weights are what execute on load: an attacker's
    adapter repo can point ``base_model_name_or_path`` at a base carrying a poisoned
    pickle or HIGH auto_map code. For a remote repo id we fetch ONLY the small
    ``adapter_config.json`` (metadata; never a weight file) and read the base. Use
    this in the gate paths so a remote LoRA base is scanned, not just the adapter.

    Returns the base model id, or ``None`` when the identifier is not a LoRA adapter
    or the base cannot be determined (the caller still scans the identifier itself).

    A genuine 404 (no ``adapter_config.json`` / repo absent) is distinguished from a
    transient error: the latter is retried once, then logged as a WARNING (a missed
    base would be scanned by neither gate), so a network blip does not silently and
    invisibly skip the base.
    """
    # Local path: reuse the existing directory reader (identical behavior).
    try:
        if is_local_path(identifier):
            return get_base_model_from_lora(identifier)
    except Exception:
        return get_base_model_from_lora(identifier)

    # Remote repo id: read base_model_name_or_path from adapter_config.json only.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    last_exc = None
    for _attempt in range(2):  # one retry: a transient blip must not skip the base
        try:
            cfg_path = hf_hub_download(
                identifier, "adapter_config.json", token = hf_token if hf_token else None
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            # No adapter_config.json -> not a resolvable LoRA; caller scans the identifier.
            return None
        except Exception as exc:  # transient / auth / network -> retry once
            last_exc = exc
            continue
        try:
            with open(cfg_path, "r") as f:
                base_model = json.load(f).get("base_model_name_or_path")
        except Exception as exc:
            logger.warning("Could not parse adapter_config.json for '%s': %s", identifier, exc)
            return None
        if base_model:
            logger.info(
                "Detected base model from remote adapter_config.json (%s): %s",
                identifier,
                base_model,
            )
        return base_model  # may be None if the key is absent (still a valid answer)

    # Both attempts failed transiently: log loudly -- a missed base is gated by neither gate.
    logger.warning(
        "Could not resolve remote LoRA base for '%s' after retry (%s); its base, if "
        "any, will not be added to the security scan targets.",
        identifier,
        type(last_exc).__name__ if last_exc else "unknown",
    )
    return None


# Status indicators that appear in UI dropdowns
UI_STATUS_INDICATORS = [" (Ready)", " (Loading...)", " (Active)", "↓ "]


def load_model_defaults(model_name: str) -> Dict[str, Any]:
    """Load default training parameters for a model from a YAML file.

    Looks in configs/model_defaults/ (incl. subfolders) by model name or its
    MODEL_NAME_MAPPING aliases, else falls back to default.yaml. Returns the
    parameter dict, or {} if none found.
    """
    # No model selected yet (or a non-string id): nothing to load. Guard before
    # the .lower() calls below so this doesn't raise and get logged as
    # "Error loading model defaults for None: 'NoneType' object has no attribute
    # 'lower'".
    if not isinstance(model_name, str) or not model_name:
        return {}
    try:
        script_dir = Path(__file__).parent.parent.parent
        defaults_dir = script_dir / "assets" / "configs" / "model_defaults"

        # Check the mapping first
        if model_name.lower() in _REVERSE_MODEL_MAPPING:
            canonical_file = _REVERSE_MODEL_MAPPING[model_name.lower()]
            for config_path in defaults_dir.rglob(canonical_file):
                if config_path.is_file():
                    with open(config_path, "r", encoding = "utf-8") as f:
                        config = yaml.safe_load(f) or {}
                        logger.info(f"Loaded model defaults from {config_path} (via mapping)")
                        return config

        # For local paths (e.g. /home/.../Spark-TTS-0.5B/LLM from
        # adapter_config.json, or C:\Users\...\model on Windows), match the
        # last 1-2 path components against the registry (e.g. "Spark-TTS-0.5B/LLM").
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

        # Exact model name match (backward compatibility). For local paths,
        # use only the dir basename to avoid passing absolute paths (e.g.
        # C:\...) into rglob, which raises "Non-relative patterns are
        # unsupported" on Windows.
        _lookup_name = Path(_normalized).name if _is_local_path else model_name
        model_filename = _lookup_name.replace("/", "_") + ".yaml"
        # Search subfolders and root
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
    """Configuration for a model to load."""

    identifier: str  # Clean model identifier (org/name or path)
    display_name: str  # Original UI display name
    path: str  # Normalized filesystem path
    is_local: bool  # Local file vs HF model?
    is_cached: bool  # Already in HF cache?
    is_vision: bool  # Vision model?
    is_lora: bool  # LoRA adapter?
    is_gguf: bool = False  # GGUF model?
    is_audio: bool = False  # TTS audio model?
    audio_type: Optional[str] = None  # Audio codec type: 'snac', 'csm', 'bicodec', 'dac'
    has_audio_input: bool = False  # Accepts audio input (ASR/speech understanding)
    gguf_file: Optional[str] = None  # Full path to the .gguf file (local mode)
    gguf_mmproj_file: Optional[str] = None  # Full path to the mmproj .gguf file (vision projection)
    gguf_mtp_file: Optional[str] = None  # Full path to the separate MTP drafter (local mode)
    gguf_hf_repo: Optional[str] = (
        None  # HF repo ID for -hf mode (e.g. "unsloth/gemma-3-4b-it-GGUF")
    )
    gguf_variant: Optional[str] = None  # Quantization variant (e.g. "Q4_K_M")
    base_model: Optional[str] = None  # Base model (for LoRAs)

    @classmethod
    def from_lora_path(
        cls,
        lora_path: str,
        hf_token: Optional[str] = None,
    ) -> Optional["ModelConfig"]:
        """Create ModelConfig from a local LoRA adapter path, auto-detecting the
        base model from adapter config.

        Args:
            lora_path: Path to the LoRA adapter directory
            hf_token: HF token for vision detection
        """
        try:
            lora_path_obj = Path(lora_path)

            if not lora_path_obj.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return None

            base_model = get_base_model_from_lora(lora_path)
            if not base_model:
                logger.error(f"Could not determine base model for LoRA: {lora_path}")
                return None

            is_vision = is_vision_model(base_model, hf_token = hf_token)
            audio_type = detect_audio_type(base_model, hf_token = hf_token)

            display_name = lora_path_obj.name
            identifier = lora_path  # path is the identifier for local LoRAs

            return cls(
                identifier = identifier,
                display_name = display_name,
                path = lora_path,
                is_local = True,
                is_cached = True,  # local LoRAs are always cached
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
        """Create ModelConfig from a clean model identifier (HF repo or local
        path), for FastAPI routes that send sanitized paths.

        Args:
            model_id: Clean model identifier (HF repo name or local path)
            hf_token: Optional HF token for vision detection on gated models
            is_lora: Whether this is a LoRA adapter
            gguf_variant: Optional GGUF quant variant (e.g. "Q4_K_M") to load
                via -hf for remote repos; None auto-selects via _pick_best_gguf().

        Returns:
            ModelConfig or None if it cannot be created.
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

        # Reuse a cached case-variant's exact repo_id spelling to avoid
        # one-time re-downloads after #2592.
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

                # Vision: check base model, then look for mmproj
                mmproj_file = None
                gguf_is_vision = False
                gguf_dir = Path(gguf_file).parent

                # Is this a vision model, per export metadata?
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

                # Direct file selections may point into a quant subdir while
                # mmproj-*.gguf lives at the snapshot root.
                companion_root = _local_gguf_companion_search_root(path, gguf_file)
                mmproj_file = detect_mmproj_file(gguf_file, search_root = companion_root)
                if mmproj_file:
                    gguf_is_vision = True
                    logger.info(f"Detected mmproj for vision: {mmproj_file}")
                elif base_is_vision:
                    logger.warning(f"Base model is vision but no mmproj file found in {gguf_dir}")

                # Separate MTP drafter sibling (Gemma 4), mirroring mmproj.
                mtp_file = detect_mtp_file(gguf_file, search_root = companion_root)
                if mtp_file:
                    logger.info(f"Detected MTP drafter: {mtp_file}")

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
                    gguf_mtp_file = mtp_file,
                )
        else:
            # Does the HF repo contain GGUF files?
            gguf_filename = detect_gguf_model_remote(identifier, hf_token = hf_token)
            if gguf_filename:
                # Preflight: verify llama-server binary exists before a multi-GB
                # download. include_denied: a transiently locked binary still
                # exists (the lock clears long before the download finishes; the
                # load itself reports a still-locked binary distinctly).
                from core.inference.llama_cpp import (
                    LLAMA_SERVER_NOT_FOUND_DETAIL,
                    LlamaCppBackend,
                    LlamaServerNotFoundError,
                )

                if not LlamaCppBackend._find_llama_server_binary(include_denied = True):
                    raise LlamaServerNotFoundError(LLAMA_SERVER_NOT_FOUND_DETAIL)

                # list_gguf_variants() detects vision & resolves the variant
                variants, has_vision = list_gguf_variants(identifier, hf_token = hf_token)
                variant = gguf_variant
                if not variant:  # auto-select best quant
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

        # Auto-detect LoRA for local paths (adapter_config.json on disk)
        if not is_lora and is_local:
            detected_base = (
                get_base_model_from_lora(path) if _looks_like_lora_adapter(Path(path)) else None
            )
            if detected_base:
                is_lora = True
                logger.info(f"Auto-detected local LoRA adapter at '{path}' (base: {detected_base})")

        # Auto-detect LoRA for remote HF models. When offline, huggingface_hub
        # raises OfflineModeIsEnabled in ~0ms; we fall through to the cache.
        if not is_lora and not is_local:
            try:
                from huggingface_hub import model_info as hf_model_info

                info = hf_model_info(identifier, token = hf_token)
                repo_files = [s.rfilename for s in info.siblings]
                if "adapter_config.json" in repo_files:
                    is_lora = True
                    logger.info(f"Auto-detected remote LoRA adapter: '{identifier}'")
            except Exception as e:
                logger.debug(f"Could not check remote LoRA status for '{identifier}': {e}")

            # API may have failed; adapter_config.json could still be cached.
            if not is_lora:
                for snap in _iter_hf_cache_snapshots(identifier):
                    if (snap / "adapter_config.json").is_file():
                        is_lora = True
                        logger.info(f"Auto-detected cached LoRA adapter: '{identifier}'")
                        break

        # Handle LoRA adapters
        base_model = None
        if is_lora:
            if is_local:
                # Local LoRA: read adapter_config.json from disk
                base_model = get_base_model_from_lora(path)
            else:
                # Remote LoRA: fetch adapter_config.json from HF
                try:
                    from huggingface_hub import hf_hub_download

                    config_path = hf_hub_download(identifier, "adapter_config.json", token = hf_token)
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
        """Create a ModelConfig from UI dropdown/search selections (base models and LoRAs)."""
        selected = None
        if search_value and search_value.strip():
            selected = search_value.strip()
        elif dropdown_value:
            selected = dropdown_value

        if not selected:
            return None

        display_name = selected

        # Resolve display names via the 'local_models' parameter
        if " (Active)" in selected or " (Ready)" in selected:
            clean_display_name = selected.replace(" (Active)", "").replace(" (Ready)", "")
            if local_models:
                for local_display, local_path in local_models:
                    if local_display == clean_display_name:
                        selected = local_path
                        break

        # Strip all UI status indicators to get the final identifier
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

        # Keep existing local GGUF selections on the llama-server path. This
        # constructor is still used by older inference helpers and must not
        # describe a .gguf weight file as loadable by FastVisionModel.
        if is_local and not is_lora and detect_gguf_model(path):
            gguf_config = cls.from_identifier(path, hf_token = hf_token)
            if gguf_config is not None:
                gguf_config.display_name = display_name
                return gguf_config

        # --- Base Model and Vision Detection ---
        base_model = None
        is_vision = False

        if is_lora:
            # A LoRA MUST have a base model.
            base_model = get_base_model_from_lora(path)
            if not base_model:
                logger.warning(
                    f"Could not determine base model for LoRA '{path}'. Cannot create config."
                )
                return None  # cannot proceed without a base model

            # A LoRA's vision capability comes from its base model.
            is_vision = is_vision_model(base_model, hf_token = hf_token)
        else:
            # Base model: check its own vision status.
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
            base_model = base_model,  # None for base models, set for LoRAs
        )
