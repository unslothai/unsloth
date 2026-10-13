# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speech-to-text checkpoints the Transformers dictation engine runs through mlx-audio (Apple Silicon,
which gets mlx-audio with mlx-vlm). Chosen by config model_type: mlx-audio loads both mlx-community
quants and the original safetensors. Whisper keeps its Transformers path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# Families proven end to end through the dictation worker; mlx-audio knows more.
MLX_STT_MODEL_TYPES = frozenset({"qwen3_asr"})


_found = False


def mlx_stt_available() -> bool:
    # Only a hit is kept: the MLX self-repair can install mlx-audio while Studio runs.
    global _found
    if not _found:
        from utils.hardware.hardware import is_apple_silicon
        try:
            _found = is_apple_silicon() and importlib.util.find_spec("mlx_audio") is not None
        except (ImportError, ValueError):
            _found = False
    return _found


def is_mlx_stt_config(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type.strip().lower() not in MLX_STT_MODEL_TYPES:
        return False
    # Qwen3-ForcedAligner shares model_type qwen3_asr but needs a transcript to align, not audio alone.
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict) and thinker.get("model_type") == "qwen3_forced_aligner":
        return False
    return mlx_stt_available()


def ships_custom_code(snapshot: Path) -> bool:
    # mlx-audio loads the tokenizer with trust_remote_code=True, so a repo's own Python would run.
    snapshot = Path(snapshot)
    if any(snapshot.rglob("*.py")):
        return True
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        try:
            data = json.loads((snapshot / name).read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "auto_map" in data:
            return True
    return False
