# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speech-to-text checkpoints the Transformers dictation engine runs through mlx-audio (Apple Silicon,
which gets mlx-audio with mlx-vlm). Chosen by config model_type: mlx-audio loads both mlx-community
quants and the original safetensors. Whisper keeps its Transformers path."""

from __future__ import annotations

import functools
import importlib.util

# Families proven end to end through the dictation worker; mlx-audio knows more.
MLX_STT_MODEL_TYPES = frozenset({"qwen3_asr"})


@functools.lru_cache(maxsize = 1)
def mlx_stt_available() -> bool:
    from utils.hardware.hardware import is_apple_silicon
    if not is_apple_silicon():
        return False
    try:
        return importlib.util.find_spec("mlx_audio") is not None
    except (ImportError, ValueError):
        return False


def is_mlx_stt_config(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type.strip().lower() not in MLX_STT_MODEL_TYPES:
        return False
    return mlx_stt_available()
