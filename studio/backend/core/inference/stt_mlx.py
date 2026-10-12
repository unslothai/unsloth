# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which cached speech-to-text checkpoints the Transformers dictation engine serves through mlx-audio.

Apple Silicon only. mlx-audio arrives with mlx-vlm in the Mac install, and loads both the
mlx-community quants and the original Qwen safetensors, so a checkpoint is chosen by its
config's model_type rather than by who published it. Whisper never comes here: it keeps the
Transformers path it has always had.
"""

from __future__ import annotations

import functools
import importlib.util

# Families proven end to end through the dictation worker; mlx-audio knows more.
MLX_STT_MODEL_TYPES = frozenset({"qwen3_asr"})


@functools.lru_cache(maxsize = 1)
def mlx_stt_available() -> bool:
    """True on Apple Silicon with mlx-audio installed. Imports nothing heavy."""
    from utils.hardware.hardware import is_apple_silicon

    if not is_apple_silicon():
        return False
    try:
        return importlib.util.find_spec("mlx_audio") is not None
    except (ImportError, ValueError):
        return False


def is_mlx_stt_config(config: object) -> bool:
    """True when this host runs the checkpoint behind ``config`` through mlx-audio."""
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type.strip().lower() not in MLX_STT_MODEL_TYPES:
        return False
    return mlx_stt_available()
