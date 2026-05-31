# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Model and LoRA configuration handling
"""

from .model_config import (
    ModelConfig,
    GgufVariantInfo,
    is_vision_model,
    is_embedding_model,
    detect_audio_type,
    is_audio_input_type,
    VALID_AUDIO_TYPES,
    scan_trained_models,
    scan_exported_models,
    get_base_model_from_checkpoint,
    load_model_defaults,
    get_base_model_from_lora,
    load_model_config,
    list_gguf_variants,
    extract_model_size_b,
    MODEL_NAME_MAPPING,
    UI_STATUS_INDICATORS,
)
from .checkpoints import scan_checkpoints

scan_trained_loras = scan_trained_models

__all__ = [
    "ModelConfig",
    "GgufVariantInfo",
    "is_vision_model",
    "is_embedding_model",
    "detect_audio_type",
    "is_audio_input_type",
    "VALID_AUDIO_TYPES",
    "scan_trained_models",
    "scan_trained_loras",
    "scan_exported_models",
    "get_base_model_from_checkpoint",
    "load_model_defaults",
    "get_base_model_from_lora",
    "load_model_config",
    "list_gguf_variants",
    "extract_model_size_b",
    "MODEL_NAME_MAPPING",
    "UI_STATUS_INDICATORS",
    "scan_checkpoints",
]
