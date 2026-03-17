# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dataset utilities package.

This package provides utilities for dataset format detection, conversion,
and processing for LLM and VLM fine-tuning workflows.

Modules:
- format_detection: Detect dataset formats (Alpaca, ShareGPT, ChatML)
- format_conversion: Convert between dataset formats
- chat_templates: Apply chat templates to datasets
- vlm_processing: Vision-Language Model processing utilities
- data_collators: Custom data collators for training
- model_mappings: Model-to-template mapping constants
"""

# Format detection
from .format_detection import (
    detect_dataset_format,
    detect_custom_format_heuristic,
    detect_multimodal_dataset,
    detect_vlm_dataset_structure,
)

# Format conversion
from .format_conversion import (
    standardize_chat_format,
    convert_chatml_to_alpaca,
    convert_alpaca_to_chatml,
    convert_to_vlm_format,
    convert_llava_to_vlm_format,
    convert_sharegpt_with_images_to_vlm_format,
)

# Chat templates
from .chat_templates import (
    apply_chat_template_to_dataset,
    get_dataset_info_summary,
    get_tokenizer_chat_template,
    DEFAULT_ALPACA_TEMPLATE,
)

# VLM processing
from .vlm_processing import (
    generate_smart_vlm_instruction,
)

# Data collators
from .data_collators import (
    DataCollatorSpeechSeq2SeqWithPadding,
    DeepSeekOCRDataCollator,
    VLMDataCollator,
)

# Model mappings (constants)
from .model_mappings import (
    TEMPLATE_TO_MODEL_MAPPER,
    MODEL_TO_TEMPLATE_MAPPER,
    TEMPLATE_TO_RESPONSES_MAPPER,
)

# Legacy imports from the original dataset_utils.py for backward compatibility
# These functions have not yet been refactored into separate modules
from .dataset_utils import (
    check_dataset_format,
    format_and_template_dataset,
    format_dataset,
)

# Public API
__all__ = [
    # Detection
    "detect_dataset_format",
    "detect_custom_format_heuristic",
    "detect_multimodal_dataset",
    "detect_vlm_dataset_structure",
    # Conversion
    "standardize_chat_format",
    "convert_chatml_to_alpaca",
    "convert_alpaca_to_chatml",
    "convert_to_vlm_format",
    "convert_llava_to_vlm_format",
    "convert_sharegpt_with_images_to_vlm_format",
    # Templates
    "apply_chat_template_to_dataset",
    "get_dataset_info_summary",
    "get_tokenizer_chat_template",
    "DEFAULT_ALPACA_TEMPLATE",
    # VLM
    "generate_smart_vlm_instruction",
    # Collators
    "DataCollatorSpeechSeq2SeqWithPadding",
    "DeepSeekOCRDataCollator",
    "VLMDataCollator",
    # Mappings
    "TEMPLATE_TO_MODEL_MAPPER",
    "MODEL_TO_TEMPLATE_MAPPER",
    "TEMPLATE_TO_RESPONSES_MAPPER",
    # Main entry points
    "check_dataset_format",
    "format_and_template_dataset",
    "format_dataset",
]
