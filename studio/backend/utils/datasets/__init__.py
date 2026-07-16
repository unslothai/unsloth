# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset utilities for LLM/VLM fine-tuning: detection, conversion, templating, collators, mappings."""

from .format_detection import (
    detect_dataset_format,
    detect_custom_format_heuristic,
    detect_multimodal_dataset,
    detect_vlm_dataset_structure,
)

from .format_conversion import (
    standardize_chat_format,
    convert_chatml_to_alpaca,
    convert_alpaca_to_chatml,
    convert_to_vlm_format,
    convert_llava_to_vlm_format,
    convert_sharegpt_with_images_to_vlm_format,
)

from .chat_templates import (
    apply_chat_template_to_dataset,
    get_dataset_info_summary,
    get_tokenizer_chat_template,
    DEFAULT_ALPACA_TEMPLATE,
)

from .vlm_processing import (
    generate_smart_vlm_instruction,
)

from .data_collators import (
    DataCollatorSpeechSeq2SeqWithPadding,
    DeepSeekOCRDataCollator,
    VLMDataCollator,
)

from .model_mappings import (
    TEMPLATE_TO_MODEL_MAPPER,
    MODEL_TO_TEMPLATE_MAPPER,
    TEMPLATE_TO_RESPONSES_MAPPER,
    is_gpt_oss_model_name,
)

# Legacy dataset_utils.py imports kept for backward compat
from .dataset_utils import (
    check_dataset_format,
    format_and_template_dataset,
    format_dataset,
)

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
    "is_gpt_oss_model_name",
    # Main entry points
    "check_dataset_format",
    "format_and_template_dataset",
    "format_dataset",
]
