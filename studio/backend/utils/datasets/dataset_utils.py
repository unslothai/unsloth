# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dataset utilities for format detection, conversion, and template application.

This module provides the main entry points for dataset processing:
- check_dataset_format: Lightweight check if manual mapping is needed (for frontend)
- format_dataset: Detects and normalizes dataset formats
- format_and_template_dataset: End-to-end processing with chat template application

All internal utilities have been moved to separate modules:
- format_detection: detect_dataset_format, detect_multimodal_dataset, etc.
- format_conversion: standardize_chat_format, convert_chatml_to_alpaca, etc.
- chat_templates: apply_chat_template_to_dataset, get_tokenizer_chat_template, etc.
- vlm_processing: generate_smart_vlm_instruction
- data_collators: DeepSeekOCRDataCollator, VLMDataCollator
- model_mappings: TEMPLATE_TO_MODEL_MAPPER
"""

import json

# Import from modular files
from .format_detection import (
    detect_dataset_format,
    detect_multimodal_dataset,
    detect_vlm_dataset_structure,
    detect_custom_format_heuristic,
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
from .raw_text import prepare_raw_text_dataset
from .vlm_processing import generate_smart_vlm_instruction
from .data_collators import DeepSeekOCRDataCollator, VLMDataCollator
from .model_mappings import TEMPLATE_TO_MODEL_MAPPER
from loggers import get_logger

logger = get_logger(__name__)


def check_dataset_format(dataset, is_vlm: bool = False) -> dict:
    """
    Lightweight format check without processing - for frontend validation.

    Use this to quickly determine if user needs to manually map columns
    before calling the full format_and_template_dataset().

    Args:
        dataset: HuggingFace dataset
        is_vlm: Whether this is a Vision-Language Model dataset

    Returns:
        dict: {
            "requires_manual_mapping": bool - True if user must map columns,
            "detected_format": str - The detected format,
            "columns": list - Available column names for mapping UI,
            "suggested_mapping": dict or None - Auto-detected mapping if available,
            "detected_image_column": str or None - For VLM only,
            "detected_text_column": str or None - For VLM only,
        }
    """
    columns = (
        list(dataset.column_names)
        if hasattr(dataset, "column_names")
        else list(next(iter(dataset)).keys())
    )

    # Auto-detect multimodal data regardless of is_vlm flag
    multimodal_info = detect_multimodal_dataset(dataset)
    is_audio = multimodal_info.get("is_audio", False)

    # Common audio fields for all return paths
    audio_fields = {
        "is_audio": is_audio,
        "detected_audio_column": multimodal_info.get("detected_audio_column"),
        "detected_speaker_column": multimodal_info.get("detected_speaker_column"),
    }

    if is_vlm:
        vlm_structure = detect_vlm_dataset_structure(dataset)
        requires_mapping = vlm_structure["format"] == "unknown"

        warning = None
        if requires_mapping:
            img_col = vlm_structure.get("image_column")
            txt_col = vlm_structure.get("text_column")
            missing = []
            if not img_col:
                missing.append("image")
            if not txt_col:
                missing.append("text")
            if missing:
                warning = (
                    f"Could not auto-detect {' or '.join(missing)} column. "
                    "Please assign image and text columns manually."
                )

        return {
            "requires_manual_mapping": requires_mapping,
            "detected_format": vlm_structure["format"],
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": vlm_structure.get("image_column"),
            "detected_text_column": vlm_structure.get("text_column"),
            "is_image": multimodal_info["is_image"],
            "multimodal_columns": multimodal_info.get("multimodal_columns"),
            "warning": warning,
            **audio_fields,
        }

    if is_audio:
        # Audio dataset — require manual mapping only when columns can't be auto-detected
        detected_audio = multimodal_info.get("detected_audio_column")
        detected_text = multimodal_info.get("detected_text_column")
        needs_mapping = not detected_audio or not detected_text
        return {
            "requires_manual_mapping": needs_mapping,
            "detected_format": "audio",
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": None,
            "detected_text_column": multimodal_info.get("detected_text_column"),
            "is_image": False,
            "multimodal_columns": multimodal_info.get("audio_columns"),
            **audio_fields,
        }

    # Text / LLM flow
    detected = detect_dataset_format(dataset)

    # If format is unknown, try heuristic detection
    if detected["format"] == "unknown":
        heuristic_mapping = detect_custom_format_heuristic(dataset)
        if heuristic_mapping:
            return {
                "requires_manual_mapping": False,
                "detected_format": "custom_heuristic",
                "columns": columns,
                "suggested_mapping": heuristic_mapping,
                "detected_image_column": None,
                "detected_text_column": None,
                "is_image": multimodal_info["is_image"],
                "multimodal_columns": multimodal_info.get("multimodal_columns"),
                **audio_fields,
            }
        else:
            # Heuristic failed — user must map manually (or use AI Assist)
            return {
                "requires_manual_mapping": True,
                "detected_format": "unknown",
                "columns": columns,
                "suggested_mapping": None,
                "detected_image_column": None,
                "detected_text_column": None,
                "is_image": multimodal_info["is_image"],
                "multimodal_columns": multimodal_info.get("multimodal_columns"),
                "warning": (
                    f"Could not auto-detect column roles for columns: {columns}. "
                    "Please assign roles manually, or use AI Assist."
                ),
                **audio_fields,
            }

    # Known format detected
    return {
        "requires_manual_mapping": False,
        "detected_format": detected["format"],
        "columns": columns,
        "suggested_mapping": None,
        "detected_image_column": None,
        "detected_text_column": None,
        "is_image": multimodal_info["is_image"],
        "multimodal_columns": multimodal_info.get("multimodal_columns"),
        **audio_fields,
    }


# Normalise any format-specific role to canonical chatml (user/assistant/system)
_TO_CHATML = {
    "user": "user",
    "human": "user",
    "instruction": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "output": "assistant",
    "system": "system",
    "input": "system",
}
_CHATML_ROLE_ORDER = ("system", "user", "assistant")
_CHATML_TO_ALPACA = {"user": "instruction", "system": "input", "assistant": "output"}


def _apply_user_mapping(dataset, mapping: dict, batch_size: int = 1000):
    """
    Apply user-provided column mapping to convert dataset to conversations format.

    Accepts chatml (user/assistant/system), sharegpt (human/gpt/system), and
    alpaca (instruction/input/output) role names — all normalised to chatml output.

    If the mapping contains ``__``-prefixed metadata keys (from the conversion
    advisor), routes to template-based conversion instead of simple role mapping.

    Returns:
        Dataset with single 'conversations' column
    """
    # Split metadata from column roles
    meta = {k: v for k, v in mapping.items() if k.startswith("__")}
    column_roles = {k: v for k, v in mapping.items() if not k.startswith("__")}

    if meta:
        return _apply_template_mapping(dataset, column_roles, meta, batch_size)

    # ── Simple mode (original logic) ──
    # Pre-compute: group columns by canonical chatml role
    role_groups: dict[str, list[str]] = {r: [] for r in _CHATML_ROLE_ORDER}
    for col_name, role in column_roles.items():
        canonical = _TO_CHATML.get(role)
        if canonical:
            role_groups[canonical].append(col_name)

    def _convert(examples):
        num = len(next(iter(examples.values())))
        conversations = []
        for i in range(num):
            convo = []
            for chatml_role in _CHATML_ROLE_ORDER:
                for col in role_groups[chatml_role]:
                    if col in examples:
                        content = examples[col][i]
                        convo.append(
                            {
                                "role": chatml_role,
                                "content": str(content) if content else "",
                            }
                        )
            conversations.append(convo)
        return {"conversations": conversations}

    return dataset.map(
        _convert,
        batched = True,
        batch_size = batch_size,
        remove_columns = dataset.column_names,
    )


def _extract_column_value(val, col: str, label_mapping: dict) -> str:
    """Extract a string value from a column, handling complex types and label mapping."""
    # Handle complex types (dicts, lists) — extract useful text instead of raw repr
    if isinstance(val, dict):
        # Common pattern: {"text": [...]} in QA datasets
        if "text" in val:
            inner = val["text"]
            str_val = inner[0] if isinstance(inner, list) and inner else str(inner)
        else:
            str_val = json.dumps(val, ensure_ascii = False)
    elif isinstance(val, list):
        str_val = val[0] if len(val) == 1 else ", ".join(str(v) for v in val)
    else:
        str_val = str(val) if val is not None else ""

    # Apply label mapping if this column has one
    if col in label_mapping and isinstance(label_mapping[col], dict):
        str_val = label_mapping[col].get(str_val, str_val)

    return str_val


def _apply_template_mapping(
    dataset, column_roles: dict, meta: dict, batch_size: int = 1000
):
    """
    Apply advisor-driven mapping for non-conversational datasets.

    Groups columns by their assigned role (user/assistant), concatenates
    values within each role into a single message, and injects an optional
    system prompt.  Label mapping is applied to convert integer labels
    to human-readable strings.

    Returns:
        Dataset with single 'conversations' column
    """
    system_prompt = meta.get("__system_prompt", "")
    label_mapping = meta.get("__label_mapping", {})  # {col: {int_str: label_str}}

    # Group columns by canonical chatml role
    role_groups: dict[str, list[str]] = {"user": [], "assistant": []}
    for col, role in column_roles.items():
        canonical = _TO_CHATML.get(role, role)
        if canonical in role_groups:
            role_groups[canonical].append(col)

    import logging as _log

    _log.getLogger(__name__).info(
        f"Applying role mapping: sys={bool(system_prompt)}, "
        f"user_cols={role_groups['user']}, asst_cols={role_groups['assistant']}, "
        f"label_map={list(label_mapping.keys())}"
    )

    def _convert(examples):
        num = len(next(iter(examples.values())))
        conversations = []
        for i in range(num):
            convo = []

            # System prompt (generated, static across all rows)
            if system_prompt:
                convo.append({"role": "system", "content": system_prompt})

            # User message: concatenate all user-role column values
            user_parts = []
            for col in role_groups["user"]:
                if col in examples:
                    user_parts.append(
                        _extract_column_value(examples[col][i], col, label_mapping)
                    )
            if user_parts:
                convo.append({"role": "user", "content": "\n".join(user_parts)})

            # Assistant message: concatenate all assistant-role column values
            asst_parts = []
            for col in role_groups["assistant"]:
                if col in examples:
                    asst_parts.append(
                        _extract_column_value(examples[col][i], col, label_mapping)
                    )
            if asst_parts:
                convo.append({"role": "assistant", "content": "\n".join(asst_parts)})

            conversations.append(convo)
        return {"conversations": conversations}

    return dataset.map(
        _convert,
        batched = True,
        batch_size = batch_size,
        remove_columns = dataset.column_names,
    )


def _apply_user_mapping_alpaca(dataset, mapping: dict, batch_size: int = 1000):
    """
    Apply user-provided column mapping to convert dataset to Alpaca format.

    Accepts any format's role names — normalises via _TO_CHATML, then maps
    user → instruction, system → input, assistant → output.

    Returns:
        Dataset with instruction/input/output columns
    """
    col_for: dict[str, str | None] = {
        "instruction": None,
        "input": None,
        "output": None,
    }
    for col_name, role in mapping.items():
        canonical = _TO_CHATML.get(role)
        alpaca_field = _CHATML_TO_ALPACA.get(canonical) if canonical else None
        if alpaca_field:
            col_for[alpaca_field] = col_name

    def _convert(examples):
        num = len(next(iter(examples.values())))
        instructions, inputs, outputs = [], [], []
        for i in range(num):
            for field, dest in (
                ("instruction", instructions),
                ("input", inputs),
                ("output", outputs),
            ):
                col = col_for[field]
                val = (
                    str(examples[col][i])
                    if col and col in examples and examples[col][i]
                    else ""
                )
                dest.append(val)
        return {"instruction": instructions, "input": inputs, "output": outputs}

    return dataset.map(
        _convert,
        batched = True,
        batch_size = batch_size,
        remove_columns = dataset.column_names,
    )


def format_dataset(
    dataset,
    format_type = "auto",
    tokenizer = None,
    aliases_for_system = [
        "system",
    ],
    aliases_for_user = [
        "user",
        "human",
        "input",
    ],
    aliases_for_assistant = [
        "gpt",
        "assistant",
        "output",
    ],
    batch_size = 1000,
    num_proc = None,
    auto_detect_custom = True,
    custom_format_mapping = None,
):
    """
    Formats dataset and returns metadata.

    Returns:
        dict: {
            "dataset": processed dataset,
            "detected_format": original format detected,
            "final_format": final format after processing,
            "chat_column": column name with chat data,
            "is_standardized": whether role names are standardized,
            "requires_manual_mapping": True if format detection failed and user must map columns,
            "warnings": list of warning messages
        }
    """

    # Detect multimodal first (needed for all flows)
    multimodal_info = detect_multimodal_dataset(dataset)

    if format_type == "raw":
        raw_result = prepare_raw_text_dataset(dataset)
        return {
            "dataset": raw_result.dataset,
            "detected_format": "raw_text",
            "final_format": "raw_text",
            "chat_column": "text",
            "is_standardized": True,
            "requires_manual_mapping": False,
            "is_image": multimodal_info["is_image"],
            "multimodal_info": multimodal_info,
            "warnings": [notice.message for notice in raw_result.notices],
        }

    # If user provided explicit mapping, skip detection and apply in the requested format
    if custom_format_mapping:
        try:
            if format_type == "alpaca":
                mapped_dataset = _apply_user_mapping_alpaca(
                    dataset, custom_format_mapping, batch_size
                )
                final_format = "alpaca"
                chat_column = None
            else:
                # auto / chatml / sharegpt / conversational — all produce chatml conversations
                # (sharegpt is always standardized to role/content internally)
                mapped_dataset = _apply_user_mapping(
                    dataset, custom_format_mapping, batch_size
                )
                final_format = "chatml_conversations"
                chat_column = "conversations"

            return {
                "dataset": mapped_dataset,
                "detected_format": "user_mapped",
                "final_format": final_format,
                "chat_column": chat_column,
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [
                    f"Applied user-provided column mapping ({format_type}): {custom_format_mapping}"
                ],
            }
        except Exception as e:
            return {
                "dataset": dataset,
                "detected_format": "user_mapped",
                "final_format": "unknown",
                "chat_column": None,
                "is_standardized": False,
                "requires_manual_mapping": True,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [f"Failed to apply user mapping: {e}"],
            }

    # Detect current format
    detected = detect_dataset_format(dataset)
    warnings = []

    # Add multimodal warning if detected
    if multimodal_info["is_image"]:
        warnings.append(
            f"Multimodal dataset detected. Found columns: {multimodal_info['multimodal_columns']}"
        )

    # AUTO MODE: Keep format but standardize if needed
    if format_type == "auto":
        # Alpaca - keep as is
        if detected["format"] == "alpaca":
            return {
                "dataset": dataset,
                "detected_format": "alpaca",
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        # ShareGPT - needs standardization
        elif detected["format"] == "sharegpt":
            try:
                standardized = standardize_chat_format(
                    dataset,
                    tokenizer,
                    aliases_for_system,
                    aliases_for_user,
                    aliases_for_assistant,
                    batch_size,
                    num_proc,
                )
                return {
                    "dataset": standardized,
                    "detected_format": "sharegpt",
                    "final_format": f"chatml_{detected['chat_column']}",
                    "chat_column": detected["chat_column"],
                    "is_standardized": True,
                    "requires_manual_mapping": False,
                    "is_image": multimodal_info["is_image"],
                    "multimodal_info": multimodal_info,
                    "warnings": [],
                }
            except Exception as e:
                warnings.append(f"Failed to standardize ShareGPT format: {e}")
                return {
                    "dataset": dataset,
                    "detected_format": "sharegpt",
                    "final_format": "sharegpt",
                    "chat_column": detected["chat_column"],
                    "is_standardized": False,
                    "requires_manual_mapping": True,
                    "is_image": multimodal_info["is_image"],
                    "multimodal_info": multimodal_info,
                    "warnings": warnings,
                }

        elif detected["format"] == "chatml" and detected["chat_column"] in [
            "conversations",
            "messages",
            "texts",
        ]:
            return {
                "dataset": dataset,
                "detected_format": f"chatml_{detected['chat_column']}",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": warnings,
            }

        # Unknown - try standardization, if fails pass as is
        else:
            warnings.append(
                f"Unknown format detected. Keys found: {detected['sample_keys']}"
            )

            # NEW: Try heuristic detection
            if auto_detect_custom:
                custom_mapping = detect_custom_format_heuristic(dataset)
                if custom_mapping:
                    warnings.append(f"Auto-detected column mapping: {custom_mapping}")

                    def _apply_auto_mapping(examples):
                        conversations = []
                        num_examples = len(examples[list(examples.keys())[0]])

                        # Preserve non-mapped columns
                        all_columns = set(examples.keys())
                        mapped_columns = set(custom_mapping.keys())
                        preserved_columns = {
                            col: examples[col] for col in all_columns - mapped_columns
                        }

                        for i in range(num_examples):
                            convo = []
                            for target_role in ["system", "user", "assistant"]:
                                for col_name, role in custom_mapping.items():
                                    if role == target_role and col_name in examples:
                                        content = examples[col_name][i]
                                        if content and str(content).strip():
                                            convo.append(
                                                {"role": role, "content": str(content)}
                                            )
                            conversations.append(convo)

                        return {"conversations": conversations, **preserved_columns}

                    try:
                        dataset = dataset.map(
                            _apply_auto_mapping, batched = True, batch_size = batch_size
                        )
                        return {
                            "dataset": dataset,
                            "detected_format": "unknown",
                            "final_format": "chatml_conversations",
                            "chat_column": "conversations",
                            "is_standardized": True,
                            "requires_manual_mapping": False,
                            "is_image": multimodal_info["is_image"],
                            "multimodal_info": multimodal_info,
                            "warnings": warnings,
                        }
                    except Exception as e:
                        warnings.append(f"Auto-detection failed: {e}")

            # Try standardization as a last resort
            if detected["chat_column"]:
                try:
                    standardized = standardize_chat_format(
                        dataset,
                        tokenizer,
                        aliases_for_system,
                        aliases_for_user,
                        aliases_for_assistant,
                        batch_size,
                        num_proc,
                    )
                    warnings.append("Successfully standardized unknown format")
                    return {
                        "dataset": standardized,
                        "detected_format": "unknown",
                        "final_format": f"chatml_{detected['chat_column']}",
                        "chat_column": detected["chat_column"],
                        "is_standardized": True,
                        "requires_manual_mapping": False,
                        "is_image": multimodal_info["is_image"],
                        "multimodal_info": multimodal_info,
                        "warnings": warnings,
                    }
                except Exception as e:
                    warnings.append(
                        f"Could not standardize: {e}. Passing dataset as-is."
                    )

            # Return as-is with warnings
            return {
                "dataset": dataset,
                "detected_format": "unknown",
                "final_format": "unknown",
                "chat_column": detected["chat_column"],
                "is_standardized": False,
                "requires_manual_mapping": True,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": warnings,
            }

    # ALPACA MODE: Convert to Alpaca
    elif format_type == "alpaca":
        if detected["format"] == "alpaca":
            return {
                "dataset": dataset,
                "detected_format": "alpaca",
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        elif detected["format"] in ["sharegpt", "chatml"]:
            # First standardize if ShareGPT
            if detected["format"] == "sharegpt":
                dataset = standardize_chat_format(
                    dataset,
                    tokenizer,
                    aliases_for_system,
                    aliases_for_user,
                    aliases_for_assistant,
                    batch_size,
                    num_proc,
                )

            # Then convert to Alpaca
            converted = convert_chatml_to_alpaca(dataset, batch_size, num_proc)
            return {
                "dataset": converted,
                "detected_format": detected["format"],
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        else:
            warnings.append(f"Cannot convert unknown format to Alpaca")
            return {
                "dataset": dataset,
                "detected_format": "unknown",
                "final_format": "unknown",
                "chat_column": detected["chat_column"],
                "is_standardized": False,
                "requires_manual_mapping": True,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": warnings,
            }

    # CHATML MODE: Convert to ChatML
    elif format_type in ["chatml", "conversational", "sharegpt"]:
        if detected["format"] == "alpaca":
            converted = convert_alpaca_to_chatml(dataset, batch_size, num_proc)
            return {
                "dataset": converted,
                "detected_format": "alpaca",
                "final_format": "chatml_conversations",
                "chat_column": "conversations",
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        elif detected["format"] == "sharegpt":
            standardized = standardize_chat_format(
                dataset,
                tokenizer,
                aliases_for_system,
                aliases_for_user,
                aliases_for_assistant,
                batch_size,
                num_proc,
            )
            return {
                "dataset": standardized,
                "detected_format": "sharegpt",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        elif detected["format"] == "chatml":
            return {
                "dataset": dataset,
                "detected_format": f"chatml_{detected['chat_column']}",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": [],
            }

        else:
            warnings.append(f"Unknown format, attempting standardization")
            if detected["chat_column"]:
                try:
                    standardized = standardize_chat_format(
                        dataset,
                        tokenizer,
                        aliases_for_system,
                        aliases_for_user,
                        aliases_for_assistant,
                        batch_size,
                        num_proc,
                    )
                    return {
                        "dataset": standardized,
                        "detected_format": "unknown",
                        "final_format": f"chatml_{detected['chat_column']}",
                        "chat_column": detected["chat_column"],
                        "is_standardized": True,
                        "requires_manual_mapping": False,
                        "is_image": multimodal_info["is_image"],
                        "multimodal_info": multimodal_info,
                        "warnings": warnings,
                    }
                except Exception as e:
                    warnings.append(f"Standardization failed: {e}")

            return {
                "dataset": dataset,
                "detected_format": "unknown",
                "final_format": "unknown",
                "chat_column": detected["chat_column"],
                "is_standardized": False,
                "requires_manual_mapping": True,
                "is_image": multimodal_info["is_image"],
                "multimodal_info": multimodal_info,
                "warnings": warnings,
            }

    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def format_and_template_dataset(
    dataset,
    model_name,
    tokenizer,
    is_vlm = False,
    format_type = "auto",
    # VLM-specific parameters
    vlm_instruction = None,  # Now optional - will auto-generate
    vlm_text_column = None,
    vlm_image_column = None,
    dataset_name = None,
    custom_prompt_template = None,
    add_eos_token = False,
    remove_bos_prefix = False,
    custom_format_mapping = None,
    auto_detect_custom = True,
    auto_detect_mapping = True,
    aliases_for_system = [
        "system",
    ],
    aliases_for_user = [
        "user",
        "human",
        "input",
    ],
    aliases_for_assistant = [
        "gpt",
        "assistant",
        "output",
    ],
    batch_size = 1000,
    num_proc = None,
    progress_callback = None,
):
    """
    Convenience function that combines format_dataset and apply_chat_template_to_dataset.
    Perfect for UI workflows - one function does everything!

    Returns:
        dict: {
            "dataset": Final dataset with 'text' column,
            "detected_format": Original format,
            "final_format": Format after processing,
            "success": Whether template application succeeded,
            "requires_manual_mapping": True if format detection failed and user must map columns,
            "warnings": List of warnings,
            "errors": List of errors,
            "summary": Human-readable summary
        }
    """

    # VLM FLOW
    if is_vlm:
        warnings = []
        errors = []

        multimodal_info = detect_multimodal_dataset(dataset)

        # NEW: If user provided explicit mapping for VLM, use it directly
        if custom_format_mapping:
            # Expect mapping like: {"image_col": "image", "caption_col": "text"}
            user_vlm_image_column = None
            user_vlm_text_column = None

            for col, role in custom_format_mapping.items():
                if role == "image":
                    user_vlm_image_column = col
                elif role in ["text", "user", "caption", "assistant"]:
                    user_vlm_text_column = col

            if user_vlm_image_column and user_vlm_text_column:
                try:
                    dataset = convert_to_vlm_format(
                        dataset,
                        instruction = vlm_instruction,
                        text_column = user_vlm_text_column,
                        image_column = user_vlm_image_column,
                        dataset_name = dataset_name,
                        progress_callback = progress_callback,
                    )
                    warnings.append(
                        f"Applied user VLM mapping: image='{user_vlm_image_column}', text='{user_vlm_text_column}'"
                    )

                    return {
                        "dataset": dataset,
                        "detected_format": "user_mapped",
                        "final_format": "vlm_messages",
                        "chat_column": "messages",
                        "is_vlm": True,
                        "is_image": True,
                        "multimodal_info": multimodal_info,
                        "success": True,
                        "requires_manual_mapping": False,
                        "warnings": warnings,
                        "errors": [],
                    }
                except Exception as e:
                    # User mapping failed — fall back to auto-detection instead
                    # of giving up (handles stale cached mappings gracefully)
                    warnings.append(
                        f"User VLM mapping (image='{user_vlm_image_column}', "
                        f"text='{user_vlm_text_column}') failed: {e} — "
                        f"falling back to auto-detection"
                    )
                    logger.info(
                        f"⚠️ User VLM mapping failed, falling back to auto-detection..."
                    )
                    custom_format_mapping = None  # clear so auto-detection runs below
            else:
                errors.append(
                    f"Invalid VLM mapping: need 'image' and 'text' roles. Got: {custom_format_mapping}"
                )
                return {
                    "dataset": dataset,
                    "detected_format": "user_mapped",
                    "final_format": "vlm_unknown",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Auto-detect VLM structure
        vlm_structure = detect_vlm_dataset_structure(dataset)

        # Handle Llava format
        if vlm_structure["format"] == "vlm_messages_llava":
            try:
                dataset = convert_llava_to_vlm_format(dataset)
                warnings.append(
                    "Converted from Llava format (image indices) to standard VLM format"
                )
            except Exception as e:
                errors.append(f"Failed to convert Llava format: {e}")
                import traceback

                traceback.print_exc()

                return {
                    "dataset": dataset,
                    "detected_format": "vlm_messages_llava",
                    "final_format": "vlm_conversion_failed",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Handle ShareGPT/ChatML + image column (e.g. ShareGPT4V, LLaVA-style)
        elif vlm_structure["format"] == "sharegpt_with_images":
            try:
                dataset = convert_sharegpt_with_images_to_vlm_format(
                    dataset,
                    image_column = vlm_structure["image_column"],
                    messages_column = vlm_structure["messages_column"],
                    dataset_name = dataset_name,
                    progress_callback = progress_callback,
                )
                warnings.append(
                    "Converted from ShareGPT+image format to standard VLM format"
                )
            except Exception as e:
                errors.append(f"Failed to convert ShareGPT+image format: {e}")
                import traceback

                traceback.print_exc()

                return {
                    "dataset": dataset,
                    "detected_format": "sharegpt_with_images",
                    "final_format": "vlm_conversion_failed",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Handle simple format
        elif vlm_structure["needs_conversion"]:
            if vlm_text_column is None:
                vlm_text_column = vlm_structure["text_column"]
            if vlm_image_column is None:
                vlm_image_column = vlm_structure["image_column"]

            if vlm_text_column is None or vlm_image_column is None:
                columns = list(next(iter(dataset)).keys()) if dataset else []
                issues = [
                    f"Could not auto-detect image and text columns from: {columns}",
                    f"VLM structure detected: {vlm_structure.get('format', 'unknown')}",
                ]
                friendly = None
                try:
                    from .llm_assist import llm_generate_dataset_warning

                    friendly = llm_generate_dataset_warning(
                        issues,
                        dataset_name = dataset_name,
                        modality = "vision",
                        column_names = columns,
                    )
                except Exception:
                    pass
                errors.append(
                    friendly
                    or f"Could not auto-detect image/text columns. Found: {vlm_structure}. "
                )
                return {
                    "dataset": dataset,
                    "detected_format": "vlm_unknown",
                    "final_format": "vlm_unknown",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

            try:
                dataset = convert_to_vlm_format(
                    dataset,
                    instruction = vlm_instruction,
                    text_column = vlm_text_column,
                    image_column = vlm_image_column,
                    dataset_name = dataset_name,
                    progress_callback = progress_callback,
                )

                if vlm_instruction:
                    warnings.append(
                        f"Using user-provided instruction: '{vlm_instruction}'"
                    )
                else:
                    warnings.append(
                        "Auto-generated instruction based on dataset analysis"
                    )

            except Exception as e:
                errors.append(f"Failed to convert to VLM format: {e}")
                import traceback

                traceback.print_exc()

                return {
                    "dataset": dataset,
                    "detected_format": vlm_structure["format"],
                    "final_format": "vlm_conversion_failed",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Already in standard VLM format
        elif vlm_structure["format"] == "vlm_messages":
            dataset = [sample for sample in dataset]
            warnings.append("Dataset already in standard VLM messages format")

        # Return as list
        return {
            "dataset": dataset,
            "detected_format": vlm_structure["format"],
            "final_format": "vlm_messages",
            "chat_column": "messages",
            "is_vlm": True,
            "is_image": multimodal_info["is_image"],
            "multimodal_info": multimodal_info,
            "vlm_structure": vlm_structure,
            "success": True,
            "requires_manual_mapping": False,
            "warnings": warnings,
            "errors": errors,
        }

    # LLM FLOW (Existing code)
    else:
        # Step 1: Format the dataset
        n_rows = len(dataset) if hasattr(dataset, "__len__") else None
        if progress_callback and n_rows:
            progress_callback(status_message = f"Formatting dataset ({n_rows:,} rows)...")
        dataset_info = format_dataset(
            dataset,
            format_type = format_type,
            tokenizer = tokenizer,
            auto_detect_custom = auto_detect_custom,
            custom_format_mapping = custom_format_mapping,
            aliases_for_system = aliases_for_system,
            aliases_for_user = aliases_for_user,
            aliases_for_assistant = aliases_for_assistant,
            batch_size = batch_size,
            num_proc = num_proc,
        )

        if dataset_info["final_format"] == "raw_text":
            summary = get_dataset_info_summary(dataset_info)
            return {
                "dataset": dataset_info["dataset"],
                "detected_format": dataset_info["detected_format"],
                "final_format": dataset_info["final_format"],
                "chat_column": dataset_info.get("chat_column"),
                "is_vlm": False,
                "success": True,
                "requires_manual_mapping": False,
                "warnings": dataset_info.get("warnings", []),
                "errors": [],
                "summary": summary,
            }

        # Step 2: Apply chat template
        detected = dataset_info.get("detected_format", "unknown")
        if progress_callback and n_rows:
            progress_callback(
                status_message = f"Applying chat template to {detected} ({n_rows:,} rows)..."
            )
        # Gemma emits a leading <bos> that must be stripped for text-only chatml/sharegpt.
        is_alpaca = format_type == "alpaca" or (
            format_type == "auto" and dataset_info["detected_format"] == "alpaca"
        )
        is_gemma = "gemma" in model_name.lower()
        if is_gemma and not dataset_info["is_image"] and not is_alpaca:
            remove_bos_prefix = True
        template_result = apply_chat_template_to_dataset(
            dataset_info = dataset_info,
            tokenizer = tokenizer,
            model_name = model_name,
            custom_prompt_template = custom_prompt_template,
            add_eos_token = add_eos_token,
            remove_bos_prefix = remove_bos_prefix,
            custom_format_mapping = custom_format_mapping,
            auto_detect_mapping = auto_detect_mapping,
            batch_size = batch_size,
            num_proc = num_proc,
            progress_callback = progress_callback,
        )

        # Step 3: Generate summary
        summary = get_dataset_info_summary(dataset_info)

        # Combine results
        all_warnings = dataset_info.get("warnings", []) + template_result.get(
            "warnings", []
        )
        all_errors = template_result.get("errors", [])

        # If format_dataset returned "unknown" but apply_chat_template rescued
        # it via heuristic detection, update final_format to reflect reality.
        final_format = dataset_info["final_format"]
        requires_manual = dataset_info.get("requires_manual_mapping", False)
        if final_format == "unknown" and template_result["success"]:
            out_ds = template_result["dataset"]
            if hasattr(out_ds, "column_names") and "text" in out_ds.column_names:
                final_format = "chatml_conversations"
                requires_manual = False

        return {
            "dataset": template_result["dataset"],
            "detected_format": dataset_info["detected_format"],
            "final_format": final_format,
            "chat_column": dataset_info.get("chat_column"),
            "is_vlm": False,  # This is LLM flow
            "success": template_result["success"],
            "requires_manual_mapping": requires_manual,
            "warnings": all_warnings,
            "errors": all_errors,
            "summary": summary,
        }
