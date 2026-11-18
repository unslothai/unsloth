# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Optional

from utils.datasets.cells import unwrap_message_column
from utils.datasets.format_detection import (
    detect_custom_format_heuristic,
    detect_dataset_format,
    detect_multimodal_dataset,
    detect_vlm_dataset_structure,
    is_plain_text_dataset,
)
from utils.datasets.pretokenized import PRETOKENIZED_FORMAT, is_pretokenized_dataset


def _first_row(dataset) -> Optional[dict]:
    try:
        row = next(iter(dataset))
    except StopIteration:
        return None
    return row if isinstance(row, dict) else None


def _column_names(dataset, sample: Optional[dict] = None) -> list[str]:
    names = getattr(dataset, "column_names", None)
    if names is not None:
        return list(names)
    return list((sample or {}).keys())


def check_dataset_format(dataset, is_vlm: bool = False) -> dict:
    sample = _first_row(dataset)
    columns = _column_names(dataset, sample)
    multimodal_info = detect_multimodal_dataset(dataset)
    is_audio = multimodal_info.get("is_audio", False)
    audio_fields = {
        "is_audio": is_audio,
        "detected_audio_column": multimodal_info.get("detected_audio_column"),
        "detected_speaker_column": multimodal_info.get("detected_speaker_column"),
    }

    if is_plain_text_dataset(dataset):
        # Before the VLM branch: text-only data trains as raw text on a vision model too.
        # Media flags off: a name match (a "speech" column) would route an audio model to its audio path.
        return {
            "requires_manual_mapping": False,
            "detected_format": "raw",
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": None,
            "detected_text_column": None,
            "is_image": False,
            "multimodal_columns": None,
            "is_audio": False,
            "detected_audio_column": None,
            "detected_speaker_column": None,
        }

    if is_vlm:
        vlm_structure = detect_vlm_dataset_structure(dataset)
        requires_mapping = vlm_structure["format"] == "unknown"
        warning = None
        if requires_mapping:
            missing = []
            if not (vlm_structure.get("image_column") or vlm_structure.get("video_column")):
                missing.append("image")
            if not vlm_structure.get("text_column"):
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
        detected_audio = multimodal_info.get("detected_audio_column")
        detected_text = multimodal_info.get("detected_text_column")
        return {
            "requires_manual_mapping": not detected_audio or not detected_text,
            "detected_format": "audio",
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": None,
            "detected_text_column": detected_text,
            "is_image": False,
            "multimodal_columns": multimodal_info.get("audio_columns"),
            **audio_fields,
        }

    detected = detect_dataset_format(dataset)
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
        if is_pretokenized_dataset(dataset):
            return {
                "requires_manual_mapping": False,
                "detected_format": PRETOKENIZED_FORMAT,
                "columns": columns,
                "suggested_mapping": None,
                "detected_image_column": None,
                "detected_text_column": None,
                "is_image": multimodal_info["is_image"],
                "multimodal_columns": multimodal_info.get("multimodal_columns"),
                **audio_fields,
            }
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

    return {
        "requires_manual_mapping": False,
        "detected_format": detected["format"],
        "columns": columns,
        "suggested_mapping": None,
        "detected_image_column": None,
        "detected_text_column": None,
        "chat_column": detected.get("chat_column"),
        "is_image": multimodal_info["is_image"],
        "multimodal_columns": multimodal_info.get("multimodal_columns"),
        **audio_fields,
    }


# Aliases standardize_data_formats accepts; keys are normalised, so use _normalize_role_alias.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "input": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "output": "assistant",
    "system": "system",
}


def _normalize_role_alias(role: Any) -> str:
    """Match aliases the way the trainer does: `role.strip().lower()`, as
    `standardize_data_formats` compares them (unslothai/unsloth-zoo#1225)."""
    if role is None:
        return ""
    return str(role).strip().lower()


def _standardize_sharegpt_row(row: dict[str, Any], chat_column: str) -> dict[str, Any]:
    chat_data = row.get(chat_column)
    if not isinstance(chat_data, list):
        return row
    messages = []
    for message in chat_data:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = message.get("content")
        if content is None:
            content = message.get("value")
        normalized = _normalize_role_alias(role)
        new_message = {key: value for key, value in message.items() if key not in ("from", "value")}
        new_message.update(
            {
                # Unknown alias shown as written; blank falls back to "user", as before.
                "role": _ROLE_MAP.get(normalized, str(role)) if normalized else "user",
                "content": "" if content is None and not message.get("tool_calls") else content,
            }
        )
        messages.append(new_message)
    return {chat_column: messages}


def format_dataset_preview(dataset):
    detected = detect_dataset_format(dataset)
    chat_column = detected.get("chat_column")
    if (
        detected.get("format") not in ("chatml", "sharegpt")
        or not isinstance(chat_column, str)
        or not hasattr(dataset, "map")
    ):
        return dataset
    dataset = unwrap_message_column(dataset, chat_column)
    if detected["format"] == "sharegpt":
        return dataset.map(lambda row: _standardize_sharegpt_row(row, chat_column))
    return dataset
