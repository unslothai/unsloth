# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from typing import Any, Optional


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


def _keyword_in_column(keyword: str, col_name: str) -> bool:
    return (
        re.search(r"\b" + re.escape(keyword) + r"\b", col_name, re.IGNORECASE)
        is not None
    )


def _unknown_dataset_format(
    chat_column: Optional[str] = None, sample_keys: Optional[list[str]] = None
) -> dict:
    return {
        "format": "unknown",
        "chat_column": chat_column,
        "needs_standardization": None,
        "sample_keys": sample_keys or [],
    }


def detect_dataset_format(dataset) -> dict:
    sample = _first_row(dataset)
    if sample is None:
        return _unknown_dataset_format()
    column_names = set(sample.keys())
    if {"instruction", "output"}.issubset(column_names):
        return {
            "format": "alpaca",
            "chat_column": None,
            "needs_standardization": False,
            "sample_keys": [],
        }

    chat_column = None
    if "messages" in column_names:
        chat_column = "messages"
    elif "conversations" in column_names:
        chat_column = "conversations"
    elif "texts" in column_names:
        chat_column = "texts"

    if not chat_column:
        return _unknown_dataset_format()

    chat_data = sample.get(chat_column)
    if not isinstance(chat_data, (list, tuple)) or not chat_data:
        return _unknown_dataset_format(chat_column)
    first_msg = chat_data[0]
    if not isinstance(first_msg, dict):
        return _unknown_dataset_format(chat_column)
    msg_keys = set(first_msg.keys())
    sample_keys = [str(key) for key in msg_keys]
    if "from" in msg_keys or "value" in msg_keys:
        return {
            "format": "sharegpt",
            "chat_column": chat_column,
            "needs_standardization": True,
            "sample_keys": sample_keys,
        }
    if "role" in msg_keys and "content" in msg_keys:
        return {
            "format": "chatml",
            "chat_column": chat_column,
            "needs_standardization": False,
            "sample_keys": sample_keys,
        }
    return _unknown_dataset_format(chat_column, sample_keys)


def detect_custom_format_heuristic(dataset):
    sample = _first_row(dataset)
    if sample is None:
        return None
    all_columns = list(sample.keys())
    mapping = {}
    assistant_words = [
        "output",
        "answer",
        "response",
        "assistant",
        "completion",
        "expected",
        "recommendation",
        "reply",
        "result",
        "target",
        "solution",
        "explanation",
        "solve",
    ]
    user_words_high_priority = [
        "input",
        "question",
        "query",
        "prompt",
        "instruction",
        "request",
        "snippet",
        "user",
        "text",
        "problem",
        "exercise",
    ]
    user_words_low_priority = ["task"]
    user_words = user_words_high_priority + user_words_low_priority
    system_words = [
        "system",
        "context",
        "description",
        "persona",
        "role",
        "template",
        "task",
    ]
    metadata_exact_match = {
        "id",
        "idx",
        "index",
        "key",
        "timestamp",
        "date",
        "metadata",
        "source",
        "kind",
        "type",
        "category",
        "score",
        "label",
        "tag",
        "inference_mode",
    }
    metadata_prefix_patterns = [
        "problem_type",
        "problem_source",
        "generation_model",
        "pass_rate",
    ]
    priority_patterns = {
        "generated": 100,
        "gen_": 90,
        "model_": 80,
        "predicted": 70,
        "completion": 60,
    }

    def has_keyword(col_name, keywords):
        col_lower = col_name.lower()
        col_normalized = col_lower.replace("_", "").replace("-", "").replace(" ", "")
        return any(
            keyword in col_lower or keyword in col_normalized for keyword in keywords
        )

    def is_metadata(col_name):
        col_lower = col_name.lower()
        if col_lower in metadata_exact_match or col_lower in metadata_prefix_patterns:
            return True
        for pattern in metadata_prefix_patterns:
            if (
                col_lower.startswith(pattern.split("_")[0] + "_")
                and col_lower != pattern
            ):
                if "_" in col_lower:
                    prefix = col_lower.split("_")[0]
                    if prefix in ["generation", "pass", "inference"]:
                        return True
        return len(col_lower) <= 2 and col_lower not in ["qa", "q", "a"]

    def get_priority_score(col_name):
        col_lower = col_name.lower()
        return sum(
            score
            for pattern, score in priority_patterns.items()
            if pattern in col_lower
        )

    def get_content_length(col_name):
        try:
            return len(str(sample[col_name])) if sample.get(col_name) else 0
        except Exception:
            return 0

    def score_column(col_name, keywords, role_type, num_candidates):
        if not has_keyword(col_name, keywords):
            return 0
        score = 10
        if role_type == "user":
            col_lower = col_name.lower()
            if "task" in col_lower and not any(
                kw in col_lower for kw in user_words_high_priority
            ):
                score -= 15
        score += get_priority_score(col_name)
        if role_type in ["assistant", "user"]:
            avg_length = get_content_length(col_name)
            if num_candidates > 1:
                if avg_length > 1000:
                    score += 50
                elif avg_length > 200:
                    score += 30
                elif avg_length > 50:
                    score += 10
                elif avg_length < 50:
                    score -= 20
            else:
                if avg_length > 1000:
                    score += 50
                elif avg_length > 200:
                    score += 30
                elif avg_length > 50:
                    score += 10
        return score

    content_columns = [col for col in all_columns if not is_metadata(col)]
    assistant_potential = [
        col for col in content_columns if has_keyword(col, assistant_words)
    ]
    user_potential = [col for col in content_columns if has_keyword(col, user_words)]
    assistant_candidates = [
        (col, score)
        for col in assistant_potential
        if (
            score := score_column(
                col, assistant_words, "assistant", len(assistant_potential)
            )
        )
        > 0
    ]
    if assistant_candidates:
        assistant_candidates.sort(key = lambda item: item[1], reverse = True)
        assistant_col = assistant_candidates[0][0]
        mapping[assistant_col] = "assistant"
    else:
        assistant_col = None

    user_candidates = []
    for col in user_potential:
        if col == assistant_col:
            continue
        score = score_column(col, user_words, "user", len(user_potential))
        if score > 0:
            user_candidates.append((col, score))
    if user_candidates:
        user_candidates.sort(key = lambda item: item[1], reverse = True)
        user_col = user_candidates[0][0]
        mapping[user_col] = "user"
    else:
        user_col = None

    remaining_columns = [col for col in content_columns if col not in mapping]
    system_col = None
    for col in remaining_columns:
        if has_keyword(col, system_words):
            mapping[col] = "system"
            system_col = col
            break
    if system_col:
        remaining_columns = [col for col in remaining_columns if col != system_col]
    if remaining_columns:
        remaining_col = remaining_columns[0]
        if not has_keyword(remaining_col, user_words + assistant_words):
            mapping[remaining_col] = "system"
        elif user_col is None:
            mapping[remaining_col] = "user"
        else:
            mapping[remaining_col] = "system"

    has_user = any(role == "user" for role in mapping.values())
    has_assistant = any(role == "assistant" for role in mapping.values())
    if not has_user:
        for col in remaining_columns:
            if col not in mapping:
                mapping[col] = "user"
                has_user = True
                break
    return mapping if has_user and has_assistant else None


_AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".webm",
)


def _is_audio_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return True
        if "bytes" in value or "path" in value:
            path = value.get("path") or ""
            return isinstance(path, str) and any(
                path.lower().endswith(ext) for ext in _AUDIO_EXTENSIONS
            )
    return False


def _has_image_header(data: bytes) -> bool:
    if len(data) < 4:
        return False
    return (
        data[:2] == b"\xff\xd8"
        or data[:4] == b"\x89PNG"
        or data[:3] == b"GIF"
        or (data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP")
        or data[:2] == b"BM"
    )


def _is_image_value(value) -> bool:
    if value is None:
        return False
    try:
        from PIL.Image import Image as PILImage
        if isinstance(value, PILImage):
            return True
    except ImportError:
        pass
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return False
        if "bytes" in value and "path" in value:
            path = value.get("path") or ""
            if isinstance(path, str) and any(
                path.lower().endswith(ext) for ext in _AUDIO_EXTENSIONS
            ):
                return False
            return True
    if isinstance(value, (bytes, bytearray)):
        return _has_image_header(value)
    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg")
    if isinstance(value, str) and len(value) < 1000:
        lower = value.strip().lower()
        if lower.startswith(("http://", "https://")):
            return any(lower.split("?")[0].endswith(ext) for ext in image_exts)
        return any(lower.endswith(ext) for ext in image_exts)
    return False


def detect_multimodal_dataset(dataset):
    sample = _first_row(dataset)
    if sample is None:
        return {
            "is_image": False,
            "multimodal_columns": [],
            "modality_types": [],
            "is_audio": False,
            "audio_columns": [],
            "detected_audio_column": None,
            "detected_text_column": None,
            "detected_speaker_column": None,
        }
    column_names = list(sample.keys())
    image_keywords = [
        "image",
        "img",
        "pixel",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "gif",
        "tiff",
        "svg",
        "photo",
        "pic",
        "picture",
        "visual",
        "file_name",
        "filename",
    ]
    audio_keywords = ["audio", "speech", "wav", "waveform", "sound"]
    multimodal_columns = []
    audio_columns = []
    modality_types = set()
    for col_name in column_names:
        if any(_keyword_in_column(keyword, col_name) for keyword in image_keywords):
            multimodal_columns.append(col_name)
            modality_types.add("image")
    for col_name in column_names:
        if col_name not in multimodal_columns and _is_image_value(sample[col_name]):
            multimodal_columns.append(col_name)
            modality_types.add("image")
    for col_name in column_names:
        if any(_keyword_in_column(keyword, col_name) for keyword in audio_keywords):
            audio_columns.append(col_name)
            modality_types.add("audio")
    for col_name in column_names:
        if col_name not in audio_columns and _is_audio_value(sample[col_name]):
            audio_columns.append(col_name)
            modality_types.add("audio")
    if audio_columns:
        multimodal_columns = [
            col for col in multimodal_columns if col not in set(audio_columns)
        ]

    detected_text_col = None
    if audio_columns:
        for col_name in column_names:
            if col_name.lower() in [
                "text",
                "sentence",
                "transcript",
                "transcription",
                "label",
            ]:
                detected_text_col = col_name
                break
    detected_speaker_col = None
    if audio_columns:
        for col_name in column_names:
            if col_name.lower() in ["source", "speaker", "speaker_id"]:
                detected_speaker_col = col_name
                break
    return {
        "is_image": len(multimodal_columns) > 0,
        "multimodal_columns": multimodal_columns,
        "modality_types": list(modality_types),
        "is_audio": len(audio_columns) > 0,
        "audio_columns": audio_columns,
        "detected_audio_column": audio_columns[0] if audio_columns else None,
        "detected_text_column": detected_text_col,
        "detected_speaker_column": detected_speaker_col,
    }


def detect_vlm_dataset_structure(dataset):
    sample = _first_row(dataset)
    if sample is None:
        return {
            "format": "unknown",
            "needs_conversion": None,
            "image_column": None,
            "text_column": None,
            "messages_column": None,
        }
    column_names = set(sample.keys())
    if "messages" in column_names:
        messages = sample["messages"]
        if messages and len(messages) > 0:
            first_msg = messages[0]
            if "content" in first_msg:
                content = first_msg["content"]
                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], dict)
                    and "type" in content[0]
                ):
                    has_index = any(
                        "index" in item for item in content if isinstance(item, dict)
                    )
                    if has_index and "images" in column_names:
                        return {
                            "format": "vlm_messages_llava",
                            "needs_conversion": True,
                            "messages_column": "messages",
                            "image_column": "images",
                            "text_column": None,
                        }
                    has_image = any(
                        "image" in item for item in content if isinstance(item, dict)
                    )
                    if has_image:
                        return {
                            "format": "vlm_messages",
                            "needs_conversion": False,
                            "messages_column": "messages",
                            "image_column": None,
                            "text_column": None,
                        }

    for chat_col in ("conversations", "messages"):
        if chat_col not in column_names:
            continue
        chat_data = sample[chat_col]
        if not isinstance(chat_data, list) or not chat_data:
            continue
        has_image_placeholder = any(
            "<image>" in str(message.get("value", "") or message.get("content", ""))
            for message in chat_data
            if isinstance(message, dict)
        )
        if not has_image_placeholder:
            continue
        image_col = next(
            (
                col
                for col in column_names
                if col != chat_col
                and (_keyword_in_column("image", col) or _keyword_in_column("img", col))
            ),
            None,
        )
        if image_col:
            return {
                "format": "sharegpt_with_images",
                "needs_conversion": True,
                "image_column": image_col,
                "text_column": None,
                "messages_column": chat_col,
            }

    metadata_suffixes = (
        "_id",
        "_url",
        "_name",
        "_filename",
        "_uri",
        "_link",
        "_key",
        "_index",
    )
    metadata_prefixes = (
        "id_",
        "url_",
        "name_",
        "filename_",
        "uri_",
        "link_",
        "key_",
        "index_",
    )
    image_keywords = [
        "image",
        "img",
        "photo",
        "picture",
        "pic",
        "visual",
        "scan",
        "file_name",
        "filename",
    ]
    text_keywords = [
        "text",
        "caption",
        "captions",
        "description",
        "answer",
        "output",
        "response",
        "label",
    ]

    def is_metadata_column(col_name):
        lower = col_name.lower()
        return any(lower.endswith(suffix) for suffix in metadata_suffixes) or any(
            lower.startswith(prefix) for prefix in metadata_prefixes
        )

    image_candidates = []
    for col in column_names:
        value = sample[col]
        if any(
            _keyword_in_column(keyword, col) for keyword in image_keywords
        ) or _is_image_value(value):
            if hasattr(value, "size") and hasattr(value, "mode"):
                score = 100
            elif isinstance(value, dict) and ("bytes" in value or "path" in value):
                score = 75
            elif isinstance(value, str):
                score = (
                    55
                    if is_metadata_column(col)
                    else 70
                    if value.startswith(("http://", "https://"))
                    else 50
                )
            else:
                score = 0
            if score > 0:
                image_candidates.append((col, score))
    image_candidates.sort(key = lambda item: item[1], reverse = True)

    text_candidates = []
    for col in column_names:
        if is_metadata_column(col) or not any(
            _keyword_in_column(keyword, col) for keyword in text_keywords
        ):
            continue
        value = sample[col]
        if isinstance(value, str) and value:
            text_candidates.append((col, min(len(value), 1000)))
        elif isinstance(value, list) and value and isinstance(value[0], str):
            text_candidates.append((col, min(len(value[0]), 1000) // 2))
    text_candidates.sort(key = lambda item: item[1], reverse = True)

    found_image = image_candidates[0][0] if image_candidates else None
    found_text = text_candidates[0][0] if text_candidates else None
    if found_image and found_text:
        return {
            "format": "simple_image_text",
            "needs_conversion": True,
            "image_column": found_image,
            "text_column": found_text,
            "messages_column": None,
        }
    return {
        "format": "unknown",
        "needs_conversion": None,
        "image_column": found_image,
        "text_column": found_text,
        "messages_column": None,
    }


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

    if is_vlm:
        vlm_structure = detect_vlm_dataset_structure(dataset)
        requires_mapping = vlm_structure["format"] == "unknown"
        warning = None
        if requires_mapping:
            missing = []
            if not vlm_structure.get("image_column"):
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
        "is_image": multimodal_info["is_image"],
        "multimodal_columns": multimodal_info.get("multimodal_columns"),
        **audio_fields,
    }


_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "input": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "output": "assistant",
    "system": "system",
}


def _standardize_sharegpt_row(row: dict[str, Any], chat_column: str) -> dict[str, Any]:
    chat_data = row.get(chat_column)
    if not isinstance(chat_data, list):
        return row
    messages = []
    for message in chat_data:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = (
            message.get("content") if "content" in message else message.get("value")
        )
        messages.append(
            {
                "role": _ROLE_MAP.get(str(role), str(role or "user")),
                "content": "" if content is None else content,
            }
        )
    return {chat_column: messages}


def format_dataset_preview(dataset):
    detected = detect_dataset_format(dataset)
    if detected.get("format") != "sharegpt":
        return dataset
    chat_column = detected.get("chat_column")
    if not isinstance(chat_column, str):
        return dataset

    if hasattr(dataset, "map"):
        return dataset.map(lambda row: _standardize_sharegpt_row(row, chat_column))
    return dataset
