# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Format detection utilities for dataset processing.

This module contains functions for detecting dataset formats (Alpaca, ShareGPT, ChatML),
detecting multimodal/VLM dataset structures, and heuristic-based column mapping.
"""

import re


def _keyword_in_column(keyword: str, col_name: str) -> bool:
    """Word-boundary keyword match to avoid false positives like 'pic' in 'topic'."""
    return (
        re.search(r"\b" + re.escape(keyword) + r"\b", col_name, re.IGNORECASE)
        is not None
    )


def detect_dataset_format(dataset):
    """
    Detects dataset format by inspecting structure.

    Returns:
        dict: {
            "format": "alpaca" | "sharegpt" | "chatml" | "unknown",
            "chat_column": "messages" | "conversations" | None,
            "needs_standardization": bool,
            "sample_keys": list of keys found in messages (for debugging)
        }
    """
    column_names = set(next(iter(dataset)).keys())

    # Check for Alpaca
    alpaca_columns = {"instruction", "output"}
    if alpaca_columns.issubset(column_names):
        return {
            "format": "alpaca",
            "chat_column": None,
            "needs_standardization": False,
            "sample_keys": [],
        }

    # Check for chat-based formats (messages or conversations)
    chat_column = None
    if "messages" in column_names:
        chat_column = "messages"
    elif "conversations" in column_names:
        chat_column = "conversations"
    elif "texts" in column_names:
        chat_column = "texts"

    if chat_column:
        # Inspect the structure to determine if ShareGPT or ChatML
        try:
            sample = next(iter(dataset))
            chat_data = sample[chat_column]

            if chat_data and len(chat_data) > 0:
                first_msg = chat_data[0]
                msg_keys = set(first_msg.keys())

                # ShareGPT uses "from" and "value"
                if "from" in msg_keys or "value" in msg_keys:
                    return {
                        "format": "sharegpt",
                        "chat_column": chat_column,
                        "needs_standardization": True,
                        "sample_keys": list(msg_keys),
                    }

                # ChatML uses "role" and "content"
                elif "role" in msg_keys and "content" in msg_keys:
                    return {
                        "format": "chatml",
                        "chat_column": chat_column,
                        "needs_standardization": False,
                        "sample_keys": list(msg_keys),
                    }

                # Unknown structure but has chat column
                else:
                    return {
                        "format": "unknown",
                        "chat_column": chat_column,
                        "needs_standardization": None,
                        "sample_keys": list(msg_keys),
                    }
        except Exception as e:
            return {
                "format": "unknown",
                "chat_column": chat_column,
                "needs_standardization": None,
                "sample_keys": [],
                "error": str(e),
            }

    # No recognized format
    return {
        "format": "unknown",
        "chat_column": None,
        "needs_standardization": None,
        "sample_keys": [],
    }


def detect_custom_format_heuristic(dataset):
    """
    Smart detection with priority scoring.

    Strategy for ambiguous keywords like 'task':
    1. Detect assistant first (unambiguous)
    2. Detect user using high-priority keywords first
    3. Check REMAINING columns for system keywords (including 'task')
    4. Only if no system match, use 'task' as fallback user
    """
    sample = next(iter(dataset))
    all_columns = list(sample.keys())

    mapping = {}

    # Keywords
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

    # Split into high/low priority
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
    user_words_low_priority = ["task"]  # Ambiguous - can be user OR system
    user_words = user_words_high_priority + user_words_low_priority

    system_words = [
        "system",
        "context",
        "description",
        "persona",
        "role",
        "template",
        "task",  # Also in system
    ]

    # Metadata columns to ignore
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
        """Check if any keyword appears in column name."""
        col_lower = col_name.lower()
        col_normalized = col_lower.replace("_", "").replace("-", "").replace(" ", "")

        for keyword in keywords:
            if keyword in col_lower or keyword in col_normalized:
                return True
        return False

    def is_metadata(col_name):
        """Check if column is likely metadata."""
        col_lower = col_name.lower()

        if col_lower in metadata_exact_match:
            return True

        if col_lower in metadata_prefix_patterns:
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

        if len(col_lower) <= 2 and not col_lower in ["qa", "q", "a"]:
            return True

        return False

    def get_priority_score(col_name):
        """Calculate priority score based on column name patterns."""
        col_lower = col_name.lower()
        score = 0

        for pattern, pattern_score in priority_patterns.items():
            if pattern in col_lower:
                score += pattern_score

        return score

    def get_content_length(col_name):
        """Get average content length for this column."""
        try:
            if col_name in sample and sample[col_name]:
                content = str(sample[col_name])
                return len(content)
            return 0
        except:
            return 0

    def score_column(col_name, keywords, role_type, num_candidates):
        """Score a column for how likely it is to be a particular role."""
        if not has_keyword(col_name, keywords):
            return 0

        score = 0
        score += 10

        # Penalize ambiguous keywords when scoring for user
        if role_type == "user":
            col_lower = col_name.lower()
            # If column is ONLY "task" (or task_xxx), give it lower priority for user role
            if "task" in col_lower and not any(
                kw in col_lower for kw in user_words_high_priority
            ):
                score -= 15  # Significant penalty so other user columns win

        priority_bonus = get_priority_score(col_name)
        score += priority_bonus

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

    # Filter out metadata columns
    content_columns = [col for col in all_columns if not is_metadata(col)]

    # Count candidates first
    assistant_potential = [
        col for col in content_columns if has_keyword(col, assistant_words)
    ]
    user_potential = [col for col in content_columns if has_keyword(col, user_words)]

    # STEP 1: Find best ASSISTANT column
    assistant_candidates = []
    for col in assistant_potential:
        score = score_column(
            col, assistant_words, "assistant", len(assistant_potential)
        )
        if score > 0:
            assistant_candidates.append((col, score))

    if assistant_candidates:
        assistant_candidates.sort(key = lambda x: x[1], reverse = True)
        assistant_col = assistant_candidates[0][0]
        mapping[assistant_col] = "assistant"
    else:
        assistant_col = None

    # STEP 2: Find best USER column (with penalty for ambiguous keywords)
    user_candidates = []
    for col in user_potential:
        if col == assistant_col:
            continue
        score = score_column(col, user_words, "user", len(user_potential))
        if score > 0:
            user_candidates.append((col, score))

    if user_candidates:
        user_candidates.sort(key = lambda x: x[1], reverse = True)
        user_col = user_candidates[0][0]
        mapping[user_col] = "user"
    else:
        user_col = None

    # STEP 3: Check ALL remaining columns for SYSTEM matches (priority check)
    remaining_columns = [col for col in content_columns if col not in mapping]

    system_col = None
    for col in remaining_columns:
        if has_keyword(col, system_words):
            # Found a system match in remaining columns
            mapping[col] = "system"
            system_col = col
            break

    # STEP 4: Handle any additional remaining columns
    if system_col:
        remaining_columns = [col for col in remaining_columns if col != system_col]

    if len(remaining_columns) >= 1:
        remaining_col = remaining_columns[0]

        # If no strong keyword match, decide based on what's missing
        if not has_keyword(remaining_col, user_words + assistant_words):
            mapping[remaining_col] = "system"
        elif user_col is None:
            # No user column yet, assign this as user
            mapping[remaining_col] = "user"
        else:
            # Already have user + assistant, treat as system context
            mapping[remaining_col] = "system"

    # VALIDATION: Ensure we have at least user + assistant
    has_user = any(role == "user" for role in mapping.values())
    has_assistant = any(role == "assistant" for role in mapping.values())

    if not has_user and len(remaining_columns) > 0:
        for col in remaining_columns:
            if col not in mapping:
                mapping[col] = "user"
                has_user = True
                break

    if has_user and has_assistant:
        return mapping

    return None


def detect_multimodal_dataset(dataset):
    """
    Detects if dataset contains multimodal data (images and/or audio).

    Two-pass approach for each modality:
      1. Column-name heuristic (fast): checks for keywords.
      2. Value-type inspection (reliable): checks actual sample values.

    Returns:
        dict: {
            "is_image": bool,
            "multimodal_columns": list of column names containing image data,
            "modality_types": list of detected types (e.g., ["image", "audio"]),
            "is_audio": bool,
            "audio_columns": list of column names containing audio data,
            "detected_audio_column": str or None,
            "detected_text_column": str or None,
        }
    """
    sample = next(iter(dataset))
    column_names = list(sample.keys())

    # Keywords that indicate image data
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

    # Keywords that indicate audio data
    audio_keywords = ["audio", "speech", "wav", "waveform", "sound"]

    multimodal_columns = []
    audio_columns = []
    modality_types = set()

    # ── Image detection ─────────────────────────────────────
    # Pass 1: column-name heuristic (word-boundary match to avoid
    #          false positives like 'pic' in 'topic')
    for col_name in column_names:
        for keyword in image_keywords:
            if _keyword_in_column(keyword, col_name):
                multimodal_columns.append(col_name)
                modality_types.add(keyword)
                break

    # Pass 2: inspect actual values
    already_detected = set(multimodal_columns)
    for col_name in column_names:
        if col_name in already_detected:
            continue
        value = sample[col_name]
        if _is_image_value(value):
            multimodal_columns.append(col_name)
            modality_types.add("image")

    # ── Audio detection ─────────────────────────────────────
    # Pass 1: column-name heuristic (word-boundary match)
    for col_name in column_names:
        for keyword in audio_keywords:
            if _keyword_in_column(keyword, col_name):
                audio_columns.append(col_name)
                modality_types.add("audio")
                break

    # Pass 2: inspect actual values (catches non-obvious column names)
    already_audio = set(audio_columns)
    for col_name in column_names:
        if col_name in already_audio:
            continue
        value = sample[col_name]
        if _is_audio_value(value):
            audio_columns.append(col_name)
            modality_types.add("audio")

    # Filter out columns that are actually audio from the image list
    # (e.g. a column named "audio" with {"bytes", "path"} could match _is_image_value)
    if audio_columns:
        audio_set = set(audio_columns)
        multimodal_columns = [c for c in multimodal_columns if c not in audio_set]

    # Detect text column for audio datasets
    detected_text_col = None
    if audio_columns:
        text_keywords = ["text", "sentence", "transcript", "transcription", "label"]
        for col_name in column_names:
            if col_name.lower() in text_keywords:
                detected_text_col = col_name
                break

    is_audio = len(audio_columns) > 0

    # Detect speaker_id column for TTS datasets (CSM, Orpheus, Spark)
    detected_speaker_col = None
    if audio_columns:
        speaker_keywords = ["source", "speaker", "speaker_id"]
        for col_name in column_names:
            if col_name.lower() in speaker_keywords:
                detected_speaker_col = col_name
                break

    return {
        "is_image": len(multimodal_columns) > 0,
        "multimodal_columns": multimodal_columns,
        "modality_types": list(modality_types),
        "is_audio": is_audio,
        "audio_columns": audio_columns,
        "detected_audio_column": audio_columns[0] if audio_columns else None,
        "detected_text_column": detected_text_col,
        "detected_speaker_column": detected_speaker_col,
    }


def _is_image_value(value) -> bool:
    """Check if a single sample value looks like image data."""
    if value is None:
        return False

    # PIL Image instance
    try:
        from PIL.Image import Image as PILImage

        if isinstance(value, PILImage):
            return True
    except ImportError:
        pass

    # HF datasets Image feature stores decoded images as PIL or dicts with
    # {"bytes": b"...", "path": "..."} when not yet decoded.
    # Exclude audio dicts (decoded audio has "array" + "sampling_rate").
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return False  # This is audio, not image
        if "bytes" in value and "path" in value:
            # Check path extension to exclude audio files
            path = value.get("path") or ""
            if isinstance(path, str) and any(
                path.lower().endswith(ext) for ext in _AUDIO_EXTENSIONS
            ):
                return False
            return True

    # Raw bytes with a known image magic header
    if isinstance(value, (bytes, bytearray)):
        return _has_image_header(value)

    # String that looks like an image file path or URL
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg")
    if isinstance(value, str) and len(value) < 1000:
        lower = value.strip().lower()
        # Image URL (http://... ending in image extension)
        if lower.startswith(("http://", "https://")) and any(
            lower.split("?")[0].endswith(ext) for ext in _IMAGE_EXTS
        ):
            return True
        # Image file path (relative or absolute path ending in image extension)
        if any(lower.endswith(ext) for ext in _IMAGE_EXTS):
            return True

    return False


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
    """Check if a single sample value looks like audio data."""
    if value is None:
        return False

    # HF datasets Audio feature: decoded → {"array": np.ndarray, "sampling_rate": int}
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return True
        # Undecoded/streaming → {"bytes": b"...", "path": "some.wav"}
        if "bytes" in value or "path" in value:
            path = value.get("path") or ""
            if isinstance(path, str) and any(
                path.lower().endswith(ext) for ext in _AUDIO_EXTENSIONS
            ):
                return True

    return False


def _has_image_header(data: bytes) -> bool:
    """Quick magic-byte check for common image formats."""
    if len(data) < 4:
        return False
    # JPEG
    if data[:2] == b"\xff\xd8":
        return True
    # PNG
    if data[:4] == b"\x89PNG":
        return True
    # GIF
    if data[:3] == b"GIF":
        return True
    # WebP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    # BMP
    if data[:2] == b"BM":
        return True
    return False


def detect_vlm_dataset_structure(dataset):
    """
    Detects if VLM dataset is:
    - Standard VLM messages format (image objects in content)
    - Llava format (image indices + separate images column)
    - Simple format needing conversion (image + text columns)
    """
    try:
        sample = next(iter(dataset))
    except StopIteration:
        return {
            "format": "unknown",
            "needs_conversion": None,
            "image_column": None,
            "text_column": None,
            "messages_column": None,
        }

    column_names = set(sample.keys())

    # Check if has messages column
    if "messages" in column_names:
        messages = sample["messages"]

        if messages and len(messages) > 0:
            first_msg = messages[0]
            if "content" in first_msg:
                content = first_msg["content"]

                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "type" in content[0]:
                        # Check for llava format
                        has_index = any(
                            "index" in item
                            for item in content
                            if isinstance(item, dict)
                        )
                        has_images_column = "images" in column_names

                        if has_index and has_images_column:
                            return {
                                "format": "vlm_messages_llava",
                                "needs_conversion": True,
                                "messages_column": "messages",
                                "image_column": "images",
                                "text_column": None,
                            }

                        # Standard VLM format
                        has_image = any(
                            "image" in item
                            for item in content
                            if isinstance(item, dict)
                        )
                        if has_image:
                            return {
                                "format": "vlm_messages",
                                "needs_conversion": False,
                                "messages_column": "messages",
                                "image_column": None,
                                "text_column": None,
                            }

    # Check for ShareGPT/ChatML conversations with <image> placeholder + companion image column
    # (e.g. Lin-Chen/ShareGPT4V, LLaVA-style datasets)
    for chat_col in ("conversations", "messages"):
        if chat_col not in column_names:
            continue
        chat_data = sample[chat_col]
        if not isinstance(chat_data, list) or len(chat_data) == 0:
            continue
        first_msg = chat_data[0]
        if not isinstance(first_msg, dict):
            continue
        # Detect ShareGPT (from/value) or ChatML (role/content) keys
        msg_text = first_msg.get("value") or first_msg.get("content")
        if not isinstance(msg_text, str):
            continue
        # Check for <image> placeholder anywhere in the conversation
        has_image_placeholder = any(
            "<image>" in str(m.get("value", "") or m.get("content", ""))
            for m in chat_data
            if isinstance(m, dict)
        )
        if not has_image_placeholder:
            continue
        # Find companion image column
        image_col = None
        for col in column_names:
            if col == chat_col:
                continue
            if _keyword_in_column("image", col) or _keyword_in_column("img", col):
                image_col = col
                break
        if image_col:
            return {
                "format": "sharegpt_with_images",
                "needs_conversion": True,
                "image_column": image_col,
                "text_column": None,
                "messages_column": chat_col,
            }

    # Find image and text columns using metadata filtering

    # Define metadata patterns to EXCLUDE
    metadata_patterns = {
        "suffixes": [
            "_id",
            "_url",
            "_name",
            "_filename",
            "_uri",
            "_link",
            "_key",
            "_index",
        ],
        "prefixes": [
            "id_",
            "url_",
            "name_",
            "filename_",
            "uri_",
            "link_",
            "key_",
            "index_",
        ],
    }

    # Image-related keywords
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

    # Text-related keywords
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
        """Check if column name looks like metadata."""
        col_lower = col_name.lower()

        # Check suffixes
        if any(col_lower.endswith(suffix) for suffix in metadata_patterns["suffixes"]):
            return True

        # Check prefixes
        if any(
            col_lower.startswith(prefix) for prefix in metadata_patterns["prefixes"]
        ):
            return True

        return False

    def _score_image_candidate(col, sample_value):
        """Score a candidate image column by how resolvable its value is."""
        # PIL Image object (highest priority - already loaded)
        if hasattr(sample_value, "size") and hasattr(sample_value, "mode"):
            return 100

        # Dict with image data (bytes/path from HF Image feature)
        if isinstance(sample_value, dict) and (
            "bytes" in sample_value or "path" in sample_value
        ):
            return 75

        if isinstance(sample_value, str):
            # URL strings
            if sample_value.startswith(("http://", "https://")):
                return 70 if not is_metadata_column(col) else 55
            # Bare file path
            if is_metadata_column(col):
                return 30
            return 50

        return 0

    def _probe_image_candidate(col, sample_value):
        """Quick probe to check if an image candidate is actually reachable.
        Returns True if likely valid, False if definitely broken."""
        import os

        # PIL / dict — already loaded, always valid
        if not isinstance(sample_value, str):
            return True

        # Local file — check it exists
        if not sample_value.startswith(("http://", "https://")):
            return os.path.exists(
                sample_value
            )  # bare filenames return False here, that's OK

        # URL — quick HEAD request with short timeout
        try:
            import urllib.request

            req = urllib.request.Request(sample_value, method = "HEAD")
            resp = urllib.request.urlopen(req, timeout = 3)
            return resp.status < 400
        except Exception:
            return False

    def find_image_column():
        """Find image column by keyword match + value-based fallback.
        When multiple candidates exist, probes them to find one that works."""
        candidates = []

        # Pass 1: keyword-matched columns
        for col in column_names:
            if any(_keyword_in_column(keyword, col) for keyword in image_keywords):
                sample_value = sample[col]
                score = _score_image_candidate(col, sample_value)
                if score > 0:
                    candidates.append((col, score))

        # Pass 2: value-based fallback — find columns with image URLs/paths
        # even if the column name doesn't match image keywords
        already = {c[0] for c in candidates}
        for col in column_names:
            if col in already:
                continue
            sample_value = sample[col]
            if _is_image_value(sample_value):
                score = _score_image_candidate(col, sample_value)
                # Slightly penalise non-keyword columns so keyword matches win on ties
                candidates.append((col, max(score - 5, 1)))

        if not candidates:
            return None

        candidates.sort(key = lambda x: x[1], reverse = True)

        # Single candidate or top candidate is PIL/dict — no probing needed
        if len(candidates) == 1 or candidates[0][1] >= 75:
            return candidates[0][0]

        # Multiple string-based candidates — probe to find one that actually works
        for col, score in candidates:
            sample_value = sample[col]
            if _probe_image_candidate(col, sample_value):
                return col

        # Nothing probed successfully — return highest-scored anyway and let
        # conversion handle the error (it may still resolve via hf_hub_download)
        return candidates[0][0]

    def find_text_column():
        """Find text column by filtering out metadata and checking keywords."""
        candidates = []

        for col in column_names:
            # Skip metadata columns
            if is_metadata_column(col):
                continue

            # Check if contains text keywords (word-boundary match)
            if any(_keyword_in_column(keyword, col) for keyword in text_keywords):
                # Verify it's actually text
                sample_value = sample[col]

                if isinstance(sample_value, str) and len(sample_value) > 0:
                    # Longer text = higher priority (likely content, not just a label)
                    priority = min(len(sample_value), 1000)  # Cap at 1000
                    candidates.append((col, priority))
                elif (
                    isinstance(sample_value, list)
                    and len(sample_value) > 0
                    and isinstance(sample_value[0], str)
                ):
                    # List of strings (e.g. captions list) — lower priority than plain strings
                    priority = min(len(sample_value[0]), 1000) // 2
                    candidates.append((col, priority))

        # Return highest priority candidate
        if candidates:
            candidates.sort(key = lambda x: x[1], reverse = True)
            return candidates[0][0]

        return None

    found_image = find_image_column()
    found_text = find_text_column()

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
