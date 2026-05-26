# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Optional


def normalize_chat_template_override(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    return None


def normalize_gpu_ids(gpu_ids: Optional[list[int]]) -> Optional[list[int]]:
    return list(gpu_ids) if gpu_ids else None


def build_transformers_load_settings(
    *,
    config: Any,
    max_seq_length: int,
    load_in_4bit: bool,
    trust_remote_code: bool,
    gpu_ids: Optional[list[int]],
    chat_template_override: Optional[str],
) -> dict[str, Any]:
    return {
        "identifier": getattr(config, "identifier", None),
        "path": getattr(config, "path", None),
        "base_model": getattr(config, "base_model", None),
        "is_lora": bool(getattr(config, "is_lora", False)),
        "is_vision": bool(getattr(config, "is_vision", False)),
        "is_audio": bool(getattr(config, "is_audio", False)),
        "audio_type": getattr(config, "audio_type", None),
        "max_seq_length": max_seq_length if max_seq_length > 0 else 2048,
        "load_in_4bit": bool(load_in_4bit),
        "trust_remote_code": bool(trust_remote_code),
        "gpu_ids": normalize_gpu_ids(gpu_ids),
        "chat_template_override": normalize_chat_template_override(
            chat_template_override
        ),
    }


def load_settings_match(
    current: Any,
    requested: dict[str, Any],
) -> bool:
    return isinstance(current, dict) and current == requested
