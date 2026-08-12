# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted policy for disabling chat history on shared Studio hosts."""

from __future__ import annotations

import os
from typing import Any

NO_CHAT_HISTORY_SETTING_KEY = "no_chat_history"
DEFAULT_NO_CHAT_HISTORY_ENABLED = False
NO_CHAT_HISTORY_ENV = "UNSLOTH_NO_CHAT_HISTORY"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def no_chat_history_forced_by_env() -> bool:
    return os.environ.get(NO_CHAT_HISTORY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_no_chat_history_enabled() -> bool:
    if no_chat_history_forced_by_env():
        return True
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(NO_CHAT_HISTORY_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_NO_CHAT_HISTORY_ENABLED


def set_no_chat_history_enabled(value: Any) -> bool:
    if no_chat_history_forced_by_env():
        raise ValueError(
            "Chat history policy is locked by UNSLOTH_NO_CHAT_HISTORY and cannot be changed.",
        )
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("No chat history setting must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({NO_CHAT_HISTORY_SETTING_KEY: parsed})
    return parsed
