# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Chat tool output limit: how many characters of one tool result the model reads; a set env var wins."""

from __future__ import annotations

import os
from typing import Any

MAX_CHARS_ENV = "UNSLOTH_TOOL_RESULT_MAX_CHARS"
MAX_CHARS_SETTING_KEY = "tool_result_max_chars"
DEFAULT_MAX_CHARS = 16000
MIN_MAX_CHARS = 2000
# Below UNSLOTH_TOOL_RESULT_HARD_CAP_CHARS (256,000), so a saved value never outgrows the floor cap on every tool.
MAX_MAX_CHARS = 200000


def _coerce_chars(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < MIN_MAX_CHARS or parsed > MAX_MAX_CHARS:
        return None
    return parsed


def locked_by_environment() -> bool:
    # Empty counts as unset, as tools._env_int reads it.
    return bool(os.environ.get(MAX_CHARS_ENV))


def environment_max_chars() -> int:
    """The env value as tools._env_int reads it: garbage or non-positive falls back to the default."""
    try:
        value = int(os.environ.get(MAX_CHARS_ENV, "") or DEFAULT_MAX_CHARS)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return value if value > 0 else DEFAULT_MAX_CHARS


def saved_max_chars() -> int | None:
    """The owner's saved value, or None when there is none (or the store cannot be read)."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Installation-wide: the owner's store, whichever account's tool call is asking.
        stored = run_as(OWNER, get_app_setting, MAX_CHARS_SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - an unreadable store keeps the shipped default
        return None
    return _coerce_chars(stored)


def effective_max_chars() -> int:
    """The cap before a small context window lowers it."""
    if locked_by_environment():
        return environment_max_chars()
    return saved_max_chars() or DEFAULT_MAX_CHARS


def set_max_chars(value: Any) -> int:
    parsed = _coerce_chars(value)
    if parsed is None:
        raise ValueError(
            f"Tool output limit must be a whole number from {MIN_MAX_CHARS} to {MAX_MAX_CHARS} characters."
        )
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {MAX_CHARS_SETTING_KEY: parsed})
    return parsed
