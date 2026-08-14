# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted preference for telling the model today's date."""

from __future__ import annotations

from datetime import date
from typing import Any

CURRENT_DATE_PROMPT_SETTING_KEY = "include_current_date_in_prompt"
# Lets callers recognise a prompt that already states a date, whoever put it there.
CURRENT_DATE_PROMPT_PREFIX = "The current date is "
# default on: date-blind models answer from their training cutoff and search for stale material.
DEFAULT_CURRENT_DATE_PROMPT_ENABLED = True


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


def get_current_date_prompt_enabled() -> bool:
    """Read the persisted preference, defaulting to enabled when it is missing or unreadable."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(CURRENT_DATE_PROMPT_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_CURRENT_DATE_PROMPT_ENABLED


def set_current_date_prompt_enabled(value: Any) -> bool:
    """Persist whether prompts should state the current date."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Include current date in prompt must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({CURRENT_DATE_PROMPT_SETTING_KEY: parsed})
    return parsed


def current_date_prompt_line(today: date | None = None) -> str:
    """Return the shared date sentence, or an empty string when the preference is off."""
    if not get_current_date_prompt_enabled():
        return ""
    return f"{CURRENT_DATE_PROMPT_PREFIX}{(today or date.today()).isoformat()}."
