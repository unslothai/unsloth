# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted preference for telling the model today's date."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CURRENT_DATE_PROMPT_SETTING_KEY = "include_current_date_in_prompt"
# Lets callers recognise a prompt that already states a date, whoever put it there.
CURRENT_DATE_PROMPT_PREFIX = "The current date is "
CURRENT_DATE_PROMPT_LINE_RE = re.compile(
    rf"(?m)^{re.escape(CURRENT_DATE_PROMPT_PREFIX)}[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.(?=\r?$)"
)
# default on: date-blind models answer from their training cutoff and search for stale material.
DEFAULT_CURRENT_DATE_PROMPT_ENABLED = True
CURRENT_DATE_TIMEZONE_HEADER = "x-unsloth-timezone"
CURRENT_DATE_TIMEZONE_OFFSET_HEADER = "x-unsloth-timezone-offset-minutes"
MAX_TIMEZONE_OFFSET_MINUTES = 14 * 60


def contains_current_date_prompt_line(text: str) -> bool:
    return CURRENT_DATE_PROMPT_LINE_RE.search(text) is not None


def replace_current_date_prompt_lines(text: str, date_line: str) -> str:
    return CURRENT_DATE_PROMPT_LINE_RE.sub(date_line, text)


def strip_current_date_prompt_lines(text: str) -> str:
    if not contains_current_date_prompt_line(text):
        return text
    return "".join(
        line
        for line in text.splitlines(keepends = True)
        if not CURRENT_DATE_PROMPT_LINE_RE.fullmatch(line.rstrip("\r\n"))
    ).strip()


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


def _request_local_date(request: Any, now: datetime | None = None) -> date:
    instant = now or datetime.now(timezone.utc)
    try:
        headers = request.headers
    except Exception:
        return instant.astimezone().date()

    timezone_name = str(headers.get(CURRENT_DATE_TIMEZONE_HEADER) or "").strip()
    if timezone_name and len(timezone_name) <= 64:
        try:
            return instant.astimezone(ZoneInfo(timezone_name)).date()
        except (ValueError, ZoneInfoNotFoundError, OSError):
            pass

    try:
        offset_minutes = int(headers.get(CURRENT_DATE_TIMEZONE_OFFSET_HEADER, ""))
    except (TypeError, ValueError):
        return instant.astimezone().date()
    if abs(offset_minutes) > MAX_TIMEZONE_OFFSET_MINUTES:
        return instant.astimezone().date()
    browser_zone = timezone(timedelta(minutes = -offset_minutes))
    return instant.astimezone(browser_zone).date()


def current_date_prompt_line(today: date | None = None, request: Any = None) -> str:
    """Return the shared date sentence, or an empty string when the preference is off."""
    if not get_current_date_prompt_enabled():
        return ""
    resolved_date = today or _request_local_date(request)
    return f"{CURRENT_DATE_PROMPT_PREFIX}{resolved_date.isoformat()}."
