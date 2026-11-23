# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted opt-in controls for Helper LLM startup pre-cache."""

from __future__ import annotations

import os
from typing import Any

HELPER_PRECACHE_SETTING_KEY = "helper_model_preload_on_startup"
DEFAULT_HELPER_PRECACHE_ENABLED = False


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


def helper_model_disabled_by_env() -> bool:
    """Return True when existing broad helper-disable env var is active."""
    return os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in {"1", "true"}


def get_helper_precache_enabled() -> bool:
    """Missing or unreadable settings default to False, so startup makes no optional network calls."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(HELPER_PRECACHE_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_HELPER_PRECACHE_ENABLED


def set_helper_precache_enabled(value: Any) -> bool:
    """Persist whether Unsloth should pre-cache the Helper LLM at startup."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Helper LLM startup pre-cache must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({HELPER_PRECACHE_SETTING_KEY: parsed})
    return parsed


def should_preload_helper_on_startup() -> bool:
    """Needs the opt-in setting and no env disable; explicit AI Assist calls bypass this gate."""
    return get_helper_precache_enabled() and not helper_model_disabled_by_env()
