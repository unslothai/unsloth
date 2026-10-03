# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted switch for keeping other models loaded when another one loads."""

from __future__ import annotations

from typing import Any

MULTI_MODEL_SETTING_KEY = "keep_multiple_models_loaded"
DEFAULT_MULTI_MODEL_ENABLED = False


def get_multi_model_enabled() -> bool:
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(MULTI_MODEL_SETTING_KEY, None)
    except Exception:
        stored = None
    return stored if isinstance(stored, bool) else DEFAULT_MULTI_MODEL_ENABLED


def set_multi_model_enabled(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Keep multiple models loaded must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({MULTI_MODEL_SETTING_KEY: value})
    return value
