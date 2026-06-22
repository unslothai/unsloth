# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle TTL for a loaded inference model.

A loaded GGUF model otherwise stays resident until a manual unload or process
exit. When the idle TTL is set (> 0), a background task unloads the model after
it has been idle (no generation activity) for that many seconds. 0 disables
eviction, preserving the historical behavior. The value can be set at startup
via the ``UNSLOTH_MODEL_IDLE_TTL`` env var or at runtime via
``PUT /api/settings/model-ttl``.
"""

from __future__ import annotations

import os
from typing import Any

MODEL_IDLE_TTL_SETTING_KEY = "model_idle_ttl_seconds"
DEFAULT_MODEL_IDLE_TTL_SECONDS = 0  # 0 = disabled (never auto-evict)
MIN_MODEL_IDLE_TTL_SECONDS = 0
MAX_MODEL_IDLE_TTL_SECONDS = 7 * 24 * 60 * 60  # one week ceiling
MODEL_IDLE_EVICTION_POLL_SECONDS = 30.0
_ENV_VAR = "UNSLOTH_MODEL_IDLE_TTL"


def _coerce_ttl_seconds(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < MIN_MODEL_IDLE_TTL_SECONDS or parsed > MAX_MODEL_IDLE_TTL_SECONDS:
        return None
    return parsed


def default_model_idle_ttl_seconds() -> int:
    """The startup default: the env override when valid, else disabled (0)."""
    env_value = _coerce_ttl_seconds(os.environ.get(_ENV_VAR))
    return env_value if env_value is not None else DEFAULT_MODEL_IDLE_TTL_SECONDS


def validate_model_idle_ttl_seconds(value: Any) -> int:
    parsed = _coerce_ttl_seconds(value)
    if parsed is None:
        raise ValueError(
            "Model idle TTL must be a whole number of seconds from "
            f"{MIN_MODEL_IDLE_TTL_SECONDS} to {MAX_MODEL_IDLE_TTL_SECONDS} (0 disables it)."
        )
    return parsed


def get_model_idle_ttl_seconds() -> int:
    """Effective TTL: the stored setting when valid, else the startup default."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(MODEL_IDLE_TTL_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_ttl_seconds(stored)
    return parsed if parsed is not None else default_model_idle_ttl_seconds()


def set_model_idle_ttl_seconds(value: Any) -> int:
    parsed = validate_model_idle_ttl_seconds(value)
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({MODEL_IDLE_TTL_SETTING_KEY: parsed})
    return parsed
