# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted "Keep multiple models loaded" mode: off, fit (When they fit) or always.

The setting belongs to the installation, so it is always read and written in the owner's
database. Each account has its own studio.db, and a managed account reading its own would
see the default and unload the owner's kept models.

Older builds stored only the ``keep_multiple_models_loaded`` bool. The bool still decides
on or off, and ``multi_model_mode`` only tells When they fit from Always, so an older build
that turns the setting off wins after an upgrade.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from utils.account_context import OWNER, run_as

MULTI_MODEL_SETTING_KEY = "keep_multiple_models_loaded"
MULTI_MODEL_MODE_SETTING_KEY = "multi_model_mode"
MULTI_MODEL_MODE_CHOICES = ("off", "fit", "always")
_VALID_MODES = frozenset(MULTI_MODEL_MODE_CHOICES)
DEFAULT_MULTI_MODEL_MODE = "off"
DEFAULT_MULTI_MODEL_ENABLED = False

# Read on every load, including media loads, so memo briefly to spare SQLite.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: tuple[float, str] | None = None
# Bumped on every write, so a read that began before a write never caches the old value.
_generation = 0
_MAX_REREADS = 3


def _mode_from_stored(stored: dict[str, Any]) -> str:
    if stored.get(MULTI_MODEL_SETTING_KEY) is not True:
        return "off"
    return "always" if stored.get(MULTI_MODEL_MODE_SETTING_KEY) == "always" else "fit"


def _read_mode() -> str:
    from storage.studio_db import get_app_settings

    stored = run_as(
        OWNER,
        get_app_settings,
        [MULTI_MODEL_SETTING_KEY, MULTI_MODEL_MODE_SETTING_KEY],
    )
    return _mode_from_stored(stored)


def get_multi_model_mode() -> str:
    """``"off"``, ``"fit"`` or ``"always"``, read as the owner."""
    global _cache
    mode = DEFAULT_MULTI_MODEL_MODE
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            if _cache is not None and time.monotonic() - _cache[0] < _CACHE_TTL_S:
                return _cache[1]
            generation = _generation
        try:
            mode = _read_mode()
        except Exception:
            # An unreadable DB must not fail a load; fall back to the default.
            return DEFAULT_MULTI_MODEL_MODE
        with _cache_lock:
            if _generation == generation:
                _cache = (time.monotonic(), mode)
                return mode
    return mode


def get_multi_model_enabled() -> bool:
    return get_multi_model_mode() != "off"


def invalidate_multi_model_cache() -> None:
    global _cache, _generation
    with _cache_lock:
        _cache = None
        _generation += 1


def set_multi_model_mode(value: Any) -> bool:
    """Store the mode and return whether it changed."""
    if not isinstance(value, str) or value not in _VALID_MODES:
        raise ValueError("Keep multiple models loaded must be off, fit or always.")

    from storage.studio_db import upsert_app_settings

    previous = _read_mode()
    run_as(
        OWNER,
        upsert_app_settings,
        {MULTI_MODEL_MODE_SETTING_KEY: value, MULTI_MODEL_SETTING_KEY: value != "off"},
    )
    invalidate_multi_model_cache()
    return previous != value


def set_multi_model_enabled(value: Any) -> bool:
    """Legacy bool write: True keeps an existing When they fit or Always, False turns it off."""
    if not isinstance(value, bool):
        raise ValueError("Keep multiple models loaded must be true or false.")
    if not value:
        set_multi_model_mode("off")
    elif _read_mode() == "off":
        set_multi_model_mode("fit")
    return value
