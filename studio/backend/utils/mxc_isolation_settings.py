# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved Settings > Sandbox MXC switches; a present env var always wins (mxc_policy / mxc_read_grants)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

DACL_SETTING_KEY = "mxc_allow_dacl_fallback"
GRANTS_SETTING_KEY = "mxc_persistent_read_grants"
DEFAULT_DACL_FALLBACK = False
DEFAULT_PERSISTENT_GRANTS = True

# A write drops the entry; the TTL only bounds staleness from another process.
_CACHE_TTL_SECONDS = 1.0
_cache_lock = threading.Lock()
_cached: tuple[float, dict[str, Any]] | None = None
# Bumped by every write: a read that started earlier must not publish what it saw.
_generation = 0


def forget_cached_setting() -> None:
    global _cached, _generation
    with _cache_lock:
        _cached = None
        _generation += 1


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


def _stored() -> dict[str, Any] | None:
    """Both keys from one owner-store snapshot, or None when the store cannot be read."""
    global _cached
    with _cache_lock:
        if _cached is not None and _cached[0] > time.monotonic():
            return _cached[1]
        generation = _generation
    try:
        from storage.studio_db import get_app_settings
        from utils.account_context import OWNER, run_as

        # Installation-wide: always the owner's store, whichever account asks.
        stored = run_as(OWNER, get_app_settings, [DACL_SETTING_KEY, GRANTS_SETTING_KEY])
    except Exception:  # noqa: BLE001 - an unreadable store falls back to the shipped defaults
        return None
    with _cache_lock:
        if generation == _generation:
            _cached = (time.monotonic() + _CACHE_TTL_SECONDS, stored)
    return stored


def _setting(key: str, default: bool) -> bool:
    stored = _stored()
    parsed = _coerce_bool(stored.get(key)) if stored is not None else None
    return default if parsed is None else parsed


def dacl_fallback_setting() -> bool:
    """Saved opt-in to MXC's AppContainer tier; off by default and when the store cannot be read."""
    return _setting(DACL_SETTING_KEY, DEFAULT_DACL_FALLBACK)


def persistent_grants_setting() -> bool:
    """Saved choice to keep Studio's runtime read grants; a read failure keeps the shipped default (on)."""
    return _setting(GRANTS_SETTING_KEY, DEFAULT_PERSISTENT_GRANTS)


def _write(key: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError("MXC isolation settings must be true or false.")
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {key: value})
    # Dropped, not replaced: a write that did not land must not be believed.
    forget_cached_setting()


def set_dacl_fallback_setting(value: bool) -> None:
    _write(DACL_SETTING_KEY, value)


def set_persistent_grants_setting(value: bool) -> None:
    _write(GRANTS_SETTING_KEY, value)


def locked_by_environment(name: str) -> bool:
    """True when the environment decides ``name``, even set to an empty string."""
    return name in os.environ
