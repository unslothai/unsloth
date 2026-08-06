# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted model-memory residency controls.

``keep_resident``  -- weights never go back to system RAM while loaded: no idle
auto-unload, and ``--mlock`` so the OS cannot page them out and re-fault them in.

``no_ram_reserve`` -- no full host-RAM copy: keeps llama.cpp's default mmap path
and drops ``--no-mmap`` / ``--mlock``.

Both on means "live in VRAM, keep no RAM copy, never idle-unload". ``--mlock`` is
itself a full-model RAM reservation, so ``no_ram_reserve`` wins on that flag.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

KEEP_RESIDENT_SETTING_KEY = "model_memory_keep_resident"
NO_RAM_RESERVE_SETTING_KEY = "model_memory_no_ram_reserve"

DEFAULT_KEEP_RESIDENT = False
DEFAULT_NO_RAM_RESERVE = False

# Read on the load path and every idle poll, so memo briefly to spare SQLite.
# Matches openai_auto_switch_settings.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def _cached_setting(key: str) -> Any:
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(key, None)
    except Exception:
        # An unreadable DB must not fail a load; fall back to the default.
        stored = None
    with _cache_lock:
        _cache[key] = (now, stored)
    return stored


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


def get_keep_resident() -> bool:
    """True when the loaded model must stay in GPU memory while it is loaded."""
    parsed = _coerce_bool(_cached_setting(KEEP_RESIDENT_SETTING_KEY))
    return parsed if parsed is not None else DEFAULT_KEEP_RESIDENT


def get_no_ram_reserve() -> bool:
    """True when no full host-RAM copy of the weights may be held."""
    parsed = _coerce_bool(_cached_setting(NO_RAM_RESERVE_SETTING_KEY))
    return parsed if parsed is not None else DEFAULT_NO_RAM_RESERVE


def should_mlock() -> bool:
    """Whether to pass ``--mlock``.

    mlock pins the whole model in host RAM, so it is emitted only when residency
    is on and no-reserve is off. The two conflict, and no-reserve wins.
    """
    return get_keep_resident() and not get_no_ram_reserve()


def get_model_memory_settings() -> tuple[bool, bool]:
    """``(keep_resident, no_ram_reserve)`` in one call for the settings route."""
    return get_keep_resident(), get_no_ram_reserve()


def set_model_memory_settings(
    keep_resident: Any = None, no_ram_reserve: Any = None
) -> tuple[bool, bool]:
    """One-transaction write; ``None`` leaves a stored value untouched."""
    updates: dict[str, bool] = {}

    if keep_resident is not None:
        parsed = _coerce_bool(keep_resident)
        if parsed is None:
            raise ValueError("Keep model in GPU memory must be true or false.")
        updates[KEEP_RESIDENT_SETTING_KEY] = parsed

    if no_ram_reserve is not None:
        parsed = _coerce_bool(no_ram_reserve)
        if parsed is None:
            raise ValueError("Do not reserve system RAM must be true or false.")
        updates[NO_RAM_RESERVE_SETTING_KEY] = parsed

    if updates:
        from storage.studio_db import upsert_app_settings
        upsert_app_settings(updates)
        for key in updates:
            _invalidate(key)

    return get_keep_resident(), get_no_ram_reserve()
