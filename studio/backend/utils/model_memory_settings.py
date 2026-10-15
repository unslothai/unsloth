# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted model-memory residency controls.

``keep_resident``  -- weights never go back to system RAM while loaded: no idle
auto-unload, and ``--mlock`` so the OS cannot page them out and re-fault them in.

``no_ram_reserve`` -- avoids locked or reserved weight buffers. Uses DirectIO on
supported Windows builds when full GPU offload is confirmed, otherwise keeps
llama.cpp's default mmap path. Required CPU buffers can still use host RAM.

Both on means "live in VRAM, keep no RAM copy, never idle-unload". ``--mlock`` is
itself a full-model RAM reservation, so ``no_ram_reserve`` wins on that flag.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from utils.account_context import OWNER, run_as

KEEP_RESIDENT_SETTING_KEY = "model_memory_keep_resident"
NO_RAM_RESERVE_SETTING_KEY = "model_memory_no_ram_reserve"

DEFAULT_KEEP_RESIDENT = False
DEFAULT_NO_RAM_RESERVE = False

# Read on hot paths, so memo briefly to spare SQLite.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
# Bumped on every write so a read started earlier does not cache a stale value.
_generation: dict[tuple[str, str], int] = {}


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


# Bound only guards against a pathological write storm.
_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
    cache_key = (OWNER.account_id, key)
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            hit = _cache.get(cache_key)
            if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_S:
                return hit[1]
            generation = _generation.get(cache_key, 0)
        try:
            from storage.studio_db import get_app_setting
            stored = run_as(OWNER, get_app_setting, key, None)
        except Exception:
            # An unreadable DB must not fail a load.
            return None
        with _cache_lock:
            if _generation.get(cache_key, 0) == generation:
                _cache[cache_key] = (time.monotonic(), stored)
                return stored
        # A write landed mid-read; read again against the new generation.
    return stored


def _invalidate(*keys: str) -> None:
    """Drop all keys in one acquisition so a load cannot pair new keep_resident with stale
    no_ram_reserve."""
    account_id = OWNER.account_id
    with _cache_lock:
        for key in keys:
            cache_key = (account_id, key)
            _cache.pop(cache_key, None)
            _generation[cache_key] = _generation.get(cache_key, 0) + 1


def get_keep_resident() -> bool:
    """True when the loaded model must stay in GPU memory while it is loaded."""
    parsed = _coerce_bool(_cached_setting(KEEP_RESIDENT_SETTING_KEY))
    return parsed if parsed is not None else DEFAULT_KEEP_RESIDENT


def get_no_ram_reserve() -> bool:
    """True when no full host-RAM copy of the weights may be held."""
    parsed = _coerce_bool(_cached_setting(NO_RAM_RESERVE_SETTING_KEY))
    return parsed if parsed is not None else DEFAULT_NO_RAM_RESERVE


def should_mlock() -> bool:
    """Needs keep_resident on and no_ram_reserve off, since mlock pins the whole model in host RAM."""
    keep_resident, no_ram_reserve = get_model_memory_settings()
    return keep_resident and not no_ram_reserve


def _pair_generations() -> tuple[int, int]:
    with _cache_lock:
        return (
            _generation.get((OWNER.account_id, KEEP_RESIDENT_SETTING_KEY), 0),
            _generation.get((OWNER.account_id, NO_RAM_RESERVE_SETTING_KEY), 0),
        )


def capture_model_memory_settings(publish) -> tuple[bool, bool]:
    """Publishes the pair, rereading if a save bumped the generation mid-read; no lock across DB I/O."""
    for _attempt in range(_MAX_REREADS):
        before = _pair_generations()
        pair = get_model_memory_settings()
        publish(pair)
        if _pair_generations() == before:
            return pair
    return pair


def get_model_memory_settings() -> tuple[bool, bool]:
    """One coherent snapshot of both flags: a save between separate reads could give a never-stored pair."""
    pair = (get_keep_resident(), get_no_ram_reserve())
    for _attempt in range(_MAX_REREADS):
        before = _pair_generations()
        pair = (get_keep_resident(), get_no_ram_reserve())
        if _pair_generations() == before:
            return pair
    return pair


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
        _invalidate(*updates)

    return get_keep_resident(), get_no_ram_reserve()


def memlock_limit_bytes() -> Optional[int]:
    """Soft RLIMIT_MEMLOCK; mlock cannot exceed it, and Linux's 8 MB default silently disables residency."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (AttributeError, ValueError, OSError):
        return None
    if soft < 0 or soft == resource.RLIM_INFINITY:
        return None
    return int(soft)
