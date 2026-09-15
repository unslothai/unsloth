# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The persisted half of the exact-concurrency switch. ``core.inference.llama_exact`` owns what
``auto | off | on`` MEAN; this file only stores one. Stored ``off`` and nothing stored are
deliberately different: nothing stored falls through to an inherited ``LLAMA_EXACT_CONCURRENCY``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from core.inference.llama_exact import (
    DEFAULT_EXACT_SETTING,
    EXACT_SETTINGS,
    normalize_setting,
)

EXACT_CONCURRENCY_SETTING_KEY = "llama_exact_concurrency"

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_generation: dict[str, int] = {}

_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
    stored = None
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_S:
                return hit[1]
            generation = _generation.get(key, 0)
        try:
            from storage.studio_db import get_app_setting
            stored = get_app_setting(key, None)
        except Exception:
            # An unreadable database must never fail a load; the caller falls back.
            return None
        with _cache_lock:
            if _generation.get(key, 0) == generation:
                _cache[key] = (time.monotonic(), stored)
                return stored
        # A write committed while this read was in flight, so `stored` predates it and a load
        # taking it would launch contradicting the setting that was just saved.
    return stored


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _generation[key] = _generation.get(key, 0) + 1


def get_exact_concurrency() -> Optional[str]:
    """The stored setting, or None when nothing valid is stored. None rather than the default,
    because the caller distinguishes them."""
    return normalize_setting(_cached_setting(EXACT_CONCURRENCY_SETTING_KEY))


def set_exact_concurrency(value: Any) -> Optional[str]:
    """Store one of auto/off/on and return it. Raises ValueError on anything else."""
    parsed = normalize_setting(value)
    if parsed is None:
        raise ValueError("Exact concurrency must be one of " + ", ".join(EXACT_SETTINGS) + ".")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({EXACT_CONCURRENCY_SETTING_KEY: parsed})
    _invalidate(EXACT_CONCURRENCY_SETTING_KEY)
    return get_exact_concurrency()


def default_exact_concurrency() -> str:
    return DEFAULT_EXACT_SETTING
