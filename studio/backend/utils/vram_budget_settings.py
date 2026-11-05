# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted fraction of each GPU a load may claim; the rest is reserved for fragmentation, the CUDA
context and MoE routing. Raising it risks llama.cpp OOM, lowering it pushes tight fits to CPU
offload. Stored value beats the environment, which beats VRAM_FRACTION_DEFAULT."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from utils.account_context import OWNER, run_as

VRAM_BUDGET_SETTING_KEY = "vram_budget_fraction"

VRAM_FRACTION_ENV_VAR = "UNSLOTH_VRAM_FRACTION"

# Mirrored in per-model-config.ts as percent for the slider, a pair test_vram_budget_settings.py pins together. The default is the historical _CTX_FIT_VRAM_FRACTION / _GPU_PIN_VRAM_FRACTION.
# 0.10, not 0: at 0 every GGUF load runs on the CPU. vLLM / SGLang cannot offload and keep engine_adapters.MANAGED_ENGINE_BUDGET_FLOOR.
VRAM_FRACTION_MIN = 0.10
VRAM_FRACTION_MAX = 1.00
VRAM_FRACTION_DEFAULT = 0.97

# The slider steps in tenths, so 0.975 is legal. Quantising to that grid keeps a stored fraction exactly representable as the percent shown.
VRAM_FRACTION_DECIMALS = 3

# Read on the load path, so memo briefly to spare SQLite, as model_memory_settings.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
# Bumped on every write so a read that began earlier does not cache a stale value.
_generation: dict[tuple[str, str], int] = {}

# Retries converge; the bound only stops a write storm spinning here forever.
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
            # An unreadable DB must not fail a load; fall back to the default.
            return None
        with _cache_lock:
            if _generation.get(cache_key, 0) == generation:
                _cache[cache_key] = (time.monotonic(), stored)
                return stored
        # A write landed mid-read, so `stored` predates it and must not be cached.
    return stored


def _invalidate(key: str) -> None:
    cache_key = (OWNER.account_id, key)
    with _cache_lock:
        _cache.pop(cache_key, None)
        _generation[cache_key] = _generation.get(cache_key, 0) + 1


def coerce_fraction(value: Any) -> Optional[float]:
    """A VRAM fraction in ``[VRAM_FRACTION_MIN, VRAM_FRACTION_MAX]``, else None. Accepts the stored JSON number and the raw environment string through the same path so a value can never be legal in one and not the other."""
    if isinstance(value, bool):
        # bool is an int subclass, and True would otherwise read as 1.0.
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    # Two-sided on purpose: NaN loses every comparison, so this rejects it, where the one-sided form would let NaN through and NaN every per-GPU budget. Mirrors _parse_mem_fraction_env.
    if not VRAM_FRACTION_MIN <= fraction <= VRAM_FRACTION_MAX:
        return None
    return round(fraction, VRAM_FRACTION_DECIMALS)


def _env_fraction() -> Optional[float]:
    """``UNSLOTH_VRAM_FRACTION``, or None when unset or unusable. Read here rather than at import so tests can monkeypatch the environment without reloading the module, and so a value exported after startup is picked up by the next load."""
    return coerce_fraction(os.environ.get(VRAM_FRACTION_ENV_VAR))


def get_vram_budget_fraction() -> float:
    """The fraction of each GPU a load may claim. Never raises and never returns a value outside the supported range: a corrupt stored value or a malformed environment variable falls through to the default rather than failing the load."""
    stored = coerce_fraction(_cached_setting(VRAM_BUDGET_SETTING_KEY))
    if stored is not None:
        return stored
    from_env = _env_fraction()
    if from_env is not None:
        return from_env
    return VRAM_FRACTION_DEFAULT


def get_vram_budget_state() -> tuple[float, bool]:
    """``(fraction, is_stored)`` for the settings route. The flag lets the UI distinguish "saved by the user" from "inherited from the environment or the default", which decides whether Reset is meaningful."""
    stored = coerce_fraction(_cached_setting(VRAM_BUDGET_SETTING_KEY))
    if stored is not None:
        return stored, True
    return get_vram_budget_fraction(), False


def set_vram_budget_fraction(fraction: Any = None) -> float:
    """Store a budget, or clear it with ``None`` so env/default applies again."""
    if fraction is None:
        from storage.studio_db import upsert_app_settings

        upsert_app_settings({VRAM_BUDGET_SETTING_KEY: None})
        _invalidate(VRAM_BUDGET_SETTING_KEY)
        return get_vram_budget_fraction()

    parsed = coerce_fraction(fraction)
    if parsed is None:
        raise ValueError(
            f"VRAM budget must be a number between {VRAM_FRACTION_MIN} and {VRAM_FRACTION_MAX}."
        )

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({VRAM_BUDGET_SETTING_KEY: parsed})
    _invalidate(VRAM_BUDGET_SETTING_KEY)
    return get_vram_budget_fraction()
