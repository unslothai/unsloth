# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted VRAM budget fraction: how much of each GPU a load may claim.

The fit reserves a slice of every card that the model and KV cache may not use,
covering fragmentation, the per-device CUDA context and MoE routing. That slice
was two hard-coded 0.97 constants in ``core.inference.llama_cpp``
(``_CTX_FIT_VRAM_FRACTION``, ``_GPU_PIN_VRAM_FRACTION``), so the only way to
trade it for context was to edit the source.

Raising the fraction hands the reserve back as context; the load can then OOM,
which llama.cpp takes as a hard crash rather than a graceful degrade. Lowering it
pushes tight fits into CPU offload, which is what 0.90 did in #5106. Neither
direction is free, so the default stays exactly where it was and an unset budget
must resolve to ``VRAM_FRACTION_DEFAULT``.

Precedence, matching ``openai_auto_switch_settings``: a stored value wins, the
environment is a standalone startup default, the constant is the last resort.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

VRAM_BUDGET_SETTING_KEY = "vram_budget_fraction"

VRAM_FRACTION_ENV_VAR = "UNSLOTH_VRAM_FRACTION"

# Mirrored in per-model-config.ts as percent for the slider, a pair
# test_vram_budget_settings.py pins together. The default is the historical
# _CTX_FIT_VRAM_FRACTION / _GPU_PIN_VRAM_FRACTION.
VRAM_FRACTION_MIN = 0.80
VRAM_FRACTION_MAX = 1.00
VRAM_FRACTION_DEFAULT = 0.97

# The slider steps in tenths, so 0.975 is legal. Quantising to that grid keeps a
# stored fraction exactly representable as the percent shown.
VRAM_FRACTION_DECIMALS = 3

# Read on the load path, so memo briefly to spare SQLite, as model_memory_settings.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
# Bumped on every write: a read that began before it must not cache its stale
# value, or the new budget would appear to revert for the rest of the TTL.
_generation: dict[str, int] = {}

# Retries converge; the bound only stops a write storm spinning here forever.
_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
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
            # An unreadable DB must not fail a load; fall back to the default.
            return None
        with _cache_lock:
            if _generation.get(key, 0) == generation:
                _cache[key] = (time.monotonic(), stored)
                return stored
        # A write landed mid-read, so `stored` predates it and must not be cached.
    return stored


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _generation[key] = _generation.get(key, 0) + 1


def coerce_fraction(value: Any) -> Optional[float]:
    """A VRAM fraction in ``[VRAM_FRACTION_MIN, VRAM_FRACTION_MAX]``, else None.

    Accepts the stored JSON number and the raw environment string through the same
    path so a value can never be legal in one and not the other.
    """
    if isinstance(value, bool):
        # bool is an int subclass, and True would otherwise read as 1.0.
        return None
    try:
        fraction = float(value)  # None -> TypeError, "" / "  " -> ValueError
    except (TypeError, ValueError):
        return None
    # Two-sided on purpose: NaN loses every comparison, so this rejects it; the
    # one-sided form would let NaN through and NaN every per-GPU budget. Mirrors
    # _parse_mem_fraction_env.
    if not VRAM_FRACTION_MIN <= fraction <= VRAM_FRACTION_MAX:
        return None
    return round(fraction, VRAM_FRACTION_DECIMALS)


def _env_fraction() -> Optional[float]:
    """``UNSLOTH_VRAM_FRACTION``, or None when unset or unusable.

    Read here rather than at import so tests can monkeypatch the environment
    without reloading the module, and so a value exported after startup is picked
    up by the next load.
    """
    return coerce_fraction(os.environ.get(VRAM_FRACTION_ENV_VAR))


def get_vram_budget_fraction() -> float:
    """The fraction of each GPU a load may claim.

    Never raises and never returns a value outside the supported range: a corrupt
    stored value or a malformed environment variable falls through to the default
    rather than failing the load.
    """
    stored = coerce_fraction(_cached_setting(VRAM_BUDGET_SETTING_KEY))
    if stored is not None:
        return stored
    from_env = _env_fraction()
    if from_env is not None:
        return from_env
    return VRAM_FRACTION_DEFAULT


def get_vram_budget_state() -> tuple[float, bool]:
    """``(fraction, is_stored)`` for the settings route.

    The flag lets the UI distinguish "saved by the user" from "inherited from the
    environment or the default", which decides whether Reset is meaningful.
    """
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
