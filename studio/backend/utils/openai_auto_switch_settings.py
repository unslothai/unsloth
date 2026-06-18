# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted opt-in controls for OpenAI-compatible model auto-switching.

Two settings, both off by default so existing API behavior is unchanged:
- ``openai_api_auto_switch_model``: when on, a ``/v1`` request whose ``model``
  names a downloaded local GGUF different from the loaded one transparently
  loads it before serving (llama-swap-style). Unknown names pass through.
- ``openai_api_auto_unload_idle_seconds``: when > 0, the loaded GGUF is
  unloaded after this many idle seconds to free VRAM.

Reads are cached for a short window because these are consulted on the
per-request hot path; writes invalidate the cache.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

OPENAI_AUTO_SWITCH_SETTING_KEY = "openai_api_auto_switch_model"
AUTO_UNLOAD_IDLE_SETTING_KEY = "openai_api_auto_unload_idle_seconds"
MODEL_OVERRIDES_SETTING_KEY = "openai_api_auto_switch_overrides"

DEFAULT_OPENAI_AUTO_SWITCH_ENABLED = False
DEFAULT_AUTO_UNLOAD_IDLE_SECONDS = 0

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


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


def _coerce_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _cached_setting(key: str, default: Any) -> Any:
    """Read an app setting, memoized for _CACHE_TTL_S to spare the hot path."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(key, None)
    except Exception:
        stored = None
    value = default if stored is None else stored
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


def get_openai_auto_switch_enabled() -> bool:
    parsed = _coerce_bool(_cached_setting(OPENAI_AUTO_SWITCH_SETTING_KEY, None))
    return parsed if parsed is not None else DEFAULT_OPENAI_AUTO_SWITCH_ENABLED


def set_openai_auto_switch_enabled(value: Any) -> bool:
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("OpenAI auto-switch must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({OPENAI_AUTO_SWITCH_SETTING_KEY: parsed})
    _invalidate(OPENAI_AUTO_SWITCH_SETTING_KEY)
    return parsed


def get_auto_unload_idle_seconds() -> int:
    parsed = _coerce_int(_cached_setting(AUTO_UNLOAD_IDLE_SETTING_KEY, None))
    return parsed if parsed is not None else DEFAULT_AUTO_UNLOAD_IDLE_SECONDS


def set_auto_unload_idle_seconds(value: Any) -> int:
    parsed = _coerce_int(value)
    if parsed is None:
        raise ValueError("Auto-unload idle seconds must be a non-negative integer.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({AUTO_UNLOAD_IDLE_SETTING_KEY: parsed})
    _invalidate(AUTO_UNLOAD_IDLE_SETTING_KEY)
    return parsed


def get_model_overrides() -> dict[str, dict]:
    """Per-model launch overrides keyed by model id ({llama_extra_args, max_seq_length})."""
    raw = _cached_setting(MODEL_OVERRIDES_SETTING_KEY, None)
    return raw if isinstance(raw, dict) else {}


def get_model_override(model_id: str) -> dict:
    """The launch override applied when auto-switch loads ``model_id`` (or empty)."""
    override = get_model_overrides().get(model_id)
    return override if isinstance(override, dict) else {}


def set_model_override(
    model_id: str,
    llama_extra_args: Optional[list[str]] = None,
    max_seq_length: Optional[int] = None,
) -> dict:
    """Upsert one model's launch override; an override with no fields removes it."""
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required.")
    entry: dict[str, Any] = {}
    if llama_extra_args:
        entry["llama_extra_args"] = [str(arg) for arg in llama_extra_args]
    if max_seq_length:
        entry["max_seq_length"] = max(0, int(max_seq_length))

    from storage.studio_db import upsert_app_setting_map_entry

    # Atomic per-entry merge so two PUTs for different models can't drop each other.
    upsert_app_setting_map_entry(MODEL_OVERRIDES_SETTING_KEY, model_id.strip(), entry or None)
    _invalidate(MODEL_OVERRIDES_SETTING_KEY)
    return entry
