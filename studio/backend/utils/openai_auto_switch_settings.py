# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted opt-in controls for OpenAI-compatible model auto-switching.

Two settings, both off by default so existing API behavior is unchanged:
- ``openai_api_auto_switch_model``: when on, a ``/v1`` request whose ``model``
  names a downloaded local GGUF different from the loaded one transparently
  loads it before serving (llama-swap-style). Unknown names pass through.
- ``openai_api_auto_unload_idle_seconds``: when > 0, the loaded GGUF is
  unloaded after this many idle seconds to free VRAM. Enabled values have a
  60s floor (0 stays "off"): a tiny TTL tears the model down between turns of
  an active chat, forcing a full weight reload + prompt re-prefill per turn.

The idle TTL can also be set at startup via the ``UNSLOTH_MODEL_IDLE_TTL`` env
var. Unlike the stored setting (which stays gated on auto-switch), the env value
is a standalone default that enables idle-unload even with auto-switch off, for
headless/container deploys; an explicit UI/API value still overrides it.

Reads are cached for a short window because these are consulted on the
per-request hot path; writes invalidate the cache.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

OPENAI_AUTO_SWITCH_SETTING_KEY = "openai_api_auto_switch_model"
AUTO_UNLOAD_IDLE_SETTING_KEY = "openai_api_auto_unload_idle_seconds"
MODEL_OVERRIDES_SETTING_KEY = "openai_api_auto_switch_overrides"
MODEL_IDLE_TTL_ENV_VAR = "UNSLOTH_MODEL_IDLE_TTL"

DEFAULT_OPENAI_AUTO_SWITCH_ENABLED = False
DEFAULT_AUTO_UNLOAD_IDLE_SECONDS = 0
MIN_AUTO_UNLOAD_IDLE_SECONDS = 60

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


def _apply_idle_floor(seconds: int) -> int:
    return 0 if seconds <= 0 else max(MIN_AUTO_UNLOAD_IDLE_SECONDS, seconds)


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


def _stored_idle_seconds() -> Optional[int]:
    """The persisted idle TTL as an int, or None when never set."""
    return _coerce_int(_cached_setting(AUTO_UNLOAD_IDLE_SETTING_KEY, None))


_env_floor_warned = False


def _env_idle_seconds() -> Optional[int]:
    """UNSLOTH_MODEL_IDLE_TTL as a non-negative seconds value, or None if unset/invalid.

    Floored to MIN_AUTO_UNLOAD_IDLE_SECONDS here (with a one-time warning) since
    headless/container deploys have no UI to surface a validation error."""
    raw = os.environ.get(MODEL_IDLE_TTL_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    parsed = _coerce_int(raw)
    if parsed is None:
        return None
    floored = _apply_idle_floor(parsed)
    if floored != parsed:
        global _env_floor_warned
        if not _env_floor_warned:
            _env_floor_warned = True
            from loggers import get_logger
            get_logger(__name__).warning(
                "%s=%s is below the %ss minimum; using %ss",
                MODEL_IDLE_TTL_ENV_VAR,
                parsed,
                MIN_AUTO_UNLOAD_IDLE_SECONDS,
                floored,
            )
    return floored


def get_stored_auto_unload_idle_seconds() -> int:
    """The persisted idle-unload TTL, independent of whether auto-switch is on.

    The settings UI reads this so it can display and round-trip the saved value;
    toggling auto-switch off must not erase it. Falls back to the env override so
    the UI shows the startup default. The idle loop uses the gated reader below.
    """
    stored = _stored_idle_seconds()
    if stored is not None:
        # Floor legacy values persisted before the minimum existed, so the UI
        # displays the effective TTL and round-trips it cleanly.
        return _apply_idle_floor(stored)
    env = _env_idle_seconds()
    return env if env is not None else DEFAULT_AUTO_UNLOAD_IDLE_SECONDS


def get_auto_unload_idle_seconds() -> int:
    """Effective idle TTL the idle loop runs on (0 = never unload)."""
    stored = _stored_idle_seconds()
    if stored is not None:
        # An explicit UI/API value stays gated on auto-switch: off reports 0 so the
        # off state is identical to pre-feature. Floored to cover values persisted
        # before the minimum existed.
        return _apply_idle_floor(stored) if get_openai_auto_switch_enabled() else 0
    # No stored value: UNSLOTH_MODEL_IDLE_TTL is a standalone startup default that
    # enables idle-unload even with auto-switch off (headless/container deploys).
    env = _env_idle_seconds()
    return env if env is not None else 0


def set_openai_auto_switch(enabled: Any, idle_seconds: Any) -> tuple[bool, int]:
    """Set both auto-switch flags in one transaction so a settings PUT can't leave
    one key updated and the other stale. Both values are coerced before any write,
    so an invalid value raises without persisting either."""
    parsed_enabled = _coerce_bool(enabled)
    if parsed_enabled is None:
        raise ValueError("OpenAI auto-switch must be true or false.")
    parsed_idle = _coerce_int(idle_seconds)
    if parsed_idle is None:
        raise ValueError("Auto-unload idle seconds must be a non-negative integer.")
    if 0 < parsed_idle < MIN_AUTO_UNLOAD_IDLE_SECONDS:
        raise ValueError(
            f"Auto-unload idle seconds must be 0 (off) or at least "
            f"{MIN_AUTO_UNLOAD_IDLE_SECONDS}."
        )
    from storage.studio_db import upsert_app_settings

    upsert_app_settings(
        {OPENAI_AUTO_SWITCH_SETTING_KEY: parsed_enabled, AUTO_UNLOAD_IDLE_SETTING_KEY: parsed_idle}
    )
    _invalidate(OPENAI_AUTO_SWITCH_SETTING_KEY)
    _invalidate(AUTO_UNLOAD_IDLE_SETTING_KEY)
    return parsed_enabled, parsed_idle


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
