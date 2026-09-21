# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-held switch for whether managed accounts may point a provider connection at a private address."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY = "managed_private_provider_urls_allowed"
DEFAULT_MANAGED_PRIVATE_PROVIDER_URLS_ALLOWED = False

# Refuses private addresses for EVERY account, the owner included, so it outranks the stored
# preference: an operator who set it in the environment does not get it undone from a settings page.
BLOCK_PRIVATE_ENV = "UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS"

# Every outbound provider request from a managed account asks this, and an uncached answer costs a
# fresh SQLite connection (~600us, nearly all of it connection setup) on the shared event loop. A
# write drops the entry, so the TTL only bounds how long ANOTHER process may still answer stale.
_CACHE_TTL_SECONDS = 1.0
_cache_lock = threading.Lock()
_cached: tuple[float, bool] | None = None


def _remembered() -> bool | None:
    with _cache_lock:
        if _cached is not None and _cached[0] > time.monotonic():
            return _cached[1]
    return None


def _remember(value: bool) -> None:
    global _cached
    with _cache_lock:
        _cached = (time.monotonic() + _CACHE_TTL_SECONDS, value)


def forget_cached_setting() -> None:
    """Drop the held answer. Called on write, and available to tests."""
    global _cached
    with _cache_lock:
        _cached = None


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


def private_urls_locked_by_environment() -> bool:
    """True when the shared-host environment opt-in decides this, whatever the stored preference."""
    return os.environ.get(BLOCK_PRIVATE_ENV) == "1"


def get_managed_private_provider_urls_allowed() -> bool:
    """Whether a managed account may use a provider base URL that resolves to a private address.

    Installation-wide, so it is read from the owner's store whoever is asking. A missing setting
    keeps the refusal that shipped, and a read failure fails closed: an unreadable settings DB must
    not quietly widen what a managed account can dial.
    """
    # Both before the cache: the strict answer is never the one being held, and a failed read is
    # never remembered.
    if private_urls_locked_by_environment():
        return False
    held = _remembered()
    if held is not None:
        return held
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        stored = run_as(OWNER, get_app_setting, MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - an unreadable settings DB keeps the stricter answer
        return False
    parsed = _coerce_bool(stored)
    allowed = parsed if parsed is not None else DEFAULT_MANAGED_PRIVATE_PROVIDER_URLS_ALLOWED
    _remember(allowed)
    return allowed


def set_managed_private_provider_urls_allowed(value: Any) -> bool:
    """Persist whether managed accounts may dial private provider addresses."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("The managed-account private provider URL setting must be true or false.")

    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    # Owner-bound like the read: the two halves have to name one store.
    run_as(OWNER, upsert_app_settings, {MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY: parsed})
    # Dropped, not replaced with `parsed`: a write that did not land must not be believed.
    forget_cached_setting()
    return parsed
