# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-held switch for whether managed accounts may point a provider connection at a private address."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY = "managed_private_provider_urls_allowed"
# Default off, which is what every installation does today: a managed account's provider base URL is
# caller-controlled server-side egress, so it reaches public addresses only until the owner says the
# people on this installation are trusted with the loopback and LAN services the host can see.
DEFAULT_MANAGED_PRIVATE_PROVIDER_URLS_ALLOWED = False

# The shared-host opt-in refuses private addresses for EVERY account, the owner included, so it is
# the stricter of the two statements and wins: an operator who set it in the environment does not
# get it undone from a settings page.
BLOCK_PRIVATE_ENV = "UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS"

# Every outbound provider request from a managed account asks this question, and the answer costs a
# fresh SQLite connection (measured at ~600us, essentially all of it connection setup) on the event
# loop shared by every concurrent request. Held briefly instead. The window is short and one-sided in
# practice: a write through this module drops the entry immediately, so the flip an owner just made
# is live in this process at once, and the TTL only bounds how long another process may still be
# answering from its own copy.
_CACHE_TTL_SECONDS = 1.0
_cache_lock = threading.Lock()
# (expiry, value); None means nothing remembered. A failed read is never remembered, so an
# unreadable settings DB keeps failing closed per call rather than pinning that answer for a second.
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

    The setting is installation-wide, so it is read from the owner's store whoever is asking. A
    *missing* setting keeps the refusal that shipped, and a *read failure* (a transient SQLite or
    permission error) fails closed for the same reason the preview kill switch does: an unreadable
    settings DB must not quietly widen what a managed account can dial.
    """
    # Read before the cache, so setting the variable takes hold at once rather than after the TTL,
    # and so the strict answer is never the one being held.
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

    # Bound to the owner like the read is. The route that calls this is owner-only, but the two
    # halves have to name one store, or a write from anywhere else lands where the reader never looks.
    run_as(OWNER, upsert_app_settings, {MANAGED_PRIVATE_PROVIDER_URLS_SETTING_KEY: parsed})
    # Dropped rather than replaced with `parsed`: the next reader then answers from the store, which
    # is the thing that decides, so a write that did not land cannot be believed for a second after.
    forget_cached_setting()
    return parsed
