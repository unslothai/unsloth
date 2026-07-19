# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-use, short-lived confirmation tokens for the llama.cpp host-binary swap.

The update runs an OS installer that replaces the binary on the machine running
Studio, so it must not fire from an unconfirmed click, a stale banner, or a replay.
Confirmation is required uniformly for every caller (not gated on "same machine",
since a headless SSH server has no host-local session). A token binds its offered
build (``target_tag``), is single-use with a short TTL, and lives in-process
(matching the single-process backend; use an HMAC stateless token for multi-worker).
"""

from __future__ import annotations

import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

# How long a freshly minted confirmation token stays valid.
CONFIRM_TOKEN_TTL_SECONDS = 300
# Cap the store so unapplied confirm calls can't grow memory unbounded; oldest first.
_MAX_TOKENS = 64

_lock = threading.Lock()
# token -> (target_tag, expires_at_monotonic)
_tokens: "dict[str, Tuple[str, float]]" = {}


def _iso(ts_epoch: float) -> str:
    return (
        datetime.fromtimestamp(ts_epoch, tz = timezone.utc)
        .replace(microsecond = 0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _purge_expired_locked(now: float) -> None:
    expired = [tok for tok, (_tag, exp) in _tokens.items() if exp <= now]
    for tok in expired:
        _tokens.pop(tok, None)


def mint_confirm_token(
    target_tag: str, *, ttl_seconds: int = CONFIRM_TOKEN_TTL_SECONDS
) -> Tuple[str, str]:
    """Mint a single-use token bound to ``target_tag``.

    Returns ``(token, expires_at_iso)``. ``expires_at_iso`` is wall-clock UTC for
    display; validity itself is tracked on a monotonic clock so a system time
    change cannot extend or shorten it.
    """
    now_mono = time.monotonic()
    now_wall = time.time()
    token = secrets.token_urlsafe(32)
    with _lock:
        _purge_expired_locked(now_mono)
        if len(_tokens) >= _MAX_TOKENS:
            oldest = min(_tokens, key = lambda t: _tokens[t][1])  # evict nearest expiry
            _tokens.pop(oldest, None)
        _tokens[token] = (target_tag, now_mono + ttl_seconds)
    return token, _iso(now_wall + ttl_seconds)


def consume_confirm_token(
    token: Optional[str], current_target_tag: str
) -> Tuple[bool, Optional[str]]:
    """Validate and consume a token for a swap to ``current_target_tag``.

    Returns ``(ok, reason)``. On success ``(True, None)`` and the token is burned
    (single use). On failure ``ok`` is False and ``reason`` is one of:
    ``invalid_token`` (missing/unknown), ``expired_token``, ``stale_target``
    (bound to a different build than the one about to install).
    """
    if not token:
        return False, "invalid_token"
    now_mono = time.monotonic()
    with _lock:
        # Pop first so any outcome consumes the token, and an expired hit still
        # reports "expired" rather than collapsing to "invalid".
        entry = _tokens.pop(token, None)
        _purge_expired_locked(now_mono)
        if entry is None:
            return False, "invalid_token"
        target_tag, expires_at = entry
        if expires_at <= now_mono:
            return False, "expired_token"
        if target_tag != (current_target_tag or ""):
            return False, "stale_target"
    return True, None


def reset_tokens_for_tests() -> None:
    """Test-only: clear the token store."""
    with _lock:
        _tokens.clear()
