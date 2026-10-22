# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Policy classes are read from the middleware, so a second per-path copy of its rules cannot drift."""

from __future__ import annotations

import math
from typing import Optional

NORMAL = "normal"
QUIET = "quiet"
LIVENESS = "liveness"
WATCHDOG = "watchdog"
QUIET_SUCCESS = "quiet_success"
EXCLUDED = "excluded"

ALL_CLASSES = (NORMAL, QUIET, LIVENESS, WATCHDOG, QUIET_SUCCESS, EXCLUDED)

NEVER_LOGGED_ON_SUCCESS = (QUIET_SUCCESS, EXCLUDED)


def watchdog_paths(handlers) -> frozenset:
    """Read the watchdog set via getattr so the harness works before and after the heartbeat change."""
    return frozenset(getattr(handlers, "_WATCHDOG_POLL_PATHS", frozenset()))


def _normalize(handlers, path: str) -> str:
    """The middleware's templated-path collapse, absent on older revisions."""
    fn = getattr(handlers, "normalize_poll_path", None)
    return fn(path) if fn else path


def classify(handlers, path: str) -> str:
    """Most specific first: liveness paths are a subset of quiet, and watchdog sits outside it."""
    if path in handlers._EXCLUDED_PATHS or path.endswith(handlers._EXCLUDED_SUFFIXES):
        return EXCLUDED
    if path.startswith("/assets/"):
        return EXCLUDED
    # _CHAT_LIST_PATHS uses the same suppressor as _QUIET_SUCCESS_PATHS.
    if (
        path in handlers._QUIET_SUCCESS_PATHS
        or path in handlers._SELF_READ_PATHS
        or path in handlers._CHAT_LIST_PATHS
    ):
        return QUIET_SUCCESS
    if path in watchdog_paths(handlers):
        return WATCHDOG
    if path in handlers._LIVENESS_POLL_PATHS:
        return LIVENESS
    if _normalize(handlers, path) in handlers._QUIET_POLL_PATHS:
        return QUIET
    return NORMAL


def window_ms(handlers, cls: str) -> Optional[int]:
    """The de-duplication window for a class, or None when 2xx never logs at all."""
    if cls in NEVER_LOGGED_ON_SUCCESS:
        return None
    if cls == WATCHDOG:
        return getattr(handlers, "_WATCHDOG_POLL_DEDUP_MS", handlers._QUIET_POLL_DEDUP_MS)
    if cls in (QUIET, LIVENESS):
        return handlers._QUIET_POLL_DEDUP_MS
    return handlers._ACCESS_LOG_DEDUP_MS


def expected_emissions(window_ms_value: Optional[int], period_s: float, duration_s: float) -> int:
    """Derive the expected line count from the window, so a poll-interval change moves the bound."""
    if window_ms_value is None:
        return 0
    if period_s <= 0:
        raise ValueError("period must be positive")
    polls = math.ceil(duration_s / period_s)
    if polls <= 0:
        return 0
    if window_ms_value <= 0:
        return polls  # window off (--verbose): every poll logs
    polls_per_emission = max(1, math.ceil((window_ms_value / 1000.0) / period_s))
    return (polls - 1) // polls_per_emission + 1


def bucket_of(handlers, path: str) -> str:
    """Liveness paths share one dedup bucket, so the guard must budget the bucket rather than each path."""
    if classify(handlers, path) == LIVENESS:
        return "\x00liveness"
    return _normalize(handlers, path)
