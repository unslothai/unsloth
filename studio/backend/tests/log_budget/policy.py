# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Log policy classes, derived from the middleware rather than duplicated from it.

A per-path budget table would be a second copy of the suppression rules, and a second copy
drifts. The class of a path is therefore read out of ``loggers.handlers`` at run time; the
only thing checked in here is how often each path is POLLED, which is the one fact the
backend genuinely does not know.

Budgeting by class rather than by path is also what stops the numbers becoming a haggling
ledger: a new endpoint picks an existing class and no number changes at all.
"""

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

# Classes whose 2xx traffic is dropped outright rather than heartbeated, so no window and
# no budget applies: the expected count is zero.
NEVER_LOGGED_ON_SUCCESS = (QUIET_SUCCESS, EXCLUDED)


def watchdog_paths(handlers) -> frozenset:
    """The watchdog set, absent until the liveness-heartbeat change lands.

    Read through ``getattr`` so this harness works on both sides of that merge instead of
    pinning the guard to one revision.
    """
    return frozenset(getattr(handlers, "_WATCHDOG_POLL_PATHS", frozenset()))


def classify(handlers, path: str) -> str:
    """Which suppression rule owns ``path``. Most specific first.

    Order matters: the liveness paths are a subset of the quiet paths, and the watchdog set
    is deliberately outside the quiet set because its window is wider.
    """
    if path in handlers._EXCLUDED_PATHS or path.endswith(handlers._EXCLUDED_SUFFIXES):
        return EXCLUDED
    if path.startswith("/assets/"):
        return EXCLUDED
    # _CHAT_LIST_PATHS rides the same suppressor as _QUIET_SUCCESS_PATHS in
    # `_is_quiet_success`, so it is the same class even though it is a separate set (it
    # carries a second rule about pre-auth 401s that the others do not).
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
    if path in handlers._QUIET_POLL_PATHS:
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
    """How many lines a periodic poll SHOULD produce. A formula, not a snapshot.

    The middleware stamps only when it emits, so the gap between two emitted lines is the
    smallest whole number of polls that spans the window. With ``n`` polls in the run and
    ``k`` polls per emission, that is ``(n - 1) // k + 1``.

    Deriving this rather than recording a measured number is what makes the guard survive a
    change to a poll interval: widen the window and the expectation moves with it, so only
    a genuine regression fails.
    """
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
    """The de-duplication bucket a path competes in.

    The liveness paths deliberately SHARE one bucket: the SPA fires all of them together
    and they all answer the same question, so the first of the burst logs with its real
    path and the rest of that window is dropped. Budgeting them per path would expect five
    lines where the design intends one, so the guard has to budget the bucket.
    """
    if classify(handlers, path) == LIVENESS:
        return "\x00liveness"
    return path
