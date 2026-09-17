# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serialization and bookkeeping for media model switches.

One switch at a time per backend, so two requests cannot race the single pipeline slot, plus a
single cross-backend lock every GPU-taking switch queues on. Without the latter two switchers
each see the other as work they would interrupt and refuse each other; queueing makes the second
a waiter instead.

The counters exist because the request that is switching is itself tracked by the middleware.
A request parked on a switch lock holds no work, and a request performing a switch is not using
the backend it is counted against, so both are discounted when a switch asks whether anything
else is running.

The locks are per running loop, like ``_auto_switch_lock`` in ``routes.inference``: a
module-level ``asyncio.Lock`` binds to the loop that first awaited it and hangs a second one.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import weakref
from typing import Optional

# not an owner: the key the cross-backend gpu switch lock is stored under
_GPU_SWITCH_KEY = "gpu-switch"

_switch_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_switch_locks_guard = threading.Lock()

_waiters: dict[str, int] = {}
_waiters_guard = threading.Lock()

_switching: dict[str, int] = {}
_switching_guard = threading.Lock()


@contextlib.contextmanager
def note_switcher(owner: str):
    """Mark this request as performing a switch on *owner*, for its whole duration."""
    with _switching_guard:
        _switching[owner] = _switching.get(owner, 0) + 1
    try:
        yield
    finally:
        with _switching_guard:
            remaining = _switching.get(owner, 0) - 1
            if remaining > 0:
                _switching[owner] = remaining
            else:
                _switching.pop(owner, None)


def switcher_count(owner: Optional[str] = None) -> int:
    """Requests currently switching *owner*, or across every backend when it is None."""
    with _switching_guard:
        if owner is None:
            return sum(_switching.values())
        return _switching.get(owner, 0)


@contextlib.contextmanager
def note_waiter(owner: str):
    """Mark this request as parked on *owner*'s switch lock, doing no work of its own."""
    with _waiters_guard:
        _waiters[owner] = _waiters.get(owner, 0) + 1
    try:
        yield
    finally:
        with _waiters_guard:
            remaining = _waiters.get(owner, 0) - 1
            if remaining > 0:
                _waiters[owner] = remaining
            else:
                _waiters.pop(owner, None)


def waiter_count(owner: str) -> int:
    """Requests parked on *owner*'s switch lock."""
    with _waiters_guard:
        return _waiters.get(owner, 0)


def gpu_switch_lock() -> asyncio.Lock:
    """The single lock every GPU-taking media switch queues on, per running loop."""
    return switch_lock(_GPU_SWITCH_KEY)


def switch_lock(owner: str) -> asyncio.Lock:
    """The switch lock for *owner* on the running loop, created on first use."""
    loop = asyncio.get_running_loop()
    # weakkeydictionary mutation is not thread-safe, so guard the get-or-create
    with _switch_locks_guard:
        per_owner = _switch_locks.get(loop)
        if per_owner is None:
            per_owner = _switch_locks[loop] = {}
        lock = per_owner.get(owner)
        if lock is None:
            lock = per_owner[owner] = asyncio.Lock()
        return lock


__all__ = [
    "gpu_switch_lock",
    "note_switcher",
    "note_waiter",
    "switch_lock",
    "switcher_count",
    "waiter_count",
]
