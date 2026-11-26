# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serializes switches: one lock per backend and one cross-backend GPU lock all GPU-taking switches
queue on; locks are per event loop, since a module-level asyncio.Lock hangs a second loop."""

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
    with _switching_guard:
        if owner is None:
            return sum(_switching.values())
        return _switching.get(owner, 0)


@contextlib.contextmanager
def note_waiter(owner: str):
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
    with _waiters_guard:
        return _waiters.get(owner, 0)


def gpu_switch_lock() -> asyncio.Lock:
    return switch_lock(_GPU_SWITCH_KEY)


def switch_lock(owner: str) -> asyncio.Lock:
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
