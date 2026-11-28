# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent callers with the same key share one in-flight call (one resolver child / GitHub read)."""

from __future__ import annotations

import threading
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_inflight: dict[Hashable, dict] = {}
# Per-cache generation keyed by id(cache), kept outside the dict so it is never mistaken for an entry.
_generations: dict[int, int] = {}


def generation(cache: dict) -> int:
    """Read before a slow fill; store only if unchanged afterwards."""
    with _lock:
        return _generations.get(id(cache), 0)


def invalidate(cache: dict) -> None:
    """Clear ``cache`` and bump its generation (callers key on it, so running fills get no joiners)."""
    with _lock:
        _generations[id(cache)] = _generations.get(id(cache), 0) + 1
        cache.clear()


def run(key: Hashable, fn: Callable[[], T]) -> T:
    """Run ``fn`` or join the running call for ``key``; errors propagate to waiters. Re-entrant calls run directly (no self-deadlock)."""
    me = threading.get_ident()
    with _lock:
        entry = _inflight.get(key)
        reentrant = entry is not None and entry["thread"] == me
        leader = entry is None
        if leader:
            entry = {"done": threading.Event(), "value": None, "error": None, "thread": me}
            _inflight[key] = entry
    if reentrant:
        return fn()
    if not leader:
        entry["done"].wait()
        if entry["error"] is not None:
            raise entry["error"]
        return entry["value"]
    try:
        entry["value"] = fn()
    except BaseException as exc:
        entry["error"] = exc
        raise
    finally:
        with _lock:
            _inflight.pop(key, None)
        entry["done"].set()
    return entry["value"]
