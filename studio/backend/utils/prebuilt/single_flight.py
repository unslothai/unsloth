# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent callers with the same key share one in-flight call. The update check, the backend picker in two tabs and the switch planner can all ask for the same resolve at once; each answer is only as fresh as the call that produced it, so a caller that arrives while one runs takes that answer instead of starting a second resolver child or GitHub read."""

from __future__ import annotations

import threading
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_inflight: dict[Hashable, dict] = {}
# Per-cache generation, bumped by invalidate(). Kept here keyed by id(cache) rather than inside the cache, so no reader of a cache dict can mistake it for an entry.
_generations: dict[int, int] = {}


def generation(cache: dict) -> int:
    """Current generation of ``cache``: read it before a slow fill and store the result only if it is unchanged, so a fill that started before an install cannot write a pre-install answer after it."""
    with _lock:
        return _generations.get(id(cache), 0)


def invalidate(cache: dict) -> None:
    """Clear ``cache`` and bump its generation, so fills already running neither store their result nor accept new joiners (callers put the generation in their key)."""
    with _lock:
        _generations[id(cache)] = _generations.get(id(cache), 0) + 1
        cache.clear()


def run(key: Hashable, fn: Callable[[], T]) -> T:
    """Call ``fn`` unless a call for ``key`` is already running; then wait for it and return its result. A raising leader re-raises in every waiter. A re-entrant call from the leader's own thread runs ``fn`` directly instead of waiting on itself forever."""
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
