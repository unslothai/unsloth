# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invalidation for the per-path "schema already created" caches.

Each store skips ``_ensure_schema`` once it has seen a database path. The path is
derived from the account name, so deleting an account and recreating the name
puts a brand new empty file at a path the cache still calls ready, and the first
query fails with ``no such table``.

Retirement is rare and ``_ensure_schema`` is idempotent, so every registered
cache is cleared wholesale rather than matched by prefix: the cost is one extra
schema check per database, and there is no path-normalisation case to get wrong.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_caches: list[set[str]] = []
# Bumped by every invalidation. A store reads it before it starts creating a
# schema and passes it back to mark_ready(), so a clear that lands while that
# work is in flight is not undone by the add that follows it: without this the
# retired path went straight back into the cache, and the namesake's brand new
# empty database was then treated as ready.
_generation = 0


def register(cache: set[str]) -> None:
    """Register a store's ready-path set so retirement can invalidate it."""
    with _lock:
        if not any(cache is known for known in _caches):
            _caches.append(cache)


def generation() -> int:
    """The current invalidation generation. Read BEFORE creating a schema."""
    with _lock:
        return _generation


def mark_ready(cache: set[str], key: str, at_generation: int) -> None:
    """Record ``key`` as schema-ready, unless an invalidation intervened.

    ``at_generation`` is what :func:`generation` returned before the schema work
    began. If it has moved since, the path this call would cache may already have
    been retired out from under it, so the store is left to check once more. That
    costs one redundant ``_ensure_schema``, which is idempotent.
    """
    with _lock:
        if at_generation == _generation:
            cache.add(key)


def forget_all() -> None:
    """Forget every cached path. Call when a workspace directory is retired."""
    global _generation
    with _lock:
        _generation += 1
        caches = list(_caches)
    for cache in caches:
        cache.clear()
