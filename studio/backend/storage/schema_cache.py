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


def register(cache: set[str]) -> None:
    """Register a store's ready-path set so retirement can invalidate it."""
    with _lock:
        if not any(cache is known for known in _caches):
            _caches.append(cache)


def forget_all() -> None:
    """Forget every cached path. Call when a workspace directory is retired."""
    with _lock:
        caches = list(_caches)
    for cache in caches:
        cache.clear()
