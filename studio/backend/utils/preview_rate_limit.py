# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coarse per-IP sliding-window rate limit for the public ``/p`` preview chat.

A signed link stops ref guessing, but anyone with a link can still drive GPU
generation. This bounds sustained abuse from a single source. In-process and
single-worker only (like the login limiter in ``routes/auth.py``); Unsloth runs as
one uvicorn process, so a shared store isn't needed.
"""

from __future__ import annotations

import threading
import time
from collections import deque

# Window / ceiling for preview chat-completions per client IP.
_WINDOW_SECONDS = 60.0
_MAX_REQUESTS = 20
# Bound memory on a public surface (many distinct IPs).
_MAX_BUCKETS = 4096

_buckets: dict[str, deque] = {}
_lock = threading.Lock()


def _prune(bucket: deque, now: float) -> None:
    while bucket and now - bucket[0] > _WINDOW_SECONDS:
        bucket.popleft()


def _evict_aged(now: float) -> None:
    """Drop only buckets that have fully aged out. Never evict an active bucket:
    evicting a throttled key would reset its counter, so a flood of distinct keys
    could cycle the table and clear a victim's (or its own) limit."""
    for key in list(_buckets.keys()):
        _prune(_buckets[key], now)
        if not _buckets[key]:
            del _buckets[key]


def check_rate_limit(key: str) -> int:
    """Record a hit for ``key``; return seconds-to-wait if over the limit, else 0."""
    now = time.monotonic()
    with _lock:
        bucket = _buckets.get(key)
        if bucket is None:
            if len(_buckets) >= _MAX_BUCKETS:
                _evict_aged(now)
            if len(_buckets) >= _MAX_BUCKETS:
                # Table is full of currently-active clients. Fail closed: deny the
                # new key rather than evict a live bucket (which would hand out a
                # rate-limit reset). Pathological only (>= _MAX_BUCKETS live IPs).
                return max(1, int(_WINDOW_SECONDS))
            bucket = _buckets[key] = deque()
        _prune(bucket, now)
        if len(bucket) >= _MAX_REQUESTS:
            return max(1, int(_WINDOW_SECONDS - (now - bucket[0])) + 1)
        bucket.append(now)
        return 0


def reset() -> None:
    """Clear all buckets (test isolation)."""
    with _lock:
        _buckets.clear()
