# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent callers with the same key share one in-flight call. The update check, the backend picker in two tabs and the switch planner can all ask for the same resolve at once; each answer is only as fresh as the call that produced it, so a caller that arrives while one runs takes that answer instead of starting a second resolver child or GitHub read."""

from __future__ import annotations

import threading
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_inflight: dict[Hashable, dict] = {}


def run(key: Hashable, fn: Callable[[], T]) -> T:
    """Call ``fn`` unless a call for ``key`` is already running; then wait for it and return its result. A raising leader re-raises in every waiter."""
    with _lock:
        entry = _inflight.get(key)
        leader = entry is None
        if leader:
            entry = {"done": threading.Event(), "value": None, "error": None}
            _inflight[key] = entry
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
