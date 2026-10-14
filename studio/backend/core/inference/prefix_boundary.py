# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a conversation with no saved thread was last cut, so the next request cuts in the same place.

An API client resends its whole history every request and has no Studio thread to save the cut on, so
the oldest kept turn moved almost every request and the prompt cache matched only the system prompt.
Remembering the cut makes it an occasional event: requests in between only append.

Process-local and in memory only: a restart forgets, which costs one recut.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Optional

_MAX_ENTRIES = 256
_IDLE_EXPIRY_S = 6 * 3600
# Turns that identify a conversation: they never change as it grows.
_KEY_TURNS = 2

_lock = threading.Lock()
_entries: "OrderedDict[str, tuple[int, str, float]]" = OrderedDict()


def _digest(*parts) -> str:
    payload = json.dumps(parts, sort_keys = True, default = str, ensure_ascii = False)
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def _split(messages: list[dict]) -> tuple[list, list[dict]]:
    system = [message.get("content") for message in messages if message.get("role") == "system"]
    return system, [message for message in messages if message.get("role") != "system"]


def _key(messages: list[dict], scope) -> Optional[str]:
    system, turns = _split(messages)
    if len(turns) <= _KEY_TURNS:
        return None
    return _digest(scope, system, turns[:_KEY_TURNS])


def _anchor(turns: list[dict], dropped: int) -> str:
    # Everything the cut removed: a conversation that only shares the opening (another user's, or this one edited)
    # differs somewhere in it, so it never borrows another's cut.
    return _digest(turns[:dropped])


def recall(messages: list[dict], scope = None) -> int:
    """The cut this conversation last took, or 0. Only replayed onto the same history."""
    key = _key(messages, scope)
    if key is None:
        return 0
    now = time.monotonic()
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return 0
        dropped, anchor, seen = entry
        if now - seen > _IDLE_EXPIRY_S:
            del _entries[key]
            return 0
        _entries.move_to_end(key)
        _entries[key] = (dropped, anchor, now)
    _, turns = _split(messages)
    if dropped >= len(turns) or _anchor(turns, dropped) != anchor:
        return 0
    return dropped


def remember(
    messages: list[dict],
    dropped: int,
    scope = None,
) -> None:
    """Record that ``messages`` (the uncut request) lost its ``dropped`` oldest non-system turns."""
    key = _key(messages, scope)
    _, turns = _split(messages)
    if key is None or not 0 < dropped < len(turns):
        return
    entry = (int(dropped), _anchor(turns, dropped), time.monotonic())
    with _lock:
        _entries[key] = entry
        _entries.move_to_end(key)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last = False)


def clear() -> None:
    with _lock:
        _entries.clear()
