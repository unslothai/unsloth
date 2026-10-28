# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a conversation with no saved thread was last cut, so later requests cut in the same place and reuse
the prompt cache. Process-local and in memory only: a restart costs one recut.
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
_entries: "OrderedDict[str, tuple[int, int, str, float]]" = OrderedDict()


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
        dropped, span, anchor, seen = entry
        if now - seen > _IDLE_EXPIRY_S:
            del _entries[key]
            return 0
        _entries.move_to_end(key)
        _entries[key] = (dropped, span, anchor, now)
    _, turns = _split(messages)
    # Up to the last removed turn: a conversation sharing only the opening (another user's, or edited) differs here.
    if span >= len(turns) or _digest(turns[:span]) != anchor:
        return 0
    return dropped


def remember(
    messages: list[dict],
    kept: list[dict],
    dropped: int,
    scope = None,
) -> None:
    """Record that ``messages`` (the uncut request) became ``kept`` by losing ``dropped`` oldest turns."""
    key = _key(messages, scope)
    _, turns = _split(messages)
    if key is None or not 0 < dropped < len(turns):
        return
    kept_ids = {id(message) for message in kept}
    # By identity, as the evictor works: kept protected turns can sit between removed ones.
    removed = [index for index, message in enumerate(turns) if id(message) not in kept_ids]
    span = removed[dropped - 1] + 1 if len(removed) >= dropped else dropped
    entry = (int(dropped), span, _digest(turns[:span]), time.monotonic())
    with _lock:
        _entries[key] = entry
        _entries.move_to_end(key)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last = False)


def clear() -> None:
    with _lock:
        _entries.clear()
