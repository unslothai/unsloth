# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs watched by a browser, keyed by (account, run) then follower; durable means
cancel_on_disconnect off, not tab gone; wait_tool_decision park ceiling relies on it; heartbeats,
not refcount"""

from __future__ import annotations

import threading
import time
import uuid

# Three follower keep-alive periods (45s); errs towards present, since a stale stamp only idles a slot but misreading a present user costs the decision.
_ATTENDED_FOR_S = 45.0

_LOCK = threading.Lock()
# (account_id, run_id) -> {follower token: monotonic seconds at that follower's last heartbeat}
_SEEN: dict[tuple[str, str], dict[str, float]] = {}


def _key(account_id: str, run_id: str) -> tuple[str, str]:
    # A missing account is its own scope, never a wildcard matching accounts that name themselves.
    return (account_id or "", run_id)


def new_follower_token() -> str:
    """Identify one follower for the life of its stream, so it can only clear its own stamp."""
    return uuid.uuid4().hex


def mark_subscriber_seen(
    run_id: str,
    follower: str,
    account_id: str = "",
) -> None:
    """Records that follower is attached to the run now; called each SSE loop iteration, so the stamp refreshes at least once per keep-alive period."""
    if not run_id or not follower:
        return
    with _LOCK:
        _SEEN.setdefault(_key(account_id, run_id), {})[follower] = time.monotonic()


def subscriber_departed(
    run_id: str,
    follower: str,
    account_id: str = "",
) -> None:
    """Drops only this follower stamp on loop exit; best effort, since an abruptly closed generator skips it and is_attended expires by age."""
    if not run_id or not follower:
        return
    key = _key(account_id, run_id)
    with _LOCK:
        followers = _SEEN.get(key)
        if followers is None:
            return
        followers.pop(follower, None)
        if not followers:
            _SEEN.pop(key, None)


def is_attended(run_id: str, account_id: str = "") -> bool:
    """Whether ANY follower of ``account_id``'s ``run_id`` has been heard from recently."""
    if not run_id:
        return False
    key = _key(account_id, run_id)
    cutoff = time.monotonic() - _ATTENDED_FOR_S
    with _LOCK:
        followers = _SEEN.get(key)
        if not followers:
            return False
        # Expired lazily; a reaper would need a thread for what this loop already does.
        for token, seen in list(followers.items()):
            if seen <= cutoff:
                del followers[token]
        if not followers:
            _SEEN.pop(key, None)
            return False
        return True


def attendance_for_tests(run_id: str, account_id: str = "") -> int:
    """How many followers are currently stamped for ``account_id``'s ``run_id``."""
    with _LOCK:
        return len(_SEEN.get(_key(account_id, run_id)) or {})


def reset_for_tests() -> None:
    with _LOCK:
        _SEEN.clear()
