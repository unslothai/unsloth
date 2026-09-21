# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which durable runs currently have a browser watching them.

"Durable" means cancel_on_disconnect is off, not that the tab is gone, so nothing downstream could
tell a watching user from an abandoned run. ``state.tool_approvals.wait_tool_decision`` needs that:
its park ceiling is only defensible for a run nobody is watching.

Keyed by (account, run) then by follower. Account, because studio.db is per-account
(``utils.paths.storage_roots.studio_db_path``) and the run id is the client's
(``CreateChatGenerationRun.runId``), so two accounts can hold the same id and one tenant's follower
would answer for another's run. Follower, because two tabs (or a reconnect overlapping the stream it
replaces) share a run, and a single stamp let either one's cleanup delete the other's heartbeat.

A heartbeat rather than a reference count: a count must be decremented by a ``finally`` an abandoned
generator may never reach, and a leaked increment would park an abandoned approval forever.
"""

from __future__ import annotations

import threading
import time
import uuid

# Three follower keep-alive periods. Errs towards present: a stale stamp costs an idle slot for 45s,
# reading a present user as absent costs them the decision.
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
    """Record that ``follower`` is attached to ``account_id``'s ``run_id`` right now.

    Called once per iteration of the run's SSE loop, so at least every keep-alive period.
    """
    if not run_id or not follower:
        return
    with _LOCK:
        _SEEN.setdefault(_key(account_id, run_id), {})[follower] = time.monotonic()


def subscriber_departed(
    run_id: str,
    follower: str,
    account_id: str = "",
) -> None:
    """Drop only ``follower``'s stamp when its loop exits, leaving any other follower's alone.

    Best effort: an abruptly closed generator never runs its cleanup, which is why ``is_attended``
    expires by age rather than trusting this. This only buys promptness on a clean close.
    """
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
