# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which durable runs currently have a browser watching them.

A durable run does not die when its tab does, which is the whole point of it. The cost is that
nothing downstream can tell the two situations apart any more: "durable" means
``cancel_on_disconnect=False``, not "disconnected". ``state.tool_approvals.wait_tool_decision``
needs that distinction, because the park ceiling it applies to an unanswered approval is only
defensible for a run nobody is watching. Applied to a user who is sitting there reading what
``terminal`` wants to run, it is a five-minute deadline on a decision that used to have an hour.

So the SSE follower stamps the run it is following, and the gate asks. A heartbeat rather than a
reference count: the follower loop in ``routes/chat_generation_runs.py`` comes round at least every
15s whatever else happens, and an entry nobody refreshes simply ages out of _ATTENDED_FOR_S. A
count would have to be decremented by a ``finally`` that a wedged or abandoned generator may never
reach, and a leaked increment fails the dangerous way round: it would park an abandoned approval
forever.

Keyed by ACCOUNT and run id, and within that by follower. The account half matters because
`studio.db` is per-account (`utils.paths.storage_roots.studio_db_path` -> `account_path`) and the
run id is chosen by the client (`CreateChatGenerationRun.runId`), so two accounts on one managed
install can hold the same id. A process-global map keyed on the bare id let one account's follower
report the other's run as attended, which would hold a stranger's approval open and keep renewing
its lease.

Keyed by follower too, because one run can have several: two tabs, or a reconnect whose
replacement stream attaches before the old one finishes unwinding. A single stamp per run made
either follower's cleanup delete a heartbeat the OTHER had just written, and the survivor does not
stamp again until its event wait turns over (up to 15s), so a short
``UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S`` -- 0 is supported and tested -- could deny an approval a
second tab was actively watching. That is the exact failure this module exists to prevent, so
attendance is tracked per follower and a departure removes only that follower's own stamp.

A plain dict plus a threading.Lock, like ``state.active_generations`` next door: no signals, no
event-loop affinity, identical on Linux, macOS, Windows and WSL.
"""

from __future__ import annotations

import threading
import time
import uuid

# Three follower keep-alive periods. Long enough that an ordinary slow round trip never reads as a
# departure, short enough that a closed tab stops holding an approval open for long. The failure
# mode this window chooses is the safe one: at worst an approval stays answerable ~45s past the
# moment its tab went away, which costs an idle inference slot, where reading a present user as
# absent costs them the decision itself.
_ATTENDED_FOR_S = 45.0

_LOCK = threading.Lock()
# (account_id, run_id) -> {follower token: monotonic seconds at that follower's last heartbeat}
_SEEN: dict[tuple[str, str], dict[str, float]] = {}


def _key(account_id: str, run_id: str) -> tuple[str, str]:
    # A missing account is its own scope rather than a wildcard: on a single-user install every
    # caller passes the same empty value, and on a managed one a caller that cannot name its
    # account must not match one that can.
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

    Called once per iteration of the run's SSE loop, which turns over at least every keep-alive
    period. Cheap enough to call unconditionally: one dict write under a lock.
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

    Best effort, and deliberately not the only way an entry goes away: a generator that is closed
    abruptly never runs its cleanup, which is exactly why ``is_attended`` expires stamps by age
    rather than trusting this. What it buys is promptness in the common case, where the user really
    did just close the tab and the ASGI server closes the generator cleanly.
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
        # Expire lazily. A reaper would need a thread to do the same job this loop does, and the
        # map is one entry per open tab on one run.
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
