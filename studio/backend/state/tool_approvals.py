# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call tool-call confirmation gate.

When a chat request sets ``confirm_tool_calls``, the agentic loop pauses
before executing each tool and waits here for the user's decision, which
arrives via ``POST /api/inference/tool-confirm`` on a separate connection.

Each gated call is identified by a unique ``approval_id`` (minted with
``new_approval_id``) that the loop both registers here and echoes in the
``tool_start`` stream event. The frontend sends that exact id back, so a
stale or duplicate confirmation -- or a second tool awaiting a decision in
the same session -- can never resolve the wrong call. ``session_id`` is
kept alongside purely as a scope check.

The slot is registered with ``begin_tool_decision`` *before* the loop
yields ``tool_start``, closing the race where a fast confirmation (or an
auto "Always allow") could otherwise arrive before the waiter exists.
``wait_tool_decision`` then blocks and cleans up its own slot.
"""

import secrets
import threading
from typing import Optional

# Generous ceiling so a user can deliberate; cancellation (stop button /
# disconnect) still breaks the wait early via ``cancel_event``.
_DECISION_TIMEOUT = 3600.0

# Fed to the model as the tool result when the user denies a call, so it
# can adapt and keep responding instead of the turn ending abruptly.
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

_lock = threading.Lock()
# approval_id -> {"event": threading.Event, "decision": str|None, "session": str}
_pending: dict[str, dict] = {}


def new_approval_id() -> str:
    """Mint an unguessable id for one pending tool-call confirmation."""
    return secrets.token_urlsafe(16)


def begin_tool_decision(session_id, approval_id) -> dict:
    """Register a pending decision slot and return it.

    Call this *before* yielding the ``tool_start`` event so the waiter
    always exists by the time the user's confirmation can arrive.
    """
    slot = {
        "event": threading.Event(),
        "decision": None,
        "session": session_id or "",
    }
    with _lock:
        _pending[approval_id] = slot
    return slot


def wait_tool_decision(
    slot,
    approval_id,
    cancel_event = None,
    timeout = _DECISION_TIMEOUT,
):
    """Block on a slot from ``begin_tool_decision`` until the user decides.

    Returns ``"allow"`` or ``"deny"``. Falls back to ``"deny"`` if the wait
    times out or generation is cancelled before the user decides. Always
    removes its own slot on exit.
    """
    try:
        waited = 0.0
        while not slot["event"].wait(timeout = 0.5):
            if cancel_event is not None and cancel_event.is_set():
                return "deny"
            waited += 0.5
            if waited >= timeout:
                return "deny"
        return slot["decision"] or "deny"
    finally:
        with _lock:
            if _pending.get(approval_id) is slot:
                _pending.pop(approval_id, None)


def abort_tool_decision(slot, approval_id) -> None:
    """Remove a slot that was announced but never entered ``wait_tool_decision``.

    Streaming wrappers may stop after ``tool_start`` is yielded and before
    the loop resumes into ``wait_tool_decision``. In that case there is no
    waiter to run the normal cleanup path, so the generator close path calls
    this explicitly.
    """
    with _lock:
        if _pending.get(approval_id) is slot:
            _pending.pop(approval_id, None)


def request_tool_decision(
    session_id,
    approval_id,
    cancel_event = None,
    timeout = _DECISION_TIMEOUT,
):
    """Register and wait in one call (when the slot is not needed early)."""
    slot = begin_tool_decision(session_id, approval_id)
    return wait_tool_decision(slot, approval_id, cancel_event = cancel_event, timeout = timeout)


def resolve_tool_decision(
    approval_id,
    decision,
    session_id = None,
) -> bool:
    """Record the user's "allow"/"deny" decision and unblock the loop.

    Returns ``True`` if a pending call matched, ``False`` otherwise (e.g. a
    stale or duplicate confirmation, or a session-scope mismatch).

    The first decision wins: once a slot's event is set, a later (duplicate or
    out-of-order) confirmation for the same id is rejected without mutating the
    recorded decision, so an Allow can never be flipped to Deny in the window
    before the waiter reads ``slot["decision"]`` and pops the slot.
    """
    if not approval_id:
        return False
    with _lock:
        slot = _pending.get(approval_id)
        if not slot:
            return False
        if session_id is not None and slot["session"] != (session_id or ""):
            return False
        if slot["event"].is_set():
            return False
        slot["decision"] = decision
        slot["event"].set()
    return True
