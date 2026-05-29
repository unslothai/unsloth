# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-session tool-call confirmation gate.

When a chat request sets ``confirm_tool_calls``, the agentic loop pauses
before executing each tool and waits here for the user's decision, which
arrives via ``POST /api/inference/tool-confirm`` on a separate connection.

The agentic loop is strictly sequential, so at most one tool awaits a
decision per session at any moment -- the gate keys on ``session_id``
alone and never needs to match individual tool-call ids (which some
models omit).
"""

import threading
from typing import Optional

# Generous ceiling so a user can deliberate; cancellation (stop button /
# disconnect) still breaks the wait early via ``cancel_event``.
_DECISION_TIMEOUT = 3600.0

# Fed to the model as the tool result when the user denies a call, so it
# can adapt and keep responding instead of the turn ending abruptly.
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

_lock = threading.Lock()
# session_key -> {"event": threading.Event, "decision": "allow"|"deny"|None}
_pending: dict[str, dict] = {}


def _key(session_id: Optional[str]) -> str:
    return session_id or ""


def request_tool_decision(session_id, cancel_event=None, timeout=_DECISION_TIMEOUT):
    """Block until the user allows/denies the pending tool call.

    Returns ``"allow"`` or ``"deny"``. Falls back to ``"deny"`` if the wait
    times out or generation is cancelled before the user decides.
    """
    key = _key(session_id)
    event = threading.Event()
    with _lock:
        _pending[key] = {"event": event, "decision": None}
    try:
        waited = 0.0
        while not event.wait(timeout=0.5):
            if cancel_event is not None and cancel_event.is_set():
                return "deny"
            waited += 0.5
            if waited >= timeout:
                return "deny"
        with _lock:
            slot = _pending.get(key)
            return (slot or {}).get("decision") or "deny"
    finally:
        with _lock:
            # Only remove our own entry — never a newer waiter's, in case the
            # same session somehow re-registered while we were waiting.
            if _pending.get(key, {}).get("event") is event:
                _pending.pop(key, None)


def resolve_tool_decision(session_id, decision) -> bool:
    """Record the user's "allow"/"deny" decision and unblock the loop.

    Returns ``True`` if a pending call matched, ``False`` otherwise (e.g. a
    stale or duplicate confirmation).
    """
    key = _key(session_id)
    with _lock:
        slot = _pending.get(key)
        if not slot:
            return False
        slot["decision"] = decision
        slot["event"].set()
    return True
