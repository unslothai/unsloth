# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call tool-call confirmation gate.

When a chat request sets ``confirm_tool_calls``, the agentic loop pauses before executing each tool and waits here for the user's decision, which arrives via ``POST /api/inference/tool-confirm`` on a separate connection.

Each gated call is identified by a unique ``approval_id`` (minted with ``new_approval_id``) that the loop registers here and echoes in the ``tool_start`` stream event. The frontend sends that exact id back, so a stale or duplicate confirmation, or a second tool awaiting a decision in the same session, can never resolve the wrong call; ``session_id`` is kept alongside purely as a scope check.

The slot is registered with ``begin_tool_decision`` *before* the loop yields ``tool_start``, closing the race where a fast confirmation (or an auto "Always allow") could arrive before the waiter exists. ``wait_tool_decision`` then blocks and cleans up its own slot.
"""

import os
import secrets
import threading
from typing import Optional

from state import run_subscribers

# The stop button or a disconnect breaks the wait early via cancel_event.
_DECISION_TIMEOUT = 3600.0

# Durable approval wait before auto-deny. 0 denies at the first 500ms poll, not instantly.
_PARK_TIMEOUT_DEFAULT_S = 300.0


def _park_timeout_from_env() -> float:
    """Read once at import so a run cannot change it mid-flight; a bad value falls back, never raises."""
    raw = os.environ.get("UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S")
    if raw is None or not raw.strip():
        return _PARK_TIMEOUT_DEFAULT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(
            f"[unsloth] UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S={raw!r} is not a number; "
            f"using {_PARK_TIMEOUT_DEFAULT_S:g}s",
        )
        return _PARK_TIMEOUT_DEFAULT_S
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        print(
            f"[unsloth] UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S={raw!r} is out of range; "
            f"using {_PARK_TIMEOUT_DEFAULT_S:g}s",
        )
        return _PARK_TIMEOUT_DEFAULT_S
    return value


_PARK_TIMEOUT_S = _park_timeout_from_env()

# Far under the 1200s lease timeout, far over the 0.5s poll.
_LEASE_RENEW_EVERY_S = 30.0

TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

# Distinct from TOOL_REJECTED_MESSAGE: the user never declined it.
TOOL_APPROVAL_EXPIRED_MESSAGE = (
    "This tool call was not run: nobody answered the approval request in time."
)

# On the slot, not the return value: tests patch wait_tool_decision with fakes returning bare
# "allow"/"deny".
DECISION_ANSWERED = "answered"
DECISION_CANCELLED = "cancelled"
DECISION_EXPIRED = "expired"


def decision_reason(slot) -> Optional[str]:
    """None means no reason was recorded, so the caller falls back to TOOL_REJECTED_MESSAGE."""
    if not isinstance(slot, dict):
        return None
    return slot.get("reason")


_lock = threading.Lock()
# approval_id -> {"event": threading.Event, "decision": str|None, "session": str}
_pending: dict[str, dict] = {}


def new_approval_id() -> str:
    """Mint an unguessable id for one pending tool-call confirmation."""
    return secrets.token_urlsafe(16)


def begin_tool_decision(session_id, approval_id) -> dict:
    """Register a pending decision slot and return it. Call this *before* yielding the ``tool_start`` event so the waiter always exists by the time the user's confirmation can arrive."""
    slot = {
        "event": threading.Event(),
        "decision": None,
        "session": session_id or "",
        "reason": None,
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
    """Records why in slot['reason']: a bare deny cannot tell a refusal from an unanswered approval."""
    park = bool(getattr(cancel_event, "durable", False))
    run_id = getattr(cancel_event, "durable_run_id", "") or ""
    # Run ids are account-local, so ask attendance under the same account.
    account_id = getattr(cancel_event, "durable_account_id", "") or ""
    renew_lease = getattr(cancel_event, "renew_lease", None)

    def _settle(verdict, reason):
        if isinstance(slot, dict):
            slot["reason"] = reason
        return verdict

    try:
        # `waited` resets when a follower is seen; `total` never does. Both bound a park.
        waited = 0.0
        total = 0.0
        last_renew: Optional[float] = None
        while not slot["event"].wait(timeout = 0.5):
            if cancel_event is not None and cancel_event.is_set():
                return _settle("deny", DECISION_CANCELLED)
            waited += 0.5
            total += 0.5
            if not park:
                if total >= timeout:
                    return _settle("deny", DECISION_EXPIRED)
                continue
            # Durable parks ignore the caller's timeout; this is the attended backstop.
            if total >= _DECISION_TIMEOUT:
                return _settle("deny", DECISION_EXPIRED)
            if run_subscribers.is_attended(run_id, account_id):
                waited = 0.0
                # Renew the RUN lease too, or the sweeper settles it at 1200s and cancels the wait.
                if renew_lease is not None and (
                    last_renew is None or total - last_renew >= _LEASE_RENEW_EVERY_S
                ):
                    last_renew = total
                    try:
                        renew_lease()
                    except Exception:
                        # A failed renewal is the sweeper's problem; keep collecting the decision.
                        pass
            elif waited >= _PARK_TIMEOUT_S:
                return _settle("deny", DECISION_EXPIRED)
        return _settle(slot["decision"] or "deny", DECISION_ANSWERED)
    finally:
        with _lock:
            if _pending.get(approval_id) is slot:
                _pending.pop(approval_id, None)


def tool_decision_is_pending(approval_id, session_id = None) -> bool:
    """Asked of the state, not inferred: a reopened tab cannot tell parked calls from answered ones."""
    if not approval_id:
        return False
    with _lock:
        slot = _pending.get(approval_id)
        if not slot:
            return False
        if session_id is not None and slot["session"] != (session_id or ""):
            return False
        return not slot["event"].is_set()


def abort_tool_decision(slot, approval_id) -> None:
    """Remove a slot that was announced but never entered ``wait_tool_decision``. Streaming wrappers may stop after ``tool_start`` is yielded and before the loop resumes into ``wait_tool_decision``, leaving no waiter to run the normal cleanup path, so the generator close path calls this explicitly."""
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
    """Record the user's "allow"/"deny" decision and unblock the loop. Returns ``True`` if a pending call matched, ``False`` otherwise (a stale or duplicate confirmation, or a session-scope mismatch). The first decision wins: once a slot's event is set, a later confirmation for the same id is rejected without mutating the recorded decision, so an Allow can never be flipped to Deny in the window before the waiter reads ``slot["decision"]`` and pops the slot."""
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
