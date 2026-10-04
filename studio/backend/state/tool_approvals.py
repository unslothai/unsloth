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

# Generous ceiling so a user can deliberate; the stop button or a disconnect still breaks the wait early via cancel_event.
_DECISION_TIMEOUT = 3600.0

# How long a durable (parked) approval waits for a human before denying and letting the agent continue.
# Default is generous enough for "grab your phone" but short enough that an unattended agentic loop
# doesn't stall for the full lease window. 0 denies at the first poll, so a decision that lands inside
# that first 500ms still wins; it is "autonomous", not "instant".
_PARK_TIMEOUT_DEFAULT_S = 300.0


def _park_timeout_from_env() -> float:
    """Read the park ceiling once, at import, tolerating a value nobody can parse.

    Read here rather than per wait so one run cannot change ceiling mid-flight. A typo must not be
    fatal: this module is imported on the chat path, so raising would take the backend down at
    startup over an environment variable, and a stuck approval is the lesser failure. A negative or
    non-finite value would disable the ceiling silently, so both fall back too.
    """
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

# Far under the lease timeout (1200s), far over the 0.5s poll: one small write every half minute.
_LEASE_RENEW_EVERY_S = 30.0

# Fed to the model as the tool result when the user denies a call, so it can adapt instead of the turn ending abruptly.
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

# The same, for a call that hit its park ceiling unanswered. Distinct from TOOL_REJECTED_MESSAGE
# because the model and the card read the result verbatim, and by then it is the only account of the
# call the returning user gets: telling them they declined something they never saw is false.
TOOL_APPROVAL_EXPIRED_MESSAGE = (
    "This tool call was not run: nobody answered the approval request in time."
)

# Why wait_tool_decision returned, on the slot rather than in the return value, so the verdict stays
# a bare "allow"/"deny": the three tool loops patch it by name in tests
# (core.inference.llama_cpp.wait_tool_decision and friends), and a fake that knows nothing about
# reasons leaves slot["reason"] untouched, which reads as the old behaviour.
DECISION_ANSWERED = "answered"
DECISION_CANCELLED = "cancelled"
DECISION_EXPIRED = "expired"


def decision_reason(slot) -> Optional[str]:
    """Why ``wait_tool_decision`` returned for ``slot``, or None if it did not say.

    None on purpose for a slot a stubbed waiter never touched: the caller then falls back to
    TOOL_REJECTED_MESSAGE, which is exactly what it did before reasons existed.
    """
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
        # Filled in by wait_tool_decision; read back with decision_reason().
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
    """Block on a slot from ``begin_tool_decision`` until the user decides. Returns ``"allow"`` or ``"deny"``, falling back to ``"deny"`` if the wait times out or generation is cancelled first. Records WHY in ``slot["reason"]`` (see ``decision_reason``), because a bare ``"deny"`` cannot tell a user who refused from an approval nobody answered, and the loops report one of those to the user as their own decision. Always removes its own slot on exit.

    The park ceiling measures time with NOBODY WATCHING, not time since the call parked. A durable run outlives its tab by design, so its cancel_event says nothing about whether a human is there; ``run_subscribers`` is what knows. While a follower is attached the deadline keeps re-arming and the wait is bounded by ``_DECISION_TIMEOUT`` exactly as a browser-owned run always was, so a user still reading what the tool wants to do does not lose the decision out from under them. Reaching that bound takes renewing the RUN's lease as well (``cancel_event.renew_lease``), since parking makes no progress and the sweeper would otherwise settle the run at its lease timeout, around 20 minutes, and cancel the wait. Once the followers go the ceiling runs (default 300s, ``UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S``) and an unattended agent adapts and continues. An explicit Stop still denies immediately.
    """
    park = bool(getattr(cancel_event, "durable", False))
    run_id = getattr(cancel_event, "durable_run_id", "") or ""
    # Run ids are account-local (per-account studio.db, client-chosen id), so attendance has to be
    # asked for under the same account or one tenant's follower answers for another's run.
    account_id = getattr(cancel_event, "durable_account_id", "") or ""
    renew_lease = getattr(cancel_event, "renew_lease", None)

    def _settle(verdict, reason):
        if isinstance(slot, dict):
            slot["reason"] = reason
        return verdict

    try:
        # `waited` is time with nobody watching and resets when a follower is seen; `total` is the
        # whole wait and never resets. A park is bounded by both.
        waited = 0.0
        total = 0.0
        last_renew: Optional[float] = None
        while not slot["event"].wait(timeout = 0.5):
            if cancel_event is not None and cancel_event.is_set():
                return _settle("deny", DECISION_CANCELLED)
            waited += 0.5
            total += 0.5
            if not park:
                # Unchanged: the caller's ceiling is the only one a browser-owned run has ever had.
                if total >= timeout:
                    return _settle("deny", DECISION_EXPIRED)
                continue
            # A durable park ignores the caller's timeout by design, so the attended backstop is this
            # module's own ceiling, the same hour a browser-owned run gets.
            if total >= _DECISION_TIMEOUT:
                return _settle("deny", DECISION_EXPIRED)
            if run_subscribers.is_attended(run_id, account_id):
                # Someone is watching, so this is deliberation, not abandonment.
                waited = 0.0
                # The RUN's lease too, not just this counter: parking makes no progress, so the
                # sweeper would otherwise settle the run at its lease timeout (1200s) and cancel the
                # wait, capping an attended deliberation near 20 minutes rather than the ceiling.
                if renew_lease is not None and (
                    last_renew is None or total - last_renew >= _LEASE_RENEW_EVERY_S
                ):
                    last_renew = total
                    try:
                        renew_lease()
                    except Exception:
                        # A lease we could not renew is the sweeper's problem, not a reason to drop
                        # the decision this wait exists to collect.
                        pass
            elif waited >= _PARK_TIMEOUT_S:
                return _settle("deny", DECISION_EXPIRED)
        return _settle(slot["decision"] or "deny", DECISION_ANSWERED)
    finally:
        with _lock:
            if _pending.get(approval_id) is slot:
                _pending.pop(approval_id, None)


def tool_decision_is_pending(approval_id, session_id = None) -> bool:
    """True while `approval_id` is still waiting on a human.

    A reopened tab cannot otherwise tell a parked call from one the user already answered: both are
    saved as a card with no result and an approval id, because the result only lands with tool_end.
    Re-arming the answered one puts Approve/Deny over a call that is already executing, and every
    press 404s. So the state is asked for rather than inferred.

    Scoped like resolve_tool_decision, and a set event counts as decided rather than pending, so the
    window between the decision and the waiter popping its own slot does not read as still-parked.
    """
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
