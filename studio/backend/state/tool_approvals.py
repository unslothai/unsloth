# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call approval gate keyed by approval_id; begin_tool_decision registers before tool_start"""

import os
import secrets
import threading
from typing import Optional

from state import run_subscribers

# Generous ceiling so a user can deliberate; the stop button or a disconnect still breaks the wait early via cancel_event.
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

# Far under the lease timeout (1200s), far over the 0.5s poll: one small write every half minute.
_LEASE_RENEW_EVERY_S = 30.0

# Fed to the model as the tool result when the user denies a call, so it can adapt instead of the turn ending abruptly.
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


class LiveToolPermissions:
    """One live tool loop's upgrade, never a process/session-wide permission grant."""

    def __init__(
        self,
        session_id,
        cancel_event = None,
    ):
        from utils.account_context import current_account

        self.session_id = session_id or ""
        self.account = current_account()
        self.account_id = self.account.account_id
        self.cancel_event = cancel_event
        self._full = threading.Event()

    def apply(self, permission_mode, bypass_permissions):
        if self._full.is_set():
            return "full", True
        return permission_mode, bypass_permissions


def bind_permission_upgrade(
    slot,
    permissions,
    *,
    eligible = True,
):
    """Bind before yielding the approval card; image disclosure alone never grants full access."""
    if not isinstance(slot, dict) or not eligible or permissions is None:
        return
    with _lock:
        if slot.get("session") == permissions.session_id and permissions.session_id:
            slot["permission_upgrade"] = permissions


def new_approval_id() -> str:
    """Mint an unguessable id for one pending tool-call confirmation."""
    return secrets.token_urlsafe(16)


def begin_tool_decision(session_id, approval_id) -> dict:
    """Register a pending decision slot and return it. Call this *before* yielding the ``tool_start`` event so the waiter always exists by the time the user's confirmation can arrive."""
    from utils.account_context import current_account_id

    slot = {
        "event": threading.Event(),
        "account_id": current_account_id(),
        "decision": None,
        "session": session_id or "",
        # Filled in by wait_tool_decision; read back with decision_reason().
        "reason": None,
    }
    with _lock:
        _pending[approval_id] = slot
    return slot


def note_host_access(approval_id, tool_name, arguments) -> None:
    """Flag a registered call whose approval would reach host paths, so ``/tool-confirm`` can refuse it to an API key."""
    if not approval_id:
        return
    from core.inference.tools import approval_grants_host_access

    try:
        grants = approval_grants_host_access(tool_name, arguments)
    except Exception:  # noqa: BLE001 - unclassifiable counts as host access here
        grants = True
    with _lock:
        slot = _pending.get(approval_id)
        if slot is not None:
            slot["host_access"] = bool(grants)


def tool_decision_grants_host_access(approval_id, session_id = None) -> bool:
    """True when allowing this pending call would let it reach host paths (``note_host_access``)."""
    if not approval_id:
        return False
    with _lock:
        slot = _pending.get(approval_id)
        if not slot:
            return False
        if session_id is not None and slot["session"] != (session_id or ""):
            return False
        return bool(slot.get("host_access"))


def wait_tool_decision(
    slot,
    approval_id,
    cancel_event = None,
    timeout = _DECISION_TIMEOUT,
):
    """Blocks until the user decides, recording why in slot reason; unattended parks expire at UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S; Stop denies at once."""
    park = bool(getattr(cancel_event, "durable", False))
    run_id = getattr(cancel_event, "durable_run_id", "") or ""
    # Run ids are account-local, so ask attendance under the same account, or one tenant follower answers for another tenant run.
    account_id = getattr(cancel_event, "durable_account_id", "") or ""
    renew_lease = getattr(cancel_event, "renew_lease", None)

    def _settle(verdict, reason):
        if not isinstance(slot, dict):
            return verdict
        # Under the resolver's lock: an Allow committed first wins, else the resolver sees the reason and refuses.
        with _lock:
            if slot["event"].is_set():
                verdict, reason = slot["decision"] or "deny", DECISION_ANSWERED
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
                # Unchanged: the caller's ceiling is the only one a browser-owned run has ever had.
                if total >= timeout:
                    return _settle("deny", DECISION_EXPIRED)
                continue
            # Durable parks ignore the caller's timeout; this is the attended backstop.
            if total >= _DECISION_TIMEOUT:
                return _settle("deny", DECISION_EXPIRED)
            if run_subscribers.is_attended(run_id, account_id):
                # Someone is watching, so this is deliberation, not abandonment.
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
    """True while approval_id still awaits a human; asked of the state, since a reopened tab cannot tell parked calls from answered ones."""
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
    *,
    full_access = False,
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
        from utils.account_context import current_account_id

        if (
            slot.get("account_id") != current_account_id()
            or slot["event"].is_set()
            or slot.get("reason") is not None
        ):
            return False
        if full_access:
            permissions = slot.get("permission_upgrade")
            if (
                decision != "allow"
                or not session_id
                or permissions is None
                or permissions.account_id != current_account_id()
                or permissions.session_id != session_id
                or (permissions.cancel_event is not None and permissions.cancel_event.is_set())
            ):
                return False
            from state.tool_policy import require_tool_access
            from utils.account_context import run_as

            require_tool_access("full")
            # The tool worker keeps the run's API/UI identity: never acknowledge a grant it would refuse.
            run_as(permissions.account, require_tool_access, "full")
            permissions._full.set()
        slot["decision"] = decision
        slot["event"].set()
    return True
