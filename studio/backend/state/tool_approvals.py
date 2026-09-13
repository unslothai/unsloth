# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call tool-call confirmation gate.

When a chat request sets ``confirm_tool_calls``, the agentic loop pauses before executing each tool and waits here for the user's decision, which arrives via ``POST /api/inference/tool-confirm`` on a separate connection.

Each gated call is identified by a unique ``approval_id`` (minted with ``new_approval_id``) that the loop registers here and echoes in the ``tool_start`` stream event. The frontend sends that exact id back, so a stale or duplicate confirmation, or a second tool awaiting a decision in the same session, can never resolve the wrong call; ``session_id`` is kept alongside purely as a scope check.

The slot is registered with ``begin_tool_decision`` *before* the loop yields ``tool_start``, closing the race where a fast confirmation (or an auto "Always allow") could arrive before the waiter exists. ``wait_tool_decision`` then blocks and cleans up its own slot.
"""

import secrets
import threading
from dataclasses import dataclass
from typing import Optional

# Generous ceiling so a user can deliberate; the stop button or a disconnect still breaks the wait early via cancel_event.
_DECISION_TIMEOUT = 3600.0

# Fed to the model as the tool result when the user denies a call, so it can adapt instead of the turn ending abruptly.
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

_lock = threading.Lock()
# approval_id -> {"event": threading.Event, "decision": str|None, "session": str}
_pending: dict[str, dict] = {}


@dataclass(frozen = True)
class McpImageDisclosureBinding:
    """Public, immutable facts covered by one image disclosure decision.

    Private bytes and encoded payloads must never be stored in approval state.
    The transport compares this complete value again when it consumes the grant.
    """

    subject: str
    session_id: str
    thread_id: str
    generation_id: str
    call_id: str
    attachment_ref: str
    message_id: str
    attachment_id: str
    attachment_sha256: str
    mime_type: str
    size_bytes: int
    server_id: str
    config_revision: int
    tool_name: str
    field: str
    encoding: str
    schema_digest: str
    public_arguments_digest: str
    recipient: str


# Image disclosure decisions deliberately use a separate namespace and state
# machine. Ordinary tool confirmations, including remembered/automatic allows,
# cannot resolve or consume these records.
_mcp_image_pending: dict[str, dict] = {}


def new_approval_id() -> str:
    """Mint an unguessable id for one pending tool-call confirmation."""
    return secrets.token_urlsafe(16)


def begin_tool_decision(session_id, approval_id) -> dict:
    """Register a pending decision slot and return it. Call this *before* yielding the ``tool_start`` event so the waiter always exists by the time the user's confirmation can arrive."""
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
    """Block on a slot from ``begin_tool_decision`` until the user decides. Returns ``"allow"`` or ``"deny"``, falling back to ``"deny"`` if the wait times out or generation is cancelled first. Always removes its own slot on exit."""
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


def begin_mcp_image_disclosure(
    binding: McpImageDisclosureBinding, approval_id: Optional[str] = None
) -> tuple[str, dict]:
    """Register a purpose-specific, one-use image disclosure decision."""
    approval_id = approval_id or new_approval_id()
    slot = {
        "event": threading.Event(),
        "decision": None,
        "state": "pending",
        "binding": binding,
    }
    with _lock:
        if approval_id in _pending or approval_id in _mcp_image_pending:
            raise ValueError("approval id is already in use")
        _mcp_image_pending[approval_id] = slot
    return approval_id, slot


def resolve_mcp_image_disclosure(
    approval_id: Optional[str],
    decision: str,
    *,
    current_subject: str,
    session_id: Optional[str] = None,
) -> bool:
    """Resolve only a matching authenticated image disclosure prompt."""
    if not approval_id or decision not in {"allow", "deny"}:
        return False
    with _lock:
        slot = _mcp_image_pending.get(approval_id)
        if not slot or slot["state"] != "pending":
            return False
        binding = slot["binding"]
        if binding.subject != current_subject:
            return False
        if session_id is not None and binding.session_id != (session_id or ""):
            return False
        slot["decision"] = decision
        slot["state"] = "allowed" if decision == "allow" else "denied"
        slot["event"].set()
        if decision == "deny":
            _mcp_image_pending.pop(approval_id, None)
    return True


def wait_mcp_image_disclosure(
    slot: dict,
    approval_id: str,
    cancel_event = None,
    timeout: float = _DECISION_TIMEOUT,
) -> str:
    """Wait for an explicit decision while retaining an allowed grant for commit."""
    waited = 0.0
    while not slot["event"].wait(timeout = 0.5):
        if cancel_event is not None and cancel_event.is_set():
            abort_mcp_image_disclosure(slot, approval_id)
            return "deny"
        waited += 0.5
        if waited >= timeout:
            abort_mcp_image_disclosure(slot, approval_id)
            return "deny"
    with _lock:
        current = _mcp_image_pending.get(approval_id)
        if current is not slot or slot["state"] != "allowed":
            return "deny"
        return "allow"


def consume_mcp_image_disclosure(
    approval_id: str, expected_binding: McpImageDisclosureBinding, recipient: str
) -> bool:
    """Atomically spend an allowed grant at the transport dispatch boundary."""
    with _lock:
        slot = _mcp_image_pending.get(approval_id)
        if not slot or slot["state"] != "allowed":
            return False
        binding = slot["binding"]
        if binding != expected_binding or binding.recipient != recipient:
            return False
        slot["state"] = "committed"
        _mcp_image_pending.pop(approval_id, None)
        return True


def abort_mcp_image_disclosure(slot: dict, approval_id: str) -> None:
    """Revoke a pending or allowed-but-uncommitted disclosure."""
    with _lock:
        if _mcp_image_pending.get(approval_id) is slot:
            slot["state"] = "revoked"
            slot["decision"] = "deny"
            slot["event"].set()
            _mcp_image_pending.pop(approval_id, None)


def revoke_mcp_image_disclosures(
    *,
    subject: Optional[str] = None,
    server_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    attachment_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """Revoke every uncommitted disclosure matching all supplied selectors."""
    selectors = {
        "subject": subject,
        "server_id": server_id,
        "thread_id": thread_id,
        "message_id": message_id,
        "attachment_id": attachment_id,
        "generation_id": generation_id,
        "session_id": session_id,
    }
    revoked = 0
    with _lock:
        for approval_id, slot in list(_mcp_image_pending.items()):
            binding = slot["binding"]
            if any(
                value is not None and getattr(binding, name) != value
                for name, value in selectors.items()
            ):
                continue
            slot["state"] = "revoked"
            slot["decision"] = "deny"
            slot["event"].set()
            _mcp_image_pending.pop(approval_id, None)
            revoked += 1
    return revoked
