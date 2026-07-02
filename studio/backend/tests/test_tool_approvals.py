# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrency tests for the per-call tool-call confirmation gate.

``state.tool_approvals`` coordinates two threads: the agentic loop thread
blocked in ``wait_tool_decision`` and the request thread that delivers the
user's choice through ``resolve_tool_decision``. Each gated call carries a
unique ``approval_id`` so a stale or concurrent confirmation can never
resolve the wrong call. These tests exercise that handshake directly --
no model, no server -- so the race windows are fast and deterministic.
"""

import threading
import time

import pytest

from state import tool_approvals
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    request_tool_decision,
    resolve_tool_decision,
    wait_tool_decision,
)


@pytest.fixture(autouse = True)
def _clear_pending():
    """Each test starts and ends with an empty ``_pending`` map."""
    with tool_approvals._lock:
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()


class _Waiter:
    """Run ``request_tool_decision`` in a thread and capture its result."""

    def __init__(
        self,
        session_id,
        approval_id,
        cancel_event = None,
        timeout = None,
    ):
        self.session_id = session_id
        self.approval_id = approval_id
        self.cancel_event = cancel_event
        self.timeout = timeout
        self.result = None
        self._thread = threading.Thread(target = self._run, daemon = True)

    def _run(self):
        kwargs = {"cancel_event": self.cancel_event}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self.result = request_tool_decision(self.session_id, self.approval_id, **kwargs)

    def start(self):
        self._thread.start()
        _wait_until(lambda: _has_pending(self.approval_id))
        return self

    def join(self, timeout = 5.0):
        self._thread.join(timeout = timeout)
        assert not self._thread.is_alive(), "waiter thread did not finish"
        return self.result


def _has_pending(approval_id) -> bool:
    with tool_approvals._lock:
        return approval_id in tool_approvals._pending


def _wait_until(
    pred,
    timeout = 2.0,
    interval = 0.005,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


# ── Basic allow / deny ───────────────────────────────────────────────


def test_allow_decision():
    aid = new_approval_id()
    w = _Waiter("sess", aid).start()
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    assert w.join() == "allow"


def test_deny_decision():
    aid = new_approval_id()
    w = _Waiter("sess", aid).start()
    assert resolve_tool_decision(aid, "deny", session_id = "sess") is True
    assert w.join() == "deny"


def test_slot_cleaned_up_after_decision():
    aid = new_approval_id()
    w = _Waiter("sess", aid).start()
    resolve_tool_decision(aid, "allow")
    w.join()
    assert _wait_until(lambda: not _has_pending(aid))


def test_abort_tool_decision_removes_unwaited_slot():
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)
    abort_tool_decision(slot, aid)
    assert not _has_pending(aid)
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is False


def test_approval_ids_are_unique():
    ids = {new_approval_id() for _ in range(1000)}
    assert len(ids) == 1000


# ── Pre-registration race (begin before wait) ────────────────────────


def test_resolve_before_wait_is_not_lost():
    """A decision delivered after ``begin`` but before ``wait`` survives.

    The loop registers the slot before it yields ``tool_start``, so even a
    confirmation that races ahead of the blocking ``wait`` is recorded on
    the slot and returned -- never dropped.
    """
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    # wait() is only entered now, after the decision already landed.
    assert wait_tool_decision(slot, aid) == "allow"
    assert not _has_pending(aid)


# ── Resolver edge cases ──────────────────────────────────────────────


def test_resolve_unknown_approval_returns_false():
    assert resolve_tool_decision(new_approval_id(), "allow") is False


def test_resolve_empty_approval_returns_false():
    assert resolve_tool_decision("", "allow") is False
    assert resolve_tool_decision(None, "allow") is False


def test_resolve_wrong_session_scope_returns_false():
    aid = new_approval_id()
    w = _Waiter("sess-a", aid).start()
    # Correct approval_id but the wrong session must not resolve it.
    assert resolve_tool_decision(aid, "allow", session_id = "sess-b") is False
    assert _has_pending(aid)
    # The right session still works.
    assert resolve_tool_decision(aid, "allow", session_id = "sess-a") is True
    assert w.join() == "allow"


def test_duplicate_resolve_after_completion_returns_false():
    aid = new_approval_id()
    w = _Waiter("sess", aid).start()
    assert resolve_tool_decision(aid, "allow") is True
    w.join()
    assert _wait_until(lambda: not _has_pending(aid))
    assert resolve_tool_decision(aid, "deny") is False


def test_first_decision_is_immutable():
    """A second confirmation cannot flip an already-recorded decision.

    The waiter reads ``slot["decision"]`` outside the lock and then cleans up,
    so a duplicate or out-of-order POST that lands in that window must be
    rejected and must not overwrite the first decision -- an Allow can never
    become a Deny. Distinct from the after-completion case above: here the slot
    is still pending (no waiter has consumed it yet).
    """
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    # Second decision, same id, before any waiter consumes/cleans the slot.
    assert resolve_tool_decision(aid, "deny", session_id = "sess") is False
    assert slot["decision"] == "allow"
    # The waiter still observes the first (immutable) decision.
    assert wait_tool_decision(slot, aid) == "allow"
    assert not _has_pending(aid)


# ── Cancellation and timeout ─────────────────────────────────────────


def test_cancel_event_breaks_wait_as_deny():
    cancel = threading.Event()
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    cancel.set()
    assert w.join(timeout = 3.0) == "deny"
    assert _wait_until(lambda: not _has_pending(aid))


def test_timeout_returns_deny():
    aid = new_approval_id()
    start = time.monotonic()
    result = request_tool_decision("sess", aid, timeout = 0.1)
    assert result == "deny"
    assert time.monotonic() - start < 2.0
    assert not _has_pending(aid)


# ── Independence across concurrent calls ─────────────────────────────


def test_two_pending_calls_same_session_are_independent():
    """Keying on approval_id, not session, keeps concurrent calls distinct.

    Resolving the first call's id must not unblock or alter the second
    call pending in the same session.
    """
    a1, a2 = new_approval_id(), new_approval_id()
    w1 = _Waiter("sess", a1).start()
    w2 = _Waiter("sess", a2).start()

    assert resolve_tool_decision(a1, "deny", session_id = "sess") is True
    assert w1.join() == "deny"
    # w2 is still waiting on its own id.
    assert _has_pending(a2)
    assert resolve_tool_decision(a2, "allow", session_id = "sess") is True
    assert w2.join() == "allow"


def test_concurrent_distinct_calls_route_their_own_decisions():
    n = 25
    waiters = {}
    for i in range(n):
        aid = new_approval_id()
        waiters[aid] = _Waiter(f"s{i}", aid).start()
    expected = {aid: ("allow" if i % 2 == 0 else "deny") for i, aid in enumerate(waiters)}
    for aid, decision in expected.items():
        assert resolve_tool_decision(aid, decision) is True
    for aid, w in waiters.items():
        assert w.join() == expected[aid]


# ── Constants ────────────────────────────────────────────────────────


def test_rejected_message_is_user_facing_text():
    assert isinstance(TOOL_REJECTED_MESSAGE, str)
    assert TOOL_REJECTED_MESSAGE.strip()
