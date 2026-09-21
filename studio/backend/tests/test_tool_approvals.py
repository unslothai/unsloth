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

from state import run_subscribers, tool_approvals
from state.tool_approvals import (
    TOOL_APPROVAL_EXPIRED_MESSAGE,
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


# ── Durable runs park the gate instead of auto-denying on timeout ───────
# A durable run's cancel_event is never set on a browser disconnect, so a pending
# approval must keep waiting for the returning session (resolved by id) rather than
# silently denying at the 3600s ceiling. An explicit Stop still denies: setting the
# event wins over parking.


def test_durable_approval_parks_past_timeout():
    """A durable gate waits past its timeout for the session to return, not deny."""
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel, timeout = 0.2).start()
    # The short ceiling has long passed; the gate must still be parked, not auto-denied.
    time.sleep(0.6)
    assert _has_pending(aid), "durable gate must park past its timeout, not auto-deny"
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    assert w.join(timeout = 3.0) == "allow"


def test_durable_cancel_still_denies():
    """An explicit Stop (cancel_event set) denies even when the run is durable."""
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    cancel.set()
    assert w.join(timeout = 3.0) == "deny"
    assert _wait_until(lambda: not _has_pending(aid))


def test_an_unanswered_park_is_released_by_the_settles_cancel_not_a_ceiling():
    """The sweeper's settle-cancel releases a park that the park timeout has not yet reached.

    At 0.6s the default park timeout (300s) has not elapsed, so the only thing that can release
    the gate is an external cancel — exactly what ``supervisor.cancel()`` does after
    ``reconcile_runs`` settles a lease-expired run. The parked gate must deny and pop its own
    slot, so the reservation unwinds.
    """
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel, timeout = 0.2).start()
    # Well within the park timeout (300s default); only an external cancel can release now.
    time.sleep(0.6)
    assert _has_pending(aid), "the park has not timed out yet; only cancel releases it"
    cancel.set()  # what reconcile_runs' settle does to a lease-expired run
    assert w.join(timeout = 3.0) == "deny", "the settle's cancel must read as deny"
    assert _wait_until(lambda: not _has_pending(aid)), "the slot must be popped on release"


def test_durable_park_denies_at_park_timeout(monkeypatch):
    """A durable park denies at the park timeout so an unattended agent adapts and continues.

    This is the release path for agentic work where the user has left: the approval times out,
    the model receives TOOL_REJECTED_MESSAGE, and the loop proceeds to the next step. The sweeper
    remains a backstop for producers wedged before they reach wait_tool_decision.
    """
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    # Park timeout (0.2s) has elapsed; the gate must deny on its own, no cancel needed.
    time.sleep(0.6)
    assert w.join(timeout = 3.0) == "deny", "the park timeout must release an unanswered approval"
    assert _wait_until(lambda: not _has_pending(aid)), "the slot must be popped on timeout"


# ── The park ceiling is configuration, and configuration must not be able to kill the backend ──


@pytest.mark.parametrize(
    "value,expected",
    [
        ("30", 30.0),
        ("0", 0.0),
        ("0.5", 0.5),
        ("  45  ", 45.0),
        # Anything unparseable, out of range, or nonsensical falls back rather than raising: this
        # module is imported on the chat path, so a typo here would otherwise take the backend down
        # at startup, and a stuck approval is the lesser failure.
        ("5m", 300.0),
        ("", 300.0),
        ("   ", 300.0),
        ("none", 300.0),
        ("-1", 300.0),
        ("nan", 300.0),
        ("inf", 300.0),
    ],
)
def test_the_park_ceiling_reads_the_env_without_ever_raising(monkeypatch, value, expected):
    monkeypatch.setenv("UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S", value)
    assert tool_approvals._park_timeout_from_env() == expected


def test_an_unset_park_ceiling_is_the_documented_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S", raising = False)
    assert tool_approvals._park_timeout_from_env() == 300.0
    assert tool_approvals._PARK_TIMEOUT_DEFAULT_S == 300.0


def test_a_zero_ceiling_denies_at_the_first_poll_not_before_it(monkeypatch):
    """The loop waits before it checks, so 0 is "autonomous", not "instant": a decision landing
    inside the first 500ms still wins. The comment on _PARK_TIMEOUT_DEFAULT_S says so; this pins it,
    because a reader who takes "denies immediately" literally would call the opposite a bug."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.0)
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    time.sleep(0.05)
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    assert w.join(timeout = 3.0) == "allow"


def test_a_returning_session_past_the_ceiling_cannot_resolve_its_own_approval(monkeypatch):
    """What the user actually experiences at the ceiling: the call is already refused, and the
    Approve they press on return has nothing left to resolve (the route turns this into a 404)."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    assert w.join(timeout = 3.0) == "deny"
    assert resolve_tool_decision(aid, "allow", session_id = "sess") is False


def test_a_pending_approval_reports_pending_and_a_decided_one_does_not():
    """A reopened tab cannot tell a parked call from one the user already answered: both are saved
    as a card with no result and an approval id, because the result only lands with tool_end. This
    is the durable signal that separates them."""
    approval_id = tool_approvals.new_approval_id()
    slot = tool_approvals.begin_tool_decision("sess-a", approval_id)
    assert tool_approvals.tool_decision_is_pending(approval_id, "sess-a") is True

    assert tool_approvals.resolve_tool_decision(approval_id, "allow", session_id = "sess-a") is True
    # Decided but not yet collected by the waiter: the window that would otherwise read as parked.
    assert tool_approvals.tool_decision_is_pending(approval_id, "sess-a") is False

    assert tool_approvals.wait_tool_decision(slot, approval_id, timeout = 30) == "allow"
    # And gone once the waiter has popped its own slot.
    assert tool_approvals.tool_decision_is_pending(approval_id, "sess-a") is False


def test_the_pending_check_is_session_scoped_and_unguessable():
    approval_id = tool_approvals.new_approval_id()
    tool_approvals.begin_tool_decision("sess-a", approval_id)
    assert tool_approvals.tool_decision_is_pending(approval_id, "sess-b") is False
    assert tool_approvals.tool_decision_is_pending(approval_id, "sess-a") is True
    # No id, no answer: this must never be a way to ask "is anything pending".
    assert tool_approvals.tool_decision_is_pending("", "sess-a") is False
    assert tool_approvals.tool_decision_is_pending(None, "sess-a") is False
    assert tool_approvals.tool_decision_is_pending("not-a-real-id", "sess-a") is False


# ── The park ceiling counts time with NOBODY WATCHING, not time since the call parked ──
# "Durable" means cancel_on_disconnect is off, not that the tab is gone. Since every
# tool-enabled turn is durable by default, keying the 300s ceiling off the durable marker
# alone put a five-minute deadline on a user who is sitting there reading what the tool
# wants to do, where a browser-owned run gave them the full hour. The gate asks
# state.run_subscribers whether a follower is attached, and re-arms while one is.


@pytest.fixture(autouse = True)
def _clear_subscribers():
    run_subscribers.reset_for_tests()
    yield
    run_subscribers.reset_for_tests()


def test_an_attended_park_does_not_expire_at_the_park_ceiling(monkeypatch):
    """A user watching the run keeps their decision past the park ceiling.

    The regression this pins: with the ceiling keyed on the durable marker alone, an Approve/Deny
    card the user was reading auto-denied at 300s and the model was told they had declined it.
    """
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    cancel = threading.Event()
    cancel.durable = True
    cancel.durable_run_id = "run-attended"
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()

    # A follower heartbeat, exactly as the SSE loop in routes/chat_generation_runs.py stamps it.
    for _ in range(6):
        run_subscribers.mark_subscriber_seen("run-attended")
        time.sleep(0.1)
    assert _has_pending(aid), "an attended park must not expire at the park ceiling"

    assert resolve_tool_decision(aid, "allow", session_id = "sess") is True
    assert w.join(timeout = 3.0) == "allow"


def test_a_park_expires_once_its_followers_go(monkeypatch):
    """The ceiling still releases an abandoned approval: presence is what re-arms it, and it ages out."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(run_subscribers, "_ATTENDED_FOR_S", 0.1)
    cancel = threading.Event()
    cancel.durable = True
    cancel.durable_run_id = "run-leaving"
    aid = new_approval_id()
    run_subscribers.mark_subscriber_seen("run-leaving")
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    # Nobody stamps again: the entry ages out and the ceiling runs.
    assert w.join(timeout = 5.0) == "deny"
    assert _wait_until(lambda: not _has_pending(aid))


def test_an_unattended_park_is_unchanged_without_a_run_id(monkeypatch):
    """A durable event carrying no run id keeps the plain park behaviour, so nothing regresses
    for a producer that never registered one."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    w = _Waiter("sess", aid, cancel_event = cancel).start()
    assert w.join(timeout = 3.0) == "deny"


def test_attendance_cannot_hold_an_approval_past_the_absolute_ceiling(monkeypatch):
    """Presence re-arms the park ceiling, never the hour a browser-owned run has always had."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 10.0)
    monkeypatch.setattr(tool_approvals, "_DECISION_TIMEOUT", 0.2)
    cancel = threading.Event()
    cancel.durable = True
    cancel.durable_run_id = "run-forever"
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)

    stop = threading.Event()

    def _stamp():
        while not stop.is_set():
            run_subscribers.mark_subscriber_seen("run-forever")
            time.sleep(0.02)

    stamper = threading.Thread(target = _stamp, daemon = True)
    stamper.start()
    try:
        verdict, reason = tool_approvals.wait_tool_decision_detail(slot, aid, cancel_event = cancel)
    finally:
        stop.set()
        stamper.join(timeout = 2.0)
    assert (verdict, reason) == ("deny", tool_approvals.DECISION_EXPIRED)


# ── An expiry is not the user's decision, and must not be reported as one ──


def test_the_reason_separates_an_expiry_from_a_deny_from_a_cancel(monkeypatch):
    """The three outcomes used to arrive as one bare "deny", so the loops had no way to tell a
    returning user that nobody answered rather than that they refused."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)

    # Expired: nobody watching, nobody answering.
    cancel = threading.Event()
    cancel.durable = True
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)
    assert tool_approvals.wait_tool_decision_detail(slot, aid, cancel_event = cancel) == (
        "deny",
        tool_approvals.DECISION_EXPIRED,
    )

    # Cancelled: an explicit Stop, or the sweeper settling a lease-expired run.
    cancel2 = threading.Event()
    cancel2.durable = True
    aid2 = new_approval_id()
    slot2 = begin_tool_decision("sess", aid2)
    cancel2.set()
    assert tool_approvals.wait_tool_decision_detail(slot2, aid2, cancel_event = cancel2) == (
        "deny",
        tool_approvals.DECISION_CANCELLED,
    )

    # Answered: the user actually pressed Deny.
    aid3 = new_approval_id()
    slot3 = begin_tool_decision("sess", aid3)
    resolve_tool_decision(aid3, "deny", session_id = "sess")
    assert tool_approvals.wait_tool_decision_detail(slot3, aid3) == (
        "deny",
        tool_approvals.DECISION_ANSWERED,
    )


def test_the_expiry_message_does_not_claim_the_user_decided():
    assert TOOL_APPROVAL_EXPIRED_MESSAGE != TOOL_REJECTED_MESSAGE
    assert TOOL_APPROVAL_EXPIRED_MESSAGE.strip()
    lowered = TOOL_APPROVAL_EXPIRED_MESSAGE.lower()
    assert "declin" not in lowered, "an expiry must not be reported as the user declining"
    assert "the user" not in lowered, "an expiry is not a statement about the user"


def test_the_legacy_wrapper_still_returns_a_bare_verdict(monkeypatch):
    """`wait_tool_decision` keeps its old signature and return, so callers that do not care about
    the reason (request_tool_decision, and anything outside the three tool loops) are untouched."""
    monkeypatch.setattr(tool_approvals, "_PARK_TIMEOUT_S", 0.2)
    aid = new_approval_id()
    slot = begin_tool_decision("sess", aid)
    resolve_tool_decision(aid, "allow", session_id = "sess")
    assert wait_tool_decision(slot, aid) == "allow"


# ── The presence registry itself ──


def test_presence_expires_by_age_so_a_leaked_stamp_cannot_park_forever(monkeypatch):
    monkeypatch.setattr(run_subscribers, "_ATTENDED_FOR_S", 0.1)
    run_subscribers.mark_subscriber_seen("run-x")
    assert run_subscribers.is_attended("run-x") is True
    time.sleep(0.2)
    assert run_subscribers.is_attended("run-x") is False


def test_presence_is_per_run_and_an_empty_id_is_never_attended():
    run_subscribers.mark_subscriber_seen("run-a")
    assert run_subscribers.is_attended("run-a") is True
    assert run_subscribers.is_attended("run-b") is False
    run_subscribers.mark_subscriber_seen("")
    assert run_subscribers.is_attended("") is False


def test_a_departing_follower_drops_its_stamp_promptly():
    run_subscribers.mark_subscriber_seen("run-c")
    assert run_subscribers.is_attended("run-c") is True
    run_subscribers.subscriber_departed("run-c")
    assert run_subscribers.is_attended("run-c") is False
