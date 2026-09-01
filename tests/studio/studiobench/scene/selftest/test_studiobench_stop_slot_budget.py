# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`stop_generation` must not spend the NEXT slot's window cleaning up after itself.

Every slot has an absolute start, so an action that overruns pushes no start. It does spend real
time: `SceneRunner._run_slot` is sequential, nothing enters the next slot until the previous action
has returned, and the next slot's remaining budget is `deadline - now`. An overrun therefore comes
straight out of the following action's window, and a big enough one records `slot_missed` there,
whose reason reads "this machine reached it at ...ms" -- the machine blamed for time the previous
action took.

WHERE THAT BECAME REACHABLE. `stop_generation` waits for the cell's own reply to drain rather than
stopping it, and that wait is bounded by the slot's remaining budget. Bounded by ALL of it, the
reply could drain at the last moment of the slot and the action would then start, stop and delete a
throwaway turn on time it no longer had. `stop_generation` closes 300 ms before `scroll_after` opens
on the fast film and 500 ms before it on the quick one, so on both of those the overrun lands inside
`scroll_after`'s own window, and on the fast film -- 1,200 ms of budget behind a 300 ms gap --
it can take the whole of it.

Reachable on supported input, and only on supported input: the drain wait exists for
`--stream-tail-chars`, whose entire purpose is a reply long enough to still be streaming when this
slot opens. A tail that puts the drain near the end of the slot is exactly the case it was written
for.

WHAT IS FAKED AND WHAT IS NOT. The action under test is the shipped `stop_generation`. The page is
a shim of the calls it makes, and the clock is a counter that the shim advances, so the test
measures the time the action ASKS the page for rather than how long this machine took to answer.

THE SHIM'S POLLS ARE NOT FREE, and that correction is why this file was rewritten. The first
version answered `isRunning()` instantly and charged nothing for a page call, so the only cost it
could see was the four fixed sleeps -- `[100 x29, 80, 600, 400, 200]`, 1,280 ms after the drain --
and a reserve sized from that number was 500 ms short of what the action really spends. Driving the
SAME shipped `stop_generation` against real chromium and a real clock, on a page standing in for
the calls it makes, the stretch after the drain costs:

    page answers instantly                                1,394 - 1,451 ms   (n = 5)
    120 ms to start, 90 ms to stop, 60 ms to delete       1,723 - 1,938 ms   (n = 5)
    300 ms to start, 200 ms to stop, 150 ms to delete     2,092 - 2,102 ms   (n = 2)

against 1,280 ms reserved, which put the action 506 - 516 ms past the end of a 3,000 ms stop slot
on both films whose gap before `scroll_after` is smaller than that. The difference is thirteen to
seventeen CDP round trips at about 4 ms each, the 50 ms granularity of the two poll loops, and the
app's own latency in answering them -- none of which a shim that answers in-process can produce.
So the shim below charges `ROUND_TRIP_MS` for every page call and holds the page to `START_MS`,
`STOP_MS` and `CLEANUP_MS` before it changes its answer. That puts the same stretch at 1,688 ms
against the middle row's 1,723 - 1,938 ms, so it is still a floor -- and with the latency switched
off it returns exactly 1,280 ms, which is the old shim's number and shows the whole correction is
the modelling and not the action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.types import ActionContext  # noqa: E402
from studiobench.scene import actions as actions_module  # noqa: E402
from studiobench.scene.actions import (  # noqa: E402
    OWN_TURN_FIXED_AFTER_SEND_MS,
    OWN_TURN_FIXED_MS,
    OWN_TURN_POLL_MS,
    OWN_TURN_RESERVE_MS,
    OWN_TURN_START_POLL_MS,
    OWN_TURN_STOP_POLL_MS,
    TURN_START_TIMEOUT_MS,
    stop_generation,
)
from studiobench.scene.schedule import FAST, QUICK, STANDARD  # noqa: E402

#: One driver call over CDP. Measured at 43 - 130 ms across nine to thirteen `page.evaluate` calls
#: in one run of the shipped action against real chromium, and the two big ones in that total are
#: page-side waits charged separately below, so about 4 ms is the round trip itself.
ROUND_TRIP_MS = 4.0

#: Enter pressed to `isRunning()` answering true: a POST to the relay and the first SSE frame back.
START_MS = 120.0

#: Stop clicked to `isRunning()` answering false: the abort round trip.
STOP_MS = 90.0

#: The delete inside `STOP_CLEANUP_JS`: the click, the removal, and the two `__sbNextPaint` frames
#: the loop waits on.
CLEANUP_MS = 90.0


class _Clock:
    """The action reads `time.monotonic`; the page's waits and calls are what move it."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def monotonic(self) -> float:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1_000.0


class _Keyboard:
    def __init__(self, page) -> None:
        self.pressed: list[str] = []
        self._page = page

    def press(self, key: str) -> None:
        self._page.charge()
        self.pressed.append(key)
        if key != "Enter" or not self._page.composer.strip():
            return
        if not self._page.accepts_send:
            # `queueDisabled` in thread.tsx: the press queued nothing and the box keeps its text.
            return
        # SENT, NOT STARTED, and the difference is the whole of the P1 above. The app takes the
        # text out of the composer and puts the user turn and its reply into the thread NOW; the
        # reply begins generating `start_ms` later, and the action has to poll for it -- which is
        # the cost the first version of this shim handed out free.
        self._page.composer = ""
        self._page.messages += 2
        self._page.sent_at_ms = self._page.elapsed_ms


class _Page:
    """The cell's own reply is streaming and drains `drain_after_ms` into the slot.

    `latency = False` restores the instant-answer page the first version of this file used, which
    is kept only as the control below: it is the floor, not the machine.

    THE COMPOSER AND THE THREAD ARE MODELLED, not stubbed. `composerText()` used to answer "" and
    `messageCount()` was never asked, so a send the app REFUSED and a send it accepted looked
    identical from the driver -- which is exactly the distinction the action has to make before it
    deletes anything, and a shim that cannot make it cannot test the code that does.
    """

    def __init__(
        self,
        clock: _Clock,
        drain_after_ms: float,
        *,
        latency: bool = True,
        start_ms: float = START_MS,
        stop_ms: float = STOP_MS,
        cleanup_ms: float = CLEANUP_MS,
        accepts_send: bool = True,
    ) -> None:
        self._clock = clock
        self._entered = clock.t
        self._drain_after_ms = drain_after_ms
        self._latency = latency
        self._start_ms = start_ms
        self._stop_ms = stop_ms
        self._cleanup_ms = cleanup_ms
        self.accepts_send = accepts_send
        self.running = True
        self.filled: list[str] = []
        self.composer = ""
        # The seeded user turn and the reply it prompted.
        self.messages = 2
        self.deleted = 0
        self.clicked = 0
        self.sent_at_ms: float | None = None
        self.clicked_at_ms: float | None = None
        self.keyboard = _Keyboard(self)

    @property
    def elapsed_ms(self) -> float:
        return (self._clock.t - self._entered) * 1_000.0

    def charge(self, ms: float = ROUND_TRIP_MS) -> None:
        if self._latency:
            self._clock.advance_ms(ms)

    def settle(self, ms: float) -> None:
        """Time passing with nobody driving: what the NEXT action walks into."""
        self._clock.advance_ms(ms)

    def _is_running(self) -> bool:
        if self.clicked_at_ms is not None:
            return self.elapsed_ms < self.clicked_at_ms + (self._stop_ms if self._latency else 0.0)
        if self.sent_at_ms is not None:
            return self.elapsed_ms >= self.sent_at_ms + (self._start_ms if self._latency else 0.0)
        if self.running and self.elapsed_ms >= self._drain_after_ms:
            self.running = False
        return self.running

    def evaluate(
        self,
        script,
        arg = None,
    ):
        self.charge()
        if arg is not None:  # STOP_CLEANUP_JS, which is the only call that takes one
            self.charge(self._cleanup_ms)
            before = self.messages
            if self.messages:
                self.messages -= 1
                self.deleted += 1
            return {
                "removed": self.messages < before,
                "before": before,
                "after": self.messages,
                "reason": None,
            }
        if "isRunning" in script:
            return self._is_running()
        if "composerText" in script:
            return self.composer
        if "messageCount" in script:
            return self.messages
        # The thread's length as well as the mounted count. `stop_generation` proves its own turn
        # was added by `threadTotal()`, so that a windowed arm whose window refills is not read as
        # a send that added nothing and left with nothing to clean up. This page models a fully
        # mounted arm, where the two are the same number.
        if "threadTotal" in script:
            return self.messages
        if "assistantChars" in script:
            return 9_200
        raise AssertionError(f"the page was asked something this shim does not model: {script}")

    def fill(self, _selector: str, text: str) -> None:
        self.charge()
        self.filled.append(text)
        self.composer = text

    def wait_for_timeout(self, ms) -> None:
        self._clock.advance_ms(ms)

    def query_selector(self, selector: str):
        self.charge()
        page = self

        class _Button:
            def click(_self) -> None:
                page.charge()
                page.clicked += 1
                page.clicked_at_ms = page.elapsed_ms

        return _Button() if "Stop generating" in selector else None


def _drive(page, monkeypatch, clock, budget_ms: int):
    monkeypatch.setattr(actions_module, "time", clock)
    return stop_generation(
        ActionContext(
            page = page,
            cdp = None,
            cell = None,
            window = None,
            args = {},
            budget_ms = budget_ms,
            dom = None,
            log = lambda _m: None,
        )
    )


def _run(
    monkeypatch,
    *,
    budget_ms: int,
    drain_after_ms: float,
    latency: bool = True,
    **page_kwargs,
):
    clock = _Clock()
    page = _Page(clock, drain_after_ms, latency = latency, **page_kwargs)
    return _drive(page, monkeypatch, clock, budget_ms), page


def _stop_slot(scene):
    """The stop slot, and the gap between its deadline and the next slot's start."""
    index = next(i for i, s in enumerate(scene.slots) if s.action == "stop_generation")
    stop, nxt = scene.slots[index], scene.slots[index + 1]
    return stop, nxt, nxt.t_start_ms - (stop.t_start_ms + stop.budget_ms)


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_reply_that_drains_at_the_end_of_the_slot_does_not_spend_the_next_one(monkeypatch, scene):
    """THE REGRESSION, in the film's own numbers.

    A reply that is still streaming 100 ms before this slot closes leaves nothing for the throwaway
    turn. The action has to stop within the gap before the next slot opens -- 300 ms on the fast
    film, 500 ms on the quick one -- or the next action's window pays for it.
    """

    stop, nxt, slack_ms = _stop_slot(scene)
    late = stop.budget_ms - 100
    result, page = _run(monkeypatch, budget_ms = stop.budget_ms, drain_after_ms = late)

    assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
        f"{scene.name}: stop_generation spent {page.elapsed_ms:.0f}ms of a {stop.budget_ms}ms "
        f"slot with only {slack_ms}ms before {nxt.action} opens"
    )
    # And it stopped by declining, not by stopping the cell's own reply, which is the whole point
    # of the wait it just gave up on.
    assert result.ran is False
    assert page.clicked == 0
    assert "one more" not in page.filled


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_no_moment_the_reply_can_drain_lets_the_action_spend_the_next_slot(monkeypatch, scene):
    """THE REGRESSION THE FIXED SLEEPS ALONE DID NOT COVER, and the reason this file was rewritten.

    The case above is refused by the drain wait's own deadline. There is a case between the two
    that is not: a reply draining just INSIDE the reserve, where the action commits to the
    throwaway turn with what the reserve says is exactly enough and then spends the two polls, the
    delete and thirteen to seventeen CDP round trips that a reserve counting only the four fixed
    sleeps never priced. Measured against real chromium that is 1,723 - 1,938 ms of work paid for
    with 1,280 ms, which lands 506 - 516 ms into `scroll_after` on both 3,000 ms films and is
    recorded there as a missed slot.

    SWEPT RATHER THAN AIMED AT ONE BOUNDARY, because the boundary is what is under test: a test
    that drains at `budget - RESERVE - 50` moves with the constant and passes on any reserve at
    all. The invariant does not depend on a number -- WHENEVER the reply drains, the action returns
    inside its slot plus the gap the film leaves before the next one -- so every 50 ms of the slot
    is tried and the assertion is the same at each.

    Two things have to hold for that, and only both together are enough. The reserve has to cover
    what the turn really costs, and the clock has to be re-read before the turn is committed to:
    the drain loop tests its deadline at the top, so the iteration that finds the reply drained has
    already spent a wait and a round trip past it, and no constant pays for time already gone.
    """

    stop, nxt, slack_ms = _stop_slot(scene)
    for drained_at in range(0, stop.budget_ms, 50):
        result, page = _run(monkeypatch, budget_ms = stop.budget_ms, drain_after_ms = drained_at)
        assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
            f"{scene.name}: a reply draining {drained_at}ms into the slot left "
            f"stop_generation spending {page.elapsed_ms:.0f}ms of a {stop.budget_ms}ms slot with "
            f"only {slack_ms}ms before {nxt.action} opens"
            + (f"; it ran the throwaway turn anyway: {result.expect}" if result.ran else "")
        )


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_send_the_app_refused_is_bounded_by_the_slot_and_not_by_eight_seconds(monkeypatch, scene):
    """The other unbounded wait after the drain, on the one path where cutting it really is free.

    `TURN_START_TIMEOUT_MS` is 8 s, which is 2.7x the whole stop slot on the fast and quick films,
    so a turn that never came up used to hold the action for eight seconds and take the next two or
    three slots with it. When the app REFUSED the send -- `queueDisabled` turns Send into Queue
    while anything is running, and a Queue press leaves the text in the box -- nothing was
    committed, there is nothing to take back, and the wait may be cut to fit the slot.

    THE CONTROL for the P1 below, and it passes on the code before that fix as well as after it:
    what changed is only which of the two cases this covers."""

    stop, nxt, slack_ms = _stop_slot(scene)
    result, page = _run(
        monkeypatch,
        budget_ms = stop.budget_ms,
        drain_after_ms = 0.0,
        accepts_send = False,
    )

    assert result.ran is False
    assert "did not start" in (result.reason or "")
    assert page.composer == "one more", "the send was refused, so the text is still in the box"
    assert page.messages == 2, "nothing was sent, so nothing may be deleted either"
    assert page.deleted == 0
    assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
        f"{scene.name}: waiting for a send the app refused cost {page.elapsed_ms:.0f}ms of a "
        f"{stop.budget_ms}ms slot with only {slack_ms}ms before {nxt.action} opens"
    )


def _thread_a_measured_turn_leaves(monkeypatch, budget_ms: int) -> int:
    """What the thread looks like after a turn that DID start, was stopped and was cleaned up.

    The reference for every give-up path below, rather than a literal: `STOP_CLEANUP_JS` is what
    decides how much of the throwaway turn comes back out, and a give-up path is required to leave
    the thread where the measured path leaves it -- not somewhere a number in this file asserts.
    """
    result, page = _run(monkeypatch, budget_ms = budget_ms, drain_after_ms = 0.0, start_ms = 0.0)
    assert result.ran is True, result.reason
    return page.messages


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
@pytest.mark.parametrize("late_by_ms", [0, 400, 800])
def test_a_turn_that_starts_after_the_slot_bound_is_not_left_generating(
    monkeypatch, scene, late_by_ms
):
    """THE P1. Enter is pressed BEFORE the turn-start wait, so the slot bound on that wait does not
    return from a decision -- it returns from a turn the app has already accepted.

    Measured in real chromium on a page taking 1,800 ms to start against the fast film's 3,000 ms
    stop slot, the action returned `not_run` after 1,759 ms and handed on a thread with two extra
    messages in it and a live stream running through `scroll_after`'s window. Every later action,
    the final census and the seeded-versus-streamed comparison then measure that scaffolding -- the
    same defect `STOP_CLEANUP_JS` was written to remove, reached by giving up instead of by
    finishing. Cutting the wait short is only free when nothing was sent, which is the control
    directly above.

    The turn has to come back stopped and deleted, and the thread has to be the one a measured turn
    would have left."""

    stop, _nxt, _slack = _stop_slot(scene)
    settled = _thread_a_measured_turn_leaves(monkeypatch, stop.budget_ms)
    # Past the slot bound, which is the budget less what the rest of the turn costs, and still
    # inside the time the turn is worth waiting for so it can be taken back.
    start_ms = stop.budget_ms - 1_000 + late_by_ms

    result, page = _run(
        monkeypatch,
        budget_ms = stop.budget_ms,
        drain_after_ms = 0.0,
        start_ms = start_ms,
    )

    assert result.ran is False
    assert page.composer == "", "the send went through, so the box is empty"
    assert page.clicked == 1, "the turn it had already sent was never stopped"
    assert page.deleted == 1, "the turn it had already sent was never deleted"
    assert page.messages == settled, (
        f"{scene.name}: a turn starting {start_ms}ms in was given up on and left the thread at "
        f"{page.messages} messages, where a measured turn leaves it at {settled}"
    )
    # And it is not about to start streaming again the moment the next action opens its window.
    page.settle(2_000)
    assert page._is_running() is False


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_turn_that_never_starts_at_all_is_still_taken_out_of_the_thread(monkeypatch, scene):
    """The far end of the same path: the send was accepted and the relay never answered.

    Nothing can be stopped, because nothing ever ran, but the app put the turn in the thread on
    send and it is still there. It comes out, the row says so, and the whole thing is bounded by
    `TURN_START_TIMEOUT_MS` -- the same eight seconds this wait cost before it was bounded by the
    slot at all, so taking the turn back is not paid for with a wait that did not exist before."""

    stop, _nxt, _slack = _stop_slot(scene)
    settled = _thread_a_measured_turn_leaves(monkeypatch, stop.budget_ms)

    result, page = _run(
        monkeypatch,
        budget_ms = stop.budget_ms,
        drain_after_ms = 0.0,
        start_ms = 10 * TURN_START_TIMEOUT_MS,
    )

    assert result.ran is False
    assert "never started" in (result.reason or ""), result.reason
    assert page.clicked == 0, "there was nothing running to stop"
    assert page.messages == settled
    assert page.elapsed_ms <= TURN_START_TIMEOUT_MS + OWN_TURN_RESERVE_MS, (
        f"{scene.name}: taking back a turn that never started cost {page.elapsed_ms:.0f}ms, more "
        f"than the {TURN_START_TIMEOUT_MS}ms this wait cost before it was bounded by the slot"
    )


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
@pytest.mark.parametrize(
    ("stop_ms", "cleanup_ms"),
    [(0.0, 0.0), (90.0, 60.0), (200.0, 150.0)],
    ids = ["instant", "local", "slow"],
)
def test_no_moment_the_turn_can_start_lets_the_action_spend_the_next_slot(
    monkeypatch, scene, stop_ms, cleanup_ms
):
    """THE P2, and the third thing the reserve has to get right.

    The drain wait reserves the whole turn and the clock is re-read before the turn is committed
    to, but the TURN-START wait is bounded separately, and a turn starting on the last millisecond
    that bound allows still has the stop-settle poll, the delete and the driver calls between them
    ahead of it. Reserving only `OWN_TURN_FIXED_MS` -- which also counts the 80 ms already spent
    settling the fill -- left that stretch unpaid. Measured against real chromium with the turn
    starting at the bound and a 3,000 ms slot:

        page answers instantly                     60 ms past the fixed sleeps    slot - 20 ms
        90 ms to stop, 60 ms to delete            227 ms past the fixed sleeps    slot + 147 ms
        200 ms to stop, 150 ms to delete          424 ms past the fixed sleeps    slot + 344 ms

    The fast film leaves 300 ms before `scroll_after` opens, so the last row was recorded there as
    `slot_missed` -- the machine blamed for time this action spent, one action later than the
    defect the reserve above already fixed. The three page latencies are the same three the totals
    in `OWN_TURN_POLL_MS` were measured at.

    SWEPT RATHER THAN AIMED AT THE BOUNDARY, for the same reason as the drain sweep: a test that
    starts the turn at `bound - 50` moves with the constant and passes on any bound at all.

    THE INVARIANT IS A PAIR, because past the bound the action stops being able to have both. A
    turn it MEASURES has to fit the slot and the gap the film leaves after it. A turn it gives up
    on does not fit -- taking a turn back costs a stop and a delete whenever the app gets round to
    starting it -- and the price of leaving instead is a live stream in every later window, so what
    is required there is that the thread comes back the way it was found."""

    stop, nxt, slack_ms = _stop_slot(scene)
    settled = _thread_a_measured_turn_leaves(monkeypatch, stop.budget_ms)
    measured = 0
    for start_ms in range(0, stop.budget_ms, 50):
        result, page = _run(
            monkeypatch,
            budget_ms = stop.budget_ms,
            drain_after_ms = 0.0,
            start_ms = start_ms,
            stop_ms = stop_ms,
            cleanup_ms = cleanup_ms,
        )
        if result.ran:
            measured += 1
            assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
                f"{scene.name}: a turn taking {start_ms}ms to start left stop_generation spending "
                f"{page.elapsed_ms:.0f}ms of a {stop.budget_ms}ms slot with only {slack_ms}ms "
                f"before {nxt.action} opens; it ran the turn anyway: {result.expect}"
            )
        else:
            assert page.messages == settled, (
                f"{scene.name}: a turn taking {start_ms}ms to start was given up on and left the "
                f"thread at {page.messages} messages for {nxt.action} to measure, where a measured "
                f"turn leaves it at {settled}"
            )
    assert measured, f"{scene.name}: the bound refused every turn, so it measures nothing"


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_reply_that_drains_early_still_gets_its_throwaway_turn(monkeypatch, scene):
    """THE CONTROL. The reserve must not turn the wait into a refusal: a marginally slow drain is
    what it was written for, and the fast film opens this slot only 400 ms after the worst-case
    drain on the ladder. A reply that finishes 500 ms in must still be stopped and measured."""

    stop, _nxt, _slack = _stop_slot(scene)
    result, page = _run(monkeypatch, budget_ms = stop.budget_ms, drain_after_ms = 500)

    assert result.ran is True, result.reason
    assert page.filled == ["one more"]
    assert page.clicked == 1, "the throwaway turn is what gets stopped"
    assert result.expect["own_generation"] is True
    assert page.elapsed_ms <= stop.budget_ms


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_nothing_running_at_the_slot_still_gets_its_throwaway_turn(monkeypatch, scene):
    """THE SECOND CONTROL, and the path every unmodified run takes: with the pinned tail the reply
    is long finished when this slot opens, the drain wait is never entered, and the turn has the
    whole budget. It must still be sent, stopped and cleaned up on every film."""

    stop, _nxt, _slack = _stop_slot(scene)
    clock = _Clock()
    monkeypatch.setattr(actions_module, "time", clock)
    page = _Page(clock, drain_after_ms = 0.0)
    page.running = False
    result = stop_generation(
        ActionContext(
            page = page,
            cdp = None,
            cell = None,
            window = None,
            args = {},
            budget_ms = stop.budget_ms,
            dom = None,
            log = lambda _m: None,
        )
    )

    assert result.ran is True, result.reason
    assert page.clicked == 1
    assert page.elapsed_ms <= stop.budget_ms


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_every_film_still_leaves_a_real_drain_wait_after_the_reserve(scene):
    """The reserve comes out of a budget, so a slot too small to hold it would silently turn the
    wait into an immediate refusal and the axis this guard protects would be unreachable again."""

    stop, _nxt, _slack = _stop_slot(scene)
    assert stop.budget_ms - OWN_TURN_RESERVE_MS >= 1_000, (
        f"{scene.name}: a {stop.budget_ms}ms stop slot leaves "
        f"{stop.budget_ms - OWN_TURN_RESERVE_MS}ms to wait for the drain"
    )


def test_the_reserve_is_still_the_whole_of_what_its_two_halves_reserve():
    """The reserve is spent at two moments and checked at two, and the two accounts have to be the
    same account. The drain wait holds back `OWN_TURN_RESERVE_MS` for the entire turn; the
    turn-start wait holds back only the part of it that is still AHEAD once the turn is sent. If
    the halves stop summing to the whole, one of the two waits is reserving for a stretch nobody
    else believes in and the film gets the difference."""

    assert OWN_TURN_RESERVE_MS == OWN_TURN_FIXED_MS + OWN_TURN_POLL_MS
    assert OWN_TURN_POLL_MS == OWN_TURN_START_POLL_MS + OWN_TURN_STOP_POLL_MS
    # The 80 ms that settles the fill is the only fixed sleep before the send.
    assert OWN_TURN_FIXED_MS - OWN_TURN_FIXED_AFTER_SEND_MS == 80
    # And what the turn-start wait holds back has to leave the start poll something to spend, or
    # the wait is an immediate refusal and the throwaway turn is unreachable on the tightest drain.
    assert (
        OWN_TURN_RESERVE_MS - 80 - OWN_TURN_FIXED_AFTER_SEND_MS - OWN_TURN_STOP_POLL_MS
        == OWN_TURN_START_POLL_MS
    )


# ── the same give-up path, on an arm that mounts a window ────────────────────────────────────


class _WindowedPage(_Page):
    """A thread whose MOUNTED count never moves because the window refills as it grows.

    `threadTotal()` is `aria-setsize`, the store's declaration of how long the thread is;
    `messageCount()` is how much of it is in the DOM. On the shipped build they are the same
    number and nothing here is visible. On an arm whose whole purpose is to mount less of the
    thread they are not, and a before/after taken on the mounted count answers "the thread did not
    grow" to a send that worked.
    """

    WINDOW = 2

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if arg is None and "messageCount" in script:
            self.charge()
            return self.WINDOW
        return super().evaluate(script, arg)


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_turn_given_up_on_is_taken_back_on_an_arm_that_mounts_a_window(monkeypatch, scene):
    """THE DEFECT. `STOP_CLEANUP_JS` already asks `threadTotal()`, but the guard deciding whether
    to RUN it compared `messageCount()` before the send with `messageCount()` after. A windowed
    mount holds that number still while the thread grows, so the cleanup was never called at all
    and the throwaway turn -- with its stream still running -- was handed to every later action
    window and to the final census."""

    stop, _nxt, _slack = _stop_slot(scene)
    settled = _thread_a_measured_turn_leaves(monkeypatch, stop.budget_ms)
    clock = _Clock()
    page = _WindowedPage(clock, 0.0, start_ms = stop.budget_ms - 1_000)
    result = _drive(page, monkeypatch, clock, stop.budget_ms)

    assert result.ran is False
    assert page.composer == "", "the send went through, so the box is empty"
    assert page.deleted == 1, "the turn it had already sent was never deleted"
    assert page.messages == settled, (
        f"{scene.name}: a give-up on a windowed arm left the thread at {page.messages} messages, "
        f"where a measured turn leaves it at {settled}"
    )
