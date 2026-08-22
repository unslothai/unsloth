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
from studiobench.scene.actions import OWN_TURN_RESERVE_MS, stop_generation  # noqa: E402
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
        if key == "Enter" and "one more" in self._page.filled:
            # Sent, not started. The turn begins generating START_MS later, and the action has to
            # poll for it -- which is the cost the first version of this shim handed out free.
            self._page.sent_at_ms = self._page.elapsed_ms


class _Page:
    """The cell's own reply is streaming and drains `drain_after_ms` into the slot.

    `latency = False` restores the instant-answer page the first version of this file used, which
    is kept only as the control below: it is the floor, not the machine.
    """

    def __init__(
        self,
        clock: _Clock,
        drain_after_ms: float,
        *,
        latency: bool = True,
    ) -> None:
        self._clock = clock
        self._entered = clock.t
        self._drain_after_ms = drain_after_ms
        self._latency = latency
        self.running = True
        self.filled: list[str] = []
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

    def _is_running(self) -> bool:
        if self.clicked_at_ms is not None:
            return self.elapsed_ms < self.clicked_at_ms + (STOP_MS if self._latency else 0.0)
        if self.sent_at_ms is not None:
            return self.elapsed_ms >= self.sent_at_ms + (START_MS if self._latency else 0.0)
        if self.running and self.elapsed_ms >= self._drain_after_ms:
            self.running = False
        return self.running

    def evaluate(
        self,
        script,
        arg = None,
    ):
        self.charge()
        if "isRunning" in script:
            return self._is_running()
        if "composerText" in script:
            return ""
        if "assistantChars" in script:
            return 9_200
        self.charge(CLEANUP_MS)
        return {"removed": True, "before": 2, "after": 1, "reason": None}

    def fill(self, _selector: str, text: str) -> None:
        self.charge()
        self.filled.append(text)

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


def _run(
    monkeypatch,
    *,
    budget_ms: int,
    drain_after_ms: float,
    latency: bool = True,
):
    clock = _Clock()
    monkeypatch.setattr(actions_module, "time", clock)
    page = _Page(clock, drain_after_ms, latency = latency)
    result = stop_generation(
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
    return result, page


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
def test_a_turn_that_never_starts_is_bounded_by_the_slot_and_not_by_eight_seconds(
    monkeypatch, scene
):
    """The other unbounded wait after the drain. `TURN_START_TIMEOUT_MS` is 8 s, which is 2.7x the
    whole stop slot on the fast and quick films, so a turn that never came up used to hold the
    action for eight seconds and take the next two or three slots with it. Nothing is committed on
    this path -- the action already reports `not_run` when the turn does not start -- so the wait
    is the one here that may be cut to fit the slot."""

    stop, nxt, slack_ms = _stop_slot(scene)
    clock = _Clock()
    monkeypatch.setattr(actions_module, "time", clock)
    page = _Page(clock, drain_after_ms = 0.0)
    # The turn is sent and never starts generating.
    page.running = False
    monkeypatch.setattr(page.keyboard, "press", lambda _key: page.charge())

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

    assert result.ran is False
    assert "did not start" in (result.reason or "")
    assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
        f"{scene.name}: waiting for a turn that never started cost {page.elapsed_ms:.0f}ms of a "
        f"{stop.budget_ms}ms slot with only {slack_ms}ms before {nxt.action} opens"
    )


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
