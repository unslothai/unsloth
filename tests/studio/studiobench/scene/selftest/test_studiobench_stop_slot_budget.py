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
throwaway turn on time it no longer had: 1,280 ms of fixed waits with nothing left to pay for them.
`stop_generation` closes 300 ms before `scroll_after` opens on the fast film and 500 ms before it on
the quick one, so on both of those the overrun lands inside `scroll_after`'s own window, and on the
fast film -- 1,200 ms of budget behind a 300 ms gap -- it can take the whole of it.

Reachable on supported input, and only on supported input: the drain wait exists for
`--stream-tail-chars`, whose entire purpose is a reply long enough to still be streaming when this
slot opens. A tail that puts the drain near the end of the slot is exactly the case it was written
for.

WHAT IS FAKED AND WHAT IS NOT. The action under test is the shipped `stop_generation`. The page is
a shim of the five calls it makes, and the clock is a counter that `wait_for_timeout` advances, so
the test measures the waits the action ASKS for rather than how long this machine took to make
them. Polling for the turn to start and for the stream to stop is free in the shim, which makes
every number below a floor: the real action spends more, never less.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.types import ActionContext  # noqa: E402
from studiobench.scene import actions as actions_module  # noqa: E402
from studiobench.scene.actions import OWN_TURN_FIXED_MS, stop_generation  # noqa: E402
from studiobench.scene.schedule import FAST, QUICK, STANDARD  # noqa: E402


class _Clock:
    """The action reads `time.monotonic`; the page's waits are what move it."""

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
        self.pressed.append(key)
        if key == "Enter" and "one more" in self._page.filled:
            self._page.running = True


class _Page:
    """The cell's own reply is streaming and drains `drain_after_ms` into the slot."""

    def __init__(self, clock: _Clock, drain_after_ms: float) -> None:
        self._clock = clock
        self._entered = clock.t
        self._drain_after_ms = drain_after_ms
        self.running = True
        self.filled: list[str] = []
        self.clicked = 0
        self.keyboard = _Keyboard(self)

    @property
    def elapsed_ms(self) -> float:
        return (self._clock.t - self._entered) * 1_000.0

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "isRunning" in script:
            if self.running and not self.filled and self.elapsed_ms >= self._drain_after_ms:
                self.running = False
            return self.running
        if "composerText" in script:
            return ""
        if "assistantChars" in script:
            return 9_200
        return {"removed": True, "before": 2, "after": 1, "reason": None}

    def fill(self, _selector: str, text: str) -> None:
        self.filled.append(text)

    def wait_for_timeout(self, ms) -> None:
        self._clock.advance_ms(ms)

    def query_selector(self, selector: str):
        page = self

        class _Button:
            def click(_self) -> None:
                page.clicked += 1
                page.running = False

        return _Button() if "Stop generating" in selector else None


def _run(monkeypatch, *, budget_ms: int, drain_after_ms: float):
    clock = _Clock()
    monkeypatch.setattr(actions_module, "time", clock)
    page = _Page(clock, drain_after_ms)
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

    A reply that is still streaming 100 ms before this slot closes leaves nothing for the 1,280 ms
    of fixed waits the throwaway turn costs. The action has to stop within the gap before the next
    slot opens -- 300 ms on the fast film, 500 ms on the quick one -- or the next action's window
    pays for it.
    """

    stop, nxt, slack_ms = _stop_slot(scene)
    result, page = _run(monkeypatch, budget_ms = stop.budget_ms, drain_after_ms = stop.budget_ms - 100)

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
def test_every_film_still_leaves_a_real_drain_wait_after_the_reserve(scene):
    """The reserve comes out of a budget, so a slot too small to hold it would silently turn the
    wait into an immediate refusal and the axis this guard protects would be unreachable again."""

    stop, _nxt, _slack = _stop_slot(scene)
    assert stop.budget_ms - OWN_TURN_FIXED_MS >= 1_000, (
        f"{scene.name}: a {stop.budget_ms}ms stop slot leaves "
        f"{stop.budget_ms - OWN_TURN_FIXED_MS}ms to wait for the drain"
    )
