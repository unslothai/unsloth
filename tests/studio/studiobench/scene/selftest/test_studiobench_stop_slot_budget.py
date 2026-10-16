# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stop_generation must fit its slot, or an overrun is charged to the next action's window."""

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

#: One CDP round trip; the large page-side waits are charged separately below.
ROUND_TRIP_MS = 4.0

START_MS = 120.0

STOP_MS = 90.0

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
            return
        # Sent, not started: the turn appears now but the reply starts `start_ms` later.
        self._page.composer = ""
        self._page.messages += 2
        self._page.sent_at_ms = self._page.elapsed_ms


class _Page:
    """Models the composer and thread, not stubs, so a refused send is told apart from an accepted one."""

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
        menu_never_opens: bool = False,
    ) -> None:
        self._clock = clock
        self._entered = clock.t
        self._drain_after_ms = drain_after_ms
        self._latency = latency
        self._start_ms = start_ms
        self._stop_ms = stop_ms
        self._cleanup_ms = cleanup_ms
        self.accepts_send = accepts_send
        # Cleanup reaches Delete via the More menu; a menu that never mounts burns the whole wait.
        self.menu_never_opens = menu_never_opens
        self.running = True
        self.filled: list[str] = []
        self.composer = ""
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
        if arg is not None:
            if self.menu_never_opens:
                self.charge(arg["menuWaitMs"])
                return {
                    "removed": False,
                    "before": self.messages,
                    "after": self.messages,
                    "reason": "no Delete control on the throwaway turn",
                }
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
        # `stop_generation` counts its own turn via `threadTotal()`; equals mounted count on a full arm.
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
    """A reply draining at the slot's end must still be stopped within the gap before the next slot."""

    stop, nxt, slack_ms = _stop_slot(scene)
    late = stop.budget_ms - 100
    result, page = _run(monkeypatch, budget_ms = stop.budget_ms, drain_after_ms = late)

    assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
        f"{scene.name}: stop_generation spent {page.elapsed_ms:.0f}ms of a {stop.budget_ms}ms "
        f"slot with only {slack_ms}ms before {nxt.action} opens"
    )
    assert result.ran is False
    assert page.clicked == 0
    assert "one more" not in page.filled


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_no_moment_the_reply_can_drain_lets_the_action_spend_the_next_slot(monkeypatch, scene):
    """Swept in 50 ms steps: whenever the reply drains, the action returns inside its slot plus the gap."""

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
def test_a_cleanup_menu_that_never_opens_still_ends_inside_the_slot(monkeypatch, scene):
    """The cleanup now opens the reply's More menu before it can select Delete, and that wait was not
    priced into `OWN_TURN_RESERVE_MS`. It is bounded by what is left of the slot instead, so the
    worst case, a menu that never mounts, is swept over every drain time like the case above."""

    stop, nxt, slack_ms = _stop_slot(scene)
    for drained_at in range(0, stop.budget_ms, 50):
        result, page = _run(
            monkeypatch,
            budget_ms = stop.budget_ms,
            drain_after_ms = drained_at,
            menu_never_opens = True,
        )
        assert page.elapsed_ms <= stop.budget_ms + slack_ms, (
            f"{scene.name}: a reply draining {drained_at}ms into the slot, then a cleanup menu "
            f"that never opened, left stop_generation spending {page.elapsed_ms:.0f}ms of a "
            f"{stop.budget_ms}ms slot with only {slack_ms}ms before {nxt.action} opens"
        )


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_send_the_app_refused_is_bounded_by_the_slot_and_not_by_eight_seconds(monkeypatch, scene):
    """A refused send is bounded by the slot, not the turn-start timeout, since nothing was committed."""

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
    """Returns the thread message count a measured turn leaves; give-up paths must match it."""
    result, page = _run(monkeypatch, budget_ms = budget_ms, drain_after_ms = 0.0, start_ms = 0.0)
    assert result.ran is True, result.reason
    return page.messages


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
@pytest.mark.parametrize("late_by_ms", [0, 400, 800])
def test_a_turn_that_starts_after_the_slot_bound_is_not_left_generating(
    monkeypatch, scene, late_by_ms
):
    """A turn that starts after the slot bound must still be stopped and deleted, not left generating."""

    stop, _nxt, _slack = _stop_slot(scene)
    settled = _thread_a_measured_turn_leaves(monkeypatch, stop.budget_ms)
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
    page.settle(2_000)
    assert page._is_running() is False


@pytest.mark.parametrize("scene", [FAST, QUICK, STANDARD], ids = lambda s: s.name)
def test_a_turn_that_never_starts_at_all_is_still_taken_out_of_the_thread(monkeypatch, scene):
    """A turn that never starts is still taken out of the thread, bounded by TURN_START_TIMEOUT_MS."""

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
    """A turn starting at the bound must still fit its stop, delete and driver calls inside the slot."""

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
    """The two halves of the turn reserve must still add up to the whole, or one wait over-reserves."""

    assert OWN_TURN_RESERVE_MS == OWN_TURN_FIXED_MS + OWN_TURN_POLL_MS
    assert OWN_TURN_POLL_MS == OWN_TURN_START_POLL_MS + OWN_TURN_STOP_POLL_MS
    assert OWN_TURN_FIXED_MS - OWN_TURN_FIXED_AFTER_SEND_MS == 80
    # The turn-start wait must leave the start poll some budget, or the throwaway turn is unreachable.
    assert (
        OWN_TURN_RESERVE_MS - 80 - OWN_TURN_FIXED_AFTER_SEND_MS - OWN_TURN_STOP_POLL_MS
        == OWN_TURN_START_POLL_MS
    )


class _WindowedPage(_Page):
    """On a windowed mount messageCount stays flat as the thread grows, so read threadTotal instead."""

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
    """Cleanup guard must compare threadTotal, not messageCount, or a windowed mount skips cleanup."""

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
