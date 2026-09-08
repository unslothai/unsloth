# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/health does not publish a verdict the MLX self-heal is about to overturn.

Detection settles before utils.mlx_repair gets its turn, so an Apple Silicon host whose MLX
stack is missing or unreadable answers chat_only=true with reason "mlx_unavailable" first.
The frontend reads that as measured, greys Train and Video and explains them with "run
`unsloth studio update`"; the background reinstall then lands, chat_only flips false and the
sidebar's recovery poll enables both rows. That is the reported "greyed out on launch, then
they come out". Health keeps replying provisionally for that window instead, through the
existing hardware_detecting shape, so the rows spin until there is a real answer.

Everything a chat-only host relies on must be unchanged: an Intel Mac, a CPU-only Linux box,
a Mac with the self-heal opted out, and a Mac whose repair has finished must all still get a
settled verdict and the tooltip that goes with it.

The hold is bounded in both halves. A live worker is waited on for as long as its install
takes; a repair that has not started is only a promise, and a promise nothing ever keeps must
expire, or the rows spin for the whole session instead of settling into the greyed state a
broken MLX stack has genuinely earned.

CPU-only, no network, no GPU, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware.hardware as hw  # noqa: E402
import utils.mlx_repair as mlx_repair  # noqa: E402


class _Worker:
    """Stand-in for the mlx-autorepair thread, alive or finished."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture
def apple_silicon(monkeypatch):
    """An Apple Silicon host with the self-heal enabled and nothing attempted yet."""
    monkeypatch.delenv(mlx_repair.DISABLE_ENV_VAR, raising = False)
    # Both modules ask the question, and hardware.py has its own copy of the helper.
    monkeypatch.setattr(mlx_repair, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlx_repair, "_attempted", False, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", None, raising = False)
    # Same reason as the hold below: a stamp left by an earlier test would decide whether
    # this one's worker still counts as in flight.
    monkeypatch.setattr(mlx_repair, "_repair_started_at", None, raising = False)
    # start_mlx_autorepair_if_needed() gates on this, and the real one imports mlx_vlm,
    # which is not installed here anyway.
    monkeypatch.setattr(mlx_repair, "mlx_stack_available", lambda: False)
    # The pre-start hold is stamped on first use and keyed by detection generation, so a
    # stamp left by an earlier test would decide this one's answer.
    import main as main_mod

    monkeypatch.setattr(main_mod, "_mlx_prestart_hold", None)


class _Clock:
    """A monotonic clock the test drives, so the hold windows cost no wall time."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    import main as main_mod

    fake = _Clock()
    monkeypatch.setattr(main_mod, "_mlx_prestart_clock", fake)
    return fake


# ------------------------------------------------------------- the predicate


def test_a_running_repair_is_in_flight(apple_silicon, monkeypatch):
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    assert mlx_repair.mlx_repair_in_flight() is True


def test_a_finished_repair_is_not_in_flight(apple_silicon, monkeypatch):
    """The worker exits only after it has published whatever it is going to publish: a
    successful re-detect, or nothing at all on a failed install. Either way the verdict
    standing when it exits is final, so a chat-only Mac gets its greyed rows and tooltip."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = False), raising = False)
    assert mlx_repair.mlx_repair_in_flight() is False


def test_a_repair_that_has_not_started_yet_is_in_flight(apple_silicon):
    """The half the window is mostly made of. The self-heal is scheduled after the torch
    warm (main._post_warm_background_work), so detection has published "mlx_unavailable"
    tens of seconds before the worker exists."""
    assert mlx_repair.mlx_repair_in_flight() is True


def test_a_worker_that_could_not_be_started_settles_the_verdict(apple_silicon, monkeypatch):
    """The latch is claimed before the thread exists, so a start() that raises must read as
    finished rather than as forever pending."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", None, raising = False)
    assert mlx_repair.mlx_repair_in_flight() is False


def test_the_opt_out_settles_the_verdict_immediately(apple_silicon, monkeypatch):
    """UNSLOTH_DISABLE_MLX_AUTOREPAIR=1 means no repair is coming, so holding the verdict
    back would spin Train and Video for the rest of the session."""
    monkeypatch.setenv(mlx_repair.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    assert mlx_repair.mlx_repair_in_flight() is False


def test_a_non_apple_host_is_never_in_flight(apple_silicon, monkeypatch):
    """The self-heal is Apple Silicon only, Intel Macs included."""
    monkeypatch.setattr(mlx_repair, "is_apple_silicon", lambda: False)
    assert mlx_repair.mlx_repair_in_flight() is False


def test_the_worker_is_published_together_with_the_latch(apple_silicon, monkeypatch):
    """A gap between claiming _attempted and recording the thread is a window where the
    predicate reads "attempted, nothing alive" and settles the verdict the repair is
    about to overturn -- the exact bug, just narrower."""
    workers: list[_Worker] = []

    def _thread(*_a, **kwargs):
        assert kwargs.get("name") == "mlx-autorepair"
        workers.append(_Worker(alive = True))
        return workers[-1]

    monkeypatch.setattr(mlx_repair.threading, "Thread", _thread)

    assert mlx_repair.start_mlx_autorepair_if_needed() is True
    assert workers and workers[0].started, "the self-heal worker was never started"
    assert mlx_repair._repair_thread is workers[0]
    assert mlx_repair.mlx_repair_in_flight() is True
    # Still one attempt per process.
    assert mlx_repair.start_mlx_autorepair_if_needed() is False
    assert len(workers) == 1


def test_started_splits_the_two_halves_of_in_flight(apple_silicon, monkeypatch):
    """The distinction the pre-start window is built on: both halves are in flight, but
    only one of them has a worker to wait for."""
    assert mlx_repair.mlx_repair_in_flight() is True
    assert mlx_repair.mlx_repair_started() is False

    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    assert mlx_repair.mlx_repair_in_flight() is True
    assert mlx_repair.mlx_repair_started() is True


# ---------------------------------------------------------- the hardware gate


def test_only_the_mlx_reason_holds_a_verdict_back(apple_silicon):
    """Every other chat-only host settles exactly as it does today: no self-heal is
    coming for an Intel Mac or a box with no GPU, and saying "still checking" would spin
    their rows forever."""
    assert hw.verdict_pending_mlx_repair(True, "mlx_unavailable") is True
    for reason in ("intel_mac", "no_gpu", "detection_failed", None):
        assert hw.verdict_pending_mlx_repair(True, reason) is False, reason


def test_a_training_capable_verdict_is_never_held_back(apple_silicon):
    """chat_only false is the answer the repair is trying to reach; publish it."""
    assert hw.verdict_pending_mlx_repair(False, None) is False
    assert hw.verdict_pending_mlx_repair(False, "mlx_unavailable") is False


def test_an_intel_mac_verdict_is_not_held_back(apple_silicon, monkeypatch):
    """Belt and braces on the platform check: the reason is derived from
    is_apple_silicon() in the first place, so the two must not disagree."""
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    assert hw.verdict_pending_mlx_repair(True, "mlx_unavailable") is False


# --------------------------------------------------- the pre-start hold window
#
# The scheduler runs in main._post_warm_background_work, after join_background_warm(), so
# on a cold Mac "not started yet" is legitimately minutes and no fixed number can stand in
# for it. main bounds that half with the warm's own progress plus a handoff grace, under an
# absolute ceiling for the warm that never ends.


def _superseded(monkeypatch, *, warming: bool) -> bool:
    """Ask main's gate directly against a settled mlx_unavailable verdict."""
    import main as main_mod

    monkeypatch.setattr(main_mod, "_torch_warm_in_progress", lambda: warming)
    return main_mod._superseded_by_mlx_repair((True, "mlx_unavailable"))


def _spend_the_handoff_grace(monkeypatch, clock) -> None:
    """Run the grace out the way a caller must: observe the warm stopped, then wait.

    The grace opens on the first stopped reading, not on the clock, so advancing time
    without asking spends nothing. Tests that want an expired window have to ask twice.
    """
    import main as main_mod

    assert _superseded(monkeypatch, warming = False) is True
    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S + 1)


def test_a_repair_that_has_not_started_holds_the_verdict_while_the_warm_runs(
    apple_silicon, clock, monkeypatch
):
    """The normal case, and the one no timeout could serve: the warm imports transformers
    and datasets before the scheduler gets its turn, which is minutes on a cold Mac."""
    assert _superseded(monkeypatch, warming = True) is True
    import main as main_mod

    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S * 10)
    assert _superseded(monkeypatch, warming = True) is True


def test_the_handoff_grace_covers_the_gap_after_the_warm(apple_silicon, clock, monkeypatch):
    """start_mlx_autorepair_if_needed() is the next statement after join_background_warm(),
    so the verdict must not settle in the instant between the two."""
    import main as main_mod

    assert _superseded(monkeypatch, warming = True) is True
    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S - 1)
    assert _superseded(monkeypatch, warming = False) is True


def test_a_repair_that_never_starts_stops_holding_the_verdict(apple_silicon, clock, monkeypatch):
    """The risk this window exists for. The warm finished and the scheduler never claimed
    the latch (its import raised and _post_warm_background_work swallowed it), so "not
    started yet" would otherwise be a permanent answer and the rows would spin for the whole
    session. Instead the Mac gets the greyed rows and the tooltip its stack has earned."""
    assert _superseded(monkeypatch, warming = True) is True
    _spend_the_handoff_grace(monkeypatch, clock)
    assert _superseded(monkeypatch, warming = False) is False


def test_a_gap_in_polling_does_not_spend_the_grace_before_the_handoff(
    apple_silicon, clock, monkeypatch
):
    """The grace has to start when the warm stops, not when a poll last saw it running.

    The warm's final stages are C-extension imports that hold the GIL for seconds at a
    time, so health requests queue behind them and the next one served can be the first in
    minutes. Measured from the last observed poll the grace would already be spent by the
    time anyone could ask, and the gate would publish the mlx_unavailable verdict during
    the very handoff it exists to cover -- the frontend then stores that as final and greys
    Train and Video until the repair flips them back, which is the reported bug.
    """
    import main as main_mod

    assert _superseded(monkeypatch, warming = True) is True
    # Nobody asks for far longer than the grace, because the warm is holding the GIL.
    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S * 6)
    assert _superseded(monkeypatch, warming = False) is True
    # And it still expires normally once the handoff has actually had its window.
    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S + 1)
    assert _superseded(monkeypatch, warming = False) is False


def test_a_warm_that_resumes_reopens_the_handoff_grace(apple_silicon, clock, monkeypatch):
    """A stopped reading between two running ones was a lull, not the handoff, so it must
    not leave a countdown running that expires while the warm is still working."""
    import main as main_mod

    assert _superseded(monkeypatch, warming = True) is True
    assert _superseded(monkeypatch, warming = False) is True
    assert _superseded(monkeypatch, warming = True) is True
    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S + 1)
    assert _superseded(monkeypatch, warming = False) is True


def test_a_warm_that_never_ends_hits_the_ceiling(apple_silicon, clock, monkeypatch):
    """_torch_warm_in_progress() goes false when the warm thread dies, but one parked
    forever inside an import never does. The ceiling is the backstop for exactly that."""
    import main as main_mod

    assert _superseded(monkeypatch, warming = True) is True
    clock.advance(main_mod._MLX_PRESTART_CEILING_S + 1)
    assert _superseded(monkeypatch, warming = True) is False


def test_a_live_worker_holds_the_verdict_past_every_window(apple_silicon, clock, monkeypatch):
    """A real reinstall is capped by _REPAIR_TIMEOUT_S, not by these windows. Expiring under
    a working install would grey the rows moments before the repair flips them back."""
    import main as main_mod

    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    clock.advance(main_mod._MLX_PRESTART_CEILING_S * 10)
    assert _superseded(monkeypatch, warming = False) is True


def test_a_worker_parked_past_its_budget_stops_holding_the_verdict(
    apple_silicon, clock, monkeypatch
):
    """attempt_mlx_repair times the uv subprocess, but not the mlx_stack_available()
    imports that verify the install nor the detect_hardware() pass after it. Those import
    mlx.core, mlx_lm and mlx_vlm, which this module already assumes can park forever on a
    broken stack, so an alive thread was an unbounded answer: the rows would spin for the
    whole session rather than settle into the chat-only verdict the stack has earned."""
    worker_clock = _Clock()
    monkeypatch.setattr(mlx_repair, "_repair_clock", worker_clock, raising = False)
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_started_at", worker_clock.now, raising = False)

    assert mlx_repair.mlx_repair_in_flight() is True
    worker_clock.advance(mlx_repair._WORKER_BUDGET_S - 1)
    assert mlx_repair.mlx_repair_in_flight() is True
    worker_clock.advance(2)
    assert mlx_repair.mlx_repair_in_flight() is False
    # And the gate follows it, so the reply settles instead of staying provisional.
    assert _superseded(monkeypatch, warming = False) is False


def test_a_worker_inside_its_budget_is_never_cut_short(apple_silicon, monkeypatch):
    """The budget is a backstop for a parked worker, not a cap on a working reinstall: a
    slow but healthy uv install must keep the verdict held for its whole run."""
    worker_clock = _Clock()
    monkeypatch.setattr(mlx_repair, "_repair_clock", worker_clock, raising = False)
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_started_at", worker_clock.now, raising = False)

    worker_clock.advance(mlx_repair._REPAIR_TIMEOUT_S)
    assert mlx_repair.mlx_repair_in_flight() is True


def test_a_worker_that_starts_late_takes_over_an_expired_window(apple_silicon, clock, monkeypatch):
    """The windows bound the promise, not the repair. A scheduler that arrives after the
    grace still gets the verdict held for as long as it is installing."""
    assert _superseded(monkeypatch, warming = True) is True
    _spend_the_handoff_grace(monkeypatch, clock)
    assert _superseded(monkeypatch, warming = False) is False

    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)
    assert _superseded(monkeypatch, warming = False) is True


def test_a_re_detected_verdict_gets_its_own_window(apple_silicon, clock, monkeypatch):
    """Detection is not once-per-process. A later pass that republishes mlx_unavailable is
    a new verdict, so it must not inherit the spent window of the one before it."""
    assert _superseded(monkeypatch, warming = True) is True
    _spend_the_handoff_grace(monkeypatch, clock)
    assert _superseded(monkeypatch, warming = False) is False

    monkeypatch.setattr(hw, "DETECTION_GENERATION", hw.DETECTION_GENERATION + 1, raising = False)
    assert _superseded(monkeypatch, warming = False) is True


def test_a_spent_window_does_not_revive_a_settled_host(apple_silicon, clock, monkeypatch):
    """The windows only ever settle a verdict earlier; they must not hold one the opt-out,
    an Intel Mac or a non-Mac had already settled."""
    import main as main_mod

    monkeypatch.setenv(mlx_repair.DISABLE_ENV_VAR, "1")
    assert _superseded(monkeypatch, warming = True) is False
    clock.advance(main_mod._MLX_PRESTART_CEILING_S + 1)
    assert _superseded(monkeypatch, warming = True) is False


def test_health_settles_the_verdict_once_the_window_is_spent(apple_silicon, clock, monkeypatch):
    """End to end: the held-back reply must not be permanent on a host whose repair never
    arrives, or Train and Video spin for the session instead of explaining themselves."""
    import main as main_mod

    monkeypatch.setattr(main_mod, "_torch_warm_in_progress", lambda: False)
    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    assert body["hardware_detecting"] is True

    clock.advance(main_mod._MLX_PRESTART_GRACE_AFTER_WARM_S + 1)
    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert "hardware_detecting" not in body
    assert body["device_type"]
    assert body["chat_only"] is True
    assert body["chat_only_reason"] == "mlx_unavailable"


def test_an_unaskable_self_heal_settles_the_verdict(apple_silicon, monkeypatch):
    """If mlx_repair cannot even be consulted, settle rather than spin forever."""

    def _boom() -> bool:
        raise RuntimeError("mlx_repair is unimportable")

    monkeypatch.setattr(mlx_repair, "mlx_repair_in_flight", _boom)
    assert hw.verdict_pending_mlx_repair(True, "mlx_unavailable") is False


# ----------------------------------------------------------------- the route


def _health(monkeypatch, *, chat_only: bool, reason: str | None) -> dict:
    """Drive /api/health as an authed caller against a settled verdict."""
    import auth.authentication as _authmod
    import main as main_mod
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    hw_mod = main_mod._hw_module
    monkeypatch.setattr(hw_mod, "DEVICE", hw_mod.DeviceType.CPU, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY", chat_only, raising = False)
    monkeypatch.setattr(hw_mod, "CHAT_ONLY_REASON", reason, raising = False)
    was_complete = hw_mod.DETECTION_COMPLETE.is_set()
    hw_mod.DETECTION_COMPLETE.set()

    async def _subject(_creds):
        return "tester"

    # health_check imports get_current_subject inside the function, so patching the module
    # attribute is enough and keeps this off the real JWT/storage path.
    monkeypatch.setattr(_authmod, "get_current_subject", _subject)
    app = FastAPI()
    app.add_api_route("/api/health", main_mod.health_check, methods = ["GET"])
    app.add_api_route("/api/liveness", main_mod.liveness_check, methods = ["GET"])
    try:
        with TestClient(app) as client:
            body = client.get("/api/health", headers = {"Authorization": "Bearer probe"}).json()
            body["_liveness"] = client.get("/api/liveness").json()
            return body
    finally:
        if not was_complete:
            hw_mod.DETECTION_COMPLETE.clear()


def test_health_replies_provisionally_while_the_repair_runs(apple_silicon, monkeypatch):
    """The fix. config/env.ts keys `fetched` on device_type, so omitting it keeps
    capabilitiesUnknown() true and resolveNavRowState spins the rows instead of greying
    them behind a tooltip the reinstall is about to make wrong."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert body["hardware_detecting"] is True, "the held-back verdict was published as measured"
    assert "device_type" not in body, (
        "the frontend caches the first reply carrying device_type as authoritative, so "
        "this one greys Train and Video for the rest of the session"
    )
    assert "chat_only_reason" not in body
    # Conservative direction, unchanged: never offer Train on a host that may not have it.
    assert body["chat_only"] is True
    assert body["version"], "the launcher-facing fields are unaffected"


def test_health_settles_the_verdict_once_the_repair_has_finished(apple_silicon, monkeypatch):
    """A Mac the self-heal could not fix must still end up with genuinely disabled rows
    and the "run `unsloth studio update`" tooltip that explains them."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = False), raising = False)

    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert "hardware_detecting" not in body
    assert body["device_type"]
    assert body["chat_only"] is True
    assert body["chat_only_reason"] == "mlx_unavailable"


def test_a_chat_only_host_with_another_reason_still_settles(apple_silicon, monkeypatch):
    """No self-heal exists for these, so nothing may hold their verdict back."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    for reason in ("intel_mac", "no_gpu"):
        body = _health(monkeypatch, chat_only = True, reason = reason)
        assert "hardware_detecting" not in body, reason
        assert body["chat_only_reason"] == reason
        assert body["device_type"]


def test_a_training_capable_verdict_still_settles(apple_silicon, monkeypatch):
    """Negative control: a repaired host publishes immediately, or Train never comes back."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    body = _health(monkeypatch, chat_only = False, reason = None)

    assert "hardware_detecting" not in body
    assert body["chat_only"] is False
    assert body["device_type"]


def test_a_held_back_verdict_is_not_reported_as_deferred(apple_silicon, monkeypatch):
    """The torch-warm kill switch does not disable the MLX self-heal, so the two can be on
    at once. "Deferred" tells env.ts nothing will ever settle, and it answers by storing the
    conservative chat_only -- which greys the rows, the outcome this change exists to stop."""
    monkeypatch.setenv("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", "1")
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert body["hardware_detecting"] is True
    assert "hardware_detection_deferred" not in body


def test_the_liveness_route_keeps_its_settled_verdict(apple_silicon, monkeypatch):
    """Scoped to health on purpose. The desktop watchdog probes /api/liveness and holds its
    startup grace open while hardware_detecting is set, so a 15-minute reinstall must not
    stretch that grace. Only the UI reads chat_only, and only the UI greys a row on it."""
    monkeypatch.setattr(mlx_repair, "_attempted", True, raising = False)
    monkeypatch.setattr(mlx_repair, "_repair_thread", _Worker(alive = True), raising = False)

    body = _health(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert body["hardware_detecting"] is True
    assert "hardware_detecting" not in body["_liveness"], (
        "the repair hold reached /api/liveness; the launcher's health watchdog would "
        "keep its startup grace open for the whole reinstall"
    )
