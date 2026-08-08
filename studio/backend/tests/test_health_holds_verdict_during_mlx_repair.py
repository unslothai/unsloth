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
    # start_mlx_autorepair_if_needed() gates on this, and the real one imports mlx_vlm,
    # which is not installed here anyway.
    monkeypatch.setattr(mlx_repair, "mlx_stack_available", lambda: False)


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
            body = client.get(
                "/api/health", headers = {"Authorization": "Bearer probe"}
            ).json()
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
