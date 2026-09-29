# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shutdown order: both trainers are asked to stop and save before anything is torn down.

Measured on the published Docker image, a save that overlapped uvicorn's lifespan teardown
lost the worker mid-write, so the wait has to come before should_exit and the force-kill.
"""

from types import SimpleNamespace


class _Server:
    def __init__(self, order):
        self._order = order

    @property
    def should_exit(self):
        return False

    @should_exit.setter
    def should_exit(self, value):
        self._order.append("should_exit")


def test_the_run_is_saved_before_the_server_stops_and_before_the_kill(monkeypatch):
    import run
    import core.training.diffusion_training_service as diffusion
    import core.training.training as training

    order = []
    fake = SimpleNamespace(
        stop_for_shutdown = lambda: order.append("stop_for_shutdown") or True,
        force_terminate = lambda: order.append("force_terminate"),
    )
    fake_diffusion = SimpleNamespace(
        stop_for_shutdown = lambda timeout: order.append(("diffusion", timeout)) or True,
    )
    monkeypatch.setattr(training, "_training_backend", fake)
    monkeypatch.setattr(diffusion, "_service", fake_diffusion)
    run._graceful_shutdown(_Server(order))
    assert order == [
        "stop_for_shutdown",
        ("diffusion", training._SHUTDOWN_STOP_TIMEOUT_S),
        "should_exit",
        "force_terminate",
    ]
