# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SIGTERM and Ctrl+C reach the same stop-and-save path as the Stop button.

The save has to finish before uvicorn's lifespan teardown starts: measured on the
published Docker image, a save that overlapped the teardown lost the worker mid-write.
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
    import core.training.training as training

    order = []
    fake = SimpleNamespace(
        stop_for_shutdown = lambda: order.append("stop_for_shutdown") or True,
        force_terminate = lambda: order.append("force_terminate"),
    )
    monkeypatch.setattr(training, "_training_backend", fake)
    run._graceful_shutdown(_Server(order))
    assert order == ["stop_for_shutdown", "should_exit", "force_terminate"]
