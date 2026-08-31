# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations


def test_sandbox_warm_probe_thread_start_failure_is_best_effort(monkeypatch):
    import run

    messages: list[str] = []

    class StartFails:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread limit reached")

    monkeypatch.setattr(run.threading, "Thread", StartFails)
    monkeypatch.setattr(
        run.logger,
        "debug",
        lambda message, exc: messages.append(message % exc),
    )

    run._start_sandbox_probe()

    assert messages == [
        "sandbox availability probe could not start: thread limit reached"
    ]
