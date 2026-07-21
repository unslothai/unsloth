# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The live progress SSE must not time out during the pre-first-step phase.

A large model load / dataset tokenization can keep a run at step 0 for longer
than the stall timeout. Treating that as a stall ends the live stream and makes a
healthy run look frozen, so the timeout must apply only once the run is stepping.
"""

import asyncio
import sys
import types

import pytest

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.training as rt


class _Progress:
    def __init__(
        self,
        step = 0,
        total_steps = 1000,
    ):
        self.step = step
        self.total_steps = total_steps
        self.loss = None
        self.learning_rate = None
        self.epoch = None
        self.grad_norm = None
        self.num_tokens = None
        self.eval_loss = None
        self.elapsed_seconds = None
        self.eta_seconds = None


class _Backend:
    def __init__(
        self,
        *,
        active_polls,
        step_history = None,
        live_step = 0,
    ):
        self.current_job_id = "job-prep"
        self.step_history = list(step_history or [])
        self.loss_history = [1.0 for _ in self.step_history]
        self.lr_history = [1e-4 for _ in self.step_history]
        self.eval_enabled = False
        self._active_calls = 0
        self._active_polls = active_polls
        self.trainer = types.SimpleNamespace(
            training_progress = _Progress(step = live_step)
        )

    def is_training_active(self):
        self._active_calls += 1
        return self._active_calls <= self._active_polls


class _FakeRequest:
    headers = {}

    async def is_disconnected(self):
        return False


class _ReconnectRequest:
    # Reconnect carrying the last step the client already received.
    headers = {"last-event-id": "10"}

    async def is_disconnected(self):
        return False


def _raw(response):
    async def _drain():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

    return asyncio.run(asyncio.wait_for(_drain(), 15))


@pytest.fixture
def _fast_short_timeout(monkeypatch):
    """Make the poll loop instant and the stall timeout tiny."""

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(rt.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(rt, "_PROGRESS_STALL_TIMEOUT_POLLS", 3)


def test_prep_phase_does_not_time_out_before_first_step(
    monkeypatch, _fast_short_timeout
):
    # Step 0 for many polls (far past the timeout), then the run ends. Pre-step
    # this is preparation, not a stall: no error event may be emitted.
    backend = _Backend(active_polls = 20, step_history = [], live_step = 0)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    raw = _raw(
        asyncio.run(
            rt.stream_training_progress(_FakeRequest(), current_subject = "tester")
        )
    )

    assert (
        backend._active_calls > rt._PROGRESS_STALL_TIMEOUT_POLLS + 1
    ), "the loop must have run past the stall threshold for this test to be meaningful"
    assert "event: heartbeat" in raw, "prep heartbeats should still flow"
    assert (
        "event: error" not in raw
    ), "a still-preparing run must not be timed out as a stall"


def test_stall_after_first_step_still_times_out(monkeypatch, _fast_short_timeout):
    # Emits a live step (so seen_live_step becomes True) then stays put: a genuine
    # post-step stall that must still trigger the timeout error.
    backend = _Backend(active_polls = 100, step_history = [1, 2], live_step = 5)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    raw = _raw(
        asyncio.run(
            rt.stream_training_progress(_FakeRequest(), current_subject = "tester")
        )
    )

    assert "event: error" in raw, "a real post-step stall should still time out"


def test_reconnect_to_stepped_run_still_times_out(monkeypatch, _fast_short_timeout):
    # Client reconnects at step 10 (Last-Event-ID) to a run that already stepped
    # then hangs (only heartbeats): the post-step stall timeout must still fire.
    # Without seeding seen_live_step from the resume point it resets to False and
    # never times out for this client.
    backend = _Backend(active_polls = 100, step_history = [10], live_step = 10)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    raw = _raw(
        asyncio.run(
            rt.stream_training_progress(_ReconnectRequest(), current_subject = "tester")
        )
    )

    assert (
        "event: error" in raw
    ), "a reconnect to an already-stepped run that then stalls must still time out"
