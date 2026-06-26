# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The SSE progress stream must follow the live progress step during
non-finite-loss stretches (loss reported as null) instead of replaying the
last finite step/loss pair from the metric histories, which skip NaN steps."""

import asyncio
import json
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
    def __init__(self):
        self.step = 5
        self.total_steps = 10
        self.loss = None  # cleared by the NaN honesty fix in core training
        self.learning_rate = 8e-5
        self.epoch = 0.1
        self.grad_norm = None
        self.num_tokens = None
        self.eval_loss = None
        self.elapsed_seconds = None
        self.eta_seconds = None


class _FakeBackend:
    """Finite history stops at step 2; live progress is at step 5 with NaN
    (loss=None). Active for a few polls, then done."""

    def __init__(self, active_polls = 2):
        self.current_job_id = "job-1"
        self.step_history = [1, 2]
        self.loss_history = [2.0, 1.5]
        self.lr_history = [1e-4, 9e-5]
        self.eval_enabled = False
        self._active_calls = 0
        self._active_polls = active_polls
        self.trainer = types.SimpleNamespace(training_progress = _Progress())

    def is_training_active(self):
        self._active_calls += 1
        return self._active_calls <= self._active_polls


class _FakeRequest:
    headers = {}

    async def is_disconnected(self):
        return False


class _DisconnectedRequest:
    headers = {}

    async def is_disconnected(self):
        return True


def _collect_events(response, timeout = 15):
    async def _drain():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

    return asyncio.run(asyncio.wait_for(_drain(), timeout))


def _progress_payloads(raw):
    payloads = []
    for block in raw.split("\n\n"):
        lines = block.strip().splitlines()
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        if data:
            payloads.append(json.loads(data))
    return payloads


def test_stream_reports_live_step_with_null_loss_during_nan(monkeypatch):
    backend = _FakeBackend(active_polls = 2)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    response = asyncio.run(rt.stream_training_progress(_FakeRequest(), current_subject = "tester"))
    raw = _collect_events(response)
    payloads = _progress_payloads(raw)
    assert payloads, f"no SSE payloads parsed from: {raw!r}"

    live = [p for p in payloads if p.get("step") == 5]
    assert live, (
        "stream never advanced to the live progress step during the NaN "
        f"stretch; steps seen: {[p.get('step') for p in payloads]}"
    )
    assert live[0]["loss"] is None
    # The stale finite pair must not be re-emitted as the latest progress.
    stale = [p for p in payloads if p.get("step") == 2 and p.get("loss") == 1.5]
    assert not stale


def test_inactive_stream_completes_with_live_step_and_null_loss(monkeypatch):
    # Fresh connection after the run already ended during a NaN stretch: the
    # immediate complete event must not replay the stale finite pair either.
    backend = _FakeBackend(active_polls = 0)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    response = asyncio.run(rt.stream_training_progress(_FakeRequest(), current_subject = "tester"))
    payloads = _progress_payloads(_collect_events(response))
    final = payloads[-1]
    assert final["step"] == 5
    assert final["loss"] is None


def test_disconnect_while_active_does_not_emit_complete(monkeypatch):
    # Client drops mid-run: the stream must end without a terminal "complete"
    # frame, which a buffered/proxy consumer could otherwise read as a finished
    # run while training is still active.
    backend = _FakeBackend(active_polls = 5)
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    response = asyncio.run(
        rt.stream_training_progress(_DisconnectedRequest(), current_subject = "tester")
    )
    raw = _collect_events(response)
    assert "event: complete" not in raw


def test_stream_uses_finite_history_when_progress_in_sync(monkeypatch):
    backend = _FakeBackend(active_polls = 2)
    # Live progress agrees with the history tail: normal finite behavior.
    backend.trainer.training_progress.step = 2
    backend.trainer.training_progress.loss = 1.5
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    response = asyncio.run(rt.stream_training_progress(_FakeRequest(), current_subject = "tester"))
    payloads = _progress_payloads(_collect_events(response))
    finite = [p for p in payloads if p.get("step") == 2]
    assert finite and finite[0]["loss"] == 1.5
