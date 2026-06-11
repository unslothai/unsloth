# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Route-level regression tests for non-finite loss reporting.

/metrics and the SSE stream used to derive "current" values from the
finite-only history arrays, which replayed the last finite loss during
NaN/Inf steps. They must read live progress instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.training import TrainingBackend
import routes.training as training_routes


def _progress_event(step: int, loss: float, lr: float = 1e-4) -> dict:
    return {
        "type": "progress",
        "step": step,
        "loss": loss,
        "learning_rate": lr,
        "epoch": 0.0,
        "total_steps": 100,
    }


def _finish_run(b: TrainingBackend) -> TrainingBackend:
    """Mark the run as finished so SSE takes the final-state path."""
    b._progress.is_training = False
    b._progress.is_completed = True
    return b


def _backend_after_nan() -> TrainingBackend:
    b = TrainingBackend()
    b._handle_event(_progress_event(step=1, loss=0.97))
    b._handle_event(_progress_event(step=2, loss=float("nan")))
    return b


class _FakeRequest:
    headers: dict = {}


def _collect_sse_events(response) -> list[tuple[str, dict]]:
    """Drain a StreamingResponse of SSE messages into (event, payload) pairs."""

    async def drain():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(chunks)

    raw = asyncio.run(drain())
    events = []
    for block in raw.split("\n\n"):
        event_name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_name is not None and data is not None:
            events.append((event_name, data))
    return events


class TestNonfiniteLossRoutes:
    def test_metrics_reports_null_loss_and_current_step_after_nan(self, monkeypatch):
        b = _backend_after_nan()
        monkeypatch.setattr(training_routes, "get_training_backend", lambda: b)
        resp = asyncio.run(
            training_routes.get_training_metrics(current_subject="test")
        )
        # Live progress, not the stale finite history point
        assert resp.current_step == 2
        assert resp.current_loss is None
        # Chart history stays finite-only
        assert resp.loss_history == [0.97]
        assert resp.step_history == [1]

    def test_metrics_falls_back_to_history_when_no_progress(self, monkeypatch):
        b = TrainingBackend()
        monkeypatch.setattr(training_routes, "get_training_backend", lambda: b)
        resp = asyncio.run(
            training_routes.get_training_metrics(current_subject="test")
        )
        assert resp.current_step is None
        assert resp.current_loss is None

    def test_sse_complete_event_reports_nan_step_with_null_loss(self, monkeypatch):
        b = _finish_run(_backend_after_nan())
        monkeypatch.setattr(training_routes, "get_training_backend", lambda: b)
        resp = asyncio.run(
            training_routes.stream_training_progress(
                _FakeRequest(), current_subject="test"
            )
        )
        events = _collect_sse_events(resp)
        completes = [payload for name, payload in events if name == "complete"]
        assert len(completes) == 1
        # The NaN step is surfaced, not the last finite one
        assert completes[0]["step"] == 2
        assert completes[0]["loss"] is None

    def test_sse_complete_event_reports_finite_loss_normally(self, monkeypatch):
        b = TrainingBackend()
        b._handle_event(_progress_event(step=1, loss=0.97))
        _finish_run(b)
        monkeypatch.setattr(training_routes, "get_training_backend", lambda: b)
        resp = asyncio.run(
            training_routes.stream_training_progress(
                _FakeRequest(), current_subject="test"
            )
        )
        events = _collect_sse_events(resp)
        completes = [payload for name, payload in events if name == "complete"]
        assert len(completes) == 1
        assert completes[0]["step"] == 1
        assert completes[0]["loss"] == pytest.approx(0.97)
