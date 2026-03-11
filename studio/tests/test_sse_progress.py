# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Tests for the SSE training progress endpoint and status fallback.

Validates:
  - SSE spec compliance: `retry:`, `id:`, `event:` fields
  - Named event types: progress, heartbeat, complete, error
  - Last-Event-ID reconnection and history replay
  - /status metric_history fallback (Option B)

All tests mock the training backend and bypass auth.
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
import re

import pytest

# ── Path setup ────────────────────────────────────────────────────
# Add backend root so bare `from routes…`, `from models…` etc. resolve.
_backend_root = Path(__file__).resolve().parent.parent / "backend"
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from fastapi.testclient import TestClient
from main import app
from auth.authentication import get_current_subject


# ── Fixtures ──────────────────────────────────────────────────────


def _bypass_auth():
    """Dependency override that skips real JWT validation."""
    return "test-user"


def _make_mock_backend(
    *,
    is_active: bool = False,
    step_history: list | None = None,
    loss_history: list | None = None,
    lr_history: list | None = None,
    total_steps: int = 100,
    epoch: int | None = 1,
    job_id: str = "job_test_001",
):
    """Build a lightweight mock that quacks like TrainingBackend."""
    backend = MagicMock()
    backend.current_job_id = job_id
    backend.step_history = step_history or []
    backend.loss_history = loss_history or []
    backend.lr_history = lr_history or []
    backend.is_training_active.return_value = is_active
    backend._training_thread = None

    # trainer.training_progress / get_training_progress()
    tp = MagicMock()
    tp.total_steps = total_steps
    tp.epoch = epoch
    tp.step = step_history[-1] if step_history else 0
    tp.loss = loss_history[-1] if loss_history else 0.0
    tp.learning_rate = lr_history[-1] if lr_history else 0.0
    tp.status_message = "Training..."
    tp.error = None
    tp.is_completed = not is_active and bool(step_history)

    backend.trainer = MagicMock()
    backend.trainer.training_progress = tp
    backend.trainer.get_training_progress.return_value = tp

    return backend


@pytest.fixture()
def client():
    """TestClient with auth bypassed."""
    app.dependency_overrides[get_current_subject] = _bypass_auth
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── SSE Parsing Helpers ───────────────────────────────────────────


def parse_sse_events(raw: str) -> list[dict]:
    """
    Parse raw SSE text into a list of event dicts.

    Each dict has optional keys: 'id', 'event', 'data', 'retry'.
    """
    events: list[dict] = []
    current: dict = {}

    for line in raw.split("\n"):
        if line.startswith("retry:"):
            # retry is a standalone directive, not part of a normal event
            events.append({"retry": line.split(":", 1)[1].strip()})
            continue
        if line.startswith("id:"):
            current["id"] = line.split(":", 1)[1].strip()
        elif line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
        elif line == "" and current:
            events.append(current)
            current = {}

    if current:
        events.append(current)
    return events


# =====================================================================
# Option A — /api/train/progress  (SSE)
# =====================================================================


class TestSSERetryDirective:
    """The first thing the stream emits must be `retry: 3000`."""

    def test_retry_is_first_event(self, client: TestClient):
        mock_backend = _make_mock_backend(is_active = False)
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = parse_sse_events(resp.text)
        assert len(events) >= 1
        assert events[0] == {"retry": "3000"}


class TestSSEEventFields:
    """Every non-retry event must include `id:`, `event:`, and `data:` fields."""

    def test_events_have_id_and_event_type(self, client: TestClient):
        mock_backend = _make_mock_backend(
            is_active = False,
            step_history = [1, 2, 3],
            loss_history = [2.0, 1.5, 1.0],
            lr_history = [1e-4, 1e-4, 1e-4],
            total_steps = 3,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        events = parse_sse_events(resp.text)
        data_events = [e for e in events if "data" in e]

        assert len(data_events) >= 1
        for evt in data_events:
            assert "id" in evt, f"Missing `id:` field in event: {evt}"
            assert "event" in evt, f"Missing `event:` field in event: {evt}"
            assert "data" in evt


class TestSSENamedEventTypes:
    """Events use the correct named types: progress, complete, heartbeat, error."""

    def test_idle_sends_progress_then_complete(self, client: TestClient):
        mock_backend = _make_mock_backend(
            is_active = False,
            step_history = [10],
            loss_history = [1.5],
            lr_history = [1e-4],
            total_steps = 10,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        events = parse_sse_events(resp.text)
        data_events = [e for e in events if "event" in e and e.get("event") != "retry"]

        event_types = [e["event"] for e in data_events]
        assert "progress" in event_types
        assert "complete" in event_types

    def test_no_history_sends_complete(self, client: TestClient):
        mock_backend = _make_mock_backend(is_active = False)
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        events = parse_sse_events(resp.text)
        data_events = [e for e in events if "event" in e]
        assert any(e["event"] == "complete" for e in data_events)


class TestSSELastEventIDResume:
    """When `Last-Event-ID` header is sent, the server replays missed steps."""

    def test_replays_steps_after_last_event_id(self, client: TestClient):
        mock_backend = _make_mock_backend(
            is_active = False,
            step_history = [1, 2, 3, 4, 5],
            loss_history = [2.5, 2.0, 1.5, 1.2, 1.0],
            lr_history = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
            total_steps = 5,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get(
                "/api/train/progress",
                headers = {"Last-Event-ID": "2"},
            )

        events = parse_sse_events(resp.text)
        # Filter to progress events (replayed ones)
        progress_events = [e for e in events if e.get("event") == "progress"]

        # Steps 3, 4, 5 should have been replayed
        replayed_ids = [int(e["id"]) for e in progress_events]
        assert 3 in replayed_ids
        assert 4 in replayed_ids
        assert 5 in replayed_ids
        # Steps 1, 2 should NOT be replayed
        assert 1 not in replayed_ids
        assert 2 not in replayed_ids

    def test_no_replay_without_header(self, client: TestClient):
        """Without Last-Event-ID, should start fresh (initial progress event)."""
        mock_backend = _make_mock_backend(
            is_active = False,
            step_history = [1, 2, 3],
            loss_history = [2.0, 1.5, 1.0],
            lr_history = [1e-4, 1e-4, 1e-4],
            total_steps = 3,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        events = parse_sse_events(resp.text)
        progress_events = [e for e in events if e.get("event") == "progress"]

        # Should have initial step=0 progress event
        assert any(e.get("id") == "0" for e in progress_events)

    def test_invalid_last_event_id_treated_as_fresh(self, client: TestClient):
        """Non-integer Last-Event-ID should be ignored gracefully."""
        mock_backend = _make_mock_backend(is_active = False)
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get(
                "/api/train/progress",
                headers = {"Last-Event-ID": "not-a-number"},
            )

        assert resp.status_code == 200
        events = parse_sse_events(resp.text)
        # Should still work — treated as a fresh connection
        assert any(
            e.get("event") == "progress" or e.get("event") == "complete" for e in events
        )


class TestSSEResponseHeaders:
    """Verify SSE response headers for proxy compatibility."""

    def test_headers(self, client: TestClient):
        mock_backend = _make_mock_backend(is_active = False)
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/progress")

        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"


# =====================================================================
# Option B — /api/train/status  (metric_history fallback)
# =====================================================================


class TestStatusMetricHistory:
    """The /status endpoint returns metric_history for chart recovery."""

    def test_metric_history_populated_when_history_exists(self, client: TestClient):
        mock_backend = _make_mock_backend(
            is_active = True,
            step_history = [1, 2, 3, 4, 5],
            loss_history = [2.5, 2.0, 1.5, 1.2, 1.0],
            lr_history = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
            total_steps = 10,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/status")

        assert resp.status_code == 200
        body = resp.json()

        assert "metric_history" in body
        mh = body["metric_history"]
        assert mh is not None
        assert mh["steps"] == [1, 2, 3, 4, 5]
        assert mh["loss"] == [2.5, 2.0, 1.5, 1.2, 1.0]
        assert mh["lr"] == [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

    def test_metric_history_null_when_no_history(self, client: TestClient):
        mock_backend = _make_mock_backend(is_active = False)
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["metric_history"] is None

    def test_status_still_returns_phase_and_details(self, client: TestClient):
        """Ensure adding metric_history didn't break existing fields."""
        mock_backend = _make_mock_backend(
            is_active = True,
            step_history = [5],
            loss_history = [1.5],
            lr_history = [1e-4],
            total_steps = 100,
        )
        with patch("routes.training.get_training_backend", return_value = mock_backend):
            resp = client.get("/api/train/status")

        body = resp.json()
        assert body["phase"] == "training"
        assert body["is_training_running"] is True
        assert body["job_id"] == "job_test_001"
        assert "details" in body
