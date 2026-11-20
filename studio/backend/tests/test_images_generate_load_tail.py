# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Once an image load has published its pipeline, /images/status reads loaded while the load still waits for its
encoder pins. A generate sent in that tail waits for it instead of getting the "model is loading" 409; a load that
has not published yet is still refused. No GPU: the backend is faked."""

from __future__ import annotations

import sys
from pathlib import Path

import core.inference.diffusion as diffusion_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_routes import client  # noqa: E402,F401 -- the route fixture with a faked backend


def _progress(phase):
    return {
        "phase": phase,
        "bytes_downloaded": 0,
        "bytes_total": 0,
        "fraction": 1.0,
        "error": None,
    }


def _arm(monkeypatch, *, tail, polls_until_ready):
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    calls, reads = [], []

    def _generate(**_kwargs):
        calls.append(len(reads))
        return {"images": [], "seed": 1, "seeds": []}

    def _load_progress():
        reads.append(1)
        return _progress("finalizing" if len(reads) <= polls_until_ready else "ready")

    monkeypatch.setattr(backend, "generate", _generate)
    monkeypatch.setattr(backend, "load_progress", _load_progress)
    monkeypatch.setattr(backend, "load_tail_pending", lambda: tail, raising = False)
    return calls


def test_generate_in_a_published_loads_tail_waits_and_runs(client, monkeypatch):
    import routes.inference as inference_routes

    monkeypatch.setattr(inference_routes, "_IMAGE_LOAD_TAIL_POLL_S", 0.0)
    calls = _arm(monkeypatch, tail = True, polls_until_ready = 3)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 200, resp.text
    # Ran once, after the load reported ready.
    assert len(calls) == 1 and calls[0] > 3


def test_generate_before_the_pipeline_is_published_is_still_refused(client, monkeypatch):
    calls = _arm(monkeypatch, tail = False, polls_until_ready = 10**6)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409, resp.text
    assert "loading" in resp.json()["detail"]
    assert calls == []


def test_a_tail_that_outlasts_the_cap_is_refused(client, monkeypatch):
    import routes.inference as inference_routes

    monkeypatch.setattr(inference_routes, "IMAGE_LOAD_TAIL_WAIT_S", 0.0)
    calls = _arm(monkeypatch, tail = True, polls_until_ready = 10**6)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409, resp.text
    assert calls == []


def test_engine_reports_the_tail_only_after_publishing():
    backend = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    import threading

    backend._load_cancel_lock = threading.Lock()
    backend._state = None
    backend._loading = diffusion_module._LoadingState(repo_id = "r", base_repo = "r")
    assert backend.load_tail_pending() is False
    state = object()
    with backend._load_cancel_lock:
        backend._publish_state_locked(state)
    assert backend._state is state
    assert backend.load_tail_pending() is True
    backend._loading.error = "boom"
    assert backend.load_tail_pending() is False
    backend._loading = None
    assert backend.load_tail_pending() is False


def test_a_synchronous_load_without_a_loading_marker_publishes():
    import threading

    backend = diffusion_module.DiffusionBackend.__new__(diffusion_module.DiffusionBackend)
    backend._load_cancel_lock = threading.Lock()
    backend._state = None
    backend._loading = None
    state = object()
    with backend._load_cancel_lock:
        backend._publish_state_locked(state)
    assert backend._state is state and backend.load_tail_pending() is False


def test_every_image_commit_site_publishes_through_the_helper():
    import inspect

    src = inspect.getsource(diffusion_module.DiffusionBackend)
    # The helper is the one place that assigns a loaded pipeline; both load paths go through it.
    assert src.count("self._state = state") == 1
    assert src.count("self._publish_state_locked(state)") >= 2


def test_a_stop_during_the_tail_wait_cancels_the_run(client, monkeypatch):
    """A Stop sent while the run waits for the load's tail never reaches the engine (no generation ticket yet), so
    the wait itself must notice it: the run is cancelled instead of rendering once the load finishes."""
    import routes.inference as inference_routes

    monkeypatch.setattr(inference_routes, "_IMAGE_LOAD_TAIL_POLL_S", 0.0)
    calls = _arm(monkeypatch, tail = True, polls_until_ready = 4)
    real_tail_only = inference_routes._image_load_tail_only
    seen = []

    def _tail_only_then_stop():
        seen.append(1)
        if len(seen) == 2:
            inference_routes._direct_stops += 1  # what /images/generate/cancel does
        return real_tail_only()

    monkeypatch.setattr(inference_routes, "_image_load_tail_only", _tail_only_then_stop)
    resp = client.post("/api/inference/images/generate", json = {"prompt": "p"})
    assert resp.status_code == 409, resp.text
    assert "cancel" in resp.json()["detail"].lower()
    assert calls == []
