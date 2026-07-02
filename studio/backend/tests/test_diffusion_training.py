# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the diffusion LoRA training service + routes.

The service's subprocess context and target are injected with in-thread fakes, so the
full start -> event-pump -> status -> complete path is exercised without real
multiprocessing or torch. The routes are hit with the FastAPI TestClient and a mocked
service, so wiring / validation / error mapping are covered without a GPU.
"""

from __future__ import annotations

import queue as _queue
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training.diffusion_training_service import DiffusionTrainingService
from routes.training import router as training_router


# ── fake spawn context (runs the "process" target on a thread) ────────────────
class _FakeQueue:
    def __init__(self) -> None:
        self._q: _queue.Queue = _queue.Queue()

    def put(self, x):
        self._q.put(x)

    def get(self, timeout = None):
        return self._q.get(timeout = timeout)  # raises queue.Empty on timeout

    def get_nowait(self):
        return self._q.get_nowait()

    def empty(self):
        return self._q.empty()


class _FakeProc:
    def __init__(self, target, kwargs, daemon):
        self._target = target
        self._kwargs = kwargs
        self._thread: threading.Thread | None = None
        self.pid = 4321

    def start(self):
        self._thread = threading.Thread(target = self._target, kwargs = self._kwargs, daemon = True)
        self._thread.start()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class _FakeCtx:
    def Queue(self):
        return _FakeQueue()

    def Process(self, target, kwargs, daemon):
        return _FakeProc(target, kwargs, daemon)


def _happy_target(*, event_queue, stop_queue, config):
    event_queue.put({"type": "model_load_started", "num_images": 3})
    event_queue.put({"type": "model_load_completed"})
    event_queue.put(
        {
            "type": "progress",
            "step": 1,
            "total_steps": 2,
            "loss": 0.5,
            "avg_loss": 0.5,
            "learning_rate": 1e-4,
        }
    )
    event_queue.put(
        {
            "type": "progress",
            "step": 2,
            "total_steps": 2,
            "loss": 0.4,
            "avg_loss": 0.45,
            "learning_rate": 1e-4,
        }
    )
    event_queue.put(
        {
            "type": "complete",
            "output_dir": config["output_dir"],
            "lora_path": config["output_dir"] + "/pytorch_lora_weights.safetensors",
            "stopped": False,
        }
    )


def _stoppable_target(*, event_queue, stop_queue, config):
    event_queue.put({"type": "model_load_completed"})
    stop_queue.get(timeout = 5.0)  # block until stop() signals
    event_queue.put(
        {"type": "complete", "output_dir": config["output_dir"], "lora_path": "x", "stopped": True}
    )


def _crashing_target(*, event_queue, stop_queue, config):
    event_queue.put({"type": "model_load_started"})
    # Exits without a terminal event -> the pump must mark it as an error.


_CFG = {"base_model": "b", "data_dir": "d", "output_dir": "/tmp/out", "train_steps": 2}


def _wait_status(
    svc,
    *terminal,
    timeout = 3.0,
):
    end = time.time() + timeout
    while time.time() < end:
        st = svc.status()
        if st["status"] in terminal:
            return st
        time.sleep(0.02)
    return svc.status()


def test_service_happy_path():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    job_id = svc.start(dict(_CFG))
    assert job_id
    st = _wait_status(svc, "completed")
    assert st["status"] == "completed"
    assert st["step"] == 2 and st["total_steps"] == 2
    assert st["num_images"] == 3
    assert st["loss"] == 0.4 and st["avg_loss"] == 0.45
    assert st["lora_path"].endswith("pytorch_lora_weights.safetensors")
    assert st["active"] is False


def test_service_rejects_bad_config_before_spawn():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    with pytest.raises(ValueError):
        svc.start({**_CFG, "train_steps": 0})
    # Nothing was spawned; still idle.
    assert svc.status()["status"] == "idle"


def test_service_rejects_second_concurrent_job():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _stoppable_target)
    svc.start(dict(_CFG))
    _wait_status(svc, "running")
    with pytest.raises(RuntimeError):
        svc.start(dict(_CFG))
    assert svc.stop() is True
    _wait_status(svc, "stopped")


def test_service_stop_marks_stopped():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _stoppable_target)
    svc.start(dict(_CFG))
    _wait_status(svc, "running")
    assert svc.stop() is True
    st = _wait_status(svc, "stopped")
    assert st["status"] == "stopped"
    assert st["active"] is False
    # Stopping again when idle is a no-op.
    assert svc.stop() is False


def test_service_crash_without_terminal_event_is_error():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _crashing_target)
    svc.start(dict(_CFG))
    st = _wait_status(svc, "error")
    assert st["status"] == "error"
    assert "unexpectedly" in st["message"]


def test_apply_event_transitions():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event({"type": "model_load_started", "num_images": 5})
    assert svc.status()["in_model_load"] is True and svc.status()["num_images"] == 5
    svc._apply_event({"type": "model_load_completed"})
    assert svc.status()["in_model_load"] is False
    svc._apply_event({"type": "error", "message": "boom"})
    assert svc.status()["status"] == "error" and svc.status()["message"] == "boom"


# ── route wiring (mocked service) ─────────────────────────────────────────────
class _FakeService:
    def __init__(self):
        self._running = False
        self.started_with = None

    def start(self, config):
        self.started_with = config
        self._running = True
        return "job-123"

    def stop(self):
        was = self._running
        self._running = False
        return was

    def status(self):
        return {
            "active": self._running,
            "job_id": "job-123" if self._running else None,
            "status": "running" if self._running else "idle",
            "message": "",
            "step": 1,
            "total_steps": 2,
            "loss": 0.5,
            "avg_loss": 0.5,
            "learning_rate": 1e-4,
            "num_images": 3,
            "in_model_load": False,
            "output_dir": None,
            "lora_path": None,
            "started_at": None,
            "updated_at": None,
        }


class _FakeLLMBackend:
    def __init__(self, active = False):
        self._active = active

    def is_training_active(self):
        return self._active


@pytest.fixture
def client(monkeypatch):
    fake = _FakeService()
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: fake
    )
    # Neutralize the LLM interlock + GPU-free for the wiring tests (their own tests below
    # exercise those behaviors). The route imports get_training_backend at module scope.
    import routes.training as tr

    monkeypatch.setattr(tr, "get_training_backend", lambda: _FakeLLMBackend(active = False))
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    c = TestClient(app)
    c._fake = fake  # type: ignore[attr-defined]
    return c


# Studio-relative paths: the route resolves/contains them before spawn.
_BODY = {
    "base_model": "stabilityai/sdxl-turbo",
    "data_dir": "uploads/my-images",
    "output_dir": "my-lora-run",
    "train_steps": 10,
}


def test_route_start_ok(client):
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    assert r.json() == {"job_id": "job-123", "status": "running"}
    assert client._fake.started_with["base_model"] == "stabilityai/sdxl-turbo"
    # Paths were resolved to absolute Studio-contained locations before spawn.
    from pathlib import Path

    assert Path(client._fake.started_with["data_dir"]).is_absolute()
    assert Path(client._fake.started_with["output_dir"]).is_absolute()


def test_route_start_forwards_extra_training_knobs(client):
    # max_grad_norm and lora_target_modules must reach the service, not be silently dropped.
    body = {**_BODY, "max_grad_norm": 0.5, "lora_target_modules": ["to_q", "to_v"]}
    r = client.post("/api/train/diffusion/start", json = body)
    assert r.status_code == 200, r.text
    assert client._fake.started_with["max_grad_norm"] == 0.5
    assert client._fake.started_with["lora_target_modules"] == ["to_q", "to_v"]


def test_route_start_rejects_uncontained_paths(client):
    # An absolute path outside the Studio dataset roots is a 400, not silently accepted.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "data_dir": "/etc"})
    assert r.status_code == 400


def test_route_start_blocked_by_active_llm_training(client, monkeypatch):
    import routes.training as tr

    monkeypatch.setattr(tr, "get_training_backend", lambda: _FakeLLMBackend(active = True))
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 409
    assert "LLM training" in r.json()["detail"]


def test_route_start_missing_required_is_422(client):
    r = client.post(
        "/api/train/diffusion/start", json = {"base_model": "x"}
    )  # no data_dir/output_dir
    assert r.status_code == 422


def test_route_start_bad_config_maps_to_400(client, monkeypatch):
    def _raise(_cfg):
        raise ValueError("resolution must be a multiple of 8")

    client._fake.start = _raise  # type: ignore[assignment]
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 400
    assert "multiple of 8" in r.json()["detail"]


def test_route_start_conflict_maps_to_409(client):
    def _raise(_cfg):
        raise RuntimeError("A diffusion training job is already running.")

    client._fake.start = _raise  # type: ignore[assignment]
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 409


def test_route_status_and_stop(client):
    client.post("/api/train/diffusion/start", json = _BODY)
    s = client.get("/api/train/diffusion/status")
    assert s.status_code == 200 and s.json()["status"] == "running"
    st = client.post("/api/train/diffusion/stop")
    assert st.status_code == 200 and st.json()["status"] == "stopping"
    # After stopping, a stop with nothing running reports idle.
    st2 = client.post("/api/train/diffusion/stop")
    assert st2.json()["status"] == "idle"


def test_service_restart_after_completion():
    # A finished job's pump is joined OUTSIDE the lock (it needs the lock for its
    # final state writes), so a second start neither stalls nor deadlocks.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc.start(dict(_CFG))
    _wait_status(svc, "completed")
    t0 = time.time()
    job2 = svc.start(dict(_CFG))
    assert job2
    assert time.time() - t0 < 4.0  # no 5s join-under-lock stall
    st = _wait_status(svc, "completed")
    assert st["status"] == "completed"


def test_stale_pump_events_cannot_corrupt_new_job():
    # An event carrying a superseded job's proc identity must be dropped, so a
    # straggler pump can never overwrite the state of a newly started job.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc.start(dict(_CFG))
    _wait_status(svc, "completed")
    current = svc._proc
    svc._apply_event({"type": "error", "message": "stale boom"}, proc = object())
    assert svc.status()["message"] != "stale boom"
    # The current job's events still apply.
    svc._apply_event({"type": "progress", "step": 9}, proc = current)
    assert svc.status()["step"] == 9
