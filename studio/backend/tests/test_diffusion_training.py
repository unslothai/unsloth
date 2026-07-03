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


@pytest.fixture(autouse = True)
def _isolated_runs_dir(monkeypatch, tmp_path):
    """Terminal service events persist a run record; point the runs dir at tmp so tests
    never write into a real studio home. Yields the dir for the history tests."""
    import core.training.diffusion_training_service as dts

    d = tmp_path / "runs" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(dts, "_runs_dir", lambda: d)
    yield d


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


def test_terminal_events_clear_model_load_flag():
    # A stop or error during model load emits complete/error WITHOUT a preceding
    # model_load_completed, so the terminal update must reset in_model_load or the
    # client shows a stale loading indicator after the job ended.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event({"type": "model_load_started"})
    assert svc.status()["in_model_load"] is True
    svc._apply_event({"type": "complete", "stopped": True})
    assert svc.status()["in_model_load"] is False and svc.status()["status"] == "stopped"

    svc2 = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc2._apply_event({"type": "model_load_started"})
    svc2._apply_event({"type": "error", "message": "load failed"})
    assert svc2.status()["in_model_load"] is False and svc2.status()["status"] == "error"


# ── route wiring (mocked service) ─────────────────────────────────────────────
class _FakeService:
    def __init__(self):
        self._running = False
        self.started_with = None
        self.stopped_with_save = None
        # Extra keys merged into status() so a test can inject metric history / perf fields.
        self.status_extra: dict = {}

    def start(self, config):
        self.started_with = config
        self._running = True
        return "job-123"

    def stop(self, save = True):
        self.stopped_with_save = save
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
            **self.status_extra,
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


def test_route_start_forwards_num_epochs(client):
    # Epochs mode: the frontend omits train_steps and sends num_epochs; it must reach the
    # service so the trainer can resolve it against the dataset size.
    body = {k: v for k, v in _BODY.items() if k != "train_steps"}
    r = client.post("/api/train/diffusion/start", json = {**body, "num_epochs": 8})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["num_epochs"] == 8


def test_request_model_num_epochs_bounds():
    # The request schema mirrors DiffusionLoraConfig's 0..1000 num_epochs range.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    base = {"base_model": "b", "data_dir": "d", "output_dir": "o"}
    assert DiffusionTrainingStartRequest(**base).num_epochs == 0  # default = use train_steps
    assert DiffusionTrainingStartRequest(**base, num_epochs = 1000).num_epochs == 1000
    for bad in (-1, 1001):
        with pytest.raises(ValidationError):
            DiffusionTrainingStartRequest(**base, num_epochs = bad)


def test_request_model_base_precision_accepts_mxfp8():
    # The base_precision Literal now includes mxfp8 (the DiT dense speed mode); a bogus
    # mode is still rejected.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    base = {"base_model": "b", "data_dir": "d", "output_dir": "o"}
    assert DiffusionTrainingStartRequest(**base).base_precision == "nf4"  # default
    assert DiffusionTrainingStartRequest(**base, base_precision = "mxfp8").base_precision == "mxfp8"
    with pytest.raises(ValidationError):
        DiffusionTrainingStartRequest(**base, base_precision = "bogus")


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


# ── /diffusion/info + /diffusion/dataset (dataset discovery + upload) ─────────
@pytest.fixture
def dataset_roots(client, monkeypatch, tmp_path):
    # The endpoints import these lazily per-request, so patching the package attr works.
    import utils.paths as up

    ds_root = tmp_path / "assets" / "datasets"
    out_root = tmp_path / "outputs"
    ds_root.mkdir(parents = True)
    out_root.mkdir(parents = True)
    monkeypatch.setattr(up, "datasets_root", lambda: ds_root)
    monkeypatch.setattr(up, "outputs_root", lambda: out_root)
    return ds_root, out_root


def test_diffusion_info_lists_image_dataset_folders(client, dataset_roots):
    ds_root, out_root = dataset_roots
    good = ds_root / "cat-photos"
    good.mkdir()
    (good / "a.png").write_bytes(b"x")
    (good / "b.jpg").write_bytes(b"x")
    (good / "a.txt").write_text("a cat")
    (ds_root / "empty-dir").mkdir()  # no images -> not a dataset
    (ds_root / "stray.txt").write_text("not a folder")

    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["datasets_root"] == str(ds_root)
    assert body["outputs_root"] == str(out_root)
    assert [d["name"] for d in body["datasets"]] == ["cat-photos"]
    assert body["datasets"][0]["image_count"] == 2
    assert body["datasets"][0]["caption_count"] == 1


def test_diffusion_dataset_upload_accumulates(client, dataset_roots):
    ds_root, _ = dataset_roots
    files = [
        ("files", ("a.png", b"png-bytes", "image/png")),
        ("files", ("b.JPG", b"jpg-bytes", "image/jpeg")),
        ("files", ("a.txt", b"a caption", "text/plain")),
    ]
    r = client.post("/api/train/diffusion/dataset", data = {"name": "my style"}, files = files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "my style"
    assert body["uploaded"] == 3
    assert body["image_count"] == 2
    assert body["caption_count"] == 1
    assert (ds_root / "my style" / "a.png").read_bytes() == b"png-bytes"

    # A second batch into the same name accumulates (large sets arrive in chunks).
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "my style"},
        files = [("files", ("c.webp", b"w", "image/webp"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["uploaded"] == 1
    assert r.json()["image_count"] == 3


def test_diffusion_dataset_upload_rejects_traversal_names(client, dataset_roots):
    for bad in ("../evil", "a/b", ".hidden", " "):
        r = client.post(
            "/api/train/diffusion/dataset",
            data = {"name": bad},
            files = [("files", ("a.png", b"x", "image/png"))],
        )
        assert r.status_code == 400, f"{bad!r}: {r.status_code}"


def test_diffusion_dataset_upload_rejects_unsupported_files(client, dataset_roots):
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "ok-name"},
        files = [("files", ("weights.exe", b"mz", "application/octet-stream"))],
    )
    assert r.status_code == 400
    assert "Unsupported file" in r.json()["detail"]


def test_route_start_refuses_non_sdxl_base_without_freeing_gpu(client, monkeypatch):
    # A doomed start (non-SDXL base) must 400 BEFORE resident GPU workloads are freed,
    # so a bad pick never unloads the user's working chat/Images model.
    import routes.training as tr

    freed = []
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append(1))
    r = client.post(
        "/api/train/diffusion/start", json = {**_BODY, "base_model": "unsloth/FLUX.1-dev-GGUF"}
    )
    assert r.status_code == 400
    assert "SDXL" in r.json()["detail"]
    assert freed == []
    assert client._fake.started_with is None


# ── metric history + perf/family fields (PR A platform) ──────────────────────
def test_apply_event_records_metric_history_and_perf():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event(
        {
            "type": "progress",
            "step": 1,
            "total_steps": 10,
            "loss": 0.5,
            "learning_rate": 1e-4,
            "samples_per_second": 3.2,
            "peak_memory_gb": 7.1,
        }
    )
    svc._apply_event(
        {"type": "progress", "step": 2, "total_steps": 10, "loss": 0.4, "learning_rate": 9e-5}
    )
    st = svc.status()
    assert st["metric_steps"] == [1, 2]
    assert st["metric_loss"] == [0.5, 0.4]
    assert st["metric_lr"] == [1e-4, 9e-5]
    assert st["samples_per_second"] == 3.2
    assert st["peak_memory_gb"] == 7.1


def test_apply_event_metric_history_skips_bad_points():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    # step 0 (warmup / no real step), a None loss, and a NaN loss must all be skipped.
    svc._apply_event({"type": "progress", "step": 0, "loss": 0.9, "learning_rate": 1e-4})
    svc._apply_event({"type": "progress", "step": 1, "loss": None, "learning_rate": 1e-4})
    svc._apply_event({"type": "progress", "step": 2, "loss": float("nan"), "learning_rate": 1e-4})
    svc._apply_event({"type": "progress", "step": 3, "loss": 0.3, "learning_rate": None})
    st = svc.status()
    assert st["metric_steps"] == [3]
    assert st["metric_loss"] == [0.3]
    assert st["metric_lr"] == [None]  # lr None is retained so the series stays index-aligned


def test_metric_history_decimates_at_cap():
    from core.training import diffusion_training_service as svc_mod

    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event({"type": "model_load_started"})  # initialise running state
    n = svc_mod._METRIC_CAP + 50
    for i in range(1, n + 1):
        svc._apply_event({"type": "progress", "step": i, "loss": 1.0 / i, "learning_rate": 1e-4})
    st = svc.status()
    # Never exceeds the cap, and stays a valid paired history with matching lengths.
    assert len(st["metric_steps"]) <= svc_mod._METRIC_CAP
    assert len(st["metric_steps"]) == len(st["metric_loss"]) == len(st["metric_lr"])
    # Decimation keeps the curve monotonic in step (still increasing, just sparser).
    assert st["metric_steps"] == sorted(st["metric_steps"])
    assert st["metric_steps"][-1] == n  # the latest point is always retained


def test_complete_event_records_family_and_catalog():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event(
        {
            "type": "complete",
            "output_dir": "/o",
            "lora_path": "/o/w.safetensors",
            "catalog_path": "/loras/w.safetensors",
            "family": "sdxl",
            "base_model": "b",
        }
    )
    st = svc.status()
    assert st["status"] == "completed"
    assert st["family"] == "sdxl"
    assert st["base_model"] == "b"
    assert st["catalog_path"] == "/loras/w.safetensors"


def test_status_route_nests_metric_history(client):
    # The status route folds the service's flat arrays into a nested metric_history object.
    client._fake.status_extra = {
        "metric_steps": [1, 2],
        "metric_loss": [0.5, 0.4],
        "metric_lr": [1e-4, 9e-5],
        "family": "sdxl",
        "samples_per_second": 2.0,
        "peak_memory_gb": 6.0,
    }
    r = client.get("/api/train/diffusion/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metric_history"]["steps"] == [1, 2]
    assert body["metric_history"]["loss"] == [0.5, 0.4]
    assert body["family"] == "sdxl"
    assert body["samples_per_second"] == 2.0


# ── /diffusion/info families + gated-repo preflight (PR B) ──────────────────────
def test_info_lists_trainable_families(client):
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    families = {f["name"]: f for f in r.json()["families"]}
    for fam in ("sdxl", "flux.1", "qwen-image", "z-image"):
        assert fam in families
    assert families["flux.1"]["default_base"] == "black-forest-labs/FLUX.1-dev"
    assert families["z-image"]["defaults"]["resolution"] == 768


def test_start_gated_base_without_access_is_400_and_keeps_gpu(client, monkeypatch):
    # A gated FLUX base with no valid token must 400 from the HEAD preflight BEFORE the GPU
    # residents are freed, so a doomed start never evicts the user's loaded model.
    import urllib.error
    import urllib.request

    import routes.training as tr

    freed: list[int] = []
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append(1))

    def _fake_urlopen(req, timeout = None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "base_model": "black-forest-labs/FLUX.1-dev"},
    )
    assert r.status_code == 400
    assert "gated" in r.json()["detail"].lower()
    assert freed == []
    assert client._fake.started_with is None


def test_start_ungated_base_preflight_is_noop(client, monkeypatch):
    # A reachable base (HEAD 200) proceeds to start normally.
    import urllib.request

    import routes.training as tr

    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout = None: object())
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "base_model": "black-forest-labs/FLUX.1-dev"},
    )
    assert r.status_code == 200, r.text
    assert client._fake.started_with["base_model"] == "black-forest-labs/FLUX.1-dev"


# ── persisted run history ──────────────────────────────────────────────────────
def test_run_record_persisted_on_complete(_isolated_runs_dir):
    # A completed run writes one JSON record: summary + scrubbed config + metric logs.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    job_id = svc.start({**_CFG, "model_family": "z-image", "hf_token": "SECRET"})
    _wait_status(svc, "completed")
    # The pump persists right after the terminal event; give the thread a beat.
    time.sleep(0.1)

    import json

    rec = json.loads((_isolated_runs_dir / f"{job_id}.json").read_text())
    assert rec["job_id"] == job_id
    assert rec["status"] == "completed"
    assert rec["saved"] is True
    assert rec["adapter"] == "out"  # basename of /tmp/out
    assert rec["family"] == "z-image"  # falls back to the config's model_family
    assert rec["step"] == 2 and rec["total_steps"] == 2
    assert rec["avg_loss"] == 0.45
    assert rec["metric_history"]["steps"] == [1, 2]
    assert rec["metric_history"]["loss"] == [0.5, 0.4]
    # Secrets never land on disk.
    assert "hf_token" not in rec["config"]
    assert rec["config"]["model_family"] == "z-image"


def test_run_record_no_save_stop_marks_unsaved(_isolated_runs_dir):
    # A cancel (stop without save) persists too, flagged as not saved.
    def _cancel_target(*, event_queue, stop_queue, config):
        event_queue.put({"type": "model_load_completed"})
        stop_queue.get(timeout = 5.0)
        event_queue.put(
            {"type": "complete", "output_dir": None, "lora_path": None, "stopped": True}
        )

    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _cancel_target)
    job_id = svc.start(dict(_CFG))
    _wait_status(svc, "running")
    svc.stop(save = False)
    _wait_status(svc, "stopped")
    time.sleep(0.1)

    import json

    rec = json.loads((_isolated_runs_dir / f"{job_id}.json").read_text())
    assert rec["status"] == "stopped"
    assert rec["saved"] is False and rec["lora_path"] is None


def test_runs_endpoints_list_and_detail(client, _isolated_runs_dir):
    # Seed two records directly (the endpoints read the persisted files, not the service).
    import json
    import os

    a = {
        "job_id": "a" * 32,
        "status": "completed",
        "adapter": "first",
        "saved": True,
        "step": 10,
        "total_steps": 10,
        "avg_loss": 0.4,
        "config": {"train_steps": 10},
        "metric_history": {"steps": [1], "loss": [0.4], "lr": [1e-4], "grad_norm": [0.2]},
    }
    b = {
        "job_id": "b" * 32,
        "status": "stopped",
        "adapter": "second",
        "saved": False,
        "step": 3,
        "total_steps": 10,
        "avg_loss": 0.6,
        "config": {"train_steps": 10},
        "metric_history": {"steps": [1], "loss": [0.6], "lr": [1e-4], "grad_norm": [0.3]},
    }
    pa = _isolated_runs_dir / f"{a['job_id']}.json"
    pb = _isolated_runs_dir / f"{b['job_id']}.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    os.utime(pa, (1000, 1000))
    os.utime(pb, (2000, 2000))  # b is newer -> listed first

    r = client.get("/api/train/diffusion/runs")
    assert r.status_code == 200, r.text
    runs = r.json()["runs"]
    assert [x["adapter"] for x in runs] == ["second", "first"]
    # Summaries stay light: no config / metric logs.
    assert "config" not in runs[0] and "metric_history" not in runs[0]

    r = client.get(f"/api/train/diffusion/runs/{a['job_id']}")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["adapter"] == "first"
    assert detail["metric_history"]["grad_norm"] == [0.2]
    assert detail["config"] == {"train_steps": 10}

    # Unknown and malformed ids 404 (malformed also covers path traversal).
    assert client.get(f"/api/train/diffusion/runs/{'c' * 32}").status_code == 404
    assert client.get("/api/train/diffusion/runs/not-a-job-id").status_code == 404
