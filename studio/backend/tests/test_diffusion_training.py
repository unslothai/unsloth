# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the diffusion LoRA training service + routes.

The service's subprocess context and target are injected with in-thread fakes, so the
full start -> event-pump -> status -> complete path is exercised without real
multiprocessing or torch. The routes are hit with the FastAPI TestClient and a mocked
service, so wiring / validation / error mapping are covered without a GPU.
"""

from __future__ import annotations

import json
import queue as _queue
import threading
import time

import contextlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
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
    # Exits without a terminal event, so the pump must mark it as an error.


def test_default_target_activates_native_tls_before_diffusion_trainer(monkeypatch):
    """The diffusion service has its own spawn target, outside training/worker.py."""
    import core.training.diffusion_training_service as service

    calls = []
    monkeypatch.setattr("utils.native_tls.activate_native_tls", lambda: calls.append("native_tls"))

    def _run_without_native_path_secret(target, **kwargs):
        calls.append("native_path_secret")
        assert target is service._run_diffusion_child
        assert kwargs == {
            "event_queue": "events",
            "stop_queue": "stop",
            "config": {"base_model": "example/model"},
        }

    monkeypatch.setattr(
        "utils.native_path_leases.run_without_native_path_secret",
        _run_without_native_path_secret,
    )

    service._default_target(
        event_queue = "events", stop_queue = "stop", config = {"base_model": "example/model"}
    )

    assert calls == ["native_tls", "native_path_secret"]


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


def test_progress_nulls_non_finite_floats_for_strict_json():
    # A divergent step can push loss / avg_loss / learning_rate to NaN or Infinity, which strict JSON forbids, so the service must null them.
    import json
    import math

    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event(
        {
            "type": "progress",
            "step": 3,
            "total_steps": 10,
            "loss": float("nan"),
            "avg_loss": float("inf"),
            "learning_rate": float("-inf"),
            "grad_norm": float("inf"),
        }
    )
    snap = svc.status()
    assert snap["loss"] is None
    assert snap["avg_loss"] is None
    assert snap["learning_rate"] is None
    # The reviewer's exact case: an inf pre-clip grad norm must not reach the status JSON.
    assert snap["grad_norm"] is None
    # The non-finite point is skipped in the history, so the loss series stays clean.
    assert snap["metric_loss"] == []
    assert snap["metric_steps"] == []
    # strict JSON (allow_nan=False) round-trips without a ValueError from NaN/Infinity.
    json.dumps(snap, allow_nan = False)

    # A finite point after the bad one is recorded and preserved verbatim.
    svc._apply_event(
        {"type": "progress", "step": 4, "total_steps": 10, "loss": 0.5, "learning_rate": 1e-4}
    )
    snap2 = svc.status()
    assert snap2["loss"] == 0.5
    assert snap2["metric_loss"] == [0.5] and snap2["metric_steps"] == [4]
    assert math.isfinite(snap2["learning_rate"])
    json.dumps(snap2, allow_nan = False)


def test_terminal_events_clear_model_load_flag():
    # A stop or error during model load emits complete/error without a preceding model_load_completed, so the terminal update must reset in_model_load.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event({"type": "model_load_started"})
    assert svc.status()["in_model_load"] is True
    svc._apply_event({"type": "complete", "stopped": True})
    assert svc.status()["in_model_load"] is False and svc.status()["status"] == "stopped"

    svc2 = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc2._apply_event({"type": "model_load_started"})
    svc2._apply_event({"type": "error", "message": "load failed"})
    assert svc2.status()["in_model_load"] is False and svc2.status()["status"] == "error"


def test_complete_event_keeps_the_ema_adapter_path():
    # A DiT run with ema_decay writes a SECOND adapter as ema_path; dropping the field leaves that adapter undiscoverable.
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    svc._apply_event(
        {
            "type": "complete",
            "output_dir": "/o",
            "lora_path": "/o/a.safetensors",
            "ema_path": "/o/ema/a.safetensors",
        }
    )
    snap = svc.status()
    assert snap["lora_path"] == "/o/a.safetensors"
    assert snap["ema_path"] == "/o/ema/a.safetensors"
    # A run without EMA leaves it null rather than carrying the previous run's path.
    svc._apply_event({"type": "complete", "output_dir": "/o2", "lora_path": "/o2/a.safetensors"})
    assert svc.status()["ema_path"] is None


# ── route wiring (mocked service) ─────────────────────────────────────────────
class _FakeService:
    def __init__(self):
        self._running = False
        self._reserved = False
        self.started_with = None
        self.stopped_with_save = None
        # Ordered log of lifecycle calls so a test can assert reserve precedes the GPU free.
        self.calls: list = []
        # Extra keys merged into status() so a test can inject metric history / perf fields.
        self.status_extra: dict = {}

    def reserve(self):
        self._reserved = True
        self.calls.append("reserve")

    def unreserve(self):
        self._reserved = False
        self.calls.append("unreserve")

    def is_active(self):
        return self._reserved or self._running

    @contextlib.contextmanager
    def dataset_mutation(self):
        # Mirrors the real interlock: refuse while a run owns the dataset, and register the mutation so a concurrent reserve() is refused.
        from core.training.diffusion_training_service import TrainingActiveError

        if self.is_active():
            raise TrainingActiveError(
                "Training images cannot be changed while diffusion training is active."
            )
        self.calls.append("dataset_mutation")
        yield

    def start(self, config):
        self.started_with = config
        self._running = True
        self.calls.append("start")
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
    # Neutralize the LLM interlock + GPU-free for the wiring tests. The route imports get_training_backend at module scope.
    import routes.training as tr

    monkeypatch.setattr(tr, "get_training_backend", lambda: _FakeLLMBackend(active = False))
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    # The dataset preflight runs the trainer discovery against _BODY's fake data_dir; stub it for the wiring tests.
    monkeypatch.setattr(
        "core.training.diffusion_train_common.discover_image_caption_pairs",
        lambda data_dir, **kw: [("img.png", "caption")],
    )
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    # Default to session (UI) auth: the API-key inference-in-flight guard is a no-op there. The guard test flips this.
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    c = TestClient(app)
    c._fake = fake  # type: ignore[attr-defined]
    c._app = app  # type: ignore[attr-defined]
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


def test_route_start_frees_gpu_off_the_coroutine_thread(client, monkeypatch):
    # The GPU cleanup can block for seconds (engine unload waits on generation locks), so the async start route must
    # offload it via asyncio.to_thread. Assert it runs on a DIFFERENT thread than the inline coroutine body.
    import threading

    import routes.training as tr

    threads: dict = {}

    def _record_cleanup():
        threads["cleanup"] = threading.current_thread()

    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", _record_cleanup)

    orig_start = client._fake.start

    def _record_start(config):
        threads["inline"] = threading.current_thread()
        return orig_start(config)

    monkeypatch.setattr(client._fake, "start", _record_start)

    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    assert threads["cleanup"] is not threads["inline"]  # offloaded to a worker, not run inline


def test_route_start_reserves_before_freeing_gpu(client, monkeypatch):
    # The training slot must be reserved (is_active -> true) BEFORE the route frees resident GPU models, so a concurrent load guard refuses during the free-then-spawn window.
    import routes.training as tr

    order: list = []

    def _record_free():
        order.append("free")
        # During the free window the service must already look active to a concurrent load guard.
        order.append(f"active={client._fake.is_active()}")

    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", _record_free)

    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    # reserve fires before the free, the free sees an active service, then start, then unreserve.
    assert client._fake.calls[0] == "reserve"
    assert client._fake.calls.index("reserve") < client._fake.calls.index("start")
    assert order == ["free", "active=True"]
    assert "unreserve" in client._fake.calls


def test_route_start_reserves_before_scanning_dataset(client, monkeypatch):
    # The dataset scan decode-probes every image, so it must run AFTER the slot is reserved or a concurrent edit could mutate the dataset the trainer is about to read.
    order: list = []

    def _record_scan(data_dir, **kw):
        client._fake.calls.append("scan")
        order.append(f"scan_active={client._fake.is_active()}")
        return [("img.png", "caption")]

    monkeypatch.setattr(
        "core.training.diffusion_train_common.discover_image_caption_pairs", _record_scan
    )

    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    assert client._fake.calls.index("reserve") < client._fake.calls.index("scan")
    assert order == ["scan_active=True"]
    assert "unreserve" in client._fake.calls


def test_route_start_unreserves_when_dataset_preflight_fails(client, monkeypatch):
    # A dataset preflight failure AFTER the reservation must roll it back, or a rejected start leaves training permanently "active".
    def _bad_scan(data_dir, **kw):
        raise ValueError("no captioned images found")

    monkeypatch.setattr(
        "core.training.diffusion_train_common.discover_image_caption_pairs", _bad_scan
    )

    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 400
    assert "no captioned images" in r.json()["detail"]
    # Reserved, then rolled back; never started, and no longer active.
    assert "reserve" in client._fake.calls and "unreserve" in client._fake.calls
    assert "start" not in client._fake.calls
    assert client._fake.is_active() is False


def test_service_reserve_marks_active_and_rolls_back():
    # The real service: reserve() flips is_active true before any proc exists, and unreserve() clears it without a live proc.
    from core.training.diffusion_training_service import DiffusionTrainingService

    svc = DiffusionTrainingService()
    assert svc.is_active() is False
    svc.reserve()
    assert svc.is_active() is True  # active with no proc, purely from the reservation
    svc.unreserve()
    assert svc.is_active() is False


def test_service_reserve_is_compare_and_set():
    # reserve() is the concurrency gate: two starts can interleave between the is_active() check and the reservation, so
    # it must reject the second atomically. Without the compare-and-set the loser 409ed only AFTER evicting the GPU.
    from core.training.diffusion_training_service import DiffusionTrainingService

    svc = DiffusionTrainingService()
    svc.reserve()
    with pytest.raises(RuntimeError, match = "already running"):
        svc.reserve()
    assert svc.is_active() is True  # the losing reserve did not clear the winner's claim
    svc.unreserve()
    svc.reserve()  # claimable again once released
    assert svc.is_active() is True


def test_route_start_preflights_gated_base_off_the_coroutine_thread(client, monkeypatch):
    # _preflight_gated_base does a blocking urlopen HEAD (up to 5s), so the start route must offload it. Assert it runs on a DIFFERENT thread.
    import threading

    import routes.training as tr

    threads: dict = {}

    def _record_preflight(base_model, hf_token):
        threads["preflight"] = threading.current_thread()

    monkeypatch.setattr(tr, "_preflight_gated_base", _record_preflight)

    orig_start = client._fake.start

    def _record_start(config):
        threads["inline"] = threading.current_thread()
        return orig_start(config)

    monkeypatch.setattr(client._fake, "start", _record_start)

    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    assert threads["preflight"] is not threads["inline"]  # offloaded to a worker, not run inline


def test_route_start_forwards_extra_training_knobs(client):
    # max_grad_norm and lora_target_modules must reach the service, not be silently dropped.
    body = {**_BODY, "max_grad_norm": 0.5, "lora_target_modules": ["to_q", "to_v"]}
    r = client.post("/api/train/diffusion/start", json = body)
    assert r.status_code == 200, r.text
    assert client._fake.started_with["max_grad_norm"] == 0.5
    assert client._fake.started_with["lora_target_modules"] == ["to_q", "to_v"]


def test_route_start_forwards_num_epochs(client):
    # Epochs mode: the frontend omits train_steps and sends num_epochs; it must reach the service.
    body = {k: v for k, v in _BODY.items() if k != "train_steps"}
    r = client.post("/api/train/diffusion/start", json = {**body, "num_epochs": 8})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["num_epochs"] == 8


def test_route_start_forwards_dit_loss_knobs(client):
    # The trainer implements these, but the request schema did not declare them, so model_dump() dropped them silently.
    body = {
        **_BODY,
        "ema_decay": 0.99,
        "cfg_dropout": 0.1,
        "weighting_scheme": "bell",
        "flow_shift": 3.0,
    }
    r = client.post("/api/train/diffusion/start", json = body)
    assert r.status_code == 200, r.text
    started = client._fake.started_with
    assert started["ema_decay"] == 0.99 and started["cfg_dropout"] == 0.1
    assert started["weighting_scheme"] == "bell" and started["flow_shift"] == 3.0


def test_request_model_dit_loss_knob_bounds():
    # Bounds mirror DiffusionLoraConfig.normalized(); flow_shift also accepts "auto".
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    base = {"base_model": "b", "data_dir": "d", "output_dir": "o"}
    defaults = DiffusionTrainingStartRequest(**base)
    assert (defaults.ema_decay, defaults.cfg_dropout) == (0.0, 0.0)
    assert defaults.weighting_scheme == "none" and defaults.flow_shift is None
    assert DiffusionTrainingStartRequest(**base, flow_shift = "auto").flow_shift == "auto"
    for bad in ({"ema_decay": 1.0}, {"cfg_dropout": 1.5}, {"weighting_scheme": "bogus"}):
        with pytest.raises(ValidationError):
            DiffusionTrainingStartRequest(**base, **bad)


def test_request_model_rejects_lora_dropout_of_one():
    # lora_dropout = 1.0 makes PEFT build nn.Dropout(p=1.0), zeroing the LoRA branch: both lora_A/lora_B get zero
    # gradient, so the run saves an untrained adapter while reporting normal progress. The schema must require < 1.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    base = {"base_model": "b", "data_dir": "d", "output_dir": "o"}
    assert DiffusionTrainingStartRequest(**base).lora_dropout == 0.0
    assert DiffusionTrainingStartRequest(**base, lora_dropout = 0.99).lora_dropout == 0.99
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            DiffusionTrainingStartRequest(**base, lora_dropout = bad)


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
    # The base_precision Literal now includes mxfp8 (the DiT dense speed mode); a bogus mode is still rejected.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    base = {"base_model": "b", "data_dir": "d", "output_dir": "o"}
    assert DiffusionTrainingStartRequest(**base).base_precision == "nf4"  # default
    assert DiffusionTrainingStartRequest(**base, base_precision = "mxfp8").base_precision == "mxfp8"
    with pytest.raises(ValidationError):
        DiffusionTrainingStartRequest(**base, base_precision = "bogus")


def test_config_from_dict_epoch_mode_drops_max_steps_sentinel():
    # The generic epoch-mode payload sends max_steps: 0 as the "use epochs" sentinel, which normalized() would reject as train_steps < 1, so _config_from_dict drops it.
    from core.training.diffusion_train_common import DiffusionLoraConfig, _config_from_dict

    cfg = _config_from_dict(
        {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "data_dir": "d",
            "output_dir": "o",
            "max_steps": 0,
            "num_epochs": 2,
        }
    )
    # 0 was dropped: the dataclass default train_steps stands in and num_epochs carries over.
    assert cfg.train_steps == DiffusionLoraConfig.train_steps
    assert cfg.num_epochs == 2
    # normalized() no longer raises on the epoch-mode payload.
    norm = cfg.normalized()
    assert norm.num_epochs == 2

    # An explicit non-zero max_steps in epochs mode is still honored, and a plain steps payload keeps 0 so normalized() surfaces it.
    cfg_explicit = _config_from_dict(
        {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "data_dir": "d",
            "output_dir": "o",
            "max_steps": 25,
            "num_epochs": 2,
        }
    )
    assert cfg_explicit.train_steps == 25


def test_permutation_sampler_covers_dataset_once_per_cycle():
    # Every index must appear exactly once per cycle before any repeat, so a short run never leaves images unseen. Consecutive cycles are reshuffled.
    import random

    from core.training.diffusion_train_common import PermutationBatchSampler

    n = 100
    sampler = PermutationBatchSampler(n, random.Random(0))

    # Draw exactly one cycle in batches of 3 (n not divisible by the batch); the first n indices must be a permutation.
    drawn: list[int] = []
    while len(drawn) < n:
        drawn.extend(sampler.next_batch(3))
    first_cycle = drawn[:n]
    assert sorted(first_cycle) == list(range(n))  # each index once, none missing

    # The next full cycle is also a permutation, and it is reshuffled (order differs).
    fresh = PermutationBatchSampler(n, random.Random(0))
    cycle_a = fresh.next_batch(n)
    cycle_b = fresh.next_batch(n)
    assert sorted(cycle_a) == list(range(n))
    assert sorted(cycle_b) == list(range(n))
    assert cycle_a != cycle_b  # cycles are reshuffled, not repeated in the same order

    # A seed replays the exact index stream (determinism for reproducible runs).
    replay = PermutationBatchSampler(n, random.Random(0))
    assert replay.next_batch(n) == cycle_a

    # A batch larger than the dataset refills across cycles so it never shrinks, repeating indices within the batch.
    big = PermutationBatchSampler(4, random.Random(1))
    batch = big.next_batch(10)
    assert len(batch) == 10
    assert set(batch) == {0, 1, 2, 3}


def test_permutation_sampler_honors_batch_on_tiny_dataset():
    # Regression for the SDXL trainer clamp: a dataset smaller than train_batch_size must still yield exactly train_batch_size indices.
    import random

    from core.training.diffusion_train_common import PermutationBatchSampler

    sampler = PermutationBatchSampler(2, random.Random(0))  # 2-image dataset
    batch = sampler.next_batch(8)  # train_batch_size = 8, not clamped to 2
    assert len(batch) == 8
    assert set(batch) == {0, 1}
    # A whole-multiple batch draws each image equally, so the effective gradient matches the configured batch rather than a shrunk one.
    assert batch.count(0) == 4 and batch.count(1) == 4


def test_route_start_accepts_zero_max_grad_norm(client):
    # 0 is the documented "disable clipping" value (the trainer skips clip_grad_norm_), so the request model must not reject it.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "max_grad_norm": 0.0})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["max_grad_norm"] == 0.0


def test_route_start_rejects_nonpositive_snr_gamma(client):
    # A gamma at or below 0 zeroes/inverts the min-SNR loss weight; null is the disable value.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "snr_gamma": 0})
    assert r.status_code == 422


def test_route_start_rejects_uncontained_paths(client):
    # An absolute path outside the Studio dataset roots is a 400, not silently accepted.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "data_dir": "/etc"})
    assert r.status_code == 400


def test_route_start_resolves_bare_name_under_image_dataset_root(client, monkeypatch, tmp_path):
    # The upload/labeling routes manage image datasets under datasets_root() and the UI passes the bare folder name back.
    # resolve_dataset_path searches the LLM roots FIRST, so the route must prefer the image dataset root for a bare name.
    import utils.paths as up

    ds_root = tmp_path / "assets" / "datasets"
    img_ds = ds_root / "my-photos"
    img_ds.mkdir(parents = True)
    (img_ds / "a.png").write_bytes(b"x")
    # Shadowing entries the generic resolver would pick first.
    (ds_root / "uploads").mkdir()
    (ds_root / "uploads" / "my-photos").write_text("an LLM dataset upload, not a folder")
    (ds_root / "recipes" / "my-photos").mkdir(parents = True)
    monkeypatch.setattr(up, "datasets_root", lambda: ds_root)

    r = client.post("/api/train/diffusion/start", json = {**_BODY, "data_dir": "my-photos"})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["data_dir"] == str(img_ds)


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


def test_route_start_over_api_with_inference_in_flight_is_409(client, monkeypatch):
    # An API-key client must not start diffusion training (which unloads chat) while an inference request is streaming; it should 409.
    client._app.dependency_overrides[authenticated_via_api_key] = lambda: True
    monkeypatch.setattr(
        "core.inference.llama_keepwarm.other_inference_request_count",
        lambda current_request_counted = False: 1,
    )
    freed = {"called": False}
    import routes.training as tr

    monkeypatch.setattr(
        tr, "_free_gpu_for_diffusion_training", lambda: freed.__setitem__("called", True)
    )
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 409
    # The guard must run BEFORE any GPU is freed, so the live inference stream survives.
    assert freed["called"] is False


def test_route_start_over_api_without_inference_proceeds(client, monkeypatch):
    # Same API-key path but no inference in flight: the start proceeds normally.
    client._app.dependency_overrides[authenticated_via_api_key] = lambda: True
    monkeypatch.setattr(
        "core.inference.llama_keepwarm.other_inference_request_count",
        lambda current_request_counted = False: 0,
    )
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200


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
    # A finished job's pump is joined OUTSIDE the lock (it needs the lock for its final writes), so a second start neither stalls nor deadlocks.
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
    # An event carrying a superseded job proc identity must be dropped, so a straggler pump cannot overwrite a new job's state.
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


def test_diffusion_info_counts_metadata_captions(client, dataset_roots):
    # A dataset captioned via metadata.jsonl must not report caption_count=0; metadata rows count like sidecars, without double-counting.
    ds_root, _ = dataset_roots
    folder = ds_root / "meta-captioned"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "b.png").write_bytes(b"x")
    (folder / "c.png").write_bytes(b"x")
    # a.png + b.png via metadata; a.png also has a sidecar (must count once); c.png none.
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "cap a"})
        + "\n"
        + json.dumps({"file_name": "b.png", "text": "cap b"})
        + "\n",
        encoding = "utf-8",
    )
    (folder / "a.txt").write_text("edited a", encoding = "utf-8")

    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "meta-captioned")
    assert summary["image_count"] == 3
    assert summary["caption_count"] == 2


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


def test_diffusion_dataset_upload_normalizes_windows_and_rejects_dotdot(client, dataset_roots):
    ds_root, _ = dataset_roots
    # A Windows client can send a backslash path in the multipart filename and POSIX Path.name does not split on it, so
    # it must be folded to the true basename, else the stored name is an orphan the grid can list but never preview.
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "winset"},
        files = [("files", ("C:\\Users\\me\\pics\\cat.png", b"png-bytes", "image/png"))],
    )
    assert r.status_code == 200, r.text
    assert (ds_root / "winset" / "cat.png").read_bytes() == b"png-bytes"
    # It is listed under the clean basename and the per-image endpoints accept it (not an orphan).
    recs = client.get("/api/train/diffusion/dataset/winset/images").json()["images"]
    assert any(rec["filename"] == "cat.png" for rec in recs)
    assert client.get("/api/train/diffusion/dataset/winset/image/cat.png").status_code == 200

    # A basename that still contains ".." is refused at upload rather than persisted as an unmanageable entry.
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "winset"},
        files = [("files", ("a..b.png", b"x", "image/png"))],
    )
    assert r.status_code == 400 and "Unsupported file" in r.json()["detail"]


def test_diffusion_dataset_upload_rejects_case_insensitive_stem_clash(client, dataset_roots):
    # Two images whose stems differ only by case map to the SAME sidecar on case-insensitive filesystems, so the clash check compares casefolded stems on any host.
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "caseset"},
        files = [
            ("files", ("sample.png", b"p", "image/png")),
            ("files", ("Sample.jpg", b"j", "image/jpeg")),
        ],
    )
    assert r.status_code == 400 and "Duplicate image name" in r.json()["detail"]

    # The same clash across batches (the new image collides with one already on disk).
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "caseset2"},
        files = [("files", ("photo.png", b"p", "image/png"))],
    )
    assert r.status_code == 200, r.text
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "caseset2"},
        files = [("files", ("PHOTO.webp", b"w", "image/webp"))],
    )
    assert r.status_code == 400 and "Duplicate image name" in r.json()["detail"]

    # A same-name case variant with the SAME extension is an overwrite, not a caption clash, so it is still allowed.
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "caseset3"},
        files = [
            ("files", ("pic.png", b"a", "image/png")),
            ("files", ("Pic.png", b"b", "image/png")),
        ],
    )
    assert r.status_code == 200, r.text


def test_diffusion_dataset_upload_over_cap_keeps_existing_example(
    client, dataset_roots, monkeypatch
):
    # A re-upload that trips the size cap mid-write must not destroy the stored file: staging to a sibling temp leaves the prior bytes intact.
    import utils.upload_limits as ul

    ds_root, _ = dataset_roots
    folder = ds_root / "my style"
    folder.mkdir()
    (folder / "cat.png").write_bytes(b"ORIGINAL-CAT-BYTES")

    monkeypatch.setattr(ul, "get_upload_limit_bytes", lambda: 8)
    monkeypatch.setattr(ul, "get_upload_limit_label", lambda: "8B")

    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "my style"},
        files = [("files", ("cat.png", b"x" * 64, "image/png"))],
    )
    assert r.status_code == 413, r.text
    # The pre-existing example survives untouched, and no temp file is left behind.
    assert (folder / "cat.png").read_bytes() == b"ORIGINAL-CAT-BYTES"
    assert sorted(p.name for p in folder.iterdir()) == ["cat.png"]


def test_diffusion_info_empty_sidecar_shadows_metadata_caption(client, dataset_roots):
    # An empty (tombstone) .txt sidecar shadows a metadata row -- the trainer skips the image -- so the summary must not count it captioned.
    ds_root, _ = dataset_roots
    folder = ds_root / "tombstoned"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "b.png").write_bytes(b"x")
    (folder / "c.png").write_bytes(b"x")
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "cap a"})
        + "\n"
        + json.dumps({"file_name": "c.png", "text": "cap c"})
        + "\n",
        encoding = "utf-8",
    )
    # a.png: metadata caption but an empty sidecar tombstone -> uncaptioned.
    (folder / "a.txt").write_text("   ", encoding = "utf-8")
    # b.png: real sidecar caption. c.png: metadata only. Both captioned.
    (folder / "b.txt").write_text("cap b", encoding = "utf-8")

    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "tombstoned")
    assert summary["image_count"] == 3
    assert summary["caption_count"] == 2


def _png_bytes(width: int, height: int) -> bytes:
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buf, format = "PNG")
    return buf.getvalue()


def test_diffusion_dataset_upload_rejects_oversized_image(client, dataset_roots):
    # A decompression bomb: a small PNG with huge dimensions passes the byte limit but would OOM the trainer, so it must 400 at upload.
    pytest.importorskip("PIL")
    big = _png_bytes(5000, 64)  # > 4096 per side
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "bomb"},
        files = [("files", ("huge.png", big, "image/png"))],
    )
    assert r.status_code == 400, r.text
    assert "too large" in r.json()["detail"]
    # An in-bounds real image still uploads fine.
    ok = _png_bytes(64, 64)
    r2 = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "bomb"},
        files = [("files", ("ok.png", ok, "image/png"))],
    )
    assert r2.status_code == 200, r2.text


def test_diffusion_dataset_upload_maps_pillow_bomb_to_400(client, dataset_roots, monkeypatch):
    # Past Pillow's bomb threshold Image.open() raises DecompressionBombError, which derives straight from Exception, so
    # the dimension guard missed it and the upload 500-ed. Shrink the limit so a small file crosses it.
    pytest.importorskip("PIL")
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 8)  # 8x8 = 64 pixels > 2 x 8
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "bomb-hard"},
        files = [("files", ("huge.png", _png_bytes(8, 8), "image/png"))],
    )
    assert r.status_code == 400, r.text
    assert "too large" in r.json()["detail"]
    # All-or-nothing: the rejected image is not left in the dataset.
    assert not (dataset_roots[0] / "bomb-hard" / "huge.png").exists()


def test_diffusion_info_tolerates_non_object_jsonl(client, dataset_roots):
    # A metadata.jsonl line that is valid JSON but not an object must be skipped per-line, not 500 the info endpoint.
    ds_root, _ = dataset_roots
    folder = ds_root / "weird-meta"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "metadata.jsonl").write_bytes(
        b"[]\n"
        b"null\n"
        b'"just a string"\n'
        b"123\n"
        b"{not json\n" + json.dumps({"file_name": "a.png", "text": "cap a"}).encode("utf-8") + b"\n"
    )
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "weird-meta")
    assert summary["caption_count"] == 1


def test_diffusion_info_skips_null_metadata_captions(client, dataset_roots):
    # A JSON null caption is "no caption": str(None) would store the literal "None" and train on that text.
    ds_root, _ = dataset_roots
    folder = ds_root / "null-caption"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "b.png").write_bytes(b"x")
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": None})
        + "\n"
        + json.dumps({"file_name": "b.png", "text": "cap b"})
        + "\n",
        encoding = "utf-8",
    )
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "null-caption")
    assert summary["caption_count"] == 1

    r2 = client.get("/api/train/diffusion/dataset/null-caption/images")
    assert r2.status_code == 200, r2.text
    by_name = {rec["filename"]: rec for rec in r2.json()["images"]}
    assert by_name["a.png"]["caption"] in (None, "")
    assert by_name["b.png"]["caption"] == "cap b"


def test_diffusion_info_tolerates_invalid_utf8_jsonl(client, dataset_roots):
    # Invalid UTF-8 in a metadata file must not 500 the info endpoint; the file is skipped.
    ds_root, _ = dataset_roots
    folder = ds_root / "bad-utf8-meta"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "metadata.jsonl").write_bytes(b"\xff\xfe not valid utf-8\n")
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "bad-utf8-meta")
    assert summary["caption_count"] == 0


def test_diffusion_info_tolerates_invalid_utf8_sidecar(client, dataset_roots):
    # Same for a .txt sidecar: read_text raises UnicodeDecodeError, not an OSError, so an unguarded read 500s the endpoint.
    ds_root, _ = dataset_roots
    folder = ds_root / "bad-utf8-sidecar"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")
    (folder / "a.txt").write_bytes(b"\xff\xfe not valid utf-8")
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    summary = next(d for d in r.json()["datasets"] if d["name"] == "bad-utf8-sidecar")
    assert summary["caption_count"] == 0


def test_diffusion_dataset_mutations_blocked_while_training_active(client, dataset_roots):
    ds_root, _ = dataset_roots
    folder = ds_root / "locked"
    folder.mkdir()
    folder.joinpath("a.png").write_bytes(b"x")
    # Flip the fake diffusion service to active.
    client._fake._running = True
    # Upload, caption, delete, and example-import must all 409 while a run is active.
    up = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "locked"},
        files = [("files", ("b.png", b"x", "image/png"))],
    )
    assert up.status_code == 409, up.text
    cap = client.put("/api/train/diffusion/dataset/locked/caption/a.png", json = {"caption": "hi"})
    assert cap.status_code == 409, cap.text
    dele = client.delete("/api/train/diffusion/dataset/locked/image/a.png")
    assert dele.status_code == 409, dele.text
    imp = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "anything"})
    assert imp.status_code == 409, imp.text


def test_diffusion_info_skips_symlinked_dataset_dir(client, dataset_roots):
    # A directory symlink under the datasets root must not be advertised as a dataset (the CRUD resolver rejects them).
    import os

    ds_root, _ = dataset_roots
    outside = ds_root.parent / "outside-images"
    outside.mkdir()
    (outside / "a.png").write_bytes(b"x")
    try:
        os.symlink(outside, ds_root / "linked")
    except OSError:
        pytest.skip("symlinks unavailable")
    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    assert "linked" not in [d["name"] for d in r.json()["datasets"]]


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


def test_free_gpu_for_diffusion_training_unloads_video(monkeypatch):
    # A resident Video pipeline loads under the VIDEO arbiter owner the Images teardown does not free, so training must unload it too.
    import routes.training as tr
    from core.inference import gpu_arbiter

    class _Exp:
        current_checkpoint = None

        def is_export_active(self):
            return False

    class _Diff:
        is_loaded = False

        def unload(self):
            pass

    unloaded = {"video": False}

    class _Vid:
        def status(self):
            return {"loaded": True}

        def unload(self):
            unloaded["video"] = True

    released = []
    monkeypatch.setattr("core.export.get_export_backend", lambda: _Exp())
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.get_active_diffusion_engine", lambda: _Diff()
    )
    monkeypatch.setattr("core.inference.video.get_video_backend", lambda: _Vid())
    monkeypatch.setattr(gpu_arbiter, "release", lambda owner: released.append(owner))

    tr._free_gpu_for_diffusion_training()

    assert unloaded["video"] is True
    assert gpu_arbiter.VIDEO in released


def test_keepwarm_tracks_image_video_generation_paths():
    # The API-key start guard uses other_inference_request_count(), so image/video generation must be tracked; the GET progress/cancel variants stay untracked.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/api/inference/images/generate")
    assert _is_inference_path("/v1/images/generations")
    assert _is_inference_path("/api/inference/video/generate")
    assert not _is_inference_path("/api/inference/images/generate-progress")
    assert not _is_inference_path("/api/inference/video/generate-progress")
    assert not _is_inference_path("/api/inference/video/generate/cancel")


def test_import_example_partial_failure_leaves_no_partial_dataset(
    client, dataset_roots, monkeypatch
):
    # A materialize that writes some images then fails must not leave a partial dataset: it stages into a discarded temp dir.
    import routes.training as tr

    ds_root, _ = dataset_roots

    def _boom(entry, dest, cap):
        (dest / "img_0000.png").write_bytes(b"x")  # partial write into staging
        raise RuntimeError("transient copy error")

    monkeypatch.setattr(tr, "_materialize_hf_dataset", _boom)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "dreambooth-dog"})
    assert r.status_code == 502
    folder = ds_root / "dreambooth-dog"
    assert not folder.exists() or not any(folder.iterdir())
    # And no leftover staging dir surfaces as a dataset.
    assert not any(p.name.startswith(".dreambooth-dog.import-") for p in ds_root.iterdir())


def test_diffusion_dataset_upload_over_cap_rolls_back_whole_batch(
    client, dataset_roots, monkeypatch
):
    # All-or-nothing: a valid image ahead of the one that trips the size cap must NOT be left on disk.
    import utils.upload_limits as ul

    monkeypatch.setattr(ul, "get_upload_limit_bytes", lambda: 100)
    ds_root, _ = dataset_roots
    files = [
        ("files", ("small.png", b"x" * 50, "image/png")),
        ("files", ("big.png", b"y" * 200, "image/png")),
    ]
    r = client.post("/api/train/diffusion/dataset", data = {"name": "rollback"}, files = files)
    assert r.status_code == 413, r.text
    folder = ds_root / "rollback"
    assert not (folder / "small.png").exists()  # the earlier valid file was rolled back
    assert not (folder / "big.png").exists()


def test_diffusion_dataset_upload_over_cap_preserves_existing_file(
    client, dataset_roots, monkeypatch
):
    # A failed batch reusing an existing filename must NOT delete the user's file: staging keeps the original until commit.
    import utils.upload_limits as ul

    monkeypatch.setattr(ul, "get_upload_limit_bytes", lambda: 100)
    ds_root, _ = dataset_roots
    folder = ds_root / "keep"
    folder.mkdir(parents = True)
    (folder / "existing.png").write_bytes(b"ORIGINAL")  # from an earlier upload
    files = [
        ("files", ("existing.png", b"NEW", "image/png")),  # re-upload, small
        ("files", ("big.png", b"y" * 200, "image/png")),  # trips the cap
    ]
    r = client.post("/api/train/diffusion/dataset", data = {"name": "keep"}, files = files)
    assert r.status_code == 413, r.text
    assert (folder / "existing.png").read_bytes() == b"ORIGINAL"  # untouched
    assert not (folder / "big.png").exists()
    assert not list(folder.glob(".*.part"))  # no leftover temp files


def test_route_start_refuses_non_sdxl_base_without_freeing_gpu(client, monkeypatch):
    # A doomed start (non-SDXL base) must 400 BEFORE resident GPU workloads are freed.
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


def test_route_start_refuses_non_bf16_gpu_without_freeing_gpu(client, monkeypatch):
    # A DiT precision the host cannot run must 400 BEFORE the GPU residents are freed. The route imports the helper locally, so patch its home module.
    import routes.training as tr

    freed = []
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append(1))
    monkeypatch.setattr(
        "core.training.diffusion_train_common.training_precision_preflight_error",
        lambda fam, prec: (
            "This trainer requires a bfloat16-capable GPU (Ampere or newer)."
            if fam != "sdxl"
            else None
        ),
    )
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "base_model": "black-forest-labs/FLUX.1-dev"},
    )
    assert r.status_code == 400
    assert "bfloat16" in r.json()["detail"]
    assert freed == []
    assert client._fake.started_with is None

    # SDXL (its own mixed_precision path) is exempt: the same probe returns None, so an SDXL start proceeds.
    r2 = client.post("/api/train/diffusion/start", json = _BODY)
    assert r2.status_code == 200, r2.text


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


def test_start_gated_base_without_access_is_400_and_keeps_gpu(client, monkeypatch, dit_train_host):
    # A gated FLUX base with no valid token must 400 from the HEAD preflight BEFORE the GPU residents are freed.
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


def test_route_start_refuses_a_dit_family_without_a_gpu_before_freeing(client, monkeypatch):
    # nf4 is not a CPU fallback: the 4-bit base load needs CUDA/XPU/MPS. Without a pre-teardown gate the default pick evicted the Images pipeline then failed in the child.
    import torch

    import routes.training as tr

    freed = []
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)

    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "base_model": "black-forest-labs/FLUX.1-dev"},
    )
    assert r.status_code == 400
    assert "GPU" in r.json()["detail"]
    assert freed == []
    assert client._fake.started_with is None

    # SDXL trains fp32 on CPU (its documented fallback), so it is not gated.
    r2 = client.post("/api/train/diffusion/start", json = _BODY)
    assert r2.status_code == 200, r2.text


def test_start_ungated_base_preflight_is_noop(client, monkeypatch):
    # A reachable base (HEAD 200) proceeds to start normally.
    import urllib.request

    import routes.training as tr

    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    # This asserts the gated-repo probe is a no-op, not device support: pin the precision preflight so a GPU-less host does not 400 for another reason.
    monkeypatch.setattr(
        "core.training.diffusion_train_common.training_precision_preflight_error",
        lambda fam, prec: None,
    )
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


def test_list_diffusion_runs_skips_wrong_shape_records(_isolated_runs_dir):
    # A valid-JSON file with the wrong shape must be skipped by list_diffusion_runs so it never takes down the whole Previous runs panel.
    import json

    from core.training.diffusion_training_service import list_diffusion_runs

    good = {"job_id": "a" * 32, "status": "completed", "adapter": "good", "saved": True}
    (_isolated_runs_dir / "good.json").write_text(json.dumps(good))
    # A JSON list (not a dict).
    (_isolated_runs_dir / "not_a_dict.json").write_text(json.dumps([1, 2, 3]))
    # A dict missing the required job_id / status.
    (_isolated_runs_dir / "no_ids.json").write_text(json.dumps({"adapter": "orphan"}))
    # A dict whose job_id / status are the wrong type.
    (_isolated_runs_dir / "bad_types.json").write_text(
        json.dumps({"job_id": 123, "status": None, "adapter": "typed"})
    )

    runs = list_diffusion_runs()
    adapters = [r.get("adapter") for r in runs]
    assert adapters == ["good"]  # only the well-shaped record survives


def test_runs_route_tolerates_bad_field_record(client, _isolated_runs_dir):
    # A record that passes the shape check but has a wrong-typed field raises ValidationError in the route, so it must catch per record.
    import json

    good = {"job_id": "a" * 32, "status": "completed", "adapter": "good", "saved": True}
    bad = {
        "job_id": "b" * 32,
        "status": "completed",
        "adapter": "bad",
        "avg_loss": "not-a-number",  # str where the summary expects Optional[float]
    }
    (_isolated_runs_dir / f"{good['job_id']}.json").write_text(json.dumps(good))
    (_isolated_runs_dir / f"{bad['job_id']}.json").write_text(json.dumps(bad))

    r = client.get("/api/train/diffusion/runs")
    assert r.status_code == 200, r.text
    adapters = [x["adapter"] for x in r.json()["runs"]]
    assert adapters == ["good"]  # the bad-field record was skipped, the good one remained


def test_run_detail_route_non_object_record_is_404(client, _isolated_runs_dir):
    # A valid-JSON but non-object record makes DiffusionTrainingRunDetail(**rec) raise TypeError, not ValidationError, so the detail route must 404.
    import json

    job_id = "a" * 32
    (_isolated_runs_dir / f"{job_id}.json").write_text(json.dumps([]))

    r = client.get(f"/api/train/diffusion/runs/{job_id}")
    assert r.status_code == 404, r.text


def test_request_model_rejects_non_finite_learning_rate():
    # 1e309 parses as inf, which satisfies a gt-only bound; AdamW would then run at an infinite rate and save a destroyed adapter.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest as R

    base = dict(base_model = "b", data_dir = "d", output_dir = "o")
    assert R(**base, learning_rate = 1e-4).learning_rate == 1e-4
    for bad in (float("inf"), 1e309, float("nan"), 1.0, 5.0, 0.0):
        with pytest.raises(ValidationError):
            R(**base, learning_rate = bad)


def test_request_model_rejects_non_finite_clipping_and_snr():
    # Same 1e309 -> inf vector as learning_rate, on the two one-sided bounds. max_grad_norm = inf passed ge = 0 and clamped
    # clip_grad_norm_'s coefficient to 1.0 (unclipped training); snr_gamma = inf passed gt = 0 and made every min-SNR weight 1.0.
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest as R

    base = dict(base_model = "b", data_dir = "d", output_dir = "o")
    # The legitimate range is untouched, including the documented disables.
    assert R(**base, max_grad_norm = 0.0).max_grad_norm == 0.0
    assert R(**base, max_grad_norm = 1.0).max_grad_norm == 1.0
    assert R(**base, snr_gamma = 5.0).snr_gamma == 5.0
    assert R(**base, snr_gamma = None).snr_gamma is None
    for bad in (float("inf"), float("-inf"), 1e309, float("nan")):
        with pytest.raises(ValidationError):
            R(**base, max_grad_norm = bad)
        with pytest.raises(ValidationError):
            R(**base, snr_gamma = bad)


def test_start_route_never_starts_a_run_with_a_non_finite_knob(client):
    # End to end: FastAPI parses the body with the stdlib json module, which accepts the Infinity literal, and 1e309 floats to inf
    # under any parser, so the schema is the only guard. Asserted on the outcome, since the 422 handler cannot serialise inf back.
    from fastapi.testclient import TestClient

    strict = TestClient(client._app, raise_server_exceptions = False)
    for raw in ('"max_grad_norm": 1e309', '"max_grad_norm": Infinity', '"snr_gamma": 1e309'):
        body = json.dumps(_BODY)[:-1] + ", " + raw + "}"
        r = strict.post(
            "/api/train/diffusion/start",
            content = body,
            headers = {"content-type": "application/json"},
        )
        assert r.status_code >= 400, (raw, r.text)
        assert client._fake.started_with is None, raw
    # Positive control, so the assertions above cannot pass vacuously.
    assert strict.post("/api/train/diffusion/start", json = _BODY).status_code == 200
    assert client._fake.started_with is not None


def test_dataset_mutation_and_reserve_refuse_each_other():
    """The route layer checked is_active() and only then handed the filesystem work to a thread, so
    a /diffusion/start could reserve inside that gap and the caption or image changed underneath the
    preflight or the live trainer. Registering the mutation under the same lock closes it from both
    sides, and neither side waits on the other (a start must not block on a long dataset import)."""
    from core.training.diffusion_training_service import (
        DatasetMutationInFlight,
        DiffusionTrainingService,
        TrainingActiveError,
    )

    svc = DiffusionTrainingService()
    # A start cannot slip in while a mutation is open.
    with svc.dataset_mutation():
        with pytest.raises(DatasetMutationInFlight):
            svc.reserve()
    svc.reserve()  # claimable again once the mutation closed
    # And a mutation is refused once the start is reserved.
    with pytest.raises(TrainingActiveError):
        with svc.dataset_mutation():
            pass
    svc.unreserve()
    with svc.dataset_mutation():
        pass


def test_dataset_mutation_releases_on_failure():
    # A mutation that raises must not leave the counter set, or every later start would 409.
    from core.training.diffusion_training_service import DiffusionTrainingService

    svc = DiffusionTrainingService()
    with pytest.raises(ValueError):
        with svc.dataset_mutation():
            raise ValueError("boom")
    svc.reserve()  # the failed mutation left nothing behind


def test_diffusion_seed_is_bounded_to_torch_range():
    """torch.manual_seed unpacks int64/uint64, so a wider value raised inside the trainer -- after
    the route had already evicted the resident image/video/chat models."""
    from pydantic import ValidationError

    from core.training.diffusion_lora_trainer import _config_from_dict
    from models.training import DiffusionTrainingStartRequest

    base = {
        "base_model": "unsloth/sdxl-turbo",
        "data_dir": "/tmp/x",
        "output_dir": "/tmp/out",
        "seed": 2**64,
    }
    with pytest.raises(ValueError, match = "seed"):
        _config_from_dict(base).normalized()
    request = {k: v for k, v in base.items() if k != "seed"}
    for bad in (2**64, -(2**63) - 1):
        with pytest.raises(ValidationError):
            DiffusionTrainingStartRequest(**request, seed = bad)
    # The extremes torch does accept stay valid.
    for good in (2**64 - 1, -(2**63)):
        assert DiffusionTrainingStartRequest(**request, seed = good).seed == good


def test_gpu_load_admission_and_reserve_exclude_each_other():
    # The load guards read is_active() and only THEN acquire the arbiter, so a start reserving inside that window freed
    # residents the load had not registered yet. The admission closes it from both sides, like the dataset interlock.
    from core.training.diffusion_training_service import TrainingActiveError

    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)

    # A start cannot reserve while a load is registering.
    with svc.gpu_load_admission():
        with pytest.raises(RuntimeError, match = "loaded onto the GPU"):
            svc.reserve()
    # ...and the admission is released afterwards, so the start goes through.
    svc.reserve()
    try:
        # A load cannot register while a start is reserved, and it says why.
        with pytest.raises(TrainingActiveError, match = "Diffusion training is running"):
            with svc.gpu_load_admission():
                pass
    finally:
        svc.unreserve()

    # Nested/concurrent admissions are counted, not boolean: the first exit must not open the door while a second load registers.
    with svc.gpu_load_admission():
        with svc.gpu_load_admission():
            pass
        with pytest.raises(RuntimeError, match = "loaded onto the GPU"):
            svc.reserve()
    svc.reserve()
    svc.unreserve()


def test_route_start_carries_and_contains_the_conditioning_cache_dir(client, dit_train_host):
    # The persistent conditioning cache (cond_cache_dir) skips the VAE and text encoders on a rerun, but the start schema
    # omitted the field, so Pydantic dropped it. It must also be contained like output_dir, and is sent against a DiT family.
    from pathlib import Path

    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "model_family": "z-image", "cond_cache_dir": "cond-cache"},
    )
    assert r.status_code == 200, r.text
    resolved = client._fake.started_with["cond_cache_dir"]
    assert Path(resolved).is_absolute()
    assert Path(resolved).name == "cond-cache"

    # Omitted or blank keeps the in-memory cache rather than resolving to the outputs root.
    r = client.post("/api/train/diffusion/start", json = _BODY)
    assert r.status_code == 200, r.text
    assert client._fake.started_with["cond_cache_dir"] is None
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "cond_cache_dir": "   "})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["cond_cache_dir"] is None


def test_route_refuses_an_output_dir_that_is_the_outputs_root(client):
    # "." / "outputs" / "./." all clean away to nothing and resolve to the outputs ROOT, where the adapter lands flat and the is_dir()-filtered listings cannot see it.
    from pathlib import Path

    for name in (".", "./", "./.", "outputs", "outputs/outputs", " . "):
        r = client.post("/api/train/diffusion/start", json = {**_BODY, "output_dir": name})
        assert r.status_code == 400, f"{name!r} -> {r.status_code} {r.text}"
        assert "not a run inside it" in r.json()["detail"]
    # A real name still works.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "output_dir": "outputs/run-1"})
    assert r.status_code == 200, r.text
    assert Path(client._fake.started_with["output_dir"]).name == "run-1"


def test_route_treats_a_root_cond_cache_dir_as_the_in_memory_cache(client, dit_train_host):
    # Same collapse on the cache side, but with an honest "off" to fall back to: the root would take one flat safetensors per cached latent.
    for name in (".", "./.", "outputs", " . "):
        r = client.post(
            "/api/train/diffusion/start",
            json = {**_BODY, "model_family": "z-image", "cond_cache_dir": name},
        )
        assert r.status_code == 200, r.text
        assert client._fake.started_with["cond_cache_dir"] is None, name


def test_route_rejects_cond_cache_dir_for_sdxl(client, dit_train_host):
    # Only the DiT trainer reads cond_cache_dir; the SDXL trainer never touches the persistent store, so refuse the option rather than ignore it.
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "model_family": "sdxl", "cond_cache_dir": "cond-cache"},
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "cond_cache_dir" in detail and "sdxl" in detail
    # It names the families that DO support it, so the message is actionable.
    assert "z-image" in detail

    # The check is on the RESOLVED family, so omitting model_family and letting an SDXL base be detected is refused too.
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "cond_cache_dir": "cond-cache"},
    )
    assert r.status_code == 400, r.text
    assert "cond_cache_dir" in r.json()["detail"]

    # A DiT family still accepts it, resolved and contained like output_dir.
    r = client.post(
        "/api/train/diffusion/start",
        json = {**_BODY, "model_family": "z-image", "cond_cache_dir": "cond-cache"},
    )
    assert r.status_code == 200, r.text
    from pathlib import Path

    assert Path(client._fake.started_with["cond_cache_dir"]).is_absolute()
    # And omitting it stays off (the trainer's in-memory default), not resolved to the outputs root.
    r = client.post("/api/train/diffusion/start", json = {**_BODY, "model_family": "sdxl"})
    assert r.status_code == 200, r.text
    assert client._fake.started_with["cond_cache_dir"] is None


def test_service_reserve_refuses_while_the_llm_trainer_holds_the_gpu(monkeypatch):
    # The route's reciprocal check runs network-bound preflights before reserve(), so reserve() re-tests the LLM backend under its own lock.
    import types

    import core.training.diffusion_training_service as dts
    from core.training.diffusion_training_service import DiffusionTrainingService

    svc = DiffusionTrainingService()
    monkeypatch.setattr(dts, "_llm_training_active", lambda: True)
    with pytest.raises(RuntimeError, match = "LLM training job is already running"):
        svc.reserve()
    assert svc.is_active() is False  # the refused start left no claim behind

    monkeypatch.setattr(dts, "_llm_training_active", lambda: False)
    svc.reserve()
    assert svc.is_active() is True


def test_llm_active_probe_fails_open(monkeypatch):
    # A chat-only install (or a wedged backend) must not block a diffusion start.
    import sys

    import core.training.diffusion_training_service as dts

    monkeypatch.setitem(sys.modules, "core.training", None)  # import raises
    assert dts._llm_training_active() is False


def test_llm_start_holds_the_diffusion_admission_across_its_spawn(monkeypatch):
    # The other half: while the LLM route is spawning it holds gpu_load_admission(), and reserve() refuses while one is open.
    from core.training.diffusion_training_service import DiffusionTrainingService
    import routes.training as tr

    svc = DiffusionTrainingService()
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: svc
    )
    with tr._diffusion_gpu_admission():
        with pytest.raises(RuntimeError, match = "being loaded onto the GPU"):
            svc.reserve()
    svc.reserve()  # released once the spawn is done


def test_llm_start_admission_refuses_when_diffusion_is_already_reserved(monkeypatch):
    from core.training.diffusion_training_service import DiffusionTrainingService
    import routes.training as tr

    svc = DiffusionTrainingService()
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: svc
    )
    svc.reserve()
    with pytest.raises(tr._DiffusionStartInFlight):
        with tr._diffusion_gpu_admission():
            pass


def test_llm_start_admission_fails_open_without_a_diffusion_stack(monkeypatch):
    import sys

    import routes.training as tr

    monkeypatch.setitem(sys.modules, "core.training.diffusion_training_service", None)
    with tr._diffusion_gpu_admission():
        pass  # a chat-only install still trains
