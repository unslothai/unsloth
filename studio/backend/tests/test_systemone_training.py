# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import contextlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from core.systemone import catalog, laya_runtime, native_worker, owned_runtime, runtime
from core.training.diffusion_training_service import DiffusionTrainingService
from .test_training_start_offload import _FakeBackend, _request
from utils import systemone_settings


@pytest.mark.parametrize("backend", ["pytorch", "llama.cpp"])
@pytest.mark.parametrize("trainer", ["llm", "diffusion"])
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_training_blocks_gpu_decisions_before_registration(monkeypatch, backend, trainer, device):
    service = DiffusionTrainingService()
    service._reserved = trainer == "diffusion"
    monkeypatch.setattr(
        "core.training.get_training_backend",
        lambda: SimpleNamespace(is_training_active = lambda: trainer == "llm"),
    )
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: service
    )
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: backend)
    monkeypatch.setattr(systemone_settings, "get_device", lambda: device)
    monkeypatch.setattr(native_worker, "native_availability", lambda: {"available": True})
    monkeypatch.setattr(laya_runtime, "status", lambda: {"loaded_model": None})
    monkeypatch.setattr(laya_runtime, "ensure_can_unload", lambda: None)
    registered = []
    monkeypatch.setattr(
        "core.inference.gpu_arbiter.acquire_for",
        lambda owner, register, **kwargs: registered.append(owner) or register(),
    )
    monkeypatch.setattr(owned_runtime, "prepare", lambda checkpoint: None)
    monkeypatch.setattr(owned_runtime, "decide", lambda *args: {"answers": {}})
    call = lambda: runtime.decide(
        catalog.CHECKPOINTS["clef-flash"], "state", {"q": {"type": "noul"}}, []
    )
    if device == "gpu":
        with pytest.raises(runtime.Unavailable, match = "training"):
            call()
    else:
        assert call()["answers"] == {}
    assert not registered


class Worker:
    def __init__(self):
        self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])

    def is_alive(self):
        return self.process.poll() is None

    def cancel(self):
        pass

    def close(self, **kwargs):
        if self.is_alive():
            self.process.terminate()
        self.process.wait(timeout = 5)
        return True


@pytest.fixture
def resident(monkeypatch):
    owned_runtime.shutdown()
    owned_runtime._shutdown_requested = False
    worker = Worker()
    monkeypatch.setattr(owned_runtime, "_worker", worker)
    monkeypatch.setattr(owned_runtime, "_loaded", catalog.CHECKPOINTS["clef-flash"])
    monkeypatch.setattr(owned_runtime, "_device_name", "cuda")
    monkeypatch.setattr(owned_runtime, "_release_gpu_if_idle", lambda: None)
    monkeypatch.setattr(laya_runtime, "status", lambda: {"device": None, "loaded_model": None})
    yield worker
    Worker.close(worker)
    owned_runtime.shutdown()
    owned_runtime._shutdown_requested = False


@pytest.fixture
def cleanup(monkeypatch):
    import routes.training as training

    engine = SimpleNamespace(
        is_loaded = False,
        runs_off_torch_device = True,
        status = lambda: {"loaded": False},
        unload = lambda: None,
    )
    monkeypatch.setattr(
        "core.export.get_export_backend",
        lambda: SimpleNamespace(current_checkpoint = None, is_export_active = lambda: False),
    )
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.get_active_diffusion_engine", lambda: engine
    )
    monkeypatch.setattr("core.inference.video.get_video_backend", lambda: engine)
    monkeypatch.setattr("routes.training_vram.coordinate_models_for_training", lambda *args: [])
    monkeypatch.setattr("routes.training_vram.summarize_resident_chat", lambda: {"any": False})
    backend = _FakeBackend()
    monkeypatch.setattr(training, "get_training_backend", lambda: backend)
    monkeypatch.setattr(training, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(training, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(
        training,
        "_reject_untrainable_model_request",
        lambda request, *_: SimpleNamespace(
            model_name = request.model_name, cached_model_pin = None, model_local_path = None
        ),
    )
    monkeypatch.setattr(training, "_preflight_hf_dataset_request", lambda *_: None)
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)

    def run(trainer):
        if trainer == "diffusion":
            training._free_gpu_for_diffusion_training()
        else:
            response = asyncio.run(
                training.start_training(
                    request = _request(), current_subject = "owner", via_api_key = False
                )
            )
            assert response.status == "queued"
            backend.hook()

    return run


@pytest.mark.parametrize("trainer", ["llm", "diffusion"])
@pytest.mark.parametrize("placement", ["gpu", "loading", "cpu"])
def test_trainers_retire_gpu_workers_and_pending_loads_not_cpu(
    resident, cleanup, monkeypatch, trainer, placement
):
    cancelled = threading.Event()
    if placement == "cpu":
        monkeypatch.setattr(owned_runtime, "_device_name", "cpu")
    elif placement == "loading":
        done = threading.Event()

        def retire_load():
            cancelled.wait()
            owned_runtime._retire(resident, None)
            done.set()

        threading.Thread(target = retire_load, daemon = True).start()
        monkeypatch.setattr(owned_runtime, "_worker", None)
        monkeypatch.setattr(owned_runtime, "_loaded", None)
        monkeypatch.setattr(owned_runtime, "_device_name", None)
        monkeypatch.setattr(
            owned_runtime, "_loading", SimpleNamespace(worker = resident, cancel = cancelled, done = done)
        )
    cleanup(trainer)
    assert resident.is_alive() == (placement == "cpu")
    if placement == "loading":
        assert cancelled.is_set()


@pytest.mark.parametrize("trainer", ["llm", "diffusion"])
def test_training_refuses_a_worker_that_has_not_reaped(resident, cleanup, monkeypatch, trainer):
    monkeypatch.setattr(resident, "close", lambda **kwargs: False)
    with pytest.raises(RuntimeError) as refused:
        cleanup(trainer)
    assert refused.value.blocks_training is True
    assert resident.is_alive()
    resident.process.terminate()
    resident.process.wait(timeout = 5)


def test_gpu_registration_rechecks_training(resident, monkeypatch):
    active = iter([False, True])
    monkeypatch.setattr(runtime, "_training_active", lambda: next(active), raising = False)
    monkeypatch.setattr(systemone_settings, "get_device", lambda: "gpu")
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: "pytorch")
    monkeypatch.setattr(
        "core.inference.gpu_arbiter.acquire_for", lambda owner, register, **kwargs: register()
    )
    monkeypatch.setattr(owned_runtime, "decide", lambda *args: {"answers": {}})
    with pytest.raises(runtime.Unavailable, match = "training"):
        runtime.decide(catalog.CHECKPOINTS["clef-flash"], "state", {"q": {"type": "noul"}}, [])


@pytest.mark.parametrize("trainer", ["llm", "diffusion"])
def test_training_preserves_active_decisions(resident, cleanup, trainer):
    with owned_runtime._run_lock:
        with pytest.raises(RuntimeError) as refused:
            cleanup(trainer)
        assert refused.value.blocks_training is True
        assert owned_runtime._worker is resident and resident.is_alive()
