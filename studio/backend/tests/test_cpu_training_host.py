# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU-only host trains on the CPU backend: health says so, the start route refuses what that backend
cannot train, and the training worker commits to the verdict before anything imports unsloth."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware as hardware_pkg  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
from models.training import DrafterTrainingConfig, TrainingStartRequest  # noqa: E402


def _restore_real_logging_modules() -> dict:
    """Drops the loggers/structlog stubs other test files install, since main needs the real packages."""
    removed = {}
    stub = sys.modules.get("loggers")
    if stub is not None and not hasattr(stub, "__path__"):
        for name in [n for n in sys.modules if n == "loggers" or n.startswith("loggers.")]:
            removed[name] = sys.modules.pop(name)
    stub = sys.modules.get("structlog")
    if stub is not None and not hasattr(stub, "BoundLogger"):
        removed["structlog"] = sys.modules.pop("structlog")
    try:
        import structlog  # noqa: F401
    except Exception:
        sys.modules.update(removed)
        pytest.skip("real structlog is required to import main")
    return removed


def _health(monkeypatch, *, reason, established) -> dict:
    """The authed /api/health body for a settled chat-only verdict."""
    swapped = _restore_real_logging_modules()
    import auth.authentication as authmod
    import main as main_mod
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(hardware_pkg, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY", True)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "_CPU_TRAINING_REASON", established)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: (reason, None))
    monkeypatch.setattr(main_mod, "_torch_warm_in_progress", lambda: False)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    hw.DETECTION_COMPLETE.set()

    async def _subject(_creds):
        return "tester"

    monkeypatch.setattr(authmod, "get_current_subject", _subject)
    app = FastAPI()
    app.add_api_route("/api/health", main_mod.health_check, methods = ["GET"])
    try:
        with TestClient(app) as client:
            return client.get("/api/health", headers = {"Authorization": "Bearer probe"}).json()
    finally:
        if not was_complete:
            hw.DETECTION_COMPLETE.clear()
        sys.modules.update(swapped)


@pytest.mark.parametrize(
    "reason, established, expected",
    [
        ("no_gpu", "no_gpu", True),
        ("no_torch", "no_torch", True),
        ("no_gpu", None, False),
        ("torch_cpu_build", "no_gpu", False),
        ("torch_cuda_unavailable", "no_gpu", False),
        ("no_torch", None, False),
        ("detection_failed", "no_gpu", False),
        ("no_torch", "no_gpu", False),
    ],
)
def test_health_reports_cpu_training_only_beside_its_verdicts(
    monkeypatch, reason, established, expected
):
    body = _health(monkeypatch, reason = reason, established = established)
    assert body["chat_only"] is True
    assert body["chat_only_reason"] == reason
    assert body["cpu_training"] is expected


def _route():
    spec = importlib.util.spec_from_file_location(
        "training_route_cpu_host", _BACKEND / "routes" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(**overrides) -> TrainingStartRequest:
    payload = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "hf_dataset": "org/dataset",
    }
    payload.update(overrides)
    return TrainingStartRequest(**payload)


@pytest.fixture
def cpu_training_host(monkeypatch):
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "_CPU_TRAINING_REASON", "no_gpu")
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))


@pytest.mark.parametrize(
    "overrides, named",
    [
        (dict(is_dataset_image = True), "Vision and audio training"),
        (dict(is_dataset_audio = True), "Vision and audio training"),
        (dict(is_embedding = True), "Embedding model training"),
        (dict(is_decision = True), "Decision model training"),
        (dict(drafter = DrafterTrainingConfig(enabled = True)), "Drafter training"),
    ],
)
def test_the_start_route_refuses_what_the_cpu_backend_cannot_train(
    cpu_training_host, overrides, named
):
    with pytest.raises(HTTPException) as exc_info:
        _route()._validate_training_platform(_request(**overrides))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail.startswith(f"{named} is not supported on CPU yet.")


@pytest.mark.parametrize(
    "training_type", ["LoRA/QLoRA", "Full Finetuning", "Continued Pretraining"]
)
def test_the_start_route_admits_text_training_on_cpu(cpu_training_host, training_type):
    _route()._validate_training_platform(_request(training_type = training_type))


def test_a_gpu_host_is_not_held_to_the_cpu_limits(monkeypatch):
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA)
    _route()._validate_training_platform(_request(is_embedding = True))


def _run_worker_until_dispatch(monkeypatch, *, reason, established):
    """Runs run_training_process() up to its first backend dispatch and returns the environment it left."""
    from core.training import worker

    monkeypatch.delenv("UNSLOTH_FORCE_CPU", raising = False)
    monkeypatch.setattr(hw, "detect_hardware", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "_CPU_TRAINING_REASON", established)
    monkeypatch.setattr(worker, "_validate_training_worker_config", lambda *_a: True)
    monkeypatch.setattr(worker, "_escaping_weights_error", lambda _c: None)
    monkeypatch.setattr(worker, "apply_gpu_ids", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_training_job_is_local", lambda _c: True)
    import core.training.training as training_mod

    monkeypatch.setattr(training_mod, "is_apple_silicon_training_platform", lambda: False)

    # The first step past the MLX dispatch; stopping there leaves only the environment to inspect.
    monkeypatch.setattr(
        worker,
        "_activate_transformers_version",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("stop here")),
    )
    events = []
    worker.run_training_process(
        event_queue = _Queue(events), stop_queue = _Queue([]), config = {"model_name": "unsloth/test"}
    )
    assert [event["error"] for event in events] == [
        "Failed to activate transformers version: stop here"
    ]
    return os.environ.get("UNSLOTH_FORCE_CPU")


class _Queue:
    def __init__(self, items):
        self.put = items.append


def test_the_worker_trains_on_cpu_where_studio_enabled_train(monkeypatch):
    # The parent's UNSLOTH_DISABLE_CPU_FALLBACK=1 would make `import unsloth` refuse this host.
    assert _run_worker_until_dispatch(monkeypatch, reason = "no_gpu", established = "no_gpu") == "1"


@pytest.mark.parametrize("reason, established", [("torch_cpu_build", "no_gpu"), ("no_gpu", None)])
def test_the_worker_leaves_every_other_host_alone(monkeypatch, reason, established):
    assert _run_worker_until_dispatch(monkeypatch, reason = reason, established = established) is None
