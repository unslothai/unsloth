# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /training/start's MLX streaming rejection must survive the warm window.

``hardware.DEVICE`` used to be set before uvicorn bound the socket. The warm thread
fills it in now, so for the first moment of serving it still holds ``None``.

``start_training`` rejects ``dataset_streaming`` on Apple Silicon by comparing ``DEVICE ==
DeviceType.MLX``. Against the default that is False, the rejection is skipped, and the
request runs on to ``_build_training_worker_config``, which detects MLX only after
validation and hands a streaming dataset to a loader that materializes the whole thing. The
guard must force detection first, and off the event loop, since detection imports torch.

The lexical half is in ``test_startup_defers_torch.py``; this file covers the behaviour.
CPU-only, no network, no GPU, no weights.
"""

from __future__ import annotations

import platform
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.training as training_routes
from auth.authentication import authenticated_via_api_key, get_current_subject
from utils.hardware import hardware as hw

# Clears every streaming precondition but the MLX guard, so only that can reject it.
# load_in_4bit is off so the latest-sidecar probe stays offline.
_STREAMING_START = {
    "model_name": "unsloth/Llama-3.2-1B-Instruct",
    "training_type": "LoRA/QLoRA",
    "format_type": "Alpaca",
    "hf_dataset": "yahma/alpaca-cleaned",
    "dataset_streaming": True,
    "max_steps": 60,
    "load_in_4bit": False,
    "eval_steps": 0,
}

_NON_STREAMING_START = {
    **_STREAMING_START,
    "dataset_streaming": False,
    "max_steps": None,
}

_MLX_REJECTION = "dataset_streaming is not yet supported on Apple Silicon (MLX)"


@pytest.fixture
def hardware_globals():
    """Restore the detection globals -- the route mutates them for real here."""
    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    try:
        yield hw
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved


@pytest.fixture(autouse = True)
def _hub_preflight_passes(monkeypatch):
    """Let the Hub preflights succeed without asking the Hub.

    This file is about the MLX guard, and its docstring already promises no network,
    but ``start_training`` verifies the model and dataset against huggingface.co on
    the way past validation. That call used to reach the real Hub, so the tests were
    quietly online and would 503 whenever it was slow or unreachable.
    """
    monkeypatch.setattr(training_routes, "_preflight_hf_dataset_request", lambda request: None)
    monkeypatch.setattr(
        training_routes,
        "_reject_untrainable_model_request",
        lambda request, *args, **kwargs: training_routes._ModelPreflightResult(
            model_name = request.model_name,
            model_local_path = None,
            cached_model_pin = None,
        ),
    )


@pytest.fixture
def spawn_calls(monkeypatch):
    """Stub the backend so a start past validation is observable without a worker."""
    backend = MagicMock()
    backend.is_training_active.return_value = False
    backend.current_job_id = ""
    backend.start_training.return_value = True
    monkeypatch.setattr(training_routes, "get_training_backend", lambda: backend)
    return backend


@pytest.fixture
def client(spawn_calls):
    app = FastAPI()
    app.include_router(training_routes.router, prefix = "/training")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app, raise_server_exceptions = False)


def _pretend_apple_silicon(monkeypatch):
    """Make detection resolve to MLX: arm64 Darwin, no torch, usable MLX stack."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)


def _pretend_cpu_linux(monkeypatch):
    """Make detection resolve to CPU: no torch, not a Mac."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)


def test_streaming_is_rejected_on_mlx_before_detection_has_run(
    monkeypatch, hardware_globals, spawn_calls, client
):
    """The regression: DEVICE is still None when the request lands."""
    _pretend_apple_silicon(monkeypatch)
    hardware_globals.DEVICE = None  # warm thread has not finished

    response = client.post("/training/start", json = _STREAMING_START)

    assert response.status_code == 400, (
        "streaming start on an Apple Silicon host was not rejected while DEVICE "
        f"was still unset (got {response.status_code}: {response.text}); the guard "
        "read the pre-detection default instead of detecting"
    )
    assert _MLX_REJECTION in response.json()["detail"]
    spawn_calls.start_training.assert_not_called()


def test_the_guard_detects_rather_than_reading_the_default(
    monkeypatch, hardware_globals, spawn_calls, client
):
    """The guard forces detection, not some later step: DEVICE goes None -> MLX across
    a request that never reaches the worker config builder."""
    _pretend_apple_silicon(monkeypatch)
    hardware_globals.DEVICE = None

    client.post("/training/start", json = _STREAMING_START)

    assert hardware_globals.DEVICE == hw.DeviceType.MLX
    spawn_calls.start_training.assert_not_called()


def test_streaming_still_starts_on_a_non_mlx_host_during_the_warm_window(
    monkeypatch, hardware_globals, spawn_calls, client
):
    """Forcing detection must not turn the guard into a blanket rejection."""
    _pretend_cpu_linux(monkeypatch)
    hardware_globals.DEVICE = None

    response = client.post("/training/start", json = _STREAMING_START)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert hardware_globals.DEVICE == hw.DeviceType.CPU
    spawn_calls.start_training.assert_called_once()


def test_rejection_still_fires_once_detection_has_already_run(
    monkeypatch, hardware_globals, spawn_calls, client
):
    """The pre-existing behaviour, unchanged: DEVICE already MLX."""
    _pretend_apple_silicon(monkeypatch)
    hardware_globals.DEVICE = hw.DeviceType.MLX

    response = client.post("/training/start", json = _STREAMING_START)

    assert response.status_code == 400
    assert _MLX_REJECTION in response.json()["detail"]
    spawn_calls.start_training.assert_not_called()


def test_non_streaming_start_detects_before_entering_the_sync_backend(
    monkeypatch, hardware_globals, spawn_calls, client
):
    """An ordinary start reaches a synchronous worker-config build that reads the
    device, so it must also detect through the route's off-loop handoff."""
    _pretend_cpu_linux(monkeypatch)
    hardware_globals.DEVICE = None

    response = client.post("/training/start", json = _NON_STREAMING_START)

    assert response.status_code == 200, response.text
    assert hardware_globals.DEVICE == hw.DeviceType.CPU
    spawn_calls.start_training.assert_called_once()
