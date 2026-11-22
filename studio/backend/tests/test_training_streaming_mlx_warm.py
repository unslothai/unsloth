# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A streaming start must force detection first, since DEVICE is None during warm-up."""

from __future__ import annotations

import platform
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.training as training_routes
from auth.authentication import authenticated_via_api_key, get_current_subject
from utils.hardware import hardware as hw

# Clears every streaming precondition; load_in_4bit off keeps the sidecar probe offline.
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
    """Stubs the Hub preflights, since start_training would otherwise reach huggingface.co."""
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


@pytest.mark.parametrize("detected", [False, True], ids = ["warm-window", "detected"])
def test_streaming_starts_on_mlx(monkeypatch, hardware_globals, spawn_calls, client, detected):
    """The MLX worker streams too, whether or not the warm thread has detected the device yet."""
    _pretend_apple_silicon(monkeypatch)
    hardware_globals.DEVICE = hw.DeviceType.MLX if detected else None

    response = client.post("/training/start", json = _STREAMING_START)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert hardware_globals.DEVICE == hw.DeviceType.MLX
    spawn_calls.start_training.assert_called_once()


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
