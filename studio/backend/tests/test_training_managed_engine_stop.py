# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A managed engine that did not stop still holds its GPUs: training must refuse to spawn
rather than log the failure and start anyway."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routes.training as training_route  # noqa: E402
import routes.training_vram as training_vram  # noqa: E402
from routes.training_vram import ManagedEngineStillRunning  # noqa: E402
from core.training.training import TrainingBackend  # noqa: E402
from test_training_before_spawn import _start  # noqa: E402


def test_a_stuck_engine_stops_the_spawn(monkeypatch):
    process = MagicMock()
    monkeypatch.setattr("core.training.training._CTX.Process", process)

    def hook():
        raise ManagedEngineStillRunning("The inference engine could not be stopped.")

    with pytest.raises(ManagedEngineStillRunning):
        _start(TrainingBackend(), hook)
    process.assert_not_called()


def test_any_other_hook_failure_is_still_best_effort():
    def hook():
        raise OSError("a GGUF server that was already gone")

    assert _start(TrainingBackend(), hook) is True


def _stuck_backend():
    return SimpleNamespace(
        active_model_name = "m",
        loading_models = set(),
        models = {"m": {}},
        _managed_engine = object(),
        _shutdown_subprocess = lambda *a, **k: False,
    )


def test_free_chat_models_raises_the_blocking_error(monkeypatch):
    import core.inference

    monkeypatch.setattr(core.inference, "get_inference_backend", _stuck_backend)
    with pytest.raises(ManagedEngineStillRunning):
        training_vram.free_chat_models_for_training(reason = "test")


def test_diffusion_training_refuses_too(monkeypatch):
    def stuck(reason):
        raise ManagedEngineStillRunning("The inference engine could not be stopped.")

    monkeypatch.setattr(training_vram, "summarize_resident_chat", lambda: {"any": True})
    monkeypatch.setattr(training_vram, "free_chat_models_for_training", stuck)
    with pytest.raises(ManagedEngineStillRunning):
        training_route._free_gpu_for_diffusion_training()
