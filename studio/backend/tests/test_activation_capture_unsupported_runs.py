# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decision, embedding and MLX runs never wire the capture callback, so their config must not claim it."""

from types import SimpleNamespace

import pytest

import core.training.training as training
from core.training.training import _build_training_worker_config


@pytest.fixture
def host(monkeypatch):
    def pin(device, apple_silicon):
        monkeypatch.setattr(training, "get_device", lambda: SimpleNamespace(value = device))
        monkeypatch.setattr(training, "is_apple_silicon_training_platform", lambda: apple_silicon)

    pin("cuda", False)
    return pin


def _config(**extra):
    return _build_training_worker_config(
        {"model_name": "m", "enable_activation_capture": True, **extra}
    )


@pytest.mark.parametrize("kind", ["is_embedding", "is_decision"])
def test_capture_is_off_for_runs_without_the_callback(host, kind):
    assert _config(**{kind: True})["enable_activation_capture"] is False


def test_capture_stays_on_for_a_text_run(host):
    assert _config()["enable_activation_capture"] is True


@pytest.mark.parametrize("device, apple_silicon", [("mlx", True), ("cpu", True), ("mlx", False)])
def test_capture_is_off_on_the_mlx_trainer(host, device, apple_silicon):
    host(device, apple_silicon)
    assert _config()["enable_activation_capture"] is False
