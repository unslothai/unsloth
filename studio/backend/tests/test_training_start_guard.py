# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""start_training() must report busy during the pre-spawn prep window."""

from unittest.mock import patch

import pytest

from core.training.training import TrainingBackend
from storage import studio_db
from utils.hardware import DeviceType


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


class _DummyProcess:
    pid = 4321

    def start(self):
        return None


class _DummyThread:
    def start(self):
        return None


def _start(backend, hook = None):
    with (
        patch("core.training.training.prepare_gpu_selection", return_value = ([0], {})),
        patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
        patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
        patch("core.training.training.threading.Thread", return_value = _DummyThread()),
    ):
        return backend.start_training(
            job_id = "start-guard-test",
            before_spawn = hook,
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
        )


def test_busy_during_prep_and_concurrent_start_refused():
    backend = TrainingBackend()
    seen = {}

    def hook():
        seen["active"] = backend.is_training_active()
        seen["second_start"] = backend.start_training(
            job_id = "concurrent",
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
        )

    assert _start(backend, hook) is True
    assert seen["active"] is True
    assert seen["second_start"] is False
    assert backend._start_in_flight is False


def test_flag_cleared_when_prep_raises():
    backend = TrainingBackend()
    with (
        patch("utils.hardware.hardware.DEVICE", DeviceType.CUDA),
        patch(
            "core.training.training.prepare_gpu_selection",
            side_effect = ValueError("Invalid gpu_ids [99]"),
        ),
    ):
        with pytest.raises(ValueError):
            backend.start_training(
                job_id = "start-guard-test",
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                gpu_ids = [99],
            )
    assert backend._start_in_flight is False
    assert backend.is_training_active() is False
