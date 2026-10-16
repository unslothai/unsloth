# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The worker files activation records under the run, so its config has to carry the run id."""

from unittest.mock import patch

from core.training.training import TrainingBackend


class _DummyProcess:
    pid = 4321

    def start(self):
        return None


class _DummyThread:
    def start(self):
        return None


def test_the_worker_config_carries_the_job_id():
    seen = {}

    def spec(module, function, cache_env, payload):
        seen.update(payload["config"])
        return (), {}

    with (
        patch("core.training.training.prepare_gpu_selection", return_value = ([0], {})),
        patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
        patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
        patch("core.training.training.threading.Thread", return_value = _DummyThread()),
        patch("core.training.training.account_process_spec", side_effect = spec),
    ):
        assert TrainingBackend().start_training(
            job_id = "run-123",
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            enable_activation_capture = True,
        )
    assert seen["job_id"] == "run-123"
