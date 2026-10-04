# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from pydantic import ValidationError

from models.training import TrainingStartRequest


def request(**overrides):
    return TrainingStartRequest(
        model_name = "unsloth/Qwen3.5-4B",
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        **overrides,
    )


def test_auto_parallelism_accepts_no_explicit_gpu_ids():
    assert request().parallelism_mode == "auto"
    assert request(gpu_ids = []).gpu_ids == []


def test_auto_parallelism_rejects_explicit_gpu_ids():
    with pytest.raises(ValidationError, match = "parallelism_mode='auto'"):
        request(gpu_ids = [0])


def test_single_parallelism_requires_one_gpu():
    assert request(parallelism_mode = "single", gpu_ids = [1]).gpu_ids == [1]
    with pytest.raises(ValidationError, match = "exactly one"):
        request(parallelism_mode = "single", gpu_ids = [0, 1])


def test_model_parallel_requires_two_gpus():
    assert request(parallelism_mode = "model_parallel", gpu_ids = [0, 1]).gpu_ids == [0, 1]
    with pytest.raises(ValidationError, match = "at least two"):
        request(parallelism_mode = "model_parallel", gpu_ids = [0])
