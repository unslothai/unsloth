# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""weight_decay must be a finite number in [0, 10]; NaN used to slip through the range check."""

import pytest
from pydantic import ValidationError

from models.training import TrainingStartRequest


def _request(**overrides):
    return TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        **overrides,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 10.5])
def test_weight_decay_rejects_non_finite_and_out_of_range(value):
    with pytest.raises(ValidationError):
        _request(weight_decay = value)


@pytest.mark.parametrize("value", [0, 0.001, 10])
def test_weight_decay_accepts_range(value):
    assert _request(weight_decay = value).weight_decay == value
