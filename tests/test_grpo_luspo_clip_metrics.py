# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO compute_loss logs the clip_ratio metrics for loss_type luspo like TRL does."""

from __future__ import annotations

import pytest
import torch

from tests.test_grpo_metrics_bucket import BATCH, _compute_loss, _inputs, _trainer


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"])
def test_clip_ratio_metrics_logged(loss_type):
    trainer = _trainer(training = True, should_evaluate = False)
    trainer.loss_type = trainer.args.loss_type = loss_type
    # luspo is sequence-level: zoo returns coef_1 as (B, 1). Row 0 low-clipped, row 1 high-clipped.
    coef_1 = torch.tensor([[0.5], [1.5]])
    inputs = _inputs()
    inputs["advantages"] = torch.tensor([-1.0, 1.0])
    assert coef_1.shape[0] == BATCH
    _compute_loss(4.0, coef_1)(trainer, trainer.model, inputs)
    trainer.log({})
    metrics = trainer._metrics["train"]
    # TRL global_masked_mean on a (B, 1) tensor: sum / B.
    assert metrics["clip_ratio/low_mean"] == [0.5]
    assert metrics["clip_ratio/high_mean"] == [0.5]
    assert metrics["clip_ratio/region_mean"] == [1.0]
    assert {"clip_ratio/low_min", "clip_ratio/high_max"} <= set(metrics)
