# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO compute_loss logs the clip_ratio metrics for loss_type luspo like TRL does."""

from __future__ import annotations

import pytest

from tests.test_grpo_metrics_bucket import _compute_loss, _inputs, _trainer

CLIP_KEYS = {
    "clip_ratio/low_mean",
    "clip_ratio/low_min",
    "clip_ratio/high_mean",
    "clip_ratio/high_max",
    "clip_ratio/region_mean",
}


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"])
def test_clip_ratio_metrics_logged(loss_type):
    trainer = _trainer(training = True, should_evaluate = False)
    trainer.loss_type = trainer.args.loss_type = loss_type
    _compute_loss(4.0)(trainer, trainer.model, _inputs())
    assert CLIP_KEYS <= set(trainer._metrics["train"])
