# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

from core.training.startup_progress import (
    ACTIVE,
    STARTING,
    WAITING_FOR_LOSS,
    TrainingStartupProgress,
)


def _start(step = 0, accumulation = 4):
    progress = TrainingStartupProgress()
    assert (
        progress.begin(
            SimpleNamespace(gradient_accumulation_steps = accumulation),
            SimpleNamespace(global_step = step),
        )
        == STARTING
    )
    return progress


def test_accumulation_batches_advance_before_the_first_optimizer_step():
    progress = _start()
    assert "batch 1 of up to 4" in progress.step_begin()
    for batch in (2, 3, 4):
        assert f"batch {batch} of up to 4" in progress.substep_end()
    assert progress.step_end(SimpleNamespace(global_step = 1)) == WAITING_FOR_LOSS
    assert progress.step_begin() is None
    assert progress.substep_end() is None
    assert progress.step_end(SimpleNamespace(global_step = 2)) is None


def test_resumed_step_is_measured_relative_to_the_session():
    progress = _start(step = 900, accumulation = 1)
    assert "batch 1 of up to 1" in progress.step_begin()
    assert progress.step_end(SimpleNamespace(global_step = 900)) is None
    assert progress.step_end(SimpleNamespace(global_step = 901)) == WAITING_FOR_LOSS


def test_first_loss_clears_startup_once_including_zero_loss():
    progress = _start()
    assert progress.log({"eval_loss": 2.0}) is None
    assert progress.log({"train_loss": 1.0}) is None
    assert progress.log({"loss": 0.0}) == ACTIVE
    assert progress.log({"loss": 0.5}) is None
    assert progress.step_end(SimpleNamespace(global_step = 1)) is None
    assert progress.step_begin() is None


def test_partial_accumulation_may_finish_before_the_target_batch_count():
    progress = _start(accumulation = 4)
    assert "up to 4" in progress.step_begin()
    progress.substep_end()
    assert progress.step_end(SimpleNamespace(global_step = 1)) == WAITING_FOR_LOSS
    assert progress.log({"loss": 1.0}) == ACTIVE


def test_callbacks_before_train_begin_do_not_claim_active_training():
    progress = TrainingStartupProgress()
    assert progress.log({"eval_loss": 1.0}) is None
    assert progress.log({"loss": 1.0}) is None
    assert progress.step_begin() is None
    assert progress.substep_end() is None
    assert progress.step_end(SimpleNamespace(global_step = 10)) is None
    assert (
        progress.begin(
            SimpleNamespace(gradient_accumulation_steps = 4), SimpleNamespace(global_step = 0)
        )
        == STARTING
    )
    assert "batch 1 of up to 4" in progress.step_begin()
    assert progress.log({"eval_loss": 1.0}) is None
    assert progress.log({"loss": 1.0}) == ACTIVE
