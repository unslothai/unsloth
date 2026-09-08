# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A status-only update must not replay the previous step's metrics.

UnslothTrainer keeps metrics and status on one TrainingProgress and notifies its
callbacks on every change, so publishing an evaluation status carries the last
logged step's loss, learning rate, grad norm and eval loss along with it. The
parent appends every progress event to loss_history / grad_norm_history /
eval_loss_history and to the metric buffer it persists, without deduplicating the
step, so a long evaluation would plot the same point once per status line.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


class _Progress:
    """The fields _create_trainer_progress_callback reads off TrainingProgress."""

    def __init__(self, **fields):
        self.step = 0
        self.total_steps = 0
        self.loss = None
        self.learning_rate = None
        self.grad_norm = None
        self.num_tokens = None
        self.epoch = None
        self.eval_loss = None
        self.elapsed_seconds = None
        self.status_message = ""
        for key, value in fields.items():
            setattr(self, key, value)


def _emitter():
    """The publish rule from worker._create_trainer_progress_callback, returning the
    steps it would have published as metric events."""
    last_metrics: list = [None]
    published: list = []

    def _on_progress(p) -> None:
        has_train_loss = p.step > 0 and p.loss is not None
        has_eval_loss = p.eval_loss is not None
        metrics = (
            p.step,
            p.loss,
            p.learning_rate,
            p.grad_norm,
            p.num_tokens,
            p.epoch,
            p.eval_loss,
        )
        is_repeat = metrics == last_metrics[0]
        if (
            (p.step == 0 and p.total_steps > 0) or has_train_loss or has_eval_loss
        ) and not is_repeat:
            last_metrics[0] = metrics
            published.append(p.step)

    return _on_progress, published


def test_evaluation_status_lines_do_not_replot_the_last_step():
    # A 4-minute evaluation after step 200 publishes a status roughly every 15s; each
    # one arrives with step 200's loss still on the shared progress object.
    on_progress, published = _emitter()
    step_200 = _Progress(step = 200, total_steps = 1000, loss = 0.42, learning_rate = 1e-4)
    on_progress(step_200)
    for seen in (8, 24, 40, 56):
        step_200.status_message = f"Evaluating... {seen} batches"
        step_200.elapsed_seconds = 900.0 + seen
        on_progress(step_200)
    step_200.status_message = "Training in progress..."
    on_progress(step_200)
    assert published == [200]


def test_a_new_step_is_still_published():
    on_progress, published = _emitter()
    on_progress(_Progress(step = 200, total_steps = 1000, loss = 0.42))
    on_progress(_Progress(step = 201, total_steps = 1000, loss = 0.41))
    assert published == [200, 201]


def test_the_same_step_with_a_new_measurement_is_still_published():
    # Evaluation ends and reports eval_loss while global_step has not moved yet; that
    # is a real new number, not a replay.
    on_progress, published = _emitter()
    on_progress(_Progress(step = 200, total_steps = 1000, loss = 0.42))
    on_progress(_Progress(step = 200, total_steps = 1000, loss = 0.42, eval_loss = 0.55))
    assert published == [200, 200]


def test_a_warning_mid_run_does_not_replot_either():
    # _record_warning notifies the same callbacks with the metrics untouched.
    on_progress, published = _emitter()
    progress = _Progress(step = 12, total_steps = 100, loss = 1.5, grad_norm = 0.9)
    on_progress(progress)
    on_progress(progress)
    assert published == [12]


def test_the_worker_publishes_only_changed_measurements():
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    body = text[text.index("def _create_trainer_progress_callback") :]
    body = body[: body.index("def _create_embedding_progress_callback")]
    assert "is_repeat = metrics == last_metrics[0]" in body
    assert "and not is_repeat" in body
    # Wall-clock fields move on every call and would defeat the comparison.
    assert "progress.elapsed_seconds," not in body[: body.index("event_queue.put")]
