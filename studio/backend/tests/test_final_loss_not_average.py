# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The run's mean loss must not be reported as the final step's loss.

HF logs the end-of-run summary as {"train_runtime": ..., "train_loss": <mean>}
with no "loss" key. `logs.get("loss", logs.get("train_loss"))` therefore fell back
to the mean and published it at the same global_step as the real last step, so:

  - the loss chart gained points stacked on the final step, the last of them the
    run average (a 30 step run charted 33 points, ending 0.3205, 0.3205, 0.3834),
  - `final_loss` on /api/train/runs became the average while
    /api/models/checkpoints reported the true last-step loss for the same run,
  - the UI stat card showed the average, so loss appeared to jump on the last step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _extract_loss(logs: dict):
    """The corrected reading of an HF on_log record: only a real per-step loss."""
    return logs.get("loss")


def test_a_step_record_still_reports_its_loss():
    logs = {"loss": 0.3205, "grad_norm": 0.4, "learning_rate": 1e-5, "epoch": 1.0}
    assert _extract_loss(logs) == 0.3205


def test_the_summary_record_reports_no_step_loss():
    logs = {"train_runtime": 23.18, "train_loss": 0.3834, "train_samples_per_second": 5.2}
    assert _extract_loss(logs) is None


class _History:
    """The append rule from TrainingManager's event pump."""

    def __init__(self):
        self.steps: list[int] = []
        self.loss: list[float] = []

    def offer(self, step, loss):
        last = self.steps[-1] if self.steps else None
        if step > 0 and loss is not None and (last is None or step > last):
            self.steps.append(step)
            self.loss.append(loss)


def test_series_ignores_repeats_at_the_same_step():
    h = _History()
    for step, loss in [(28, 0.27), (29, 0.32), (30, 0.3205), (30, 0.3205), (30, None)]:
        h.offer(step, loss)
    assert h.steps == [28, 29, 30]
    assert h.loss[-1] == 0.3205


def test_series_never_ends_on_the_average():
    h = _History()
    # The exact tail a 30 step run produced before the fix.
    for step, loss in [(30, 0.3205), (30, 0.3205), (30, 0.3834), (30, 0.3834)]:
        h.offer(step, loss)
    assert h.steps == [30]
    assert h.loss == [0.3205]


def test_a_step_zero_record_is_still_ignored():
    h = _History()
    h.offer(0, 1.23)
    assert h.steps == []


def test_normal_monotonic_run_is_unchanged():
    h = _History()
    for step in range(1, 31):
        h.offer(step, 1.0 / step)
    assert h.steps == list(range(1, 31))
    assert len(h.loss) == 30


def test_the_shipped_call_sites_no_longer_fall_back_to_train_loss():
    # Guard the actual source: the fallback is what caused this.
    for rel in ("core/training/trainer.py", "core/training/worker.py"):
        text = (_BACKEND / rel).read_text(encoding = "utf-8")
        assert 'logs.get("loss", logs.get("train_loss", None))' not in text, rel


def test_the_terminal_summary_still_reports_elapsed_time():
    # The summary record has no step loss, so the progress filter dropped it; the
    # elapsed time it carries (final eval, checkpoint save, best-model reload) is the
    # run's real duration and must still reach the parent.
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from core.training.worker import _create_trainer_progress_callback

    class _P:
        step = 30
        total_steps = 30
        loss = None
        eval_loss = None
        epoch = 3.0
        learning_rate = 0.0
        elapsed_seconds = 412.5
        eta_seconds = None
        grad_norm = None
        num_tokens = 12345
        status_message = ""
        warnings: list = []

    events = []

    class _Q:
        def put(self, e):
            events.append(e)

    _create_trainer_progress_callback(_Q())(_P())
    progress = [e for e in events if e.get("type") == "progress"]
    assert progress, events
    assert progress[0]["elapsed_seconds"] == 412.5
    assert progress[0]["loss"] is None


def test_a_lossless_mid_run_record_is_still_dropped():
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from core.training.worker import _create_trainer_progress_callback

    class _P:
        step = 12
        total_steps = 30
        loss = None
        eval_loss = None
        epoch = 1.0
        learning_rate = 0.0
        elapsed_seconds = 40.0
        eta_seconds = None
        grad_norm = None
        num_tokens = 1
        status_message = ""
        warnings: list = []

    events = []

    class _Q:
        def put(self, e):
            events.append(e)

    _create_trainer_progress_callback(_Q())(_P())
    assert [e for e in events if e.get("type") == "progress"] == []
