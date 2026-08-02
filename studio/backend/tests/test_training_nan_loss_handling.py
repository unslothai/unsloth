# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pin Unsloth's behavior when a training event reports non-finite (NaN/Inf) loss.

The training event handler used to filter NaN/Inf to None silently while
leaving the previous finite loss in progress.loss — so the API kept reporting
the stale value as if everything were fine. We now drop the stale value:
clients see loss=None at the affected step and a one-shot warning is logged.
Training continues; the run is not marked failed.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.training import TrainingBackend


def _make_backend() -> TrainingBackend:
    return TrainingBackend()


def _progress_event(
    step: int,
    loss: float,
    lr: float = 1e-4,
) -> dict:
    return {
        "type": "progress",
        "step": step,
        "loss": loss,
        "learning_rate": lr,
        "epoch": 0.0,
        "total_steps": 100,
    }


class TestNonfiniteLossSoftHandling:
    def test_finite_loss_updates_progress_normally(self):
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = 0.97))
        assert b._progress.loss == pytest.approx(0.97)
        assert b._progress.error is None
        assert b._should_stop is False
        assert getattr(b._progress, "_nonfinite_loss_warned", False) is False

    def test_nan_loss_clears_progress_loss(self):
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = 0.97))
        assert b._progress.loss == pytest.approx(0.97)
        b._handle_event(_progress_event(step = 2, loss = float("nan")))
        # Stale finite loss must NOT leak through
        assert b._progress.loss is None
        # Run is not marked failed
        assert b._progress.error is None
        assert b._should_stop is False
        # Warning flag is set so we don't re-log on every subsequent NaN step
        assert b._progress._nonfinite_loss_warned is True

    def test_inf_loss_clears_progress_loss(self):
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = float("inf")))
        assert b._progress.loss is None
        assert b._progress.error is None
        assert b._should_stop is False
        assert b._progress._nonfinite_loss_warned is True

    def test_negative_inf_loss_clears_progress_loss(self):
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = float("-inf")))
        assert b._progress.loss is None
        assert b._progress.error is None
        assert b._should_stop is False
        assert b._progress._nonfinite_loss_warned is True

    def test_repeated_nan_only_warns_once(self):
        """Subsequent NaN events must not re-fire the warning flag setter.
        The flag should already be True after the first NaN."""
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = 0.97))
        b._handle_event(_progress_event(step = 2, loss = float("nan")))
        assert b._progress._nonfinite_loss_warned is True
        # Further NaN steps don't change anything we care about
        b._handle_event(_progress_event(step = 3, loss = float("nan")))
        b._handle_event(_progress_event(step = 4, loss = float("nan")))
        assert b._progress._nonfinite_loss_warned is True
        assert b._progress.loss is None
        assert b._progress.error is None
        assert b._should_stop is False

    def test_recovery_updates_loss_when_finite_again(self):
        """If a NaN step is followed by a finite step, progress.loss must
        reflect the new finite value (not stay stuck at None)."""
        b = _make_backend()
        b._handle_event(_progress_event(step = 1, loss = 0.97))
        b._handle_event(_progress_event(step = 2, loss = float("nan")))
        assert b._progress.loss is None
        b._handle_event(_progress_event(step = 3, loss = 0.85))
        assert b._progress.loss == pytest.approx(0.85)
        # Warning flag stays set (we don't reset it on recovery)
        assert b._progress._nonfinite_loss_warned is True
