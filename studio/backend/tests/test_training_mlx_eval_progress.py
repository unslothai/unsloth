# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.training as tr
import storage.studio_db as sdb
from core.training.training import TrainingBackend


@pytest.fixture
def backend(monkeypatch):
    finished = []
    monkeypatch.setattr(tr, "account_is_retired", lambda: False)
    monkeypatch.setattr(sdb, "finish_run", lambda **kw: finished.append(kw))
    monkeypatch.setattr(sdb, "insert_metrics_batch", lambda *a, **k: None)
    monkeypatch.setattr(sdb, "update_run_progress", lambda **kw: None)
    b = TrainingBackend()
    monkeypatch.setattr(b, "_start_stop_watchdog", lambda *a, **k: None)
    b.current_job_id = "job-mlx-eval"
    b._db_run_created = True
    b._db_total_steps_set = True
    b._db_config = {"model_name": "m", "hf_dataset": "d"}
    return b, finished


def _step_event(step: int) -> dict:
    return {
        "type": "progress",
        "step": step,
        "epoch": 1.0,
        "loss": 0.9,
        "learning_rate": 1e-4,
        "total_steps": 60,
        "elapsed_seconds": 12.0,
        "eta_seconds": 0,
        "session_start_step": 0,
        "grad_norm": 0.5,
        "num_tokens": 500,
        "eval_loss": None,
        "status_message": None,
        "peak_memory_gb": 3.0,
    }


def test_final_mlx_eval_event_keeps_elapsed_and_run_duration(backend):
    b, finished = backend
    b._handle_event(_step_event(60))
    b._handle_event({"type": "progress", "step": 60, "eval_loss": 1.2})
    assert b._progress.eval_loss == 1.2
    assert b._progress.elapsed_seconds == 12.0
    assert b._progress.eta_seconds == 0
    assert b._progress.grad_norm == 0.5
    assert b._progress.num_tokens == 500
    b._handle_event({"type": "complete", "output_dir": "/tmp/out"})
    assert finished[0]["duration_seconds"] == 12.0


def test_explicit_none_still_clears_progress_fields(backend):
    b, _ = backend
    b._handle_event(_step_event(10))
    event = _step_event(11)
    event.update(grad_norm = None, num_tokens = None, eta_seconds = None)
    b._handle_event(event)
    assert b._progress.grad_norm is None
    assert b._progress.num_tokens is None
    assert b._progress.eta_seconds is None
