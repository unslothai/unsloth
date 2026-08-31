# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

from core.training.training import TrainingBackend


_LIVE_OUTPUT_DIR = Path("outputs") / "live"
_CANCEL_OUTPUT_DIR = Path("outputs") / "cancel"
_CONFIG_OUTPUT_DIR = Path("outputs") / "config"
_RESUME_OUTPUT_DIR = Path("outputs") / "resume"
_RESUME_CHECKPOINT = _RESUME_OUTPUT_DIR / "checkpoint-5"


@pytest.mark.parametrize(
    (
        "live_output_dir",
        "cancel_cleanup_output_dir",
        "config_output_dir",
        "resume_from_checkpoint",
        "expected",
    ),
    [
        (
            _LIVE_OUTPUT_DIR,
            str(_CANCEL_OUTPUT_DIR),
            str(_CONFIG_OUTPUT_DIR),
            str(_RESUME_CHECKPOINT),
            str(_LIVE_OUTPUT_DIR),
        ),
        (
            None,
            _CANCEL_OUTPUT_DIR,
            str(_CONFIG_OUTPUT_DIR),
            str(_RESUME_CHECKPOINT),
            str(_CANCEL_OUTPUT_DIR),
        ),
        (
            None,
            None,
            _CONFIG_OUTPUT_DIR,
            str(_RESUME_CHECKPOINT),
            str(_CONFIG_OUTPUT_DIR),
        ),
        (
            None,
            None,
            None,
            str(_RESUME_CHECKPOINT),
            str(_RESUME_OUTPUT_DIR),
        ),
        (None, None, None, None, None),
    ],
)
def test_active_output_dir_uses_live_state_precedence(
    monkeypatch,
    live_output_dir,
    cancel_cleanup_output_dir,
    config_output_dir,
    resume_from_checkpoint,
    expected,
):
    backend = TrainingBackend()
    monkeypatch.setattr(backend, "is_training_active", lambda: True)
    backend._output_dir = live_output_dir
    backend._cancel_cleanup_output_dir = cancel_cleanup_output_dir
    backend._db_config = {
        "output_dir": config_output_dir,
        "resume_from_checkpoint": resume_from_checkpoint,
    }

    assert backend.active_output_dir() == expected


def test_active_output_dir_hides_inactive_run(monkeypatch):
    backend = TrainingBackend()
    monkeypatch.setattr(backend, "is_training_active", lambda: False)
    backend._output_dir = str(_LIVE_OUTPUT_DIR)
    backend._cancel_cleanup_output_dir = str(_CANCEL_OUTPUT_DIR)
    backend._db_config = {
        "output_dir": str(_CONFIG_OUTPUT_DIR),
        "resume_from_checkpoint": str(_RESUME_CHECKPOINT),
    }

    assert backend.active_output_dir() is None


def test_active_output_dir_keeps_cancel_cleanup_path_while_worker_exits():
    class ActiveProcess:
        @staticmethod
        def is_alive():
            return True

    backend = TrainingBackend()
    backend._proc = ActiveProcess()
    backend._should_stop = True
    backend._cancel_requested = True
    backend._cancel_cleanup_output_dir = str(_CANCEL_OUTPUT_DIR)
    backend._db_config = {}

    assert backend.active_output_dir() == str(_CANCEL_OUTPUT_DIR)
