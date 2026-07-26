# SPDX-License-Identifier: AGPL-3.0-only
"""Static contract coverage for the credential-free upload lifecycle."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util

_spec = importlib.util.spec_from_file_location("training_models", Path(__file__).resolve().parent.parent / "models" / "training.py")
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
CheckpointUploadProgress = _module.CheckpointUploadProgress
TrainingProgress = _module.TrainingProgress


@pytest.mark.parametrize("state", ["idle", "preparing", "uploading", "completed", "skipped", "error"])
def test_upload_lifecycle_states_and_unknown_totals(state):
    upload = CheckpointUploadProgress(state=state, message="safe")
    assert upload.state == state
    assert upload.total_bytes is None
    assert upload.percentage is None


def test_upload_percentage_is_bounded():
    for percentage in (-1, 101):
        with pytest.raises(ValidationError):
            CheckpointUploadProgress(state="uploading", percentage=percentage)


def test_upload_error_does_not_become_training_error():
    progress = TrainingProgress(
        job_id="job", step=1, total_steps=1, progress_percent=100,
        checkpoint_upload={"state": "error", "message": "Upload failed", "error": "Network unavailable"},
    )
    assert progress.checkpoint_upload.state == "error"
    assert not hasattr(progress, "error")


def test_checkpoint_upload_preserves_checkpoint_directory():
    """Regression: checkpoint-30 files must not be flattened into repo root."""
    trainer_source = (
        Path(__file__).resolve().parent.parent / "core" / "training" / "trainer.py"
    ).read_text()
    assert "path_in_repo = checkpoint.name" in trainer_source
    assert "folder_path = str(checkpoint)" in trainer_source
    assert 'config["push_to_hub"] = False' in trainer_source


def test_completed_checkpoint_upload_does_not_stick_in_training_status():
    """Terminal Hub progress must restore the header and ignore closed tqdm bars."""
    worker_source = (
        Path(__file__).resolve().parent.parent / "core" / "training" / "worker.py"
    ).read_text()
    assert '_send_status(event_queue, "Training...")' in worker_source
    assert '0 < n < total and desc' in worker_source


def test_disabled_checkpoint_upload_does_not_emit_skipped_event():
    """Resuming locally must not surface a misleading upload notification."""
    trainer_source = (
        Path(__file__).resolve().parent.parent / "core" / "training" / "trainer.py"
    ).read_text()
    disabled_branch = trainer_source.split("if not push_to_hub:", 1)[1].split(
        "if not repository_id:", 1
    )[0]
    assert "emit(" not in disabled_branch
    assert "return" in disabled_branch
