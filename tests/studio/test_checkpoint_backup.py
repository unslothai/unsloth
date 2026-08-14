import sys
import time
import types
from pathlib import Path

import pytest
from pydantic import ValidationError


BACKEND = Path(__file__).parents[2] / "studio" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
# The production training package eagerly imports optional runtime logging dependencies.
# Keep this unit test focused on the isolated backup package.
training_package = types.ModuleType("core.training")
training_package.__path__ = [str(BACKEND / "core" / "training")]
sys.modules.setdefault("core.training", training_package)

from core.training.checkpoint_backup.config import CheckpointBackupConfig
from core.training.checkpoint_backup.manager import CheckpointBackupManager
from core.training.checkpoint_backup.manifest import build_resume_manifest, upload_files


def enabled(interval: int = 400, **changes) -> CheckpointBackupConfig:
    values = dict(enabled = True, repo_id = "user/checkpoints", interval_steps = interval)
    values.update(changes)
    return CheckpointBackupConfig(**values)


@pytest.mark.parametrize(("interval", "eligible"), [
    (200, [200, 400, 600]),
    (400, [400, 800]),
    (600, [600, 1200]),
])
def test_cadence_is_independent_positive_multiple(interval, eligible):
    config = enabled(interval).validate_for_save_steps(200)
    assert [step for step in range(200, max(eligible) + 1, 200) if step % config.interval_steps == 0] == eligible


def test_invalid_cadence_is_not_rounded():
    with pytest.raises(ValueError, match = "must be a multiple"):
        enabled(300).validate_for_save_steps(200)


def test_periodic_backup_requires_local_checkpoints():
    with pytest.raises(ValueError, match = "greater than zero"):
        enabled().validate_for_save_steps(0)


def test_disabled_and_legacy_config_remain_valid():
    assert CheckpointBackupConfig().enabled is False
    assert CheckpointBackupConfig(enabled = False, interval_steps = 3).validate_for_save_steps(0)


def test_private_default_and_no_token_field():
    config = enabled()
    assert config.private is True
    assert "token" not in config.model_dump()
    with pytest.raises(ValidationError):
        CheckpointBackupConfig(enabled = True, repo_id = "../escape", interval_steps = 200)


def test_manifest_allowlist_and_sensitive_upload_filter(tmp_path):
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}")
    (checkpoint / "studio.db").write_text("secret")
    (checkpoint / "hf_token.txt").write_text("secret")
    manifest = build_resume_manifest({"run_id": "r1", "global_step": 200, "hf_token": "secret"})
    assert manifest == {"run_id": "r1", "global_step": 200}
    assert [p.name for p in upload_files(checkpoint)] == ["trainer_state.json"]


class RecordingTransport:
    def __init__(self, *, fail_once = False):
        self.steps = []
        self.fail_once = fail_once

    def upload_checkpoint(self, run_id, path, progress):
        self.steps.append(int(path.name.split("-")[-1]))
        if self.fail_once:
            self.fail_once = False
            raise OSError("network lost")
        progress(1, 1, 10, 10)


def checkpoint(root: Path, step: int) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text("{}")
    return path


def wait_status(manager, run_id, expected, timeout = 2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.progress.snapshot(run_id)
        if snapshot.status == expected:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(manager.progress.snapshot(run_id))


def test_manager_retries_without_failing_training(tmp_path):
    transport = RecordingTransport(fail_once = True)
    manager = CheckpointBackupManager(
        enabled(200), tmp_path, transport,
        checkpoint_validator = lambda path: (path / "trainer_state.json").is_file(),
        backoff_seconds = 0.01,
    )
    path = checkpoint(tmp_path, 200)
    assert manager.on_checkpoint_saved("run", 200, path)
    result = wait_status(manager, "run", "success")
    assert result.attempt == 2
    assert result.progress_percent == 100
    assert manager.shutdown()


def test_manager_rejects_incomplete_and_outside_paths(tmp_path):
    manager = CheckpointBackupManager(
        enabled(200), tmp_path, RecordingTransport(),
        checkpoint_validator = lambda path: (path / "trainer_state.json").is_file(),
    )
    incomplete = tmp_path / "checkpoint-200.part"
    incomplete.mkdir()
    outside = tmp_path.parent / "checkpoint-200"
    outside.mkdir(exist_ok = True)
    assert not manager.on_checkpoint_saved("run", 200, incomplete)
    assert not manager.on_checkpoint_saved("run", 200, outside)
    assert manager.shutdown()
