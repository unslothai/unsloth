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
from core.training.checkpoint_backup.huggingface import HuggingFaceCheckpointTransport
from core.training.checkpoint_backup.manager import CheckpointBackupManager
from core.training.checkpoint_backup.manifest import build_resume_manifest, upload_files


def enabled(interval: int = 2, **changes) -> CheckpointBackupConfig:
    values = dict(enabled = True, repo_id = "user/checkpoints", interval_checkpoints = interval)
    values.update(changes)
    return CheckpointBackupConfig(**values)


@pytest.mark.parametrize(("interval", "eligible"), [
    (1, [200, 400, 600]),
    (2, [400, 800]),
    (3, [600, 1200]),
])
def test_cadence_is_independent_positive_multiple(interval, eligible):
    config = enabled(interval).validate_for_save_steps(200)
    cadence = config.effective_backup_steps(200)
    assert [step for step in range(200, max(eligible) + 1, 200) if step % cadence == 0] == eligible


def test_invalid_cadence_is_rejected():
    with pytest.raises(ValidationError):
        enabled(0)


def test_periodic_backup_requires_local_checkpoints():
    with pytest.raises(ValueError, match = "greater than zero"):
        enabled().validate_for_save_steps(0)


def test_disabled_and_legacy_config_remain_valid():
    assert CheckpointBackupConfig().enabled is False
    assert CheckpointBackupConfig(enabled = False, interval_checkpoints = 3).validate_for_save_steps(0)


def test_legacy_private_is_ignored_and_not_serialized():
    config = enabled()
    assert "private" not in CheckpointBackupConfig(private = True).model_dump()
    assert "token" not in config.model_dump()
    with pytest.raises(ValidationError):
        CheckpointBackupConfig(enabled = True, repo_id = "../escape", interval_checkpoints = 2)


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
        enabled(1), tmp_path, transport,
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
        enabled(1), tmp_path, RecordingTransport(),
        checkpoint_validator = lambda path: (path / "trainer_state.json").is_file(),
    )
    incomplete = tmp_path / "checkpoint-200.part"
    incomplete.mkdir()
    outside = tmp_path.parent / "checkpoint-200"
    outside.mkdir(exist_ok = True)
    assert not manager.on_checkpoint_saved("run", 200, incomplete)
    assert not manager.on_checkpoint_saved("run", 200, outside)
    assert manager.shutdown()


class FakeHubError(Exception):
    pass


class FakeRepositoryNotFoundError(FakeHubError):
    pass


class RecordingApi:
    calls = []
    info_error = None
    user = {"name": "owner", "orgs": []}

    def __init__(self, token):
        self.token = token

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        if self.info_error:
            raise self.info_error
        return types.SimpleNamespace(private = self.visibility)

    def whoami(self):
        self.calls.append(("whoami", {}))
        return self.user

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))


@pytest.fixture
def fake_hub(monkeypatch):
    RecordingApi.calls = []
    RecordingApi.info_error = None
    RecordingApi.user = {"name": "owner", "orgs": []}
    module = types.ModuleType("huggingface_hub")
    module.HfApi = RecordingApi
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = FakeHubError
    errors.RepositoryNotFoundError = FakeRepositoryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


@pytest.mark.parametrize("visibility", [True, False])
def test_public_and_private_repositories_use_the_same_upload(fake_hub, visibility, tmp_path):
    RecordingApi.visibility = visibility
    path = checkpoint(tmp_path, 1)
    transport = HuggingFaceCheckpointTransport("token", "owner/backups")
    transport.validate_access()
    transport.upload_checkpoint("run", path)

    call_names = [name for name, _ in RecordingApi.calls]
    assert call_names == ["repo_info", "whoami", "upload_file"]
    assert "create_repo" not in call_names


def test_missing_repository_is_not_created(fake_hub):
    RecordingApi.info_error = FakeRepositoryNotFoundError()
    transport = HuggingFaceCheckpointTransport("token", "owner/missing")

    with pytest.raises(ValueError, match = "Create it on Hugging Face"):
        transport.validate_access()
    assert [name for name, _ in RecordingApi.calls] == ["repo_info"]


def test_repository_outside_writable_namespaces_is_rejected(fake_hub):
    transport = HuggingFaceCheckpointTransport("token", "someone-else/backups")

    with pytest.raises(PermissionError, match = "No write permission"):
        transport.validate_access()
