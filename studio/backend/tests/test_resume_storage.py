from pathlib import Path
import os
import shutil
import importlib.util
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "resume_storage_under_test", Path(__file__).parents[1] / "core/training/resume_storage.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
COMPLETE_MARKER = _MODULE.COMPLETE_MARKER
latest_synchronized_checkpoint = _MODULE.latest_synchronized_checkpoint
stage_checkpoint = _MODULE.stage_checkpoint
synchronize_checkpoints = _MODULE.synchronize_checkpoints

def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"; checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step": %d}' % step)
    return checkpoint

def test_read_only_source_can_be_staged(tmp_path):
    source = _checkpoint(tmp_path / "mounted", 1); (source / "weights").write_bytes(b"state"); source.chmod(0o555)
    staged = Path(stage_checkpoint(str(source), str(tmp_path / "local"), lambda _: None))
    assert (staged / "weights").read_bytes() == b"state" and (staged / COMPLETE_MARKER).is_file()

def test_distinct_destination_is_atomically_synchronized(tmp_path):
    local, persistent = tmp_path / "local", tmp_path / "persistent"; _checkpoint(local, 2)
    synchronize_checkpoints(str(local), str(persistent), lambda _: None)
    assert latest_synchronized_checkpoint(str(persistent)) == str(persistent / "checkpoint-2")

def test_in_place_conflict_requires_explicit_selection(tmp_path):
    source = _checkpoint(tmp_path / "run", 2)
    destination = source.parent
    conflicts = destination == source.parent or source.parent in destination.parents
    assert conflicts, "route validation must reject this unless in_place_continuation is true"

def test_interrupted_copy_never_publishes_destination(tmp_path, monkeypatch):
    source = _checkpoint(tmp_path / "source", 1)
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): stage_checkpoint(str(source), str(tmp_path / "local"), lambda _: None)
    assert not list((tmp_path / "local").glob("checkpoint-1-*"))

def test_interrupted_synchronization_is_not_resumable(tmp_path, monkeypatch):
    source, persistent = tmp_path / "local", tmp_path / "persistent"; _checkpoint(source, 3)
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("terminated")))
    with pytest.raises(OSError, match="terminated"): synchronize_checkpoints(str(source), str(persistent), lambda _: None)
    assert latest_synchronized_checkpoint(str(persistent)) is None

def test_storage_exhaustion_does_not_publish_checkpoint(tmp_path, monkeypatch):
    source, persistent = tmp_path / "local", tmp_path / "persistent"; _checkpoint(source, 4)
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left")))
    with pytest.raises(OSError, match="No space"): synchronize_checkpoints(str(source), str(persistent), lambda _: None)
    assert latest_synchronized_checkpoint(str(persistent)) is None

def test_restart_selects_latest_valid_synchronized_checkpoint(tmp_path):
    _checkpoint(tmp_path, 7).joinpath(COMPLETE_MARKER).write_text("complete\n")
    _checkpoint(tmp_path, 9)
    _checkpoint(tmp_path, 8).joinpath(COMPLETE_MARKER).write_text("complete\n")
    assert latest_synchronized_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-8")
