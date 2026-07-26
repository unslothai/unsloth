# SPDX-License-Identifier: AGPL-3.0-only

import json
import importlib.util
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.manifest import atomic_write_manifest, build_manifest

_BACKEND = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "checkpoint_import_resume_under_test", _BACKEND / "core/training/resume.py"
)
resume = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(resume)
CheckpointImportError = resume.CheckpointImportError
inspect_import_checkpoint = resume.inspect_import_checkpoint
validate_import_compatibility = resume.validate_import_compatibility
preserve_checkpoint_training_target = resume.preserve_checkpoint_training_target


def _archive(path, *, tensor = False):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"N.")
        if tensor:
            archive.writestr("archive/data/0", b"tensor")


def _checkpoint(root, step = 7, max_steps = None):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True)
    state = {"global_step": step}
    if max_steps is not None:
        state["max_steps"] = max_steps
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    _archive(checkpoint / "adapter_model.bin", tensor = True)
    _archive(checkpoint / "optimizer.pt")
    _archive(checkpoint / "scheduler.pt")
    return checkpoint


def test_import_valid_checkpoint_without_database_row(tmp_path):
    checkpoint = _checkpoint(tmp_path / "run")
    result = inspect_import_checkpoint(str(checkpoint), [tmp_path])
    assert result["selected_checkpoint"] == str(checkpoint)
    assert result["global_step"] == 7
    assert result["backend_type"] == "transformers"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: (p / "trainer_state.json").write_text("{"), "trainer_state.json"),
        (lambda p: (p / "optimizer.pt").unlink(), "optimizer.pt"),
        (lambda p: (p / "scheduler.pt").unlink(), "scheduler.pt"),
    ],
)
def test_import_reports_precise_corruption(tmp_path, mutation, message):
    checkpoint = _checkpoint(tmp_path / "run")
    mutation(checkpoint)
    with pytest.raises(CheckpointImportError, match = message):
        inspect_import_checkpoint(str(checkpoint), [tmp_path])


def test_import_rejects_traversal_and_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    outside = _checkpoint(tmp_path / "outside")
    allowed.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory = True)
    with pytest.raises(CheckpointImportError, match = "traversal"):
        inspect_import_checkpoint(str(allowed / ".." / "outside"), [allowed])
    with pytest.raises(CheckpointImportError, match = "approved browse roots"):
        inspect_import_checkpoint(str(allowed / "escape"), [allowed])


def test_import_rejects_incompatible_backend(tmp_path):
    info = inspect_import_checkpoint(str(_checkpoint(tmp_path / "run")), [tmp_path])
    with pytest.raises(CheckpointImportError, match = "incompatible"):
        validate_import_compatibility(info, object(), "mlx")


def test_import_rejects_checkpoint_that_already_reached_max_steps(tmp_path):
    info = inspect_import_checkpoint(str(_checkpoint(tmp_path / "run", step = 30)), [tmp_path])
    request = SimpleNamespace(model_name = None, max_steps = 30)

    with pytest.raises(CheckpointImportError, match = "already reached Max Steps"):
        validate_import_compatibility(info, request, "transformers")


def test_import_allows_checkpoint_with_steps_remaining(tmp_path):
    info = inspect_import_checkpoint(str(_checkpoint(tmp_path / "run", step = 20)), [tmp_path])
    request = SimpleNamespace(
        model_name = None,
        max_steps = 30,
        confirm_import_differences = True,
    )

    assert validate_import_compatibility(info, request, "transformers") == [
        "No portable training manifest was found; compatibility checks are limited."
    ]


def test_import_preserves_original_trainer_target(tmp_path):
    info = inspect_import_checkpoint(
        str(_checkpoint(tmp_path / "run", step = 30, max_steps = 2000)),
        [tmp_path],
    )
    request = SimpleNamespace(max_steps = 30)

    preserve_checkpoint_training_target(info, request)

    assert info["max_steps"] == 2000
    assert request.max_steps == 2000


def test_import_rejects_manifest_step_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path / "run")
    atomic_write_manifest(
        checkpoint,
        build_manifest({"model_name": "m", "device_backend": "cuda"}, expected_checkpoint_step = 6),
    )
    with pytest.raises(CheckpointImportError, match = "expected checkpoint step"):
        inspect_import_checkpoint(str(checkpoint), [tmp_path])


def test_import_rejects_manifest_adapter_model_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path / "run")
    atomic_write_manifest(
        checkpoint,
        build_manifest(
            {"model_name": "org/model-a", "device_backend": "cuda"}, expected_checkpoint_step = 7
        ),
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/model-b"})
    )
    with pytest.raises(CheckpointImportError, match = "base model"):
        inspect_import_checkpoint(str(checkpoint), [tmp_path])


def test_concurrent_import_output_claim_is_unique(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    kwargs = dict(
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
        output_dir = str(tmp_path / "run"),
        imported_checkpoint = str(tmp_path / "run/checkpoint-7"),
        import_source_output_dir = str(tmp_path / "run"),
    )
    studio_db.create_run(id = "first", **kwargs)
    with pytest.raises(sqlite3.IntegrityError):
        studio_db.create_run(id = "second", **kwargs)
