# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import json
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _load_training_history_module():
    spec = importlib.util.spec_from_file_location(
        "training_history_under_test",
        _BACKEND / "routes" / "training_history.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_resume_module():
    spec = importlib.util.spec_from_file_location(
        "training_resume_artifacts_under_test",
        _BACKEND / "core" / "training" / "resume.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training_history = _load_training_history_module()
resume = _load_resume_module()


@pytest.fixture(autouse = True)
def _run_thread_offloads_inline(monkeypatch):
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(training_history.asyncio, "to_thread", inline)


def _run_row(**overrides) -> dict:
    row = {
        "id": "run-1",
        "status": "stopped",
        "model_name": "unsloth/test-model",
        "dataset_name": "test-dataset",
        "started_at": "2026-01-01T00:00:00Z",
        "output_dir": "/tmp/run-1",
        "resumed_later": False,
        "config_json": json.dumps({"hf_dataset": "org/dataset"}),
    }
    row.update(overrides)
    return row


def test_artifacts_present_truth_table(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)

    monkeypatch.setattr(resume, "outputs_root", lambda: outputs)
    monkeypatch.setattr(resume, "resolve_output_dir", lambda value: Path(value))

    assert resume.artifacts_present(str(run_dir)) is True
    assert resume.artifacts_present(str(outputs / "missing")) is False
    assert resume.artifacts_present(str(tmp_path / "elsewhere")) is False
    assert resume.artifacts_present(None) is False
    assert resume.artifacts_present("") is False


def test_artifacts_present_tolerates_unresolvable_dirs(monkeypatch):
    def _raise(value):
        raise ValueError("escapes outputs root")

    monkeypatch.setattr(resume, "resolve_output_dir", _raise)
    assert resume.artifacts_present("D:\\other-drive\\run") is False


def test_get_resume_checkpoint_path_tolerates_unresolvable_dirs(monkeypatch):
    def _raise(value):
        raise ValueError("escapes outputs root")

    monkeypatch.setattr(resume, "resolve_output_dir", _raise)
    assert resume.get_resume_checkpoint_path("D:\\other-drive\\run") is None


def test_summary_reports_artifacts_available(monkeypatch):
    monkeypatch.setattr(training_history, "can_resume_run", lambda run: False)
    monkeypatch.setattr(training_history, "artifacts_present", lambda path: path == "/tmp/run-1")

    with_artifacts = training_history._summary_from_row(_run_row(), sharing_on = False)
    without_artifacts = training_history._summary_from_row(
        _run_row(output_dir = "/tmp/gone"), sharing_on = False
    )

    assert with_artifacts.artifacts_available is True
    assert without_artifacts.artifacts_available is False


def test_resource_resume_validation_is_cached_per_request(monkeypatch):
    from core.training import provenance

    calls: list[dict] = []
    monkeypatch.setattr(resume, "has_resume_state", lambda output_dir: True)
    monkeypatch.setattr(
        provenance,
        "resource_provenance_allows_resume",
        lambda config: calls.append(config) or True,
    )
    cache: dict[str, bool] = {}
    first = _run_row(output_dir = "/tmp/first")
    second = _run_row(output_dir = "/tmp/second")

    assert resume.can_resume_run(first, resource_cache = cache) is True
    assert resume.can_resume_run(second, resource_cache = cache) is True
    assert len(calls) == 1
    assert resume.can_resume_run(second, resource_cache = {}) is True
    assert len(calls) == 2


def test_resource_resume_cache_key_tracks_snapshot_paths(monkeypatch):
    from core.training import provenance

    calls: list[dict] = []
    monkeypatch.setattr(resume, "has_resume_state", lambda output_dir: True)
    monkeypatch.setattr(
        provenance,
        "resource_provenance_allows_resume",
        lambda config: calls.append(config) or True,
    )
    cache: dict[str, bool] = {}
    first = _run_row(
        config_json = {
            "hf_dataset": "org/dataset",
            "dataset_snapshot_path": "/cache/first",
        }
    )
    second = _run_row(
        config_json = {
            "hf_dataset": "org/dataset",
            "dataset_snapshot_path": "/cache/second",
        }
    )

    assert resume.can_resume_run(first, resource_cache = cache) is True
    assert resume.can_resume_run(second, resource_cache = cache) is True
    assert len(calls) == 2


def test_nonserializable_resume_key_bypasses_cache(monkeypatch):
    from core.training import provenance

    calls: list[dict] = []
    monkeypatch.setattr(resume, "has_resume_state", lambda output_dir: True)
    monkeypatch.setattr(
        provenance,
        "resource_provenance_allows_resume",
        lambda config: calls.append(config) or True,
    )
    row = _run_row(config_json = {"hf_dataset": object()})
    cache: dict[str, bool] = {}

    assert resume.can_resume_run(row, resource_cache = cache) is True
    assert resume.can_resume_run(row, resource_cache = cache) is True
    assert len(calls) == 2
    assert cache == {}


def test_summary_batch_preserves_order_and_uses_request_local_cache(monkeypatch):
    observed_caches: list[dict[str, bool]] = []

    def fake_can_resume(run, *, resource_cache):
        observed_caches.append(resource_cache)
        return False

    monkeypatch.setattr(training_history, "can_resume_run", fake_can_resume)
    monkeypatch.setattr(training_history, "artifacts_present", lambda path: False)
    rows = [_run_row(id = run_id, output_dir = f"/tmp/{run_id}") for run_id in ("b", "a", "c")]

    first = training_history._summaries_from_rows(rows, sharing_on = False)
    first_cache = observed_caches[0]
    second = training_history._summaries_from_rows(rows, sharing_on = False)
    second_cache = observed_caches[len(rows)]

    assert [summary.id for summary in first] == ["b", "a", "c"]
    assert [summary.id for summary in second] == ["b", "a", "c"]
    assert all(cache is first_cache for cache in observed_caches[: len(rows)])
    assert all(cache is second_cache for cache in observed_caches[len(rows) :])
    assert first_cache is not second_cache


def test_list_training_runs_offloads_summary_batch(monkeypatch):
    offloads: list[tuple[object, tuple[object, ...]]] = []
    rows = [_run_row(id = run_id, output_dir = f"/tmp/{run_id}") for run_id in ("b", "a", "c")]

    async def record_offload(function, *args):
        offloads.append((function, args))
        return function(*args)

    monkeypatch.setattr(training_history.asyncio, "to_thread", record_offload)
    monkeypatch.setattr(
        training_history,
        "list_runs",
        lambda **kwargs: {"runs": rows, "total": len(rows)},
    )
    monkeypatch.setattr(
        training_history,
        "get_preview_sharing_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        training_history,
        "can_resume_run",
        lambda run, *, resource_cache: False,
    )
    monkeypatch.setattr(training_history, "artifacts_present", lambda path: False)

    response = asyncio.run(
        training_history.list_training_runs(
            limit = 50,
            offset = 0,
            current_subject = "test-user",
        )
    )

    assert [summary.id for summary in response.runs] == ["b", "a", "c"]
    assert offloads == [(training_history._summaries_from_rows, (rows, False))]


def _delete(
    monkeypatch,
    run_row,
    *,
    delete_artifacts,
    active_output_dir = None,
    sibling_paths = None,
):
    deleted_runs: list[str] = []
    monkeypatch.setattr(training_history, "get_run", lambda run_id: dict(run_row))
    monkeypatch.setattr(training_history, "delete_run", deleted_runs.append)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: active_output_dir)
    monkeypatch.setattr(
        training_history,
        "list_other_run_output_dirs",
        lambda exclude_id: list(sibling_paths or []),
    )

    response = asyncio.run(
        training_history.delete_training_run(
            "run-1",
            delete_artifacts = delete_artifacts,
            current_subject = "test-user",
        )
    )
    return response, deleted_runs


def test_delete_with_artifacts_removes_dir_under_outputs_root(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    (run_dir / "adapter_model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    response, deleted_runs = _delete(
        monkeypatch, _run_row(output_dir = str(run_dir)), delete_artifacts = True
    )

    assert response.status == "deleted"
    assert response.artifacts_deleted is True
    assert response.artifacts_kept_reason is None
    assert deleted_runs == ["run-1"]
    assert not run_dir.exists()


def test_delete_refuses_dirs_outside_outputs_root(monkeypatch, tmp_path):
    from fastapi import HTTPException

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    (foreign_dir / "keep.txt").write_text("keep")

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    deleted_runs: list[str] = []
    monkeypatch.setattr(
        training_history,
        "get_run",
        lambda run_id: _run_row(output_dir = str(foreign_dir)),
    )
    monkeypatch.setattr(training_history, "delete_run", deleted_runs.append)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: None)
    monkeypatch.setattr(training_history, "list_other_run_output_dirs", lambda run_id: [])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "training_artifact_deletion_failed"
    assert deleted_runs == []
    assert foreign_dir.exists()


def test_canonical_output_dir_rejects_foreign_absolute_paths(monkeypatch, tmp_path):
    windows_path = r"C:\Users\alice\Unsloth\outputs\run-1"
    foreign_path = (
        windows_path
        if not Path(windows_path).is_absolute()
        else "/home/alice/.unsloth/outputs/run-1"
    )
    assert not Path(foreign_path).is_absolute()

    monkeypatch.setattr(
        training_history,
        "resolve_output_dir",
        lambda _value: pytest.fail("foreign absolute path must be rejected first"),
    )
    monkeypatch.setattr(training_history, "outputs_root", lambda: tmp_path / "outputs")

    assert training_history._canonical_output_dir(foreign_path) is None


def test_delete_with_missing_dir_still_deletes_row(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    response, deleted_runs = _delete(
        monkeypatch, _run_row(output_dir = str(outputs / "gone")), delete_artifacts = True
    )

    assert response.status == "deleted"
    assert response.artifacts_deleted is True
    assert deleted_runs == ["run-1"]


def test_delete_with_missing_shared_dir_still_deletes_row(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    missing = outputs / "gone"

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    response, deleted_runs = _delete(
        monkeypatch,
        _run_row(output_dir = str(missing)),
        delete_artifacts = True,
        sibling_paths = [str(missing)],
    )

    assert response.status == "deleted"
    assert response.artifacts_deleted is True
    assert deleted_runs == ["run-1"]


def test_delete_without_flag_leaves_artifacts(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    response, deleted_runs = _delete(
        monkeypatch, _run_row(output_dir = str(run_dir)), delete_artifacts = False
    )

    assert response.status == "deleted"
    assert response.artifacts_deleted is False
    assert deleted_runs == ["run-1"]
    assert run_dir.exists()


def test_delete_rejects_running_run(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(training_history, "get_run", lambda run_id: _run_row(status = "running"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409


def test_delete_artifacts_refused_while_dir_in_use_by_active_run(monkeypatch, tmp_path):
    from fastapi import HTTPException

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    deleted_runs: list[str] = []
    monkeypatch.setattr(
        training_history, "get_run", lambda run_id: _run_row(output_dir = str(run_dir))
    )
    monkeypatch.setattr(training_history, "delete_run", deleted_runs.append)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: str(run_dir))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "training_artifacts_in_use"
    assert deleted_runs == []
    assert run_dir.exists()


def test_delete_row_without_artifacts_allowed_while_dir_in_use(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    response, deleted_runs = _delete(
        monkeypatch,
        _run_row(output_dir = str(run_dir)),
        delete_artifacts = False,
        active_output_dir = str(run_dir),
    )

    assert response.status == "deleted"
    assert deleted_runs == ["run-1"]
    assert run_dir.exists()


def test_active_dir_guard_compares_resolved_paths(monkeypatch, tmp_path):
    from fastapi import HTTPException

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    unnormalized = str(outputs / "x" / ".." / "run-1")
    monkeypatch.setattr(
        training_history, "get_run", lambda run_id: _run_row(output_dir = str(run_dir))
    )
    monkeypatch.setattr(training_history, "delete_run", lambda run_id: None)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: unnormalized)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert run_dir.exists()


def test_same_output_dir_canonicalizes_legacy_parent_segments(monkeypatch, tmp_path):
    from utils.paths import storage_roots

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(storage_roots, "outputs_root", lambda: outputs)
    monkeypatch.setattr(
        training_history,
        "resolve_output_dir",
        storage_roots.resolve_output_dir,
    )

    legacy_alias = str(outputs / "old" / ".." / "run-1")

    assert training_history._same_output_dir(str(run_dir), legacy_alias) is True


def test_output_dir_overlap_detects_ancestors(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    checkpoint = run_dir / "checkpoint-5"
    checkpoint.mkdir(parents = True)
    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    assert training_history._output_dirs_overlap(str(run_dir), str(checkpoint)) is True
    assert training_history._output_dirs_overlap(str(checkpoint), str(run_dir)) is True


def test_delete_artifacts_refused_when_finished_sibling_shares_dir(monkeypatch, tmp_path):
    from fastapi import HTTPException

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    (run_dir / "adapter_model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))

    deleted_runs: list[str] = []
    monkeypatch.setattr(
        training_history,
        "get_run",
        lambda run_id: _run_row(output_dir = str(run_dir)),
    )
    monkeypatch.setattr(training_history, "delete_run", deleted_runs.append)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: None)
    monkeypatch.setattr(
        training_history,
        "list_other_run_output_dirs",
        lambda run_id: [str(run_dir)],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "training_artifacts_shared"
    assert deleted_runs == []
    assert run_dir.exists()


def test_list_other_run_output_dirs_ignores_unstamped_running_rows(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    shared = "/tmp/outputs/shared-run"
    studio_db.create_run(
        id = "run-a",
        model_name = "unsloth/test-model",
        dataset_name = "test-dataset",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    studio_db.finish_run(
        "run-a",
        status = "stopped",
        ended_at = "2026-01-01T01:00:00Z",
        final_step = 5,
        final_loss = 0.5,
        duration_seconds = 60.0,
        output_dir = shared,
    )
    studio_db.create_run(
        id = "run-b",
        model_name = "unsloth/test-model",
        dataset_name = "test-dataset",
        config_json = "{}",
        started_at = "2026-01-01T02:00:00Z",
        total_steps = 10,
    )

    assert studio_db.list_other_run_output_dirs(exclude_id = "run-a") == []
    assert studio_db.list_other_run_output_dirs(exclude_id = "run-b") == [shared]

    studio_db.finish_run(
        "run-b",
        status = "stopped",
        ended_at = "2026-01-01T03:00:00Z",
        final_step = 5,
        final_loss = 0.4,
        duration_seconds = 60.0,
        output_dir = shared,
    )

    assert studio_db.list_other_run_output_dirs(exclude_id = "run-a") == [shared]


def test_delete_failure_retains_history_row(monkeypatch, tmp_path):
    from fastapi import HTTPException

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    deleted_runs: list[str] = []

    monkeypatch.setattr(
        training_history,
        "get_run",
        lambda run_id: _run_row(output_dir = str(run_dir)),
    )
    monkeypatch.setattr(training_history, "delete_run", deleted_runs.append)
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: None)
    monkeypatch.setattr(training_history, "list_other_run_output_dirs", lambda run_id: [])
    monkeypatch.setattr(training_history, "_delete_run_output_dir", lambda *args: False)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            training_history.delete_training_run(
                "run-1",
                delete_artifacts = True,
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "training_artifact_deletion_failed"
    assert deleted_runs == []
    assert run_dir.exists()


def test_delete_artifacts_uses_thread_offload(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    calls: list[tuple] = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        # Staging returns (outcome, original, staged); the purge after the row delete returns nothing.
        return ("deleted", None, None) if len(calls) == 1 else None

    monkeypatch.setattr(training_history.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(training_history, "_delete_run_output_dir", lambda *args: True)

    response, deleted_runs = _delete(
        monkeypatch,
        _run_row(output_dir = str(run_dir)),
        delete_artifacts = True,
    )

    assert response.artifacts_deleted is True
    assert deleted_runs == ["run-1"]
    # One offload, not two: nothing was staged here, so there is no purge phase.
    assert len(calls) == 1


def test_guarded_delete_prevents_resume_from_spawning_after_artifacts_are_removed(
    monkeypatch, tmp_path
):
    from core.training.training import TrainingBackend

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    (run_dir / "adapter_model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: None)
    monkeypatch.setattr(training_history, "list_other_run_output_dirs", lambda run_id: [])

    delete_started = threading.Event()
    allow_delete = threading.Event()
    internal_start_called = threading.Event()
    original_delete = training_history._delete_run_output_dir

    def blocking_delete(run_id, output_dir):
        delete_started.set()
        assert allow_delete.wait(timeout = 2)
        return original_delete(run_id, output_dir)

    monkeypatch.setattr(training_history, "_delete_run_output_dir", blocking_delete)
    monkeypatch.setattr(
        "core.training.resume.get_resume_checkpoint_path",
        lambda path: str(path) if Path(path).exists() else None,
    )

    backend = TrainingBackend()
    monkeypatch.setattr(
        backend,
        "_start_training_with_lifecycle_reserved",
        lambda *args, **kwargs: internal_start_called.set() or True,
    )
    outcomes: dict[str, object] = {}

    delete_thread = threading.Thread(
        target = lambda: outcomes.update(
            delete = training_history._delete_run_output_dir_guarded(
                "run-1",
                str(run_dir),
            )
        )
    )
    delete_thread.start()
    assert delete_started.wait(timeout = 2)

    start_thread = threading.Thread(
        target = lambda: outcomes.update(
            start = backend.start_training(
                "resume-job",
                model_name = "unsloth/test",
                resume_from_checkpoint = str(run_dir),
            )
        )
    )
    start_thread.start()
    assert internal_start_called.wait(timeout = 0.1) is False

    allow_delete.set()
    delete_thread.join(timeout = 2)
    start_thread.join(timeout = 2)

    assert outcomes["delete"][0] == "deleted"
    assert outcomes["start"] is False
    assert internal_start_called.is_set() is False
    assert run_dir.exists() is False


def test_guarded_delete_rechecks_shared_output_after_waiting_for_lifecycle(monkeypatch, tmp_path):
    from core.training.lifecycle import training_lifecycle_guard

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    run_dir.mkdir(parents = True)
    (run_dir / "adapter_model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))
    monkeypatch.setattr(training_history, "_active_training_output_dir", lambda: None)

    sibling_paths: list[str] = []
    monkeypatch.setattr(
        training_history,
        "list_other_run_output_dirs",
        lambda run_id: list(sibling_paths),
    )
    delete_attempted = threading.Event()
    outcome: list[str] = []

    def guarded_delete():
        delete_attempted.set()
        outcome.append(training_history._delete_run_output_dir_guarded("run-1", str(run_dir))[0])

    with training_lifecycle_guard():
        delete_thread = threading.Thread(target = guarded_delete)
        delete_thread.start()
        assert delete_attempted.wait(timeout = 2)
        sibling_paths.append(str(run_dir))

    delete_thread.join(timeout = 2)

    assert outcome == ["shared"]
    assert run_dir.exists() is True


def test_shared_guard_compares_canonical_paths(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    alias = outputs / "alias"
    run_dir.mkdir(parents = True)
    alias.symlink_to(run_dir, target_is_directory = True)

    monkeypatch.setattr(training_history, "outputs_root", lambda: outputs)
    monkeypatch.setattr(training_history, "resolve_output_dir", lambda value: Path(value))
    monkeypatch.setattr(
        training_history,
        "list_other_run_output_dirs",
        lambda run_id: [str(alias)],
    )

    assert training_history._output_dir_shared(str(run_dir), "run-1") is True
