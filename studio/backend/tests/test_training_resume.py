# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for resumable training run eligibility."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file


_BACKEND = Path(__file__).resolve().parents[1]


def _load_resume_module():
    spec = importlib.util.spec_from_file_location(
        "training_resume_under_test",
        _BACKEND / "core" / "training" / "resume.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resume = _load_resume_module()


@pytest.fixture(autouse = True)
def _pytorch_backend(monkeypatch):
    # Host-independent default: the MLX cases override explicitly.
    monkeypatch.setattr(resume, "current_training_backend", lambda: "pt")


def test_resume_request_accepts_sanitized_null_target_modules():
    from models.training import TrainingStartRequest
    request = TrainingStartRequest(
        model_name = "unsloth/Qwen3-0.6B",
        training_type = "Full Finetuning",
        format_type = "alpaca",
        target_modules = None,
    )

    assert request.target_modules == []


def _write_checkpoint(out: Path, step: int) -> Path:
    checkpoint = out / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True, exist_ok = True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding = "utf-8"
    )
    torch.save({"weight": torch.ones(1)}, checkpoint / "adapter_model.bin")
    torch.save({"state": {0: torch.ones(1)}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
    return checkpoint


def _write_mlx_checkpoint(out: Path, step: int) -> Path:
    checkpoint = out / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True, exist_ok = True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding = "utf-8"
    )
    save_file({"weight": np.ones(1, dtype = np.float32)}, checkpoint / "adapters.safetensors")
    save_file(
        {"state": np.ones(1, dtype = np.float32)},
        checkpoint / "optimizer_state.safetensors",
    )
    return checkpoint


def _write_checkpoint_after_rewind(out: Path, step: int) -> Path:
    """A checkpoint the resumed run wrote, i.e. dated after the recorded rewind."""
    checkpoint = _write_checkpoint(out, step)
    marker = json.loads((out / "resume_rewind.json").read_text(encoding = "utf-8"))
    stamp = float(marker["recorded_at"]) + 1
    os.utime(checkpoint / "trainer_state.json", (stamp, stamp))
    return checkpoint


def _stopped_run(**overrides):
    run = {
        "status": "stopped",
        "final_step": 5,
        "total_steps": 10,
        "output_dir": "/tmp/unsloth-output",
        "resumed_later": False,
        "config_json": json.dumps({"hf_dataset": "org/dataset"}),
    }
    run.update(overrides)
    return run


def test_can_resume_run_allows_checkpointed_non_s3_run(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    assert resume.can_resume_run(_stopped_run()) is True


def test_can_resume_run_allows_errored_run_with_checkpoint(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    assert resume.can_resume_run(_stopped_run(status = "error")) is True


def test_can_resume_run_rejects_errored_run_without_checkpoint(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: False)

    assert resume.can_resume_run(_stopped_run(status = "error")) is False


def test_can_resume_run_allows_errored_run_at_final_step(monkeypatch):
    # A save-time crash records final_step == total_steps; resuming re-runs the
    # final-save path from the checkpoint.
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(status = "error", final_step = 10, total_steps = 10)

    assert resume.can_resume_run(run) is True


def test_can_resume_run_rejects_stopped_run_at_final_step(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(final_step = 10, total_steps = 10)

    assert resume.can_resume_run(run) is False


def test_can_resume_run_rejects_s3_dataset_source(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(
        config_json = json.dumps(
            {
                "dataset_source": "s3",
                "s3_dataset": {
                    "bucket": "training-data",
                    "prefix": "datasets/",
                    "region": "us-east-1",
                    "use_iam_role": True,
                },
            }
        )
    )

    assert resume.can_resume_run(run) is False


def test_can_resume_run_rejects_s3_metadata_marker(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(config_json = json.dumps({"s3_dataset": {"bucket": "training-data"}}))

    assert resume.can_resume_run(run) is False


def test_list_runs_includes_config_json_for_resume_policy(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    config_json = json.dumps({"dataset_source": "s3", "s3_dataset": {"bucket": "training-data"}})

    studio_db.create_run(
        id = "run-s3",
        model_name = "unsloth/test-model",
        dataset_name = "s3://training-data",
        config_json = config_json,
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )

    result = studio_db.list_runs()

    assert result["runs"][0]["config_json"] == config_json


def test_crashed_run_with_persisted_output_dir_is_resumable(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 10)

    studio_db.create_run(
        id = "run-crash",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    studio_db.update_run_output_dir("run-crash", str(out))
    conn = studio_db.get_connection()
    conn.execute("UPDATE training_runs SET status = 'error' WHERE id = 'run-crash'")
    conn.commit()
    conn.close()

    run = studio_db.get_run("run-crash")
    assert run["output_dir"] == str(out)
    assert resume.can_resume_run(run) is True


def test_checkpoint_discovery_skips_malformed_newest(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_x"
    valid = _write_checkpoint(out, 5)
    (_write_checkpoint(out, 8) / "scheduler.pt").unlink()
    malformed = out / "checkpoint-10"
    malformed.mkdir()
    (malformed / "trainer_state.json").write_text(json.dumps({"global_step": 10}), encoding = "utf-8")
    (malformed / "adapter_model.bin").write_bytes(b"not a torch archive")
    (malformed / "optimizer.pt").write_bytes(b"not a torch archive")

    assert resume.get_resume_checkpoint_path(str(out)) == str(valid)


def test_completed_run_keeps_output_dir_and_rejects_stale_cancel(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "r",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    studio_db.update_run_output_dir("r", "/out/x")
    studio_db.finish_run(
        id = "r",
        status = "completed",
        ended_at = "t",
        final_step = 2,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = "/out/x",
        error_message = None,
    )

    assert studio_db.get_run("r")["output_dir"] == "/out/x"
    assert studio_db.mark_run_cancel_requested("r") is False
    assert studio_db.get_run("r")["output_dir"] == "/out/x"
    assert studio_db.get_run("r")["resume_blocked"] == 0


def test_finish_run_clears_output_dir_for_stop_without_save(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "r",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    studio_db.update_run_output_dir("r", "/out/x")
    studio_db.finish_run(
        id = "r",
        status = "stopped",
        ended_at = "t",
        final_step = 2,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = None,
        clear_output_dir = True,
    )

    assert studio_db.get_run("r")["output_dir"] is None
    conn = studio_db.get_connection()
    conn.execute(
        "UPDATE training_runs SET status = 'running', output_dir = '/out/x', resume_blocked = 0 WHERE id = 'r'"
    )
    conn.commit()
    conn.close()
    studio_db.mark_run_cancel_requested("r")
    studio_db.cleanup_orphaned_runs()
    assert studio_db.get_run("r")["status"] == "stopped"
    assert studio_db.get_run("r")["output_dir"] is None


def test_finish_run_clears_output_dir_on_cancel_error_finalize(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "r",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    studio_db.update_run_output_dir("r", "/out/x")
    studio_db.finish_run(
        id = "r",
        status = "stopped",
        ended_at = "t",
        final_step = 2,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = "/out/x",
        error_message = "worker failed during cancel",
        clear_output_dir = True,
    )

    assert studio_db.get_run("r")["output_dir"] is None


def test_finish_run_preserves_output_dir_for_interrupted_stop_and_save(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "r",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    studio_db.update_run_output_dir("r", "/out/x")
    studio_db.finish_run(
        id = "r",
        status = "stopped",
        ended_at = "t",
        final_step = 2,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = None,
    )

    assert studio_db.get_run("r")["output_dir"] == "/out/x"


def test_resume_run_dir_maps_checkpoint_to_its_parent():
    assert resume.resume_run_dir("/outputs/run_x/checkpoint-5") == "/outputs/run_x"
    assert resume.resume_run_dir("/outputs/run_x") == "/outputs/run_x"


def test_find_resumable_run_accepts_checkpoint_path(monkeypatch, tmp_path):
    # The DB stores the parent run dir; a checkpoint-N target must still match.
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

    studio_db.create_run(
        id = "run-ckpt",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    studio_db.update_run_output_dir("run-ckpt", str(out))
    studio_db.finish_run(
        id = "run-ckpt",
        status = "stopped",
        ended_at = "2026-01-01T00:05:00Z",
        final_step = 10,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = str(out),
        error_message = None,
    )

    assert resume.find_resumable_run(str(out))["id"] == "run-ckpt"
    assert resume.find_resumable_run(str(ckpt))["id"] == "run-ckpt"
    assert resume.find_resumable_run(str(out / "checkpoint-99"))["id"] == "run-ckpt"


def test_resumed_errored_run_is_not_offered_again(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 10)

    studio_db.create_run(
        id = "run-old",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    studio_db.update_run_output_dir("run-old", str(out))
    studio_db.finish_run(
        id = "run-old",
        status = "error",
        ended_at = "2026-01-01T00:05:00Z",
        final_step = 10,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = "killed",
    )
    studio_db.create_run(
        id = "run-new",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-02T00:00:00Z",
        total_steps = 20,
        output_dir = str(out),
        resumed_from_run_id = "run-old",
    )
    with pytest.raises(RuntimeError, match = "no longer available"):
        studio_db.create_run(
            id = "run-duplicate",
            model_name = "m",
            dataset_name = "d",
            config_json = "{}",
            started_at = "2026-01-02T00:00:01Z",
            total_steps = 20,
            output_dir = str(out),
            resumed_from_run_id = "run-old",
        )
    assert studio_db.get_run("run-duplicate") is None
    studio_db.finish_run(
        id = "run-new",
        status = "error",
        ended_at = "2026-01-02T00:05:00Z",
        final_step = 15,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = "killed again",
    )

    old_run = studio_db.get_run("run-old")
    new_run = studio_db.get_run("run-new")
    assert old_run["resumed_later"] == 1
    assert resume.can_resume_run(old_run) is False
    assert new_run["resumed_later"] == 0
    assert resume.can_resume_run(new_run) is True
    assert studio_db.get_resumable_run_by_output_dir(str(out))["id"] == "run-new"


def test_running_continuation_blocks_older_resume(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 10)

    studio_db.create_run(
        id = "run-old",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    studio_db.update_run_output_dir("run-old", str(out))
    studio_db.finish_run(
        id = "run-old",
        status = "error",
        ended_at = "2026-01-01T00:05:00Z",
        final_step = 10,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = "killed",
    )
    studio_db.create_run(
        id = "run-new",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-02T00:00:00Z",
        total_steps = 20,
        output_dir = str(out),
        resumed_from_run_id = "run-old",
    )

    old_run = studio_db.get_run("run-old")
    assert old_run["resumed_later"] == 1
    assert resume.can_resume_run(old_run) is False
    assert studio_db.get_resumable_run_by_output_dir(str(out)) is None


def test_stop_save_checkpoint_failure_keeps_error_status(monkeypatch, tmp_path):
    # A stop-and-save whose checkpoint write failed must finalize as an error so
    # history explains the missing resume state (keep_error_status flag).
    from core.training.training import TrainingBackend
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "run-failed-save",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    backend = TrainingBackend()
    backend.current_job_id = "run-failed-save"
    backend._db_run_created = True
    backend._should_stop = True
    backend._handle_event(
        {
            "type": "error",
            "error": "Failed to save a resumable checkpoint after stop.",
            "keep_error_status": True,
        }
    )

    run = studio_db.get_run("run-failed-save")
    assert run["status"] == "error"
    assert "resumable checkpoint" in run["error_message"]


def test_can_resume_run_rejects_resume_blocked_run(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    assert resume.can_resume_run(_stopped_run(status = "error", resume_blocked = 1)) is False


def test_stop_save_checkpoint_failure_with_stale_checkpoint_is_not_resumable(monkeypatch, tmp_path):
    # A failed stop-and-save must not offer Resume from an older periodic checkpoint;
    # that would roll back past the recorded final step.
    from core.training.training import TrainingBackend
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 10)

    studio_db.create_run(
        id = "run-stale-ckpt",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    studio_db.update_run_output_dir("run-stale-ckpt", str(out))
    backend = TrainingBackend()
    backend.current_job_id = "run-stale-ckpt"
    backend._db_run_created = True
    backend._should_stop = True
    backend._output_dir = str(out)
    backend._handle_event(
        {
            "type": "error",
            "error": "Failed to save a resumable checkpoint after stop.",
            "keep_error_status": True,
            "resume_blocked": True,
        }
    )

    run = studio_db.get_run("run-stale-ckpt")
    assert run["status"] == "error"
    assert run["resume_blocked"] == 1
    assert run["output_dir"] == str(out)
    assert resume.can_resume_run(run) is False


def test_user_stop_error_without_checkpoint_ack_is_blocked(monkeypatch, tmp_path):
    from core.training.training import TrainingBackend
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.create_run(
        id = "run-user-stop",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )
    backend = TrainingBackend()
    backend.current_job_id = "run-user-stop"
    backend._db_run_created = True
    backend._should_stop = True
    backend._handle_event({"type": "error", "error": "interrupted"})

    run = studio_db.get_run("run-user-stop")
    assert run["status"] == "error" and run["resume_blocked"] == 1


def test_terminal_fallback_keeps_resumable_when_current_checkpoint_landed(monkeypatch, tmp_path):
    # Worker died before its terminal event, but a valid current-step checkpoint
    # is on disk: the fallback must keep the run resumable, not block it.
    from core.training.training import TrainingBackend

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_ok"
    _write_checkpoint(out, 7)

    backend = TrainingBackend()
    backend.current_job_id = "run-ok"
    backend._should_stop = True
    backend._output_dir = str(out)
    backend._progress.step = 7

    kwargs = backend._terminal_finalize_kwargs()
    assert kwargs["status"] == "stopped"
    assert kwargs["resume_blocked"] is False


def test_terminal_fallback_blocks_when_no_current_checkpoint(monkeypatch, tmp_path):
    # Same path, but only a stale (older-step) checkpoint exists: must block.
    from core.training.training import TrainingBackend

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_stale"
    _write_checkpoint(out, 5)

    backend = TrainingBackend()
    backend.current_job_id = "run-stale"
    backend._should_stop = True
    backend._output_dir = str(out)
    backend._progress.step = 7

    kwargs = backend._terminal_finalize_kwargs()
    assert kwargs["status"] == "error"
    assert kwargs["resume_blocked"] is True


def test_can_resume_run_rejects_a_bundle_from_the_other_backend(monkeypatch, tmp_path):
    # History must not offer Resume for a run this host's backend cannot load.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_mlx"
    _write_mlx_checkpoint(out, 5)
    run = _stopped_run(output_dir = str(out))

    assert resume.can_resume_run(run) is False

    monkeypatch.setattr(resume, "current_training_backend", lambda: "mlx")
    assert resume.can_resume_run(run) is True


def test_start_validates_the_resume_checkpoint_against_the_studio_backend(monkeypatch, tmp_path):
    # The mismatch has to be caught here, not after model and dataset loading.
    from unittest.mock import MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.training as training_routes
    from auth.authentication import authenticated_via_api_key, get_current_subject

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_mlx"
    checkpoint = _write_mlx_checkpoint(out, 5)

    backend = MagicMock()
    backend.is_training_active.return_value = False
    backend.current_job_id = ""
    backend.start_training.return_value = True
    monkeypatch.setattr(training_routes, "get_training_backend", lambda: backend)
    monkeypatch.setattr(training_routes, "find_resumable_run", lambda _dir: {"id": "run-mlx"})
    monkeypatch.setattr(training_routes, "can_resume_run", lambda _run: True)
    monkeypatch.setattr(training_routes, "current_training_backend", lambda: "pt")

    app = FastAPI()
    app.include_router(training_routes.router, prefix = "/training")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    client = TestClient(app, raise_server_exceptions = False)
    payload = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "training_type": "LoRA/QLoRA",
        "format_type": "Alpaca",
        "hf_dataset": "yahma/alpaca-cleaned",
        "load_in_4bit": False,
        "eval_steps": 0,
        "resume_from_checkpoint": str(out),
    }

    response = client.post("/training/start", json = payload)

    assert response.status_code == 400, response.text
    assert "training backend" in response.json()["detail"]
    backend.start_training.assert_not_called()

    monkeypatch.setattr(training_routes, "current_training_backend", lambda: "mlx")
    response = client.post("/training/start", json = payload)

    assert response.status_code == 200, response.text
    assert backend.start_training.call_args.kwargs["resume_from_checkpoint"] == str(checkpoint)


def test_resume_after_a_rewind_stays_off_the_abandoned_checkpoint(monkeypatch, tmp_path):
    # The rewound run stopped before passing checkpoint-10; a later plain resume
    # must continue the rewound timeline instead of jumping to the abandoned one.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_x"
    rewound = _write_checkpoint(out, 5)
    _write_checkpoint(out, 10)

    resume.record_resume_rewind(str(rewound), backend = "pt")

    assert resume.get_resume_checkpoint_path(str(out)) == str(rewound)
    assert resume.has_resume_state(str(out)) is True


def test_checkpoint_written_after_a_rewind_lifts_the_cap_to_itself(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_x"
    rewound = _write_checkpoint(out, 5)
    _write_checkpoint(out, 10)
    resume.record_resume_rewind(str(rewound), backend = "pt")

    # The new timeline reached step 8: still short of the abandoned checkpoint-10.
    fresh = _write_checkpoint_after_rewind(out, 8)
    assert resume.get_resume_checkpoint_path(str(out)) == str(fresh)

    # Once it writes past checkpoint-10 the cap no longer constrains anything.
    passed = _write_checkpoint_after_rewind(out, 12)
    assert resume.get_resume_checkpoint_path(str(out)) == str(passed)


def test_explicitly_resuming_the_newest_checkpoint_clears_the_rewind(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    out = tmp_path / "outputs" / "run_x"
    rewound = _write_checkpoint(out, 5)
    newest = _write_checkpoint(out, 10)
    resume.record_resume_rewind(str(rewound), backend = "pt")

    # Targeting checkpoint-10 explicitly re-adopts that timeline.
    assert resume.get_resume_checkpoint_path(str(newest)) == str(newest)
    resume.record_resume_rewind(str(newest), backend = "pt")

    assert (out / "resume_rewind.json").exists() is False
    assert resume.get_resume_checkpoint_path(str(out)) == str(newest)


def test_invalid_post_rewind_checkpoint_does_not_lift_the_cap(monkeypatch, tmp_path):
    # An interrupted save after a rewind writes trainer_state.json without model
    # state; lifting the cap on it would re-admit the abandoned valid sibling.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    run = tmp_path / "outputs" / "run"
    _write_checkpoint(run, 5)
    _write_checkpoint(run, 10)
    resume.record_resume_rewind(str(run / "checkpoint-5"), backend = "pt")
    partial = run / "checkpoint-8"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(json.dumps({"global_step": 8}))
    assert resume.resume_step_cap(run, "pt") == 5
    assert resume.get_resume_checkpoint_path(str(run), backend = "pt") == str(run / "checkpoint-5")


def test_has_resume_state_is_false_without_a_usable_backend(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_checkpoint(run, 5)
    monkeypatch.setattr(resume, "current_training_backend", lambda: None)
    assert resume.has_resume_state(str(run)) is False
