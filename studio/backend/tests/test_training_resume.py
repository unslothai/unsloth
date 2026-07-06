# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for resumable training run eligibility."""

import importlib.util
import json
from pathlib import Path


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
    # A save-time crash records final_step == total_steps; resume re-runs the final save.
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
    (out / "checkpoint-10").mkdir(parents = True)
    (out / "checkpoint-10" / "trainer_state.json").write_text("{}", encoding = "utf-8")

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


def test_finish_run_does_not_erase_persisted_output_dir(monkeypatch, tmp_path):
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
        status = "error",
        ended_at = "t",
        final_step = 2,
        final_loss = None,
        duration_seconds = 1,
        loss_sparkline = "[]",
        output_dir = None,
        error_message = "killed",
    )

    assert studio_db.get_run("r")["output_dir"] == "/out/x"


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
    (out / "checkpoint-10").mkdir(parents = True)
    (out / "checkpoint-10" / "trainer_state.json").write_text("{}", encoding = "utf-8")

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
    )
    studio_db.update_run_output_dir("run-new", str(out))
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
    (out / "checkpoint-10").mkdir(parents = True)
    (out / "checkpoint-10" / "trainer_state.json").write_text("{}", encoding = "utf-8")

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
    )
    studio_db.update_run_output_dir("run-new", str(out))

    old_run = studio_db.get_run("run-old")
    assert old_run["resumed_later"] == 1
    assert resume.can_resume_run(old_run) is False
    assert studio_db.get_resumable_run_by_output_dir(str(out)) is None


def test_stop_save_checkpoint_failure_keeps_error_status(monkeypatch, tmp_path):
    # A failed stop-and-save checkpoint must finalize as an error, not a stopped run.
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
    # A failed stop-and-save must not offer Resume from an older, stale checkpoint.
    from core.training.training import TrainingBackend
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    out = tmp_path / "outputs" / "run_x"
    (out / "checkpoint-10").mkdir(parents = True)
    (out / "checkpoint-10" / "trainer_state.json").write_text("{}", encoding = "utf-8")

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


def test_user_stop_error_without_flag_still_finalizes_stopped(monkeypatch, tmp_path):
    # Errors surfaced while honouring a plain user stop keep the stopped status.
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

    assert studio_db.get_run("run-user-stop")["status"] == "stopped"
