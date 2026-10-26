# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A multi-LoRA batch: one training_runs row per member, the batch id itself never gets one."""

from __future__ import annotations

import json
import queue
from unittest.mock import patch

import pytest

import core.training.training as training_mod
from core.training.training import TrainingBackend, TrainingProgress

BATCH = "job_batch"
MEMBERS = ("job_a", "job_b", "job_c")


@pytest.fixture
def db(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    return studio_db


class _AliveProc:
    pid = 4321

    def is_alive(self):
        return True


def _member_config(run_id):
    return training_mod._sanitize_db_config(
        training_mod._build_training_worker_config(
            {
                "model_name": "unsloth/tiny",
                "local_datasets": [f"/data/{run_id}.jsonl"],
                "hf_token": "hf_secret",
                "wandb_token": "wb_secret",
                "learning_rate": "1e-4",
            }
        )
    )


def _batch_backend(tmp_path):
    b = TrainingBackend()
    b.current_job_id = BATCH
    b._proc = _AliveProc()
    b._stop_queue = queue.Queue()
    b._start_stop_watchdog = lambda **kw: None
    b._progress = TrainingProgress(is_training = True)
    b._db_config = None
    b._batch_members = {
        run_id: b._new_batch_member(_member_config(run_id), "2026-10-26T00:00:00+00:00")
        for run_id in MEMBERS
    }
    return b


def _progress(run_id, step, loss):
    return {
        "type": "progress",
        "member_run_id": run_id,
        "step": step,
        "loss": loss,
        "learning_rate": 1e-4,
        "total_steps": 4,
        "epoch": 0.5,
        "num_tokens": step * 10,
        "elapsed_seconds": float(step),
    }


def _exported(tmp_path, run_id):
    out = tmp_path / "out" / run_id
    out.mkdir(parents = True, exist_ok = True)
    (out / "adapter_config.json").write_text("{}")
    return str(out)


def test_member_events_get_their_own_rows_metrics_and_terminal_states(db, tmp_path):
    b = _batch_backend(tmp_path)
    outs = {r: _exported(tmp_path, r) for r in MEMBERS}
    for run_id in MEMBERS:
        b._handle_event({"type": "output_dir", "member_run_id": run_id, "output_dir": outs[run_id]})
    for step in (1, 2):
        for i, run_id in enumerate(MEMBERS):
            b._handle_event(_progress(run_id, step, 1.0 + i + step / 10))
    assert b.batch_member_ids() == list(MEMBERS)

    b._handle_event({"type": "complete", "member_run_id": "job_a", "output_dir": outs["job_a"]})
    assert b.batch_member_ids() == ["job_b", "job_c"]
    b._handle_event({"type": "error", "member_run_id": "job_a", "error": "late"})
    b._handle_event(_progress("job_a", 3, 9.0))
    b._handle_event(
        {"type": "stopped", "member_run_id": "job_b", "output_dir": outs["job_b"], "saved": True}
    )
    assert b.batch_member_ids() == ["job_c"]
    b._handle_event({"type": "error", "error": "CUDA error: boom"})
    assert b.batch_member_ids() == []

    assert db.get_run(BATCH) is None
    rows = {r: db.get_run(r) for r in MEMBERS}
    assert rows["job_a"]["status"] == "completed"
    assert rows["job_a"]["error_message"] is None
    assert rows["job_b"]["status"] == "stopped"
    assert rows["job_c"]["status"] == "error"
    assert rows["job_c"]["error_message"] == "CUDA error: boom"
    for i, run_id in enumerate(MEMBERS):
        row = rows[run_id]
        assert row["dataset_name"] == f"/data/{run_id}.jsonl"
        config = json.loads(row["config_json"])
        assert config["local_datasets"] == [f"/data/{run_id}.jsonl"]
        assert "hf_token" not in config and "wandb_token" not in config
        assert "hf_secret" not in row["config_json"]
        assert row["final_step"] == 2
        metrics = db.get_run_metrics(run_id)
        assert metrics["step_history"] == [1, 2]
        assert metrics["loss_history"] == pytest.approx([1.1 + i, 1.2 + i])
    assert rows["job_a"]["output_dir"] == outs["job_a"]
    assert rows["job_b"]["output_dir"] == outs["job_b"]


def test_process_complete_finalizes_only_running_members(db, tmp_path):
    b = _batch_backend(tmp_path)
    saved = _exported(tmp_path, "job_b")
    for run_id in MEMBERS:
        b._handle_event(_progress(run_id, 1, 2.0))
    b._handle_event({"type": "error", "member_run_id": "job_a", "error": "bad dataset"})
    b._handle_event({"type": "output_dir", "member_run_id": "job_b", "output_dir": saved})
    b._handle_event(
        {"type": "complete", "output_dir": None, "status_message": "Training completed"}
    )

    assert db.get_run(BATCH) is None
    assert db.get_run("job_a")["status"] == "error"
    assert db.get_run("job_a")["error_message"] == "bad dataset"
    assert db.get_run("job_b")["status"] == "completed"
    assert db.get_run("job_c")["status"] == "error"
    assert "without saving" in db.get_run("job_c")["error_message"]


def test_member_finalize_survives_a_transient_db_error(db, tmp_path, monkeypatch):
    b = _batch_backend(tmp_path)
    out = _exported(tmp_path, "job_a")
    b._handle_event({"type": "output_dir", "member_run_id": "job_a", "output_dir": out})
    b._handle_event(_progress("job_a", 1, 2.0))
    real, calls = db.finish_run, []

    def flaky(**kwargs):
        calls.append(kwargs["id"])
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return real(**kwargs)

    monkeypatch.setattr(db, "finish_run", flaky)
    monkeypatch.setattr(training_mod, "_DB_FINALIZE_RETRY_S", 0)
    b._handle_event({"type": "complete", "member_run_id": "job_a", "output_dir": out})
    assert calls == ["job_a", "job_a"]
    assert db.get_run("job_a")["status"] == "completed"


def test_member_error_before_progress_still_records_its_row(db, tmp_path):
    b = _batch_backend(tmp_path)
    b._handle_event({"type": "error", "member_run_id": "job_c", "error": "empty dataset"})
    row = db.get_run("job_c")
    assert row["status"] == "error" and row["error_message"] == "empty dataset"
    assert db.get_run("job_a") is None
    assert b.batch_member_ids() == ["job_a", "job_b"]


def test_stop_member_sends_member_stop_only(db, tmp_path):
    b = _batch_backend(tmp_path)
    assert b.stop_member("job_b") is True
    assert b._stop_queue.get_nowait() == {"type": "stop_member", "run_id": "job_b", "save": True}
    assert b._stop_queue.empty()
    assert b._should_stop is False and b._cancel_requested is False
    assert b.stop_member("job_unknown") is False
    b._handle_event({"type": "complete", "member_run_id": "job_a"})
    assert b.stop_member("job_a") is False
    assert b._stop_queue.empty()
    b._handle_event(_progress("job_b", 1, 1.0))
    b._handle_event({"type": "complete", "member_run_id": "job_b", "output_dir": "/x"})
    assert db.get_run("job_b")["status"] == "stopped"


def test_batch_cancel_marks_member_rows(db, tmp_path):
    b = _batch_backend(tmp_path)
    b._handle_event(_progress("job_a", 1, 1.0))
    assert b.stop_training(save = False, expected_job_id = BATCH) is True
    assert b._stop_queue.get_nowait() == {"type": "stop", "save": False}
    assert db.get_run("job_a")["resume_blocked"] == 1
    b._handle_event({"type": "complete", "status_message": "Training cancelled."})
    assert db.get_run(BATCH) is None
    for run_id in ("job_a",):
        row = db.get_run(run_id)
        assert row["status"] == "stopped" and row["output_dir"] is None


def test_single_job_events_unchanged(db, tmp_path):
    b = TrainingBackend()
    b.current_job_id = "job_single"
    b._proc = _AliveProc()
    b._start_stop_watchdog = lambda **kw: None
    b._progress = TrainingProgress(is_training = True)
    b._db_config = {"model_name": "unsloth/tiny", "local_datasets": ["/d.jsonl"]}
    b._handle_event({"type": "progress", "step": 1, "loss": 1.5, "total_steps": 2})
    b._handle_event(
        {"type": "complete", "output_dir": "/o", "status_message": "Training completed"}
    )
    row = db.get_run("job_single")
    assert row["status"] == "completed"
    assert db.get_run_metrics("job_single")["step_history"] == [1]


class _DummyProcess:
    pid = 4321

    def start(self):
        return None

    def is_alive(self):
        return True


class _DummyThread:
    def start(self):
        return None


def test_start_training_builds_batch_members_config(db, tmp_path):
    captured = {}
    real_spec = training_mod.account_process_spec

    def _spy(module, fn, cache_env, kwargs):
        captured["config"] = kwargs["config"]
        return real_spec(module, fn, cache_env, kwargs)

    backend = TrainingBackend()
    with (
        patch("core.training.training.prepare_gpu_selection", return_value = ([0], {})),
        patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
        patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
        patch("core.training.training.threading.Thread", return_value = _DummyThread()),
        patch("core.training.training.account_process_spec", side_effect = _spy),
    ):
        ok = backend.start_training(
            job_id = BATCH,
            batch_members = [
                {
                    "run_id": run_id,
                    "model_name": "unsloth/test",
                    "training_type": "LoRA/QLoRA",
                    "hf_token": "hf_secret",
                    "local_datasets": [f"/data/{run_id}.jsonl"],
                    "lora_r": 8 * (i + 1),
                }
                for i, run_id in enumerate(MEMBERS)
            ],
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            hf_token = "hf_secret",
            local_datasets = ["/data/job_a.jsonl"],
            lora_r = 8,
        )
    assert ok
    config = captured["config"]
    assert config["batch_id"] == BATCH
    assert config["hf_token"] == "hf_secret"
    assert [m["run_id"] for m in config["batch_members"]] == list(MEMBERS)
    for i, member in enumerate(config["batch_members"]):
        assert member["config"]["lora_r"] == 8 * (i + 1)
        assert member["config"]["local_datasets"] == [f"/data/{MEMBERS[i]}.jsonl"]
        assert "hf_token" not in member["config"]
        assert "subject" not in member["config"]
    assert backend.batch_member_ids() == list(MEMBERS)
    assert backend._db_config is None
    assert db.get_run(BATCH) is None


def test_start_training_without_batch_is_unchanged(db, tmp_path):
    captured = {}
    real_spec = training_mod.account_process_spec

    def _spy(module, fn, cache_env, kwargs):
        captured["config"] = kwargs["config"]
        return real_spec(module, fn, cache_env, kwargs)

    backend = TrainingBackend()
    with (
        patch("core.training.training.prepare_gpu_selection", return_value = ([0], {})),
        patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
        patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
        patch("core.training.training.threading.Thread", return_value = _DummyThread()),
        patch("core.training.training.account_process_spec", side_effect = _spy),
    ):
        assert backend.start_training(
            job_id = "job_single", model_name = "unsloth/test", training_type = "LoRA/QLoRA"
        )
    assert "batch_members" not in captured["config"]
    assert "batch_id" not in captured["config"]
    assert backend.batch_member_ids() == []
    assert db.get_run("job_single")["status"] == "running"
