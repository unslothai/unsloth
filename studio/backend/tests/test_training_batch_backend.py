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


def test_member_error_before_progress_survives_a_transient_db_error(db, tmp_path, monkeypatch):
    b = _batch_backend(tmp_path)
    real, calls = db.create_run, []

    def flaky(**kwargs):
        calls.append(kwargs["id"])
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return real(**kwargs)

    monkeypatch.setattr(db, "create_run", flaky)
    monkeypatch.setattr(training_mod, "_DB_FINALIZE_RETRY_S", 0)
    b._handle_event({"type": "error", "member_run_id": "job_c", "error": "empty dataset"})
    assert calls == ["job_c", "job_c"]
    row = db.get_run("job_c")
    assert row["status"] == "error" and row["error_message"] == "empty dataset"


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

    def _select(gpu_ids, **kwargs):
        captured["sizing"] = kwargs
        return [0], {}

    backend = TrainingBackend()
    with (
        patch("core.training.training.prepare_gpu_selection", side_effect = _select),
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
                    "batch_size": i + 1,
                }
                for i, run_id in enumerate(MEMBERS)
            ],
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            hf_token = "hf_secret",
            local_datasets = ["/data/job_a.jsonl"],
            lora_r = 8,
            gpu_ids = [0],
        )
    assert ok
    # The GPU is chosen for every member's rows and adapters together, not the first member's.
    assert captured["sizing"]["batch_size"] == 1 + 2 + 3
    assert captured["sizing"]["lora_rank"] == 8 + 16 + 24
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


def test_batch_start_sizes_chat_coexistence_for_every_member(monkeypatch):
    import asyncio
    import contextlib

    import routes.training as tr
    import routes.training_vram as vram
    from models import TrainingStartRequest

    class _Backend:
        current_job_id = None

        def is_training_active(self):
            return False

        def start_training(
            self,
            job_id,
            *,
            before_spawn = None,
            **kwargs,
        ):
            self.current_job_id = job_id
            before_spawn()
            return True

    asked = []
    monkeypatch.setattr(tr, "get_training_backend", lambda: _Backend())
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(tr, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "_remote_untrainable_model_format", lambda *a, **k: None)
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)
    monkeypatch.setattr(vram, "coordinate_models_for_training", lambda keep: [keep()])
    monkeypatch.setattr(
        vram, "can_keep_chat_during_training", lambda **kw: asked.append(kw) or True
    )

    def request(r, batch_size):
        return TrainingStartRequest(
            model_name = "unsloth/tiny-model",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            load_in_4bit = False,
            lora_r = r,
            batch_size = batch_size,
        )

    started = asyncio.run(
        tr.start_training_batch(
            [request(8, 2), request(16, 4), request(32, 1)],
            current_subject = "alice",
            via_api_key = True,
        )
    )
    assert [s.status for s in started] == ["queued"] * 3
    assert [(kw["batch_size"], kw["lora_rank"]) for kw in asked] == [(2 + 4 + 1, 8 + 16 + 32)]


def test_xet_respawn_replays_pending_member_stops(monkeypatch, tmp_path):
    class _Queue:
        def __init__(self):
            self.items = []

        def put(self, item, *a, **k):
            self.items.append(item)

    class _Proc:
        pid = 4322

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout = None):
            pass

    class _Ctx:
        def __init__(self):
            self.spawned = []

        def Queue(self):
            return _Queue()

        def Process(self, **kwargs):
            self.spawned.append(kwargs)
            return _Proc(**kwargs)

    b = _batch_backend(tmp_path)
    b._last_full_config = {"model_name": "unsloth/tiny", "batch_members": []}
    b._last_hf_cache_env = {"HF_HOME": str(tmp_path / "hf")}
    b._proc = _Proc()
    b._batch_members["job_a"]["stop_requested"] = True
    b._batch_members["job_c"].update(stop_requested = True, terminal = True)
    ctx = _Ctx()
    monkeypatch.setattr(training_mod, "_CTX", ctx)
    monkeypatch.setattr(b, "_pump_loop", lambda: None)
    monkeypatch.setattr(b, "_open_worker_stderr_capture", lambda: None)

    assert b._respawn_worker_disable_xet() is True
    stop_queue = b._stop_queue
    assert stop_queue.items == [{"type": "stop_member", "run_id": "job_a", "save": True}]
