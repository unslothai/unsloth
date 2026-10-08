# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import math
import sqlite3

import pytest

from storage import studio_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.get_connection().close()
    yield tmp_path / "studio.db"
    studio_db.close_wal_keeper()


def _run(run_id = "r1"):
    studio_db.create_run(
        run_id, "unsloth/Qwen3-0.6B", "openai/gsm8k", "{}", "2026-10-02T00:00:00", 3
    )


def test_rl_metrics_round_trip_into_history(db):
    _run()
    studio_db.insert_metrics_batch(
        "r1",
        [
            {
                "step": 1,
                "loss": 0.0,
                "rl": {"reward": 0.5, "kl": 0.0, "rewards/exact_answer/mean": 1.0},
            },
            {
                "step": 2,
                "loss": 0.1,
                "rl": {"reward": float("nan"), "kl": math.inf, "reward_std": 0.3},
            },
            {"step": 3, "loss": 0.2},
        ],
    )
    assert studio_db.get_run_metrics("r1")["rl_history"] == [
        {"step": 1, "reward": 0.5, "kl": 0.0, "rewards/exact_answer/mean": 1.0},
        {"step": 2, "reward_std": 0.3},
    ]


def test_a_later_upsert_without_rl_keeps_the_rl_values(db):
    _run()
    studio_db.insert_metrics_batch("r1", [{"step": 1, "rl": {"reward": 2.0}}])
    studio_db.insert_metrics_batch("r1", [{"step": 1, "loss": 0.4}])
    assert studio_db.get_run_metrics("r1")["rl_history"] == [{"step": 1, "reward": 2.0}]


def test_sft_runs_have_no_rl_history(db):
    _run()
    studio_db.insert_metrics_batch("r1", [{"step": 1, "loss": 1.2}])
    assert studio_db.get_run_metrics("r1")["rl_history"] == []


def test_an_old_database_gains_the_column(tmp_path, monkeypatch):
    path = tmp_path / "studio.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE training_metrics (id INTEGER PRIMARY KEY, run_id TEXT, step INTEGER)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: path)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.get_connection().close()
    studio_db.close_wal_keeper()
    cols = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(training_metrics)")}
    assert "rl_json" in cols


def test_sse_replay_sends_each_step_its_own_rl_metrics(monkeypatch):
    import asyncio
    import types

    import routes.training as rt

    progress = types.SimpleNamespace(
        step = 3,
        total_steps = 3,
        loss = 0.5,
        learning_rate = 1e-5,
        epoch = 0.1,
        grad_norm = None,
        num_tokens = None,
        eval_loss = None,
        elapsed_seconds = None,
        eta_seconds = None,
        rl_metrics = {"reward": 9.0},
    )
    backend = types.SimpleNamespace(
        current_job_id = "job-1",
        step_history = [1, 2, 3],
        loss_history = [0.7, 0.6, 0.5],
        lr_history = [1e-5] * 3,
        rl_metric_history = [{"step": 1, "reward": 1.0}, {"step": 2, "reward": 2.0}],
        eval_enabled = False,
        trainer = types.SimpleNamespace(training_progress = progress),
        is_training_active = lambda: False,
    )
    monkeypatch.setattr(rt, "get_training_backend", lambda: backend)

    class _Request:
        headers = {"last-event-id": "0"}

        async def is_disconnected(self):
            return False

    async def _drain(response):
        out = []
        async for chunk in response.body_iterator:
            out.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(out)

    response = asyncio.run(rt.stream_training_progress(_Request(), current_subject = "tester"))
    raw = asyncio.run(asyncio.wait_for(_drain(response), 15))
    replayed = {}
    for block in raw.split("\n\n"):
        lines = block.strip().splitlines()
        if "event: progress" in lines:
            data = json.loads(next(l[6:] for l in lines if l.startswith("data: ")))
            replayed.setdefault(data["step"], data.get("rl_metrics"))
    assert replayed[1] == {"reward": 1.0}
    assert replayed[2] == {"reward": 2.0}
