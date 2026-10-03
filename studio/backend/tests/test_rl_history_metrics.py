# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
