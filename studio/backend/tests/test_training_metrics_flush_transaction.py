# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A metric flush writes its batch and the run's progress through one connection and one commit."""

import pytest

import core.training.training as tr
from core.training.training import TrainingBackend
from storage import studio_db


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(tr, "account_is_retired", lambda: False)
    studio_db.create_run(
        id = "run-1",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00+00:00",
        total_steps = 100,
    )
    b = TrainingBackend()
    b.current_job_id = "run-1"
    b._db_run_created = True
    yield b
    studio_db.close_wal_keeper()


def _buffer(b, steps):
    b._metric_buffer[:] = [
        {"step": s, "loss": 1.0 / s, "learning_rate": 1e-4, "grad_norm": 0.5} for s in steps
    ]
    b._progress.step = steps[-1]
    b._progress.loss = 1.0 / steps[-1]
    b._progress.elapsed_seconds = float(steps[-1])


def _rows():
    conn = studio_db.get_connection()
    try:
        metrics = [
            tuple(r)
            for r in conn.execute(
                "SELECT step, loss, learning_rate, grad_norm FROM training_metrics "
                "WHERE run_id = 'run-1' ORDER BY step"
            )
        ]
        run = tuple(
            conn.execute(
                "SELECT final_step, final_loss, duration_seconds FROM training_runs "
                "WHERE id = 'run-1'"
            ).fetchone()
        )
    finally:
        conn.close()
    return metrics, run


def test_flush_opens_one_connection_and_commits_once(backend, monkeypatch):
    opened = []
    real = studio_db.get_connection

    def counting(*a, **k):
        conn = real(*a, **k)
        opened.append(conn)
        return conn

    monkeypatch.setattr(studio_db, "get_connection", counting)
    _buffer(backend, list(range(1, 11)))
    backend._flush_metrics_to_db()
    assert len(opened) == 1
    monkeypatch.setattr(studio_db, "get_connection", real)
    metrics, run = _rows()
    assert [m[0] for m in metrics] == list(range(1, 11))
    assert run == (10, 0.1, 10.0)


def test_failed_progress_write_leaves_metrics_unwritten_and_requeued(backend):
    conn = studio_db.get_connection()
    conn.execute(
        "CREATE TRIGGER fail_progress BEFORE UPDATE OF final_step ON training_runs "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    conn.commit()
    conn.close()
    _buffer(backend, [1, 2, 3])
    backend._flush_metrics_to_db()
    assert [m["step"] for m in backend._metric_buffer] == [1, 2, 3]
    assert _rows()[0] == []

    conn = studio_db.get_connection()
    conn.execute("DROP TRIGGER fail_progress")
    conn.commit()
    conn.close()
    backend._flush_metrics_to_db()
    assert backend._metric_buffer == []
    metrics, run = _rows()
    assert [m[0] for m in metrics] == [1, 2, 3]
    assert run == (3, pytest.approx(1 / 3), 3.0)


def test_reflushed_step_keeps_upsert_semantics(backend):
    _buffer(backend, [1, 2])
    backend._flush_metrics_to_db()
    backend._metric_buffer[:] = [{"step": 2, "eval_loss": 0.7}]
    backend._flush_metrics_to_db()
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT loss, eval_loss FROM training_metrics WHERE run_id = 'run-1' AND step = 2"
        ).fetchone()
    finally:
        conn.close()
    assert tuple(row) == (0.5, 0.7)
