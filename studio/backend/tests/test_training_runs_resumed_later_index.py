# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training history's resumed-later check is answered from an index, and nothing else changes plan."""

import pytest

from storage import studio_db

_INDEX = "idx_training_runs_resumed_later"
_EXISTS = (
    "SELECT r.id FROM training_runs r WHERE EXISTS ("
    "SELECT 1 FROM training_runs newer WHERE newer.output_dir = r.output_dir "
    "AND newer.status IN ('stopped', 'completed', 'error', 'running') "
    "AND newer.started_at > r.started_at)"
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield
    studio_db.close_wal_keeper()


def _run(
    i,
    status,
    output_dir,
    started_at,
    model = "org/m",
):
    studio_db.create_run(
        id = f"run-{i}",
        model_name = model,
        dataset_name = "d",
        config_json = "{}",
        started_at = started_at,
        total_steps = 10,
        output_dir = output_dir,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE training_runs SET status = ? WHERE id = ?", (status, f"run-{i}"))
        conn.commit()
    finally:
        conn.close()


def _plan(sql, args = ()):
    conn = studio_db.get_connection()
    try:
        return " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, args))
    finally:
        conn.close()


def test_resumed_later_subquery_searches_the_index(db):
    plan = _plan(_EXISTS)
    # SQLite >= 3.53 prints "SEARCH newer EXISTS USING INDEX".
    assert "SEARCH newer" in plan and f"USING INDEX {_INDEX}" in plan
    assert "SCAN newer" not in plan


def test_other_training_run_queries_keep_their_plans(db):
    for sql, args in (
        ("SELECT * FROM training_runs WHERE output_dir = ? ORDER BY started_at DESC", ("x",)),
        (
            "SELECT model_name FROM training_runs WHERE output_dir IN (?, ?) ORDER BY started_at DESC",
            ("a", "b"),
        ),
        (
            "SELECT * FROM training_runs WHERE output_dir = ? AND status IN ('stopped', 'error') "
            "ORDER BY started_at DESC LIMIT 1",
            ("x",),
        ),
        ("SELECT id, output_dir FROM training_runs WHERE output_dir IS NOT NULL", ()),
        ("SELECT id FROM training_runs ORDER BY started_at DESC", ()),
    ):
        assert _INDEX not in _plan(sql, args), sql


def test_resumed_later_and_order_match_a_direct_reading(db):
    rows = [
        (0, "stopped", "/o/a", "2026-01-01T00:00:01"),
        (1, "error", "/o/a", "2026-01-01T00:00:02"),
        (2, "completed", "/o/a", "2026-01-01T00:00:02"),
        (3, "stopped", "/o/b", "2026-01-01T00:00:03"),
        (4, "cancelled", "/o/b", "2026-01-01T00:00:04"),
        (5, "error", None, "2026-01-01T00:00:05"),
        (6, "stopped", "/o/c", "2026-01-01T00:00:05"),
        (7, "running", "/o/c", "2026-01-01T00:00:06"),
        (8, "stopped", "/o/a/", "2026-01-01T00:00:07"),
    ]
    for row in rows:
        _run(*row)
    expected = {}
    for i, status, out, started in rows:
        expected[f"run-{i}"] = int(
            status in ("stopped", "error")
            and out is not None
            and any(
                o == out and s in ("stopped", "completed", "error", "running") and t > started
                for _, s, o, t in rows
            )
        )
    listed = studio_db.list_runs(limit = 50)["runs"]
    assert {r["id"]: r["resumed_later"] for r in listed} == expected
    assert [r["started_at"] for r in listed] == sorted((r[3] for r in rows), reverse = True)
    assert studio_db.get_run("run-0")["resumed_later"] == 1
    assert studio_db.get_run("run-3")["resumed_later"] == 0
    # run-2 started at the same second as run-1, so it is not newer and run-1 stays resumable.
    assert studio_db.get_resumable_run_by_output_dir("/o/a")["id"] == "run-1"
    assert studio_db.get_resumable_run_by_output_dir("/o/b")["id"] == "run-3"


def test_index_is_created_once_on_a_database_made_without_it(db):
    _run(0, "stopped", "/o/a", "2026-01-01T00:00:01")
    conn = studio_db.get_connection()
    conn.execute(f"DROP INDEX {_INDEX}")
    conn.commit()
    conn.close()
    studio_db.reset_schema_state_for_tests()
    for _ in range(2):
        studio_db.reset_schema_state_for_tests()
        conn = studio_db.get_connection()
        try:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?", (_INDEX,)
                )
            ]
        finally:
            conn.close()
        assert names == [_INDEX]
    assert studio_db.list_runs()["runs"][0]["id"] == "run-0"
