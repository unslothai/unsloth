# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """studio_db pointed at a throwaway sqlite file.

    The DB path is env-driven: studio_db_path() returns
    studio_root() / "studio.db", where studio_root() reads
    UNSLOTH_STUDIO_HOME. Setting that env var to tmp_path and
    resetting _schema_ready ensures get_connection() opens the
    temp DB and re-runs _ensure_schema there — never touching
    the user's real studio.db.
    """
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return studio_db


def test_create_and_get_eval_run(db):
    db.create_eval_run(
        id="run1", model_identifier="hf/model", dataset_ref="data.jsonl",
        metric_name="exact_match", config_json="{}", started_at="2026-05-26T00:00:00",
        num_examples=2,
    )
    run = db.get_eval_run("run1")
    assert run["id"] == "run1"
    assert run["status"] == "running"
    assert run["metric_name"] == "exact_match"
    assert run["num_examples"] == 2


def test_insert_results_and_finish(db):
    db.create_eval_run(
        id="run2", model_identifier="m", dataset_ref="d", metric_name="exact_match",
        config_json="{}", started_at="2026-05-26T00:00:00", num_examples=2,
    )
    db.insert_eval_result(run_id="run2", idx=0, input_text="a", prediction_text="x",
                          reference_text="x", score=1.0, breakdown_json=None, error=None)
    db.insert_eval_result(run_id="run2", idx=1, input_text="b", prediction_text="y",
                          reference_text="z", score=0.0, breakdown_json=None, error=None)
    db.finish_eval_run(id="run2", status="completed", ended_at="2026-05-26T00:01:00",
                       avg_score=0.5, error_message=None)
    run = db.get_eval_run("run2")
    assert run["status"] == "completed"
    assert run["avg_score"] == 0.5
    results = db.get_eval_results("run2", limit=10, offset=0)
    assert results["total"] == 2
    assert [r["idx"] for r in results["results"]] == [0, 1]
    assert results["results"][0]["score"] == 1.0


def test_list_eval_runs_newest_first(db):
    db.create_eval_run(id="r_old", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.create_eval_run(id="r_new", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T01:00:00", num_examples=1)
    listing = db.list_eval_runs(limit=50, offset=0)
    assert listing["total"] == 2
    assert listing["runs"][0]["id"] == "r_new"  # newest first


def test_cleanup_marks_running_eval_interrupted(db):
    db.create_eval_run(id="stuck", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.cleanup_orphaned_runs()
    assert db.get_eval_run("stuck")["status"] == "interrupted"


def test_insert_eval_result_upserts_same_idx(db):
    db.create_eval_run(id="up", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.insert_eval_result(run_id="up", idx=0, input_text="i", prediction_text="first",
                          reference_text="r", score=0.0, breakdown_json=None, error="e")
    # re-insert same (run_id, idx) -> overwrite, not duplicate
    db.insert_eval_result(run_id="up", idx=0, input_text="i", prediction_text="second",
                          reference_text="r", score=1.0, breakdown_json=None, error=None)
    results = db.get_eval_results("up", limit=10, offset=0)
    assert results["total"] == 1
    assert results["results"][0]["prediction_text"] == "second"
    assert results["results"][0]["score"] == 1.0
    assert results["results"][0]["error"] is None


def test_deleting_eval_run_cascades_results(db):
    db.create_eval_run(id="cas", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.insert_eval_result(run_id="cas", idx=0, input_text="i", prediction_text="p",
                          reference_text="r", score=1.0, breakdown_json=None, error=None)
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM eval_runs WHERE id=?", ("cas",))
        conn.commit()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM eval_results WHERE run_id=?", ("cas",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert remaining == 0  # ON DELETE CASCADE removed the child rows
