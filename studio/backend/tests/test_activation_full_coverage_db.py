# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full-coverage counts and pinned columns survive the DB, including a DB from an earlier build."""

import sqlite3

from storage import studio_db

_COVERAGE = {
    "0": {
        "channels": 1024,
        "dead": 2,
        "constant": 40,
        "outliers": 3,
        "top_outliers": [[35, 37.5], [7, float("inf")]],
    }
}


def test_coverage_and_pinned_channels_round_trip():
    studio_db.create_run("run-cov", "unsloth/test", "data", "{}", "2026-10-29T00:00:00Z", 10)
    studio_db.upsert_activation_metadata(
        "run-cov",
        {
            "model_name": "tiny",
            "captured_channels": [1, 35],
            "pinned_channels": [35],
            "full_coverage": True,
        },
    )
    studio_db.insert_activation_record(
        "run-cov", 0, 1.0, {"0": {"mean_abs": [0.1]}}, None, None, _COVERAGE
    )
    studio_db.insert_activation_record("run-cov", 3, 0.9, {"0": {"mean_abs": [0.1]}}, None, None)

    meta = studio_db.get_activation_metadata("run-cov")
    assert meta["pinned_channels"] == [35]
    assert meta["full_coverage"] is True
    first, second = studio_db.get_activation_records("run-cov")
    assert first["coverage"]["0"]["dead"] == 2
    assert first["coverage"]["0"]["top_outliers"] == [[35, 37.5], [7, None]]
    assert second["coverage"] is None


def test_a_db_from_before_full_coverage_gains_the_columns():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE activation_run_metadata (job_id TEXT PRIMARY KEY, model_name TEXT, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE activation_records (id INTEGER PRIMARY KEY, job_id TEXT, step INTEGER, captured_at TEXT)"
    )
    studio_db._ensure_schema(conn)
    meta_cols = {r[1] for r in conn.execute("PRAGMA table_info(activation_run_metadata)")}
    record_cols = {r[1] for r in conn.execute("PRAGMA table_info(activation_records)")}
    assert {"pinned_channels_json", "full_coverage"} <= meta_cols
    assert "coverage_json" in record_cols
