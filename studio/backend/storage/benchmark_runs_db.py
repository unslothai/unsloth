# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark runs in studio.db, stored like training runs: one row per run, one row per
measurement. `kind` is the benchmark family (config sweeps today; llama-bench and quality
suites later), so the same tables carry every phase."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from storage.studio_db import get_connection

RESULT_COLUMNS = (
    "variant",
    "rep",
    "warmup",
    "prompt_index",
    "tps",
    "prompt_tps",
    "prompt_tokens",
    "gen_tokens",
    "ttft_ms",
    "wall_ms",
    "client_tps",
    "draft_n",
    "draft_accepted",
    "load_ms",
    "at",
)

# The wire shape is the frontend's camelCase; the table keeps snake_case like its siblings.
_RESULT_KEYS = {
    "variant": "variant",
    "rep": "rep",
    "warmup": "warmup",
    "prompt_index": "promptIndex",
    "tps": "tps",
    "prompt_tps": "promptTps",
    "prompt_tokens": "promptTokens",
    "gen_tokens": "genTokens",
    "ttft_ms": "ttftMs",
    "wall_ms": "wallMs",
    "client_tps": "clientTps",
    "draft_n": "draftN",
    "draft_accepted": "draftAccepted",
    "load_ms": "loadMs",
    "at": "at",
}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id TEXT NOT NULL PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT 'sweep',
            sweep TEXT NOT NULL,
            model TEXT NOT NULL,
            gguf_variant TEXT,
            kv TEXT,
            context INTEGER,
            config_json TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            base_json TEXT NOT NULL,
            outcomes_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            finished_at INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at ON benchmark_runs(created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
            variant TEXT NOT NULL,
            rep INTEGER NOT NULL,
            warmup INTEGER NOT NULL DEFAULT 0,
            prompt_index INTEGER NOT NULL DEFAULT 0,
            tps REAL,
            prompt_tps REAL,
            prompt_tokens INTEGER,
            gen_tokens INTEGER,
            ttft_ms REAL,
            wall_ms REAL NOT NULL,
            client_tps REAL,
            draft_n INTEGER,
            draft_accepted INTEGER,
            load_ms REAL,
            at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_benchmark_results_run ON benchmark_results(run_id, variant, rep)"
    )


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii = False, separators = (",", ":"))


def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "sweep": row["sweep"],
        "model": row["model"],
        "ggufVariant": row["gguf_variant"],
        "kv": row["kv"],
        "context": row["context"],
        "config": _loads(row["config_json"], {}),
        "meta": _loads(row["meta_json"], {}),
        "base": _loads(row["base_json"], []),
        "outcomes": _loads(row["outcomes_json"], []),
        "createdAt": row["created_at"],
        "finishedAt": row["finished_at"],
    }


def _result_from_row(row: sqlite3.Row) -> dict[str, Any]:
    out = {key: row[col] for col, key in _RESULT_KEYS.items()}
    out["warmup"] = bool(row["warmup"])
    return out


def _result_params(run_id: str, result: dict[str, Any]) -> tuple:
    values: list[Any] = [run_id]
    for col in RESULT_COLUMNS:
        value = result.get(_RESULT_KEYS[col])
        values.append(int(bool(value)) if col == "warmup" else value)
    return tuple(values)


def list_runs(limit: int = 100) -> list[dict[str, Any]]:
    """Newest first, results left out: the list is for picking a run, not reading it."""
    conn = get_connection()
    try:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
        counts = dict(
            conn.execute(
                "SELECT run_id, COUNT(*) FROM benchmark_results WHERE warmup = 0 GROUP BY run_id"
            ).fetchall()
        )
        out = []
        for row in rows:
            summary = _summary_from_row(row)
            summary["resultCount"] = int(counts.get(row["id"], 0))
            out.append(summary)
        return out
    finally:
        conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        ensure_schema(conn)
        row = conn.execute("SELECT * FROM benchmark_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        run = _summary_from_row(row)
        run["results"] = [
            _result_from_row(r)
            for r in conn.execute(
                "SELECT * FROM benchmark_results WHERE run_id = ? ORDER BY at, rep", (run_id,)
            ).fetchall()
        ]
        return run
    finally:
        conn.close()


def upsert_run(run: dict[str, Any]) -> dict[str, Any]:
    """Save a run whole. The frontend sends the run again as rows land, so the results are
    replaced rather than appended: a partial save after a crash never doubles a row."""
    conn = get_connection()
    try:
        ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO benchmark_runs (
                id, kind, sweep, model, gguf_variant, kv, context, config_json, meta_json,
                base_json, outcomes_json, created_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                sweep = excluded.sweep,
                model = excluded.model,
                gguf_variant = excluded.gguf_variant,
                kv = excluded.kv,
                context = excluded.context,
                config_json = excluded.config_json,
                meta_json = excluded.meta_json,
                base_json = excluded.base_json,
                outcomes_json = excluded.outcomes_json,
                finished_at = excluded.finished_at
            """,
            (
                run["id"],
                run.get("kind") or "sweep",
                run["sweep"],
                run["model"],
                run.get("ggufVariant"),
                run.get("kv"),
                run.get("context"),
                _dumps(run.get("config") or {}),
                _dumps(run.get("meta") or {}),
                _dumps(run.get("base") or []),
                _dumps(run.get("outcomes") or []),
                int(run["createdAt"]),
                run.get("finishedAt"),
            ),
        )
        conn.execute("DELETE FROM benchmark_results WHERE run_id = ?", (run["id"],))
        conn.executemany(
            f"INSERT INTO benchmark_results (run_id, {', '.join(RESULT_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in range(len(RESULT_COLUMNS) + 1))})",
            [_result_params(run["id"], r) for r in run.get("results") or []],
        )
        conn.commit()
        return get_run(run["id"]) or run
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_run(run_id: str) -> bool:
    conn = get_connection()
    try:
        ensure_schema(conn)
        cur = conn.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
