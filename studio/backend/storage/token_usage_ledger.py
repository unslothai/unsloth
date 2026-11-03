# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sqlite3

LEDGER_TABLE = "token_usage_ledger"


def _count(row: str, path: str) -> str:
    return (
        f"(CASE WHEN json_type({row}, '{path}') IN ('integer', 'real') "
        f"THEN MAX(CAST(json_extract({row}, '{path}') AS INTEGER), 0) ELSE 0 END)"
    )


def _usage_columns(row: str) -> str:
    usage_prompt = _count(row, "$.contextUsage.promptTokens")
    server_prompt = (
        f"{_count(row, '$.serverTimings.prompt_n')} + {_count(row, '$.serverTimings.cache_n')}"
    )
    usage_completion = _count(row, "$.contextUsage.completionTokens")
    server_completion = _count(row, "$.serverTimings.predicted_n")
    timing_completion = _count(row, "$.timing.tokenCount")
    return f"""
        CASE WHEN {usage_prompt} > 0 THEN {usage_prompt} ELSE {server_prompt} END AS prompt_tokens,
        CASE WHEN {usage_completion} > 0 THEN {usage_completion}
             WHEN {server_completion} > 0 THEN {server_completion}
             ELSE {timing_completion} END AS completion_tokens,
        {_count(row, "$.contextUsage.totalTokens")} AS reported_total,
        CASE WHEN COALESCE(json_extract({row}, '$.responseDetails.providerType'), 'local')
                  IN ('local', '') THEN 'local' ELSE 'external' END AS source
    """


def _ledger_insert(message: str, rows: str = "") -> str:
    # Fork copies seed incremental context without charging it; MAX prevents refunds.
    return f"""
        INSERT INTO {LEDGER_TABLE}
            (message_id, thread_id, created_at, source,
             prompt_tokens, completion_tokens, total_tokens)
        SELECT u.id, u.thread_id, u.created_at,
               CASE WHEN EXISTS (
                   SELECT 1 FROM chat_threads t
                   WHERE t.id = u.thread_id AND t.forked_from_thread_id IS NOT NULL
                     AND u.created_at < t.created_at
               ) THEN 'inherited' ELSE u.source END,
               u.prompt_tokens, u.completion_tokens,
               CASE WHEN u.reported_total > 0 THEN u.reported_total
                    ELSE u.prompt_tokens + u.completion_tokens END
        FROM (
            SELECT {message}.id AS id, {message}.thread_id AS thread_id,
                   {message}.created_at AS created_at,
                   {_usage_columns(f"{message}.metadata_json")}
            {rows}
        ) AS u
        WHERE (u.prompt_tokens + u.completion_tokens > 0 OR u.reported_total > 0)
        ON CONFLICT(message_id) DO UPDATE SET
            source = excluded.source,
            prompt_tokens = MAX(prompt_tokens, excluded.prompt_tokens),
            completion_tokens = MAX(completion_tokens, excluded.completion_tokens),
            total_tokens = MAX(total_tokens, excluded.total_tokens)
    """


def ensure_token_usage_ledger(conn: sqlite3.Connection) -> None:
    existed = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (LEDGER_TABLE,)
        ).fetchone()
        is not None
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
            message_id TEXT NOT NULL PRIMARY KEY,
            thread_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_created_at ON {LEDGER_TABLE}(created_at)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_thread_created_at "
        f"ON {LEDGER_TABLE}(thread_id, created_at)"
    )
    for event, timing in (
        ("insert", "AFTER INSERT ON chat_messages"),
        ("update", "AFTER UPDATE OF metadata_json ON chat_messages"),
    ):
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {LEDGER_TABLE}_{event}
            {timing}
            WHEN NEW.role = 'assistant' AND json_valid(NEW.metadata_json)
            BEGIN
                {_ledger_insert("NEW")};
            END
            """
        )
    if not existed:
        conn.execute(
            _ledger_insert(
                "m",
                "FROM chat_messages m WHERE m.role = 'assistant' AND json_valid(m.metadata_json)",
            )
        )


def record_run_usage(conn: sqlite3.Connection, run_id: str, chunk: dict) -> None:
    metadata = {}
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        metadata["contextUsage"] = {
            camel: usage.get(snake, 0)
            for camel, snake in (
                ("promptTokens", "prompt_tokens"),
                ("completionTokens", "completion_tokens"),
                ("totalTokens", "total_tokens"),
            )
        }
    if isinstance(chunk.get("timings"), dict):
        metadata["serverTimings"] = chunk["timings"]
    if metadata:
        conn.execute(
            _ledger_insert(
                "m",
                """
            FROM (
                SELECT c.id, c.thread_id, c.created_at,
                       json_patch(c.metadata_json, ?) AS metadata_json
                FROM chat_messages c JOIN chat_generation_runs r ON r.assistant_message_id = c.id
                WHERE r.id = ? AND COALESCE(json_extract(r.request_json, '$.api_monitor_via_api_key'), 0) = 0
            ) m
        """,
            ),
            (json.dumps(metadata), run_id),
        )


def window_usage(conn: sqlite3.Connection, start_ms: int, method: str) -> dict[str, int]:
    if method == "total":
        usage = dict(
            conn.execute(
                f"SELECT source, SUM(total_tokens) FROM {LEDGER_TABLE} "
                "WHERE created_at >= ? AND source != 'inherited' GROUP BY source",
                (start_ms,),
            )
        )
    else:
        usage = dict(
            conn.execute(
                f"""
        WITH turns AS (
            SELECT source, created_at, completion_tokens,
                   prompt_tokens - LAG(prompt_tokens + completion_tokens, 1, 0) OVER (
                       PARTITION BY thread_id ORDER BY created_at, message_id
                   ) AS prompt_delta
            FROM {LEDGER_TABLE}
            WHERE thread_id IN (
                SELECT DISTINCT thread_id FROM {LEDGER_TABLE} WHERE created_at >= ?
            )
        )
        SELECT source, SUM(MAX(0, prompt_delta) + completion_tokens) FROM turns
        WHERE created_at >= ? AND source != 'inherited' GROUP BY source
        """,
                (start_ms, start_ms),
            )
        )
    usage["api"] = int(
        conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM api_usage_events WHERE created_at >= ?",
            (start_ms,),
        ).fetchone()[0]
    )
    return usage
