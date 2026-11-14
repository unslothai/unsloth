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
    # Live snapshots carry only resumeTiming; both counts include the continued reply's tokens.
    saved_count = _count(row, "$.timing.tokenCount")
    live_count = _count(row, "$.resumeTiming.tokenCount")
    timing_completion = (
        f"MAX((CASE WHEN {saved_count} > 0 THEN {saved_count} ELSE {live_count} END)"
        f" - {_count(row, '$.tokenUsagePriorCount')}, 0)"
    )
    return f"""
        CASE WHEN {usage_prompt} > 0 THEN {usage_prompt}
             WHEN {server_prompt} > 0 THEN {server_prompt}
             ELSE {_count(row, "$.tokenUsageEstimatedPrompt")} END AS prompt_tokens,
        CASE WHEN {usage_completion} > 0 THEN {usage_completion}
             WHEN {server_completion} > 0 THEN {server_completion}
             ELSE {timing_completion} END AS completion_tokens,
        {_count(row, "$.contextUsage.totalTokens")} AS reported_total,
        NULLIF({_count(row, "$.contextUsage.contextTokens")}, 0) AS context_tokens,
        CASE WHEN {usage_prompt} + {usage_completion} + {server_completion} + {server_prompt} > 0
             THEN 0 ELSE 1 END AS estimated,
        json_extract({row}, '$.tokenUsageTemporaryKey') AS temporary_key,
        CASE WHEN json_extract({row}, '$.tokenUsageCharged') = 1 THEN 'inherited'
             WHEN COALESCE(json_extract({row}, '$.responseDetails.providerType'),
                           json_extract({row}, '$.tokenUsageProvider'), 'local')
                  IN ('local', '') THEN 'local' ELSE 'external' END AS source
    """


def _ledger_insert(message: str, rows: str = "") -> str:
    # Fork copies seed incremental context without charging it; MAX prevents refunds.
    return f"""
        INSERT INTO {LEDGER_TABLE}
            (message_id, thread_id, created_at, source,
             prompt_tokens, completion_tokens, total_tokens, context_tokens, parent_message_id,
             estimated, temporary_key)
        SELECT u.id, u.thread_id, u.created_at, u.source,
               u.prompt_tokens, u.completion_tokens,
               CASE WHEN u.reported_total > 0 THEN u.reported_total
                    ELSE u.prompt_tokens + u.completion_tokens END, u.context_tokens,
               u.parent_message_id, u.estimated, u.temporary_key
        FROM (
            SELECT {message}.id AS id, {message}.thread_id AS thread_id,
                   {message}.created_at AS created_at,
                   COALESCE(NULLIF(json_extract({message}.metadata_json, '$.tokenUsageParentId'), ''),
                   CASE WHEN {message}.parent_id IS NULL THEN NULL ELSE COALESCE((
                       SELECT p.parent_id FROM chat_messages p WHERE p.id = {message}.parent_id
                   ), '') END) AS parent_message_id,
                   {_usage_columns(f"{message}.metadata_json")}
            {rows}
        ) AS u
        WHERE (u.prompt_tokens + u.completion_tokens > 0 OR u.reported_total > 0)
          AND NOT EXISTS (
              SELECT 1 FROM chat_generation_runs r WHERE r.assistant_message_id = u.id
              AND json_extract(r.request_json, '$.api_monitor_via_api_key') = 1
          )
        ON CONFLICT(message_id) DO UPDATE SET
            -- An existing debit is never re-labelled exempt by later metadata.
            source = CASE WHEN source IN ('inherited', 'api') OR excluded.source = 'inherited'
                THEN source ELSE excluded.source END,
            -- A reported count replaces a text-length estimate and is never moved by one.
            prompt_tokens = CASE WHEN estimated > excluded.estimated THEN excluded.prompt_tokens
                WHEN estimated < excluded.estimated THEN prompt_tokens
                ELSE MAX(prompt_tokens, excluded.prompt_tokens) END,
            completion_tokens = CASE WHEN estimated > excluded.estimated
                THEN excluded.completion_tokens
                WHEN estimated < excluded.estimated THEN completion_tokens
                ELSE MAX(completion_tokens, excluded.completion_tokens) END,
            total_tokens = CASE WHEN estimated > excluded.estimated THEN excluded.total_tokens
                WHEN estimated < excluded.estimated THEN total_tokens
                ELSE MAX(total_tokens, excluded.total_tokens) END,
            estimated = MIN(estimated, excluded.estimated),
            temporary_key = COALESCE(excluded.temporary_key, temporary_key),
            context_tokens = CASE WHEN EXISTS (
                SELECT 1 FROM {LEDGER_TABLE} later
                WHERE later.thread_id = {LEDGER_TABLE}.thread_id
                AND (later.created_at, later.message_id) > ({LEDGER_TABLE}.created_at, {LEDGER_TABLE}.message_id)
            ) THEN context_tokens ELSE COALESCE(excluded.context_tokens, context_tokens) END,
            parent_message_id = excluded.parent_message_id
    """


def ensure_token_usage_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_generation_runs_assistant "
        "ON chat_generation_runs(assistant_message_id)"
    )
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
            total_tokens INTEGER NOT NULL,
            context_tokens INTEGER,
            parent_message_id TEXT,
            estimated INTEGER NOT NULL DEFAULT 0,
            temporary_key TEXT
        ) WITHOUT ROWID
        """
    )
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({LEDGER_TABLE})")}
    missing = {
        name: kind
        for name, kind in (
            ("context_tokens", "INTEGER"),
            ("parent_message_id", "TEXT"),
            ("estimated", "INTEGER NOT NULL DEFAULT 0"),
            ("temporary_key", "TEXT"),
        )
        if name not in columns
    }
    for name, column_type in missing.items():
        conn.execute(f"ALTER TABLE {LEDGER_TABLE} ADD COLUMN {name} {column_type}")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_created_at ON {LEDGER_TABLE}(created_at)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_temporary_key "
        f"ON {LEDGER_TABLE}(temporary_key) WHERE temporary_key IS NOT NULL"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LEDGER_TABLE}_thread_created_at "
        f"ON {LEDGER_TABLE}(thread_id, created_at)"
    )
    for event, timing in (
        ("insert", "AFTER INSERT ON chat_messages"),
        ("update", "AFTER UPDATE OF metadata_json ON chat_messages"),
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {LEDGER_TABLE}_{event}")
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
    if not existed or missing:
        conn.execute(
            _ledger_insert(
                "m",
                "FROM chat_messages m WHERE m.role = 'assistant' AND json_valid(m.metadata_json)",
            )
        )
        conn.execute(f"""
            WITH RECURSIVE inherited(id, parent_id, thread_id) AS (
                SELECT m.id, m.parent_id, m.thread_id FROM chat_messages m
                JOIN chat_threads t ON t.fork_boundary_message_id = m.id
                WHERE t.forked_from_thread_id IS NOT NULL
                UNION
                SELECT m.id, m.parent_id, m.thread_id FROM chat_messages m
                JOIN inherited i ON m.id = i.parent_id AND m.thread_id = i.thread_id
            )
            UPDATE {LEDGER_TABLE} SET source = 'inherited'
            WHERE message_id IN (SELECT id FROM inherited)
        """)


def record_run_usage(conn: sqlite3.Connection, run_id: str, chunk: dict) -> None:
    metadata = {}
    usage = chunk.get("usage")
    if chunk.get("type") == "context_usage":
        # A pass reports before an approval wait the run may never leave; debit what the final
        # frame would: this prompt plus every pass's completion.
        (completion,) = conn.execute(
            """SELECT COALESCE(SUM(CAST(json_extract(payload_json, '$.completion_tokens') AS INTEGER)), 0)
               FROM chat_generation_events WHERE run_id = ? AND event_type = 'chunk'
               AND json_extract(payload_json, '$.type') = 'context_usage'""",
            (run_id,),
        ).fetchone()
        prompt = chunk.get("prompt_tokens") or 0
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "context_tokens": chunk.get("context_tokens"),
        }
    if isinstance(usage, dict):
        metadata["contextUsage"] = {
            camel: usage.get(snake, 0)
            for camel, snake in (
                ("promptTokens", "prompt_tokens"),
                ("completionTokens", "completion_tokens"),
                ("totalTokens", "total_tokens"),
                ("contextTokens", "context_tokens"),
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
                SELECT c.id, c.thread_id, c.created_at, c.parent_id,
                       json_patch(c.metadata_json, ?) AS metadata_json
                FROM chat_messages c JOIN chat_generation_runs r ON r.assistant_message_id = c.id
                WHERE r.id = ?
            ) m
        """,
            ),
            (json.dumps(metadata), run_id),
        )


def _text_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return 0


def settle_interrupted_run(conn: sqlite3.Connection, run_id: str) -> None:
    """A run stopped before its next usage frame (e.g. unattended) still debits an estimate of the
    output it persisted after the last one; reported usage, if it ever arrives, replaces it."""
    output = 0
    reported = False
    for (payload,) in conn.execute(
        "SELECT payload_json FROM chat_generation_events WHERE run_id = ? AND event_type = 'chunk'"
        " ORDER BY seq",
        (run_id,),
    ):
        try:
            chunk = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(chunk, dict):
            continue
        if (
            chunk.get("type") == "context_usage"
            or isinstance(chunk.get("usage"), dict)
            or isinstance(chunk.get("timings"), dict)
        ):
            output, reported = 0, True
            continue
        if chunk.get("type") == "tool_args":
            # Generated arguments (e.g. render_html), sent as additive slices; tool output is not.
            output += _text_chars(chunk.get("text"))
            continue
        choices = chunk.get("choices")
        for choice in choices if isinstance(choices, list) else ():
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                output += _text_chars(delta.get("content")) + _text_chars(
                    delta.get("reasoning_content")
                )
    if not output:
        return
    tail = -(-output // 4)
    if reported:
        # The last frame's counts stand; add only what streamed after it.
        conn.execute(
            f"""UPDATE {LEDGER_TABLE} SET completion_tokens = completion_tokens + ?,
                       total_tokens = total_tokens + ?
                WHERE message_id = (
                    SELECT assistant_message_id FROM chat_generation_runs WHERE id = ?
                )""",
            (tail, tail, run_id),
        )
        return
    row = conn.execute(
        "SELECT request_json FROM chat_generation_runs WHERE id = ?", (run_id,)
    ).fetchone()
    try:
        messages = json.loads(row[0]).get("messages") or [] if row else []
    except (TypeError, ValueError, AttributeError):
        messages = []
    prompt = sum(_text_chars(m.get("content")) for m in messages if isinstance(m, dict))
    metadata = {"timing": {"tokenCount": tail}, "tokenUsageEstimatedPrompt": -(-prompt // 4)}
    conn.execute(
        _ledger_insert(
            "m",
            """
            FROM (
                SELECT c.id, c.thread_id, c.created_at, c.parent_id,
                       json_patch(c.metadata_json, ?) AS metadata_json
                FROM chat_messages c JOIN chat_generation_runs r ON r.assistant_message_id = c.id
                WHERE r.id = ?
            ) m
        """,
        ),
        (json.dumps(metadata), run_id),
    )


def record_temporary_usage(receipt) -> bool:
    from storage.studio_db import get_connection
    conn = get_connection()
    try:
        key = receipt.temporary_message_id or receipt.id
        saved = conn.execute(
            f"SELECT message_id FROM {LEDGER_TABLE} WHERE temporary_key = ?", (key,)
        ).fetchone()
        if saved is not None:
            # Saved before this receipt landed: its row already debits the reply, so merge.
            conn.execute(
                f"""UPDATE {LEDGER_TABLE} SET
                        prompt_tokens = MAX(prompt_tokens, ?),
                        completion_tokens = MAX(completion_tokens, ?),
                        total_tokens = MAX(total_tokens, ?)
                    WHERE message_id = ?""",
                (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens, saved[0]),
            )
            conn.commit()
            return True
        conn.execute(
            f"""INSERT INTO {LEDGER_TABLE}
                (message_id, thread_id, created_at, source, prompt_tokens, completion_tokens, total_tokens, context_tokens, parent_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    prompt_tokens = MAX(prompt_tokens, excluded.prompt_tokens),
                    completion_tokens = MAX(completion_tokens, excluded.completion_tokens),
                    total_tokens = MAX(total_tokens, excluded.total_tokens),
                    context_tokens = COALESCE(context_tokens, excluded.context_tokens)""",
            (
                receipt.temporary_message_id or receipt.id,
                receipt.temporary_thread,
                receipt.created_at,
                receipt.temporary_source,
                receipt.prompt_tokens,
                receipt.completion_tokens,
                receipt.total_tokens,
                receipt.context_tokens,
                receipt.temporary_parent_id,
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


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
            SELECT t.source, t.created_at, t.completion_tokens,
                   t.prompt_tokens - CASE WHEN t.parent_message_id IS NULL THEN
                       LAG(COALESCE(t.context_tokens, t.prompt_tokens + t.completion_tokens), 1, 0)
                       OVER (PARTITION BY t.thread_id ORDER BY t.created_at, t.message_id)
                   ELSE COALESCE(p.context_tokens, p.prompt_tokens + p.completion_tokens, 0)
                   END AS prompt_delta
            FROM {LEDGER_TABLE} t
            LEFT JOIN {LEDGER_TABLE} p ON p.message_id = t.parent_message_id
            WHERE t.thread_id IN (
                SELECT DISTINCT thread_id FROM {LEDGER_TABLE} WHERE created_at >= ?
            )
        )
        SELECT source, SUM(MAX(0, prompt_delta) + completion_tokens) FROM turns
        WHERE created_at >= ? AND source != 'inherited' GROUP BY source
        """,
                (start_ms, start_ms),
            )
        )
    # API receipts supply the debit; ledger rows only retain origin.
    usage["api"] = int(
        conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM api_usage_events WHERE created_at >= ?",
            (start_ms,),
        ).fetchone()[0]
    )
    return usage
