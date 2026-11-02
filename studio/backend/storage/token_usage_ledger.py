# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Append-only token ledger behind the usage limiter. Triggers copy each assistant reply's token
counts out of chat_messages, so deleting a chat cannot hand its tokens back to the limit."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

LEDGER_TABLE = "token_usage_ledger"

# Same fallbacks as the Profile stats fold: the usage chunk first, llama.cpp's serverTimings when no
# usage chunk was sent, and the adapter's own token count for the completion as a last resort.


def _count(row: str, path: str) -> str:
    """A non-negative integer at ``path``, or 0 for anything that is not a JSON number."""
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
    """INSERT .. SELECT of chat_messages rows: ``NEW`` in a trigger, or alias ``m`` over ``rows`` in the
    backfill. Fork copies keep their source's timestamps, which predate the fork thread; the original
    row already counted them, so they are skipped. MAX on conflict keeps a rewrite from shrinking a
    counted reply."""
    return f"""
        INSERT INTO {LEDGER_TABLE}
            (message_id, thread_id, created_at, source,
             prompt_tokens, completion_tokens, total_tokens)
        SELECT u.id, u.thread_id, u.created_at, u.source, u.prompt_tokens, u.completion_tokens,
               CASE WHEN u.reported_total > 0 THEN u.reported_total
                    ELSE u.prompt_tokens + u.completion_tokens END
        FROM (
            SELECT {message}.id AS id, {message}.thread_id AS thread_id,
                   {message}.created_at AS created_at,
                   {_usage_columns(f"{message}.metadata_json")}
            {rows}
        ) AS u
        WHERE (u.prompt_tokens + u.completion_tokens > 0 OR u.reported_total > 0)
          AND NOT EXISTS (
              SELECT 1 FROM chat_threads t
              WHERE t.id = u.thread_id
                AND t.forked_from_thread_id IS NOT NULL
                AND u.created_at < t.created_at
          )
        ON CONFLICT(message_id) DO UPDATE SET
            source = excluded.source,
            prompt_tokens = MAX(prompt_tokens, excluded.prompt_tokens),
            completion_tokens = MAX(completion_tokens, excluded.completion_tokens),
            total_tokens = MAX(total_tokens, excluded.total_tokens)
    """


def ensure_token_usage_ledger(conn: sqlite3.Connection) -> None:
    """Create the ledger and its triggers; the first creation backfills from the stored history."""
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
    # json_valid in WHEN: a malformed metadata blob must never abort the chat write that carries it.
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


@dataclass
class WindowUsage:
    """Tokens since a period start, per source: ``local`` and ``external`` chat replies and ``api``
    requests made with an API key."""

    total: dict[str, int] = field(default_factory = lambda: {"local": 0, "external": 0, "api": 0})
    incremental: dict[str, int] = field(
        default_factory = lambda: {"local": 0, "external": 0, "api": 0}
    )


def window_usage(conn: sqlite3.Connection, start_ms: int) -> WindowUsage:
    """Usage from ``start_ms`` on. Incremental counting charges a chat reply only for the prompt tokens
    the previous reply in its thread had not already seen, plus its completion; API requests are
    stateless, so both methods charge them in full."""
    usage = WindowUsage()
    rows = conn.execute(
        f"""
        SELECT thread_id, created_at, source, prompt_tokens, completion_tokens, total_tokens
        FROM {LEDGER_TABLE}
        WHERE thread_id IN (
            SELECT DISTINCT thread_id FROM {LEDGER_TABLE} WHERE created_at >= ?
        )
        ORDER BY thread_id, created_at, message_id
        """,
        (start_ms,),
    ).fetchall()
    thread = None
    seen = 0
    for thread_id, created_at, source, prompt, completion, total in rows:
        if thread_id != thread:
            thread, seen = thread_id, 0
        if created_at >= start_ms:
            usage.total[source] = usage.total.get(source, 0) + total
            usage.incremental[source] = (
                usage.incremental.get(source, 0) + max(0, prompt - seen) + completion
            )
        seen = prompt + completion
    api_total = conn.execute(
        "SELECT COALESCE(SUM(total_tokens), 0) FROM api_usage_events WHERE created_at >= ?",
        (start_ms,),
    ).fetchone()[0]
    usage.total["api"] = int(api_total)
    usage.incremental["api"] = int(api_total)
    return usage
