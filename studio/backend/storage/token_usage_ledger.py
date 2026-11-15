# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Token allowance usage: one row per finished non-API-key monitor request. API-key requests
already persist theirs in api_usage_events; keeping these apart leaves API usage stats unchanged."""

from __future__ import annotations

import sqlite3

USAGE_TABLE = "token_usage_events"


def ensure_token_usage_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {USAGE_TABLE} (
            id TEXT NOT NULL PRIMARY KEY,
            created_at INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL
        ) WITHOUT ROWID"""
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{USAGE_TABLE}_created_at ON {USAGE_TABLE}(created_at)"
    )


def record_session_usage(conn: sqlite3.Connection, receipt) -> bool:
    # Keyed by monitor id, so a repeated terminal callback cannot charge twice.
    inserted = conn.execute(
        f"INSERT OR IGNORE INTO {USAGE_TABLE} (id, created_at, total_tokens) VALUES (?, ?, ?)",
        (receipt.id, receipt.created_at, receipt.total_tokens),
    ).rowcount
    conn.commit()
    return bool(inserted)


def window_usage(conn: sqlite3.Connection, start_ms: int) -> int:
    (used,) = conn.execute(
        f"""SELECT COALESCE((SELECT SUM(total_tokens) FROM {USAGE_TABLE} WHERE created_at >= ?), 0)
                 + COALESCE((SELECT SUM(total_tokens) FROM api_usage_events
                             WHERE created_at >= ?), 0)""",
        (start_ms, start_ms),
    ).fetchone()
    return int(used)
