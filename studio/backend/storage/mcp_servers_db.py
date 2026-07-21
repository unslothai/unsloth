# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id TEXT NOT NULL PRIMARY KEY,
            display_name TEXT NOT NULL,
            url TEXT NOT NULL,
            headers_json TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            use_oauth INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Backfill use_oauth for pre-existing DBs.
    cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(mcp_servers)").fetchall()
    }
    if "use_oauth" not in cols:
        conn.execute(
            "ALTER TABLE mcp_servers ADD COLUMN use_oauth INTEGER NOT NULL DEFAULT 0"
        )


def get_connection() -> sqlite3.Connection:
    global _schema_ready
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready = True
                except Exception:
                    conn.close()
                    raise
    return conn


def create_server(
    id: str,
    display_name: str,
    url: str,
    headers_json: Optional[str] = None,
    is_enabled: bool = True,
    use_oauth: bool = False,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO mcp_servers
                (id, display_name, url, headers_json,
                 is_enabled, use_oauth, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                display_name,
                url,
                headers_json,
                int(is_enabled),
                int(use_oauth),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_server(id: str, changes: dict) -> bool:
    """Apply column updates and bump ``updated_at``. Returns True on a hit."""
    if not changes:
        return False
    bool_cols = {"is_enabled", "use_oauth"}
    sets, params = [], []
    for col, value in changes.items():
        sets.append(f"{col} = ?")
        params.append(int(value) if col in bool_cols else value)
    sets.append("updated_at = ?")
    params.extend([datetime.now(timezone.utc).isoformat(), id])

    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE mcp_servers SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_server(id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM mcp_servers WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_server(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM mcp_servers WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_servers() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM mcp_servers ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
