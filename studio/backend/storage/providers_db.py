# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SQLite storage for external LLM provider configurations.

Follows the same pattern as studio_db.py — module-level functions,
raw sqlite3, WAL mode, per-function connections.

NOTE: API keys are NOT stored here. They live only in the browser
(localStorage) and are sent encrypted per-request.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the llm_providers table if it doesn't exist. Called once per process."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_providers (
            id TEXT NOT NULL PRIMARY KEY,
            provider_type TEXT NOT NULL,
            display_name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def get_connection() -> sqlite3.Connection:
    """Open studio.db with WAL mode, create table once per process."""
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


def create_provider(
    id: str,
    provider_type: str,
    display_name: str,
    base_url: str,
) -> None:
    """Insert a new provider configuration."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO llm_providers (id, provider_type, display_name, base_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id, provider_type, display_name, base_url, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def update_provider(
    id: str,
    display_name: Optional[str] = None,
    base_url: Optional[str] = None,
    is_enabled: Optional[bool] = None,
) -> bool:
    """Update fields on an existing provider. Returns True if a row was updated."""
    updates = []
    params = []
    if display_name is not None:
        updates.append("display_name = ?")
        params.append(display_name)
    if base_url is not None:
        updates.append("base_url = ?")
        params.append(base_url)
    if is_enabled is not None:
        updates.append("is_enabled = ?")
        params.append(1 if is_enabled else 0)
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(id)

    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE llm_providers SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_provider(id: str) -> bool:
    """Delete a provider by ID. Returns True if a row was deleted."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM llm_providers WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_provider(id: str) -> Optional[dict]:
    """Fetch a single provider by ID."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM llm_providers WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_providers() -> list[dict]:
    """List all provider configurations, ordered by creation time."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM llm_providers ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
