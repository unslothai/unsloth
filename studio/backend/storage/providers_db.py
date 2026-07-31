# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for external LLM provider configurations.

Same pattern as studio_db.py (module-level functions, raw sqlite3, WAL,
per-function connections). API keys are NOT stored here: they live only in
the browser (localStorage) and are sent encrypted per-request.

Enabled model selections and discovered catalog IDs are stored server-side so
remote Studio clients see the same connection state (#7281).
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready = False


def _encode_models_json(models: Optional[list[str]]) -> str:
    if not models:
        return "[]"
    return json.dumps([str(model).strip() for model in models if str(model).strip()])


def _decode_models_json(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(model).strip() for model in parsed if str(model).strip()]


def _row_models(row: sqlite3.Row) -> tuple[list[str], list[str]]:
    return (
        _decode_models_json(row["models_json"] if "models_json" in row.keys() else None),
        _decode_models_json(
            row["available_models_json"] if "available_models_json" in row.keys() else None
        ),
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the llm_providers table if absent. Called once per process."""
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
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_providers)").fetchall()}
    if "models_json" not in existing_cols:
        conn.execute("ALTER TABLE llm_providers ADD COLUMN models_json TEXT NOT NULL DEFAULT '[]'")
    if "available_models_json" not in existing_cols:
        conn.execute(
            "ALTER TABLE llm_providers ADD COLUMN available_models_json TEXT NOT NULL DEFAULT '[]'"
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
    models: Optional[list[str]] = None,
    available_models: Optional[list[str]] = None,
) -> None:
    """Insert a new provider configuration."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO llm_providers (
                id, provider_type, display_name, base_url,
                models_json, available_models_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                provider_type,
                display_name,
                base_url,
                _encode_models_json(models),
                _encode_models_json(available_models),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_provider(
    id: str,
    display_name: Optional[str] = None,
    base_url: Optional[str] = None,
    is_enabled: Optional[bool] = None,
    models: Optional[list[str]] = None,
    available_models: Optional[list[str]] = None,
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
    if models is not None:
        updates.append("models_json = ?")
        params.append(_encode_models_json(models))
    if available_models is not None:
        updates.append("available_models_json = ?")
        params.append(_encode_models_json(available_models))
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
        if not row:
            return None
        data = dict(row)
        models, available_models = _row_models(row)
        data["models"] = models
        data["available_models"] = available_models
        return data
    finally:
        conn.close()


def list_providers() -> list[dict]:
    """List all provider configurations, ordered by creation time."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM llm_providers ORDER BY created_at").fetchall()
        providers: list[dict] = []
        for row in rows:
            data = dict(row)
            models, available_models = _row_models(row)
            data["models"] = models
            data["available_models"] = available_models
            providers.append(data)
        return providers
    finally:
        conn.close()
