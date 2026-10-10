# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Other Unsloth Studio instances linked to this one by URL and API key.

The key is encrypted in ``credential_secrets``; this table holds only the name and URL.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage import credential_secrets
from utils.paths import ensure_dir, studio_db_path

LINKED_INSTANCE_API_KEY_KIND = "linked_instance_api_key"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()


class DuplicateName(ValueError):
    pass


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS linked_instances (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            base_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Tools stay off for a linked instance until the owner turns them on for that instance.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(linked_instances)").fetchall()}
    if "allow_tools" not in cols:
        conn.execute(
            "ALTER TABLE linked_instances ADD COLUMN allow_tools INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), timeout = 5.0)
    conn.row_factory = sqlite3.Row
    if db_path not in _schema_ready:
        with _schema_lock:
            if db_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready.add(db_path)
                except Exception:
                    conn.close()
                    raise
    return conn


def _row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "base_url": row["base_url"],
        "allow_tools": bool(row["allow_tools"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def validate_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not NAME_PATTERN.match(name):
        raise ValueError("Name must be 1-32 characters: lowercase letters, digits, '-' or '_'.")
    return name


def list_instances() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM linked_instances ORDER BY name").fetchall()
        return [_row(r) for r in rows]
    finally:
        conn.close()


def get_instance(instance_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM linked_instances WHERE id = ?", (instance_id,)).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def get_instance_by_name(name: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM linked_instances WHERE name = ?", (name.lower(),)
        ).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def create_instance(name: str, base_url: str, api_key: str) -> dict:
    name = validate_name(name)
    now = datetime.now(timezone.utc).isoformat()
    instance_id = uuid.uuid4().hex
    credential_secrets.ensure_schema()
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO linked_instances (id, name, base_url, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (instance_id, name, base_url, now, now),
            )
            credential_secrets.upsert_secret(
                LINKED_INSTANCE_API_KEY_KIND, instance_id, api_key, connection = conn
            )
    except sqlite3.IntegrityError as exc:
        raise DuplicateName(f"A linked instance named '{name}' already exists.") from exc
    finally:
        conn.close()
    return get_instance(instance_id)


def update_instance(
    instance_id: str,
    *,
    name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    allow_tools: Optional[bool] = None,
) -> Optional[dict]:
    if get_instance(instance_id) is None:
        return None
    fields: dict[str, object] = {}
    if name is not None:
        fields["name"] = validate_name(name)
    if base_url is not None:
        fields["base_url"] = base_url
    if allow_tools is not None:
        fields["allow_tools"] = 1 if allow_tools else 0
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    credential_secrets.ensure_schema()
    conn = get_connection()
    try:
        with conn:
            assignments = ", ".join(f"{column} = ?" for column in fields)
            conn.execute(
                f"UPDATE linked_instances SET {assignments} WHERE id = ?",
                (*fields.values(), instance_id),
            )
            if api_key:
                credential_secrets.upsert_secret(
                    LINKED_INSTANCE_API_KEY_KIND, instance_id, api_key, connection = conn
                )
    except sqlite3.IntegrityError as exc:
        raise DuplicateName(
            f"A linked instance named '{fields.get('name')}' already exists."
        ) from exc
    finally:
        conn.close()
    return get_instance(instance_id)


def delete_instance(instance_id: str) -> bool:
    conn = get_connection()
    try:
        with conn:
            deleted = conn.execute(
                "DELETE FROM linked_instances WHERE id = ?", (instance_id,)
            ).rowcount
    finally:
        conn.close()
    credential_secrets.delete_secret(LINKED_INSTANCE_API_KEY_KIND, instance_id)
    return bool(deleted)


def get_api_key(instance_id: str) -> Optional[str]:
    return credential_secrets.get_secret(LINKED_INSTANCE_API_KEY_KIND, instance_id)
