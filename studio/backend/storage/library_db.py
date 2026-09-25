# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Library state in studio.db.

The Library lists files that already live elsewhere (chat uploads, generated images, sandbox
files), so it only stores what those sources cannot: folders, and a per-item overlay of display
name, favorite flag and folder. Files uploaded straight into the Library are the one source it
owns; their bytes sit under ``account_path("library")``.
"""

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from utils.paths import ensure_dir, studio_db_path

_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS library_folders (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS library_entries (
            item_id TEXT NOT NULL PRIMARY KEY,
            name TEXT,
            favorite INTEGER NOT NULL DEFAULT 0,
            folder_id TEXT,
            updated_at INTEGER NOT NULL
        )
        """
    )
    entry_columns = {row["name"] for row in conn.execute("PRAGMA table_info(library_entries)")}
    if "fingerprint" not in entry_columns:
        try:
            conn.execute("ALTER TABLE library_entries ADD COLUMN fingerprint TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS library_uploads (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    # One key for the check and the add, or a home reached through a link never finds its entry
    # and runs the schema again on every connection.
    schema_path = db_path.resolve()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if schema_path not in _schema_ready:
        with _schema_lock:
            if schema_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready.add(schema_path)
                except Exception:
                    conn.close()
                    raise
    return conn


def _lock(conn: sqlite3.Connection) -> None:
    """Take the write lock before the checks, so nothing they read can change before the write."""
    conn.execute("BEGIN IMMEDIATE")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _folder(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "parentId": row["parent_id"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_folders() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM library_folders ORDER BY updated_at DESC").fetchall()
        return [_folder(row) for row in rows]
    finally:
        conn.close()


def get_folder(folder_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM library_folders WHERE id = ?", (folder_id,)).fetchone()
        return _folder(row) if row else None
    finally:
        conn.close()


def _is_descendant(conn: sqlite3.Connection, folder_id: str, ancestor_id: str) -> bool:
    """Whether ``folder_id`` sits anywhere under ``ancestor_id`` (or is it)."""
    current: Optional[str] = folder_id
    seen: set[str] = set()
    while current and current not in seen:
        if current == ancestor_id:
            return True
        seen.add(current)
        row = conn.execute(
            "SELECT parent_id FROM library_folders WHERE id = ?", (current,)
        ).fetchone()
        current = row["parent_id"] if row else None
    return False


def create_folder(name: str, parent_id: Optional[str] = None) -> dict:
    now = _now_ms()
    folder_id = uuid.uuid4().hex
    conn = get_connection()
    try:
        _lock(conn)
        if (
            parent_id
            and not conn.execute(
                "SELECT 1 FROM library_folders WHERE id = ?", (parent_id,)
            ).fetchone()
        ):
            raise KeyError(parent_id)
        conn.execute(
            "INSERT INTO library_folders (id, name, parent_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (folder_id, name, parent_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": folder_id,
        "name": name,
        "parentId": parent_id,
        "createdAt": now,
        "updatedAt": now,
    }


def update_folder(
    folder_id: str,
    *,
    name: Optional[str] = None,
    parent_id: Optional[str] = None,
    move: bool = False,
) -> Optional[dict]:
    """Rename and/or move a folder. ``move`` distinguishes "move to the root" from "leave it"."""
    conn = get_connection()
    try:
        _lock(conn)
        if not conn.execute("SELECT 1 FROM library_folders WHERE id = ?", (folder_id,)).fetchone():
            return None
        if move and parent_id:
            if not conn.execute(
                "SELECT 1 FROM library_folders WHERE id = ?", (parent_id,)
            ).fetchone():
                raise KeyError(parent_id)
            if _is_descendant(conn, parent_id, folder_id):
                raise ValueError("A folder cannot be moved into itself")
        now = _now_ms()
        if name is not None:
            conn.execute(
                "UPDATE library_folders SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, folder_id),
            )
        if move:
            conn.execute(
                "UPDATE library_folders SET parent_id = ?, updated_at = ? WHERE id = ?",
                (parent_id, now, folder_id),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM library_folders WHERE id = ?", (folder_id,)).fetchone()
        return _folder(row)
    finally:
        conn.close()


def delete_folder(folder_id: str) -> bool:
    """Delete a folder; what it held moves up to its parent rather than disappearing."""
    conn = get_connection()
    try:
        _lock(conn)
        row = conn.execute(
            "SELECT parent_id FROM library_folders WHERE id = ?", (folder_id,)
        ).fetchone()
        if row is None:
            return False
        parent_id = row["parent_id"]
        conn.execute(
            "UPDATE library_folders SET parent_id = ? WHERE parent_id = ?", (parent_id, folder_id)
        )
        conn.execute(
            "UPDATE library_entries SET folder_id = ? WHERE folder_id = ?", (parent_id, folder_id)
        )
        conn.execute("DELETE FROM library_folders WHERE id = ?", (folder_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def list_entries() -> dict[str, dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM library_entries").fetchall()
        return {
            row["item_id"]: {
                "name": row["name"],
                "favorite": bool(row["favorite"]),
                "folderId": row["folder_id"],
                "updatedAt": row["updated_at"],
                "fingerprint": row["fingerprint"],
            }
            for row in rows
        }
    finally:
        conn.close()


def update_entry(
    item_id: str,
    *,
    name: Optional[str] = None,
    favorite: Optional[bool] = None,
    folder_id: Optional[str] = None,
    move: bool = False,
    fingerprint: Optional[str] = None,
) -> None:
    """Write an item's overlay. ``fingerprint`` is the file a path-derived id names right now: a
    row kept for another file at that path is dropped before the write, not carried over."""
    conn = get_connection()
    try:
        _lock(conn)
        if (
            move
            and folder_id
            and not conn.execute(
                "SELECT 1 FROM library_folders WHERE id = ?", (folder_id,)
            ).fetchone()
        ):
            raise KeyError(folder_id)
        now = _now_ms()
        if fingerprint is not None:
            conn.execute(
                "DELETE FROM library_entries WHERE item_id = ? AND fingerprint IS NOT NULL AND fingerprint != ?",
                (item_id, fingerprint),
            )
        conn.execute(
            "INSERT OR IGNORE INTO library_entries (item_id, updated_at) VALUES (?, ?)",
            (item_id, now),
        )
        if fingerprint is not None:
            conn.execute(
                "UPDATE library_entries SET fingerprint = ? WHERE item_id = ?",
                (fingerprint, item_id),
            )
        if name is not None:
            conn.execute("UPDATE library_entries SET name = ? WHERE item_id = ?", (name, item_id))
        if favorite is not None:
            conn.execute(
                "UPDATE library_entries SET favorite = ? WHERE item_id = ?",
                (int(favorite), item_id),
            )
        if move:
            conn.execute(
                "UPDATE library_entries SET folder_id = ? WHERE item_id = ?", (folder_id, item_id)
            )
        conn.execute("UPDATE library_entries SET updated_at = ? WHERE item_id = ?", (now, item_id))
        conn.commit()
    finally:
        conn.close()


def reconcile_entries(adopt: list[tuple[str, str]], stale: list[tuple[str, str]]) -> None:
    """Fingerprint legacy rows with the file found at their path (``adopt``), and drop rows kept
    for a file since replaced (``stale``). Each only while the row still reads as the listing saw
    it, so a write that landed in between is kept."""
    if not adopt and not stale:
        return
    conn = get_connection()
    try:
        conn.executemany(
            "UPDATE library_entries SET fingerprint = ? WHERE item_id = ? AND fingerprint IS NULL",
            [(fingerprint, item_id) for item_id, fingerprint in adopt],
        )
        conn.executemany(
            "DELETE FROM library_entries WHERE item_id = ? AND fingerprint = ?",
            stale,
        )
        conn.commit()
    finally:
        conn.close()


def delete_entry(item_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM library_entries WHERE item_id = ?", (item_id,))
        conn.commit()
    finally:
        conn.close()


def _upload(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "contentType": row["content_type"],
        "sizeBytes": row["size_bytes"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_uploads() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM library_uploads").fetchall()
        return [_upload(row) for row in rows]
    finally:
        conn.close()


def get_upload(upload_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM library_uploads WHERE id = ?", (upload_id,)).fetchone()
        return _upload(row) if row else None
    finally:
        conn.close()


def insert_upload(upload_id: str, name: str, content_type: str, size_bytes: int) -> dict:
    now = _now_ms()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO library_uploads (id, name, content_type, size_bytes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (upload_id, name, content_type, size_bytes, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": upload_id,
        "name": name,
        "contentType": content_type,
        "sizeBytes": size_bytes,
        "createdAt": now,
        "updatedAt": now,
    }


def touch_upload(upload_id: str, size_bytes: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE library_uploads SET size_bytes = ?, updated_at = ? WHERE id = ?",
            (size_bytes, _now_ms(), upload_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_upload(upload_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM library_uploads WHERE id = ?", (upload_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
