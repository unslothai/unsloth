# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lazy process-wide rag.db connection with sqlite-vec loaded."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from loggers import get_logger
from utils.paths.storage_roots import ensure_dir, rag_root

logger = get_logger(__name__)

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def rag_db_path() -> Path:
    return rag_root() / "rag.db"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        conn.enable_load_extension(True)
    except AttributeError as exc:
        raise RuntimeError(
            "This Python build cannot load SQLite extensions "
            "(connection.enable_load_extension is unavailable). RAG "
            "requires sqlite-vec, which loads as a SQLite extension. "
            "Re-install studio via install.sh so the venv uses uv's "
            "managed Python (python-build-standalone), compiled with "
            "--enable-loadable-sqlite-extensions."
        ) from exc
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rag_vectors (
            chunk_id     TEXT PRIMARY KEY,
            scope        TEXT NOT NULL,
            document_id  TEXT NOT NULL,
            chunk_index  INTEGER NOT NULL,
            kind         TEXT NOT NULL DEFAULT 'text',
            dim          INTEGER NOT NULL,
            vector       BLOB NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_rag_vectors_scope
            ON rag_vectors(scope);
        CREATE INDEX IF NOT EXISTS idx_rag_vectors_scope_doc
            ON rag_vectors(scope, document_id);
        """
    )
    conn.commit()


def get_rag_connection() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            ensure_dir(rag_root())
            conn = sqlite3.connect(
                str(rag_db_path()),
                check_same_thread = False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            _load_sqlite_vec(conn)
            _ensure_schema(conn)
            _conn = conn
            logger.info("RAG vector store opened", path = str(rag_db_path()))
        return _conn


def _reset_for_tests() -> None:
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
