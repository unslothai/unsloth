# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for the RAG engine.

Same pattern as providers_db.py / studio_db.py (module functions, raw sqlite3,
WAL, per-call connections, lazy schema), but every connection also loads
sqlite-vec (vec0 needs it per-connection). If it cannot load, RAG_AVAILABLE is
False and get_connection() raises rather than failing import.

One rag.db holds the ``documents`` / ``chunks`` model, the FTS5 lexical index
(``chunks_fts``) and the sqlite-vec dense index (``chunks_vec``, created lazily
by ensure_vec once the embedding dim is known, since vec0 bakes the dim into the
column type).
"""

import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

from utils.paths import rag_db_path, ensure_dir

# Optional dep: import must never crash this module (Studio imports it unconditionally).
try:
    import sqlite_vec

    RAG_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - any import failure disables RAG
    sqlite_vec = None
    RAG_AVAILABLE = False
    logger.warning("RAG unavailable: sqlite-vec could not be imported (%s)", exc)

_RAG_UNAVAILABLE_MSG = "RAG unavailable: sqlite-vec extension could not be loaded"

_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the RAG tables if absent (once per process). ``chunks_vec`` is
    skipped: its column type needs the embedding dim, so ensure_vec() makes it
    lazily at first ingest."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            embedding_model TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT NOT NULL PRIMARY KEY,
            scope TEXT NOT NULL,
            kb_id TEXT,
            thread_id TEXT,
            filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            num_chunks INTEGER NOT NULL DEFAULT 0,
            stored_path TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(scope);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(scope, sha256);

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT NOT NULL PRIMARY KEY,
            document_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            page_number INTEGER,
            source_page_index INTEGER,
            token_count INTEGER,
            kind TEXT NOT NULL DEFAULT 'text',
            pdf_regions_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT NOT NULL PRIMARY KEY,
            document_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            stage TEXT,
            progress REAL NOT NULL DEFAULT 0.0,
            error TEXT,
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            chunk_id UNINDEXED,
            scope UNINDEXED,
            tokenize='porter unicode61'
        );
        """
    )


def get_connection() -> sqlite3.Connection:
    """Open rag.db (WAL + sqlite-vec loaded, schema created once). Raises if the extension is unavailable."""
    global _schema_ready
    if not RAG_AVAILABLE:
        raise RuntimeError(_RAG_UNAVAILABLE_MSG)

    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:  # noqa: BLE001
        conn.close()
        raise RuntimeError(_RAG_UNAVAILABLE_MSG) from exc

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


def ensure_vec(conn: sqlite3.Connection, dim: int) -> None:
    """Create the dense ``chunks_vec`` table once the embedding dim is known
    (vec0 bakes it into the column type). Idempotent; dim fixed per db."""
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"scope TEXT partition key, "
        f"chunk_id TEXT, "
        f"embedding float[{int(dim)}] distance_metric=cosine)"
    )


def vec_table_exists(conn: sqlite3.Connection) -> bool:
    """True if the dense ``chunks_vec`` table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    return row is not None
