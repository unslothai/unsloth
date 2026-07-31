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
import re
import sqlite3
import threading

logger = logging.getLogger(__name__)

from utils.paths import rag_db_path, ensure_dir

# Optional dep: import must never crash this module (imported unconditionally).
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
            project_id TEXT,
            filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            num_chunks INTEGER NOT NULL DEFAULT 0,
            stored_path TEXT,
            created_at TEXT NOT NULL,
            embedding_model TEXT
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
    # Lazy upgrade for databases created before project sources existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "project_id" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN project_id TEXT")
    # Lazy upgrade: which embedder produced a document's vectors (NULL = legacy,
    # assumed current). Dedupe re-ingests when it no longer matches.
    if "embedding_model" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT")


def get_connection() -> sqlite3.Connection:
    """Open rag.db (WAL + sqlite-vec loaded, schema created once). Raises if the extension is unavailable."""
    global _schema_ready
    if not RAG_AVAILABLE:
        raise RuntimeError(_RAG_UNAVAILABLE_MSG)

    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Wait for a lock instead of erroring immediately: a figure/scan-heavy ingest can
    # hold its connection across many seconds of vision calls, and a concurrent ingest
    # or autoinject read would otherwise hit "database is locked".
    conn.execute("PRAGMA busy_timeout = 5000")
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


def vec_table_dim(conn: sqlite3.Connection) -> int | None:
    """Embedding width baked into ``chunks_vec``, or None when absent."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if row is None or not row["sql"]:
        return None
    m = re.search(r"float\[(\d+)\]", row["sql"])
    return int(m.group(1)) if m else None


def ensure_vec(conn: sqlite3.Connection, dim: int) -> None:
    """Create the dense ``chunks_vec`` table once the embedding dim is known
    (vec0 bakes it into the column type). A width change (embedding model
    switched in Settings) drops the table: the old vectors live in a foreign
    space and would only block inserts, while lexical search keeps serving old
    chunks until they are re-uploaded."""
    existing = vec_table_dim(conn)
    if existing is not None and existing != int(dim):
        logger.warning(
            "chunks_vec dim changed %d -> %d (embedding model switched); dropping "
            "stale dense index. Re-upload documents to restore dense search.",
            existing,
            int(dim),
        )
        conn.execute("DROP TABLE chunks_vec")
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


def _delete_document_chunks(conn, document_id: str) -> None:
    """Delete a document's chunk rows (chunks/chunks_fts/chunks_vec), keeping the
    documents row. Used when reconciling a half-ingested doc to failed: retrieval
    filters by scope not status, so leftover chunks would stay citable."""
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id=?", (document_id,)
        ).fetchall()
    ]
    if not chunk_ids:
        return
    has_vec = vec_table_exists(conn)
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        if has_vec:
            conn.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))


def reconcile_orphaned_ingestion_jobs() -> int:
    """Fail ingestion jobs/documents left mid-flight by a crash so they stop
    showing as stuck "processing" and become re-ingestible. Run at startup.
    No-op without RAG. Returns the number of jobs reset.
    """
    if not RAG_AVAILABLE:
        return 0
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, document_id FROM ingestion_jobs "
            "WHERE status NOT IN ('completed', 'failed')"
        ).fetchall()
        for row in rows:
            doc = conn.execute(
                "SELECT status FROM documents WHERE id=?", (row["document_id"],)
            ).fetchone()
            if doc is not None and doc["status"] == "completed":
                # Worker finished indexing before the crash but didn't retire the
                # job row. Mark the job completed (not failed) and keep its chunks,
                # so the UI's getJob fallback after restart doesn't flag a
                # searchable document as a failed ingestion.
                conn.execute(
                    "UPDATE ingestion_jobs SET status='completed', stage='done', "
                    "progress=1.0, error=NULL WHERE id=?",
                    (row["id"],),
                )
                continue
            conn.execute(
                "UPDATE ingestion_jobs SET status='failed', stage='error', "
                "error='Server restarted during ingestion' WHERE id=?",
                (row["id"],),
            )
            conn.execute(
                "UPDATE documents SET status='failed' "
                "WHERE id=? AND status NOT IN ('completed', 'failed')",
                (row["document_id"],),
            )
            # A failed or still-in-flight doc must not leave citable chunks
            # (retrieval filters by scope, not status); also drops any chunks of a
            # doc already 'failed' before the crash.
            _delete_document_chunks(conn, row["document_id"])
        conn.commit()
        return len(rows)
    finally:
        conn.close()
