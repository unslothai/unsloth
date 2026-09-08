# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for the RAG engine.

Same pattern as providers_db.py / studio_db.py (module functions, raw sqlite3,
WAL, per-call connections, lazy schema), but every connection also loads
sqlite-vec (vec0 needs it per-connection). If it cannot load, get_connection()
raises RagExtensionUnavailable rather than failing import, and rag_available()
reports the machine as one where RAG cannot run.

One rag.db holds the ``documents`` / ``chunks`` model, the FTS5 lexical index
(``chunks_fts``) and the sqlite-vec dense index (``chunks_vec``, created lazily
by ensure_vec once the embedding dim is known, since vec0 bakes the dim into the
column type).
"""

import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone

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


class RagExtensionUnavailable(RuntimeError):
    """sqlite-vec is installed but its native library will not load (a missing
    vec0 binary in the venv is the common macOS case). Subclasses RuntimeError so
    existing ``except RuntimeError`` callers are unaffected; it exists so a caller
    can tell "RAG is switched off on this machine" from a real database error and
    degrade instead of returning 500 on every poll."""


_schema_lock = threading.Lock()
_schema_ready = False
# The dylib is either there or it is not, and the UI polls the KB list on a timer, so one warning
# per process says everything the repeats would.
_unavailable_lock = threading.Lock()
_unavailable_warned = False
# Only the positive verdict is kept: a failure stays retried per connection, so a one-off cannot
# latch RAG off for the rest of the session.
_extension_loaded = False


def _warn_unavailable_once(exc: BaseException | None = None) -> None:
    """Log the sqlite-vec unavailability at most once per process."""
    global _unavailable_warned
    with _unavailable_lock:
        if _unavailable_warned:
            return
        _unavailable_warned = True
    logger.warning(
        "%s; RAG features are disabled for this session%s",
        _RAG_UNAVAILABLE_MSG,
        f" ({exc})" if exc is not None else "",
    )


def rag_available() -> bool:
    """Whether RAG can actually run in this process.

    RAG_AVAILABLE only records that ``import sqlite_vec`` worked. The vec0 native
    library it loads is a separate file, and a venv can have the package without it
    (the common macOS case), which nothing finds out until a connection tries. So try,
    unless one already got through: a machine where RAG works answers from the flag
    instead of opening a second connection per request, and a machine where it does not
    pays the same failed connect it paid before, quietly.

    A genuine database error (locked, corrupt, bad schema) is not an answer to this
    question, so it propagates instead of being reported as "RAG is off here".
    """
    if not RAG_AVAILABLE:
        return False
    if _extension_loaded:
        return True
    try:
        conn = get_connection()
    except RagExtensionUnavailable:
        return False
    conn.close()
    return True


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

        CREATE TABLE IF NOT EXISTS rag_job_leases (
            kind TEXT NOT NULL,
            job_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY(kind, job_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rag_job_leases_expiry
            ON rag_job_leases(expires_at);

        CREATE TABLE IF NOT EXISTS linked_folders (
            id TEXT NOT NULL PRIMARY KEY,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            root_device INTEGER,
            root_inode INTEGER,
            delete_remove_index INTEGER,
            auto_sync INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT,
            last_scan_at TEXT,
            withheld_paths TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(scope, path)
        );
        CREATE INDEX IF NOT EXISTS idx_linked_folders_scope ON linked_folders(scope);

        CREATE TABLE IF NOT EXISTS linked_folder_retired_scopes (
            scope TEXT NOT NULL PRIMARY KEY,
            retired_at TEXT NOT NULL,
            purged_at TEXT
        );

        CREATE TABLE IF NOT EXISTS linked_folder_files (
            folder_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            device INTEGER,
            inode INTEGER,
            document_id TEXT NOT NULL,
            content_hash TEXT,
            synced_at TEXT NOT NULL,
            PRIMARY KEY(folder_id, relative_path)
        );
        CREATE INDEX IF NOT EXISTS idx_linked_folder_files_document
            ON linked_folder_files(document_id);

        CREATE TABLE IF NOT EXISTS linked_folder_sync_jobs (
            id TEXT NOT NULL PRIMARY KEY,
            folder_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'sync',
            status TEXT NOT NULL DEFAULT 'pending',
            stage TEXT,
            progress REAL NOT NULL DEFAULT 0.0,
            discovered INTEGER NOT NULL DEFAULT 0,
            added INTEGER NOT NULL DEFAULT 0,
            changed INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0,
            renamed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            successor_kind TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_linked_folder_jobs_queue
            ON linked_folder_sync_jobs(status, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_linked_folder_jobs_active
            ON linked_folder_sync_jobs(folder_id)
            WHERE status IN ('pending','running');

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
    # Which embedder produced a document's vectors (NULL = legacy, assumed current); dedupe re-ingests
    # when it no longer matches.
    if "embedding_model" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT")
    # Folder ownership makes crash cleanup unambiguous without changing retrieval.
    if "linked_folder_id" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN linked_folder_id TEXT")
    if "linked_relative_path" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN linked_relative_path TEXT")
    # NULL for everything else and for older archives, which fall back to counting role labels in the rendered text.
    if "archive_messages" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN archive_messages INTEGER")
    # NULL for older archives, which fall back to created_at ordering. Not backfilled: created_at cannot
    # recover the order within a compaction epoch.
    if "archive_ordinal" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN archive_ordinal INTEGER")
    # Partial, so it is empty until a chat is compacted and the MAX() that allocates the next ordinal is
    # an index probe rather than a scan.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_archive_ordinal "
        "ON documents(scope, archive_ordinal) WHERE archive_ordinal IS NOT NULL"
    )
    # After the ALTER that adds the column on an older database. Partial, so it holds only
    # folder-owned rows and is empty with nothing linked, which keeps the lexical fast-path gate an
    # index probe rather than a scan.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_linked_folder "
        "ON documents(linked_folder_id) WHERE linked_folder_id IS NOT NULL"
    )
    ensure_linked_folder_columns(conn)
    conn.commit()


def ensure_linked_folder_columns(conn: sqlite3.Connection) -> None:
    """Add the linked-folder columns a database created by an earlier build is missing.

    Also called for the metadata connection, which skips _ensure_schema so that scope
    retirement keeps working when the vector extension cannot load.
    """
    job_cols = {r[1] for r in conn.execute("PRAGMA table_info(linked_folder_sync_jobs)").fetchall()}
    # the queued follow-up request; it replaced a flag that only recorded rebuilds
    if job_cols and "successor_kind" not in job_cols:
        conn.execute("ALTER TABLE linked_folder_sync_jobs ADD COLUMN successor_kind TEXT")
        if "rebuild_requested" in job_cols:
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET successor_kind='rebuild' "
                "WHERE rebuild_requested=1"
            )
    # vanished paths already granted their one grace pass before removal
    folder_cols = {r[1] for r in conn.execute("PRAGMA table_info(linked_folders)").fetchall()}
    if folder_cols and "withheld_paths" not in folder_cols:
        conn.execute("ALTER TABLE linked_folders ADD COLUMN withheld_paths TEXT")


def get_connection() -> sqlite3.Connection:
    """Open rag.db (WAL + sqlite-vec loaded, schema created once). Raises if the extension is unavailable."""
    global _schema_ready, _extension_loaded
    if not RAG_AVAILABLE:
        raise RagExtensionUnavailable(_RAG_UNAVAILABLE_MSG)

    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Wait for a lock instead of erroring immediately: a figure-heavy ingest can hold its connection
    # across many seconds of vision calls, and a concurrent read would hit "database is locked".
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:  # noqa: BLE001
        conn.close()
        _warn_unavailable_once(exc)
        raise RagExtensionUnavailable(_RAG_UNAVAILABLE_MSG) from exc
    # Set before the schema step: the library loaded, so RAG runs on this machine whatever a broken
    # database does next. A monotonic flip, so no lock.
    _extension_loaded = True

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


def get_metadata_connection() -> sqlite3.Connection:
    """Open rag.db without loading sqlite-vec.

    This connection is only for ordinary SQLite metadata tables. It lets lifecycle
    tombstones remain writable when the optional native vector extension is temporarily
    unavailable. Callers must not query or mutate the vec0 virtual table.
    """
    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
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
    showing as stuck "processing" and become re-ingestible. Work owned by another
    live backend is left alone until its lease expires. No-op without RAG. Returns
    the number of jobs reset.
    """
    # rag_available(), not RAG_AVAILABLE: a venv with the package but no vec0 binary would raise out of
    # startup and be logged as a reconcile failure when there is nothing to reconcile.
    if not rag_available():
        return 0
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = datetime.now(timezone.utc).isoformat()
        # "cancelled" is terminal too: the job stopped because its document was deleted, so rewriting it to
        # failed would report a deliberate cancellation as an indexing failure.
        rows = conn.execute(
            "SELECT j.id, j.document_id FROM ingestion_jobs j "
            "WHERE j.status NOT IN ('completed', 'failed', 'cancelled') AND NOT EXISTS ("
            "SELECT 1 FROM rag_job_leases l WHERE l.kind='ingestion' "
            "AND l.job_id=j.id AND l.expires_at>?)",
            (now,),
        ).fetchall()
        for row in rows:
            doc = conn.execute(
                "SELECT status FROM documents WHERE id=?", (row["document_id"],)
            ).fetchone()
            if doc is not None and doc["status"] == "completed":
                # The worker finished indexing before the crash but did not retire the job row: mark it completed
                # and keep its chunks, so the UI does not flag a searchable document as a failed ingestion.
                conn.execute(
                    "UPDATE ingestion_jobs SET status='completed', stage='done', "
                    "progress=1.0, error=NULL WHERE id=?",
                    (row["id"],),
                )
            else:
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
                # A failed or still-in-flight doc must not leave citable chunks, since retrieval filters by scope
                # and not status.
                _delete_document_chunks(conn, row["document_id"])
            conn.execute(
                "DELETE FROM rag_job_leases WHERE kind='ingestion' AND job_id=?",
                (row["id"],),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()
