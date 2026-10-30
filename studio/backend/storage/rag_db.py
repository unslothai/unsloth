# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every connection loads sqlite-vec; if it fails get_connection raises RagExtensionUnavailable,
never at import; rag_available reports it; chunks_vec made lazily by ensure_vec"""

import logging
import re
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from storage import sqlite_gate
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
    """sqlite-vec is installed but its native library will not load (a missing vec0 binary in the venv is the
    common macOS case). Subclasses RuntimeError so existing ``except RuntimeError`` callers are unaffected; it
    exists so a caller can tell "RAG is switched off on this machine" from a real database error and degrade
    instead of returning 500 on every poll."""


_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()
# Held idle so per-operation closes do not delete and recreate WAL/SHM (#15552).
_wal_keepers: dict[Path, sqlite3.Connection] = {}
_wal_keeper_paths: dict[Path, Path] = {}
_wal_keeper_lock = threading.Lock()
_wal_keepers_enabled = True
# Warn once per process: the UI polls the KB list on a timer.
_unavailable_lock = threading.Lock()
_unavailable_warned = False
# Only success is cached, so a one-off failure cannot latch RAG off.
_extension_loaded = False


def _warn_unavailable_once(exc: BaseException | None = None) -> None:
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
    """RAG_AVAILABLE only proves the import, so the vec0 library is probed; real DB errors propagate."""
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
    """Create the RAG tables if absent (once per process). ``chunks_vec`` is skipped: its column type needs the
    embedding dim, so ensure_vec() makes it lazily at first ingest."""
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
            page_char_start INTEGER,
            page_char_end INTEGER,
            token_count INTEGER,
            whole_document_token_count INTEGER,
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
            failed_files TEXT,
            scan_report TEXT,
            last_change_at TEXT,
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

        -- No INSERT trigger: any row trigger on chunks costs ~20 us per insert. Inserts show up as a new
        -- max(rowid) per scope instead; a reused rowid only follows a delete, which bumps the epoch.
        CREATE TABLE IF NOT EXISTS rag_vec_epoch (
            scope TEXT NOT NULL PRIMARY KEY,
            n INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TRIGGER IF NOT EXISTS rag_vec_epoch_chunk_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO rag_vec_epoch(scope, n) VALUES (OLD.scope, 1)
                ON CONFLICT(scope) DO UPDATE SET n = n + 1;
        END;
        CREATE TRIGGER IF NOT EXISTS rag_vec_epoch_document_model
            AFTER UPDATE OF embedding_model ON documents BEGIN
            INSERT INTO rag_vec_epoch(scope, n) VALUES (NEW.scope, 1)
                ON CONFLICT(scope) DO UPDATE SET n = n + 1;
        END;
        CREATE TRIGGER IF NOT EXISTS rag_vec_epoch_document_delete AFTER DELETE ON documents BEGIN
            INSERT INTO rag_vec_epoch(scope, n) VALUES (OLD.scope, 1)
                ON CONFLICT(scope) DO UPDATE SET n = n + 1;
        END;

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
    # NULL = legacy (assumed current); dedupe re-ingests when the embedder changes.
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
    # NULL for older archives falls back to created_at ordering; not backfilled, since created_at cannot recover order within a compaction epoch.
    if "archive_ordinal" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN archive_ordinal INTEGER")
    chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "page_char_start" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN page_char_start INTEGER")
    if "page_char_end" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN page_char_end INTEGER")
    if "whole_document_token_count" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN whole_document_token_count INTEGER")
    # Partial: empty until a chat is compacted; keeps the MAX() ordinal lookup an index probe.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_archive_ordinal "
        "ON documents(scope, archive_ordinal) WHERE archive_ordinal IS NOT NULL"
    )
    # After the ALTER on older dbs. Partial, so the lexical fast-path gate stays an index probe.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_linked_folder "
        "ON documents(linked_folder_id) WHERE linked_folder_id IS NOT NULL"
    )
    ensure_linked_folder_columns(conn)
    conn.commit()


def ensure_linked_folder_columns(conn: sqlite3.Connection) -> None:
    """Adds linked-folder columns missing from older builds; also called on the metadata connection,
    which skips _ensure_schema so scope retirement works without the vector extension"""
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
    if folder_cols and "failed_files" not in folder_cols:
        conn.execute("ALTER TABLE linked_folders ADD COLUMN failed_files TEXT")
    if folder_cols and "scan_report" not in folder_cols:
        conn.execute("ALTER TABLE linked_folders ADD COLUMN scan_report TEXT")
    if folder_cols and "last_change_at" not in folder_cols:
        conn.execute("ALTER TABLE linked_folders ADD COLUMN last_change_at TEXT")


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    """Open rag.db (WAL + sqlite-vec loaded, schema created once). Raises if the extension is unavailable."""
    global _extension_loaded
    if not RAG_AVAILABLE:
        raise RagExtensionUnavailable(_RAG_UNAVAILABLE_MSG)

    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite_gate.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Wait for a lock instead of erroring: figure-heavy ingests hold connections across many seconds.
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:  # noqa: BLE001
        conn.close()
        _warn_unavailable_once(exc)
        raise RagExtensionUnavailable(_RAG_UNAVAILABLE_MSG) from exc
    # Set before the schema step: the library loaded, whatever the db does next. Monotonic, no lock.
    _extension_loaded = True

    if db_path not in _schema_ready:
        with _schema_lock:
            schema_path = db_path.resolve()
            if schema_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready.add(schema_path)
                except Exception:
                    conn.close()
                    raise
    try:
        _ensure_wal_keeper(db_path)
    except Exception:
        conn.close()
        raise
    return conn


def get_metadata_connection() -> sqlite3.Connection:
    """Metadata-only connection without sqlite-vec: tombstones stay writable; do not touch vec0 tables."""
    db_path = rag_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite_gate.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        _ensure_wal_keeper(db_path)
    except Exception:
        conn.close()
        raise
    return conn


def _ensure_wal_keeper(db_path: Path) -> None:
    if not _wal_keepers_enabled or db_path in _wal_keeper_paths:
        return
    with _wal_keeper_lock:
        if not _wal_keepers_enabled or _account_retired():
            return
        resolved = db_path.resolve()
        if resolved not in _wal_keepers:
            # No vec0 load: metadata-only callers must work without sqlite-vec.
            keeper = sqlite_gate.connect(db_path, check_same_thread = False)
            try:
                mode = keeper.execute("PRAGMA journal_mode").fetchone()[0]
                if not isinstance(mode, str) or mode.lower() != "wal":
                    keeper.close()
                    # Not cached: a metadata open can precede RAG switching the file to WAL.
                    return
                keeper.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchall()
            except Exception:
                keeper.close()
                raise
            _wal_keepers[resolved] = keeper
        _wal_keeper_paths[db_path] = resolved
        _wal_keeper_paths[resolved] = resolved


def _account_retired() -> bool:
    # Under _wal_keeper_lock: a late open must not recreate a keeper retirement closed (blocks the Windows rename).
    from utils.account_context import is_owner_context

    if is_owner_context():
        return False
    from core.training.account_jobs import account_is_retired

    return account_is_retired()


def start_wal_keepers() -> None:
    """Enable lazy keepers for a new backend lifespan without initializing RAG."""
    global _wal_keepers_enabled
    with _wal_keeper_lock:
        _wal_keepers_enabled = True


def _close_keeper(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except Exception as exc:
        logger.warning("Could not close the rag.db WAL keeper: %s", exc)


def close_wal_keeper_for(path: str | Path) -> None:
    """Release one account's RAG database before its directories are retired."""
    resolved = Path(path).resolve()
    with _wal_keeper_lock:
        keeper = _wal_keepers.pop(resolved, None)
        if keeper is not None:
            _close_keeper(keeper)
        for spelling, target in tuple(_wal_keeper_paths.items()):
            if target == resolved:
                del _wal_keeper_paths[spelling]


def close_wal_keeper() -> None:
    """Release all keepers; late workers cannot reopen them after shutdown."""
    global _wal_keepers_enabled
    with _wal_keeper_lock:
        _wal_keepers_enabled = False
        for keeper in _wal_keepers.values():
            _close_keeper(keeper)
        _wal_keepers.clear()
        _wal_keeper_paths.clear()


def vec_table_dim(conn: sqlite3.Connection) -> int | None:
    """Embedding width baked into ``chunks_vec``, or None when absent."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if row is None or not row["sql"]:
        return None
    m = re.search(r"float\[(\d+)\]", row["sql"])
    return int(m.group(1)) if m else None


# Prefixed to the identity of a document whose vectors were dropped, so it matches no embedder.
VECTORS_DROPPED = "vectors-dropped:"


def ensure_vec(
    conn: sqlite3.Connection,
    dim: int,
    writing: str | None = None,
) -> None:
    """Create the dense ``chunks_vec`` table once the embedding dim is known (vec0 bakes it into the column
    type). A width change (embedding model switched in Settings) drops the table: the old vectors live in a
    foreign space and would only block inserts. Lexical search keeps serving the old chunks, and every
    document but ``writing``, completed or not (ingestion commits its vectors before completing), gets
    ``VECTORS_DROPPED`` prefixed to its identity until it writes vectors again or is re-embedded."""
    existing = vec_table_dim(conn)
    if existing is not None and existing != int(dim):
        logger.warning(
            "chunks_vec dim changed %d -> %d (embedding model switched); dropping "
            "stale dense index. Re-embed documents to restore dense search.",
            existing,
            int(dim),
        )
        conn.execute("DROP TABLE chunks_vec")
        # The drop moves no chunks row; the global epoch invalidates every cached scope.
        conn.execute(
            "INSERT INTO rag_vec_epoch(scope, n) VALUES ('', 1) "
            "ON CONFLICT(scope) DO UPDATE SET n = n + 1"
        )
        conn.execute(
            "UPDATE documents SET embedding_model=? || COALESCE(embedding_model, '') "
            "WHERE id IS NOT ? AND COALESCE(embedding_model, '') NOT LIKE ?",
            (VECTORS_DROPPED, writing, VECTORS_DROPPED + "%"),
        )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"scope TEXT partition key, "
        f"chunk_id TEXT, "
        f"embedding float[{int(dim)}] distance_metric=cosine)"
    )


def restore_dropped_identity(conn: sqlite3.Connection, document_id: str) -> None:
    """Undo ``ensure_vec``'s prefix once ``document_id`` has written vectors into the current table."""
    conn.execute(
        "UPDATE documents SET embedding_model=NULLIF(substr(embedding_model, ?), '') "
        "WHERE id=? AND embedding_model LIKE ?",
        (len(VECTORS_DROPPED) + 1, document_id, VECTORS_DROPPED + "%"),
    )


def vec_table_exists(conn: sqlite3.Connection) -> bool:
    """True if the dense ``chunks_vec`` table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    return row is not None


def _delete_document_chunks(conn, document_id: str) -> None:
    """Deletes chunk rows, keeps the documents row; retrieval filters by scope, so leftovers stay
    citable."""
    if (
        conn.execute("SELECT 1 FROM chunks WHERE document_id=? LIMIT 1", (document_id,)).fetchone()
        is None
    ):
        return
    # chunk_id is unindexed in fts5 and vec0: one IN scan per table, not one scan per chunk.
    owned = "SELECT id FROM chunks WHERE document_id=?"
    conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({owned})", (document_id,))
    if vec_table_exists(conn):
        conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({owned})", (document_id,))
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))


def reconcile_orphaned_ingestion_jobs() -> int:
    """Fails crash-orphaned ingestion jobs so they can re-run; another live backend's work is left
    alone until its lease expires; no-op without RAG"""
    # rag_available(): the package may be installed without the vec0 binary.
    if not rag_available():
        return 0
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = datetime.now(timezone.utc).isoformat()
        # 'cancelled' is terminal: the document was deleted, not an indexing failure.
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
                # Indexed before the crash but job row not retired: mark completed and keep chunks, so a searchable document is not shown as a failed ingestion.
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
                # Retrieval filters by scope, not status, so drop chunks of failed or in-flight docs.
                _delete_document_chunks(conn, row["document_id"])
            conn.execute(
                "DELETE FROM rag_job_leases WHERE kind='ingestion' AND job_id=?",
                (row["id"],),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()
