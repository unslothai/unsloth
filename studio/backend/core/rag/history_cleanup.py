# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Destructive cleanup for RAG data hidden by the no-chat-history policy."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone

from storage import rag_db
from utils.paths import rag_uploads_root

from . import job_leases, store


def _table_exists(conn, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _managed_path(path: str | None) -> str | None:
    if not path:
        return None
    root = os.path.normcase(os.path.realpath(str(rag_uploads_root())))
    candidate = os.path.abspath(os.path.expanduser(path))
    try:
        resolved = os.path.normcase(os.path.realpath(candidate))
        if os.path.dirname(resolved) != root:
            return None
        if stat.S_ISLNK(os.lstat(candidate).st_mode):
            return None
    except FileNotFoundError:
        return resolved
    except OSError:
        return None
    return resolved


def _remove_managed_path(path: str | None) -> None:
    candidate = _managed_path(path)
    if candidate is None:
        return
    try:
        os.remove(candidate)
    except FileNotFoundError:
        pass


def clear_non_knowledge_base_data() -> int:
    """Delete hidden durable RAG data while preserving explicit global KBs.

    The text-bearing metadata cleanup works without sqlite-vec. When vec0 is
    available, its corresponding embeddings are removed in the same transaction.
    """
    vectors_available = True
    try:
        conn = rag_db.get_connection()
    except rag_db.RagExtensionUnavailable:
        vectors_available = False
        conn = rag_db.get_metadata_connection()

    try:
        if not _table_exists(conn, "documents"):
            return 0
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TEMP TABLE no_chat_history_documents(id TEXT NOT NULL PRIMARY KEY)")
        conn.execute("CREATE TEMP TABLE no_chat_history_folders(id TEXT NOT NULL PRIMARY KEY)")
        conn.execute("CREATE TEMP TABLE no_chat_history_scopes(scope TEXT NOT NULL PRIMARY KEY)")

        has_kbs = _table_exists(conn, "knowledge_bases")
        document_rows = conn.execute("SELECT id, scope, kb_id FROM documents").fetchall()
        hidden_document_ids = [
            (row["id"],)
            for row in document_rows
            if not has_kbs or store.document_knowledge_base_id(conn, dict(row)) is None
        ]
        conn.executemany("INSERT INTO no_chat_history_documents(id) VALUES(?)", hidden_document_ids)

        if _table_exists(conn, "linked_folders"):
            folder_without_kb_owner = (
                "NOT EXISTS (SELECT 1 FROM knowledge_bases kb WHERE "
                "(f.scope_type='knowledge_base' AND f.scope_id=kb.id) "
                "OR f.scope='kb_' || kb.id)"
                if has_kbs
                else "1=1"
            )
            conn.execute(
                "INSERT INTO no_chat_history_folders(id) "
                "SELECT f.id FROM linked_folders f WHERE "
                f"{folder_without_kb_owner}"
            )

        conn.execute(
            "INSERT OR IGNORE INTO no_chat_history_scopes(scope) "
            "SELECT DISTINCT d.scope FROM documents d "
            "JOIN no_chat_history_documents t ON t.id=d.id "
            "WHERE d.scope!=''"
        )
        if _table_exists(conn, "linked_folders"):
            conn.execute(
                "INSERT OR IGNORE INTO no_chat_history_scopes(scope) "
                "SELECT DISTINCT f.scope FROM linked_folders f "
                "JOIN no_chat_history_folders t ON t.id=f.id "
                "WHERE f.scope!=''"
            )

        documents = conn.execute(
            "SELECT d.id, d.stored_path FROM documents d "
            "JOIN no_chat_history_documents t ON t.id=d.id"
        ).fetchall()
        preserved_paths = {
            managed
            for row in conn.execute(
                "SELECT d.stored_path FROM documents d "
                "LEFT JOIN no_chat_history_documents t ON t.id=d.id "
                "WHERE t.id IS NULL AND d.stored_path IS NOT NULL"
            ).fetchall()
            if (managed := _managed_path(row["stored_path"])) is not None
        }
        for path in dict.fromkeys(row["stored_path"] for row in documents):
            managed = _managed_path(path)
            if managed is not None and managed not in preserved_paths:
                _remove_managed_path(managed)

        conn.execute(
            "CREATE TABLE IF NOT EXISTS linked_folder_retired_scopes("
            "scope TEXT NOT NULL PRIMARY KEY, retired_at TEXT NOT NULL, purged_at TEXT)"
        )
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO linked_folder_retired_scopes(scope, retired_at, purged_at) "
            "SELECT scope, ?, ? FROM no_chat_history_scopes",
            (now, now),
        )

        if _table_exists(conn, "ingestion_jobs"):
            if _table_exists(conn, "rag_job_leases"):
                conn.execute(
                    "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN ("
                    "SELECT j.id FROM ingestion_jobs j "
                    "LEFT JOIN no_chat_history_documents d ON d.id=j.document_id "
                    "LEFT JOIN no_chat_history_scopes s ON s.scope=j.scope "
                    "WHERE d.id IS NOT NULL OR s.scope IS NOT NULL)",
                    (job_leases.INGESTION,),
                )
            conn.execute(
                "DELETE FROM ingestion_jobs WHERE document_id IN "
                "(SELECT id FROM no_chat_history_documents) OR scope IN "
                "(SELECT scope FROM no_chat_history_scopes)"
            )

        if _table_exists(conn, "linked_folder_sync_jobs"):
            if _table_exists(conn, "rag_job_leases"):
                conn.execute(
                    "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN ("
                    "SELECT j.id FROM linked_folder_sync_jobs j "
                    "JOIN no_chat_history_folders f ON f.id=j.folder_id)",
                    (job_leases.FOLDER_SYNC,),
                )
            conn.execute(
                "DELETE FROM linked_folder_sync_jobs WHERE folder_id IN "
                "(SELECT id FROM no_chat_history_folders)"
            )
        if _table_exists(conn, "linked_folder_files"):
            conn.execute(
                "DELETE FROM linked_folder_files WHERE folder_id IN "
                "(SELECT id FROM no_chat_history_folders)"
            )
        if _table_exists(conn, "linked_folders"):
            conn.execute(
                "DELETE FROM linked_folders WHERE id IN (SELECT id FROM no_chat_history_folders)"
            )

        if _table_exists(conn, "chunks"):
            if _table_exists(conn, "chunks_fts"):
                conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id IN ("
                    "SELECT c.id FROM chunks c "
                    "JOIN no_chat_history_documents d ON d.id=c.document_id)"
                )
            if vectors_available and rag_db.vec_table_exists(conn):
                conn.execute(
                    "DELETE FROM chunks_vec WHERE chunk_id IN ("
                    "SELECT c.id FROM chunks c "
                    "JOIN no_chat_history_documents d ON d.id=c.document_id)"
                )
            conn.execute(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM no_chat_history_documents)"
            )
        conn.execute("DELETE FROM documents WHERE id IN (SELECT id FROM no_chat_history_documents)")
        conn.commit()
        return len(documents)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
