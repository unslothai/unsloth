# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unified SQLite store: relational chunks + FTS5 lexical + sqlite-vec dense.

Module-level functions in the Studio idiom: each takes a ``conn`` the caller
opens (``rag_db.get_connection()``) and closes. Inserts are incremental:
``add_chunks`` appends one document's rows without rebuilding the scope, so the
Nth upload costs O(its own chunks). Scope ("kb_<id>" / "thread_<id>") is a column
on every table and the vec0 partition key.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import uuid
from datetime import datetime, timezone

from storage import rag_db


# --------------------------------------------------------------------------
# Scope + serialization helpers
# --------------------------------------------------------------------------
def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def _f32(vector) -> bytes:
    """Pack a vector into little-endian float32 bytes for vec0."""
    return struct.pack(f"{len(vector)}f", *(float(x) for x in vector))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _match_query(query: str) -> str:
    """User text -> safe FTS5 OR-of-quoted-terms query. Quoting defuses FTS5
    operators in the input; "" (no tokens) means no lexical results."""
    toks = _TOKEN.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in toks)


# --------------------------------------------------------------------------
# Knowledge bases
# --------------------------------------------------------------------------
def create_kb(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str | None = None,
    embedding_model: str | None = None,
    kb_id: str | None = None,
) -> str:
    kb_id = kb_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO knowledge_bases(id, name, description, embedding_model, created_at) "
        "VALUES(?,?,?,?,?)",
        (kb_id, name, description, embedding_model, _now()),
    )
    conn.commit()
    return kb_id


def list_kbs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def get_kb(conn: sqlite3.Connection, kb_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    return dict(row) if row else None


def delete_kb(conn: sqlite3.Connection, kb_id: str) -> None:
    """Delete a knowledge base and every document (+ chunks) under it."""
    scope = kb_scope(kb_id)
    doc_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    ]
    for doc_id in doc_ids:
        delete_document(conn, doc_id)
    conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
    conn.commit()


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def create_document(
    conn: sqlite3.Connection,
    *,
    scope: str,
    filename: str,
    sha256: str,
    kb_id: str | None = None,
    thread_id: str | None = None,
    status: str = "pending",
    stored_path: str | None = None,
    document_id: str | None = None,
) -> str:
    document_id = document_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO documents(id, scope, kb_id, thread_id, filename, sha256, status, "
        "stored_path, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            document_id,
            scope,
            kb_id,
            thread_id,
            filename,
            sha256,
            status,
            stored_path,
            _now(),
        ),
    )
    conn.commit()
    return document_id


def set_document_status(
    conn: sqlite3.Connection,
    document_id: str,
    status: str,
    *,
    num_chunks: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE documents SET status=?, num_chunks=COALESCE(?, num_chunks), error=? WHERE id=?",
        (status, num_chunks, error, document_id),
    )
    conn.commit()


def list_documents(conn: sqlite3.Connection, scope: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, scope, kb_id, thread_id, filename, sha256, status, error, num_chunks, created_at "
        "FROM documents WHERE scope=? ORDER BY created_at DESC",
        (scope,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else None


def document_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM documents WHERE scope=? AND sha256=?", (scope, sha256)
    ).fetchone()
    return row["id"] if row else None


# --------------------------------------------------------------------------
# Chunks (incremental writes)
# --------------------------------------------------------------------------
def add_chunks(
    conn: sqlite3.Connection,
    scope: str,
    document_id: str,
    chunks,
    vectors,
    regions = None,
) -> None:
    """Incrementally index one document's chunks into chunks + FTS5 + vec0.
    ``vectors`` parallels ``chunks``; optional ``regions`` parallels them too,
    storing per-chunk PDF highlight rects as JSON for citation preview."""
    if len(vectors):
        rag_db.ensure_vec(conn, len(vectors[0]))
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{document_id}:{chunk.chunk_index}"
        chunk_regions = regions[i] if regions and i < len(regions) else None
        regions_json = json.dumps(chunk_regions) if chunk_regions else None
        conn.execute(
            "INSERT OR REPLACE INTO chunks("
            "id, document_id, scope, chunk_index, text, page_number, "
            "source_page_index, token_count, kind, pdf_regions_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                chunk_id,
                document_id,
                scope,
                chunk.chunk_index,
                chunk.text,
                chunk.page_number,
                chunk.source_page_index,
                chunk.token_count,
                getattr(chunk, "kind", "text"),
                regions_json,
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts(text, chunk_id, scope) VALUES(?,?,?)",
            (chunk.text, chunk_id, scope),
        )
        conn.execute(
            "INSERT INTO chunks_vec(scope, chunk_id, embedding) VALUES(?,?,?)",
            (scope, chunk_id, _f32(vector)),
        )
    conn.commit()


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    """Remove a document and all its chunks (+ fts + vec rows)."""
    ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id=?", (document_id,)
        ).fetchall()
    ]
    has_vec = rag_db.vec_table_exists(conn)
    for chunk_id in ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        if has_vec:
            conn.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()


# --------------------------------------------------------------------------
# Retrieval primitives
# --------------------------------------------------------------------------
def search_lexical(conn: sqlite3.Connection, scope: str, query: str, k: int):
    """BM25 lexical search. Returns [(chunk_id, score)], higher = better."""
    mq = _match_query(query)
    if not mq:
        return []
    rows = conn.execute(
        "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
        "WHERE chunks_fts MATCH ? AND scope=? ORDER BY s LIMIT ?",
        (mq, scope, k),
    ).fetchall()
    # bm25() is negative (more negative = better); flip to higher-is-better.
    return [(r["chunk_id"], -r["s"]) for r in rows]


def search_dense(conn: sqlite3.Connection, scope: str, vector, k: int):
    """Cosine KNN over vec0. Returns [(chunk_id, 1 - distance)]."""
    if not rag_db.vec_table_exists(conn):
        return []
    rows = conn.execute(
        "SELECT chunk_id, distance FROM chunks_vec "
        "WHERE scope=? AND embedding MATCH ? ORDER BY distance LIMIT ?",
        (scope, _f32(vector), k),
    ).fetchall()
    return [(r["chunk_id"], 1.0 - r["distance"]) for r in rows]


def chunks_by_id(conn: sqlite3.Connection, ids) -> dict:
    """Hydrate chunk rows (joined with document filename), keyed by id."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT c.id, c.text, c.document_id, c.chunk_index, c.page_number, "
        f"c.source_page_index, d.filename "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.id IN ({placeholders})",
        list(ids),
    ).fetchall()
    return {r["id"]: r for r in rows}
