# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unified SQLite store: relational chunks + FTS5 lexical + sqlite-vec dense.

Module-level functions each take a ``conn`` the caller opens and closes. Inserts
are incremental: ``add_chunks`` appends one document's rows without rebuilding the
scope. Scope ("kb_<id>" / "thread_<id>") is a column on every table and the vec0
partition key.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import uuid
from datetime import datetime, timezone

from storage import rag_db

from . import config


def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def project_scope(project_id: str) -> str:
    return f"project_{project_id}"


CONVERSATION_ARCHIVE_PREFIX = "convarchive_"


def conversation_archive_scope(thread_id: str) -> str:
    """Scope holding the turns a thread's rolling context window has evicted.

    Deliberately NOT ``thread_scope``. That scope is the user's attached documents, and
    with ``config.THREAD_WHOLE_DOC`` on, ``tool.whole_document_context`` renders every
    chunk of it into every request -- archiving turns there would re-inject the entire
    history each turn and defeat the compaction that produced it. Keeping the archive in
    its own scope also keeps it out of the attachments UI and the citation panel.
    """
    return f"{CONVERSATION_ARCHIVE_PREFIX}{thread_id}"


def _scopes(scope) -> list[str]:
    """Search helpers accept one scope or several (e.g. project + thread)."""
    return [scope] if isinstance(scope, str) else list(scope)


def _f32(vector) -> bytes:
    """Pack a vector into float32 bytes for vec0."""
    return struct.pack(f"{len(vector)}f", *(float(x) for x in vector))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _match_query(query: str) -> str:
    """User text -> safe FTS5 OR-of-quoted-terms query; quoting defuses FTS5
    operators. "" (no tokens) means no lexical results."""
    toks = _TOKEN.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in toks)


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


def delete_kb(
    conn: sqlite3.Connection,
    kb_id: str,
    *,
    commit: bool = True,
    delete_documents: bool = True,
) -> None:
    """Delete a knowledge base, optionally retaining documents for durable cleanup."""
    try:
        if commit:
            conn.execute("BEGIN IMMEDIATE")
        scope = kb_scope(kb_id)
        if delete_documents:
            doc_ids = [
                r["id"]
                for r in conn.execute("SELECT id FROM documents WHERE scope=?", (scope,)).fetchall()
            ]
            for doc_id in doc_ids:
                delete_document(conn, doc_id, commit = False)
        conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise


def create_document(
    conn: sqlite3.Connection,
    *,
    scope: str,
    filename: str,
    sha256: str,
    kb_id: str | None = None,
    thread_id: str | None = None,
    project_id: str | None = None,
    status: str = "pending",
    stored_path: str | None = None,
    document_id: str | None = None,
    embedding_model: str | None = None,
    linked_folder_id: str | None = None,
    linked_relative_path: str | None = None,
    archive_messages: int | None = None,
    commit: bool = True,
) -> str:
    document_id = document_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO documents(id, scope, kb_id, thread_id, project_id, filename, sha256, "
        "status, stored_path, created_at, embedding_model, linked_folder_id, "
        "linked_relative_path, archive_messages) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            document_id,
            scope,
            kb_id,
            thread_id,
            project_id,
            filename,
            sha256,
            status,
            stored_path,
            _now(),
            embedding_model,
            linked_folder_id,
            linked_relative_path,
            archive_messages,
        ),
    )
    if commit:
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


def set_document_embedding_model(
    conn: sqlite3.Connection, document_id: str, embedding_model: str
) -> None:
    """Record which embedder actually produced this document's vectors. Written after
    the encode, because the process can swap backends part way through a job."""
    conn.execute(
        "UPDATE documents SET embedding_model=? WHERE id=?", (embedding_model, document_id)
    )
    conn.commit()


def list_documents(conn: sqlite3.Connection, scope: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, scope, kb_id, thread_id, project_id, filename, sha256, status, error, "
        "num_chunks, created_at, linked_folder_id "
        "FROM documents d WHERE scope=? AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        "ORDER BY created_at DESC",
        (scope,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_all_documents(conn: sqlite3.Connection) -> list[dict]:
    """Every uploaded document across all scopes (KBs, threads, projects).

    Archived conversation turns are excluded: they are written by the rolling context
    window rather than uploaded by anyone, so listing them here would show a chat's own
    history back to the user as a pile of files they never added.
    """
    rows = conn.execute(
        "SELECT id, scope, kb_id, thread_id, project_id, filename, sha256, status, error, "
        "num_chunks, stored_path, created_at, linked_folder_id "
        "FROM documents d WHERE NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        "AND d.scope NOT LIKE 'convarchive#_%' ESCAPE '#' "
        "ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_visible_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    """Return a document only while its owning scope is available to readers."""
    row = conn.execute(
        "SELECT d.* FROM documents d WHERE d.id=? AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope)",
        (document_id,),
    ).fetchone()
    return dict(row) if row else None


def document_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM documents WHERE scope=? AND sha256=? AND status!='failed' "
        "AND linked_folder_id IS NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (scope, sha256),
    ).fetchone()
    return row["id"] if row else None


def failed_documents_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, stored_path FROM documents WHERE scope=? AND sha256=? AND status='failed' "
        "AND linked_folder_id IS NULL",
        (scope, sha256),
    ).fetchall()
    return [dict(r) for r in rows]


def add_chunks(
    conn: sqlite3.Connection,
    scope: str,
    document_id: str,
    chunks,
    vectors,
    regions = None,
) -> None:
    """Incrementally index one document's chunks into chunks + FTS5 + vec0.
    ``vectors`` parallels ``chunks``; optional ``regions`` (also parallel) holds
    per-chunk PDF highlight rects, stored as JSON."""
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


def delete_document(
    conn: sqlite3.Connection,
    document_id: str,
    *,
    commit: bool = True,
) -> None:
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
    if commit:
        conn.commit()


def linked_folder_rows_exist(conn: sqlite3.Connection) -> bool:
    """Whether anything here can be hidden by the linked-folder filters.

    One EXISTS per thing they hide, so with all three empty the plain query returns the
    same rows straight out of the FTS index.

    A purged tombstone does not count: every knowledge base delete leaves one for good
    and its scope keeps no documents, so counting it would end the fast path on the first
    delete. Folder-owned documents are counted directly, not via `linked_folders`: a
    crash before `_install_mapping` leaves one that outlives its folder row.
    """
    return bool(
        conn.execute(
            "SELECT EXISTS(SELECT 1 FROM linked_folders) "
            "OR EXISTS(SELECT 1 FROM linked_folder_retired_scopes WHERE purged_at IS NULL) "
            "OR EXISTS(SELECT 1 FROM documents WHERE linked_folder_id IS NOT NULL)"
        ).fetchone()[0]
    )


def search_lexical(conn: sqlite3.Connection, scope, query: str, k: int):
    """BM25 lexical search over one scope or several. Returns
    [(chunk_id, score)], higher = better."""
    mq = _match_query(query)
    if not mq:
        return []
    scopes = _scopes(scope)
    if not scopes:
        return []
    placeholders = ",".join("?" * len(scopes))
    # One snapshot for the gate and the read: WAL pins it at the transaction's first
    # read, so a scope retired in between cannot land rows in a result the gate already
    # decided to run unfiltered. A caller's own transaction is used instead.
    own_read_txn = not conn.in_transaction
    if own_read_txn:
        conn.execute("BEGIN")
    try:
        # The filtered form joins chunks and documents and runs both subqueries for every
        # matched row BEFORE the LIMIT, so it costs more the commoner the query terms are.
        # With nothing linked that work is provably wasted (linked_folder_rows_exist).
        if linked_folder_rows_exist(conn):
            sql = (
                f"SELECT chunks_fts.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"JOIN chunks c ON c.id=chunks_fts.chunk_id "
                f"JOIN documents d ON d.id=c.document_id "
                f"WHERE chunks_fts MATCH ? AND chunks_fts.scope IN ({placeholders}) "
                f"AND NOT EXISTS "
                f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
                f"AND (d.linked_folder_id IS NULL OR EXISTS "
                f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
                f"ORDER BY s LIMIT ?"
            )
        else:
            sql = (
                f"SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? AND scope IN ({placeholders}) "
                f"ORDER BY s LIMIT ?"
            )
        rows = conn.execute(sql, (mq, *scopes, k)).fetchall()
    finally:
        # Read-only, but it has to end: an open snapshot blocks WAL checkpointing.
        if own_read_txn:
            conn.commit()
    # bm25() is negative (more negative = better); flip to higher-is-better.
    return [(r["chunk_id"], -r["s"]) for r in rows]


def search_dense(
    conn: sqlite3.Connection,
    scope,
    vector,
    k: int,
    *,
    embedding_model: str | None = None,
):
    """Cosine KNN over vec0 for one scope or several. Returns
    [(chunk_id, 1 - distance)]. vec0 KNN constrains its partition key by
    equality, so multi-scope runs one query per scope and merges by score.
    ``embedding_model`` is the querying embedder's identity (backend plus model, see
    ``embeddings.embedding_identity``); it drops hits from documents indexed by a
    different embedder of the same width, whose vectors live in another space. Rows
    written before identities carried a backend match on the model name alone, and
    NULL-model legacy documents are assumed current, matching the ingestion dedupe
    rule."""
    if not rag_db.vec_table_exists(conn):
        return []
    dim = rag_db.vec_table_dim(conn)
    if dim is not None and dim != len(vector):
        # Embedding model switched widths and nothing re-indexed yet; the stale
        # table cannot answer new-model queries (vec0 errors on the MATCH).
        return []
    # The pre-tag spelling of the same request, kept acceptable so an existing index
    # keeps answering after an upgrade.
    untagged = config.embedding_identity_model(embedding_model) or embedding_model
    # dict.fromkeys keeps the caller's order and collapses a scope named twice, which
    # is now load-bearing: widening carries per-scope state, and a repeat would both
    # multiply that scope's fetch twice per round and emit its hits twice into the merge.
    scopes = list(
        dict.fromkeys(
            s
            for s in _scopes(scope)
            if not conn.execute(
                "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?", (s,)
            ).fetchone()
        )
    )
    # Over-fetch when filtering so stale-model hits don't starve the top-k. Their
    # distances come from another space, so they can fill every fetched slot while
    # compatible chunks sit further down the KNN list: widen until k of them survive
    # the filter or the scope has nothing left to give.
    #
    # Per scope, not across the merge. vec0 constrains its partition key by equality,
    # so each scope is its own KNN list with its own stale prefix; a project scope
    # buried under another embedder's vectors would otherwise stop widening the
    # moment the thread scope handed over k weak hits, and the merge would then rank
    # a stronger project chunk it never fetched.
    kept: dict[str, list[tuple[str, float]]] = {}
    fetches = dict.fromkeys(scopes, max(k * 3, k + 10))
    pending = list(scopes)
    while pending:
        widen: list[str] = []
        for s in pending:
            fetch = fetches[s]
            rows = conn.execute(
                "SELECT chunk_id, distance FROM chunks_vec "
                "WHERE scope=? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                (s, _f32(vector), fetch),
            ).fetchall()
            kept[s] = _drop_incompatible(
                conn,
                [(r["chunk_id"], 1.0 - r["distance"]) for r in rows],
                embedding_model,
                untagged,
            )
            if len(kept[s]) < k and len(rows) >= fetch and fetch < _MAX_DENSE_FETCH:
                fetches[s] = min(fetch * 4, _MAX_DENSE_FETCH)
                widen.append(s)
        pending = widen
    out = [hit for s in scopes for hit in kept[s]]
    out.sort(key = lambda t: t[1], reverse = True)
    return out[:k]


# Widening is bounded: past this many nearest neighbours per scope the scope is
# effectively another embedder's, and a re-upload is the answer, not a longer scan.
_MAX_DENSE_FETCH = 4096
# One id per bound parameter, kept under the oldest SQLITE_MAX_VARIABLE_NUMBER.
_ID_BATCH = 900


def _drop_incompatible(
    conn: sqlite3.Connection,
    candidates: list[tuple[str, float]],
    embedding_model: str | None,
    untagged: str | None,
) -> list[tuple[str, float]]:
    """Keep the KNN candidates whose document is still live and whose vectors this
    query can be compared against."""
    valid: set[str] = set()
    ids = [cid for cid, _ in candidates]
    for start in range(0, len(ids), _ID_BATCH):
        batch = ids[start : start + _ID_BATCH]
        placeholders = ",".join("?" * len(batch))
        valid.update(
            r["id"]
            for r in conn.execute(
                f"SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id "
                f"WHERE c.id IN ({placeholders}) AND NOT EXISTS "
                f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
                f"AND (d.linked_folder_id IS NULL OR EXISTS "
                f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
                f"AND (? IS NULL OR d.embedding_model IS NULL OR d.embedding_model=? "
                f"OR d.embedding_model=?)",
                (*batch, embedding_model, embedding_model, untagged),
            ).fetchall()
        )
    return [t for t in candidates if t[0] in valid]


def count_untagged_documents(conn: sqlite3.Connection) -> int:
    """Documents whose ``embedding_model`` predates backend tagging.

    Either backend could have written them, because the llama-server fallback never
    recorded that it had taken over, and nothing in the row says which pooling the
    vectors came from. We keep serving them rather than drop a corpus or re-embed one
    behind the user's back, so this exists to say how many are in that state."""
    tags = " ".join(f"AND embedding_model NOT LIKE '{t}:%'" for t in config.EMBEDDING_IDENTITY_TAGS)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM documents WHERE embedding_model IS NOT NULL {tags}"
    ).fetchone()
    return int(row["n"]) if row else 0


def chunks_by_id(conn: sqlite3.Connection, ids) -> dict:
    """Hydrate chunk rows (joined with document filename), keyed by id."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT c.id, c.text, c.document_id, c.chunk_index, c.page_number, "
        f"c.source_page_index, d.filename "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.id IN ({placeholders}) AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id))",
        list(ids),
    ).fetchall()
    return {r["id"]: r for r in rows}


def all_chunks_for_scope(conn: sqlite3.Connection, scope) -> list[dict]:
    """Every completed-document chunk for a scope, ordered document-then-index and
    joined with the document filename. Backs whole-document context injection, so
    it does no retrieval or embedding."""
    scopes = _scopes(scope)
    if not scopes:
        return []
    placeholders = ",".join("?" * len(scopes))
    rows = conn.execute(
        f"SELECT c.id, c.text, c.document_id, c.chunk_index, c.page_number, "
        f"c.token_count, d.filename, d.created_at "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.scope IN ({placeholders}) AND d.status='completed' "
        f"AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
        f"ORDER BY d.created_at, c.document_id, c.chunk_index",
        list(scopes),
    ).fetchall()
    return [dict(r) for r in rows]


def scope_token_estimate(conn: sqlite3.Connection, scope) -> int:
    """Upper-bound token total for a scope's completed chunks without hydrating text.
    Mirrors ``all_chunks_for_scope`` + the ``tool._row_token_count`` fallback (stored
    count, else length/4), so the whole-doc budget can be checked before loading text."""
    scopes = _scopes(scope)
    if not scopes:
        return 0
    placeholders = ",".join("?" * len(scopes))
    row = conn.execute(
        f"SELECT COALESCE(SUM(CASE WHEN c.token_count > 0 THEN c.token_count "
        f"ELSE MAX(1, length(COALESCE(c.text, '')) / 4) END), 0) AS total "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.scope IN ({placeholders}) AND d.status='completed' "
        f"AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id))",
        list(scopes),
    ).fetchone()
    return int(row["total"] or 0)
