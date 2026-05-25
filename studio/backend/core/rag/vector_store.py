# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite-backed vector store for RAG, distance math via sqlite-vec.

Replaces the previous Qdrant local-mode store. Vectors are stored as
BLOBs in a single `rag_vectors` table inside rag.db, keyed by
chunk_id; cosine distance is computed by sqlite-vec's
`vec_distance_cosine(blob, blob)` scalar function.

A "scope" is `kb_<uuid>` for standalone knowledge bases or
`thread_<uuid>` for per-thread document sets. The scope column is
indexed; queries filter by scope before computing distances so
different scopes can hold vectors of different dimensions without
breaking the cosine math. The per-scope embedder resolver
(routes/rag.py:_resolve_scope_embedder) guarantees one embedder per
scope, so dims within a scope are always consistent.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def collection_exists(scope: str) -> bool:
    """Whether the scope has any indexed vectors.

    Used by callers (notably retrieval.retrieve_dense) to short-circuit
    when the scope was never populated. Cheap — a covering index hit.
    """
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    row = conn.execute(
        "SELECT 1 FROM rag_vectors WHERE scope = ? LIMIT 1",
        (scope,),
    ).fetchone()
    return row is not None


def ensure_collection(scope: str, dim: int) -> None:
    """No-op for sqlite-vec — vectors get inserted directly into the
    shared table. Kept for API parity with the previous Qdrant store
    so ingestion callers don't need conditional logic.
    """
    _ = scope, dim  # unused; signature preserved


def upsert_chunks(scope: str, points: Iterable[dict]) -> None:
    """Insert/update vectors. Each point: {id, vector, payload}.

    Conflict resolution is per chunk_id (the primary key): re-ingesting
    overwrites in place. Payload is round-tripped as JSON so the
    Qdrant-shaped {filename, page_number, kind, ...} dicts callers
    already build can be reused unchanged.
    """
    import sqlite_vec

    from core.rag.db import get_rag_connection

    rows = []
    for p in points:
        payload = p.get("payload") or {}
        vec = list(p["vector"])
        rows.append(
            (
                p["id"],
                scope,
                str(payload.get("document_id") or ""),
                int(payload.get("chunk_index") or 0),
                str(payload.get("kind") or "text"),
                len(vec),
                sqlite_vec.serialize_float32(vec),
                json.dumps(payload, default = str),
            )
        )
    if not rows:
        return
    conn = get_rag_connection()
    conn.executemany(
        """
        INSERT INTO rag_vectors
            (chunk_id, scope, document_id, chunk_index, kind, dim, vector, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            scope        = excluded.scope,
            document_id  = excluded.document_id,
            chunk_index  = excluded.chunk_index,
            kind         = excluded.kind,
            dim          = excluded.dim,
            vector       = excluded.vector,
            payload_json = excluded.payload_json
        """,
        rows,
    )
    conn.commit()


def search(
    scope: str,
    query_vector: list[float],
    *,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Cosine-distance search filtered to a single scope.

    Returns rows in the same shape as the old Qdrant path —
    {chunk_id, score, payload} — where ``score`` is cosine similarity
    in [0, 1] (vec_distance_cosine returns 1 - similarity, so we
    invert). Filtered-by-document_ids variant for the per-thread
    "search only these uploads" case.
    """
    import sqlite_vec

    from core.rag.db import get_rag_connection

    if not collection_exists(scope):
        return []

    serialized = sqlite_vec.serialize_float32(list(query_vector))
    sql = (
        "SELECT chunk_id, payload_json, "
        "       vec_distance_cosine(vector, ?) AS distance "
        "FROM rag_vectors WHERE scope = ?"
    )
    params: list = [serialized, scope]
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        sql += f" AND document_id IN ({placeholders})"
        params.extend(document_ids)
    sql += " ORDER BY distance ASC LIMIT ?"
    params.append(int(top_k))

    conn = get_rag_connection()
    rows = conn.execute(sql, params).fetchall()

    out: list[dict] = []
    for row in rows:
        score = 1.0 - float(row["distance"])
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "score": score,
                "payload": payload,
            }
        )
    return out


def delete_scope(scope: str) -> None:
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    conn.execute("DELETE FROM rag_vectors WHERE scope = ?", (scope,))
    conn.commit()


def delete_document(scope: str, document_id: str) -> None:
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    conn.execute(
        "DELETE FROM rag_vectors WHERE scope = ? AND document_id = ?",
        (scope, document_id),
    )
    conn.commit()
