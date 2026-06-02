# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sqlite-vec vector store. Scope filter (kb_<id>/thread_<id>) keeps mixed-dim
scopes safe — per-scope embedder resolver guarantees one dim per scope."""

from __future__ import annotations

import json
import re
from typing import Iterable

from loggers import get_logger

logger = get_logger(__name__)


def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def collection_exists(scope: str) -> bool:
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    row = conn.execute(
        "SELECT 1 FROM rag_vectors WHERE scope = ? LIMIT 1",
        (scope,),
    ).fetchone()
    return row is not None


def ensure_collection(scope: str, dim: int) -> None:
    """No-op; kept for API parity. Vectors go straight into the shared table."""
    _ = scope, dim


def upsert_chunks(scope: str, points: Iterable[dict]) -> None:
    """Insert/update vectors + the lexical FTS5 index. Each point: {id, vector, payload}."""
    import sqlite_vec

    from core.rag.db import get_rag_connection

    rows = []
    fts_rows: list[tuple[str, str, str]] = []  # (text, chunk_id, scope)
    fts_delete: list[tuple[str]] = []
    for p in points:
        payload = p.get("payload") or {}
        vec = list(p["vector"])
        kind = str(payload.get("kind") or "text")
        rows.append(
            (
                p["id"],
                scope,
                str(payload.get("document_id") or ""),
                int(payload.get("chunk_index") or 0),
                kind,
                len(vec),
                sqlite_vec.serialize_float32(vec),
                json.dumps(payload, default = str),
            )
        )
        text = payload.get("text")
        if kind in ("text", "caption") and text:
            # FTS5 has no UPSERT; delete-then-insert keeps re-ingest idempotent.
            fts_delete.append((p["id"],))
            fts_rows.append((text, p["id"], scope))
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
    if fts_rows:
        conn.executemany(
            "DELETE FROM rag_chunks_fts WHERE chunk_id = ?", fts_delete
        )
        conn.executemany(
            "INSERT INTO rag_chunks_fts (text, chunk_id, scope) VALUES (?, ?, ?)",
            fts_rows,
        )
    conn.commit()


def search(
    scope: str,
    query_vector: list[float],
    *,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Cosine search; returns {chunk_id, score, payload} with score = 1 - distance."""
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


def update_chunk_payload_fields(
    scope: str,
    updates: dict[str, dict],
) -> None:
    """Merge locator fields into existing vector payload JSON by chunk id."""
    from core.rag.db import get_rag_connection

    if not updates:
        return
    conn = get_rag_connection()
    rows = conn.execute(
        f"""
        SELECT chunk_id, payload_json
        FROM rag_vectors
        WHERE scope = ? AND chunk_id IN ({",".join("?" for _ in updates)})
        """,
        [scope, *updates.keys()],
    ).fetchall()
    payload_rows: list[tuple[str, str]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        payload.update(updates.get(row["chunk_id"], {}))
        payload_rows.append((json.dumps(payload, default = str), row["chunk_id"], scope))
    if not payload_rows:
        return
    conn.executemany(
        """
        UPDATE rag_vectors
        SET payload_json = ?
        WHERE chunk_id = ? AND scope = ?
        """,
        payload_rows,
    )
    conn.commit()


def delete_scope(scope: str) -> None:
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    conn.execute("DELETE FROM rag_vectors WHERE scope = ?", (scope,))
    conn.execute("DELETE FROM rag_chunks_fts WHERE scope = ?", (scope,))
    conn.commit()


def delete_document(scope: str, document_id: str) -> None:
    from core.rag.db import get_rag_connection

    conn = get_rag_connection()
    chunk_ids = [
        row["chunk_id"]
        for row in conn.execute(
            "SELECT chunk_id FROM rag_vectors WHERE scope = ? AND document_id = ?",
            (scope, document_id),
        ).fetchall()
    ]
    conn.execute(
        "DELETE FROM rag_vectors WHERE scope = ? AND document_id = ?",
        (scope, document_id),
    )
    if chunk_ids:
        conn.executemany(
            "DELETE FROM rag_chunks_fts WHERE chunk_id = ?",
            [(cid,) for cid in chunk_ids],
        )
    conn.commit()


_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def _match_query(query: str) -> str:
    """User text → safe FTS5 OR-of-quoted-terms; quoting defuses FTS5 operators."""
    tokens = _FTS_TOKEN.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


def search_lexical(scope: str, query: str, k: int) -> list[tuple[str, float]]:
    """BM25 lexical search via SQLite FTS5. Returns [(chunk_id, score)], higher = better."""
    from core.rag.db import get_rag_connection

    match = _match_query(query)
    if not match:
        return []
    conn = get_rag_connection()
    rows = conn.execute(
        "SELECT chunk_id, bm25(rag_chunks_fts) AS s FROM rag_chunks_fts "
        "WHERE rag_chunks_fts MATCH ? AND scope = ? ORDER BY s LIMIT ?",
        (match, scope, int(k)),
    ).fetchall()
    # bm25() is negative (more negative = better); flip to higher-is-better.
    return [(row["chunk_id"], -float(row["s"])) for row in rows]
