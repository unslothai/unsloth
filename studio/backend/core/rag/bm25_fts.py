# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Incremental BM25 on SQLite FTS5, living in the shared rag.db.

Drop-in for the bm25s-backed `bm25.py` (same `rebuild_index` / `search` /
`delete_scope` surface) plus an incremental `add_chunks`. FTS5 supports row-level
INSERT/DELETE, so a new document inserts only its own rows -- no scope-wide
rebuild. `MATCH` returns only rows that contain a query term (no zero-score
padding), and the `porter` tokenizer adds stemming. Scope is an UNINDEXED column
filtered in the WHERE clause; the dense leg stays in sqlite-vec untouched.
"""

from __future__ import annotations

import re
import threading

from core.rag.db import get_rag_connection
from loggers import get_logger

logger = get_logger(__name__)

_schema_lock = threading.Lock()
_schema_ready = False

# FTS5 query terms: alphanumeric runs only. Anything else (quotes, hyphens,
# operators) is dropped so a raw user query can never form invalid MATCH syntax.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn = get_rag_connection()
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(
                chunk_id UNINDEXED,
                scope UNINDEXED,
                text,
                tokenize = 'porter unicode61'
            );
            """
        )
        conn.commit()
        _schema_ready = True


def add_chunks(scope: str, chunks: list[dict]) -> None:
    """Insert only these chunks' rows. Each chunk: {id, text}. Incremental O(len)."""
    if not chunks:
        return
    _ensure_schema()
    conn = get_rag_connection()
    conn.executemany(
        "INSERT INTO rag_fts (chunk_id, scope, text) VALUES (?, ?, ?)",
        [(c["id"], scope, c["text"]) for c in chunks],
    )
    conn.commit()


def rebuild_index(scope: str, chunks: list[dict]) -> None:
    """Compat path: replace a scope's rows. Empty list clears the scope."""
    _ensure_schema()
    conn = get_rag_connection()
    conn.execute("DELETE FROM rag_fts WHERE scope = ?", (scope,))
    conn.commit()
    add_chunks(scope, chunks)


def _match_query(query: str) -> str | None:
    terms = _TOKEN_RE.findall(query.lower())
    if not terms:
        return None
    # OR the terms (recall-oriented, like bm25s default); quote each so FTS5
    # treats it as a bare token, never an operator.
    return " OR ".join(f'"{t}"' for t in terms)


def search(scope: str, query: str, k: int) -> list[tuple[str, float]]:
    """Best-first (chunk_id, score). score = -bm25() so higher is better."""
    _ensure_schema()
    match = _match_query(query)
    if match is None:
        return []
    conn = get_rag_connection()
    rows = conn.execute(
        """
        SELECT chunk_id, bm25(rag_fts) AS score
        FROM rag_fts
        WHERE scope = ? AND rag_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (scope, match, k),
    ).fetchall()
    # FTS5 bm25() is negative with more-negative = better; negate so callers see
    # higher = better, list already best-first from ORDER BY score ASC.
    return [(row["chunk_id"], -float(row["score"])) for row in rows]


def delete_scope(scope: str) -> None:
    _ensure_schema()
    conn = get_rag_connection()
    conn.execute("DELETE FROM rag_fts WHERE scope = ?", (scope,))
    conn.commit()


def delete_document(document_id: str, chunk_ids: list[str]) -> None:
    """Incremental per-document delete by chunk ids (FTS has no document_id col)."""
    if not chunk_ids:
        return
    _ensure_schema()
    conn = get_rag_connection()
    conn.executemany(
        "DELETE FROM rag_fts WHERE chunk_id = ?",
        [(cid,) for cid in chunk_ids],
    )
    conn.commit()
