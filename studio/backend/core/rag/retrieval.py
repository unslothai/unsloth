# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lexical (FTS5) and dense (vec0 cosine) retrieval fused by Reciprocal Rank Fusion; dense_score is
kept so callers can apply a similarity floor."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sqlite3
from dataclasses import dataclass

from . import config, embeddings, reembed, store

logger = logging.getLogger(__name__)

_query_vectors: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "rag_query_vectors", default = None
)


@contextlib.contextmanager
def query_vector_memo():
    """Within the block, each (model, query) is encoded once; a nested entry shares the outer memo."""
    if _query_vectors.get() is not None:
        yield
        return
    token = _query_vectors.set({})
    try:
        yield
    finally:
        _query_vectors.reset(token)


@dataclass
class Hit:
    chunk_id: str
    score: float
    lexical_score: float | None = None
    dense_score: float | None = None


def retrieve_lexical(
    conn: sqlite3.Connection,
    scope: str | list[str],
    query: str,
    k: int | None = None,
    *,
    match_query: str | None = None,
    newest_first: bool = False,
    oldest_first: bool = False,
) -> list[Hit]:
    k = k or config.TOP_K_LEXICAL
    return [
        Hit(cid, s, lexical_score = s)
        for cid, s in store.search_lexical(
            conn,
            scope,
            query,
            k,
            match_query = match_query,
            newest_first = newest_first,
            oldest_first = oldest_first,
        )
    ]


def retrieve_dense(
    conn: sqlite3.Connection,
    scope: str | list[str],
    query: str,
    k: int | None = None,
    *,
    model_name: str | None = None,
) -> list[Hit]:
    k = k or config.TOP_K_DENSE
    effective = model_name or config.effective_embedding_model()
    memo = _query_vectors.get()
    key = (effective, query)
    if memo is not None and key in memo:
        vec, identity = memo[key]
    else:
        # Take identity from the encode, not the process: a concurrent ST failure may have swapped it.
        vectors, identity = embeddings.encode_with_identity(
            [query], model_name = effective, normalize = True
        )
        vec = vectors[0]
        if memo is not None:
            memo[key] = (vec, identity)
    _warn_once_on_untagged(conn)
    try:
        reembed.start_after_backend_change(identity)
    except Exception:  # noqa: BLE001 - re-embedding must never break retrieval
        logger.warning("could not start re-embedding after an embedder change", exc_info = True)
    return [
        Hit(cid, s, dense_score = s)
        for cid, s in store.search_dense(conn, scope, vec, k, embedding_model = identity)
    ]


_untagged_warned = False


def _warn_once_on_untagged(conn: sqlite3.Connection) -> None:
    """Warns once that untagged documents are served as current; reports rather than re-embeds unasked."""
    global _untagged_warned
    if _untagged_warned:
        return
    _untagged_warned = True
    try:
        stale = store.count_untagged_documents(conn)
    except Exception:  # noqa: BLE001 - a diagnostic must never break retrieval
        return
    if stale:
        logger.warning(
            "%d document(s) were indexed before the embedder was recorded; they are "
            "searched as if current. Re-upload them if dense results look wrong.",
            stale,
        )


def _rrf(rankings: list[list[Hit]], rrf_k: int, top_k: int) -> list[Hit]:
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            cur = best.get(hit.chunk_id)
            if cur is None:
                best[hit.chunk_id] = Hit(hit.chunk_id, 0.0, hit.lexical_score, hit.dense_score)
            else:
                cur.lexical_score = (
                    cur.lexical_score if cur.lexical_score is not None else hit.lexical_score
                )
                cur.dense_score = (
                    cur.dense_score if cur.dense_score is not None else hit.dense_score
                )
    out: list[Hit] = []
    for cid, s in sorted(fused.items(), key = lambda kv: kv[1], reverse = True)[:top_k]:
        h = best[cid]
        h.score = s
        out.append(h)
    return out


def retrieve_hybrid(
    conn: sqlite3.Connection,
    scope: str | list[str],
    query: str,
    *,
    k: int | None = None,
    model_name: str | None = None,
    mode: str = "hybrid",
    lexical_query: str | None = None,
    lexical_k: int | None = None,
    dense_k: int | None = None,
) -> list[Hit]:
    """lexical_query changes only the FTS5 leg; the dense leg always encodes the natural-language query.
    lexical_k / dense_k size the legs (default TOP_K_LEXICAL / TOP_K_DENSE): fused output never exceeds their sum."""
    k = k if k is not None else config.TOP_K_HYBRID
    k = int(k)  # tool-call / scope top_k may arrive as a float; LIMIT + slice need int
    if mode == "lexical":
        return retrieve_lexical(conn, scope, query, k, match_query = lexical_query)
    if mode == "dense":
        return retrieve_dense(conn, scope, query, k, model_name = model_name)
    lexical = retrieve_lexical(
        conn, scope, query, lexical_k or config.TOP_K_LEXICAL, match_query = lexical_query
    )
    dense = retrieve_dense(conn, scope, query, dense_k or config.TOP_K_DENSE, model_name = model_name)
    return _rrf([lexical, dense], config.RRF_K, k)


def filter_min_score(hits: list[Hit], min_score: float) -> list[Hit]:
    """Cosine floor; gates only hits with a dense_score (lexical-only pass)."""
    if min_score <= 0:
        return hits
    return [h for h in hits if h.dense_score is None or h.dense_score >= min_score]
