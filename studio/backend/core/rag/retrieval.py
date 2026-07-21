# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lexical (FTS5) + dense (vec0 cosine) retrieval fused via Reciprocal Rank
Fusion. ``dense_score`` is carried so callers can apply a similarity floor."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import config, embeddings, store


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
) -> list[Hit]:
    k = k or config.TOP_K_LEXICAL
    return [
        Hit(cid, s, lexical_score = s)
        for cid, s in store.search_lexical(conn, scope, query, k)
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
    vec = embeddings.encode([query], model_name = effective, normalize = True)[0]
    return [
        Hit(cid, s, dense_score = s)
        for cid, s in store.search_dense(conn, scope, vec, k, embedding_model = effective)
    ]


def _rrf(rankings: list[list[Hit]], rrf_k: int, top_k: int) -> list[Hit]:
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            cur = best.get(hit.chunk_id)
            if cur is None:
                best[hit.chunk_id] = Hit(
                    hit.chunk_id, 0.0, hit.lexical_score, hit.dense_score
                )
            else:
                cur.lexical_score = (
                    cur.lexical_score
                    if cur.lexical_score is not None
                    else hit.lexical_score
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
) -> list[Hit]:
    """``mode`` picks the backend: lexical-only, dense-only, or RRF of both
    (default). Pool sizes and the RRF constant come from config."""
    k = k if k is not None else config.TOP_K_HYBRID
    k = int(k)  # tool-call / scope top_k may arrive as a float; LIMIT + slice need int
    if mode == "lexical":
        return retrieve_lexical(conn, scope, query, k)
    if mode == "dense":
        return retrieve_dense(conn, scope, query, k, model_name = model_name)
    lexical = retrieve_lexical(conn, scope, query, config.TOP_K_LEXICAL)
    dense = retrieve_dense(
        conn, scope, query, config.TOP_K_DENSE, model_name = model_name
    )
    return _rrf([lexical, dense], config.RRF_K, k)


def filter_min_score(hits: list[Hit], min_score: float) -> list[Hit]:
    """Cosine floor; gates only hits with a dense_score (lexical-only pass)."""
    if min_score <= 0:
        return hits
    return [h for h in hits if h.dense_score is None or h.dense_score >= min_score]
