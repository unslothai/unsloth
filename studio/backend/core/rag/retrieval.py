# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""High-level retrieval surface for RAG: BM25, dense, and RRF hybrid.

Reciprocal Rank Fusion is parameter-light: each candidate's fused score
is the sum of ``1 / (rrf_k + rank)`` across rankers. It avoids the
need to calibrate score scales between BM25 (raw, unbounded) and cosine
similarity (-1..1).

Hits also carry the raw dense cosine score when available so callers
can apply a meaningful similarity threshold (e.g., "drop hits below
0.3 cosine") — the fused RRF score isn't on a comparable scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.rag.config import (
    RAG_RRF_K,
    RAG_TOP_K_BM25,
    RAG_TOP_K_DENSE,
    RAG_TOP_K_HYBRID,
)

from . import bm25, embeddings, vector_store


@dataclass(frozen = True)
class Hit:
    chunk_id: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None
    kind: str = "text"
    # Raw cosine similarity from the dense retriever (0..1 for
    # normalized embeddings). None when this chunk wasn't returned by
    # the dense pass (BM25-only hit) — callers applying a similarity
    # floor should treat None as "no signal" and exclude it.
    dense_score: float | None = None


def retrieve_bm25(scope: str, query: str, k: int | None = None) -> list[Hit]:
    limit = k or RAG_TOP_K_BM25
    return [Hit(chunk_id = cid, score = s) for cid, s in bm25.search(scope, query, limit)]


def retrieve_dense(
    scope: str,
    query: str,
    k: int | None = None,
    *,
    document_ids: list[str] | None = None,
    embedder_model: str | None = None,
) -> list[Hit]:
    """Dense retrieval. `embedder_model` MUST match the model that
    populated this scope's vectors — using a different one yields a
    dim mismatch (shape (N, scope_dim) vs (query_dim,)) at distance
    compute time. Callers should resolve from the KB / thread settings
    before passing.
    """
    limit = k or RAG_TOP_K_DENSE
    vector = embeddings.encode(
        [query],
        normalize = True,
        model_name = embedder_model,
    )[0].tolist()
    raw = vector_store.search(
        scope,
        query_vector = vector,
        top_k = limit,
        document_ids = document_ids,
    )
    out: list[Hit] = []
    for r in raw:
        payload = r["payload"]
        out.append(
            Hit(
                chunk_id = r["chunk_id"],
                score = r["score"],
                document_id = payload.get("document_id"),
                chunk_index = payload.get("chunk_index"),
                kind = payload.get("kind", "text"),
                dense_score = r["score"],
            )
        )
    return out


def _rrf_fuse(
    rankings: list[list[Hit]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[Hit]:
    fused: dict[str, float] = {}
    seen: dict[str, Hit] = {}
    # Track the dense cosine score per chunk so it survives fusion —
    # callers downstream filter on this, not the RRF score.
    dense_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            if hit.chunk_id not in seen:
                seen[hit.chunk_id] = hit
            if hit.dense_score is not None:
                dense_scores[hit.chunk_id] = hit.dense_score
    ordered = sorted(fused.items(), key = lambda kv: kv[1], reverse = True)[:top_k]
    return [
        Hit(
            chunk_id = cid,
            score = score,
            document_id = seen[cid].document_id,
            chunk_index = seen[cid].chunk_index,
            kind = seen[cid].kind,
            dense_score = dense_scores.get(cid),
        )
        for cid, score in ordered
    ]


def retrieve_hybrid(
    scope: str,
    query: str,
    *,
    k: int | None = None,
    k_bm25: int | None = None,
    k_dense: int | None = None,
    document_ids: list[str] | None = None,
    embedder_model: str | None = None,
) -> list[Hit]:
    bm25_hits = retrieve_bm25(scope, query, k_bm25 or RAG_TOP_K_BM25)
    dense_hits = retrieve_dense(
        scope,
        query,
        k_dense or RAG_TOP_K_DENSE,
        document_ids = document_ids,
        embedder_model = embedder_model,
    )
    return _rrf_fuse(
        [bm25_hits, dense_hits],
        rrf_k = RAG_RRF_K,
        top_k = k or RAG_TOP_K_HYBRID,
    )


def filter_by_min_score(hits: list[Hit], min_score: float) -> list[Hit]:
    """Drop hits whose dense cosine score is below ``min_score``.

    Hits without a dense score (BM25-only) are dropped too — there's
    no comparable signal to evaluate them against the similarity floor.
    Use ``min_score = 0.0`` (or negative) to disable the filter.
    """
    if min_score <= 0.0:
        return hits
    return [h for h in hits if h.dense_score is not None and h.dense_score >= min_score]
