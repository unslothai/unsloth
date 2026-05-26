# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG retrieval: BM25, dense, and RRF hybrid. Hits carry dense_score for thresholding."""

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
    # Raw cosine; None for BM25-only hits.
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
    """Dense retrieval. embedder_model MUST match the model that populated this scope."""
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
    # Preserve dense_score through fusion for downstream thresholding.
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
    """Drop hits whose dense_score < min_score; BM25-only hits dropped too."""
    if min_score <= 0.0:
        return hits
    return [h for h in hits if h.dense_score is not None and h.dense_score >= min_score]
