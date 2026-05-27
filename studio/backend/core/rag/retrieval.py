# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG retrieval: BM25, dense, and RRF hybrid. Hits carry dense_score for thresholding."""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.rag.config import (
    RAG_RRF_K,
    RAG_TOP_K_BM25,
    RAG_TOP_K_DENSE,
    RAG_TOP_K_HYBRID,
)

from . import bm25, embeddings, vector_store

# Match "Figure 1", "Figure 1.2", "Figure B.1", "Table 4", "Fig. 5" anywhere
# in the query. Used to inject a third retrieval source that directly looks
# up chunks anchored by these references — dense vectors don't preserve
# figure numbers, so without this an exact-numbered query gets out-ranked
# by chunks describing other figures that share more vocabulary with the
# question.
_FIGURE_REF_RE = re.compile(
    r"\b(Figure|Fig\.|Table|Tab\.)\s+([A-Z]?\.?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _extract_figure_refs(query: str) -> list[str]:
    """Return normalized 'Figure N' / 'Table N' references found in query."""
    refs: list[str] = []
    seen: set[str] = set()
    for m in _FIGURE_REF_RE.finditer(query):
        head = m.group(1).lower()
        label = "Figure" if head.startswith("fig") else "Table"
        ref = f"{label} {m.group(2)}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


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


def retrieve_figure_refs(
    scope: str,
    query: str,
    *,
    k: int = 5,
    document_ids: list[str] | None = None,
) -> list[Hit]:
    """Look up chunks anchored at a 'Figure N:' / 'Table N:' caption that
    the query references. Returns at most ``k`` hits — usually 0 or 1.

    Chunks produced by the figure-boundary chunker start with the literal
    caption, so a SQL prefix match is enough; we don't need full-text
    search here.
    """
    refs = _extract_figure_refs(query)
    if not refs:
        return []

    from .db import get_rag_connection

    placeholders_docs = ""
    params: list = [scope]
    if document_ids:
        placeholders_docs = (
            f" AND document_id IN ({','.join('?' * len(document_ids))})"
        )
        params.extend(document_ids)

    like_clauses: list[str] = []
    for ref in refs:
        # Match "Figure 1:" and "Figure 1." (period-terminated captions).
        like_clauses.append(
            "json_extract(payload_json, '$.text') LIKE ?"
            " OR json_extract(payload_json, '$.text') LIKE ?"
        )
        params.extend([f"{ref}:%", f"{ref}.%"])

    sql = (
        "SELECT chunk_id, document_id, chunk_index, kind"
        " FROM rag_vectors"
        f" WHERE scope = ?{placeholders_docs}"
        " AND kind = 'text'"
        f" AND ({' OR '.join(like_clauses)})"
        f" LIMIT {int(k)}"
    )

    out: list[Hit] = []
    with get_rag_connection() as conn:
        for row in conn.execute(sql, params):
            out.append(
                Hit(
                    chunk_id = row[0],
                    score = 1.0,
                    document_id = row[1],
                    chunk_index = row[2],
                    kind = row[3] or "text",
                )
            )
    return out


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
    rankings = [bm25_hits, dense_hits]
    fig_hits = retrieve_figure_refs(scope, query, document_ids = document_ids)
    if fig_hits:
        rankings.append(fig_hits)
    return _rrf_fuse(
        rankings,
        rrf_k = RAG_RRF_K,
        top_k = k or RAG_TOP_K_HYBRID,
    )


def filter_by_min_score(hits: list[Hit], min_score: float) -> list[Hit]:
    """Drop hits whose dense_score < min_score; BM25-only hits dropped too."""
    if min_score <= 0.0:
        return hits
    return [h for h in hits if h.dense_score is not None and h.dense_score >= min_score]
