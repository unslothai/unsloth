# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``search_knowledge_base`` LLM tool: scope resolution + hit formatting.

KB scope wins over thread scope. Hits render as ``<chunk>`` blocks for the model
plus a parallel citation source-map for clickable sources in the chat layer.
Opens/closes its own ``rag_db`` connection per call.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from storage import rag_db

from . import config, retrieval
from .store import kb_scope, thread_scope

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's uploaded documents and knowledge bases for "
            "relevant passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max chunks to return.",
                },
            },
            "required": ["query"],
        },
    },
}


def _resolve_scope(scope_kb_id: str | None, scope_thread_id: str | None) -> str | None:
    if scope_kb_id:
        return kb_scope(scope_kb_id)
    if scope_thread_id:
        return thread_scope(scope_thread_id)
    return None


def _format(rows, hits) -> tuple[str, list[dict]]:
    """Render hits as ``<chunk>`` blocks and build a citation source-map."""
    if not hits:
        return "No matching chunks were found in the knowledge base.", []
    blocks: list[str] = []
    sources: list[dict] = []
    for i, h in enumerate(hits, 1):
        r = rows.get(h.chunk_id)
        filename = (r["filename"] if r else None) or "unknown"
        page = r["page_number"] if r else None
        text = r["text"] if r else ""
        src = quoteattr(filename)
        page_attr = f" page={quoteattr(str(page))}" if page else ""
        blocks.append(f'<chunk id="{i}" source={src}{page_attr}>\n{text}\n</chunk>')
        sources.append(
            {
                "citationId": i,
                "chunkId": h.chunk_id,
                "documentId": r["document_id"] if r else None,
                "filename": filename,
                "page": page,
                "text": text,
                "score": round(float(h.score), 4) if h.score is not None else None,
            }
        )
    return "\n\n".join(blocks), sources


def search_knowledge_base_with_sources(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    top_k: int | None = None,
    min_score: float = 0.0,
    model_name: str | None = None,
    mode: str = "hybrid",
    rrf_k: int | None = None,
    top_k_lexical: int | None = None,
    top_k_dense: int | None = None,
) -> tuple[str, list[dict]]:
    """Search -> ``(rendered_text, citation_sources)``; sources align with each
    rendered ``<chunk>`` block's ``id``. Retrieval knobs fall back to config."""
    if not query or not query.strip():
        return "Error: query is empty.", []
    scope = _resolve_scope(scope_kb_id, scope_thread_id)
    if scope is None:
        return "No documents are attached to this chat.", []

    conn = rag_db.get_connection()
    try:
        hits = retrieval.retrieve_hybrid(
            conn,
            scope,
            query,
            k = top_k or config.TOP_K_HYBRID,
            model_name = model_name,
            mode = mode,
            rrf_k = rrf_k,
            top_k_lexical = top_k_lexical,
            top_k_dense = top_k_dense,
        )
        hits = retrieval.filter_min_score(hits, min_score)
        rows = store_rows(conn, hits)
    finally:
        conn.close()
    return _format(rows, hits)


def store_rows(conn, hits):
    """Hydrate chunk rows for a list of hits."""
    from . import store

    return store.chunks_by_id(conn, [h.chunk_id for h in hits])


def search_for_autoinject(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    top_k: int | None = None,
    min_dense_score: float = 0.55,
    model_name: str | None = None,
    mode: str = "hybrid",
    rrf_k: int | None = None,
    top_k_lexical: int | None = None,
    top_k_dense: int | None = None,
) -> tuple[str, list[dict]] | None:
    """Forced-retrieval variant for auto-injection.

    Returns ``(rendered_text, sources)`` only when some hit's dense (cosine)
    similarity clears ``min_dense_score``; else ``None`` (inject nothing). The
    dense gate keeps weak lexical-only matches from polluting unrelated answers
    (e.g. agriculture docs vs "capital of France").
    """
    if not query or not query.strip():
        return None
    scope = _resolve_scope(scope_kb_id, scope_thread_id)
    if scope is None:
        return None
    k = top_k or config.TOP_K_HYBRID
    conn = rag_db.get_connection()
    try:
        hits = retrieval.retrieve_hybrid(
            conn,
            scope,
            query,
            k = k,
            model_name = model_name,
            mode = mode,
            rrf_k = rrf_k,
            top_k_lexical = top_k_lexical,
            top_k_dense = top_k_dense,
        )
        strong = [
            h
            for h in hits
            if h.dense_score is not None and h.dense_score >= min_dense_score
        ][:k]
        if not strong:
            return None
        rows = store_rows(conn, strong)
    finally:
        conn.close()
    text, sources = _format(rows, strong)
    return (text, sources) if sources else None


def search_knowledge_base(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    top_k: int | None = None,
    min_score: float = 0.0,
    model_name: str | None = None,
) -> str:
    """Text-only variant of :func:`search_knowledge_base_with_sources`."""
    text, _sources = search_knowledge_base_with_sources(
        query = query,
        scope_kb_id = scope_kb_id,
        scope_thread_id = scope_thread_id,
        top_k = top_k,
        min_score = min_score,
        model_name = model_name,
    )
    return text
