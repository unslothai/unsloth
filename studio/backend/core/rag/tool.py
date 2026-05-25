# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`search_knowledge_base` tool — RAG retrieval surfaced to the LLM.

Invoked from `core/inference/tools.execute_tool` when the local model
emits a `search_knowledge_base` call. The handler runs the existing
hybrid retrieval, hydrates chunk text + filename + page number from
sqlite, and returns a Markdown-with-numbered-citations string that
the LLM consumes as the tool-result message.

Scope (`kb_id` / `thread_id`) is not exposed as a tool argument — it
comes from the chat-completions request body (`rag_scope`) so the
LLM doesn't need to know about KB UUIDs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's attached documents for information relevant to "
            "the user's question. Call this when the user references content "
            "from their docs, asks fact-heavy questions, or needs grounded "
            "citations. Returns numbered chunks with source filenames; cite "
            "them in your reply as [1], [2], etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused search query — phrase it as the question "
                        "you want answered, not as a keyword list."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": (
                        "How many chunks to retrieve (default 5). Higher = "
                        "more grounding, more tokens."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _format_hits_for_llm(hits: list[Any]) -> str:
    """Render hits as numbered Markdown citations for the LLM.

    Empty results produce a one-line message rather than an empty
    string — the model needs to know the search ran but found nothing
    so it can fall back to its own knowledge or ask the user.
    """
    if not hits:
        return (
            "No matching chunks were found in the attached documents. "
            "Either nothing in this scope is relevant, or no documents "
            "have been ingested yet."
        )
    lines: list[str] = []
    for index, hit in enumerate(hits, start = 1):
        name = hit.get("filename") or "unknown source"
        page = hit.get("page_number")
        suffix = f" (page {page})" if page is not None else ""
        text = (hit.get("text") or "").strip()
        lines.append(f"[{index}] {name}{suffix}: {text}")
    return "\n\n".join(lines)


def search_knowledge_base(
    *,
    query: str,
    top_k: int | None = None,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    enable_rerank: bool = False,
    reranker_model: str | None = None,
    default_top_k: int = 5,
    min_score: float = 0.0,
) -> str:
    """Execute the RAG search and return a tool-result string.

    `kb_id` takes precedence over `thread_id` when both are set —
    matches the create/upload contract that a document belongs to one
    or the other, never both. ``min_score`` is a cosine-similarity
    floor on dense hits; chunks below it are dropped.
    """
    if not query or not query.strip():
        return "Error: empty query."

    if not scope_kb_id and not scope_thread_id:
        return (
            "No knowledge base or thread documents are configured for "
            "retrieval. Ask the user to upload a document or select a "
            "knowledge base in the chat settings."
        )

    from core.rag import retrieval
    from core.rag.vector_store import kb_scope, thread_scope
    from storage.studio_db import get_connection

    scope = kb_scope(scope_kb_id) if scope_kb_id else thread_scope(scope_thread_id)
    k = top_k if top_k is not None else default_top_k

    if enable_rerank:
        from utils.rag.config import RAG_RERANK_CANDIDATE_K

        candidate_k = max(k, RAG_RERANK_CANDIDATE_K)
    else:
        candidate_k = k

    # Resolve the scope's embedder (the one that populated its
    # vectors). Inline import to avoid a backend-route → core
    # dependency cycle at module load.
    from routes.rag import _resolve_scope_embedder

    scope_embedder = _resolve_scope_embedder(scope)

    logger.info(
        "search_knowledge_base: scope=%s embedder=%s top_k=%d min_score=%.3f rerank=%s query=%r",
        scope,
        scope_embedder or "<default>",
        k,
        min_score,
        enable_rerank,
        query[:120],
    )

    try:
        hits = retrieval.retrieve_hybrid(
            scope,
            query.strip(),
            k = candidate_k,
            embedder_model = scope_embedder,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_knowledge_base retrieval failed")
        return f"Error: retrieval failed ({type(exc).__name__})."

    retrieved_count = len(hits)
    if min_score > 0.0:
        hits = retrieval.filter_by_min_score(hits, min_score)
        logger.info(
            "search_knowledge_base: retrieved=%d met_threshold=%d (min_score=%.3f)",
            retrieved_count,
            len(hits),
            min_score,
        )
    else:
        logger.info(
            "search_knowledge_base: retrieved=%d (no threshold)", retrieved_count
        )

    chunk_ids = [h.chunk_id for h in hits]
    lookup: dict[str, dict] = {}
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.text, c.page_number,
                       c.kind, d.filename
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        for row in rows:
            lookup[row["chunk_id"]] = dict(row)

    if enable_rerank and hits:
        from core.rag import reranker

        pairs = [
            (hit, lookup[hit.chunk_id]["text"])
            for hit in hits
            if hit.chunk_id in lookup
        ]
        try:
            hits = reranker.rerank(
                query.strip(),
                pairs,
                model_name = reranker_model,
                top_k = k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rerank failed in search_knowledge_base: %s", exc)
            hits = hits[:k]
    else:
        hits = hits[:k]

    # Image-kind hits don't carry LLM-friendly text — skip them. The
    # paired caption (linked_chunk_id) usually surfaces separately.
    formatted = [
        lookup[hit.chunk_id]
        for hit in hits
        if hit.chunk_id in lookup and lookup[hit.chunk_id].get("kind") != "image"
    ]
    return _format_hits_for_llm(formatted)
