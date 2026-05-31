# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`search_knowledge_base` tool — RAG retrieval surfaced to the LLM.

Scope comes from the request body (`rag_scope`), not from the tool args,
so the model never sees KB UUIDs.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from loggers import get_logger

logger = get_logger(__name__)

# Per-request chunk-id counter. Task-local (FastAPI runs each request in its own
# asyncio task → its own Context). Keeps citation ids unique across multiple
# search_knowledge_base calls in a turn: call 1 → 1..N, call 2 → N+1..N+M, etc.
_chunk_id_counter: ContextVar[int] = ContextVar("rag_chunk_id_counter", default = 0)


SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "ALWAYS CALL THIS TOOL FIRST before answering any user question. "
            "It searches the user's attached documents and returns the chunks "
            "you must ground your reply in. Do not answer from your own "
            "knowledge until you have called this tool with a focused query "
            "derived from the user's latest message. Returns chunks wrapped in "
            '<chunk id="N" source="..." page="...">...</chunk> '
            "tags. CITE each chunk you use with its LITERAL id attribute, "
            'e.g. `<chunk id="7">` is cited as `[7]`. IDs are unique across '
            "all calls in this turn — never renumber, never reuse."
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


def _xml_attr(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_hits_for_llm(hits: list[dict], start_id: int = 0) -> str:
    """Render hits as fenced <chunk> blocks with metadata.

    ``start_id`` offsets the citation id so multiple calls in the same
    request produce globally unique ids (call 1: 1..N, call 2: N+1..N+M).
    """
    if not hits:
        return (
            "No matching chunks were found in the attached documents. "
            "Either nothing in this scope is relevant, or no documents "
            "have been ingested yet."
        )
    blocks: list[str] = []
    for index, hit in enumerate(hits, start = start_id + 1):
        attrs = [
            f'id="{index}"',
            f'source="{_xml_attr(hit.get("filename") or "unknown")}"',
        ]
        # Durable backend ids — additive per contracts.md §3.1 (T3). ``id`` stays
        # the visible citation (model's `[N]`); ``document_id`` + ``chunk_id`` feed
        # the preview route. Old XML without these still parses (hover-only) per §3.2.
        document_id = hit.get("document_id")
        if document_id:
            attrs.append(f'document_id="{_xml_attr(document_id)}"')
        backend_chunk_id = hit.get("chunk_id")
        if backend_chunk_id:
            attrs.append(f'chunk_id="{_xml_attr(backend_chunk_id)}"')
        page = hit.get("page_number")
        if page is not None:
            attrs.append(f'page="{page}"')
        chunk_index = hit.get("chunk_index")
        if chunk_index is not None:
            attrs.append(f'chunk_index="{chunk_index}"')
        for attr_name in (
            "source_page_index",
            "page_char_start",
            "page_char_end",
            "line_start",
            "line_end",
        ):
            value = hit.get(attr_name)
            if value is not None:
                attrs.append(f'{attr_name}="{value}"')
        tokens = hit.get("token_count")
        if tokens:
            attrs.append(f'tokens="{tokens}"')
        kind = hit.get("kind")
        if kind and kind != "text":
            attrs.append(f'kind="{_xml_attr(kind)}"')
        image_path = hit.get("image_path")
        if kind == "image" and image_path and document_id:
            # Mirror routes/rag.py search-response shape so the frontend tool card
            # can render the image inline via the same route.
            image_url = f"/api/rag/images/{document_id}/{Path(image_path).name}"
            attrs.append(f'image_url="{_xml_attr(image_url)}"')
        text = (hit.get("text") or "").strip()
        blocks.append(f"<chunk {' '.join(attrs)}>\n{text}\n</chunk>")
    return "\n\n".join(blocks)


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
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
) -> str:
    """Run RAG and return a tool-result string. kb_id takes precedence over thread_id."""
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
    from storage.studio_db import closing_connection

    scope = kb_scope(scope_kb_id) if scope_kb_id else thread_scope(scope_thread_id)
    k = top_k if top_k is not None else default_top_k

    if enable_rerank:
        from utils.rag.config import RAG_RERANK_CANDIDATE_K

        candidate_k = max(k, RAG_RERANK_CANDIDATE_K)
    else:
        candidate_k = k

    from core.rag.scope import resolve_scope_embedder

    scope_embedder = resolve_scope_embedder(scope)

    logger.info(
        "search_knowledge_base: scope=%s embedder=%s mode=%s top_k=%d min_score=%.3f rerank=%s query=%r",
        scope,
        scope_embedder or "<default>",
        mode,
        k,
        min_score,
        enable_rerank,
        query[:120],
    )

    try:
        if mode == "bm25":
            hits = retrieval.retrieve_bm25(scope, query.strip(), candidate_k)
        elif mode == "dense":
            hits = retrieval.retrieve_dense(
                scope,
                query.strip(),
                candidate_k,
                embedder_model = scope_embedder,
            )
        else:
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
        with closing_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.text, c.page_number,
                       c.token_count, c.kind, c.image_path,
                       c.source_page_index, c.page_char_start,
                       c.page_char_end, c.line_start, c.line_end,
                       c.document_id, d.filename
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

    # Merge Hit metadata (score, dense_score, chunk_index) into the sqlite row so
    # the formatter sees one flat dict per chunk. Image-kind hits flow through so
    # the multimodal match reaches the LLM; their image_url lets the UI render it.
    formatted: list[dict] = []
    for hit in hits:
        row = lookup.get(hit.chunk_id)
        if row is None:
            continue
        formatted.append(
            {
                **row,
                "score": hit.score,
                "dense_score": hit.dense_score,
                "chunk_index": hit.chunk_index,
            }
        )
    start_id = _chunk_id_counter.get()
    rendered = _format_hits_for_llm(formatted, start_id = start_id)
    _chunk_id_counter.set(start_id + len(formatted))
    return rendered
