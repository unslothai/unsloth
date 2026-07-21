# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ephemeral web-RAG for deep research auto-read.

Deep research auto-reads the top search results so synthesis is grounded in page text rather
than short snippets. Whole pages make a small local model loop on boilerplate, so the scraped
pages go through the *same* retrieval pipeline the knowledge base uses and only the most
relevant passages are folded into the evidence.

Nothing here re-implements chunking, embedding, retrieval, ranking, or rendering; it wires
Studio's existing KB components (``chunk_pages``, ``embeddings.encode``, ``store.add_chunks``,
``retrieval.retrieve_hybrid``, ``retrieval.filter_min_score``, ``tool._format``) to the live
scrape. The only difference from a persisted KB is the corpus: pages are ingested under a
unique throwaway scope deleted in a ``finally`` block, so an auto-read never pollutes a user's
knowledge base, exactly like Studio's per-thread attachment RAG on the same store.
"""

from __future__ import annotations

import hashlib
import uuid

from loggers import get_logger
from storage import rag_db

from . import config, embeddings, retrieval, store, tool
from .chunking import chunk_pages
from .parsers import Page

logger = get_logger(__name__)


def _fit_to_budget(hits, rows, char_budget):
    """Keep the best (already ranked) hits whose cumulative chunk text fits ``char_budget``,
    always keeping at least the top hit so a single long passage is not dropped whole."""
    if char_budget is None:
        return hits
    kept = []
    used = 0
    for hit in hits:
        row = rows.get(hit.chunk_id)
        text = (row["text"] if row else "") or ""
        if kept and used + len(text) > char_budget:
            break
        kept.append(hit)
        used += len(text)
    return kept


def retrieve_web_chunks(
    pages: list[dict],
    query: str,
    *,
    top_n: int,
    min_score: float,
    char_budget: int | None = None,
    max_tokens: int | None = None,
    overlap: int | None = None,
    model_name: str | None = None,
) -> tuple[str, list[dict]]:
    """Ingest scraped pages into an ephemeral RAG scope, hybrid-retrieve the passages most
    relevant to ``query``, and return ``(rendered_chunks, sources)`` using Studio's KB
    formatter.

    ``pages`` is a list of dicts with ``text`` (required) and optional ``title`` / ``url``
    (``title`` becomes the ``<chunk source>``). Returns ``("", [])`` when there is nothing
    usable or RAG is unavailable, so the caller can fall back to snippet evidence. The scope
    is always deleted before returning, so nothing is left in the store."""
    query = (query or "").strip()
    if not query or top_n <= 0 or not pages or not rag_db.RAG_AVAILABLE:
        return "", []
    model = model_name or config.effective_embedding_model()
    max_tokens = max_tokens or config.CHUNK_TOKENS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    count = embeddings.token_counter(model)

    conn = rag_db.get_connection()
    scope = f"research_scrape_{uuid.uuid4().hex}"
    doc_ids: list[str] = []
    try:
        for page in pages:
            text = str(page.get("text") or "").strip()
            if not text:
                continue
            source = str(page.get("title") or page.get("url") or "web").strip() or "web"
            chunks = chunk_pages(
                [Page(text = text, page_number = None, char_count = len(text))],
                max_tokens = max_tokens,
                overlap = overlap,
                count = count,
            )
            if not chunks:
                continue
            vectors = embeddings.encode(
                [chunk.text for chunk in chunks], model_name = model, normalize = True
            )
            doc_id = store.create_document(
                conn,
                scope = scope,
                filename = source,
                sha256 = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest(),
                status = "ready",
                embedding_model = model,
            )
            doc_ids.append(doc_id)
            store.add_chunks(conn, scope, doc_id, chunks, vectors)

        if not doc_ids:
            return "", []
        hits = retrieval.retrieve_hybrid(
            conn, scope, query, k = top_n, model_name = model, mode = "hybrid"
        )
        hits = retrieval.filter_min_score(hits, min_score)
        if not hits:
            return "", []
        rows = store.chunks_by_id(conn, [hit.chunk_id for hit in hits])
        hits = _fit_to_budget(hits, rows, char_budget)
        return tool._format(rows, hits)
    except Exception:
        logger.warning("research.web_rank_failed", exc_info = True)
        return "", []
    finally:
        for doc_id in doc_ids:
            try:
                store.delete_document(conn, doc_id)
            except Exception:
                logger.warning("research.web_rank_cleanup_failed doc_id=%s", doc_id)
        conn.close()
