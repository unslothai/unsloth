# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``search_knowledge_base`` LLM tool: scope resolution + hit formatting.

KB scope wins; otherwise project and thread scopes combine so project chats also
see their own attachments. Hits render as ``<chunk>`` blocks for the model,
plus a parallel citation source-map for clickable sources. Each call opens and
closes its own ``rag_db`` connection.
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from storage import rag_db

from . import config, retrieval
from .store import (
    all_chunks_for_scope,
    conversation_archive_scope,
    kb_scope,
    project_scope,
    scope_token_estimate,
    thread_scope,
)

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's uploaded documents and knowledge bases for relevant passages."
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


def _resolve_scope(
    scope_kb_id: str | None,
    scope_thread_id: str | None,
    scope_project_id: str | None = None,
    scope_conversation_id: str | None = None,
) -> str | list[str] | None:
    """The archive is exclusive and wins over KB; project and thread scopes combine."""
    if scope_conversation_id:
        return conversation_archive_scope(scope_conversation_id)
    if scope_kb_id:
        return kb_scope(scope_kb_id)
    scopes = []
    if scope_project_id:
        scopes.append(project_scope(scope_project_id))
    if scope_thread_id:
        scopes.append(thread_scope(scope_thread_id))
    if not scopes:
        return None
    return scopes[0] if len(scopes) == 1 else scopes


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


CONVERSATION_RECALL_HEADER = (
    "These are earlier turns of THIS conversation, quoted verbatim and listed oldest "
    "first. The turn number is each one's position in the conversation; they are not "
    "consecutive. Where two turns state different things about the same subject, the one "
    "with the HIGHER turn number was said later and supersedes the earlier one."
)


def format_conversation_recall(rows, hits) -> tuple[str, list[dict]]:
    """Blocks carry their turn; with 2+ passages a header says a later turn supersedes an earlier one."""
    if not hits:
        return "No matching turns were found in this conversation.", []
    blocks: list[str] = []
    sources: list[dict] = []
    for i, h in enumerate(hits, 1):
        r = rows.get(h.chunk_id)
        filename = (r["filename"] if r else None) or "unknown"
        text = r["text"] if r else ""
        ordinal = _row_value(r, "archive_ordinal")
        turn_attr = f" turn={quoteattr(str(int(ordinal) + 1))}" if ordinal is not None else ""
        blocks.append(f'<chunk id="{i}" source={quoteattr(filename)}{turn_attr}>\n{text}\n</chunk>')
        sources.append(
            {
                "citationId": i,
                "chunkId": h.chunk_id,
                "documentId": r["document_id"] if r else None,
                "filename": filename,
                "page": None,
                "text": text,
                "turn": int(ordinal) + 1 if ordinal is not None else None,
                "chunkIndex": _row_value(r, "chunk_index"),
                "createdAt": _row_value(r, "created_at"),
                "documentRowid": _row_value(r, "document_rowid"),
                "score": round(float(h.score), 4) if h.score is not None else None,
            }
        )
    body = "\n\n".join(blocks)
    if len(hits) >= 2:
        body = f"{CONVERSATION_RECALL_HEADER}\n\n{body}"
    return body, sources


def render_conversation_sources(sources: list[dict]) -> str:
    """Renders already-built recalled sources, keeping turn and header; no rows remain to rebuild them."""
    blocks: list[str] = []
    for i, s in enumerate(sources, 1):
        s["citationId"] = i
        turn = s.get("turn")
        turn_attr = f" turn={quoteattr(str(turn))}" if turn is not None else ""
        blocks.append(
            f'<chunk id="{i}" source={quoteattr(s.get("filename") or "unknown")}'
            f'{turn_attr}>\n{s.get("text") or ""}\n</chunk>'
        )
    body = "\n\n".join(blocks)
    return f"{CONVERSATION_RECALL_HEADER}\n\n{body}" if len(sources) >= 2 else body


def _row_value(row, key: str):
    """A column that may not exist on an older row object, without raising."""
    if row is None:
        return None
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def render_sources(sources: list[dict]) -> str:
    """Renumbers citationId by position so separately built source lists merge under one numbering."""
    blocks: list[str] = []
    for i, s in enumerate(sources, 1):
        s["citationId"] = i
        src = quoteattr(s.get("filename") or "unknown")
        page = s.get("page")
        page_attr = f" page={quoteattr(str(page))}" if page else ""
        blocks.append(f'<chunk id="{i}" source={src}{page_attr}>\n{s.get("text") or ""}\n</chunk>')
    return "\n\n".join(blocks)


def _row_token_count(row) -> int:
    """Chunk token count for budgeting, falling back to a length estimate when the
    stored count is missing or zero, so a malformed chunk cannot bypass the budget."""
    tc = row["token_count"]
    if tc:
        return int(tc)
    return max(1, len(row["text"] or "") // 4)


def _whole_document_token_count(row) -> int:
    token_count = _row_value(row, "whole_document_token_count")
    return max(0, int(token_count)) if token_count is not None else _row_token_count(row)


def search_knowledge_base_with_sources(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    scope_project_id: str | None = None,
    scope_conversation_id: str | None = None,
    top_k: int | None = None,
    min_score: float = 0.0,
    model_name: str | None = None,
    mode: str = "hybrid",
) -> tuple[str, list[dict]]:
    """Search -> ``(rendered_text, citation_sources)``; each source aligns with a
    rendered ``<chunk>`` block's ``id``."""
    if not query or not query.strip():
        return "Error: query is empty.", []
    scope = _resolve_scope(scope_kb_id, scope_thread_id, scope_project_id, scope_conversation_id)
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
        )
        hits = retrieval.filter_min_score(hits, min_score)
        rows = store_rows(conn, hits)
    finally:
        conn.close()
    return _format(rows, hits)


def store_rows(conn, hits):
    from . import store
    return store.chunks_by_id(conn, [h.chunk_id for h in hits])


def search_for_autoinject(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    scope_project_id: str | None = None,
    top_k: int | None = None,
    min_dense_score: float | None = 0.70,
    model_name: str | None = None,
    mode: str = "hybrid",
) -> tuple[str, list[dict]] | None:
    """Returns None, meaning inject nothing, unless some hit's cosine clears min_dense_score."""
    if not query or not query.strip():
        return None
    scope = _resolve_scope(scope_kb_id, scope_thread_id, scope_project_id)
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
        )
        strong = (
            hits[:k]
            if min_dense_score is None
            else [
                h for h in hits if h.dense_score is not None and h.dense_score >= min_dense_score
            ][:k]
        )
        if min_dense_score is not None and not strong and hits and mode == "lexical":
            probe = retrieval.retrieve_dense(conn, scope, query, 1, model_name = model_name)
            if (
                probe
                and probe[0].dense_score is not None
                and (probe[0].dense_score >= min_dense_score)
            ):
                strong = hits[:k]
        if not strong:
            return None
        rows = store_rows(conn, strong)
    finally:
        conn.close()
    text, sources = _format(rows, strong)
    return (text, sources) if sources else None


def _drop_chunk_overlap(text: str, start: int | None, prev_end: int | None) -> str:
    if not isinstance(start, int) or not isinstance(prev_end, int) or start >= prev_end:
        return text
    return text[min(prev_end - start, len(text)) :].lstrip("\r\n")


def whole_document_context(
    *, scope_thread_id: str | None = None, max_tokens: int
) -> tuple[str, list[dict]] | None:
    """render ordered thread attachment chunks with citation blocks; KB and project corpora stay search-only; return None when absent or over max_tokens."""
    if not scope_thread_id:
        return None
    # a non-positive budget disables whole-document injection rather than removing the limit
    if max_tokens <= 0:
        return None
    scope = thread_scope(scope_thread_id)
    conn = rag_db.get_connection()
    try:
        # reject oversized attachments before all_chunks_for_scope hydrates the corpus
        if scope_token_estimate(conn, scope) > max_tokens:
            return None
        rows = all_chunks_for_scope(conn, scope)
    finally:
        conn.close()
    if not rows:
        return None
    total = sum(_whole_document_token_count(r) for r in rows)
    if total > max_tokens:
        return None

    sources: list[dict] = []
    prev_page, prev_end = None, None
    for i, r in enumerate(rows, 1):
        page = (r["document_id"], r["source_page_index"])
        text = r["text"] or ""
        trimmed = (
            _drop_chunk_overlap(text, r["page_char_start"], prev_end) if page == prev_page else text
        )
        sources.append(
            {
                "citationId": i,
                "chunkId": r["id"],
                "documentId": r["document_id"],
                "filename": r["filename"] or "unknown",
                "page": r["page_number"],
                "text": trimmed,
                "score": None,
            }
        )
        prev_page, prev_end = page, r["page_char_end"]
    rendered = render_sources(sources)
    if max(1, len(rendered) // 4) > max_tokens:
        return None
    return rendered, sources


def search_knowledge_base(
    *,
    query: str,
    scope_kb_id: str | None = None,
    scope_thread_id: str | None = None,
    scope_project_id: str | None = None,
    top_k: int | None = None,
    min_score: float = 0.0,
    model_name: str | None = None,
) -> str:
    text, _sources = search_knowledge_base_with_sources(
        query = query,
        scope_kb_id = scope_kb_id,
        scope_thread_id = scope_thread_id,
        scope_project_id = scope_project_id,
        top_k = top_k,
        min_score = min_score,
        model_name = model_name,
    )
    return text
