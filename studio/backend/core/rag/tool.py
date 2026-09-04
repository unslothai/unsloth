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
    """KB (an explicit pick) is exclusive; project and thread scopes combine so a
    project chat also retrieves from its own attached documents.

    The conversation archive is exclusive too, and takes precedence: it is a different
    corpus (this chat's evicted turns), so sharing a top-K would have old turns and
    document passages crowd each other out."""
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
    """`_format`, plus what a recalled conversation needs and a knowledge base does not.

    Two additions, both presentation only:

    * each block carries ``turn``, the position of that turn in the conversation, so the
      passages can be told apart in time. Omitted where the archive predates the column,
      because a missing ordinal is not a position of zero.
    * a one-line header, emitted only when there are at least two passages, stating that
      they are oldest first and that a later turn supersedes an earlier one. It says
      SUPERSEDES rather than "is the answer", so a question about what was originally
      said still reads the first block as the original. With a single passage the header
      would be an ordering claim about nothing, and it would spend tokens on the rung the
      over-budget backoff falls to when there is least room.
    """
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
                # createdAt is the tie-breaker _conversation_order needs: pre-ordinal rows have turn None and
                # ordinals are not UNIQUE.
                "chunkIndex": _row_value(r, "chunk_index"),
                "createdAt": _row_value(r, "created_at"),
                "score": round(float(h.score), 4) if h.score is not None else None,
            }
        )
    body = "\n\n".join(blocks)
    if len(hits) >= 2:
        body = f"{CONVERSATION_RECALL_HEADER}\n\n{body}"
    return body, sources


def render_conversation_sources(sources: list[dict]) -> str:
    """`render_sources` for recalled conversation: keeps ``turn`` and the header.

    Used when two searches are merged into one block, where the sources are already built
    and there are no rows left to read them from.
    """
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
    """Render a citation-source list to sequentially-numbered ``<chunk>`` blocks,
    rewriting each source's ``citationId`` to match its 1-based position. Lets
    independently-built source lists (a whole-document thread attachment plus
    retrieved project passages) be merged under one citation numbering."""
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
    """Hydrate chunk rows for a list of hits."""
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
    """Forced-retrieval variant for auto-injection.

    Returns ``(rendered_text, sources)`` only if some hit's cosine clears
    ``min_dense_score``, else ``None`` (inject nothing). ``None`` keeps the
    retrieved top-K without the optional-auto relevance gate. In ``lexical``
    mode gated hits fall back to a dense 1-NN probe.
    """
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


def whole_document_context(
    *, scope_thread_id: str | None = None, max_tokens: int
) -> tuple[str, list[dict]] | None:
    """Render EVERY chunk of the THREAD's attached documents (in order) as the same
    ``<chunk>`` blocks + citation source-map as retrieval, so the model reads the whole
    file rather than top-K passages. Thread-attached files only: KB and project corpora
    are search corpora, never whole-document, so this resolves the thread scope alone.
    ``None`` (caller falls back to retrieval) when there is no thread scope, no completed
    chunks, or the total exceeds ``max_tokens``."""
    if not scope_thread_id:
        return None
    # A non-positive budget means "never inject" (disable whole-doc via RAG_THREAD_WHOLE_DOC=0), not
    # "inject the whole corpus unbounded".
    if max_tokens <= 0:
        return None
    scope = thread_scope(scope_thread_id)
    conn = rag_db.get_connection()
    try:
        # Cheap SUM pre-check so an oversized attachment is rejected before the whole corpus is hydrated;
        # all_chunks_for_scope runs only once it fits.
        if scope_token_estimate(conn, scope) > max_tokens:
            return None
        rows = all_chunks_for_scope(conn, scope)
    finally:
        conn.close()
    if not rows:
        return None
    total = sum(_row_token_count(r) for r in rows)
    if total > max_tokens:
        return None

    sources: list[dict] = [
        {
            "citationId": i,
            "chunkId": r["id"],
            "documentId": r["document_id"],
            "filename": r["filename"] or "unknown",
            "page": r["page_number"],
            "text": r["text"] or "",
            "score": None,
        }
        for i, r in enumerate(rows, 1)
    ]
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
    """Text-only variant of :func:`search_knowledge_base_with_sources`."""
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
