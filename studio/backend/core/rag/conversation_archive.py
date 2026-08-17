# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the turns the rolling context window evicts, and hand them back on request.

The rolling window drops complete oldest turns so a long chat keeps working. Without
somewhere to put them, an evicted turn is gone for the rest of the session and the model
will cheerfully claim the conversation started wherever its visible context starts.

This archives each evicted turn into a per-thread RAG scope, reusing the store, chunker,
embedder and hybrid retrieval that back attached documents. It never touches the stored
transcript; the archive is a search index over turns the projection can no longer carry.

Two properties are load-bearing and easy to break:

* The archive is CUMULATIVE. Every compaction adds to it and nothing is ever cleared or
  re-scoped per compaction, so the fifth compaction can still find what the first one
  evicted. Benchmarked on MRCR v2 over 11 compaction events: cumulative scored 0.450
  against 0.058 for an archive holding only the latest compaction's evictions. "Clear it
  each time" is the obvious-looking simplification that reduces the feature to nothing.
* Nothing here may break a chat. Every entry point returns a no-op on failure, because
  the alternative to a degraded archive is a working conversation, not an error.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from storage import rag_db

from . import config, embeddings, retrieval, store, tool
from .chunking import chunk_pages
from .parsers import Page

logger = logging.getLogger(__name__)

# Roles whose content is instructions rather than conversation. Archiving a system prompt
# would let it come back as a quoted "earlier turn" and read as user-authored text.
_SKIP_ROLES = frozenset({"system", "developer"})

# Prefixes of the tool call ids this feature and the RAG auto-inject generate. Their
# results are retrieved passages, not things anyone said, so archiving them would feed
# retrieved text back into the index it came from.
_INJECTED_CALL_PREFIXES = ("rag_auto_", "conv_recall_")

_MAX_TOOL_RESULT_CHARS = 4000


def _text_of(content) -> str:
    """Flatten OpenAI message content to plain text, dropping non-text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text") or "text" in part:
                parts.append(str(part.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _is_injected(message: dict) -> bool:
    call_id = str(message.get("tool_call_id") or "")
    if call_id.startswith(_INJECTED_CALL_PREFIXES):
        return True
    for call in message.get("tool_calls") or []:
        if str((call or {}).get("id") or "").startswith(_INJECTED_CALL_PREFIXES):
            return True
    return False


def render_turn(group: list[dict]) -> str:
    """Render one evicted turn group as the text that gets indexed and quoted back.

    Both sides are kept in a single document on purpose: retrieval then returns the
    question together with its answer, which is what makes a recalled turn usable rather
    than a floating fragment.
    """
    lines: list[str] = []
    for message in group:
        role = str(message.get("role") or "")
        if role in _SKIP_ROLES:
            continue
        text = _text_of(message.get("content")).strip()
        calls = message.get("tool_calls") or []
        if calls:
            names = ", ".join(
                str((call or {}).get("function", {}).get("name") or "tool") for call in calls
            )
            lines.append(f"assistant called {names}")
            continue
        if not text:
            continue
        if role == "tool":
            # A tool result can be enormous and is the least useful thing to quote back
            # verbatim; keep enough to be searchable without bloating the index.
            if len(text) > _MAX_TOOL_RESULT_CHARS:
                text = text[:_MAX_TOOL_RESULT_CHARS] + " ..."
            lines.append(f"tool result: {text}")
        else:
            lines.append(f"{role or 'message'}: {text}")
    return "\n".join(lines).strip()


def _archivable(group: list[dict]) -> bool:
    if any(str(message.get("role") or "") in _SKIP_ROLES for message in group):
        return False
    return not any(_is_injected(message) for message in group)


def enabled() -> bool:
    return bool(config.CONVERSATION_ARCHIVE) and bool(rag_db.RAG_AVAILABLE)


def archive_turns(thread_id: str, evicted: list[dict]) -> int:
    """Index the evicted turns for ``thread_id``. Returns how many were newly written.

    Idempotent by content hash: the same turns are evicted again on every later request
    in the session, so re-archiving has to be free. After the first write each repeat
    costs one indexed SELECT and writes nothing.
    """
    if not thread_id or not evicted or not enabled():
        return 0

    # Lazy, and pointed at the evictor's own grouper on purpose: an archived unit has to
    # be exactly the unit the rolling window drops. Imported here rather than at module
    # scope because the inference layer imports this module.
    from core.inference.context_window import group_turns

    groups = [group for group in group_turns(evicted) if _archivable(group)]
    if not groups:
        return 0

    model = config.effective_embedding_model()
    scope = store.conversation_archive_scope(thread_id)
    written = 0
    conn = None
    try:
        count = embeddings.token_counter(model)
        conn = rag_db.get_connection()
        for group in groups:
            text = render_turn(group)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            if store.document_by_hash(conn, scope, digest):
                continue
            chunks = chunk_pages(
                [Page(text = text, page_number = None, char_count = len(text))],
                max_tokens = config.CHUNK_TOKENS,
                overlap = config.CHUNK_OVERLAP,
                count = count,
            )
            if not chunks:
                continue
            # The identity must come from the encode that produced these vectors: a
            # concurrent embedder swap would otherwise label them with a space they were
            # never in, which the query side then filters against.
            vectors, identity = embeddings.encode_with_identity(
                [chunk.text for chunk in chunks], model_name = model, normalize = True
            )
            roles = " + ".join(
                dict.fromkeys(str(message.get("role") or "message") for message in group)
            )
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
            )
            store.add_chunks(conn, scope, document_id, chunks, vectors)
            written += 1
    except Exception:
        # A chat that cannot archive is worse off than one that can, but far better off
        # than one that raises. Whatever was written before the failure stays searchable.
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return written


def has_archive(thread_id: str) -> bool:
    """Whether anything has ever been archived for this thread."""
    if not thread_id or not enabled():
        return False
    conn = None
    try:
        conn = rag_db.get_connection()
        row = conn.execute(
            "SELECT 1 FROM documents WHERE scope=? LIMIT 1",
            (store.conversation_archive_scope(thread_id),),
        ).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def recall(
    thread_id: str,
    query: str,
    *,
    top_k: Optional[int] = None,
) -> Optional[tuple[str, list[dict]]]:
    """Most relevant archived turns for ``query``, rendered like any other RAG hit.

    No relevance floor, unlike ``tool.search_for_autoinject``. That 0.70 cosine gate
    exists to keep off-topic document passages out of answers; here the passages ARE this
    conversation, and the alternative to a weak match is a model answering with no memory
    at all. It also means lexical-only hits survive, since ``filter_min_score`` only gates
    hits carrying a dense score -- which is what makes the degradation path below work.
    """
    query = (query or "").strip()
    if not thread_id or not query or not enabled():
        return None

    scope = store.conversation_archive_scope(thread_id)
    limit = top_k or config.CONVERSATION_ARCHIVE_TOP_K
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        try:
            hits = retrieval.retrieve_hybrid(
                conn, scope, query, k = limit, model_name = model, mode = "hybrid"
            )
        except Exception:
            # Dense retrieval raises rather than degrading when no embedder can start.
            # The FTS rows written by earlier successful compactions still answer, so
            # fall back rather than losing the archive entirely.
            logger.warning(
                "conversation_archive.dense_unavailable thread_id=%s", thread_id, exc_info = True
            )
            hits = retrieval.retrieve_hybrid(
                conn, scope, query, k = limit, model_name = model, mode = "lexical"
            )
        if not hits:
            return None
        rows = store.chunks_by_id(conn, [hit.chunk_id for hit in hits])
        text, sources = tool._format(rows, hits)
        return (text, sources) if sources else None
    except Exception:
        logger.warning("conversation_archive.recall_failed thread_id=%s", thread_id, exc_info = True)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def delete_for_thread(thread_id: str) -> int:
    """Drop a thread's archive. Called when the thread itself is deleted."""
    if not thread_id or not rag_db.RAG_AVAILABLE:
        return 0
    conn = None
    removed = 0
    try:
        conn = rag_db.get_connection()
        scope = store.conversation_archive_scope(thread_id)
        for row in conn.execute("SELECT id FROM documents WHERE scope=?", (scope,)).fetchall():
            store.delete_document(conn, row["id"])
            removed += 1
    except Exception:
        logger.warning("conversation_archive.delete_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return removed
