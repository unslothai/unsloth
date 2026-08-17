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
import re
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
# Tool arguments are usually short (a path, a command, a query). The cap is only
# there so a pathological blob cannot dominate the archived turn.
_MAX_TOOL_ARGS_CHARS = 1000
# Retrieval fetches this multiple of the requested number before the live-branch
# filter runs. Filtering AFTER a k-sized fetch means stale turns from an abandoned
# branch can occupy the whole result and starve live matches sitting just below,
# so recall returns nothing while the answer is in the archive.
_BRANCH_FILTER_OVERFETCH = 4


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
            # The arguments, bounded, not just the name. The searchable substance of a
            # tool turn is the command, query, path or snippet that was run -- "assistant
            # called terminal" cannot answer "what did you run earlier?", which is exactly
            # what this archive exists to answer. Any assistant text on the same message
            # is kept for the same reason.
            for call in calls:
                function = (call or {}).get("function") or {}
                name = str(function.get("name") or "tool")
                arguments = str(function.get("arguments") or "").strip()
                if len(arguments) > _MAX_TOOL_ARGS_CHARS:
                    arguments = arguments[:_MAX_TOOL_ARGS_CHARS] + " ..."
                lines.append(
                    f"assistant called {name}: {arguments}"
                    if arguments
                    else f"assistant called {name}"
                )
            if text:
                lines.append(f"{role or 'assistant'}: {text}")
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
    """Whether the archive can actually run here.

    ``rag_available()`` and not ``RAG_AVAILABLE``: the flag only records that
    ``import sqlite_vec`` worked, while the vec0 native library it loads is a separate
    file a venv can be missing (the common macOS case). Trusting the flag there is worse
    than having no feature at all -- the fit holds a recall reserve back, so every
    overflowing prompt evicts extra history, and then both the archive write and the
    recall fail at ``get_connection()``, so the user pays for content they never get.
    """
    return bool(config.CONVERSATION_ARCHIVE) and bool(rag_db.rag_available())


def archive_turns(thread_id: str, evicted: list[dict]) -> int:
    """Index the evicted turns for ``thread_id``. Returns how many were newly written.

    Idempotent by content hash: the same turns are evicted again on every later request
    in the session, so re-archiving has to be free. After the first write each repeat
    costs one indexed SELECT and writes nothing.
    """
    if not thread_id or not evicted or not enabled():
        return 0
    # A temporary (incognito) chat is never written to studio.db and is documented to
    # vanish on reload, but the frontend still sends its thread_id and the request model
    # carries no incognito flag, so nothing here would otherwise tell the two apart.
    # Archiving one would persist exactly the content the user asked not to keep, into a
    # scope with no thread row -- so no deletion flow could ever reach it either.
    #
    # A thread with saved messages is the signal, and by the time a chat is long enough
    # to compact its earlier turns are always persisted. An API client that sends a
    # thread_id without persisting anything is excluded for the same reason: its archive
    # would be equally unreachable by any delete.
    if not _live_transcript(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
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
            # commit=False, then commit once the chunks are in. Committing the
            # document first leaves an empty row marked "completed" if the chunk or
            # vector write fails, and `document_by_hash` then skips that turn on every
            # later compaction: the turn is silently unarchivable forever, and it looks
            # finished, so nothing retries it.
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, vectors)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
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


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _live_transcript(thread_id: str) -> Optional[str]:
    """The thread's saved messages as one normalised blob, or None if it has none.

    Used to keep recall on the branch the user is actually on. Editing an earlier message
    rewinds a thread and continues down a new branch, but the archive is append-only and
    still holds the turns the abandoned continuation produced -- so without this, asking
    the right question after a rewind can pull back a turn that, on this branch, never
    happened. Verified: after rewinding past a turn, querying its distinctive text still
    returned it.

    None means "this thread has no saved transcript" -- an API client passing a thread_id
    without persisting messages -- and the caller then does not filter, because an empty
    transcript is absence of evidence, not evidence the turns are gone.
    """
    try:
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    parts = [_text_of(message.get("content")) for message in messages]
    blob = _normalise("\n".join(part for part in parts if part))
    return blob or None


_ROLE_PREFIX = re.compile(
    r"^(?:user|assistant|system|developer|tool result|message):\s*", re.IGNORECASE
)


def _on_live_branch(text: str, transcript: str) -> bool:
    """Whether an archived turn still exists in the saved thread.

    Substring containment on a normalised prefix rather than a digest match: the archived
    text was rendered from the inference projection and the saved copy comes back through
    the message store, so exact equality is too brittle to bet the feature on.

    The role labels ``render_turn`` writes ("user: ...") exist only in the archived copy,
    so they are stripped first -- leaving them in made every probe miss and filtered out
    the turns this feature exists to return.
    """
    probes = [_normalise(_ROLE_PREFIX.sub("", line))[:160] for line in (text or "").splitlines()]
    probes = [probe for probe in probes if probe]
    if not probes:
        return False
    # EVERY line, not the first one that matches. A turn is archived as a unit, so
    # editing only the assistant half leaves the user line matching and would keep
    # serving the answer that no longer exists. Requiring all of them means an edit to
    # any part of a turn retires the whole archived copy, which is what "this turn is
    # still on the branch" has to mean.
    return all(probe in transcript for probe in probes)


def recall(
    thread_id: str,
    query: str,
    *,
    top_k: Optional[int] = None,
) -> Optional[tuple[str, list[dict]]]:
    """Most relevant archived turns for ``query``, rendered like any other RAG hit.

    LEXICAL FIRST, then hybrid for the rest of the budget. Recalling your own conversation
    is mostly an exact-match problem -- a name, a number, an identifier, a code someone
    pasted twenty turns ago -- and those live or die on rare-token matching. Measured on a
    real 30-turn document walkthrough where every turn shared the same boilerplate wrapper:
    the one chunk holding the needle ranked 3rd lexically at any k, was never returned by
    dense retrieval at all (near-identical text embeds near-identically, and a rare token
    barely moves a 384-dim vector), and RRF fusion pushed it down to 16th because it had 30
    useless dense hits to fuse with. Taking hybrid alone lost the answer outright.

    Dense still earns its place for paraphrased recall ("that thing about pickling"), so it
    fills whatever the lexical pass leaves.

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
    # Ask for more than we intend to keep, because the live-branch filter below can
    # reject any of them and there is no second fetch.
    fetch = limit * _BRANCH_FILTER_OVERFETCH
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        hits = retrieval.retrieve_hybrid(
            conn, scope, query, k = fetch, model_name = model, mode = "lexical"
        )
        if len(hits) < fetch:
            try:
                seen = {hit.chunk_id for hit in hits}
                for hit in retrieval.retrieve_hybrid(
                    conn, scope, query, k = fetch, model_name = model, mode = "hybrid"
                ):
                    if hit.chunk_id not in seen:
                        hits.append(hit)
                        seen.add(hit.chunk_id)
                    if len(hits) >= fetch:
                        break
            except Exception:
                # Dense retrieval raises rather than degrading when no embedder can start.
                # The lexical hits above already stand on their own, so this is a top-up
                # that is allowed to fail.
                logger.warning(
                    "conversation_archive.dense_unavailable thread_id=%s",
                    thread_id,
                    exc_info = True,
                )
        if not hits:
            return None
        rows = store.chunks_by_id(conn, [hit.chunk_id for hit in hits])
        transcript = _live_transcript(thread_id)
        if transcript:
            kept = [
                hit
                for hit in hits
                if hit.chunk_id in rows and _on_live_branch(rows[hit.chunk_id]["text"], transcript)
            ][:limit]
            if len(kept) != len(hits):
                logger.info(
                    "conversation_archive.branch_filtered thread_id=%s kept=%d of %d",
                    thread_id,
                    len(kept),
                    len(hits),
                )
            hits = kept
        else:
            hits = hits[:limit]
        if not hits:
            return None
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
