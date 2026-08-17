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
import json
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

# What render_turn appends where it cut something short. Named because the branch
# check has to recognise it: a probe carrying this marker can only ever be a PREFIX of
# the live text, never equal to it.
_TRUNCATION_MARKER = " ..."
_MAX_TOOL_RESULT_CHARS = 4000
# Tool arguments are usually short (a path, a command, a query). The cap is only
# there so a pathological blob cannot dominate the archived turn.
_MAX_TOOL_ARGS_CHARS = 1000
# Retrieval fetches this multiple of the requested number before the live-branch
# filter runs. Filtering AFTER a k-sized fetch means stale turns from an abandoned
# branch can occupy the whole result and starve live matches sitting just below,
# so recall returns nothing while the answer is in the archive.
_BRANCH_FILTER_OVERFETCH = 4
# One over-fetch is not always enough: rewinding or retrying a long continuation that
# had already been compacted leaves an abandoned branch big enough to fill any fixed
# candidate window, and every one of those is rejected while live matches sitting just
# below the cut-off are never looked at. So widen and re-ask instead of giving up.
# Bounded because this is a chat request, not a crawl: the widening stops as soon as
# enough live hits are in hand, when the archive stops yielding new candidates, or here.
_BRANCH_FILTER_MAX_CANDIDATES = 256


def _text_of(content, *, include_tool_calls: bool = False) -> str:
    """Flatten OpenAI message content to plain text, dropping non-text parts.

    ``include_tool_calls`` also flattens assistant-ui's persisted ``tool-call`` parts,
    which carry the call as structured ``toolName``/``args``/``result`` fields rather
    than text. The branch check needs them: ``render_turn`` archives a tool turn as
    "assistant called X: args" and "tool result: ...", so a transcript built without
    them can never contain those lines, and every archived tool turn would be filtered
    out as though it had been rolled back.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text") or "text" in part:
                parts.append(str(part.get("text") or ""))
            elif include_tool_calls and part.get("type") == "tool-call":
                for value in (part.get("toolName"), part.get("args"), part.get("result")):
                    if value in (None, "", {}, []):
                        continue
                    parts.append(value if isinstance(value, str) else json.dumps(value))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _probe_text(message: dict) -> str:
    """One message flattened in the ORDER ``render_turn`` writes it.

    The branch check compares the archived rendering against a transcript, and now
    compares it in order, so the two have to agree on what that order is. render_turn
    writes a tool call before any assistant text on the same message, and the tool's
    result after it (the result arrives as the next message). Both message shapes are
    laid out that way here -- the request's `tool_calls`, and the store's `tool-call`
    content parts, which carry the call and its result together on one part.
    """
    calls: list[str] = []
    texts: list[str] = []
    results: list[str] = []

    def rendered(value) -> list[str]:
        """A structured value as the strings a wire-format copy of it could look like.

        The store keeps a tool call's arguments as an object, while the archived copy is
        the raw string the model emitted, whose spacing is its own. This is a haystack,
        so offering both spacings costs nothing and cannot cause a false rejection.
        """
        if isinstance(value, str):
            return [value]
        try:
            spaced = json.dumps(value, ensure_ascii = False)
            compact = json.dumps(value, ensure_ascii = False, separators = (",", ":"))
        except Exception:
            return [str(value)]
        return [spaced] if spaced == compact else [spaced, compact]

    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool-call":
                for value in (part.get("toolName"), part.get("args")):
                    if value not in (None, "", {}, []):
                        calls.extend(rendered(value))
                result = part.get("result")
                if result not in (None, "", {}, []):
                    results.extend(rendered(result))
            elif part.get("type") in ("text", "input_text") or "text" in part:
                texts.append(str(part.get("text") or ""))
    else:
        texts.append(_text_of(content))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        for value in (function.get("name"), function.get("arguments")):
            if value:
                calls.append(str(value))
    return "\n".join(part for part in calls + texts + results if part)


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
                    arguments = arguments[:_MAX_TOOL_ARGS_CHARS] + _TRUNCATION_MARKER
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
                text = text[:_MAX_TOOL_RESULT_CHARS] + _TRUNCATION_MARKER
            lines.append(f"tool result: {text}")
        else:
            lines.append(f"{role or 'message'}: {text}")
    return "\n".join(lines).strip()


def _archivable(group: list[dict]) -> list[dict]:
    """The part of an evicted turn worth archiving, or an empty list.

    Our own injections come out, the rest stays. Rejecting the whole group instead threw
    away real answers: group_turns keeps an assistant tool call, its result and the reply
    that follows in ONE group, so a forced recall or a RAG auto-inject on that turn took
    the model's actual answer down with it. The question was archived (its own group) and
    the answer was not, so a later search could find what was asked and never what was
    said -- on compaction turns specifically, which are the ones this feature is for.
    """
    if any(str(message.get("role") or "") in _SKIP_ROLES for message in group):
        return []
    return [message for message in group if not _is_injected(message)]


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


def can_archive(thread_id: Optional[str]) -> bool:
    """Whether this thread's evicted turns can be archived at all.

    A temporary (incognito) chat is never written to studio.db and is documented to
    vanish on reload, but the frontend still sends its thread_id and the request model
    carries no incognito flag, so nothing else here would tell the two apart. Archiving
    one would persist exactly the content the user asked not to keep, into a scope with
    no thread row, which no deletion flow could ever reach. An API client that sends a
    thread_id without persisting anything is excluded for the same reason.

    Also what decides whether the fit should hold a recall reserve back: a thread that
    can never be archived can never be recalled either, and the reserve is subtracted
    from the trim target, so charging it there evicts history to make room for content
    that cannot arrive.
    """
    if not thread_id or not enabled():
        return False
    try:
        from storage import studio_db
        return bool(studio_db.chat_thread_has_messages(thread_id))
    except Exception:
        return False


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
    if not can_archive(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
        return 0

    # Lazy, and pointed at the evictor's own grouper on purpose: an archived unit has to
    # be exactly the unit the rolling window drops. Imported here rather than at module
    # scope because the inference layer imports this module.
    from core.inference.context_window import group_turns

    groups = [
        archivable
        for archivable in (_archivable(group) for group in group_turns(evicted))
        if archivable
    ]
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
            # Re-checked here, under a write lock, and not only before the embedding
            # pass above: two generations compacting the same thread both clear that
            # first check while the other is still embedding, and `(scope, sha256)`
            # carries a plain index rather than a unique one, so both insert. The turn
            # is then stored twice and its copies take two of the few recall slots,
            # displacing other turns. Reproduced with two concurrent archive passes.
            _write_lock = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                _write_lock = True
            except Exception:
                # Already inside a transaction: the insert below is still atomic with
                # respect to the re-check, which is what this is for.
                logger.debug("conversation_archive.no_write_lock", exc_info = True)
            if store.document_by_hash(conn, scope, digest):
                if _write_lock:
                    conn.rollback()
                continue
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
                # The turn's real size, so the branch check can bound its run by the
                # messages this document came from. Counting role labels in the rendered
                # text is only an approximation of that: a pasted transcript writes lines
                # that look exactly like the renderer's own.
                archive_messages = len(group),
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, vectors)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
            written += 1
            if _INGEST_FAILED:
                globals()["_INGEST_FAILED"] = False
    except Exception:
        # A chat that cannot archive is worse off than one that can, but far better off
        # than one that raises. Whatever was written before the failure stays searchable.
        globals()["_INGEST_FAILED"] = True
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Deleting a thread cancels its generation, but cancellation is cooperative and the
    # work between the liveness check above and the commit is the slowest part of this
    # function (chunking, then an embedding pass). A delete landing in that window
    # removes the thread's rows and sweeps its archive scope BEFORE this commit puts
    # rows back, leaving content the user deleted persisted in a scope no later delete
    # can reach -- the same unreachable-archive problem as an unpersisted thread.
    #
    # Re-checking after the commit converges either way, because the delete route drops
    # the thread's rows first and sweeps archives last: a sweep that ran before this
    # commit is caught here, and one that runs after removes these rows itself.
    if written and not can_archive(thread_id):
        logger.info("conversation_archive.thread_deleted_mid_ingest thread_id=%s", thread_id)
        delete_for_thread(thread_id)
        return 0
    return written


# Set when an archive write fails, cleared by the next one that succeeds. Process-wide on
# purpose: what fails here is the embedder or the store, not one thread.
_INGEST_FAILED = False


def degraded() -> bool:
    """Whether the last archive attempt failed outright.

    The rolling window holds room back for a recall BEFORE any of this runs, and every
    failure in ``archive_turns`` is swallowed so a chat that cannot archive can still
    answer. Together that means a machine where the embedder cannot start pays the
    reserve on every compaction and gets nothing back for it, forgetting substantially
    more than plain rolling eviction would. The caller checks this and stops reserving.
    """
    return _INGEST_FAILED


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


def _live_transcript(thread_id: str) -> Optional[list[str]]:
    """The thread's saved messages, one normalised string each, or None if it has none.

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
    texts = [_normalise(_probe_text(message)) for message in messages]
    return [text for text in texts if text] or None


def branch_message_texts(messages: Optional[list[dict]]) -> Optional[list[str]]:
    """The ACTIVE branch, one normalised string PER MESSAGE, from the request's own messages.

    Per message rather than one blob, because the branch check has to stay inside the turn
    it is checking. Flattened together, a probe can be satisfied by any later message that
    happens to repeat the words: an archived "Should I deploy? / No" whose answer was
    edited to "Yes" still matched, because an unrelated later turn said "No". Reproduced
    against this code before the split.

    Preferred over ``_live_transcript`` wherever the caller has it, because the stored
    rows are the whole message DAG, not a branch. Retry and regenerate keep the abandoned
    response as a sibling node on purpose -- that is what the branch arrows navigate
    between -- so a thread-wide blob still contains a response the user replaced, and an
    archived copy of it passes the live-branch check and can be recalled into a branch
    where it never happened. The client sends exactly one branch per request, so the
    messages being fitted are the authoritative answer to "what is on this branch".

    They are also the same projection ``render_turn`` archived from, so the probe is
    comparing like with like rather than crossing the store's message format.
    """
    if not messages:
        return None
    texts = [_normalise(_probe_text(message)) for message in messages]
    return [text for text in texts if text] or None


def message_text(content) -> str:
    """One stored message's content, normalised the way the branch check normalises it.

    Exposed so the rolling window can tell two stored messages apart on exactly the terms
    this module compares them on, rather than inventing a second notion of "same text".
    """
    return _normalise(_probe_text({"content": content}))


def content_on_branch(content, transcript: Optional[list[str]]) -> bool:
    """Whether one stored message's text appears on the branch ``transcript`` describes.

    Shared with the rolling window, which has the same problem recall does: a thread's
    stored rows are the whole DAG, so "the newest assistant turn" can belong to a sibling
    the user is not on. Empty text is treated as on-branch, because a message with
    nothing to compare is not evidence of a different branch.
    """
    if not transcript:
        return True
    text = _normalise(_probe_text({"content": content}))
    return not text or any(text in message for message in transcript)


_ROLE_PREFIX = re.compile(
    r"^(?:user|assistant|system|developer|tool result|message):\s*", re.IGNORECASE
)
# render_turn labels a tool call "assistant called <name>: <args>". The label is ours,
# not the stored message's, so it has to come off before the probe like any other role
# prefix -- otherwise every archived tool line misses and the turn looks rolled back.
# The name goes with it when arguments follow (the transcript carries both separately);
# with no arguments the bare name is what remains, and that is what the transcript has.
_TOOL_CALL_PREFIX = re.compile(r"^assistant called (?:[^:\n]+:\s*)?", re.IGNORECASE)


def _probes_for(text: str) -> list[str]:
    """The lines of an archived chunk, normalised into things to look for on the branch.

    Substring containment on a normalised prefix rather than a digest match: the archived
    text was rendered from the inference projection and the saved copy comes back through
    the message store, so exact equality is too brittle to bet the feature on.

    The role labels ``render_turn`` writes ("user: ...") exist only in the archived copy,
    so they are stripped first -- leaving them in made every probe miss and filtered out
    the turns this feature exists to return.
    """
    probes = []
    for line in (text or "").splitlines():
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", _ROLE_PREFIX.sub("", line)))
        # The WHOLE line, except where render_turn cut something short. A prefix probe
        # cannot see an edit past its cut-off, so rewriting the tail of a long answer
        # would leave the stale copy eligible.
        #
        # Keying off the marker rather than off a "tool result:" label matters: a long
        # tool result is ONE appended string containing many newlines, so only its first
        # line carries that label. Every continuation line took the full-string path,
        # including the last one, which is the only line the marker is actually on -- and
        # since nothing in a real transcript ends in that marker, every archived tool
        # turn over the cap was rejected as rolled back. Reproduced on a 400-line result.
        if stripped.endswith(_TRUNCATION_MARKER.strip()):
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        probes.append(stripped)
    return [probe for probe in probes if probe]


def _on_live_branch(text: str, transcript: Optional[list[str]]) -> bool:
    """Whether one archived chunk still exists in the saved thread."""
    probes = _probes_for(text)
    if not probes or not transcript:
        return False
    # EVERY line, IN ORDER, and WITHIN ONE RUN OF ADJACENT MESSAGES.
    #
    # All of a turn: it is archived as a unit, so editing only the assistant half leaves
    # the user line matching and would keep serving an answer that no longer exists.
    # In order: independent membership accepts a turn whose lines were merely rearranged.
    # In one bounded run: a global scan lets a missing line be supplied by any later
    # message that repeats the words, which is how "Should I deploy? / No" survived its
    # answer being edited to "Yes".
    #
    # The window is the probe count, and a group of m messages always renders at least m
    # lines, so the real turn always fits inside it.
    window = len(probes)
    return any(
        _probes_match_from(probes, transcript, start, window) for start in range(len(transcript))
    )


def _scan_probes(probes: list[str], messages: list[str], start: int, last: int) -> Optional[int]:
    """Index of the message where ``probes`` finish matching in order, or None.

    Returned rather than a bool so the chunks of one document can be scanned as a single
    pass: the next chunk continues from where the previous one stopped instead of
    restarting, which is what stops two chunks of the same turn matching two different
    places. The cursor within that message is deliberately NOT carried over, because
    chunks overlap by ``CHUNK_OVERLAP`` and the next chunk legitimately repeats the tail
    of the last one.
    """
    index = start
    cursor = 0
    for probe in probes:
        while index < last:
            found = messages[index].find(probe, cursor)
            if found >= 0:
                cursor = found + len(probe)
                break
            index += 1
            cursor = 0
        else:
            return None
    return index


def _probes_match_from(probes: list[str], messages: list[str], start: int, window: int) -> bool:
    """Whether ``probes`` appear in order within ``messages[start:start + window]``."""
    return _scan_probes(probes, messages, start, min(len(messages), start + window)) is not None


def _rendered_message_count(rows) -> int:
    """How many messages the archived turn was rendered from.

    ``render_turn`` labels every message it writes, so counting the labelled lines counts
    the messages. That is the run this document is allowed to occupy on the branch, and it
    is far tighter than the number of lines: a two-message turn may be a hundred lines
    long, and a hundred-message window will find its tail almost anywhere.

    Chunks overlap, so a labelled line can be counted twice. That only widens the window,
    which is the safe direction: too wide costs strictness, too narrow retires turns that
    are still live.
    """
    total = 0
    for row in rows:
        for line in (row["text"] or "").splitlines():
            stripped = line.strip()
            if _ROLE_PREFIX.match(stripped) or _TOOL_CALL_PREFIX.match(stripped):
                total += 1
    return total


def _document_on_live_branch(conn, document_id: str, transcript: list[str], cache: dict) -> bool:
    """Whether EVERY chunk of an archived turn is still on the branch.

    Per chunk is not enough. A turn longer than CHUNK_TOKENS is stored as several
    chunks of one document, so editing the second half of a long answer retires only
    the chunks that carry the edit, and an untouched earlier chunk of the same retired
    turn stays eligible on its own. The unit that was archived is the turn, and the
    comment on ``_on_live_branch`` says an edit to any part of a turn retires the whole
    copy -- which is only true if the whole document is checked.

    Cached per call: candidates from one turn share a document, and this is the only
    query in the filter.
    """
    if document_id in cache:
        return cache[document_id]
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        # NULL for archives written before the column existed, which fall back to
        # counting labels.
        row = conn.execute(
            "SELECT archive_messages FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        message_count = row["archive_messages"] if row else None
    except Exception:
        # Never fail a recall on the strictness pass: fall back to what the candidate
        # chunk itself said, which is the previous behaviour.
        cache[document_id] = True
        return True
    cache[document_id] = _document_matches_one_run(rows, transcript, message_count)
    return cache[document_id]


def _document_matches_one_run(
    rows,
    transcript: Optional[list[str]],
    message_count: Optional[int] = None,
) -> bool:
    """Every chunk of the turn, found within ONE run of adjacent messages.

    Chunk by chunk independently is not enough. The chunks are consecutive slices of a
    single rendered turn, so letting each pick its own place on the branch lets a turn be
    reassembled out of parts that never sat together: the head matching the question and
    its current answer, the tail matching some later message that happens to repeat the
    passage the edit removed. Reproduced against this code before the run was shared.

    The run is bounded by the document's total line count, which is at least the number of
    messages the turn was rendered from, so a turn that really is still there always fits.
    """
    if not rows or not transcript:
        return False
    probe_lists = [_probes_for(row["text"]) for row in rows]
    if any(not probes for probes in probe_lists):
        return False
    # The messages the turn was rendered from, not the lines it produced. Bounding by
    # lines let the tail of a long answer be satisfied by a message far outside the turn.
    #
    # Recorded at archive time where it is known. The label count is the fallback for
    # documents written before that, and it is only an approximation: a pasted transcript
    # contains lines that look exactly like the ones the renderer writes, and each one
    # widens the run by a message.
    window = (
        int(message_count)
        if message_count
        else (_rendered_message_count(rows) or sum(len(probes) for probes in probe_lists))
    )

    def _one_run_from(start: int) -> bool:
        last = min(len(transcript), start + window)
        position = start
        for probes in probe_lists:
            found = _scan_probes(probes, transcript, position, last)
            if found is None:
                return False
            position = found
        return True

    return any(_one_run_from(start) for start in range(len(transcript)))


def _candidates(conn, scope: str, query: str, model, fetch: int, thread_id: str) -> list:
    """Up to ``fetch`` archive chunks for ``query``: lexical first, hybrid for the rest."""
    hits = retrieval.retrieve_hybrid(conn, scope, query, k = fetch, model_name = model, mode = "lexical")
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
                "conversation_archive.dense_unavailable thread_id=%s", thread_id, exc_info = True
            )
    return hits


def recall(
    thread_id: str,
    query: str,
    *,
    top_k: Optional[int] = None,
    branch_messages: Optional[list[dict]] = None,
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
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        # The request's own branch first: the stored rows are the whole DAG, siblings
        # included. Falling back to them is still better than not filtering at all, for
        # a caller that has no branch to offer.
        transcript = branch_message_texts(branch_messages) or _live_transcript(thread_id)
        fetch = limit * _BRANCH_FILTER_OVERFETCH
        rows: dict = {}
        hits: list = []
        live_documents: dict = {}
        while True:
            candidates = _candidates(conn, scope, query, model, fetch, thread_id)
            if not candidates:
                return None
            rows = store.chunks_by_id(conn, [hit.chunk_id for hit in candidates])
            if not transcript:
                hits = candidates[:limit]
                break
            hits = [
                hit
                for hit in candidates
                if hit.chunk_id in rows
                and _document_on_live_branch(
                    conn, rows[hit.chunk_id]["document_id"], transcript, live_documents
                )
            ]
            if len(hits) != len(candidates):
                logger.info(
                    "conversation_archive.branch_filtered thread_id=%s kept=%d of %d",
                    thread_id,
                    len(hits),
                    len(candidates),
                )
            # Enough live hits, or nothing more to widen into: an abandoned branch can
            # outrank the live one, but it cannot outrank it forever.
            if (
                len(hits) >= limit
                or len(candidates) < fetch
                or fetch >= _BRANCH_FILTER_MAX_CANDIDATES
            ):
                hits = hits[:limit]
                break
            fetch = min(_BRANCH_FILTER_MAX_CANDIDATES, fetch * _BRANCH_FILTER_OVERFETCH)
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


def _delete_scope_without_vec(scope: str, thread_id: str) -> int:
    """Delete a scope's text-bearing rows over a connection with no sqlite-vec.

    Deletion must not depend on the optional native extension. An archive is only ever
    WRITTEN while vec0 loads, but the library can stop loading afterwards (a venv change,
    the common macOS case), and a delete that quietly does nothing then leaves the turns
    of a deleted conversation on disk, ready to answer again the day vec0 loads.

    The embedding rows in chunks_vec cannot be reached from here and are left behind.
    They carry vectors, not text, and every read path resolves a hit back through
    ``chunks`` joined to ``documents``, both of which are gone, so an orphan can be
    retrieved by nothing.
    """
    conn = None
    removed = 0
    try:
        conn = rag_db.get_metadata_connection()
        documents = [
            row["id"]
            for row in conn.execute("SELECT id FROM documents WHERE scope=?", (scope,)).fetchall()
        ]
        for document_id in documents:
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (document_id,),
            )
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            removed += 1
        conn.commit()
    except Exception:
        logger.warning(
            "conversation_archive.delete_without_vec_failed thread_id=%s", thread_id, exc_info = True
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return removed


def delete_for_thread(thread_id: str) -> int:
    """Drop a thread's archive. Called when the thread itself is deleted."""
    if not thread_id:
        return 0
    scope = store.conversation_archive_scope(thread_id)
    conn = None
    removed = 0
    try:
        try:
            conn = rag_db.get_connection()
        except Exception:
            # No vec0 here. Delete what can be deleted rather than nothing at all.
            return _delete_scope_without_vec(scope, thread_id)
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
