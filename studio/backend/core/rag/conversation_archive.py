# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the turns the rolling context window evicts, and hand them back on request.

Each evicted turn is indexed into a per-thread RAG scope, reusing the store, chunker,
embedder and hybrid retrieval that back attached documents. The stored transcript is
never touched; this is only a search index over turns the projection cannot carry.

Two properties are load-bearing and easy to break:

* The archive is CUMULATIVE. Nothing is cleared or re-scoped per compaction, so the fifth
  compaction can still find what the first evicted. On MRCR v2 over 11 compaction events,
  cumulative scored 0.450 against 0.058 for latest-compaction-only.
* Nothing here may break a chat. Every entry point no-ops on failure, because the
  alternative to a degraded archive is a working conversation, not an error.
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

# Instructions, not conversation. An archived system prompt could come back quoted as an
# "earlier turn" and read as user-authored text.
_SKIP_ROLES = frozenset({"system", "developer"})

# Tool call ids this feature and the RAG auto-inject generate. Their results are retrieved
# passages, so archiving them would feed retrieved text back into its own index.
_INJECTED_CALL_PREFIXES = ("rag_auto_", "conv_recall_")

# What render_turn appends where it cut something short. Named because the branch check
# has to recognise it: a probe carrying it can only be a PREFIX of the live text.
_TRUNCATION_MARKER = " ..."
_MAX_TOOL_RESULT_CHARS = 4000
# Tool arguments are usually short; the cap only stops a pathological blob dominating.
_MAX_TOOL_ARGS_CHARS = 1000
# Over-fetch multiple ahead of the live-branch filter. Filtering after a k-sized fetch
# lets stale turns from an abandoned branch fill the result and starve live matches just
# below, so recall returns nothing while the answer is in the archive.
_BRANCH_FILTER_OVERFETCH = 4
# One over-fetch is not always enough: rewinding a long compacted continuation leaves an
# abandoned branch big enough to fill any fixed candidate window, so widen and re-ask.
# Bounded because this is a chat request, not a crawl.
_BRANCH_FILTER_MAX_CANDIDATES = 256


def _text_of(content, *, include_tool_calls: bool = False) -> str:
    """Flatten OpenAI message content to plain text, dropping non-text parts.

    ``include_tool_calls`` also flattens assistant-ui's persisted ``tool-call`` parts,
    whose call lives in structured ``toolName``/``args``/``result`` fields. The branch
    check needs them, or a transcript can never contain the lines ``render_turn`` wrote
    and every archived tool turn looks rolled back.
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

    The branch check matches in order, so both must agree on it: a tool call before any
    assistant text on the same message, its result after. Laid out that way here for both
    shapes, the request's ``tool_calls`` and the store's ``tool-call`` content parts.
    """
    calls: list[str] = []
    texts: list[str] = []
    results: list[str] = []

    def rendered(value) -> list[str]:
        """A structured value as the strings a wire-format copy of it could look like.

        The store keeps tool arguments as an object; the archived copy is the model's raw
        string, with its own spacing. This is a haystack, so offering both is free.
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

    Both sides go in one document on purpose, so retrieval returns the question with its
    answer rather than a floating fragment.
    """
    lines: list[str] = []
    for message in group:
        role = str(message.get("role") or "")
        if role in _SKIP_ROLES:
            continue
        text = _text_of(message.get("content")).strip()
        calls = message.get("tool_calls") or []
        if calls:
            # The arguments, bounded, not just the name: the searchable substance is the
            # command or query that ran, and "assistant called terminal" cannot answer
            # "what did you run earlier?". Assistant text on the same message likewise.
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
            # Tool results are huge and the least useful thing to quote back verbatim:
            # keep enough to be searchable without bloating the index.
            if len(text) > _MAX_TOOL_RESULT_CHARS:
                text = text[:_MAX_TOOL_RESULT_CHARS] + _TRUNCATION_MARKER
            lines.append(f"tool result: {text}")
        else:
            lines.append(f"{role or 'message'}: {text}")
    return "\n".join(lines).strip()


def _archivable(group: list[dict]) -> list[dict]:
    """The part of an evicted turn worth archiving, or an empty list.

    Our own injections come out, the rest stays. Rejecting the whole group threw away real
    answers: ``group_turns`` keeps a tool call, its result and the following reply in ONE
    group, so an injection on that turn took the model's answer down with it, leaving the
    question archived and the answer not -- on compaction turns, which are the point here.
    """
    if any(str(message.get("role") or "") in _SKIP_ROLES for message in group):
        return []
    return [message for message in group if not _is_injected(message)]


def enabled() -> bool:
    """Whether the archive can actually run here.

    ``rag_available()`` and not ``RAG_AVAILABLE``: the flag only records that
    ``import sqlite_vec`` worked, while the vec0 native library is a separate file a venv
    can be missing (the common macOS case). Trusting it there is worse than no feature at
    all: the fit still reserves room, then write and recall both fail, so the user pays
    extra eviction for content that never arrives.
    """
    return bool(config.CONVERSATION_ARCHIVE) and bool(rag_db.rag_available())


def can_archive(thread_id: Optional[str]) -> bool:
    """Whether this thread's evicted turns can be archived at all.

    A temporary (incognito) chat is never written to studio.db, yet the frontend still
    sends its thread_id and the request carries no incognito flag, so saved messages are
    the only signal. Archiving one would persist exactly what the user asked not to keep,
    into a scope with no thread row that no deletion flow could reach. An API client that
    sends a thread_id without persisting anything is excluded for the same reason.

    Also gates the fit's recall reserve: a thread that cannot be archived cannot be
    recalled, so reserving there would evict history for content that cannot arrive.
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
    # Incognito and API-only threads are excluded; see can_archive. By the time a chat is
    # long enough to compact, its earlier turns are always persisted.
    if not can_archive(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
        return 0

    # The evictor's own grouper, so an archived unit is exactly the unit the window drops.
    # Imported lazily because the inference layer imports this module.
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
        # What the query side will ask for. Only a prediction until the encode reports
        # what it actually used, so the authoritative check happens under the write lock.
        expected_identity = embeddings.embedding_identity(model)

        # Chunk everything first, then embed it in ONE pass. Per group, a first
        # compaction of a long chat ran dozens of one-item embedding jobs back to back
        # before the reply could start, and both backends serialise them.
        pending = []
        for group in groups:
            text = render_turn(group)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            if _archived_under(conn, scope, digest, expected_identity):
                continue
            chunks = chunk_pages(
                [Page(text = text, page_number = None, char_count = len(text))],
                max_tokens = config.CHUNK_TOKENS,
                overlap = config.CHUNK_OVERLAP,
                count = count,
            )
            if chunks:
                pending.append((group, digest, chunks))
        if not pending:
            return 0

        # Identity from the encode that produced these vectors: a concurrent embedder
        # swap would otherwise label them with a space they were never in.
        vectors, identity = embeddings.encode_with_identity(
            [chunk.text for group_chunks in pending for chunk in group_chunks[2]],
            model_name = model,
            normalize = True,
        )
        offset = 0
        for group, digest, chunks in pending:
            group_vectors = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            roles = " + ".join(
                dict.fromkeys(str(message.get("role") or "message") for message in group)
            )
            # commit=False, then commit once the chunks are in: committing the document
            # first leaves an empty row marked "completed" if the chunk write fails, and
            # `document_by_hash` skips that turn forever after without retrying.
            # Re-checked here under a write lock, not only before the embedding pass: two
            # generations compacting one thread both clear that first check, and
            # `(scope, sha256)` is a plain index, so both insert and the duplicate takes
            # two of the few recall slots. Reproduced with two concurrent archive passes.
            _write_lock = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                _write_lock = True
            except Exception:
                # Already in a transaction: the insert is still atomic with the re-check.
                logger.debug("conversation_archive.no_write_lock", exc_info = True)
            stale = _stale_document(conn, scope, digest, identity)
            if stale is _ARCHIVED:
                if _write_lock:
                    conn.rollback()
                continue
            ordinal = None
            if stale is not None:
                # Same turn, vectors from an embedder the query side no longer asks for.
                # Skipping it would leave the turn invisible to dense search forever, so
                # the copy is replaced rather than deduplicated, as ingestion does.
                #
                # The turn KEEPS its position. Re-embedding walks the whole archive, so
                # taking a fresh ordinal here would renumber an entire conversation into
                # the order its vectors were rebuilt, which is not an order at all.
                previous = store.get_document(conn, stale) or {}
                ordinal = previous.get("archive_ordinal")
                store.delete_document(conn, stale, commit = False)
            if ordinal is None:
                ordinal = store.next_archive_ordinal(conn, scope)
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
                # The turn's real size, so the branch check can bound its run exactly.
                # Counting role labels only approximates it, since a pasted transcript
                # writes lines that look exactly like the renderer's own.
                archive_messages = len(group),
                # Where this turn sits in the conversation. Allocated inside the write
                # lock, in `group_turns` order, so it is conversation order within an
                # epoch and across epochs -- which `created_at` cannot be, since one
                # compaction writes every turn it evicts microseconds apart. Written
                # unconditionally, with no knob: a period with ordering switched off must
                # not punch permanent holes in the sequence.
                archive_ordinal = ordinal,
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, group_vectors)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
            written += 1
            if _INGEST_FAILED:
                globals()["_INGEST_FAILED"] = False
    except Exception:
        # A chat that cannot archive still beats one that raises. Whatever was written
        # before the failure stays searchable.
        globals()["_INGEST_FAILED"] = True
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Cancellation is cooperative, and chunking plus embedding is the slowest stretch of
    # this function. A delete landing in that window sweeps the archive scope BEFORE this
    # commit puts rows back, stranding deleted content in a scope no later delete reaches.
    # Re-checking after the commit converges either way: the delete route drops thread
    # rows first and sweeps archives last, so an earlier sweep is caught here and a later
    # one removes these rows itself.
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

    The window reserves recall room before any of this runs and ``archive_turns``
    swallows its failures, so a machine whose embedder cannot start would pay the reserve
    on every compaction for nothing. The caller checks this and stops reserving.
    """
    return _INGEST_FAILED


# Returned by ``_stale_document`` for a turn that is already archived under vectors the
# query side still accepts.
_ARCHIVED = "archived"


def _stale_document(conn, scope: str, digest: str, identity: str):
    """The document id to replace, ``_ARCHIVED`` to skip, or None to write a new one.

    Hash alone is not enough. Dense search only reads documents whose recorded embedder
    matches the query's, so a turn archived under the previous model stays hashed-and-
    skipped while being invisible to every paraphrased search. Ingestion re-indexes in
    that case; so does this.
    """
    existing = store.document_by_hash(conn, scope, digest)
    if existing is None:
        return None
    document = store.get_document(conn, existing)
    recorded = (document or {}).get("embedding_model")
    if config.embedding_identity_matches(recorded, identity):
        return _ARCHIVED
    return existing


def _archived_under(conn, scope: str, digest: str, identity: str) -> bool:
    """Cheap pre-check before the chunking and embedding pass."""
    return _stale_document(conn, scope, digest, identity) is _ARCHIVED


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

    Keeps recall on the branch the user is on. Editing a message rewinds the thread, but
    the archive is append-only and still holds the abandoned continuation's turns, so
    without this a recall can pull back a turn that never happened on this branch.
    Verified: after rewinding past a turn, querying its text still returned it.

    None means the thread has no saved transcript (an API client passing a thread_id
    without persisting), and the caller then does not filter: absence of evidence.
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


def branch_message_texts(
    messages: Optional[list[dict]], roles: Optional[tuple[str, ...]] = None
) -> Optional[list[str]]:
    """The ACTIVE branch, one normalised string PER MESSAGE, from the request's own messages.

    Per message rather than one blob, so the check stays inside the turn it is checking.
    Flattened, a probe can be satisfied by any later message repeating the words: an
    archived "Should I deploy? / No" whose answer was edited to "Yes" still matched
    because an unrelated later turn said "No". Reproduced before the split.

    Preferred over ``_live_transcript`` wherever available: the stored rows are the whole
    message DAG, and retry/regenerate keep the replaced response as a sibling, so a
    thread-wide blob can validate a turn that is not on this branch. The client sends
    exactly one branch per request, and it is the same projection ``render_turn``
    archived from, so the probe compares like with like.

    ``roles`` narrows it to messages of those roles. The rolling window compares stored
    ASSISTANT rows, and against every role a short abandoned reply ("Done") matches a live
    user message that merely contains it ("not done yet").
    """
    if roles:
        messages = [
            message for message in (messages or []) if str(message.get("role") or "") in roles
        ]
    if not messages:
        return None
    texts = [_normalise(_probe_text(message)) for message in messages]
    return [text for text in texts if text] or None


def message_text(content) -> str:
    """One stored message's content, normalised the way the branch check normalises it.

    Exposed so the rolling window compares stored messages on exactly these terms rather
    than inventing a second notion of "same text".
    """
    return _normalise(_probe_text({"content": content}))


def content_on_branch(content, transcript: Optional[list[str]]) -> bool:
    """Whether one stored message's text appears on the branch ``transcript`` describes.

    Shared with the rolling window, which has recall's problem: the stored rows are the
    whole DAG, so "the newest assistant turn" can belong to a sibling branch. Empty text
    counts as on-branch, since nothing to compare is not evidence of another branch.
    """
    if not transcript:
        return True
    text = _normalise(_probe_text({"content": content}))
    return not text or any(text in message for message in transcript)


_ROLE_PREFIX = re.compile(
    r"^(?:user|assistant|system|developer|tool result|message):\s*", re.IGNORECASE
)
# render_turn labels a tool call "assistant called <name>: <args>". The label is ours,
# not the stored message's, so it comes off before the probe like any role prefix, or
# every archived tool line misses and the turn looks rolled back. The name goes with it
# when arguments follow; with none, the bare name is what the transcript has.
_TOOL_CALL_PREFIX = re.compile(r"^assistant called (?:[^:\n]+:\s*)?", re.IGNORECASE)


def _probes_for(text: str) -> list[str]:
    """The lines of an archived chunk, normalised into things to look for on the branch.

    Substring containment on a normalised prefix rather than a digest match: the archived
    text came from the inference projection and the saved copy from the message store, so
    exact equality is too brittle. The role labels ``render_turn`` writes exist only in
    the archived copy, so they are stripped first, or every probe misses.
    """
    probes = []
    for line in (text or "").splitlines():
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", _ROLE_PREFIX.sub("", line)))
        # The WHOLE line, except where render_turn cut something short: a prefix probe
        # cannot see an edit past its cut-off, leaving a rewritten tail eligible.
        # Keyed off the marker, not a "tool result:" label: a long tool result is ONE
        # string of many newlines, so only its first line carries the label while the
        # marker is on its last, and nothing in a real transcript ends in the marker, so
        # every over-cap tool turn was rejected. Reproduced on a 400-line result.
        if stripped.endswith(_TRUNCATION_MARKER.strip()):
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        probes.append(stripped)
    return [probe for probe in probes if probe]


def _probe_entries(text: str) -> list[tuple[str, bool]]:
    """``_probes_for`` with a flag per line: was it cut short, or is it a tool call.

    Both mean the live message may legitimately hold more than the probe: a cut line is a
    prefix by construction, and a tool call is matched against a haystack that renders its
    arguments twice, spaced and compact, so only one of the two can ever be covered.
    """
    entries = []
    for line in (text or "").splitlines():
        without_role = _ROLE_PREFIX.sub("", line)
        is_call = bool(_TOOL_CALL_PREFIX.match(without_role.strip()))
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", without_role))
        truncated = stripped.endswith(_TRUNCATION_MARKER.strip())
        if truncated:
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        if stripped:
            entries.append((stripped, truncated or is_call))
    return entries


def _on_live_branch(text: str, transcript: Optional[list[str]]) -> bool:
    """Whether one archived chunk still exists in the saved thread."""
    probes = _probes_for(text)
    if not probes or not transcript:
        return False
    # EVERY line, IN ORDER, WITHIN ONE RUN OF ADJACENT MESSAGES. All of a turn, because
    # it is archived as a unit and editing only the assistant half leaves the user line
    # matching. In order, because independent membership accepts rearranged lines. In one
    # bounded run, because a global scan lets any later message supply a missing line,
    # which is how "Should I deploy? / No" survived its answer becoming "Yes".
    # The window is the probe count; m messages render at least m lines, so a real turn
    # always fits.
    window = len(probes)
    return any(
        _probes_match_from(probes, transcript, start, window) for start in range(len(transcript))
    )


def _scan_probes(
    entries: list[tuple[str, bool]], messages: list[str], start: int, last: int
) -> Optional[tuple[int, int, int, bool]]:
    """Where the probes finish: message index, end offset, opening offset, tail-is-partial.

    An index rather than a bool so one document's chunks scan as a single pass, each
    continuing where the last stopped, which stops two chunks of a turn matching two
    places. The cursor within that message is NOT carried over: chunks overlap by
    ``CHUNK_OVERLAP``, so the next one legitimately repeats the previous tail.
    """
    index = start
    cursor = 0
    opened_at = None
    partial = False
    fresh = False
    for probe, partial_ok in entries:
        while index < last:
            found = messages[index].find(probe, cursor)
            if found >= 0:
                # A message the run stepped INTO has to be accounted for from its first
                # character: an edit that prepends to it ("no" becoming "correction: no")
                # otherwise leaves every probe matching. A tool call is exempt, and the
                # exemption then covers the rest of that message: the store keeps a call
                # as a structured part, so the live text carries the tool name and BOTH
                # spellings of its arguments while the archived copy has one line of one
                # of them. Nothing there can line up character for character.
                if fresh and found != 0 and not partial_ok:
                    return None
                if opened_at is None:
                    # Same exemption at the front of the run as inside it.
                    opened_at = 0 if partial_ok else found
                cursor = found + len(probe)
                partial = partial or partial_ok
                fresh = False
                break
            # And leaving one it had entered: whatever is left over is text an edit added.
            if cursor and not partial and cursor < len(messages[index]):
                return None
            index += 1
            cursor = 0
            partial = False
            fresh = True
        else:
            return None
    return index, cursor, (opened_at or 0), partial


def _probes_match_from(probes: list[str], messages: list[str], start: int, window: int) -> bool:
    """Whether ``probes`` appear in order within ``messages[start:start + window]``."""
    return (
        _scan_probes(
            [(probe, True) for probe in probes],
            messages,
            start,
            min(len(messages), start + window),
        )
        is not None
    )


def _rendered_message_count(rows) -> int:
    """How many messages the archived turn was rendered from.

    ``render_turn`` labels every message it writes, so labelled lines count messages. That
    bounds the run this document may occupy, far tighter than a line count: a two-message
    turn can be a hundred lines, and a hundred-message window finds its tail anywhere.

    Overlapping chunks can double-count a label, which only widens the window: the safe
    direction, since too narrow retires turns that are still live.
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

    Per chunk is not enough: a turn over CHUNK_TOKENS spans several chunks, so editing the
    second half of a long answer leaves the untouched earlier chunks eligible on their
    own. The archived unit is the turn, so an edit to any part retires the whole copy.

    Cached per call: candidates from one turn share a document, and this is the filter's
    only query.
    """
    if document_id in cache:
        return cache[document_id]
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        # NULL for archives predating the column, which fall back to counting labels.
        row = conn.execute(
            "SELECT archive_messages FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        message_count = row["archive_messages"] if row else None
    except Exception:
        # Never fail a recall on the strictness pass: fall back to what the candidate
        # chunk itself said.
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

    Chunk by chunk independently is not enough: they are consecutive slices of one turn,
    so letting each pick its own place reassembles a turn from parts that never sat
    together (head on the current answer, tail on a later message repeating what the edit
    removed). Reproduced before the run was shared. The run is bounded by at least the
    number of messages the turn was rendered from, so a live turn always fits.
    """
    if not rows or not transcript:
        return False
    probe_lists = [_probe_entries(row["text"]) for row in rows]
    if any(not probes for probes in probe_lists):
        return False
    # The messages the turn was rendered from, not the lines it produced: bounding by
    # lines let a long answer's tail be satisfied far outside the turn. Recorded at
    # archive time; the label count is an approximate fallback for older documents, since
    # a pasted transcript contains lines that look like the renderer's own.
    window = (
        int(message_count)
        if message_count
        else (_rendered_message_count(rows) or sum(len(probes) for probes in probe_lists))
    )

    def _one_run_from(start: int) -> bool:
        last = min(len(transcript), start + window)
        position = start
        cursor = 0
        opened_at = None
        partial_tail = False
        for probes in probe_lists:
            found = _scan_probes(probes, transcript, position, last)
            if found is None:
                return False
            position, cursor, chunk_opened_at, partial_tail = found
            # Where the whole run opened, which is the first chunk's answer: an edit that
            # prepends to the turn's FIRST message shows up here and nowhere else.
            if opened_at is None:
                opened_at = chunk_opened_at
        # And the turn has to cover the messages it claims, end to end. An edit that
        # keeps the old text and adds to it leaves every probe matching, whichever side it
        # adds on and whichever message it touches: "No" becoming "No, correction: yes" or
        # "Correction: no", or the question itself growing a clause. Either way the
        # pre-edit copy stayed eligible and could be recalled as the answer.
        if opened_at:
            return False
        return partial_tail or cursor >= len(transcript[position])

    return any(_one_run_from(start) for start in range(len(transcript)))


def _conversation_order(row) -> tuple:
    """Sort key putting recalled turns in the order they were said.

    NULL ordinals sort FIRST, and that is not a fallback so much as a fact: they were
    written by a build that had no such column, so they genuinely predate every numbered
    turn in the same scope. Within a turn, `chunk_index` keeps a long message's pieces
    contiguous and in order, which relevance ordering gets wrong today. `created_at`
    breaks ties, because the ordinal is deliberately not UNIQUE: the write lock is
    best-effort, so two concurrent archive passes can compute the same MAX + 1 and must
    tie-break rather than raise.
    """
    if row is None:
        return (2, 0, "", 0)
    ordinal = tool._row_value(row, "archive_ordinal")
    created = tool._row_value(row, "created_at") or ""
    index = tool._row_value(row, "chunk_index") or 0
    if ordinal is None:
        return (0, 0, created, index)
    return (1, int(ordinal), created, index)


def _candidates(conn, scope: str, query: str, model, fetch: int, thread_id: str) -> list:
    """Up to ``fetch`` archive chunks for ``query``: lexical first, hybrid for the rest.

    The lexical pass runs TWICE when the question contains an identifier: once requiring
    all of them, then once over the content words to top up. Requiring them first is what
    stops an incidental word in the question outranking the subject of the whole
    conversation (see `store.conversation_match_queries`); running the permissive pass
    afterwards is what stops a conjunction that matches nothing from reading as an empty
    archive. The merged list is what the caller's widening loop measures, so a
    conjunction returning few rows cannot be mistaken for "nothing left to widen into".
    """
    expressions = (store.conversation_match_queries(query)
                   if config.CONVERSATION_QUERY_FOCUS else [None])
    hits: list = []
    seen: set = set()
    for expression in expressions:
        if len(hits) >= fetch:
            break
        for hit in retrieval.retrieve_hybrid(
            conn, scope, query, k = fetch, model_name = model, mode = "lexical",
            lexical_query = expression,
        ):
            if hit.chunk_id not in seen:
                hits.append(hit)
                seen.add(hit.chunk_id)
            if len(hits) >= fetch:
                break
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
            # The lexical hits stand on their own, so this top-up may fail.
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
    extra_queries: Optional[list[str]] = None,
    forced: bool = False,
) -> Optional[tuple[str, list[dict]]]:
    """Most relevant archived turns for ``query``, rendered like any other RAG hit.

    LEXICAL FIRST, then hybrid for the rest of the budget. Recalling your own conversation
    is mostly exact match (a name, a number, a code pasted twenty turns ago), which lives
    or dies on rare tokens. Measured on a 30-turn walkthrough with identical boilerplate
    per turn: the needle chunk ranked 3rd lexically at any k, was never returned by dense
    retrieval at all, and RRF fusion pushed it to 16th behind 30 useless dense hits.
    Hybrid alone lost the answer. Dense still earns its place for paraphrased recall, so
    it fills whatever the lexical pass leaves.

    No relevance floor, unlike ``tool.search_for_autoinject``: its 0.70 cosine gate keeps
    off-topic documents out, but here the passages ARE this conversation and the
    alternative to a weak match is no memory at all. It also keeps lexical-only hits,
    since ``filter_min_score`` only gates hits carrying a dense score.
    """
    query = (query or "").strip()
    if not thread_id or not query or not enabled():
        return None
    min_dense_score = (
        config.CONVERSATION_FORCED_MIN_SCORE if forced else 0.0
    )

    scope = store.conversation_archive_scope(thread_id)
    limit = top_k or config.CONVERSATION_ARCHIVE_TOP_K
    # Two queries rather than one concatenated string. Concatenating would recreate the
    # very defect the shaped query fixes: the filler's tokens dilute the instruction's
    # identifiers, and the conjunctive pass would AND identifiers drawn from two
    # unrelated intents. Run separately, each is shaped on its own and each spends its
    # own half of the budget. At a limit of one the anchor takes the slot outright: one
    # chunk retrieved for the word "continue" is worth nothing.
    queries = [query] + [q.strip() for q in (extra_queries or []) if (q or "").strip()]
    if len(queries) > 1:
        share = max(1, -(-limit // len(queries)))
        merged: list = []
        seen_ids: set = set()
        for index, one in enumerate(reversed(queries)):
            # Anchors first: they are the reason the extra query was added at all.
            room = limit - len(merged) if index == len(queries) - 1 else share
            if room <= 0:
                break
            found = recall(thread_id, one, top_k = room,
                           branch_messages = branch_messages, forced = forced)
            if not found:
                continue
            for source in found[1]:
                if source["chunkId"] in seen_ids:
                    continue
                seen_ids.add(source["chunkId"])
                merged.append(source)
        if not merged:
            return None
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            merged.sort(key = lambda source: (source.get("turn") is None,
                                              source.get("turn") or 0))
            kept = merged[:limit]
            return tool.render_conversation_sources(kept), kept
        kept = merged[:limit]
        return tool.render_sources(kept), kept
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        # The request's own branch first: the stored rows are the whole DAG. Falling back
        # to them still beats not filtering when the caller has no branch to offer.
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
        if min_dense_score > 0:
            # The FORCED path only. An automatic lookup that returns whatever happens to
            # share a stopword with the question is worse than none under checkpoint
            # compaction, because that block is also the model's first sight of the search
            # tool. Lexical-only hits are kept: they matched real tokens, and gating them
            # on a similarity they never carried would delete the exact-identifier hits
            # this archive is best at. A model that wanted more can still search.
            strong = [
                hit for hit in hits
                if hit.dense_score is None or hit.dense_score >= min_dense_score
            ]
            if not strong:
                logger.info(
                    "conversation_archive.recall_below_floor thread_id=%s floor=%.2f",
                    thread_id, min_dense_score,
                )
                return None
            hits = strong
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            # AFTER the top-k slice, never before. Sorting first would make the slice take
            # the OLDEST turns rather than the most relevant ones, which is a different
            # feature and a worse one.
            hits.sort(key = lambda hit: _conversation_order(rows.get(hit.chunk_id)))
            text, sources = tool.format_conversation_recall(rows, hits)
        else:
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

    Deletion must not depend on the optional native extension. Archives are only WRITTEN
    while vec0 loads, but it can stop loading afterwards (a venv change, common on macOS),
    and a delete that silently does nothing leaves a deleted conversation on disk ready to
    answer again once vec0 returns.

    The chunks_vec rows are unreachable from here and left behind. They carry vectors, not
    text, and every read path resolves through ``chunks`` joined to ``documents``, both
    gone, so nothing can retrieve an orphan.
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
