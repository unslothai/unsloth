# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the turns the rolling context window evicts, and hand them back on request.

Each evicted turn is indexed into a per-thread RAG scope, reusing the store, chunker, embedder and
hybrid retrieval that back attached documents. The stored transcript is never touched; this is only
a search index over turns the projection cannot carry.

Two properties are load-bearing and easy to break. The archive is CUMULATIVE: nothing is cleared or
re-scoped per compaction, so the fifth compaction can still find what the first evicted (on MRCR v2
over 11 compaction events, cumulative scored 0.450 against 0.058 for latest-compaction-only). And
nothing here may break a chat: every entry point no-ops on failure, because the alternative to a
degraded archive is a working conversation, not an error.
"""

from __future__ import annotations

import hashlib
import logging
import bisect
import json
import re
from typing import Optional

from storage import rag_db

from . import config, embeddings, retrieval, store, tool
from .chunking import chunk_pages
from .parsers import Page

logger = logging.getLogger(__name__)

# An archived system prompt could be recalled as user-authored text.
_SKIP_ROLES = frozenset({"system", "developer"})

# Auto-inject/recall call ids: their results are retrieved passages, so archiving them would self-feed the index.
_INJECTED_CALL_PREFIXES = ("rag_auto_", "conv_recall_")

# A probe carrying this marker can only be a PREFIX of the live text.
_TRUNCATION_MARKER = " ..."
_MAX_TOOL_RESULT_CHARS = 4000
_MAX_TOOL_ARGS_CHARS = 1000
# Over-fetch before the branch filter so stale abandoned-branch turns cannot starve live hits.
_BRANCH_FILTER_OVERFETCH = 4
# Long abandoned branches can fill any fixed window, so widen up to this bound.
_BRANCH_FILTER_MAX_CANDIDATES = 256


def _text_of(content, *, include_tool_calls: bool = False) -> str:
    """Tool calls are flattened only when include_tool_calls is set, which the branch check needs."""
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
    """Matches render_turn's order: text after a tool call flushes the pending calls first."""
    chunks: list[str] = []
    calls: list[str] = []
    texts: list[str] = []
    results: list[str] = []

    def flush() -> None:
        chunks.extend(part for part in calls + texts + results if part)
        calls.clear()
        texts.clear()
        results.clear()

    def rendered(value) -> list[str]:
        """A structured value as the strings a wire-format copy of it could look like. The store
        keeps tool arguments as an object; the archived copy is the model's raw string, with its
        own spacing. This is a haystack, so offering both is free."""
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
            elif part.get("type") == "reasoning":
                # Wire copy has no reasoning part (sent in reasoning_content), so skip it or probes never match.
                continue
            elif part.get("type") in ("text", "input_text") or "text" in part:
                if calls:
                    flush()
                texts.append(str(part.get("text") or ""))
    else:
        texts.append(_text_of(content))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        for value in (function.get("name"), function.get("arguments")):
            if value:
                calls.append(str(value))
    flush()
    return "\n".join(chunks)


def _is_injected(message: dict) -> bool:
    call_id = str(message.get("tool_call_id") or "")
    if call_id.startswith(_INJECTED_CALL_PREFIXES):
        return True
    for call in message.get("tool_calls") or []:
        if str((call or {}).get("id") or "").startswith(_INJECTED_CALL_PREFIXES):
            return True
    return False


def render_turn(group: list[dict]) -> str:
    """Render one evicted turn group as the text that gets indexed and quoted back. Both sides go in
    one document on purpose, so retrieval returns the question with its answer rather than a
    floating fragment."""
    lines: list[str] = []
    for message in group:
        role = str(message.get("role") or "")
        if role in _SKIP_ROLES:
            continue
        text = _text_of(message.get("content")).strip()
        calls = message.get("tool_calls") or []
        if calls:
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
            if len(text) > _MAX_TOOL_RESULT_CHARS:
                text = text[:_MAX_TOOL_RESULT_CHARS] + _TRUNCATION_MARKER
            lines.append(f"tool result: {text}")
        else:
            lines.append(f"{role or 'message'}: {text}")
    return "\n".join(lines).strip()


def _archivable(group: list[dict]) -> list[dict]:
    """Drops only injected messages, keeping the real reply that shares a group with an injection."""
    if any(str(message.get("role") or "") in _SKIP_ROLES for message in group):
        return []
    kept = [message for message in group if not _is_injected(message)]
    return _without_retrieval(kept)


def _retrieval_names() -> frozenset:
    """The tool names whose results are retrieved passages, not conversation."""
    try:
        from core.inference.tool_call_parser import RAG_SEARCH_TOOLS
        return frozenset(RAG_SEARCH_TOOLS)
    except Exception:  # noqa: BLE001 -- fall back to the name this module owns
        return frozenset({"search_conversation", "search_knowledge_base"})


def _without_retrieval(group: list[dict]) -> list[dict]:
    """Retrieval calls match by name: model searches carry plain call_N ids that _is_injected misses."""
    names = _retrieval_names()
    dropped_ids = set()
    out = []
    for message in group:
        calls = message.get("tool_calls") or []
        if calls:
            keep_calls = []
            for call in calls:
                function = (call or {}).get("function") or {}
                if str(function.get("name") or "") in names:
                    dropped_ids.add(str((call or {}).get("id") or ""))
                else:
                    keep_calls.append(call)
            if len(keep_calls) != len(calls):
                if not keep_calls and not _text_of(message.get("content")).strip():
                    continue
                message = {**message, "tool_calls": keep_calls}
        if str(message.get("role") or "") == "tool":
            if str(message.get("tool_call_id") or "") in dropped_ids:
                continue
        message = _without_folded_retrieval(message, dropped_ids)
        if message is None:
            continue
        out.append(message)
    return out


def _without_folded_retrieval(message: dict, dropped_ids: set):
    """Matches the dropped tool_call_ids, not the tool name, so user text that looks like a fold
    survives."""
    if str(message.get("role") or "") != "user":
        return message
    content = message.get("content")
    if isinstance(content, str):
        kept = _text_without_folded_retrieval(content, dropped_ids)
        if kept is None:
            return None
        return {**message, "content": kept}
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) and part.get("type") == "text" else None
        if not isinstance(text, str):
            parts.append(part)
            continue
        kept = _text_without_folded_retrieval(text, dropped_ids)
        if kept is None:
            continue
        parts.append({**part, "text": kept})
    if not parts:
        return None
    return {**message, "content": parts}


def _text_without_folded_retrieval(text: str, dropped_ids: set):
    """``text`` with folded retrieval blocks removed, or None if that was all of it."""
    if '"tool_response"' not in text:
        return text
    kept = []
    for segment in text.split("\n\n"):
        try:
            payload = json.loads(segment)
        except (ValueError, TypeError):
            kept.append(segment)
            continue
        response = payload.get("tool_response") if isinstance(payload, dict) else None
        if not isinstance(response, dict):
            kept.append(segment)
            continue
        call_id = str(response.get("tool_call_id") or "")
        if call_id and call_id in dropped_ids:
            continue
        kept.append(segment)
    if not kept:
        return None
    return "\n\n".join(kept)


def enabled() -> bool:
    """Uses rag_available(), not the RAG_AVAILABLE flag: the vec0 native library can be missing."""
    return bool(config.CONVERSATION_ARCHIVE) and bool(rag_db.rag_available())


def can_archive(thread_id: Optional[str]) -> bool:
    """Saved messages are the only signal: an incognito chat sends a thread_id but is never persisted."""
    if not thread_id or not enabled():
        return False
    try:
        from storage import studio_db
        return bool(studio_db.chat_thread_has_messages(thread_id))
    except Exception:
        return False


def archive_turns(
    thread_id: str,
    evicted: list[dict],
    live: Optional[list[dict]] = None,
    branch: Optional[list[dict]] = None,
) -> int:
    """Idempotent by content hash, so re-archiving evicted turns on later requests writes nothing."""
    if not thread_id or not evicted or not enabled():
        return 0
    if not can_archive(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
        return 0

    # Evictor's own grouper so archived units match dropped units; lazy import avoids a cycle.
    from core.inference.context_window import group_turns

    # Track the turn's span separately: _archivable may drop messages, and the branch check needs it.
    groups = [
        (archivable, len(group))
        for group, archivable in ((group, _archivable(group)) for group in group_turns(evicted))
        if archivable
    ]
    if not groups:
        return 0

    model = config.effective_embedding_model()
    scope = store.conversation_archive_scope(thread_id)
    positions = _transcript_positions(thread_id, branch or live)
    live_positions = _live_positions(live)
    written = 0
    conn = None
    try:
        count = embeddings.token_counter(model)
        conn = rag_db.get_connection()
        # Only a prediction; the authoritative identity check happens under the write lock.
        expected_identity = embeddings.embedding_identity(model)

        # Embed in ONE pass: per-group jobs serialise and delay the reply.
        pending = []
        for group, span in groups:
            text = render_turn(group)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            seats = _occurrences(positions, group)
            budget = _write_budget(positions, seats, live_positions, group)
            if _archived_under(conn, scope, digest, expected_identity, occurrences = budget):
                # Commit here: this path may return before the write lock, its only chance to converge.
                # Bounded by the live-aware budget so no duplicate copy of in-prompt text takes a recall slot.
                _restamp(
                    conn,
                    scope,
                    digest,
                    seats[: max(1, budget)] if seats else seats,
                    commit = True,
                )
                _widen_span(conn, scope, digest, span)
                continue
            chunks = chunk_pages(
                [Page(text = text, page_number = None, char_count = len(text))],
                max_tokens = config.CHUNK_TOKENS,
                overlap = config.CHUNK_OVERLAP,
                count = count,
            )
            if chunks:
                pending.append((group, span, digest, chunks, seats, budget))
        if not pending:
            return 0

        # Identity from the encode itself, so a concurrent embedder swap cannot mislabel vectors.
        vectors, identity = embeddings.encode_with_identity(
            [chunk.text for entry in pending for chunk in entry[3]],
            model_name = model,
            normalize = True,
        )
        offset = 0
        for group, span, digest, chunks, seats, budget in pending:
            group_vectors = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            roles = " + ".join(
                dict.fromkeys(str(message.get("role") or "message") for message in group)
            )
            # Commit after chunks, or a failed chunk write leaves a 'completed' empty row skipped forever.
            # Re-checked under the write lock: concurrent compactions would both insert.
            _write_lock = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                _write_lock = True
            except Exception:
                logger.debug("conversation_archive.no_write_lock", exc_info = True)
            copies = store.documents_by_hash(conn, scope, digest)
            stale = _stale_document(conn, scope, digest, identity, occurrences = budget)
            if stale is _ARCHIVED:
                _restamp(conn, scope, digest, seats, copies = copies)
                if _write_lock:
                    conn.commit()
                # Widen here too: both occurrences can arrive in one compaction.
                _widen_span(conn, scope, digest, span)
                continue
            ordinal = None
            archived_at = None
            archived_rowid = None
            if stale is not None:
                # Replace stale-embedder copy, keeping position, timestamp and rowid: re-embed changes how, not when.
                # NULL ordinal stays NULL; rowid reuse is safe since the row is deleted in this transaction.
                previous = store.document_rewrite_identity(conn, stale) or {}
                ordinal = previous.get("archive_ordinal")
                archived_at = previous.get("created_at")
                archived_rowid = previous.get("rowid")
                store.delete_document(conn, stale, commit = False)
            else:
                # The nth copy of a repeated turn takes the nth occurrence's position.
                ordinal = (
                    seats[len(copies)]
                    if seats and len(copies) < len(seats)
                    else _fallback_ordinal(conn, scope, positions)
                )
            document_id = store.create_document(
                conn,
                scope = scope,
                thread_id = thread_id,
                filename = f"earlier turn ({roles})",
                sha256 = digest,
                status = "completed",
                embedding_model = identity,
                # Real transcript size so the branch check can bound its run exactly.
                archive_messages = span,
                # Allocated under the write lock in group_turns order; created_at cannot order one compaction.
                archive_ordinal = ordinal,
                created_at = archived_at,
                rowid = archived_rowid,
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, group_vectors)
            except Exception:
                conn.rollback()
                raise
            # Surplus only, not a restamp: restamping would renumber a pre-column row.
            _retire_surplus(conn, scope, digest, seats)
            # Restamp ordinals against the remaining seats once a copy is retired, skipping NULLs.
            if stale is not None:
                _restamp(conn, scope, digest, seats, skip_null = True)
            conn.commit()
            written += 1
            # Top copies up to the budget: re-embed replaces, never adds, so evicted repeats were missed.
            while (
                stale is not None
                and live_positions is not None
                and len(store.documents_by_hash(conn, scope, digest)) < budget
            ):
                if not _write_copy(
                    conn,
                    scope = scope,
                    thread_id = thread_id,
                    roles = roles,
                    digest = digest,
                    identity = identity,
                    group = group,
                    span = span,
                    chunks = chunks,
                    vectors = group_vectors,
                    seats = seats,
                ):
                    break
                _retire_surplus(conn, scope, digest, seats)
                conn.commit()
                written += 1
            if _INGEST_FAILED:
                globals()["_INGEST_FAILED"] = False
    except embeddings.EmbeddingModelDownloadRequiredError as exc:
        globals()["_INGEST_FAILED"] = True
        _log_embedder_pending(model, exc)
    except Exception:
        globals()["_INGEST_FAILED"] = True
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Re-check after commit: a concurrent delete sweep could otherwise strand deleted content.
    if written and not can_archive(thread_id):
        logger.info("conversation_archive.thread_deleted_mid_ingest thread_id=%s", thread_id)
        delete_for_thread(thread_id)
        return 0
    return written


_EMBEDDER_PENDING_LOGGED: set = set()


def _log_embedder_pending(model, exc: Exception) -> None:
    if model in _EMBEDDER_PENDING_LOGGED:
        return
    _EMBEDDER_PENDING_LOGGED.add(model)
    logger.info("conversation_archive.embedder_pending: %s", exc)


# Process-wide on purpose: failures are in the embedder or store, not one thread.
_INGEST_FAILED = False


def degraded() -> bool:
    """Whether the last archive write failed; callers stop reserving recall room while it is set."""
    return _INGEST_FAILED


def reachable() -> bool:
    """Probes the store and embedder with a real encode; never memoised, since a cached yes outlives
    them."""
    if not enabled():
        return False
    conn = None
    try:
        conn = rag_db.get_connection()
        # Check WRITABLE with a real statement: BEGIN IMMEDIATE alone succeeds on a read-only connection.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS _archive_write_probe(x)")
        finally:
            conn.rollback()
        model = config.effective_embedding_model()
        embeddings.embedding_identity(model)
        embeddings.token_counter(model)("")
        embeddings.encode(["x"], model_name = model, normalize = True)
        return True
    except Exception:  # noqa: BLE001 -- an unreachable archive is "no", never an error
        logger.debug("conversation_archive.unreachable", exc_info = True)
        return False
    finally:
        # Close explicitly; GC-held connections leak descriptors on rag.db.
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


_ARCHIVED = "archived"


# High cap: the branch the user went back to is older than every abandoned retry leaf.
_BRANCH_SEED_MAX_LEAVES = 512


def _walk_from(by_id: dict, parent_of: dict, leaf) -> list[dict]:
    """The rows from ``leaf`` back to the root, oldest first."""
    chain: list[dict] = []
    seen: set = set()
    current = leaf
    while current is not None and current not in seen:
        seen.add(current)
        record = by_id.get(current)
        if record is None:
            break
        chain.append(record)
        current = parent_of.get(current)
    chain.reverse()
    return chain


def _branch_seed(
    messages: list[dict],
    by_id: dict,
    parent_of: dict,
    branch,
    *,
    require_unique: bool = False,
) -> Optional[str]:
    """Matches the request's text, not the newest stored row, which can belong to an abandoned branch."""
    if not branch:
        return None

    # Ordered list, not set: sets let an abandoned sibling with the same texts tie and win.
    # System/developer messages excluded: prepended instructions are not in the stored chain.
    def _key(message):
        """What a message matches ON. Role-blind let a rewound, unstored "Continue." match an
        ABANDONED assistant reply of the same text and adopt its boundary."""
        text = _normalise_cased(_probe_text(message))
        if not text:
            return None
        return (str(message.get("role") or ""), text) if require_unique else text

    wanted = [
        key
        for key in (
            _key(message)
            for message in _as_wire(branch)
            if str(message.get("role") or "") not in ("system", "developer")
        )
        if key
    ]
    if not wanted:
        return None
    # Lets the scan SKIP entries with no stored counterpart instead of stalling.
    where: dict = {}
    for index, text in enumerate(wanted):
        where.setdefault(text, []).append(index)
    parents = {parent for parent in parent_of.values() if parent is not None}
    order = {message.get("id"): index for index, message in enumerate(messages)}
    leaves = sorted(
        (identifier for identifier in by_id if identifier not in parents),
        key = lambda identifier: order.get(identifier, -1),
        reverse = True,
    )
    if require_unique and len(leaves) > _BRANCH_SEED_MAX_LEAVES:
        return None
    best = None
    best_matched = None
    best_score = 0
    best_tied = False
    rendered: dict = {}

    def _texts_of(record: dict) -> list:
        identifier = record.get("id")
        if identifier is None:
            return [_key(m) for m in _as_wire([record])]
        if identifier not in rendered:
            rendered[identifier] = [_key(m) for m in _as_wire([record])]
        return rendered[identifier]

    for leaf in leaves[:_BRANCH_SEED_MAX_LEAVES]:
        # Gaps allowed on both sides: neither side is a subsequence of the other.
        cursor = 0
        score = 0
        last_matched = None
        for record in _walk_from(by_id, parent_of, leaf):
            for key in _texts_of(record):
                spots = where.get(key) if key else None
                if not spots:
                    continue
                index = bisect.bisect_left(spots, cursor)
                if index < len(spots):
                    cursor = spots[index] + 1
                    score += 1
                    last_matched = record.get("id")
        if score > best_score:
            best, best_score = leaf, score
            best_matched = last_matched
            best_tied = False
        elif require_unique and score == best_score and score > 0:
            best_tied = best_tied or last_matched != best_matched
        if best_score >= len(wanted) and not require_unique:
            break
    if require_unique:
        return None if best_tied else best_matched
    return best


def _active_chain(
    messages: list[dict],
    branch = None,
    *,
    fallback: bool = True,
    require_unique: bool = False,
) -> list[dict]:
    """Walks newest leaf back to root: a flat list would mix in siblings left by Retry."""
    if not messages:
        return []
    by_id: dict = {}
    parent_of: dict = {}
    synthesized: set = set()
    previous = None
    for message in messages:
        identifier = message.get("id")
        if identifier is None:
            continue
        by_id[identifier] = message
        parent = message.get("parentId") or message.get("parent_id")
        # A row that has the column with null is an intended root; pre-column rows use storage order.
        stated = require_unique and ("parentId" in message or "parent_id" in message)
        if parent is None and previous is not None and not stated:
            synthesized.add(identifier)
            parent = previous
        parent_of[identifier] = parent
        previous = identifier
    if not by_id:
        return list(messages) if fallback else []
    if require_unique and synthesized:
        # Storage order is not ancestry: decline invented links between indistinguishable twins only.
        seen: set = set()
        for message in messages:
            if message.get("id") is None:
                continue
            key = (message.get("role"), _normalise_cased(_probe_text(message)))
            if key[1] and key in seen:
                return []
            seen.add(key)
    # Fall back to the newest row: empty positions send every turn to MAX + 1.
    seed = _branch_seed(messages, by_id, parent_of, branch, require_unique = require_unique)
    if seed is None:
        if not fallback:
            return []
        seed = messages[-1].get("id")
    return _walk_from(by_id, parent_of, seed) or list(messages)


# JSON.stringify spelling (no space after colon); comparisons downstream are exact.
_EMPTY_TOOL_RESULT = '{"result":""}'


# Mirrors SERVER_SIDE_BUILTIN_TOOL_NAMES (chat-adapter.ts); name alone never decides, as the frontend also
# requires.
_SERVER_BUILTIN_NAMES = frozenset({"web_search", "web_fetch", "code_execution", "image_generation"})
# `SANDBOX_FILE_TOOLS`, and `tool_loop_controller._FILE_SENTINEL_TOOLS`. Only these wrap.
_SANDBOX_TOOL_NAMES = frozenset({"python", "terminal", "edit_file"})

# Mirrors search-images.ts. re.ASCII: Python IGNORECASE folds Unicode, JS /i does not.
_SEARCH_IMAGE_ID = re.compile(r"[0-9a-f]{12}")
_SEARCH_IMAGE_SOURCE = re.compile(r"https?://", re.IGNORECASE | re.ASCII)
_SEARCH_IMAGE_TOKEN = re.compile(
    r"\n\n[ \t]*\[\[img:[0-9a-f]{12}\]\][ \t]*(?=\n\n|\n?\Z)|\[\[img:[0-9a-f]{12}\]\]"
)
# Mirrors sanitizeAssistantReplayText: inline audio data URIs are sent as [audio].
_REPLAY_AUDIO_DATA_URI = re.compile(r"data:audio/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def _server_builtin(part: dict) -> tuple[bool, bool]:
    """Flags provider builtins and native parts; reading one as a local call adds a phantom exchange."""
    args = part.get("args")
    args = args if isinstance(args, dict) else {}
    if str(part.get("toolName") or "").lower() not in _SERVER_BUILTIN_NAMES:
        return False, False
    google = args.get("google")
    native = isinstance(google, dict) and isinstance(google.get("native_part"), dict)
    return bool(args.get("_server_tool") is True or native), native


# Mirrors frontend predicates; sessionId/subject tested for ABSENCE (frontend checks undefined).
def _mcp_image_result(result) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        return False
    images = result.get("images")
    return (
        "sessionId" not in result
        and isinstance(images, list)
        and bool(images)
        and all(
            isinstance(image, dict)
            and isinstance(image.get("data"), str)
            and isinstance(image.get("mimeType"), str)
            for image in images
        )
    )


def _search_image_entry(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        isinstance(entry.get("id"), str)
        and _SEARCH_IMAGE_ID.fullmatch(entry["id"]) is not None
        and isinstance(entry.get("title"), str)
        and isinstance(entry.get("domain"), str)
        and isinstance(entry.get("source"), str)
        and _SEARCH_IMAGE_SOURCE.match(entry["source"]) is not None
        and ("subject" not in entry or isinstance(entry["subject"], str))
    )


def _search_images_result(result) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        return False
    entries = result.get("webImages")
    return (
        isinstance(entries, list)
        and bool(entries)
        and all(_search_image_entry(entry) for entry in entries)
    )


def _sandbox_wrapper(result, tool_name: str) -> bool:
    # Name gates it too: another tool with this shape is not ours to unwrap.
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        return False
    if tool_name not in _SANDBOX_TOOL_NAMES or not isinstance(result.get("sessionId"), str):
        return False
    if not isinstance(result.get("images"), list):
        return False
    files = result.get("files")
    return files is None or (
        isinstance(files, list)
        and all(isinstance(f, dict) and isinstance(f.get("name"), str) for f in files)
    )


def _strip_search_image_tokens(text: str) -> str:
    """Drops [[img: tokens, which resolve only against the message that produced them."""
    if "[[img:" not in text:
        return text
    return _SEARCH_IMAGE_TOKEN.sub("", text)


def _sanitised_assistant_text(text: str) -> str:
    """Strips image tokens and audio data URIs, as the serializer does, so replies match their
    transcript."""
    return _REPLAY_AUDIO_DATA_URI.sub("[audio]", _strip_search_image_tokens(text))


def _sanitised_assistant_content(content):
    """`content` with every part `_probe_text` reads as text sanitised, others untouched. Keyed on
    the `text` field rather than on the type, as `_probe_text` is, so a part shape it renders
    cannot slip past this one."""
    if isinstance(content, str):
        return _sanitised_assistant_text(content)
    if not isinstance(content, list):
        return content
    return [
        {**part, "text": _sanitised_assistant_text(part["text"])}
        if isinstance(part, dict)
        and part.get("type") not in ("reasoning", "tool-call")
        and isinstance(part.get("text"), str)
        else part
        for part in content
    ]


def _unwrapped(result, tool_name: str):
    """Reduces an app wrapper to the text the model saw, since the adapter replays only that text."""
    if not (
        _mcp_image_result(result)
        or _search_images_result(result)
        or _sandbox_wrapper(result, tool_name)
    ):
        return None
    text = result["text"]
    return _strip_search_image_tokens(text) if _search_images_result(result) else text


def _tool_result_content(result, tool_name: str = "") -> str:
    """Empty results become a sentinel, since the backend rejects empty tool content."""
    if isinstance(result, str):
        return result if result else _EMPTY_TOOL_RESULT
    unwrapped = _unwrapped(result, tool_name)
    if unwrapped is not None:
        return unwrapped if unwrapped else _EMPTY_TOOL_RESULT
    # ensure_ascii=False to match JSON.stringify, or non-ASCII results lose their transcript seat.
    return json.dumps(result, ensure_ascii = False, separators = (",", ":"))


def _replayable(part: dict) -> bool:
    """Drops a call with no result unless it is a builtin with a native part, as the frontend does."""
    builtin, native = _server_builtin(part)
    if builtin:
        return native
    return part.get("result") is not None


def _local_round_id(part: dict):
    """`codexLocalToolRoundId`: the round a LOCAL tool call belongs to, or None."""
    provenance = part.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "local":
        return None
    round_id = provenance.get("round_id")
    if isinstance(round_id, bool) or not isinstance(round_id, int):
        return None
    return round_id


def _flushes_local_pair(part: dict) -> bool:
    """A completed local call is flushed before and after, so adjacent ones replay as separate rounds."""
    provenance = part.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "local":
        return False
    if _server_builtin(part)[0]:
        return False
    return part.get("result") is not None


def _as_wire(messages: list[dict], sanitise_assistant: bool = True) -> list[dict]:
    """Splits stored tool-call parts into wire form; pass sanitise_assistant=False for request messages."""
    wire: list[dict] = []
    for message in messages:
        content = message.get("content")
        parts = content if isinstance(content, list) else None
        # Drop reasoning parts: _probe_text would render them inline and the turn matches no seat.
        if parts is not None and any(
            isinstance(part, dict) and part.get("type") == "reasoning" for part in parts
        ):
            parts = [
                part
                for part in parts
                if not (isinstance(part, dict) and part.get("type") == "reasoning")
            ]
            content = parts
        if sanitise_assistant and str(message.get("role") or "") == "assistant":
            content = _sanitised_assistant_content(content)
            parts = content if isinstance(content, list) else None
        # A builtin with no native part is not replayed, so it is not a call here either.
        calls = [
            part
            for part in (parts or [])
            if isinstance(part, dict) and part.get("type") == "tool-call"
        ]
        if not calls:
            wire.append(
                {
                    "role": message.get("role"),
                    "content": content,
                    "tool_calls": message.get("tool_calls"),
                }
            )
            continue
        # Replayed like chat-adapter.ts: parts in order, flushing pending calls whenever text arrives.
        pending_calls: list[dict] = []
        pending_text: list = []
        # One-slot box: _flush resets it and closures cannot rebind.
        pending_round: list = [None]

        def _flush(role = message.get("role")) -> None:
            pending_round[0] = None
            if not pending_calls and not pending_text:
                return
            body: list = [
                {key: value for key, value in call.items() if key != "result"}
                for call in pending_calls
            ] + pending_text
            entry: dict = {"role": role, "content": body}
            if pending_calls:
                entry["tool_calls"] = [
                    {"id": call.get("toolCallId"), "function": {}} for call in pending_calls
                ]
            wire.append(entry)
            for call in pending_calls:
                if _server_builtin(call)[0]:
                    # A builtin with a native part replays as a call with no tool message.
                    continue
                if "result" not in call or call.get("result") is None:
                    # Only undefined/null are absent (serializer rule); "" / {} / [] are carried.
                    continue
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("toolCallId"),
                        "content": _tool_result_content(
                            call.get("result"), str(call.get("toolName") or "")
                        ),
                    }
                )
            pending_calls.clear()
            pending_text.clear()

        for part in parts:
            if isinstance(part, dict) and part.get("type") == "tool-call":
                if not _replayable(part):
                    continue
                round_id = _local_round_id(part)
                if (
                    pending_calls
                    and pending_round[0] is not None
                    and round_id is not None
                    and pending_round[0] != round_id
                ):
                    # startsNewCodexToolRound: a new local round is a new group.
                    _flush()
                if round_id is not None:
                    pending_round[0] = round_id
                flush_pair = round_id is None and _flushes_local_pair(part)
                if flush_pair and pending_calls:
                    _flush()
                pending_calls.append(part)
                if flush_pair:
                    _flush()
                continue
            if pending_calls:
                _flush()
            pending_text.append(part)
        _flush()
    return wire


def _fallback_ordinal(conn, scope: str, positions: Optional[list[list[str]]]) -> int:
    """A turn matching no seat takes the larger of the archive's next ordinal and the transcript length."""
    return max(store.next_archive_ordinal(conn, scope), len(positions or []))


def _transcript_positions(thread_id: str, branch = None) -> Optional[list[str]]:
    """Numbers come from the saved transcript, not archive time, since eviction is not oldest-first."""
    try:
        from core.inference.context_window import group_turns
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    wire = _as_wire(_active_chain(messages, branch))
    return [
        [_normalise_cased(_probe_text(message)) for message in group]
        for group in group_turns(wire)
        if group
    ]


def _occurrences(positions: Optional[list[list[str]]], group: list[dict]) -> list[int]:
    """Matches the whole turn, not its first line, so turns that merely start alike do not share seats."""
    if not positions or not group:
        return []
    texts = [_normalise_cased(_probe_text(message)) for message in group]
    if not texts or not texts[0]:
        return []
    calls = [bool(message.get("tool_calls")) for message in group]

    def _same(stored: str, live: str, is_call: bool) -> bool:
        # Tool calls compared as needle in haystack (stored args are an object, live a raw string);
        # split once since _probe_text inserts the second spelling mid-string.
        if is_call:
            if not live:
                return False
            if live in stored:
                return True
            words = live.split(" ")
            for cut in range(1, len(words)):
                head = " ".join(words[:cut])
                tail = " ".join(words[cut:])
                at = stored.find(head)
                if at >= 0 and stored.find(tail, at + len(head)) >= 0:
                    return True
            return False
        return stored == live

    return [
        index
        for index, position in enumerate(positions)
        if position
        # Shorter positions may only match the trailing one, or an orphan user row takes a second seat.
        and (len(position) >= len(texts) or index == len(positions) - 1)
        and all(
            _same(stored, live, is_call) for stored, live, is_call in zip(position, texts, calls)
        )
    ]


def _write_copy(
    conn,
    *,
    scope: str,
    thread_id: str,
    roles: str,
    digest: str,
    identity: str,
    group: list[dict],
    span: int,
    chunks,
    vectors,
    seats: list[int],
) -> bool:
    """span is the turn's transcript size, not len(group), since archiving strips retrieval parts."""
    copies = store.documents_by_hash(conn, scope, digest)
    if not seats or len(copies) >= len(seats):
        return False
    document_id = store.create_document(
        conn,
        scope = scope,
        thread_id = thread_id,
        filename = f"earlier turn ({roles})",
        sha256 = digest,
        status = "completed",
        embedding_model = identity,
        archive_messages = span,
        archive_ordinal = seats[len(copies)],
        commit = False,
    )
    try:
        store.add_chunks(conn, scope, document_id, chunks, vectors)
    except Exception:
        conn.rollback()
        raise
    return True


def _live_positions(live: Optional[list[dict]]) -> Optional[list[list[str]]]:
    """The fitted conversation grouped exactly as ``_transcript_positions`` groups the stored one,
    so a live turn can be matched by the same rules a stored seat is."""
    if live is None:
        return None
    try:
        from core.inference.context_window import group_turns
    except Exception:
        return None
    positions = [
        [_normalise_cased(_probe_text(message)) for message in group]
        for group in group_turns(_as_wire(live))
        if group
    ]
    return positions or None


def _write_budget(
    positions: Optional[list[list[str]]],
    seats: list[int],
    live_positions: Optional[list[list[str]]],
    group: Optional[list[dict]] = None,
) -> int:
    """Counts occurrences still in the live prompt, so a turn still shown is not archived again."""
    if not seats:
        return 1
    if not live_positions or not positions:
        return len(seats)
    live = len(_occurrences(live_positions, group)) if group else 0
    return max(len(seats) - live, 1)


def _retire_surplus(
    conn,
    scope: str,
    digest: str,
    seats: list[int],
    *,
    rows = None,
) -> bool:
    """Deletes surplus copies after a rewind; with no seats it does nothing, so no copy is lost."""
    if not seats:
        return False
    try:
        copies = rows if rows is not None else store.documents_by_hash(conn, scope, digest)
        surplus = copies[len(seats) :]
        for copy in surplus:
            store.delete_document(conn, copy["id"], commit = False)
        return bool(surplus)
    except Exception:  # noqa: BLE001 -- tidying an archive is not worth a chat
        logger.debug("conversation_archive.retire_surplus_failed", exc_info = True)
        return False


def _restamp(
    conn,
    scope: str,
    digest: str,
    seats: list[int],
    *,
    copies = None,
    commit: bool = False,
    skip_null: bool = False,
) -> None:
    """Converges archives numbered by arrival or not at all, via UPDATE, so nothing is duplicated."""
    if not seats:
        return
    try:
        rows = copies if copies is not None else store.documents_by_hash(conn, scope, digest)
        moved = False
        for seat, copy in zip(seats, rows):
            if skip_null and copy.get("archive_ordinal") is None:
                # Pre-column rows stay unnumbered, or the oldest statement is shown as the latest.
                continue
            if copy.get("archive_ordinal") != seat:
                store.set_archive_ordinal(conn, copy["id"], seat)
                moved = True
        moved = _retire_surplus(conn, scope, digest, seats, rows = rows) or moved
        if moved and commit:
            conn.commit()
    except Exception:  # noqa: BLE001 -- ordering an old archive is not worth a chat
        logger.debug("conversation_archive.restamp_failed", exc_info = True)


def _stale_document(
    conn,
    scope: str,
    digest: str,
    identity: str,
    *,
    occurrences: int = 1,
):
    """Hash alone is not enough: the embedder must match, and each repeated occurrence needs a copy."""
    copies = store.documents_by_hash(conn, scope, digest)
    if not copies:
        return None
    for copy in copies:
        if not config.embedding_identity_matches(copy.get("embedding_model"), identity):
            return copy["id"]
    return _ARCHIVED if len(copies) >= max(1, occurrences) else None


def _archived_under(
    conn,
    scope: str,
    digest: str,
    identity: str,
    *,
    occurrences: int = 1,
) -> bool:
    """Cheap pre-check before the chunking and embedding pass."""
    return _stale_document(conn, scope, digest, identity, occurrences = occurrences) is _ARCHIVED


def _widen_span(conn, scope: str, digest: str, span: int) -> None:
    """Only widens: a stored window is a maximum, so a longer span still matches the shorter turn."""
    try:
        conn.execute(
            "UPDATE documents SET archive_messages = ? "
            "WHERE scope = ? AND sha256 = ? "
            "AND (archive_messages IS NULL OR archive_messages < ?)",
            (int(span), scope, digest, int(span)),
        )
        conn.commit()
    except Exception:
        logger.debug("conversation_archive.widen_span_failed", exc_info = True)


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
    """Trims both ends and keeps case: a capitalisation-only edit must retire the stale copy."""
    return (text or "").strip()


def _normalise_cased(text: str) -> str:
    """Collapses whitespace but keeps case, since casing-only variants are different turns."""
    return " ".join((text or "").split())


def _live_transcript(thread_id: str) -> Optional[list[str]]:
    """Keeps recall on the user's branch, since the append-only archive also holds abandoned turns."""
    try:
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    # Same reconstruction, or persisted tool calls render in a different order.
    texts = [_normalise(_probe_text(message)) for message in _as_wire(messages)]
    return [text for text in texts if text] or None


def branch_message_texts(
    messages: Optional[list[dict]], roles: Optional[tuple[str, ...]] = None
) -> Optional[list[str]]:
    """Taken from the request, not the stored DAG, which keeps abandoned siblings; one string per
    message."""
    if roles:
        messages = [
            message for message in (messages or []) if str(message.get("role") or "") in roles
        ]
    if not messages:
        return None
    texts = [_normalise(_probe_text(message)) for message in messages]
    return [text for text in texts if text] or None


def message_text(content) -> str:
    """One stored message's content, normalised the way the branch check normalises it. Exposed so
    the rolling window compares stored messages on exactly these terms rather than inventing a
    second notion of "same text"."""
    return _normalise(_probe_text({"content": content}))


def content_on_branch(content, transcript: Optional[list[str]]) -> bool:
    """Stored rows span all branches, so a newest reply may be a sibling; empty text counts as on-branch."""
    if not transcript:
        return True
    text = _normalise(_probe_text({"content": content}))
    return not text or any(text in message for message in transcript)


_ROLE_PREFIX = re.compile(
    r"^(?:user|assistant|system|developer|tool result|message):\s*", re.IGNORECASE
)
# Strip our 'assistant called <name>:' label before probing (name kept if no args).
_TOOL_CALL_PREFIX = re.compile(r"^assistant called (?:[^:\n]+:\s*)?", re.IGNORECASE)
# Name captured: a retry that changed only the tool must not match.
_TOOL_CALL_NAME = re.compile(r"^assistant called ([^:\n]+):\s*", re.IGNORECASE)
# Between probes of one message only a stripped label may sit; other text is unseen content.
_GAP_IS_LABEL = re.compile(
    r"^\s*(?:(?:user|assistant|system|developer|tool result|message):"
    r"|assistant called (?:[^:\n]+:)?)?\s*$",
    re.IGNORECASE,
)


def _probes_for(text: str) -> list[str]:
    """Strips the role labels render_turn writes, which exist only in the archived copy, before matching."""
    probes = []
    for line in (text or "").splitlines():
        without_role = _ROLE_PREFIX.sub("", line)
        named = _TOOL_CALL_NAME.match(without_role.strip())
        if named:
            # The tool name is its own probe; except 'tool', render_turn's fallback for nameless calls.
            name = _normalise(named.group(1))
            if name and name != "tool":
                probes.append(name)
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", without_role))
        # Whole line unless truncated; keyed off the marker since long tool results span many lines.
        if stripped.endswith(_TRUNCATION_MARKER.strip()):
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        probes.append(stripped)
    return [probe for probe in probes if probe]


def _probe_entries(text: str) -> list[tuple[str, bool]]:
    """Flags lines cut short or tool calls, since the live message may hold more than the probe."""
    entries = []
    for line in (text or "").splitlines():
        without_role = _ROLE_PREFIX.sub("", line)
        is_call = bool(_TOOL_CALL_PREFIX.match(without_role.strip()))
        named = _TOOL_CALL_NAME.match(without_role.strip())
        if named:
            # See _probes_for: the tool name is a probe; 'tool' is render_turn's nameless fallback.
            name = _normalise(named.group(1))
            if name and name != "tool":
                entries.append((name, True))
        stripped = _normalise(_TOOL_CALL_PREFIX.sub("", without_role))
        truncated = stripped.endswith(_TRUNCATION_MARKER.strip())
        if truncated:
            stripped = stripped[: -len(_TRUNCATION_MARKER.strip())].strip()
        if stripped:
            entries.append((stripped, truncated or is_call))
        elif truncated and entries:
            # Cut on a line boundary: the marker is its own line, so flag the previous line as truncated.
            entries[-1] = (entries[-1][0], True)
    return entries


def _on_live_branch(text: str, transcript: Optional[list[str]]) -> bool:
    """Whether one archived chunk still exists in the saved thread."""
    probes = _probes_for(text)
    if not probes or not transcript:
        return False
    # Every line, in order, within one bounded run of adjacent messages, so edits cannot slip past.
    window = len(probes)
    return any(
        _probes_match_from(probes, transcript, start, window) for start in range(len(transcript))
    )


def _scan_probes(
    entries: list[tuple[str, bool]], messages: list[str], start: int, last: int
) -> Optional[tuple[int, int, int, bool, int]]:
    """Reports the message a run opened on: an overlapping chunk's tail can begin in an earlier message."""
    index = start
    cursor = 0
    opened_at = None
    opened_index = None
    partial = False
    fresh = False
    for probe, partial_ok in entries:
        while index < last:
            found = messages[index].find(probe, cursor)
            if found >= 0:
                # A message stepped into must match from char 0, except tool calls (stored structured).
                if fresh and found != 0 and not partial_ok:
                    return None
                # Reject text inserted between two probes unless it is a render_turn label the probe stripped.
                if (
                    not fresh
                    and cursor
                    and not (partial or partial_ok)
                    and not _GAP_IS_LABEL.match(messages[index][cursor:found])
                ):
                    return None
                if opened_at is None:
                    opened_at = 0 if partial_ok else found
                    opened_index = index
                cursor = found + len(probe)
                # The exemption belongs to the call only; after a text probe the cursor is exact again.
                partial = partial_ok
                fresh = False
                break
            if cursor and not partial and cursor < len(messages[index]):
                return None
            index += 1
            cursor = 0
            partial = False
            fresh = True
        else:
            return None
    return (
        index,
        cursor,
        (opened_at or 0),
        partial,
        (opened_index if opened_index is not None else start),
    )


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
    """Counts labelled lines, since render_turn labels every message; bounds the run a turn may occupy."""
    total = 0
    for row in rows:
        for line in (row["text"] or "").splitlines():
            stripped = line.strip()
            if _ROLE_PREFIX.match(stripped) or _TOOL_CALL_PREFIX.match(stripped):
                total += 1
    return total


def _document_on_live_branch(conn, document_id: str, transcript: list[str], cache: dict) -> bool:
    """Every chunk must still be on the branch: an edit to any part of a turn retires the whole copy."""
    if document_id in cache:
        return cache[document_id]
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        row = conn.execute(
            "SELECT archive_messages FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        message_count = row["archive_messages"] if row else None
    except Exception:
        cache[document_id] = True
        return True
    cache[document_id] = _document_matches_one_run(rows, transcript, message_count)
    return cache[document_id]


def _document_matches_one_run(
    rows,
    transcript: Optional[list[str]],
    message_count: Optional[int] = None,
) -> bool:
    """Every chunk must match within one run of adjacent messages, not be assembled from scattered parts."""
    if not rows or not transcript:
        return False
    probe_lists = [_probe_entries(row["text"]) for row in rows]
    if any(not probes for probes in probe_lists):
        return False
    # Bound by source messages, not lines; label count is a fallback for older documents.
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
        # Next chunk may restart where the previous opened: its overlap can carry a whole short message.
        floor = start
        for probes in probe_lists:
            found = None
            for candidate in range(position, floor - 1, -1):
                found = _scan_probes(probes, transcript, candidate, last)
                if found is not None:
                    break
            if found is None:
                return False
            position, cursor, chunk_opened_at, partial_tail, floor = found
            # Only place that sees an edit prepending to the turn's first message.
            if opened_at is None:
                opened_at = chunk_opened_at
        # The turn must cover its messages end to end, or an appended correction leaves it eligible.
        if opened_at:
            return False
        return partial_tail or cursor >= len(transcript[position])

    return any(_one_run_from(start) for start in range(len(transcript)))


def _order_key(ordinal, created_at, document_rowid, chunk_index) -> tuple:
    """One key for both snake_case columns and camelCase keys, so the recall order cannot drift."""
    created = created_at or ""
    rowid = document_rowid or 0
    index = chunk_index or 0
    if ordinal is None:
        return (0, 0, created, rowid, index)
    return (1, int(ordinal), created, rowid, index)


def _conversation_order(row) -> tuple:
    """NULL ordinals sort first; then created_at, then rowid, since coarse clocks tie created_at."""
    if row is None:
        return (2, 0, "", 0, 0)
    return _order_key(
        tool._row_value(row, "archive_ordinal"),
        tool._row_value(row, "created_at"),
        tool._row_value(row, "document_rowid"),
        tool._row_value(row, "chunk_index"),
    )


def _above_floor(hits: list, min_dense_score: float) -> list:
    """Applies only to forced lookups; lexical-only hits are kept, since they have no score to gate."""
    if min_dense_score <= 0:
        return hits
    return [hit for hit in hits if hit.dense_score is None or hit.dense_score >= min_dense_score]


def _ends_first_within_ties(conn, hits: list) -> list:
    """Alternates newest and oldest within tied scores, since newest-first alone loses the original
    value."""
    if not config.CONVERSATION_QUERY_FOCUS:
        # Reordering is selection, so it stays behind the rollback knob.
        return hits
    if len(hits) < 2:
        return hits
    if len({hit.lexical_score for hit in hits}) == len(hits):
        return hits
    try:
        rows = store.chunks_by_id(conn, [hit.chunk_id for hit in hits])
    except Exception:  # noqa: BLE001 -- ordering must never break a recall
        return hits
    ordered: list = []
    run: list = []

    def _flush():
        if not run:
            return
        by_time = sorted(run, key = lambda hit: _conversation_order(rows.get(hit.chunk_id)))
        while by_time:
            ordered.append(by_time.pop())
            if by_time:
                ordered.append(by_time.pop(0))

    for hit in hits:
        if run and run[-1].lexical_score != hit.lexical_score:
            _flush()
            run = []
        run.append(hit)
    _flush()
    return ordered


def _lexical_pass(
    conn,
    scope: str,
    query: str,
    model,
    k: int,
    expression,
    *,
    newest_first: bool = False,
    oldest_first: bool = False,
) -> list:
    if newest_first or oldest_first:
        return _ends_first_within_ties(
            conn,
            retrieval.retrieve_lexical(
                conn,
                scope,
                query,
                k,
                match_query = expression,
                newest_first = newest_first,
                oldest_first = oldest_first,
            ),
        )
    return _ends_first_within_ties(
        conn,
        retrieval.retrieve_hybrid(
            conn,
            scope,
            query,
            k = k,
            model_name = model,
            mode = "lexical",
            lexical_query = expression,
        ),
    )


def _both_ends(oldest: list, newest: list) -> list:
    """One candidate list from a run fetched from each end, oldest side first, no repeats."""
    seen = {hit.chunk_id for hit in oldest}
    return oldest + [hit for hit in newest if hit.chunk_id not in seen]


def _focused_lexical(conn, scope: str, query: str, model, fetch: int) -> list:
    """Identifiers filter candidates and content words rank them, as BM25 is flat on common identifiers."""
    expressions = (
        store.conversation_match_queries(query) if config.CONVERSATION_QUERY_FOCUS else [None]
    )
    # Only the kill switch takes the plain path; a single-identifier query needs two-ended fetch.
    if not expressions or expressions[0] is None:
        return _lexical_pass(conn, scope, query, model, fetch, None)
    # Fetch both ends of the tied run: SQLite returns ties in rowid order, so LIMIT hides the newest.
    _newest_half = _BRANCH_FILTER_MAX_CANDIDATES // 2
    # Re-order over the merged run, not per half.
    strict = _ends_first_within_ties(
        conn,
        _both_ends(
            _lexical_pass(
                conn,
                scope,
                query,
                model,
                _BRANCH_FILTER_MAX_CANDIDATES - _newest_half,
                expressions[0],
                oldest_first = True,
            ),
            _lexical_pass(
                conn, scope, query, model, _newest_half, expressions[0], newest_first = True
            ),
        ),
    )
    if len(expressions) < 2:
        return strict
    # Rank pass fetched to the filter bound, not fetch, or it can be filled by ineligible chunks.
    _loose_k = max(fetch, _BRANCH_FILTER_MAX_CANDIDATES)
    _loose_newest = _loose_k // 2
    loose = _ends_first_within_ties(
        conn,
        _both_ends(
            _lexical_pass(
                conn,
                scope,
                query,
                model,
                _loose_k - _loose_newest,
                expressions[-1],
                oldest_first = True,
            ),
            _lexical_pass(
                conn, scope, query, model, _loose_newest, expressions[-1], newest_first = True
            ),
        ),
    )
    # Ask the index for eligibility: the strict pass is capped and arbitrarily ordered.
    eligible = {hit.chunk_id for hit in strict}
    try:
        eligible |= store.lexical_matching_ids(
            conn, [hit.chunk_id for hit in loose], expressions[0]
        )
    except Exception:
        logger.warning("conversation_archive.eligibility_probe_failed", exc_info = True)
    ranked = [hit for hit in loose if hit.chunk_id in eligible]
    already = {hit.chunk_id for hit in ranked}
    ranked += [hit for hit in strict if hit.chunk_id not in already]
    already |= {hit.chunk_id for hit in strict}
    ranked += [hit for hit in loose if hit.chunk_id not in already]
    return ranked


def _candidates(conn, scope: str, query: str, model, fetch: int, thread_id: str) -> list:
    """Lexical runs twice with an identifier: once requiring it, then over content words for order."""
    hits: list = []
    seen: set = set()
    for hit in _focused_lexical(conn, scope, query, model, fetch):
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
        except embeddings.EmbeddingModelDownloadRequiredError as exc:
            _log_embedder_pending(model, exc)
        except Exception:
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
    """Lexical first, since recall is mostly exact match; no relevance floor, as weak matches beat none."""
    query = (query or "").strip()
    if not thread_id or not query or not enabled():
        return None
    min_dense_score = config.CONVERSATION_FORCED_MIN_SCORE if forced else 0.0

    scope = store.conversation_archive_scope(thread_id)
    limit = top_k or config.CONVERSATION_ARCHIVE_TOP_K
    # Separate queries: concatenating would dilute identifiers and AND unrelated intents.
    queries = [query] + [q.strip() for q in (extra_queries or []) if (q or "").strip()]
    if len(queries) > 1:
        share = max(1, -(-limit // len(queries)))
        merged: list = []
        seen_ids: set = set()
        for index, one in enumerate(reversed(queries)):
            room = limit - len(merged) if index == len(queries) - 1 else share
            if room <= 0:
                break
            # Over-fetch by what is held: overlapping queries otherwise lose slots to dedup.
            found = recall(
                thread_id,
                one,
                top_k = room + len(seen_ids),
                branch_messages = branch_messages,
                forced = forced,
            )
            if not found:
                continue
            fresh = [source for source in found[1] if source["chunkId"] not in seen_ids]
            # Sort by retrieval rank first: rounded scores tie and preserve chronological order.
            fresh.sort(
                key = lambda source: (
                    source.get("rank") if source.get("rank") is not None else 1 << 30,
                    -(source.get("score") or 0.0),
                )
            )
            for source in fresh[:room]:
                seen_ids.add(source["chunkId"])
                merged.append(source)
        if not merged:
            return None
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            # Must be the exact key _conversation_order uses.
            merged.sort(
                key = lambda source: _order_key(
                    source.get("turn"),
                    source.get("createdAt"),
                    source.get("documentRowid"),
                    source.get("chunkIndex"),
                )
            )
            kept = merged[:limit]
            return tool.render_conversation_sources(kept), kept
        kept = merged[:limit]
        return tool.render_sources(kept), kept
    conn = None
    try:
        conn = rag_db.get_connection()
        model = config.effective_embedding_model()
        transcript = branch_message_texts(branch_messages) or _live_transcript(thread_id)
        fetch = limit * _BRANCH_FILTER_OVERFETCH
        rows: dict = {}
        hits: list = []
        live_documents: dict = {}
        while True:
            fetched = _candidates(conn, scope, query, model, fetch, thread_id)
            if not fetched:
                return None
            # Floor filters candidates before the limit cut; exhausted reads the raw fetch.
            exhausted = len(fetched) < fetch
            candidates = _above_floor(fetched, min_dense_score)
            if not candidates:
                if exhausted or fetch >= _BRANCH_FILTER_MAX_CANDIDATES:
                    logger.info(
                        "conversation_archive.recall_below_floor thread_id=%s floor=%.2f",
                        thread_id,
                        min_dense_score,
                    )
                    return None
                fetch = min(_BRANCH_FILTER_MAX_CANDIDATES, fetch * _BRANCH_FILTER_OVERFETCH)
                continue
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
            if len(hits) >= limit or exhausted or fetch >= _BRANCH_FILTER_MAX_CANDIDATES:
                hits = hits[:limit]
                break
            fetch = min(_BRANCH_FILTER_MAX_CANDIDATES, fetch * _BRANCH_FILTER_OVERFETCH)
        if not hits:
            return None
        if config.CONVERSATION_RECALL_ORDER == "chronological":
            # Keep rank: rounded scores tie, so the merge refill would take the oldest.
            rank_of = {hit.chunk_id: rank for rank, hit in enumerate(hits)}
            # Sort AFTER the top-k slice, or the slice takes the oldest turns.
            hits.sort(key = lambda hit: _conversation_order(rows.get(hit.chunk_id)))
            text, sources = tool.format_conversation_recall(rows, hits)
            for source in sources:
                source["rank"] = rank_of.get(source.get("chunkId"))
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


def _scope_select(scope: str, created_before: Optional[str]) -> tuple:
    """The scope's documents, optionally cut at ``created_before`` (see `delete_for_thread`)."""
    if not created_before:
        return ("SELECT id, stored_path FROM documents WHERE scope=?", (scope,))
    return (
        "SELECT id, stored_path FROM documents WHERE scope=? AND created_at<?",
        (scope, created_before),
    )


def _delete_scope_without_vec(
    scope: str,
    thread_id: str,
    *,
    created_before: Optional[str] = None,
) -> list:
    """Must not depend on sqlite-vec: a silent no-op here would leave deleted turns retrievable."""
    conn = None
    removed = []
    try:
        conn = rag_db.get_metadata_connection()
        documents = conn.execute(*_scope_select(scope, created_before)).fetchall()
        for row in documents:
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (row["id"],),
            )
            conn.execute("DELETE FROM chunks WHERE document_id=?", (row["id"],))
            conn.execute("DELETE FROM documents WHERE id=?", (row["id"],))
        conn.commit()
        removed = [row["stored_path"] for row in documents]
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


def delete_for_thread(thread_id: str, *, created_before: Optional[str] = None) -> int:
    """Bounded by created_before, so a reused thread id keeps the archive written since the delete."""
    if not thread_id:
        return 0
    scope = store.conversation_archive_scope(thread_id)
    return len(_delete_scope(scope, thread_id, created_before = created_before))


def delete_thread_documents(thread_id: str, *, created_before: Optional[str] = None) -> int:
    """Drop a thread's uploaded documents and their stored files, cut as in `delete_for_thread`."""
    if not thread_id:
        return 0
    from .ingestion import _remove_upload

    scope = store.thread_scope(thread_id)
    removed = _delete_scope(scope, thread_id, created_before = created_before)
    for stored_path in removed:
        _remove_upload(stored_path)
    return len(removed)


def copy_thread_documents(source_thread_id: str, thread_id: str) -> tuple[dict[str, str], bool]:
    """Copies files before the transaction, so rag.db's write lock is never held across file I/O."""
    from .ingestion import _copy_upload, _remove_upload

    scope = store.thread_scope(thread_id)
    copied: list = []
    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE scope=? ORDER BY created_at",
            (store.thread_scope(source_thread_id),),
        ).fetchall()
        documents = [dict(row) for row in rows if row["status"] == "completed"]
        for document in documents:
            copied.append(_copy_upload(document["stored_path"]))
        conn.execute("BEGIN IMMEDIATE")
        sources = []
        for document, stored_path in zip(documents, copied):
            source = store.get_document(conn, document["id"])
            if source is None or source["status"] != "completed":
                raise RuntimeError("Source document changed while its file was being copied")
            sources.append((source, stored_path))
        document_ids = store.copy_documents(conn, sources, scope, thread_id = thread_id)
        conn.commit()
    except Exception:
        conn.rollback()
        for stored_path in copied:
            _remove_upload(stored_path)
        raise
    finally:
        conn.close()
    return document_ids, any(row["status"] in ("pending", "running") for row in rows)


def thread_has_documents(thread_id: str) -> bool:
    """Whether the thread has uploads that did not fail. Read over a metadata connection so a fork
    can still report them when vec0 cannot load and nothing can be copied."""
    if not rag_db.rag_db_path().is_file():
        return False
    conn = rag_db.get_metadata_connection()
    try:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone():
            return False
        return (
            conn.execute(
                "SELECT 1 FROM documents WHERE scope=? AND status != 'failed' LIMIT 1",
                (store.thread_scope(thread_id),),
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def _delete_scope(
    scope: str,
    thread_id: str,
    *,
    created_before: Optional[str] = None,
) -> list:
    conn = None
    removed = []
    try:
        try:
            conn = rag_db.get_connection()
        except Exception:
            return _delete_scope_without_vec(scope, thread_id, created_before = created_before)
        for row in conn.execute(*_scope_select(scope, created_before)).fetchall():
            store.delete_document(conn, row["id"])
            removed.append(row["stored_path"])
    except Exception:
        logger.warning("conversation_archive.delete_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return removed
