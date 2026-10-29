# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Search index over evicted turns in a per-thread RAG scope; transcript untouched; cumulative,
never reset per compaction; entry points no-op on failure."""

from __future__ import annotations

import hashlib
import logging
import bisect
import json
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Optional

from storage import rag_db

from . import config, embeddings, retrieval, store, tool
from .chunking import chunk_pages
from .parsers import Page

logger = logging.getLogger(__name__)

_ARCHIVE_EMBED_BATCH = 64
# Per call; the rest is archived by later requests, which re-evict the same turns.
_ARCHIVE_EMBED_MAX_CHUNKS_DEFAULT = 256
# Background pre-embedding runs only while no inference runs (tool calls, between turns); small batches so
# a new generation waits at most one of them.
_PREWARM_BATCH = 16
_PREWARM_MAX_CHUNKS = 2048
_VECTOR_CACHE_MAX = 8192
_NOTED_MAX = 8
_PAUSE_JOIN_S = 10.0

# Instructions, not conversation. An archived system prompt could come back quoted as an "earlier turn" and read as
# user-authored text.
_SKIP_ROLES = frozenset({"system", "developer"})

# Auto-inject/recall call ids: their results are retrieved passages, so archiving them would self-feed the index.
_INJECTED_CALL_PREFIXES = ("rag_auto_", "conv_recall_")

# A probe carrying this marker can only be a PREFIX of the live text.
_TRUNCATION_MARKER = " ..."
_MAX_TOOL_RESULT_CHARS = 4000
# Tool arguments are usually short; the cap only stops a pathological blob dominating.
_MAX_TOOL_ARGS_CHARS = 1000
# Over-fetch before the branch filter so stale abandoned-branch turns cannot starve live hits.
_BRANCH_FILTER_OVERFETCH = 4
# Long abandoned branches can fill any fixed window, so widen and re-ask, up to this bound: a chat request, not a crawl.
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
            # Index bounded tool arguments, not names: the command run is searchable, as is assistant text.
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
            # Tool results are too large to quote whole; keep a bounded prefix so the index stays small.
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
    """Matched by name (model searches use plain call_N ids); drop only retrieval parts, since the reply after a search must still be archived."""
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
                    # Nothing left of it but the search.
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
    """Cuts only the folded retrieval result out of a user message, matched by dropped tool_call_ids not tool name, so the question around it survives."""
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


def _archive_embed_max_chunks() -> int:
    raw = os.environ.get("RAG_ARCHIVE_EMBED_MAX_CHUNKS", "").strip()
    if not raw:
        return _ARCHIVE_EMBED_MAX_CHUNKS_DEFAULT
    try:
        return max(1, int(raw))
    except ValueError:
        return _ARCHIVE_EMBED_MAX_CHUNKS_DEFAULT


def _chunk_key(identity: str, text: str) -> tuple[str, str]:
    return identity, hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


_vector_cache: "OrderedDict[tuple[str, str], object]" = OrderedDict()
_vector_cache_lock = threading.Lock()


def _cached_vectors(identity: Optional[str], texts: list[str]) -> list:
    """The cached vector per text, None for a miss. Keyed by content, so an edited turn simply misses."""
    if not identity:
        return [None] * len(texts)
    with _vector_cache_lock:
        out = []
        for text in texts:
            key = _chunk_key(identity, text)
            vector = _vector_cache.get(key)
            if vector is not None:
                _vector_cache.move_to_end(key)
            out.append(vector)
        return out


def _remember_vectors(identity: str, texts: list[str], vectors) -> None:
    with _vector_cache_lock:
        for text, vector in zip(texts, vectors):
            key = _chunk_key(identity, text)
            # A row of an (N, dim) batch is a view: cached as is, it would keep the whole batch alive.
            copy = getattr(vector, "copy", None)
            _vector_cache[key] = copy() if copy is not None else vector
            _vector_cache.move_to_end(key)
        while len(_vector_cache) > _VECTOR_CACHE_MAX:
            _vector_cache.popitem(last = False)


def _turn_chunks(text: str, chunk_limit: int, count) -> list:
    """The exact chunking archive_turns stores, shared so pre-embedded vectors match it."""
    return chunk_pages(
        [Page(text = text, page_number = None, char_count = len(text))],
        max_tokens = chunk_limit,
        overlap = config.CHUNK_OVERLAP,
        count = count,
    )


def _embed_pending_chunks(
    texts: list[str],
    *,
    model_name: str,
    cancel_check: Optional[Callable[[], bool]] = None,
    expected_identity: Optional[str] = None,
    cached: Optional[list] = None,
    max_encode: Optional[int] = None,
):
    """``(vectors, identity, cancelled, error)``, partial after a cancel or a late failure:
    write only groups ``vectors`` fully covers. A failing first batch raises. Only cache misses
    are encoded."""
    # `cached` is the lookup the caller's cap was computed from; a second lookup could see a later eviction.
    vectors = list(cached) if cached is not None else _cached_vectors(expected_identity, texts)
    identity = expected_identity if any(v is not None for v in vectors) else None
    misses = [i for i, v in enumerate(vectors) if v is None]
    restarted = cancelled = False
    error = None
    k = 0
    while k < len(misses):
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        idx = misses[k : k + _ARCHIVE_EMBED_BATCH]
        try:
            out, batch_identity = embeddings.encode_with_identity(
                [texts[i] for i in idx], model_name = model_name, normalize = True
            )
        except Exception as exc:
            # Keep finished batches, or a late timeout makes every request redo the whole pass.
            error = exc
            break
        if identity is not None and batch_identity != identity:
            if restarted:
                break
            # An ST failure swaps to llama-server for good: restart in the new space, as ingestion does.
            logger.warning("conversation_archive.embedder_changed_mid_pass; re-embedding")
            vectors, identity, restarted, k = [None] * len(texts), None, True, 0
            # Former cache hits are re-encoded too, so the restart honours the request's cap itself.
            misses = list(range(len(texts) if max_encode is None else min(len(texts), max_encode)))
            continue
        identity = batch_identity
        for i, vector in zip(idx, out):
            vectors[i] = vector
        _remember_vectors(batch_identity, [texts[i] for i in idx], out)
        k += _ARCHIVE_EMBED_BATCH
    prefix = []
    for vector in vectors:
        if vector is None:
            break
        prefix.append(vector)
    if error is not None and not prefix:
        raise error
    return prefix, identity or embeddings.embedding_identity(model_name), cancelled, error


_gap = threading.Lock()
_inference_depth = 0
_noted_branches: dict = {}
_prewarm_worker: Optional[threading.Thread] = None
_prewarm_stop = threading.Event()


def note_branch(
    thread_id: Optional[str],
    messages: list[dict],
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Remember a thread's latest transcript for the next idle gap; replaces the previous snapshot."""
    if not thread_id or not messages or not enabled():
        return
    from utils.account_context import current_account

    # Captured here: the worker is a plain thread, which would otherwise read the owner's databases.
    account = current_account()
    with _gap:
        # A stopped request whose stream never opened would otherwise keep its transcript here indefinitely.
        for key, job in list(_noted_branches.items()):
            if job[3] is not None and job[3]():
                del _noted_branches[key]
        key = (account.account_id, thread_id)
        _noted_branches.pop(key, None)
        while len(_noted_branches) >= _NOTED_MAX:
            del _noted_branches[next(iter(_noted_branches))]
        _noted_branches[key] = (
            account,
            thread_id,
            list(messages),
            cancel_check,
        )


@contextmanager
def _busy(restart: bool):
    global _inference_depth
    with _gap:
        _inference_depth += 1
    _pause_prewarm()
    try:
        yield
    finally:
        with _gap:
            _inference_depth -= 1
        if restart:
            _start_prewarm()


def inference_running():
    """Hold while llama.cpp generates. Stops background embedding and waits out its batch in flight."""
    return _busy(restart = True)


def prewarm_running() -> bool:
    worker = _prewarm_worker
    return worker is not None and worker.is_alive()


def hold_inference() -> Callable[[], None]:
    """``inference_running`` for the event loop: never blocks (the batch in flight, at most ``_PREWARM_BATCH``
    chunks, finishes on its own) and returns the release, to be called exactly once."""
    global _inference_depth
    with _gap:
        _inference_depth += 1
    _prewarm_stop.set()

    def release() -> None:
        global _inference_depth
        with _gap:
            _inference_depth -= 1
        _start_prewarm()

    return release


def _pause_prewarm() -> None:
    """Stop the gap worker and wait out its batch in flight; it restarts at the next stream's end. Bounded, so a
    wedged embedder can never hold a generation back."""
    _prewarm_stop.set()
    worker = _prewarm_worker
    if worker is not None and worker is not threading.current_thread():
        worker.join(_PAUSE_JOIN_S)
        if worker.is_alive():
            logger.warning("conversation_archive.prewarm_pause_timed_out after=%ss", _PAUSE_JOIN_S)


def _start_prewarm() -> None:
    global _prewarm_worker
    with _gap:
        if _inference_depth or not _noted_branches:
            return
        if _prewarm_worker is not None and _prewarm_worker.is_alive():
            return
        jobs = list(_noted_branches.values())
        _noted_branches.clear()
        _prewarm_stop.clear()
        _prewarm_worker = threading.Thread(
            target = _prewarm_jobs, args = (jobs,), daemon = True, name = "archive-prewarm"
        )
        _prewarm_worker.start()


def _other_work_busy() -> bool:
    """Image / video generation or a training run holds the accelerator. Read only from modules already
    imported (never constructs a service), and an unreadable probe counts as busy."""
    import sys

    try:
        for module, attr, probe in (
            ("core.training.training", "_training_backend", "is_training_active"),
            ("core.training.diffusion_training_service", "_service", "is_active"),
        ):
            service = getattr(sys.modules.get(module), attr, None)
            if service is not None and getattr(service, probe)():
                return True
        residency = sys.modules.get("core.inference.gpu_residency")
        if residency is not None:
            for owner in residency.MEDIA_OWNERS:
                if any(residency._media_busy(owner)):
                    return True
    except Exception:  # noqa: BLE001 -- fail closed: skipping a warm-up costs nothing
        return True
    return False


def _gap_closed() -> bool:
    return _prewarm_stop.is_set() or _inference_depth > 0 or _other_work_busy()


def _requeue(jobs: list) -> None:
    """Put back what a closed gap interrupted, behind nothing newer: a fresh note for the same chat wins, and a
    stopped request stays dropped."""
    with _gap:
        kept = {
            (job[0].account_id, job[1]): job
            for job in jobs
            if (job[0].account_id, job[1]) not in _noted_branches and not (job[3] and job[3]())
        }
        merged = {**kept, **_noted_branches}
        _noted_branches.clear()
        _noted_branches.update(list(merged.items())[-_NOTED_MAX:])


def _prewarm_jobs(jobs: list) -> None:
    from utils.account_context import run_as
    for index, (account, thread_id, messages, cancel_check) in enumerate(jobs):
        if _gap_closed():
            _requeue(jobs[index:])
            return
        # Chat Stop ends this request's background work too, not only its foreground pass.
        stopped = cancel_check or (lambda: False)
        if stopped():
            continue
        try:
            run_as(account, _prewarm_thread, thread_id, messages, stopped)
        except Exception:  # noqa: BLE001 -- a cache warm-up is never worth a log storm
            logger.debug(
                "conversation_archive.prewarm_failed thread_id=%s", thread_id, exc_info = True
            )
        if _gap_closed():
            # Possibly cut short; cached vectors make the redo cheap.
            _requeue(jobs[index:])
            return


def _prewarm_thread(
    thread_id: str,
    messages: list[dict],
    stopped: Callable[[], bool] = lambda: False,
) -> None:
    """Embed this thread's not-yet-archived turns into the vector cache, oldest first (the next to be
    evicted). Writes nothing to the archive, so an edit or rewind cannot strand a stale row."""
    if not can_archive(thread_id):
        return
    from core.inference.context_window import group_turns

    model = config.effective_embedding_model()
    if not _embedder_free():
        return
    identity = embeddings.embedding_identity(model)
    scope = store.conversation_archive_scope(thread_id)
    count = _yielding(embeddings.token_counter(model))
    chunk_limit = embeddings.chunk_tokens(model)
    conn = rag_db.get_connection()
    try:
        budget, todo, done = _PREWARM_MAX_CHUNKS, [], 0
        for group in group_turns(messages):
            if _gap_closed() or stopped() or budget <= 0:
                break
            archivable = _archivable(group)
            text = render_turn(archivable) if archivable else ""
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
            if _archived_under(conn, scope, digest, identity):
                continue
            texts = [chunk.text for chunk in _turn_chunks(text, chunk_limit, count)]
            misses = [t for t, v in zip(texts, _cached_vectors(identity, texts)) if v is None]
            todo += misses[:budget]
            budget -= len(misses)
            while len(todo) >= _PREWARM_BATCH and not (_gap_closed() or stopped()):
                done += _prewarm_batch(model, identity, todo[:_PREWARM_BATCH])
                todo = todo[_PREWARM_BATCH:]
        while todo and not (_gap_closed() or stopped()):
            done += _prewarm_batch(model, identity, todo[:_PREWARM_BATCH])
            todo = todo[_PREWARM_BATCH:]
    except _GapClosed:
        pass
    finally:
        conn.close()
    if done:
        logger.info("conversation_archive.prewarmed thread_id=%s chunks=%s", thread_id, done)


class _GapClosed(Exception):
    pass


def _yielding(count: Callable[[str], int]) -> Callable[[str], int]:
    def call(text: str) -> int:
        if not _embedder_free():
            raise _GapClosed
        return count(text)

    return call


def _embedder_busy() -> bool:
    """Whether another caller holds a lock the encode path would wait on: model load / backend build, the ST
    compute lock, or the llama-server backend's serve lock. Probed without blocking."""
    locks = [embeddings._backend_lock, embeddings._lock, embeddings._compute_lock]
    serve = getattr(embeddings._backend, "_serve_lock", None)
    if serve is not None:
        locks.append(serve)
    for lock in locks:
        if not lock.acquire(blocking = False):
            return True
        lock.release()
    return False


def _embedder_free() -> bool:
    """Wait for the embedder without queueing on its locks: a queued worker cannot see a stop, so a generation
    would wait behind another caller's (e.g. ingestion's) whole batch."""
    while _embedder_busy():
        if _gap_closed():
            return False
        time.sleep(0.05)
    return not _gap_closed()


def _prewarm_batch(model: str, identity: str, texts: list[str]) -> int:
    if not _embedder_free():
        return 0
    vectors, got = embeddings.encode_with_identity(texts, model_name = model, normalize = True)
    if got != identity:
        # The embedder changed under us; the next request's archive re-embeds in the new space.
        _prewarm_stop.set()
        return 0
    _remember_vectors(got, texts, vectors)
    return len(texts)


def archive_turns(
    thread_id: str,
    evicted: list[dict],
    live: Optional[list[dict]] = None,
    branch: Optional[list[dict]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """Index the evicted turns for ``thread_id``. Returns how many were newly written.

    Idempotent by content hash: the same turns are evicted again on every later request in the
    session, so re-archiving has to be free. After the first write each repeat costs one indexed
    SELECT and writes nothing.

    ``live`` is the fitted conversation, i.e. what the model can still see. It bounds how many
    copies of a repeated turn the archive is allowed to hold right now; see ``_write_budget``.
    Optional, and omitting it keeps the previous behaviour.

    ``cancel_check`` (Chat Stop) is polled before chunking and between embedding batches.
    """
    if not thread_id or not evicted or not enabled():
        return 0
    # Incognito and API-only threads are skipped (see can_archive); compacted chats are always persisted.
    if not can_archive(thread_id):
        logger.debug("conversation_archive.skipped_unpersisted_thread thread_id=%s", thread_id)
        return 0
    # A request's own compaction goes first and stays marked busy: another slot's stream ending must not
    # restart the gap worker under it. Only a stream's end restarts it, so none starts before this stream.
    with _busy(restart = False):
        return _archive_turns(thread_id, evicted, live, branch, cancel_check)


def _archive_turns(
    thread_id: str,
    evicted: list[dict],
    live: Optional[list[dict]],
    branch: Optional[list[dict]],
    cancel_check: Optional[Callable[[], bool]],
) -> int:
    # The evictor's own grouper, so an archived unit is exactly the unit the window drops. Imported lazily because the
    # inference layer imports this module.
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
    # Conversation order from the persisted thread, not arrival order (`_transcript_positions`).
    positions = _transcript_positions(thread_id, branch or live)
    live_positions = _live_positions(live)
    written = 0
    conn = None
    try:
        if cancel_check is not None and cancel_check():
            return 0
        count = embeddings.token_counter(model)
        chunk_limit = None
        conn = rag_db.get_connection()
        # Only a prediction; the authoritative identity check happens under the write lock.
        expected_identity = embeddings.embedding_identity(model)

        # Chunk everything first, then embed it in ONE pass. Per group, a first compaction of a long chat ran dozens
        # of one-item embedding jobs back to back before the reply could start, and both backends serialise them.
        pending, needs, snapshots = [], [], []
        max_chunks = _archive_embed_max_chunks()
        collected = skipped = 0
        for group, span in groups:
            if cancel_check is not None and cancel_check():
                return 0
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
            if collected >= max_chunks:
                skipped += 1
                continue
            if chunk_limit is None:
                chunk_limit = embeddings.chunk_tokens(model)
            chunks = _turn_chunks(text, chunk_limit, count)
            if chunks:
                # The cap counts what must be encoded now; chunks pre-embedded in an idle gap are free.
                hits = _cached_vectors(expected_identity, [chunk.text for chunk in chunks])
                need = sum(v is None for v in hits)
                pending.append((group, span, digest, chunks, seats, budget))
                needs.append(need)
                snapshots.append(hits)
                collected += need
        if not pending:
            return 0

        if cancel_check is not None and cancel_check():
            logger.info(
                "conversation_archive.embed_cancelled_before_encode thread_id=%s groups=%s",
                thread_id,
                len(pending),
            )
            return 0

        chunk_total = sum(needs)
        if chunk_total > max_chunks or skipped:
            kept, used = [], 0
            for entry, need in zip(pending, needs):
                # On what was kept, not on chunk count: a fully cached first turn adds nothing to `used`.
                if kept and used + need > max_chunks:
                    break
                if need > max_chunks and not kept:
                    # An oversized first turn still goes, or it would block the backlog forever.
                    kept.append(entry)
                    used += need
                    break
                kept.append(entry)
                used += need
            deferred = len(pending) - len(kept) + skipped
            logger.info(
                "conversation_archive.embed_capped thread_id=%s groups=%s chunks=%s "
                "cap=%s deferred_groups=%s",
                thread_id,
                len(pending),
                chunk_total,
                max_chunks,
                deferred,
            )
            snapshots = snapshots[: len(kept)]
            pending = kept
            if not pending:
                return 0

        texts = [chunk.text for entry in pending for chunk in entry[3]]
        started = time.monotonic()
        logger.info(
            "conversation_archive.embed_start thread_id=%s groups=%s chunks=%s batch=%s",
            thread_id,
            len(pending),
            len(texts),
            _ARCHIVE_EMBED_BATCH,
        )
        # Identity from the encode that produced these vectors: a concurrent embedder swap would otherwise label them
        # with a space they were never in.
        vectors, identity, cancelled, embed_error = _embed_pending_chunks(
            texts,
            model_name = model,
            cancel_check = cancel_check,
            expected_identity = expected_identity,
            cached = [vector for hits in snapshots for vector in hits],
            max_encode = max(max_chunks, sum(needs[: len(pending)])),
        )
        logger.info(
            "conversation_archive.embed_done thread_id=%s vectors=%s/%s cancelled=%s elapsed_ms=%s",
            thread_id,
            len(vectors),
            len(texts),
            cancelled,
            int((time.monotonic() - started) * 1000),
        )
        if not vectors:
            return 0

        # Only fully embedded groups; a partial one is left for the next request.
        complete: list = []
        offset = 0
        for entry in pending:
            n = len(entry[3])
            if offset + n > len(vectors):
                break
            complete.append((entry, vectors[offset : offset + n]))
            offset += n
        if len(complete) < len(pending):
            logger.info(
                "conversation_archive.embed_partial thread_id=%s written_groups=%s/%s",
                thread_id,
                len(complete),
                len(pending),
            )

        for (group, span, digest, chunks, seats, budget), group_vectors in complete:
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
                # Already in a transaction: the insert is still atomic with the re-check.
                logger.debug("conversation_archive.no_write_lock", exc_info = True)
            copies = store.documents_by_hash(conn, scope, digest)
            stale = _stale_document(conn, scope, digest, identity, occurrences = budget)
            if stale is _ARCHIVED:
                _restamp(conn, scope, digest, seats, copies = copies)
                if _write_lock:
                    conn.commit()
                # Widen here too, after the commit above, since widen carries its own: both occurrences can arrive in one compaction.
                _widen_span(conn, scope, digest, span)
                continue
            ordinal = None
            archived_at = None
            archived_rowid = None
            if stale is not None:
                # Replace stale copy, keeping position, timestamp and rowid (same-tick turns tie on created_at); NULL stays NULL or the oldest turn sorts last.
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
                # Allocated under the write lock in group_turns order, so it holds across epochs. Written unconditionally, so toggling ordering leaves no holes.
                archive_ordinal = ordinal,
                # Archive time, not row write time; None for a first-seen turn, which uses the clock.
                created_at = archived_at,
                # Likewise None for a first sighting: SQLite then assigns the next rowid.
                rowid = archived_rowid,
                commit = False,
            )
            try:
                store.add_chunks(conn, scope, document_id, chunks, group_vectors)
            except Exception:
                conn.rollback()
                raise
            # Surplus only, not a restamp (restamping would renumber a pre-column row). Runs after the new row exists, so the count is what the scope holds.
            _retire_surplus(conn, scope, digest, seats)
            # Restamp ordinals against the remaining seats once a copy is retired, skipping NULLs.
            if stale is not None:
                _restamp(conn, scope, digest, seats, skip_null = True)
            conn.commit()
            written += 1
            # Top up copies to budget (re-embed replaces, never adds). Only when the budget counts evicted occurrences; else it would copy a turn still in the prompt.
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
        if embed_error is not None:
            raise embed_error
    except embeddings.EmbeddingModelDownloadRequiredError as exc:
        globals()["_INGEST_FAILED"] = True
        _log_embedder_pending(model, exc)
    except Exception:
        # Failing to archive must not raise; what was written before the failure stays searchable.
        globals()["_INGEST_FAILED"] = True
        logger.warning("conversation_archive.ingest_failed thread_id=%s", thread_id, exc_info = True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Re-check after commit: a concurrent delete sweep could strand deleted content. Delete drops rows first, sweeps archives last, so either order converges.
    if written and not can_archive(thread_id):
        logger.info("conversation_archive.thread_deleted_mid_ingest thread_id=%s", thread_id)
        delete_for_thread(thread_id)
        return 0
    return written


# A Settings pick whose download has not finished is not a fault; logged once, not per compaction.
_EMBEDDER_PENDING_LOGGED: set = set()


def _log_embedder_pending(model, exc: Exception) -> None:
    if model in _EMBEDDER_PENDING_LOGGED:
        return
    _EMBEDDER_PENDING_LOGGED.add(model)
    logger.info("conversation_archive.embedder_pending: %s", exc)


# Set when an archive write fails, cleared by the next success. Process-wide: the embedder or store fails, not one thread.
_INGEST_FAILED = False


def degraded() -> bool:
    """Whether the last archive write failed; callers stop reserving recall room while it is set."""
    return _INGEST_FAILED


# Short enough to notice a returning store within a turn or two; long enough to share one probe.
def reachable() -> bool:
    """Probes with a real encode of x, never empty (empty input upsets llama); never memoised across requests, callers memoise per fit."""
    if not enabled():
        return False
    conn = None
    try:
        conn = rag_db.get_connection()
        # Check WRITABLE with a real statement, rolled back: BEGIN IMMEDIATE alone succeeds on a read-only connection.
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


# Returned by ``_stale_document`` for a turn that is already archived under vectors the query side still accepts.
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
        # A capped search cannot prove that an unvisited sibling does not tie the winner.
        return None
    best = None
    best_matched = None
    best_score = 0
    best_tied = False
    # Rendered once per STORED ROW, not per leaf: `_as_wire` expands a row the same way in any chain.
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
            # Tied on the SAME endpoint is not ambiguity: two retries fork past the proof.
            best_tied = best_tied or last_matched != best_matched
        if best_score >= len(wanted) and not require_unique:
            # Every message the request carries is on this chain; nothing can beat it.
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
    """Walks newest leaf to root, so Retry siblings do not mix in. fallback=False declines instead of reading the newest row; require_unique declines on tied leaves."""
    if not messages:
        return []
    by_id: dict = {}
    parent_of: dict = {}
    # Rows whose parent is storage order, not a real link; the root's absent parent is not synthesized.
    synthesized: set = set()
    previous = None
    for message in messages:
        identifier = message.get("id")
        if identifier is None:
            continue
        by_id[identifier] = message
        parent = message.get("parentId") or message.get("parent_id")
        # A row carrying the column with null is an intended root; pre-column rows use storage order. Rooting a null on the legacy path renumbers the whole archive.
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
    """Drops image tokens, which resolve only against their own message. The code-block carve-out is not mirrored: nothing here writes tokens into code."""
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


def _with_attachments(message: dict):
    """Content plus attachments, as `collectTextParts` (chat-adapter.ts) sends it; else the turn takes `_fallback_ordinal`."""
    content = message.get("content")
    attachments = message.get("attachments")
    if not isinstance(attachments, list) or str(message.get("role") or "") == "assistant":
        return content
    attached = [
        part
        for attachment in attachments
        if isinstance(attachment, dict)
        and attachment.get("mcpToolOnly") is not True
        and isinstance(attachment.get("content"), list)
        for part in attachment["content"]
        if isinstance(part, dict) and part.get("type") in ("text", "image")
    ]
    if not attached:
        return content
    if isinstance(content, list):
        return [*content, *attached]
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}, *attached]
    return attached


def _as_wire(messages: list[dict], sanitise_assistant: bool = True) -> list[dict]:
    """Splits stored tool-call parts into wire form; pass sanitise_assistant=False for request messages."""
    wire: list[dict] = []
    for message in messages:
        content = _with_attachments(message)
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
        # A provider builtin with no native part is not replayed, so not a call here. Any other tool-call part takes the replay loop even if the serializer drops it.
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
        # A one-slot box because `_flush` resets it and Python closures cannot rebind.
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
                    # Dropped whole, call and result, exactly as the serializer does.
                    continue
                round_id = _local_round_id(part)
                if (
                    pending_calls
                    and pending_round[0] is not None
                    and round_id is not None
                    and pending_round[0] != round_id
                ):
                    # `startsNewCodexToolRound`: a new local round is a new group.
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


# `withSandboxAttachmentPaths` (sandbox-attachments.ts) path notes, as `checkpoint._ATTACHMENT` matches them.
_SANDBOX_NOTE = re.compile(
    r"^\[[^\n]*(?: is saved at \.unsloth_attachments/[0-9a-f]{12}/[^\n]* in the python tool's working directory"
    r"|: its text is below, so answer from it\. For calculations, the python tool has the file at "
    r"path = \"\.unsloth_attachments/[0-9a-f]{12}/)[^\n]*\]$",
    re.MULTILINE,
)


def _seat_text(message: dict) -> str:
    """The seat-matching probe minus python path notes, which only the request carries."""
    text = _probe_text(message)
    if str(message.get("role") or "") == "user":
        text = _SANDBOX_NOTE.sub("", text)
    return _normalise_cased(text)


def _fallback_ordinal(conn, scope: str, positions: Optional[list[list[str]]]) -> int:
    """Past both the archive's next ordinal and the transcript length: the two numbering spaces differ, and a gap in ordinals costs nothing."""
    return max(store.next_archive_ordinal(conn, scope), len(positions or []))


def _transcript_positions(thread_id: str, branch = None) -> Optional[list[str]]:
    """Numbers come from the saved transcript, not archive time (eviction is not oldest-first); read it, not the request, whose indices shift as groups drop out."""
    try:
        from core.inference.context_window import group_turns
        from storage import studio_db
        messages = studio_db.list_chat_messages(thread_id)
    except Exception:
        return None
    if not messages:
        return None
    # Grouped like the evictor, so a position is a TURN index, matching what ordinals compare against.
    wire = _as_wire(_active_chain(messages, branch))
    return [[_seat_text(message) for message in group] for group in group_turns(wire) if group]


def _occurrences(positions: Optional[list[list[str]]], group: list[dict]) -> list[int]:
    """Matches the whole turn, not its first line, so turns that start alike do not share seats. Compared as a prefix, since the transcript lags the evicted group."""
    if not positions or not group:
        return []
    texts = [_seat_text(message) for message in group]
    if not texts or not texts[0]:
        return []
    calls = [bool(message.get("tool_calls")) for message in group]

    def _same(stored: str, live: str, is_call: bool) -> bool:
        # A tool call is compared as needle in haystack, never for equality: the store keeps arguments as an OBJECT
        # and the request carries the model's raw string, so `_probe_text` offers both JSON spellings for the stored
        # copy and one for the live one. Every other message is compared exactly.
        # In two pieces when the whole needle does not fit, since the second spelling is inserted BETWEEN the
        # arguments and what followed them: a call with a preamble renders live as name/args/text and stored as
        # name/args/args/text, so the live string is no longer contiguous. One split point, so this tolerates exactly
        # the one insertion `_probe_text` makes.
        # Space offsets keep split(" ") semantics. Head presence only falls and tail presence only rises as the cut
        # moves right, so both bounds are binary-searched; the ordered match is not monotone, so every cut between
        # them is still checked against the earliest head and the latest tail.
        if not is_call:
            return stored == live

        if not live:
            return False

        if live in stored:
            return True

        # Splitting removes one space; the remaining characters must fit without overlap.
        if len(stored) < len(live) - 1:
            return False

        cuts = [index for index, char in enumerate(live) if char == " "]
        if not cuts:
            return False

        # Find the exclusive upper bound of splits whose head occurs in stored.
        left, right = 0, len(cuts)
        while left < right:
            mid = (left + right) // 2
            if live[: cuts[mid]] in stored:
                left = mid + 1
            else:
                right = mid

        prefix_stop = left
        if prefix_stop == 0:
            return False

        # Within that range, find the first split whose tail occurs in stored.
        left, right = 0, prefix_stop
        while left < right:
            mid = (left + right) // 2
            if live[cuts[mid] + 1 :] in stored:
                right = mid
            else:
                left = mid + 1

        for index in range(left, prefix_stop):
            cut = cuts[index]
            head_at = stored.find(live[:cut])
            tail_at = stored.rfind(live[cut + 1 :])

            if tail_at >= head_at + cut:
                return True

        return False

    return [
        index
        for index, position in enumerate(positions)
        if position
        # Shorter positions may only match the trailing one, or an orphan user row takes a second seat. Longer ones are fine (>=, not ==).
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
    """One more copy at the next unfilled seat; False when none is left. span is the transcript size, not len(group), since archiving strips retrieval parts."""
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
        [_seat_text(message) for message in group] for group in group_turns(_as_wire(live)) if group
    ]
    return positions or None


def _write_budget(
    positions: Optional[list[list[str]]],
    seats: list[int],
    live_positions: Optional[list[list[str]]],
    group: Optional[list[dict]] = None,
) -> int:
    """Counts occurrences still in the live prompt, so a shown turn is not re-archived. Floors at 1 so a turn that cannot be told apart is still stored."""
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
    """Converges archives numbered by arrival or unnumbered, via UPDATE, so nothing duplicates. Must run from both archived paths, not only the write-lock one."""
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
            # Re-index the stale copy before considering writing a new one.
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
    """Trims both ends (leading space would retire a live turn) and keeps case: a capitalisation-only edit must retire the stale copy."""
    return (text or "").strip()


def _normalise_cased(text: str) -> str:
    """Collapses whitespace but keeps case, since casing-only variants are different turns."""
    return " ".join((text or "").split())


def _live_transcript(thread_id: str) -> Optional[list[str]]:
    """Keeps recall on the user's branch, as the archive also holds abandoned turns. None means no saved transcript, so the caller does not filter."""
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
    """Reports the message a run opened on: an overlapping chunk's tail can begin in an earlier message. The cursor is not carried over between chunks."""
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
                    # Same exemption at the front of the run as inside it.
                    opened_at = 0 if partial_ok else found
                    opened_index = index
                cursor = found + len(probe)
                # The exemption belongs to the call only; after a text probe the cursor is exact again.
                partial = partial_ok
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
        # NULL for archives predating the column, which fall back to counting labels.
        row = conn.execute(
            "SELECT archive_messages FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        message_count = row["archive_messages"] if row else None
    except Exception:
        # Never fail a recall on the strictness pass: fall back to what the candidate chunk itself said.
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
        # Next chunk may restart where the previous opened: its overlap can carry a whole short message. Never earlier than that, and forward is tried first.
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
    """NULL ordinals sort first; then created_at, then rowid (coarse clocks tie created_at); chunk_index last, as a position within one document."""
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
        # Nothing tied: skip the row fetch entirely, which is the common case.
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
        # Archive legs are always "lexical", so this is the same call with the tie-break reversed.
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
    # One expression: the filter pass already IS the ranking pass, so skip the second query.
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
        # Exact membership is an improvement, not a dependency; on failure the strict rows alone stand.
        logger.warning("conversation_archive.eligibility_probe_failed", exc_info = True)
    ranked = [hit for hit in loose if hit.chunk_id in eligible]
    already = {hit.chunk_id for hit in ranked}
    ranked += [hit for hit in strict if hit.chunk_id not in already]
    already |= {hit.chunk_id for hit in strict}
    ranked += [hit for hit in loose if hit.chunk_id not in already]
    return ranked


def _candidates(conn, scope: str, query: str, model, fetch: int, thread_id: str) -> list:
    """Lexical runs twice with an identifier: requiring it, then over content words for order, so an empty filter is not read as an empty archive."""
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
            # Legs sized to the fetch: at the default 30 each, widening past 60 reads as an exhausted archive.
            legs = max(fetch, config.TOP_K_LEXICAL, config.TOP_K_DENSE)
            for hit in retrieval.retrieve_hybrid(
                conn,
                scope,
                query,
                k = fetch,
                model_name = model,
                mode = "hybrid",
                lexical_k = legs,
                dense_k = legs,
            ):
                if hit.chunk_id not in seen:
                    hits.append(hit)
                    seen.add(hit.chunk_id)
                if len(hits) >= fetch:
                    break
        except embeddings.EmbeddingModelDownloadRequiredError as exc:
            _log_embedder_pending(model, exc)
        except Exception:
            # Dense retrieval raises if no embedder starts; lexical hits stand alone, so this top-up may fail.
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
    with retrieval.query_vector_memo():
        return _recall(
            thread_id,
            query,
            top_k = top_k,
            branch_messages = branch_messages,
            extra_queries = extra_queries,
            forced = forced,
        )


def _recall(
    thread_id: str,
    query: str,
    *,
    top_k: Optional[int] = None,
    branch_messages: Optional[list[dict]] = None,
    extra_queries: Optional[list[str]] = None,
    forced: bool = False,
) -> Optional[tuple[str, list[dict]]]:
    query = (query or "").strip()
    if not thread_id or not query or not enabled():
        return None
    min_dense_score = config.CONVERSATION_FORCED_MIN_SCORE if forced else 0.0

    scope = store.conversation_archive_scope(thread_id)
    limit = top_k or config.CONVERSATION_ARCHIVE_TOP_K
    # Separate queries: concatenating would dilute identifiers and AND unrelated intents. At a limit of one the anchor wins.
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
        # Request's branch first; stored rows are the whole DAG, a fallback that still beats no filtering.
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
            # Stop on enough hits or no wider fetch: an abandoned branch can outrank the live one, not forever.
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


def copy_thread_documents(
    source_thread_id: str,
    thread_id: str,
    *,
    created_before: Optional[str] = None,
) -> tuple[dict[str, str], bool]:
    """Copy a thread's finished uploads into another thread. Returns the source-to-copy document id
    map and whether an upload still being ingested was left behind. Failed uploads are not copied
    and not reported: the documents bar never shows them. ``created_before`` (ISO-8601 UTC) leaves
    out uploads made after a fork point.

    Files are copied before the transaction so rag.db's write lock is never held across file I/O.
    """
    from .ingestion import _copy_upload, _remove_upload

    scope = store.thread_scope(thread_id)
    copied: list = []
    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE scope=? AND (? IS NULL OR created_at<?) "
            "ORDER BY created_at",
            (store.thread_scope(source_thread_id), created_before, created_before),
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
            # No vec0 here. Delete what can be deleted rather than nothing at all.
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
