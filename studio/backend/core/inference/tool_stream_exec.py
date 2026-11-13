# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streams tool output and heartbeats (proxies cap idle at ~100 s); returns the result unchanged."""

from __future__ import annotations

import contextvars
import inspect
import queue
import threading
import time
from typing import Any, Callable, Generator, Sequence

from loggers import get_logger

# Split across two chunks, neither half matches the key pattern and the credential is painted whole.
_SECRET_PREFIXES = ("sk-unsloth-", "desktop-")
# A prefix plus a generous token: bounds the reverse search on a multi-megabyte chunk.
_SECRET_SCAN_TAIL = 128
# What ends a credential token in tool output: whitespace, quoting and closing punctuation.
_SECRET_TERMINATORS = frozenset(" \t\r\n\"'`,;)]}")


def _hold_back_partial_secret(text: str) -> int:
    """Index where a credential token still open at the end of `text` begins; the tail is held back."""
    tail_start = max(0, len(text) - _SECRET_SCAN_TAIL)
    tail = text[tail_start:]
    best = len(text)
    for prefix in _SECRET_PREFIXES:
        index = tail.rfind(prefix)
        while index != -1:
            absolute = tail_start + index
            rest = text[absolute + len(prefix) :]
            # A terminator anywhere after the token ends it, so the text is safe to emit.
            if not rest or not _SECRET_TERMINATORS.intersection(rest):
                best = min(best, absolute)
            # Keep going left: an earlier occurrence may still be open.
            index = tail.rfind(prefix, 0, index)
    # Also a tail that is a PREFIX of a prefix ("...sk-unslo"), which no rfind above can see.
    for prefix in _SECRET_PREFIXES:
        for length in range(min(len(prefix) - 1, len(text)), 0, -1):
            if text.endswith(prefix[:length]):
                best = min(best, len(text) - length)
                break
    return best


logger = get_logger(__name__)


def accepts_kwarg(func: Callable[..., str], name: str) -> bool:
    """Injected execute_tool fakes may lack the kwarg, so it is passed only when declared or **kwargs."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def accepts_output_callback(func: Callable[..., str]) -> bool:
    return accepts_kwarg(func, "output_callback")


def search_images_kwargs(func: Callable[..., str], tool_name: str) -> dict[str, bool]:
    """Read per call so the Settings toggle applies to the next search without a reload."""
    if tool_name != "web_search" or not accepts_kwarg(func, "search_images"):
        return {}
    from .search_images import search_images_enabled

    return {"search_images": True} if search_images_enabled() else {}


# Under common proxy idle caps (Cloudflare ~100 s, nginx 60 s).
TOOL_HEARTBEAT_INTERVAL_S = 10.0

# delay before the first approval keepalive so it cannot coalesce with the gated card.
TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

# How often the wrapper wakes to poll for output / completion / cancellation.
_POLL_INTERVAL_S = 0.25

# A cancel-ignoring tool is left as a daemon rather than blocking teardown.
_WORKER_JOIN_TIMEOUT_S = 5.0

# Caps live output per tool call so a tight print loop cannot flood SSE; above the model result cap since the UI shows it on truncation.
TOOL_OUTPUT_STREAM_MAX_CHARS = 400_000

_STREAM_CAPPED_NOTICE = "\n... (further live output not streamed)\n"


def _drain_queue(
    q: "queue.Queue",
    sentinel: object,
    max_chars: int | None,
    held: list | None = None,
) -> tuple[str, bool]:
    """Joins queued chunks FIFO up to max_chars, discarding the rest but still scanning it so
    completion is detected; crossing chunk sliced one char past budget; dict events held, never
    joined or dropped"""
    parts: list[str] = []
    total = 0
    dropping = False
    hit_sentinel = False
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is sentinel:
            hit_sentinel = True
            break
        if isinstance(item, dict):
            if held is not None:
                held.append(item)
            break
        if dropping:
            continue
        if max_chars is not None and total + len(item) > max_chars:
            # Keep one char past the budget as the overflow signal; drop the rest.
            parts.append(item[: max(0, max_chars - total) + 1])
            dropping = True
            continue
        parts.append(item)
        total += len(item)
    return "".join(parts), hit_sentinel


def stream_tool_execution(
    invoke: Callable[[Callable[[str], None]], str],
    *,
    tool_name: str,
    tool_call_id: str = "",
    cancel_event: Any = None,
    heartbeat_interval_s: float = TOOL_HEARTBEAT_INTERVAL_S,
    poll_interval_s: float = _POLL_INTERVAL_S,
    started_events: Sequence[dict] = (),
) -> Generator[dict, None, str]:
    """Run ``invoke(output_callback)`` in a thread; yield live events; return the result.

    ``invoke`` receives a thread-safe ``callable(str)`` it may call with
    incremental output chunks (or ignore entirely). Exceptions raised by the
    tool propagate to the caller unchanged after the worker thread finishes.

    ``cancel_event`` is the request-level cancellation signal already handed to
    the tool. If the consumer closes this generator early (an SSE disconnect
    calls ``gen.close()``, raising ``GeneratorExit`` at a ``yield``), the wrapper
    sets it so a cancel-observing tool stops, then joins the worker with a bounded
    timeout. Set ONLY on that abnormal-exit path, never on a clean finish, because
    the event is shared across a turn's tool calls and setting it early would
    abort the next tool.

    ``started_events`` are yielded once the worker is running, so a slow consumer never delays the tool.
    """
    output_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()
    outcome: dict[str, Any] = {}

    # Caps at the producer so a fast worker cannot queue unboundedly behind a slow client; one char past the cap keeps the notice; result stays byte-identical.
    accepted_output_chars = 0
    accepted_output_lock = threading.Lock()

    def _on_output(text: "str | dict") -> None:
        nonlocal accepted_output_chars
        if isinstance(text, dict):
            # A structured request for the client (a browser action), outside the text budget.
            output_queue.put(dict(text))
            return
        if not text:
            return
        with accepted_output_lock:
            remaining = TOOL_OUTPUT_STREAM_MAX_CHARS + 1 - accepted_output_chars
            if remaining <= 0:
                return
            accepted = text[:remaining]
            accepted_output_chars += len(accepted)
        output_queue.put(accepted)

    def _run() -> None:
        try:
            outcome["result"] = invoke(_on_output)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller side
            outcome["error"] = exc
        finally:
            # Posted after the result or error is recorded; wakes the consumer at once so fast tools pay no poll latency.
            output_queue.put(done_sentinel)

    # The worker runs in the caller's context, so a tool started for one account cannot resolve another's roots.
    worker = threading.Thread(
        target = contextvars.copy_context().run,
        args = (_run,),
        daemon = True,
        name = f"tool-exec-{tool_name or 'unknown'}",
    )
    worker.start()

    # Paced by idle polls, not a clock: tests patch ``time.monotonic`` globally.
    idle_polls_per_heartbeat = max(1, int(round(heartbeat_interval_s / poll_interval_s)))
    idle_polls = 0
    streamed_chars = 0
    stream_capped = False
    finished = False

    # Structured events pulled while joining text, delivered before the queue is read again.
    held: list[dict] = []

    def _drain_pending(max_chars: int | None = None) -> str:
        nonlocal finished
        text, hit_sentinel = _drain_queue(output_queue, done_sentinel, max_chars, held)
        if hit_sentinel:
            finished = True
        return text

    def _drain_and_drop() -> None:
        """Past the cap, drops chunks without joining them; completion is still detected so the loop
        exits."""
        nonlocal finished
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                return
            if item is done_sentinel:
                finished = True
                return
            if isinstance(item, dict):
                held.append(item)

    # Chunks reach the frontend and storage before tool_end is masked, so a key straddling chunks is carried into the next one.
    carry = ""

    def _masked(text: str, *, final: bool) -> "tuple[str, str]":
        """Return (emit, carry) for `text`, with credentials masked and partials held back."""
        from core.inference.tool_loop_controller import redact_studio_credentials

        combined = carry + text
        if final:
            return redact_studio_credentials(combined), ""
        split = _hold_back_partial_secret(combined)
        return redact_studio_credentials(combined[:split]), combined[split:]

    abnormal_exit = False
    try:
        for event in started_events:
            yield event
        while not finished or held:
            try:
                item = held.pop(0) if held else output_queue.get(timeout = poll_interval_s)
            except queue.Empty:
                # A disconnect sets cancel_event; heartbeat now so the route tears down at once.
                if cancel_event is not None and cancel_event.is_set():
                    yield {"type": "heartbeat"}
                    continue
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            if item is done_sentinel:
                break

            if isinstance(item, dict):
                idle_polls = 0
                yield {
                    **item,
                    "type": "tool_output",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "text": "",
                }
                continue

            if stream_capped:
                # Past the cap: drop queued chunks. One sleep per poll (not monotonic; tests patch it), counted
                # as idle so heartbeats keep flowing.
                _drain_and_drop()
                if finished:
                    continue
                time.sleep(poll_interval_s)
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            # Bound the join to the remaining budget so the crossing batch cannot overallocate; surplus is truncated anyway and output stays byte-identical.
            budget = TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars
            chunk = item + _drain_pending(max_chars = budget - len(item))
            idle_polls = 0
            if streamed_chars + len(chunk) > TOOL_OUTPUT_STREAM_MAX_CHARS:
                chunk = chunk[: max(0, TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars)]
                chunk += _STREAM_CAPPED_NOTICE
                stream_capped = True
            streamed_chars += len(chunk)
            if chunk:
                # The capped chunk is the last one sent, so flush the carry with it.
                emit, carry = _masked(chunk, final = stream_capped)
                if emit:
                    yield {
                        "type": "tool_output",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "text": emit,
                    }
        # Deferred text that turns out to be ordinary: emit it, or the output is truncated.
        if carry:
            emit, carry = _masked("", final = True)
            if emit:
                yield {
                    "type": "tool_output",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "text": emit,
                }
    except BaseException:
        # Only reached on early close by the consumer. Set cancel only here so the shared event is
        # never set under the next tool in a clean turn; re-raise (GeneratorExit must not be swallowed).
        abnormal_exit = True
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception:
                pass
        raise
    finally:
        # Abnormal exit: zero-timeout join so teardown never blocks the event loop.
        worker.join(timeout = 0 if abnormal_exit else _WORKER_JOIN_TIMEOUT_S)

    error = outcome.get("error")
    if error is not None:
        raise error
    # Returned verbatim so the result matches a direct execute_tool call.
    return outcome.get("result")
