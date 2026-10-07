# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming wrapper around blocking server-side tool execution.

``stream_tool_execution`` runs a blocking tool call in a worker thread and
turns it into a generator that yields:

* ``{"type": "tool_output", "tool_name", "tool_call_id", "text"}`` -- an
  incremental stdout/stderr chunk (python/terminal tools) for live UI output;
* ``{"type": "heartbeat"}`` -- emitted whenever nothing else has been yielded
  for ``heartbeat_interval_s`` seconds, so the SSE route can write a
  keepalive and reverse proxies (Cloudflare tunnels cap idle streams at
  ~100 s) never see a silent connection while a tool runs;

and *returns* the tool's final result string via ``StopIteration.value``
(``result = yield from stream_tool_execution(...)``). The returned result is
byte-identical to calling the tool directly, so tool-result parsing, nudging,
and healing downstream are untouched.
"""

from __future__ import annotations

import contextvars
import inspect
import queue
import threading
import time
from typing import Any, Callable, Generator

from loggers import get_logger

# A key split across chunks would otherwise evade the pattern and be painted whole.
_SECRET_PREFIXES = ("sk-unsloth-", "desktop-")
_SECRET_SCAN_TAIL = 128
_SECRET_TERMINATORS = frozenset(" \t\r\n\"'`,;)]}")


def _hold_back_partial_secret(text: str) -> int:
    """Index at which `text` may hold a credential token still open at its end.

    Everything from the returned index is carried to the next chunk. Returns ``len(text)`` when the
    tail is safe to emit whole. Conservative on purpose: holding back a few bytes that turn out to
    be ordinary text only delays them until the next chunk or the final flush.
    """
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
    """Whether an injectable ``execute_tool`` supports the keyword ``name``.

    ``execute_tool`` is replaceable (tests inject fakes / the pre-PR signature),
    so forward a kwarg only when the callable declares it or takes ``**kwargs``
    (passing it unconditionally would ``TypeError`` on an old signature).
    """
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
    """``{"search_images": True}`` when web_search should also return images, else ``{}``.

    Read per call rather than per request so the Settings toggle applies to the
    next search without a reload, and only for web_search so other tools never
    pay the settings read.
    """
    if tool_name != "web_search" or not accepts_kwarg(func, "search_images"):
        return {}
    from .search_images import search_images_enabled

    return {"search_images": True} if search_images_enabled() else {}


# Under common proxy idle caps (Cloudflare ~100 s, nginx 60 s).
TOOL_HEARTBEAT_INTERVAL_S = 10.0

# Delay so the first approval keepalive cannot coalesce with the gated card.
TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

_POLL_INTERVAL_S = 0.25

# A cancel-ignoring tool is left as a daemon rather than blocking teardown.
_WORKER_JOIN_TIMEOUT_S = 5.0

# Bounds the live UI stream; above the model result cap since the UI shows it on truncation.
TOOL_OUTPUT_STREAM_MAX_CHARS = 400_000

_STREAM_CAPPED_NOTICE = "\n... (further live output not streamed)\n"


def _drain_queue(q: "queue.Queue", sentinel: object, max_chars: int | None) -> tuple[str, bool]:
    """Pull every currently-queued item, joining chunks in FIFO order.

    With ``max_chars`` set, stop concatenating at the budget and discard the
    remaining chunks in place, bounding peak allocation when a chatty tool queues
    far more than the cap before the consumer wakes. The crossing chunk is sliced
    to one char past the budget, enough for the caller's truncation to stay
    byte-identical. Returns ``(joined_text, hit_sentinel)``; the surplus is still
    scanned so completion is detected promptly.
    """
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
    """
    output_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()
    outcome: dict[str, Any] = {}

    # Cap at the producer so a fast worker cannot enqueue unboundedly behind a slow client; keep one
    # char past the cap so the consumer still emits the notice.
    accepted_output_chars = 0
    accepted_output_lock = threading.Lock()

    def _on_output(text: str) -> None:
        nonlocal accepted_output_chars
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
            # Wakes the consumer at once so fast tools pay no poll latency.
            output_queue.put(done_sentinel)

    # The worker runs in the caller's context so it resolves only that account's roots.
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

    def _drain_pending(max_chars: int | None = None) -> str:
        nonlocal finished
        text, hit_sentinel = _drain_queue(output_queue, done_sentinel, max_chars)
        if hit_sentinel:
            finished = True
        return text

    def _drain_and_drop() -> None:
        """Discard the current and every queued chunk without concatenating.

        Past the cap every chunk is dropped, so don't pay to build a combined
        string only to drop it. Still detect completion so the loop can exit.
        """
        nonlocal finished
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                return
            if item is done_sentinel:
                finished = True
                return

    # A key can straddle chunks; carry a still-open token into the next one.
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
        while not finished:
            try:
                item = output_queue.get(timeout = poll_interval_s)
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

            if stream_capped:
                # Past the cap: drop queued chunks. One sleep per poll (not monotonic; tests patch it), counted
                # as idle so heartbeats keep flowing.
                _drain_and_drop()
                if finished:
                    break
                time.sleep(poll_interval_s)
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            # Bound the join so the crossing batch cannot allocate far past the cap.
            budget = TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars
            chunk = item + _drain_pending(max_chars = budget - len(item))
            idle_polls = 0
            if streamed_chars + len(chunk) > TOOL_OUTPUT_STREAM_MAX_CHARS:
                chunk = chunk[: max(0, TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars)]
                chunk += _STREAM_CAPPED_NOTICE
                stream_capped = True
            streamed_chars += len(chunk)
            if chunk:
                emit, carry = _masked(chunk, final = stream_capped)
                if emit:
                    yield {
                        "type": "tool_output",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "text": emit,
                    }
        # Deferred ordinary text must still be emitted.
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
