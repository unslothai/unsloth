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

import inspect
import queue
import threading
import time
from typing import Any, Callable, Generator

from loggers import get_logger

logger = get_logger(__name__)


def accepts_output_callback(func: Callable[..., str]) -> bool:
    """Whether an injectable ``execute_tool`` supports ``output_callback``.

    ``execute_tool`` is replaceable (tests inject fakes / the pre-PR signature),
    so forward the kwarg only when the callable declares it or takes ``**kwargs``
    (passing it unconditionally would ``TypeError`` on an old signature).
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if "output_callback" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# Cadence of heartbeat events while a tool blocks with no output. Well under
# common proxy idle caps (Cloudflare ~100 s, nginx default 60 s).
TOOL_HEARTBEAT_INTERVAL_S = 10.0

# How often the wrapper wakes to poll for output / completion / cancellation.
_POLL_INTERVAL_S = 0.25

# Upper bound on how long teardown waits for the worker once the stream is
# closed or errors. A cancel-observing tool returns within this after
# ``cancel_event`` is set; a cancel-ignoring one is a daemon left to finish on
# its own rather than blocking teardown for the tool's full timeout.
_WORKER_JOIN_TIMEOUT_S = 5.0

# Cap on total streamed live-output characters per tool call, bounding the
# transient UI stream so a tight print loop cannot flood the SSE channel. Much
# higher than the model-visible result cap (tools._MAX_OUTPUT_CHARS) since the
# UI keeps the live stream as the displayed output when the result is truncated.
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

    # Bound accepted output at the PRODUCER boundary: the consumer-side cap alone
    # wouldn't stop a fast worker enqueuing unboundedly while a slow SSE client
    # backpressures. Accept at most one char past the cap (so the consumer still
    # emits the capped notice) and drop the rest. The final result is captured
    # independently, so this never changes the byte-identical result.
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
            # Posted after the result/error is recorded; wakes the consumer
            # immediately so fast tools pay no poll-interval latency.
            output_queue.put(done_sentinel)

    worker = threading.Thread(
        target = _run,
        daemon = True,
        name = f"tool-exec-{tool_name or 'unknown'}",
    )
    worker.start()

    # Heartbeats are paced by counting idle queue polls rather than a wall clock
    # (tests patch ``time.monotonic`` globally, so the wrapper must not read it).
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

    abnormal_exit = False
    try:
        while not finished:
            try:
                item = output_queue.get(timeout = poll_interval_s)
            except queue.Empty:
                # A disconnect sets cancel_event while the worker is silent;
                # surface a heartbeat this poll so the route regains control and
                # tears down at once, not after a full heartbeat interval.
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
                # Past the cap: drop this chunk and every queued sibling (see
                # _drain_and_drop). Pace with one time.sleep per poll (not
                # time.monotonic -- tests patch the clock), counted as an idle
                # poll so heartbeats keep flowing while the queue stays non-empty.
                _drain_and_drop()
                if finished:
                    break
                time.sleep(poll_interval_s)
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            # Bound the join to the remaining budget so the crossing batch can't
            # allocate far past the cap (surplus is truncated below anyway); the
            # prefix is long enough that truncation stays byte-identical.
            budget = TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars
            chunk = item + _drain_pending(max_chars = budget - len(item))
            idle_polls = 0
            if streamed_chars + len(chunk) > TOOL_OUTPUT_STREAM_MAX_CHARS:
                chunk = chunk[: max(0, TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars)]
                chunk += _STREAM_CAPPED_NOTICE
                stream_capped = True
            streamed_chars += len(chunk)
            if chunk:
                yield {
                    "type": "tool_output",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "text": chunk,
                }
    except BaseException:
        # The loop only raises when the consumer closes us early: an SSE
        # disconnect calls gen.close() (GeneratorExit at the yield) or the route
        # throws in. Signal cancellation so a cancel-observing tool returns; the
        # daemon worker is then abandoned (see finally). Re-raise so the caller
        # sees the real cause (GeneratorExit must not be swallowed). Runs ONLY on
        # abnormal exit, so the shared cancel_event is never set out from under
        # the next tool in a clean multi-tool turn.
        abnormal_exit = True
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception:
                pass
        raise
    finally:
        # Clean finish: the worker already recorded its result and queued the
        # sentinel we consumed, so this join returns at once. Abnormal exit:
        # cancel_event is set and the daemon worker abandoned, so join with a zero
        # timeout -- teardown never blocks the caller (the route may close this
        # generator on the event loop), and the daemon cannot outlive the process.
        worker.join(timeout = 0 if abnormal_exit else _WORKER_JOIN_TIMEOUT_S)

    error = outcome.get("error")
    if error is not None:
        raise error
    # Returned verbatim (the loop's record_result handles non-str), so the
    # final tool result is byte-identical to a direct execute_tool call.
    return outcome.get("result")
