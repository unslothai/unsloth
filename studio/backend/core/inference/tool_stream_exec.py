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

import queue
import threading
import time
from typing import Any, Callable, Generator

from loggers import get_logger

logger = get_logger(__name__)

# Cadence of heartbeat events while a tool blocks with no output. Well under
# common proxy idle caps (Cloudflare ~100 s, nginx default 60 s).
TOOL_HEARTBEAT_INTERVAL_S = 10.0

# How often the wrapper wakes to poll for output / completion / cancellation.
_POLL_INTERVAL_S = 0.25

# Cap on total streamed live-output characters per tool call. The final
# result is truncated separately (tools._MAX_OUTPUT_CHARS); this only bounds
# the transient UI stream so a tight print loop cannot flood the SSE channel.
# Deliberately much higher than the model-visible result cap: when the final
# result is truncated, the UI keeps the accumulated live stream as the
# displayed output, so this is the ceiling on what the user can see. Chunks
# are batched per poll interval (~4 events/s), so a large cap stays cheap on
# the SSE channel.
TOOL_OUTPUT_STREAM_MAX_CHARS = 400_000

_STREAM_CAPPED_NOTICE = "\n... (further live output not streamed)\n"


def stream_tool_execution(
    invoke: Callable[[Callable[[str], None]], str],
    *,
    tool_name: str,
    tool_call_id: str = "",
    heartbeat_interval_s: float = TOOL_HEARTBEAT_INTERVAL_S,
    poll_interval_s: float = _POLL_INTERVAL_S,
) -> Generator[dict, None, str]:
    """Run ``invoke(output_callback)`` in a thread; yield live events; return the result.

    ``invoke`` receives a thread-safe ``callable(str)`` it may call with
    incremental output chunks (or ignore entirely). Exceptions raised by the
    tool propagate to the caller unchanged after the worker thread finishes.
    """
    output_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()
    outcome: dict[str, Any] = {}

    def _on_output(text: str) -> None:
        if text:
            output_queue.put(text)

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

    # Heartbeats are paced by counting idle queue polls (each poll waits
    # ``poll_interval_s``) rather than reading a wall clock: tests patch
    # ``time.monotonic`` globally, and the wrapper must not consume their
    # scripted values.
    idle_polls_per_heartbeat = max(1, int(round(heartbeat_interval_s / poll_interval_s)))
    idle_polls = 0
    streamed_chars = 0
    stream_capped = False
    finished = False

    def _drain_pending() -> str:
        nonlocal finished
        parts: list[str] = []
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                return "".join(parts)
            if item is done_sentinel:
                finished = True
                return "".join(parts)
            parts.append(item)

    def _drain_and_drop() -> None:
        """Discard the current and every queued chunk without concatenating.

        Once the live-output cap is tripped every chunk is thrown away, so a
        chatty tool (``yes``, a tight print loop) that enqueues far more than
        one poll interval's worth of text must not pay to build a combined
        string only to drop it -- that would let it blow past the memory/CPU
        ceiling the cap exists to enforce. Still detect completion so the loop
        can exit promptly.
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

    while not finished:
        try:
            item = output_queue.get(timeout = poll_interval_s)
        except queue.Empty:
            idle_polls += 1
            if idle_polls >= idle_polls_per_heartbeat:
                idle_polls = 0
                yield {"type": "heartbeat"}
            continue

        if item is done_sentinel:
            break

        if stream_capped:
            # Past the cap this chunk and every queued sibling are discarded.
            # Drop them without concatenating (see _drain_and_drop): a chatty
            # tool can enqueue far more than one poll interval's worth of text,
            # and building a combined string just to throw it away would defeat
            # the cap's memory/CPU bound. The keepalive cadence must still
            # survive: draining without pacing would neither yield anything nor
            # let an idle poll fire (a chatty tool keeps the queue non-empty),
            # so the SSE stream would go silent past proxy idle timeouts. Sleep
            # one poll interval per drain (time.sleep, not time.monotonic --
            # tests patch the clock) and count it as an idle poll so heartbeats
            # keep flowing at the configured cadence.
            _drain_and_drop()
            if finished:
                break
            time.sleep(poll_interval_s)
            idle_polls += 1
            if idle_polls >= idle_polls_per_heartbeat:
                idle_polls = 0
                yield {"type": "heartbeat"}
            continue

        chunk = item + _drain_pending()
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

    worker.join()
    error = outcome.get("error")
    if error is not None:
        raise error
    # Returned verbatim (the loop's record_result handles non-str), so the
    # final tool result is byte-identical to a direct execute_tool call.
    return outcome.get("result")
