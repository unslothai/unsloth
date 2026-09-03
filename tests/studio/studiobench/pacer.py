# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An OpenAI-compatible SSE server that paces a reply the way a real backend does.

WHY THIS EXISTS AT ALL. The harness this replaces measured a backend-free smoke page driven by a
local `ChatModelAdapter`. Two whole mechanisms therefore never executed: the cumulative
`<think>` re-parse in `chat-adapter.ts`, which re-parses the ENTIRE growing buffer on every delta,
and the autoscroll `MutationObserver` in `use-intent-aware-autoscroll.tsx`, which answers every
streamed character with a synchronous `scrollHeight` read over the whole thread. Both are O(thread)
per chunk and neither is reachable without real bytes arriving over a real transport. So Unsloth is
pointed at THIS, as an external provider, and the bytes go out over the wire, through the Unsloth
backend's own relay, into the app's own `TextDecoder`, its own SSE framing and its own delta
accumulation. Nothing is stubbed and there is no `page.route` anywhere near the primary transport.

THREE things here are not incidental.

**Threaded.** `ThreadingHTTPServer`, not `HTTPServer`. A single-threaded server previously lost 11
cells of a matrix to `goto` timeouts: the browser opens the SPA's own requests while a stream is in
flight, and one blocked handler stalls all of them, so the page never finishes navigating and the
cell dies without a number.

**Deficit-scheduled cadence.** Each tick computes `floor((now - t0) / gap)` and sends the SHORTFALL
in one burst, rather than sleeping a gap per chunk. Two consequences, both wanted. Stream duration
becomes a function of wall clock alone, so a 90K reply takes the same 276 seconds on a fast machine
and a slow one and a tier's time budget is honest. And a renderer that jams gets a BURST when it
recovers, which is exactly what a real backend does: the model keeps generating while the socket
backs up, and the queue drains at once. `sleep(gap)` per chunk instead makes the SERVER slow down
whenever the client does, which quietly converts a rendering problem into a shorter benchmark.

**The exact chunk shapes the app parses.** The backend's own `_gguf_chat_delta_line` emits
`reasoning_content` WITH `content: ""` alongside, so that is what goes out here. The terminal chunk
carries `finish_reason: "stop"` AND is followed by `data: [DONE]`: `streamChatCompletions` in
`chat-api.ts` throws `StreamInterruptedError` at EOF if it saw neither, and a harness that omits
one measures error handling. A usage chunk follows because the app always sends
`stream_options: {include_usage: true}` and its context bar reads the result.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

# The field's own arrival rate: 90,262 characters in 276 seconds is 327 characters a second,
# which at 24 characters a chunk is one chunk every 73ms. It is part of the fixture and getting it
# wrong is how the earlier harness failed to reproduce anything: at a 2ms gap the renderer is the
# bottleneck from the first chunk, the run opens at 45 fps instead of 59, and there is no healthy
# baseline left to degrade FROM. The trace's whole shape is "idle between chunks at the start, not
# idle at the end".
CAD_FIELD = (24, 73)
# For the small rungs, where the field cadence would spend four minutes streaming a reply nobody
# is measuring the cadence of.
CAD_FAST = (64, 8)

CADENCES = {"field": CAD_FIELD, "fast": CAD_FAST}

KEEPALIVE_S = 5.0


@dataclass
class Script:
    """One reply for the pacer to serve. Set by the driver before it presses send."""

    reasoning: str = ""
    content: str = ""
    chunk_chars: int = CAD_FIELD[0]
    gap_ms: int = CAD_FIELD[1]
    model: str = "studiobench-pacer"
    # Emitted before anything else, so a driver can prove the stream it is watching is the one it
    # asked for rather than a leftover from the previous cell.
    tag: str = ""
    # Set by the driver to make the pacer stop mid-reply, for the stop-generation action. The
    # pacer keeps the socket OPEN and idle rather than closing it, because a close is a different
    # event from a cancel and the app distinguishes them.
    hold_after_chars: Optional[int] = None


@dataclass
class StreamStats:
    """What the pacer ACTUALLY did, which is the only honest record of the cadence.

    Kept because the driver cannot see it any other way: what reaches the page has been through a
    backend relay and a browser, so "did the fixture send what it meant to" and "did the page
    receive it" are two different questions and only one of them is about the app.
    """

    request_id: str = ""
    tag: str = ""
    started_at: float = 0.0
    first_chunk_ms: Optional[float] = None
    last_chunk_ms: Optional[float] = None
    chunks_sent: int = 0
    chars_sent: int = 0
    bytes_sent: int = 0
    reasoning_chars_sent: int = 0
    content_chars_sent: int = 0
    # How often the deficit scheduler had to send more than one chunk to catch up, and the worst
    # shortfall it saw. A run where these stay at zero had no backpressure at all; a run where
    # max_deficit is 40 had the socket blocked for three seconds.
    bursts: int = 0
    max_deficit: int = 0
    # Time spent inside a blocking write. This IS the backpressure, measured rather than inferred.
    write_block_ms: float = 0.0
    keepalives: int = 0
    completed: bool = False
    disconnected: bool = False
    held: bool = False
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    # The cadence achieved, against the one asked for. A gap that came out at 91ms when 73 was
    # requested is a machine that could not keep up, and it belongs in the report next to the
    # numbers it distorted.
    requested_gap_ms: int = 0
    achieved_gap_ms: Optional[float] = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("started_at", None)
        for k in (
            "first_chunk_ms",
            "last_chunk_ms",
            "write_block_ms",
            "duration_ms",
            "achieved_gap_ms",
        ):
            if isinstance(d.get(k), float):
                d[k] = round(d[k], 2)
        return d


class PacerState:
    """The single-slot script plus the log of what each request did. Guarded by one lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.script = Script()
        self.stats: list[StreamStats] = []
        self.requests_seen = 0
        self.model_ids: list[str] = ["studiobench-pacer"]
        # Flipped by the driver to release a held stream (the stop-generation action asserts the
        # app's cancel, so the pacer must be able to sit still while it happens).
        self.release = threading.Event()

    def set_script(self, script: Script) -> None:
        with self.lock:
            self.script = script
            self.release.clear()

    def get_script(self) -> Script:
        with self.lock:
            return self.script

    def record(self, stats: StreamStats) -> None:
        with self.lock:
            self.stats.append(stats)

    def last(self) -> Optional[StreamStats]:
        with self.lock:
            return self.stats[-1] if self.stats else None

    def snapshot(self) -> list[StreamStats]:
        with self.lock:
            return list(self.stats)


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators = (",", ":")).encode("utf-8") + b"\n\n"


def _chunk_frame(
    request_id: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "studiobench-pacer"

    # Quiet. A per-request line on stderr at 24 characters a chunk is 4,000 lines a reply.
    def log_message(self, fmt, *args) -> None:  # noqa: A003
        pass

    @property
    def state(self) -> PacerState:
        return self.server.state  # type: ignore[attr-defined]

    # ── plumbing ────────────────────────────────────────────────────

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            return {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/healthz", "/_pacer/healthz"):
            self._json(200, {"ok": True})
        elif path.endswith("/models"):
            with self.state.lock:
                ids = list(self.state.model_ids)
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": m, "object": "model", "owned_by": "studiobench"} for m in ids],
                },
            )
        elif path == "/_pacer/stats":
            with self.state.lock:
                self._json(
                    200,
                    {
                        "requests_seen": self.state.requests_seen,
                        "streams": [s.as_dict() for s in self.state.stats],
                    },
                )
        elif path == "/_pacer/last":
            last = self.state.last()
            self._json(200, last.as_dict() if last else {})
        else:
            self._json(404, {"error": {"message": f"no route {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        body = self._read_body()
        if path == "/_pacer/script":
            self.state.set_script(
                Script(
                    reasoning = body.get("reasoning", ""),
                    content = body.get("content", ""),
                    chunk_chars = int(body.get("chunk_chars", CAD_FIELD[0])),
                    gap_ms = int(body.get("gap_ms", CAD_FIELD[1])),
                    model = body.get("model", "studiobench-pacer"),
                    tag = body.get("tag", ""),
                    hold_after_chars = body.get("hold_after_chars"),
                )
            )
            self._json(200, {"ok": True})
        elif path == "/_pacer/release":
            self.state.release.set()
            self._json(200, {"ok": True})
        elif path == "/_pacer/reset":
            with self.state.lock:
                self.state.stats.clear()
                self.state.requests_seen = 0
            self._json(200, {"ok": True})
        elif path.endswith("/chat/completions"):
            self._stream(body)
        else:
            self._json(404, {"error": {"message": f"no route {path}"}})

    # ── the stream ──────────────────────────────────────────────────

    def _stream(self, body: dict) -> None:
        script = self.state.get_script()
        with self.state.lock:
            self.state.requests_seen += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        model = body.get("model") or script.model
        stats = StreamStats(
            request_id = request_id,
            tag = script.tag,
            started_at = time.monotonic(),
            requested_gap_ms = script.gap_ms,
        )

        if not body.get("stream", True):
            # A non-streaming request is a bug in the driver, not a mode: every mechanism this
            # tool measures is on the streaming path. Say so rather than serve it.
            self._json(
                400,
                {
                    "error": {
                        "message": "the studiobench pacer only serves stream=true; a non-streaming "
                        "request would skip every mechanism this benchmark exists to measure",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        # CHUNKED, explicitly. On HTTP/1.1 a response with neither Content-Length nor
        # Transfer-Encoding is a keep-alive response of unknown length, and the reader blocks
        # forever waiting for a body that already arrived -- measured here as a client that hung
        # past a 60s timeout having received every byte. Uvicorn, which is what Unsloth's own
        # backend streams through, sends chunked for a StreamingResponse, so this is also the
        # framing the relay and the browser see in production.
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # The role chunk first, as every OpenAI-compatible server sends it. The app tolerates its
        # absence, but its presence is what makes the first delta an APPEND rather than a create,
        # which is the path the cumulative buffer takes for every chunk after it.
        chunks: list[tuple[str, str]] = []
        for piece in _split(script.reasoning, script.chunk_chars):
            chunks.append(("reasoning", piece))
        for piece in _split(script.content, script.chunk_chars):
            chunks.append(("content", piece))

        gap_s = max(script.gap_ms, 1) / 1000.0
        t0 = time.monotonic()
        sent = 0
        try:
            self._write(_sse(_chunk_frame(request_id, model, {"role": "assistant"})), stats)
            last_activity = time.monotonic()
            while sent < len(chunks):
                now = time.monotonic()
                # DEFICIT SCHEDULING. How many chunks SHOULD have gone out by now, from wall
                # clock alone; send the shortfall in one burst. Never sleep(gap) per chunk.
                should_have_sent = int((now - t0) / gap_s)
                deficit = min(should_have_sent, len(chunks)) - sent
                if deficit <= 0:
                    if now - last_activity >= KEEPALIVE_S:
                        self._write(b": keep-alive\n\n", stats)
                        stats.keepalives += 1
                        last_activity = now
                    # Sleep to the next boundary, not a fixed tick: waking late is the SIGNAL, and
                    # the burst on the next pass is what a real backend does when a socket that
                    # was backed up clears.
                    target = t0 + (sent + 1) * gap_s
                    time.sleep(max(0.0, min(target - now, KEEPALIVE_S)))
                    continue
                if deficit > 1:
                    stats.bursts += 1
                    stats.max_deficit = max(stats.max_deficit, deficit)
                payload = bytearray()
                for _ in range(deficit):
                    kind, piece = chunks[sent]
                    if kind == "reasoning":
                        # `content: ""` alongside, which is exactly what the backend's own
                        # `_gguf_chat_delta_line` emits. Without it the app takes a different
                        # branch in delta accumulation and the cumulative <think> buffer is
                        # assembled by a path the shipping build never uses.
                        delta = {"reasoning_content": piece, "content": ""}
                        stats.reasoning_chars_sent += len(piece)
                    else:
                        delta = {"content": piece}
                        stats.content_chars_sent += len(piece)
                    payload += _sse(_chunk_frame(request_id, model, delta))
                    sent += 1
                    stats.chunks_sent += 1
                    stats.chars_sent += len(piece)
                    if (
                        script.hold_after_chars is not None
                        and stats.chars_sent >= script.hold_after_chars
                    ):
                        break
                self._write(bytes(payload), stats)
                last_activity = time.monotonic()
                if stats.first_chunk_ms is None:
                    stats.first_chunk_ms = (last_activity - t0) * 1000
                stats.last_chunk_ms = (last_activity - t0) * 1000
                if (
                    script.hold_after_chars is not None
                    and stats.chars_sent >= script.hold_after_chars
                ):
                    # HOLD, do not close. The stop-generation action needs the app to be visibly
                    # mid-stream while it presses stop; closing the socket instead would end the
                    # stream on its own and the action would measure a stream that had already
                    # finished. The keep-alives make the hold indistinguishable from a model
                    # thinking, which is what it is standing in for.
                    stats.held = True
                    while not self.state.release.wait(KEEPALIVE_S):
                        self._write(b": keep-alive\n\n", stats)
                        stats.keepalives += 1
                    break

            # BOTH terminal signals. `streamChatCompletions` throws StreamInterruptedError at EOF
            # unless it saw a finish_reason or a [DONE]; sending one and not the other measures the
            # app's error path and calls it a benchmark.
            self._write(_sse(_chunk_frame(request_id, model, {}, finish_reason = "stop")), stats)
            prompt_tokens = max(1, len(script.reasoning) // 4)
            completion_tokens = max(1, stats.chars_sent // 4)
            self._write(
                _sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                ),
                stats,
            )
            self._write(b"data: [DONE]\n\n", stats)
            self._end_chunked(stats)
            stats.completed = True
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as exc:
            # The app cancelled, or the relay went away. A first-class outcome, not an error: the
            # stop-generation action produces exactly this.
            stats.disconnected = True
            stats.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            stats.duration_ms = (time.monotonic() - t0) * 1000
            if stats.chunks_sent > 1 and stats.last_chunk_ms and stats.first_chunk_ms is not None:
                span = stats.last_chunk_ms - stats.first_chunk_ms
                stats.achieved_gap_ms = span / max(1, stats.chunks_sent - 1)
            self.state.record(stats)

    def _write(self, raw: bytes, stats: StreamStats) -> None:
        """Write, and CHARGE THE TIME to the stats.

        A blocking write is backpressure: the relay's receive buffer is full because the browser
        has not drained it because the main thread has not run. That is the same jam the frame
        recorder sees from the other side, and having both makes it attributable rather than
        merely visible.
        """
        frame = b"%x\r\n" % len(raw) + raw + b"\r\n"
        started = time.monotonic()
        self.wfile.write(frame)
        self.wfile.flush()
        stats.write_block_ms += (time.monotonic() - started) * 1000
        stats.bytes_sent += len(frame)

    def _end_chunked(self, stats: StreamStats) -> None:
        started = time.monotonic()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        stats.write_block_ms += (time.monotonic() - started) * 1000


class Pacer:
    """The server, its thread, and the driver-side handle on both."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.state = PacerState()
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._server.state = self.state  # type: ignore[attr-defined]
        self.host, self.port = self._server.server_address[:2]
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        """What Unsloth's provider config points at. The `/v1` is load-bearing: the backend appends
        `/chat/completions` to it verbatim."""
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> "Pacer":
        self._thread = threading.Thread(
            target = self._server.serve_forever, kwargs = {"poll_interval": 0.1}, daemon = True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._server.server_close()
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout = 5)

    # ── driver-side control ─────────────────────────────────────────

    def load(
        self,
        reasoning: str,
        content: str,
        *,
        cadence: str = "field",
        tag: str = "",
        hold_after_chars: Optional[int] = None,
        model: str = "studiobench-pacer",
        chunk_chars: Optional[int] = None,
        gap_ms: Optional[int] = None,
    ) -> None:
        default_chunk, default_gap = CADENCES[cadence]
        chunk_chars = default_chunk if chunk_chars is None else chunk_chars
        gap_ms = default_gap if gap_ms is None else gap_ms
        self.state.model_ids = [model]
        self.state.set_script(
            Script(
                reasoning = reasoning,
                content = content,
                chunk_chars = chunk_chars,
                gap_ms = gap_ms,
                model = model,
                tag = tag,
                hold_after_chars = hold_after_chars,
            )
        )

    def release(self) -> None:
        self.state.release.set()

    def reset(self) -> None:
        with self.state.lock:
            self.state.stats.clear()
            self.state.requests_seen = 0

    def last_stats(self) -> Optional[dict]:
        last = self.state.last()
        return last.as_dict() if last else None

    def all_stats(self) -> list[dict]:
        """Every stream this pacer served since the last `reset`, in order.

        `last_stats` alone cannot answer "did the cell stream what it planned": a cell streams an
        opening reply and then one follow-up per `send_turn`, and the LAST of those is the only
        one it describes. See `check_planned_streams`.
        """
        return [s.as_dict() for s in self.state.snapshot()]

    def expected_duration_ms(
        self,
        reasoning: str,
        content: str,
        cadence: str = "field",
        chunk_chars: Optional[int] = None,
        gap_ms: Optional[int] = None,
    ) -> float:
        """What the cadence COMMITS to, before anything runs. The tier budget is built from this,
        and the deficit scheduler is what makes it true on a slow machine as well as a fast one."""
        default_chunk, default_gap = CADENCES[cadence]
        chunk_chars = default_chunk if chunk_chars is None else chunk_chars
        gap_ms = default_gap if gap_ms is None else gap_ms
        n = len(_split(reasoning, chunk_chars)) + len(_split(content, chunk_chars))
        return n * gap_ms


def check_planned_streams(streams: list[dict], planned: list[dict]) -> dict:
    """Did every turn the cell PLANNED actually stream, in full?

    A cell is not one stream. It opens with a reply and then streams one follow-up per `send_turn`,
    and until this existed the only record kept was `last_stats()` -- the LAST of them. An opening
    reply that disconnected, or delivered 4,624 of the 10,000 characters its rung is named for, was
    therefore erased by whichever turn happened to finish last, and the cell was scored COMPLETE
    against a thread thousands of characters short of the one it claims. That is the one failure a
    benchmark must never have: under-measuring and reporting success, because a reader acts on it.

    Matching is by `tag`, first unmatched stream wins, because a tag is not unique. The
    `stop_generation` action sends its OWN throwaway turn against whatever script is loaded, so it
    produces a second, deliberately cancelled stream carrying the tag of the turn before it. Those
    land in `extra` and are reported rather than validated: an aborted throwaway is the action
    working, not the cell failing.

    Returns a dict; `ok` is False when any planned turn is missing, unfinished, or short.
    """
    remaining = list(streams)
    turns: list[dict] = []
    ok = True
    for want in planned:
        tag = want.get("tag")
        chars = int(want.get("chars") or 0)
        got = next((s for s in remaining if s.get("tag") == tag), None)
        if got is not None:
            remaining.remove(got)
        sent = int((got or {}).get("chars_sent") or 0)
        complete = bool((got or {}).get("completed"))
        entry = {
            "tag": tag,
            "turn": want.get("turn"),
            "planned_chars": chars,
            "chars_sent": sent if got is not None else None,
            "completed": complete if got is not None else None,
            "disconnected": bool((got or {}).get("disconnected")) if got is not None else None,
            "found": got is not None,
        }
        if got is None:
            entry["reason"] = (
                f"no stream carried the tag {tag!r}, so this turn never reached the pacer"
            )
        elif not complete:
            entry["reason"] = (
                f"the stream tagged {tag!r} did not complete "
                f"({sent} of {chars} characters sent, "
                f"{'the client disconnected' if entry['disconnected'] else 'no terminal frame'})"
            )
        elif sent < chars:
            entry["reason"] = (
                f"the stream tagged {tag!r} delivered {sent} of the {chars} characters planned"
            )
        else:
            entry["reason"] = None
        entry["ok"] = entry["reason"] is None
        ok = ok and entry["ok"]
        turns.append(entry)
    failures = [t for t in turns if not t["ok"]]
    return {
        "ok": ok,
        "checked": bool(planned),
        "planned_turns": len(planned),
        "turns": turns,
        # The throwaway turns and any retry, kept so a reader can see WHAT else the fixture served
        # without them being mistaken for a planned turn that failed.
        "extra": remaining,
        "reason": None if ok else "; ".join(t["reason"] for t in failures),
    }


def _selftest() -> int:
    """Serve one scripted reply to a plain socket client and check the wire bytes.

    Run with `python -m tests.studio.studiobench.pacer`. No browser, no Unsloth, no Playwright: the
    pacer's contract is with the wire, and the wire is checkable on its own.
    """
    import urllib.request

    pacer = Pacer().start()
    try:
        reasoning = "R" * 480
        content = "C" * 240
        pacer.load(reasoning, content, cadence = "fast", tag = "selftest")
        req = urllib.request.Request(
            f"{pacer.base_url}/chat/completions",
            data = json.dumps(
                {
                    "model": "studiobench-pacer",
                    "stream": True,
                    "messages": [{"role": "user", "content": "go"}],
                    "stream_options": {"include_usage": True},
                }
            ).encode(),
            headers = {"Content-Type": "application/json", "Authorization": "Bearer sb-local"},
        )
        started = time.monotonic()
        raw = urllib.request.urlopen(req, timeout = 60).read().decode("utf-8")
        elapsed_ms = (time.monotonic() - started) * 1000

        events = [e for e in raw.split("\n\n") if e.strip()]
        datas = [e[len("data: ") :] for e in events if e.startswith("data: ")]
        assert datas[-1] == "[DONE]", f"no [DONE] terminator, got {datas[-1][:80]!r}"
        frames = [json.loads(d) for d in datas[:-1]]
        finish = [
            f for f in frames if f.get("choices") and f["choices"][0].get("finish_reason") == "stop"
        ]
        assert finish, "no finish_reason:stop chunk"
        usage = [f for f in frames if f.get("usage")]
        assert usage, "no usage chunk"
        reasoning_frames = [
            f
            for f in frames
            if f.get("choices") and "reasoning_content" in f["choices"][0]["delta"]
        ]
        assert reasoning_frames, "no reasoning_content deltas"
        for f in reasoning_frames:
            assert (
                f["choices"][0]["delta"].get("content") == ""
            ), "reasoning_content must carry content:'' alongside, as _gguf_chat_delta_line does"
        got_reasoning = "".join(
            f["choices"][0]["delta"]["reasoning_content"] for f in reasoning_frames
        )
        assert got_reasoning == reasoning, "reasoning did not round-trip"
        content_frames = [
            f for f in frames if f.get("choices") and f["choices"][0]["delta"].get("content")
        ]
        got_content = "".join(f["choices"][0]["delta"]["content"] for f in content_frames)
        assert got_content == content, "content did not round-trip"

        stats = pacer.last_stats()
        assert stats and stats["completed"], f"stream did not complete: {stats}"
        expected = pacer.expected_duration_ms(reasoning, content, "fast")
        # The deficit scheduler's whole promise: duration is set by wall clock, not by how fast
        # the reader is. Generous bound because a loaded CI box is exactly the case it must hold in.
        assert (
            elapsed_ms >= expected * 0.75
        ), f"stream finished in {elapsed_ms:.0f}ms, faster than the {expected:.0f}ms cadence"
        assert (
            elapsed_ms <= expected * 2.5 + 1000
        ), f"stream took {elapsed_ms:.0f}ms against a {expected:.0f}ms cadence"

        # /v1/models, which the SPA's model list may probe.
        models = json.loads(urllib.request.urlopen(f"{pacer.base_url}/models", timeout = 10).read())
        assert models["data"][0]["id"] == "studiobench-pacer", models

        print(
            f"pacer selftest OK: {stats['chunks_sent']} chunks, {stats['chars_sent']} chars, "
            f"{elapsed_ms:.0f}ms against a {expected:.0f}ms cadence, "
            f"achieved gap {stats['achieved_gap_ms']}ms against {stats['requested_gap_ms']}ms, "
            f"{stats['bursts']} bursts, max deficit {stats['max_deficit']}"
        )

        # PHASE 2: the deficit claim itself. A reader that stalls must NOT slow the stream down;
        # it must come back to a burst and the run must still end on wall clock. This is the
        # difference between measuring a renderer and measuring a server that waits for one.
        pacer.reset()
        # Big enough to exhaust the loopback send buffer, which autotunes to `wmem_max` (4MB on
        # this kernel). A 400KB reply does not block a loopback write no matter how long the
        # reader sleeps, so a smaller body tests nothing.
        body = "D" * 12_000_000
        cad = {"chunk_chars": 12_000, "gap_ms": 1}
        pacer.load("", body, tag = "backpressure", **cad)
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # A small receive buffer is what makes this a test. With the default buffer the kernel
        # absorbs a 17KB reply whole, the pacer's writes never block, it is never late, and there
        # is correctly no deficit to schedule -- which is what the first version of this check
        # measured and wrongly called a failure. Backpressure needs a reader that cannot keep up
        # AND a pipe that fills, which is exactly the case a jammed renderer produces.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        conn.settimeout(60)
        conn.connect((pacer.host, pacer.port))
        payload = json.dumps(
            {
                "model": "studiobench-pacer",
                "stream": True,
                "messages": [{"role": "user", "content": "go"}],
            }
        ).encode()
        conn.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + b"Content-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
        )
        started = time.monotonic()
        # Read nothing at all for a stretch several gaps long, then drain. The socket buffer
        # absorbs it, so the pacer keeps its own clock and owes a shortfall when it next looks.
        time.sleep(0.4)
        conn.setblocking(True)
        seen = b""
        while b"[DONE]" not in seen:
            got = conn.recv(65536)
            if not got:
                break
            seen += got
        stalled_ms = (time.monotonic() - started) * 1000
        conn.close()
        expected2 = pacer.expected_duration_ms("", body, **cad)
        s2 = pacer.last_stats()
        assert s2 and s2["completed"], f"stalled stream did not complete: {s2}"
        assert s2["bursts"] > 0, (
            "a 400ms reader stall produced no deficit burst, so the pacer is sleeping per chunk "
            f"rather than scheduling against wall clock: {s2}"
        )
        # The honest form of the machine-independence claim. No scheduler can make a stream finish
        # faster than the reader consumes it, so total duration is NOT bounded by the cadence when
        # the reader is the throughput limit. What the deficit scheduler promises is narrower and
        # is exactly what is checked here: the pacer itself adds no delay beyond the cadence, so
        # every millisecond of overrun is accounted for by time blocked in a write. If it were
        # sleeping a gap per chunk instead, the overrun would exceed the blocked time by the
        # stall, since each late chunk would then push the NEXT one late as well.
        unblocked_ms = stalled_ms - s2["write_block_ms"]
        assert unblocked_ms <= expected2 * 2.5 + 1000, (
            f"the stream spent {unblocked_ms:.0f}ms not blocked on a write, against a "
            f"{expected2:.0f}ms cadence: the pacer is adding delay of its own"
        )
        print(
            f"pacer backpressure OK: {s2['bursts']} bursts, max deficit {s2['max_deficit']}, "
            f"{stalled_ms:.0f}ms total of which {s2['write_block_ms']:.0f}ms blocked on a "
            f"jammed reader, leaving {unblocked_ms:.0f}ms against a {expected2:.0f}ms cadence"
        )
        return 0
    finally:
        pacer.stop()


if __name__ == "__main__":
    raise SystemExit(_selftest())
