# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Byte splicing between two asyncio streams, used by the agent and the relay alike."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import ssl
import time
from typing import Callable, Dict, List, Optional

CHUNK = 256 * 1024
STREAM_LIMIT = 256 * 1024
CLOSE_TIMEOUT_S = 2.0

_STREAM_ERRORS = (OSError, ssl.SSLError, asyncio.IncompleteReadError, ConnectionError)

# Diagnostics only: UNSLOTH_CLUSTER_LINK_TRACE=<file> appends one JSON line per spliced session with
# per-direction chunk counts and the turnaround times this hop sees. Unset, nothing is recorded.
TRACE_ENV = "UNSLOTH_CLUSTER_LINK_TRACE"
TINY_CHUNK = 64
MAX_TURNAROUNDS = 100_000


class _Trace:
    def __init__(self, path: str, keys) -> None:
        self.path = path
        self.started = time.perf_counter_ns()
        self.dirs = {k: {"chunks": 0, "bytes": 0, "tiny": 0, "max_chunk": 0} for k in keys}
        self.turnarounds: Dict[str, List[int]] = {k: [] for k in keys}
        self.last_key: Optional[str] = None
        self.last_ns = 0

    def chunk(self, key: str, n: int) -> None:
        now = time.perf_counter_ns()
        d = self.dirs[key]
        d["chunks"] += 1
        d["bytes"] += n
        d["tiny"] += n <= TINY_CHUNK
        d["max_chunk"] = max(d["max_chunk"], n)
        # Direction switch: the time since the other direction's last chunk (a round trip's far side).
        if self.last_key is not None and self.last_key != key:
            spans = self.turnarounds[key]
            if len(spans) < MAX_TURNAROUNDS:
                spans.append(now - self.last_ns)
        self.last_key, self.last_ns = key, now

    def write(self, clean: bool) -> None:
        def pct(xs: List[int], q: float) -> Optional[float]:
            if not xs:
                return None
            xs = sorted(xs)
            return round(xs[min(len(xs) - 1, int(q * len(xs)))] / 1000.0, 1)

        row = {
            "t": time.time(),
            "pid": os.getpid(),
            "seconds": round((time.perf_counter_ns() - self.started) / 1e9, 3),
            "clean": clean,
            "dirs": self.dirs,
            "turnaround_us": {
                k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9)}
                for k, v in self.turnarounds.items()
            },
        }
        try:
            with open(self.path, "a", encoding = "utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass


KEEPALIVE_IDLE_S = 30
KEEPALIVE_INTERVAL_S = 10
KEEPALIVE_PROBES = 6


def tune_socket(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    # ggml-rpc is request/response with small headers: Nagle would hold every command for an ACK.
    # Keepalive: a master that loses power or network sends no FIN, and without probes the worker
    # would keep the session, and report busy, until it is restarted.
    options = [
        (socket.IPPROTO_TCP, getattr(socket, "TCP_NODELAY", None), 1),
        (socket.SOL_SOCKET, getattr(socket, "SO_KEEPALIVE", None), 1),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", None), KEEPALIVE_IDLE_S),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", None), KEEPALIVE_INTERVAL_S),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", None), KEEPALIVE_PROBES),
    ]
    for level, name, value in options:
        if name is None:
            continue
        try:
            sock.setsockopt(level, name, value)
        except (OSError, AttributeError):
            pass


def is_private_address(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_private or addr.is_loopback or addr.is_link_local


async def close_writer(writer: Optional[asyncio.StreamWriter], abort: bool = False) -> None:
    if writer is None:
        return
    transport = writer.transport
    if abort:
        transport.abort()
        return
    try:
        writer.close()
        # A TLS close_notify the peer never answers would otherwise hang here.
        await asyncio.wait_for(writer.wait_closed(), CLOSE_TIMEOUT_S)
    except (asyncio.TimeoutError, *_STREAM_ERRORS):
        transport.abort()
    except asyncio.CancelledError:
        transport.abort()
        raise


class _Pipe(asyncio.Protocol):
    """Takes over one transport once the session starts: every chunk goes straight to the other
    transport from data_received, with no StreamReader buffer or task switch between them. That is
    most of a relay hop's cost on ggml-rpc's many small request/response round trips."""

    def __init__(
        self,
        transport,
        old_protocol,
        counters,
        key,
        done: asyncio.Future,
        trace = None,
    ) -> None:
        self.trace = trace
        self.transport = transport
        self.old_protocol = old_protocol
        self.peer: Optional["_Pipe"] = None
        self.counters = counters
        self.key = key
        self.done = done
        self.lost: asyncio.Future = done.get_loop().create_future()

    def _finish(self, exc: Optional[BaseException]) -> None:
        if not self.done.done():
            self.done.set_result(exc)

    def data_received(self, data: bytes) -> None:
        if self.counters is not None:
            self.counters[self.key] = self.counters.get(self.key, 0) + len(data)
        if self.trace is not None:
            self.trace.chunk(self.key, len(data))
        assert self.peer is not None
        self.peer.transport.write(data)

    def eof_received(self) -> bool:
        self._finish(None)
        return False

    # The peer's write buffer filled or drained: flow control for the transport feeding it.
    def pause_writing(self) -> None:
        if self.peer is not None and not self.peer.transport.is_closing():
            self.peer.transport.pause_reading()

    def resume_writing(self) -> None:
        if self.peer is not None and not self.peer.transport.is_closing():
            self.peer.transport.resume_reading()

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        self._finish(exc if exc is not None else None)
        if not self.lost.done():
            self.lost.set_result(None)
        # The StreamWriter's wait_closed() still waits on the protocol it was created with.
        try:
            self.old_protocol.connection_lost(exc)
        except Exception:
            pass


def _take_over(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    counters,
    key,
    done,
    trace = None,
) -> tuple:
    transport = writer.transport
    pipe = _Pipe(transport, transport.get_protocol(), counters, key, done, trace)
    # Bytes the StreamReader already holds were read before the switch and must go first.
    pending = bytes(getattr(reader, "_buffer", b""))
    getattr(reader, "_buffer", bytearray()).clear()
    transport.set_protocol(pipe)
    return pipe, pending, reader.at_eof() or getattr(reader, "_eof", False)


async def _close(pipe: _Pipe, abort: bool) -> None:
    transport = pipe.transport
    if abort:
        transport.abort()
        return
    transport.close()
    try:
        # A TLS close_notify the peer never answers would otherwise hang here.
        await asyncio.wait_for(asyncio.shield(pipe.lost), CLOSE_TIMEOUT_S)
    except asyncio.TimeoutError:
        transport.abort()
    except asyncio.CancelledError:
        transport.abort()
        raise


async def splice(
    a_reader: asyncio.StreamReader,
    a_writer: asyncio.StreamWriter,
    b_reader: asyncio.StreamReader,
    b_writer: asyncio.StreamWriter,
    counters: Optional[Dict[str, int]] = None,
    a_to_b: str = "a_to_b",
    b_to_a: str = "b_to_a",
    on_done: Optional[Callable[[], None]] = None,
) -> None:
    """Copy both ways until either side ends, then close both. TLS has no half-close, and an RPC
    session is over once either end goes, so the first EOF or error ends the whole splice.
    ``on_done`` runs once no more bytes can move, before the (possibly slow) TLS close."""
    tune_socket(a_writer)
    tune_socket(b_writer)
    done: asyncio.Future = asyncio.get_running_loop().create_future()
    trace_path = os.environ.get(TRACE_ENV)
    trace = _Trace(trace_path, (a_to_b, b_to_a)) if trace_path else None
    a, a_pending, a_eof = _take_over(a_reader, a_writer, counters, a_to_b, done, trace)
    b, b_pending, b_eof = _take_over(b_reader, b_writer, counters, b_to_a, done, trace)
    a.peer, b.peer = b, a
    clean = False
    try:
        for pipe, pending, eof in ((a, a_pending, a_eof), (b, b_pending, b_eof)):
            if pending:
                pipe.data_received(pending)
            if eof:
                pipe._finish(None)
            elif pipe.transport.is_closing():
                pipe._finish(ConnectionResetError("closed before the splice started"))
            else:
                pipe.transport.resume_reading()
        clean = (await done) is None
    finally:
        if trace is not None:
            trace.write(clean)
        if on_done is not None:
            on_done()
        # A clean EOF: flush what is buffered, then close. Errors and cancellation abort.
        await asyncio.gather(_close(a, not clean), _close(b, not clean), return_exceptions = True)
