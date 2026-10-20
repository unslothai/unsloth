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


# Threaded sessions (the default): once an RPC session is admitted its bytes move in one plain thread
# over two non-blocking sockets, with no event loop, protocol callbacks or MemoryBIO copies between
# them. asyncio's TLS layer roughly doubled the cost of the TLS link on ggml-rpc's round trips.
# UNSLOTH_CLUSTER_LINK_ASYNC_SPLICE=1 restores the asyncio path above.
ASYNC_SPLICE_ENV = "UNSLOTH_CLUSTER_LINK_ASYNC_SPLICE"
WRITE_MAX = 256 * 1024
OUT_LIMIT = 4 * 1024 * 1024
SELECT_TIMEOUT_S = 1.0


def threaded_sessions() -> bool:
    return os.environ.get(ASYNC_SPLICE_ENV, "").strip().lower() not in ("1", "true", "yes", "on")


def tune_raw_socket(sock: socket.socket) -> None:
    class _W:  # tune_socket reads the socket from a StreamWriter
        def get_extra_info(
            self,
            name,
            default = None,
        ):
            return sock if name == "socket" else default

    tune_socket(_W())


class SpliceSession:
    """One session's two sockets; ``abort`` may be called from any thread (revocation, stop)."""

    def __init__(self) -> None:
        self._r, self._w = socket.socketpair()
        self._r.setblocking(False)
        self._w.setblocking(False)
        self.aborted = False
        self.sockets: List[socket.socket] = []

    def abort(self) -> None:
        self.aborted = True
        try:
            self._w.send(b"x")
        except OSError:
            pass

    def close(self) -> None:
        for s in (self._r, self._w):
            try:
                s.close()
            except OSError:
                pass


class _Dir:
    __slots__ = ("src", "dst", "key", "out", "inflight", "eof")

    def __init__(self, src, dst, key) -> None:
        self.src, self.dst, self.key = src, dst, key
        self.out = bytearray()
        self.inflight = 0
        self.eof = False


def _send_some(d: _Dir) -> bool:
    """Push d.out into d.dst until it would block; True if it blocked with bytes still queued."""
    while d.out:
        # An SSL write that wanted to wait must be retried with the same length.
        n_try = d.inflight or min(len(d.out), WRITE_MAX)
        mv = memoryview(d.out)
        part = mv[:n_try]
        try:
            n = d.dst.send(part)
        except (BlockingIOError, InterruptedError, ssl.SSLWantWriteError, ssl.SSLWantReadError):
            d.inflight = n_try if isinstance(d.dst, ssl.SSLSocket) else 0
            return True
        finally:
            part.release()
            mv.release()
        d.inflight = 0
        del d.out[:n]
    return False


def _recv_some(d: _Dir, view: memoryview, counters, trace) -> None:
    while len(d.out) < OUT_LIMIT:
        try:
            n = d.src.recv_into(view)
        except (BlockingIOError, InterruptedError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return
        if n == 0:
            d.eof = True
            return
        d.out += view[:n]
        if counters is not None:
            counters[d.key] = counters.get(d.key, 0) + n
        if trace is not None:
            trace.chunk(d.key, n)


def _shut(sock: socket.socket, clean: bool) -> None:
    try:
        if clean:
            sock.settimeout(CLOSE_TIMEOUT_S)
            if isinstance(sock, ssl.SSLSocket):
                try:
                    sock.unwrap()  # close_notify; a peer that never answers costs CLOSE_TIMEOUT_S at most
                except (OSError, ValueError, ssl.SSLError):
                    pass
            else:
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
    finally:
        try:
            sock.close()
        except OSError:
            pass


def socket_splice(
    a: socket.socket,
    b: socket.socket,
    counters: Optional[Dict[str, int]] = None,
    a_to_b: str = "a_to_b",
    b_to_a: str = "b_to_a",
    on_done: Optional[Callable[[], None]] = None,
    session: Optional[SpliceSession] = None,
    pending_a: bytes = b"",
) -> bool:
    """splice() for the threaded path: copy both ways in the calling thread until either side ends or
    ``session.abort()`` is called, then close both. One thread drives both sockets, so an SSL object is
    never read and written concurrently (OpenSSL forbids it). Returns True on a clean EOF."""
    import selectors

    session = session or SpliceSession()
    session.sockets = [a, b]
    for s in (a, b):
        tune_raw_socket(s)
        s.setblocking(False)
    trace_path = os.environ.get(TRACE_ENV)
    trace = _Trace(trace_path, (a_to_b, b_to_a)) if trace_path else None
    da, db = _Dir(a, b, a_to_b), _Dir(b, a, b_to_a)
    if pending_a:
        da.out += pending_a
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    sel = selectors.DefaultSelector()
    sel.register(session._r, selectors.EVENT_READ, None)
    masks = {a: 0, b: 0}
    clean = False
    try:
        while not session.aborted:
            for d in (da, db):
                if not d.eof:
                    _recv_some(d, view, counters, trace)
            blocked = {da.dst: _send_some(da), db.dst: _send_some(db)}
            ended = [d for d in (da, db) if d.eof]
            if ended and not any(d.out for d in ended):
                # The first EOF ends the session (TLS has no half-close); what it queued is delivered.
                clean = True
                break
            # SSL may already hold decrypted bytes the fd will never signal: read those without waiting.
            if any(
                isinstance(d.src, ssl.SSLSocket)
                and not d.eof
                and len(d.out) < OUT_LIMIT
                and d.src.pending()
                for d in (da, db)
            ):
                continue
            for d in (da, db):
                want = 0
                if not d.eof and len(d.out) < OUT_LIMIT:
                    want |= selectors.EVENT_READ
                want_w = (
                    selectors.EVENT_WRITE if blocked[d.src] else 0
                )  # d.src is the other dir's dst
                mask = want | want_w
                if mask != masks[d.src]:
                    if masks[d.src] and mask:
                        sel.modify(d.src, mask)
                    elif mask:
                        sel.register(d.src, mask)
                    else:
                        sel.unregister(d.src)
                    masks[d.src] = mask
            sel.select(SELECT_TIMEOUT_S)
    except (OSError, ssl.SSLError, ValueError):
        clean = False
    finally:
        sel.close()
        if trace is not None:
            trace.write(clean)
        if on_done is not None:
            on_done()
        if clean and not session.aborted:
            for d in (da, db):
                if d.out:
                    try:
                        d.dst.settimeout(CLOSE_TIMEOUT_S)
                        d.dst.sendall(bytes(d.out))
                    except (OSError, ssl.SSLError):
                        clean = False
        for s in (a, b):
            _shut(s, clean and not session.aborted)
        session.close()
    return clean and not session.aborted
