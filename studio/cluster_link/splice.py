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
import threading
import time
from typing import Any, Callable, Dict, List, Optional

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


# Threaded sessions (the default): once an RPC session is admitted its bytes move in two plain
# threads, one per direction, blocked in recv on each socket, with no event loop, selector, protocol
# callbacks or MemoryBIO copies in between. asyncio's TLS layer cost several times the TLS crypto on
# ggml-rpc's round trips. UNSLOTH_CLUSTER_LINK_ASYNC_SPLICE=1 restores the asyncio path above.
ASYNC_SPLICE_ENV = "UNSLOTH_CLUSTER_LINK_ASYNC_SPLICE"
WRITE_MAX = 256 * 1024
WAIT_S = 1.0


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
        self.aborted = False
        self.sockets: List[socket.socket] = []

    def abort(self) -> None:
        self.aborted = True
        _shutdown_all(self.sockets)

    def close(self) -> None:
        pass


def _shutdown_all(socks) -> None:
    for sock in list(socks):
        try:
            # The base method: SSLSocket.shutdown would also drop its SSL object under a running reader.
            socket.socket.shutdown(sock, socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass


TLS_RECORD = 16 * 1024 + 256


def _readable_now(sock: socket.socket) -> bool:
    import select
    if hasattr(select, "poll"):
        poller = select.poll()
        poller.register(sock.fileno(), select.POLLIN)
        return bool(poller.poll(0))
    return bool(select.select([sock], [], [], 0)[0])


def _wait_fd(sock: socket.socket, write: bool, timeout: float) -> None:
    """Block until ``sock`` is readable / writable (or ``timeout``), without touching its SSL object."""
    import select
    if hasattr(select, "poll"):
        poller = select.poll()
        poller.register(sock.fileno(), select.POLLOUT if write else select.POLLIN)
        poller.poll(timeout * 1000.0)
    elif write:
        select.select([], [sock], [sock], timeout)
    else:
        select.select([sock], [], [sock], timeout)


class _End:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.first: Optional[bool] = None  # True: a clean EOF ended the session; False: an error

    def set(self, clean: bool) -> bool:
        with self.lock:
            if self.first is None:
                self.first = clean
                return True
            return False


def _pump(
    src,
    dst,
    locks,
    counters,
    key,
    trace,
    session,
    end,
    pending: bytes = b"",
) -> None:
    """One direction: read src, write dst, until either ends. A TLS socket is non-blocking and only
    touched under its lock, and nothing waits while holding it, so the other direction's thread can
    write it meanwhile. Plain sockets just block in recv / sendall."""
    src_lock, dst_lock = locks.get(id(src)), locks.get(id(dst))
    buf = bytearray(CHUNK)
    view = memoryview(buf)

    def write(data) -> None:
        if dst_lock is None:
            dst.sendall(data)
            return
        mv = memoryview(data)
        off, inflight = 0, 0
        while off < len(mv):
            # An SSL write that had to wait must be retried with the same length.
            n_try = inflight or min(len(mv) - off, WRITE_MAX)
            with dst_lock:
                try:
                    n = dst.send(mv[off : off + n_try])
                except (
                    ssl.SSLWantWriteError,
                    ssl.SSLWantReadError,
                    BlockingIOError,
                    InterruptedError,
                ):
                    n = None
            if n is None:
                if session.aborted or end.first is not None:
                    raise ConnectionAbortedError("session ended")
                inflight = n_try
                _wait_fd(dst, True, WAIT_S)
                continue
            inflight = 0
            off += n

    try:
        if pending:
            write(pending)
        while not session.aborted and end.first is None:
            if src_lock is None:
                n = src.recv_into(view)
            else:
                with src_lock:
                    ready = src.pending() > 0
                if not ready:
                    _wait_fd(src, False, WAIT_S)
                with src_lock:
                    try:
                        n = src.recv_into(view)
                    except (
                        ssl.SSLWantReadError,
                        ssl.SSLWantWriteError,
                        BlockingIOError,
                        InterruptedError,
                    ):
                        continue
                    # SSL hands out one record (<= 16 KiB) per read: take every record already here so
                    # the plain side gets one write per message, not one per record.
                    while 0 < n <= len(view) - TLS_RECORD:
                        if not src.pending() and not _readable_now(src):
                            break
                        try:
                            more = src.recv_into(view[n:])
                        except (
                            ssl.SSLWantReadError,
                            ssl.SSLWantWriteError,
                            BlockingIOError,
                            InterruptedError,
                        ):
                            break
                        if not more:
                            break  # EOF: the bytes in hand go first, the next read sees it again
                        n += more
            if n == 0:
                end.set(True)
                break
            if counters is not None:
                counters[key] = counters.get(key, 0) + n
            if trace is not None:
                with locks["trace"]:
                    trace.chunk(key, n)
            write(view[:n])
            if end.first is not None:
                break
    except (OSError, ssl.SSLError, ValueError):
        end.set(False)
    finally:
        # Either direction ending ends the session (TLS has no half-close): wake the other thread.
        _shutdown_all(session.sockets)


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
    """splice() for the threaded path: one thread per direction (the caller runs a -> b), plain
    sockets blocking, TLS sockets non-blocking behind a lock each, so an SSL object is never read and
    written at the same moment (OpenSSL forbids it) yet neither direction waits on the other. Returns
    True when a clean EOF ended the session, False on an error or ``session.abort()``."""
    session = session or SpliceSession()
    locks: Dict[Any, Any] = {"trace": threading.Lock()}
    for s in (a, b):
        tune_raw_socket(s)
        if isinstance(s, ssl.SSLSocket):
            s.setblocking(False)
            locks[id(s)] = threading.Lock()
        else:
            s.settimeout(None)
    session.sockets = [a, b]
    if session.aborted:
        _shutdown_all(session.sockets)
    trace_path = os.environ.get(TRACE_ENV)
    trace = _Trace(trace_path, (a_to_b, b_to_a)) if trace_path else None
    end = _End()
    other = threading.Thread(
        target = _pump,
        args = (b, a, locks, counters, b_to_a, trace, session, end),
        name = "unsloth-cluster-splice",
        daemon = True,
    )
    other.start()
    _pump(a, b, locks, counters, a_to_b, trace, session, end, pending_a)
    other.join()
    clean = end.first is True and not session.aborted
    if trace is not None:
        trace.write(clean)
    if on_done is not None:
        on_done()
    # No close_notify: SSLSocket.unwrap() drops its SSL object, and a TLS peer here treats a plain EOF
    # as the end of the session, which is all an RPC session's end means.
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass
    return clean
