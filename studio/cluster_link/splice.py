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
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import peer_owner

CHUNK = 256 * 1024
STREAM_LIMIT = 256 * 1024
CLOSE_TIMEOUT_S = 2.0

_STREAM_ERRORS = (OSError, ssl.SSLError, asyncio.IncompleteReadError, ConnectionError)

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


def _new_trace(a_to_b: str, b_to_a: str) -> Optional[_Trace]:
    path = os.environ.get(TRACE_ENV)
    return _Trace(path, (a_to_b, b_to_a)) if path else None


KEEPALIVE_IDLE_S = 30
KEEPALIVE_INTERVAL_S = 10
KEEPALIVE_PROBES = 6


def tune_socket(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is not None:
        tune_raw_socket(sock)


def tune_raw_socket(sock: socket.socket) -> None:
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
    found = _endpoint(host, 0)
    if found is None:
        return False
    addr = found[0]
    return addr.is_private or addr.is_loopback or addr.is_link_local


def drop_socket(sock: Any) -> None:
    for step in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
        try:
            step()
        except (OSError, ValueError):
            pass


def accept_loop(
    listener: socket.socket, alive: Callable[[], bool], handle: Callable[..., None], name: str
) -> None:
    """Hand each accepted (conn, addr) to ``handle`` on its own thread while ``alive()``."""
    while alive():
        try:
            conn, addr = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            if not alive():
                return
            time.sleep(0.05)  # EMFILE and the like: do not spin
            continue
        threading.Thread(target = handle, args = (conn, addr), name = name, daemon = True).start()


def authorized(authorize: Callable[[Any], bool], pid: Any) -> bool:
    """``authorize(pid)``, False for a missing identity or a check that raises."""
    try:
        return pid is not None and bool(authorize(pid))
    except Exception:
        return False


async def authorized_async(authorize: Callable[[Any], bool], pid: Any) -> bool:
    """authorized() off the event loop."""
    try:
        return await asyncio.to_thread(authorized, authorize, pid)
    except Exception:
        return False


async def until_unauthorized(authorize: Callable[[Any], bool], pid: Any, every: float) -> None:
    """Returns once a periodic re-check finds ``pid`` no longer paired."""
    while True:
        await asyncio.sleep(every)
        if not await authorized_async(authorize, pid):
            return


def watch_authorized(
    authorize: Callable[[Any], bool],
    pid: Any,
    every: float,
    ended: threading.Event,
    on_revoked: Callable[[], None],
    name: str,
) -> threading.Thread:
    """until_unauthorized for the threaded path: calls ``on_revoked`` unless ``ended`` is set first."""

    def watch() -> None:
        while not ended.wait(every):
            if not authorized(authorize, pid):
                on_revoked()
                return

    thread = threading.Thread(target = watch, name = name, daemon = True)
    thread.start()
    return thread


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
    """Copy both ways until either side ends, then close both."""
    tune_socket(a_writer)
    tune_socket(b_writer)
    done: asyncio.Future = asyncio.get_running_loop().create_future()
    trace = _new_trace(a_to_b, b_to_a)
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
        await asyncio.gather(_close(a, not clean), _close(b, not clean), return_exceptions = True)


THREADED_SPLICE_ENV = "UNSLOTH_CLUSTER_LINK_THREADED_SPLICE"
WRITE_MAX = 256 * 1024
WAIT_S = 1.0


def threaded_sessions() -> bool:
    return os.environ.get(THREADED_SPLICE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


_TCP_ESTABLISHED = "01"  # the `st` column of /proc/net/tcp{,6}


def _endpoint(addr: str, port: int) -> Optional[tuple]:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip, int(port)


def _proc_endpoint(field: str) -> Optional[tuple]:
    try:
        addr_hex, port_hex = field.rsplit(":", 1)
        if len(addr_hex) not in (8, 32):
            return None
        packed = b"".join(
            int(addr_hex[i : i + 8], 16).to_bytes(4, sys.byteorder)
            for i in range(0, len(addr_hex), 8)
        )
        ip = ipaddress.ip_address(packed)
        return _endpoint(str(ip), int(port_hex, 16))
    except ValueError:
        return None


def _loopback_owner_uid(client: tuple, server: tuple) -> Optional[int]:
    want_local = _endpoint(*client[:2])
    want_remote = _endpoint(*server[:2])
    if want_local is None or want_remote is None:
        return None
    uids = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding = "ascii") as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) < 8 or cols[3].upper() != _TCP_ESTABLISHED:
                continue
            if _proc_endpoint(cols[1]) != want_local or _proc_endpoint(cols[2]) != want_remote:
                continue
            try:
                uids.add(int(cols[7]))
            except ValueError:
                return None
    return uids.pop() if len(uids) == 1 else None


def local_client_is_ours(sock: socket.socket) -> bool:
    """Whether a client of a loopback listener runs as this user.

    Linux reads /proc/net/tcp{,6}; macOS and Windows ask the OS which process owns the client end
    (``peer_owner``). Every platform refuses when the owner cannot be established.
    """
    try:
        client = sock.getpeername()
        server = sock.getsockname()
    except OSError:
        return False
    if sys.platform.startswith("linux"):
        return _loopback_owner_uid(client, server) == os.getuid()
    return peer_owner.client_is_ours(client, server)


class SpliceSession:
    """One session's two sockets; ``abort`` may be called from any thread (revocation, stop)."""

    def __init__(self) -> None:
        self.aborted = False
        self.sockets: List[socket.socket] = []

    def abort(self) -> None:
        self.aborted = True
        _shutdown_all(self.sockets)


def _shutdown_all(socks) -> None:
    for sock in list(socks):
        try:
            socket.socket.shutdown(sock, socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass


TLS_RECORD = 16 * 1024 + 256


_WOULD_BLOCK = (ssl.SSLWantReadError, ssl.SSLWantWriteError, BlockingIOError, InterruptedError)


def _wait_fd(sock: socket.socket, write: bool, timeout: float) -> bool:
    """Wait until ``sock`` is writable (or readable); whether it is."""
    import select

    if hasattr(select, "poll"):
        poller = select.poll()
        poller.register(sock.fileno(), select.POLLOUT if write else select.POLLIN)
        return bool(poller.poll(timeout * 1000.0))
    if write:
        return bool(select.select([], [sock], [sock], timeout)[1])
    return bool(select.select([sock], [], [sock], timeout)[0])


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
            n_try = inflight or min(len(mv) - off, WRITE_MAX)
            with dst_lock:
                try:
                    n = dst.send(mv[off : off + n_try])
                except _WOULD_BLOCK:
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
                    except _WOULD_BLOCK:
                        continue
                    while 0 < n <= len(view) - TLS_RECORD:
                        if not src.pending() and not _wait_fd(src, False, 0):
                            break
                        try:
                            more = src.recv_into(view[n:])
                        except _WOULD_BLOCK:
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
    """splice() for the threaded path."""
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
    trace = _new_trace(a_to_b, b_to_a)
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
    # No close_notify: unwrap() drops the SSL object; a TLS peer here treats plain EOF as session end.
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass
    return clean
