# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Byte splicing between two asyncio streams, used by the agent and the relay alike."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
from typing import Callable, Dict, Optional

CHUNK = 256 * 1024
STREAM_LIMIT = 256 * 1024
CLOSE_TIMEOUT_S = 2.0

_STREAM_ERRORS = (OSError, ssl.SSLError, asyncio.IncompleteReadError, ConnectionError)


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

    def __init__(self, transport, old_protocol, counters, key, done: asyncio.Future) -> None:
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
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, counters, key, done
) -> tuple:
    transport = writer.transport
    pipe = _Pipe(transport, transport.get_protocol(), counters, key, done)
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
    a, a_pending, a_eof = _take_over(a_reader, a_writer, counters, a_to_b, done)
    b, b_pending, b_eof = _take_over(b_reader, b_writer, counters, b_to_a, done)
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
        if on_done is not None:
            on_done()
        # A clean EOF: flush what is buffered, then close. Errors and cancellation abort.
        await asyncio.gather(_close(a, not clean), _close(b, not clean), return_exceptions = True)
