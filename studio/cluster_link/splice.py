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


def set_nodelay(writer: asyncio.StreamWriter) -> None:
    # ggml-rpc is request/response with small headers: Nagle would hold every command for an ACK.
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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


async def _pump(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    counters: Optional[Dict[str, int]],
    key: str,
) -> None:
    while True:
        data = await reader.read(CHUNK)
        if not data:
            return
        if counters is not None:
            counters[key] = counters.get(key, 0) + len(data)
        writer.write(data)
        await writer.drain()


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
    set_nodelay(a_writer)
    set_nodelay(b_writer)
    tasks = [
        asyncio.ensure_future(_pump(a_reader, b_writer, counters, a_to_b)),
        asyncio.ensure_future(_pump(b_reader, a_writer, counters, b_to_a)),
    ]
    clean = False
    try:
        done, _ = await asyncio.wait(tasks, return_when = asyncio.FIRST_COMPLETED)
        clean = all(not t.cancelled() and t.exception() is None for t in done)
    finally:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions = True)
        clean = clean and not any(
            isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)
            for r in results
        )
        if on_done is not None:
            on_done()
        # A clean EOF was fully written and drained by its pump: flush-close. Errors abort.
        await asyncio.gather(
            close_writer(a_writer, abort = not clean),
            close_writer(b_writer, abort = not clean),
            return_exceptions = True,
        )
