# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The replica front end exists so that one dead engine does not reach the client.

It did not do that. Round robin chose a backend, the connect failed, and the connection was
closed, so with two replicas every other request failed while a healthy engine sat idle. These
tests drive a real socket through a real event loop rather than stubbing `open_connection`,
because the behaviour under test is what a client sees.
"""

from __future__ import annotations

import asyncio
import importlib.util
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _lb_module():
    spec = importlib.util.spec_from_file_location("spark_lb", REPO / "studio" / "spark_lb.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    """A port nothing is listening on: bound, read back, then released."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _echo_backend(tag: bytes):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(64)
        writer.write(tag)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _ask(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"ping")
    await writer.drain()
    body = await reader.read(64)
    writer.close()
    return body


async def _front_end(lb, backends):
    server = await asyncio.start_server(
        lb._handler(backends, __import__("itertools").count()), "127.0.0.1", 0
    )
    return server, server.sockets[0].getsockname()[1]


def test_every_request_is_served_while_one_replica_is_down() -> None:
    """Four requests over two replicas, one of them dead. All four must be answered."""

    async def scenario() -> list:
        lb = _lb_module()
        alive, alive_port = await _echo_backend(b"alive")
        dead_port = _free_port()
        front, front_port = await _front_end(
            lb, [("127.0.0.1", dead_port), ("127.0.0.1", alive_port)]
        )
        try:
            return [await _ask(front_port) for _ in range(4)]
        finally:
            front.close()
            alive.close()

    replies = asyncio.run(scenario())
    assert replies == [b"alive"] * 4, replies


def test_a_healthy_pair_still_alternates() -> None:
    """The retry must not collapse the balancing it was added to protect."""

    async def scenario() -> list:
        lb = _lb_module()
        first, first_port = await _echo_backend(b"first")
        second, second_port = await _echo_backend(b"second")
        front, front_port = await _front_end(
            lb, [("127.0.0.1", first_port), ("127.0.0.1", second_port)]
        )
        try:
            return [await _ask(front_port) for _ in range(4)]
        finally:
            front.close()
            first.close()
            second.close()

    replies = asyncio.run(scenario())
    assert set(replies) == {b"first", b"second"}, replies
    assert replies[0] != replies[1], f"consecutive requests went to the same engine: {replies}"


def test_the_client_is_released_when_no_replica_answers() -> None:
    """With nothing to serve, the front end must close rather than hang."""

    async def scenario() -> bytes:
        lb = _lb_module()
        ports = [_free_port(), _free_port()]
        front, front_port = await _front_end(lb, [("127.0.0.1", p) for p in ports])
        try:
            return await asyncio.wait_for(_ask(front_port), timeout = 5)
        finally:
            front.close()

    assert asyncio.run(scenario()) == b""
