# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The round-robin front end over the llama.cpp engines."""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason = "runs the peer's bash script")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_review_fixes", "studio/spark_pipeline.py")


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_review_fixes", "studio/spark_cluster.py")


@pytest.fixture
def lb():
    return _load("spark_lb_for_review_fixes", "studio/spark_lb.py")


def test_a_half_closed_client_still_gets_the_whole_response(lb) -> None:
    async def scenario():
        async def backend(reader, writer):
            await reader.read(1024)
            await asyncio.sleep(0.1)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(backend, "127.0.0.1", 0)
        port = upstream.sockets[0].getsockname()[1]
        front = await asyncio.start_server(
            lb._handler([("127.0.0.1", port)], itertools.count()), "127.0.0.1", 0
        )
        r, w = await asyncio.open_connection("127.0.0.1", front.sockets[0].getsockname()[1])
        w.write(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        await w.drain()
        w.write_eof()
        body = await asyncio.wait_for(r.read(), 5)
        w.close()
        front.close()
        upstream.close()
        return body

    assert asyncio.run(scenario()).endswith(b"OK")


def test_a_lost_backend_is_not_retried_on_every_request(lb, monkeypatch) -> None:
    """With the peer gone every other request waited out the full connect timeout first."""
    monkeypatch.setattr(lb, "CONNECT_TIMEOUT", 0.3)

    async def scenario():
        async def ok(reader, writer):
            await reader.read(100)
            writer.write(b"HTTP/1.0 200 OK\r\n\r\nok")
            await writer.drain()
            writer.close()

        good = await asyncio.start_server(ok, "127.0.0.1", 0)
        good_port = good.sockets[0].getsockname()[1]
        attempts = []
        real_open = asyncio.open_connection

        async def open_connection(host, port, *a, **k):
            if port == 1:
                attempts.append(port)
                await asyncio.sleep(10)  # dropped SYNs: the connect hangs
            return await real_open(host, port, *a, **k)

        monkeypatch.setattr(lb.asyncio, "open_connection", open_connection)
        front = await asyncio.start_server(
            lb._handler([("192.0.2.1", 1), ("127.0.0.1", good_port)], itertools.count()),
            "127.0.0.1",
            0,
        )
        front_port = front.sockets[0].getsockname()[1]
        started = time.monotonic()
        for _ in range(6):
            r, w = await real_open("127.0.0.1", front_port)
            w.write(b"GET / HTTP/1.0\r\n\r\n")
            await w.drain()
            assert (await asyncio.wait_for(r.read(), 5)).endswith(b"ok")
            w.close()
        elapsed = time.monotonic() - started
        front.close()
        good.close()
        return attempts, elapsed

    attempts, elapsed = asyncio.run(scenario())
    assert len(attempts) == 1, attempts
    assert elapsed < 6 * 0.3 / 2 + 0.6, elapsed
