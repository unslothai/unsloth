# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the mid-stream stall watchdog.

httpcore binds the socket read timeout once when body iteration starts, so the
old "lower the read timeout after the first chunk" trick was a no-op and a real
no-byte stall hung for the whole ~20 min first-token deadline. These tests drive
a genuine ``httpx``/``httpcore`` stream through a raw socket that emits one chunk
then goes silent, and assert the consumer raises within the short stall window,
NOT the far-off first-token deadline. No model, subprocess, or GPU needed.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
import routes.inference as inf_mod  # noqa: E402


_ONE_CHUNK = b"data: hello\n\n"


class _StubUpstream:
    """A raw HTTP/1.1 server that streams headers + one chunked SSE body chunk,
    then behaves per ``mode``:

    - ``"stall"``: hold the socket open forever, sending nothing more.
    - ``"no_first_token"``: send only headers, never any body chunk.
    - ``"complete"``: send the chunk then the terminating 0-length chunk.
    - ``"two_chunks"``: send the chunk, a short healthy gap, a second chunk,
      then the terminator (upstream is never silent for long).
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._serve, daemon = True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout = 5)

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            # Drain the whole request (head + body) so a clean close can't RST
            # the client on its unread request bytes.
            conn.settimeout(5)
            try:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    data = conn.recv(4096)
                    if not data:
                        return
                    buf += data
                head, _, body = buf.partition(b"\r\n\r\n")
                content_length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        content_length = int(line.split(b":", 1)[1].strip())
                        break
                while len(body) < content_length:
                    data = conn.recv(4096)
                    if not data:
                        break
                    body += data
            except OSError:
                return
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"\r\n"
            )
            if self.mode != "no_first_token":
                chunk = _ONE_CHUNK
                conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            if self.mode == "complete":
                conn.sendall(b"0\r\n\r\n")
                return
            if self.mode == "two_chunks":
                self._stop.wait(timeout = 0.2)
                conn.sendall(b"%x\r\n%s\r\n" % (len(_ONE_CHUNK), _ONE_CHUNK))
                conn.sendall(b"0\r\n\r\n")
                return
            # stall / no_first_token: hold the connection open and silent until
            # the client tears it down (the watchdog closes its side).
            while not self._stop.wait(timeout = 0.05):
                try:
                    conn.settimeout(0.05)
                    if conn.recv(1) == b"":
                        return
                except socket.timeout:
                    continue
                except OSError:
                    return


def test_sync_stall_raises_at_stall_window_not_first_token_deadline():
    # Bound read timeout is large (proxy for the real ~1200s) and the
    # first-token deadline is far away; only the 0.5s stall watchdog should fire.
    with _StubUpstream("stall") as server:
        with httpx.Client(timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)) as client:
            with client.stream("POST", server.url, json = {}) as response:
                started = time.monotonic()
                chunks = []
                with pytest.raises(httpx.ReadTimeout) as excinfo:
                    for chunk in LlamaCppBackend._iter_text_cancellable(
                        response,
                        stall_timeout_s = 0.5,
                        first_token_deadline = started + 100,
                        client = client,
                    ):
                        chunks.append(chunk)
                elapsed = time.monotonic() - started
    assert "".join(chunks) == _ONE_CHUNK.decode()
    assert "mid-response" in str(excinfo.value)
    # Fires on the 0.5s stall window, not the 100s first-token deadline.
    assert elapsed < 30, elapsed


def test_sync_first_token_deadline_fires_when_no_chunk_arrives():
    with _StubUpstream("no_first_token") as server:
        with httpx.Client(timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)) as client:
            with client.stream("POST", server.url, json = {}) as response:
                started = time.monotonic()
                with pytest.raises(httpx.ReadTimeout) as excinfo:
                    for _ in LlamaCppBackend._iter_text_cancellable(
                        response,
                        stall_timeout_s = 100,
                        first_token_deadline = started + 0.5,
                        client = client,
                    ):
                        pass
                elapsed = time.monotonic() - started
    assert "first token" in str(excinfo.value)
    assert elapsed < 30, elapsed


def test_sync_slow_consumer_does_not_trip_stall():
    # A consumer that sleeps longer than the stall window between chunks must
    # NOT trip the watchdog: the window measures upstream silence, not consumer
    # latency (client backpressure). Pre-fix, the watchdog counted this sleep and
    # shut the live socket down with a spurious "mid-response".
    with _StubUpstream("two_chunks") as server:
        with httpx.Client(timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)) as client:
            with client.stream("POST", server.url, json = {}) as response:
                started = time.monotonic()
                chunks = []
                for chunk in LlamaCppBackend._iter_text_cancellable(
                    response,
                    stall_timeout_s = 0.5,
                    first_token_deadline = started + 100,
                    client = client,
                ):
                    chunks.append(chunk)
                    if len(chunks) == 1:
                        time.sleep(1.2)  # > stall window, while upstream is healthy
    assert "".join(chunks) == (_ONE_CHUNK * 2).decode()


def test_async_slow_consumer_does_not_trip_stall():
    async def _run():
        with _StubUpstream("two_chunks") as server:
            async with httpx.AsyncClient(
                timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)
            ) as client:
                async with client.stream("POST", server.url, json = {}) as response:
                    started = time.monotonic()
                    chunks = []
                    async for chunk in inf_mod._aiter_llama_stream_items(
                        response.aiter_text(),
                        response = response,
                        first_token_deadline = started + 100,
                        post_first_item_read_timeout_s = 0.5,
                    ):
                        chunks.append(chunk)
                        if len(chunks) == 1:
                            await asyncio.sleep(1.2)
        assert "".join(chunks) == (_ONE_CHUNK * 2).decode()

    asyncio.run(_run())


def test_sync_complete_stream_yields_and_ends_cleanly():
    with _StubUpstream("complete") as server:
        with httpx.Client(timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)) as client:
            with client.stream("POST", server.url, json = {}) as response:
                started = time.monotonic()
                chunks = list(
                    LlamaCppBackend._iter_text_cancellable(
                        response,
                        stall_timeout_s = 5,
                        first_token_deadline = started + 100,
                        client = client,
                    )
                )
    assert "".join(chunks) == _ONE_CHUNK.decode()


def test_async_stall_raises_at_stall_window_not_first_token_deadline():
    async def _run():
        with _StubUpstream("stall") as server:
            async with httpx.AsyncClient(
                timeout = httpx.Timeout(connect = 5, read = 100, write = 5, pool = 5)
            ) as client:
                async with client.stream("POST", server.url, json = {}) as response:
                    started = time.monotonic()
                    chunks = []
                    with pytest.raises(httpx.ReadTimeout) as excinfo:
                        async for chunk in inf_mod._aiter_llama_stream_items(
                            response.aiter_text(),
                            response = response,
                            first_token_deadline = started + 100,
                            post_first_item_read_timeout_s = 0.5,
                        ):
                            chunks.append(chunk)
                    elapsed = time.monotonic() - started
        assert "".join(chunks) == _ONE_CHUNK.decode()
        assert "mid-response" in str(excinfo.value)
        assert elapsed < 30, elapsed

    asyncio.run(_run())
