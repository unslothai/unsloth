# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A `: preempted` that arrives in the same read as the response headers sits in h11's buffer
before the park-aware read is installed. The read below the httpx iterators still has to know the
stream is parked, or a stall timeout shorter than the server's park heartbeat ends the answer."""

import asyncio
import socket
import threading
import time

import httpx
import pytest

import routes.inference as inference

_HEADERS = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Connection: close\r\n\r\n"
)
_FIRST = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
_LAST = b'data: {"choices":[{"delta":{"content":"b"}}]}\n\ndata: [DONE]\n\n'


def _serve(with_headers: bytes, after_silence: bytes, silence_s: float, then = None):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def run():
        conn, _ = listener.accept()
        try:
            conn.settimeout(5)
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(65536)
            conn.sendall(with_headers)
            time.sleep(silence_s)
            conn.sendall(after_silence)
            if then is not None:
                time.sleep(then[1])
                conn.sendall(then[0])
        finally:
            conn.close()
            listener.close()

    threading.Thread(target = run, daemon = True).start()
    return listener.getsockname()[1]


async def _relay(port: int, stall_s: float) -> bytes:
    async with httpx.AsyncClient(timeout = httpx.Timeout(10.0)) as client:
        request = client.build_request("POST", f"http://127.0.0.1:{port}/v1/chat/completions")
        response = await client.send(request, stream = True)
        try:
            seen = b""
            async for item in inference._aiter_llama_stream_items(
                response.aiter_bytes(),
                first_token_deadline = time.monotonic() + 10.0,
                response = response,
                post_first_item_read_timeout_s = stall_s,
                # A build that writes the notices: the aggregate probe is retired.
                stall_grace = lambda: False,
            ):
                if isinstance(item, bytes):
                    seen += item
            return seen
        finally:
            await response.aclose()


def test_a_park_read_with_the_headers_waits_for_the_resume():
    port = _serve(_HEADERS + _FIRST + b": preempted\n\n", b": resumed\n\n" + _LAST, 1.0)
    seen = asyncio.run(_relay(port, stall_s = 0.3))
    assert seen.endswith(b"data: [DONE]\n\n")


def test_a_park_read_on_its_own_still_waits_for_the_resume():
    """The control: the same park in a read of its own was always excused."""
    port = _serve(_HEADERS + _FIRST, b": preempted\n\n", 0.1, then = (b": resumed\n\n" + _LAST, 1.0))
    seen = asyncio.run(_relay(port, stall_s = 0.3))
    assert seen.endswith(b"data: [DONE]\n\n")


def test_a_resume_read_with_the_headers_leaves_a_later_stall_a_stall():
    """Read once: the parser feeding the read's tracker must not leave a resumed park parked."""
    port = _serve(_HEADERS + _FIRST + b": preempted\n\n: resumed\n\n", _LAST, 1.0)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(_relay(port, stall_s = 0.3))


def _relay_sync(port: int, stall_s: float) -> str:
    from core.inference.llama_cpp import LlamaCppBackend

    with httpx.Client(
        timeout = httpx.Timeout(10.0), limits = httpx.Limits(max_keepalive_connections = 0)
    ) as client:
        with client.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions") as response:
            LlamaCppBackend._install_cancel_aware_read(
                client, threading.Event(), response, stall_grace = lambda: False
            )
            return "".join(
                LlamaCppBackend._iter_text_cancellable(
                    response,
                    stall_timeout_s = stall_s,
                    post_first_chunk_read_timeout_s = stall_s,
                    first_token_deadline = time.monotonic() + 10.0,
                )
            )


def test_the_synchronous_read_waits_out_a_park_read_with_the_headers():
    port = _serve(_HEADERS + _FIRST + b": preempted\n\n", b": resumed\n\n" + _LAST, 1.0)
    assert _relay_sync(port, stall_s = 0.3).endswith("data: [DONE]\n\n")


def test_the_synchronous_read_still_times_out_a_resumed_park():
    port = _serve(_HEADERS + _FIRST + b": preempted\n\n: resumed\n\n", _LAST, 1.0)
    with pytest.raises(httpx.ReadTimeout):
        _relay_sync(port, stall_s = 0.3)
