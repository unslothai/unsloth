# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parking-capable raw stream (`/v1/responses`, `/v1/messages`) waits out a park below the httpx
iterators. Its first read used to be armed with the longer of the first-token and stall bounds, and
the relay then added the park cap on top, so a request that was never parked waited for the stall
bound instead of its first-token deadline (30 s became 120 s). The first read now ends at the
first-token deadline unless the server really holds the slot parked."""

import asyncio
import socket
import threading
import time

import httpx
import pytest

import routes.inference as inference

_HEADERS = (
    b"HTTP/1.1 200 OK\r\n" b"Content-Type: text/event-stream\r\n" b"Connection: close\r\n\r\n"
)
_ANSWER = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n'
_FIRST_TOKEN_S = 0.15
_STALL_S = 1.5


def _serve(silence_s: float):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def run():
        conn, _ = listener.accept()
        try:
            conn.settimeout(10)
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(65536)
            conn.sendall(_HEADERS)
            time.sleep(silence_s)
            conn.sendall(_ANSWER)
        except OSError:
            pass
        finally:
            conn.close()
            listener.close()

    threading.Thread(target = run, daemon = True).start()
    return listener.getsockname()[1]


async def _relay(port: int, parked: bool) -> bytes:
    async with httpx.AsyncClient(timeout = httpx.Timeout(10.0)) as client:
        request = client.build_request("POST", f"http://127.0.0.1:{port}/v1/responses")
        response = await client.send(request, stream = True)
        try:
            seen = b""
            async for item in inference._aiter_llama_stream_items(
                response.aiter_bytes(),
                first_token_deadline = time.monotonic() + _FIRST_TOKEN_S,
                response = response,
                post_first_item_read_timeout_s = _STALL_S,
                # The aggregate probe of a build that sends no notices.
                stall_grace = lambda: parked,
            ):
                if isinstance(item, bytes):
                    seen += item
            return seen
        finally:
            await response.aclose()


def test_an_unparked_request_times_out_at_its_first_token_deadline():
    port = _serve(silence_s = 3.0)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(_relay(port, parked = False))
    waited = time.monotonic() - started
    assert waited < _FIRST_TOKEN_S + 0.5, (
        f"waited {waited:.2f}s against a {_FIRST_TOKEN_S}s first-token deadline: "
        f"the read was stretched to the {_STALL_S}s stall bound"
    )


def test_a_parked_request_still_waits_for_its_first_token():
    port = _serve(silence_s = 0.6)
    seen = asyncio.run(_relay(port, parked = True))
    assert seen.endswith(b"data: [DONE]\n\n")
