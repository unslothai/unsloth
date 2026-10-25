# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parking build that predates the stream notices is silent for the whole park, headers
included, when the slot is parked during prompt processing. The park-aware read is installed
only once the headers are in, so the header wait itself has to ask the `/metrics` probe at the
first-token deadline, or a healthy parked request ends in a ReadTimeout. An unparked request
still ends at its first-token deadline, and Stop still ends a wait the probe excuses."""

import asyncio
import contextlib
import socket
import sys
import threading
import time

import httpx
import pytest

import routes.inference as inference
from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend, _LlamaStreamCancelled

_HEADERS = (
    b"HTTP/1.1 200 OK\r\n" b"Content-Type: text/event-stream\r\n" b"Connection: close\r\n\r\n"
)
_ANSWER = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n'
_FIRST_TOKEN_S = 0.1
_HEADERS_AFTER_S = 0.3

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "a Winsock recv before headers cannot be woken, so the sync header wait keeps its bound",
)


def _serve(headers_after_s: float):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def run():
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        try:
            conn.settimeout(10)
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(65536)
            head, _, body = request.partition(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            # Read the whole request: closing with bytes unread resets the connection.
            while len(body) < length:
                body += conn.recv(65536)
            time.sleep(headers_after_s)
            conn.sendall(_HEADERS + _ANSWER)
        except OSError:
            pass
        finally:
            conn.close()
            listener.close()

    threading.Thread(target = run, daemon = True).start()
    return listener.getsockname()[1]


class _Probe:
    def __init__(self, parked: bool):
        self.parked = parked
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.parked


# ── async: the raw relays (/v1/responses, /v1/messages, /v1/chat/completions passthrough) ──


async def _relay(
    port: int,
    probe: _Probe,
    cancel_after_s = None,
):
    cancel = threading.Event()
    if cancel_after_s is not None:
        threading.Timer(cancel_after_s, cancel.set).start()
    async with httpx.AsyncClient(
        timeout = httpx.Timeout(10.0, read = _FIRST_TOKEN_S), trust_env = False
    ) as client:
        req = client.build_request("POST", f"http://127.0.0.1:{port}/v1/responses")
        first_token_deadline = time.monotonic() + _FIRST_TOKEN_S
        resp = await inference._send_stream_with_preheader_cancel(
            client, req, cancel, stall_grace = probe
        )
        if resp is None:
            return None
        try:
            seen = b""
            async for item in inference._aiter_llama_stream_items(
                resp.aiter_bytes(),
                first_token_deadline = first_token_deadline,
                response = resp,
                stall_grace = probe,
            ):
                if isinstance(item, bytes):
                    seen += item
            return seen
        finally:
            await resp.aclose()


def test_async_a_park_during_the_header_wait_is_waited_out():
    probe = _Probe(parked = True)
    seen = asyncio.run(_relay(_serve(_HEADERS_AFTER_S), probe))
    assert probe.calls > 0, "the header wait never asked the park probe"
    assert seen.endswith(b"data: [DONE]\n\n")


def test_async_an_unparked_header_wait_keeps_its_first_token_deadline():
    probe = _Probe(parked = False)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(_relay(_serve(3.0), probe))
    waited = time.monotonic() - started
    assert waited < _FIRST_TOKEN_S + 0.5, f"waited {waited:.2f}s past a {_FIRST_TOKEN_S}s deadline"


def test_async_stop_ends_an_excused_header_wait():
    probe = _Probe(parked = True)
    started = time.monotonic()
    assert asyncio.run(_relay(_serve(3.0), probe, cancel_after_s = 0.25)) is None
    assert time.monotonic() - started < 1.5


# ── sync: Studio's own chat streams (`_open_stream` -> `_stream_with_retry`) ──


class _Backend:
    """Just what `_open_stream` reads off a backend."""

    _auth_headers: dict = {}
    server_may_park = True
    _stream_with_retry = staticmethod(LlamaCppBackend._stream_with_retry)

    def __init__(self, probe: _Probe):
        self._server_park_grace = probe


def _relay_sync(
    port: int,
    probe: _Probe,
    cancel_after_s = None,
) -> str:
    cancel = threading.Event()
    if cancel_after_s is not None:
        threading.Timer(cancel_after_s, cancel.set).start()
    opened = LlamaCppBackend._open_stream.__wrapped__(
        _Backend(probe), f"http://127.0.0.1:{port}/v1/chat/completions", {}, cancel
    )
    with contextlib.contextmanager(lambda: opened)() as (response, first_token_deadline):
        return "".join(
            LlamaCppBackend._iter_text_cancellable(
                response, cancel, first_token_deadline = first_token_deadline
            )
        )


@pytest.fixture
def short_first_token(monkeypatch):
    real = llama_cpp.start_first_token_wait
    monkeypatch.setattr(
        llama_cpp,
        "start_first_token_wait",
        lambda seconds = None: real(_FIRST_TOKEN_S if seconds is None else seconds),
    )


@posix_only
def test_sync_a_park_during_the_header_wait_is_waited_out(short_first_token):
    probe = _Probe(parked = True)
    seen = _relay_sync(_serve(_HEADERS_AFTER_S), probe)
    assert probe.calls > 0, "the header wait never asked the park probe"
    assert seen.endswith("data: [DONE]\n\n")


def test_sync_an_unparked_header_wait_keeps_its_first_token_deadline(short_first_token):
    probe = _Probe(parked = False)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        _relay_sync(_serve(3.0), probe)
    waited = time.monotonic() - started
    assert waited < _FIRST_TOKEN_S + 0.5, f"waited {waited:.2f}s past a {_FIRST_TOKEN_S}s deadline"


@posix_only
def test_sync_stop_ends_an_excused_header_wait(short_first_token):
    probe = _Probe(parked = True)
    started = time.monotonic()
    with pytest.raises(_LlamaStreamCancelled):
        _relay_sync(_serve(3.0), probe, cancel_after_s = 0.25)
    assert time.monotonic() - started < 1.5
