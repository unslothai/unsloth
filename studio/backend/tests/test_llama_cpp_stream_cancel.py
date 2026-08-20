# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import contextlib
import os
import socket
import sys
import threading
import time

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_cpp import LlamaCppBackend, _LlamaStreamCancelled


def _backend_stub() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48848
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    return backend


def test_stream_cancel_uses_internal_exception_not_generator_exit():
    class FakeResponse:
        status_code = 200

        def close(self):
            pass

    class FakeStream:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

    cancel_event = threading.Event()

    with pytest.raises(Exception) as exc_info:
        with LlamaCppBackend._stream_with_retry(
            FakeClient(),
            "http://llama.test/v1/chat/completions",
            {},
            cancel_event,
        ):
            cancel_event.set()
            raise httpx.ReadError("client closed")

    assert exc_info.type is _LlamaStreamCancelled
    assert not issubclass(exc_info.type, GeneratorExit)


def test_generate_chat_completion_swallows_internal_stream_cancel(monkeypatch):
    backend = _backend_stub()

    @contextlib.contextmanager
    def fake_open_stream(*_args, **_kwargs):
        raise _LlamaStreamCancelled

    monkeypatch.setattr(backend, "_open_stream", fake_open_stream)

    chunks = list(
        backend.generate_chat_completion(
            [{"role": "user", "content": "hi"}],
            cancel_event = threading.Event(),
        )
    )

    assert chunks == []


class _StallUpstream:
    """Raw HTTP/1.1 server that streams one chunked SSE chunk, then holds the
    socket open and silent so the client's next read blocks in recv() until its
    side is torn down. Reproduces a mid-stream stall (llama-server goes quiet)."""

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._serve, daemon = True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
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
            chunk = b"data: hello\n\n"
            conn.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            # Stall: stay open and silent until the client shuts its side down.
            while not self._stop.wait(timeout = 0.05):
                try:
                    conn.settimeout(0.05)
                    if conn.recv(1) == b"":
                        return
                except socket.timeout:
                    continue
                except OSError:
                    return


def test_cancel_interrupts_a_read_blocked_on_a_mid_stream_stall():
    # Mid-stream stall: the reader is parked in recv() on a long bound read timeout,
    # so response.close() alone can't wake it; the watcher must shut the socket down.
    # Assert cancel lands in seconds, not at the far-off deadline (pre-fix: hung ~30s).
    with _StallUpstream() as server:
        cancel_event = threading.Event()

        def _cancel_soon():
            time.sleep(0.3)
            cancel_event.set()

        threading.Thread(target = _cancel_soon, daemon = True).start()

        started = time.monotonic()
        with httpx.Client(
            limits = httpx.Limits(max_keepalive_connections = 0), trust_env = False
        ) as client:
            with pytest.raises(_LlamaStreamCancelled):
                with LlamaCppBackend._stream_with_retry(
                    client,
                    server.url,
                    {},
                    cancel_event,
                    first_token_deadline = started + 30,
                ) as response:
                    for _chunk in response.iter_text():
                        pass  # first chunk arrives, then the read blocks silently
        elapsed = time.monotonic() - started

    assert elapsed < 10, f"cancel took {elapsed:.1f}s; the blocked read was not interrupted"
