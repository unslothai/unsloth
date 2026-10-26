# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A direct request that gives up on a silent server must hang up, not only stop waiting.

`_post_through_park` runs the POST on a worker thread and returns to its caller at the deadline or
on cancel, and the caller then releases the slot and cache charge it held. `client.close()` does
not wake a recv() another thread is blocked in, so the request stayed open upstream, still
generating into the cache that had just been handed back. Checked against a real loopback socket."""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

import core.inference.llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend, _post_through_park


class _SilentServer:
    """Accepts one request, reads it, never answers; records when the client hangs up."""

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        self.received = threading.Event()
        self.hung_up = threading.Event()
        self._conn = None
        threading.Thread(target = self._serve, daemon = True).start()

    def _serve(self):
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        self._conn = conn
        conn.settimeout(30)
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                self.received.set()
        except OSError:
            pass
        self.hung_up.set()

    def close(self):
        for s in (self._conn, self._sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


@pytest.fixture
def server():
    srv = _SilentServer()
    yield srv
    srv.close()


def _post_workers():
    return [t for t in threading.enumerate() if t.name == "park-aware-post" and t.is_alive()]


def _wait_for(predicate, timeout = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_a_timed_out_direct_post_disconnects(server):
    before = set(_post_workers())
    with httpx.Client(trust_env = False) as client:
        with pytest.raises(httpx.ReadTimeout):
            _post_through_park(
                client,
                f"{server.url}/v1/chat/completions",
                json = {"messages": []},
                window = 0.3,
                stall_grace = lambda: False,
            )
    assert server.received.is_set()
    assert server.hung_up.wait(3.0), "the request stayed open after its caller gave up"
    assert _wait_for(lambda: not (set(_post_workers()) - before)), "the POST worker still reads"


def test_a_cancelled_direct_post_disconnects(server):
    before = set(_post_workers())
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with httpx.Client(trust_env = False) as client:
        with pytest.raises(httpx.ReadError):
            _post_through_park(
                client,
                f"{server.url}/completion",
                json = {"prompt": "x"},
                window = 60.0,
                stall_grace = lambda: True,
                cancel_event = cancel,
            )
    assert server.hung_up.wait(3.0), "the cancelled request stayed open upstream"
    assert _wait_for(lambda: not (set(_post_workers()) - before)), "the POST worker still reads"


def test_a_stopped_gguf_tts_render_disconnects(server, monkeypatch):
    """TTS closes the client from its own cancel watcher too; that must not empty the pool the
    helper's hang-up reads, or Stop releases the slot while the render goes on."""
    monkeypatch.setattr(llama_mod, "_local_ssl_context", lambda: None)
    fake = SimpleNamespace(
        _TTS_PROMPTS = LlamaCppBackend._TTS_PROMPTS,
        _auth_headers = None,
        server_may_park = True,
        _server_park_grace_any = lambda: True,
        base_url = server.url,
        _shutdown_active_httpx_sockets = LlamaCppBackend._shutdown_active_httpx_sockets,
    )
    before = set(_post_workers())
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(RuntimeError, match = "cancelled"):
        LlamaCppBackend.generate_audio_response(
            fake, "hello", "snac", max_new_tokens = 16, cancel_event = cancel
        )
    assert server.hung_up.wait(3.0), "Stop returned while the render stayed open upstream"
    assert _wait_for(lambda: not (set(_post_workers()) - before)), "the POST worker still reads"
