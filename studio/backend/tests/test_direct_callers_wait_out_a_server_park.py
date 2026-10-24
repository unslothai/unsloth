# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The non-streaming calls that reach llama-server directly, a caption or OCR transcription and a
GGUF TTS render, wait out a server park longer than their read timeout, as the chat surfaces do:
the server parks the smallest slot first, which is usually theirs. Without a park, or on a server
that does not park, their own timeout still holds."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
import time

import pytest

from core.inference import llama_cpp as llama_mod
from core.rag import captioner

_CHAT_BODY = {
    "id": "chatcmpl-caption",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "a cat"}, "finish_reason": "stop"}],
}


@contextlib.contextmanager
def _upstream(park_s: float, body: dict):
    """A fake llama-server that sits silent for ``park_s`` on every POST, then answers."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            time.sleep(park_s)
            payload = json.dumps(body).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                pass

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class _ChatBackend:
    def __init__(self, port: int, *, server: bool, parked: bool):
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_may_park = server
        self._parked = parked
        self.probes = 0

    def _server_park_grace_any(self) -> bool:
        self.probes += 1
        return self._parked


def _caption(monkeypatch, *, server: bool, parked: bool, park_s: float):
    import routes.inference as inf_mod

    monkeypatch.setattr(captioner, "_direct_llama_slot", lambda **_k: contextlib.nullcontext())
    with _upstream(park_s, _CHAT_BODY) as port:
        backend = _ChatBackend(port, server = server, parked = parked)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        started = time.monotonic()
        text = captioner._vision_complete(
            backend.base_url, "m", b"png", prompt = "caption", timeout = 1.0, max_tokens = 8
        )
        return text, backend, time.monotonic() - started


@pytest.fixture
def short_cap(monkeypatch):
    monkeypatch.setattr(llama_mod, "_SERVER_PARK_STALL_CAP_S", 1.0)


def test_a_caption_waits_out_a_park_longer_than_its_timeout(monkeypatch):
    text, backend, elapsed = _caption(monkeypatch, server = True, parked = True, park_s = 2.5)
    assert text == "a cat"
    assert backend.probes >= 2
    assert elapsed >= 2.0


def test_a_caption_without_a_park_still_times_out(monkeypatch):
    text, backend, elapsed = _caption(monkeypatch, server = True, parked = False, park_s = 6.0)
    assert text is None
    assert backend.probes == 1
    assert elapsed < 3.0


def test_a_caption_on_a_server_that_does_not_park_is_unchanged(monkeypatch):
    text, backend, elapsed = _caption(monkeypatch, server = False, parked = True, park_s = 6.0)
    assert text is None
    assert backend.probes == 0
    assert elapsed < 3.0


def test_the_caption_wait_is_bounded_by_the_park_cap(monkeypatch, short_cap):
    text, _backend, elapsed = _caption(monkeypatch, server = True, parked = True, park_s = 8.0)
    assert text is None
    assert 1.9 <= elapsed < 5.0


class _Codec:
    def __init__(self):
        self.decoded = []

    def decode(self, audio_type, device, *, token_ids = None, text = ""):
        self.decoded.append(text)
        return b"RIFF", 24000


def _tts(monkeypatch, *, server: bool, parked: bool, park_s: float, cancel = None):
    codec = _Codec()
    monkeypatch.setattr(llama_mod.LlamaCppBackend, "_codec_mgr", codec)
    monkeypatch.setattr(llama_mod, "_GGUF_AUDIO_READ_TIMEOUT", 1.0)
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend, "server_may_park", property(lambda self: server)
    )
    with _upstream(park_s, {"content": "<audio>", "stop_type": "eos"}) as port:
        backend = llama_mod.LlamaCppBackend.__new__(llama_mod.LlamaCppBackend)
        backend._port = port
        backend._api_key = None
        probes = []

        def grace():
            probes.append(1)
            return parked

        backend._server_park_grace_any = grace
        started = time.monotonic()
        try:
            result = backend.generate_audio_response(
                "hello", "bicodec", max_new_tokens = 8, cancel_event = cancel
            )
        except Exception as exc:  # noqa: BLE001 - the outcome under test
            result = exc
        return result, codec, len(probes), time.monotonic() - started


def test_a_tts_render_waits_out_a_park_longer_than_its_timeout(monkeypatch):
    result, codec, probes, elapsed = _tts(monkeypatch, server = True, parked = True, park_s = 2.5)
    assert result == (b"RIFF", 24000), result
    assert codec.decoded == ["<audio>"]
    assert probes >= 2
    assert elapsed >= 2.0


def test_a_tts_render_without_a_park_still_times_out(monkeypatch):
    import httpx

    result, codec, probes, elapsed = _tts(monkeypatch, server = True, parked = False, park_s = 6.0)
    assert isinstance(result, httpx.TimeoutException), result
    assert codec.decoded == []
    assert probes == 1
    assert elapsed < 3.0


def test_a_tts_render_on_a_server_that_does_not_park_is_unchanged(monkeypatch):
    import httpx

    result, _codec, probes, elapsed = _tts(monkeypatch, server = False, parked = True, park_s = 6.0)
    assert isinstance(result, httpx.TimeoutException), result
    assert probes == 0
    assert elapsed < 3.0


def test_a_parked_tts_render_still_stops_on_cancel(monkeypatch):
    cancel = threading.Event()
    threading.Timer(1.5, cancel.set).start()
    result, codec, _probes, elapsed = _tts(
        monkeypatch, server = True, parked = True, park_s = 6.0, cancel = cancel
    )
    assert isinstance(result, RuntimeError) and "cancelled" in str(result), result
    assert codec.decoded == []
    assert elapsed < 3.0
