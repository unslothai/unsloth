# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from core.inference import external_provider as provider_module
from core.inference.external_provider import ExternalProviderClient


_PCM = b"\x01\x00\xff\x7f\x00\x80"
_MODEL = "microsoft/mai-voice-2.1-flash"
_VOICE = "de-DE-Mia:MAI-Voice-2.1-Flash"


def _speak(
    monkeypatch,
    provider,
    model,
    response_format,
    accepted,
    payload,
    content_type,
    repeat = 1,
):
    sent = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            sent.append(body)
            allowed = body["response_format"] == accepted
            self.send_response(200 if allowed else 400)
            if content_type is not None:
                self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(payload if allowed else b"response_format: expected mp3 or pcm")

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target = server.serve_forever, kwargs = {"poll_interval": 0.01})
    worker.start()

    async def run():
        async with httpx.AsyncClient(trust_env = False) as transport:
            monkeypatch.setattr(provider_module, "_http_client", transport)
            client = ExternalProviderClient(
                provider, f"http://127.0.0.1:{server.server_port}/v1", "test"
            )
            results = [
                await client.create_speech(
                    text = "Hallo",
                    model = model,
                    voice = _VOICE,
                    response_format = response_format,
                )
                for _ in range(repeat)
            ]
            assert all(result == results[0] for result in results)
            return results[0]

    try:
        result = asyncio.run(run())
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    assert len(sent) == repeat
    assert sent[0]["voice"] == _VOICE
    assert sent[0]["input"] == "Hallo"
    return result


@pytest.mark.parametrize("content_type", ["audio/pcm", "Audio/PCM", "audio/pcm; rate=24000", None])
def test_openrouter_mai_returns_playable_wav(monkeypatch, content_type):
    audio, media_type = _speak(
        monkeypatch, "openrouter", _MODEL, "wav", "pcm", _PCM, content_type, repeat = 2
    )
    assert media_type == "audio/wav"
    with wave.open(io.BytesIO(audio), "rb") as reader:
        assert reader.getframerate() == 24_000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.readframes(reader.getnframes()) == _PCM


@pytest.mark.parametrize(
    "provider, model, fmt, content_type",
    [
        ("custom", _MODEL, "wav", "audio/wav"),
        ("openai", "gpt-4o-mini-tts", "wav", "audio/wav"),
        ("openrouter", "openai/gpt-4o-mini-tts", "wav", "audio/wav"),
        ("openrouter", _MODEL, "mp3", "audio/mpeg"),
        ("openrouter", _MODEL, "pcm", "audio/pcm"),
    ],
)
def test_other_requests_keep_format_and_bytes(monkeypatch, provider, model, fmt, content_type):
    audio, media_type = _speak(monkeypatch, provider, model, fmt, fmt, _PCM, content_type)
    assert audio == _PCM
    assert media_type == content_type


@pytest.mark.parametrize(
    "payload, content_type",
    [(b"", "audio/pcm"), (b"\x01", "audio/pcm"), (b"ID3mp3", "audio/mpeg")],
)
def test_malformed_pcm_is_a_provider_http_error(monkeypatch, payload, content_type):
    with pytest.raises(httpx.HTTPError):
        _speak(monkeypatch, "openrouter", _MODEL, "wav", "pcm", payload, content_type)
