# SPDX-License-Identifier: AGPL-3.0-only
"""/audio/generate reports what a request spent before and during synthesis."""

import asyncio
import base64
import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.inference as inference_route
from auth.authentication import get_current_subject
from models.inference import ChatCompletionRequest
from routes.inference import router
from utils.api_errors import install_api_error_handlers

_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-payload"


def _fake_backend(monkeypatch, *, resolve_s = 0.0, synth_s = 0.0):
    class _Llama:
        is_loaded = False
        _is_audio = False

    class _Backend:
        active_model_name = "some/custom-tts"
        models = {"some/custom-tts": {"is_audio": True, "audio_type": "snac"}}

        def generate_audio_response(self, **_kwargs):
            time.sleep(synth_s)
            return _WAV, 24000

    async def _slow_switch(*_args, **_kwargs):
        await asyncio.sleep(resolve_s)

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _slow_switch)
    monkeypatch.setattr(inference_route, "_persist_tts_clip", lambda *_args: None)


def _generate(payload):
    response = asyncio.run(
        inference_route.generate_audio(payload, request = None, current_subject = "t")
    )
    return json.loads(response.body)


def _payload():
    return ChatCompletionRequest(
        model = "some/custom-tts",
        messages = [{"role": "user", "content": "hello"}],
    )


def test_generate_reports_resolution_and_synthesis_time(monkeypatch):
    _fake_backend(monkeypatch, resolve_s = 0.03, synth_s = 0.03)

    body = _generate(_payload())

    assert set(body["timing"]) == {"resolve_ms", "synth_ms"}
    # Each stage is timed on its own: the slow switch counts as resolution, the model as synthesis.
    assert body["timing"]["resolve_ms"] >= 25
    assert body["timing"]["synth_ms"] >= 25


def test_timing_leaves_the_audio_as_it_was(monkeypatch):
    _fake_backend(monkeypatch)

    body = _generate(_payload())

    assert body["audio"] == {
        "data": base64.b64encode(_WAV).decode("ascii"),
        "format": "wav",
        "sample_rate": 24000,
    }
    assert body["choices"][0]["message"]["content"] == "hello"


def test_the_speech_route_still_returns_the_raw_wav(monkeypatch):
    _fake_backend(monkeypatch)
    monkeypatch.setattr(inference_route, "_persist_tts_clip", lambda *_args, **_kwargs: None)
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    resp = TestClient(app).post(
        "/v1/audio/speech", json = {"input": "hello", "model": "some/custom-tts"}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content == _WAV
