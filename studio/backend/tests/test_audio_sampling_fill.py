# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio (TTS) generation applies recommended sampling + operator pins, like chat.

Regression guard for the fix that moved the sampling fill ahead of the audio generators: a
prior version resolved sampling only after the audio branches returned, so `unsloth run
--temperature` (UNSLOTH_SAMPLING_*) and per-model recommendations never reached audio
generation. These exercise the transformers TTS path of ``generate_audio`` (the direct
``/audio/generate`` route, which the chat-completions audio branches also delegate to).
"""

import asyncio
import json

import pytest

import routes.inference as inference_route
from fastapi import HTTPException
from models.inference import AudioSpeechRequest, ChatCompletionRequest
from starlette.requests import Request
from utils.inference import inference_config as ic


def _request(path = "/v1/audio/speech"):
    """/v1/audio/speech opens an API monitor row, so it needs a real request."""
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
        }
    )


class _FakeLlama:
    # is_loaded False forces the transformers (non-GGUF) TTS branch in generate_audio.
    is_loaded = False
    _is_audio = False


class _FakeTransformersBackend:
    def __init__(self, audio_type = "snac"):
        self.active_model_name = "some/custom-tts"
        self.models = {"some/custom-tts": {"is_audio": True, "audio_type": audio_type}}
        self.captured = {}

    def generate_audio_response(self, **kwargs):
        self.captured.update(kwargs)
        return (b"RIFFfake", 24000)


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    ic._recommended_sampling.cache_clear()
    for field in ic.SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(ic._SAMPLING_FIELDS[field][0], raising = False)
    yield
    ic._recommended_sampling.cache_clear()


def _run_generate_audio(
    monkeypatch,
    *,
    recommended = None,
    temperature = None,
):
    backend = _FakeTransformersBackend()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)

    async def _noop_switch(*a, **k):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)

    # Recommendation source == the Chat UI's .inference block.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(recommended or {}))
    ic._recommended_sampling.cache_clear()

    kwargs = {"model": "some/custom-tts", "messages": [{"role": "user", "content": "hi"}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    payload = ChatCompletionRequest(**kwargs)

    asyncio.run(inference_route.generate_audio(payload, request = None, current_subject = "t"))
    return backend.captured


def test_audio_uses_recommended_sampling_when_omitted(monkeypatch):
    captured = _run_generate_audio(monkeypatch, recommended = {"temperature": 1.0, "top_k": 64})
    assert captured["temperature"] == 1.0
    assert captured["top_k"] == 64


@pytest.mark.parametrize(
    "model_id",
    (
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "OpenMOSS-Team/MOSS-TTS-Nano-100M",
    ),
)
def test_moss_uses_the_published_audio_sampling_defaults(model_id):
    expected = {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    assert ic.get_family_inference_params(model_id) == expected
    resolved = ic.load_inference_config(model_id)
    assert {key: resolved[key] for key in ("temperature", "top_p", "top_k", "min_p")} == {
        key: expected[key] for key in ("temperature", "top_p", "top_k", "min_p")
    }


def test_audio_operator_pin_overrides_client(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    captured = _run_generate_audio(monkeypatch, recommended = {"temperature": 1.0}, temperature = 0.2)
    assert captured["temperature"] == 0.9  # operator pin wins even over an explicit client value


def test_audio_client_explicit_preserved(monkeypatch):
    captured = _run_generate_audio(monkeypatch, recommended = {"temperature": 1.0}, temperature = 0.2)
    assert captured["temperature"] == 0.2  # explicit client value preserved over recommendation


def test_audio_generate_returns_the_exact_persisted_clip_id(monkeypatch):
    backend = _FakeTransformersBackend()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)

    async def _noop_switch(*a, **k):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = "some/custom-tts", messages = [{"role": "user", "content": "hi"}]
    )
    response = asyncio.run(
        inference_route.generate_audio(payload, request = None, current_subject = "t")
    )
    body = json.loads(response.body)
    assert body["clip_id"]
    assert len(body["clip_id"]) == 32


def test_whisper_is_rejected_cleanly_by_both_tts_endpoints(monkeypatch):
    backend = _FakeTransformersBackend(audio_type = "whisper")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)

    async def _noop_switch(*a, **k):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    payload = ChatCompletionRequest(
        model = "some/custom-tts", messages = [{"role": "user", "content": "hi"}]
    )
    speech = AudioSpeechRequest(input = "hi", model = "some/custom-tts")
    for request in (
        inference_route.generate_audio(payload, request = None, current_subject = "t"),
        inference_route.openai_audio_speech(speech, request = _request(), current_subject = "t"),
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(request)
        assert exc.value.status_code == 400
        assert "does not support text-to-speech" in exc.value.detail
    assert backend.captured == {}
