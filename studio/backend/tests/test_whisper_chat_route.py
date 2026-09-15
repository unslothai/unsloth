# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resident Whisper rejects text turns and transcribes chat audio uploads."""

import base64
import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.inference as inference_route
from auth.authentication import get_current_subject
from core.inference.api_monitor import ApiMonitor
from utils.api_errors import install_api_error_handlers


@pytest.fixture
def whisper_chat(monkeypatch):
    calls = []

    class ResidentWhisper:
        active_model_name = "unsloth/whisper-large-v3"
        models = {
            active_model_name: {
                "is_audio": True,
                "audio_type": "whisper",
                "has_audio_input": True,
            }
        }

        def generate_whisper_response(self, **kwargs):
            calls.append(kwargs)
            yield "The quick brown fox"
            yield " jumps over the lazy dog."

    backend = ResidentWhisper()
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )

    # Keep this a route test: decoding and model inference have their own tests.
    def decode_audio(encoded):
        with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as recording:
            assert recording.getframerate() == 16000
            return np.zeros(recording.getnframes(), dtype = np.float32)

    monkeypatch.setattr(inference_route, "_decode_audio_base64", decode_audio)
    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    with TestClient(app) as client:
        yield client, backend, calls, monitor


@pytest.mark.parametrize("stream", [False, True])
def test_resident_whisper_text_turn_requires_audio(whisper_chat, stream):
    client, backend, calls, monitor = whisper_chat
    response = client.post(
        "/v1/chat/completions",
        json = {
            "model": backend.active_model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Whisper models require audio input. Please upload an audio file."
    )
    assert calls == []
    assert backend.active_model_name == "unsloth/whisper-large-v3"
    assert monitor.active_count() == 0


@pytest.mark.parametrize("stream", [False, True])
def test_resident_whisper_chat_audio_returns_transcript(whisper_chat, stream):
    client, backend, calls, monitor = whisper_chat
    wav = io.BytesIO()
    with wave.open(wav, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\x00\x00" * 1600)
    response = client.post(
        "/v1/chat/completions",
        json = {
            "model": backend.active_model_name,
            "messages": [{"role": "user", "content": "Transcribe this"}],
            "audio_base64": base64.b64encode(wav.getvalue()).decode(),
            "stream": stream,
        },
    )
    assert response.status_code == 200
    if stream:
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        transcript = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    else:
        choice = response.json()["choices"][0]
        transcript = choice["message"]["content"]
        assert choice["finish_reason"] == "stop"
    assert transcript == "The quick brown fox jumps over the lazy dog."
    assert len(calls) == 1
    assert calls[0]["audio_array"].shape == (1600,)
    assert backend.active_model_name == "unsloth/whisper-large-v3"
    assert monitor.active_count() == 0
    [entry] = monitor.snapshot()
    assert entry["status"] == "completed"
    assert entry["reply"] == transcript
