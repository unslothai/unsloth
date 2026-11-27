# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The raw transcription route's opt-in ``timestamps`` reaches only whisper.cpp."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import inference as ri

_SEGMENTS = [{"start": 0.0, "end": 1.5, "text": "Hello there."}]


@pytest.fixture
def transcriptions(monkeypatch):
    calls = []

    def transcribe(*args, **kwargs):
        calls.append(kwargs)
        result = {"text": "Hello there.", "model": "small", "language": "en", "duration": 1.5}
        if kwargs.get("timestamps"):
            result["segments"] = _SEGMENTS
        return result

    sidecar = SimpleNamespace(loaded_model = None, device = None, transcribe = transcribe)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: engine)
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *args: None)
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (lambda *a, **k: None, lambda *a: []))
    return calls


@pytest.fixture
def client():
    app = FastAPI()
    app.add_api_route("/raw", ri.transcribe_audio_raw, methods = ["POST"])
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_timestamps_reach_whisper_cpp_and_return_segments(client, transcriptions):
    response = client.post(
        "/raw", params = {"engine": "gguf", "model": "small", "timestamps": "true"}, content = b"a"
    )
    assert response.status_code == 200, response.text
    assert transcriptions[0]["timestamps"] is True
    assert response.json()["segments"] == _SEGMENTS


@pytest.mark.parametrize("engine", ["gguf", "mtmd", "transformers", "audiocpp"])
def test_default_request_is_unchanged(client, transcriptions, engine):
    response = client.post("/raw", params = {"engine": engine, "model": "small"}, content = b"a")
    assert response.status_code == 200, response.text
    assert "timestamps" not in transcriptions[0]
    assert response.json() == {
        "text": "Hello there.",
        "model": "small",
        "language": "en",
        "duration": 1.5,
    }


@pytest.mark.parametrize("engine", ["mtmd", "transformers", "audiocpp"])
def test_other_engines_ignore_timestamps(client, transcriptions, engine):
    response = client.post(
        "/raw", params = {"engine": engine, "model": "small", "timestamps": "true"}, content = b"a"
    )
    assert response.status_code == 200, response.text
    assert "timestamps" not in transcriptions[0]
    assert "segments" not in response.json()
