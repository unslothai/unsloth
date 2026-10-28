# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice context preferences are validated and reach only the llama.cpp ASR sidecar."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auth.authentication import get_current_subject
from models.inference import (
    AudioInputTranscribeRequest,
    SttLoadRequest,
    TranscribeRequest,
    TranscribeSourceRequest,
)
from routes import inference as ri


@pytest.mark.parametrize(
    "model, fields",
    [
        (SttLoadRequest, {}),
        (TranscribeRequest, {"audio": "YQ=="}),
        (AudioInputTranscribeRequest, {"model": "qwen3-asr-0.6b"}),
        (TranscribeSourceRequest, {"model": "qwen3-asr-0.6b", "source": {"input_id": "a" * 32}}),
    ],
)
def test_context_settings_are_strict_and_default_to_auto(model, fields):
    assert model(**fields).context_length == 0
    for value in (0, 1024, 4096, 65536):
        assert model(**fields, context_length = value).context_length == value
    for value in (True, False, None, "4096", 4096.0, 4096.5, -1, 1, 1023, 65537):
        with pytest.raises(ValidationError, match = "Voice context length"):
            model(**fields, context_length = value)


@pytest.fixture
def sidecar_calls(monkeypatch):
    loads, transcriptions = [], []

    def transcribe(*args, **kwargs):
        transcriptions.append((args, kwargs))
        return {"text": "hello", "model": "qwen3-asr-0.6b"}

    sidecar = SimpleNamespace(loaded_model = None, device = None, transcribe = transcribe)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: engine or "mtmd")
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *args: None)
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(
        ri,
        "_stt_lifecycle",
        lambda: (lambda *args, **kwargs: loads.append((args, kwargs)), lambda *args: []),
    )
    return loads, transcriptions


@pytest.mark.parametrize("engine", ["mtmd", "gguf", "transformers", "audiocpp"])
@pytest.mark.parametrize("streaming", [False, True])
def test_only_mtmd_receives_context_on_implicit_load_and_transcription(
    sidecar_calls, engine, streaming
):
    loads, transcriptions = sidecar_calls
    result = asyncio.run(
        ri._transcribe_audio_result(
            b"audio",
            "qwen3-asr-0.6b",
            None,
            False,
            engine,
            on_progress = (lambda event: None) if streaming else None,
            context_length = 4096,
        )
    )
    assert result["text"] == "hello"
    for kwargs in (loads[0][1], transcriptions[0][1]):
        if engine == "mtmd":
            assert kwargs["context_length"] == 4096
        else:
            assert "context_length" not in kwargs


def test_an_omitted_api_preference_requests_auto(sidecar_calls):
    loads, transcriptions = sidecar_calls
    asyncio.run(ri._transcribe_audio_result(b"audio", "qwen3-asr-0.6b", None, False, "mtmd"))
    assert loads[0][1]["context_length"] == 0
    assert transcriptions[0][1]["context_length"] == 0


@pytest.mark.parametrize("error_name", ["SttContextSizeError", "SttIncompleteTranscriptionError"])
def test_context_failures_are_actionable_422_errors(sidecar_calls, monkeypatch, error_name):
    from core.inference import stt_sidecar

    def fail(*args, **kwargs):
        raise getattr(stt_sidecar, error_name)(
            "Choose a larger voice context or a shorter recording."
        )

    monkeypatch.setattr(ri._stt_sidecar_for("mtmd"), "transcribe", fail)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ri._transcribe_audio_result(b"audio", "qwen3-asr-0.6b", None, False, "mtmd"))
    assert exc.value.status_code == 422
    assert "larger voice context" in exc.value.detail


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.add_api_route("/load", ri.stt_load, methods = ["POST"])
    app.add_api_route("/transcribe", ri.transcribe_audio, methods = ["POST"])
    app.add_api_route("/raw", ri.transcribe_audio_raw, methods = ["POST"])
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.mark.parametrize("engine", ["mtmd", "gguf"])
def test_explicit_load_forwards_context_only_to_mtmd(client, sidecar_calls, engine):
    response = client.post(
        "/load",
        json = {
            "model": "qwen3-asr-0.6b",
            "engine": engine,
            "context_length": 8192,
        },
    )
    assert response.status_code == 200, response.text
    kwargs = sidecar_calls[0][0][1]
    assert kwargs.get("context_length") == (8192 if engine == "mtmd" else None)


def test_load_context_failure_is_422(client, sidecar_calls, monkeypatch):
    from core.inference.stt_sidecar import SttContextSizeError

    def fail(*args, **kwargs):
        raise SttContextSizeError("The requested context is unavailable.")

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (fail, lambda *args: []))
    response = client.post("/load", json = {"model": "qwen3-asr-0.6b", "engine": "mtmd"})
    assert response.status_code == 422
    assert response.json()["detail"] == "The requested context is unavailable."


@pytest.mark.parametrize(
    "value", ["true", "false", "4096.0", "4096.5", "1e4", "-1", "1023", "65537"]
)
def test_raw_query_rejects_invalid_context_before_transcribing(client, monkeypatch, value):
    async def unexpected(*args, **kwargs):
        pytest.fail("invalid context reached transcription")

    monkeypatch.setattr(ri, "_transcribe_audio_result", unexpected)
    response = client.post("/raw", params = {"context_length": value}, content = b"audio")
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("endpoint", ["json", "raw", "stream"])
def test_request_preference_reaches_transcription(client, monkeypatch, endpoint):
    calls = []

    async def transcribe(*args, **kwargs):
        calls.append(kwargs)
        return {"text": "hello", "model": "qwen3-asr-0.6b", "duration": 1.0}

    monkeypatch.setattr(ri, "_transcribe_audio_result", transcribe)
    if endpoint == "json":
        response = client.post("/transcribe", json = {"audio": "YQ==", "context_length": 8192})
    else:
        response = client.post(
            "/raw",
            params = {
                "context_length": "8192",
                "stream": endpoint == "stream",
            },
            content = b"audio",
        )
    assert response.status_code == 200, response.text
    assert calls[0]["context_length"] == 8192


@pytest.mark.parametrize("error_name", ["SttContextSizeError", "SttIncompleteTranscriptionError"])
def test_streamed_context_failures_preserve_detail_without_completing(
    client, sidecar_calls, monkeypatch, error_name
):
    from core.inference import stt_sidecar

    message = "Choose a larger voice context or a shorter recording."

    def fail(*args, **kwargs):
        raise getattr(stt_sidecar, error_name)(message)

    monkeypatch.setattr(ri._stt_sidecar_for("mtmd"), "transcribe", fail)
    response = client.post("/raw", params = {"engine": "mtmd", "stream": True}, content = b"audio")
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "error", "message": message}
    assert not any(event["type"] == "complete" for event in events)
