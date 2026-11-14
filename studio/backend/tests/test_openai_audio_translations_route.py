# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Round-trip tests for POST /v1/audio/translations with _transcribe_audio_result faked; no GPU."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.inference as routes_module
from core.inference.api_monitor import api_monitor
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers


def _make_client(monkeypatch, translate = None):
    calls = []

    async def _fake_transcribe(
        raw,
        model,
        language,
        fast,
        engine = None,
        request = None,
        **kwargs,
    ):
        calls.append({"raw": raw, "model": model, "language": language, "engine": engine, **kwargs})
        if translate is not None:
            return await translate(raw)
        return {"text": "hello sloth", "language": None, "duration": 1.5, "model": "small"}

    monkeypatch.setattr(routes_module, "_transcribe_audio_result", _fake_transcribe)

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), calls


def _post(
    cli,
    data = None,
    content = b"RIFFfake",
):
    return cli.post(
        "/v1/audio/translations",
        files = {"file": ("clip.wav", content, "audio/wav")},
        data = data or {},
    )


def test_json_response_is_text_only_and_asks_for_a_translation(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"model": "whisper-1", "prompt": "ignored", "temperature": "0.2"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello sloth"}
    assert calls[0]["raw"] == b"RIFFfake"
    assert calls[0]["model"] is None
    assert calls[0]["language"] is None
    assert calls[0]["translate"] is True


def test_text_response_is_plain_body(monkeypatch):
    cli, _ = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "text"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello sloth"


def test_verbose_json_validates_against_the_openai_client_model(monkeypatch):
    openai_types = pytest.importorskip("openai.types.audio.translation_verbose")
    cli, _ = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "verbose_json"})
    assert resp.status_code == 200
    assert resp.json() == {
        "task": "translate",
        "language": "english",
        "duration": 1.5,
        "text": "hello sloth",
    }
    openai_types.TranslationVerbose.model_validate(resp.json())


def test_explicit_whisper_model_passes_through(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    assert _post(cli, data = {"model": "large-v3"}).status_code == 200
    assert calls[0]["model"] == "large-v3"


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-asr-0.6b",
        "audio-cpp/audio.cpp-gguf/Qwen3-ASR-0.6B",
        # Fine-tuned for transcription only: it answers in the source language.
        "large-v3-turbo",
        "unsloth/whisper-large-v3-turbo",
    ],
)
def test_models_without_a_translate_task_are_refused_before_any_work(monkeypatch, model):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"model": model})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "model"
    assert calls == []


@pytest.mark.parametrize(
    ("data", "param"),
    [
        ({"response_format": "srt"}, "response_format"),
        ({"response_format": "diarized_json"}, "response_format"),
        ({"provider_id": "conn-1"}, "provider_id"),
    ],
)
def test_unsupported_fields_are_400(monkeypatch, data, param):
    cli, calls = _make_client(monkeypatch)
    api_monitor.clear()
    resp = _post(cli, data = data)
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == param
    assert calls == []
    assert api_monitor.snapshot(include_details = False) == []


def test_sidecar_errors_keep_their_status(monkeypatch):
    # An English-only checkpoint is the sidecar's SttLanguageError, mapped to 422 by the helper.
    async def _english_only(raw):
        raise HTTPException(status_code = 422, detail = "English-only STT model 'x' cannot translate.")

    cli, _ = _make_client(monkeypatch, translate = _english_only)
    resp = _post(cli, data = {"model": "owner/whisper-small.en"})
    assert resp.status_code == 422
    assert "cannot translate" in resp.json()["error"]["message"]


def test_translation_opens_a_monitor_row(monkeypatch):
    cli, _ = _make_client(monkeypatch)
    api_monitor.clear()
    assert _post(cli).status_code == 200
    rows = api_monitor.snapshot(include_details = False)
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/audio/translations"
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_preview"] == "clip.wav"
    assert rows[0]["reply_preview"] == "hello sloth"
    assert rows[0]["model"] == "small"


def test_the_helper_hands_whisper_the_translate_task(monkeypatch):
    """The real _transcribe_audio_result, with only the sidecar faked."""
    seen = {}

    def _transcribe(
        raw,
        model,
        language,
        fast,
        cancel_event = None,
        **kwargs,
    ):
        seen.update(kwargs, language = language)
        return {"text": "hi", "model": "small"}

    sidecar = SimpleNamespace(transcribe = _transcribe, loaded_model = None)
    monkeypatch.setattr(routes_module, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(routes_module, "_prepare_runtime_fallback_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        routes_module, "_stt_lifecycle", lambda: (lambda *a, **k: None, lambda *a, **k: [])
    )

    result = asyncio.run(
        routes_module._transcribe_audio_result(b"audio", None, None, False, translate = True)
    )
    assert result["text"] == "hi"
    assert seen == {"task": "translate", "language": None}

    seen.clear()
    asyncio.run(routes_module._transcribe_audio_result(b"audio", None, None, False))
    assert "task" not in seen
