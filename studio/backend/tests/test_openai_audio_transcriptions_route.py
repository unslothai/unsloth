# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/transcriptions.

The STT sidecar call (_transcribe_audio_result) is replaced with a light fake,
so these exercise the multipart wiring, model-id mapping, response formats, and
error propagation without whisper, llama.cpp, or a GPU."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.inference as routes_module
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers


def _make_client(monkeypatch, transcribe = None):
    calls = []

    async def _fake_transcribe(raw, model, language, fast, engine = None):
        calls.append(
            {"raw": raw, "model": model, "language": language, "fast": fast, "engine": engine}
        )
        if transcribe is not None:
            return await transcribe(raw)
        return {"text": "hello sloth", "language": "en", "duration": 1.2, "model": "small"}

    monkeypatch.setattr(routes_module, "_transcribe_audio_result", _fake_transcribe)

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), calls


def _post(cli, data = None, filename = "clip.wav", content = b"RIFFfake"):
    return cli.post(
        "/v1/audio/transcriptions",
        files = {"file": (filename, content, "audio/wav")},
        data = data or {},
    )


def test_json_response_is_text_only(monkeypatch):
    # OpenAI's json shape carries only the text; the sidecar's extra fields stay internal.
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli)
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello sloth"}
    assert calls[0]["raw"] == b"RIFFfake"
    assert calls[0]["fast"] is False


def test_text_response_is_plain_body(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "text"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello sloth"


def test_whisper1_and_missing_model_map_to_sidecar_default(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    assert _post(cli, data = {"model": "whisper-1"}).status_code == 200
    assert _post(cli).status_code == 200
    assert [c["model"] for c in calls] == [None, None]


def test_explicit_model_passes_through(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"model": "large-v3-turbo", "language": "de"})
    assert resp.status_code == 200
    assert calls[0]["model"] == "large-v3-turbo"
    assert calls[0]["language"] == "de"


def test_unknown_response_format_is_400(monkeypatch):
    cli, calls = _make_client(monkeypatch)
    resp = _post(cli, data = {"response_format": "srt"})
    assert resp.status_code == 400
    assert "srt" in resp.json()["error"]["message"]
    assert calls == []


def test_missing_file_is_rejected(monkeypatch):
    # install_api_error_handlers maps validation errors to a 400 OpenAI envelope on /v1.
    cli, calls = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/transcriptions", data = {"model": "whisper-1"})
    assert resp.status_code == 400
    assert calls == []


def test_sidecar_errors_keep_their_status(monkeypatch):
    # The shared error mapping (SttModelIdError -> 422, empty audio -> 400, ...) sits inside
    # _transcribe_audio_result; the route must not swallow or rewrap what it raises.
    async def _bad_model(raw):
        raise HTTPException(status_code = 422, detail = "Unknown STT model id.")

    cli, calls = _make_client(monkeypatch, transcribe = _bad_model)
    resp = _post(cli, data = {"model": "not-a-model"})
    assert resp.status_code == 422
    assert "Unknown STT model id." in resp.json()["error"]["message"]
