# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastAPI round-trip tests for the OpenAI-compatible POST /v1/audio/speech.

The TTS core (_generate_tts_wav) is replaced with a light fake, so these
exercise the route wiring, validation, gallery persistence, and the raw-WAV
response shape without torch, llama.cpp, weights, or a GPU."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery_module
import routes.inference as routes_module
from auth.authentication import get_current_subject
from routes.inference import router
from utils.api_errors import install_api_error_handlers

_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-payload"


def _make_client(monkeypatch, generate = None):
    calls = []

    async def _fake_generate(text, payload, request, current_subject):
        calls.append({"text": text, "payload": payload})
        if generate is not None:
            return await generate(text)
        return _WAV, 24000, "unsloth/orpheus-3b-0.1-ft", "snac"

    saved = []

    def _save(wav_bytes, meta):
        saved.append({"bytes": wav_bytes, "meta": meta})
        return {**meta, "id": "aud0", "url": "/api/inference/audio/gallery/aud0/file"}

    monkeypatch.setattr(routes_module, "_generate_tts_wav", _fake_generate)
    monkeypatch.setattr(gallery_module, "save", _save)

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), calls, saved


def test_returns_raw_wav_bytes(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hello sloth"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content == _WAV
    assert calls[0]["text"] == "hello sloth"


def test_persists_clip_to_gallery(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "persist me"})
    assert resp.status_code == 200
    assert len(saved) == 1
    meta = saved[0]["meta"]
    assert meta["prompt"] == "persist me"
    assert meta["model"] == "unsloth/orpheus-3b-0.1-ft"
    assert meta["audio_type"] == "snac"
    assert meta["sample_rate"] == 24000
    assert isinstance(meta["duration_s"], float)
    assert meta["created_at"]


def test_gallery_persist_failure_still_serves_audio(monkeypatch):
    # Persistence is best-effort: a full disk must not fail the request that produced the audio.
    cli, calls, saved = _make_client(monkeypatch)

    def _boom(wav_bytes, meta):
        raise OSError("disk full")

    monkeypatch.setattr(gallery_module, "save", _boom)
    resp = cli.post("/v1/audio/speech", json = {"input": "still speaks"})
    assert resp.status_code == 200
    assert resp.content == _WAV


def test_voice_and_speed_accepted_and_ignored(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post(
        "/v1/audio/speech",
        json = {"input": "hi", "voice": "alloy", "speed": 1.25, "model": "tts-1"},
    )
    assert resp.status_code == 200


def test_non_wav_response_format_is_400(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": "mp3"})
    assert resp.status_code == 400
    assert "mp3" in resp.json()["error"]["message"]
    assert calls == []  # rejected before any generation


def test_null_response_format_means_wav(monkeypatch):
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi", "response_format": None})
    assert resp.status_code == 200


def test_empty_input_is_rejected(monkeypatch):
    # install_api_error_handlers maps validation errors to a 400 OpenAI envelope on /v1.
    cli, calls, saved = _make_client(monkeypatch)
    resp = cli.post("/v1/audio/speech", json = {"input": ""})
    assert resp.status_code == 400
    assert calls == []


def test_core_error_propagates(monkeypatch):
    # "No model loaded" from the TTS core keeps its status through the route.
    async def _no_model(text):
        raise HTTPException(status_code = 400, detail = "No model loaded.")

    cli, calls, saved = _make_client(monkeypatch, generate = _no_model)
    resp = cli.post("/v1/audio/speech", json = {"input": "hi"})
    assert resp.status_code == 400
    assert saved == []


def test_wav_duration_seconds_reads_header():
    # A real 1-second 24 kHz mono WAV reports ~1.0s.
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x00\x00" * 24000)
    assert routes_module._wav_duration_seconds(buf.getvalue(), 24000) == 1.0
    # Unreadable bytes fall back to the 16-bit mono PCM estimate.
    fallback = routes_module._wav_duration_seconds(b"\x00" * (44 + 48000), 24000)
    assert fallback == 1.0
