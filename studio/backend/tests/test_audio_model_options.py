# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.audio_cpp_backend as backend
from auth.authentication import get_current_subject
from core.inference import audio_cpp_models
from utils.api_errors import install_api_error_handlers

URL = "/api/inference/audio/model-options"
KOKORO = "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"
OPTIONS = [
    {"name": "voice", "type": "enum", "values": ["af_heart", "am_adam"], "default": "af_heart"},
    {"name": "speed", "type": "float", "min": 0.5, "max": 2.0, "default": 1.0},
]


@pytest.fixture
def client():
    from routes import inference

    app = FastAPI()
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(inference.router, prefix = "/v1")
    yield TestClient(app)


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple[str, bool]] = []
    downloaded = {KOKORO}
    known = {KOKORO, "audio-cpp/audio.cpp-gguf/OmniVoice-GGUF"}

    def fake_resolve(identifier, variant = None, hf_token = None, *, network = True, **_):
        seen.append((identifier, network))
        if identifier not in known or (not network and identifier not in downloaded):
            return None
        return SimpleNamespace(id = identifier, unsupported = False)

    monkeypatch.setattr(audio_cpp_models, "resolve", fake_resolve)
    monkeypatch.setattr(
        backend,
        "model_info_fields",
        lambda model: {"audio_family": "kokoro_tts", "audio_options": OPTIONS, "gguf_variant": "Q8_0"},
    )
    return seen


def test_a_downloaded_model_is_read_from_disk_without_the_hub(client, calls):
    res = client.get(URL, params = {"model": KOKORO})
    assert res.status_code == 200
    body = res.json()
    assert body == {"model": KOKORO, "audio_family": "kokoro_tts", "audio_options": OPTIONS}
    assert calls == [(KOKORO, False)]


def test_a_model_not_yet_downloaded_reads_its_spec_from_the_hub(client, calls):
    omni = "audio-cpp/audio.cpp-gguf/OmniVoice-GGUF"
    res = client.get(URL, params = {"model": omni})
    assert res.status_code == 200
    assert res.json() == {"model": omni, "audio_family": "kokoro_tts", "audio_options": OPTIONS}


def test_an_unknown_or_unsupported_model_is_404(client, calls, monkeypatch):
    assert client.get(URL, params = {"model": "someone/not-audio"}).status_code == 404
    monkeypatch.setattr(
        audio_cpp_models,
        "resolve",
        lambda *a, **k: SimpleNamespace(id = "x", unsupported = True),
    )
    assert client.get(URL, params = {"model": KOKORO}).status_code == 404


def test_a_resolve_error_is_404_not_a_500_with_its_text(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path /home/someone")

    monkeypatch.setattr(audio_cpp_models, "resolve", boom)
    res = client.get(URL, params = {"model": KOKORO})
    assert res.status_code == 404
    assert "secret" not in res.text


@pytest.mark.parametrize("model", ["", "   ", "x" * 513])
def test_a_blank_or_oversized_id_is_refused(client, calls, model):
    assert client.get(URL, params = {"model": model}).status_code == 400
    assert calls == []


def test_it_is_a_studio_route_not_part_of_the_openai_api(client, calls):
    assert client.get("/v1/audio/model-options", params = {"model": KOKORO}).status_code == 404
