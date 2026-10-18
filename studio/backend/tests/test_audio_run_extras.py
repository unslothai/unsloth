# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/run says what it changed (applied, warnings) and where each clip came from."""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import audio_inputs
from models.inference import AudioRunResponse
from routes import inference
from utils.account_context import bind_account, reset_account

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_account_media_isolation import ALICE, isolated  # noqa: E402,F401

MUSIC = "audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF"
TTS = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF"
MUSIC_INFO = {
    "is_audio": True,
    "audio_type": "audiocpp_music",
    "audio_family": "stable_audio",
    "audio_workflows": ["music"],
    "audio_music": {"modes": [{"id": "song", "duration": {"min": 1, "max": 10, "default": 5}}]},
}
TTS_INFO = {
    "is_audio": True,
    "audio_type": "audiocpp_tts",
    "audio_family": "qwen3_tts",
    "audio_workflows": ["clone", "speak"],
    "audio_reference_text": "required",
    "audio_required_inputs": [],
    "audio_clone": {"reference_text": "required", "emotion_audio": False},
}


def wav_bytes(rate = 24000, frames = 2400) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


class Backend:
    """The orchestrator as /audio/run sees it; ``run`` replaces the generation."""

    def __init__(
        self,
        name,
        info,
        ignored = (),
        run = None,
    ):
        self.active_model_name = name
        self.models = {name: info}
        self.ignored = list(ignored)
        self.run = run
        self.calls: list[dict] = []

    def generate_audio_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.ignored:
            kwargs["stats_holder"]["stats"] = {"ignored_options": self.ignored}
        if self.run is not None:
            self.run(kwargs)
        return wav_bytes(), 24000


@pytest.fixture
def use_backend(monkeypatch):
    class _Llama:
        is_loaded = False
        _is_audio = False

    async def _noop_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference, "_maybe_auto_switch_model", _noop_switch)

    def use(backend):
        monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
        return backend

    return use


def app_for(account) -> FastAPI:
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    for prefix in ("/api/inference", "/v1"):
        app.include_router(inference.router, prefix = prefix)
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return app


@pytest.mark.parametrize("path", ["/api/inference/audio/run", "/v1/audio/run"])
def test_a_clamped_music_length_is_reported_on_both_mounts(use_backend, path):
    backend = use_backend(Backend(MUSIC, MUSIC_INFO))
    with TestClient(app_for(ALICE)) as client:
        song = {"workflow": "music", "mode": "song", "text": "house"}
        long = client.post(path, json = {**song, "duration_s": 75})
        short = client.post(path, json = {**song, "duration_s": 5})
    assert long.status_code == 200, long.text
    assert backend.calls[0]["music"]["duration_s"] == 10.0
    assert long.json()["applied"] == {"duration_s": 10.0}
    assert long.json()["warnings"] == ["duration_s 75 clamped to 10"]
    assert (short.json()["applied"], short.json()["warnings"]) == ({"duration_s": 5.0}, [])


def test_an_ignored_option_gives_exactly_one_warning(use_backend):
    use_backend(Backend(MUSIC, MUSIC_INFO, ignored = ["cfg_scale"]))
    with TestClient(app_for(ALICE)) as client:
        response = client.post(
            "/v1/audio/run",
            json = {
                "workflow": "music",
                "mode": "song",
                "text": "house",
                "options": {"cfg_scale": 3},
            },
        )
    body = response.json()
    assert body["ignored_options"] == ["cfg_scale"]
    assert body["warnings"] == ["Ignored option cfg_scale: the model does not take it."]
    # The route builds the response and FastAPI validates it again: still one line.
    again = AudioRunResponse.model_validate(body)
    assert again.warnings == body["warnings"]


@pytest.mark.parametrize("kind, field", [("voice", "voice_id"), ("clip", "source_clip_id")])
def test_clip_rows_carry_their_lineage(use_backend, monkeypatch, tmp_path, kind, field):
    use_backend(Backend(TTS, TTS_INFO))
    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav_bytes())
    source_id = "c" * 32
    monkeypatch.setattr(
        audio_inputs,
        "prepare_reference",
        lambda _ref: (audio_inputs.Source(kind, source_id, ref, "Alice"), ref),
    )
    with TestClient(app_for(ALICE)) as client:
        response = client.post(
            "/api/inference/audio/run",
            json = {
                "workflow": "clone",
                "text": "Hello.",
                "inputs": {"reference": {f"{kind}_id": source_id}, "reference_text": "Hi."},
            },
        )
    assert response.status_code == 200, response.text
    (clip,) = response.json()["clips"]
    other = ({"voice_id", "source_clip_id"} - {field}).pop()
    assert (clip[field], clip[other]) == (source_id, None)


def test_a_length_the_model_never_receives_is_not_reported_as_applied(use_backend):
    info = {**MUSIC_INFO, "audio_family": "yue2"}
    use_backend(Backend("audio-cpp/audio.cpp-gguf/YuE2-GGUF", info))
    with TestClient(app_for(ALICE)) as client:
        song = {"workflow": "music", "mode": "song", "text": "house"}
        asked = client.post("/api/inference/audio/run", json = {**song, "duration_s": 10})
        plain = client.post("/api/inference/audio/run", json = song)
    assert asked.status_code == 200, asked.text
    assert asked.json()["applied"] == {}
    assert asked.json()["warnings"] == ["duration_s ignored: YuE2 sets its own length"]
    assert (plain.json()["applied"], plain.json()["warnings"]) == ({}, [])
