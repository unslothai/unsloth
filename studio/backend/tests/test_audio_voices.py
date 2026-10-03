# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved voices: CRUD through the routes, the 30-second cut and 200-voice cap, unsafe ids and
sidecar ownership."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery
from auth.authentication import get_current_subject
from core.inference import audio_inputs, audio_voices

# Sibling imports need the tests dir on the path: pytest inserts rootdir, not this package.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_inputs import _chunks, encode, wav_bytes  # noqa: E402


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _client() -> TestClient:
    from routes import inference

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


def _input(
    seconds = 2.0,
    rate = 48000,
    layout = "stereo",
) -> str:
    data = encode("wav", "pcm_s16le", rate, layout, seconds)
    record, _ = asyncio.run(audio_inputs.save_stream(_chunks([data]), "me.wav"))
    return record["id"]


def test_save_list_rename_and_delete_a_voice():
    input_id = _input(2.0)
    with _client() as client:
        created = client.post(
            "/api/inference/audio/voices",
            json = {
                "source": {"input_id": input_id, "trim": {"start_s": 0.5, "end_s": 1.5}},
                "name": "  My   voice ",
                "transcript": "Okay, I'm Cemo.",
                "language": "English",
            },
        )
        assert created.status_code == 201, created.text
        voice = created.json()
        assert set(voice) == {
            "id",
            "name",
            "transcript",
            "language",
            "duration_s",
            "sample_rate",
            "created_at",
            "url",
        }
        assert voice["name"] == "My voice" and voice["transcript"] == "Okay, I'm Cemo."
        # Stored 24 kHz mono, trim applied.
        assert (voice["sample_rate"], voice["duration_s"]) == (24000, 1.0)
        path = audio_voices.voice_path(voice["id"])
        assert audio_inputs.wav_info(path)["channels"] == 1
        assert path.parent == gallery.gallery_dir() / "voices"
        assert voice["url"] == f"/api/inference/audio/voices/{voice['id']}/file"
        served = client.get(voice["url"])
        assert served.status_code == 200 and served.headers["content-type"] == "audio/wav"

        second = client.post(
            "/api/inference/audio/voices",
            json = {"source": {"input_id": input_id}, "name": "Second"},
        ).json()
        listed = client.get("/api/inference/audio/voices").json()["voices"]
        assert [v["id"] for v in listed] == [second["id"], voice["id"]]  # newest first

        patched = client.patch(
            f"/api/inference/audio/voices/{voice['id']}",
            json = {"name": "Renamed", "transcript": None},
        )
        assert patched.status_code == 200
        assert patched.json()["name"] == "Renamed" and patched.json()["transcript"] is None
        assert patched.json()["language"] == "English"

        assert client.delete(f"/api/inference/audio/voices/{voice['id']}").json() == {
            "removed": True
        }
        assert client.get(voice["url"]).status_code == 404
        assert client.delete(f"/api/inference/audio/voices/{voice['id']}").status_code == 404
        assert (
            client.patch(
                f"/api/inference/audio/voices/{voice['id']}", json = {"name": "x"}
            ).status_code
            == 404
        )


def test_a_voice_from_a_history_clip():
    clip = gallery.save(
        wav_bytes(rate = 22050, seconds = 1.0),
        {
            "prompt": "said this",
            "model": "m",
            "audio_type": "audiocpp_tts",
            "sample_rate": 22050,
            "duration_s": 1.0,
            "created_at": "2026-10-02T00:00:00Z",
        },
    )
    with _client() as client:
        response = client.post(
            "/api/inference/audio/voices",
            json = {"source": {"clip_id": clip["id"]}, "name": "From history"},
        )
    assert response.status_code == 201 and response.json()["sample_rate"] == 24000


def test_a_voice_keeps_the_first_30_seconds():
    # A clone run sends at most 30 s of a reference, so a longer clip is cut, not refused.
    input_id = _input(45.0, rate = 8000, layout = "mono")
    with _client() as client:
        response = client.post(
            "/api/inference/audio/voices",
            json = {"source": {"input_id": input_id}, "name": "Long"},
        )
        assert response.status_code == 201 and response.json()["duration_s"] == 30.0
        trimmed = client.post(
            "/api/inference/audio/voices",
            json = {
                "source": {"input_id": input_id, "trim": {"start_s": 1.0, "end_s": 11.0}},
                "name": "Trimmed",
            },
        )
        assert trimmed.status_code == 201 and trimmed.json()["duration_s"] == 10.0
    assert not [p for p in audio_voices.voices_dir().iterdir() if p.name.startswith(".")]


def test_the_two_hundredth_voice_is_the_last(monkeypatch):
    monkeypatch.setattr(audio_voices, "MAX_VOICES", 2)
    input_id = _input(0.5)
    with _client() as client:
        for name in ("a", "b"):
            assert (
                client.post(
                    "/api/inference/audio/voices",
                    json = {"source": {"input_id": input_id}, "name": name},
                ).status_code
                == 201
            )
        full = client.post(
            "/api/inference/audio/voices", json = {"source": {"input_id": input_id}, "name": "c"}
        )
    assert full.status_code == 400 and "2 saved voices" in full.json()["detail"]
    assert audio_voices.MAX_VOICES == 2 and len(audio_voices.list_voices()) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"source": {"input_id": "x" * 32}, "name": ""},
        {"source": {"input_id": "x" * 32, "clip_id": "y" * 32}, "name": "two"},
        {"source": {"voice_id": "x" * 32}, "name": "from a voice"},
        {"source": {"path": "/etc/passwd"}, "name": "path"},
        {"source": {"input_id": "../../etc"}, "name": "escape"},
        {"source": {"input_id": "x" * 32}, "name": "n", "file": "/tmp/x.wav"},
    ],
)
def test_a_voice_names_its_source_by_one_safe_id(body):
    with _client() as client:
        assert client.post("/api/inference/audio/voices", json = body).status_code == 422


def test_an_unknown_source_is_404():
    with _client() as client:
        response = client.post(
            "/api/inference/audio/voices", json = {"source": {"input_id": "f" * 32}, "name": "gone"}
        )
    assert response.status_code == 404


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "x" * 129])
def test_unsafe_voice_ids_resolve_to_nothing(bad):
    assert audio_voices.voice_path(bad) is None
    assert audio_voices.get(bad) is None
    assert audio_voices.update(bad, {"name": "x"}) is None
    assert audio_voices.delete(bad) is False


def test_a_lone_wav_or_foreign_sidecar_is_not_a_voice():
    directory = audio_voices.voices_dir()
    (directory / "orphan.wav").write_bytes(wav_bytes())
    (directory / "foreign.wav").write_bytes(wav_bytes())
    (directory / "foreign.json").write_text(json.dumps({"title": "not ours"}), encoding = "utf-8")
    assert audio_voices.list_voices() == []
    for voice_id in ("orphan", "foreign"):
        assert audio_voices.voice_path(voice_id) is None
        assert audio_voices.delete(voice_id) is False
    assert (directory / "orphan.wav").is_file() and (directory / "foreign.wav").is_file()


def test_deleting_a_voice_drops_its_prepared_copies():
    input_id = _input(1.0)
    voice = audio_voices.create(audio_inputs.input_path(input_id), {"name": "v"})
    _source, prepared = audio_inputs.prepare_reference({"voice_id": voice["id"]})
    assert prepared.is_file() and prepared.name.startswith(f"v-{voice['id']}.")
    assert audio_voices.delete(voice["id"])
    assert not prepared.exists()
