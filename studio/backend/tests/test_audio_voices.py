# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

import core.inference.audio_gallery as gallery
from core.inference import audio_inputs, audio_voices

# pytest inserts rootdir, not this dir, so sibling imports need it on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_audio_inputs import _clip, _save, _tmp_gallery, client, encode, wav_bytes  # noqa: E402,F401

VOICES = "/api/inference/audio/voices"


def _input(seconds = 2.0, rate = 48000) -> str:
    return _save(encode("wav", "pcm_s16le", rate, "stereo", seconds), "me.wav")["id"]


def test_save_list_rename_and_delete_a_voice(client):
    input_id = _input()
    body = {"source": {"input_id": input_id}, "name": "  My   voice "}
    created = client.post(
        VOICES, json = {**body, "transcript": "Okay, I'm Cemo.", "language": "English"}
    )
    assert created.status_code == 201, created.text
    voice = created.json()
    keys = "id name transcript language duration_s sample_rate created_at url"
    assert set(voice) == set(keys.split())
    assert (voice["name"], voice["transcript"]) == ("My voice", "Okay, I'm Cemo.")
    assert (voice["sample_rate"], voice["duration_s"]) == (24000, 2.0)
    path = audio_voices.voice_path(voice["id"])
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1
    assert path.parent == gallery.gallery_dir() / "voices"
    assert voice["url"] == f"{VOICES}/{voice['id']}/file"
    served = client.get(voice["url"])
    assert served.status_code == 200 and served.headers["content-type"] == "audio/wav"

    second = client.post(VOICES, json = {**body, "name": "Second"}).json()
    assert [v["id"] for v in client.get(VOICES).json()["voices"]] == [second["id"], voice["id"]]

    url = f"{VOICES}/{voice['id']}"
    patched = client.patch(url, json = {"name": "Renamed", "transcript": None})
    assert patched.status_code == 200
    p = patched.json()
    assert (p["name"], p["transcript"], p["language"]) == ("Renamed", None, "English")

    assert client.delete(url).json() == {"removed": True}
    assert client.get(voice["url"]).status_code == 404
    assert client.delete(url).status_code == 404
    assert client.patch(url, json = {"name": "x"}).status_code == 404


@pytest.mark.parametrize(
    "make_source, duration",
    [
        (lambda: {"input_id": _input(45.0, rate = 8000)}, 30.0),
        (lambda: {"clip_id": _clip(rate = 22050, audio_type = "audiocpp_tts")["id"]}, 1.0),
    ],
)
def test_a_voice_is_24k_and_keeps_the_first_30_seconds(client, make_source, duration):
    response = client.post(VOICES, json = {"source": make_source(), "name": "v"})
    assert response.status_code == 201, response.text
    assert (response.json()["sample_rate"], response.json()["duration_s"]) == (24000, duration)
    assert not [p for p in audio_voices.voices_dir().iterdir() if p.name.startswith(".")]


def test_the_voice_cap_is_enforced(client, monkeypatch):
    monkeypatch.setattr(audio_voices, "MAX_VOICES", 2)
    source = {"input_id": _input(0.5)}
    codes = [client.post(VOICES, json = {"source": source, "name": n}).status_code for n in "ab"]
    assert codes == [201, 201]
    full = client.post(VOICES, json = {"source": source, "name": "c"})
    assert full.status_code == 400 and "2 saved voices" in full.json()["detail"]
    assert len(audio_voices.list_voices()) == 2


@pytest.mark.parametrize(
    "body, status",
    [
        ({"source": {"input_id": "x" * 32}, "name": ""}, 422),
        ({"source": {"input_id": "x" * 32, "clip_id": "y" * 32}, "name": "two"}, 422),
        ({"source": {"voice_id": "x" * 32}, "name": "from a voice"}, 422),
        ({"source": {"path": "/etc/passwd"}, "name": "path"}, 422),
        ({"source": {"input_id": "../../etc"}, "name": "escape"}, 422),
        ({"source": {"input_id": "x" * 32}, "name": "n", "file": "/tmp/x.wav"}, 422),
        ({"source": {"input_id": "f" * 32}, "name": "gone"}, 404),
    ],
)
def test_a_voice_names_its_source_by_one_safe_known_id(client, body, status):
    assert client.post(VOICES, json = body).status_code == status


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
    voice = audio_voices.create(audio_inputs.input_path(_input(1.0)), {"name": "v"})
    _source, prepared = audio_inputs.prepare_reference({"voice_id": voice["id"]})
    assert prepared.is_file() and prepared.name.startswith(f"v-{voice['id']}.")
    assert audio_voices.delete(voice["id"])
    assert not prepared.exists()
