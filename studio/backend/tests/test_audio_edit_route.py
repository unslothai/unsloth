# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/run with workflow edit, on test_audio_run_route's stub orchestrator: never a client
path in, never a server path in a response or sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core.inference import audio_gallery
from routes import inference
from utils.account_context import run_as

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_run_route import (  # noqa: E402,F401 - fixtures
    ALICE,
    BOB,
    _client,
    _input,
    _wav,
    isolated,
    stub,
)

ORIGINAL = "Okay, I'm Cemo and what you just heard wasn't a human voice."
EDITED = "Okay, I'm Sam and what you just heard wasn't a robot voice."
MARKUP = (
    "Okay, I'm <sub targ=\"Sam\">Cemo</sub> and what you just heard wasn't a "
    '<sub targ="robot">human</sub> voice.'
)
DOTS = "audio-cpp/audio.cpp-gguf/DotTTS-Edit-GGUF"
FIRERED = "audio-cpp/audio.cpp-gguf/FireRedAudio-GGUF"


def _edit_info(
    family,
    workflows,
    style,
    delivery = False,
    max_changes = None,
):
    return {
        "is_audio": True,
        "audio_type": "audiocpp_tts",
        "audio_family": family,
        "audio_workflows": list(workflows),
        "audio_reference_text": None,
        "audio_required_inputs": [],
        "audio_clone": None,
        "audio_edit": {
            "style": style,
            "delivery": delivery,
            "max_changes": max_changes,
        },
    }


def _dots(stub):
    return stub["use"](DOTS, _edit_info("dots_tts", ("speak", "edit"), "markup"))


def _firered(stub):
    return stub["use"](
        FIRERED, _edit_info("firered_audio", ("clone", "edit"), "instructions", True, 5)
    )


def _edit(
    client,
    source = None,
    edit = None,
    **body,
):
    payload = {
        "workflow": "edit",
        "text": EDITED,
        "inputs": {"reference_text": ORIGINAL, **({"source": source} if source else {})},
        "edit": edit if edit is not None else {"mode": "words", "markup": MARKUP},
        **body,
    }
    return client.post("/api/inference/audio/run", json = payload)


MISMATCH = "The changes do not match the transcript. Check ① and ② again."


def _detail(response):
    assert response.status_code == 400, response.text
    return response.json()["detail"]


def _sidecar(tmp_path, clip_id):
    return (tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip_id}.json").read_text(
        encoding = "utf-8"
    )


@pytest.mark.parametrize(
    "body",
    [
        {"inputs": {"source": {"path": "/etc/passwd"}}},
        {"inputs": {"source": {"input_id": "a" * 32, "source_audio": "/etc/passwd"}}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": {"mode": "words", "path": "/x"}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": {"source_audio": "/x.wav"}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": {"mode": "rewrite"}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": {"speed": 3.0}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": {"instructions": ["x"] * 9}},
        {"workflow": "clone", "inputs": {"reference": {"input_id": "a" * 32}}},
        {"inputs": {"source": {"input_id": "a" * 32}}, "edit": None},
        {"workflow": "speak", "inputs": {"source": {"input_id": "a" * 32}}, "edit": None},
        {"inputs": {"source": {"input_id": "a" * 32}, "reference": {"input_id": "a" * 32}}},
        {"options": {"source_audio": "/etc/passwd"}},
    ],
)
def test_client_paths_and_misplaced_edit_fields_are_422(stub, body):
    _dots(stub)
    payload = {"workflow": "edit", "text": EDITED, "edit": {"mode": "words", "markup": MARKUP}}
    payload.update(body)
    if payload.get("edit") is None:
        payload.pop("edit")
    with _client(ALICE) as client:
        response = client.post("/api/inference/audio/run", json = payload)
    assert response.status_code == 422, response.text
    assert stub["backend"].calls == []


def test_an_edit_needs_a_recording_and_not_a_saved_voice(stub):
    from core.inference import audio_inputs, audio_voices

    _dots(stub)
    input_id = _input(ALICE)
    voice = run_as(
        ALICE, lambda: audio_voices.create(audio_inputs.input_path(input_id), {"name": "Mine"})
    )
    with _client(ALICE) as client:
        missing = _edit(client)
        voiced = _edit(client, {"voice_id": voice["id"]})
    assert _detail(missing) == "Add a recording to edit."
    assert _detail(voiced) == "Pick a recording or a history clip to edit."
    assert stub["backend"].calls == []


def test_a_recording_over_30_seconds_is_refused(stub):
    _dots(stub)
    input_id = _input(ALICE, 31.0)
    with _client(ALICE) as client:
        long = _edit(client, {"input_id": input_id})
    assert _detail(long) == "Edit works on recordings up to 30 s. Record or upload a shorter take."
    assert stub["backend"].calls == []


def test_a_model_that_cannot_edit_is_a_400(stub, monkeypatch):
    input_id = _input(ALICE)
    stub["use"](
        "audio-cpp/audio.cpp-gguf/DotTTS-MF-GGUF",
        {
            "is_audio": True,
            "audio_type": "audiocpp_tts",
            "audio_workflows": ["speak"],
            "audio_reference_text": None,
            "audio_required_inputs": [],
            "audio_edit": None,
        },
    )
    with _client(ALICE) as client:
        response = _edit(client, {"input_id": input_id})
    assert _detail(response) == "Load a model that can edit speech."
    assert stub["backend"].calls == []

    # A GGUF speech model on llama.cpp has no workflows to read.
    from utils.audio_tokens import GGUF_TTS_AUDIO_TYPES

    calls = []

    class _Llama:
        is_loaded = True
        _is_audio = True
        _audio_type = sorted(GGUF_TTS_AUDIO_TYPES)[0]
        model_identifier = "someone/tts-GGUF"
        context_length = None

        def generate_audio_response(self, **kwargs):
            calls.append(kwargs)
            return _wav(), 24000

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
    with _client(ALICE) as client:
        direct = _edit(client, {"input_id": input_id})
    assert _detail(direct) == "Load a model that can edit speech."
    assert calls == []


@pytest.mark.parametrize(
    "family, edit, text, detail",
    [
        ("dots", {"mode": "words", "markup": MARKUP.replace("Sam", "Sammy")}, EDITED, MISMATCH),
        ("dots", {"mode": "words", "markup": "Okay <b>x</b>"}, EDITED, MISMATCH),
        ("dots", {"mode": "words"}, EDITED, "DotTTS-Edit needs the marked-up changes."),
        ("dots", {"mode": "words"}, ORIGINAL, "Change at least one word."),
        ("dots", {"mode": "delivery", "speed": 1.5}, EDITED, "Delivery changes need FireRedAudio."),
        (
            "firered",
            {"mode": "words", "instructions": ["Replace 'human' with 'robot'."] * 6},
            EDITED,
            "FireRedAudio applies at most 5 changes. Make fewer changes, or use DotTTS Edit.",
        ),
        (
            "firered",
            {"mode": "words", "instructions": ["Make it sound happy."]},
            EDITED,
            "That change is not one FireRedAudio understands.",
        ),
        ("firered", {"mode": "words"}, EDITED, "Change at least one word."),
        ("firered", {"mode": "delivery", "speed": 1.0}, EDITED, "Pick a speed or a pitch change."),
    ],
)
def test_changes_that_do_not_fit_the_model_are_400(stub, family, edit, text, detail):
    (_dots if family == "dots" else _firered)(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _edit(client, {"input_id": input_id}, edit, text = text)
    assert _detail(response) == detail
    assert stub["backend"].calls == []


def test_an_edit_run_hands_the_worker_paths_and_keeps_the_source(stub, tmp_path):
    backend = _dots(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _edit(client, {"input_id": input_id}, options = {"num_inference_steps": 10})
        assert response.status_code == 200, response.text
        body = response.json()
        files = [client.get(clip["url"]).status_code for clip in body["clips"]]
    (call,) = backend.calls
    source = Path(call["audio_inputs"]["source"])
    inputs_root = tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    assert source.is_absolute() and source.parent == inputs_root
    assert source.name.startswith(f"{input_id}.24000.mono")
    assert (call["workflow"], call["reference_text"], call["edit"]) == (
        "edit",
        ORIGINAL,
        {"mode": "words", "markup": MARKUP},
    )
    assert call["text"] == EDITED and "speed" not in call
    assert body["audio"] is None
    output, original = body["clips"]
    assert (output["role"], output["workflow"]) == ("output", "edit")
    assert (original["role"], original["workflow"]) == ("source", "edit")
    assert files == [200, 200]
    out_meta = json.loads(_sidecar(tmp_path, output["id"]))
    src_meta = json.loads(_sidecar(tmp_path, original["id"]))
    assert src_meta["role"] == "source"
    assert out_meta["source_clip_id"] == original["id"]
    assert out_meta["prompt"] == EDITED and out_meta["reference_name"] == "me.webm"
    assert out_meta["settings"] == {
        "mode": "words",
        "changes": 2,
        "original_text": ORIGINAL,
        "options": {"num_inference_steps": 10},
    }
    assert (src_meta["role"], src_meta["model"], src_meta["prompt"]) == (
        "source",
        "Recording",
        ORIGINAL,
    )
    for clip in (output, original):
        text = _sidecar(tmp_path, clip["id"])
        assert str(tmp_path) not in text and "inputs" not in text
    assert str(tmp_path) not in response.text


def test_the_same_upload_reuses_its_source_clip(stub):
    _dots(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        first = _edit(client, {"input_id": input_id}).json()
        second = _edit(client, {"input_id": input_id}).json()
    assert first["clips"][1]["id"] == second["clips"][1]["id"]
    assert first["clips"][0]["id"] != second["clips"][0]["id"]
    clips = run_as(ALICE, audio_gallery.list_audio)
    assert sum(c.get("role") == "source" for c in clips) == 1


def test_the_cap_keeps_a_source_its_edits_still_play(stub, monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    _dots(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        _edit(client, {"input_id": input_id})
        source = _edit(client, {"input_id": input_id}).json()["clips"][1]
        assert client.get(source["url"]).status_code == 200


def test_clearing_edit_keeps_the_source_of_an_archived_edit(stub):
    _dots(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        output, source = _edit(client, {"input_id": input_id}).json()["clips"]
        r = client.patch(f"/api/inference/audio/gallery/{output['id']}", json = {"archived": True})
        assert r.status_code == 200, r.text
        assert client.delete("/api/inference/audio/gallery?workflow=edit").status_code == 200
        assert client.get(source["url"]).status_code == 200
        # Once nothing plays it, the source goes with the next clear.
        client.patch(f"/api/inference/audio/gallery/{output['id']}", json = {"archived": False})
        assert client.delete("/api/inference/audio/gallery?workflow=edit").status_code == 200
        assert client.get(source["url"]).status_code == 404


def test_a_history_clip_source_makes_no_copy(stub, tmp_path):
    _firered(stub)
    clip = run_as(
        ALICE,
        audio_gallery.save,
        _wav(),
        {
            "prompt": ORIGINAL,
            "model": "m",
            "audio_type": "audiocpp_tts",
            "workflow": "clone",
            "sample_rate": 24000,
            "duration_s": 0.1,
            "created_at": "2026-10-02T00:00:00Z",
        },
    )
    with _client(ALICE) as client:
        response = _edit(
            client,
            {"clip_id": clip["id"]},
            {"mode": "delivery", "speed": 1.5, "pitch_steps": 3},
            text = ORIGINAL,
        )
    assert response.status_code == 200, response.text
    (output,) = response.json()["clips"]
    meta = json.loads(_sidecar(tmp_path, output["id"]))
    assert meta["source_clip_id"] == clip["id"]
    assert meta["settings"] == {
        "mode": "delivery",
        "changes": 2,
        "original_text": ORIGINAL,
        "options": {},
        "speed": 1.5,
        "pitch_steps": 3,
    }
    assert len(run_as(ALICE, audio_gallery.list_audio)) == 2


def test_another_accounts_recording_is_404(stub):
    _dots(stub)
    input_id = _input(ALICE)
    with _client(BOB) as client:
        response = _edit(client, {"input_id": input_id})
    assert response.status_code == 404
    assert stub["backend"].calls == []


def test_the_orchestrator_and_worker_carry_the_edit():
    import queue
    import threading

    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Backend:
        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

    edit = {"mode": "words", "instructions": ["Replace 'human' with 'robot'."]}
    _handle_generate_audio(
        _Backend(),
        {
            "request_id": "r1",
            "text": EDITED,
            "workflow": "edit",
            "audio_inputs": {"source": "/abs/source.wav"},
            "reference_text": ORIGINAL,
            "edit": edit,
        },
        queue.Queue(),
        threading.Event(),
    )
    _handle_generate_audio(
        _Backend(), {"request_id": "r2", "text": "hi"}, queue.Queue(), threading.Event()
    )
    assert seen[0]["edit"] == edit and seen[0]["audio_inputs"] == {"source": "/abs/source.wav"}
    assert "edit" not in seen[1]
