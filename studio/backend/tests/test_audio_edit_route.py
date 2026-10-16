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
    _inputs_root,
    _sidecar,
    _voice,
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
MISMATCH = "The changes do not match the transcript. Check ① and ② again."
CANNOT = "Load a model that can edit speech."
_MODELS = {
    "dots": ("DotTTS-Edit-GGUF", ["speak", "edit"], ("markup", False, None)),
    "firered": ("FireRedAudio-GGUF", ["clone", "edit"], ("instructions", True, 5)),
    "mf": ("DotTTS-MF-GGUF", ["speak"], None),
}


def _load(stub, name = "dots"):
    folder, workflows, rules = _MODELS[name]
    info = {
        "is_audio": True,
        "audio_type": "audiocpp_tts",
        "audio_workflows": workflows,
        "audio_reference_text": None,
        "audio_required_inputs": [],
        "audio_clone": None,
        "audio_edit": rules and dict(zip(("style", "delivery", "max_changes"), rules)),
    }
    return stub["use"](f"audio-cpp/audio.cpp-gguf/{folder}", info)


def _edit(
    client,
    source = None,
    edit = None,
    **body,
):
    if isinstance(source, str):
        source = {"input_id": source}
    payload = {
        "workflow": "edit",
        "text": EDITED,
        "inputs": {"reference_text": ORIGINAL, **({"source": source} if source else {})},
        "edit": edit or {"mode": "words", "markup": MARKUP},
        **body,
    }
    return client.post("/api/inference/audio/run", json = payload)


_SRC = {"source": {"input_id": "a" * 32}}


# fmt: off
@pytest.mark.parametrize("body", [
    {"inputs": {"source": {"path": "/etc/passwd"}}},
    {"inputs": {"source": {"input_id": "a" * 32, "source_audio": "/etc/passwd"}}},
    {"inputs": _SRC, "edit": {"mode": "words", "path": "/x"}},
    {"inputs": _SRC, "edit": {"source_audio": "/x.wav"}},
    {"inputs": _SRC, "edit": {"mode": "rewrite"}},
    {"inputs": _SRC, "edit": {"speed": 3.0}},
    {"inputs": _SRC, "edit": {"instructions": ["x"] * 9}},
    {"workflow": "clone", "inputs": {"reference": {"input_id": "a" * 32}}},
    {"inputs": _SRC, "edit": None},
    {"inputs": {**_SRC, "reference": {"input_id": "a" * 32}}},
    {"options": {"source_audio": "/etc/passwd"}},
])
# fmt: on
def test_client_paths_and_misplaced_edit_fields_are_422(stub, body):
    _load(stub)
    payload = {"workflow": "edit", "text": EDITED, "edit": {"mode": "words", "markup": MARKUP}}
    payload.update(body)
    if payload["edit"] is None:
        payload.pop("edit")
    with _client(ALICE) as client:
        response = client.post("/api/inference/audio/run", json = payload)
    assert response.status_code == 422, response.text
    assert stub["backend"].calls == []


W = {"mode": "words"}


# fmt: off
@pytest.mark.parametrize("model, source, edit, text, detail", [
    ("dots", None, None, EDITED, "Add a recording to edit."),
    ("dots", "voice", None, EDITED, "Pick a recording or a history clip to edit."),
    ("dots", 31.0, None, EDITED, "Edit works on recordings up to 30 s. Record or upload a shorter take."),
    ("mf", 1.0, None, EDITED, CANNOT),
    ("llama", 1.0, None, EDITED, CANNOT),
    ("dots", 1.0, {**W, "markup": MARKUP.replace("Sam", "Sammy")}, EDITED, MISMATCH),
    ("dots", 1.0, {**W, "markup": "Okay <b>x</b>"}, EDITED, MISMATCH),
    ("dots", 1.0, W, EDITED, "DotTTS-Edit needs the marked-up changes."),
    ("dots", 1.0, W, ORIGINAL, "Change at least one word."),
    ("dots", 1.0, {"mode": "delivery", "speed": 1.5}, EDITED, "Delivery changes need FireRedAudio."),
    ("firered", 1.0, {**W, "instructions": ["Replace 'human' with 'robot'."] * 6}, EDITED,
     "FireRedAudio applies at most 5 changes. Make fewer changes, or use DotTTS Edit."),
    ("firered", 1.0, {**W, "instructions": ["Make it sound happy."]}, EDITED,
     "That change is not one FireRedAudio understands."),
    ("firered", 1.0, W, EDITED, "Change at least one word."),
    ("firered", 1.0, {"mode": "delivery", "speed": 1.0}, EDITED, "Pick a speed or a pitch change."),
])
# fmt: on
def test_edits_the_route_refuses_are_400(stub, monkeypatch, model, source, edit, text, detail):
    _load(stub, "mf" if model == "llama" else model)
    if model == "llama":
        from utils.audio_tokens import GGUF_TTS_AUDIO_TYPES
        class _Llama:
            is_loaded = _is_audio = True
            _audio_type = sorted(GGUF_TTS_AUDIO_TYPES)[0]
            model_identifier = "someone/tts-GGUF"
            context_length = None

            def generate_audio_response(self, **kwargs):
                raise AssertionError("llama.cpp was asked to edit")

        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
    if source == "voice":
        source = {"voice_id": _voice(ALICE, _input(ALICE))["id"]}
    elif source:
        source = _input(ALICE, source)
    with _client(ALICE) as client:
        response = _edit(client, source, edit, text = text)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == detail
    assert stub["backend"].calls == []


def test_an_edit_run_hands_the_worker_paths_and_keeps_the_source(stub, tmp_path):
    backend = _load(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _edit(client, input_id, options = {"num_inference_steps": 10})
        assert response.status_code == 200, response.text
        body = response.json()
        assert [client.get(clip["url"]).status_code for clip in body["clips"]] == [200, 200]
    (call,) = backend.calls
    source = Path(call["audio_inputs"]["source"])
    assert source.is_absolute() and source.parent == _inputs_root(tmp_path)
    assert source.name.startswith(f"{input_id}.24000.mono")
    assert "speed" not in call and body["audio"] is None
    output, original = body["clips"]
    out_meta, src_meta = (json.loads(_sidecar(tmp_path, c["id"])) for c in (output, original))
    # fmt: off
    assert (call["workflow"], call["text"], call["reference_text"], call["edit"]) == (
        "edit", EDITED, ORIGINAL, {"mode": "words", "markup": MARKUP})
    assert [(c["role"], c["workflow"]) for c in body["clips"]] == [("output", "edit"), ("source", "edit")]
    assert (out_meta["source_clip_id"], out_meta["prompt"], out_meta["reference_name"]) == (
        original["id"], EDITED, "me.webm")
    assert out_meta["settings"] == {
        "mode": "words", "changes": 2, "original_text": ORIGINAL, "options": {"num_inference_steps": 10}}
    assert (src_meta["role"], src_meta["model"], src_meta["prompt"]) == ("source", "Recording", ORIGINAL)
    # fmt: on
    for clip in body["clips"]:
        text = _sidecar(tmp_path, clip["id"])
        assert str(tmp_path) not in text and "inputs" not in text
    assert str(tmp_path) not in response.text


def test_one_source_clip_per_upload_kept_while_an_edit_plays_it(stub, monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "2")
    _load(stub)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        first = _edit(client, input_id).json()["clips"]
        output, source = _edit(client, input_id).json()["clips"]
        assert first[1]["id"] == source["id"] and first[0]["id"] != output["id"]
        assert client.get(source["url"]).status_code == 200
        assert sum(c.get("role") == "source" for c in run_as(ALICE, audio_gallery.list_audio)) == 1
        clip = f"/api/inference/audio/gallery/{output['id']}"
        assert client.patch(clip, json = {"archived": True}).status_code == 200
        assert client.delete("/api/inference/audio/gallery?workflow=edit").status_code == 200
        assert client.get(source["url"]).status_code == 200
        client.patch(clip, json = {"archived": False})
        assert client.delete("/api/inference/audio/gallery?workflow=edit").status_code == 200
        assert client.get(source["url"]).status_code == 404


def test_a_history_clip_source_makes_no_copy(stub, tmp_path):
    _load(stub, "firered")
    meta = {"prompt": ORIGINAL, "model": "m", "audio_type": "audiocpp_tts", "workflow": "clone"}
    meta |= {"sample_rate": 24000, "duration_s": 0.1, "created_at": "2026-10-02T00:00:00Z"}
    clip = run_as(ALICE, audio_gallery.save, _wav(), meta)
    delivery = {"mode": "delivery", "speed": 1.5, "pitch_steps": 3}
    with _client(ALICE) as client:
        response = _edit(client, {"clip_id": clip["id"]}, delivery, text = ORIGINAL)
    assert response.status_code == 200, response.text
    (output,) = response.json()["clips"]
    meta = json.loads(_sidecar(tmp_path, output["id"]))
    assert meta["source_clip_id"] == clip["id"]
    assert meta["settings"] == {**delivery, "changes": 2, "original_text": ORIGINAL, "options": {}}
    assert len(run_as(ALICE, audio_gallery.list_audio)) == 2


def test_another_accounts_recording_is_404(stub):
    _load(stub)
    input_id = _input(ALICE)
    with _client(BOB) as client:
        assert _edit(client, input_id).status_code == 404
    assert stub["backend"].calls == []


def test_the_worker_carries_the_edit():
    import queue
    import threading

    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Backend:
        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

    edit = {"mode": "words", "instructions": ["Replace 'human' with 'robot'."]}
    source = {"source": "/abs/source.wav"}
    for cmd in (
        {"text": EDITED, "workflow": "edit", "audio_inputs": source, "edit": edit},
        {"text": "hi"},
    ):
        _handle_generate_audio(
            _Backend(), {"request_id": "r", **cmd}, queue.Queue(), threading.Event()
        )
    assert seen[0]["edit"] == edit and seen[0]["audio_inputs"] == source
    assert "edit" not in seen[1]
