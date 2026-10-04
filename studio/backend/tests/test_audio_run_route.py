# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/run: ids in, server-chosen account paths out, never a client path or audio bytes."""

from __future__ import annotations

import asyncio
import contextlib
import base64
import io
import json
import queue
import sys
import threading
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import audio_gallery, audio_inputs, audio_voices
from routes import inference
from utils.account_context import bind_account, reset_account, run_as

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_account_media_isolation import ALICE, BOB, isolated  # noqa: E402,F401
from test_audio_inputs import _chunks, encode  # noqa: E402

QWEN3_BASE = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF"
CLONE_FIELDS = {"workflow", "audio_inputs", "reference_text", "speed"}


def _wav(rate = 24000, frames = 2400) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


def _clone_info(
    workflows = ("clone",),
    reference_text = "required",
    emotion = False,
    waived = None,
):
    return {
        "is_audio": True,
        "audio_type": "audiocpp_tts",
        "audio_family": "qwen3_tts",
        "audio_workflows": list(workflows),
        "audio_reference_text": reference_text,
        "audio_required_inputs": [],
        "audio_clone": {
            "reference_text": reference_text,
            "reference_text_waived": waived or [["x_vector_only_mode", ["true"]]],
            "emotion_audio": emotion,
        },
    }


def _speak_info(required = ()):
    return {
        "is_audio": True,
        "audio_type": "audiocpp_tts",
        "audio_workflows": ["speak"],
        "audio_reference_text": None,
        "audio_required_inputs": list(required),
    }


class _Backend:
    def __init__(self, name, info):
        self.active_model_name = name
        self.models = {name: info}
        self.calls: list[dict] = []

    def generate_audio_response(self, **kwargs):
        self.calls.append(kwargs)
        return _wav(), 24000


@pytest.fixture
def stub(monkeypatch):
    class _Llama:
        is_loaded = False
        _is_audio = False

    async def _noop_switch(*_args, **_kwargs):
        return None

    holder = {"backend": _Backend(QWEN3_BASE, _clone_info())}
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference, "get_inference_backend", lambda: holder["backend"])
    monkeypatch.setattr(inference, "_maybe_auto_switch_model", _noop_switch)

    def use(name, info):
        holder["backend"] = _Backend(name, info)
        return holder["backend"]

    holder["use"] = use
    return holder


def _client(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


def _input(account, seconds = 1.0) -> str:
    data = encode("webm", "libopus", 48000, "stereo", seconds)
    save = lambda: asyncio.run(audio_inputs.save_stream(_chunks([data]), "me.webm"))[0]["id"]
    return run_as(account, save)


def _voice(account, input_id, **meta):
    path = audio_inputs.input_path
    return run_as(account, lambda: audio_voices.create(path(input_id), {"name": "Alice", **meta}))


def _run(client, **body):
    payload = {"workflow": "clone", "text": "Hello in my voice.", **body}
    return client.post("/api/inference/audio/run", json = payload)


def _sidecar(tmp_path, clip_id) -> str:
    path = tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip_id}.json"
    return path.read_text(encoding = "utf-8")


def _inputs_root(tmp_path):
    return tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"


def test_a_clone_run_hands_the_worker_an_account_path_and_saves_the_clip(stub, tmp_path):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _run(
            client,
            language = "en",
            inputs = {"reference": {"input_id": input_id}, "reference_text": "Okay, I'm Cemo."},
            options = {"x_vector_only_mode": False},
            seed = 7,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert client.get(body["clips"][0]["url"]).status_code == 200
    (call,) = stub["backend"].calls
    reference = Path(call["audio_inputs"]["reference"])
    assert reference.parent == _inputs_root(tmp_path)
    assert reference.name.startswith(f"{input_id}.24000.mono")
    with wave.open(str(reference)) as w:
        assert (w.getframerate(), w.getnchannels()) == (24000, 1)
    assert (call["workflow"], call["reference_text"], call["language"], call["seed"]) == (
        "clone",
        "Okay, I'm Cemo.",
        "en",
        7,
    )
    assert "speed" not in call
    assert all(not isinstance(v, (bytes, bytearray)) for v in call.values())
    assert body["audio"] is None
    (clip,) = body["clips"]
    assert set(clip) == {"id", "role", "url", "sample_rate", "duration_s", "workflow"}
    assert (clip["role"], clip["workflow"], clip["sample_rate"]) == ("output", "clone", 24000)
    sidecar = _sidecar(tmp_path, clip["id"])
    meta = json.loads(sidecar)
    assert meta["reference_name"] == "me.webm"
    assert meta["settings"] == {
        "language": "en",
        "instructions": None,
        "options": {"x_vector_only_mode": False},
        "reference_text_used": True,
        "speed": None,
    }
    assert str(tmp_path) not in sidecar and "inputs" not in sidecar


@pytest.mark.parametrize(
    "body",
    [
        {"path": "/etc/passwd"},
        {"voice_ref": "/etc/passwd"},
        {"file": "x.wav"},
        {"url": "http://x"},
        {"model": "other/model"},
        {"inputs": {"reference": {"path": "/etc/passwd"}}},
        {"inputs": {"reference": {"input_id": "a" * 32, "voice_ref": "/etc/passwd"}}},
        {"inputs": {"reference": {"input_id": "a" * 32}, "voice_ref": "/x.wav"}},
        {"inputs": {"reference": {"input_id": "a" * 32, "clip_id": "b" * 32}}},
        {"inputs": {"reference": {}}},
        {"inputs": {"reference": {"input_id": "../../etc/passwd"}}},
        {"inputs": {"reference": {"input_id": "a" * 32}, "emotion": {"url": "http://x"}}},
        {"options": {"voice_ref": "/etc/passwd"}},
        {"options": {"source_audio": "/etc/passwd"}},
        {"options": {"codec_model_path": "/etc/passwd"}},
        {"options": {"video": "/etc/passwd"}},  # ControlFoley
        {"options": {"reference_image": "/etc/passwd"}},
        {"options": {"nested": {"a": 1}}},
        {"workflow": "music"},
        {"workflow": "transcribe"},
        {"text": ""},
    ],
)
def test_client_paths_and_unknown_fields_are_422(stub, body):
    with _client(ALICE) as client:
        assert _run(client, **body).status_code == 422
    assert stub["backend"].calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"min_new_audio_steps": 10, "max_new_audio_steps": 900},  # FireRedAudio
        {"no_ref": True},  # Irodori
        {"audio_chunk_threshold_sec": 30, "audio_chunk_duration_sec": 20},  # DramaBox
        {"use_video": True},
    ],
)
def test_settings_named_after_audio_are_not_file_options(options):
    from models.inference import AudioRunRequest
    request = AudioRunRequest(workflow = "speak", text = "hi", options = options)
    assert request.options == options


@pytest.mark.parametrize("kind", ["input_id", "clip_id", "voice_id"])
def test_another_accounts_ids_are_404(stub, kind):
    input_id = _input(ALICE)
    meta = {
        "prompt": "alice said",
        "model": "m",
        "audio_type": "audiocpp_tts",
        "sample_rate": 24000,
        "duration_s": 0.1,
        "created_at": "2026-10-02T00:00:00Z",
    }
    ids = {
        "input_id": input_id,
        "clip_id": run_as(ALICE, audio_gallery.save, _wav(), meta)["id"],
        "voice_id": _voice(ALICE, input_id)["id"],
    }
    source = {kind: ids[kind]}
    with _client(BOB) as client:
        assert _run(client, inputs = {"reference": source, "reference_text": "hi"}).status_code == 404
        emotion = {"reference": source, "emotion": source, "reference_text": "hi"}
        assert _run(client, inputs = emotion).status_code == 404
    assert stub["backend"].calls == []
    with _client(ALICE) as client:
        assert _run(client, inputs = {"reference": source, "reference_text": "hi"}).status_code == 200


@pytest.mark.parametrize(
    "model, info, workflow, transcript",
    [
        (QWEN3_BASE, _clone_info(), "clone", "Okay, I'm Cemo."),
        (
            "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
            _clone_info(workflows = ("speak", "clone"), reference_text = "optional"),
            "speak",
            None,
        ),
    ],
)
def test_a_saved_voice_resolves_to_an_account_path_and_is_recorded(
    stub, tmp_path, model, info, workflow, transcript
):
    backend = stub["use"](model, info)
    voice = _voice(ALICE, _input(ALICE), **({"transcript": transcript} if transcript else {}))
    with _client(ALICE) as client:
        response = _run(client, workflow = workflow, inputs = {"reference": {"voice_id": voice["id"]}})
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    assert call["workflow"] == workflow and call.get("reference_text") == transcript
    reference = Path(call["audio_inputs"]["reference"])
    assert reference.parent == _inputs_root(tmp_path)
    assert reference.name.startswith(f"v-{voice['id']}.")
    meta = json.loads(_sidecar(tmp_path, response.json()["clips"][0]["id"]))
    assert (meta["voice_id"], meta["workflow"], meta["reference_name"]) == (
        voice["id"],
        workflow,
        "Alice",
    )


def test_a_cleared_transcript_is_not_refilled_from_the_saved_voice(stub):
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
        _clone_info(workflows = ("speak", "clone"), reference_text = "optional"),
    )
    voice = _voice(ALICE, _input(ALICE), transcript = "Okay, I'm Cemo.")
    with _client(ALICE) as client:
        response = _run(
            client, inputs = {"reference": {"voice_id": voice["id"]}, "reference_text": ""}
        )
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    assert call.get("reference_text") is None


_COSY = _clone_info(waived = [["template_name", ["cross_lingual", "instruct"]]])


@pytest.mark.parametrize(
    "info, options, status",
    [
        (_clone_info(), None, 400),
        (_clone_info(), {"x_vector_only_mode": True}, 200),
        (_clone_info(reference_text = "unused"), None, 200),
        (_COSY, None, 400),
        (_COSY, {"template_name": "zero_shot"}, 400),
        (_COSY, {"template_name": "cross_lingual"}, 200),
        (_COSY, {"template_name": "instruct"}, 200),
    ],
)
def test_a_missing_transcript_is_a_400_unless_the_family_waives_it(stub, info, options, status):
    stub["use"]("x/Model-GGUF", info)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"input_id": input_id}}, options = options)
    assert response.status_code == status, response.text
    if status == 400:
        assert response.json()["detail"] == "Type what's said in the reference clip."


_GENERATE = ("/api/inference/audio/generate", {"messages": [{"role": "user", "content": "hi"}]})
_SPEECH = ("/api/inference/audio/speech", {"input": "hi"})
_KOKORO = ("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", _speak_info())
_MAYA1 = ("audio-cpp/audio.cpp-gguf/Maya1-GGUF", _speak_info(["instruct"]))
_REF = {"reference": {"input_id": "REF"}, "reference_text": "x"}
_MAYA1_DETAIL = "Maya1 needs a voice description."


@pytest.mark.parametrize(
    "model, request_, detail",
    [
        (_KOKORO, {"inputs": _REF}, "Load a model that can clone a voice."),
        (_KOKORO, {"workflow": "speak", "inputs": _REF}, "Load a model that can clone a voice."),
        (_KOKORO, {"workflow": "speak"}, None),
        (None, {}, "Add a reference clip to clone."),
        (None, {"inputs": {**_REF, "emotion": {"input_id": "REF"}}}, "does not take an emotion"),
        (None, _GENERATE, "Open Clone"),
        (_MAYA1, {"workflow": "speak"}, _MAYA1_DETAIL),
        (_MAYA1, _GENERATE, _MAYA1_DETAIL),
        (_MAYA1, _SPEECH, _MAYA1_DETAIL),
        (_MAYA1, {"workflow": "speak", "instructions": "A calm, low male voice."}, None),
    ],
)
def test_requests_the_loaded_model_cannot_serve_are_400(stub, model, request_, detail):
    if model:
        stub["use"](*model)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        if isinstance(request_, tuple):
            response = client.post(request_[0], json = request_[1])
        else:
            body = json.loads(json.dumps(request_).replace('"REF"', json.dumps(input_id)))
            response = _run(client, **body)
    if detail is None:
        assert response.status_code == 200 and len(stub["backend"].calls) == 1
    else:
        assert response.status_code == 400 and detail in response.json()["detail"]
        assert stub["backend"].calls == []


def test_an_emotion_clip_reaches_a_family_that_takes_one(stub):
    first, second = _input(ALICE, 0.5), _input(ALICE, 0.7)
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/IndexTTS2-GGUF",
        _clone_info(reference_text = "unused", emotion = True),
    )
    with _client(ALICE) as client:
        ok = _run(
            client, inputs = {"reference": {"input_id": first}, "emotion": {"input_id": second}}
        )
    assert ok.status_code == 200, ok.text
    (call,) = backend.calls
    assert set(call["audio_inputs"]) == {"reference", "emotion"}
    assert Path(call["audio_inputs"]["emotion"]).name.startswith(f"{second}.24000.mono")


def test_a_failed_gallery_save_returns_the_audio_inline(stub, monkeypatch):
    input_id = _input(ALICE)
    monkeypatch.setattr(inference, "_persist_tts_clip", lambda *a, **k: None)
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"input_id": input_id}, "reference_text": "x"})
    body = response.json()
    assert response.status_code == 200 and body["clips"] == []
    assert body["audio"]["sample_rate"] == 24000
    assert base64.b64decode(body["audio"]["data"])[:4] == b"RIFF"


def test_the_orchestrator_command_carries_paths_not_bytes(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []
    wav = base64.b64encode(_wav()).decode()

    def read_one(*, timeout):
        rid = sent[-1]["request_id"]
        return {"type": "audio_done", "request_id": rid, "wav_base64": wav, "sample_rate": 24000}

    for name, value in {
        "_ensure_subprocess_alive": lambda: True,
        "_send_cmd": sent.append,
        "_direct_reader": lambda _rid, _cancel = None: (read_one, lambda **_k: None, lambda: None),
        "_reserve_worker": lambda _why: contextlib.nullcontext(),
        "_wait_worker_idle": lambda **_k: True,
        "_claim_worker": lambda _cancel: None,
        "_release_worker": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(orchestrator, name, value, raising = False)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    reference = "/home/x/.unsloth/studio/accounts/a/audio/inputs/abc.24000.mono.m30.wav"
    fields = {
        "workflow": "clone",
        "audio_inputs": {"reference": reference},
        "reference_text": "hi",
        "speed": 1.2,
    }
    orchestrator.generate_audio_response("hello", **fields)
    orchestrator.generate_audio_response("plain")
    assert {k: sent[0][k] for k in fields} == fields
    assert all(not isinstance(v, (bytes, bytearray)) for v in sent[0].values())
    assert not CLONE_FIELDS & set(sent[1])


def test_the_worker_forwards_clone_fields_only_when_present():
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Recorder:
        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

    fields = {
        "workflow": "clone",
        "audio_inputs": {"reference": "/abs/ref.wav"},
        "reference_text": "x",
        "speed": 1.1,
    }
    for cmd in ({"request_id": "r1", "text": "hi", **fields}, {"request_id": "r2", "text": "hi"}):
        _handle_generate_audio(_Recorder(), cmd, queue.Queue(), threading.Event())
    assert {k: seen[0][k] for k in fields} == fields
    assert not CLONE_FIELDS & set(seen[1])


HTDEMUCS_6 = "audio-cpp/audio.cpp-gguf/HTDemucs-6stems-GGUF"
BS_ROFORMER = "audio-cpp/audio.cpp-gguf/BS-RoFormer-ep368-GGUF"
SIX_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def _sep_info(family = "htdemucs_6stems"):
    return {
        "is_audio": True,
        "audio_type": "audiocpp_sep",
        "audio_family": family,
        "audio_workflows": ["separate"],
        "audio_options": [],
        "audio_clone": None,
    }


def _stem(
    path,
    stem = "vocals",
    data = None,
    duration_s = 0.01,
) -> dict:
    """A worker stem entry; writes ``data`` (a WAV by default) to ``path``."""
    Path(path).write_bytes(_wav() if data is None else data)
    return {
        "id": stem,
        "path": str(path),
        "sample_rate": 44100,
        "channels": 2,
        "duration_s": duration_s,
    }


class _SepBackend(_Backend):
    def __init__(
        self,
        name,
        info,
        stems = SIX_STEMS,
    ):
        super().__init__(name, info)
        self.stems = stems
        self.separations: list[dict] = []
        self.outputs = None

    def separate_audio_response(
        self,
        source_path,
        output_dir,
        audio_options = None,
        cancel_event = None,
    ):
        self.separations.append(
            {"source_path": source_path, "output_dir": output_dir, "audio_options": audio_options}
        )
        if self.outputs is not None:
            return self.outputs(output_dir)
        return [_stem(Path(output_dir) / f"{s}.wav", s) for s in self.stems]


@pytest.fixture
def sep(stub):
    backend = _SepBackend(HTDEMUCS_6, _sep_info())
    stub["backend"] = backend
    return backend


def _separate(
    client,
    source = None,
    **body,
):
    if source is not None:
        body["inputs"] = {"source": source}
    return client.post("/api/inference/audio/run", json = {"workflow": "separate", **body})


def _wav_input(
    account,
    seconds,
    rate = 8000,
    layout = "mono",
) -> str:
    data = encode("wav", "pcm_s16le", rate, layout, seconds)
    save = lambda: asyncio.run(audio_inputs.save_stream(_chunks([data]), "track.wav"))[0]["id"]
    return run_as(account, save)


def _gallery_root(tmp_path) -> Path:
    return tmp_path / "accounts" / ALICE.account_id / "audio"


def _wav_info(path) -> tuple:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def _alice_sources() -> dict:
    """An upload and a history clip in Alice's account."""
    meta = {
        "prompt": "alice said",
        "model": "m",
        "audio_type": "audiocpp_tts",
        "sample_rate": 24000,
        "duration_s": 0.1,
        "created_at": "2026-10-02T00:00:00Z",
    }
    clip = run_as(ALICE, audio_gallery.save, _wav(), meta)
    return {"input_id": _input(ALICE), "clip_id": clip["id"]}


def _staging_and_stems(tmp_path):
    root = _gallery_root(tmp_path)
    return list(root.glob(".separate-*")), sorted(p.name for p in root.glob("*.wav"))


def test_a_separation_prepares_44k_and_saves_every_stem_as_one_group(sep, tmp_path):
    input_id = _input(ALICE, 2.0)  # 48 kHz stereo
    with _client(ALICE) as client:
        response = _separate(client, {"input_id": input_id}, seed = 3)
        assert response.status_code == 200, response.text
        body = response.json()
        stem_bytes = client.get(body["clips"][0]["url"])
        listed = client.get("/api/inference/audio/gallery").text
    (call,) = sep.separations
    assert sep.calls == []
    source = Path(call["source_path"])
    assert source.is_absolute() and source.parent == _gallery_root(tmp_path) / "inputs"
    assert source.name.startswith(f"{input_id}.44100.stereo")
    rate, channels, frames = _wav_info(source)
    assert (rate, channels) == (44100, 2) and abs(frames - 2 * 44100) <= 2
    staging = Path(call["output_dir"])
    assert staging.parent == _gallery_root(tmp_path) and staging.name.startswith(".separate-")
    assert not staging.exists()
    assert call["audio_options"] is None
    order = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    assert [clip["role"] for clip in body["clips"]] == order
    assert body["audio"] is None and body["model"] and len(body["group_id"]) == 32
    assert stem_bytes.status_code == 200 and stem_bytes.content[:4] == b"RIFF"
    for clip in body["clips"]:
        assert set(clip) == {"id", "role", "url", "sample_rate", "duration_s", "workflow"}
        assert (clip["workflow"], clip["sample_rate"], clip["duration_s"]) == (
            "separate",
            44100,
            0.01,
        )
        sidecar = (_gallery_root(tmp_path) / f"{clip['id']}.json").read_text(encoding = "utf-8")
        meta = json.loads(sidecar)
        assert meta["workflow"] == "separate" and meta["audio_type"] == "audiocpp_sep"
        assert meta["role"] == clip["role"] and meta["group_id"] == body["group_id"]
        assert meta["settings"] == {"stems": order, "num_overlap": None}
        assert meta["prompt"] == "me.webm"
        assert not any(isinstance(v, str) and v.startswith("/") for v in meta.values())
        assert str(tmp_path) not in sidecar and ".separate-" not in sidecar
    for text in (response.text, listed):
        assert str(tmp_path) not in text and ".separate-" not in text
    # The listing keeps the group, so history shows the run as one item.
    stems = [c for c in json.loads(listed)["audio"] if c["workflow"] == "separate"]
    assert {c["group_id"] for c in stems} == {body["group_id"]}
    assert sorted(c["role"] for c in stems) == sorted(SIX_STEMS)


def test_a_mono_track_reaches_the_runtime_as_44k_mono(sep):
    input_id = _wav_input(ALICE, 1.0, rate = 22050, layout = "mono")
    with _client(ALICE) as client:
        assert _separate(client, {"input_id": input_id}).status_code == 200
    rate, channels, frames = _wav_info(sep.separations[0]["source_path"])
    assert (rate, channels) == (44100, 1) and abs(frames - 44100) <= 1


def test_a_history_clip_can_be_separated_and_is_recorded(sep, tmp_path):
    clip_id = _alice_sources()["clip_id"]
    with _client(ALICE) as client:
        response = _separate(client, {"clip_id": clip_id})
    assert response.status_code == 200, response.text
    sidecar = _gallery_root(tmp_path) / f"{response.json()['clips'][0]['id']}.json"
    meta = json.loads(sidecar.read_text(encoding = "utf-8"))
    assert meta["source_clip_id"] == clip_id and meta["prompt"] == "alice said"


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"text": "hello"}, "Separate takes no text."),
        ({}, "Add a track to separate."),
        (
            {"inputs": {"source": {"voice_id": "a" * 32}}},
            "Pick a track to separate, not a saved voice.",
        ),
        (
            {"inputs": {"source": {"input_id": "a" * 32}, "reference": {"input_id": "a" * 32}}},
            "Separate takes only a track.",
        ),
    ],
)
def test_a_separation_request_without_a_track_is_a_400(sep, body, detail):
    with _client(ALICE) as client:
        response = _separate(client, **body)
    assert (response.status_code, response.json()["detail"]) == (400, detail)
    assert sep.separations == []


def test_a_clone_or_speak_run_still_needs_text(stub):
    with _client(ALICE) as client:
        response = client.post("/api/inference/audio/run", json = {"workflow": "speak"})
    assert response.status_code == 422
    assert stub["backend"].calls == []


def test_separation_needs_a_separation_model(stub):
    input_id = _input(ALICE)
    stub["use"](*_KOKORO)
    with _client(ALICE) as client:
        response = _separate(client, {"input_id": input_id})
    assert (response.status_code, response.json()["detail"]) == (
        400,
        "Load a model that can separate audio.",
    )
    assert stub["backend"].calls == []


def test_options_only_for_a_family_that_takes_them(sep, stub):
    source = {"input_id": _input(ALICE)}
    with _client(ALICE) as client:
        refused = _separate(client, source, options = {"num_overlap": 1})
        assert (refused.status_code, refused.json()["detail"]) == (
            400,
            "HTDemucs-6stems has no separation options.",
        )
    roformer = _SepBackend(BS_ROFORMER, _sep_info("bs_roformer"), ["vocals", "instrumental"])
    stub["backend"] = roformer
    with _client(ALICE) as client:
        unknown = _separate(client, source, options = {"shifts": 2})
        assert (unknown.status_code, unknown.json()["detail"]) == (400, "Unknown option 'shifts'.")
        assert _separate(client, source, options = {"num_overlap": 9}).status_code == 400
        ok = _separate(client, source, options = {"num_overlap": 1})
    assert ok.status_code == 200, ok.text
    (call,) = roformer.separations
    assert call["audio_options"] == {"num_overlap": 1}
    assert [clip["role"] for clip in ok.json()["clips"]] == ["vocals", "instrumental"]
    assert sep.separations == []


def test_a_track_over_ten_minutes_is_refused(sep):
    input_id = _wav_input(ALICE, 601.0)
    with _client(ALICE) as client:
        long = _separate(client, {"input_id": input_id})
        assert (long.status_code, long.json()["detail"]) == (
            400,
            "Separate tracks up to 10 minutes long. This one is 10:01. Pick a shorter track.",
        )
        ten = _separate(client, {"input_id": _wav_input(ALICE, 600.0)})
    assert ten.status_code == 200, ten.text
    (call,) = sep.separations
    assert abs(_wav_info(call["source_path"])[2] - 600 * 44100) <= 600


def test_another_accounts_track_is_404(sep):
    sources = _alice_sources()
    with _client(BOB) as client:
        for kind in ("input_id", "clip_id"):
            response = _separate(client, {kind: sources[kind]})
            assert response.status_code == 404, response.text
        expired = _separate(client, {"input_id": "c" * 32})
        assert expired.json()["detail"] == "This track expired. Add it again."
    assert sep.separations == []


def test_another_account_cannot_fetch_a_stem(sep):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        url = _separate(client, {"input_id": input_id}).json()["clips"][0]["url"]
        assert client.get(url).status_code == 200
    with _client(BOB) as client:
        assert client.get(url).status_code == 404


@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param(lambda tmp, out: [], id = "no_stems"),
        pytest.param(lambda tmp, out: [_stem(tmp / "elsewhere.wav")], id = "outside_staging"),
        pytest.param(lambda tmp, out: [_stem(Path(out) / "v.wav", data = b"junk")], id = "not_a_wav"),
    ],
)
def test_unusable_stems_are_a_502(sep, tmp_path, outputs):
    sep.outputs = lambda out: outputs(tmp_path, out)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _separate(client, {"input_id": input_id})
    assert (response.status_code, response.json()["detail"]) == (
        502,
        "The audio runtime returned no stems.",
    )
    assert _staging_and_stems(tmp_path) == ([], [])


def test_a_cancel_is_a_499_and_leaves_no_staging(sep, tmp_path):
    from core.inference.audio_errors import AudioGenerationCancelledError

    def cancelled(out):
        _stem(Path(out) / "drums.wav")
        raise AudioGenerationCancelledError("Audio generation cancelled")

    sep.outputs = cancelled
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        assert _separate(client, {"input_id": input_id}).status_code == 499
    assert _staging_and_stems(tmp_path) == ([], [])


def test_a_failed_save_rolls_back_the_whole_group(sep, tmp_path, monkeypatch):
    real = audio_gallery.save_file
    saved = []

    def flaky(
        src,
        meta,
        *,
        prune = True,
    ):
        assert prune is False
        if len(saved) == 2:
            raise OSError(28, "No space left on device")
        record = real(src, meta, prune = prune)
        saved.append(record["id"])
        return record

    monkeypatch.setattr(audio_gallery, "save_file", flaky)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _separate(client, {"input_id": input_id})
    assert (response.status_code, response.json()["detail"]) == (
        500,
        "Could not save the stems to history: [Errno 28] No space left on device",
    )
    assert len(saved) == 2
    assert _staging_and_stems(tmp_path) == ([], [])
    assert not list(_gallery_root(tmp_path).glob("*.json"))


def test_a_separation_model_refuses_speech_and_points_at_separate(sep):
    with _client(ALICE) as client:
        generate = client.post(_GENERATE[0], json = _GENERATE[1])
        speech = client.post(_SPEECH[0], json = _SPEECH[1])
        speak = _run(client, workflow = "speak")
    assert (generate.status_code, generate.json()["detail"]) == (
        400,
        "HTDemucs-6stems separates audio into stems. Open Audio, then Separate.",
    )
    for response in (speech, speak):
        assert response.status_code == 400 and "Separate" in response.json()["detail"]
    assert sep.calls == [] and sep.separations == []


def test_the_separate_scope_clears_only_separations(sep, tmp_path):
    _alice_sources()
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        assert _separate(client, {"input_id": input_id}).status_code == 200
        cleared = client.delete("/api/inference/audio/gallery", params = {"workflow": "separate"})
    assert cleared.status_code == 200 and cleared.json()["removed"] == 6
    assert len(_staging_and_stems(tmp_path)[1]) == 1


def _orchestrator(monkeypatch, reply):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []

    def read_one(*, timeout):
        return {"type": "audio_done", "request_id": sent[-1]["request_id"], **reply}

    for name, value in {
        "_ensure_subprocess_alive": lambda: True,
        "_send_cmd": sent.append,
        "_direct_reader": lambda _rid, _cancel = None: (read_one, lambda **_k: None, lambda: None),
        "_reserve_worker": lambda _why: contextlib.nullcontext(),
        "_wait_worker_idle": lambda **_k: True,
        "_claim_worker": lambda _cancel: None,
        "_release_worker": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(orchestrator, name, value, raising = False)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"audio_type": "audiocpp_sep"}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    return orchestrator, sent


_VOCALS = {
    "id": "vocals",
    "path": "/x/.separate-1/vocals.wav",
    "sample_rate": 44100,
    "channels": 2,
    "duration_s": 1.0,
}


def test_the_orchestrator_sends_paths_and_returns_the_stem_list(monkeypatch):
    from core.inference.orchestrator import _audio_generation_timeout

    orchestrator, sent = _orchestrator(monkeypatch, {"outputs": [_VOCALS], "sample_rate": 44100})
    result = orchestrator.separate_audio_response(
        "/x/inputs/a.44100.stereo.wav", "/x/.separate-1", {"num_overlap": 1}
    )
    assert result == [_VOCALS]
    (cmd,) = sent
    assert cmd["workflow"] == "separate" and cmd["text"] == ""
    assert cmd["audio_inputs"] == {"source": "/x/inputs/a.44100.stereo.wav"}
    assert cmd["output_dir"] == "/x/.separate-1"
    assert cmd["audio_options"] == {"num_overlap": 1}
    assert cmd["max_new_tokens"] == 8192 and "seed" not in cmd
    assert _audio_generation_timeout(8192) == 3600.0
    orchestrator, sent = _orchestrator(
        monkeypatch, {"wav_base64": base64.b64encode(_wav()).decode(), "sample_rate": 24000}
    )
    orchestrator.generate_audio_response("plain")
    assert "output_dir" not in sent[0]


def test_the_worker_separates_into_paths_not_bytes():
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Separator:
        def separate_audio(self, **kwargs):
            seen.append(kwargs)
            return [_VOCALS]

        def generate_audio_response(self, **kwargs):
            raise AssertionError("a separation is not speech")

    class _Speech:
        def generate_audio_response(self, **kwargs):
            return _wav(), 24000

    responses: queue.Queue = queue.Queue()
    cancel = threading.Event()
    cmd = {
        "request_id": "r1",
        "text": "",
        "workflow": "separate",
        "audio_inputs": {"source": "/abs/in.wav"},
        "output_dir": "/x/.separate-1",
        "audio_options": {"num_overlap": 1},
    }
    _handle_generate_audio(_Separator(), cmd, responses, cancel)
    reply = responses.get_nowait()
    assert reply["type"] == "audio_done" and "wav_base64" not in reply
    assert reply["outputs"] == [_VOCALS]
    assert seen == [
        {
            "source_path": "/abs/in.wav",
            "output_dir": "/x/.separate-1",
            "options": {"num_overlap": 1},
            "cancel_event": cancel,
        }
    ]
    _handle_generate_audio(_Speech(), {**cmd, "request_id": "r2"}, responses, threading.Event())
    refused = responses.get_nowait()
    assert refused["type"] == "audio_error" and refused["code"] == "audio_unsupported_backend"


def test_the_worker_keeps_run_fields_off_a_backend_without_them():
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Native:
        def generate_audio_response(
            self,
            text,
            temperature = 0.6,
            top_p = 0.95,
            top_k = 50,
            min_p = 0.0,
            max_new_tokens = 2048,
            repetition_penalty = 1.0,
            use_adapter = None,
            cancel_event = None,
            instructions = None,
            language = None,
            seed = None,
        ):
            seen.append(text)
            return _wav(), 24000

    replies = queue.Queue()
    speak = {"request_id": "r1", "text": "hi", "workflow": "speak", "speed": 1.1}
    _handle_generate_audio(_Native(), speak, replies, threading.Event())
    assert replies.get_nowait()["type"] == "audio_done" and seen == ["hi"]
    clone = {"request_id": "r2", "text": "hi", "audio_inputs": {"reference": "/abs/ref.wav"}}
    _handle_generate_audio(_Native(), clone, replies, threading.Event())
    error = replies.get_nowait()
    assert error["type"] == "audio_error" and error["status"] == 400
    assert seen == ["hi"]


def test_a_float_wav_separation_source_is_probed_through_ffmpeg(tmp_path):
    import struct

    import numpy as np

    from routes.inference import _probe_audio_with_av

    frames = np.zeros((22050, 2), dtype = np.float32).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, 44100, 44100 * 8, 8, 32)
    path = tmp_path / "stem.wav"
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(frames))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", len(frames))
        + frames
    )
    channels, seconds = _probe_audio_with_av(path)
    assert channels == 2 and abs(seconds - 0.5) < 0.01
    (tmp_path / "junk.wav").write_bytes(b"not audio")
    assert _probe_audio_with_av(tmp_path / "junk.wav") is None
