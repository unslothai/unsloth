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
        {"options": {"nested": {"a": 1}}},
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


def _alice_sources():
    input_id = _input(ALICE)
    clip = run_as(
        ALICE,
        audio_gallery.save,
        _wav(),
        {
            "prompt": "alice said",
            "model": "m",
            "audio_type": "audiocpp_tts",
            "sample_rate": 24000,
            "duration_s": 0.1,
            "created_at": "2026-10-02T00:00:00Z",
        },
    )
    voice = _voice(ALICE, input_id)
    return {"input_id": input_id, "clip_id": clip["id"], "voice_id": voice["id"]}


STABLE_AUDIO = "audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF"
ACE_STEP = "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF"


def _music_info(family = "stable_audio", folder = "Stable-Audio-3-Small-Music-GGUF"):
    from types import SimpleNamespace

    from core.inference import audio_cpp_music
    from core.inference.audio_cpp_models import family_policy

    policy = family_policy(family, None, (folder,))
    model = SimpleNamespace(music = policy.music, task = "music")
    return {
        "is_audio": True,
        "audio_type": "audiocpp_music",
        "audio_family": family,
        "audio_workflows": ["music"],
        "audio_reference_text": None,
        "audio_required_inputs": [],
        "audio_music": audio_cpp_music.music_rules(model),
        "audio_cpp_backend": "cuda",
    }


class _MusicBackend(_Backend):
    """Writes ``takes`` outputs and the manifest into the run folder, as the worker does."""

    def __init__(
        self,
        name,
        info,
        takes = 1,
    ):
        super().__init__(name, info)
        self.takes = takes
        self.run_dirs: list[Path] = []

    def generate_audio_response(self, **kwargs):
        self.calls.append(kwargs)
        out = Path(kwargs["output_dir"])
        self.run_dirs.append(out)
        manifest = []
        for index in range(self.takes):
            name = f"{index:02d}.wav"
            (out / name).write_bytes(_wav(rate = 44100, frames = 4410 * (index + 1)))
            manifest.append(
                {"id": f"audio_{index}", "file": name, "sample_rate": 44100, "seed": 40 + index}
            )
        (out / "outputs.json").write_text(json.dumps(manifest), encoding = "utf-8")
        return _wav(rate = 44100), 44100


def _music(client, **body):
    payload = {"workflow": "music", "mode": "song", "text": "uplifting house", **body}
    return client.post("/api/inference/audio/run", json = payload)


def test_music_variations_are_saved_as_one_group(stub, tmp_path):
    backend = stub["use"](STABLE_AUDIO, _music_info())
    backend.__class__ = _MusicBackend
    backend.takes, backend.run_dirs = 3, []
    with _client(ALICE) as client:
        response = _music(client, duration_s = 10, variations = 3, options = {"sampler": "euler"})
        assert response.status_code == 200, response.text
        listing = client.get("/api/inference/audio/gallery").json()
    body = response.json()
    assert len(body["clips"]) == 3 and body["group_id"]
    assert [c["role"] for c in body["clips"]] == ["variation"] * 3
    assert {c["workflow"] for c in body["clips"]} == {"music"}
    (call,) = backend.calls
    assert call["workflow"] == "music" and "audio_inputs" not in call
    music = call["music"]
    assert (music["mode"], music["duration_s"], music["variations"]) == ("song", 10.0, 3)
    assert music["timeout_s"] >= 300 and call["max_new_tokens"] >= 750
    (run_dir,) = backend.run_dirs
    inputs_root = tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    assert run_dir.parent == inputs_root / "runs" and not run_dir.exists()
    metas = [json.loads(_sidecar(tmp_path, c["id"])) for c in body["clips"]]
    assert {m["group_id"] for m in metas} == {body["group_id"]}
    assert [m["settings"]["variation"] for m in metas] == [1, 2, 3]
    assert [m["settings"]["seed"] for m in metas] == [40, 41, 42]
    assert metas[0]["settings"]["mode"] == "song" and metas[0]["settings"]["duration_s"] == 10.0
    assert metas[0]["settings"]["options"] == {"sampler": "euler"}
    items = {i["id"]: i for i in listing["audio"]}
    for clip in body["clips"]:
        assert items[clip["id"]]["group_id"] == body["group_id"]
        assert items[clip["id"]]["role"] == "variation"
        assert items[clip["id"]]["settings"]["mode"] == "song"
    assert str(tmp_path) not in json.dumps(listing)
    for clip in body["clips"]:
        sidecar = _sidecar(tmp_path, clip["id"])
        assert str(tmp_path) not in sidecar and "runs" not in sidecar


def test_a_single_take_is_an_output_with_no_group(stub, tmp_path):
    backend = stub["use"](STABLE_AUDIO, _music_info())
    backend.__class__ = _MusicBackend
    backend.takes, backend.run_dirs = 1, []
    with _client(ALICE) as client:
        response = _music(client)
    body = response.json()
    assert response.status_code == 200 and body["group_id"] is None
    assert [c["role"] for c in body["clips"]] == ["output"]
    assert backend.calls[0]["music"]["duration_s"] == 30.0


def test_a_manifest_entry_outside_the_run_folder_is_never_read(stub, tmp_path):
    backend = stub["use"](STABLE_AUDIO, _music_info())
    secret = tmp_path / "secret.wav"
    secret.write_bytes(_wav(rate = 8000))

    def escape(**kwargs):
        backend.calls.append(kwargs)
        out = Path(kwargs["output_dir"])
        manifest = [{"id": "x", "file": "../../../../secret.wav"}, {"id": "y", "file": str(secret)}]
        (out / "outputs.json").write_text(json.dumps(manifest), encoding = "utf-8")
        return _wav(rate = 44100), 44100

    backend.generate_audio_response = escape
    with _client(ALICE) as client:
        response = _music(client)
    body = response.json()
    assert response.status_code == 200 and len(body["clips"]) == 1
    assert body["clips"][0]["sample_rate"] == 44100


def _edit(
    client,
    source,
    action = "inpaint",
    ranges = None,
    **body,
):
    edit = {
        "action": action,
        "ranges": ranges if ranges is not None else [{"start_s": 0.02, "end_s": 0.08}],
    }
    edit.update(body.pop("edit", {}))
    return _music(
        client, mode = "edit", text = "add birds", inputs = {"source": source}, edit = edit, **body
    )


def test_an_edit_prepares_the_source_at_the_family_rate(stub, tmp_path):
    backend = stub["use"](STABLE_AUDIO, _music_info())
    backend.__class__ = _MusicBackend
    backend.takes, backend.run_dirs = 1, []
    sources = _alice_sources()
    with _client(ALICE) as client:
        response = _edit(client, {"clip_id": sources["clip_id"]})
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    source = Path(call["audio_inputs"]["source"])
    assert source.parent == tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    info = audio_inputs.wav_info(source)
    assert (info["sample_rate"], info["channels"]) == (44100, 2)
    clip = response.json()["clips"][0]
    assert clip["role"] == "edit"
    meta = json.loads(_sidecar(tmp_path, clip["id"]))
    assert meta["source_clip_id"] == sources["clip_id"]
    assert meta["settings"]["edit"]["action"] == "inpaint"
    assert meta["settings"]["edit"]["ranges"] == [{"start_s": 0.02, "end_s": 0.08}]


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"mode": "sfx"}, "does not make sound effects"),
        ({"variations": 5}, None),
        ({"inputs": {"reference": {"input_id": "a" * 32}}}, "not a voice reference"),
        ({"inputs": {"reference_text": "hi"}}, "not a voice reference"),
        ({"edit": {"action": "inpaint"}}, "apply to Edit only"),
        ({"mode": "edit"}, "Add a clip to edit"),
        ({"text": ""}, "needs a description"),
    ],
)
def test_music_refusals_name_the_fix(stub, body, detail):
    stub["use"](STABLE_AUDIO, _music_info())
    with _client(ALICE) as client:
        response = _music(client, **body)
    if detail is None:
        assert response.status_code == 422
    else:
        assert response.status_code == 400, response.text
        assert detail in response.json()["detail"]
    assert stub["backend"].calls == []


def test_edit_refusals_need_the_source(stub):
    stub["use"](STABLE_AUDIO, _music_info())
    sources = _alice_sources()
    clip = {"clip_id": sources["clip_id"]}  # 0.1 s long
    with _client(ALICE) as client:
        past = _edit(client, clip, ranges = [{"start_s": 0.2, "end_s": 0.4}])
        assert past.status_code == 400 and "after the clip ends" in past.json()["detail"]
        empty = _edit(client, clip, ranges = [])
        assert empty.status_code == 400 and "Select the part" in empty.json()["detail"]
        backwards = _edit(client, clip, ranges = [{"start_s": 0.05, "end_s": 0.01}])
        assert backwards.status_code == 422
        repaint = _edit(client, clip, action = "repaint")
        assert repaint.status_code == 400 and "cannot repaint" in repaint.json()["detail"]
        voice = _edit(client, {"voice_id": sources["voice_id"]})
        assert voice.status_code == 400 and "Saved voices" in voice.json()["detail"]
    assert stub["backend"].calls == []


def test_a_source_longer_than_the_model_can_return_is_refused(stub, monkeypatch):
    stub["use"](STABLE_AUDIO, _music_info())
    sources = _alice_sources()
    real = audio_inputs.wav_info
    monkeypatch.setattr(audio_inputs, "wav_info", lambda path: {**real(path), "duration_s": 121.0})
    with _client(ALICE) as client:
        response = _edit(client, {"clip_id": sources["clip_id"]})
    assert response.status_code == 400
    # Stable Audio Small returns at most ~120 s, so a longer edit would come back cut.
    assert response.json()["detail"] == "Edit clips up to 2 minutes. Trim it first."
    assert stub["backend"].calls == []


def test_ace_step_extend_past_the_song_limit_is_refused(stub):
    stub["use"](ACE_STEP, _music_info("ace_step", "ACE-Step1.5-GGUF"))
    sources = _alice_sources()
    with _client(ALICE) as client:
        response = _edit(
            client,
            {"clip_id": sources["clip_id"]},
            action = "extend",
            ranges = [],
            edit = {"extend_s": 300},
        )
    assert response.status_code == 400 and "Add fewer seconds" in response.json()["detail"]


def test_another_accounts_clip_is_404_for_an_edit(stub):
    stub["use"](STABLE_AUDIO, _music_info())
    sources = _alice_sources()
    with _client(BOB) as client:
        response = _edit(client, {"clip_id": sources["clip_id"]})
        upload = _edit(client, {"input_id": sources["input_id"]})
    assert response.status_code == 404 and upload.status_code == 404
    assert stub["backend"].calls == []


def test_music_on_a_speech_model_or_native_minimax_is_a_400(stub):
    with _client(ALICE) as client:
        response = _music(client)
    assert response.status_code == 400 and "Music studio" in response.json()["detail"]
    stub["use"](
        "MiniMax/Music3",
        {"is_audio": True, "audio_type": "minimax_music3", "audio_workflows": ["music"]},
    )
    with _client(ALICE) as client:
        native = _music(client, lyrics = "la")
    assert native.status_code == 400 and "Music studio" in native.json()["detail"]
    assert stub["backend"].calls == []


def test_music_fields_on_clone_or_speak_are_a_400(stub):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        for extra in (
            {"mode": "song"},
            {"lyrics": "la"},
            {"variations": 2},
            {"edit": {"action": "cover"}},
            {"inputs": {"reference": {"input_id": input_id}, "source": {"input_id": input_id}}},
        ):
            response = _run(client, **extra)
            assert response.status_code == 400, extra
            assert "Music only" in response.json()["detail"]
    assert stub["backend"].calls == []


def test_a_music_runtime_refusal_keeps_the_runtime_reason(stub):
    from core.inference.audio_errors import AudioRuntimeError

    backend = stub["use"](STABLE_AUDIO, _music_info())

    def refuse(**_kwargs):
        raise AudioRuntimeError(
            "The audio runtime could not generate audio: Stable Audio inpaint regions must be finite and ordered",
            status = 400,
        )

    backend.generate_audio_response = refuse
    with _client(ALICE) as client:
        response = _music(client)
    assert response.status_code == 400
    assert "inpaint regions must be finite" in response.json()["detail"]


def test_the_worker_and_orchestrator_carry_music_fields_only_when_present(monkeypatch):
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Backend2:
        patch = {"audio_music": {"modes": []}}

        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

        def take_status_patch(self):
            patch, self.patch = self.patch, None
            return patch

    responses: queue.Queue = queue.Queue()
    music = {"mode": "song", "variations": 3}
    backend = _Backend2()
    _handle_generate_audio(
        backend,
        {
            "request_id": "r1",
            "text": "hi",
            "workflow": "music",
            "music": music,
            "output_dir": "/x/runs/1",
        },
        responses,
        threading.Event(),
    )
    _handle_generate_audio(
        backend, {"request_id": "r2", "text": "hi"}, responses, threading.Event()
    )
    assert seen[0]["music"] == music and seen[0]["output_dir"] == "/x/runs/1"
    assert not {"music", "output_dir"} & set(seen[1])
    sent = []
    while not responses.empty():
        sent.append(responses.get())
    done = [m for m in sent if m.get("type") == "audio_done"]
    assert done[0]["status_patch"] == {"audio_music": {"modes": []}}
    assert "status_patch" not in done[1]


def test_the_orchestrator_sends_music_and_merges_the_reload_status(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []
    wav = base64.b64encode(_wav()).decode()
    reloaded = {"modes": [{"id": "song", "variations": {"max": 4, "how": "batch", "loaded": 4}}]}

    def read_one(*, timeout):
        return {
            "type": "audio_done",
            "request_id": sent[-1]["request_id"],
            "wav_base64": wav,
            "sample_rate": 24000,
            "status_patch": {"audio_music": reloaded, "is_audio": False},
        }

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    for name, value in {
        "_ensure_subprocess_alive": lambda: True,
        "_send_cmd": lambda cmd: sent.append(cmd),
        "_direct_reader": lambda _rid, _cancel = None: (read_one, lambda **_k: None, lambda: None),
        "_reserve_worker": lambda _why: _Null(),
        "_wait_worker_idle": lambda **_k: True,
        "_claim_worker": lambda _cancel: None,
        "_release_worker": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(orchestrator, name, value, raising = False)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"is_audio": True, "audio_music": {"modes": []}}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    music = {"mode": "song", "variations": 3, "timeout_s": 900.0}
    orchestrator.generate_audio_response(
        "", workflow = "music", music = music, output_dir = "/acct/audio/inputs/runs/abc"
    )
    (cmd,) = sent
    assert cmd["music"] == music and cmd["output_dir"] == "/acct/audio/inputs/runs/abc"
    assert orchestrator.models["m"] == {"is_audio": True, "audio_music": reloaded}


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
