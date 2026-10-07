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
    sent.clear()
    edit = {"mode": "words", "instructions": ["Replace 'human' with 'robot'."]}
    orchestrator.generate_audio_response(
        "edited", workflow = "edit", audio_inputs = {"source": reference}, edit = edit
    )
    (cmd,) = sent
    assert (cmd["workflow"], cmd["audio_inputs"], cmd["edit"]) == (
        "edit",
        {"source": reference},
        edit,
    )


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
    # Stable Audio always plays instrumental and ignores lyrics; history says what ran.
    assert (metas[0]["settings"]["instrumental"], metas[0]["settings"]["lyrics"]) == (True, None)
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


def test_ace_step_repaint_reaches_at_most_30_s_past_the_end(stub):
    stub["use"](ACE_STEP, _music_info("ace_step", "ACE-Step1.5-GGUF"))
    sources = _alice_sources()
    with _client(ALICE) as client:
        response = _edit(
            client,
            {"clip_id": sources["clip_id"]},
            action = "repaint",
            ranges = [{"start_s": 0.02, "end_s": 40.0}],
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "A repaint can reach at most 30 s past the end."


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
        (
            {"inputs": {"source": {"input_id": "a" * 32}}, "mode": "song"},
            "A music mode, lyrics, variations and edits apply to Music only.",
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


def test_speak_in_a_saved_voice_on_a_speak_and_clone_model(stub, tmp_path):
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
        _clone_info(workflows = ("speak", "clone"), reference_text = "optional"),
    )
    input_id = _input(ALICE)
    voice = run_as(
        ALICE, lambda: audio_voices.create(audio_inputs.input_path(input_id), {"name": "Mine"})
    )
    with _client(ALICE) as client:
        response = _run(client, workflow = "speak", inputs = {"reference": {"voice_id": voice["id"]}})
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    assert call["workflow"] == "speak"
    reference = Path(call["audio_inputs"]["reference"])
    assert reference.parent == tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    clip = response.json()["clips"][0]
    assert clip["workflow"] == "speak"
    meta = json.loads(
        (tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip['id']}.json").read_text(
            encoding = "utf-8"
        )
    )
    assert meta["voice_id"] == voice["id"] and meta["workflow"] == "speak"


def _history(tmp_path, account):
    return sorted((tmp_path / "accounts" / account.account_id / "audio").glob("*.wav"))


def _generate(client, **body):
    payload = {"messages": [{"role": "user", "content": "Read me aloud."}], **body}
    return client.post("/api/inference/audio/generate", json = payload)


@pytest.mark.parametrize("persist", [True, False])
def test_read_aloud_keeps_nothing_in_history_and_generate_still_does(stub, tmp_path, persist):
    stub["use"](*_KOKORO)
    with _client(ALICE) as client:
        response = _generate(client, **({} if persist else {"persist": False}))
    assert response.status_code == 200, response.text
    body = response.json()
    assert base64.b64decode(body["audio"]["data"])[:4] == b"RIFF"
    assert (body["clip_id"] is not None) is persist
    assert len(_history(tmp_path, ALICE)) == (1 if persist else 0)


@pytest.mark.parametrize("persist", [True, False])
def test_read_aloud_speaks_in_a_saved_voice(stub, tmp_path, persist):
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
        _clone_info(workflows = ("speak", "clone"), reference_text = "optional"),
    )
    voice = _voice(ALICE, _input(ALICE), transcript = "Okay, I'm Cemo.")
    with _client(ALICE) as client:
        response = _generate(client, voice_id = voice["id"], persist = persist)
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    assert call["workflow"] == "speak" and call["reference_text"] == "Okay, I'm Cemo."
    reference = Path(call["audio_inputs"]["reference"])
    assert reference.parent == _inputs_root(tmp_path)
    assert reference.name.startswith(f"v-{voice['id']}.")
    assert base64.b64decode(response.json()["audio"]["data"])[:4] == b"RIFF"
    assert len(_history(tmp_path, ALICE)) == (1 if persist else 0)


def test_a_saved_voice_keeps_the_request_settings(stub):
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
        _clone_info(workflows = ("speak", "clone"), reference_text = "optional"),
    )
    voice = _voice(ALICE, _input(ALICE))
    with _client(ALICE) as client:
        response = _generate(
            client, voice_id = voice["id"], seed = 7, audio_language = "English", max_tokens = 50
        )
        empty = client.post(
            "/api/inference/audio/generate",
            json = {"messages": [{"role": "user", "content": ""}], "voice_id": voice["id"]},
        )
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    assert (call["seed"], call["language"]) == (7, "English")
    assert call["max_new_tokens"] <= 50
    assert empty.status_code == 422, empty.text


def test_read_aloud_voice_refusals(stub):
    voice = _voice(ALICE, _input(ALICE))
    with _client(BOB) as client:
        assert _generate(client, voice_id = voice["id"]).status_code == 404
    stub["use"](*_KOKORO)
    with _client(ALICE) as client:
        refused = _generate(client, voice_id = voice["id"])
        assert refused.status_code == 400
        assert refused.json()["detail"] == "Load a model that can clone a voice."
        assert _generate(client, voice_id = "../voice").status_code == 422
    assert stub["backend"].calls == []


def _wav_params(path):
    import wave
    with wave.open(str(path)) as w:
        return {"sample_rate": w.getframerate(), "channels": w.getnchannels()}


def _convert_info(family, folder):
    from core.inference import audio_cpp_backend as acb
    from core.inference import audio_cpp_models as acm

    policy = acm.FAMILIES[family]
    variant = acm.AudioCppVariant("Q8_0", (acm.RepoFile("m.gguf", 1),), "m.gguf")
    model = acm.AudioCppModel(
        id = f"audio-cpp/audio.cpp-gguf/{folder}",
        repo_id = acm.AUDIO_CPP_REPO,
        folder = folder,
        display_name = folder,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        speaks = policy.speaks,
        clone = policy.clone,
        convert = policy.convert,
    )
    return f"audio-cpp/audio.cpp-gguf/{folder}", {
        "is_audio": True,
        "audio_type": "audiocpp_tts",
        **acb.model_info_fields(model),
    }


def _use(stub, family, folder):
    return stub["use"](*_convert_info(family, folder))


def _convert(client, **body):
    return client.post("/api/inference/audio/run", json = {"workflow": "convert", **body})


def _meta(tmp_path, account, clip_id):
    return json.loads(
        (tmp_path / "accounts" / account.account_id / "audio" / f"{clip_id}.json").read_text(
            encoding = "utf-8"
        )
    )


def test_an_rvc_run_converts_an_upload_to_a_builtin_voice(stub, tmp_path):
    backend = _use(stub, "rvc", "RVC-GGUF")
    input_id = _input(ALICE, seconds = 2.0)
    with _client(ALICE) as client:
        response = _convert(
            client,
            inputs = {"source": {"input_id": input_id}},
            convert = {"voice": "manthos", "pitch": 3},
            options = {"retrieval_blend": 0.5},
            seed = 7,
        )
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    inputs_root = tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    source = Path(call["audio_inputs"]["source"])
    assert set(call["audio_inputs"]) == {"source"}
    assert source.parent == inputs_root and source.name == f"{input_id}.16000.mono.m300.wav"
    info = _wav_params(source)
    assert (info["sample_rate"], info["channels"]) == (16000, 1)
    assert call["workflow"] == "convert" and call["seed"] == 7
    assert call["convert"] == {
        "mode": "speech",
        "pitch": 3,
        "pitch_auto": False,
        "style": "source",
        "voice": "manthos",
        "source_text": None,
    }
    assert call["audio_options"] == {"retrieval_blend": 0.5}
    assert "reference_text" not in call and "speed" not in call
    (clip,) = response.json()["clips"]
    assert clip["workflow"] == "convert" and clip["role"] == "output"
    meta = _meta(tmp_path, ALICE, clip["id"])
    assert meta["workflow"] == "convert" and meta["prompt"] == "me.webm → Manthos"
    assert (meta["source_input_id"], meta["source_name"]) == (input_id, "me.webm")
    assert (meta["reference_name"], meta["target_builtin"]) == ("Manthos", "manthos")
    assert meta["settings"] == {
        "mode": "speech",
        "pitch": 3,
        "pitch_auto": False,
        "style": "source",
        "options": {"retrieval_blend": 0.5},
    }
    assert "voice_id" not in meta and "source_clip_id" not in meta
    sidecar = json.dumps(meta)
    assert str(tmp_path) not in sidecar and "inputs" not in sidecar
    # The upload expires, so the clip keeps the 16 kHz copy the model heard beside it.
    assert meta["source_saved"] is True
    kept = tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip['id']}.source.wav"
    assert kept.read_bytes() == source.read_bytes()
    source_url = f"/api/inference/audio/gallery/{clip['id']}/source/file"
    with _client(BOB) as client:
        assert client.get(source_url).status_code == 404
    with _client(ALICE) as client:
        served = client.get(source_url)
        assert served.status_code == 200 and served.content == kept.read_bytes()
        listed = client.get("/api/inference/audio/gallery").json()["audio"]
        assert [c["id"] for c in listed] == [clip["id"]] and listed[0]["source_saved"] is True
        assert client.delete(f"/api/inference/audio/gallery/{clip['id']}").status_code == 200
        assert client.get(source_url).status_code == 404
    assert not kept.exists()


def test_a_seed_vc_run_takes_a_history_clip_and_a_saved_voice_at_its_rates(stub, tmp_path):
    backend = _use(stub, "seed_vc", "SeedVC-MLX-GGUF")
    sources = _alice_sources()
    with _client(ALICE) as client:
        response = _convert(
            client,
            inputs = {
                "source": {"clip_id": sources["clip_id"]},
                "target": {"voice_id": sources["voice_id"]},
            },
            convert = {"mode": "singing", "pitch_auto": True},
        )
    assert response.status_code == 200, response.text
    (call,) = backend.calls
    source, target = Path(call["audio_inputs"]["source"]), Path(call["audio_inputs"]["target"])
    assert source.name == f"c-{sources['clip_id']}.44100.mono.m300.wav"
    assert target.name == f"v-{sources['voice_id']}.44100.mono.m30.wav"
    assert _wav_params(target)["sample_rate"] == 44100
    assert call["convert"]["mode"] == "singing" and call["convert"]["pitch_auto"] is True
    meta = _meta(tmp_path, ALICE, response.json()["clips"][0]["id"])
    assert meta["prompt"] == "alice said → Alice"
    assert (meta["source_clip_id"], meta["voice_id"]) == (sources["clip_id"], sources["voice_id"])
    assert meta["reference_name"] == "Alice" and "target_builtin" not in meta
    # A history clip stays in history: no copy, and no source file to serve.
    assert "source_saved" not in meta
    clip_id = response.json()["clips"][0]["id"]
    audio_dir = tmp_path / "accounts" / ALICE.account_id / "audio"
    assert not (audio_dir / f"{clip_id}.source.wav").exists()
    with _client(ALICE) as client:
        assert client.get(f"/api/inference/audio/gallery/{clip_id}/source/file").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"inputs": {"source": {"path": "/etc/passwd"}}},
        {"inputs": {"source": {"input_id": "a" * 32}, "target": {"url": "http://x"}}},
        {"options": {"voice_model_path": "/x.pth"}},
        {"options": {"target_voice": "/etc/passwd"}},
        {"text": "hello"},
        {"convert": {"engine": "v2"}},
        {"convert": {"pitch": 25}},
        {"convert": {"mode": "rap"}},
        {"convert": {"voice": "nope"}},
        {"inputs": {"source": {"input_id": "a" * 32}, "reference": {"input_id": "a" * 32}}},
        {"source_audio": "/etc/passwd"},
    ],
)
def test_convert_client_paths_and_unknown_fields_are_422(stub, body):
    _use(stub, "seed_vc", "SeedVC-MLX-GGUF")
    with _client(ALICE) as client:
        response = _convert(client, **body)
    assert response.status_code == 422, response.text
    assert stub["backend"].calls == []


def test_convert_fields_on_clone_and_speak_are_422(stub):
    with _client(ALICE) as client:
        assert _run(client, convert = {"mode": "speech"}).status_code == 422
        # A source on clone or speak reaches the route, which refuses it as Music-only.
        assert _run(client, inputs = {"source": {"input_id": "a" * 32}}).status_code == 400
        assert _run(client, workflow = "speak", text = None).status_code == 422
    assert stub["backend"].calls == []


@pytest.mark.parametrize(
    "family, folder, body, detail",
    [
        (None, None, {"target": True}, "Load a model that can convert a voice."),
        ("seed_vc", "SeedVC-MLX-GGUF", {"source": False}, "Add the recording to convert."),
        ("meanvc2", "MeanVC2-GGUF", {}, "Add the target voice."),
        (
            "rvc",
            "RVC-GGUF",
            {"target": True},
            "RVC converts to its built-in voices; pick one under Built-in.",
        ),
        (
            "meanvc2",
            "MeanVC2-GGUF",
            {"target": True, "convert": {"mode": "singing"}},
            "MeanVC2 does not convert singing.",
        ),
        (
            "vevo2",
            "Vevo2-GGUF",
            {"target": True, "convert": {"style": "target"}},
            "Type what's said in the recording, or press Transcribe.",
        ),
    ],
)
def test_convert_problems_are_400s_in_words(stub, family, folder, body, detail):
    if family is not None:
        _use(stub, family, folder)
    input_id = _input(ALICE)
    inputs = {} if body.get("source") is False else {"source": {"input_id": input_id}}
    if body.get("target"):
        inputs["target"] = {"input_id": input_id}
    with _client(ALICE) as client:
        response = _convert(client, inputs = inputs, convert = body.get("convert") or {})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == detail
    assert stub["backend"].calls == []


def test_vevo2_takes_the_target_style_with_a_transcript_and_clone_still_works(stub, tmp_path):
    backend = _use(stub, "vevo2", "Vevo2-GGUF")
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        converted = _convert(
            client,
            inputs = {
                "source": {"input_id": input_id},
                "target": {"input_id": input_id},
                "source_text": "  Okay, I'm Cemo. ",
            },
            convert = {"style": "target"},
        )
        cloned = _run(client, inputs = {"reference": {"input_id": input_id}})
    assert converted.status_code == 200 and cloned.status_code == 200, cloned.text
    convert_call, clone_call = backend.calls
    assert convert_call["convert"]["source_text"] == "Okay, I'm Cemo."
    assert {Path(p).name.split(".")[1] for p in convert_call["audio_inputs"].values()} == {"24000"}
    assert clone_call["workflow"] == "clone" and "convert" not in clone_call


def test_a_convert_only_model_refuses_plain_speech(stub):
    _use(stub, "rvc", "RVC-GGUF")
    with _client(ALICE) as client:
        generate = client.post(
            "/api/inference/audio/generate", json = {"messages": [{"role": "user", "content": "hi"}]}
        )
        speak = _run(client, workflow = "speak")
    for response in (generate, speak):
        assert response.status_code == 400
        assert response.json()["detail"] == "RVC converts recordings. Open Convert and add one."
    assert stub["backend"].calls == []


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("kind", ["input_id", "clip_id", "voice_id"])
def test_another_accounts_ids_are_404_on_convert(stub, role, kind):
    _use(stub, "seed_vc", "SeedVC-MLX-GGUF")
    sources = _alice_sources()
    own = _input(BOB)
    inputs = {"source": {"input_id": own}, "target": {"input_id": own}}
    inputs[role] = {kind: sources[kind]}
    with _client(BOB) as client:
        response = _convert(client, inputs = inputs)
    assert response.status_code == 404, response.text
    assert stub["backend"].calls == []


def test_the_worker_and_orchestrator_carry_convert_and_the_running_task(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Backend:
        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

        def runtime_fields(self):
            return {"audio_server_task": "vc", "audio_convert_route": None}

    responses: queue.Queue = queue.Queue()
    convert = {"mode": "speech", "pitch": None, "pitch_auto": False, "style": "source"}
    _handle_generate_audio(
        _Backend(),
        {
            "request_id": "r1",
            "text": "x",
            "workflow": "convert",
            "audio_inputs": {"source": "/abs/s.wav", "target": "/abs/t.wav"},
            "convert": convert,
        },
        responses,
        threading.Event(),
    )
    assert seen[0]["convert"] == convert and seen[0]["workflow"] == "convert"
    done = responses.get_nowait()
    assert done["audio_runtime"] == {"audio_server_task": "vc", "audio_convert_route": None}

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []

    def read_one(*, timeout):
        return {**done, "request_id": sent[0]["request_id"]}

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
    orchestrator.models = {"m": {"audio_server_task": "clon"}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    orchestrator.generate_audio_response(
        "x",
        workflow = "convert",
        audio_inputs = {"source": "/abs/s.wav"},
        convert = convert,
    )
    (cmd,) = sent
    assert cmd["convert"] == convert and cmd["audio_inputs"] == {"source": "/abs/s.wav"}
    # The status mirror follows the server the run left running.
    assert orchestrator.models["m"]["audio_server_task"] == "vc"


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


def test_a_repaint_past_the_end_sizes_its_work_to_the_range_end(stub):
    backend = stub["use"](ACE_STEP, _music_info("ace_step", "ACE-Step1.5-GGUF"))
    backend.__class__ = _MusicBackend
    backend.takes, backend.run_dirs = 1, []
    sources = _alice_sources()
    with _client(ALICE) as client:
        short = _edit(client, {"clip_id": sources["clip_id"]}, action = "repaint")
        long = _edit(
            client,
            {"clip_id": sources["clip_id"]},
            action = "repaint",
            ranges = [{"start_s": 0.02, "end_s": 25.0}],
        )
    assert short.status_code == 200 and long.status_code == 200, long.text
    first, second = backend.calls
    assert second["music"]["timeout_s"] > first["music"]["timeout_s"]


def test_a_convert_expiry_names_the_side_that_expired():
    """Both uploads shared "This reference expired", so the page marked both cards when one went."""
    from core.inference.audio_inputs import AudioInputError
    from routes.inference import CONVERT_EXPIRED_DETAIL, _convert_role_error

    gone = AudioInputError(404, "This reference expired. Add it again.")
    assert _convert_role_error(gone, "source").detail == CONVERT_EXPIRED_DETAIL["source"]
    assert _convert_role_error(gone, "target").detail == CONVERT_EXPIRED_DETAIL["target"]
    other = AudioInputError(404, "That clip is gone.")
    assert _convert_role_error(other, "target") is other


def _v1(account):
    from utils.api_errors import install_api_error_handlers

    client = _client(account)
    install_api_error_handlers(client.app)
    client.app.include_router(inference.router, prefix = "/v1")
    return client


@pytest.fixture
def switches(stub, monkeypatch):
    seen = []

    async def record(model, *_args, **kwargs):
        seen.append((model, kwargs.get("require_audio_workflow")))

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", record)
    return seen


def test_a_named_model_is_switched_to_with_the_workflow_it_must_run(stub, switches):
    voice = _voice(ALICE, _input(ALICE), transcript = "hi")
    with _client(ALICE) as client:
        clone = _run(client, model = QWEN3_BASE, inputs = {"reference": {"voice_id": voice["id"]}})
        speak = _run(
            client,
            workflow = "speak",
            model = QWEN3_BASE,
            inputs = {"reference": {"voice_id": voice["id"]}},
        )
        unnamed = _run(client, inputs = {"reference": {"voice_id": voice["id"]}})
    assert [r.status_code for r in (clone, speak, unnamed)] == [200, 200, 200]
    assert switches == [
        (QWEN3_BASE, "clone"),
        (QWEN3_BASE, "clone"),
        (inference._RELOAD_ONLY_MODEL, "clone"),
    ]


def test_a_named_separation_model_is_switched_to_before_the_track_is_read(sep, switches):
    with _client(ALICE) as client:
        response = _separate(client, {"input_id": _input(ALICE)}, model = HTDEMUCS_6)
    assert response.status_code == 200, response.text
    assert switches == [(HTDEMUCS_6, "separate")]


def test_a_v1_run_is_monitored_and_its_clips_are_served_under_v1(stub):
    from core.inference.api_monitor import api_monitor

    voice = _voice(ALICE, _input(ALICE), transcript = "hi")
    body = {
        "workflow": "clone",
        "text": "Hello.",
        "inputs": {"reference": {"voice_id": voice["id"]}},
    }
    api_monitor.clear()
    with _v1(ALICE) as client:
        studio = client.post("/api/inference/audio/run", json = body)
        assert api_monitor.snapshot(include_details = False) == []
        api = client.post("/v1/audio/run", json = body)
        assert api.status_code == 200, api.text
        url = api.json()["clips"][0]["url"]
        assert url.startswith("/v1/audio/gallery/")
        assert client.get(url).content[:4] == b"RIFF"
    assert studio.json()["clips"][0]["url"].startswith("/api/inference/audio/gallery/")
    (row,) = api_monitor.snapshot(include_details = False)
    assert (row["endpoint"], row["status"], row["model"]) == (
        "/v1/audio/run",
        "completed",
        QWEN3_BASE,
    )


def test_a_bad_v1_generate_in_a_saved_voice_opens_no_monitor_row(stub):
    from core.inference.api_monitor import api_monitor

    voice = _voice(ALICE, _input(ALICE))
    api_monitor.clear()
    with _v1(ALICE) as client:
        response = client.post(
            "/v1/audio/generate",
            json = {"messages": [{"role": "user", "content": ""}], "voice_id": voice["id"]},
        )
    assert 400 <= response.status_code < 500, response.text
    assert api_monitor.snapshot(include_details = False) == []


@pytest.mark.parametrize("as_object", [False, True])
def test_speech_in_a_saved_voice_clones_with_its_transcript(stub, tmp_path, as_object):
    voice = _voice(ALICE, _input(ALICE), transcript = "Okay, I'm Cemo.")
    ref = {"id": voice["id"]} if as_object else voice["id"]
    with _v1(ALICE) as client:
        response = client.post("/v1/audio/speech", json = {"input": "Hello.", "voice": ref})
    assert response.status_code == 200, response.text
    assert response.content[:4] == b"RIFF"
    (call,) = stub["backend"].calls
    assert (call["workflow"], call["reference_text"]) == ("speak", "Okay, I'm Cemo.")
    assert Path(call["audio_inputs"]["reference"]).name.startswith(f"v-{voice['id']}.")
    (meta,) = [json.loads(p.read_text()) for p in _gallery_root(tmp_path).glob("*.json")]
    assert (meta["voice_id"], meta["workflow"]) == (voice["id"], "speak")


def test_speech_clones_a_reference_and_refuses_it_beside_a_saved_voice(stub):
    input_id = _input(ALICE)
    voice = _voice(ALICE, input_id)
    reference = {"reference": {"input_id": input_id}, "reference_text": "Hi there."}
    with _v1(ALICE) as client:
        both = client.post(
            "/v1/audio/speech", json = {"input": "Hello.", "voice": voice["id"], **reference}
        )
        clone = client.post(
            "/v1/audio/speech", json = {"input": "Hello.", "voice": "alloy", **reference}
        )
    assert (both.status_code, both.json()["error"]["param"]) == (400, "reference")
    assert clone.status_code == 200, clone.text
    (call,) = stub["backend"].calls
    assert (call["workflow"], call["reference_text"]) == ("clone", "Hi there.")
    assert Path(call["audio_inputs"]["reference"]).name.startswith(f"{input_id}.24000.mono")


def test_speech_uses_a_builtin_speaker_the_model_lists(stub):
    voices = {"name": "voice", "type": "enum", "values": ["Ryan", "Vivian"]}
    backend = stub["use"](QWEN3_BASE, {**_speak_info(), "audio_options": [voices]})
    with _v1(ALICE) as client:
        for body in (
            {"voice": "Ryan"},
            {"voice": "alloy"},
            {"voice": "Ryan", "audio_options": {"voice": "Vivian"}},
        ):
            response = client.post("/v1/audio/speech", json = {"input": "Hello.", **body})
            assert response.status_code == 200, response.text
    assert [call.get("audio_options") for call in backend.calls] == [
        {"voice": "Ryan"},
        None,
        {"voice": "Vivian"},
    ]


@pytest.mark.parametrize(
    "fmt, media_type, magic",
    [
        ("mp3", "audio/mpeg", (b"ID3", b"\xff\xfb", b"\xff\xf3")),
        ("flac", "audio/flac", (b"fLaC",)),
        ("opus", "audio/ogg", (b"OggS",)),
        ("aac", "audio/aac", (b"\xff\xf1", b"\xff\xf9")),
    ],
)
def test_speech_encodes_each_format_and_history_keeps_the_wav(
    stub, tmp_path, fmt, media_type, magic
):
    stub["use"](QWEN3_BASE, _speak_info())
    with _v1(ALICE) as client:
        response = client.post("/v1/audio/speech", json = {"input": "Hello.", "response_format": fmt})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media_type
    assert response.content.startswith(magic)
    (clip,) = _gallery_root(tmp_path).glob("*.wav")
    assert clip.read_bytes() == _wav()


@pytest.mark.parametrize("rate, channels", [(24000, 1), (44100, 2)])
def test_speech_pcm_is_openais_24k_mono_samples(stub, monkeypatch, rate, channels):
    stub["use"](QWEN3_BASE, _speak_info())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * rate * channels)
    monkeypatch.setattr(
        _Backend, "generate_audio_response", lambda self, **kw: (buf.getvalue(), rate)
    )
    with _v1(ALICE) as client:
        response = client.post(
            "/v1/audio/speech", json = {"input": "Hello.", "response_format": "pcm"}
        )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/pcm"
    # One second of audio: 24000 two-byte mono samples, whatever the model's rate and layout.
    assert abs(len(response.content) - 48000) <= 2 * 64
    if (rate, channels) == (24000, 1):
        assert response.content == buf.getvalue()[44:]
