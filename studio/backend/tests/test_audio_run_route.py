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
    voice = run_as(
        ALICE,
        lambda: audio_voices.create(audio_inputs.input_path(input_id), {"name": "Alice"}),
    )
    return {"input_id": input_id, "clip_id": clip["id"], "voice_id": voice["id"]}


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
        {"convert": {"pitch": 13}},
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
        assert _run(client, inputs = {"source": {"input_id": "a" * 32}}).status_code == 422
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
