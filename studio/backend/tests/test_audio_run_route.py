# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/run: ids in, server-chosen account paths out, never a client path or audio bytes.

A stub orchestrator stands in for the worker and records what the route hands it; the account
boundary is the real one (ALICE and BOB as in test_account_media_isolation).
"""

from __future__ import annotations

import asyncio
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

from auth import policy, storage as auth_storage
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import audio_gallery, audio_inputs, audio_voices
from routes import inference
from utils.account_context import AccountContext, bind_account, reset_account, run_as

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_inputs import _chunks, encode  # noqa: E402

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
QWEN3_BASE = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF"


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    connection = auth_storage.get_connection()
    with connection:
        for account in (ALICE, BOB):
            connection.execute(
                "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret,"
                " account_id, role, is_active) VALUES (?, 'salt', 'hash', 'secret', ?, 'user', 1)",
                (account.username, account.account_id),
            )
    connection.close()
    yield
    policy.invalidate_account_cache()


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
    **extra,
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
            "reference_text_waived": [["x_vector_only_mode", ["true"]]],
            "emotion_audio": emotion,
            "input_rate": 24000,
        },
        **extra,
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

    def save():
        record, _ = asyncio.run(audio_inputs.save_stream(_chunks([data]), "me.webm"))
        return record["id"]

    return run_as(account, save)


def _run(client, **body):
    payload = {"workflow": "clone", "text": "Hello in my voice.", **body}
    return client.post("/api/inference/audio/run", json = payload)


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
        clip_bytes = client.get(body["clips"][0]["url"])
    ((call,),) = [stub["backend"].calls]
    reference = Path(call["audio_inputs"]["reference"])
    inputs_root = tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
    # An absolute path inside ALICE's inputs, a prepared 24 kHz mono copy; never bytes.
    assert reference.is_absolute() and reference.parent == inputs_root
    assert reference.name.startswith(f"{input_id}.24000.mono")
    info = audio_inputs.wav_info(reference)
    assert (info["sample_rate"], info["channels"]) == (24000, 1)
    assert (call["workflow"], call["reference_text"], call["language"], call["seed"]) == (
        "clone",
        "Okay, I'm Cemo.",
        "en",
        7,
    )
    assert "speed" not in call
    assert all(not isinstance(v, (bytes, bytearray)) for v in call.values())
    assert body["model"] and body["audio"] is None and body["group_id"] is None
    (clip,) = body["clips"]
    assert set(clip) == {"id", "role", "url", "sample_rate", "duration_s", "workflow"}
    assert (clip["role"], clip["workflow"], clip["sample_rate"]) == ("output", "clone", 24000)
    assert clip["url"] == f"/api/inference/audio/gallery/{clip['id']}/file"
    assert clip_bytes.status_code == 200
    sidecar = (tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip['id']}.json").read_text(
        encoding = "utf-8"
    )
    meta = json.loads(sidecar)
    assert meta["workflow"] == "clone" and meta["role"] == "output"
    assert meta["reference_name"] == "me.webm"
    assert meta["settings"] == {
        "language": "en",
        "instructions": None,
        "options": {"x_vector_only_mode": False},
        "reference_text_used": True,
        "speed": None,
    }
    # No server path in the recipe.
    assert str(tmp_path) not in sidecar and "inputs" not in sidecar
    assert not any(isinstance(v, str) and v.startswith("/") for v in meta.values())


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
        response = _run(client, **body)
    assert response.status_code == 422, response.text
    assert stub["backend"].calls == []


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


@pytest.mark.parametrize("kind", ["input_id", "clip_id", "voice_id"])
def test_another_accounts_ids_are_404(stub, kind):
    sources = _alice_sources()
    with _client(BOB) as client:
        response = _run(
            client,
            inputs = {"reference": {kind: sources[kind]}, "reference_text": "hello"},
        )
        assert response.status_code == 404, response.text
        emotion = _run(
            client,
            inputs = {
                "reference": {kind: sources[kind]},
                "emotion": {kind: sources[kind]},
                "reference_text": "hello",
            },
        )
        assert emotion.status_code == 404
    assert stub["backend"].calls == []
    # ALICE runs with every one of them.
    with _client(ALICE) as client:
        assert (
            _run(
                client, inputs = {"reference": {kind: sources[kind]}, "reference_text": "hi"}
            ).status_code
            == 200
        )


def test_a_saved_voice_brings_its_transcript_and_is_recorded(stub, tmp_path):
    input_id = _input(ALICE)
    voice = run_as(
        ALICE,
        lambda: audio_voices.create(
            audio_inputs.input_path(input_id), {"name": "Alice", "transcript": "Okay, I'm Cemo."}
        ),
    )
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"voice_id": voice["id"]}})
    assert response.status_code == 200, response.text
    (call,) = stub["backend"].calls
    assert call["reference_text"] == "Okay, I'm Cemo."
    assert Path(call["audio_inputs"]["reference"]).name.startswith(f"v-{voice['id']}.")
    clip_id = response.json()["clips"][0]["id"]
    meta = json.loads(
        (tmp_path / "accounts" / ALICE.account_id / "audio" / f"{clip_id}.json").read_text(
            encoding = "utf-8"
        )
    )
    assert meta["voice_id"] == voice["id"] and meta["reference_name"] == "Alice"


def test_a_missing_transcript_is_a_400_unless_timbre_only(stub):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        missing = _run(client, inputs = {"reference": {"input_id": input_id}})
        assert missing.status_code == 400
        assert missing.json()["detail"] == "Type what's said in the reference clip."
        timbre = _run(
            client,
            inputs = {"reference": {"input_id": input_id}},
            options = {"x_vector_only_mode": True},
        )
        assert timbre.status_code == 200, timbre.text
    assert len(stub["backend"].calls) == 1
    stub["use"]("x/Chatterbox-GGUF", _clone_info(reference_text = "unused"))
    with _client(ALICE) as client:
        assert _run(client, inputs = {"reference": {"input_id": input_id}}).status_code == 200


def test_a_speak_model_cannot_clone(stub):
    input_id = _input(ALICE)
    stub["use"](
        "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
        {
            "is_audio": True,
            "audio_type": "audiocpp_tts",
            "audio_workflows": ["speak"],
            "audio_reference_text": None,
            "audio_required_inputs": [],
        },
    )
    with _client(ALICE) as client:
        clone = _run(client, inputs = {"reference": {"input_id": input_id}, "reference_text": "x"})
        assert clone.status_code == 400
        assert clone.json()["detail"] == "Load a model that can clone a voice."
        # Speak in a saved voice needs a model that clones, too.
        speak_ref = _run(client, workflow = "speak", inputs = {"reference": {"input_id": input_id}})
        assert speak_ref.status_code == 400
        # Plain speak still works through /audio/run.
        assert _run(client, workflow = "speak").status_code == 200
    assert len(stub["backend"].calls) == 1


def test_a_clone_run_needs_a_reference(stub):
    with _client(ALICE) as client:
        response = _run(client)
    assert (
        response.status_code == 400
        and response.json()["detail"] == "Add a reference clip to clone."
    )


def test_an_emotion_clip_only_for_a_family_that_takes_one(stub):
    first, second = _input(ALICE, 0.5), _input(ALICE, 0.7)
    with _client(ALICE) as client:
        refused = _run(
            client,
            inputs = {
                "reference": {"input_id": first},
                "emotion": {"input_id": second},
                "reference_text": "x",
            },
        )
        assert refused.status_code == 400 and "emotion" in refused.json()["detail"]
    backend = stub["use"](
        "audio-cpp/audio.cpp-gguf/IndexTTS2-GGUF",
        _clone_info(reference_text = "unused", emotion = True),
    )
    with _client(ALICE) as client:
        ok = _run(
            client,
            inputs = {"reference": {"input_id": first}, "emotion": {"input_id": second}},
        )
    assert ok.status_code == 200, ok.text
    (call,) = backend.calls
    assert set(call["audio_inputs"]) == {"reference", "emotion"}
    assert Path(call["audio_inputs"]["emotion"]).name.startswith(f"{second}.24000.mono")


def test_maya1_without_a_description_is_a_400_on_every_speech_route(stub):
    stub["use"](
        "audio-cpp/audio.cpp-gguf/Maya1-GGUF",
        {
            "is_audio": True,
            "audio_type": "audiocpp_tts",
            "audio_workflows": ["speak"],
            "audio_reference_text": None,
            "audio_required_inputs": ["instruct"],
        },
    )
    with _client(ALICE) as client:
        run = _run(client, workflow = "speak")
        assert (run.status_code, run.json()["detail"]) == (400, "Maya1 needs a voice description.")
        generate = client.post(
            "/api/inference/audio/generate", json = {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert (generate.status_code, generate.json()["detail"]) == (
            400,
            "Maya1 needs a voice description.",
        )
        speech = client.post("/api/inference/audio/speech", json = {"input": "hi"})
        assert speech.status_code == 400
        described = _run(client, workflow = "speak", instructions = "A calm, low male voice.")
        assert described.status_code == 200
    assert len(stub["backend"].calls) == 1


def test_a_clone_only_model_refuses_plain_speech(stub):
    with _client(ALICE) as client:
        generate = client.post(
            "/api/inference/audio/generate", json = {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert generate.status_code == 400 and "Open Clone" in generate.json()["detail"]
    assert stub["backend"].calls == []


def test_a_failed_gallery_save_returns_the_audio_inline(stub, monkeypatch):
    input_id = _input(ALICE)
    monkeypatch.setattr(inference, "_persist_tts_clip", lambda *a, **k: None)
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"input_id": input_id}, "reference_text": "x"})
    body = response.json()
    assert response.status_code == 200 and body["clips"] == []
    assert body["audio"]["format"] == "wav" and body["audio"]["sample_rate"] == 24000
    assert base64.b64decode(body["audio"]["data"])[:4] == b"RIFF"


def test_a_runtime_refusal_keeps_the_runtime_reason(stub):
    from core.inference.audio_errors import AudioRuntimeError

    def refuse(**_kwargs):
        raise AudioRuntimeError("Qwen3 voice clone ICL mode requires reference text", status = 500)

    stub["backend"].generate_audio_response = refuse
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"input_id": input_id}, "reference_text": "x"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Qwen3 voice clone ICL mode requires reference text"


def test_the_orchestrator_command_carries_paths_not_bytes(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    sent = []
    wav = base64.b64encode(_wav()).decode()

    def read_one(*, timeout):
        return {
            "type": "audio_done",
            "request_id": sent[0]["request_id"],
            "wav_base64": wav,
            "sample_rate": 24000,
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
    orchestrator.models = {"m": {}}
    orchestrator._gen_lock = threading.Lock()
    orchestrator._send_order_lock = threading.Lock()
    orchestrator._unload_pending = False
    reference = "/home/x/.unsloth/studio/accounts/a/audio/inputs/abc.24000.mono.m30.wav"
    orchestrator.generate_audio_response(
        "hello",
        workflow = "clone",
        audio_inputs = {"reference": reference},
        reference_text = "hi",
        speed = 1.2,
    )
    (cmd,) = sent
    assert cmd["audio_inputs"] == {"reference": reference}
    assert (cmd["workflow"], cmd["reference_text"], cmd["speed"]) == ("clone", "hi", 1.2)
    assert all(not isinstance(v, (bytes, bytearray)) for v in cmd.values())
    sent.clear()
    orchestrator.generate_audio_response("plain")
    assert not {"workflow", "audio_inputs", "reference_text", "speed"} & set(sent[0])


def test_the_worker_forwards_clone_fields_only_when_present():
    from core.inference.worker import _handle_generate_audio

    seen = []

    class _Backend:
        def generate_audio_response(self, **kwargs):
            seen.append(kwargs)
            return _wav(), 24000

    responses: queue.Queue = queue.Queue()
    _handle_generate_audio(
        _Backend(),
        {
            "request_id": "r1",
            "text": "hi",
            "workflow": "clone",
            "audio_inputs": {"reference": "/abs/ref.wav"},
            "reference_text": "x",
            "speed": 1.1,
        },
        responses,
        threading.Event(),
    )
    _handle_generate_audio(
        _Backend(), {"request_id": "r2", "text": "hi"}, responses, threading.Event()
    )
    assert seen[0]["audio_inputs"] == {"reference": "/abs/ref.wav"}
    assert (seen[0]["workflow"], seen[0]["reference_text"], seen[0]["speed"]) == ("clone", "x", 1.1)
    assert not {"workflow", "audio_inputs", "reference_text", "speed"} & set(seen[1])


def _cosy_info():
    info = _clone_info()
    info["audio_clone"]["reference_text_waived"] = [
        ["template_name", ["cross_lingual", "instruct"]]
    ]
    return info


@pytest.mark.parametrize(
    "options, status",
    [
        (None, 400),
        ({"template_name": "zero_shot"}, 400),
        ({"template_name": "cross_lingual"}, 200),
        ({"template_name": "instruct"}, 200),
    ],
)
def test_cosyvoice3_needs_a_transcript_only_for_zero_shot(stub, options, status):
    stub["use"]("audio-cpp/audio.cpp-gguf/CosyVoice3-GGUF", _cosy_info())
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _run(client, inputs = {"reference": {"input_id": input_id}}, options = options)
    assert response.status_code == status, response.text


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
    info = audio_inputs.wav_info(source)
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
    assert audio_inputs.wav_info(target)["sample_rate"] == 44100
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
