# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact requests audio.cpp receives for a voice conversion, per family, on a fake server."""

from __future__ import annotations

import base64
import io
import json
import wave

import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_convert as acc
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AudioCppModel,
    AudioCppVariant,
    RepoFile,
)
from core.inference.audio_errors import AudioRuntimeError

SOURCE = "/srv/accounts/a/audio/inputs/0123.16000.mono.m300.wav"
TARGET = "/srv/accounts/a/audio/inputs/4567.16000.mono.m30.wav"
TRANSCRIPT = "Okay, I'm Cemo and what you just heard wasn't a human voice."

# What the runtime specs declare (model_specs/*.json at the pin), as option_schema cleans them.
_RVC_OPTIONS = (
    {"name": "voice_id", "type": "enum", "values": ["default", "manthos", "chocola", "fraise"]},
    {"name": "retrieval_blend", "type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
    {"name": "semitone_shift", "type": "int", "default": 0},
    {"name": "rms_mix_rate", "type": "float", "default": 0.25},
    {"name": "unvoiced_protection", "type": "float", "min": 0.0, "max": 1.0, "default": 0.33},
)
_SEED_VC_OPTIONS = (
    {
        "name": "route",
        "type": "enum",
        "values": ["v2_vc", "v1_svc", "v1_whisper_bigvgan_vc", "v1_xlsr_hift_vc"],
    },
    {"name": "length_adjust", "type": "float", "min": 0.0, "default": 1.0},
    {"name": "similarity_guidance_scale", "type": "float", "min": 0.0, "default": 0.7},
    {"name": "voice_anonymization", "type": "bool", "default": False},
    {"name": "auto_f0_adjust", "type": "bool", "default": False},
    {"name": "semitone_shift", "type": "int", "default": 0},
)


_BLANK_OPTION = {
    "description": "",
    "required": False,
    "default": None,
    "min": None,
    "max": None,
    "values": None,
}


def _wav(rate) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * 240)
    return buf.getvalue()


def _model(family, options = ()) -> AudioCppModel:
    policy = acm.FAMILIES[family]
    variant = AudioCppVariant("Q8_0", (RepoFile("m-q8_0.gguf", 1),), "m-q8_0.gguf")
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{family}",
        repo_id = AUDIO_CPP_REPO,
        folder = family,
        display_name = family,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        options = tuple({**_BLANK_OPTION, **o} for o in options),
        speaks = policy.speaks,
        clone = policy.clone,
        convert = policy.convert,
    )


class _Server:
    model_id = "studio-test"
    backend = "cpu"

    def __init__(
        self,
        model,
        fail = None,
    ):
        self.model = model
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def alive(self):
        return True

    def stop(self):
        pass

    def post_json(self, path, payload, **_kwargs):
        self.calls.append((path, json.loads(json.dumps(payload))))
        if self.fail is not None:
            raise self.fail
        if path == "/v1/tasks/run":
            body = {"audio": base64.b64encode(_wav(16000)).decode(), "sample_rate": 16000}
            return "application/json", json.dumps(body).encode()
        return "audio/wav", _wav(24000)


@pytest.fixture
def started(monkeypatch):
    servers: list[_Server] = []

    def start(served, path, **_kwargs):
        servers.append(_Server(served))
        return servers[-1]

    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    monkeypatch.setattr(audio_cpp_files, "materialize", lambda model: "/models/m.gguf")
    return servers


def _backend(model, started):
    backend = acb.AudioCppBackend()
    backend._model = model
    backend.models = {model.id: {"is_audio": True, **acb.model_info_fields(model)}}
    backend.active_model_name = model.id
    backend._start_server(model)
    started.clear()
    return backend


def _convert(
    backend,
    inputs = None,
    options = None,
    seed = 7,
    **convert,
):
    wav, rate = backend.generate_audio_response(
        "ignored",
        workflow = "convert",
        audio_inputs = {"source": SOURCE, "target": TARGET} if inputs is None else inputs,
        audio_options = options,
        seed = seed,
        convert = convert,
    )
    assert wav[:4] == b"RIFF" and rate == 16000
    path, body = backend._server.calls[-1]
    assert path == "/v1/tasks/run"
    assert body["model"] == "studio-test"
    return body["request"]


def test_rvc_converts_to_a_builtin_voice_by_options_voice_id(started):
    backend = _backend(_model("rvc", _RVC_OPTIONS), started)
    request = _convert(
        backend,
        inputs = {"source": SOURCE},
        options = {"retrieval_blend": 0.6, "voice_id": "fraise", "semitone_shift": 9},
        voice = "manthos",
        pitch = 3,
    )
    # Strict: no seed, no voice_ref, no language; the typed fields win over driven options.
    assert request == {
        "audio": SOURCE,
        "options": {"retrieval_blend": "0.6", "voice_id": "manthos", "semitone_shift": "3"},
    }
    assert _convert(backend, inputs = {"source": SOURCE}) == {
        "audio": SOURCE,
        "options": {"voice_id": "default"},
    }
    # Auto is not offered on RVC, so it never becomes auto_f0_adjust.
    request = _convert(backend, inputs = {"source": SOURCE}, pitch = -5, pitch_auto = True)
    assert request["options"] == {"voice_id": "default", "semitone_shift": "-5"}
    assert started == []


@pytest.mark.parametrize(
    "family, target, match",
    [("rvc", TARGET, "built-in voices"), ("meanvc2", None, "Add the target voice.")],
)
def test_a_target_the_family_cannot_take_is_refused(family, target, match):
    with pytest.raises(acc.ConvertRequestError, match = match):
        acc.convert_request(_model(family), mode = "speech", source = SOURCE, target = target)


@pytest.mark.parametrize("family", ["meanvc2", "tone_color_vc"])
def test_a_reference_family_sends_audio_voice_ref_and_seed(family):
    request = acc.convert_request(
        _model(family), mode = "speech", source = SOURCE, target = TARGET, pitch = 4, seed = 7
    )
    assert request == {"audio": SOURCE, "voice_ref": TARGET, "seed": "7"}


def test_seed_vc_speech_sends_no_pitch_and_its_route_by_model_entry(started):
    model = _model("seed_vc", _SEED_VC_OPTIONS)
    backend = _backend(model, started)
    request = _convert(
        backend,
        options = {
            "length_adjust": 1.25,
            "similarity_guidance_scale": 1.2,
            "semitone_shift": 5,
            "auto_f0_adjust": True,
            "f0_condition": True,
        },
        pitch = 5,
        pitch_auto = True,
    )
    assert request == {
        "audio": SOURCE,
        "voice_ref": TARGET,
        "options": {"length_adjust": "1.25", "similarity_guidance_scale": "1.2"},
        "seed": "7",
    }
    # The default engine (v2_vc) runs on the loaded session: no restart, no route sent.
    assert started == []
    # Another engine reloads the model with that route as its default, never in the request.
    request = _convert(backend, options = {"route": "v1_xlsr_hift_vc"})
    assert "route" not in request and "route" not in request.get("options", {})
    (server,) = started
    assert server.model.server_task == "vc"
    assert server.model.model_options["default_request_options"] == {"route": "v1_xlsr_hift_vc"}
    assert backend.models[model.id]["audio_convert_route"] == "v1_xlsr_hift_vc"
    _convert(backend, options = {"route": "v1_xlsr_hift_vc"})
    assert len(started) == 1
    _convert(backend)
    assert len(started) == 2 and "default_request_options" not in started[-1].model.model_options
    assert backend.models[model.id]["audio_convert_route"] == "v2_vc"


def test_seed_vc_singing_reloads_under_svc_and_sends_its_pitch(started):
    model = _model("seed_vc", _SEED_VC_OPTIONS)
    backend = _backend(model, started)
    # The runtime applies the manual shift on top of the matched pitch, as seed-vc does.
    request = _convert(backend, mode = "singing", pitch = 12, pitch_auto = True)
    assert request["options"] == {"auto_f0_adjust": "true", "semitone_shift": "12"}
    request = _convert(backend, mode = "singing", pitch_auto = True)
    assert request["options"] == {"auto_f0_adjust": "true"}
    (server,) = started
    assert server.model.server_task == "svc"
    assert backend.models[model.id]["audio_server_task"] == "svc"
    assert backend.models[model.id]["audio_convert_route"] == "v1_svc"
    # A speech route sent in singing is dropped: v1_svc is the only singing route.
    request = _convert(backend, mode = "singing", pitch = -3, options = {"route": "v1_xlsr_hift_vc"})
    assert request["options"] == {"semitone_shift": "-3"}
    assert len(started) == 1
    _convert(backend)
    assert [s.model.server_task for s in started] == ["svc", "vc"]


def test_chatterbox_converts_under_vc_and_clones_back_under_clon(started):
    model = _model("chatterbox")
    backend = _backend(model, started)
    request = _convert(
        backend,
        options = {"s3gen_cfg_rate": 1.0, "num_inference_steps": 6, "exaggeration": 0.9},
    )
    # Exaggeration is clone-only and never reaches a conversion.
    assert request == {
        "audio": SOURCE,
        "voice_ref": TARGET,
        "options": {"s3gen_cfg_rate": "1", "num_inference_steps": "6"},
        "seed": "7",
    }
    assert [s.model.server_task for s in started] == ["vc"]
    assert backend.models[model.id]["audio_server_task"] == "vc"
    _convert(backend)
    assert len(started) == 1
    backend.generate_audio_response(
        "Hello there.", workflow = "clone", audio_inputs = {"reference": TARGET}
    )
    assert [s.model.server_task for s in started] == ["vc", "clon"]
    assert backend.models[model.id]["audio_server_task"] == "clon"
    assert started[-1].calls[-1][0] == "/v1/audio/speech"


def test_vevo2_keeps_the_source_style_with_source_audio_and_target_voice(started):
    model = _model("vevo2")
    backend = _backend(model, started)
    assert _convert(backend, pitch_auto = True, pitch = 4) == {
        "source_audio": SOURCE,
        "target_voice": TARGET,
        "seed": "7",
    }
    assert _convert(backend, pitch = 4)["options"] == {
        "use_pitch_shift": "true",
        "source_shift_steps": "4",
    }
    assert _convert(backend, pitch = 0)["options"] == {"use_pitch_shift": "false"}
    assert _convert(backend, options = {"num_inference_steps": 20})["options"] == {
        "num_inference_steps": "20"
    }
    _convert(backend, mode = "singing", pitch = -2)
    assert [s.model.server_task for s in started] == ["vc", "svc"]
    assert started[-1].calls[-1][1]["request"]["options"] == {
        "use_pitch_shift": "true",
        "source_shift_steps": "-2",
    }
    assert backend.models[model.id]["audio_convert_route"] is None


def test_vevo2_takes_the_target_style_through_the_source_transcript():
    model = _model("vevo2")
    request = acc.convert_request(
        model,
        mode = "speech",
        source = SOURCE,
        target = TARGET,
        style = "target",
        source_text = f"  {TRANSCRIPT} ",
        pitch = 5,
        seed = 7,
    )
    # No pitch under the target's style: its prosody is the target's.
    assert request == {
        "source_audio": SOURCE,
        "target_voice": TARGET,
        "route": "style_converted_vc",
        "target_text": TRANSCRIPT,
        "style_ref": TARGET,
        "seed": "7",
    }
    with pytest.raises(acc.ConvertRequestError, match = "Type what's said in the recording"):
        acc.convert_request(model, mode = "speech", source = SOURCE, target = TARGET, style = "target")
    singing = acc.convert_request(
        model, mode = "singing", source = SOURCE, target = TARGET, style = "target"
    )
    assert "route" not in singing and "target_text" not in singing


def test_a_mode_the_model_lacks_is_refused(started):
    backend = _backend(_model("meanvc2"), started)
    with pytest.raises(RuntimeError, match = "does not convert singing"):
        _convert(backend, mode = "singing")
    with pytest.raises(acc.ConvertRequestError):
        acc.served_model(_model("kokoro_tts"), "speech", {})


def test_a_runtime_refusal_is_an_audio_runtime_error(started):
    backend = _backend(_model("meanvc2"), started)
    backend._server.fail = srv.AudioCppRequestError(500, "MeanVC2 requires --voice-ref target")
    with pytest.raises(AudioRuntimeError, match = "MeanVC2 requires --voice-ref target") as info:
        _convert(backend)
    assert info.value.status == 500


def test_status_fields_describe_convert_per_family():
    rvc = acb.model_info_fields(_model("rvc", _RVC_OPTIONS))
    assert rvc["audio_workflows"] == ["convert"]
    assert rvc["audio_convert"] == {
        "modes": ["speech"],
        "target": "builtin",
        "builtin_voices": [
            {"id": "default", "label": "Default"},
            {"id": "manthos", "label": "Manthos"},
            {"id": "chocola", "label": "Chocola"},
            {"id": "fraise", "label": "Fraise"},
        ],
        "pitch": {"speech": {"auto": False, "shift_with_auto": False}},
        "style": False,
        "route_reloads": False,
        "source_max_seconds": 300,
    }
    assert rvc["audio_workflow_tasks"] == {"convert": "vc"}
    assert (rvc["audio_server_task"], rvc["audio_convert_route"]) == ("vc", None)
    assert [o["name"] for o in rvc["audio_options_by_workflow"]["convert"]] == [
        "retrieval_blend",
        "rms_mix_rate",
        "unvoiced_protection",
    ]
    assert rvc["audio_convert_rules"] == {"source_rate": 16000, "target_rate": None}

    seed_vc = acb.model_info_fields(_model("seed_vc", _SEED_VC_OPTIONS))
    assert seed_vc["audio_convert"]["modes"] == ["speech", "singing"]
    assert seed_vc["audio_convert"]["pitch"] == {"singing": {"auto": True, "shift_with_auto": True}}
    assert seed_vc["audio_convert"]["route_reloads"] is True
    assert seed_vc["audio_workflow_tasks"] == {"convert": "vc", "convert:singing": "svc"}
    assert seed_vc["audio_convert_route"] == "v2_vc"
    assert seed_vc["audio_convert_rules"] == {"source_rate": 44100, "target_rate": 44100}

    vevo2 = acb.model_info_fields(_model("vevo2"))
    assert vevo2["audio_workflows"] == ["clone", "convert"]
    assert vevo2["audio_convert"]["pitch"] == {
        "speech": {"auto": True, "shift_with_auto": False},
        "singing": {"auto": True, "shift_with_auto": False},
    }
    assert vevo2["audio_convert"]["style"] is True
    assert vevo2["audio_workflow_tasks"] == {
        "clone": "tts",
        "convert": "vc",
        "convert:singing": "svc",
    }
    assert vevo2["audio_server_task"] == "tts"

    chatterbox = acb.model_info_fields(_model("chatterbox"))
    assert chatterbox["audio_workflow_tasks"] == {"clone": "clon", "convert": "vc"}
    assert chatterbox["audio_convert"]["pitch"] == {}
    assert chatterbox["audio_convert_rules"] == {"source_rate": 16000, "target_rate": 24000}
    assert acb.model_info_fields(_model("meanvc2"))["audio_convert"]["pitch"] == {}
    tone_color = acb.model_info_fields(_model("tone_color_vc"))
    assert tone_color["audio_workflow_tasks"] == {"convert": "vc"}
    assert tone_color["audio_convert_rules"] == {"source_rate": 22050, "target_rate": 22050}

    kokoro = acb.model_info_fields(_model("kokoro_tts"))
    assert kokoro["audio_convert"] is None and kokoro["audio_options_by_workflow"] is None
    assert kokoro["audio_workflow_tasks"] == {"speak": "tts"}


def test_the_seed_vc_server_entry_carries_its_route(monkeypatch, tmp_path):
    seen: dict = {}
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")

    def spy(command, *_args, **_kwargs):
        with open(command[command.index("--config") + 1], encoding = "utf-8") as f:
            seen.update(json.load(f))
        raise OSError("not launched in tests")

    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    served, rest = acc.served_model(
        _model("seed_vc", _SEED_VC_OPTIONS), "speech", {"route": "v1_whisper_bigvgan_vc", "x": 1}
    )
    assert rest == {"x": 1}
    with pytest.raises(srv.AudioCppUnavailableError):
        srv.AudioCppServer.start(served, "/models/m.gguf")
    (entry,) = seen["models"]
    assert entry["task"] == "vc"
    assert entry["default_request_options"] == {"route": "v1_whisper_bigvgan_vc"}


def test_a_five_minute_source_transcript_fits_the_request():
    from models.inference import AudioRunInputs

    # ~150 words a minute for the 300 s Convert source cap is ~5000 characters.
    text = "word " * 1100
    inputs = AudioRunInputs(source = {"input_id": "a" * 32}, source_text = text)
    assert inputs.source_text == text
