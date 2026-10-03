# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact requests audio.cpp receives for a voice clone, per family (spike S1's bodies).

A fake server records every ``post_json``; no runtime runs. The reference is always a
server-local path the route resolved, sent as ``voice_ref``; options the family's spec does not
declare reach the runtime only when the family's Studio tool schema lists them.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AudioCppModel,
    AudioCppVariant,
    RepoFile,
)

REF = "/srv/accounts/a/audio/inputs/0123.24000.mono.m30.wav"
EMOTION = "/srv/accounts/a/audio/inputs/4567.24000.mono.m30.wav"
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."
TRANSCRIPT = "Okay, I'm Cemo and what you just heard wasn't a human voice."
# Bound before any test swaps it for a recorder.
_REAL_START = srv.AudioCppServer.start


def _wav(rate = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * 240)
    return buf.getvalue()


def _model(
    family,
    names = (),
    options = (),
    spec = None,
    display = None,
) -> AudioCppModel:
    policy = acm.family_policy(family, spec, names)
    variant = AudioCppVariant("Q8_0", (RepoFile("m-q8_0.gguf", 1),), "m-q8_0.gguf")
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{display or family}",
        repo_id = AUDIO_CPP_REPO,
        folder = display or family,
        display_name = display or family,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        request_defaults = dict(policy.request_defaults),
        model_options = dict(policy.model_options),
        options = tuple(options),
        speaks = policy.speaks,
        clone = policy.clone,
        companions = policy.companions,
        required_inputs = acm._required_inputs(spec, None, policy),
    )


class _Recorder:
    model_id = "studio-test"

    def __init__(self, model):
        self.model = model
        self.calls: list[tuple[str, dict]] = []

    def alive(self):
        return True

    def stop(self):
        pass

    def post_json(self, path, payload, **_kwargs):
        self.calls.append((path, json.loads(json.dumps(payload))))
        if path == "/v1/tasks/run":
            body = {"audio": base64.b64encode(_wav(22050)).decode(), "sample_rate": 22050}
            return "application/json", json.dumps(body).encode()
        return "audio/wav", _wav()


def _backend(model):
    backend = acb.AudioCppBackend()
    server = _Recorder(model)
    backend._server = server
    backend._model = model
    backend.models = {model.id: {"is_audio": True}}
    backend.active_model_name = model.id
    return backend, server


def _clone(model, **kwargs):
    backend, server = _backend(model)
    inputs = kwargs.pop("audio_inputs", {"reference": REF})
    wav, rate = backend.generate_audio_response(
        TEXT, workflow = "clone", audio_inputs = inputs, seed = 7, **kwargs
    )
    assert wav[:4] == b"RIFF" and rate > 0
    (call,) = server.calls
    return call


def _qwen3_base():
    return _model(
        "qwen3_tts",
        names = ["Qwen3-TTS-12Hz-0.6B-Base-GGUF", "qwen3-tts-12hz-0.6b-base-q8_0.gguf"],
        display = "Qwen3-TTS-12Hz-0.6B-Base-GGUF",
    )


def test_qwen3_base_sends_the_reference_its_transcript_and_a_language_name():
    path, body = _clone(_qwen3_base(), reference_text = TRANSCRIPT, language = "en")
    assert path == "/v1/audio/speech"
    assert body == {
        "model": "studio-test",
        "input": TEXT,
        "voice_ref": REF,
        "reference_text": TRANSCRIPT,
        "language": "English",
        "seed": "7",
    }
    # Full names pass through; Auto and an unknown code are left out rather than refused at runtime.
    assert _clone(_qwen3_base(), language = "Japanese")[1]["language"] == "Japanese"
    assert _clone(_qwen3_base(), language = "pt-BR")[1]["language"] == "Portuguese"
    assert "language" not in _clone(_qwen3_base(), language = "auto")[1]
    assert "language" not in _clone(_qwen3_base(), language = "xx")[1]


def test_qwen3_timbre_only_sends_the_flag_as_a_string_and_no_transcript():
    _path, body = _clone(
        _qwen3_base(),
        reference_text = TRANSCRIPT,
        audio_options = {"x_vector_only_mode": True, "bogus_key": "1"},
    )
    assert body["options"] == {"x_vector_only_mode": "true"}
    assert "reference_text" not in body


def test_chatterbox_loads_as_a_cloning_session_and_sends_its_expressiveness():
    model = _model("chatterbox", display = "Chatterbox-GGUF")
    assert model.server_task == "clon"
    _path, body = _clone(
        model,
        reference_text = TRANSCRIPT,
        audio_options = {"exaggeration": 0.7, "guidance_scale": 9, "temperature": 0.1},
    )
    # The transcript is unused, guidance is clamped to its 0-5 range, undeclared options are dropped.
    assert body == {
        "model": "studio-test",
        "input": TEXT,
        "voice_ref": REF,
        "options": {"exaggeration": "0.7", "guidance_scale": "5"},
        "seed": "7",
    }
    turbo = _model("chatterbox_turbo", display = "Chatterbox-Turbo-GGUF")
    assert turbo.server_task == "tts" and list(turbo.workflows) == ["speak"]


def _server_entry(
    monkeypatch,
    tmp_path,
    model,
    model_options = None,
):
    """The model entry ``AudioCppServer.start`` writes to the server config."""
    seen = {}
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")

    def spy(command, *_args, **_kwargs):
        config = command[command.index("--config") + 1]
        with open(config, encoding = "utf-8") as f:
            seen.update(json.load(f))
        raise OSError("not launched in tests")

    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    if model_options is not None:
        from dataclasses import replace
        model = replace(model, model_options = model_options)
    with pytest.raises(srv.AudioCppUnavailableError):
        _REAL_START(model, "/models/m-q8_0.gguf")
    (entry,) = seen["models"]
    return entry


def test_the_chatterbox_server_entry_asks_for_the_clon_task(monkeypatch, tmp_path):
    entry = _server_entry(monkeypatch, tmp_path, _model("chatterbox", display = "Chatterbox-GGUF"))
    assert (entry["family"], entry["task"]) == ("chatterbox", "clon")
    turbo = _server_entry(
        monkeypatch, tmp_path, _model("chatterbox_turbo", display = "Chatterbox-Turbo-GGUF")
    )
    assert turbo["task"] == "tts"


def test_index_tts2_sends_the_emotion_mixer_on_the_speech_endpoint():
    model = _model("index_tts2", display = "IndexTTS2-GGUF")
    _path, body = _clone(
        model,
        reference_text = TRANSCRIPT,
        audio_options = {"emotion_vector": "0.8,0,0,0,0,0,0.2,0", "emotion_alpha": 0.7},
    )
    assert _path == "/v1/audio/speech"
    assert body["options"] == {"emotion_vector": "0.8,0,0,0,0,0,0.2,0", "emotion_alpha": "0.7"}
    assert "reference_text" not in body
    _path, text_emotion = _clone(
        model, audio_options = {"use_emotion_text": True, "emotion_text": "Excited and happy."}
    )
    assert text_emotion["options"] == {
        "use_emotion_text": "true",
        "emotion_text": "Excited and happy.",
    }


def test_index_tts2_emotion_audio_goes_to_the_tasks_endpoint_with_top_level_audio():
    model = _model("index_tts2", display = "IndexTTS2-GGUF")
    path, body = _clone(
        model,
        audio_inputs = {"reference": REF, "emotion": EMOTION},
        audio_options = {"emotion_alpha": 0.7},
    )
    assert path == "/v1/tasks/run"
    assert body == {
        "model": "studio-test",
        "text": TEXT,
        "voice_ref": REF,
        "audio": EMOTION,
        "options": {"emotion_alpha": "0.7"},
        "seed": "7",
    }


def test_an_emotion_clip_is_ignored_by_a_family_without_emotion_audio():
    path, body = _clone(
        _model("voxcpm2", display = "VoxCPM2-GGUF"),
        audio_inputs = {"reference": REF, "emotion": EMOTION},
    )
    assert path == "/v1/audio/speech" and "audio" not in body


COSY_OPTIONS = (
    {
        "name": "template_name",
        "type": "enum",
        "description": "",
        "required": False,
        "default": None,
        "min": None,
        "max": None,
        "values": ["zero_shot", "cross_lingual", "instruct"],
    },
)


def test_cosyvoice3_instruct_sends_top_level_instructions():
    model = _model("cosyvoice3", options = COSY_OPTIONS, display = "CosyVoice3-GGUF")
    _path, body = _clone(
        model,
        instructions = "Speak very slowly and sadly.",
        audio_options = {"template_name": "instruct", "bogus_key": "x"},
    )
    # Schema-strict family: only the declared template reaches it.
    assert body["options"] == {"template_name": "instruct"}
    assert body["instructions"] == "Speak very slowly and sadly."
    _path, zero_shot = _clone(model, reference_text = TRANSCRIPT)
    assert zero_shot["reference_text"] == TRANSCRIPT and "instructions" not in zero_shot


def test_f5_sends_a_top_level_speed_and_its_transcript():
    model = _model("f5_tts", display = "F5-TTS-GGUF")
    _path, body = _clone(model, reference_text = TRANSCRIPT, speed = 1.3)
    assert body["speed"] == 1.3 and body["reference_text"] == TRANSCRIPT
    assert "options" not in body
    assert _clone(model, reference_text = TRANSCRIPT, speed = 9)[1]["speed"] == 2.0
    # Families without a speed field never get one.
    assert "speed" not in _clone(_qwen3_base(), reference_text = TRANSCRIPT, speed = 1.3)[1]


def test_voxcpm2_speak_body_is_unchanged_by_clone_support():
    model = _model("voxcpm2", display = "VoxCPM2-GGUF")
    backend, server = _backend(model)
    backend.generate_audio_response(TEXT, seed = 7, language = "en", instructions = "warm")
    ((path, body),) = server.calls
    assert path == "/v1/audio/speech"
    assert body == {
        "model": "studio-test",
        "input": TEXT,
        "options": {"instruct": "warm", "language": "en"},
        "seed": "7",
    }


@pytest.mark.parametrize(
    "text, sent",
    [
        ("Hello there.", "Speaker 1: Hello there."),
        ("Speaker 1: Hi.\nSpeaker 2: Hello.", "Speaker 1: Hi.\nSpeaker 2: Hello."),
        ("  speaker 2 : already labelled", "  speaker 2 : already labelled"),
    ],
)
def test_vibevoice_gets_a_speaker_label_only_when_it_has_none(text, sent):
    model = _model("vibevoice", display = "VibeVoice-1.5B-GGUF")
    backend, server = _backend(model)
    backend.generate_audio_response(text)
    assert server.calls[0][1]["input"] == sent
    # Other families speak the text as written.
    kokoro, other = _backend(_model("kokoro_tts", display = "Kokoro-82M-GGUF"))
    kokoro.generate_audio_response("Hello there.")
    assert other.calls[0][1]["input"] == "Hello there."


def test_a_speak_only_model_refuses_a_clone_request():
    backend, server = _backend(_model("kokoro_tts", display = "Kokoro-82M-GGUF"))
    with pytest.raises(RuntimeError, match = "cannot clone"):
        backend.generate_audio_response(TEXT, workflow = "clone", audio_inputs = {"reference": REF})
    assert server.calls == []


def test_status_fields_name_the_clone_workflow_and_transcript_rule():
    fields = acb.model_info_fields(_qwen3_base())
    assert fields["audio_workflows"] == ["clone"]
    assert fields["audio_reference_text"] == "required"
    assert fields["audio_required_inputs"] == []
    assert fields["audio_clone"]["reference_text_waived"] == [["x_vector_only_mode", ["true"]]]
    vox = acb.model_info_fields(_model("voxcpm2", display = "VoxCPM2-GGUF"))
    assert (
        vox["audio_workflows"] == ["speak", "clone"] and vox["audio_reference_text"] == "optional"
    )
    kokoro = acb.model_info_fields(_model("kokoro_tts", display = "Kokoro-82M-GGUF"))
    assert kokoro["audio_workflows"] == ["speak"] and kokoro["audio_reference_text"] is None
    maya = _model(
        "maya1",
        spec = {
            "tasks": ["tts"],
            "options": {"request": [{"name": "instruct", "type": "string", "required": True}]},
        },
        display = "Maya1-GGUF",
    )
    assert acb.model_info_fields(maya)["audio_required_inputs"] == ["instruct"]


def test_miotts_downloads_mio_codec_and_starts_with_its_path(monkeypatch, tmp_path):
    model = _model("miotts", display = "MioTTS-1.7B-GGUF")
    assert list(model.workflows) == ["clone"]
    codec_variant = AudioCppVariant(
        "Q8_0",
        (RepoFile("MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf", 299),),
        "MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf",
    )
    codec = AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/MioCodec-25Hz-44.1kHz-v2-GGUF",
        repo_id = AUDIO_CPP_REPO,
        folder = "MioCodec-25Hz-44.1kHz-v2-GGUF",
        display_name = "MioCodec-25Hz-44.1kHz-v2-GGUF",
        family = "miocodec",
        task = "",
        server_task = "",
        variant = codec_variant,
        variants = (codec_variant,),
        default_variant = "Q8_0",
        unsupported = "MioCodec is the codec MioTTS loads with; pick MioTTS to use it.",
    )
    resolved = []

    def resolve(
        identifier,
        variant = None,
        hf_token = None,
        **kwargs,
    ):
        resolved.append((identifier, variant))
        return codec if identifier == codec.id else None

    downloads = []
    monkeypatch.setattr(acb, "resolve", resolve)
    monkeypatch.setattr(
        audio_cpp_files,
        "missing_files",
        lambda m: [(m.variant.primary, 1)] if m.family == "miocodec" else [],
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, path, **kw: downloads.append((repo, path))
    )
    codec_path = str(tmp_path / "farm" / "miocodec-25hz-44khz-v2-q8_0.gguf")
    monkeypatch.setattr(
        audio_cpp_files,
        "materialize",
        lambda m: codec_path if m.family == "miocodec" else str(tmp_path / "farm" / "miotts.gguf"),
    )
    started = []

    def start(served, path, **kwargs):
        started.append((served, path))
        server = _Recorder(served)
        server.backend = "cpu"
        return server

    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    backend = acb.AudioCppBackend()
    backend._ensure_downloaded(model, None)
    assert downloads == [(AUDIO_CPP_REPO, codec_variant.primary)]
    assert (codec.id, "Q8_0") in resolved
    backend._start_server(model)
    ((served, path),) = started
    assert path.endswith("miotts.gguf")
    assert served.model_options["session_options"] == {"miotts.codec_model_path": codec_path}
    # The server equals the model it was asked for, so the next request does not restart it.
    assert served == model
    entry = _server_entry(monkeypatch, tmp_path, model, served.model_options)
    assert entry["session_options"] == {"miotts.codec_model_path": codec_path}
    assert entry["task"] == "tts"


def test_a_model_without_companions_starts_unchanged(monkeypatch, tmp_path):
    model = _model("kokoro_tts", display = "Kokoro-82M-GGUF")
    assert acb._with_companions(model) is model


def test_clone_models_stay_hashable():
    # Models are compared and hashed as cache keys; the tool option dicts are kept out of the hash.
    for model in (_qwen3_base(), _model("chatterbox"), _model("index_tts2"), _model("miotts")):
        hash(model)
    hash(acm.FAMILIES["chatterbox"])


def test_cosyvoice3_transcript_rules_per_template():
    model = _model("cosyvoice3", options = COSY_OPTIONS, display = "CosyVoice3-GGUF")
    # Cross-lingual sends no transcript; instruct keeps one that was given.
    cross = _clone(
        model, reference_text = TRANSCRIPT, audio_options = {"template_name": "cross_lingual"}
    )[1]
    assert "reference_text" not in cross
    instruct = _clone(
        model,
        reference_text = TRANSCRIPT,
        instructions = "Whisper.",
        audio_options = {"template_name": "instruct"},
    )[1]
    assert instruct["reference_text"] == TRANSCRIPT
    rules = acb.model_info_fields(model)["audio_clone"]
    assert rules["reference_text_waived"] == [["template_name", ["cross_lingual", "instruct"]]]
