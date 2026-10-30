# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import replace

import huggingface_hub
import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppModel, AudioCppVariant, RepoFile

REF = "/srv/accounts/a/audio/inputs/0123.24000.mono.m30.wav"
EMOTION = "/srv/accounts/a/audio/inputs/4567.24000.mono.m30.wav"
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."
T = "Okay, I'm Cemo and what you just heard wasn't a human voice."
ABSENT = object()
# Bound before any test swaps it for a recorder.
_REAL_START = srv.AudioCppServer.start


def _wav(rate = 24000) -> bytes:
    with wave.open(buf := io.BytesIO(), "wb") as w:
        w.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        w.writeframes(b"\x00\x00" * 240)
    return buf.getvalue()


# fmt: off
def _bare(display, family, variant, **fields) -> AudioCppModel:
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{display}", repo_id = AUDIO_CPP_REPO, folder = display,
        display_name = display, family = family, variant = variant, variants = (variant,),
        default_variant = "Q8_0", **fields,
    )


def _model(family, display = None, spec = None, **kw,) -> AudioCppModel:
    policy = acm.family_policy(family, spec, kw.get("names", ()))
    variant = AudioCppVariant("Q8_0", (RepoFile("m-q8_0.gguf", 1),), "m-q8_0.gguf")
    return _bare(
        display or family, family, variant, task = policy.task, server_task = policy.default_server_task,
        request_defaults = dict(policy.request_defaults), model_options = dict(policy.model_options),
        options = tuple(kw.get("options", ())), speaks = policy.speaks, clone = policy.clone,
        companions = policy.companions, required_inputs = acm._required_inputs(spec, None, policy),
    )


COSY_OPTIONS = ({
    "name": "template_name", "type": "enum", "description": "", "required": False, "default": None,
    "min": None, "max": None, "values": ["zero_shot", "cross_lingual", "instruct"],
},)
SPEED_OPTIONS = ({
    "name": "speed", "type": "float", "description": "", "required": False, "default": 1.0,
    "min": 0.5, "max": 2.0, "values": None,
},)
# fmt: on
QWEN = "Qwen3-TTS-12Hz-0.6B-Base-GGUF"
MODELS = {
    "qwen3": lambda: _model("qwen3_tts", QWEN, names = [QWEN, "qwen3-tts-12hz-0.6b-base-q8_0.gguf"]),
    "chatterbox": lambda: _model("chatterbox", "Chatterbox-GGUF"),
    "index": lambda: _model("index_tts2", "IndexTTS2-GGUF"),
    "vox": lambda: _model("voxcpm2", "VoxCPM2-GGUF"),
    "cosy": lambda: _model("cosyvoice3", "CosyVoice3-GGUF", options = COSY_OPTIONS),
    "f5": lambda: _model("f5_tts", "F5-TTS-GGUF"),
    "kokoro": lambda: _model("kokoro_tts", "Kokoro-82M-GGUF"),
    "vibevoice": lambda: _model("vibevoice", "VibeVoice-1.5B-GGUF"),
    "omnivoice": lambda: _model("omnivoice", "OmniVoice-GGUF"),
    "omnivoice_speed": lambda: _model("omnivoice", "OmniVoice-GGUF", options = SPEED_OPTIONS),
    "dots": lambda: _model("dots_tts", "DotTTS-SOAR-GGUF"),
    "irodori": lambda: _model("irodori_tts", "Irodori-TTS-v4-Small-GGUF"),
}


class _Recorder:
    model_id, backend = "studio-test", "cpu"
    alive = staticmethod(lambda: True)
    stop = staticmethod(lambda: None)

    def __init__(self, model):
        self.model, self.calls = model, []

    def post_json(self, path, payload, **_kwargs):
        self.calls.append((path, json.loads(json.dumps(payload))))
        if path == "/v1/tasks/run":
            body = {"audio": base64.b64encode(_wav(22050)).decode(), "sample_rate": 22050}
            return "application/json", json.dumps(body).encode()
        return "audio/wav", _wav()


def _backend(model):
    backend = acb.AudioCppBackend()
    server = _Recorder(model)
    backend._server, backend._model = server, model
    backend.models, backend.active_model_name = {model.id: {"is_audio": True}}, model.id
    return backend, server


def _clone(key, **kwargs):
    backend, server = _backend(MODELS[key]())
    inputs = kwargs.pop("audio_inputs", {"reference": REF})
    wav, rate = backend.generate_audio_response(
        TEXT, workflow = "clone", audio_inputs = inputs, seed = 7, **kwargs
    )
    assert wav[:4] == b"RIFF" and rate > 0
    (call,) = server.calls
    return call


BASE = {"model": "studio-test", "input": TEXT, "voice_ref": REF, "seed": "7"}
SPEECH, TASKS = "/v1/audio/speech", "/v1/tasks/run"


# fmt: off
@pytest.mark.parametrize("key, kwargs, expect", [
    ("qwen3", {"reference_text": T, "language": "en"}, {**BASE, "reference_text": T, "language": "English"}),
    ("chatterbox", {"reference_text": T, "audio_options": {"exaggeration": 0.7, "guidance_scale": 9, "temperature": 0.1}},
     {**BASE, "options": {"exaggeration": "0.7", "guidance_scale": "5"}}),
    ("index", {"audio_inputs": {"reference": REF, "emotion": EMOTION}, "audio_options": {"emotion_alpha": 0.7}},
     {"path": TASKS, "model": "studio-test", "text": TEXT, "voice_ref": REF, "audio": EMOTION,
      "options": {"emotion_alpha": "0.7"}, "seed": "7"}),
    ("qwen3", {"language": "Japanese"}, {"language": "Japanese"}),
    ("qwen3", {"language": "pt-BR"}, {"language": "Portuguese"}),
    ("qwen3", {"language": "auto"}, {"language": ABSENT}),
    ("qwen3", {"language": "xx"}, {"language": ABSENT}),
    ("qwen3", {"reference_text": T, "audio_options": {"x_vector_only_mode": True, "bogus_key": "1"}},
     {"options": {"x_vector_only_mode": "true"}, "reference_text": ABSENT}),
    ("qwen3", {"reference_text": T, "speed": 1.3}, {"speed": ABSENT}),
    ("index", {"reference_text": T, "audio_options": {"emotion_vector": "0.8,0,0,0,0,0,0.2,0", "emotion_alpha": 0.7}},
     {"options": {"emotion_vector": "0.8,0,0,0,0,0,0.2,0", "emotion_alpha": "0.7"}, "reference_text": ABSENT}),
    ("index", {"audio_options": {"use_emotion_text": True, "emotion_text": "Excited."}},
     {"options": {"use_emotion_text": "true", "emotion_text": "Excited."}}),
    ("vox", {"audio_inputs": {"reference": REF, "emotion": EMOTION}}, {"audio": ABSENT}),
    ("cosy", {"instructions": "Sadly.", "audio_options": {"template_name": "instruct", "bogus_key": "x"}},
     {"options": {"template_name": "instruct"}, "instructions": "Sadly."}),
    ("cosy", {"reference_text": T}, {"reference_text": T, "instructions": ABSENT}),
    ("cosy", {"reference_text": T, "audio_options": {"template_name": "cross_lingual"}}, {"reference_text": ABSENT}),
    ("cosy", {"reference_text": T, "instructions": "Whisper.", "audio_options": {"template_name": "instruct"}},
     {"reference_text": T}),
    ("f5", {"reference_text": T, "speed": 1.3}, {"speed": 1.3, "reference_text": T, "options": ABSENT}),
    ("f5", {"reference_text": T, "speed": 9}, {"speed": 2.0}),
    ("omnivoice", {"reference_text": T}, {**BASE, "reference_text": T}),
    ("omnivoice_speed", {"reference_text": T, "speed": 1.7}, {"speed": ABSENT, "options": {"speed": "1.7"}}),
    ("omnivoice_speed", {"reference_text": T, "speed": 1.7, "audio_options": {"speed": 0.8}},
     {"options": {"speed": "0.8"}}),
    ("dots", {"reference_text": T}, {**BASE, "reference_text": T}),
    ("dots", {}, BASE),
    # Irodori refuses a transcript as an unknown option.
    ("irodori", {"reference_text": T}, BASE),
])
# fmt: on
def test_clone_request_body(key, kwargs, expect):
    path, body = _clone(key, **kwargs)
    expect = dict(expect)
    assert path == expect.pop("path", SPEECH)
    assert {k: body.get(k, ABSENT) for k in expect} == expect
    if "model" in expect:
        assert body.keys() == expect.keys()


def test_a_top_level_speed_reaches_a_speak_model_as_its_speed_option():
    backend, server = _backend(_model("kokoro_tts", "Kokoro-82M-GGUF", options = SPEED_OPTIONS))
    backend.generate_audio_response(TEXT, seed = 7, speed = 1.5)
    backend.generate_audio_response(TEXT, seed = 7)
    assert [body["options"] for _, body in server.calls] == [{"speed": 1.5}, {"speed": 1.0}]


def test_voxcpm2_speak_body_is_unchanged_by_clone_support():
    backend, server = _backend(MODELS["vox"]())
    backend.generate_audio_response(TEXT, seed = 7, language = "en", instructions = "warm")
    body = {"model": "studio-test", "input": TEXT, "options": {"instruct": "warm", "language": "en"}}
    assert server.calls == [(SPEECH, {**body, "seed": "7"})]


class _StrictRecorder(_Recorder):
    """Refuses an option the model's contract does not declare, as the runtime does."""

    def post_json(self, path, payload, **kwargs):
        keys = self.model.request_keys
        undeclared = sorted(set(payload.get("options") or {}) - set(keys or ()))
        if keys is not None and undeclared:
            self.calls.append((path, json.loads(json.dumps(payload))))
            raise srv.AudioCppRequestError(500, f"unknown request option: {undeclared[0]}")
        return super().post_json(path, payload, **kwargs)


# Request option names each published GGUF's embedded (strict) spec declares.
STRICT_KEYS = {
    "Breeze-TTS-2-GGUF": ("breeze_tts", {"instruction", "reference_text", "guidance_scale", "seed"}),
    "Irodori-TTS-v4-Small-GGUF": ("irodori_tts", {"instruction", "language", "no_ref", "seed"}),
    "DramaBox-GGUF": ("dramabox", {"negative_prompt", "target_voice", "guidance_scale", "seed"}),
    "DotTTS-MF-GGUF": ("dots_tts", {"language", "reference_text", "template_name", "seed"}),
    "DotTTS-Edit-GGUF": ("dots_tts", {"instruct", "instruction", "language", "template_name"}),
    "Maya1-GGUF": ("maya1", {"instruct", "max_tokens", "temperature", "seed"}),
}


# fmt: off
@pytest.mark.parametrize("display, language, options", [
    ("Breeze-TTS-2-GGUF", None, {"instruction": "warm"}),
    ("Breeze-TTS-2-GGUF", "en", {"instruction": "warm"}),
    ("Irodori-TTS-v4-Small-GGUF", None, {"instruction": "warm"}),
    ("Irodori-TTS-v4-Small-GGUF", "ja", {"instruction": "warm", "language": "ja"}),
    ("DramaBox-GGUF", "en", ABSENT),
    ("DotTTS-MF-GGUF", "en", {"language": "en"}),
    ("DotTTS-Edit-GGUF", None, {"instruct": "warm"}),
    ("Maya1-GGUF", "en", {"instruct": "warm"}),
])
# fmt: on
def test_speak_sends_the_description_under_the_name_the_contract_declares(display, language, options):
    family, keys = STRICT_KEYS[display]
    model = replace(_model(family, display), request_keys = frozenset(keys))
    backend = acb.AudioCppBackend()
    server = _StrictRecorder(model)
    backend._server, backend._model = server, model
    backend.models, backend.active_model_name = {model.id: {"is_audio": True}}, model.id
    backend.generate_audio_response(TEXT, language = language, instructions = "warm")
    # One request: nothing the contract lacks is sent, so there is no refusal to retry after.
    ((path, body),) = server.calls
    assert path == SPEECH and body.get("options", ABSENT) == options


def _spec(*names, strict = True):
    spec = {"family": "dots_tts", "options": {"request": [{"name": n, "type": "string"} for n in names]}}
    return {**spec, "schema_version": 1} if strict else spec


# fmt: off
@pytest.mark.parametrize("runtime, embedded, keys", [
    # DotTTS-MF: model_specs/ declares instruct, the GGUF's v1 contract (what the server checks) does not.
    (_spec("instruct", "instruction", "language"), _spec("language", "seed"), {"language", "seed"}),
    # A legacy GGUF is checked against model_specs/, so that list still wins.
    (_spec("instruct", "language"), _spec("language", strict = False), {"instruct", "language"}),
    # A v1 GGUF that lists no request options leaves the runtime's list in charge, as before.
    (_spec("instruct", "language"), {**_spec(), "options": {}}, {"instruct", "language"}),
    (None, _spec("language"), {"language"}),
    (None, _spec("language", strict = False), None),
])
# fmt: on
def test_request_keys_follow_the_contract_the_server_checks(runtime, embedded, keys):
    assert acm._request_keys(runtime, embedded) == (frozenset(keys) if keys else None)


# fmt: off
@pytest.mark.parametrize("runtime, embedded, keys", [
    # v0.9.2 checks model_specs/ before the GGUF's v1 contract.
    (_spec("instruct", "instruction", "language"), _spec("language", "seed"), {"instruct", "instruction", "language"}),
    # A legacy model_specs/ entry is not strict, even over a v1 GGUF.
    (_spec("instruct", "language", strict = False), _spec("language", "seed"), None),
    # No runtime spec, or one without request options: the GGUF's own contract.
    (None, _spec("language"), {"language"}),
    ({**_spec(), "options": {}}, _spec("language"), {"language"}),
    (None, _spec("language", strict = False), None),
])
# fmt: on
def test_request_keys_on_a_spec_first_runtime(runtime, embedded, keys):
    got = acm._request_keys(runtime, embedded, runtime_first = True)
    assert got == (frozenset(keys) if keys else None)


def test_speak_on_a_runtime_with_model_specs_sends_only_what_the_gguf_declares():
    runtime = _spec("instruct", "instruction", "language")
    embedded = _spec("language", "template_name", "seed")
    model = replace(_model("dots_tts", "DotTTS-MF-GGUF"), request_keys = acm._request_keys(runtime, embedded))
    backend = acb.AudioCppBackend()
    server = _StrictRecorder(model)
    backend._server, backend._model = server, model
    backend.models, backend.active_model_name = {model.id: {"is_audio": True}}, model.id
    backend.generate_audio_response(TEXT, language = "en", instructions = "warm")
    ((path, body),) = server.calls
    assert path == SPEECH and body["options"] == {"language": "en"}


# fmt: off
@pytest.mark.parametrize("key, text, sent", [
    ("vibevoice", "Hello there.", "Speaker 1: Hello there."),
    ("vibevoice", "Speaker 1: Hi.\nSpeaker 2: Hello.", "Speaker 1: Hi.\nSpeaker 2: Hello."),
    ("vibevoice", "  speaker 2 : already labelled", "  speaker 2 : already labelled"),
    ("kokoro", "Hello there.", "Hello there."),
])
# fmt: on
def test_speaker_label_added_only_for_unlabelled_vibevoice(key, text, sent):
    backend, server = _backend(MODELS[key]())
    backend.generate_audio_response(text)
    assert server.calls[0][1]["input"] == sent


def test_a_speak_only_model_refuses_a_clone_request():
    backend, server = _backend(MODELS["kokoro"]())
    with pytest.raises(RuntimeError, match = "cannot clone"):
        backend.generate_audio_response(TEXT, workflow = "clone", audio_inputs = {"reference": REF})
    assert server.calls == []


def test_status_fields_name_the_clone_workflow_and_transcript_rule():
    f = acb.model_info_fields(MODELS["qwen3"]())
    assert (f["audio_workflows"], f["audio_reference_text"]) == (["clone"], "required")
    assert f["audio_required_inputs"] == []
    assert f["audio_clone"]["reference_text_waived"] == [["x_vector_only_mode", ["true"]]]
    for key, workflows, rt in (
        ("vox", ["speak", "clone"], "optional"),
        ("omnivoice", ["speak", "clone"], "required"),
        ("irodori", ["speak", "clone"], "unused"),
        ("kokoro", ["speak"], None),
    ):
        f = acb.model_info_fields(MODELS[key]())
        assert (f["audio_workflows"], f["audio_reference_text"]) == (workflows, rt)
    cosy = acb.model_info_fields(MODELS["cosy"]())["audio_clone"]
    assert cosy["reference_text_waived"] == [["template_name", ["cross_lingual", "instruct"]]]
    instruct = {"name": "instruct", "type": "string", "required": True}
    maya = _model("maya1", "Maya1-GGUF", {"tasks": ["tts"], "options": {"request": [instruct]}})
    assert acb.model_info_fields(maya)["audio_required_inputs"] == ["instruct"]
    # Models are cache keys; tool option dicts must stay out of the hash.
    for model in (MODELS["qwen3"](), _model("chatterbox"), _model("index_tts2"), _model("miotts")):
        hash(model)
    hash(acm.FAMILIES["chatterbox"])
    for model in (MODELS["qwen3"](), _model("chatterbox"), _model("index_tts2"), _model("miotts")):
        hash(model)
    hash(acm.FAMILIES["chatterbox"])


def _server_entry(monkeypatch, tmp_path, model):
    seen = {}
    (binary := tmp_path / srv.BINARY_NAME).write_bytes(b"")

    def spy(command, *_args, **_kwargs):
        with open(command[command.index("--config") + 1], encoding = "utf-8") as f:
            seen.update(json.load(f))
        raise OSError("not launched in tests")

    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    with pytest.raises(srv.AudioCppUnavailableError):
        _REAL_START(model, "/models/m-q8_0.gguf")
    (entry,) = seen["models"]
    return entry


# fmt: off
@pytest.mark.parametrize("family, display, task", [
    ("chatterbox", "Chatterbox-GGUF", "clon"), ("chatterbox_turbo", "Chatterbox-Turbo-GGUF", "tts"),
])
# fmt: on
def test_chatterbox_server_task(monkeypatch, tmp_path, family, display, task):
    model = _model(family, display)
    assert model.server_task == task
    if task == "tts":
        assert list(model.workflows) == ["speak"]
    entry = _server_entry(monkeypatch, tmp_path, model)
    assert (entry["family"], entry["task"]) == (family, task)


def test_miotts_downloads_mio_codec_and_starts_with_its_path(monkeypatch, tmp_path):
    model = _model("miotts", "MioTTS-1.7B-GGUF")
    assert list(model.workflows) == ["clone"]
    codec_file = "MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf"
    codec_variant = AudioCppVariant("Q8_0", (RepoFile(codec_file, 299),), codec_file)
    codec = _bare(
        "MioCodec-25Hz-44.1kHz-v2-GGUF", "miocodec", codec_variant, task = "", server_task = "",
        unsupported = "MioCodec is the codec MioTTS loads with; pick MioTTS to use it.",
    )
    resolved, downloads, started = [], [], []

    def resolve(identifier, *args, **_kw):
        resolved.append((identifier, *args[:1]))
        return codec if identifier == codec.id else None

    codec_path = str(tmp_path / "farm" / "miocodec-25hz-44khz-v2-q8_0.gguf")
    is_codec = lambda m: m.family == "miocodec"
    monkeypatch.setattr(acb, "resolve", resolve)
    monkeypatch.setattr(audio_cpp_files, "missing_files", lambda m: [(codec_file, 1)] * is_codec(m))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda r, p, **kw: downloads.append((r, p)))
    tts_path = str(tmp_path / "farm" / "miotts.gguf")
    monkeypatch.setattr(audio_cpp_files, "materialize", lambda m: codec_path if is_codec(m) else tts_path)
    start = lambda served, path, **_kw: started.append((served, path)) or _Recorder(served)
    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    backend = acb.AudioCppBackend()
    backend._ensure_downloaded(model, None)
    assert downloads == [(AUDIO_CPP_REPO, codec_file)]
    assert (codec.id, "Q8_0") in resolved
    backend._start_server(model)
    ((served, path),) = started
    assert path.endswith("miotts.gguf")
    assert served.model_options["session_options"] == {"miotts.codec_model_path": codec_path}
    assert served == model
    entry = _server_entry(monkeypatch, tmp_path, replace(model, model_options = served.model_options))
    assert entry["session_options"] == {"miotts.codec_model_path": codec_path}
    assert entry["task"] == "tts"
    kokoro = MODELS["kokoro"]()
    assert acb._with_companions(kokoro) is kokoro


def test_a_blank_clone_seed_is_picked_by_the_server(monkeypatch):
    monkeypatch.setattr(acb.cm, "random_seed", lambda: 99)
    backend, server = _backend(MODELS["chatterbox"]())
    backend.generate_audio_response(TEXT, workflow = "clone", audio_inputs = {"reference": REF})
    assert server.calls[-1][1]["seed"] == "99" and backend.last_generation_stats["seed"] == 99
