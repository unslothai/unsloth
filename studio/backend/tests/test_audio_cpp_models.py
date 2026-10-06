# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp GGUFs: recognising them by header, resolving ids and variants, their files in the HF
cache, and the worker backend's request shaping."""

import asyncio
import base64
import io
import json
import os
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AudioCppModel,
    AudioCppVariant,
    RepoFile,
)

# ---------------------------------------------------------------------------
# Fixtures: GGUF headers and a fake HF cache


def _gguf_bytes(
    arch = "audiocpp",
    family = None,
    spec = None,
    *,
    blob_before_spec = 0,
    extra = (),
) -> bytes:
    kv = [("general.architecture", 8, arch)]
    kv.append(("general.name", 8, "test"))
    if blob_before_spec:
        # audiocpp.embedded_files.data: a u8 array some writers place before the spec keys.
        kv.append(("audiocpp.embedded_files.data", 9, blob_before_spec))
    if family is not None:
        kv.append(("audiocpp.model_spec.version", 4, 1))
        kv.append(("audiocpp.model_spec.family", 8, family))
    if spec is not None:
        kv.append(("audiocpp.model_spec.json", 8, json.dumps(spec)))
    kv.extend(extra)
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)))
    for key, vtype, value in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            out += struct.pack("<Q", len(v)) + v
        elif vtype == 4:
            out += struct.pack("<I", value)
        elif vtype == 9:
            out += struct.pack("<IQ", 0, value) + b"\0" * value
    return bytes(out) + b"\0" * 64


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An empty HF hub cache every audio.cpp lookup reads, with the Hub and runtime specs out of reach."""
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    yield root
    acm.forget()


def _snapshot(
    root,
    repo = AUDIO_CPP_REPO,
    sha = "a" * 40,
    main = True,
):
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents = True, exist_ok = True)
    if main:
        (repo_dir / "refs").mkdir(exist_ok = True)
        (repo_dir / "refs" / "main").write_text(sha)
    return snap


def _put(
    snap,
    rel,
    data = b"x",
):
    path = snap / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)
    return path


KOKORO_SPEC = {
    "family": "kokoro_tts",
    "tasks": ["tts"],
    "ui": {"default_voice": "af_heart", "builtin_voices": ["af_heart", "am_adam"]},
    "options": {
        "request": [
            {"name": "language", "type": "string"},
            {"name": "seed", "type": "int"},
            {"name": "speed", "type": "float", "default": 1, "min": 0.5, "max": 2},
            {"name": "phonemes", "type": "string_list"},
            {"name": "text_chunk_size", "type": "int", "default": 240, "min": 1},
            {"name": "mode", "type": "enum", "values": ["a", "b"], "default": "a"},
            {"name": "voice_dir", "type": "path"},
        ]
    },
}


def _kokoro(root):
    snap = _snapshot(root)
    _put(
        snap,
        "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
        _gguf_bytes(family = "kokoro_tts", spec = KOKORO_SPEC),
    )
    _put(snap, "Kokoro-82M-GGUF/README.md", b"readme")
    return snap


def _model(
    folder,
    family,
    task,
    *,
    files = None,
    primary = None,
    key = "Q8_0",
    **kw,
):
    files = files or (RepoFile(f"{folder}/{folder.lower()}-q8_0.gguf", 4),)
    primary = files[0].path if primary is None else primary
    variant = AudioCppVariant(key, tuple(files), primary)
    fields = dict(
        id = f"{AUDIO_CPP_REPO}/{folder}",
        repo_id = AUDIO_CPP_REPO,
        folder = folder,
        display_name = folder,
        family = family,
        task = task,
        server_task = "gen" if task == "music" else task,
        variant = variant,
        variants = (variant,),
        default_variant = key,
    )
    fields.update(kw)
    return AudioCppModel(**fields)


CANARY = _model("Canary-180M-Flash-GGUF", "canary_asr", "asr")


# ---------------------------------------------------------------------------
# Ids


def test_ids_legacy_keys_and_subfolders_parse_without_io():
    ref = acm.parse_identifier("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF")
    assert (ref.repo_id, ref.folder, ref.variant_hint) == (AUDIO_CPP_REPO, "Kokoro-82M-GGUF", None)
    assert ref.display_name == "Kokoro-82M-GGUF"
    legacy = acm.parse_identifier("audiocpp-ace-step-1.5-turbo")
    assert legacy.id == f"{AUDIO_CPP_REPO}/ACE-Step1.5-GGUF" and legacy.variant_hint == "turbo/Q8_0"
    # The sub-folder ids of earlier builds name the folder row and a variant.
    old = acm.parse_identifier("audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english")
    assert old.id == f"{AUDIO_CPP_REPO}/PocketTTS-GGUF" and old.variant_hint == "english"
    moon = acm.parse_identifier("Audio-CPP/audio.cpp-gguf/Moonshine-Streaming-GGUF/tiny")
    assert moon.variant_hint == "tiny/Q8_0"
    dedicated = acm.parse_identifier("audio-cpp/MiniMax-Music3-GGUF")
    assert (dedicated.repo_id, dedicated.folder) == ("audio-cpp/MiniMax-Music3-GGUF", "")
    assert dedicated.display_name == "MiniMax-Music3-GGUF"
    for junk in (None, "", "no-slash", "audio-cpp/audio.cpp-gguf/../x", "a/b/c"):
        assert acm.parse_identifier(junk) is None
    assert acm.split_variant_ref("audio-cpp/Yue2-3B-GGUF:BF16") == (
        "audio-cpp/Yue2-3B-GGUF",
        "BF16",
    )
    assert acm.split_variant_ref(r"C:\models\x.gguf") == (r"C:\models\x.gguf", None)
    assert acm.is_umbrella_id("audiocpp-qwen3-asr-0.6b")
    assert not acm.is_umbrella_id("unsloth/Qwen3-0.6B-GGUF")
    assert acm.repo_of("audiocpp-kokoro-82m") == AUDIO_CPP_REPO
    assert acm.repo_of("audio-cpp/Yue2-3B-GGUF:Q8_0") == "audio-cpp/Yue2-3B-GGUF"


# ---------------------------------------------------------------------------
# Headers


def test_header_reads_family_and_spec_past_embedded_data():
    data = _gguf_bytes(family = "moonshine_asr", spec = {"tasks": ["asr"]}, blob_before_spec = 5000)
    header = acm.parse_header(io.BytesIO(data))
    assert (
        header.is_audio_cpp
        and header.family == "moonshine_asr"
        and header.spec == {"tasks": ["asr"]}
    )
    # A prefix that stops inside the embedded data still answers "audio.cpp", without a family.
    short = acm.parse_header(io.BytesIO(data[:2000]))
    assert short.is_audio_cpp and short.family is None and not short.complete
    llama = acm.parse_header(io.BytesIO(_gguf_bytes(arch = "llama")))
    assert llama is not None and not llama.is_audio_cpp
    assert acm.parse_header(io.BytesIO(b"not a gguf at all, just bytes...")) is None


def test_quant_labels():
    assert acm.quant_label("kokoro-82m-q8_0.gguf") == "Q8_0"
    assert acm.quant_label("kitten-tts-mini-0.8-orig.gguf") == "orig"
    assert acm.quant_label("x/pocket-tts-english-bf16.gguf") == "BF16"
    assert acm.quant_label("voxtral-mini-4b-realtime-2602-q4_k.gguf") == "Q4_K"


# ---------------------------------------------------------------------------
# Resolution, from the cache alone


def test_umbrella_folder_resolves_from_its_cached_header(hub):
    _kokoro(hub)
    model = acm.resolve(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF", network = False)
    assert model.family == "kokoro_tts" and model.task == "tts" and model.server_task == "tts"
    assert model.audio_type == "audiocpp_tts" and model.needs_espeak
    assert model.display_name == "Kokoro-82M-GGUF" and model.variant.key == "Q8_0"
    assert model.files == (
        "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
    )  # the README is not a model file
    assert model.unsupported is None
    # The same model through its legacy key and a quant spelled any way.
    assert acm.resolve("audiocpp-kokoro-82m", network = False).id == model.id
    assert acm.resolve(model.id, "q8_0", network = False).variant.key == "Q8_0"
    missing = acm.resolve(model.id, "Q2_K", network = False)
    assert "not found" in missing.unsupported and "Q8_0" in missing.unsupported
    assert acm.looks_like_audio_cpp(model.id)


def test_options_come_from_the_spec_minus_what_studio_drives(hub):
    _kokoro(hub)
    model = acm.resolve(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF", network = False)
    by_name = {o["name"]: o for o in model.options}
    # Language and seed have their own controls; lists and paths cannot be rendered.
    assert list(by_name) == ["voice", "speed", "text_chunk_size", "mode"]
    assert by_name["voice"] == {
        "name": "voice",
        "type": "enum",
        "description": "Built-in speaker.",
        "required": False,
        "default": "af_heart",
        "min": None,
        "max": None,
        "values": ["af_heart", "am_adam"],
    }
    assert by_name["speed"]["default"] == 1 and by_name["speed"]["max"] == 2
    assert by_name["mode"]["values"] == ["a", "b"]


def test_validate_options_drops_unknowns_and_clamps():
    schema = acm.option_schema(acm.FAMILIES["kokoro_tts"], KOKORO_SPEC, None)
    assert acm.validate_options(
        schema,
        {
            "speed": 9,
            "text_chunk_size": "12.6",
            "mode": "c",
            "voice": "am_adam",
            "seed": 4,
            "rm -rf": 1,
        },
    ) == {"speed": 2.0, "text_chunk_size": 13, "voice": "am_adam"}
    assert acm.validate_options(schema, None) == {}


def test_a_llama_gguf_is_not_audio_cpp(hub):
    snap = _snapshot(hub, "unsloth/Qwen3-0.6B-GGUF")
    _put(snap, "Qwen3-0.6B-Q4_K_M.gguf", _gguf_bytes(arch = "qwen3"))
    assert acm.resolve("unsloth/Qwen3-0.6B-GGUF", network = False) is None
    assert (
        acm.resolve("unsloth/Qwen3-0.6B-GGUF", network = False, gguf_hint = "Qwen3-0.6B-Q4_K_M.gguf")
        is None
    )
    assert not acm.looks_like_audio_cpp("unsloth/Qwen3-0.6B-GGUF")


def test_a_dedicated_repo_resolves_by_header_with_its_quants(hub):
    repo = "someone/AudioCPP-Voxtral-GGUF"
    snap = _snapshot(hub, repo)
    header = _gguf_bytes(family = "voxtral_realtime")
    _put(snap, "voxtral-q8_0.gguf", header)
    _put(snap, "voxtral-q4_k.gguf", header)
    model = acm.resolve(repo, network = False)
    assert (
        model.task == "asr"
        and model.audio_type is None
        and model.display_name == "AudioCPP-Voxtral-GGUF"
    )
    assert [v.key for v in model.variants] == ["Q8_0", "Q4_K"]
    with pytest.raises(acm.AudioCppModelError, match = "speech-to-text"):
        acm.require_runnable(model, "tts")
    acm.require_runnable(model, "asr")


def test_an_unknown_family_follows_its_spec_tasks_or_is_refused(hub):
    for family, tasks, task, server in (
        ("new_tts", ["tts", "clone"], "tts", "tts"),
        ("new_music", ["music"], "music", "gen"),
        ("new_clone_music", ["clone", "music"], "music", "gen"),
        ("new_clone_asr", ["clone", "asr"], "asr", "asr"),
        ("new_clone_only", ["clone"], "tts", "clon"),
        ("new_sep", ["sep"], "sep", "sep"),
        ("new_vad", ["vad"], "", ""),
    ):
        repo = f"someone/{family}-GGUF"
        _put(_snapshot(hub, repo), "m-q8_0.gguf", _gguf_bytes(family = family, spec = {"tasks": tasks}))
        model = acm.resolve(repo, network = False)
        assert (model.task, model.server_task) == (task, server)
        assert (model.unsupported is None) == bool(task)
    sep = acm.resolve("someone/new_sep-GGUF", network = False)
    assert list(sep.workflows) == ["separate"] and sep.separation is not None
    acm.require_runnable(sep, "tts")
    vad = acm.resolve("someone/new_vad-GGUF", network = False)
    assert "Voice activity detection" in vad.unsupported
    clone_only = acm.resolve("someone/new_clone_only-GGUF", network = False)
    assert list(clone_only.workflows) == ["clone"] and clone_only.clone.reference_text == "optional"


def test_unsupported_task_tokens_are_refused_by_a_readable_name(hub):
    for family, tasks, message in (
        ("smart_turn", ["turn"], "Turn detection models are not supported in Studio yet."),
        ("new_svc", ["svc"], "Singing voice conversion models are not supported in Studio yet."),
        ("new_s2s", ["s2s"], "Speech-to-speech models are not supported in Studio yet."),
        ("new_odd", ["odd_task"], "Studio does not support the 'odd_task' task yet."),
    ):
        repo = f"someone/{family}-GGUF"
        _put(_snapshot(hub, repo), "m-q8_0.gguf", _gguf_bytes(family = family, spec = {"tasks": tasks}))
        assert acm.resolve(repo, network = False).unsupported == message


_REAL_RUNTIME_KNOWS_FAMILY = acm.runtime_knows_family


def _fake_runtime(
    monkeypatch,
    tmp_path,
    *,
    tag = None,
    specs = None,
):
    """An installed audiocpp_server with a prebuilt install record for ``tag``, and a
    ``model_specs/`` folder beside it when ``specs`` is given."""
    from core.inference import audio_cpp_server

    root = tmp_path / "audio.cpp"
    root.mkdir(parents = True)
    binary = root / "audiocpp_server"
    binary.write_bytes(b"\x7fELF")
    if tag is not None:
        (root / audio_cpp_server.INSTALL_RECORD).write_text(json.dumps({"release_tag": tag}))
    if specs is not None:
        (root / "model_specs").mkdir()
        for name in specs:
            (root / "model_specs" / f"{name}.json").write_text(json.dumps({"family": name}))
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(binary))
    return binary


def test_the_spec_family_list_matches_the_pinned_runtime():
    import importlib.util

    from core.inference.audio_cpp_spec_families import (
        AUDIO_CPP_SPEC_FAMILIES,
        AUDIO_CPP_SPEC_FAMILIES_TAG,
    )

    installer = Path(__file__).resolve().parents[2] / "install_audio_cpp_prebuilt.py"
    spec = importlib.util.spec_from_file_location("_audio_cpp_installer", installer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert AUDIO_CPP_SPEC_FAMILIES_TAG == module.DEFAULT_TAG
    for family in ("crisperwhisper", "index_echo", "audio_flamingo", "kugelaudio", "smart_turn"):
        assert family not in AUDIO_CPP_SPEC_FAMILIES
    for family in ("samsone", "gigaam_asr", "maya1"):
        assert family in AUDIO_CPP_SPEC_FAMILIES
    assert set(acm.FAMILIES) - AUDIO_CPP_SPEC_FAMILIES == {"marblenet_vad", "silero_vad"}


def test_runtime_knows_a_family_from_its_specs_or_the_pinned_list(monkeypatch, tmp_path):
    from core.inference import audio_cpp_server
    from core.inference.audio_cpp_spec_families import AUDIO_CPP_SPEC_FAMILIES_TAG

    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: None)
    assert acm.runtime_knows_family("samsone") is None
    _fake_runtime(monkeypatch, tmp_path / "pinned", tag = AUDIO_CPP_SPEC_FAMILIES_TAG)
    assert acm.runtime_knows_family("samsone") is True
    assert acm.runtime_knows_family("crisperwhisper") is False
    _fake_runtime(monkeypatch, tmp_path / "newer", tag = "v9.9.9-unsloth.1")
    assert acm.runtime_knows_family("crisperwhisper") is None
    _fake_runtime(monkeypatch, tmp_path / "custom")
    assert acm.runtime_knows_family("crisperwhisper") is None
    _fake_runtime(
        monkeypatch, tmp_path / "specs", tag = AUDIO_CPP_SPEC_FAMILIES_TAG, specs = ("crisperwhisper",)
    )
    assert acm.runtime_knows_family("crisperwhisper") is True
    assert acm.runtime_knows_family("samsone") is False


def test_a_spec_fallback_family_the_runtime_lacks_is_refused(hub, monkeypatch, tmp_path):
    from core.inference.audio_cpp_spec_families import AUDIO_CPP_SPEC_FAMILIES_TAG

    # The hub fixture leaves the runtime out of reach; this test installs one.
    monkeypatch.setattr(acm, "runtime_knows_family", _REAL_RUNTIME_KNOWS_FAMILY)
    _fake_runtime(monkeypatch, tmp_path, tag = AUDIO_CPP_SPEC_FAMILIES_TAG)
    for family, folder, tasks in (
        ("gigaam_asr", "GigaAM-v3-GGUF", ["asr"]),
        ("crisperwhisper", "CrisperWhisper2.0-GGUF", ["asr"]),
        ("kugelaudio", "KugelAudio-0-Open-GGUF", ["tts"]),
    ):
        _put(
            _snapshot(hub),
            f"{folder}/m-q8_0.gguf",
            _gguf_bytes(family = family, spec = {"tasks": tasks}),
        )
    gigaam = acm.resolve(f"{AUDIO_CPP_REPO}/GigaAM-v3-GGUF", network = False)
    assert (gigaam.task, gigaam.unsupported) == ("asr", None)
    crisper = acm.resolve(f"{AUDIO_CPP_REPO}/CrisperWhisper2.0-GGUF", network = False)
    assert crisper.task == "" and crisper.workflows == {}
    assert crisper.unsupported == (
        "CrisperWhisper2.0-GGUF needs a newer audio runtime than the one installed."
    )
    kugel = acm.resolve(f"{AUDIO_CPP_REPO}/KugelAudio-0-Open-GGUF", network = False)
    assert "needs a newer audio runtime" in kugel.unsupported
    assert acm.family_policy("silero_vad").unsupported.startswith("Voice activity detection")
    assert acm.family_policy("kokoro_tts").task == "tts"
    assert acm.family_policy("crisperwhisper", {"tasks": ["asr"]}).unsupported == (
        "crisperwhisper needs a newer audio runtime than the one installed."
    )
    _fake_runtime(monkeypatch, tmp_path / "newer", tag = "v9.9.9-unsloth.1")
    assert acm.family_policy("crisperwhisper", {"tasks": ["asr"]}).task == "asr"


def test_known_task_families_are_refused_with_a_reason(hub):
    _put(
        _snapshot(hub),
        "Sortformer-Diar-GGUF/sortformer-diar-q8_0.gguf",
        _gguf_bytes(family = "sortformer_diar"),
    )
    model = acm.resolve(f"{AUDIO_CPP_REPO}/Sortformer-Diar-GGUF", network = False)
    assert "Speaker diarization" in model.unsupported
    with pytest.raises(acm.AudioCppModelError):
        acm.require_runnable(model, "tts")


@pytest.mark.parametrize(
    "folder, family",
    [
        ("HTDemucs-GGUF", "htdemucs"),
        ("HTDemucs-6stems-GGUF", "htdemucs_6stems"),
        ("BS-RoFormer-ep368-GGUF", "bs_roformer"),
        ("Mel-Band-RoFormer-GGUF", "mel_band_roformer"),
    ],
)
def test_separation_families_resolve_runnable(hub, folder, family):
    _put(_snapshot(hub), f"{folder}/{folder.lower()}-q8_0.gguf", _gguf_bytes(family = family))
    model = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", network = False)
    assert (model.family, model.task, model.server_task) == (family, "sep", "sep")
    assert model.unsupported is None and model.options == ()
    assert (model.audio_type, model.hub_task) == ("audiocpp_sep", "audio-to-audio")
    assert model.workflows == {"separate": acm.WorkflowBinding("sep", "tasks", None, ("audio",))}
    assert model.separation == acm.FAMILIES[family].separation
    acm.require_runnable(model, "tts")
    with pytest.raises(acm.AudioCppModelError, match = "not a speech-to-text model"):
        acm.require_runnable(model, "asr")
    assert acm.family_from_names([folder]) == family


def test_qwen3_tts_package_kind_comes_from_its_name(hub):
    snap = _snapshot(hub)
    for folder in (
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF",
        "Qwen3-TTS-12Hz-0.6B-Base-GGUF",
    ):
        _put(snap, f"{folder}/{folder.lower()}-q8_0.gguf", _gguf_bytes(family = "qwen3_tts"))
    design = acm.resolve(f"{AUDIO_CPP_REPO}/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF", network = False)
    custom = acm.resolve(f"{AUDIO_CPP_REPO}/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF", network = False)
    base = acm.resolve(f"{AUDIO_CPP_REPO}/Qwen3-TTS-12Hz-0.6B-Base-GGUF", network = False)
    assert design.server_task == "vdes" and design.request_defaults["options"]["instruct"]
    assert custom.server_task == "tts" and custom.request_defaults["voice"] == "vivian"
    # The published GGUF embeds no speaker list, so the picker offers the package's own nine.
    voice = next(o for o in custom.options if o["name"] == "voice")
    assert (
        voice["default"] == "vivian" and len(voice["values"]) == 9 and "uncle_fu" in voice["values"]
    )
    assert not any(o["name"] == "voice" for o in design.options)
    assert base.unsupported is None and list(base.workflows) == ["clone"]
    assert base.clone.reference_text == "required" and base.clone.language_names
    assert [o["name"] for o in base.clone.tool_options] == ["x_vector_only_mode"]
    assert list(custom.workflows) == ["speak"] and list(design.workflows) == ["speak"]


# family -> (server task, speaks, reference_text), from the audio.cpp sources.
_CLONE_TABLE = {
    "chatterbox": ("clon", False, "unused"),
    "f5_tts": ("tts", False, "required"),
    "index_tts2": ("tts", False, "unused"),
    "cosyvoice3": ("tts", False, "required"),
    "voxcpm2": ("tts", True, "optional"),
    "fish_audio": ("tts", True, "required"),
    "echo_tts": ("clon", False, "unused"),
    "confucius4_tts": ("clon", False, "unused"),
    "fireredtts3": ("tts", False, "optional"),
    "firered_audio": ("tts", False, "optional"),
    "vevo2": ("tts", False, "unused"),
    "miotts": ("tts", False, "unused"),
    "breeze_tts": ("tts", True, "required"),
    "omnivoice": ("tts", True, "required"),
    "voxcpm1": ("tts", True, "required"),
    "dots_tts": ("tts", True, "optional"),
    "higgs_audio_tts": ("tts", True, "optional"),
    "moss_tts_local": ("tts", True, "unused"),
    "moss_tts_nano": ("tts", True, "unused"),
    "irodori_tts": ("tts", True, "unused"),
    "pocket_tts": ("tts", True, "unused"),
}


# family -> (load task, mode -> server task, target, source rate)
_CONVERT_TABLE = {
    "rvc": ("vc", {"speech": "vc"}, "builtin", 16000),
    "seed_vc": ("vc", {"speech": "vc", "singing": "svc"}, "audio", 44100),
    "meanvc2": ("vc", {"speech": "vc"}, "audio", 16000),
    "chatterbox": ("clon", {"speech": "vc"}, "audio", 16000),
    "vevo2": ("tts", {"speech": "vc", "singing": "svc"}, "audio", 24000),
    "tone_color_vc": ("vc", {"speech": "vc"}, "audio", 22050),
}


def test_every_task_family_binds_its_workflows():
    workflow_for = {"tts": "speak", "music": "music", "asr": "transcribe", "sep": "separate"}
    endpoint_for = {
        "speak": "speech",
        "music": "tasks",
        "transcribe": "transcriptions",
        "separate": "tasks",
    }
    for family in acm.FAMILIES.values():
        # Edit is checked by its own table below.
        bindings = {k: v for k, v in family.workflows.items() if k != "edit"}
        if not family.task:
            assert bindings == {}, family.family
            continue
        if family.convert is not None:
            load_task, modes, target, source_rate = _CONVERT_TABLE[family.family]
            assert family.default_server_task == load_task, family.family
            assert family.convert.server_tasks == modes, family.family
            assert family.convert.target == target, family.family
            convert = bindings["convert"]
            assert (convert.endpoint, convert.server_task, family.convert.source_rate) == (
                "tasks",
                modes["speech"],
                source_rate,
            ), family.family
            assert convert.inputs == ("source", "target", "source_text")
            if family.clone is None:
                assert list(bindings) == ["convert"], family.family
                continue
        if family.clone is not None:
            server_task, speaks, reference_text = _CLONE_TABLE[family.family]
            expected = (["speak"] if speaks else []) + ["clone"]
            if family.convert is not None:
                expected.append("convert")
            assert list(bindings) == expected, family.family
            assert family.default_server_task == server_task, family.family
            assert family.clone.reference_text == reference_text, family.family
            clone = bindings["clone"]
            assert (clone.endpoint, clone.server_task) == ("speech", server_task)
            continue
        assert family.family not in _CLONE_TABLE
        ((workflow, binding),) = bindings.items()
        assert workflow == workflow_for[family.task], family.family
        assert binding.endpoint == endpoint_for[workflow]
        assert binding.server_task == family.default_server_task
    assert set(_CLONE_TABLE) == {f.family for f in acm.FAMILIES.values() if f.clone is not None}
    assert set(_CONVERT_TABLE) == {f.family for f in acm.FAMILIES.values() if f.convert is not None}
    # Chatterbox-Turbo is its own family and still speaks.
    assert list(acm.FAMILIES["chatterbox_turbo"].workflows) == ["speak"]
    assert acm.FAMILIES["chatterbox_turbo"].default_server_task == "tts"
    assert acm.family_from_names(["Chatterbox-Turbo-GGUF", "chatterbox-turbo-q8_0.gguf"]) == (
        "chatterbox_turbo"
    )
    base = acm.family_policy("fireredtts3", names = ["FireRedTTS3-Base-GGUF"])
    assert base.default_server_task == "clon" and list(base.workflows) == ["clone"]
    (codec,) = acm.FAMILIES["miotts"].companions
    assert codec.id == f"{AUDIO_CPP_REPO}/MioCodec-25Hz-44.1kHz-v2-GGUF"
    assert (codec.variant, codec.session_option) == ("Q8_0", "miotts.codec_model_path")
    assert "pick MioTTS" in acm.family_policy("miocodec").unsupported
    assert acm.FAMILIES["kokoro_tts"].workflows["speak"].server_task == "tts"
    assert acm.FAMILIES["moss_voicegen"].workflows["speak"].server_task == "vdes"
    assert acm.FAMILIES["ace_step"].workflows["music"] == acm.WorkflowBinding(
        "gen", "tasks", None, ("text", "lyrics", "duration_seconds")
    )
    assert acm.FAMILIES["moonshine_asr"].workflows["transcribe"].inputs == ("audio",)
    design = acm.family_policy("qwen3_tts", names = ["Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF"])
    assert design.workflows["speak"].server_task == "vdes"
    assert acm.family_policy("htdemucs").workflows == {
        "separate": acm.WorkflowBinding("sep", "tasks", None, ("audio",))
    }


def test_edit_families_bind_the_edit_workflow():
    edit_names = ["DotTTS-Edit-GGUF", "dots-tts-edit-q8_0.gguf"]
    dots_edit = acm.family_policy("dots_tts", names = edit_names)
    binding = dots_edit.workflows["edit"]
    assert list(dots_edit.workflows) == ["speak", "clone", "edit"]
    assert (binding.server_task, binding.endpoint) == ("tts", "tasks")
    for names in (["DotTTS-MF-GGUF"], ["DotTTS-SOAR-GGUF"], []):
        assert list(acm.family_policy("dots_tts", names = names).workflows) == [
            "speak",
            "clone",
        ], names
    vevo2, firered = acm.FAMILIES["vevo2"], acm.FAMILIES["firered_audio"]
    assert list(firered.workflows) == ["clone", "edit"]
    # Vevo2 also converts a voice.
    assert list(vevo2.workflows) == ["clone", "edit", "convert"]
    # Vevo2 edits in an s2s session though it loads as tts.
    assert vevo2.default_server_task == "tts"
    assert (vevo2.workflows["edit"].server_task, vevo2.workflows["edit"].route) == (
        "s2s",
        "editing",
    )
    assert firered.workflows["edit"].server_task == "tts"
    assert {n for n, f in acm.FAMILIES.items() if f.edit is not None} == {"vevo2", "firered_audio"}
    assert acm.family_policy("auk").unsupported
    for family in (dots_edit, vevo2, firered):
        hash(family)


def test_a_resolved_model_carries_its_workflow_binding(hub):
    snap = _snapshot(hub)
    folder = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF"
    _put(snap, f"{folder}/{folder.lower()}-q8_0.gguf", _gguf_bytes(family = "qwen3_tts"))
    model = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", network = False)
    assert list(model.workflows) == ["speak"] and model.workflows["speak"].server_task == "vdes"
    _put(snap, "HTDemucs-GGUF/htdemucs-q8_0.gguf", _gguf_bytes(family = "htdemucs"))
    assert list(acm.resolve(f"{AUDIO_CPP_REPO}/HTDemucs-GGUF", network = False).workflows) == [
        "separate"
    ]


def test_sub_folders_and_same_quant_files_become_named_variants(hub):
    snap = _snapshot(hub)
    ace = _gguf_bytes(family = "ace_step")
    _put(snap, "ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-q8_0.gguf", ace)
    _put(snap, "ACE-Step1.5-GGUF/base/ace-step-1.5-base-q8_0.gguf", ace)
    moon = _gguf_bytes(family = "moonshine_asr")
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", moon)
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf", moon)
    model = acm.resolve(f"{AUDIO_CPP_REPO}/ACE-Step1.5-GGUF", network = False)
    assert [v.key for v in model.variants] == ["turbo/Q8_0", "base/Q8_0"]
    assert model.variant.key == "turbo/Q8_0" and model.variants[0].label == "Q8_0 · turbo"
    # A bare quant shared by both picks the default; the legacy id names its sub-folder.
    assert acm.resolve(model.id, "Q8_0", network = False).variant.key == "turbo/Q8_0"
    assert acm.resolve("audiocpp-ace-step-1.5-base", network = False).variant.key == "base/Q8_0"
    moonshine = acm.resolve("audiocpp-moonshine-small", network = False)
    assert moonshine.variant.key == "small/Q8_0" and moonshine.variant.label == "Q8_0 · small"
    assert moonshine.default_variant == "tiny/Q8_0"
    assert acm.resolve(
        f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF/tiny", network = False
    ).variant.key == ("tiny/Q8_0")


def _keys(folder, *names):
    files = [RepoFile(f"{folder}/{name}", 1) for name in names]
    return [v.key for v in sorted(acm._single_file_variants(files, folder), key = acm._variant_rank)]


def test_umbrella_folders_key_and_rank_their_models_by_the_published_names():
    # Q8_0_V2 is the Q8_0 export and FP32 is F32: both outrank the bigger or unranked files.
    assert _keys(
        "Qwen3-TTS-12Hz-1.7B-Base-GGUF",
        "qwen3-tts-12hz-1.7b-base-bf16.gguf",
        "qwen3-tts-12hz-1.7b-base-orig.gguf",
        "qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
    ) == ["Q8_0_V2", "BF16", "orig"]
    assert _keys(
        "MeanVC2-GGUF", "meanvc2-120ms-40ms-q4_k.gguf", "meanvc2-120ms-40ms-fp32.gguf"
    ) == ["FP32", "Q4_K"]
    # Six models in one folder: an F32 row is named like its F16 sibling.
    gigaam = [
        f"gigaam-{name}-f16.gguf"
        for name in (
            "multilingual-ctc",
            "multilingual-large-ctc",
            "v3-ctc",
            "v3-e2e-ctc",
            "v3-e2e-rnnt",
            "v3-rnnt",
        )
    ]
    assert _keys(
        "GigaAM-ASR-GGUF",
        *gigaam,
        "gigaam-multilingual-ctc-f32.gguf",
        "gigaam-multilingual-large-ctc-f32.gguf",
    ) == [
        "multilingual-ctc/F16",
        "multilingual-large-ctc/F16",
        "v3-ctc/F16",
        "v3-e2e-ctc/F16",
        "v3-e2e-rnnt/F16",
        "v3-rnnt/F16",
        "multilingual-ctc/F32",
        "multilingual-large-ctc/F32",
    ]
    # A dotted version stays one word, and the model with one quant is named too.
    assert _keys(
        "Irodori-TTS-v4-Small-GGUF",
        "irodori-tts-v4-small-f16.gguf",
        "irodori-tts-v4.1-anime-q8_0.gguf",
        "irodori-tts-v4-small-q8_0.gguf",
    ) == ["v4-small/Q8_0", "v4.1-anime/Q8_0", "v4-small/F16"]


def test_umbrella_keys_saved_settings_name_stay_the_same():
    assert _keys(
        "Moonshine-Streaming-GGUF",
        "moonshine-streaming-tiny-q8_0.gguf",
        "moonshine-streaming-medium-q8_0.gguf",
        "moonshine-streaming-small-q8_0.gguf",
    ) == ["tiny/Q8_0", "medium/Q8_0", "small/Q8_0"]
    assert _keys(
        "Niagara-ASR-GGUF", "niagara-19m-batch.en-f32.gguf", "niagara-38m-batch.en-f32.gguf"
    ) == ["19m/F32", "38m/F32"]
    assert _keys(
        "Samsone-GGUF",
        *(
            f"samsone-{size}-{quant}.gguf"
            for size in ("99m", "134m", "356m")
            for quant in ("q8_0", "bf16")
        ),
    ) == ["134m/Q8_0", "356m/Q8_0", "99m/Q8_0", "134m/BF16", "356m/BF16", "99m/BF16"]
    assert _keys("UniverSR-GGUF", "universr-audio-orig.gguf", "universr-speech-orig.gguf") == [
        "audio/orig",
        "speech/orig",
    ]
    assert _keys(
        "ACE-Step1.5-GGUF",
        "turbo/ace-step-1.5-turbo-q8_0.gguf",
        "turbo/ace-step-1.5-turbo-bf16.gguf",
        "base/ace-step-1.5-base-q8_0.gguf",
    ) == ["turbo/Q8_0", "turbo/BF16", "base/Q8_0"]
    assert _keys(
        "PocketTTS-GGUF",
        "english/pocket-tts-english-q8_0.gguf",
        "english/pocket-tts-english-bf16.gguf",
        "german/pocket-tts-german-q8_0.gguf",
    ) == ["english/Q8_0", "english/BF16", "german/Q8_0"]


def _minimax(root, *, quant = "q4_0"):
    snap = _snapshot(root, "audio-cpp/MiniMax-Music3-GGUF")
    header = _gguf_bytes(family = "minimax_music3")
    for rel in (
        "config.json",
        "config/language_model.json",
        "config/rvq_depth_decoder.json",
        "config/condition_encoder.json",
        "config/transformer.json",
        "config/vocoder.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        _put(snap, rel, b"{}")
    for rel in (
        f"language_model_{quant}.gguf",
        "rvq_depth_decoder_q8_0.gguf",
        f"transformer_{quant}.gguf",
        "condition_encoder.gguf",
        "vocoder.gguf",
    ):
        _put(snap, rel, header)
    return snap


def test_a_package_repo_lists_its_mixes_and_picks_its_components(hub):
    _minimax(hub)
    model = acm.resolve("audio-cpp/MiniMax-Music3-GGUF", network = False)
    assert model.family == "minimax_music3" and model.task == "music" and model.server_task == "gen"
    assert model.is_package and [v.key for v in model.variants] == ["Q4_0"]  # only Q4_0 is complete
    assert len(model.files) == 13 and model.gguf_file == "language_model_q4_0.gguf"
    assert model.model_options["session_options"] == {
        "minimax_music3.language_model_gguf": "language_model_q4_0.gguf",
        "minimax_music3.rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
        "minimax_music3.flow_transformer_gguf": "transformer_q4_0.gguf",
    }
    # The Studio-side list stands in for the runtime's spec; lyrics and duration have their own fields.
    assert [o["name"] for o in model.options] == [
        "num_inference_steps",
        "guidance_scale",
        "ar_guidance_scale",
        "top_k",
    ]
    assert audio_cpp_files.cached_files(model) is not None


def test_package_layouts_plan_whole_mixes(hub):
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    def sib(name, size = 10):
        return SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": name})

    names = [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-3b-q8_0.gguf",
        "yue2-3b-bf16.gguf",
        "yue2-vae-f16.gguf",
        "yue2-vae-f32.gguf",
        "README.md",
    ]
    plans = build_gguf_variant_plans([sib(n) for n in names])
    assert sorted(plans) == ["bf16", "q8_0"]
    plan = plan_for_variant(plans, "BF16")
    assert set(plan.target_filenames) == set(names[:4]) | {"yue2-3b-bf16.gguf", "yue2-vae-f16.gguf"}
    # Files every mix shares are companions, never one mix's own weights.
    assert plan.main_filenames == frozenset({"yue2-3b-bf16.gguf"})
    assert "yue2-vae-f16.gguf" in plan.companion_hashes


def test_voice_embeddings_ride_with_the_gguf_they_sit_beside():
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    def sib(name):
        return SimpleNamespace(rfilename = name, size = 5, lfs = {"sha256": name})

    names = [
        "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
        "PocketTTS-GGUF/english/embeddings/alba.safetensors",
        "PocketTTS-GGUF/english/embeddings/anna.safetensors",
        "PocketTTS-GGUF/german/pocket-tts-german-q8_0.gguf",
        "PocketTTS-GGUF/german/embeddings/karl.safetensors",
    ]
    plans = build_gguf_variant_plans([sib(n) for n in names])
    english = plan_for_variant(plans, "PocketTTS-GGUF/english/pocket-tts-english-q8_0")
    assert set(english.target_filenames) == set(names[:3])
    assert english.download_size_bytes == 15


GIGAAM_FILES = (
    *(
        f"gigaam-{name}-f16.gguf"
        for name in (
            "multilingual-ctc",
            "multilingual-large-ctc",
            "v3-ctc",
            "v3-e2e-ctc",
            "v3-e2e-rnnt",
            "v3-rnnt",
        )
    ),
    "gigaam-multilingual-ctc-f32.gguf",
    "gigaam-multilingual-large-ctc-f32.gguf",
)
GIGAAM_SPEC = {"family": "gigaam_asr", "tasks": ["asr"]}


@pytest.mark.parametrize(
    "folder, spec, published, cached",
    [
        ("GigaAM-ASR-GGUF", GIGAAM_SPEC, GIGAAM_FILES, GIGAAM_FILES[2:3]),
        ("GigaAM-ASR-GGUF", GIGAAM_SPEC, GIGAAM_FILES, (GIGAAM_FILES[0], GIGAAM_FILES[1])),
        (
            "Irodori-TTS-v4-Small-GGUF",
            {"family": "irodori_tts", "tasks": ["tts"]},
            (
                "irodori-tts-v4-small-f16.gguf",
                "irodori-tts-v4.1-anime-q8_0.gguf",
                "irodori-tts-v4-small-q8_0.gguf",
            ),
            ("irodori-tts-v4-small-q8_0.gguf", "irodori-tts-v4.1-anime-q8_0.gguf"),
        ),
    ],
)
def test_a_hub_variant_key_loads_its_file_from_a_partial_cache(
    hub, folder, spec, published, cached
):
    # STT loads and offline loads list only the cache, where the downloaded files alone name the rows.
    online = acm._single_file_variants(
        [RepoFile(f"{folder}/{name}", 1) for name in published], folder
    )
    snap = _snapshot(hub)
    for name in cached:
        _put(snap, f"{folder}/{name}", _gguf_bytes(family = spec["family"], spec = spec))
    for name in cached:
        key = next(v.key for v in online if v.primary == f"{folder}/{name}")
        acm.forget()
        model = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", key, network = False)
        assert model.unsupported is None, key
        assert model.variant.primary == f"{folder}/{name}"
        assert model.variant.key == key


def test_the_folder_name_never_names_a_variant(hub):
    # Irodori-TTS-v4-Small-GGUF carries the words of v4-small: with only the anime model cached,
    # v4-small/Q8_0 is missing, not the anime file under that name.
    folder = "Irodori-TTS-v4-Small-GGUF"
    spec = {"family": "irodori_tts", "tasks": ["tts"]}
    _put(
        _snapshot(hub),
        f"{folder}/irodori-tts-v4.1-anime-q8_0.gguf",
        _gguf_bytes(family = "irodori_tts", spec = spec),
    )
    model = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", "v4-small/Q8_0", network = False)
    assert "not found" in model.unsupported
    anime = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", "v4.1-anime/Q8_0", network = False)
    assert anime.unsupported is None and anime.variant.key == "v4.1-anime/Q8_0"


MIOCODEC_Q8 = "MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf"


def test_miotts_downloads_its_codec_with_every_variant():
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    def sib(name):
        return SimpleNamespace(rfilename = name, size = 5, lfs = {"sha256": name})

    names = [
        "MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf",
        "MioTTS-1.7B-GGUF/miotts-1.7b-bf16.gguf",
        MIOCODEC_Q8,
        "MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-f16.gguf",
        "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
    ]
    assert acm.companion_files(names[1], names) == (MIOCODEC_Q8,)
    assert acm.companion_files(names[4], names) == ()
    plans = build_gguf_variant_plans([sib(n) for n in names])
    bf16 = plan_for_variant(plans, "MioTTS-1.7B-GGUF/miotts-1.7b-bf16")
    assert set(bf16.target_filenames) == {names[1], MIOCODEC_Q8}
    assert bf16.main_filenames == frozenset({names[1]}) and bf16.download_size_bytes == 10
    kokoro = plan_for_variant(plans, "Kokoro-82M-GGUF/kokoro-82m-q8_0")
    assert kokoro.target_filenames == (names[4],)


def test_miotts_lists_as_downloaded_only_with_its_codec(hub):
    from hub.services.models import gguf_variants as gv

    snap = _snapshot(hub)
    _put(snap, "MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf", _gguf_bytes(family = "miotts"))

    def answer():
        acm.forget()
        result = asyncio.run(
            gv._audio_cpp_variants_answer(f"{AUDIO_CPP_REPO}/MioTTS-1.7B-GGUF", None, True)
        )
        return result.response.variants[0]

    row = answer()
    assert row.quant == "Q8_0" and not row.downloaded
    codec = _put(snap, MIOCODEC_Q8, _gguf_bytes(family = "miocodec"))
    row = answer()
    assert row.downloaded
    assert row.download_size_bytes == row.size_bytes + codec.stat().st_size


def test_download_target_maps_a_folder_row_onto_the_planner_key(hub):
    _kokoro(hub)
    assert acm.download_target(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF", "Q8_0") == (
        AUDIO_CPP_REPO,
        "Kokoro-82M-GGUF/kokoro-82m-q8_0",
    )
    assert acm.download_target("audio-cpp/MiniMax-Music3-GGUF", "Q4_0") is None


# ---------------------------------------------------------------------------
# Files in the cache


def test_cached_files_require_every_file_of_the_variant(hub):
    snap = _snapshot(hub)
    gguf = RepoFile("PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf", 4)
    voice = RepoFile("PocketTTS-GGUF/english/embeddings/alba.safetensors", 1)
    m = _model("PocketTTS-GGUF", "pocket_tts", "tts", files = (gguf, voice), key = "english/Q8_0")
    _put(snap, gguf.path, b"GGUF")
    # An interrupted download missing a voice is not a complete model.
    assert audio_cpp_files.cached_files(m) is None
    assert audio_cpp_files.missing_files(m) == [(voice.path, 1)]
    _put(snap, voice.path, b"x")
    assert len(audio_cpp_files.cached_files(m)) == 2
    # A truncated file is not complete either.
    _put(snap, gguf.path, b"GG")
    assert audio_cpp_files.cached_files(m) is None


def test_farm_layout_keeps_the_voice_folder_beside_the_gguf(hub, monkeypatch):
    snap = _snapshot(hub)
    gguf = RepoFile("PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf", 4)
    voice = RepoFile("PocketTTS-GGUF/english/embeddings/alba.safetensors", 1)
    m = _model("PocketTTS-GGUF", "pocket_tts", "tts", files = (gguf, voice), key = "english/Q8_0")
    _put(snap, gguf.path, b"GGUF")
    _put(snap, voice.path, b"x")
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    served = audio_cpp_files.materialize(m, hub_cache = hub)
    farm = audio_cpp_files._link_farm_root(hub) / m.key
    assert served == str(farm / "english" / "pocket-tts-english-q8_0.gguf")
    assert (farm / "english" / "embeddings" / "alba.safetensors").is_file()
    marker = json.loads((farm / audio_cpp_files._SOURCE_MARKER).read_text(encoding = "utf-8"))
    assert marker["repo_id"] == AUDIO_CPP_REPO and len(marker["files"]) == 2


def test_a_package_is_served_as_its_directory(hub, monkeypatch):
    _minimax(hub)
    model = acm.resolve("audio-cpp/MiniMax-Music3-GGUF", network = False)
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    served = Path(audio_cpp_files.materialize(model, hub_cache = hub))
    assert served == audio_cpp_files._link_farm_root(hub) / model.key
    assert (served / "tokenizer" / "tokenizer.json").is_file() and (
        served / "vocoder.gguf"
    ).is_file()


def test_windows_refuses_a_model_path_the_server_cannot_open(hub, monkeypatch):
    from core.inference import audio_cpp_server as srv
    from core.inference.audio_cpp_server import AudioCppUnavailableError

    _put(_snapshot(hub), CANARY.gguf_file, b"GGUF")
    binary = str(hub / srv.BINARY_NAME)
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    assert srv.model_runtime_problem(CANARY, binary) is None
    monkeypatch.setattr(audio_cpp_files, "_WINDOWS_MAX_MODEL_PATH", 10)
    # Refused before download or eviction, not first at materialize.
    assert "shorter" in srv.model_runtime_problem(CANARY, binary)
    with pytest.raises(AudioCppUnavailableError, match = "shorter"):
        audio_cpp_files.materialize(CANARY, hub_cache = hub)
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "linux")
    assert srv.model_runtime_problem(CANARY, binary) is None


def test_an_unsupported_model_is_a_runtime_problem(hub):
    from core.inference import audio_cpp_server as srv
    refused = _model(
        "Silero-VAD-GGUF",
        "silero_vad",
        "",
        unsupported = "Voice activity detection models are not supported in Studio yet.",
    )
    assert "Voice activity detection" in srv.model_runtime_problem(refused, "x")


def test_files_split_across_snapshots_are_found_in_the_one_that_holds_them(hub):
    _snapshot(hub, sha = "a" * 40)
    other = _snapshot(hub, sha = "b" * 40, main = False)
    _put(other, CANARY.gguf_file, b"GGUF")
    assert audio_cpp_files.cached_files(CANARY) == [other / CANARY.gguf_file]


def test_materialize_of_a_missing_model_raises(hub):
    _snapshot(hub)
    with pytest.raises(FileNotFoundError):
        audio_cpp_files.materialize(CANARY, hub_cache = hub)


def _symlinked(
    hub,
    model = CANARY,
    data = b"GGUF-weights",
):
    snap = _snapshot(hub)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir(exist_ok = True)
    blob = blobs / "deadbeef"
    blob.write_bytes(data)
    link = snap / model.gguf_file
    link.parent.mkdir(parents = True, exist_ok = True)
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
    except OSError:
        pytest.skip("this filesystem or account cannot create symlinks")
    return link, blob


def test_symlinked_cache_entries_are_hardlinked_under_their_real_names(hub):
    model = _model(
        "Canary-180M-Flash-GGUF", "canary_asr", "asr", files = (RepoFile(CANARY.gguf_file, 12),)
    )
    _link, blob = _symlinked(hub, model)
    path = audio_cpp_files.materialize(model, hub_cache = hub)
    assert path.endswith(".gguf") and not os.path.islink(path)
    assert os.path.samefile(path, blob)
    assert audio_cpp_files.materialize(model, hub_cache = hub) == path


def test_prune_drops_farm_entries_whose_files_were_deleted(hub):
    model = _model(
        "Canary-180M-Flash-GGUF", "canary_asr", "asr", files = (RepoFile(CANARY.gguf_file, 4),)
    )
    link, blob = _symlinked(hub, model, b"GGUF")
    path = audio_cpp_files.materialize(model, hub_cache = hub)
    assert audio_cpp_files.prune_link_farm(hub) == 0  # still referenced
    link.unlink()
    blob.unlink()
    assert audio_cpp_files.prune_link_farm(hub) == 2  # the file and its source record
    assert not os.path.exists(path)


def test_prune_removes_old_unrecorded_farm_dirs(hub):
    farm = audio_cpp_files._link_farm_root(hub)
    legacy = farm / "audiocpp-canary-180m-flash"
    legacy.mkdir(parents = True)
    (legacy / "canary.gguf").write_bytes(b"x")
    # A young unrecorded folder may be a materialize in flight.
    assert audio_cpp_files.prune_link_farm(hub) == 0
    os.utime(legacy, (1, 1))
    assert audio_cpp_files.prune_link_farm(hub) == 1
    assert not legacy.exists()


def test_a_copied_file_is_not_recopied(hub, monkeypatch):
    model = _model(
        "Canary-180M-Flash-GGUF", "canary_asr", "asr", files = (RepoFile(CANARY.gguf_file, 4),)
    )
    _symlinked(hub, model, b"GGUF")

    def no_link(*_a, **_k):
        raise OSError("cross-device link")

    copies = []
    real_copy = audio_cpp_files.shutil.copyfile
    monkeypatch.setattr(audio_cpp_files.os, "link", no_link)
    monkeypatch.setattr(
        audio_cpp_files.shutil, "copyfile", lambda a, b: copies.append(a) or real_copy(a, b)
    )
    for _ in range(3):
        audio_cpp_files.materialize(model, hub_cache = hub)
    assert len(copies) == 1


def test_an_unwritable_farm_location_is_an_unavailable_error(hub, monkeypatch):
    from core.inference.audio_cpp_server import AudioCppUnavailableError

    model = _model(
        "Canary-180M-Flash-GGUF", "canary_asr", "asr", files = (RepoFile(CANARY.gguf_file, 12),)
    )
    _symlinked(hub, model)
    farm_root = audio_cpp_files._link_farm_root(hub)
    real_mkdir = Path.mkdir

    def mkdir(self, *a, **kw):
        if farm_root in (self, *self.parents):
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(AudioCppUnavailableError, match = "writable folder"):
        audio_cpp_files.materialize(model, hub_cache = hub)


def _dir_link(link, target):
    """A directory symlink, or a Windows junction when this account cannot create symlinks."""
    try:
        os.symlink(target, link, target_is_directory = True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("this filesystem or account cannot create symlinks")
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def test_prune_never_follows_a_link_planted_in_the_farm(hub, tmp_path):
    farm = audio_cpp_files._link_farm_root(hub)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep", encoding = "utf-8")
    farm.mkdir(parents = True)
    _dir_link(farm / "not-a-model", victim)
    (farm / "some-model").mkdir()
    _dir_link(farm / "some-model" / "sub", victim)
    os.utime(farm / "some-model", (1, 1))
    audio_cpp_files.prune_link_farm(hub)
    assert (victim / "notes.txt").read_text(encoding = "utf-8") == "keep"
    assert not (farm / "not-a-model").exists()


def test_materialize_does_not_write_through_a_planted_model_folder_link(hub, tmp_path, monkeypatch):
    snap = _snapshot(hub)
    _put(snap, CANARY.gguf_file, b"GGUF")
    farm = audio_cpp_files._link_farm_root(hub)
    farm.mkdir(parents = True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _dir_link(farm / CANARY.key, elsewhere)
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")  # always goes through the farm
    served = audio_cpp_files.materialize(CANARY, hub_cache = hub)
    assert list(elsewhere.iterdir()) == []
    assert os.path.samefile(served, snap / CANARY.gguf_file)


def test_materialize_refuses_a_repo_file_name_that_climbs_out_of_the_farm(
    hub, tmp_path, monkeypatch
):
    from dataclasses import replace

    from core.inference.audio_cpp_server import AudioCppUnavailableError

    snap = _snapshot(hub)
    _put(snap, CANARY.gguf_file, b"GGUF")
    evil = acm.RepoFile("Canary-180M-Flash-GGUF/embeddings/..\\..\\..\\victim", 4)
    _put(snap, evil.path, b"EVIL")
    victim = audio_cpp_files._link_farm_root(hub).parent / "victim"
    variant = replace(CANARY.variant, files = (*CANARY.variant.files, evil))
    model = replace(CANARY, variant = variant, variants = (variant,))
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")  # always goes through the farm
    with pytest.raises(AudioCppUnavailableError, match = "Refusing"):
        audio_cpp_files.materialize(model, hub_cache = hub)
    assert not victim.exists()


def test_v1_models_lists_speech_and_separation_models_with_their_workflows(hub, monkeypatch):
    from core.inference import audio_cpp_server
    from routes import inference as ri

    snap = _kokoro(hub)
    _put(snap, "HTDemucs-GGUF/htdemucs-q8_0.gguf", _gguf_bytes(family = "htdemucs"))
    assert {m.id: m.task for m in acm.downloaded_models()}[
        f"{AUDIO_CPP_REPO}/HTDemucs-GGUF"
    ] == "sep"
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: "audiocpp_server")
    monkeypatch.setattr(audio_cpp_server, "model_runtime_problem", lambda model, binary = None: None)
    listed = {o["id"]: o for o in ri._audio_cpp_speech_model_objects(0)}
    kokoro = listed[f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"]
    assert kokoro["task"] == "text-to-speech" and "speak" in kokoro["audio_workflows"]
    # /v1/audio/speech cannot serve it, so it is not text-to-speech; /v1/audio/run loads it by name.
    demucs = listed[f"{AUDIO_CPP_REPO}/HTDemucs-GGUF"]
    assert demucs["task"] == "audio-to-audio" and demucs["audio_workflows"] == ["separate"]
    assert len(listed) == 2


def test_downloaded_models_are_found_by_header(hub):
    _kokoro(hub)
    snap = _snapshot(hub)
    _put(snap, "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf", _gguf_bytes(family = "qwen3_asr"))
    _put(_snapshot(hub, "someone/llama-GGUF"), "llama-q4_k_m.gguf", _gguf_bytes(arch = "llama"))
    _put(
        _snapshot(hub, "someone/Parakeet-GGUF"),
        "parakeet-q8_0.gguf",
        _gguf_bytes(family = "parakeet_tdt"),
    )
    found = {m.id: m.task for m in acm.downloaded_models()}
    assert found == {
        f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF": "tts",
        f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF": "asr",
        "someone/Parakeet-GGUF": "asr",
    }
    from core.inference import stt_audiocpp_sidecar as s

    assert sorted(s.downloaded_model_ids()) == [
        f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF",
        # The legacy key of that downloaded folder, which Settings > Voice compares against.
        "audiocpp-qwen3-asr-0.6b",
        "someone/Parakeet-GGUF",
    ]


# ---------------------------------------------------------------------------
# Worker backend request shaping, against a recording fake server


def _wav(seconds = 0.1, rate = 24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


class _FakeServer:
    def __init__(self, model, reply):
        self.model = model
        self.model_id = "studio-test"
        self.backend = "cuda"
        self.reply = reply
        self.calls = []

    def alive(self):
        return True

    def post_json(self, path, payload, **kwargs):
        self.calls.append((path, payload))
        return self.reply

    def stop(self):
        pass


def _backend_with(model, reply):
    b = audio_cpp_backend.AudioCppBackend()
    b._server = _FakeServer(model, reply)
    b._model = model
    b.models = {model.id: {"is_audio": True, "audio_type": model.audio_type}}
    b.active_model_name = model.id
    return b, b._server


def _speech_model(family, **kw):
    policy = acm.family_policy(family, None, (kw.pop("name", family),))
    return _model(
        f"{family}-GGUF",
        family,
        policy.task,
        server_task = policy.default_server_task,
        request_defaults = dict(policy.request_defaults),
        **kw,
    )


SPEECH_WAV = ("audio/wav", _wav())
MUSIC_REPLY = (
    "application/json",
    json.dumps({"audio": base64.b64encode(_wav()).decode()}).encode(),
)


def test_speech_never_forwards_generic_sampling():
    b, srv = _backend_with(_speech_model("moss_tts_nano"), SPEECH_WAV)
    wav, rate = b.generate_audio_response("hello", temperature = 0.6, top_p = 0.9, seed = 7)
    path, body = srv.calls[0]
    assert path == "/v1/audio/speech" and rate == 24000 and wav[:4] == b"RIFF"
    assert body == {"model": "studio-test", "input": "hello", "seed": "7"}


def test_speech_passes_instruction_language_and_default_voice():
    b, srv = _backend_with(_speech_model("pocket_tts"), SPEECH_WAV)
    b.generate_audio_response("hi", instructions = " warm ", language = "en")
    body = srv.calls[0][1]
    assert body["voice"] == "alba"
    assert body["options"] == {"instruct": "warm", "language": "en"}


def test_audio_options_reach_the_request_validated():
    schema = acm.option_schema(acm.FAMILIES["kokoro_tts"], KOKORO_SPEC, None)
    b, srv = _backend_with(_speech_model("kokoro_tts", options = schema), SPEECH_WAV)
    b.generate_audio_response(
        "hi", audio_options = {"voice": "am_adam", "speed": 1.5, "bogus": 1, "mode": "zzz"}
    )
    body = srv.calls[0][1]
    assert body["voice"] == "am_adam" and body["options"] == {"speed": 1.5}


def test_speech_retry_keeps_package_defaults_and_user_options_and_covers_500():
    from core.inference.audio_cpp_server import AudioCppRequestError

    design = _speech_model("qwen3_tts", name = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF")
    b, srv = _backend_with(design, SPEECH_WAV)
    calls = []

    def post_json(path, payload, **kwargs):
        calls.append(json.loads(json.dumps(payload)))
        if len(calls) == 1:
            raise AudioCppRequestError(500, "unknown option: language")
        return SPEECH_WAV

    srv.post_json = post_json
    b.generate_audio_response("hi", language = "en")
    assert calls[0]["options"]["language"] == "en" and calls[0]["options"]["instruct"]
    # The retry drops the request's hints, never the package's own instruction.
    assert calls[1]["options"] == design.request_defaults["options"]


def test_a_busy_server_is_not_retried():
    from core.inference.audio_cpp_server import AudioCppRequestError

    b, srv = _backend_with(_speech_model("kokoro_tts"), SPEECH_WAV)

    def post_json(path, payload, **kwargs):
        raise AudioCppRequestError(503, "server_busy")

    srv.post_json = post_json
    with pytest.raises(RuntimeError, match = "server_busy"):
        b.generate_audio_response("hi", instructions = "warm")


def test_music_maps_description_lyrics_and_duration():
    b, srv = _backend_with(_speech_model("ace_step"), MUSIC_REPLY)
    wav, _ = b.generate_audio_response(
        "[verse] la la", instructions = "synth pop", max_new_tokens = 1500
    )
    path, body = srv.calls[0]
    assert path == "/v1/tasks/run" and wav[:4] == b"RIFF"
    assert body["request"] == {
        "text": "synth pop",
        "duration_seconds": 60.0,
        "lyrics": "[verse] la la",
    }


def test_music_without_a_description_uses_the_prompt_and_stable_audio_drops_lyrics():
    b, srv = _backend_with(_speech_model("stable_audio"), MUSIC_REPLY)
    b.generate_audio_response("uplifting house", max_new_tokens = 0)
    assert srv.calls[0][1]["request"] == {"text": "uplifting house", "duration_seconds": 30.0}


def test_minimax_takes_the_caption_as_input_and_lyrics_and_duration_as_options():
    minimax = _speech_model("minimax_music3", options = acm.FAMILIES["minimax_music3"].options)
    b, srv = _backend_with(minimax, MUSIC_REPLY)
    b.generate_audio_response(
        "[verse] hi", instructions = "lofi", max_new_tokens = 500, seed = 3, audio_options = {"top_k": 9}
    )
    request = srv.calls[0][1]["request"]
    assert request["text"] == "lofi" and request["seed"] == "3"
    # The runtime rejects duration_sec beside duration_seconds ("conflicting option values").
    assert request["options"] == {"top_k": 9, "lyrics": "[verse] hi"}
    assert request["duration_seconds"] == 20.0
    with pytest.raises(RuntimeError, match = "needs lyrics"):
        b.generate_audio_response("", instructions = "lofi")


def test_yue2_takes_the_description_as_its_required_style():
    b, srv = _backend_with(_speech_model("yue2"), MUSIC_REPLY)
    b.generate_audio_response("[chorus] oh", instructions = "rock, male vocal")
    request = srv.calls[0][1]["request"]
    assert request["text"] == "[chorus] oh"
    assert request["options"] == {
        "style": "rock, male vocal",
        "lyrics": "[chorus] oh",
        "semantic_max_tokens": 2048,
        "semantic_min_tokens": 200,
    }
    with pytest.raises(RuntimeError, match = "style"):
        b.generate_audio_response("[chorus] oh")


def test_task_response_without_audio_is_an_error():
    with pytest.raises(RuntimeError, match = "no audio"):
        audio_cpp_backend._audio_from_task_response("application/json", b'{"text": "x"}')


def test_generation_needs_an_active_model():
    b = audio_cpp_backend.AudioCppBackend()
    with pytest.raises(RuntimeError, match = "No active audio model"):
        b.generate_audio_response("x")


def test_load_rejects_speech_to_text_models():
    b = audio_cpp_backend.AudioCppBackend()
    with pytest.raises(RuntimeError, match = "speech-to-text"):
        b.load_model(SimpleNamespace(identifier = CANARY.id, audio_cpp = CANARY))


def test_status_fields_of_a_loaded_model():
    schema = acm.option_schema(acm.FAMILIES["kokoro_tts"], KOKORO_SPEC, None)
    fields = audio_cpp_backend.model_info_fields(_speech_model("kokoro_tts", options = schema))
    assert fields["audio_family"] == "kokoro_tts" and fields["gguf_variant"] == "Q8_0"
    assert [o["name"] for o in fields["audio_options"]] == [
        "voice",
        "speed",
        "text_chunk_size",
        "mode",
    ]


# ---------------------------------------------------------------------------
# Routing, config and classification


def test_stt_routing_forces_the_audiocpp_engine_for_its_models(hub):
    import routes.inference as ri

    _put(_snapshot(hub), CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))
    assert ri._stt_engine_for_model("audiocpp-canary-180m-flash") == "audiocpp"
    assert ri._stt_engine_for_model(CANARY.id) == "audiocpp"
    for alias in ("audiocpp", "audio_cpp", "audio.cpp", " AudioCpp "):
        assert ri._resolve_stt_engine(alias) == "audiocpp"
    assert ri._stt_repo_reference("audiocpp-canary-180m-flash", "audiocpp") == AUDIO_CPP_REPO
    # Saved legacy keys report as themselves (Settings compares against them); folder ids as rows.
    assert ri._stt_resolved_model_id("audiocpp-canary-180m-flash", "audiocpp") == (
        "audiocpp-canary-180m-flash"
    )
    assert ri._stt_resolved_model_id(CANARY.id, "audiocpp") == CANARY.id


def test_stt_sidecar_resolves_any_cached_asr_and_rejects_the_rest(hub):
    from core.inference import stt_audiocpp_sidecar as s
    from core.inference.stt_sidecar import SttModelIdError, SttModelNotDownloadedError

    _kokoro(hub)
    _put(_snapshot(hub), CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))
    assert s.resolve_audio_cpp_stt_model_id(None) == f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF"
    assert s.resolve_audio_cpp_stt_model("audiocpp-canary-180m-flash").family == "canary_asr"
    assert s.is_model_downloaded(CANARY.id)
    with pytest.raises(SttModelIdError, match = "not a speech-to-text"):
        s.resolve_audio_cpp_stt_model(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF")
    with pytest.raises(SttModelIdError):
        s.resolve_audio_cpp_stt_model("small")
    with pytest.raises(SttModelNotDownloadedError):
        s.resolve_audio_cpp_stt_model(f"{AUDIO_CPP_REPO}/Parakeet-TDT-0.6B-v3-GGUF")


def test_registry_keeps_the_resident_engine_when_audiocpp_runtime_is_missing(monkeypatch):
    from core.inference import stt_audiocpp_sidecar, stt_registry

    monkeypatch.setattr(stt_audiocpp_sidecar, "is_available", lambda: False)
    monkeypatch.setattr(stt_audiocpp_sidecar, "is_model_downloaded", lambda model: True)
    assert stt_registry._model_is_downloaded("audiocpp", "audiocpp-canary-180m-flash") is False


def test_model_config_answers_umbrella_ids_without_the_hub(hub, monkeypatch):
    from utils.models import model_config

    snap = _snapshot(hub)
    _put(snap, "Supertonic-3-GGUF/supertonic-3-f16.gguf", _gguf_bytes(family = "supertonic"))
    _put(snap, CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))

    def refuse(*args, **kwargs):
        raise AssertionError("an umbrella id must not reach a Hub probe")

    monkeypatch.setattr(model_config, "detect_gguf_model_remote", refuse)
    monkeypatch.setattr(model_config, "is_vision_model", refuse)
    config = model_config.ModelConfig.from_identifier(f"{AUDIO_CPP_REPO}/Supertonic-3-GGUF")
    assert config.identifier == f"{AUDIO_CPP_REPO}/Supertonic-3-GGUF"
    assert config.is_audio and config.audio_type == "audiocpp_tts" and config.gguf_variant == "F16"
    assert not config.is_gguf and not config.is_lora and not config.is_vision
    assert config.audio_cpp.family == "supertonic" and config.is_cached
    # A dictation model is not a main-slot model.
    with pytest.raises(ValueError, match = "speech-to-text"):
        model_config.ModelConfig.from_identifier("audiocpp-canary-180m-flash")
    _put(snap, "HTDemucs-GGUF/htdemucs-q8_0.gguf", _gguf_bytes(family = "htdemucs"))
    sep = model_config.ModelConfig.from_identifier(f"{AUDIO_CPP_REPO}/HTDemucs-GGUF")
    assert sep.is_audio and sep.audio_type == "audiocpp_sep" and sep.audio_cpp.task == "sep"


def test_model_config_routes_a_hub_repo_by_its_gguf_header(hub, monkeypatch):
    from utils.models import model_config

    repo = "audio-cpp/Yue2-3B-GGUF"
    snap = _snapshot(hub, repo)
    for rel in (
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
    ):
        _put(snap, rel, b"{}")
    _put(snap, "yue2-3b-q8_0.gguf", _gguf_bytes(family = "yue2"))
    _put(snap, "yue2-vae-f16.gguf", _gguf_bytes())  # the VAE carries no spec
    monkeypatch.setattr(
        model_config, "detect_gguf_model_remote", lambda *a, **k: "yue2-3b-q8_0.gguf"
    )

    def no_llama(*a, **k):
        raise AssertionError("an audio.cpp GGUF must not reach the llama.cpp preflight")

    monkeypatch.setattr(model_config, "list_gguf_variants", no_llama)
    config = model_config.ModelConfig.from_identifier(repo)
    assert config.audio_type == "audiocpp_music" and not config.is_gguf
    assert config.gguf_variant == "Q8_0" and config.audio_cpp.is_package
    assert config.audio_cpp.model_options["session_options"]["yue2.vae_gguf"] == "yue2-vae-f16.gguf"


def test_native_audio_helpers_point_at_the_holding_repo(hub):
    from core.inference import native_audio

    _put(
        _snapshot(hub),
        "ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-q8_0.gguf",
        _gguf_bytes(family = "ace_step"),
    )
    music = f"{AUDIO_CPP_REPO}/ACE-Step1.5-GGUF"
    assert native_audio.is_native_audio_model(music)
    assert native_audio.audio_cpp_audio_type(music) == "audiocpp_music"
    assert native_audio.native_audio_security_targets(music) == [AUDIO_CPP_REPO]
    assert native_audio.native_audio_security_targets(
        "audio-cpp/Yue2-3B-GGUF", "audiocpp_music"
    ) == ["audio-cpp/Yue2-3B-GGUF"]
    assert "audiocpp_tts" in native_audio.NATIVE_AUDIO_TYPES
    with pytest.raises(ValueError, match = "GGUF variant"):
        native_audio.native_audio_download_plan(music)


def test_audio_cpp_ggufs_are_classified_off_chat(hub):
    from hub.services.models import catalog_classification as cc
    from utils.gguf_archs import is_audio_cpp_gguf_architecture

    assert is_audio_cpp_gguf_architecture(" AudioCPP ") and not is_audio_cpp_gguf_architecture(
        "llama"
    )
    snap = _kokoro(hub)
    kokoro = snap / "Kokoro-82M-GGUF" / "kokoro-82m-q8_0.gguf"
    asr = _put(snap, CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))
    sep = _put(snap, "HTDemucs-GGUF/htdemucs-q8_0.gguf", _gguf_bytes(family = "htdemucs"))
    music = _put(snap, "Yue2/yue2-3b-q8_0.gguf", _gguf_bytes(family = "yue2"))
    assert cc._gguf_path_task(kokoro) == "text-to-speech"
    assert cc._gguf_path_audio_type(kokoro) == "audiocpp_tts"
    assert cc._gguf_path_task(music) == "text-to-audio"
    assert cc._gguf_path_audio_type(music) == "audiocpp_music"
    assert cc._gguf_path_task(asr) == "automatic-speech-recognition"
    assert cc._gguf_path_task(sep) == "audio-to-audio"
    assert cc._gguf_path_audio_type(sep) == "audiocpp_sep"
    assert cc._gguf_path_audio_workflows(sep) == ["separate"]
    # Architecture alone (a remote prefix): the family comes from the names.
    assert cc._arch_to_task("audiocpp", ("audio-cpp/MiniMax-Music3-GGUF",)) == "text-to-audio"
    assert cc._arch_to_audio_type("audiocpp", ("Kokoro-82M-GGUF",)) == "audiocpp_tts"


def test_llama_refuses_an_audio_cpp_gguf_with_a_pointer_to_the_audio_page():
    from core.inference.llama_cpp import LlamaCppBackend

    probe = object.__new__(LlamaCppBackend)
    probe._architecture = "audiocpp"
    probe._gguf_header_parsed = True
    probe._model_identifier = "someone/x-GGUF"
    assert "Audio page" in probe._non_chat_gguf_refusal("x.gguf")


def test_speech_api_music_default_is_thirty_seconds():
    import routes.inference as ri
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "x"}], max_tokens = 8192)
    assert (
        ri._tts_max_new_tokens(
            payload, None, audio_type = "audiocpp_music", speech_api_default_max_tokens = True
        )
        == 750
    )
    assert ri._tts_max_new_tokens(payload, None, audio_type = "audiocpp_music") == 240 * 25


def test_public_id_of_an_audio_cpp_model_is_the_id_itself():
    from core.inference.model_ids import public_model_id
    for mid in (f"{AUDIO_CPP_REPO}/ACE-Step1.5-GGUF/turbo", f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"):
        assert public_model_id(mid) == mid
    # Unrelated three-segment paths are still paths.
    assert public_model_id("some/local/dir") != "some/local/dir"


def test_umbrella_repo_is_hidden_from_chat():
    from utils.hidden_models import is_hidden_model
    assert is_hidden_model(AUDIO_CPP_REPO)
    assert is_hidden_model("Audio-CPP/audio.cpp-GGUF")


def test_resident_estimate_scales_with_the_variant():
    import routes.inference as ri

    small = ri.audio_cpp_resident_gb(237 * 1024**2)
    big = ri.audio_cpp_resident_gb(5899 * 1024**2)
    assert 1.5 < small < 3 and 15 < big < 18


def test_one_failed_transcription_does_not_hide_the_engine(monkeypatch):
    from core.inference import stt_audiocpp_sidecar as s

    monkeypatch.setattr(s.audio_cpp_server, "is_available", lambda: True)
    s.note_runtime_inference_failure("HTTP 500 on a bad clip")
    try:
        assert s.is_available() is True
    finally:
        s.clear_runtime_inference_failure()


def test_child_home_is_private_and_per_user(tmp_path, monkeypatch):
    from core.inference import audio_cpp_server as srv
    import utils.paths.storage_roots as roots

    monkeypatch.setattr(roots, "studio_root", lambda: tmp_path / "studio")
    monkeypatch.setattr(srv, "_MAX_CHILD_HOME_LEN", len(str(tmp_path)) + 40)
    assert srv._child_home_dir() == tmp_path / "studio" / "cache" / "audiocpp-home"
    monkeypatch.setattr(srv.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    # Too long for eSpeak-ng's data-dir buffer: the short per-user temp home instead.
    monkeypatch.setattr(roots, "studio_root", lambda: tmp_path / ("x" * 60) / "studio")
    assert srv._child_home_dir().parent == tmp_path / "tmp"

    def broken():
        raise OSError("no studio root")

    monkeypatch.setattr(roots, "studio_root", broken)
    home = srv._child_home_dir()
    assert home.parent == tmp_path / "tmp" and home.name.startswith("unsloth-audiocpp-home-")


def test_espeak_models_need_an_espeak_build(tmp_path, monkeypatch):
    from core.inference import audio_cpp_server as srv

    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"x")
    kokoro = _speech_model("kokoro_tts", needs_espeak = True)
    assert "eSpeak-ng" in srv.model_runtime_problem(kokoro, str(binary))
    assert srv.model_runtime_problem(CANARY, str(binary)) is None
    (tmp_path / "espeak-ng-data.bin").write_bytes(b"x")
    assert srv.model_runtime_problem(kokoro, str(binary)) is None
    # A record that says eSpeak is not enough once the data was removed or quarantined.
    monkeypatch.setattr(srv, "read_install_record", lambda _binary: {"espeak": True})
    (tmp_path / "espeak-ng-data.bin").unlink()
    assert "eSpeak-ng" in srv.model_runtime_problem(kokoro, str(binary))
    monkeypatch.setattr(srv, "find_audio_cpp_server_binary", lambda: None)
    assert "not installed" in srv.model_runtime_problem(kokoro)


def test_child_home_is_outside_the_managed_install(monkeypatch, tmp_path):
    from core.inference import audio_cpp_server as srv

    monkeypatch.setattr(srv, "managed_audio_cpp_dir", lambda: tmp_path / "audio.cpp")
    env = srv.child_env(str(tmp_path / srv.BINARY_NAME))
    homes = {v for k, v in env.items() if k in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")}
    assert homes and not any(str(tmp_path / "audio.cpp") in h for h in homes)


def test_a_binary_that_cannot_spawn_is_a_typed_error_and_leaks_nothing(monkeypatch, tmp_path):
    import tempfile as _tempfile

    from core.inference import audio_cpp_server as srv

    made = []
    real_mkdtemp = _tempfile.mkdtemp
    monkeypatch.setattr(
        srv.tempfile,
        "mkdtemp",
        lambda **kw: made.append(real_mkdtemp(dir = tmp_path, **kw)) or made[-1],
    )
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(tmp_path / "missing" / srv.BINARY_NAME))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    with pytest.raises(srv.AudioCppUnavailableError):
        srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    assert made and not os.path.exists(made[0])


def test_a_package_config_carries_its_session_options(tmp_path, monkeypatch):
    from core.inference import audio_cpp_server as srv

    written = {}

    class Stop(Exception):
        pass

    def fake_popen(command, *a, **k):
        config = command[command.index("--config") + 1]
        written.update(json.loads(Path(config).read_text(encoding = "utf-8")))
        raise Stop()

    monkeypatch.setattr(srv, "ensure_binary", lambda: str(tmp_path / srv.BINARY_NAME))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv.subprocess, "Popen", fake_popen)
    yue2 = _model(
        "Yue2-3B-GGUF",
        "yue2",
        "music",
        primary = "",
        model_options = {"session_options": {"yue2.model_gguf": "yue2-3b-bf16.gguf"}},
    )
    with pytest.raises(Stop):
        srv.AudioCppServer.start(yue2, str(tmp_path / "pkg"))
    entry = written["models"][0]
    assert (
        entry["family"] == "yue2"
        and entry["task"] == "gen"
        and entry["path"] == str(tmp_path / "pkg")
    )
    assert entry["session_options"] == {"yue2.model_gguf": "yue2-3b-bf16.gguf"}


# ---------------------------------------------------------------------------
# Hub deletion


def test_clearing_the_hub_cache_prunes_the_link_farm(hub, monkeypatch):
    from utils import cache_inventory

    snap = _snapshot(hub)
    _put(snap, CANARY.gguf_file, b"GGUF")
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    served = audio_cpp_files.materialize(CANARY, hub_cache = hub)
    pruned = []
    real = audio_cpp_files.prune_link_farm
    monkeypatch.setattr(
        audio_cpp_files, "prune_link_farm", lambda root = None: pruned.append(root) or real(root)
    )
    monkeypatch.setattr(cache_inventory, "_training_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_inference_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_link_mode_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_reserve_downloads", lambda key: ([], None))
    monkeypatch.setattr(cache_inventory, "_release_downloads", lambda reserved: None)
    monkeypatch.setattr(cache_inventory, "_resolve_roots", lambda definition: [hub])

    def empty(root, **_kwargs):
        import shutil
        shutil.rmtree(root / ("models--" + AUDIO_CPP_REPO.replace("/", "--")))
        return cache_inventory.PurgeOutcome()

    monkeypatch.setattr(cache_inventory, "empty_cache_root", empty)
    cache_inventory.purge_cache("hf_hub")
    assert pruned == [hub]
    assert not os.path.exists(served)


def test_deleting_any_repo_prunes_the_farm_of_the_cache_it_was_in(tmp_path, monkeypatch):
    import asyncio

    from hub.services.models import deletion
    from hub.utils import hf_cache_state

    other_root = tmp_path / "old" / "hub"
    repo = "audio-cpp/Yue2-3B-GGUF"
    repo_folder = other_root / ("models--" + repo.replace("/", "--"))
    (repo_folder / "snapshots" / ("a" * 40)).mkdir(parents = True)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [other_root])
    monkeypatch.setattr(deletion.account_access, "require_installation_owner", lambda: None)
    for guard in (
        "_llama_cpp_blocks_delete",
        "_inference_backend_blocks_delete",
        "_audio_cpp_blocks_delete",
    ):
        monkeypatch.setattr(deletion, guard, lambda *a: False)
    for guard in ("_diffusion_blocks_delete", "_video_blocks_delete"):
        monkeypatch.setattr(deletion, guard, lambda *a: None)
    monkeypatch.setattr(deletion, "resolve_cached_repo_id_case", lambda repo_id, repo_type: repo_id)
    monkeypatch.setattr(deletion.downloads.registry, "begin_delete", lambda *a: True)
    monkeypatch.setattr(deletion.downloads.registry, "end_delete", lambda *a: None)
    monkeypatch.setattr(
        deletion, "_delete_cached_model_blocking", lambda *a, **k: {"deleted": True}
    )
    monkeypatch.setattr(deletion.cache_inventory, "invalidate_hf_cache_scans", lambda: None)
    pruned = []
    monkeypatch.setattr(audio_cpp_files, "prune_link_farm", lambda root = None: pruned.append(root))
    asyncio.run(deletion.delete_cached_model_response(repo, cache_path = str(repo_folder)))
    # The Hub row sends the repo folder; the farm sits beside that folder's hub root.
    assert pruned == [other_root.resolve()]


@pytest.mark.parametrize(
    "state", ["idle", "stt_loading", "stt_loaded", "speech_loading", "speech_active"]
)
def test_delete_guard_covers_resident_and_loading_audio_models(monkeypatch, state):
    from core.inference import orchestrator, stt_audiocpp_sidecar
    from hub.services.models import deletion

    side = stt_audiocpp_sidecar.AudioCppSttSidecar()
    if state == "stt_loading":
        side._loading, side._loading_model = True, f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    if state == "stt_loaded":
        side._model_id = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
        side._server = SimpleNamespace(alive = lambda: True)
    speech = "audiocpp-kokoro-82m"
    backend = SimpleNamespace(
        active_model_name = speech if state == "speech_active" else None,
        loading_models = {speech} if state == "speech_loading" else set(),
    )
    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", lambda: side)
    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
    assert deletion._audio_cpp_blocks_delete(AUDIO_CPP_REPO) is (state != "idle")
    assert deletion._audio_cpp_blocks_delete("someone/other-repo") is False


def test_umbrella_download_jobs_report_under_their_folder_row(hub):
    snap = _snapshot(hub)
    moon = _gguf_bytes(family = "moonshine_asr")
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", moon)
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf", moon)
    assert acm.folder_row_for_download(
        AUDIO_CPP_REPO, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0"
    ) == (f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF", "small/Q8_0")
    # Not cached yet: the planner key's own path still names the row and quant.
    assert acm.folder_row_for_download(
        AUDIO_CPP_REPO, "PocketTTS-GGUF/english/pocket-tts-english-q8_0"
    ) == (f"{AUDIO_CPP_REPO}/PocketTTS-GGUF", "english/Q8_0")
    assert acm.folder_row_for_download("audio-cpp/Yue2-3B-GGUF", "q8_0") is None


def test_active_downloads_name_umbrella_jobs_by_row(hub, monkeypatch):
    import asyncio

    from hub.schemas.downloads import ActiveDownload
    from hub.services.models import downloads

    job = ActiveDownload.model_construct(
        repo_id = AUDIO_CPP_REPO, variant = "Kokoro-82M-GGUF/kokoro-82m-q8_0", state = "downloading"
    )
    other = ActiveDownload.model_construct(repo_id = "a/b", variant = "Q4_K_M", state = "downloading")
    monkeypatch.setattr(
        downloads.download_lifecycle, "active_download_refs", lambda *a, **k: [job, other]
    )
    monkeypatch.setattr(
        downloads, "resolve_cached_repo_id_case", lambda repo_id, repo_type: repo_id
    )
    rows = asyncio.run(downloads.get_active_downloads_response()).downloads
    assert [(r.repo_id, r.variant) for r in rows] == [
        (f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF", "Q8_0"),
        ("a/b", "Q4_K_M"),
    ]
    only = asyncio.run(downloads.get_active_downloads_response(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"))
    assert [r.repo_id for r in only.downloads] == [f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"]


def test_stt_requests_carry_their_variant_in_the_model_id():
    from models.inference import SttLoadRequest

    folder = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    assert (
        SttLoadRequest(model = folder, engine = "audiocpp", gguf_variant = "tiny").model
        == f"{folder}:tiny"
    )
    assert SttLoadRequest(model = "audiocpp-moonshine-small", gguf_variant = "Q8_0").model == (
        "audiocpp-moonshine-small:Q8_0"
    )
    assert SttLoadRequest(model = folder, engine = "audiocpp").model == folder
    # Other engines keep their own ids untouched.
    assert SttLoadRequest(
        model = "openai/whisper-small", engine = "transformers", gguf_variant = "x"
    ).model == ("openai/whisper-small")


def test_stt_sidecar_picks_a_named_sub_variant(hub):
    from core.inference import stt_audiocpp_sidecar as s

    snap = _snapshot(hub)
    moon = _gguf_bytes(family = "moonshine_asr")
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", moon)
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf", moon)
    folder = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    assert s.resolve_audio_cpp_stt_model(f"{folder}:small").variant.key == "small/Q8_0"
    assert s.resolve_audio_cpp_stt_model(folder).variant.key == "tiny/Q8_0"
    assert s.resolve_audio_cpp_stt_model_id(f"{folder}:small") == folder


def _fake_cache_repo(snap):
    files = [
        SimpleNamespace(file_name = path.name, file_path = str(path), blob_path = None)
        for path in sorted(snap.rglob("*"))
        if path.is_file()
    ]
    return SimpleNamespace(
        repo_path = str(snap.parent.parent),
        revisions = [SimpleNamespace(snapshot_path = str(snap), files = files)],
    )


def _files_under(snap):
    return {p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file()}


def test_deleting_a_package_mix_keeps_what_another_downloaded_mix_loads(hub, monkeypatch):
    from hub.services.models import deletion

    snap = _minimax(hub, quant = "q4_0")
    header = _gguf_bytes(family = "minimax_music3")
    for rel in ("language_model_q8_0.gguf", "transformer_q8_0.gguf"):
        _put(snap, rel, header)  # Q8_0 is complete too, sharing rvq_depth_decoder_q8_0 with Q4_0
    monkeypatch.setattr(
        deletion.gguf_variants,
        "delete_variant_incomplete_blobs_result",
        lambda *a, **k: SimpleNamespace(unresolved = False, deleted = 0, deleted_bytes = 0),
    )
    deletion._delete_gguf_variant_from_repos(
        "audio-cpp/MiniMax-Music3-GGUF", "Q8_0", [_fake_cache_repo(snap)], None, root = hub
    )
    left = _files_under(snap)
    # Q8_0's own weights are gone; every file the Q4_0 mix loads is still there.
    assert "language_model_q8_0.gguf" not in left and "transformer_q8_0.gguf" not in left
    assert set(acm.package_variant_files(left)["Q4_0"]) <= left
    assert "rvq_depth_decoder_q8_0.gguf" in left
    # The last mix takes its shared files with it.
    deletion._delete_gguf_variant_from_repos(
        "audio-cpp/MiniMax-Music3-GGUF", "Q4_0", [_fake_cache_repo(snap)], None, root = hub
    )
    assert not snap.exists() or not [p for p in snap.rglob("*.gguf")]


def test_package_mixes_show_only_the_studio_option_list():
    runtime = {
        "options": {
            "request": [
                {"name": "cot", "type": "enum", "values": ["off", "full"], "default": "full"},
                {"name": "abc_temperature", "type": "float"},
                {"name": "semantic_penalty_window", "type": "int"},
            ]
        }
    }
    names = [o["name"] for o in acm.option_schema(acm.FAMILIES["yue2"], runtime, None)]
    assert names == [
        "cot",
        "num_inference_steps",
        "guidance_scale",
        "semantic_temperature",
        "semantic_top_p",
        "semantic_top_k",
    ]
    # Other families still follow the runtime's spec.
    assert [o["name"] for o in acm.option_schema(acm.FAMILIES["heartmula"], runtime, None)] == [
        "cot",
        "abc_temperature",
        "semantic_penalty_window",
    ]


def test_yue2_lists_only_its_package_mixes(hub):
    snap = _snapshot(hub, "audio-cpp/Yue2-3B-GGUF")
    for rel in (
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
    ):
        _put(snap, rel, b"{}")
    for quant in ("q8_0", "bf16", "q4_0", "ios-q4_0"):
        _put(snap, f"yue2-3b-{quant}.gguf", _gguf_bytes(family = "yue2"))
    for vae in ("f16", "f32"):
        _put(snap, f"yue2-vae-{vae}.gguf", _gguf_bytes())
    model = acm.resolve("audio-cpp/Yue2-3B-GGUF", network = False)
    assert [v.key for v in model.variants] == ["Q8_0", "BF16", "Q4_0"]
    assert all("yue2-vae-f16.gguf" in [f.path for f in v.files] for v in model.variants)


def _audio_load(
    tmp_path,
    arch,
    name = "model-Q4_K_M.gguf",
):
    path = tmp_path / name
    path.write_bytes(_gguf_bytes(arch = arch))
    config = SimpleNamespace(
        identifier = "someone/model-GGUF",
        gguf_file = str(path),
        gguf_verified = None,
        gguf_hf_repo = None,
        gguf_variant = None,
    )
    return config, SimpleNamespace(audio_device = "auto")


def test_an_audio_page_load_of_a_chat_gguf_is_refused(tmp_path, monkeypatch):
    import routes.inference as ri
    from utils.models import gguf_metadata

    config, request = _audio_load(tmp_path, "qwen3")
    monkeypatch.setattr(gguf_metadata, "read_gguf_tts_audio_type", lambda path: None)
    assert "not an audio model" in ri._audio_intent_gguf_refusal(config, request)
    # Chat loads (no audio_device) are not this check's business.
    assert ri._audio_intent_gguf_refusal(config, SimpleNamespace(audio_device = None)) is None
    # A speech-codec GGUF the page runs through llama-server passes, by vocab or by name.
    monkeypatch.setattr(gguf_metadata, "read_gguf_tts_audio_type", lambda path: "snac")
    assert ri._audio_intent_gguf_refusal(config, request) is None
    monkeypatch.setattr(gguf_metadata, "read_gguf_tts_audio_type", lambda path: None)
    orpheus, _ = _audio_load(tmp_path, "llama", name = "orpheus-3b-0.1-ft-Q4_K_M.gguf")
    assert ri._audio_intent_gguf_refusal(orpheus, request) is None
    # A file not on disk yet is left to the load's own checks.
    config.gguf_file = str(tmp_path / "absent.gguf")
    assert ri._audio_intent_gguf_refusal(config, request) is None


def test_stt_takes_the_variant_keys_gguf_variants_lists_even_from_a_partial_cache(hub):
    from core.inference import stt_audiocpp_sidecar as s
    from models.inference import SttLoadRequest

    folder = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    assert acm.split_variant_ref(f"{folder}:small/Q8_0") == (folder, "small/Q8_0")
    request = SttLoadRequest(model = folder, engine = "audiocpp", gguf_variant = "small/Q8_0")
    assert request.model == f"{folder}:small/Q8_0"
    # Only small is downloaded, so the cache alone would call it plain Q8_0.
    _put(
        _snapshot(hub),
        "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf",
        _gguf_bytes(family = "moonshine_asr"),
    )
    for wanted in ("small/Q8_0", "small", "Q8_0"):
        model = s.resolve_audio_cpp_stt_model(f"{folder}:{wanted}")
        assert model.unsupported is None
        assert model.gguf_file.endswith("moonshine-streaming-small-q8_0.gguf")
    assert s.resolve_audio_cpp_stt_model(f"{folder}:small/Q8_0").variant.key == "small/Q8_0"
    assert s.resolve_audio_cpp_stt_model(f"{folder}:small").variant.key == "small/Q8_0"
    assert s.is_model_downloaded(f"{folder}:small/Q8_0")
    # A variant that is not on disk is still refused rather than swapped for another.
    tiny = acm.resolve(folder, "tiny/Q8_0", network = False)
    assert "not found" in tiny.unsupported
    # To dictation that is a download to offer, not a bad model id.
    from core.inference.stt_sidecar import SttModelIdError, SttModelNotDownloadedError

    with pytest.raises(SttModelNotDownloadedError, match = r"\(tiny/Q8_0\) is not downloaded"):
        s.resolve_audio_cpp_stt_model(f"{folder}:tiny/Q8_0")
    with pytest.raises(SttModelNotDownloadedError):
        s.resolve_audio_cpp_stt_model(folder, "tiny/Q8_0")
    assert not s.is_model_downloaded(f"{folder}:tiny/Q8_0")
    # Another task's folder is still the wrong model, quant or not.
    _kokoro(hub)
    with pytest.raises(SttModelIdError):
        s.resolve_audio_cpp_stt_model(f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF:Q8_0")
    # A file stem is a spelling of its own row, not a new scope.
    stem = acm.resolve(folder, "moonshine-streaming-small-q8_0", network = False)
    assert stem.variant.key == "Q8_0" and stem.unsupported is None


def test_yue2_length_follows_the_requested_duration():
    b, srv = _backend_with(_speech_model("yue2"), MUSIC_REPLY)
    b.generate_audio_response("[chorus] oh", instructions = "rock", max_new_tokens = 125)
    options = srv.calls[0][1]["request"]["options"]
    # 5 s at 25 frames per second, with the default 200-frame floor lowered to fit.
    assert options["semantic_max_tokens"] == 125 and options["semantic_min_tokens"] == 125
    b.generate_audio_response("[chorus] oh", instructions = "rock", max_new_tokens = 1500)
    options = srv.calls[1][1]["request"]["options"]
    assert options["semantic_max_tokens"] == 1500 and options["semantic_min_tokens"] == 200


def test_music_requests_missing_a_required_field_are_a_400_message():
    import routes.inference as ri

    assert ri._audio_cpp_music_request_problem("yue2", "[verse] la", "") == (
        "YuE2 needs a style description."
    )
    assert ri._audio_cpp_music_request_problem("minimax_music3", " ", "lofi") == (
        "MiniMax Music 3 needs lyrics."
    )
    assert ri._audio_cpp_music_request_problem("yue2", "", "rock") is None
    assert ri._audio_cpp_music_request_problem("ace_step", "", "") is None


def test_a_variantless_stt_id_keeps_the_loaded_variant_and_an_explicit_one_switches(
    hub, monkeypatch
):
    from core.inference import stt_audiocpp_sidecar as s

    snap = _snapshot(hub)
    moon = _gguf_bytes(family = "moonshine_asr")
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", moon)
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf", moon)
    folder = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    small = s.resolve_audio_cpp_stt_model(f"{folder}:small/Q8_0")

    started = []

    class _Server:
        def __init__(self, model):
            self.model, self.backend = model, "cpu"

        def alive(self):
            return True

        def stop(self):
            pass

    def start(model, path, **kwargs):
        started.append(model.variant.key)
        return _Server(model)

    monkeypatch.setattr(s, "ensure_engine_available", lambda: "audiocpp_server")
    monkeypatch.setattr(s.AudioCppServer, "start", start)
    side = s.AudioCppSttSidecar()
    try:
        side.load(f"{folder}:small/Q8_0")
        assert started == ["small/Q8_0"]
        # Settings and dictation send the bare row id: that is the loaded small, not the default tiny.
        assert side.keep_loaded_variant(folder) == small.canonical_id
        assert side.keep_loaded_variant("audiocpp-moonshine-tiny") != small.canonical_id
        side.load(folder)
        assert started == ["small/Q8_0"] and side.loaded_variant == "small/Q8_0"
        # An explicit variant still switches.
        side.load(f"{folder}:tiny/Q8_0")
        assert started == ["small/Q8_0", "tiny/Q8_0"] and side.loaded_variant == "tiny/Q8_0"
    finally:
        side.unload()
    # Nothing loaded: the bare row id means the row's default.
    assert side.keep_loaded_variant(folder) == folder


def test_stt_status_speaks_legacy_keys_to_the_clients_that_saved_them(hub, monkeypatch):
    from core.inference import stt_audiocpp_sidecar as s

    snap = _snapshot(hub)
    moon = _gguf_bytes(family = "moonshine_asr")
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", moon)
    _put(snap, "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf", moon)
    _put(snap, CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))
    downloaded = s.downloaded_model_ids()
    folder = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    # Folder ids for the Audio page, and each key whose own folder and variant is on disk.
    for name in (
        folder,
        CANARY.id,
        "audiocpp-canary-180m-flash",
        "audiocpp-moonshine-tiny",
        "audiocpp-moonshine-small",
    ):
        assert name in downloaded
    assert "audiocpp-parakeet-tdt-0.6b-v3" not in downloaded

    class _Server:
        def __init__(self, model):
            self.model, self.backend = model, "cpu"

        def alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(s, "ensure_engine_available", lambda: "audiocpp_server")
    monkeypatch.setattr(s.AudioCppServer, "start", lambda model, path, **kw: _Server(model))
    side = s.AudioCppSttSidecar()
    try:
        side.load("audiocpp-moonshine-small")
        assert (
            side.loaded_model == "audiocpp-moonshine-small" and side.loaded_variant == "small/Q8_0"
        )
        # The same model named by its folder row reports the row, without a restart.
        side.load(f"{folder}:small/Q8_0")
        assert side.loaded_model == folder
        side.load("audiocpp-moonshine-small")
        assert side.loaded_model == "audiocpp-moonshine-small"
        # Unloading by either name finds it.
        side.unload(expected_model = folder)
        assert side.loaded_model is None
    finally:
        side.unload()


def test_stt_errors_do_not_name_the_engine(hub):
    from core.inference import stt_audiocpp_sidecar as s
    from core.inference.stt_sidecar import SttModelIdError

    with pytest.raises(SttModelIdError) as refused:
        s.resolve_audio_cpp_stt_model("small")
    assert "audio.cpp" not in str(refused.value) and "audio runtime" in str(refused.value)
    unknown = acm.family_policy("brand_new_family", None)
    assert "audio.cpp" not in unknown.unsupported


def test_standalone_music_gguf_rows_are_not_offered_to_chat(hub):
    from huggingface_hub import scan_cache_dir

    from hub.services.models import cache_inventory

    _put(_snapshot(hub, "audio-cpp/Yue2-3B-GGUF"), "yue2-3b-q8_0.gguf", _gguf_bytes(family = "yue2"))
    # The fixture cache passed in: left to itself the scanner walks the machine's own caches.
    scans = [scan_cache_dir(hub)]
    rows = {
        r["repo_id"]: r
        for r in cache_inventory._scan_cached_gguf(cache_scans = scans, active_hub_cache = hub)
    }
    assert rows["audio-cpp/Yue2-3B-GGUF"]["task"] == "text-to-audio"
    assert rows["audio-cpp/Yue2-3B-GGUF"]["capabilities"]["can_chat"] is False


def test_a_miss_without_a_token_is_not_served_to_a_caller_with_one(monkeypatch):
    seen = []

    def fake(ref, wanted, hf_token, network, tags):
        seen.append(hf_token)
        return CANARY if hf_token else None

    monkeypatch.setattr(acm, "_resolve_uncached", fake)
    acm._resolve_cache.clear()
    assert acm.resolve("someone/Gated-ASR-GGUF") is None
    assert acm.resolve("someone/Gated-ASR-GGUF", hf_token = "hf_secret") is CANARY
    assert acm.resolve("someone/Gated-ASR-GGUF", hf_token = "hf_secret") is CANARY
    assert seen == [None, "hf_secret"]


def test_deleting_from_an_inactive_cache_prunes_that_caches_farm(monkeypatch, tmp_path):
    import asyncio

    from hub.services.models import deletion
    from hub.utils import hf_cache_state

    active, old = tmp_path / "active" / "hub", tmp_path / "old" / "hub"
    pruned = []
    monkeypatch.setattr(deletion.account_access, "require_installation_owner", lambda: None)
    for name in (
        "_llama_cpp_blocks_delete",
        "_inference_backend_blocks_delete",
        "_audio_cpp_blocks_delete",
    ):
        monkeypatch.setattr(deletion, name, lambda *a: False)
    monkeypatch.setattr(deletion, "_diffusion_blocks_delete", lambda *a: None)
    monkeypatch.setattr(deletion, "_video_blocks_delete", lambda *a: None)
    monkeypatch.setattr(deletion, "resolve_cached_repo_id_case", lambda r, **k: r)
    # The repo lives only in the remembered, inactive cache; cache_path is omitted.
    monkeypatch.setattr(
        deletion, "_delete_cached_model_blocking", lambda *a, **k: {"status": "deleted"}
    )
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda *a: [active, old])
    monkeypatch.setattr(
        audio_cpp_files, "prune_link_farm", lambda root = None: pruned.append(root) or 0
    )
    asyncio.run(deletion.delete_cached_model_response("audio-cpp/Yue2-3B-GGUF"))
    assert old in pruned


def test_a_dictation_server_training_moved_to_cpu_returns_to_the_gpu_after(hub, monkeypatch):
    from core.inference import stt_audiocpp_sidecar as s

    _put(_snapshot(hub), CANARY.gguf_file, _gguf_bytes(family = "canary_asr"))
    started = []

    class _Server:
        def __init__(self, cpu):
            self.backend = "cpu" if cpu else "cuda"

        def alive(self):
            return True

        def stop(self):
            pass

    def start(
        model,
        path,
        force_cpu = False,
        **kwargs,
    ):
        started.append(force_cpu)
        return _Server(force_cpu)

    training = [True]
    monkeypatch.setattr(s, "ensure_engine_available", lambda: "audiocpp_server")
    monkeypatch.setattr(s, "_training_active", lambda: training[0])
    monkeypatch.setattr(s.AudioCppServer, "start", start)
    side = s.AudioCppSttSidecar()
    model = f"{AUDIO_CPP_REPO}/Canary-180M-Flash-GGUF"
    try:
        side.load(model, device = "gpu")
        side.load(model, device = "gpu")  # still training: kept on the CPU, no restart
        assert started == [True]
        training[0] = False
        side.load(model, device = "gpu")
        assert started == [True, False] and side.device == "cuda"
    finally:
        side.unload()


def test_a_remote_header_miss_without_a_token_is_not_served_to_one_with_it(monkeypatch):
    from core.inference import diffusion_compat

    seen = []

    def read(repo, filename, token, **kw):
        seen.append(token)
        return _gguf_bytes(family = "canary_asr") if token else None

    monkeypatch.setattr(diffusion_compat, "_read_gguf_header", read)
    assert acm.read_remote_header("someone/Gated-GGUF", "a.gguf", None, size = 64) is None
    header = acm.read_remote_header("someone/Gated-GGUF", "a.gguf", "hf_secret", size = 64)
    assert header is not None and header.family == "canary_asr" and seen == [None, "hf_secret"]


def test_a_finished_download_is_listed_on_the_next_status_poll(hub):
    """Status polls during the download cache a listing without the model; the download's own
    forget() must drop it, or the first poll after completion reads as a failed download."""
    assert acm.downloaded_models("asr") == []
    _put(
        _snapshot(hub),
        "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
        _gguf_bytes(family = "qwen3_asr"),
    )
    acm.forget(f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF")
    assert [m.id for m in acm.downloaded_models("asr")] == [f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF"]


def test_a_variant_download_reports_the_row_the_caller_tracks():
    """The Audio page tracks the bare row; a variant pick reaches the worker as ``row:variant``."""
    import threading

    from core.inference.stt_audiocpp_sidecar import _AudioCppDownloadState

    state = _AudioCppDownloadState()
    release = threading.Event()
    state._model_id = f"{AUDIO_CPP_REPO}/Moonshine-GGUF:tiny/Q8_0"
    state._thread = threading.Thread(target = release.wait, daemon = True)
    state._thread.start()
    try:
        assert state.status()["model"] == f"{AUDIO_CPP_REPO}/Moonshine-GGUF"
    finally:
        release.set()


def test_missing_files_are_counted_in_the_snapshot_downloads_land_in(hub):
    from dataclasses import replace

    old = _snapshot(hub, sha = "b" * 40, main = False)
    main = _snapshot(hub)  # refs/main
    gguf = RepoFile("PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf", 4)
    voice = RepoFile("PocketTTS-GGUF/english/embeddings/alba.safetensors", 1)
    _put(old, gguf.path, b"GGUF")  # an older revision holds more of the files
    variant = acm.AudioCppVariant("english/Q8_0", (gguf, voice), gguf.path)
    model = replace(CANARY, folder = "PocketTTS-GGUF", variant = variant, variants = (variant,))
    # Downloads go to refs/main, so that is where the gap is measured; the old snapshot would
    # report only the voice missing and never become complete.
    assert audio_cpp_files.missing_files(model, hub_cache = hub) == [(gguf.path, 4), (voice.path, 1)]
    _put(main, gguf.path, b"GGUF")
    _put(main, voice.path, b"v")
    assert audio_cpp_files.missing_files(model, hub_cache = hub) == []


def test_required_inputs_come_from_the_raw_spec(hub):
    snap = _snapshot(hub)
    maya_spec = {
        "family": "maya1",
        "tasks": ["tts"],
        "options": {
            "request": [
                {"name": "instruct", "type": "string", "required": True},
                {"name": "temperature", "type": "float"},
            ]
        },
    }
    _put(snap, "Maya1-GGUF/maya1-q8_0.gguf", _gguf_bytes(family = "maya1", spec = maya_spec))
    maya = acm.resolve(f"{AUDIO_CPP_REPO}/Maya1-GGUF", network = False)
    # instruct travels as the request's description, not as an option.
    assert maya.required_inputs == ("instruct",)
    assert [o["name"] for o in maya.options] == ["temperature"]
    # A voice-design package fills instruct itself.
    folder = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF"
    design_spec = {**maya_spec, "family": "qwen3_tts"}
    _put(snap, f"{folder}/m-q8_0.gguf", _gguf_bytes(family = "qwen3_tts", spec = design_spec))
    assert acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", network = False).required_inputs == ()


def _opt_spec(name, type_, **kw):
    return {"name": name, "type": type_, "description": name, **kw}


# The request options of the runtime's rvc and seed_vc specs (model_specs/*.json at the pin).
RVC_SPEC = {
    "family": "rvc",
    "tasks": ["vc"],
    "options": {
        "request": [
            _opt_spec("voice_id", "enum", values = ["default", "manthos", "chocola", "fraise"]),
            _opt_spec("voice_model_path", "path"),
            _opt_spec("pitch_extractor", "enum", values = ["rmvpe"], default = "rmvpe"),
            _opt_spec("pitch_path", "path"),
            _opt_spec("retrieval_index_path", "path"),
            _opt_spec("retrieval_blend", "float", default = 0.0, min = 0.0, max = 1.0),
            _opt_spec("semitone_shift", "int", default = 0),
            _opt_spec("pitch_filter_radius", "int", default = 3, min = 0),
            _opt_spec("output_sample_rate", "int", default = 0, min = 0),
            _opt_spec("rms_mix_rate", "float", default = 0.25),
            _opt_spec("unvoiced_protection", "float", default = 0.33, min = 0.0, max = 1.0),
            _opt_spec("speaker_id", "int", default = 0, min = 0),
            _opt_spec("audio_pad_duration_sec", "int", default = 1, min = 1),
            _opt_spec("split_query_sec", "int", default = 5, min = 1),
        ]
    },
}
SEED_VC_SPEC = {
    "family": "seed_vc",
    "tasks": ["vc", "svc"],
    "options": {
        "request": [
            _opt_spec(
                "route",
                "enum",
                values = ["v2_vc", "v1_svc", "v1_whisper_bigvgan_vc", "v1_xlsr_hift_vc"],
            ),
            _opt_spec("length_adjust", "float", default = 1.0, min = 0.0),
            _opt_spec("num_inference_steps", "int", default = 30, min = 1),
            _opt_spec("inference_guidance_scale", "float", default = 0.7, min = 0.0),
            _opt_spec("intelligibility_guidance_scale", "float", default = 0.7, min = 0.0),
            _opt_spec("similarity_guidance_scale", "float", default = 0.7, min = 0.0),
            _opt_spec("voice_anonymization", "bool", default = False),
            _opt_spec("seed", "int", min = 0),
            _opt_spec("noise_path", "path"),
            _opt_spec("f0_condition", "bool"),
            _opt_spec("auto_f0_adjust", "bool", default = False),
            _opt_spec("semitone_shift", "int", default = 0),
        ]
    },
}
MEANVC2_SPEC = {
    "family": "meanvc2",
    "tasks": ["vc"],
    "options": {"request": [_opt_spec("seed", "int", default = 42, min = 0)]},
}
_NEVER_IN_CONVERT = {
    "voice_model_path",
    "retrieval_index_path",
    "pitch_path",
    "noise_path",
    "audio_pad_duration_sec",
    "seed",
}


def test_voice_conversion_families_resolve_and_offer_convert_alone(hub):
    snap = _snapshot(hub)
    _put(snap, "RVC-GGUF/rvc-f16.gguf", _gguf_bytes(family = "rvc", spec = RVC_SPEC))
    for name in ("seed-vc-mlx-q8_0", "seed-vc-mlx-f16", "seed-vc-mlx-q4_k"):
        _put(
            snap,
            f"SeedVC-MLX-GGUF/{name}.gguf",
            _gguf_bytes(family = "seed_vc", spec = SEED_VC_SPEC),
        )
    for name in ("meanvc2-120ms-40ms-fp32", "meanvc2-120ms-40ms-q4_k"):
        _put(
            snap,
            f"MeanVC2-GGUF/{name}.gguf",
            _gguf_bytes(family = "meanvc2", spec = MEANVC2_SPEC),
        )
    rvc = acm.resolve(f"{AUDIO_CPP_REPO}/RVC-GGUF", network = False)
    seed_vc = acm.resolve(f"{AUDIO_CPP_REPO}/SeedVC-MLX-GGUF", network = False)
    meanvc2 = acm.resolve(f"{AUDIO_CPP_REPO}/MeanVC2-GGUF", network = False)
    for model in (rvc, seed_vc, meanvc2):
        assert model.unsupported is None, model.family
        assert (model.task, model.server_task, model.speaks) == ("tts", "vc", False)
        assert list(model.workflows) == ["convert"]
        acm.require_runnable(model, "tts")
        names = {o["name"] for o in model.convert_options}
        assert not names & _NEVER_IN_CONVERT, model.family
        assert not names & acm.CONVERT_DRIVEN_OPTIONS, model.family
    assert seed_vc.variant.key == "Q8_0"
    assert meanvc2.variant.key == "FP32"
    assert [o["name"] for o in rvc.convert_options] == [
        "retrieval_blend",
        "pitch_filter_radius",
        "output_sample_rate",
        "rms_mix_rate",
        "unvoiced_protection",
        "split_query_sec",
    ]
    assert {"route", "length_adjust", "similarity_guidance_scale"} <= {
        o["name"] for o in seed_vc.convert_options
    }
    assert "f0_condition" not in {o["name"] for o in seed_vc.options}
    assert meanvc2.convert_options == ()


def test_samsone_is_refused_for_transcription(hub):
    from core.inference import stt_audiocpp_sidecar as s
    from core.inference.stt_sidecar import SttModelIdError

    spec = {"category": "asr", "tasks": ["asr"]}
    _put(_snapshot(hub), "Samsone-GGUF/s-q8_0.gguf", _gguf_bytes(family = "samsone", spec = spec))
    with pytest.raises(SttModelIdError, match = "describes audio"):
        s.resolve_audio_cpp_stt_model(f"{AUDIO_CPP_REPO}/Samsone-GGUF")
    assert f"{AUDIO_CPP_REPO}/Samsone-GGUF" not in s.downloaded_model_ids()
