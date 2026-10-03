# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stt_capabilities: what each speech-to-text model adds to a transcript, answered offline."""

import struct

import pytest

from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import stt_capabilities as caps
from core.inference.audio_cpp_models import AUDIO_CPP_REPO


def _gguf_bytes(family) -> bytes:
    kv = [
        ("general.architecture", 8, "audiocpp"),
        ("audiocpp.model_spec.version", 4, 1),
        ("audiocpp.model_spec.family", 8, family),
    ]
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)))
    for key, vtype, value in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            out += struct.pack("<Q", len(v)) + v
        else:
            out += struct.pack("<I", value)
    return bytes(out) + b"\0" * 64


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    acm.forget()
    yield root
    acm.forget()


def _put(hub, rel, family):
    repo_dir = hub / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    path = repo_dir / "snapshots" / ("a" * 40) / rel
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text("a" * 40)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_gguf_bytes(family))
    acm.forget()


def _row(folder):
    return f"{AUDIO_CPP_REPO}/{folder}"


@pytest.mark.parametrize(
    "folder,family,timestamps,speakers",
    [
        ("Qwen3-ASR-0.6B-GGUF", "qwen3_asr", "on_request", False),
        ("MOSS-Transcribe-Diarize-GGUF", "moss_transcribe_diarize", "always", True),
        ("VibeVoice-ASR-GGUF", "vibevoice_asr", "always", True),
        (
            "VibeVoice-ASR-Streaming-GGUF",
            "vibevoice_asr_streaming",
            "unsupported",
            False,
        ),
        ("Parakeet-TDT-0.6B-v3-GGUF", "parakeet_tdt", "always", False),
        ("Kroko-ASR-GGUF", "kroko_asr", "always", False),
        # Its spans are sub-word pieces, so Studio treats it as plain text.
        ("Nemotron-3.5-ASR-Streaming-0.6B-GGUF", "nemotron_asr", "unsupported", False),
        ("Canary-180M-Flash-GGUF", "canary_asr", "unsupported", False),
    ],
)
def test_the_table_for_downloaded_audio_cpp_models(hub, folder, family, timestamps, speakers):
    _put(hub, f"{folder}/{folder.lower()}-q8_0.gguf", family)
    for engine in ("audiocpp", None):
        result = caps.capabilities_for(_row(folder), engine)
        assert result["engine"] == "audiocpp" and result["family"] == family
        assert (result["timestamps"], result["speakers"]) == (timestamps, speakers)
        assert result["cpu_only"] is False
        assert (result["aligner"] is not None) == (timestamps == "on_request")


def test_qwen3_reports_whether_its_aligner_is_downloaded(hub):
    _put(hub, "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf", "qwen3_asr")
    assert caps.capabilities_for(_row("Qwen3-ASR-0.6B-GGUF"), "audiocpp")["aligner"] == {
        "downloaded": False,
        "size_bytes": 1_129_966_496,
    }
    _put(
        hub,
        "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf",
        "qwen3_forced_aligner",
    )
    aligner = caps.capabilities_for(_row("Qwen3-ASR-0.6B-GGUF"), "audiocpp")["aligner"]
    assert aligner["downloaded"] is True and aligner["size_bytes"] > 0


def test_an_undownloaded_model_is_judged_by_its_name(hub):
    moss = caps.capabilities_for(_row("MOSS-Transcribe-Diarize-GGUF"), None)
    assert (moss["engine"], moss["family"]) == ("audiocpp", "moss_transcribe_diarize")
    assert (moss["timestamps"], moss["speakers"]) == ("always", True)
    qwen3 = caps.capabilities_for(_row("Qwen3-ASR-1.7B-GGUF"), "audiocpp")
    assert qwen3["timestamps"] == "on_request" and qwen3["aligner"]["downloaded"] is False


def test_llama_cpp_qwen3_and_whisper_engines_have_no_timestamps(hub):
    for model, engine in (
        ("qwen3-asr-0.6b", "mtmd"),
        ("qwen3-asr-0.6b", None),
        ("unslothai/Qwen3-ASR-0.6B-GGUF", "audiocpp"),
    ):
        result = caps.capabilities_for(model, engine)
        assert result["engine"] == "mtmd" and result["family"] is None
        assert (result["timestamps"], result["speakers"], result["aligner"]) == (
            "unsupported",
            False,
            None,
        )
    for engine in ("transformers", "gguf", None):
        result = caps.capabilities_for("small", engine)
        assert result["timestamps"] == "unsupported" and result["speakers"] is False


def test_niagara_runs_on_the_cpu(hub):
    result = caps.capabilities_for(_row("Niagara-ASR-GGUF"), "audiocpp")
    assert result["family"] == "niagara_asr" and result["cpu_only"] is True
    assert result["timestamps"] == "unsupported"


@pytest.mark.parametrize(
    "model,engine",
    [
        (None, None),
        ("", "audiocpp"),
        ("nobody/Nothing-Here", None),
        ("nobody/Nothing-Here", "audiocpp"),
        (_row("Samsone-GGUF"), "audiocpp"),
        (_row("Qwen3-ASR-0.6B-GGUF"), "no-such-engine"),
    ],
)
def test_unknown_models_are_plain_text_not_errors(hub, model, engine):
    result = caps.capabilities_for(model, engine)
    assert set(result) == {"engine", "family", "timestamps", "speakers", "aligner", "cpu_only"}
    assert (result["timestamps"], result["speakers"], result["aligner"]) == (
        "unsupported",
        False,
        None,
    )
    assert result["family"] is None


def test_a_failing_lookup_still_answers(hub, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("cache unreadable")

    monkeypatch.setattr(acm, "resolve", broken)
    monkeypatch.setattr(acm, "family_from_names", broken)
    result = caps.capabilities_for(_row("MOSS-Transcribe-Diarize-GGUF"), "audiocpp")
    assert result["timestamps"] == "unsupported" and result["speakers"] is False


def test_speakers_are_labelled_in_the_order_they_first_speak():
    assert caps.label_speakers(["S03", "S01", "S03", "0"]) == [
        {"id": "S03", "label": "Speaker 1"},
        {"id": "S01", "label": "Speaker 2"},
        {"id": "0", "label": "Speaker 3"},
    ]
    assert caps.label_speakers([]) == [] and caps.label_speakers(None) == []
