# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stt_capabilities: what each speech-to-text model adds to a transcript, answered offline."""

import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from core.inference import stt_capabilities as caps
from core.inference.audio_cpp_models import AUDIO_CPP_REPO as REPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401


def _add(hub, folder, family):
    _put(_snapshot(hub), f"{folder}/{folder.lower()}-q8_0.gguf", _gguf_bytes(family = family))
    acm.forget()


@pytest.mark.parametrize(
    "folder,family,timestamps,speakers,cpu_only",
    [
        ("Qwen3-ASR-0.6B-GGUF", "qwen3_asr", "on_request", False, False),
        ("MOSS-Transcribe-Diarize-GGUF", "moss_transcribe_diarize", "always", True, False),
        ("VibeVoice-ASR-GGUF", "vibevoice_asr", "always", True, False),
        ("VibeVoice-ASR-Streaming-GGUF", "vibevoice_asr_streaming", "unsupported", False, False),
        ("Parakeet-TDT-0.6B-v3-GGUF", "parakeet_tdt", "always", False, False),
        ("Kroko-ASR-GGUF", "kroko_asr", "always", False, False),
        # Its spans are sub-word pieces, so Studio treats it as plain text.
        ("Nemotron-3.5-ASR-Streaming-0.6B-GGUF", "nemotron_asr", "unsupported", False, False),
        ("Canary-180M-Flash-GGUF", "canary_asr", "unsupported", False, False),
        ("Niagara-ASR-GGUF", "niagara_asr", "unsupported", False, True),
    ],
)
@pytest.mark.parametrize("downloaded", [True, False])
def test_audio_cpp_table(hub, folder, family, timestamps, speakers, cpu_only, downloaded):
    if downloaded:
        _add(hub, folder, family)
    for engine in ("audiocpp", None):
        result = caps.capabilities_for(f"{REPO}/{folder}", engine)
        assert (result["engine"], result["family"]) == ("audiocpp", family)
        assert (result["timestamps"], result["speakers"]) == (timestamps, speakers)
        assert result["cpu_only"] is cpu_only
        assert (result["aligner"] is not None) == (timestamps == "on_request")


def test_qwen3_reports_whether_its_aligner_is_downloaded(hub):
    def aligner():
        return caps.capabilities_for(f"{REPO}/Qwen3-ASR-1.7B-GGUF", "audiocpp")["aligner"]

    assert aligner() == {"downloaded": False, "size_bytes": 1_129_966_496}
    _add(hub, "Qwen3-ForcedAligner-0.6B-GGUF", "qwen3_forced_aligner")
    assert aligner()["downloaded"] is True and aligner()["size_bytes"] > 0


@pytest.mark.parametrize(
    "model,engine,expected_engine",
    [
        ("qwen3-asr-0.6b", "mtmd", "mtmd"),
        ("qwen3-asr-0.6b", None, "mtmd"),
        ("unslothai/Qwen3-ASR-0.6B-GGUF", "audiocpp", "mtmd"),
        ("small", "transformers", "transformers"),
        ("small", "gguf", "gguf"),
        ("small", None, "transformers"),
        (None, None, "transformers"),
        ("", "audiocpp", "audiocpp"),
        ("nobody/Nothing-Here", None, "transformers"),
        ("nobody/Nothing-Here", "audiocpp", "audiocpp"),
        (f"{REPO}/Samsone-GGUF", "audiocpp", "audiocpp"),
        (f"{REPO}/Qwen3-ASR-0.6B-GGUF", "no-such-engine", None),
    ],
)
def test_other_engines_and_unknown_models_are_plain_text(hub, model, engine, expected_engine):
    result = caps.capabilities_for(model, engine)
    assert result == {
        "engine": expected_engine,
        "family": None,
        "timestamps": "unsupported",
        "speakers": False,
        "aligner": None,
        "cpu_only": False,
    }


def test_a_failing_lookup_still_answers(hub, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("cache unreadable")

    monkeypatch.setattr(acm, "resolve", broken)
    monkeypatch.setattr(acm, "family_from_names", broken)
    result = caps.capabilities_for(f"{REPO}/MOSS-Transcribe-Diarize-GGUF", "audiocpp")
    assert result["timestamps"] == "unsupported" and result["speakers"] is False


def test_speakers_are_labelled_in_the_order_they_first_speak():
    assert caps.label_speakers(["S03", "S01", "S03", "0"]) == [
        {"id": "S03", "label": "Speaker 1"},
        {"id": "S01", "label": "Speaker 2"},
        {"id": "0", "label": "Speaker 3"},
    ]
    assert caps.label_speakers(None) == []
