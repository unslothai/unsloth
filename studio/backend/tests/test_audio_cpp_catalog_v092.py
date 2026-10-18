# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Folders audio.cpp v0.9.2 runs that Unsloth Studio now offers: their name hints and the language
each transcription request carries."""

import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from core.inference import stt_audiocpp_sidecar as stt
from core.inference.audio_cpp_models import AUDIO_CPP_REPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401
from test_stt_audiocpp_details import ABSENT, fake, side, source  # noqa: E402, F401


def _names(folder, gguf = None):
    return (f"{AUDIO_CPP_REPO}/{folder}", folder, *([gguf] if gguf else []))


@pytest.mark.parametrize(
    ("names", "family"),
    [
        # Hub names on audio-cpp/audio.cpp-gguf; Voxtral TTS speaks, it is not the realtime ASR.
        (_names("Voxtral-4B-TTS-2603-GGUF"), None),
        (_names("Voxtral-4B-TTS-2603-GGUF", "voxtral-4b-tts-2603-q8_0.gguf"), None),
        (_names("Voxtral-4B-TTS-2603-GGUF", "voxtral-4b-tts-2603-bf16.gguf"), None),
        (_names("Voxtral-Mini-4B-Realtime-2602-GGUF"), "voxtral_realtime"),
        (
            _names("Voxtral-Mini-4B-Realtime-2602-GGUF", "voxtral-mini-4b-realtime-2602-q4_k.gguf"),
            "voxtral_realtime",
        ),
        (_names("KittenTTS-GGUF"), "kitten_tts"),
        (("dignome/kitten_tts2", "kitten-tts2", "kitten-tts2-native-q8-multilingual.gguf"), None),
        # A version dot is dropped before matching: IndexTTS2.5 is still index_tts2.
        (_names("IndexTTS2.5-GGUF"), "index_tts2"),
        (_names("IndexTTS2.5-GGUF", "index-tts2_5-q8_0.gguf"), "index_tts2"),
    ],
)
def test_name_hints(names, family):
    assert acm.family_from_names(names) == family


OWSM = f"{AUDIO_CPP_REPO}/OWSM-GGUF"
HVISKE_V6 = f"{AUDIO_CPP_REPO}/Hviske-v6-GGUF"


def _asr_spec(*options):
    return {
        "schema_version": 1,
        "tasks": ["asr"],
        "options": {"request": [{"name": n, "type": "string"} for n in options]},
    }


@pytest.fixture
def v092_asr(fake, hub, monkeypatch):
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: True)
    snap = _snapshot(hub)
    _put(
        snap,
        "OWSM-GGUF/owsm-v4-base-102m-q8_0.gguf",
        _gguf_bytes(family = "owsm", spec = _asr_spec("language", "max_tokens")),
    )
    _put(
        snap,
        "Hviske-v6-GGUF/hviske-v6-q8_0.gguf",
        _gguf_bytes(family = "hviske_asr_v6", spec = _asr_spec("max_tokens", "num_beams")),
    )
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    return fake


@pytest.mark.parametrize("language", ["fr", "da-DK", None])
def test_owsm_detects_the_language_rather_than_a_two_letter_code(v092_asr, side, source, language):
    result = side.transcribe_path(source, OWSM, language)
    (body,) = v092_asr.bodies
    assert body["language"] == "auto"
    # The language the model reports hearing (the fake server answers English), not the ignored choice.
    assert result["language"] == "English"


@pytest.mark.parametrize("family", ["owsm", "owsm_ctc"])
def test_owsm_families_always_read_auto(family):
    assert stt._runtime_language(family, "fr") == "auto"
    assert stt._runtime_language(family, None) == "auto"


def test_a_spec_without_a_language_option_gets_none(v092_asr, side, source):
    assert acm.resolve(HVISKE_V6, network = False).request_keys == {"max_tokens", "num_beams"}
    side.transcribe_path(source, HVISKE_V6, "da")
    (body,) = v092_asr.bodies
    assert body.get("language", ABSENT) is ABSENT
