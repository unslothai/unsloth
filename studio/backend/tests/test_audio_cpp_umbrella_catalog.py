# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every folder of the audio.cpp-gguf repo lands on the Audio pages its family runs on, and the
Audio pickers offer exactly the folders Studio runs under the pinned runtime."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_spec_families import AUDIO_CPP_SPEC_FAMILIES

# Folder -> (family its GGUF header names, Audio pages), audio-cpp/audio.cpp-gguf at e36610ac.
UMBRELLA = {
    "ACE-Step1.5-GGUF": ("ace_step", ["music"]),
    "Apollo-GGUF": ("apollo", []),
    "AudioSR-GGUF": ("audiosr", []),
    "BS-RoFormer-ep368-GGUF": ("bs_roformer", ["separate"]),
    "Breeze-TTS-2-GGUF": ("breeze_tts", ["speak", "clone"]),
    "Canary-180M-Flash-GGUF": ("canary_asr", ["transcribe"]),
    "Chatterbox-GGUF": ("chatterbox", ["clone", "convert"]),
    "Chatterbox-Turbo-GGUF": ("chatterbox_turbo", ["speak"]),
    "Citrinet-ASR-GGUF": ("citrinet_asr", ["transcribe"]),
    "Cohere-Transcribe-GGUF": ("cohere_asr", ["transcribe"]),
    "Confucius4-TTS-GGUF": ("confucius4_tts", ["clone"]),
    "ControlFoley-GGUF": ("controlfoley", ["music"]),
    "CosyVoice3-GGUF": ("cosyvoice3", ["clone"]),
    "CrisperWhisper2.0-GGUF": ("crisperwhisper", []),
    "DotTTS-Edit-GGUF": ("dots_tts", ["speak", "clone", "edit"]),
    "DotTTS-MF-GGUF": ("dots_tts", ["speak", "clone"]),
    "DotTTS-SOAR-GGUF": ("dots_tts", ["speak", "clone"]),
    "DramaBox-GGUF": ("dramabox", ["speak"]),
    "FireRedAudio-GGUF": ("firered_audio", ["clone", "edit"]),
    "FireRedTTS3-Base-GGUF": ("fireredtts3", ["clone"]),
    "FireRedTTS3-Instruct-GGUF": ("fireredtts3", ["clone"]),
    "Fish-Audio-S2-Pro-GGUF": ("fish_audio", ["speak", "clone"]),
    "Fun-ASR-Nano-2512-GGUF": ("fun_asr_nano", ["transcribe"]),
    "GigaAM-ASR-GGUF": ("gigaam_asr", ["transcribe"]),
    "Granite-Speech-5.0-470M-TurboCTC-GGUF": ("granite5asr", ["transcribe"]),
    "HTDemucs-6stems-GGUF": ("htdemucs_6stems", ["separate"]),
    "HTDemucs-GGUF": ("htdemucs", ["separate"]),
    "HeartMuLa-GGUF": ("heartmula", ["music"]),
    "Higgs-Audio-v3-STT-GGUF": ("higgs_audio_stt", ["transcribe"]),
    "Higgs-Audio-v3-TTS-4B-GGUF": ("higgs_audio_tts", ["speak", "clone"]),
    "Hviske-v5.3-GGUF": ("hviske_asr", ["transcribe"]),
    "IndexTTS2-GGUF": ("index_tts2", ["clone"]),
    "IndexTTS2.5-GGUF": ("index_tts2", ["clone"]),
    "Inflect-Micro-v2-GGUF": ("inflect_v2", ["speak"]),
    "Irodori-TTS-500M-v3-GGUF": ("irodori_tts", ["speak", "clone"]),
    "Irodori-TTS-600M-v3-VoiceDesign-GGUF": ("irodori_tts", ["speak", "clone"]),
    "Irodori-TTS-v4-Small-GGUF": ("irodori_tts", ["speak", "clone"]),
    "KittenTTS-GGUF": ("kitten_tts", ["speak"]),
    "Kokoro-82M-GGUF": ("kokoro_tts", ["speak"]),
    "Kroko-ASR-GGUF": ("kroko_asr", ["transcribe"]),
    "KugelAudio-0-Open-GGUF": ("kugelaudio", []),
    "MMS-Forced-Aligner-GGUF": ("mms_forced_aligner", []),
    "MOSS-TTS-Local-v1.5-GGUF": ("moss_tts_local", ["speak", "clone"]),
    "MOSS-TTS-Nano-100M-GGUF": ("moss_tts_nano", ["speak", "clone"]),
    "MOSS-Transcribe-Diarize-GGUF": ("moss_transcribe_diarize", ["transcribe"]),
    "MOSS-VoiceGenerator-GGUF": ("moss_voicegen", ["speak"]),
    "MagpieTTS-Multilingual-357M-GGUF": ("magpie_tts", ["speak"]),
    "Maya1-GGUF": ("maya1", ["speak"]),
    "MeanVC2-GGUF": ("meanvc2", ["convert"]),
    "Mel-Band-RoFormer-GGUF": ("mel_band_roformer", ["separate"]),
    "MiDashengLM-Gen-GGUF": ("midashenglm_gen", ["music"]),
    "MiniMax-H3-Q4-GGUF": ("minimax_h3", []),
    "MioCodec-25Hz-44.1kHz-v2-GGUF": ("miocodec", []),
    "MioTTS-1.7B-GGUF": ("miotts", ["clone"]),
    "Moonshine-Streaming-GGUF": ("moonshine_asr", ["transcribe"]),
    "MuScriptor-Small-GGUF": ("muscriptor", []),
    "Nemotron-3.5-ASR-Streaming-0.6B-GGUF": ("nemotron_asr", ["transcribe"]),
    "NeuTTS-2E-GGUF": ("neutts", ["speak"]),
    "Niagara-ASR-GGUF": ("niagara_asr", ["transcribe"]),
    "OWSM-CTC-GGUF": ("owsm_ctc", []),
    "OWSM-GGUF": ("owsm", []),
    "OmniVoice-GGUF": ("omnivoice", ["speak", "clone"]),
    "Parakeet-TDT-0.6B-v3-GGUF": ("parakeet_tdt", ["transcribe"]),
    "PersonaPlex-GGUF": ("personaplex", []),
    "Piper-TTS-GGUF": ("piper_tts", ["speak"]),
    "PocketTTS-GGUF": ("pocket_tts", ["speak", "clone"]),
    "PulseVAD-GGUF": ("pulsevad", []),
    "Qwen3-ASR-0.6B-GGUF": ("qwen3_asr", ["transcribe"]),
    "Qwen3-ASR-1.7B-GGUF": ("qwen3_asr", ["transcribe"]),
    "Qwen3-ForcedAligner-0.6B-GGUF": ("qwen3_forced_aligner", []),
    "Qwen3-TTS-12Hz-0.6B-Base-GGUF": ("qwen3_tts", ["clone"]),
    "Qwen3-TTS-12Hz-1.7B-Base-GGUF": ("qwen3_tts", ["clone"]),
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF": ("qwen3_tts", ["speak"]),
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF": ("qwen3_tts", ["speak"]),
    "RVC-GGUF": ("rvc", ["convert"]),
    "Samsone-GGUF": ("samsone", []),
    "SeedVC-MLX-GGUF": ("seed_vc", ["convert"]),
    "Sidon-GGUF": ("sidon", []),
    "Smart-Turn-v3-GGUF": ("smart_turn", []),
    "Sortformer-Diar-4spk-v1-GGUF": ("sortformer_diar", []),
    "Stable-Audio-3-Medium-GGUF": ("stable_audio", ["music"]),
    "Stable-Audio-3-Small-Music-GGUF": ("stable_audio", ["music"]),
    "Stable-Audio-3-Small-SFX-GGUF": ("stable_audio", ["music"]),
    "Supertonic-3-GGUF": ("supertonic", ["speak"]),
    "Tone-Color-VC-GGUF": ("tone_color_vc", ["convert"]),
    "UniverSR-GGUF": ("universr", []),
    "Vevo2-GGUF": ("vevo2", ["clone", "edit", "convert"]),
    "VibeVoice-1.5B-GGUF": ("vibevoice", ["speak"]),
    "VibeVoice-ASR-GGUF": ("vibevoice_asr", ["transcribe"]),
    "VoxCPM1-GGUF": ("voxcpm1", ["speak", "clone"]),
    "VoxCPM2-GGUF": ("voxcpm2", ["speak", "clone"]),
    "Voxtral-Mini-4B-Realtime-2602-GGUF": ("voxtral_realtime", ["transcribe"]),
}

# Families outside acm.FAMILIES are classified by the tasks their spec lists.
SPEC_TASKS = {
    "gigaam_asr": ["asr"],
    "maya1": ["tts"],
    "crisperwhisper": ["asr", "align"],
    "kugelaudio": ["tts"],
    "owsm": ["asr"],
    "owsm_ctc": ["asr"],
    "sidon": ["s2s"],
    "smart_turn": ["turn"],
}

TASK_PAGE = {"tts": "speak", "music": "music", "asr": "transcribe", "sep": "separate"}

CATALOG = Path(__file__).resolve().parents[2] / "frontend/src/features/audio/audio-cpp-catalog.ts"


@pytest.fixture(autouse = True)
def pinned_runtime(monkeypatch):
    monkeypatch.setattr(
        acm, "runtime_knows_family", lambda family: family in AUDIO_CPP_SPEC_FAMILIES
    )


@pytest.mark.parametrize("folder", sorted(UMBRELLA))
def test_each_folder_lands_on_its_pages(folder):
    family, pages = UMBRELLA[folder]
    spec = {"tasks": SPEC_TASKS[family]} if family in SPEC_TASKS else None
    policy = acm.family_policy(family, spec, (f"{acm.AUDIO_CPP_REPO}/{folder}", folder), folder)
    assert list(policy.workflows) == pages
    # A folder with no page is refused, should it be loaded by id.
    assert (policy.unsupported is None) == bool(pages)


def _catalog() -> tuple[dict[str, list[str]], set[str]]:
    source = CATALOG.read_text(encoding = "utf-8")
    seeded = {}
    for folder, task, workflows in re.findall(
        r'id: folder\("([^"]+)"\),\s*task: "(\w+)"(?:,\s*workflows: \[([^\]]*)\])?', source
    ):
        seeded[folder] = re.findall(r'"(\w+)"', workflows) or [TASK_PAGE[task]]
    block = re.search(r"AUDIO_CPP_UNOFFERED_FOLDERS[^=]*=\s*\{(.*?)\};", source, re.S).group(1)
    return seeded, set(re.findall(r'^\s*"([^"]+)":', block, re.M))


def test_the_pickers_offer_every_runnable_folder_on_its_pages():
    seeded, unoffered = _catalog()
    assert seeded == {folder: pages for folder, (_, pages) in UMBRELLA.items() if pages}
    assert unoffered == {folder for folder, (_, pages) in UMBRELLA.items() if not pages}


@pytest.mark.parametrize(
    "folder", sorted(f for f, (_, pages) in UMBRELLA.items() if "transcribe" in pages)
)
def test_dictation_routes_every_transcribe_folder_to_audiocpp(folder):
    """Settings > Voice lists every Transcribe folder; dictation must send each to the audio runtime."""
    from core.inference.stt_audiocpp_sidecar import resolve_audio_cpp_stt_model_id
    from routes.inference import _stt_engine_for_model

    row = f"{acm.AUDIO_CPP_REPO}/{folder}"
    assert _stt_engine_for_model(row) == "audiocpp"
    assert resolve_audio_cpp_stt_model_id(row) == row


def test_the_settings_dictation_keys_are_the_backend_asr_keys():
    keys = dict(
        re.findall(r'dictation\(\s*"([^"]+)",\s*"([^"]+)"', CATALOG.read_text(encoding = "utf-8"))
    )
    asr = {
        key: folder
        for key, (folder, _) in acm._LEGACY_KEYS.items()
        if "transcribe" in UMBRELLA[folder][1]
    }
    assert keys == asr
    assert acm.parse_identifier("audiocpp-moonshine-medium").variant_hint == "medium/Q8_0"
