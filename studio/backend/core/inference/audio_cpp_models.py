# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated audio.cpp models: one table from a Studio id to the files audio.cpp loads.

audio.cpp publishes its native GGUF packages as subfolders of one Hub repo, so a
Studio id here is that repo plus the subfolder (``audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF``).
Three path segments cannot collide with a real ``owner/name`` repo id, which is what
lets the load path tell these apart without a Hub request. Everything that needs the
real repo, the files, or the family goes through ``lookup``; nothing else parses the id.

Speech-to-text models are also reachable by a short key (``audiocpp-qwen3-asr-0.6b``),
which is what the dictation settings store, the way whisper.cpp and mtmd use keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

AUDIO_CPP_REPO = "audio-cpp/audio.cpp-gguf"
# The repo revision the catalog was tested against with the pinned runtime. The repo updates packages
# in place, so Studio's own downloads fetch this commit; bump it together with the runtime tag.
AUDIO_CPP_REVISION = "83c5d96c03023ff5a7712570d057ce26c8769f98"

# The audio_type Studio records for a model served by audio.cpp, per task.
AUDIO_CPP_TTS_AUDIO_TYPE = "audiocpp_tts"
AUDIO_CPP_MUSIC_AUDIO_TYPE = "audiocpp_music"
AUDIO_CPP_AUDIO_TYPES = frozenset((AUDIO_CPP_TTS_AUDIO_TYPE, AUDIO_CPP_MUSIC_AUDIO_TYPE))


@dataclass(frozen = True)
class AudioCppModel:
    # Studio id: ``<AUDIO_CPP_REPO>/<folder>``.
    id: str
    # Short key, used by dictation settings for ASR models.
    key: str
    display_name: str
    # audio.cpp ``--family``.
    family: str
    # Studio task: ``tts`` (speech), ``music`` (generation) or ``asr``.
    task: str
    # Repo-relative files to download. The first is the GGUF the server loads.
    files: tuple[str, ...]
    size_bytes: int
    # Values injected into the server config's model entry (load/session options, voice presets).
    model_options: dict = field(default_factory = dict, hash = False, compare = False)
    # Default request fields for speech (e.g. a built-in ``voice``). An ``options`` dict here is merged
    # under the request's own options.
    request_defaults: dict = field(default_factory = dict, hash = False, compare = False)
    # audiocpp_server task when it is not the Studio task's default (a voice-design-only package runs "vdes").
    task_override: Optional[str] = None
    # Phonemizes with eSpeak-ng, so it needs a bundle built with AUDIOCPP_STATIC_ESPEAK.
    needs_espeak: bool = False
    # Fewest files each glob in ``files`` must match before the model counts as downloaded.
    min_glob_matches: int = 1

    @property
    def repo_id(self) -> str:
        return AUDIO_CPP_REPO

    @property
    def gguf_file(self) -> str:
        return self.files[0]

    @property
    def server_task(self) -> str:
        """The task name the audiocpp_server config expects."""
        if self.task_override:
            return self.task_override
        return "gen" if self.task == "music" else self.task

    @property
    def audio_type(self) -> Optional[str]:
        if self.task == "tts":
            return AUDIO_CPP_TTS_AUDIO_TYPE
        if self.task == "music":
            return AUDIO_CPP_MUSIC_AUDIO_TYPE
        return None


def _m(
    folder: str, key: str, name: str, family: str, task: str, files, size_mb: float, **kw
) -> AudioCppModel:
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{folder}",
        key = key,
        display_name = name,
        family = family,
        task = task,
        files = tuple(f"{folder.split('/')[0]}/{f}" for f in files),
        size_bytes = int(size_mb * 1024 * 1024),
        **kw,
    )


_MODELS: tuple[AudioCppModel, ...] = (
    # Text to speech. Every entry speaks without a reference clip; cloning is a later step.
    # Kokoro, KittenTTS, Piper and Inflect phonemize with eSpeak-ng, which the Unsloth bundles link
    # statically (GPL-3.0-or-later) with its data file beside the server.
    _m(
        "Kokoro-82M-GGUF",
        "audiocpp-kokoro-82m",
        "Kokoro 82M",
        "kokoro_tts",
        "tts",
        ["kokoro-82m-q8_0.gguf"],
        180.8,
        needs_espeak = True,
    ),
    _m(
        "KittenTTS-GGUF",
        "audiocpp-kitten-tts-mini",
        "KittenTTS Mini 0.8",
        "kitten_tts",
        "tts",
        ["kitten-tts-mini-0.8-orig.gguf"],
        288.2,
        needs_espeak = True,
    ),
    _m(
        "Piper-TTS-GGUF",
        "audiocpp-piper-en-us-lessac",
        "Piper (en-US Lessac)",
        "piper_tts",
        "tts",
        ["piper-en-us-lessac-medium-orig.gguf"],
        59.8,
        needs_espeak = True,
    ),
    _m(
        "Inflect-Micro-v2-GGUF",
        "audiocpp-inflect-micro-v2",
        "Inflect Micro v2",
        "inflect_v2",
        "tts",
        ["inflect-micro-v2-orig.gguf"],
        68.7,
        needs_espeak = True,
    ),
    _m(
        "PocketTTS-GGUF/english",
        "audiocpp-pocket-tts-en",
        "PocketTTS (English)",
        "pocket_tts",
        "tts",
        ["english/pocket-tts-english-q8_0.gguf", "english/embeddings/*"],
        280.0,
        request_defaults = {"voice": "alba"},
        min_glob_matches = 26,
    ),
    _m(
        "MOSS-TTS-Nano-100M-GGUF",
        "audiocpp-moss-tts-nano",
        "MOSS-TTS Nano 100M",
        "moss_tts_nano",
        "tts",
        ["moss-tts-nano-100m-q8_0.gguf"],
        184.4,
    ),
    _m(
        "Supertonic-3-GGUF",
        "audiocpp-supertonic-3",
        "Supertonic 3",
        "supertonic",
        "tts",
        ["supertonic-3-f16.gguf"],
        298.3,
        request_defaults = {"voice": "F1"},
    ),
    _m(
        "Chatterbox-Turbo-GGUF",
        "audiocpp-chatterbox-turbo",
        "Chatterbox Turbo",
        "chatterbox_turbo",
        "tts",
        ["chatterbox-turbo-q8_0.gguf"],
        666.7,
    ),
    _m(
        "VoxCPM2-GGUF",
        "audiocpp-voxcpm2",
        "VoxCPM2 2B",
        "voxcpm2",
        "tts",
        ["voxcpm2-q8_0.gguf"],
        2818.1,
    ),
    _m(
        "Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF",
        "audiocpp-qwen3-tts-1.7b-customvoice",
        "Qwen3-TTS 1.7B CustomVoice",
        "qwen3_tts",
        "tts",
        ["qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"],
        2686.5,
        # A speaker is mandatory; the package's spk_id table names vivian, serena, ryan, aiden, ono_anna, sohee,
        # uncle_fu, eric and dylan.
        request_defaults = {"voice": "vivian"},
    ),
    _m(
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF",
        "audiocpp-qwen3-tts-1.7b-voicedesign",
        "Qwen3-TTS 1.7B VoiceDesign",
        "qwen3_tts",
        "tts",
        ["qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf"],
        2686.5,
        # The package accepts only the voice-design task, and designs every voice from an instruction.
        task_override = "vdes",
        request_defaults = {
            "options": {
                "instruct": "A warm, clear, natural adult voice speaking at a relaxed pace."
            }
        },
    ),
    # Music generation.
    _m(
        "ACE-Step1.5-GGUF/turbo",
        "audiocpp-ace-step-1.5-turbo",
        "ACE-Step 1.5 Turbo",
        "ace_step",
        "music",
        ["turbo/ace-step-1.5-turbo-q8_0.gguf"],
        5898.9,
    ),
    _m(
        "ACE-Step1.5-GGUF/base",
        "audiocpp-ace-step-1.5-base",
        "ACE-Step 1.5 Base",
        "ace_step",
        "music",
        ["base/ace-step-1.5-base-q8_0.gguf"],
        5898.9,
    ),
    _m(
        "Stable-Audio-3-Small-Music-GGUF",
        "audiocpp-stable-audio-3-small-music",
        "Stable Audio 3 Small (Music)",
        "stable_audio",
        "music",
        ["stable-audio-3-small-music-q8_0.gguf"],
        1605.6,
    ),
    # Speech to text.
    _m(
        "Qwen3-ASR-0.6B-GGUF",
        "audiocpp-qwen3-asr-0.6b",
        "Qwen3-ASR 0.6B",
        "qwen3_asr",
        "asr",
        ["qwen3-asr-0.6b-q8_0.gguf"],
        1097.9,
    ),
    _m(
        "Qwen3-ASR-1.7B-GGUF",
        "audiocpp-qwen3-asr-1.7b",
        "Qwen3-ASR 1.7B",
        "qwen3_asr",
        "asr",
        ["qwen3-asr-1.7b-q8_0.gguf"],
        2358.4,
    ),
    _m(
        "Parakeet-TDT-0.6B-v3-GGUF",
        "audiocpp-parakeet-tdt-0.6b-v3",
        "Parakeet TDT 0.6B v3",
        "parakeet_tdt",
        "asr",
        ["parakeet-tdt-0.6b-v3-q8_0.gguf"],
        873.3,
    ),
    _m(
        "Canary-180M-Flash-GGUF",
        "audiocpp-canary-180m-flash",
        "Canary 180M Flash",
        "canary_asr",
        "asr",
        ["canary-180m-flash-q8_0.gguf"],
        237.9,
    ),
    _m(
        "Moonshine-Streaming-GGUF/tiny",
        "audiocpp-moonshine-tiny",
        "Moonshine Tiny",
        "moonshine_asr",
        "asr",
        ["moonshine-streaming-tiny-q8_0.gguf"],
        57.6,
    ),
    _m(
        "Moonshine-Streaming-GGUF/small",
        "audiocpp-moonshine-small",
        "Moonshine Small",
        "moonshine_asr",
        "asr",
        ["moonshine-streaming-small-q8_0.gguf"],
        286.7,
    ),
    _m(
        "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
        "audiocpp-nemotron-3.5-asr-0.6b",
        "Nemotron 3.5 ASR 0.6B",
        "nemotron_asr",
        "asr",
        ["nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"],
        887.5,
    ),
)

_BY_ID = {m.id.lower(): m for m in _MODELS}
_BY_KEY = {m.key: m for m in _MODELS}

DEFAULT_AUDIO_CPP_STT_MODEL = "audiocpp-qwen3-asr-0.6b"


def all_models() -> tuple[AudioCppModel, ...]:
    return _MODELS


def lookup(identifier: Optional[str]) -> Optional[AudioCppModel]:
    """The curated model for a Studio id or short key, else None. Never touches the network."""
    if not identifier:
        return None
    text = str(identifier).strip()
    return _BY_KEY.get(text) or _BY_ID.get(text.lower().rstrip("/"))


def is_audio_cpp_model(identifier: Optional[str]) -> bool:
    return lookup(identifier) is not None


def is_audio_cpp_generation_model(identifier: Optional[str]) -> bool:
    """A TTS or music model, i.e. one that loads into the main audio slot."""
    model = lookup(identifier)
    return model is not None and model.task in ("tts", "music")


def stt_models() -> tuple[AudioCppModel, ...]:
    return tuple(m for m in _MODELS if m.task == "asr")
