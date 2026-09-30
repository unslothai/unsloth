# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recognise a GGUF that audio.cpp runs, and work out what Studio needs to run it.

Every GGUF audio.cpp writes declares ``general.architecture = "audiocpp"`` and, in
most cases, names its loader in ``audiocpp.model_spec.family`` and embeds the full
model spec in ``audiocpp.model_spec.json``. The server has no auto-detection, so
Studio reads the family itself and hands it over together with a runtime task token.

A model is any Hub repo (or local file) holding such GGUFs. The ``audio-cpp/audio.cpp-gguf``
repo publishes many models side by side, one per top-level folder, so a model there is
named by three segments (``audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF``); three segments
cannot collide with a real ``owner/name`` repo id. Variants are the quants in that repo
or folder, like any llama.cpp GGUF repo. A few families (MiniMax Music 3, YuE2) are
packages of several GGUFs plus config files that load as one directory; their variants
are the component mixes the package defines.

Everything that needs the repo, the files, the family or the variant goes through
``resolve``; nothing else parses the id.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

AUDIO_CPP_REPO = "audio-cpp/audio.cpp-gguf"
AUDIO_CPP_ARCHITECTURE = "audiocpp"
_UMBRELLA_PREFIX = AUDIO_CPP_REPO.lower() + "/"

# The audio_type Studio records for a model served by audio.cpp, per task.
AUDIO_CPP_TTS_AUDIO_TYPE = "audiocpp_tts"
AUDIO_CPP_MUSIC_AUDIO_TYPE = "audiocpp_music"
AUDIO_CPP_AUDIO_TYPES = frozenset((AUDIO_CPP_TTS_AUDIO_TYPE, AUDIO_CPP_MUSIC_AUDIO_TYPE))

# Studio task -> the Hub pipeline task its rows carry.
HUB_TASKS = {
    "tts": "text-to-speech",
    "music": "text-to-audio",
    "asr": "automatic-speech-recognition",
}

DEFAULT_AUDIO_CPP_STT_MODEL = f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF"


# ---------------------------------------------------------------------------
# Families


@dataclass(frozen = True)
class AudioCppPackageVariant:
    """One component mix of a package family: its files and the session options that pick them."""

    key: str
    files: tuple[str, ...]
    session_options: dict = field(default_factory = dict, hash = False, compare = False)


@dataclass(frozen = True)
class AudioCppFamily:
    family: str
    # Studio task: ``tts`` (speech), ``music`` (generation) or ``asr``; empty when Studio has no feature for it.
    task: str
    # audiocpp_server task when it is not the Studio task's default (a voice-design package runs "vdes").
    server_task: Optional[str] = None
    # Phonemizes with eSpeak-ng, so it needs a bundle built with AUDIOCPP_STATIC_ESPEAK.
    needs_espeak: bool = False
    # Default request fields for speech (e.g. a built-in ``voice``). An ``options`` dict here is merged
    # under the request's own options.
    request_defaults: dict = field(default_factory = dict, hash = False, compare = False)
    # Values injected into the server config's model entry.
    model_options: dict = field(default_factory = dict, hash = False, compare = False)
    # Package families load a directory of several GGUFs; their variants are the package's mixes.
    package: tuple[AudioCppPackageVariant, ...] = ()
    # Request options Studio shows when neither the runtime nor the GGUF offers a current spec.
    options: tuple[dict, ...] = ()
    # Why Studio refuses this family, when it does.
    unsupported: Optional[str] = None

    @property
    def default_server_task(self) -> str:
        if self.server_task:
            return self.server_task
        return "gen" if self.task == "music" else self.task


def _minimax_package(key: str, lm: str, rvq: str, flow: str) -> AudioCppPackageVariant:
    files = (
        "config.json",
        "config/language_model.json",
        "config/rvq_depth_decoder.json",
        "config/condition_encoder.json",
        "config/transformer.json",
        "config/vocoder.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        f"language_model_{lm}.gguf",
        f"rvq_depth_decoder_{rvq}.gguf",
        "condition_encoder.gguf",
        f"transformer_{flow}.gguf",
        "vocoder.gguf",
    )
    return AudioCppPackageVariant(
        key,
        files,
        {
            "minimax_music3.language_model_gguf": f"language_model_{lm}.gguf",
            "minimax_music3.rvq_depth_decoder_gguf": f"rvq_depth_decoder_{rvq}.gguf",
            "minimax_music3.flow_transformer_gguf": f"transformer_{flow}.gguf",
        },
    )


def _yue2_package(key: str, quant: str) -> AudioCppPackageVariant:
    main = f"yue2-3b-{quant}.gguf"
    return AudioCppPackageVariant(
        key,
        (
            "sidecars/yue2-model-config.json",
            "sidecars/yue2-generation-config.json",
            "sidecars/yue2-qwen.tiktoken",
            "sidecars/yue2-vae-config.json",
            main,
            "yue2-vae-f16.gguf",
        ),
        {"yue2.model_gguf": main, "yue2.vae_gguf": "yue2-vae-f16.gguf"},
    )


def _opt(name: str, type_: str, description: str, **kw) -> dict:
    return {"name": name, "type": type_, "description": description, **kw}


_FAMILY_LIST: tuple[AudioCppFamily, ...] = (
    # Text to speech. Kokoro, KittenTTS, Piper and Inflect phonemize with eSpeak-ng, which the Unsloth
    # bundles link statically (GPL-3.0-or-later) with its data file beside the server.
    AudioCppFamily("kokoro_tts", "tts", needs_espeak = True),
    AudioCppFamily("kitten_tts", "tts", needs_espeak = True),
    AudioCppFamily("piper_tts", "tts", needs_espeak = True),
    AudioCppFamily("inflect_v2", "tts", needs_espeak = True),
    AudioCppFamily("pocket_tts", "tts", request_defaults = {"voice": "alba"}),
    AudioCppFamily("supertonic", "tts", request_defaults = {"voice": "F1"}),
    AudioCppFamily("qwen3_tts", "tts"),
    *(
        AudioCppFamily(name, "tts")
        for name in (
            "moss_tts_nano",
            "moss_tts_local",
            "moss_tts_v15",
            "moss_ttsd",
            "chatterbox",
            "chatterbox_turbo",
            "voxcpm1",
            "voxcpm2",
            "neutts",
            "magpie_tts",
            "miotts",
            "cosyvoice3",
            "fish_audio",
            "higgs_audio_tts",
            "index_tts2",
            "irodori_tts",
            "breeze_tts",
            "dots_tts",
            "dramabox",
            "fireredtts3",
            "firered_audio",
            "omnivoice",
            "confucius4_tts",
            "vibevoice",
            "vevo2",
            "echo_tts",
            "f5_tts",
            "glm_tts",
            "outetts",
            "sanotts",
            "soprano_tts",
            "zipvoice",
            "vieneu_v3_turbo",
            "audio8_tts",
        )
    ),
    # Voice design only: every voice is designed from the instruction.
    AudioCppFamily(
        "moss_voicegen",
        "tts",
        server_task = "vdes",
        request_defaults = {
            "options": {
                "instruct": "A warm, clear, natural adult voice speaking at a relaxed pace."
            }
        },
    ),
    # Music generation.
    *(
        AudioCppFamily(name, "music")
        for name in ("ace_step", "stable_audio", "heartmula", "midashenglm_gen", "controlfoley")
    ),
    AudioCppFamily(
        "minimax_music3",
        "music",
        package = (
            _minimax_package("Q4_0", "q4_0", "q8_0", "q4_0"),
            _minimax_package("Q8_0", "q8_0", "q8_0", "q8_0"),
            _minimax_package("BF16", "bf16", "bf16", "bf16"),
        ),
        # The runtime's current spec; the one embedded in the published GGUFs predates it.
        options = (
            _opt(
                "num_inference_steps",
                "int",
                "Flow matching Euler steps per chunk.",
                min = 1,
                default = 30,
            ),
            _opt(
                "guidance_scale",
                "float",
                "Flow transformer classifier-free guidance scale.",
                min = 0.0,
                default = 1.7,
            ),
            _opt(
                "ar_guidance_scale",
                "float",
                "Autoregressive semantic and depth CFG scale.",
                min = 0.0,
                default = 1.5,
            ),
            _opt(
                "top_k",
                "int",
                "Top-k sampling for semantic and residual code sampling.",
                min = 1,
                default = 50,
            ),
        ),
    ),
    AudioCppFamily(
        "yue2",
        "music",
        package = (
            _yue2_package("Q8_0", "q8_0"),
            _yue2_package("BF16", "bf16"),
            _yue2_package("Q4_0", "q4_0"),
        ),
        options = (
            _opt(
                "cot",
                "enum",
                "Symbolic planning route. off skips ABC generation; melody/full generate ABC before music tokens.",
                values = ["off", "melody", "full"],
                default = "full",
            ),
            _opt("num_inference_steps", "int", "NAR midpoint ODE steps.", min = 1, default = 8),
            _opt(
                "guidance_scale",
                "float",
                "Semantic classifier-free guidance scale.",
                min = 0.0,
                max = 20.0,
            ),
            _opt(
                "semantic_temperature",
                "float",
                "Semantic codec sampling temperature.",
                min = 0.0,
                max = 5.0,
            ),
            _opt(
                "semantic_top_p",
                "float",
                "Semantic codec nucleus sampling probability.",
                min = 0.0,
                max = 1.0,
            ),
            _opt("semantic_top_k", "int", "Semantic codec top-k sampling limit.", min = 1),
        ),
    ),
    # Speech to text.
    *(
        AudioCppFamily(name, "asr")
        for name in (
            "qwen3_asr",
            "parakeet_tdt",
            "canary_asr",
            "citrinet_asr",
            "cohere_asr",
            "higgs_audio_stt",
            "granite5asr",
            "kroko_asr",
            "moonshine_asr",
            "moss_transcribe_diarize",
            "nemotron_asr",
            "niagara_asr",
            "voxtral_realtime",
            "vibevoice_asr",
            "vibevoice_asr_streaming",
            "hviske_asr",
            "fun_asr_nano",
            "sense_asr",
            "confucius4_r2t2",
            "audio8_asr",
        )
    ),
    # Package families Studio does not lay out yet.
    *(
        AudioCppFamily(
            name,
            "",
            unsupported = "This model ships as a multi-file package Studio cannot load yet.",
        )
        for name in ("minimax_h3", "auk", "liveavatar", "vibeasr")
    ),
    # Tasks Studio has no page for.
    *(
        AudioCppFamily(name, "", unsupported = f"{what} models are not supported in Studio yet.")
        for names, what in (
            (("qwen3_forced_aligner", "mms_forced_aligner"), "Forced-alignment"),
            (("sortformer_diar", "sortformer_diar_v2", "nemotron_3_diar"), "Speaker diarization"),
            (("htdemucs", "bs_roformer", "mel_band_roformer"), "Source separation"),
            (("pulsevad", "silero_vad", "marblenet_vad"), "Voice activity detection"),
            (("muscriptor", "sheetsage2"), "Music transcription"),
            (("meanvc2", "rvc", "seed_vc", "miocodec"), "Voice conversion"),
            (("apollo", "audiosr", "universr", "personaplex"), "Speech-to-speech"),
        )
        for name in names
    ),
)

FAMILIES: dict[str, AudioCppFamily] = {f.family: f for f in _FAMILY_LIST}

# Spec task names that are not runtime task tokens.
_SPEC_TO_SERVER_TASK = {
    "music": "gen",
    "sfx": "gen",
    "edit": "gen",
    "clone": "clon",
    "design": "vdes",
    "speaker": "spk",
}
_SERVER_TO_STUDIO_TASK = {"tts": "tts", "gen": "music", "asr": "asr"}
_TASK_NAMES = {
    "align": "Forced-alignment",
    "diar": "Speaker diarization",
    "sep": "Source separation",
    "vad": "Voice activity detection",
    "midi": "Music transcription",
    "vc": "Voice conversion",
    "svc": "Voice conversion",
    "s2s": "Speech-to-speech",
    "spk": "Speaker embedding",
}


def _family_from_spec_tasks(family: str, spec: Optional[dict]) -> AudioCppFamily:
    """Policy for a family Studio does not list, from the tasks its spec declares."""
    tasks = [str(t).lower() for t in (spec or {}).get("tasks") or [] if isinstance(t, str)]
    tokens = [_SPEC_TO_SERVER_TASK.get(t, t) for t in tasks]
    # Speech first, then music, then transcription: a model that speaks is most useful spoken.
    for token in ("tts", "vdes", "clon", "gen", "asr"):
        if token in tokens:
            if token == "clon":
                break
            studio = "tts" if token == "vdes" else _SERVER_TO_STUDIO_TASK[token]
            return AudioCppFamily(family, studio, server_task = token)
    if "clon" in tokens:
        return AudioCppFamily(
            family,
            "",
            unsupported = "This model only clones a reference voice, which Studio does not send yet.",
        )
    if tokens:
        what = _TASK_NAMES.get(tokens[0], tokens[0])
        return AudioCppFamily(
            family, "", unsupported = f"{what} models are not supported in Studio yet."
        )
    return AudioCppFamily(
        family,
        "",
        unsupported = f"Studio does not know what the audio.cpp family '{family}' does.",
    )


def family_policy(
    family: str,
    spec: Optional[dict] = None,
    names: Iterable[str] = (),
) -> AudioCppFamily:
    """What Studio does with ``family``: the table entry, else what its spec says it does.

    ``names`` (repo, folder and file names) pick the Qwen3-TTS package kind, which the family
    alone does not say: CustomVoice needs a speaker, VoiceDesign designs one, Base only clones.
    """
    policy = FAMILIES.get(family) or _family_from_spec_tasks(family, spec)
    if family == "qwen3_tts":
        text = " ".join(names).lower()
        if "voicedesign" in text or "voice-design" in text:
            return AudioCppFamily(
                family,
                "tts",
                server_task = "vdes",
                request_defaults = {
                    "options": {
                        "instruct": "A warm, clear, natural adult voice speaking at a relaxed pace."
                    }
                },
            )
        if "customvoice" in text or "custom-voice" in text:
            # A speaker is mandatory; the package's spk_id table names vivian, serena, ryan, aiden,
            # ono_anna, sohee, uncle_fu, eric and dylan.
            return AudioCppFamily(family, "tts", request_defaults = {"voice": "vivian"})
        if "base" in text:
            return AudioCppFamily(
                family,
                "",
                unsupported = "Qwen3-TTS Base only clones a reference voice, which Studio does not send yet.",
            )
    return policy


# Legacy dictation keys and the sub-folder ids of earlier builds, mapped to the folder row and
# the variant they named. Saved settings keep working through these.
_LEGACY_KEYS: dict[str, tuple[str, Optional[str]]] = {
    "audiocpp-kokoro-82m": ("Kokoro-82M-GGUF", None),
    "audiocpp-kitten-tts-mini": ("KittenTTS-GGUF", None),
    "audiocpp-piper-en-us-lessac": ("Piper-TTS-GGUF", None),
    "audiocpp-inflect-micro-v2": ("Inflect-Micro-v2-GGUF", None),
    "audiocpp-pocket-tts-en": ("PocketTTS-GGUF", "english/Q8_0"),
    "audiocpp-moss-tts-nano": ("MOSS-TTS-Nano-100M-GGUF", None),
    "audiocpp-supertonic-3": ("Supertonic-3-GGUF", None),
    "audiocpp-chatterbox-turbo": ("Chatterbox-Turbo-GGUF", None),
    "audiocpp-voxcpm2": ("VoxCPM2-GGUF", None),
    "audiocpp-qwen3-tts-1.7b-customvoice": ("Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF", None),
    "audiocpp-qwen3-tts-1.7b-voicedesign": ("Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF", None),
    "audiocpp-ace-step-1.5-turbo": ("ACE-Step1.5-GGUF", "turbo/Q8_0"),
    "audiocpp-ace-step-1.5-base": ("ACE-Step1.5-GGUF", "base/Q8_0"),
    "audiocpp-stable-audio-3-small-music": ("Stable-Audio-3-Small-Music-GGUF", None),
    "audiocpp-qwen3-asr-0.6b": ("Qwen3-ASR-0.6B-GGUF", None),
    "audiocpp-qwen3-asr-1.7b": ("Qwen3-ASR-1.7B-GGUF", None),
    "audiocpp-parakeet-tdt-0.6b-v3": ("Parakeet-TDT-0.6B-v3-GGUF", None),
    "audiocpp-canary-180m-flash": ("Canary-180M-Flash-GGUF", None),
    "audiocpp-moonshine-tiny": ("Moonshine-Streaming-GGUF", "tiny/Q8_0"),
    "audiocpp-moonshine-small": ("Moonshine-Streaming-GGUF", "small/Q8_0"),
    "audiocpp-nemotron-3.5-asr-0.6b": ("Nemotron-3.5-ASR-Streaming-0.6B-GGUF", None),
}
_LEGACY_SUBFOLDER_VARIANTS = {
    "moonshine-streaming-gguf/tiny": "tiny/Q8_0",
    "moonshine-streaming-gguf/small": "small/Q8_0",
}


@dataclass(frozen = True)
class AudioCppRef:
    """Where a Studio id points: a repo, a folder in it (umbrella rows), or a local GGUF."""

    id: str
    repo_id: Optional[str]
    folder: str = ""
    variant_hint: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def display_name(self) -> str:
        if self.local_path:
            return Path(self.local_path).stem
        if self.folder:
            return self.folder
        return str(self.repo_id).split("/")[-1]


_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def is_umbrella_id(identifier: Optional[str]) -> bool:
    """A folder row of the audio.cpp umbrella repo, or a legacy dictation key. No I/O."""
    if not identifier:
        return False
    text = str(identifier).strip()
    return text in _LEGACY_KEYS or text.lower().startswith(_UMBRELLA_PREFIX)


def parse_identifier(identifier: Optional[str]) -> Optional[AudioCppRef]:
    """The ref ``identifier`` names, without any I/O; None when it cannot name an audio.cpp model."""
    if not identifier:
        return None
    text = str(identifier).strip().rstrip("/")
    if not text:
        return None
    legacy = _LEGACY_KEYS.get(text)
    if legacy is not None:
        folder, hint = legacy
        return AudioCppRef(f"{AUDIO_CPP_REPO}/{folder}", AUDIO_CPP_REPO, folder, hint)
    if text.lower().startswith(_UMBRELLA_PREFIX):
        rest = text[len(_UMBRELLA_PREFIX) :].replace("\\", "/").strip("/")
        parts = [p for p in rest.split("/") if p]
        if not parts or any(p in (".", "..") for p in parts):
            return None
        folder = parts[0]
        sub = "/".join(parts[1:])
        hint = None
        if sub:
            hint = _LEGACY_SUBFOLDER_VARIANTS.get(f"{folder}/{sub}".lower(), sub)
        return AudioCppRef(f"{AUDIO_CPP_REPO}/{folder}", AUDIO_CPP_REPO, folder, hint)
    if text.lower().endswith(".gguf"):
        path = Path(text).expanduser()
        try:
            if path.is_file():
                return AudioCppRef(str(path), None, local_path = str(path))
        except OSError:
            return None
        return None
    if _REPO_ID_RE.match(text):
        return AudioCppRef(text, text)
    return None


def split_variant_ref(identifier: str) -> tuple[str, Optional[str]]:
    """``repo:Q8_0`` -> ``("repo", "Q8_0")``, ``row:tiny/Q8_0`` -> ``("row", "tiny/Q8_0")``; a bare id
    keeps no variant. Paths are left whole."""
    text = str(identifier or "").strip()
    if ":" in text and not re.match(r"^[A-Za-z]:[\\/]", text):
        base, _, variant = text.rpartition(":")
        if base and variant and not variant.startswith(("/", "\\")):
            return base, variant
    return text, None


# ---------------------------------------------------------------------------
# GGUF headers

_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_WANTED_KEYS = (
    "general.architecture",
    "audiocpp.model_spec.family",
    "audiocpp.model_spec.json",
)


@dataclass(frozen = True)
class AudioCppHeader:
    architecture: Optional[str]
    family: Optional[str]
    spec: Optional[dict] = field(default = None, hash = False, compare = False)
    # False when the read stopped before the header's end (a remote prefix): absent keys may lie beyond it.
    complete: bool = True

    @property
    def is_audio_cpp(self) -> bool:
        return (self.architecture or "").strip().lower() == AUDIO_CPP_ARCHITECTURE


def parse_header(stream: BinaryIO) -> Optional[AudioCppHeader]:
    """The audio.cpp keys of a GGUF header, reading only up to the last one wanted.

    Truncation-safe: a prefix that stops early answers with what it read and ``complete=False``.
    """
    try:
        head = stream.read(24)
        if len(head) < 24:
            return None
        magic, _version, _tensors, kv_count = struct.unpack("<IIQQ", head)
        if magic != 0x46554747 or kv_count > 1 << 20:
            return None
        found: dict[str, str] = {}
        for _ in range(kv_count):
            raw = stream.read(8)
            if len(raw) < 8:
                return _header(found, complete = False)
            (klen,) = struct.unpack("<Q", raw)
            if klen > 1 << 16:
                return _header(found, complete = False)
            key_bytes = stream.read(klen)
            type_bytes = stream.read(4)
            if len(key_bytes) < klen or len(type_bytes) < 4:
                return _header(found, complete = False)
            key = key_bytes.decode("utf-8", "replace")
            (vtype,) = struct.unpack("<I", type_bytes)
            if key in _WANTED_KEYS and vtype == 8:
                raw = stream.read(8)
                if len(raw) < 8:
                    return _header(found, complete = False)
                (slen,) = struct.unpack("<Q", raw)
                if slen > 1 << 26:
                    return _header(found, complete = False)
                value = stream.read(slen)
                if len(value) < slen:
                    return _header(found, complete = False)
                found[key] = value.decode("utf-8", "replace")
                if all(k in found for k in _WANTED_KEYS):
                    break
                continue
            if not _skip_value(stream, vtype):
                return _header(found, complete = False)
        return _header(found, complete = True)
    except (OSError, struct.error, ValueError):
        return None


def _skip_value(stream: BinaryIO, vtype: int) -> bool:
    if vtype == 8:
        raw = stream.read(8)
        if len(raw) < 8:
            return False
        stream.seek(struct.unpack("<Q", raw)[0], 1)
        return True
    if vtype == 9:
        raw = stream.read(12)
        if len(raw) < 12:
            return False
        atype, alen = struct.unpack("<IQ", raw)
        if atype == 8:
            for _ in range(alen):
                raw = stream.read(8)
                if len(raw) < 8:
                    return False
                stream.seek(struct.unpack("<Q", raw)[0], 1)
            return True
        size = _FIXED_SIZES.get(atype)
        if size is None:
            return False
        stream.seek(size * alen, 1)
        return True
    size = _FIXED_SIZES.get(vtype)
    if size is None:
        return False
    stream.seek(size, 1)
    return True


def _header(found: dict[str, str], *, complete: bool) -> AudioCppHeader:
    spec = None
    raw = found.get("audiocpp.model_spec.json")
    if raw:
        try:
            parsed = json.loads(raw)
            spec = parsed if isinstance(parsed, dict) else None
        except ValueError:
            spec = None
    family = found.get("audiocpp.model_spec.family") or (spec or {}).get("family")
    return AudioCppHeader(
        architecture = found.get("general.architecture"),
        family = str(family).strip() if family else None,
        spec = spec,
        complete = complete,
    )


_header_cache: dict[tuple, Optional[AudioCppHeader]] = {}
_header_lock = threading.Lock()
_HEADER_CACHE_MAX = 512


def _cached_header(key: tuple, compute) -> Optional[AudioCppHeader]:
    with _header_lock:
        if key in _header_cache:
            return _header_cache[key]
    value = compute()
    with _header_lock:
        while len(_header_cache) >= _HEADER_CACHE_MAX:
            _header_cache.pop(next(iter(_header_cache)))
        _header_cache[key] = value
    return value


def read_local_header(path: str | Path) -> Optional[AudioCppHeader]:
    """The audio.cpp header keys of a local GGUF, memoized on its size and mtime. Never raises."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = ("local", os.path.realpath(path), stat.st_size, int(stat.st_mtime))

    def compute():
        try:
            with open(path, "rb") as f:
                return parse_header(f)
        except OSError:
            return None

    return _cached_header(key, compute)


def read_remote_header(
    repo_id: str,
    filename: str,
    hf_token: Optional[str] = None,
    *,
    size: Optional[int] = None,
) -> Optional[AudioCppHeader]:
    """The audio.cpp header keys of a Hub GGUF from a ranged read of its first bytes.

    The architecture is the first key, so 64 KiB answers "is this audio.cpp". Some writers put the
    spec after the embedded sidecar files, so an audio.cpp file whose prefix held no family is read
    again further in.
    """

    def compute():
        from core.inference.diffusion_compat import _read_gguf_header

        header = None
        for max_bytes in (1 << 16, 1 << 22):
            if size is not None and 0 < size <= max_bytes // 2 and header is not None:
                break
            data = _read_gguf_header(
                repo_id, filename, hf_token, max_bytes = max_bytes, timeout_seconds = 10
            )
            if not data:
                return header
            header = parse_header(io.BytesIO(data))
            if header is None or not header.is_audio_cpp or header.family or header.complete:
                return header
        return header

    return _cached_header(("remote", repo_id.lower(), filename, size), compute)


# ---------------------------------------------------------------------------
# Listing


@dataclass(frozen = True)
class RepoFile:
    path: str
    size: int = 0


def hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _hub_cache() -> Path:
    from utils.hf_cache_settings import active_hf_hub_cache
    return Path(active_hf_hub_cache())


def snapshot_dirs(repo_id: str, hub_cache: Optional[Path] = None) -> list[Path]:
    """Cached snapshots of ``repo_id``: ``refs/main`` first, then newest."""
    root = hub_cache if hub_cache is not None else _hub_cache()
    repo_dir = root / ("models--" + repo_id.replace("/", "--"))
    try:
        candidates = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()]
    except OSError:
        if hub_cache is not None:
            return []
        # A cached repo whose casing differs from the request.
        try:
            wanted = repo_dir.name.lower()
            match = next((p for p in root.iterdir() if p.name.lower() == wanted), None)
        except OSError:
            match = None
        if match is None:
            return []
        return snapshot_dirs(match.name[len("models--") :].replace("--", "/"), root)
    try:
        main_sha = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
    except OSError:
        main_sha = ""

    def order(p: Path) -> tuple[int, float]:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (0 if p.name == main_sha else 1, -mtime)

    return sorted(candidates, key = order)


def _cached_listing(repo_id: str, folder: str) -> list[RepoFile]:
    """Every file under ``folder`` in any cached snapshot, from disk."""
    seen: dict[str, RepoFile] = {}
    for snapshot in snapshot_dirs(repo_id):
        base = snapshot / folder if folder else snapshot
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                full = Path(dirpath) / name
                rel = full.relative_to(snapshot).as_posix()
                if rel in seen:
                    continue
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                seen[rel] = RepoFile(rel, size)
    return sorted(seen.values(), key = lambda f: f.path)


def _hub_listing(repo_id: str, folder: str, hf_token: Optional[str]) -> list[RepoFile]:
    from huggingface_hub import HfApi

    api = HfApi(token = hf_token or None)
    entries = api.list_repo_tree(
        repo_id, path_in_repo = folder or None, recursive = True, repo_type = "model"
    )
    out = []
    for entry in entries:
        size = getattr(entry, "size", None)
        if size is None:  # a folder
            continue
        out.append(RepoFile(str(entry.path), int(size or 0)))
    return sorted(out, key = lambda f: f.path)


_listing_cache: dict[tuple, tuple[float, list[RepoFile], bool]] = {}
_listing_lock = threading.Lock()
_LISTING_TTL_SECONDS = 300.0


def list_files(
    repo_id: str,
    folder: str = "",
    hf_token: Optional[str] = None,
    *,
    network: bool = True,
) -> tuple[list[RepoFile], bool]:
    """``(files, from_hub)`` under ``folder``: the Hub listing, else the cached copy."""
    from hub.utils.hf_tokens import normalize_token

    token = normalize_token(hf_token) if hf_token is not False else None
    use_hub = network and not hub_offline()
    key = (repo_id.lower(), folder.lower(), hashlib.sha1(str(token).encode()).hexdigest()[:8])
    if use_hub:
        with _listing_lock:
            hit = _listing_cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < _LISTING_TTL_SECONDS:
            return list(hit[1]), hit[2]
        try:
            files = _hub_listing(repo_id, folder, token)
            with _listing_lock:
                _listing_cache[key] = (time.monotonic(), files, True)
            return list(files), True
        except Exception as exc:  # noqa: BLE001 - offline, refused or gone: the cache may still hold it
            logger.debug("audio.cpp: Hub listing of %s/%s failed: %s", repo_id, folder, exc)
    return _cached_listing(repo_id, folder), False


# ---------------------------------------------------------------------------
# Variants

_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:i?q\d(?:_[a-z0-9]+)*)|bf16|f16|f32|fp16|fp32|orig)(?=$|[-_.])", re.IGNORECASE
)
_QUANT_PREFERENCE = ("Q8_0", "F16", "BF16", "ORIG", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "F32")
_PREFERRED_SCOPES = ("turbo", "english", "tiny")
_SKIP_FILE_RE = re.compile(
    r"(^|/)(readme[^/]*|license[^/]*|\.gitattributes|[^/]*\.(md|mp4|mp3|wav|png|jpe?g|gif))$", re.I
)


def _is_quant(word: str) -> bool:
    """Whether ``word`` is a quant label (``Q8_0``, ``BF16``, ``orig``) rather than a name."""
    return bool(_QUANT_RE.fullmatch(f"-{word}"))


def quant_label(filename: str) -> str:
    """``kokoro-82m-q8_0.gguf`` -> ``Q8_0``; the stem's last word when no quant is named."""
    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    matches = list(_QUANT_RE.finditer(stem))
    if matches:
        token = matches[-1].group(1)
        # "orig" (unquantized as published) is a word, not a quant token.
        return token.lower() if token.lower() == "orig" else token.upper()
    return re.split(r"[-_.]", stem)[-1].upper() or stem.upper()


@dataclass(frozen = True)
class AudioCppVariant:
    key: str
    files: tuple[RepoFile, ...]
    # Repo path the server opens; empty for a package, which loads as a directory.
    primary: str
    session_options: dict = field(default_factory = dict, hash = False, compare = False)
    label: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def main_file(self) -> str:
        """The GGUF that stands for this variant in listings."""
        if self.primary:
            return self.primary
        ggufs = [f for f in self.files if f.path.lower().endswith(".gguf")]
        return max(ggufs, key = lambda f: f.size).path if ggufs else self.files[0].path


def _single_file_variants(files: Sequence[RepoFile], folder: str) -> list[AudioCppVariant]:
    prefix = f"{folder}/" if folder else ""
    ggufs = [
        f
        for f in files
        if f.path.lower().endswith(".gguf")
        and "mmproj" not in f.path.lower()
        and "imatrix" not in f.path.lower()
    ]
    rows = []
    for gguf in ggufs:
        rel = gguf.path[len(prefix) :] if gguf.path.startswith(prefix) else gguf.path
        sub = rel.rpartition("/")[0]
        rows.append((gguf, sub, quant_label(rel)))
    groups: dict[str, list[str]] = {}
    for gguf, sub, quant in rows:
        k = f"{sub}/{quant}" if sub else quant
        groups.setdefault(k.lower(), []).append(Path(gguf.path).name[: -len(".gguf")])
    variants = []
    for gguf, sub, quant in rows:
        key = f"{sub}/{quant}" if sub else quant
        label = None
        stems = groups[key.lower()]
        if len(stems) > 1:
            # Several files at one quant in one place (Moonshine tiny, small and medium): the words their
            # names do not share tell them apart, and name the row like a sub-folder would.
            scope = _distinct_part(Path(gguf.path).name[: -len(".gguf")], stems)
            key = f"{sub}/{scope}/{quant}" if sub else f"{scope}/{quant}"
            label = f"{quant} · {scope}"
        elif sub:
            label = f"{quant} · {sub}"
        gguf_dir = gguf.path.rpartition("/")[0]
        # Voices and other extras audio.cpp reads beside the GGUF (PocketTTS's embeddings/*.safetensors).
        extras = tuple(
            f
            for f in files
            if f.path.startswith(f"{gguf_dir}/embeddings/" if gguf_dir else "embeddings/")
        )
        variants.append(AudioCppVariant(key, (gguf, *extras), gguf.path, label = label))
    return variants


def _distinct_part(stem: str, stems: Sequence[str]) -> str:
    """The words of ``stem`` its siblings do not share: ``moonshine-streaming-tiny-q8_0`` among
    ``...-small-q8_0`` gives ``tiny``. The whole stem when nothing is left."""
    split = [re.split(r"[-_.]", s) for s in stems]
    words = re.split(r"[-_.]", stem)
    head = 0
    while all(len(w) > head for w in split) and len({w[head].lower() for w in split}) == 1:
        head += 1
    tail = 0
    while (
        all(len(w) > head + tail for w in split) and len({w[-1 - tail].lower() for w in split}) == 1
    ):
        tail += 1
    middle = words[head : len(words) - tail]
    return "-".join(middle) or stem


def _package_variants(
    policy: AudioCppFamily, files: Sequence[RepoFile], folder: str
) -> list[AudioCppVariant]:
    prefix = f"{folder}/" if folder else ""
    by_path = {f.path: f for f in files}
    variants = []
    for package in policy.package:
        found = [by_path.get(prefix + rel) for rel in package.files]
        if any(f is None for f in found):
            continue
        variants.append(
            AudioCppVariant(
                package.key, tuple(found), "", session_options = dict(package.session_options)
            )
        )
    return variants


def _variant_rank(variant: AudioCppVariant) -> tuple:
    sub = variant.key.rpartition("/")[0].lower()
    scope_rank = (
        0 if not sub else (1 + _PREFERRED_SCOPES.index(sub) if sub in _PREFERRED_SCOPES else 10)
    )
    quant = quant_label(variant.main_file)
    quant_rank = (
        _QUANT_PREFERENCE.index(quant) if quant in _QUANT_PREFERENCE else len(_QUANT_PREFERENCE)
    )
    stem = Path(variant.main_file).name.lower()
    stem_rank = min(
        (i for i, word in enumerate(_PREFERRED_SCOPES) if word in stem),
        default = len(_PREFERRED_SCOPES),
    )
    return (scope_rank, stem_rank, quant_rank, variant.key.lower())


def match_variant(
    variants: Sequence[AudioCppVariant], wanted: Optional[str]
) -> Optional[AudioCppVariant]:
    """The variant ``wanted`` names: its key, its file, or a quant that picks one row.

    A bare quant shared by several rows (ACE-Step ``turbo/Q8_0`` and ``base/Q8_0``) picks the default
    ordering's first, which is what the folder row loads by default.
    """
    text = (wanted or "").strip().replace("\\", "/")
    if not text:
        return None
    low = text.lower()
    for variant in variants:
        if variant.key.lower() == low:
            return variant
    for variant in variants:
        main = variant.main_file.lower()
        if main == low or main.rsplit("/", 1)[-1] in (low, f"{low}.gguf"):
            return variant
    quant = low.rsplit("/", 1)[-1]
    matches = [
        v
        for v in variants
        if quant_label(v.main_file).lower() == quant
        and ("/" not in low or v.key.lower().endswith(low))
    ]
    if matches:
        return sorted(matches, key = _variant_rank)[0]
    # A sub-folder alone (legacy ``.../PocketTTS-GGUF/english``).
    scoped = [v for v in variants if v.key.lower().startswith(low + "/")]
    if scoped:
        return sorted(scoped, key = _variant_rank)[0]
    return _match_by_words(variants, low)


def _match_by_words(variants: Sequence[AudioCppVariant], low: str) -> Optional[AudioCppVariant]:
    """A key named by the words of its file (``tiny/Q8_0`` or ``tiny``), whatever the listing keyed it.

    Which rows need a name beyond their quant depends on the listing: with only Moonshine tiny in
    the cache it is simply ``Q8_0``, while the Hub listing (and /gguf-variants) calls it
    ``tiny/Q8_0``. Matching the scope words against the file keeps both spellings loading it.
    """
    parts = [part for part in low.split("/") if part]
    if not parts:
        return None
    quant = parts[-1] if _is_quant(parts[-1]) else None
    scopes = parts[:-1] if quant else parts
    if not scopes:
        return None
    matches = [
        v
        for v in variants
        if all(scope in re.split(r"[-_./]", v.main_file.lower()) for scope in scopes)
        and (quant is None or quant_label(v.main_file).lower() == quant)
    ]
    return sorted(matches, key = _variant_rank)[0] if matches else None


# ---------------------------------------------------------------------------
# Options


_STUDIO_DRIVEN_OPTIONS = frozenset(
    {
        "input",
        "text",
        "language",
        "seed",
        "instruct",
        "instruction",
        "instructions",
        "reference_text",
        "voice_ref",
        "phonemes",
        "return_timestamps",
        "stop_after",
        "export_semantic",
        "abc",
        # A JSON array of codec indices: a debugging hook, not a setting.
        "semantic_prefix",
    }
)
# The music form fills these from its lyrics, description and duration fields.
_MUSIC_DRIVEN_OPTIONS = frozenset(
    {"lyrics", "style", "caption", "prompt", "tags", "duration", "duration_sec", "duration_seconds"}
)
_RENDERABLE_TYPES = frozenset({"bool", "int", "float", "string", "enum"})
_MAX_STRING_OPTION = 4000


def runtime_spec(family: str) -> Optional[dict]:
    """``model_specs/<family>.json`` of the installed runtime, the spec the binary was built with."""
    if not family or not re.fullmatch(r"[a-z0-9_]+", family):
        return None
    try:
        from core.inference.audio_cpp_server import find_audio_cpp_server_binary
        binary = find_audio_cpp_server_binary()
    except Exception:  # noqa: BLE001 - no runtime, no runtime spec
        return None
    if not binary:
        return None
    for parent in list(Path(binary).resolve().parents)[:3]:
        path = parent / "model_specs" / f"{family}.json"
        try:
            with open(path, "r", encoding = "utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            continue
    return None


def _clean_option(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    kind = str(raw.get("type") or "").strip().lower()
    if not name or "." in name or kind not in _RENDERABLE_TYPES:
        return None
    if name in _STUDIO_DRIVEN_OPTIONS or name.endswith(("_file", "_path", "_dir")):
        return None
    option: dict[str, Any] = {
        "name": name,
        "type": kind,
        "description": str(raw.get("description") or ""),
        "required": bool(raw.get("required", False)),
        "default": raw.get("default"),
        "min": raw.get("min"),
        "max": raw.get("max"),
        "values": None,
    }
    if kind == "enum":
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            return None
        option["values"] = [str(v) for v in values]
    for bound in ("min", "max"):
        if not isinstance(option[bound], (int, float)) or isinstance(option[bound], bool):
            option[bound] = None
    if kind in ("int", "float") and (
        isinstance(option["default"], bool) or not isinstance(option["default"], (int, float))
    ):
        option["default"] = None
    if kind == "bool" and not isinstance(option["default"], bool):
        option["default"] = None
    if kind in ("string", "enum") and option["default"] is not None:
        option["default"] = str(option["default"])
    return option


def option_schema(
    policy: AudioCppFamily, spec: Optional[dict], embedded: Optional[dict]
) -> tuple[dict, ...]:
    """The request options Studio offers for this model, in the spec's order.

    A family with a Studio-side list (MiniMax Music 3, YuE2) shows that list: their specs carry
    dozens of planner and debugging knobs, and the published GGUFs embed a stale copy. Everyone
    else gets the runtime's spec, else the one embedded in the GGUF.
    """
    if policy.options:
        source = {"options": {"request": list(policy.options)}}
    else:
        source = spec
        if source is None or not ((source.get("options") or {}).get("request")):
            source = embedded
    raw = ((source or {}).get("options") or {}).get("request") or []
    driven = _MUSIC_DRIVEN_OPTIONS if policy.task == "music" else frozenset()
    out = []
    names = set()
    for item in raw:
        option = _clean_option(item)
        if option is None or option["name"] in driven or option["name"] in names:
            continue
        if option["name"] == "voice":
            continue
        names.add(option["name"])
        out.append(option)
    if policy.task == "tts" and policy.default_server_task != "vdes":
        ui = (source or {}).get("ui") or (embedded or {}).get("ui") or {}
        voices = ui.get("builtin_voices") if isinstance(ui, dict) else None
        if isinstance(voices, list) and voices and all(isinstance(v, str) for v in voices):
            default = policy.request_defaults.get("voice") or ui.get("default_voice")
            out.insert(
                0,
                {
                    "name": "voice",
                    "type": "enum",
                    "description": "Built-in speaker.",
                    "required": False,
                    "default": str(default) if default in voices else None,
                    "min": None,
                    "max": None,
                    "values": list(voices),
                },
            )
    return tuple(out)


def validate_options(schema: Sequence[dict], values: Optional[dict]) -> dict:
    """``values`` checked against ``schema``: unknown names and values of the wrong kind are dropped,
    numbers are clamped to their bounds."""
    if not isinstance(values, dict):
        return {}
    by_name = {o["name"]: o for o in schema}
    out: dict[str, Any] = {}
    for name, value in values.items():
        option = by_name.get(str(name))
        if option is None or value is None:
            continue
        kind = option["type"]
        try:
            if kind == "bool":
                if isinstance(value, str):
                    if value.strip().lower() not in ("true", "false", "1", "0"):
                        continue
                    value = value.strip().lower() in ("true", "1")
                elif not isinstance(value, (bool, int)):
                    continue
                out[name] = bool(value)
            elif kind in ("int", "float"):
                if isinstance(value, bool):
                    continue
                number = float(value)
                if number != number or number in (float("inf"), float("-inf")):
                    continue
                if option.get("min") is not None:
                    number = max(float(option["min"]), number)
                if option.get("max") is not None:
                    number = min(float(option["max"]), number)
                out[name] = int(round(number)) if kind == "int" else number
            elif kind == "enum":
                if str(value) in (option.get("values") or []):
                    out[name] = str(value)
            elif kind == "string":
                text = str(value)
                if text.strip():
                    out[name] = text[:_MAX_STRING_OPTION]
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Resolution


@dataclass(frozen = True)
class AudioCppModel:
    """One loadable audio.cpp model: a repo (or folder, or local file) at one variant."""

    id: str
    repo_id: Optional[str]
    folder: str
    display_name: str
    family: str
    task: str
    server_task: str
    variant: AudioCppVariant
    variants: tuple[AudioCppVariant, ...]
    default_variant: str
    needs_espeak: bool = False
    request_defaults: dict = field(default_factory = dict, hash = False, compare = False)
    model_options: dict = field(default_factory = dict, hash = False, compare = False)
    options: tuple[dict, ...] = field(default = (), hash = False, compare = False)
    unsupported: Optional[str] = None
    local_path: Optional[str] = None
    tags: tuple[str, ...] = ()

    @property
    def is_package(self) -> bool:
        return not self.variant.primary

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.variant.files)

    @property
    def gguf_file(self) -> str:
        return self.variant.main_file

    @property
    def size_bytes(self) -> int:
        return self.variant.size_bytes

    @property
    def audio_type(self) -> Optional[str]:
        if self.task == "tts":
            return AUDIO_CPP_TTS_AUDIO_TYPE
        if self.task == "music":
            return AUDIO_CPP_MUSIC_AUDIO_TYPE
        return None

    @property
    def hub_task(self) -> Optional[str]:
        return HUB_TASKS.get(self.task)

    @property
    def key(self) -> str:
        """A short name, unique per model and variant, for the link farm."""
        stem = re.sub(r"[^a-z0-9]+", "-", self.display_name.lower()).strip("-")[:24] or "model"
        digest = hashlib.sha1(f"{self.id.lower()}@{self.variant.key.lower()}".encode()).hexdigest()
        return f"{stem}-{digest[:8]}"

    @property
    def canonical_id(self) -> str:
        """``id:variant``: what dictation records, so a reload picks the same files."""
        return f"{self.id}:{self.variant.key}"

    def with_variant(self, variant: AudioCppVariant) -> "AudioCppModel":
        from dataclasses import replace

        model_options = dict(self.model_options)
        if variant.session_options:
            model_options["session_options"] = {
                **dict(model_options.get("session_options") or {}),
                **variant.session_options,
            }
        return replace(self, variant = variant, model_options = model_options)


class AudioCppModelError(ValueError):
    """The id names an audio.cpp model Studio cannot run (task, package or variant)."""


def _names_for(ref: AudioCppRef, files: Sequence[RepoFile]) -> tuple[str, ...]:
    return (
        ref.id,
        ref.folder,
        *(Path(f.path).name for f in files if f.path.lower().endswith(".gguf")),
    )


_TAG_FAMILY_RE = re.compile(r"^[a-z0-9_]+$")


def family_from_names(names: Iterable[str]) -> Optional[str]:
    text = " ".join(names).lower().replace(".", "")
    for family in sorted(FAMILIES, key = len, reverse = True):
        pattern = re.escape(family).replace(r"\_", "[-_]?")
        if re.search(pattern, text):
            return family
    hints = (
        (r"minimax[-_]?music[-_]?3", "minimax_music3"),
        (r"yue[-_]?2", "yue2"),
        (r"moonshine", "moonshine_asr"),
        (r"qwen3[-_]?asr", "qwen3_asr"),
        (r"qwen3[-_]?tts", "qwen3_tts"),
        (r"parakeet", "parakeet_tdt"),
        (r"canary", "canary_asr"),
        (r"nemotron.*asr", "nemotron_asr"),
        (r"pocket[-_]?tts", "pocket_tts"),
        (r"kokoro", "kokoro_tts"),
        (r"kitten", "kitten_tts"),
        (r"piper", "piper_tts"),
        (r"inflect", "inflect_v2"),
        (r"ace[-_]?step", "ace_step"),
        (r"stable[-_]?audio", "stable_audio"),
        (r"voxtral", "voxtral_realtime"),
    )
    for pattern, family in hints:
        if re.search(pattern, text):
            return family
    return None


def _probe_header(
    ref: AudioCppRef, candidates: Sequence[RepoFile], hf_token: Optional[str], network: bool
) -> Optional[AudioCppHeader]:
    """The first header among ``candidates`` that names a family, else the first one read."""
    first = None
    local_first: list[tuple[RepoFile, Optional[Path]]] = []
    for f in candidates:
        local = None
        if ref.local_path:
            local = Path(ref.local_path)
        elif ref.repo_id:
            for snapshot in snapshot_dirs(ref.repo_id):
                path = snapshot / f.path
                if path.is_file():
                    local = path
                    break
        local_first.append((f, local))
    local_first.sort(key = lambda item: item[1] is None)
    for f, local in local_first[:4]:
        header = None
        if local is not None:
            header = read_local_header(local)
        elif network and not hub_offline() and ref.repo_id:
            header = read_remote_header(ref.repo_id, f.path, hf_token, size = f.size or None)
        if header is None:
            continue
        if not header.is_audio_cpp:
            return header
        if header.family:
            return header
        first = first or header
    return first


_resolve_cache: dict[tuple, tuple[float, Optional["AudioCppModel"]]] = {}
_resolve_lock = threading.Lock()
_RESOLVE_TTL_SECONDS = 300.0
_known_audio_cpp_ids: set[str] = set()


def looks_like_audio_cpp(identifier: Optional[str]) -> bool:
    """An umbrella row, a legacy key, or an id this process already resolved as audio.cpp. No I/O."""
    if is_umbrella_id(identifier):
        return True
    base, _variant = split_variant_ref(str(identifier or ""))
    return base.lower() in _known_audio_cpp_ids


def resolve(
    identifier: Optional[str],
    variant: Optional[str] = None,
    hf_token: Optional[str] = None,
    *,
    network: bool = True,
    tags: Sequence[str] = (),
    gguf_hint: Optional[str] = None,
) -> Optional[AudioCppModel]:
    """The audio.cpp model ``identifier`` names at ``variant`` (default when None), else None.

    None means "not an audio.cpp model". A model Studio cannot run (unsupported task, missing
    variant) still resolves, with ``unsupported`` set, so callers can refuse it clearly.
    ``network=False`` answers from the HF cache only. ``gguf_hint`` names a GGUF the caller
    already found in the repo: its header alone rules out an ordinary llama.cpp repo, before
    any listing.
    """
    if identifier is None:
        return None
    base, ref_variant = split_variant_ref(str(identifier))
    ref = parse_identifier(base)
    if ref is None:
        return None
    wanted = variant or ref_variant or ref.variant_hint
    key = (ref.id.lower(), (wanted or "").lower(), bool(network))
    with _resolve_lock:
        hit = _resolve_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < (_RESOLVE_TTL_SECONDS if network else 20.0):
        return hit[1]
    if gguf_hint and ref.repo_id and not is_umbrella_id(ref.id):
        header = _probe_header(ref, [RepoFile(gguf_hint)], hf_token, network)
        if header is not None and not header.is_audio_cpp:
            with _resolve_lock:
                _resolve_cache[key] = (time.monotonic(), None)
            return None
    model = _resolve_uncached(ref, wanted, hf_token, network, tuple(tags))
    with _resolve_lock:
        if len(_resolve_cache) > 256:
            _resolve_cache.clear()
        _resolve_cache[key] = (time.monotonic(), model)
        if model is not None:
            _known_audio_cpp_ids.add(model.id.lower())
    return model


def forget(identifier: Optional[str] = None) -> None:
    """Drop memoized resolutions (after a download or delete changed the cache)."""
    with _resolve_lock:
        if identifier is None:
            _resolve_cache.clear()
            return
        low = split_variant_ref(str(identifier))[0].lower()
        for key in [k for k in _resolve_cache if k[0] == low]:
            _resolve_cache.pop(key, None)


def _resolve_uncached(
    ref: AudioCppRef,
    wanted: Optional[str],
    hf_token: Optional[str],
    network: bool,
    tags: tuple[str, ...],
) -> Optional[AudioCppModel]:
    if ref.local_path:
        files = [RepoFile(Path(ref.local_path).name, _size(ref.local_path))]
    else:
        files, _from_hub = list_files(ref.repo_id, ref.folder, hf_token, network = network)
    files = [f for f in files if not _SKIP_FILE_RE.search(f.path)]
    ggufs = [f for f in files if f.path.lower().endswith(".gguf")]
    if not ggufs:
        return None
    header = _probe_header(ref, ggufs, hf_token, network)
    names = _names_for(ref, files)
    umbrella = ref.repo_id is not None and ref.repo_id.lower() == AUDIO_CPP_REPO.lower()
    if header is not None and not header.is_audio_cpp:
        return None
    family = header.family if header is not None else None
    if header is None:
        # Unread (offline, not cached): only a repo that says it is audio.cpp counts.
        owner = str(ref.repo_id or "").split("/")[0].lower()
        text = " ".join(names).lower()
        tagged = any(t.lower() in ("audio.cpp", "audiocpp") for t in tags)
        if not (
            umbrella or owner == "audio-cpp" or tagged or "audiocpp" in text or "audio.cpp" in text
        ):
            return None
    if not family:
        family = next(
            (t for t in tags if _TAG_FAMILY_RE.match(t) and t in FAMILIES), None
        ) or family_from_names(names)
    if not family:
        reason = "Studio could not tell which audio.cpp model family this GGUF is."
        variant = AudioCppVariant(quant_label(ggufs[0].path), (ggufs[0],), ggufs[0].path)
        return AudioCppModel(
            id = ref.id,
            repo_id = ref.repo_id,
            folder = ref.folder,
            display_name = ref.display_name,
            family = "",
            task = "",
            server_task = "",
            variant = variant,
            variants = (variant,),
            default_variant = variant.key,
            unsupported = reason,
            local_path = ref.local_path,
        )
    spec = runtime_spec(family)
    embedded = header.spec if header is not None else None
    policy = family_policy(family, spec or embedded, names)
    if policy.package:
        # In the package's own order: its first mix is the one the runtime defaults to.
        variants = _package_variants(policy, files, ref.folder)
    else:
        variants = _single_file_variants(files, ref.folder)
        if ref.local_path:
            variants = [
                AudioCppVariant(v.key, v.files, ref.local_path, label = v.label) for v in variants
            ]
        variants.sort(key = _variant_rank)
    unsupported = policy.unsupported
    if not variants:
        unsupported = (
            unsupported
            or "This repository does not publish every file the audio.cpp package needs."
        )
        variants = [AudioCppVariant(quant_label(ggufs[0].path), (ggufs[0],), ggufs[0].path)]
    default = variants[0]
    chosen = match_variant(variants, wanted) if wanted else default
    if (
        chosen is not None
        and wanted
        and "/" not in chosen.key
        and chosen.key.lower() != wanted.lower()
    ):
        # A partial cache listing named the row by quant alone; keep the name the full listing gives
        # it (``tiny/Q8_0``), so status and /gguf-variants agree on the variant that is loaded.
        text = wanted.strip().strip("/")
        head, _, tail = text.rpartition("/")
        scope = head if tail.lower() == chosen.key.lower() else ""
        if not scope and re.fullmatch(r"[A-Za-z0-9]+", text) and not _is_quant(text):
            scope = text  # a bare sub-variant name ("tiny"), not a quant or a file stem
        if scope:
            chosen = AudioCppVariant(
                f"{scope}/{chosen.key}",
                chosen.files,
                chosen.primary,
                dict(chosen.session_options),
                f"{chosen.key} · {scope.rsplit('/', 1)[-1]}",
            )
    if chosen is None:
        unsupported = unsupported or (
            f"Variant '{wanted}' not found. Available variants: "
            + ", ".join(v.key for v in variants)
        )
        chosen = default
    model = AudioCppModel(
        id = ref.id,
        repo_id = ref.repo_id,
        folder = ref.folder,
        display_name = ref.display_name,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = default,
        variants = tuple(variants),
        default_variant = default.key,
        needs_espeak = policy.needs_espeak,
        request_defaults = dict(policy.request_defaults),
        model_options = dict(policy.model_options),
        options = option_schema(policy, spec, embedded),
        unsupported = unsupported,
        local_path = ref.local_path,
        tags = tuple(tags),
    )
    return model.with_variant(chosen)


def _size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def package_variant_files(names: Iterable[str]) -> Optional[dict[str, tuple[str, ...]]]:
    """``{variant key: repo paths}`` when ``names`` (a repo's root listing) is an audio.cpp package
    layout Studio knows, else None. A pure function of the names, for the GGUF download planner."""
    present = set(names)
    for policy in _FAMILY_LIST:
        if not policy.package:
            continue
        found = {
            variant.key: variant.files
            for variant in policy.package
            if all(path in present for path in variant.files)
        }
        if found:
            return found
    return None


def download_target(
    identifier: Optional[str],
    variant: Optional[str],
    hf_token: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """``(repo, variant key)`` the Hub GGUF downloader fetches for an umbrella folder row, else None.

    The frontend names an umbrella model as ``audio-cpp/audio.cpp-gguf/<Folder>`` plus a quant, like
    any GGUF repo; the downloader and its progress work on the real repo and the path-qualified key
    its variant planner derives from the file.
    """
    if not is_umbrella_id(identifier):
        return None
    model = resolve(identifier, variant, hf_token, network = not hub_offline())
    if model is None or not model.repo_id or not model.variant.primary:
        return None
    from hub.utils.gguf import gguf_variant_key

    return model.repo_id, gguf_variant_key(model.variant.primary)


def folder_row_for_download(repo_id: str, variant: Optional[str]) -> Optional[tuple[str, str]]:
    """``(folder row id, variant key)`` for an umbrella download job, the inverse of
    ``download_target``: jobs run on the umbrella repo, but Studio names them by row. None for
    any other job. Answered from the cache, falling back to reading the planner key's path."""
    if str(repo_id or "").lower() != AUDIO_CPP_REPO.lower() or not variant or "/" not in variant:
        return None
    parts = variant.replace("\\", "/").split("/")
    row_id = f"{AUDIO_CPP_REPO}/{parts[0]}"
    try:
        model = resolve(row_id, network = False)
    except Exception:  # noqa: BLE001 - the path-derived key below still names it
        model = None
    if model is not None:
        from hub.utils.gguf import gguf_variant_key
        for candidate in model.variants:
            if candidate.primary and gguf_variant_key(candidate.primary).lower() == variant.lower():
                return row_id, candidate.key
    return row_id, "/".join([*parts[1:-1], quant_label(f"{parts[-1]}.gguf")])


def require_runnable(model: AudioCppModel, task: Optional[str] = None) -> None:
    """Raise ``AudioCppModelError`` when Studio cannot run ``model`` (for ``task`` when given)."""
    if model.unsupported:
        raise AudioCppModelError(model.unsupported)
    if task == "asr" and model.task != "asr":
        raise AudioCppModelError(f"{model.display_name} is not a speech-to-text model.")
    if task in ("tts", "music") and model.task not in ("tts", "music"):
        if model.task == "asr":
            raise AudioCppModelError(
                f"{model.display_name} is a speech-to-text model; choose it for dictation in "
                "Settings, then Voice."
            )
        raise AudioCppModelError(f"{model.display_name} is not a speech or music model.")


def repo_of(identifier: Optional[str]) -> Optional[str]:
    """The Hub repo that holds ``identifier``'s files. No I/O."""
    base, _variant = split_variant_ref(str(identifier or ""))
    ref = parse_identifier(base) if not base.lower().endswith(".gguf") else None
    return ref.repo_id if ref is not None else None


# Recommended ASR rows for dictation, in the order the Voice settings list them.
RECOMMENDED_STT_MODELS: tuple[str, ...] = tuple(
    f"{AUDIO_CPP_REPO}/{folder}"
    for folder in (
        "Qwen3-ASR-0.6B-GGUF",
        "Qwen3-ASR-1.7B-GGUF",
        "Parakeet-TDT-0.6B-v3-GGUF",
        "Canary-180M-Flash-GGUF",
        "Moonshine-Streaming-GGUF",
        "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
    )
)


_downloaded_cache: dict[Optional[str], tuple[float, list[AudioCppModel]]] = {}
_DOWNLOADED_TTL_SECONDS = 10.0


def downloaded_models(task: Optional[str] = None) -> list[AudioCppModel]:
    """audio.cpp models with a variant fully in the HF cache, found by header.

    Walks the active hub cache's repos, reading only GGUF headers (memoized). Umbrella folders
    are listed per folder. Status polls call this, so the answer is kept for a few seconds.
    """
    with _resolve_lock:
        hit = _downloaded_cache.get(task)
    if hit is not None and time.monotonic() - hit[0] < _DOWNLOADED_TTL_SECONDS:
        return list(hit[1])
    out = _scan_downloaded(task)
    with _resolve_lock:
        _downloaded_cache[task] = (time.monotonic(), out)
    return list(out)


def _scan_downloaded(task: Optional[str]) -> list[AudioCppModel]:
    from core.inference import audio_cpp_files

    out: list[AudioCppModel] = []
    root = _hub_cache()
    try:
        repo_dirs = [p for p in root.iterdir() if p.name.startswith("models--") and p.is_dir()]
    except OSError:
        return out
    for repo_dir in repo_dirs:
        repo_id = repo_dir.name[len("models--") :].replace("--", "/")
        snapshots = snapshot_dirs(repo_id, root)
        if not snapshots:
            continue
        ids: list[str] = []
        if repo_id.lower() == AUDIO_CPP_REPO.lower():
            folders = set()
            for snapshot in snapshots:
                try:
                    folders.update(p.name for p in snapshot.iterdir() if p.is_dir())
                except OSError:
                    continue
            ids = [f"{AUDIO_CPP_REPO}/{name}" for name in sorted(folders)]
        else:
            first_gguf = None
            for snapshot in snapshots:
                # The root or one folder down, where audio.cpp repos keep their GGUFs; a full walk of
                # every cached checkpoint on each status poll would cost more than it finds.
                first_gguf = next(
                    iter(sorted([*snapshot.glob("*.gguf"), *snapshot.glob("*/*.gguf")])), None
                )
                if first_gguf is not None:
                    break
            if first_gguf is None:
                continue
            header = read_local_header(first_gguf)
            if header is None or not header.is_audio_cpp:
                continue
            ids = [repo_id]
        for model_id in ids:
            try:
                model = resolve(model_id, network = False)
            except Exception:  # noqa: BLE001 - one unreadable repo never hides the rest
                model = None
            if model is None or (model.unsupported and not model.family):
                continue
            if task is not None and model.task != task:
                continue
            ready = next(
                (v for v in model.variants if audio_cpp_files.cached_files(model.with_variant(v))),
                None,
            )
            if ready is not None:
                out.append(model.with_variant(ready))
    return out
