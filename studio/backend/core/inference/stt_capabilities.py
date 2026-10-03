# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""STT model capabilities, answered offline (cache, else name); never raises."""

from __future__ import annotations

from typing import Optional

from core.inference.stt_details import ALWAYS_TIMESTAMPED, ON_REQUEST_TIMESTAMPS
from loggers import get_logger

logger = get_logger(__name__)

_SPEAKER_FAMILIES = frozenset({"moss_transcribe_diarize", "vibevoice_asr"})
_ALIGNER_SIZE_BYTES = 1_129_966_496

_ENGINE_ALIASES = {
    "": "transformers",
    "transformers": "transformers",
    "whisper": "transformers",
    "gguf": "gguf",
    "ggml": "gguf",
    "whisper_cpp": "gguf",
    "whisper.cpp": "gguf",
    "mtmd": "mtmd",
    "llama_cpp": "mtmd",
    "llama.cpp": "mtmd",
    "audiocpp": "audiocpp",
    "audio_cpp": "audiocpp",
    "audio.cpp": "audiocpp",
}


def label_speakers(ids) -> list[dict]:
    """``[{id, label}]`` with labels "Speaker 1", "Speaker 2"… in the order the ids first speak."""
    unique = list(dict.fromkeys(str(i) for i in ids or () if str(i).strip()))
    return [{"id": sid, "label": f"Speaker {n}"} for n, sid in enumerate(unique, start = 1)]


def _is_mtmd(model: str) -> bool:
    from core.inference.stt_mtmd_sidecar import MTMD_STT_MODELS
    low = model.lower()
    return model in MTMD_STT_MODELS or any(m.repo.lower() == low for m in MTMD_STT_MODELS.values())


def _engine_for(model: str, engine: Optional[str]) -> Optional[str]:
    """The engine that would serve ``model``: the one asked for, else the one its id requires."""
    # The llama.cpp Qwen3-ASR repos are not audio.cpp models, whatever engine was named.
    if model and _is_mtmd(model):
        return "mtmd"
    if engine is not None and str(engine).strip():
        return _ENGINE_ALIASES.get(str(engine).strip().lower())
    if not model:
        return "transformers"
    from core.inference.audio_cpp_models import looks_like_audio_cpp, resolve

    if looks_like_audio_cpp(model):
        return "audiocpp"
    try:
        found = resolve(model, network = False)
    except Exception:  # noqa: BLE001 - unreadable from the cache: the default engine
        found = None
    return "audiocpp" if found is not None and found.task == "asr" else "transformers"


def _audio_cpp_family(model: str) -> Optional[str]:
    """The family of a downloaded audio.cpp ASR model, else the one its name suggests."""
    from core.inference.audio_cpp_models import (
        family_from_names,
        family_policy,
        parse_identifier,
        resolve,
        split_variant_ref,
    )

    base, _variant = split_variant_ref(model)
    ref = parse_identifier(base)
    if ref is None:
        return None
    try:
        found = resolve(model, network = False)
    except Exception:  # noqa: BLE001 - the name still says something
        found = None
    if found is not None:
        return found.family if found.task == "asr" and not found.unsupported else None
    family = family_from_names((ref.id, ref.folder or ""))
    if family is None:
        return None
    policy = family_policy(family)
    return family if policy.task == "asr" and not policy.unsupported else None


def _aligner_state() -> dict:
    from core.inference import audio_cpp_files
    from core.inference.audio_cpp_models import resolve
    from core.inference.stt_audiocpp_sidecar import QWEN3_ALIGNER

    try:
        aligner = resolve(QWEN3_ALIGNER.id, QWEN3_ALIGNER.variant, network = False)
    except Exception:  # noqa: BLE001 - not in the cache
        aligner = None
    if aligner is None:
        return {"downloaded": False, "size_bytes": _ALIGNER_SIZE_BYTES}
    return {
        "downloaded": audio_cpp_files.is_downloaded(aligner),
        "size_bytes": aligner.size_bytes or _ALIGNER_SIZE_BYTES,
    }


def capabilities_for(model: Optional[str], engine: Optional[str]) -> dict:
    """``timestamps``: "on_request" | "always" | "unsupported"; ``aligner`` only for "on_request"."""
    model = str(model or "").strip()
    result = {
        "engine": None,
        "family": None,
        "timestamps": "unsupported",
        "speakers": False,
        "aligner": None,
        "cpu_only": False,
    }
    try:
        resolved_engine = _engine_for(model, engine)
        result["engine"] = resolved_engine
        if resolved_engine != "audiocpp" or not model:
            return result
        family = _audio_cpp_family(model)
        if family is None:
            return result
        from core.inference.audio_cpp_models import CPU_ONLY_FAMILIES

        result["family"] = family
        result["cpu_only"] = family in CPU_ONLY_FAMILIES
        result["speakers"] = family in _SPEAKER_FAMILIES
        if family in ALWAYS_TIMESTAMPED:
            result["timestamps"] = "always"
        elif family in ON_REQUEST_TIMESTAMPS:
            result["timestamps"] = "on_request"
            result["aligner"] = _aligner_state()
    except Exception as exc:  # noqa: BLE001 - a capability probe never fails the page
        logger.info("STT capabilities for %r unavailable (%s)", model, type(exc).__name__)
        result.update(timestamps = "unsupported", speakers = False, aligner = None)
    return result
