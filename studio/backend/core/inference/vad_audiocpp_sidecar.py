# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.inference.audio_cpp_clip_sidecar import (
    CLIP_SAMPLE_RATE,
    AudioCppClipSidecar,
    ClipModelNotDownloadedError,
    ClipError,
    ClipUnavailableError,
    resolve_hub_model,
)
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile
from core.inference.audio_cpp_server import find_audio_cpp_server_binary, runtime_release_key
from loggers import get_logger

logger = get_logger(__name__)

# Voiced time, in ms, a clip needs to count as speech: a whole utterance, and the short probe of a
# barge-in. These are the Hub detectors' bars (PulseVAD counts 100 ms windows).
_HUB_BARS = (300.0, 200.0)


@dataclass(frozen = True)
class BuiltinDetector:
    """A detector whose weights are not on the Hub: they ship with Unsloth Studio, or with the audio
    runtime's own asset bundle. These are small CPU networks, so they never take the GPU."""

    id: str
    family: str
    display_name: str
    bars: tuple[float, float] = _HUB_BARS
    # The first audio runtime release that runs the family, where an older supported one does not.
    min_runtime: Optional[tuple[int, int, int]] = None


BUILTIN_PREFIX = "builtin/"
BUILTIN_DETECTORS: dict[str, BuiltinDetector] = {
    d.id: d
    for d in (
        # Earshot times speech in 16 ms frames, so the same short word reads 220-285 ms where
        # PulseVAD says 300: its bars sit lower to keep the same clips.
        BuiltinDetector(
            "builtin/earshot-vad", "earshot_vad", "Earshot VAD", (200.0, 150.0), (0, 9, 3)
        ),
        # Their weights come only with the runtime's asset bundle, so a runtime that has them runs them.
        BuiltinDetector("builtin/silero-vad", "silero_vad", "Silero VAD"),
        BuiltinDetector("builtin/marblenet-vad", "marblenet_vad", "MarbleNet VAD"),
    )
}
# Scores 16 ms frames in about 2 ms per clip on the CPU, with nothing to download.
DEFAULT_VAD_MODEL = "builtin/earshot-vad"
_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "audio"
_WEIGHT_SUFFIXES = (".safetensors", ".gguf")


# The names the routes and callers know these by.
VadError = ClipError
VadUnavailableError = ClipUnavailableError


def _canonical(model: Optional[str]) -> str:
    """The id as the table keys it; no id means the default detector."""
    return ((model or "").strip() or DEFAULT_VAD_MODEL).lower()


def is_builtin_id(model: Optional[str]) -> bool:
    """Whether ``model`` is in the built-in namespace, known or not: such an id is never a Hub repo."""
    return _canonical(model).startswith(BUILTIN_PREFIX)


def builtin_detector(model: Optional[str]) -> Optional[BuiltinDetector]:
    return BUILTIN_DETECTORS.get(_canonical(model))


def _weights(detector: BuiltinDetector, binary: Optional[str]) -> list[Path]:
    """The detector's weight files: Studio's own copy, else the runtime's asset bundle."""
    roots = [_ASSETS / detector.family]
    if binary:
        runtime = Path(binary).resolve().parent
        roots.append(runtime / "assets" / "framework" / "models" / detector.family)
    for root in roots:
        try:
            files = sorted(
                p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
            )
        except OSError:
            continue
        if files:
            return files
    return []


def speech_bars(model_id: str) -> dict:
    """How much voiced time the detector ``resolve_vad_model`` named needs before a clip counts as speech."""
    detector = BUILTIN_DETECTORS.get(model_id)
    utterance_ms, barge_ms = detector.bars if detector else _HUB_BARS
    return {"utterance_ms": utterance_ms, "barge_ms": barge_ms}


def detectors() -> list[dict]:
    """The built-in detectors and whether this install can run each. Hub detectors are in the audio catalogue."""
    listed = []
    for d in BUILTIN_DETECTORS.values():
        try:
            _builtin_model(d)
            available = True
        except VadUnavailableError:
            available = False
        listed.append({"id": d.id, "label": d.display_name, "available": available})
    return listed


def _builtin_model(detector: BuiltinDetector) -> AudioCppModel:
    binary = find_audio_cpp_server_binary()
    # An unknown release (a custom build) is not an old one.
    release = runtime_release_key(binary) if binary else None
    if detector.min_runtime and release is not None and release < detector.min_runtime:
        raise VadUnavailableError(
            f"{detector.display_name} needs a newer audio runtime. Update the audio runtime from "
            "the update banner, or run `unsloth studio update`."
        )
    weights = _weights(detector, binary)
    if not weights:
        raise VadUnavailableError(f"{detector.display_name} is not included in this install.")
    files = tuple(RepoFile(p.name, p.stat().st_size) for p in weights)
    # A package variant: the runtime loads the folder, not one file in it.
    variant = AudioCppVariant("builtin", files, "")
    folder = weights[0].parent
    return AudioCppModel(
        id = detector.id,
        repo_id = None,
        folder = "",
        display_name = detector.display_name,
        family = detector.family,
        task = "vad",
        server_task = "vad",
        variant = variant,
        variants = (variant,),
        default_variant = variant.key,
        local_path = str(folder),
    )


def resolve_vad_model(
    model: Optional[str],
    *,
    network: bool = False,
    hf_token: Optional[str] = None,
) -> AudioCppModel:
    identifier = (model or "").strip() or DEFAULT_VAD_MODEL
    builtin = builtin_detector(identifier)
    if builtin is not None:
        return _builtin_model(builtin)
    if is_builtin_id(identifier):
        # Shaped like a Hub repo id, but it names nothing there: never look it up.
        raise VadError(f"'{model}' is not a detector Unsloth Studio includes.")
    return resolve_hub_model(
        identifier, "vad", "detect speech with", network = network, hf_token = hf_token
    )


def segments_from_result(payload: dict, fallback_rate: int) -> dict:
    rate = int(payload.get("sample_rate") or fallback_rate or 16000)
    segments = []
    speech_ms = 0.0
    for raw in payload.get("segments") or []:
        start = int(raw.get("start_sample", 0))
        end = int(raw.get("end_sample", start))
        if end <= start:
            continue
        segments.append(
            {
                "start_s": start / rate,
                "end_s": end / rate,
                "confidence": float(raw.get("confidence", 1.0)),
            }
        )
        speech_ms += (end - start) * 1000.0 / rate
    return {"segments": segments, "speech_ms": round(speech_ms, 1), "sample_rate": rate}


def _debug_dump(wav_bytes: bytes, payload: dict, result: dict) -> None:
    import os
    import time

    folder = os.environ.get("UNSLOTH_VAD_DEBUG_DIR")
    if not folder:
        return
    try:
        root = Path(folder)
        root.mkdir(parents = True, exist_ok = True)
        stem = root / f"clip-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
        stem.with_suffix(".wav").write_bytes(wav_bytes)
        stem.with_suffix(".json").write_text(
            json.dumps({"raw": payload, "result": result}, indent = 1), encoding = "utf-8"
        )
    except OSError:
        pass


class AudioCppVadSidecar(AudioCppClipSidecar):
    label = "speech detector"

    def _resolve(
        self, model: Optional[str], *, network: bool, hf_token: Optional[str] = None
    ) -> AudioCppModel:
        return resolve_vad_model(model, network = network, hf_token = hf_token)

    def _cpu_only(self, entry: AudioCppModel) -> bool:
        return entry.id in BUILTIN_DETECTORS

    def _known_cpu_id(self, model: Optional[str]) -> Optional[str]:
        builtin = builtin_detector(model)
        return builtin.id if builtin else None

    def _weights_path(self, entry: AudioCppModel):
        # A built-in's weights are a folder of this install, not a Hub download.
        return entry.local_path if entry.id in BUILTIN_DETECTORS else super()._weights_path(entry)

    def download(
        self,
        model: Optional[str],
        hf_token: Optional[str] = None,
    ) -> str:
        builtin = builtin_detector(model)
        if builtin is not None:
            # Nothing to fetch, but an install that cannot run it still says so.
            return _builtin_model(builtin).id
        return super().download(model, hf_token)

    def status(self) -> dict:
        return {**self.state(), "default_model": DEFAULT_VAD_MODEL, "detectors": detectors()}

    def detect(
        self,
        audio: bytes,
        model: Optional[str] = None,
    ) -> dict:
        model_id, payload, wav_bytes = self.run_clip(audio, model)
        result = segments_from_result(payload, CLIP_SAMPLE_RATE)
        result["model"] = model_id
        result["bars"] = speech_bars(model_id)
        _debug_dump(wav_bytes, payload, result)
        return result


_sidecar: Optional[AudioCppVadSidecar] = None
_sidecar_lock = threading.Lock()


def get_audio_cpp_vad_sidecar() -> AudioCppVadSidecar:
    global _sidecar
    with _sidecar_lock:
        if _sidecar is None:
            _sidecar = AudioCppVadSidecar()
        return _sidecar
