# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import tempfile
import threading
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from core.inference import audio_cpp_files
from core.inference.audio_cpp_models import (
    AudioCppModel,
    AudioCppModelError,
    AudioCppVariant,
    RepoFile,
    forget,
    parse_identifier,
    require_runnable,
    resolve,
    split_variant_ref,
)
from core.inference.audio_cpp_server import (
    AudioCppServer,
    AudioCppUnavailableError,
    find_audio_cpp_server_binary,
    runtime_release_key,
)
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
_RUN_PATH = "/v1/tasks/run"
_DETECT_TIMEOUT_SECONDS = 30.0
MAX_CLIP_BYTES = 16 * 1024 * 1024


class VadError(Exception):
    """A request-shaped failure (bad model id, not downloaded); the route maps it to a 4xx."""


class VadModelNotDownloadedError(VadError):
    pass


class VadUnavailableError(VadError):
    """The audio runtime cannot run VAD here."""


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


def resolve_vad_model(model: Optional[str], *, network: bool = False) -> AudioCppModel:
    identifier = (model or "").strip() or DEFAULT_VAD_MODEL
    builtin = builtin_detector(identifier)
    if builtin is not None:
        return _builtin_model(builtin)
    if is_builtin_id(identifier):
        # Shaped like a Hub repo id, but it names nothing there: never look it up.
        raise VadError(f"'{model}' is not a detector Unsloth Studio includes.")
    base, variant = split_variant_ref(identifier)
    if parse_identifier(base) is None:
        raise VadError(f"'{model}' is not an audio GGUF the audio runtime can detect speech with.")
    found = resolve(base, variant, network = network)
    if found is None and not network:
        raise VadModelNotDownloadedError(f"VAD model '{base}' is not downloaded yet.")
    if found is None:
        raise VadError(f"'{model}' is not an audio GGUF the audio runtime can detect speech with.")
    try:
        require_runnable(found, "vad")
    except AudioCppModelError as exc:
        raise VadError(str(exc)) from exc
    return found


@contextmanager
def _temp_wav(wav_bytes: bytes) -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(prefix = "unsloth-vad-", suffix = ".wav", delete = False) as f:
        f.write(wav_bytes)
        path = Path(f.name)
    try:
        yield path
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _wav_rate(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate()
    except (wave.Error, OSError):
        return 16000


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


class AudioCppVadSidecar:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._server: Optional[AudioCppServer] = None
        self._model: Optional[AudioCppModel] = None
        self._loading: Optional[str] = None
        self._on_cpu = False

    @property
    def loaded_model(self) -> Optional[str]:
        with self._lock:
            return self._model.id if self._model and self._server_alive() else None

    @property
    def on_cpu(self) -> bool:
        """Whether the live server runs on the CPU (it holds no VRAM through training)."""
        with self._lock:
            return self._on_cpu

    @property
    def is_loading(self) -> bool:
        return self._loading is not None

    def status(self) -> dict:
        # No lock: a clip check holds it for its whole run, and the picker must not wait behind one.
        model = self._model
        return {
            "available": find_audio_cpp_server_binary() is not None,
            "loaded_model": model.id if model is not None and self._server_alive() else None,
            "loading": self._loading,
            "default_model": DEFAULT_VAD_MODEL,
            "detectors": detectors(),
        }

    def _server_alive(self) -> bool:
        server = self._server
        return server is not None and server.process.poll() is None

    def _release_locked(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:  # noqa: BLE001 - a dying child is already gone
                pass
        self._server = None
        self._model = None

    def unload(self) -> bool:
        with self._lock:
            had = self._server is not None
            self._release_locked()
            return had

    def load(
        self,
        model: Optional[str],
        hf_token: Optional[str] = None,
    ) -> str:
        builtin = builtin_detector(model)
        # Every clip comes through here. A live built-in has nothing to re-check: it is always on the CPU.
        if builtin is not None and self.loaded_model == builtin.id:
            return builtin.id
        entry = resolve_vad_model(model, network = False)
        from core.inference.stt_sidecar import _training_active

        with self._lock:
            # Under the lock training admission's release takes, so a load cannot start on a GPU admission just freed.
            on_cpu = builtin is not None or _training_active()
            if (
                self._model is not None
                and self._model.id == entry.id
                and self._server_alive()
                and self._on_cpu == on_cpu
            ):
                return entry.id
            try:
                path = (
                    entry.local_path if builtin is not None else audio_cpp_files.materialize(entry)
                )
            except FileNotFoundError:
                raise VadModelNotDownloadedError(
                    f"VAD model '{entry.display_name}' is not downloaded yet."
                ) from None
            except AudioCppUnavailableError as exc:
                raise VadUnavailableError(str(exc)) from exc
            self._loading = entry.id
            try:
                self._release_locked()
                try:
                    server = AudioCppServer.start(entry, path, force_cpu = on_cpu)
                except AudioCppUnavailableError as exc:
                    raise VadUnavailableError(str(exc)) from exc
                self._server = server
                self._model = entry
                self._on_cpu = on_cpu
            finally:
                self._loading = None
        return entry.id

    def download(
        self,
        model: Optional[str],
        hf_token: Optional[str] = None,
    ) -> str:
        from core.inference import audio_cpp_backend

        builtin = builtin_detector(model)
        if builtin is not None:
            return builtin.id
        entry = resolve_vad_model(model, network = True)
        audio_cpp_backend.AudioCppBackend._download_missing(entry, hf_token)
        # The load right after reads the cache-only resolution, which still remembers the miss.
        forget(entry.id)
        return entry.id

    def detect(
        self,
        wav_bytes: bytes,
        model: Optional[str] = None,
    ) -> dict:
        if len(wav_bytes) > MAX_CLIP_BYTES:
            raise VadError("Audio clip is too long for a speech check.")
        from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
        from core.inference.stt_sidecar import _decode_audio_bounded

        wav_bytes = _pcm_to_wav_bytes(_decode_audio_bounded(wav_bytes))
        # Loaded under the same hold as the run: another model's load cannot swap the server between.
        with self._lock:
            model_id = self.load(model)
            server = self._server
            if server is None or not self._server_alive():
                raise VadUnavailableError("The speech detector is not running.")
            with _temp_wav(wav_bytes) as path:
                body = {"model": server.model_id, "request": {"audio": str(path)}}
                _, data = server.post_json(_RUN_PATH, body, timeout = _DETECT_TIMEOUT_SECONDS)
                payload = json.loads(data.decode("utf-8"))
                result = segments_from_result(payload, _wav_rate(path))
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
