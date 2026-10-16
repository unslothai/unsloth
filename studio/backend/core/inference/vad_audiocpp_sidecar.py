# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Voice activity detection on audio.cpp, for voice mode.

The browser's loudness gate decides when an utterance ended; this answers whether the clip it
cut actually holds a voice (a cough, a hum and a chair do not), through one offline VAD session
on its own audio.cpp server. One model resident at a time, same as the dictation sidecar.
"""

from __future__ import annotations

import json
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from core.inference import audio_cpp_files
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AudioCppModel,
    AudioCppModelError,
    parse_identifier,
    require_runnable,
    resolve,
    split_variant_ref,
)
from core.inference.audio_cpp_server import (
    AudioCppServer,
    AudioCppUnavailableError,
    find_audio_cpp_server_binary,
)
from loggers import get_logger

logger = get_logger(__name__)

DEFAULT_VAD_MODEL = f"{AUDIO_CPP_REPO}/PulseVAD-GGUF"
_RUN_PATH = "/v1/tasks/run"
_DETECT_TIMEOUT_SECONDS = 30.0
# Voice mode sends clips of a few seconds; anything longer is not a turn.
MAX_CLIP_BYTES = 16 * 1024 * 1024


class VadError(Exception):
    """A request-shaped failure (bad model id, not downloaded); the route maps it to a 4xx."""


class VadModelNotDownloadedError(VadError):
    pass


class VadUnavailableError(VadError):
    """The audio runtime cannot run VAD here."""


def resolve_vad_model(model: Optional[str], *, network: bool = False) -> AudioCppModel:
    identifier = (model or "").strip() or DEFAULT_VAD_MODEL
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
    """Shape the server's ``segments`` (sample offsets) into seconds, plus the voiced total."""
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
    """With UNSLOTH_VAD_DEBUG_DIR set, keep every checked clip and its verdict for offline tuning."""
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
        stem.with_suffix(".json").write_text(json.dumps({"raw": payload, "result": result}, indent = 1))
    except OSError:
        pass


class AudioCppVadSidecar:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._server: Optional[AudioCppServer] = None
        self._model: Optional[AudioCppModel] = None
        self._loading: Optional[str] = None

    @property
    def loaded_model(self) -> Optional[str]:
        with self._lock:
            return self._model.id if self._model and self._server_alive() else None

    @property
    def is_loading(self) -> bool:
        return self._loading is not None

    def status(self) -> dict:
        with self._lock:
            return {
                "available": find_audio_cpp_server_binary() is not None,
                "loaded_model": self.loaded_model,
                "loading": self._loading,
                "default_model": DEFAULT_VAD_MODEL,
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

    def load(self, model: Optional[str], hf_token: Optional[str] = None) -> str:
        entry = resolve_vad_model(model, network = False)
        with self._lock:
            if self._model is not None and self._model.id == entry.id and self._server_alive():
                return entry.id
            try:
                path = audio_cpp_files.materialize(entry)
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
                    server = AudioCppServer.start(entry, path)
                except AudioCppUnavailableError as exc:
                    raise VadUnavailableError(str(exc)) from exc
                self._server = server
                self._model = entry
            finally:
                self._loading = None
        return entry.id

    def download(self, model: Optional[str], hf_token: Optional[str] = None) -> str:
        from core.inference import audio_cpp_backend

        entry = resolve_vad_model(model, network = True)
        audio_cpp_backend.AudioCppBackend._download_missing(entry, hf_token)
        return entry.id

    def detect(self, wav_bytes: bytes, model: Optional[str] = None) -> dict:
        if len(wav_bytes) > MAX_CLIP_BYTES:
            raise VadError("Audio clip is too long for a speech check.")
        # Browsers capture at the device rate (48 kHz on a Mac); the detectors take 16 kHz mono only.
        from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
        from core.inference.stt_sidecar import _decode_audio_bounded

        wav_bytes = _pcm_to_wav_bytes(_decode_audio_bounded(wav_bytes))
        model_id = self.load(model)
        with self._lock:
            server = self._server
            if server is None or not self._server_alive():
                raise VadUnavailableError("The speech detector is not running.")
            with _temp_wav(wav_bytes) as path:
                body = {"model": server.model_id, "request": {"audio": str(path)}}
                _, data = server.post_json(_RUN_PATH, body, timeout = _DETECT_TIMEOUT_SECONDS)
                payload = json.loads(data.decode("utf-8"))
                result = segments_from_result(payload, _wav_rate(path))
        result["model"] = model_id
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
