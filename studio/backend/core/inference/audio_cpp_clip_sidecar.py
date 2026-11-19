# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from core.inference import audio_cpp_files
from core.inference.audio_cpp_models import (
    AudioCppModel,
    AudioCppModelError,
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
)

_RUN_PATH = "/v1/tasks/run"
_RUN_TIMEOUT_SECONDS = 30.0
MAX_CLIP_BYTES = 16 * 1024 * 1024
CLIP_SAMPLE_RATE = 16000
_MAX_THREADS = 4


class ClipError(Exception):
    """Request-shaped failure: the route maps it to 4xx."""


class ClipModelNotDownloadedError(ClipError):
    pass


class ClipUnavailableError(ClipError):
    """The audio runtime cannot run the task here."""


def resolve_hub_model(
    identifier: str,
    task: str,
    purpose: str,
    *,
    network: bool,
    hf_token: Optional[str] = None,
) -> AudioCppModel:
    refusal = f"'{identifier}' is not an audio GGUF the audio runtime can {purpose}."
    base, variant = split_variant_ref(identifier)
    if parse_identifier(base) is None:
        raise ClipError(refusal)
    found = resolve(base, variant, hf_token, network = network)
    if found is None and not network:
        raise ClipModelNotDownloadedError(f"'{base}' is not downloaded yet.")
    if found is None:
        raise ClipError(refusal)
    try:
        require_runnable(found, task)
    except AudioCppModelError as exc:
        raise ClipError(str(exc)) from exc
    return found


@contextmanager
def _temp_wav(wav_bytes: bytes) -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(prefix = "unsloth-clip-", suffix = ".wav", delete = False) as f:
        f.write(wav_bytes)
        path = Path(f.name)
    try:
        yield path
    finally:
        try:
            path.unlink()
        except OSError:
            pass


class AudioCppClipSidecar:
    label = "audio helper"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._server: Optional[AudioCppServer] = None
        self._model: Optional[AudioCppModel] = None
        self._loading: Optional[str] = None
        self._on_cpu = False

    def _resolve(
        self, model: Optional[str], *, network: bool, hf_token: Optional[str] = None
    ) -> AudioCppModel:
        raise NotImplementedError

    def _cpu_only(self, entry: AudioCppModel) -> bool:
        return False

    def _known_cpu_id(self, model: Optional[str]) -> Optional[str]:
        return None

    def _weights_path(self, entry: AudioCppModel):
        return audio_cpp_files.materialize(entry)

    @property
    def loaded_model(self) -> Optional[str]:
        with self._lock:
            return self._model.id if self._model and self._server_alive() else None

    @property
    def on_cpu(self) -> bool:
        with self._lock:
            return self._on_cpu

    @property
    def is_loading(self) -> bool:
        return self._loading is not None

    def state(self) -> dict:
        # No lock: a status read must not wait behind a clip run.
        model = self._model
        return {
            "available": find_audio_cpp_server_binary() is not None,
            "loaded_model": model.id if model is not None and self._server_alive() else None,
            "loading": self._loading,
        }

    def _server_alive(self) -> bool:
        server = self._server
        return server is not None and server.process.poll() is None

    def _release_locked(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:  # noqa: BLE001 - child may already be dead
                pass
        self._server = None
        self._model = None

    def unload(self, keep: Optional[Callable[[Optional[str]], bool]] = None) -> bool:
        with self._lock:
            # keep() runs under the load lock: a model swapped in meanwhile is never stopped.
            if keep is not None and keep(self.loaded_model):
                return False
            had = self._server is not None
            self._release_locked()
            return had

    def load(
        self,
        model: Optional[str],
        hf_token: Optional[str] = None,
    ) -> str:
        known = self._known_cpu_id(model)
        with self._lock:
            if (
                known
                and self._model is not None
                and self._model.id == known
                and self._server_alive()
            ):
                return known
        entry = self._resolve(model, network = False)
        from core.inference.stt_sidecar import _training_active

        with self._lock:
            # Under the lock that training admission's release takes: a freed GPU is seen.
            on_cpu = self._cpu_only(entry) or _training_active()
            if (
                self._model is not None
                and self._model.id == entry.id
                and self._server_alive()
                and self._on_cpu == on_cpu
            ):
                return entry.id
            try:
                path = self._weights_path(entry)
            except FileNotFoundError:
                raise ClipModelNotDownloadedError(
                    f"The {self.label} '{entry.display_name}' is not downloaded yet."
                ) from None
            except AudioCppUnavailableError as exc:
                raise ClipUnavailableError(str(exc)) from exc
            self._loading = entry.id
            try:
                self._release_locked()
                try:
                    server = AudioCppServer.start(
                        entry, path, force_cpu = on_cpu, max_threads = _MAX_THREADS
                    )
                except AudioCppUnavailableError as exc:
                    raise ClipUnavailableError(str(exc)) from exc
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

        entry = self._resolve(model, network = True, hf_token = hf_token)
        audio_cpp_backend.AudioCppBackend._download_missing(entry, hf_token)
        # Drop the cached miss, or the load right after still sees it.
        forget(entry.id)
        return entry.id

    def run_clip(
        self,
        audio: bytes,
        model: Optional[str] = None,
        *,
        tail_seconds: Optional[float] = None,
    ) -> tuple[str, dict, bytes]:
        if len(audio) > MAX_CLIP_BYTES:
            raise ClipError("Audio clip is too long to check.")
        from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
        from core.inference.stt_sidecar import _decode_audio_bounded

        samples = _decode_audio_bounded(audio)
        if tail_seconds:
            samples = samples[-int(tail_seconds * CLIP_SAMPLE_RATE) :]
        wav_bytes = _pcm_to_wav_bytes(samples)
        with self._lock:
            model_id = self.load(model)
            server = self._server
            if server is None or not self._server_alive():
                raise ClipUnavailableError(f"The {self.label} is not running.")
            with _temp_wav(wav_bytes) as path:
                body = {"model": server.model_id, "request": {"audio": str(path)}}
                _, data = server.post_json(_RUN_PATH, body, timeout = _RUN_TIMEOUT_SECONDS)
                payload = json.loads(data.decode("utf-8"))
        return model_id, payload, wav_bytes
