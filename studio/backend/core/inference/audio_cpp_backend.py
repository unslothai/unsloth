# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worker backend for audio.cpp speech and music models in the main audio slot.

Selected by the inference worker in place of ``NativeAudioBackend`` when the model
is a curated audio.cpp TTS or music model, so loading, auto-switching, idle
eviction, cancellation and the Audio page, read-aloud and ``/v1/audio/speech`` all
work through the machinery every other main-slot audio model uses. The weights run
in an ``audiocpp_server`` child; this class only downloads, starts, proxies and stops.

A cancelled request cannot interrupt GPU work already queued inside the server, and
the server would make the next request wait for it (minutes, for music). So a cancel
stops the server and the next request starts it again from the warm file cache.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from typing import Any, Optional, Tuple

from core.inference import audio_cpp_files
from core.inference.audio_cpp_models import AudioCppModel, lookup
from core.inference.audio_cpp_server import (
    AudioCppRequestCancelledError,
    AudioCppRequestError,
    AudioCppServer,
    AudioCppStartCancelledError,
)
from loggers import get_logger
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory

logger = get_logger(__name__)

# Upper bound for one request; music generation at long durations takes minutes.
_GENERATE_TIMEOUT_SECONDS = 3600.0
# Frames of a 30-second clip, the MiniMax convention the Audio page's music budget speaks in.
_DEFAULT_MUSIC_SECONDS = 30.0
_MAX_MUSIC_SECONDS = 240.0


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        from core.inference.audio_errors import AudioGenerationCancelledError
        raise AudioGenerationCancelledError("Audio generation cancelled")


def _wav_sample_rate(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return int(w.getframerate())


def _music_seconds(max_new_tokens: Optional[int]) -> float:
    """Map the shared speech-token budget onto a clip length.

    The Audio page sends MiniMax's 25-frames-per-second budget for music; the same
    number here means the same length, so one duration control drives every model.
    """
    try:
        frames = int(max_new_tokens or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        return _DEFAULT_MUSIC_SECONDS
    return max(5.0, min(_MAX_MUSIC_SECONDS, frames / 25.0))


class AudioCppBackend:
    """One-model backend proxying to an ``audiocpp_server`` child."""

    def __init__(self, device_preference: Optional[str] = None) -> None:
        from core.inference.audio_device import audio_device_forces_cpu

        self.device_preference = device_preference
        self._force_cpu = audio_device_forces_cpu(device_preference)
        self.device = "cpu" if self._force_cpu else "audio.cpp"
        self.models: dict[str, dict[str, Any]] = {}
        self.active_model_name: Optional[str] = None
        self.loading_models: set[str] = set()
        self._server: Optional[AudioCppServer] = None
        self._server_lock = threading.RLock()

    # Loading

    @_invalidates_gpu_memory("audio load")
    def load_model(
        self,
        config,
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = False,
        gpu_ids: Optional[list[int]] = None,
        **_ignored,
    ) -> bool:
        del max_seq_length, dtype, load_in_4bit, trust_remote_code
        model_name = config.identifier
        model = lookup(model_name)
        if model is None or model.task not in ("tts", "music"):
            raise RuntimeError(f"'{model_name}' is not a curated audio.cpp speech or music model.")
        if gpu_ids is not None and len(gpu_ids) > 1:
            raise RuntimeError(
                "audio.cpp models run on a single GPU; multi-GPU sharding is not supported."
            )
        if model_name in self.models and self._server is not None and self._server.alive():
            self.active_model_name = model_name
            return True
        # Before any download: a runtime that cannot serve this model must fail fast, not after gigabytes.
        from core.inference.audio_cpp_server import model_runtime_problem

        problem = model_runtime_problem(model)
        if problem:
            raise RuntimeError(problem)
        self.loading_models.add(model_name)
        try:
            self._ensure_downloaded(model, hf_token)
            self._start_server(model)
            self.models = {
                model_name: {
                    "is_audio": True,
                    "audio_type": model.audio_type,
                    "has_audio_input": False,
                    "model_path": model.id,
                    # No token window: speech length is bounded by the server, music by duration.
                    "context_length": 0,
                    "audio_cpp_family": model.family,
                    "audio_cpp_backend": self._server.backend if self._server else None,
                }
            }
            self.active_model_name = model_name
            return True
        finally:
            self.loading_models.discard(model_name)

    def _ensure_downloaded(self, model: AudioCppModel, hf_token: Optional[str]) -> None:
        if audio_cpp_files.is_downloaded(model):
            return
        from huggingface_hub import hf_hub_download

        from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AUDIO_CPP_REVISION
        from utils.hf_cache_settings import active_hf_hub_cache

        cache_dir = str(active_hf_hub_cache())
        for path, _size in audio_cpp_files.expand_repo_files(model, hf_token):
            logger.info("audio.cpp: downloading %s", path)
            hf_hub_download(
                AUDIO_CPP_REPO,
                path,
                revision = AUDIO_CPP_REVISION,
                token = hf_token or None,
                cache_dir = cache_dir,
            )

    def _start_server(
        self,
        model: AudioCppModel,
        cancel_event = None,
    ) -> None:
        with self._server_lock:
            self._stop_server_locked()
            model_path = audio_cpp_files.materialize(model)
            try:
                self._server = AudioCppServer.start(
                    model, model_path, force_cpu = self._force_cpu, cancel_event = cancel_event
                )
            except AudioCppStartCancelledError as exc:
                _raise_if_cancelled(cancel_event)
                raise RuntimeError(str(exc)) from exc
            self.device = "cpu" if self._server.backend == "cpu" else self._server.backend

    def _stop_server_locked(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.stop()

    def _running_server(self, model: AudioCppModel, cancel_event) -> AudioCppServer:
        with self._server_lock:
            if (
                self._server is None
                or not self._server.alive()
                or self._server.model.id != model.id
            ):
                logger.info("audio.cpp: (re)starting the server for %s", model.id)
                self._start_server(model, cancel_event)
            return self._server

    # Generation

    def generate_audio_response(
        self,
        text: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        use_adapter = None,
        cancel_event = None,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        del top_k, min_p, repetition_penalty, use_adapter
        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        model = lookup(self.active_model_name)
        if model is None:
            raise RuntimeError("No active audio model")
        _raise_if_cancelled(cancel_event)
        server = self._running_server(model, cancel_event)
        try:
            if model.task == "music":
                wav = self._generate_music(
                    server, model, text, instructions, max_new_tokens, seed, cancel_event
                )
            else:
                wav = self._generate_speech(
                    server,
                    model,
                    text,
                    instructions,
                    language,
                    temperature,
                    top_p,
                    seed,
                    cancel_event,
                )
        except AudioCppRequestCancelledError:
            self._restart_after_cancel()
            _raise_if_cancelled(cancel_event)
            raise
        except AudioCppRequestError as exc:
            raise RuntimeError(f"audio.cpp could not generate audio: {exc.detail}") from exc
        _raise_if_cancelled(cancel_event)
        return wav, _wav_sample_rate(wav)

    def _restart_after_cancel(self) -> None:
        # The server keeps computing the abandoned request; stopping it frees the GPU now and the next
        # request starts a fresh one.
        with self._server_lock:
            self._stop_server_locked()

    @staticmethod
    def _generate_speech(
        server: AudioCppServer,
        model: AudioCppModel,
        text: str,
        instructions: Optional[str],
        language: Optional[str],
        temperature: float,
        top_p: float,
        seed: Optional[int],
        cancel_event,
    ) -> bytes:
        defaults = dict(model.request_defaults)
        default_options = dict(defaults.pop("options", None) or {})
        body: dict[str, Any] = {"model": server.model_id, "input": text, **defaults}
        # Sampling stays with each family's own defaults. Studio's generic speech temperature is tuned for the
        # token-codec models; handed to audio.cpp's families it can keep one from ever emitting its stop token
        # (MOSS-TTS-Nano samples at 1.5 and runs to its length cap at 0.6).
        del temperature, top_p
        requested: dict[str, Any] = {}
        if instructions and str(instructions).strip():
            # Voice-design and style-capable families read the description as the instruction.
            requested["instruct"] = str(instructions).strip()
        if language and str(language).strip():
            requested["language"] = str(language).strip()
        options = {**default_options, **requested}
        if options:
            body["options"] = options
        if seed is not None:
            body["seed"] = str(int(seed))
        try:
            _ctype, data = server.post_json(
                "/v1/audio/speech",
                body,
                timeout = _GENERATE_TIMEOUT_SECONDS,
                cancel_event = cancel_event,
            )
        except AudioCppRequestError as exc:
            # 503 is a busy model, not a refused option. The server answers an option a family lacks with a
            # 4xx or a 500 depending on where the family validates it, so both count as a refusal here.
            if not requested or exc.status == 503 or not 400 <= exc.status < 600:
                raise
            # The request's own description and language are hints; the package's defaults are not.
            logger.info(
                "audio.cpp: %s rejected %s (%s); retrying without them",
                model.family,
                sorted(requested),
                exc.detail,
            )
            if default_options:
                body["options"] = default_options
            else:
                body.pop("options", None)
            _ctype, data = server.post_json(
                "/v1/audio/speech",
                body,
                timeout = _GENERATE_TIMEOUT_SECONDS,
                cancel_event = cancel_event,
            )
        return data

    @staticmethod
    def _generate_music(
        server: AudioCppServer,
        model: AudioCppModel,
        text: str,
        instructions: Optional[str],
        max_new_tokens: Optional[int],
        seed: Optional[int],
        cancel_event,
    ) -> bytes:
        # The Audio page's music form sends the description as instructions and the lyrics as text (the MiniMax
        # convention). A lone prompt with no description is the description.
        description = str(instructions or "").strip()
        lyrics = str(text or "").strip()
        if not description:
            description, lyrics = lyrics, ""
        request: dict[str, Any] = {
            "text": description,
            "duration_seconds": _music_seconds(max_new_tokens),
        }
        if lyrics and model.family != "stable_audio":
            request["lyrics"] = lyrics
        if seed is not None:
            request["seed"] = str(int(seed))
        ctype, data = server.post_json(
            "/v1/tasks/run",
            {"model": server.model_id, "request": request},
            timeout = _GENERATE_TIMEOUT_SECONDS,
            cancel_event = cancel_event,
        )
        return _audio_from_task_response(ctype, data)

    # Unloading

    @_invalidates_gpu_memory("audio unload")
    def unload_model(self, model_name: str) -> bool:
        self.models.pop(model_name, None)
        if self.active_model_name == model_name:
            self.active_model_name = None
        with self._server_lock:
            self._stop_server_locked()
        return True

    def reset_generation_state(self, caller_cancel_event = None) -> None:
        del caller_cancel_event


def _audio_from_task_response(content_type: str, data: bytes) -> bytes:
    """WAV bytes from ``/v1/tasks/run``: raw audio, or JSON carrying base64 audio."""
    if content_type.startswith("audio/") or data[:4] == b"RIFF":
        return data
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError("audio.cpp returned an unreadable music response.") from exc
    for candidate in _audio_candidates(payload):
        try:
            decoded = base64.b64decode(candidate, validate = False)
        except Exception:  # noqa: BLE001 - not base64
            continue
        if decoded[:4] == b"RIFF":
            return decoded
    raise RuntimeError("audio.cpp returned no audio for the music request.")


def _audio_candidates(node: Any):
    if isinstance(node, dict):
        for key in ("audio", "wav", "data", "b64_json", "audio_base64"):
            value = node.get(key)
            if isinstance(value, str):
                yield value.split(",", 1)[-1]
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _audio_candidates(value)
    elif isinstance(node, list):
        for item in node:
            yield from _audio_candidates(item)
