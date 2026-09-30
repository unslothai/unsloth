# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worker backend for audio.cpp speech and music models in the main audio slot.

Selected by the inference worker in place of ``NativeAudioBackend`` when the model
is an audio.cpp TTS or music GGUF, so loading, auto-switching, idle eviction,
cancellation and the Audio page, read-aloud and ``/v1/audio/speech`` all work
through the machinery every other main-slot audio model uses. The weights run in
an ``audiocpp_server`` child; this class only downloads, starts, proxies and stops.

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
from core.inference.audio_cpp_models import (
    AudioCppModel,
    AudioCppModelError,
    require_runnable,
    resolve,
    validate_options,
)
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


def model_info_fields(model: AudioCppModel) -> dict[str, Any]:
    """What status reports for a loaded audio.cpp model, beyond the common audio fields."""
    return {
        "audio_family": model.family,
        "audio_options": [dict(option) for option in model.options],
        "gguf_variant": model.variant.key,
    }


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
        self._model: Optional[AudioCppModel] = None
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
        model = getattr(config, "audio_cpp", None) or resolve(
            model_name, getattr(config, "gguf_variant", None), hf_token
        )
        if model is None:
            raise RuntimeError(f"'{model_name}' is not an audio.cpp GGUF.")
        try:
            require_runnable(model, "tts")
        except AudioCppModelError as exc:
            raise RuntimeError(str(exc)) from exc
        if gpu_ids is not None and len(gpu_ids) > 1:
            raise RuntimeError(
                "Audio GGUF models run on a single GPU; multi-GPU sharding is not supported."
            )
        if (
            model_name in self.models
            and self._model == model
            and self._server is not None
            and self._server.alive()
        ):
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
            self._model = model
            self.models = {
                model_name: {
                    "is_audio": True,
                    "audio_type": model.audio_type,
                    "has_audio_input": False,
                    "model_path": model.id,
                    # No token window: speech length is bounded by the server, music by duration.
                    "context_length": 0,
                    **model_info_fields(model),
                    "audio_cpp_backend": self._server.backend if self._server else None,
                }
            }
            self.active_model_name = model_name
            return True
        finally:
            self.loading_models.discard(model_name)

    def _ensure_downloaded(self, model: AudioCppModel, hf_token: Optional[str]) -> None:
        missing = audio_cpp_files.missing_files(model)
        if not missing:
            return
        from huggingface_hub import hf_hub_download

        from utils.hf_cache_settings import active_hf_hub_cache

        cache_dir = str(active_hf_hub_cache())
        for path, _size in missing:
            logger.info("audio.cpp: downloading %s from %s", path, model.repo_id)
            hf_hub_download(
                model.repo_id,
                path,
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
            if self._server is None or not self._server.alive() or self._server.model != model:
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
        audio_options: Optional[dict] = None,
    ) -> Tuple[bytes, int]:
        del top_k, min_p, repetition_penalty, use_adapter
        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        model = self._model
        if model is None:
            raise RuntimeError("No active audio model")
        _raise_if_cancelled(cancel_event)
        options = validate_options(model.options, audio_options)
        server = self._running_server(model, cancel_event)
        try:
            if model.task == "music":
                wav = self._generate_music(
                    server, model, text, instructions, max_new_tokens, seed, options, cancel_event
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
                    options,
                    cancel_event,
                )
        except AudioCppRequestCancelledError:
            self._restart_after_cancel()
            _raise_if_cancelled(cancel_event)
            raise
        except AudioCppRequestError as exc:
            raise RuntimeError(f"The audio runtime could not generate audio: {exc.detail}") from exc
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
        options: dict,
        cancel_event,
    ) -> bytes:
        defaults = dict(model.request_defaults)
        default_options = dict(defaults.pop("options", None) or {})
        body: dict[str, Any] = {"model": server.model_id, "input": text, **defaults}
        # Sampling stays with each family's own defaults unless the user set it for this model. Studio's
        # generic speech temperature is tuned for the token-codec models; handed to audio.cpp's families it
        # can keep one from ever emitting its stop token (MOSS-TTS-Nano samples at 1.5 and runs to its length
        # cap at 0.6).
        del temperature, top_p
        chosen = dict(options)
        voice = chosen.pop("voice", None)
        if voice:
            body["voice"] = voice
        default_options.update(chosen)
        requested: dict[str, Any] = {}
        if instructions and str(instructions).strip():
            # Voice-design and style-capable families read the description as the instruction.
            requested["instruct"] = str(instructions).strip()
        if language and str(language).strip():
            requested["language"] = str(language).strip()
        merged = {**default_options, **requested}
        if merged:
            body["options"] = merged
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
            # The request's own description and language are hints; the package's defaults and the
            # user's options are not.
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
        options: dict,
        cancel_event,
    ) -> bytes:
        # The Audio page's music form sends the description as instructions and the lyrics as text (the MiniMax
        # convention).
        description = str(instructions or "").strip()
        lyrics = str(text or "").strip()
        seconds = _music_seconds(max_new_tokens)
        request: dict[str, Any] = {}
        request_options: dict[str, Any] = dict(options)
        if model.family == "minimax_music3":
            if not lyrics:
                raise RuntimeError("MiniMax Music 3 needs lyrics.")
            # The caption is the input; lyrics and the frame budget are options (the request fields too,
            # which the task route maps onto the same options).
            request.update(
                {"text": description or lyrics, "lyrics": lyrics, "duration_seconds": seconds}
            )
            request_options.update({"lyrics": lyrics, "duration_sec": seconds})
        elif model.family == "yue2":
            if not description:
                raise RuntimeError("YuE2 needs a style description.")
            request["text"] = lyrics or description
            request_options["style"] = description
            if lyrics:
                request_options["lyrics"] = lyrics
            # YuE2's length is its semantic token budget at 25 frames per second (default 9000, six
            # minutes), and its default floor of 200 frames would outlast a short request.
            frames = int(round(seconds * 25))
            request_options["semantic_max_tokens"] = frames
            request_options["semantic_min_tokens"] = min(200, frames)
        else:
            # A lone prompt with no description is the description.
            if not description:
                description, lyrics = lyrics, ""
            request["text"] = description
            request["duration_seconds"] = seconds
            if lyrics and model.family != "stable_audio":
                request["lyrics"] = lyrics
        if request_options:
            request["options"] = request_options
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
            self._model = None
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
        raise RuntimeError("The audio runtime returned an unreadable music response.") from exc
    for candidate in _audio_candidates(payload):
        try:
            decoded = base64.b64decode(candidate, validate = False)
        except Exception:  # noqa: BLE001 - not base64
            continue
        if decoded[:4] == b"RIFF":
            return decoded
    raise RuntimeError("The audio runtime returned no audio for the music request.")


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
