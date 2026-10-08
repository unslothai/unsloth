# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worker backend for audio.cpp speech, music and separation models in the main audio slot.

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
import os
import re
import tempfile
import threading
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Tuple

from core.inference import audio_cpp_convert, audio_cpp_files
from core.inference import audio_cpp_music as cm
from core.inference.audio_cpp_models import (
    MUSIC_MAX_VARIATIONS,
    AudioCppModel,
    AudioCppModelError,
    CloneSpec,
    forget,
    option_matches,
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
from core.inference.audio_cpp_music import music_rules
from core.inference.audio_task_outputs import task_outputs, wav_header
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


_inflight_lock = threading.Lock()
_inflight: dict[tuple[str, str, str], tuple[threading.Event, list]] = {}


def _download_or_cancel(key, download, cancel_event) -> None:
    """hf_hub_download has no cancel hook: it runs on its own thread and the caller stops waiting on
    cancel. The thread still finishes into the Hub cache; a retry meanwhile joins it."""
    with _inflight_lock:
        flight = _inflight.get(key)
        if flight is None:
            flight = _inflight[key] = (threading.Event(), [])
            done, failure = flight

            def run():
                try:
                    download()
                except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                    failure.append(exc)
                finally:
                    with _inflight_lock:
                        _inflight.pop(key, None)
                    done.set()

            threading.Thread(target = run, name = "audio-cpp-download", daemon = True).start()
    done, failure = flight
    while not done.wait(0.2):
        if cancel_event.is_set():
            raise AudioCppRequestCancelledError("Request cancelled.")
    if failure:
        raise failure[0]


def _wav_seconds(path: str) -> float:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        return round(w.getnframes() / rate, 3) if rate else 0.0


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


# Qwen3-TTS answers "unsupported language: en"; it wants the language's English name.
_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
_SPEAKER_LINE_RE = re.compile(r"^\s*Speaker\s*\d+\s*:", re.IGNORECASE | re.MULTILINE)
# F5's usable speed range.
_MIN_SPEED, _MAX_SPEED = 0.5, 2.0


def model_info_fields(model: AudioCppModel) -> dict[str, Any]:
    """What status reports for a loaded audio.cpp model, beyond the common audio fields."""
    return {
        "audio_family": model.family,
        "audio_options": [dict(option) for option in model.options],
        "gguf_variant": model.variant.key,
        "audio_workflows": list(model.workflows),
        "audio_reference_text": model.clone.reference_text if model.clone else None,
        "audio_required_inputs": list(model.required_inputs),
        "audio_clone": clone_rules(model),
        "audio_edit": edit_rules(model),
        "audio_music": music_rules(model),
        "audio_options_by_workflow": (
            {"convert": [dict(option) for option in model.convert_options]}
            if model.convert is not None
            else None
        ),
        "audio_workflow_tasks": audio_cpp_convert.workflow_tasks(model),
        "audio_convert": audio_cpp_convert.convert_caps(model),
        "audio_convert_rules": model.convert
        and {"source_rate": model.convert.source_rate, "target_rate": model.convert.target_rate},
        **server_runtime_fields(model, None),
    }


def server_runtime_fields(model: AudioCppModel, running: Optional[AudioCppModel]) -> dict[str, Any]:
    """Task and Seed-VC route of the running server (``model``'s own when none runs)."""
    served = running or model
    return {
        "audio_server_task": served.server_task,
        "audio_convert_route": audio_cpp_convert.served_route(served),
    }


def clone_rules(model: AudioCppModel) -> Optional[dict[str, Any]]:
    clone = model.clone
    if clone is None:
        return None
    return {
        "reference_text": clone.reference_text,
        "reference_text_waived": [
            [name, list(values)] for name, values in clone.reference_text_waived
        ],
        "emotion_audio": clone.emotion_audio,
    }


def edit_rules(model: AudioCppModel) -> Optional[dict[str, Any]]:
    edit = getattr(model, "edit", None)
    if edit is None:
        return None
    return {
        "style": edit.style,
        "delivery": edit.delivery_template is not None,
        "max_changes": edit.max_changes,
    }


def edit_options(model: AudioCppModel) -> tuple[dict, ...]:
    claims = set(model.edit.claims) if model.edit is not None else set()
    return tuple(option for option in model.options if option["name"] not in claims)


def edit_request_bodies(
    model: AudioCppModel,
    text: str,
    reference_text: Optional[str],
    edit: Optional[dict],
    options: dict,
    seed: Optional[int],
    model_id: str,
) -> list[dict[str, Any]]:
    """The ``/v1/tasks/run`` bodies an edit posts, in order, without the recording field: the
    first call reads the source, each later one the previous call's output."""
    spec = model.edit
    if spec is None:
        raise RuntimeError(f"{model.display_name} cannot edit speech.")
    edit = edit or {}
    advanced = {name: _option_string(value) for name, value in options.items()}
    top: dict[str, Any] = {"model": model_id}
    if seed is not None:
        top["seed"] = str(int(seed))
    if spec.style == "markup":
        markup = str(edit.get("markup") or "")
        if not markup.strip():
            raise RuntimeError(f"{model.display_name} needs the marked-up changes.")
        return [{**top, "text": markup, "options": {"template_name": spec.template, **advanced}}]
    if spec.style == "sentence":
        body = {**top, "target_text": text}
        if spec.route:
            body["route"] = spec.route
        original = str(reference_text or "").strip()
        if original:
            body["reference_text"] = original
        if advanced:
            body["options"] = advanced
        return [body]
    if edit.get("mode") == "delivery":
        from core.inference.audio_edit import delivery_instructions

        template = spec.delivery_template
        if template is None:
            raise RuntimeError("Delivery changes need FireRedAudio.")
        instructions = delivery_instructions(edit.get("speed"), edit.get("pitch_steps"))
    else:
        template = spec.template
        instructions = [str(item) for item in edit.get("instructions") or ()]
    if not instructions:
        raise RuntimeError("Change at least one word.")
    return [
        {**top, "options": {"template_name": template, "instruction": line, **advanced}}
        for line in instructions
    ]


def language_name(language: Optional[str]) -> Optional[str]:
    """``en`` -> ``English``; None for Auto or an unknown code."""
    text = str(language or "").strip()
    if not text or text.lower() == "auto":
        return None
    code = text.lower().replace("_", "-").split("-", 1)[0]
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    for name in _LANGUAGE_NAMES.values():
        if name.lower() == text.lower():
            return name
    logger.info("audio.cpp: no language name for %r; sending Auto", text)
    return None


def speech_input(model: AudioCppModel, text: str) -> str:
    """VibeVoice refuses a script with no ``Speaker N:`` line."""
    if model.family == "vibevoice" and not _SPEAKER_LINE_RE.search(text or ""):
        return f"Speaker 1: {text}"
    return text


def _option_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if abs(value) < 1e15 else repr(value)
    return str(value)


class AudioCppBackend:
    """One-model backend proxying to an ``audiocpp_server`` child."""

    def __init__(self, device_preference: Optional[str] = None) -> None:
        from core.inference.audio_device import audio_device_forces_cpu

        self.device_preference = device_preference
        self._force_cpu = audio_device_forces_cpu(device_preference)
        self.device = "cpu" if self._force_cpu else "gpu"
        self.models: dict[str, dict[str, Any]] = {}
        self.active_model_name: Optional[str] = None
        self.loading_models: set[str] = set()
        self._model: Optional[AudioCppModel] = None
        self._server: Optional[AudioCppServer] = None
        self._server_lock = threading.RLock()
        # ``model_options`` is not part of model equality: a changed overlap is only seen here.
        self._served_session: dict = {}
        # Request-raised session options (Stable Audio max_batch); cleared when another model loads.
        self._session_overrides: dict[str, str] = {}
        self._status_patch: Optional[dict[str, Any]] = None

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
            raise RuntimeError(f"'{model_name}' is not an audio GGUF.")
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
            self._session_overrides = {}
            self._status_patch = None
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
        self._download_missing(model, hf_token)
        for companion in model.companions:
            companion_model = _resolve_companion(model, companion, hf_token, network = True)
            if self._download_missing(companion_model, hf_token):
                forget(companion.id)

    @staticmethod
    def _download_missing(
        model: AudioCppModel,
        hf_token: Optional[str],
        cancel_event = None,
    ) -> bool:
        missing = audio_cpp_files.missing_files(model)
        if not missing:
            return False
        from huggingface_hub import hf_hub_download

        from utils.hf_cache_settings import active_hf_hub_cache

        cache_dir = str(active_hf_hub_cache())
        for path, _size in missing:
            if cancel_event is not None and cancel_event.is_set():
                raise AudioCppRequestCancelledError("Request cancelled.")
            logger.info("audio.cpp: downloading %s from %s", path, model.repo_id)

            def fetch(path = path):
                hf_hub_download(
                    model.repo_id,
                    path,
                    token = hf_token or None,
                    cache_dir = cache_dir,
                )

            if cancel_event is None:
                fetch()
            else:
                # Cache root in the key: a retry after the cache moved must not join the old transfer.
                _download_or_cancel((cache_dir, model.repo_id, path), fetch, cancel_event)
        return True

    def _start_server(
        self,
        model: AudioCppModel,
        cancel_event = None,
    ) -> None:
        with self._server_lock:
            self._stop_server_locked()
            model_path = audio_cpp_files.materialize(model)
            served = _with_session_overrides(_with_companions(model), self._session_overrides)
            try:
                self._server = AudioCppServer.start(
                    served, model_path, force_cpu = self._force_cpu, cancel_event = cancel_event
                )
            except AudioCppStartCancelledError as exc:
                _raise_if_cancelled(cancel_event)
                raise RuntimeError(str(exc)) from exc
            self._served_session = dict((served.model_options or {}).get("session_options") or {})
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
                or self._server.model != model
                # Seed-VC's route: model equality leaves model_options out.
                or _request_defaults(self._server.model) != _request_defaults(model)
            ):
                logger.info("audio.cpp: (re)starting the server for %s", model.id)
                try:
                    self._start_server(model, cancel_event)
                finally:
                    self._record_runtime()
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
        workflow: Optional[str] = None,
        audio_inputs: Optional[dict] = None,
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
        convert: Optional[dict] = None,
        edit: Optional[dict] = None,
        music: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        del top_k, min_p, repetition_penalty, use_adapter
        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        model = self._model
        if model is None:
            raise RuntimeError("No active audio model")
        _raise_if_cancelled(cancel_event)
        if model.task == "sep":
            raise RuntimeError(f"{model.display_name} separates audio; open Separate.")
        if workflow == "music" and music is not None:
            options = validate_options(model.options, audio_options)
            return self._run_music(
                model,
                music,
                (audio_inputs or {}).get("source"),
                options,
                seed,
                output_dir,
                cancel_event,
            )
        editing = workflow == "edit"
        converting = workflow == "convert"
        # Speak in a saved voice sends a reference with workflow "speak": a clone too.
        cloning = not (editing or converting) and (workflow == "clone" or bool(audio_inputs))
        if editing and model.edit is None:
            raise RuntimeError(f"{model.display_name} cannot edit speech.")
        if converting:
            if model.convert is None:
                raise RuntimeError(f"{model.display_name} cannot convert a voice.")
            convert = convert or {}
            mode = str(convert.get("mode") or "speech")
            try:
                served, options = audio_cpp_convert.served_model(
                    model, mode, validate_options(model.convert_options, audio_options)
                )
                request = audio_cpp_convert.convert_request(
                    model,
                    mode = mode,
                    source = (audio_inputs or {}).get("source"),
                    target = (audio_inputs or {}).get("target"),
                    voice = convert.get("voice"),
                    pitch = convert.get("pitch"),
                    pitch_auto = bool(convert.get("pitch_auto")),
                    style = str(convert.get("style") or "source"),
                    source_text = convert.get("source_text"),
                    options = {name: _option_string(value) for name, value in options.items()},
                    seed = seed,
                )
            except audio_cpp_convert.ConvertRequestError as exc:
                raise RuntimeError(str(exc)) from exc
        elif editing:
            options = validate_options(edit_options(model), audio_options)
            served = _served_for_edit(model)
        else:
            if cloning and model.clone is None:
                raise RuntimeError(f"{model.display_name} cannot clone a voice.")
            served = model
            options = validate_options(
                model.clone_options if cloning else model.options, audio_options
            )
        server = self._running_server(served, cancel_event)
        try:
            if converting:
                ctype, data = server.post_json(
                    "/v1/tasks/run",
                    {"model": server.model_id, "request": request},
                    timeout = _GENERATE_TIMEOUT_SECONDS,
                    cancel_event = cancel_event,
                )
                wav = _audio_from_task_response(ctype, data)
            elif editing:
                wav = self._generate_edit(
                    server,
                    model,
                    text,
                    audio_inputs or {},
                    reference_text,
                    edit,
                    options,
                    seed,
                    cancel_event,
                )
            elif cloning:
                wav = self._generate_clone(
                    server,
                    model,
                    text,
                    audio_inputs or {},
                    reference_text = reference_text,
                    instructions = instructions,
                    language = language,
                    speed = speed,
                    seed = seed,
                    options = options,
                    cancel_event = cancel_event,
                )
            elif model.task == "music":
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
            from core.inference.audio_errors import AudioRuntimeError
            raise AudioRuntimeError(
                f"The audio runtime could not generate audio: {exc.detail}", status = exc.status
            ) from exc
        _raise_if_cancelled(cancel_event)
        return wav, _wav_sample_rate(wav)

    def runtime_fields(self) -> dict[str, Any]:
        model = self._model
        if model is None:
            return {}
        with self._server_lock:
            server = self._server
            running = server.model if server is not None and server.alive() else None
        return server_runtime_fields(model, running)

    def _record_runtime(self) -> None:
        entry = self.models.get(self.active_model_name or "")
        if entry is not None:
            entry.update(self.runtime_fields())

    def separate_audio(
        self,
        source_path: str,
        output_dir: str,
        options: Optional[dict] = None,
        cancel_event = None,
    ) -> list[dict[str, Any]]:
        """Split a 44.1 kHz WAV into stems under ``output_dir``; [] when none decodes."""
        from core.inference.audio_cpp_outputs import SeparationOutputError, extract_named_outputs

        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        model = self._model
        if model is None:
            raise RuntimeError("No active audio model")
        spec = model.separation
        if spec is None:
            raise RuntimeError(f"{model.display_name} cannot separate audio.")
        _raise_if_cancelled(cancel_event)
        session = dict((model.model_options or {}).get("session_options") or {})
        overlap = (options or {}).get("num_overlap")
        if spec.overlap_option and overlap is not None:
            session[spec.overlap_option] = str(max(1, min(8, int(overlap))))
        with self._server_lock:
            if self._server is None or not self._server.alive() or self._served_session != session:
                logger.info("audio.cpp: (re)starting the server for %s with %s", model.id, session)
                self._start_server(
                    replace(
                        model, model_options = {**model.model_options, "session_options": session}
                    ),
                    cancel_event,
                )
            server = self._server
        response_path = Path(output_dir) / ".response.json"
        try:
            # The separation families refuse any other key.
            server.post_json_to_file(
                "/v1/tasks/run",
                {"model": server.model_id, "audio": str(source_path)},
                response_path,
                timeout = _GENERATE_TIMEOUT_SECONDS,
                cancel_event = cancel_event,
            )
            _raise_if_cancelled(cancel_event)
            outputs = extract_named_outputs(response_path, output_dir)
        except AudioCppRequestCancelledError:
            self._restart_after_cancel()
            _raise_if_cancelled(cancel_event)
            raise
        except AudioCppRequestError as exc:
            from core.inference.audio_errors import AudioRuntimeError
            raise AudioRuntimeError(
                f"The audio runtime could not separate the track: {exc.detail}", status = exc.status
            ) from exc
        except SeparationOutputError as exc:
            logger.warning("audio.cpp: %s answered without decodable stems: %s", model.id, exc)
            return []
        finally:
            try:
                response_path.unlink(missing_ok = True)
            except OSError:
                pass
        _raise_if_cancelled(cancel_event)
        return outputs

    def _restart_after_cancel(self) -> None:
        # The server keeps computing the abandoned request; stopping it frees the GPU now and the next
        # request starts a fresh one.
        with self._server_lock:
            self._stop_server_locked()
            self._record_runtime()

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
        body: dict[str, Any] = {
            "model": server.model_id,
            "input": speech_input(model, text),
            **defaults,
        }
        # Family defaults win: Studio's generic temperature can stop one emitting its stop token (MOSS-TTS-Nano).
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
    def _generate_clone(
        server: AudioCppServer,
        model: AudioCppModel,
        text: str,
        audio_inputs: dict,
        *,
        reference_text: Optional[str],
        instructions: Optional[str],
        language: Optional[str],
        speed: Optional[float],
        seed: Optional[int],
        options: dict,
        cancel_event,
    ) -> bytes:
        """Speak ``text`` in the voice of ``audio_inputs["reference"]`` (a server-local WAV path).

        An emotion clip has no field on the speech endpoint, so that request goes to /v1/tasks/run
        with the clip as top-level ``audio``."""
        clone = model.clone or CloneSpec()
        reference = audio_inputs.get("reference")
        if not reference:
            raise RuntimeError(f"{model.display_name} needs a reference clip to clone.")
        emotion = audio_inputs.get("emotion") if clone.emotion_audio else None
        request_options = {name: _option_string(value) for name, value in options.items()}
        body: dict[str, Any] = {"model": server.model_id}
        spoken = speech_input(model, text)
        if emotion:
            body["text"] = spoken
        else:
            body["input"] = spoken
        body["voice_ref"] = str(reference)
        if emotion:
            body["audio"] = str(emotion)
        transcript = str(reference_text or "").strip()
        if (
            transcript
            and clone.reference_text != "unused"
            and not option_matches(clone.reference_text_dropped, options)
        ):
            body["reference_text"] = transcript
        if language and str(language).strip():
            chosen = language_name(language) if clone.language_names else str(language).strip()
            if chosen:
                body["language"] = chosen
        if clone.speed and speed is not None:
            body["speed"] = max(_MIN_SPEED, min(_MAX_SPEED, float(speed)))
        if clone.instructions and instructions and str(instructions).strip():
            body["instructions"] = str(instructions).strip()
        if request_options:
            body["options"] = request_options
        if seed is not None:
            body["seed"] = str(int(seed))
        if emotion:
            ctype, data = server.post_json(
                "/v1/tasks/run",
                body,
                timeout = _GENERATE_TIMEOUT_SECONDS,
                cancel_event = cancel_event,
            )
            return _audio_from_task_response(ctype, data)
        _ctype, data = server.post_json(
            "/v1/audio/speech",
            body,
            timeout = _GENERATE_TIMEOUT_SECONDS,
            cancel_event = cancel_event,
        )
        return data

    @staticmethod
    def _generate_edit(
        server: AudioCppServer,
        model: AudioCppModel,
        text: str,
        audio_inputs: dict,
        reference_text: Optional[str],
        edit: Optional[dict],
        options: dict,
        seed: Optional[int],
        cancel_event,
    ) -> bytes:
        """Edit ``audio_inputs["source"]`` (a server-local WAV). A chain hands each output to the
        next call through a temporary WAV. The runtime's answer ``text`` is not the edited
        transcript (FireRedAudio), so it is never read."""
        source = audio_inputs.get("source")
        if not source:
            raise RuntimeError(f"{model.display_name} needs a recording to edit.")
        bodies = edit_request_bodies(
            model, text, reference_text, edit, options, seed, server.model_id
        )
        with tempfile.TemporaryDirectory(prefix = "unsloth-audio-edit-") as scratch:
            path = str(source)
            for index, body in enumerate(bodies):
                _raise_if_cancelled(cancel_event)
                if index:
                    path = os.path.join(scratch, f"step{index}.wav")
                    with open(path, "wb") as handle:
                        handle.write(wav)
                ctype, data = server.post_json(
                    "/v1/tasks/run",
                    {**body, model.edit.source_field: path},
                    timeout = _GENERATE_TIMEOUT_SECONDS,
                    cancel_event = cancel_event,
                )
                wav = _audio_from_task_response(ctype, data)
        return wav

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
        try:
            request = cm.legacy_song_request(
                model, text, instructions, _music_seconds(max_new_tokens), options, seed
            )
        except cm.MusicRequestError as exc:
            raise RuntimeError(str(exc)) from exc
        ctype, data = server.post_json(
            "/v1/tasks/run",
            {"model": server.model_id, "request": request},
            timeout = _GENERATE_TIMEOUT_SECONDS,
            cancel_event = cancel_event,
        )
        return _audio_from_task_response(ctype, data)

    def take_status_patch(self) -> Optional[dict[str, Any]]:
        patch, self._status_patch = self._status_patch, None
        return patch

    def _max_batch(self) -> int:
        try:
            return max(1, int(self._session_overrides.get(_MAX_BATCH_OPTION, "1")))
        except ValueError:
            return 1

    def _server_for_batch(self, model: AudioCppModel, batch: int, cancel_event) -> AudioCppServer:
        """The running server, restarted once with max_batch raised to the cap, so later batches
        never reload."""
        with self._server_lock:
            if batch <= self._max_batch():
                return self._running_server(model, cancel_event)
            logger.info(
                "audio.cpp: reloading %s with %s=%d for %d variations",
                model.id,
                _MAX_BATCH_OPTION,
                MUSIC_MAX_VARIATIONS,
                batch,
            )
            previous = dict(self._session_overrides)
            self._session_overrides[_MAX_BATCH_OPTION] = str(MUSIC_MAX_VARIATIONS)
            try:
                self._start_server(model, cancel_event)
            except BaseException:
                self._session_overrides = previous
                raise
            rules = music_rules(model, self._max_batch())
            if self.active_model_name in self.models:
                self.models[self.active_model_name]["audio_music"] = rules
            self._status_patch = {"audio_music": rules}
            return self._server

    def _run_music(
        self,
        model: AudioCppModel,
        music: dict,
        source: Optional[str],
        options: dict,
        seed: Optional[int],
        output_dir: Optional[str],
        cancel_event,
    ) -> Tuple[bytes, int]:
        """Every output goes to ``output_dir`` with an ``outputs.json`` manifest; the first is
        returned."""
        if model.task != "music" or model.music is None:
            raise RuntimeError(f"{model.display_name} does not make music in the Music studio.")
        mode_id = str(music.get("mode") or "song")
        try:
            mode = cm.song_mode(model, mode_id)
        except cm.MusicRequestError as exc:
            raise RuntimeError(str(exc)) from exc
        variations = int(music.get("variations") or 1) if mode_id != "edit" else 1
        if variations > 1 and (not mode.variations or variations > MUSIC_MAX_VARIATIONS):
            raise RuntimeError(
                f"{model.display_name} makes at most "
                f"{MUSIC_MAX_VARIATIONS if mode.variations else 1} variation(s) at a time."
            )
        if seed is None and model.music.fixed_seed:
            seed = cm.random_seed()
        batch = variations if mode.variations == "batch" else 1
        if mode_id == "edit":
            if not source:
                raise RuntimeError("Add a clip to edit.")
            seconds = _wav_seconds(source)
        else:
            seconds = cm.clamp_seconds(mode, music.get("duration_s"))
        try:
            if mode_id == "edit":
                requests = [
                    cm.edit_request(
                        model,
                        text = music.get("text") or "",
                        edit = music.get("edit") or {},
                        source = source,
                        source_seconds = seconds,
                        duration_s = music.get("duration_s"),
                        options = options,
                        seed = seed,
                    )
                ]
            else:
                instrumental = bool(music.get("instrumental")) and mode.instrumental != "never"
                takes = 1 if batch > 1 else variations
                requests = [
                    cm.song_request(
                        model,
                        description = music.get("text") or "",
                        lyrics = "" if mode.lyrics == "unused" else (music.get("lyrics") or ""),
                        seconds = seconds,
                        options = options,
                        seed = cm.take_seed(seed, index),
                        instrumental = instrumental or mode.instrumental == "always",
                        batch = batch,
                    )
                    for index in range(takes)
                ]
        except cm.MusicRequestError as exc:
            raise RuntimeError(str(exc)) from exc
        server = self._server_for_batch(model, batch, cancel_event)
        # The route sizes the wait from the full work (extend, continue length); the orchestrator
        # outlasts that same budget.
        timeout = float(music.get("timeout_s") or 0) or cm.timeout_seconds(
            model.family, seconds, variations, server.backend == "cpu"
        )
        outputs: list[tuple[str, bytes, Optional[int]]] = []
        try:
            for index, request in enumerate(requests):
                _raise_if_cancelled(cancel_event)
                ctype, data = server.post_json(
                    "/v1/tasks/run",
                    {"model": server.model_id, "request": request},
                    timeout = timeout,
                    cancel_event = cancel_event,
                )
                take_seed = cm.take_seed(seed, index)
                for output_id, wav in task_outputs(ctype, data):
                    name = output_id if len(requests) == 1 else f"take_{index}"
                    outputs.append((name, wav, take_seed))
        except AudioCppRequestCancelledError:
            self._restart_after_cancel()
            _raise_if_cancelled(cancel_event)
            raise
        except AudioCppRequestError as exc:
            from core.inference.audio_errors import AudioRuntimeError
            raise AudioRuntimeError(
                f"The audio runtime could not generate audio: {exc.detail}", status = exc.status
            ) from exc
        _raise_if_cancelled(cancel_event)
        if output_dir:
            _write_outputs(output_dir, outputs)
        first = outputs[0][1]
        return first, _wav_sample_rate(first)

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


def _resolve_companion(
    model: AudioCppModel,
    companion,
    hf_token: Optional[str] = None,
    *,
    network: bool = True,
) -> AudioCppModel:
    found = resolve(companion.id, companion.variant, hf_token, network = network)
    if found is None:
        name = companion.id.rsplit("/", 1)[-1]
        raise RuntimeError(f"{model.display_name} needs {name}, which Studio could not find.")
    return found


def _served_for_edit(model: AudioCppModel) -> AudioCppModel:
    edit = model.edit
    if edit is None or edit.server_task == model.server_task:
        return model
    return replace(model, server_task = edit.server_task)


def _with_companions(model: AudioCppModel) -> AudioCppModel:
    """``model`` with each companion's served path in its session options (MioTTS's codec)."""
    if not model.companions:
        return model
    session = dict((model.model_options or {}).get("session_options") or {})
    for companion in model.companions:
        companion_model = _resolve_companion(model, companion, network = False)
        session[companion.session_option] = audio_cpp_files.materialize(companion_model)
    return replace(model, model_options = {**model.model_options, "session_options": session})


def _request_defaults(model: AudioCppModel) -> Any:
    return (model.model_options or {}).get("default_request_options")


_MAX_BATCH_OPTION = "stable_audio.max_batch"
_MANIFEST = "outputs.json"


def _with_session_overrides(model: AudioCppModel, overrides: dict[str, str]) -> AudioCppModel:
    if not overrides:
        return model
    session = dict((model.model_options or {}).get("session_options") or {})
    session.update(overrides)
    return replace(model, model_options = {**model.model_options, "session_options": session})


def _write_outputs(output_dir: str, outputs) -> None:
    """Bare file names only: the route refuses any path outside ``output_dir``."""
    directory = Path(output_dir)
    directory.mkdir(parents = True, exist_ok = True)
    manifest = []
    for index, (output_id, wav, seed) in enumerate(outputs):
        name = f"{index:02d}.wav"
        (directory / name).write_bytes(wav)
        rate, duration = wav_header(wav)
        manifest.append(
            {
                "id": str(output_id),
                "file": name,
                "sample_rate": rate,
                "duration_s": duration,
                "seed": seed,
            }
        )
    (directory / _MANIFEST).write_text(json.dumps(manifest), encoding = "utf-8")


def _audio_from_task_response(content_type: str, data: bytes) -> bytes:
    """WAV bytes from ``/v1/tasks/run``: raw audio, or JSON carrying base64 audio."""
    try:
        return task_outputs(content_type, data)[0][1]
    except RuntimeError:
        pass
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
