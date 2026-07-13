# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Standalone speech-to-text (STT) sidecar for dictation.

Loads a faster-whisper (CTranslate2) model in the backend process, separate
from the chat model that runs in the inference subprocess. This lets a user
dictate into any chat model, including text-only ones, without evicting it.

faster-whisper is torch-free, so this works on GGUF-only installs too. Model
weights download lazily on first use and memory is released after dictation.
"""

from __future__ import annotations

import gc
import io
import threading
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Curated, fast, accurate models. Keys are the ids the API/UI use; values are
# the faster-whisper model names (auto-downloaded from the HF hub, then cached).
STT_MODELS: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "distil-large-v3": "distil-large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "large-v3": "large-v3",
}
DEFAULT_STT_MODEL = "base"
ENGLISH_ONLY_STT_MODELS = frozenset({"distil-large-v3"})

# Bound decoded audio so a crafted upload cannot exhaust memory. Callers also
# cap the encoded bytes; this bounds the decoded PCM length.
_MAX_AUDIO_SECONDS = 30 * 60
_TARGET_SAMPLE_RATE = 16000


class SttUnavailableError(RuntimeError):
    """faster-whisper is not installed in this environment."""


class SttAudioDecodeError(ValueError):
    """The uploaded bytes could not be decoded as audio."""


class SttLanguageError(ValueError):
    """The requested language is not supported by the selected STT model."""


_WHISPER_LANGUAGE_ALIASES = {
    # Common legacy/browser BCP-47 primaries whose Whisper code differs.
    "cmn": "zh",
    "fil": "tl",
    "in": "id",
    "iw": "he",
    "ji": "yi",
    "nb": "no",
    "nn": "no",
}


def normalize_whisper_language(language: Optional[str]) -> Optional[str]:
    """Convert a BCP-47 locale into the short code faster-whisper expects."""
    if not language:
        return None
    normalized = language.strip().replace("_", "-").lower()
    if not normalized or normalized == "auto":
        return None
    primary = normalized.split("-", 1)[0]
    return _WHISPER_LANGUAGE_ALIASES.get(primary, primary)


def is_available() -> bool:
    """True when the faster-whisper backend can be imported."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def resolve_model_id(model: Optional[str]) -> str:
    """Map a requested id to a supported one, falling back to the default."""
    if model and model in STT_MODELS:
        return model
    return DEFAULT_STT_MODEL


def _pick_device() -> tuple[str, str]:
    """Return (device, compute_type) for CTranslate2.

    CTranslate2 supports CUDA and CPU only (no MPS/XPU), so Apple Silicon and
    Intel GPUs run on CPU with int8, which is fast and low-memory.
    """
    try:
        from utils.hardware.hardware import DeviceType, get_device
        if get_device() == DeviceType.CUDA:
            return "cuda", "float16"
    except Exception as exc:
        logger.debug("STT device detection failed, using CPU: %s", exc)
    return "cpu", "int8"


class WhisperSttSidecar:
    """Lazily loaded faster-whisper model with explicit release. Thread-safe."""

    def __init__(self) -> None:
        self._model = None
        self._model_id: Optional[str] = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded_model(self) -> Optional[str]:
        return self._model_id

    @property
    def device(self) -> Optional[str]:
        return self._device

    def is_loading(self) -> bool:
        # Loading holds the lock; a non-blocking acquire tells the UI to wait.
        acquired = self._lock.acquire(blocking = False)
        if acquired:
            self._lock.release()
            return False
        return True

    def load(self, model: Optional[str] = None):
        """Load (or switch to) a model, reusing it if already resident."""
        model_id = resolve_model_id(model)
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise SttUnavailableError(
                    "Speech-to-text needs the faster-whisper package. "
                    "Run `unsloth studio update` to install it."
                ) from exc

            device, compute_type = _pick_device()
            # Drop the old model before loading a new one to free memory.
            self._model = None
            self._model_id = None
            self._device = None
            logger.info("Loading STT model %s (%s/%s)", model_id, device, compute_type)
            whisper_name = STT_MODELS[model_id]
            try:
                self._model = WhisperModel(whisper_name, device = device, compute_type = compute_type)
            except Exception as exc:
                # CUDA libraries can be missing even when a GPU is present; the
                # CPU path always works, so retry there before giving up.
                if device != "cpu":
                    logger.warning("STT load on %s failed (%s); retrying on CPU", device, exc)
                    self._model = WhisperModel(whisper_name, device = "cpu", compute_type = "int8")
                    device, compute_type = "cpu", "int8"
                else:
                    raise
            self._model_id = model_id
            self._device = device
            logger.info("STT model %s ready on %s", model_id, device)
            return self._model

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
    ) -> dict:
        """Transcribe encoded audio bytes to text.

        Accepts any container faster-whisper (PyAV) can decode: wav, mp3,
        opus/webm, ogg, m4a/aac. Returns {text, language, duration, model}.
        """
        whisper_model = self.load(model)
        # A specific language is faster and more accurate than auto-detect;
        # "auto" (or unset) lets Whisper detect it. The API accepts BCP-47
        # locales, while faster-whisper accepts short codes such as en or fr.
        lang = normalize_whisper_language(language)
        supported_languages = getattr(whisper_model, "supported_languages", None)
        model_id = self._model_id or resolve_model_id(model)
        if lang is not None and model_id in ENGLISH_ONLY_STT_MODELS and lang != "en":
            raise SttLanguageError(
                f"STT model '{model_id}' only supports English. "
                "Choose a multilingual model for this dictation language."
            )
        if lang is not None and supported_languages is not None and lang not in supported_languages:
            if list(supported_languages) == ["en"]:
                raise SttLanguageError(
                    f"STT model '{model_id}' only supports English. "
                    "Choose a multilingual model for this dictation language."
                )
            raise SttLanguageError(
                f"Language '{language}' is not supported by STT model '{model_id}'."
            )
        decode_options = {
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if fast:
            # Composer dictation is already split into short voiced clips. A
            # single greedy candidate avoids five-way beam search and all
            # temperature fallback passes. Skip the second Silero VAD pass and
            # timestamp tokens because the caller already supplies voiced,
            # pause-separated clips and consumes only text.
            decode_options.update(
                beam_size = 1,
                best_of = 1,
                temperature = 0.0,
                without_timestamps = True,
                vad_filter = False,
            )
        try:
            segments, info = whisper_model.transcribe(
                io.BytesIO(audio),
                language = lang,
                **decode_options,
            )
        except (ValueError, RuntimeError) as exc:
            # PyAV raises on undecodable input (e.g. truncated or non-audio).
            raise SttAudioDecodeError("Could not decode the audio.") from exc
        # Guard against pathologically long inputs slipping past the byte cap.
        text_parts: list[str] = []
        for segment in segments:
            if segment.start > _MAX_AUDIO_SECONDS:
                break
            text_parts.append(segment.text)
        text = "".join(text_parts).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "model": self._model_id,
        }

    def unload(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
            self._model_id = None
            self._device = None
        # Drop CTranslate2's native allocations promptly instead of waiting for
        # a later cyclic-GC pass. An in-flight transcription keeps its own
        # reference and releases safely when that request finishes.
        del model
        gc.collect()


_sidecar: Optional[WhisperSttSidecar] = None


def get_stt_sidecar() -> WhisperSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = WhisperSttSidecar()
    return _sidecar
