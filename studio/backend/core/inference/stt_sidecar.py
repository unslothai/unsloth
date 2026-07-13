# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Standalone speech-to-text (STT) sidecar for dictation.

Loads a Whisper model (via Transformers) in the backend process, separate from
the chat model that runs in the inference subprocess. This lets a user dictate
into any chat model, including text-only ones, without evicting it.

Only weights Unsloth has uploaded are used. They are downloaded explicitly
through Studio's Model Hub, and memory is released after dictation. CUDA runs
in float16; Apple Silicon (MPS) and CPU run in float32.
"""

from __future__ import annotations

import gc
import io
import threading
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Multilingual Whisper models. Keys are the API/UI ids; values are the Unsloth
# HF repos downloaded through Studio's Model Hub. Only Unsloth uploads are
# offered, so Studio never pulls third-party weights.
STT_MODELS: dict[str, str] = {
    "small": "unsloth/whisper-small",
    "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
    "large-v3": "unsloth/whisper-large-v3",
}
DEFAULT_STT_MODEL = "small"

# Bound decoded audio so a crafted upload cannot exhaust memory. Callers also
# cap the encoded bytes; this bounds the decoded PCM length.
_MAX_AUDIO_SECONDS = 30 * 60
_TARGET_SAMPLE_RATE = 16000


class SttUnavailableError(RuntimeError):
    """The STT backend (PyTorch/Transformers or PyAV) is not installed."""


class SttModelNotDownloadedError(RuntimeError):
    """The selected model is not complete in the shared Hub cache."""


class SttAudioDecodeError(ValueError):
    """The uploaded bytes could not be decoded as audio."""


class SttAudioTooLongError(ValueError):
    """The decoded audio exceeds the bounded transcription duration."""


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
    """Convert a BCP-47 locale into the short code Whisper expects."""
    if not language:
        return None
    normalized = language.strip().replace("_", "-").lower()
    if not normalized or normalized == "auto":
        return None
    primary = normalized.split("-", 1)[0]
    return _WHISPER_LANGUAGE_ALIASES.get(primary, primary)


def _known_whisper_languages() -> Optional[frozenset[str]]:
    """Return Whisper's language codes without constructing/loading a model."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
    except Exception:
        # Preserve the normal 501 response when Transformers is unavailable,
        # and tolerate a future release moving this constant.
        return None
    return frozenset(LANGUAGES)


def is_available() -> bool:
    """True when the Transformers Whisper backend can be imported."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def resolve_model_id(model: Optional[str]) -> str:
    """Map a requested id to a supported one, falling back to the default."""
    if model and model in STT_MODELS:
        return model
    return DEFAULT_STT_MODEL


def _is_missing_local_model_error(exc: BaseException) -> bool:
    """Recognize a local-cache-only miss without importing HF internals.

    The Model Hub owns downloads, so loading is local-only. Name and message
    based checks tolerate huggingface_hub/Transformers moving the exception.
    """
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in ("LocalEntryNotFoundError", "EntryNotFoundError"):
            return True
        message = str(current).lower()
        if "local_files_only" in message or "does not appear to have a file" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _pick_device():
    """Return (device, torch_dtype) for the Whisper model.

    CUDA uses float16. MPS and CPU use float32: Whisper's decoder is unstable in
    float16 on MPS and degenerates into repeated tokens.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", torch.float32
        return "cpu", torch.float32
    except Exception as exc:
        logger.debug("STT device detection failed, using CPU: %s", exc)
        import torch
        return "cpu", torch.float32


def _decode_audio_bounded(audio: bytes):
    """Decode to 16 kHz mono PCM without ever buffering unbounded audio.

    A small, highly-compressed upload can expand far beyond the encoded request
    limit once decoded. Decode frame-by-frame and enforce the sample cap as
    frames arrive, then hand the array straight to Whisper (no second decode).
    """
    try:
        import av
        import numpy as np
        from av.error import FFmpegError, InvalidDataError
    except ImportError as exc:
        raise SttUnavailableError(
            "Speech-to-text needs the PyAV package to decode audio. "
            "Run `unsloth studio update` to install it."
        ) from exc

    max_samples = _MAX_AUDIO_SECONDS * _TARGET_SAMPLE_RATE
    sample_count = 0
    raw_buffer = io.BytesIO()
    resampler = av.audio.resampler.AudioResampler(
        format = "s16",
        layout = "mono",
        rate = _TARGET_SAMPLE_RATE,
    )
    # Group frames before resampling so short dictation clips usually need only
    # one resampler call instead of one call per codec frame.
    fifo = av.audio.fifo.AudioFifo()

    def write_frame(frame) -> None:
        nonlocal sample_count
        array = frame.to_ndarray()
        sample_count += array.size
        if sample_count > max_samples:
            max_minutes = _MAX_AUDIO_SECONDS // 60
            unit = "minute" if max_minutes == 1 else "minutes"
            raise SttAudioTooLongError(f"Audio must be {max_minutes} {unit} or shorter.")
        raw_buffer.write(array)

    try:
        with av.open(io.BytesIO(audio), mode = "r", metadata_errors = "ignore") as container:
            frames = iter(container.decode(audio = 0))
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except InvalidDataError:
                    # Skip a corrupt frame when the rest of the stream remains
                    # decodable, rather than failing the whole transcription.
                    continue
                frame.pts = None
                fifo.write(frame)
                if fifo.samples >= 500000:
                    for resampled in resampler.resample(fifo.read()):
                        write_frame(resampled)
            if fifo.samples > 0:
                for resampled in resampler.resample(fifo.read()):
                    write_frame(resampled)
            for resampled in resampler.resample(None):
                write_frame(resampled)
    except SttAudioTooLongError:
        raise
    except (FFmpegError, ValueError, RuntimeError) as exc:
        raise SttAudioDecodeError("Could not decode the audio.") from exc
    finally:
        del fifo, resampler

    if sample_count == 0:
        raise SttAudioDecodeError("Could not decode the audio.")
    decoded = np.frombuffer(raw_buffer.getbuffer(), dtype = np.int16).astype(np.float32)
    decoded /= 32768.0
    return decoded


class WhisperSttSidecar:
    """Lazily loaded Transformers Whisper model with explicit release. Thread-safe."""

    def __init__(self) -> None:
        self._engine = None
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

    def _build_model(self, repo: str, device: str, dtype):
        """Load a Whisper model + processor from the local Hub cache.

        local_files_only keeps the Model Hub the only download path; a cache
        miss raises so the caller can surface SttModelNotDownloadedError.
        """
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(repo, local_files_only = True)
        model = WhisperForConditionalGeneration.from_pretrained(
            repo, torch_dtype = dtype, local_files_only = True
        )
        model.to(torch.device(device))
        model.eval()
        return model, processor

    def load(self, model: Optional[str] = None):
        """Load (or switch to) a model, reusing it if already resident.

        Returns a ``(model, processor)`` pair.
        """
        model_id = resolve_model_id(model)
        with self._lock:
            if self._engine is not None and self._model_id == model_id:
                return self._engine
            try:
                import torch
            except Exception as exc:
                raise SttUnavailableError(
                    "Speech-to-text needs PyTorch and Transformers. "
                    "Run `unsloth studio update` to install them."
                ) from exc

            device, dtype = _pick_device()
            # Drop the old model before loading a new one to free memory.
            self._engine = None
            self._model_id = None
            self._device = None
            gc.collect()
            repo = STT_MODELS[model_id]
            logger.info("Loading STT model %s (%s) on %s", model_id, repo, device)

            def not_downloaded(cause: BaseException) -> SttModelNotDownloadedError:
                return SttModelNotDownloadedError(
                    f"STT model '{model_id}' is not downloaded. "
                    "Download it in Settings, then Voice, before loading it."
                )

            try:
                self._engine = self._build_model(repo, device, dtype)
            except Exception as exc:
                if _is_missing_local_model_error(exc):
                    raise not_downloaded(exc) from exc
                # A GPU/MPS build can fail (missing CUDA libs, unsupported op);
                # CPU float32 always works, so retry there before giving up.
                if device != "cpu":
                    logger.warning("STT load on %s failed (%s); retrying on CPU", device, exc)
                    try:
                        self._engine = self._build_model(repo, "cpu", torch.float32)
                    except Exception as cpu_exc:
                        if _is_missing_local_model_error(cpu_exc):
                            raise not_downloaded(cpu_exc) from cpu_exc
                        raise
                    device = "cpu"
                else:
                    raise
            self._model_id = model_id
            self._device = device
            logger.info("STT model %s ready on %s", model_id, device)
            return self._engine

    def _transcribe_decoded(self, model_id: str, decoded_audio, generate_kwargs: dict) -> str:
        """Run Whisper on already-decoded 16 kHz mono PCM and return text.

        Feeds the processor a pre-decoded array so nothing here touches the
        Transformers audio path (torchcodec/ffmpeg). Splits into 30s windows
        (Whisper's receptive field); short dictation clips take one pass.
        """
        import torch

        model, processor = self.load(model_id)
        window = 30 * _TARGET_SAMPLE_RATE
        target_dtype = getattr(model, "dtype", None)
        parts: list[str] = []
        with torch.no_grad():
            for start in range(0, max(len(decoded_audio), 1), window):
                segment = decoded_audio[start : start + window]
                if segment.size == 0:
                    continue
                inputs = processor(
                    segment,
                    sampling_rate = _TARGET_SAMPLE_RATE,
                    return_tensors = "pt",
                )
                features = inputs.input_features.to(model.device)
                if target_dtype is not None:
                    features = features.to(target_dtype)
                generated = model.generate(features, **generate_kwargs)
                text = processor.batch_decode(generated, skip_special_tokens = True)
                parts.append(text[0] if text else "")
        return " ".join(part.strip() for part in parts if part.strip()).strip()

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
    ) -> dict:
        """Transcribe encoded audio bytes to text.

        Accepts any container PyAV can decode: wav, mp3, opus/webm, ogg,
        m4a/aac. Returns {text, language, duration, model}.
        """
        # A set language is faster/more accurate than auto-detect. The API takes
        # BCP-47 locales; Whisper wants short codes like en or fr.
        lang = normalize_whisper_language(language)
        # Pin the requested id: another request may switch the resident model
        # mid-transcription, so sidecar state is not this request's identity.
        model_id = resolve_model_id(model)
        known_languages = _known_whisper_languages()
        if lang is not None and known_languages is not None and lang not in known_languages:
            raise SttLanguageError(
                f"Language '{language}' is not supported by STT model '{model_id}'."
            )
        decoded_audio = _decode_audio_bounded(audio)
        # condition_on_prev_tokens=False stops a fresh clip inheriting prior
        # context, which otherwise causes runaway repeats.
        generate_kwargs = {
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "num_beams": 5,
        }
        if lang is not None:
            generate_kwargs["language"] = lang
        if fast:
            # Dictation clips are short and already voiced, so greedy decoding
            # drops the five-way beam search for much lower latency.
            generate_kwargs["num_beams"] = 1
        text = self._transcribe_decoded(model_id, decoded_audio, generate_kwargs)
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": model_id,
        }

    def unload(self) -> None:
        with self._lock:
            engine = self._engine
            device = self._device
            self._engine = None
            self._model_id = None
            self._device = None
        # Drop the model promptly and free device memory instead of waiting for
        # a later cyclic-GC pass. An in-flight transcription keeps its own
        # reference and releases safely when that request finishes.
        del engine
        try:
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()


_sidecar: Optional[WhisperSttSidecar] = None


def get_stt_sidecar() -> WhisperSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = WhisperSttSidecar()
    return _sidecar
