# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Standalone speech-to-text (STT) sidecar for dictation.

Loads a Whisper model (via Transformers) in the backend process, separate from
the chat model that runs in the inference subprocess. This lets a user dictate
into any chat model, including text-only ones, without evicting it.

Five Unsloth-curated Whisper models are offered by default. Users may also
select another Transformers-compatible Whisper repository from Hugging Face.
Weights are downloaded explicitly through Studio's Model Hub and kept warm
briefly between dictations. CUDA runs in float16; Apple Silicon (MPS) and CPU
run in float32.
"""

from __future__ import annotations

import gc
import io
import json
import re
import threading
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Multilingual Whisper defaults. Keys are stable API/UI ids; values are the
# repositories downloaded through Studio's Model Hub. A request may also use a
# validated Hugging Face `owner/model` id for another Whisper-compatible model.
STT_MODELS: dict[str, str] = {
    "tiny": "unsloth/whisper-tiny",
    "base": "unsloth/whisper-base",
    "small": "unsloth/whisper-small",
    "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
    "large-v3": "unsloth/whisper-large-v3",
}
DEFAULT_STT_MODEL = "small"
STT_KEEP_ALIVE_SECONDS = 5 * 60
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

# Bound decoded audio so a crafted upload cannot exhaust memory. Callers also
# cap the encoded bytes; this bounds the decoded PCM length.
_MAX_AUDIO_SECONDS = 30 * 60
_TARGET_SAMPLE_RATE = 16000


class SttUnavailableError(RuntimeError):
    """The STT backend (PyTorch/Transformers or PyAV) is not installed."""


class SttLoadCancelledError(RuntimeError):
    """An in-flight STT model load was cancelled for training."""


class SttModelNotDownloadedError(RuntimeError):
    """The selected model is not complete in the shared Hub cache."""


class SttModelIdError(ValueError):
    """The requested custom model is not a valid Hugging Face repository id."""


class SttModelCompatibilityError(ValueError):
    """The requested repository is not a Transformers Whisper checkpoint."""


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


def ensure_stt_available() -> None:
    """Raise when the complete local Whisper backend cannot be imported."""
    try:
        import av  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise SttUnavailableError(
            "Speech-to-text needs PyTorch, Transformers, and PyAV. "
            "Run `unsloth studio update` to install them."
        ) from exc


def is_available() -> bool:
    """True when the complete local Whisper backend can be imported."""
    try:
        ensure_stt_available()
    except SttUnavailableError:
        return False
    return True


def resolve_model_id(model: Optional[str]) -> str:
    """Resolve a curated id or validate a custom Hugging Face repository."""
    if not model:
        return DEFAULT_STT_MODEL
    normalized = model.strip()
    if normalized in STT_MODELS:
        return normalized
    if _HF_REPO_ID.fullmatch(normalized):
        return normalized
    raise SttModelIdError(
        "STT model must be one of Studio's defaults or a Hugging Face "
        "repository in 'owner/model' form."
    )


def resolve_model_repo(model_id: str) -> str:
    """Return the Hub repository for a curated or custom model id."""
    resolved = resolve_model_id(model_id)
    return STT_MODELS.get(resolved, resolved)


def _is_whisper_config(config: object) -> bool:
    """True when Hub/local config metadata identifies a Whisper ASR model."""
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip().lower() == "whisper":
        return True
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(
        isinstance(name, str) and name == "WhisperForConditionalGeneration"
        for name in architectures
    )


def _read_json_object(path: Path) -> dict:
    try:
        with open(path, "r", encoding = "utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def validate_remote_model(model: Optional[str], hf_token: Optional[str] = None) -> dict:
    """Verify a custom Hub repository is Whisper-compatible without downloading weights."""
    model_id = resolve_model_id(model)
    repo = resolve_model_repo(model_id)
    if model_id in STT_MODELS:
        return {"model": model_id, "repo": repo}

    try:
        from huggingface_hub import HfApi
        info = HfApi(token = hf_token or False).model_info(
            repo,
            expand = ["config"],
            timeout = 10,
        )
    except Exception as exc:
        raise SttModelCompatibilityError(
            f"Could not verify STT model '{model_id}'. "
            "Check that the repository exists and your Hugging Face token can access it."
        ) from exc

    if not _is_whisper_config(getattr(info, "config", None)):
        raise SttModelCompatibilityError(
            f"STT model '{model_id}' is not a compatible Transformers Whisper model."
        )
    return {"model": model_id, "repo": repo}


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


def _snapshot_is_complete(snapshot: Path) -> bool:
    """True when a cached snapshot holds every file loading actually needs.

    An aborted download can leave a snapshot directory holding only small
    metadata files, and an offline cache lookup cannot know the repository's
    full file list -- so verify config, preprocessor, and weights directly.
    is_file() follows the cache symlinks, so a link left behind by an
    interrupted blob download does not count.
    """
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        # Sharded checkpoint: one present shard is not enough; every
        # shard in the index must exist.
        weight_map = _read_json_object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        has_weights = all((snapshot / shard).is_file() for shard in set(weight_map.values()))
    else:
        has_weights = any(
            p.is_file()
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for p in snapshot.glob(pattern)
        )
    return (
        has_weights
        and (snapshot / "config.json").is_file()
        and (snapshot / "preprocessor_config.json").is_file()
    )


def is_model_downloaded(model: Optional[str]) -> bool:
    """True when a usable Whisper snapshot exists in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id = resolve_model_repo(resolve_model_id(model)),
                local_files_only = True,
            )
        )
        return _snapshot_is_complete(snapshot)
    except Exception:
        return False


class _SnapshotDownloadState:
    """Tracks one background snapshot_download of a dictation repository.

    Mirrors stt_ggml_sidecar's per-file tracker, but a Transformers checkpoint
    is a whole repository, so progress is the byte count of its cache blobs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._model_id: Optional[str] = None
        self._repo: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            return {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "bytes_total": self._total_bytes if downloading else None,
                "bytes_done": self._blob_bytes() if downloading else None,
            }

    def _blob_bytes(self) -> Optional[int]:
        """Best-effort progress: bytes in the repository's HF cache blobs.

        Counts finished and ``.incomplete`` blobs alike; the repositories are
        dedicated Whisper checkpoints, so every blob belongs to this download.
        """
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Callers may already hold self._lock (non-reentrant); a bare
            # attribute read is safe without it.
            repo = self._repo
            if not repo:
                return None
            blobs = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs"
            if not blobs.is_dir():
                return None
            return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file())
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = resolve_model_id(model_id)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    return
                raise SttModelIdError(
                    f"Another dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._repo = resolve_model_repo(model_id)
            self._error = None
            self._total_bytes = None
            thread = threading.Thread(target = self._run, args = (self._repo, hf_token), daemon = True)
            self._thread = thread
            thread.start()

    def _run(self, repo: str, hf_token: Optional[str]) -> None:
        try:
            from huggingface_hub import HfApi, snapshot_download
            try:
                info = HfApi(token = hf_token or None).model_info(
                    repo, files_metadata = True, timeout = 30
                )
                total = sum(s.size or 0 for s in info.siblings or [])
                with self._lock:
                    self._total_bytes = total or None
            except Exception:
                pass
            snapshot_download(repo_id = repo, token = hf_token or None)
        except Exception as exc:
            logger.warning("STT snapshot download failed for %s: %s", repo, exc)
            with self._lock:
                self._error = f"Download failed for '{repo}'."


_download_state = _SnapshotDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> None:
    _download_state.start(resolve_model_id(model), hf_token)


def download_status() -> dict:
    return _download_state.status()


def _training_active() -> bool:
    try:
        from core.training import get_training_backend
        return bool(get_training_backend().is_training_active())
    except Exception:
        return False


def _clear_device_cache(device: Optional[str]) -> None:
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _pick_device():
    """Return (device, torch_dtype) for the Whisper model.

    CUDA uses float16. MPS and CPU use float32: Whisper's decoder is unstable in
    float16 on MPS and degenerates into repeated tokens.
    """
    try:
        import torch

        # New STT loads use CPU during training. A resident GPU model may stay
        # loaded when the training admission check confirms enough headroom.
        training_active = _training_active()
        if not training_active and torch.cuda.is_available():
            return "cuda", torch.float16
        if (
            not training_active
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
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
            if not container.streams.audio:
                raise SttAudioDecodeError("Could not decode the audio.")
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
    except (SttAudioDecodeError, SttAudioTooLongError):
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
    """Lazily loaded Whisper model with idle eviction. Thread-safe."""

    def __init__(self, keep_alive_seconds: float = STT_KEEP_ALIVE_SECONDS) -> None:
        self._engine = None
        self._model_id: Optional[str] = None
        self._device: Optional[str] = None
        self._lock = threading.RLock()
        self._load_state_lock = threading.Lock()
        self._loading = False
        self._load_cancel_event: Optional[threading.Event] = None
        self._keep_alive_seconds = max(0.0, keep_alive_seconds)
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0

    @property
    def loaded_model(self) -> Optional[str]:
        return self._model_id

    @property
    def device(self) -> Optional[str]:
        return self._device

    def is_loading(self) -> bool:
        with self._load_state_lock:
            return self._loading

    def cancel_pending_load(self) -> bool:
        """Cancel a model load without waiting for the model lock."""
        with self._load_state_lock:
            event = self._load_cancel_event
            if not self._loading or event is None:
                return False
            event.set()
            return True

    def wait_for_load_to_settle(self) -> None:
        """Block until any in-flight load() has exited and released its memory.

        load() holds self._lock for its whole duration, including the blocking
        from_pretrained()/.to(device) allocation and the cancel cleanup, so
        acquiring the lock here waits for that memory to be freed.
        """
        with self._lock:
            pass

    def _begin_load(self) -> threading.Event:
        event = threading.Event()
        with self._load_state_lock:
            self._load_cancel_event = event
            self._loading = True
        return event

    def _end_load(self, event: threading.Event) -> None:
        with self._load_state_lock:
            if self._load_cancel_event is event:
                self._load_cancel_event = None
                self._loading = False

    @staticmethod
    def _raise_if_load_cancelled(event: threading.Event) -> None:
        if event.is_set():
            raise SttLoadCancelledError("STT model loading was cancelled so training could start.")

    @property
    def keep_alive_seconds(self) -> float:
        return self._keep_alive_seconds

    def _cancel_idle_unload_locked(self) -> None:
        self._idle_generation += 1
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if self._engine is None or self._keep_alive_seconds <= 0:
            return
        generation = self._idle_generation
        timer = threading.Timer(
            self._keep_alive_seconds,
            self._idle_unload,
            args = (generation,),
        )
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        with self._lock:
            if generation != self._idle_generation or self._engine is None:
                return
            logger.info("Unloading idle STT model %s", self._model_id)
            self._release_engine_locked()

    def _release_engine_locked(self) -> None:
        self._cancel_idle_unload_locked()
        engine = self._engine
        device = self._device
        self._engine = None
        self._model_id = None
        self._device = None
        del engine
        _clear_device_cache(device)

    def _build_model(self, repo: str, device: str, dtype, cancel_event: threading.Event):
        """Load a Whisper model + processor from the local Hub cache.

        local_files_only keeps the Model Hub the only download path; a cache
        miss raises so the caller can surface SttModelNotDownloadedError.
        """
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = None
        model = None
        try:
            processor = WhisperProcessor.from_pretrained(repo, local_files_only = True)
            self._raise_if_load_cancelled(cancel_event)
            model = WhisperForConditionalGeneration.from_pretrained(
                repo, torch_dtype = dtype, local_files_only = True
            )
            self._raise_if_load_cancelled(cancel_event)
            model.to(torch.device(device))
            self._raise_if_load_cancelled(cancel_event)
            model.eval()
            return model, processor
        except SttLoadCancelledError:
            model = None
            processor = None
            _clear_device_cache(device)
            raise

    def _ensure_model_downloaded(self, model_id: str) -> Optional[bool]:
        """Validate the local snapshot before decode or resident-model replacement.

        Returns the checkpoint's multilingual flag when local metadata provides
        it. Studio's curated defaults are known multilingual.
        """
        model_id = resolve_model_id(model_id)
        with self._lock:
            if self._engine is not None and self._model_id == model_id:
                resident_model = (
                    self._engine[0] if isinstance(self._engine, (tuple, list)) else self._engine
                )
                generation_config = getattr(resident_model, "generation_config", None)
                is_multilingual = getattr(generation_config, "is_multilingual", None)
                return is_multilingual if isinstance(is_multilingual, bool) else None
        try:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(
                repo_id = resolve_model_repo(model_id),
                local_files_only = True,
            )
        except Exception as exc:
            if _is_missing_local_model_error(exc):
                raise SttModelNotDownloadedError(
                    f"STT model '{model_id}' is not downloaded. "
                    "Download it in Settings, then Voice, before loading it."
                ) from exc
            raise

        snapshot_path = Path(snapshot)
        # A resolvable snapshot can still be partial (aborted download); fail
        # here so callers do not decode audio before load() hits missing files.
        if not _snapshot_is_complete(snapshot_path):
            raise SttModelNotDownloadedError(
                f"STT model '{model_id}' is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            )

        if model_id in STT_MODELS:
            return True

        if not _is_whisper_config(_read_json_object(snapshot_path / "config.json")):
            raise SttModelCompatibilityError(
                f"STT model '{model_id}' is not a compatible Transformers Whisper model."
            )
        generation_config = _read_json_object(snapshot_path / "generation_config.json")
        is_multilingual = generation_config.get("is_multilingual")
        if isinstance(is_multilingual, bool):
            return is_multilingual
        if resolve_model_repo(model_id).lower().endswith(".en"):
            return False
        return None

    def load(self, model: Optional[str] = None):
        """Load (or switch to) a model, reusing it if already resident.

        Returns a ``(model, processor)`` pair.
        """
        model_id = resolve_model_id(model)
        with self._lock:
            ensure_stt_available()
            if self._engine is not None and self._model_id == model_id:
                self._schedule_idle_unload_locked()
                return self._engine
            import torch

            cancel_event = self._begin_load()
            candidate = None
            device: Optional[str] = None
            try:
                self._ensure_model_downloaded(model_id)
                self._raise_if_load_cancelled(cancel_event)
                device, dtype = _pick_device()
                self._release_engine_locked()
                repo = resolve_model_repo(model_id)
                logger.info("Loading STT model %s (%s) on %s", model_id, repo, device)

                def not_downloaded(cause: BaseException) -> SttModelNotDownloadedError:
                    return SttModelNotDownloadedError(
                        f"STT model '{model_id}' is not downloaded. "
                        "Download it in Settings, then Voice, before loading it."
                    )

                try:
                    candidate = self._build_model(repo, device, dtype, cancel_event)
                    self._raise_if_load_cancelled(cancel_event)
                except SttLoadCancelledError:
                    raise
                except Exception as exc:
                    if _is_missing_local_model_error(exc):
                        raise not_downloaded(exc) from exc
                    # Retry on CPU when the accelerator cannot load the model.
                    if device != "cpu":
                        logger.warning("STT load on %s failed (%s); retrying on CPU", device, exc)
                        # The traceback pins frames that still reference the
                        # partly loaded accelerator model; drop it so the cache
                        # clear can release that memory before the CPU retry.
                        exc = exc.with_traceback(None)
                        _clear_device_cache(device)
                        try:
                            candidate = self._build_model(
                                repo,
                                "cpu",
                                torch.float32,
                                cancel_event,
                            )
                            self._raise_if_load_cancelled(cancel_event)
                        except SttLoadCancelledError:
                            raise
                        except Exception as cpu_exc:
                            if _is_missing_local_model_error(cpu_exc):
                                raise not_downloaded(cpu_exc) from cpu_exc
                            raise
                        device = "cpu"
                    else:
                        raise
                with self._load_state_lock:
                    self._raise_if_load_cancelled(cancel_event)
                    self._engine = candidate
                    self._model_id = model_id
                    self._device = device
                    self._load_cancel_event = None
                    self._loading = False
                self._schedule_idle_unload_locked()
                logger.info("STT model %s ready on %s", model_id, device)
                return self._engine
            except SttLoadCancelledError:
                candidate = None
                self._release_engine_locked()
                _clear_device_cache(device)
                raise
            finally:
                self._end_load(cancel_event)

    def _transcribe_decoded(self, model_id: str, decoded_audio, generate_kwargs: dict) -> str:
        """Run Whisper on already-decoded 16 kHz mono PCM and return text.

        Feeds the processor a pre-decoded array so nothing here touches the
        Transformers audio path (torchcodec/ffmpeg). Splits into 30s windows
        (Whisper's receptive field); short dictation clips take one pass.
        """
        import torch

        model, processor = self.load(model_id)
        effective_generate_kwargs = dict(generate_kwargs)
        generation_config = getattr(model, "generation_config", None)
        if getattr(generation_config, "is_multilingual", None) is False:
            # Transformers rejects both controls for English-only Whisper
            # checkpoints; their generation config already fixes the language
            # and transcription task.
            effective_generate_kwargs.pop("task", None)
            effective_generate_kwargs.pop("language", None)
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
                generated = model.generate(features, **effective_generate_kwargs)
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
        # Reject a missing runtime before the model cache and the bounded decode,
        # so an unavailable server 501s up front instead of decoding first.
        ensure_stt_available()
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
        is_multilingual = self._ensure_model_downloaded(model_id)
        if is_multilingual is False and lang not in (None, "en"):
            raise SttLanguageError(
                f"Language '{language}' is not supported by English-only STT model '{model_id}'."
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
        # Serialize inference with model switches and unloads.
        with self._lock:
            try:
                text = self._transcribe_decoded(model_id, decoded_audio, generate_kwargs)
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": model_id,
        }

    def unload(self) -> None:
        with self._lock:
            self._release_engine_locked()


_sidecar: Optional[WhisperSttSidecar] = None


def get_stt_sidecar() -> WhisperSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = WhisperSttSidecar()
    return _sidecar
