# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Standalone speech-to-text (STT) sidecar for dictation.

Loads a Whisper model (via Transformers) in the backend process, separate from
the chat model's inference subprocess, so dictation works with any chat model
without evicting it. Curated defaults plus any Transformers-compatible Whisper
repo; weights come through Studio's Model Hub and stay warm briefly between
dictations. CUDA runs float16; MPS and CPU run float32.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Multilingual Whisper defaults: stable API/UI id -> Hub repository. A request
# may instead pass a validated Hugging Face `owner/model` id.
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
_HF_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Bound decoded PCM length so a crafted upload cannot exhaust memory (callers
# also cap the encoded bytes).
_MAX_AUDIO_SECONDS = 30 * 60
_TARGET_SAMPLE_RATE = 16000

# Non-weight files WhisperProcessor/WhisperForConditionalGeneration may load.
# Weight selection is built from pinned Hub metadata. A custom repo id is
# attacker-controllable, so only safetensors weights are accepted: a
# pytorch_model.bin is a pickle and executes code while Transformers
# deserializes it (see utils/security/file_security.py), and this path skips
# the malware gate the normal model loader applies.
_STT_SNAPSHOT_SUPPORT_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
_STT_SAFETENSORS_INDEX = "model.safetensors.index.json"
_STT_SAFETENSORS_WEIGHTS = "model.safetensors"
_STT_REVISION_RECORD_VERSION = 1


@dataclass(frozen = True)
class _SelectedHubFile:
    path: str
    size: int
    blob_key: Optional[str]


@dataclass(frozen = True)
class _CachedSttSnapshot:
    path: Optional[Path]
    is_multilingual: Optional[bool]


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
    # Legacy/browser BCP-47 primaries whose Whisper code differs.
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
        # Transformers unavailable or the constant moved: skip the check.
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


def _active_hf_hub_cache() -> Path:
    """Return the active Hub cache while respecting runtime test overrides."""
    explicit = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def _repo_cache_dir(repo: str) -> Path:
    return _active_hf_hub_cache() / f"models--{repo.replace('/', '--')}"


def _revision_record_path(repo: str) -> Path:
    from utils.paths.storage_roots import cache_root
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()
    return cache_root() / "stt-revisions" / f"{digest}.json"


def _write_revision_record(repo: str, revision: str) -> None:
    """Persist immutable identity only, never an HF-cache absolute path."""
    if not _HF_COMMIT_SHA.fullmatch(revision):
        return
    path = _revision_record_path(repo)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        with tmp.open("w", encoding = "utf-8") as handle:
            json.dump(
                {
                    "version": _STT_REVISION_RECORD_VERSION,
                    "repo": repo,
                    "revision": revision,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not persist STT revision for %s: %s", repo, exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


def _read_revision_record(repo: str) -> Optional[str]:
    payload = _read_json_object(_revision_record_path(repo))
    if payload.get("version") != _STT_REVISION_RECORD_VERSION or payload.get("repo") != repo:
        return None
    revision = payload.get("revision")
    return revision if isinstance(revision, str) and _HF_COMMIT_SHA.fullmatch(revision) else None


def _safe_snapshot_for_revision(repo: str, revision: str) -> Optional[Path]:
    """Resolve a canonical SHA below this repository's active snapshots dir."""
    if not _HF_COMMIT_SHA.fullmatch(revision):
        return None
    snapshots = _repo_cache_dir(repo) / "snapshots"
    candidate = snapshots / revision
    try:
        snapshots_resolved = snapshots.resolve()
        candidate_resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    if snapshots_resolved not in candidate_resolved.parents or not candidate_resolved.is_dir():
        return None
    return candidate_resolved


def _snapshot_usable(model_id: str, snapshot: Path) -> bool:
    if not _snapshot_is_complete(snapshot):
        return False
    if model_id not in STT_MODELS:
        return _is_whisper_config(_read_json_object(snapshot / "config.json"))
    return True


def _find_complete_cached_snapshot(model: Optional[str]) -> Optional[Path]:
    """Find one complete local snapshot without contacting the Hub."""
    model_id = resolve_model_id(model)
    repo = resolve_model_repo(model_id)

    recorded = _read_revision_record(repo)
    if recorded:
        snapshot = _safe_snapshot_for_revision(repo, recorded)
        if snapshot is not None and _snapshot_usable(model_id, snapshot):
            return snapshot

    ref = _repo_cache_dir(repo) / "refs" / "main"
    try:
        revision = ref.read_text(encoding = "utf-8").strip()
    except OSError:
        revision = ""
    snapshot = _safe_snapshot_for_revision(repo, revision)
    if snapshot is not None and _snapshot_usable(model_id, snapshot):
        _write_revision_record(repo, revision)
        return snapshot

    snapshots = _repo_cache_dir(repo) / "snapshots"
    try:
        revisions = sorted(
            (
                (path.stat().st_mtime_ns, path.name)
                for path in snapshots.iterdir()
                if path.is_dir() and _HF_COMMIT_SHA.fullmatch(path.name)
            ),
            reverse = True,
        )
    except OSError:
        return None
    for _mtime, revision in revisions:
        snapshot = _safe_snapshot_for_revision(repo, revision)
        if snapshot is not None and _snapshot_usable(model_id, snapshot):
            _write_revision_record(repo, revision)
            return snapshot
    return None


def _selected_file_from_sibling(sibling) -> _SelectedHubFile:
    lfs = getattr(sibling, "lfs", None)
    blob_key = getattr(lfs, "sha256", None) or getattr(sibling, "blob_id", None)
    return _SelectedHubFile(
        path = sibling.rfilename,
        size = max(0, int(getattr(sibling, "size", 0) or 0)),
        blob_key = blob_key if isinstance(blob_key, str) and blob_key else None,
    )


def _select_snapshot_files(info, load_index) -> tuple[_SelectedHubFile, ...]:
    """Select support files and one complete safetensors weight set. Pickle
    (pytorch_model.bin) weights are never selected: they are an RCE sink on a
    custom repo id (see _STT_SNAPSHOT_SUPPORT_FILES)."""
    siblings = {
        sibling.rfilename: sibling
        for sibling in (getattr(info, "siblings", None) or [])
        if isinstance(getattr(sibling, "rfilename", None), str)
    }
    selected = {name for name in _STT_SNAPSHOT_SUPPORT_FILES if name in siblings}

    index_name: Optional[str] = None
    if _STT_SAFETENSORS_INDEX in siblings:
        index_name = _STT_SAFETENSORS_INDEX
    elif _STT_SAFETENSORS_WEIGHTS in siblings:
        selected.add(_STT_SAFETENSORS_WEIGHTS)
    else:
        raise SttModelCompatibilityError(
            "The STT repository has no safetensors model weights. Only safetensors "
            "checkpoints are supported; convert the model with save_pretrained(safe_serialization=True)."
        )

    if index_name is not None:
        weight_map = load_index(index_name).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SttModelCompatibilityError(f"Invalid checkpoint index '{index_name}'.")
        shards = set(weight_map.values())
        if not all(isinstance(shard, str) and shard in siblings for shard in shards):
            raise SttModelCompatibilityError(f"Checkpoint index '{index_name}' has missing shards.")
        # The index JSON is attacker-controlled: a safetensors index can name
        # pytorch_model-*.bin shards, which Transformers still loads through
        # torch.load (pickle) since it dispatches per shard by file extension.
        # Require every shard to be safetensors so no pickle file is selected.
        if not all(shard.endswith(".safetensors") for shard in shards):
            raise SttModelCompatibilityError(
                f"Checkpoint index '{index_name}' references non-safetensors shards."
            )
        selected.add(index_name)
        selected.update(shards)

    return tuple(_selected_file_from_sibling(siblings[name]) for name in sorted(selected))


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
            expand = ["config", "sha"],
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
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not _HF_COMMIT_SHA.fullmatch(revision):
        raise SttModelCompatibilityError(
            f"Could not resolve an immutable revision for STT model '{model_id}'."
        )
    # The commit that was validated; the download pins to it so the repo cannot
    # be swapped between validation and snapshot_download (TOCTOU).
    return {"model": model_id, "repo": repo, "revision": revision}


def _is_missing_local_model_error(exc: BaseException) -> bool:
    """Recognize a local-cache-only miss by name/message, without importing HF
    internals (tolerates huggingface_hub/Transformers moving the exception)."""
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
    """True when a cached snapshot holds every file loading needs.

    An aborted download can leave only metadata behind, and an offline lookup
    cannot know the repo's full file list, so verify config, preprocessor,
    tokenizer, and weights directly. is_file() follows cache symlinks, so a
    link from an interrupted blob download does not count.
    """
    # Safetensors only: a cached pytorch_model.bin is a pickle load path and is
    # never treated as a usable snapshot (a repo shipping only pickle weights
    # re-resolves and fails closed in _select_snapshot_files).
    index = snapshot / _STT_SAFETENSORS_INDEX
    if index.is_file():
        # Sharded safetensors checkpoint: every shard must exist and be
        # safetensors (a safe index naming .bin shards would still pickle-load
        # them, matching the _select_snapshot_files guard).
        weight_map = _read_json_object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shards = set(weight_map.values())
        if not all(isinstance(shard, str) and shard.endswith(".safetensors") for shard in shards):
            return False
        has_weights = all((snapshot / shard).is_file() for shard in shards)
    else:
        has_weights = (snapshot / _STT_SAFETENSORS_WEIGHTS).is_file()
    # WhisperProcessor needs the tokenizer: either the fast tokenizer.json or
    # the slow vocab.json + merges.txt pair.
    has_tokenizer = (snapshot / "tokenizer.json").is_file() or (
        (snapshot / "vocab.json").is_file() and (snapshot / "merges.txt").is_file()
    )
    return (
        has_weights
        and has_tokenizer
        and (snapshot / "config.json").is_file()
        and (snapshot / "preprocessor_config.json").is_file()
    )


def is_model_downloaded(model: Optional[str]) -> bool:
    """True when a usable Whisper snapshot exists in the local HF cache."""
    try:
        return _find_complete_cached_snapshot(model) is not None
    except Exception:
        return False


class _SnapshotDownloadState:
    """Tracks one background snapshot_download of a dictation repository.

    Like stt_ggml_sidecar's tracker, but a Transformers checkpoint is a whole
    repo, so progress is the byte count of its cache blobs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._model_id: Optional[str] = None
        self._repo: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._selected_files: tuple[_SelectedHubFile, ...] = ()
        self._complete = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            show_progress = downloading or self._complete
            return {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "bytes_total": self._total_bytes if show_progress else None,
                "bytes_done": self._blob_bytes() if show_progress else None,
            }

    def _blob_bytes(self) -> Optional[int]:
        """Best-effort progress: bytes in the repo's HF cache blobs.

        Counts only the selected support files and one selected weight format,
        including in-progress ``.incomplete`` blobs.
        """
        try:
            # Caller may hold the non-reentrant self._lock; a bare read is safe.
            repo = self._repo
            selected_files = self._selected_files
            if not repo or not selected_files:
                return None
            blobs = _repo_cache_dir(repo) / "blobs"
            if not blobs.is_dir():
                return 0
            done = 0
            for selected in selected_files:
                if not selected.blob_key:
                    continue
                complete = blobs / selected.blob_key
                incomplete = blobs / f"{selected.blob_key}.incomplete"
                candidate = complete if complete.is_file() else incomplete
                if candidate.is_file():
                    done += min(candidate.stat().st_size, selected.size)
            total = self._total_bytes
            return min(done, total) if total is not None else done
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
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
            self._selected_files = ()
            self._complete = False
            thread = threading.Thread(
                target = self._run, args = (self._repo, hf_token, revision), daemon = True
            )
            self._thread = thread
            thread.start()

    def _run(
        self,
        repo: str,
        hf_token: Optional[str],
        revision: Optional[str] = None,
    ) -> None:
        try:
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download

            info = HfApi(token = hf_token or None).model_info(
                repo,
                revision = revision,
                files_metadata = True,
                timeout = 30,
            )
            if not revision:
                revision = getattr(info, "sha", None)
            if not isinstance(revision, str) or not _HF_COMMIT_SHA.fullmatch(revision):
                raise SttModelCompatibilityError(
                    f"Could not resolve an immutable revision for STT model '{repo}'."
                )

            def load_index(filename: str) -> dict:
                path = hf_hub_download(
                    repo_id = repo,
                    filename = filename,
                    revision = revision,
                    token = hf_token or None,
                )
                return _read_json_object(Path(path))

            selected_files = _select_snapshot_files(info, load_index)
            total = sum(selected.size for selected in selected_files)
            with self._lock:
                self._selected_files = selected_files
                self._total_bytes = total or None
            snapshot = Path(
                snapshot_download(
                    repo_id = repo,
                    revision = revision,
                    allow_patterns = [selected.path for selected in selected_files],
                    token = hf_token or None,
                )
            )
            if not _snapshot_is_complete(snapshot):
                raise SttModelCompatibilityError(
                    f"Downloaded STT snapshot for '{repo}' is incomplete."
                )
            _write_revision_record(repo, revision)
            with self._lock:
                self._complete = True
        except Exception as exc:
            logger.warning("STT snapshot download failed for %s: %s", repo, exc)
            with self._lock:
                self._error = f"Download failed for '{repo}'."


_download_state = _SnapshotDownloadState()


def start_model_download(
    model: Optional[str],
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
) -> None:
    _download_state.start(resolve_model_id(model), hf_token, revision = revision)


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

        # New loads use CPU during training; a resident GPU model may stay put
        # when the training admission check confirms enough headroom.
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
    """Decode to 16 kHz mono PCM without buffering unbounded audio.

    A small, highly-compressed upload can expand far past the encoded request
    limit once decoded, so decode frame-by-frame and enforce the sample cap as
    frames arrive, then hand the array straight to Whisper.
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
    # Group frames before resampling so short clips need one resampler call
    # rather than one per codec frame.
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
                    # Skip a corrupt frame rather than fail the whole transcription.
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
        """Block until any in-flight load() has exited and freed its memory.

        load() holds self._lock throughout, including the from_pretrained()/
        .to(device) allocation and cancel cleanup, so acquiring the lock here
        waits for that memory to be freed.
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

    def _build_model(self, snapshot_path: str, device: str, dtype, cancel_event: threading.Event):
        """Load a Whisper model + processor from the local Hub cache.

        local_files_only keeps the Model Hub the only download path; a cache
        miss raises so the caller can surface SttModelNotDownloadedError.
        """
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = None
        model = None
        try:
            processor = WhisperProcessor.from_pretrained(snapshot_path, local_files_only = True)
            self._raise_if_load_cancelled(cancel_event)
            # use_safetensors forces the pickle-free load path even if a
            # pytorch_model.bin somehow reached the cache; the selector and the
            # completeness check already exclude pickle weights upstream.
            model = WhisperForConditionalGeneration.from_pretrained(
                snapshot_path, torch_dtype = dtype, local_files_only = True, use_safetensors = True
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

    def _ensure_model_downloaded(self, model_id: str) -> _CachedSttSnapshot:
        """Validate the local snapshot before decode or model replacement.

        Returns the checkpoint's multilingual flag when local metadata provides
        it. Curated defaults are known multilingual.
        """
        model_id = resolve_model_id(model_id)
        with self._lock:
            if self._engine is not None and self._model_id == model_id:
                resident_model = (
                    self._engine[0] if isinstance(self._engine, (tuple, list)) else self._engine
                )
                generation_config = getattr(resident_model, "generation_config", None)
                is_multilingual = getattr(generation_config, "is_multilingual", None)
                return _CachedSttSnapshot(
                    path = None,
                    is_multilingual = is_multilingual if isinstance(is_multilingual, bool) else None,
                )
        snapshot_path = _find_complete_cached_snapshot(model_id)
        if snapshot_path is None:
            raise SttModelNotDownloadedError(
                f"STT model '{model_id}' is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            )

        if model_id in STT_MODELS:
            return _CachedSttSnapshot(path = snapshot_path, is_multilingual = True)

        if not _is_whisper_config(_read_json_object(snapshot_path / "config.json")):
            raise SttModelCompatibilityError(
                f"STT model '{model_id}' is not a compatible Transformers Whisper model."
            )
        generation_config = _read_json_object(snapshot_path / "generation_config.json")
        is_multilingual = generation_config.get("is_multilingual")
        if isinstance(is_multilingual, bool):
            return _CachedSttSnapshot(path = snapshot_path, is_multilingual = is_multilingual)
        if resolve_model_repo(model_id).lower().endswith(".en"):
            return _CachedSttSnapshot(path = snapshot_path, is_multilingual = False)
        return _CachedSttSnapshot(path = snapshot_path, is_multilingual = None)

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
                cached = self._ensure_model_downloaded(model_id)
                snapshot_path = cached.path
                if snapshot_path is None:
                    raise SttModelNotDownloadedError(
                        f"STT model '{model_id}' is not downloaded. "
                        "Download it in Settings, then Voice, before loading it."
                    )
                self._raise_if_load_cancelled(cancel_event)
                device, dtype = _pick_device()
                self._release_engine_locked()
                logger.info("Loading STT model %s (%s) on %s", model_id, snapshot_path, device)

                def not_downloaded(cause: BaseException) -> SttModelNotDownloadedError:
                    return SttModelNotDownloadedError(
                        f"STT model '{model_id}' is not downloaded. "
                        "Download it in Settings, then Voice, before loading it."
                    )

                retry_on_cpu = False
                try:
                    candidate = self._build_model(str(snapshot_path), device, dtype, cancel_event)
                    self._raise_if_load_cancelled(cancel_event)
                except SttLoadCancelledError:
                    raise
                except Exception as exc:
                    if _is_missing_local_model_error(exc):
                        raise not_downloaded(exc) from exc
                    if device == "cpu":
                        raise
                    logger.warning("STT load on %s failed (%s); retrying on CPU", device, exc)
                    retry_on_cpu = True
                if retry_on_cpu:
                    # Retry outside the handler: live exception state pins frames
                    # referencing the partly loaded model, so leave it before
                    # clearing the cache to release that memory.
                    _clear_device_cache(device)
                    try:
                        candidate = self._build_model(
                            str(snapshot_path),
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

        Feeds a pre-decoded array so nothing here touches the Transformers audio
        path (torchcodec/ffmpeg). Splits into 30s windows (Whisper's receptive
        field); short clips take one pass.
        """
        import torch

        model, processor = self.load(model_id)
        effective_generate_kwargs = dict(generate_kwargs)
        generation_config = getattr(model, "generation_config", None)
        if getattr(generation_config, "is_multilingual", None) is False:
            # English-only checkpoints fix language and task in their generation
            # config, and Transformers rejects passing them here.
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
        # Reject a missing runtime up front, before the cache and bounded decode.
        ensure_stt_available()
        # A set language beats auto-detect. API takes BCP-47; Whisper wants short
        # codes like en or fr.
        lang = normalize_whisper_language(language)
        # Pin the requested id: another request may switch the resident model
        # mid-transcription, so sidecar state is not this request's identity.
        model_id = resolve_model_id(model)
        known_languages = _known_whisper_languages()
        if lang is not None and known_languages is not None and lang not in known_languages:
            raise SttLanguageError(
                f"Language '{language}' is not supported by STT model '{model_id}'."
            )
        cached = self._ensure_model_downloaded(model_id)
        if cached.is_multilingual is False and lang not in (None, "en"):
            raise SttLanguageError(
                f"Language '{language}' is not supported by English-only STT model '{model_id}'."
            )
        decoded_audio = _decode_audio_bounded(audio)
        # condition_on_prev_tokens=False stops a fresh clip inheriting prior
        # context, which causes runaway repeats.
        generate_kwargs = {
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "num_beams": 5,
        }
        if lang is not None:
            generate_kwargs["language"] = lang
        if fast:
            # Short voiced clips: greedy decoding drops beam search for latency.
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
