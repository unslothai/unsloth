# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Standalone speech-to-text (STT) sidecar for dictation.

Loads a Whisper model (via Transformers) in a spawn child of its own, separate
from the chat model's inference subprocess, so dictation works with any chat
model without evicting it. Curated defaults plus any Transformers-compatible
Whisper repo; weights come through Unsloth's Model Hub and stay warm briefly
between dictations. CUDA runs float16; MPS and CPU run float32.

Everything except the model itself stays here: device choice, the Hub cache,
audio decoding, windowing and the idle timer. Only the load and the generate
happen in core/inference/stt_transformers_worker.py, because an accelerator
context is never returned while the process holding it lives and the backend
must not be the process that takes one.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import threading
import time
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


class SttDownloadCacheError(RuntimeError):
    """The shared Hub cache cannot safely admit an STT HTTP writer."""


def _capture_stt_hub_cache() -> Path:
    """Capture the configured Hub cache once for one STT download run."""
    from utils.hf_cache_settings import get_hf_cache_paths
    return Path(get_hf_cache_paths().hub_cache).expanduser()


def _claim_stt_repository(repo: str) -> tuple[object, object]:
    """Reserve *repo* against every Model Hub writer and delete."""
    from hub.utils.download_registry import get_models_registry

    registry = get_models_registry()
    owner = object()
    claimed, state = registry.claim_repository_owner(repo, owner)
    if not claimed:
        raise SttDownloadCacheError(
            f"'{repo}' is already active in the Model Hub ({state}); wait for it to finish."
        )
    return registry, owner


def _prepare_stt_cache_for_http(repo: str, hub_cache: Path) -> None:
    """Prepare *repo* for HTTP, rejecting unverified transport markers."""
    from hub.utils.download_registry import TRANSPORT_HTTP, prepare_cache_for_transport
    from hub.utils.hf_cache_state import (
        TRANSPORT_MARKER_NAME,
        iter_active_repo_cache_dirs,
        repo_cache_dir_name,
    )

    canonical = hub_cache / repo_cache_dir_name("model", repo)
    try:
        canonical.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        raise SttDownloadCacheError(
            f"Could not make the Hugging Face cache safe for '{repo}'. "
            "Close other programs using that cache and try again."
        ) from exc
    prepare_cache_for_transport("model", repo, TRANSPORT_HTTP, root = hub_cache)
    entries = list(iter_active_repo_cache_dirs("model", repo, root = hub_cache))
    try:
        all_marked_http = bool(entries) and all(
            (entry / TRANSPORT_MARKER_NAME).read_text(encoding = "utf-8").strip() == TRANSPORT_HTTP
            for entry in (*entries, canonical)
        )
    except (OSError, UnicodeDecodeError):
        all_marked_http = False
    if not all_marked_http:
        raise SttDownloadCacheError(
            f"Could not make the Hugging Face cache safe for '{repo}'. "
            "Close other programs using that cache and try again."
        )


def _downloaded_file_bytes(
    *,
    hub_cache: Path,
    repo: str,
    filename: str,
    size: int,
    blob_key: Optional[str],
    revision: Optional[str],
) -> int:
    """Count one selected file across partial, finalized, and snapshot forms."""
    repo_cache = _repo_cache_dir(repo, hub_cache = hub_cache)
    candidates: list[Path] = []
    if blob_key:
        blobs = repo_cache / "blobs"
        candidates.extend((blobs / f"{blob_key}.incomplete", blobs / blob_key))
    if revision:
        candidates.append(repo_cache / "snapshots" / revision / filename)
    sizes: list[int] = []
    for candidate in candidates:
        try:
            if candidate.is_file():
                sizes.append(max(0, int(candidate.stat().st_size)))
        except OSError:
            continue
    done = max(sizes, default = 0)
    return min(done, max(0, int(size)))


class SttUnavailableError(RuntimeError):
    """The STT backend (PyTorch/Transformers or PyAV) is not installed."""


class SttLoadCancelledError(RuntimeError):
    """An in-flight STT model load was cancelled for training."""


class SttTranscriptionCancelledError(RuntimeError):
    """An in-flight transcription was cancelled by its client."""


class SttModelNotDownloadedError(RuntimeError):
    """The selected model is not complete in the shared Hub cache."""


class SttModelBusyError(RuntimeError):
    """The current model cannot make way yet, so the switch has to be retried."""


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


def _close_connection_on_cancel(connection, cancel_event, done_event) -> None:
    """Abandon one blocked sidecar HTTP request, leaving its server resident.

    Shutting the socket unblocks the read without touching the process, so a cancelled
    dictation does not cost the next one a server relaunch and model load. Shared by the
    whisper.cpp and llama.cpp sidecars.
    """
    while not done_event.is_set():
        if not cancel_event.wait(0.05):
            continue
        while not done_event.is_set():
            sock = connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
                return
            time.sleep(0.01)
        return


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
        "STT model must be one of Unsloth's defaults or a Hugging Face "
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
    """Return the currently configured Hub cache, including live relocation."""
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        paths = get_hf_cache_paths()
        # Unsloth's live cache setting takes precedence over environment defaults.
        if paths.source == "studio":
            return Path(paths.hub_cache)
        explicit = (os.environ.get("HF_HUB_CACHE") or "").strip()
        if explicit:
            return Path(explicit).expanduser()
        hf_home = (os.environ.get("HF_HOME") or "").strip()
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        return Path(paths.hub_cache)
    except Exception:
        explicit = (os.environ.get("HF_HUB_CACHE") or "").strip()
        if explicit:
            return Path(explicit).expanduser()
        hf_home = (os.environ.get("HF_HOME") or "").strip()
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)


def _repo_cache_dir(repo: str, *, hub_cache: Optional[Path] = None) -> Path:
    root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
    return root / f"models--{repo.replace('/', '--')}"


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


def _fallback_revisions(repo: str, *, hub_cache: Optional[Path] = None) -> list[str]:
    """Cached commits to try when no revision record survives, newest first.

    Pinned downloads never write refs/main, so without this a lost record would
    hide an already downloaded model and re-download it on every launch.
    """
    repo_cache = _repo_cache_dir(repo, hub_cache = hub_cache)
    candidates: list[str] = []
    try:
        main_revision = (repo_cache / "refs" / "main").read_text(encoding = "utf-8").strip()
        if _HF_COMMIT_SHA.fullmatch(main_revision):
            candidates.append(main_revision)
    except OSError:
        pass
    try:
        snapshots = sorted(
            (p for p in (repo_cache / "snapshots").iterdir() if _HF_COMMIT_SHA.fullmatch(p.name)),
            key = lambda p: p.stat().st_mtime,
            reverse = True,
        )
    except OSError:
        snapshots = []
    candidates.extend(p.name for p in snapshots if p.name not in candidates)
    return candidates


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
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._repo: Optional[str] = None
        self._revision: Optional[str] = None
        self._hub_cache: Optional[Path] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._selected_files: tuple[_SelectedHubFile, ...] = ()
        self._complete = False
        self._cancelled = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            show_progress = downloading or self._complete
            snapshot = {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "cancelled": self._cancelled,
                # Which model the cancel applies to. "model" goes None once the worker
                # thread stops, so a settled cancellation was indistinguishable from an
                # unrelated one and a deferred load restarted the whole download.
                "cancelled_model": self._model_id if self._cancelled else None,
                "bytes_total": self._total_bytes if show_progress else None,
            }
            captured = (
                self._repo,
                self._revision,
                self._hub_cache,
                self._selected_files,
                self._total_bytes,
            )
        # Outside the lock: _downloaded_bytes() stats the cache, and a cancel must not queue.
        snapshot["bytes_done"] = self._downloaded_bytes(*captured) if show_progress else None
        return snapshot

    def cancel(self) -> bool:
        """Stop an in-flight download. False when none was running.

        Partial blobs stay cached, so a restart resumes from them.
        """
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            from core.inference.stt_download_worker import terminate_download
            terminate_download(process)
        return True

    def _downloaded_bytes(
        self,
        repo: Optional[str] = None,
        revision: Optional[str] = None,
        hub_cache: Optional[Path] = None,
        selected_files: Optional[tuple[_SelectedHubFile, ...]] = None,
        total: Optional[int] = None,
    ) -> Optional[int]:
        """Count selected files across partial, blob, and snapshot locations.

        status() captures these under the lock and passes them in: reading them
        here would let a run that starts mid-probe pair its bytes with the total
        of the run that just ended.
        """
        try:
            repo = repo if repo is not None else self._repo
            revision = revision if revision is not None else self._revision
            hub_cache = hub_cache or self._hub_cache or _active_hf_hub_cache()
            selected_files = selected_files if selected_files is not None else self._selected_files
            total = total if total is not None else self._total_bytes
            if not repo or hub_cache is None or not selected_files:
                return None
            done = sum(
                _downloaded_file_bytes(
                    hub_cache = hub_cache,
                    repo = repo,
                    filename = selected.path,
                    size = selected.size,
                    blob_key = selected.blob_key,
                    revision = revision,
                )
                for selected in selected_files
            )
            return min(done, total) if total is not None else done
        except Exception:
            return None

    def _run_worker(
        self, args: list[str], hf_token: Optional[str], hub_cache: Path
    ) -> tuple[bool, bytes]:
        from core.inference.stt_download_worker import (
            reap_download,
            spawn_download,
            terminate_download,
        )

        process = spawn_download(args, hf_token = hf_token or None, hub_cache = hub_cache)
        with self._lock:
            if self._cancelled:
                terminate_download(process)
            self._process = process
        stderr = reap_download(process)
        with self._lock:
            if self._process is process:
                self._process = None
            cancelled = self._cancelled
        if process.returncode == 0 and not cancelled:
            return True, stderr
        if cancelled or process.returncode < 0:
            with self._lock:
                self._cancelled = True
            return False, stderr
        detail = stderr.decode("utf-8", "replace").strip()
        raise SttModelCompatibilityError(f"Download worker failed for '{self._repo}': {detail}")

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        model_id = resolve_model_id(model_id)
        hub_cache = _capture_stt_hub_cache()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    # Joining a cancelling run would silently download nothing.
                    if not self._cancelled:
                        return
                    raise SttModelIdError(
                        f"'{model_id}' is still cancelling; try again in a moment."
                    )
                raise SttModelIdError(
                    f"Another dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._repo = resolve_model_repo(model_id)
            self._revision = None
            self._hub_cache = hub_cache
            self._error = None
            self._total_bytes = None
            self._selected_files = ()
            self._complete = False
            self._cancelled = False
            self._process = None
            thread = threading.Thread(
                target = self._run,
                args = (self._repo, hf_token, revision),
                daemon = True,
            )
            self._thread = thread
            thread.start()

    def _run(
        self,
        repo: str,
        hf_token: Optional[str],
        revision: Optional[str] = None,
        hub_cache: Optional[Path] = None,
    ) -> None:
        hub_cache = hub_cache or self._hub_cache or _active_hf_hub_cache()
        registry = None
        owner = None
        try:
            from huggingface_hub import HfApi, hf_hub_download

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

            # A cancel during metadata has no child to stop. Without these the
            # run still reserves the repo and rewrites the cache after the stop.
            with self._lock:
                if self._cancelled:
                    return
            registry, owner = _claim_stt_repository(repo)
            with self._lock:
                if self._cancelled:
                    return
            _prepare_stt_cache_for_http(repo, hub_cache)
            with self._lock:
                self._revision = revision

            def load_index(filename: str) -> dict:
                completed, _stderr = self._run_worker(
                    ["--repo-id", repo, "--revision", revision, "--filename", filename],
                    hf_token,
                    hub_cache,
                )
                if not completed:
                    return {}
                path = hf_hub_download(
                    repo_id = repo,
                    filename = filename,
                    revision = revision,
                    local_files_only = True,
                    cache_dir = str(hub_cache),
                )
                return _read_json_object(Path(path))

            selected_files = _select_snapshot_files(info, load_index)
            total = sum(selected.size for selected in selected_files)
            with self._lock:
                self._selected_files = selected_files
                self._total_bytes = total or None
            args = ["--repo-id", repo, "--revision", revision]
            for selected in selected_files:
                args += ["--filename", selected.path]
            completed, _stderr = self._run_worker(args, hf_token, hub_cache)
            if not completed:
                return
            snapshot = _repo_cache_dir(repo, hub_cache = hub_cache) / "snapshots" / revision
            if not _snapshot_is_complete(snapshot):
                raise SttModelCompatibilityError(
                    f"Downloaded STT snapshot for '{repo}' is incomplete."
                )
            _write_revision_record(repo, revision)
            with self._lock:
                self._complete = True
        except Exception as exc:
            with self._lock:
                if not self._cancelled:
                    logger.warning("STT snapshot download failed for %s: %s", repo, exc)
                    self._error = f"Download failed for '{repo}'."
        finally:
            if registry is not None and owner is not None:
                registry.release_repository_owner(repo, owner)


_download_state = _SnapshotDownloadState()


def start_model_download(
    model: Optional[str],
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
) -> None:
    _download_state.start(resolve_model_id(model), hf_token, revision = revision)


def download_status() -> dict:
    return _download_state.status()


def cancel_model_download() -> bool:
    return _download_state.cancel()


def _training_active() -> bool:
    try:
        from core.training import get_training_backend
        return bool(get_training_backend().is_training_active())
    except Exception:
        return False


def _clear_device_cache(device: Optional[str], collect: bool = True) -> None:
    """Drop the unreferenced model, then hand its blocks back to the allocator.

    ``collect = False`` for a caller that has just collected and dropped nothing since: a full
    collection is not free (a long-lived backend reaches millions of tracked objects, where one
    pass costs about a second), and the cancel path below runs this twice in a row while it
    holds the model lock that ``wait_for_load_to_settle`` waits on."""
    if collect:
        gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _reported_device(device: Optional[str]) -> Optional[str]:
    """Device name for status. Torch calls the HIP device "cuda", which is right for the
    API and wrong on screen: an AMD card reported as cuda reads like a bug."""
    if device != "cuda":
        return device
    try:
        import torch
        if getattr(torch.version, "hip", None):
            return "rocm"
    except Exception:  # noqa: BLE001 - a label must never fail a status call
        pass
    return device


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


def _dtype_name(dtype) -> str:
    """Name a dtype for the worker command: torch.float16 becomes float16.

    Takes the plain strings tests use as readily as a real torch dtype, so the
    command carries no torch object across the process boundary.
    """
    return str(dtype).rsplit(".", 1)[-1]


def _engine_is_alive(engine) -> bool:
    """False only for a worker whose process is confirmed dead.

    Anything without a liveness check counts as live, so a caller holding a
    plain object (tests, or a future in-process engine) is unaffected. So does
    a probe that cannot answer: absence of liveness evidence is not evidence
    that the accelerator context was released, and reporting nothing resident
    is what lets training be admitted against memory that is not free.
    """
    is_alive = getattr(engine, "is_alive", None)
    if is_alive is None:
        return True
    try:
        return bool(is_alive())
    except Exception as exc:  # noqa: BLE001 - an unanswerable probe must not fail a status read
        logger.warning("Could not check whether the STT worker is alive: %s", exc)
        return True


def _engine_survived_kill(engine) -> bool:
    """Whether a handle says its own child outlived terminate and kill.

    close() reports that to its caller, but a cancelled or timed-out command
    closes the worker from inside the handle and raises over the answer, so the
    only record that reaches here is the one the handle keeps on itself.
    """
    return bool(getattr(engine, "survived_kill", False))


def _close_engine(engine) -> bool:
    """End the worker behind an engine handle, if it has one.

    Ending the worker is what returns its accelerator context; dropping the
    handle and emptying the cache cannot. A plain object (tests, or a future
    in-process engine) has no close and needs none.

    False when the engine says so itself, which WhisperWorker does for a child
    that outlived terminate and kill and is therefore still holding the memory
    this call was made to release, and False when close() raises out of a
    process operation: nothing was confirmed dead, so the handle has to be kept
    rather than the memory advertised as free. A close that raised over a child
    already gone still counts as released, so bookkeeping that failed after the
    death cannot wedge every later load.
    """
    close = getattr(engine, "close", None)
    if close is None:
        return True
    try:
        return close() is not False
    except Exception as exc:  # noqa: BLE001 - a stuck worker must not block the unload
        logger.warning("Could not stop the STT worker: %s", exc)
        return not _engine_is_alive(engine)


def _decode_audio_bounded(audio: bytes, cancel_event = None):
    """Decode to 16 kHz mono PCM without buffering unbounded audio.

    A small, highly-compressed upload can expand far past the encoded request
    limit once decoded, so decode frame-by-frame and enforce the sample cap as
    frames arrive, then hand the array straight to Whisper.

    ``cancel_event`` is polled inside the frame loop: checking only after the decode
    returned let an abandoned upload run to EOF or the sample cap, and several of them
    could do that at once.
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
                if cancel_event is not None and cancel_event.is_set():
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
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
    except (SttAudioDecodeError, SttAudioTooLongError, SttTranscriptionCancelledError):
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
        self._load_owner_cancel_event: Optional[threading.Event] = None
        self._keep_alive_seconds = max(0.0, keep_alive_seconds)
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0
        # Held only to keep its memory accounted: a worker that outlived its own
        # kill answers nothing, so a later dictation cannot be handed it.
        self._survivor = False
        # A child that outlived the kill start() gave it, handed from _build_model
        # to the load that called it. Written and read under _lock.
        self._start_survivor = None

    @property
    def loaded_model(self) -> Optional[str]:
        # A worker that died holds nothing, so reporting its model would make
        # training admission reserve memory for a model that is not there.
        engine = self._engine
        if engine is not None and not _engine_is_alive(engine):
            return None
        return self._model_id

    @property
    def device(self) -> Optional[str]:
        # Reported, so name the backend a user recognises. Torch's ROCm build keeps the
        # "cuda" device name for HIP, which made an AMD box report "Transformers - cuda".
        return _reported_device(self._device)

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

    def _cancel_owned_load(self, owner: threading.Event) -> bool:
        """Cancel startup only when it belongs to this transcription."""
        with self._load_state_lock:
            event = self._load_cancel_event
            if not self._loading or event is None or self._load_owner_cancel_event is not owner:
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

    def _begin_load(self, owner: Optional[threading.Event] = None) -> threading.Event:
        event = owner if owner is not None else threading.Event()
        with self._load_state_lock:
            self._load_cancel_event = event
            self._load_owner_cancel_event = owner
            self._loading = True
        return event

    def _end_load(self, event: threading.Event) -> None:
        with self._load_state_lock:
            if self._load_cancel_event is event:
                self._load_cancel_event = None
                self._load_owner_cancel_event = None
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

    def _release_engine_locked(self) -> bool:
        """Release the resident engine. False if its child outlived the kill.

        Such a child still holds its accelerator memory, so forgetting it here
        would report the model unloaded and let training be admitted against
        memory that is not free. Keep it resident instead and rearm the idle
        timer, so the release is tried again rather than stranded.

        The fields are cleared only once the worker is confirmed dead. close()
        can take the full shutdown wait, and loaded_model reads the fields
        without this lock, so clearing them first would report nothing resident
        for that whole window.

        A worker kept this way is flagged a survivor: it was asked to shut down,
        terminated and killed, so it is held for its memory and not for its
        answers, and a later dictation must load one of its own rather than be
        handed this one and wait out the command timeout on it.
        """
        self._cancel_idle_unload_locked()
        engine = self._engine
        device = self._device
        released = _close_engine(engine)
        if released:
            self._engine = None
            self._model_id = None
            self._device = None
            self._survivor = False
        else:
            self._survivor = True
        del engine
        _clear_device_cache(device)
        if not released:
            self._schedule_idle_unload_locked()
        return released

    def _keep_survivor_locked(
        self,
        engine,
        model_id: str,
        device: Optional[str] = None,
    ) -> None:
        """Hold an engine whose child outlived its close, so it stays accounted.

        Its device is the one the child reports, which after a CPU retry is not
        the one this load started on; a child that never finished its load
        reports none, so the device the attempt was made on stands in. The idle
        timer is rearmed, so the release is tried again rather than the survivor
        being stranded here.

        Held for its memory, not for its answers: it is flagged so a later
        dictation loads a worker of its own instead of being handed one that is
        wedged, which would cost the caller the whole command timeout.
        """
        self._engine = engine
        self._model_id = model_id
        self._survivor = True
        self._device = getattr(engine, "device", None) or device
        logger.error(
            "The dictation worker for %s outlived the kill and still holds its memory; "
            "keeping it resident so it is not reported unloaded",
            model_id,
        )
        self._schedule_idle_unload_locked()

    def _is_survivor_locked(self) -> bool:
        """Whether the resident engine is held for its memory, not its answers.

        Folds in the flag the handle raised on itself: a command that was
        cancelled or timed out closes the worker from inside the handle, so
        close()'s False never reaches the sidecar and this is the only way it
        learns the child outlived both signals. Handing such a worker to the
        next dictation would spend the whole command timeout on it under the
        model lock; refusing lets the idle timer retry the kill instead.
        """
        if self._survivor:
            return True
        if self._engine is not None and _engine_survived_kill(self._engine):
            self._survivor = True
            return True
        return False

    def _release_dead_engine_locked(self) -> None:
        """Drop a worker whose process is gone, so the next use loads a fresh one."""
        if self._engine is not None and not _engine_is_alive(self._engine):
            logger.warning("STT worker for %s exited; it will be reloaded", self._model_id)
            self._release_engine_locked()

    def _build_model(self, snapshot_path: str, device: str, dtype, cancel_event: threading.Event):
        """Start a worker process holding this model and return its handle.

        Out of process because an accelerator context is never given back while
        the process holding it lives, so an in-process load made the backend
        permanently heavier even after unload.

        A host that cannot create a child at all (a sandbox, or a frozen POSIX
        build) falls back to loading here instead, on the CPU: this move may
        take work out of the backend, never take dictation away from someone
        who had it. The fallback waits for the CPU attempt, so a spawn failure
        on an accelerator still goes through the caller's own CPU retry rather
        than downgrading the user here.

        A child that outlived start()'s own kill is left in ``_start_survivor``
        for the caller. start() ends its child on every failure, so a handle
        still reporting a live process is one holding memory that nothing else
        knows about: dropping it here is what would let this failed load read as
        nothing resident.
        """
        from core.inference.stt_transformers_worker import (
            InProcessWhisperEngine,
            SttWorkerSpawnError,
            WhisperWorker,
        )

        worker = WhisperWorker()
        try:
            # start() kills its own child on any failure, including cancellation.
            worker.start(str(snapshot_path), device, _dtype_name(dtype), cancel_event)
        except SttWorkerSpawnError as exc:
            if device != "cpu":
                raise
            logger.warning(
                "No dictation worker process could be started (%s); "
                "loading the model in the backend on the CPU instead",
                exc,
            )
            engine = InProcessWhisperEngine()
            engine.start(str(snapshot_path), "cpu", _dtype_name(dtype), cancel_event)
            return engine
        except BaseException:
            if _engine_is_alive(worker):
                self._start_survivor = worker
            raise
        return worker

    def _ensure_model_downloaded(self, model_id: str) -> _CachedSttSnapshot:
        """Validate the local snapshot before decode or model replacement.

        Returns the checkpoint's multilingual flag when local metadata provides
        it. Curated defaults are known multilingual.

        A survivor is held for its memory alone, so it does not answer for the
        model the way a resident one does: the snapshot is looked up on disk, or
        the load it precedes would be turned away as a checkpoint that is not
        downloaded.
        """
        model_id = resolve_model_id(model_id)
        with self._lock:
            reusable = self._engine is not None and self._model_id == model_id
            if reusable and not self._is_survivor_locked():
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

    def load(
        self,
        model: Optional[str] = None,
        request_cancel_event: Optional[threading.Event] = None,
    ):
        """Load (or switch to) a model, reusing it if already resident.

        Returns a ``(model, processor)`` pair.
        """
        if request_cancel_event is not None and request_cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        model_id = resolve_model_id(model)
        with self._lock:
            if request_cancel_event is not None and request_cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            ensure_stt_available()
            self._release_dead_engine_locked()
            reusable = self._engine is not None and self._model_id == model_id
            if reusable and not self._is_survivor_locked():
                self._schedule_idle_unload_locked()
                return self._engine
            import torch

            cancel_event = self._begin_load(request_cancel_event)
            candidate = None
            device: Optional[str] = None
            resident_released = False
            self._start_survivor = None
            try:
                self._raise_if_load_cancelled(cancel_event)
                cached = self._ensure_model_downloaded(model_id)
                snapshot_path = cached.path
                if snapshot_path is None:
                    raise SttModelNotDownloadedError(
                        f"STT model '{model_id}' is not downloaded. "
                        "Download it in Settings, then Voice, before loading it."
                    )
                self._raise_if_load_cancelled(cancel_event)
                device, dtype = _pick_device()
                if not self._release_engine_locked():
                    # Starting a second child over one that never exited doubles
                    # the memory this release was meant to give back.
                    raise SttModelBusyError(
                        "The previous dictation worker did not exit and still holds its "
                        "memory. Try again shortly."
                    )
                resident_released = True
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
                    # The worker classifies a cache miss; the exception cannot cross processes.
                    if isinstance(exc, SttModelNotDownloadedError) or _is_missing_local_model_error(
                        exc
                    ):
                        raise not_downloaded(exc) from exc
                    if device == "cpu":
                        raise
                    if self._start_survivor is not None:
                        # The attempt left a child that outlived its own kill and still
                        # holds the device. A second child would sit beside it, and
                        # installing that one would forget this one, which is what lets
                        # training be admitted against memory that is not free. Refuse as
                        # a release that could not kill its worker does; the timer retries.
                        raise SttModelBusyError(
                            "The previous dictation worker did not exit and still holds its "
                            "memory. Try again shortly."
                        ) from exc
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
                        if isinstance(
                            cpu_exc, SttModelNotDownloadedError
                        ) or _is_missing_local_model_error(cpu_exc):
                            raise not_downloaded(cpu_exc) from cpu_exc
                        raise
                    device = "cpu"
                with self._load_state_lock:
                    self._raise_if_load_cancelled(cancel_event)
                    self._engine = candidate
                    self._model_id = model_id
                    self._device = device
                    self._survivor = False
                    self._load_cancel_event = None
                    self._load_owner_cancel_event = None
                    self._loading = False
                self._schedule_idle_unload_locked()
                logger.info("STT model %s ready on %s", model_id, device)
                return self._engine
            except SttLoadCancelledError:
                # cancel_pending_load() does not wait for the model lock, so the cancel
                # can land after start() came back with a live child. Nothing installed
                # the candidate, and dropping the handle does not end the process holding
                # the context training is waiting for, so close it here.
                if self._start_survivor is not None:
                    # start() ends its own child, so this one outlived terminate and kill
                    # inside it and never became a candidate. It holds its memory all the
                    # same and this is the only handle on the process, so keep it rather
                    # than let the cancel report the memory given back; the timer retries.
                    self._keep_survivor_locked(self._start_survivor, model_id, device)
                    _clear_device_cache(device)
                    raise
                if not _close_engine(candidate):
                    # It outlived terminate and kill, so it still holds the memory this
                    # cancel was made to free. Keep it, for the same reason
                    # _release_engine_locked keeps its own survivor: reporting nothing
                    # resident is what lets training be admitted against memory that is
                    # not free. Nothing is installed over, since a candidate exists only
                    # after the resident was released.
                    self._keep_survivor_locked(candidate, model_id)
                    candidate = None
                    _clear_device_cache(device)
                    raise
                candidate = None
                if resident_released:
                    # _release_engine_locked already collected, and the candidate was dropped
                    # before it ran, so nothing has become garbage since. This second call is
                    # only here to empty the cache of the device this LOAD picked, which need
                    # not be the resident's, so it keeps the sweep and skips the collection.
                    self._release_engine_locked()
                    _clear_device_cache(device, collect = False)
                else:
                    _clear_device_cache(device)
                raise
            except BaseException:
                # Same reasoning for any other failed load: this is the only handle on a
                # child that outlived start()'s own kill and still holds its memory, and
                # reporting nothing resident lets training be admitted against it.
                if self._start_survivor is not None and self._engine is None:
                    self._keep_survivor_locked(self._start_survivor, model_id, device)
                    _clear_device_cache(device)
                raise
            finally:
                self._start_survivor = None
                self._end_load(cancel_event)

    def _transcribe_decoded(
        self,
        model_id: str,
        decoded_audio,
        generate_kwargs: dict,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, Optional[list[dict]]]:
        """Run Whisper on already-decoded 16 kHz mono PCM and return (text, segments).

        Splits into 30s windows (Whisper's receptive field) and sends one window
        at a time to the worker; short clips take one pass. Windowing stays here
        so a cancelled dictation stops between windows even while the worker is
        busy, and so no single message carries more than 30 seconds of audio.

        ``segments`` marks each window's own audio bounds. That is coarser than
        whisper.cpp's native per-utterance segmentation (see the ggml sidecar),
        since asking Transformers for finer timestamps means changing the
        generation call itself, but it is still real audio-time, not fabricated.
        """
        import numpy as np

        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        if cancel_event is None:
            engine = self.load(model_id)
        else:
            engine = self.load(model_id, request_cancel_event = cancel_event)
        effective_generate_kwargs = dict(generate_kwargs)
        generation_config = getattr(engine, "generation_config", None)
        if getattr(generation_config, "is_multilingual", None) is False:
            # English-only checkpoints fix language and task in their generation
            # config, and Transformers rejects passing them here.
            effective_generate_kwargs.pop("task", None)
            effective_generate_kwargs.pop("language", None)
        window = 30 * _TARGET_SAMPLE_RATE
        parts: list[str] = []
        segments: list[dict] = []
        for start in range(0, max(len(decoded_audio), 1), window):
            if cancel_event is not None and cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            segment = decoded_audio[start : start + window]
            if segment.size == 0:
                continue
            pcm = np.ascontiguousarray(segment, dtype = np.float32).tobytes()
            text = engine.transcribe_window(pcm, effective_generate_kwargs, cancel_event)
            if cancel_event is not None and cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            stripped = text.strip() if text else ""
            if stripped:
                parts.append(stripped)
                segments.append(
                    {
                        "start": start / _TARGET_SAMPLE_RATE,
                        "end": min(start + window, len(decoded_audio)) / _TARGET_SAMPLE_RATE,
                        "text": stripped,
                    }
                )
        joined = " ".join(parts).strip()
        return joined, (segments or None)

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Transcribe encoded audio bytes to text.

        Accepts any container PyAV can decode: wav, mp3, opus/webm, ogg,
        m4a/aac. Returns {text, language, duration, model, segments}, where
        ``segments`` is a list of {start, end, text} at window granularity
        (see ``_transcribe_decoded``).
        """
        # Reject a missing runtime up front, before the cache and bounded decode.
        ensure_stt_available()
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
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
        decoded_audio = _decode_audio_bounded(audio, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
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
                if cancel_event is None:
                    text, segments = self._transcribe_decoded(
                        model_id, decoded_audio, generate_kwargs
                    )
                else:
                    text, segments = self._transcribe_decoded(
                        model_id,
                        decoded_audio,
                        generate_kwargs,
                        cancel_event,
                    )
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": model_id,
            "segments": segments,
        }

    def cancel_transcription(self, cancel_event: threading.Event) -> bool:
        """Ask this request's Transformers generation or load to stop.

        Only this request's own event is set. The thread waiting on the worker
        mirrors it into the child within a poll, so a cancel never reaches a
        window belonging to a different request.
        """
        already_cancelled = cancel_event.is_set()
        cancel_event.set()
        return self._cancel_owned_load(cancel_event) or not already_cancelled

    def _holds_expected_model(self, expected: Optional[str]) -> bool:
        """Whether the resident model is the one the caller claimed. Call under ``_lock``.

        A caller that owns a specific model must not release whatever happens to be
        resident: another surface can switch the engine between the ownership check and
        the request reaching the sidecar, and the queued unload then tears down a model
        it never owned.
        """
        if expected is None:
            return True
        current = self._model_id
        if current is None:
            return False
        if current == expected:
            return True
        try:
            return current == resolve_model_id(expected)
        except Exception:  # noqa: BLE001 - an unresolvable name is not this model
            return False

    def unload(
        self,
        wait: bool = True,
        expected_model: Optional[str] = None,
    ) -> None:
        """Release the resident model. ``wait=False`` skips a sidecar mid-request.

        A transcription holds ``_lock`` throughout, so a caller releasing engines it does
        not own must be able to leave a busy one alone. ``expected_model`` scopes the
        release to one model, compared under the lock.
        """
        if not self._lock.acquire(blocking = wait):
            return
        try:
            if not self._holds_expected_model(expected_model):
                return
            self._release_engine_locked()
        finally:
            self._lock.release()


_sidecar: Optional[WhisperSttSidecar] = None


def get_stt_sidecar() -> WhisperSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = WhisperSttSidecar()
    return _sidecar
