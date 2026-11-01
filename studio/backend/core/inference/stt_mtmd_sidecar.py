# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""mtmd STT sidecar for non-Whisper ASR: one llama-server on an ephemeral 127.0.0.1 port; two GGUFs."""

from __future__ import annotations

import base64
import http.client
import json
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from loggers import get_logger

from hub.utils.hf_errors import modelscope_missing
from hub.utils.hf_tokens import normalize_token
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
)

from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttLoadCancelledError,
    SttModelBusyError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    SttTranscriptionCancelledError,
    _SelectedHubFile,
    _capture_stt_hub_cache,
    _claim_stt_repository,
    _close_connection_on_cancel,
    _decode_audio_bounded,
    _downloaded_file_bytes,
    _fallback_revisions,
    _HF_COMMIT_SHA,
    _prepare_stt_cache_for_http,
    _read_revision_record,
    _remember_completed_download,
    _TARGET_SAMPLE_RATE,
    _training_active,
    _write_revision_record,
    QWEN3_ASR_LANGUAGE_NAMES,
    normalize_whisper_language,
)

logger = get_logger(__name__)

# Bounded so training is not stalled by a long recording.
_ACTIVE_REQUEST_DRAIN_TIMEOUT = 30.0


@dataclass(frozen = True)
class MtmdSttModel:
    repo: str
    model_file: str
    mmproj_file: str
    label: str
    # Qwen3-ASR emits "language English<asr_text>" before the transcript.
    transcript_marker: Optional[str] = None


# Every entry takes Qwen3-ASR's assistant-prefill language hint; see _post_transcribe before adding another family.
MTMD_STT_MODELS: dict[str, MtmdSttModel] = {
    "qwen3-asr-0.6b": MtmdSttModel(
        repo = "unslothai/Qwen3-ASR-0.6B-GGUF",
        model_file = "Qwen3-ASR-0.6B-Q8_0.gguf",
        mmproj_file = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        label = "Qwen3-ASR 0.6B",
        transcript_marker = "<asr_text>",
    ),
    "qwen3-asr-1.7b": MtmdSttModel(
        repo = "unslothai/Qwen3-ASR-1.7B-GGUF",
        model_file = "Qwen3-ASR-1.7B-Q8_0.gguf",
        mmproj_file = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
        label = "Qwen3-ASR 1.7B",
        transcript_marker = "<asr_text>",
    ),
}
# Voxtral Mini answers audio instead of transcribing in chat mode; Parakeet and Nemotron lack llama.cpp text architectures.

_TRANSCRIBE_PROMPT = "Transcribe the audio."
# Generous cap; it only stops a looping model before the request timeout.
_TRANSCRIPT_TOKENS_PER_SECOND = 30
_MIN_TRANSCRIPT_TOKENS = 512
# Well under trained context, which must also hold the audio; llama-server keeps its default.
_MAX_TRANSCRIPT_TOKENS = 16384


def _transcript_token_budget(audio_seconds: Optional[float]) -> int:
    """Output cap for a clip. A fixed one silently truncated long audio."""
    if not audio_seconds or audio_seconds <= 0:
        return _MIN_TRANSCRIPT_TOKENS
    scaled = int(audio_seconds * _TRANSCRIPT_TOKENS_PER_SECOND)
    return max(_MIN_TRANSCRIPT_TOKENS, min(scaled, _MAX_TRANSCRIPT_TOKENS))


_SERVER_START_TIMEOUT_SECONDS = 180.0
_TRANSCRIBE_TIMEOUT_SECONDS = 600.0


def resolve_mtmd_model_id(model: Optional[str]) -> str:
    """Validate a curated mtmd model id. Custom repos are not supported here."""
    normalized = (model or "").strip()
    if normalized in MTMD_STT_MODELS:
        return normalized
    raise SttModelIdError(
        f"STT model '{model}' is not a curated llama.cpp dictation model. "
        f"Choose one of: {', '.join(MTMD_STT_MODELS)}."
    )


def is_mtmd_model(model: Optional[str]) -> bool:
    return (model or "").strip() in MTMD_STT_MODELS


def find_llama_server_binary() -> Optional[str]:
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend._find_llama_server_binary()


def is_available() -> bool:
    """True when llama-server is installed and audio can be decoded."""
    if find_llama_server_binary() is None:
        return False
    try:
        import av  # noqa: F401
    except Exception:
        # No PyAV means every transcription 501s on decode, so offering a multi-gigabyte download here would be a waste.
        return False
    return True


def _llama_server_child_env(binary: str) -> dict:
    """The chat backend's llama-server environment, for the same binary."""
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend._llama_server_env_for_binary(binary)


def ensure_engine_available() -> str:
    binary = find_llama_server_binary()
    if not binary:
        raise SttUnavailableError(
            "llama.cpp is not installed, so these dictation models cannot run. "
            "Run `unsloth studio update` to install it."
        )
    return binary


def _reap(process: Optional[subprocess.Popen]) -> None:
    """Waits for exit, since terminate alone returns before the child's port and VRAM are actually freed."""
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout = 10)
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        logger.warning("Could not reap llama-server (pid %s): %s", process.pid, exc)
    finally:
        # Forget the pid only once dead: a live child must stay in the shutdown sweep record.
        if process.poll() is not None:
            forget_pid(process.pid)
        else:
            logger.warning(
                "llama-server (pid %s) survived the reap; leaving it adopted so a later "
                "sweep can still find it",
                process.pid,
            )


def _cached_file(
    model_id: str,
    filename: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    from huggingface_hub import hf_hub_download

    from core.inference.stt_sidecar import _active_hf_hub_cache

    root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
    try:
        return hf_hub_download(
            repo_id = MTMD_STT_MODELS[model_id].repo,
            filename = filename,
            revision = revision,
            local_files_only = True,
            cache_dir = str(root),
        )
    except Exception:
        return None


def _cached_model_paths(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Both cached files, or None when either is missing."""
    spec = MTMD_STT_MODELS[model_id]
    from core.inference.stt_sidecar import _active_hf_hub_cache

    root = hub_cache if hub_cache is not None else _active_hf_hub_cache()

    def cached_at(candidate_revision: str) -> Optional[tuple[str, str]]:
        model = _cached_file(
            model_id,
            spec.model_file,
            hub_cache = root,
            revision = candidate_revision,
        )
        mmproj = _cached_file(
            model_id,
            spec.mmproj_file,
            hub_cache = root,
            revision = candidate_revision,
        )
        if model is None or mmproj is None:
            return None
        return model, mmproj

    # Explicit revisions keep both files on the downloaded commit.
    if revision is not None:
        return cached_at(revision)

    recorded = _read_revision_record(spec.repo)
    if recorded:
        cached = cached_at(recorded)
        if cached is not None:
            return cached

    for candidate in _fallback_revisions(spec.repo, hub_cache = root):
        cached = cached_at(candidate)
        if cached is not None:
            _write_revision_record(spec.repo, candidate)
            return cached
    return None


# The panel polls every 750ms and each probe stats the snapshot twice, so memoise briefly.
_DOWNLOADED_PROBE_TTL_SECONDS = 2.0
_downloaded_probe_lock = threading.Lock()
_downloaded_probe: dict[str, tuple[float, bool]] = {}
# Bumped on every invalidation so an in-flight probe cannot overwrite a cleared entry.
_downloaded_probe_generation = 0


def _forget_downloaded_probe(model_id: Optional[str] = None) -> None:
    """Drop memoised answers, for one model or all of them."""
    global _downloaded_probe_generation
    with _downloaded_probe_lock:
        _downloaded_probe_generation += 1
        if model_id is None:
            _downloaded_probe.clear()
        else:
            _downloaded_probe.pop(model_id, None)


def is_model_downloaded(model_id: str) -> bool:
    if model_id not in MTMD_STT_MODELS:
        return False
    with _downloaded_probe_lock:
        cached = _downloaded_probe.get(model_id)
        if cached is not None and time.monotonic() - cached[0] < _DOWNLOADED_PROBE_TTL_SECONDS:
            return cached[1]
        generation = _downloaded_probe_generation
    downloaded = _cached_model_paths(model_id) is not None
    with _downloaded_probe_lock:
        # Timestamp now, not before the probe: a slow cache would store a near-expired entry.
        if generation == _downloaded_probe_generation:
            _downloaded_probe[model_id] = (time.monotonic(), downloaded)
    return downloaded


def _hub_entry_missing(exc: BaseException) -> bool:
    """The Hub answered that this file (or revision) does not exist, as opposed to a network, auth or gate error."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"EntryNotFoundError", "RemoteEntryNotFoundError", "RevisionNotFoundError"})


class _MtmdDownloadState:
    """Tracks the two-file download of one mtmd dictation model.

    ``specs`` / ``cached_paths`` / ``resolve`` let another llama.cpp sidecar (the FLUX.3 prompt upsampler) reuse the
    same pinned two-file download; None keeps the dictation catalogue, looked up at call time. A spec with a ``files``
    tuple downloads those files instead of its model and projector (the Images text-to-mask segmenter).
    """

    def __init__(
        self,
        specs: Optional[dict] = None,
        cached_paths = None,
        resolve = None,
        label: str = "mtmd STT",
        describe_error = None,
    ) -> None:
        self._specs = specs
        self._cached_paths_fn = cached_paths
        self._resolve_fn = resolve
        self._label = label
        # (model_id, exception or worker stderr) -> a message for the UI, or None for the generic one.
        self._describe_error_fn = describe_error
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._download_id: Optional[str] = None
        self._completed_download_ids: list[str] = []
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._selected_files: tuple[_SelectedHubFile, ...] = ()
        self._revision: Optional[str] = None
        self._hub_cache: Optional[Path] = None
        self._cancelled = False

    def _spec(self, model_id: str):
        return (self._specs if self._specs is not None else MTMD_STT_MODELS)[model_id]

    def _resolve(self, model_id: Optional[str]) -> str:
        return (self._resolve_fn or resolve_mtmd_model_id)(model_id)

    def _describe_error(self, model_id: str, error: object) -> Optional[str]:
        if self._describe_error_fn is None:
            return None
        try:
            return self._describe_error_fn(model_id, error)
        except Exception:  # noqa: BLE001 -- the generic message still reaches the UI
            return None

    def _cached_paths(self, model_id: str, **kwargs):
        return (self._cached_paths_fn or _cached_model_paths)(model_id, **kwargs)

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            model_id = self._model_id
            snapshot = {
                "downloading": downloading,
                "model": model_id if downloading else None,
                "download_id": self._download_id,
                "completed_download_ids": list(self._completed_download_ids),
                "error": self._error,
                "cancelled": self._cancelled,
                # "model" goes None once the worker stops, so record which model a cancel applied to.
                "cancelled_model": self._model_id if self._cancelled else None,
                "bytes_total": self._total_bytes if downloading else None,
            }
            captured = (
                model_id,
                self._hub_cache,
                self._selected_files,
                self._revision,
                self._total_bytes,
            )
        # Outside the lock: _downloaded_bytes() stats the cache, and a cancel must not queue.
        snapshot["bytes_done"] = self._downloaded_bytes(*captured) if downloading else None
        return snapshot

    def cancel(
        self,
        model_id: Optional[str] = None,
        download_id: Optional[str] = None,
    ) -> bool:
        """Stop an in-flight download. False when none was running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            if self._download_id in self._completed_download_ids:
                return False
            if download_id is not None:
                if self._download_id != download_id:
                    return False
            elif model_id is not None and self._model_id != model_id:
                return False
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            from core.inference.stt_download_worker import terminate_download
            terminate_download(process)
        return True

    def _downloaded_bytes(
        self,
        model_id: Optional[str] = None,
        hub_cache: Optional[Path] = None,
        selected_files: Optional[tuple[_SelectedHubFile, ...]] = None,
        revision: Optional[str] = None,
        total: Optional[int] = None,
    ) -> Optional[int]:
        """Counts the two selected files; values come from status() under the lock so runs cannot
        mix totals."""
        try:
            model_id = model_id or self._model_id
            hub_cache = hub_cache if hub_cache is not None else self._hub_cache
            selected_files = selected_files if selected_files is not None else self._selected_files
            revision = revision if revision is not None else self._revision
            total = total if total is not None else self._total_bytes
            if not model_id or hub_cache is None or not selected_files:
                return None
            repo = self._spec(model_id).repo
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

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> str:
        model_id = self._resolve(model_id)
        hub_cache = _capture_stt_hub_cache()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    if not self._cancelled:
                        return str(self._download_id)
                    raise SttModelIdError(
                        f"'{model_id}' is still cancelling; try again in a moment."
                    )
                raise SttModelIdError(
                    f"Another dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._download_id = uuid.uuid4().hex
            self._error = None
            self._total_bytes = None
            self._selected_files = ()
            self._revision = None
            self._hub_cache = hub_cache
            self._cancelled = False
            self._process = None
            thread = threading.Thread(
                target = self._run,
                args = (model_id, hf_token),
                daemon = True,
            )
            self._thread = thread
            thread.start()
            return str(self._download_id)

    def _run(
        self,
        model_id: str,
        hf_token: Optional[str],
        hub_cache: Optional[Path] = None,
    ) -> None:
        if hub_cache is None:
            from core.inference.stt_sidecar import _active_hf_hub_cache
            hub_cache = self._hub_cache or _active_hf_hub_cache()
        spec = self._spec(model_id)
        registry = None
        owner = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            selected: list[_SelectedHubFile] = []
            # A spec may pin a commit (the segmenter's variants); none, or a branch such as "main", resolves the newest.
            pinned = getattr(spec, "revision", None)
            revision: Optional[str] = (
                pinned if isinstance(pinned, str) and _HF_COMMIT_SHA.fullmatch(pinned) else None
            )
            for filename in getattr(spec, "files", None) or (spec.model_file, spec.mmproj_file):
                try:
                    meta = get_hf_file_metadata(
                        hf_hub_url(spec.repo, filename, revision = revision),
                        token = normalize_token(hf_token),
                    )
                except Exception as exc:
                    fallback = getattr(spec, "fallback", None)
                    if not fallback or not _hub_entry_missing(exc):
                        raise
                    # A spec whose file is not published (yet) at its revision downloads its fallback instead:
                    # the segmenter's pre-quantized file falls back to the bf16 one.
                    logger.info(
                        "%s: %s is not on %s at %s (%s); downloading %s instead",
                        self._label,
                        filename,
                        spec.repo,
                        revision or "main",
                        type(exc).__name__,
                        fallback,
                    )
                    with self._lock:
                        if self._cancelled:
                            return
                        self._model_id = fallback
                    self._run(fallback, hf_token, hub_cache)
                    return
                if revision is None:
                    revision = meta.commit_hash
                    if not isinstance(revision, str) or not _HF_COMMIT_SHA.fullmatch(revision):
                        raise RuntimeError("could not resolve an immutable mtmd revision")
                if meta.commit_hash != revision:
                    raise RuntimeError("could not pin both mtmd files to one revision")
                selected.append(
                    _SelectedHubFile(
                        path = filename,
                        size = max(0, int(meta.size or 0)),
                        blob_key = meta.etag,
                    )
                )
            # A cancel during metadata has no child to stop; mark it so the run does not rewrite the cache.
            with self._lock:
                if self._cancelled:
                    return
            registry, owner = _claim_stt_repository(spec.repo)
            with self._lock:
                if self._cancelled:
                    return
            _prepare_stt_cache_for_http(spec.repo, hub_cache)
            with self._lock:
                self._total_bytes = sum(item.size for item in selected) or None
                self._selected_files = tuple(selected)
                self._revision = revision

            from core.inference.stt_download_worker import (
                reap_download,
                spawn_download,
                terminate_download,
            )

            args = ["--repo-id", spec.repo, "--revision", revision]
            for item in selected:
                args.extend(("--filename", item.path))
            process = spawn_download(args, hf_token = normalize_token(hf_token), hub_cache = hub_cache)
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
                if (
                    self._cached_paths(
                        model_id,
                        hub_cache = hub_cache,
                        revision = revision,
                    )
                    is None
                ):
                    raise RuntimeError("downloaded mtmd files are missing from the captured cache")
                _write_revision_record(spec.repo, revision)
                with self._lock:
                    _remember_completed_download(
                        self._completed_download_ids,
                        self._download_id,
                        cancelled = self._cancelled,
                    )
                return
            with self._lock:
                if cancelled or process.returncode < 0:
                    self._cancelled = True
                    return
            detail = stderr.decode("utf-8", "replace").strip()
            logger.warning("%s download failed for %s: %s", self._label, model_id, detail)
            with self._lock:
                self._error = (
                    self._describe_error(model_id, detail)
                    or modelscope_missing(detail)
                    or f"Download failed for '{model_id}'."
                )
        except Exception as exc:
            with self._lock:
                if not self._cancelled:
                    logger.warning("%s download failed for %s: %s", self._label, model_id, exc)
                    self._error = (
                        self._describe_error(model_id, exc)
                        or modelscope_missing(exc)
                        or f"Download failed for '{model_id}'."
                    )
        finally:
            # Release first: it is the half that would wedge the repository.
            if registry is not None and owner is not None:
                registry.release_repository_owner(spec.repo, owner)
            # The memo is stale however this ended; dropping it now lets the next poll settle.
            _forget_downloaded_probe(model_id)


_download_state = _MtmdDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> str:
    return _download_state.start(resolve_mtmd_model_id(model), hf_token)


def download_status() -> dict:
    return _download_state.status()


def cancel_model_download(model: Optional[str] = None, download_id: Optional[str] = None) -> bool:
    try:
        model_id = resolve_mtmd_model_id(model) if model is not None else None
    except SttModelIdError:
        return False
    return _download_state.cancel(model_id, download_id)


class MtmdSttSidecar:
    """One llama-server process serving the selected mtmd dictation model."""

    def __init__(self, keep_alive_seconds: float = STT_KEEP_ALIVE_SECONDS) -> None:
        self._lock = threading.RLock()
        # Held across a whole startup; _lock is released while llama-server boots.
        self._start_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._api_key: Optional[str] = None
        self._model_id: Optional[str] = None
        self._binary_path_revision: Optional[int] = None
        self._loading = False
        # Published as soon as Popen returns so a startup can be preempted before _process is set.
        self._starting_process: Optional[subprocess.Popen] = None
        # Resident server launched with GPU off for training; kept so later dictation is not stuck on CPU.
        self._gpu_disabled = False
        # Apart from _gpu_disabled, so training restarts while the preference survives.
        self._forced_cpu = False
        self._load_cancel_event: Optional[threading.Event] = None
        self._load_owner_cancel_event: Optional[threading.Event] = None
        self._update_in_progress = False
        self._keep_alive_seconds = keep_alive_seconds
        self._idle_timer: Optional[threading.Timer] = None
        self._generation = 0
        # Transcription runs outside _lock; the idle timer stays disarmed while requests are in flight.
        self._active_requests = 0

    @property
    def loaded_model(self) -> Optional[str]:
        # Lock-free: the status route reads this on the event loop while _lock may be held.
        return self._model_id if self._process_alive() else None

    @property
    def device(self) -> Optional[str]:
        # derived, not a second probe: two probes can straddle the publish and report a device with no model
        return "llama.cpp" if self.loaded_model else None

    def is_loading(self) -> bool:
        # Bare bool read, for the same reason as loaded_model.
        return self._loading

    @property
    def keep_alive_seconds(self) -> float:
        return self._keep_alive_seconds

    def _process_alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def _cancel_idle_unload_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if self._keep_alive_seconds <= 0 or self._active_requests:
            return
        generation = self._generation
        timer = threading.Timer(self._keep_alive_seconds, self._idle_unload, args = (generation,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._release_locked()

    def _release_locked(self) -> None:
        self._cancel_idle_unload_locked()
        self._generation += 1
        process = self._process
        try:
            # Reap before clearing so lock-free readers see the model until its VRAM is freed.
            _reap(process)
        finally:
            self._process = None
            self._port = None
            self._api_key = None
            self._model_id = None
            self._binary_path_revision = None

    def _drain_active_requests(self, deadline: float) -> None:
        """Must not hold _lock: transcribe claims _active_requests under it, so the wait would block
        itself."""
        while self._active_requests and time.monotonic() < deadline:
            time.sleep(0.1)

    def _holds_expected_model(self, expected: Optional[str]) -> bool:
        """Compared under the lock: another surface can switch the resident model before the release
        arrives."""
        if expected is None:
            return True
        current = self._model_id
        if current is None:
            return False
        if current == expected:
            return True
        try:
            return current == resolve_mtmd_model_id(expected)
        except Exception:  # noqa: BLE001 - an unresolvable name is not this model
            return False

    def unload(
        self,
        wait: bool = True,
        expected_model: Optional[str] = None,
    ) -> None:
        """Busy is judged by _active_requests, since transcribe runs outside _lock; the drain is bounded."""
        if not wait and (self.is_loading() or self._active_requests):
            return
        # Bounded drain: do not kill llama-server under a live transcription, nor wait forever.
        drain_deadline = time.monotonic() + _ACTIVE_REQUEST_DRAIN_TIMEOUT
        if wait:
            self._drain_active_requests(drain_deadline)
        # Cancel first: a startup not yet assigned _process would republish the unloaded model. Outside _lock, since load() takes _start_lock then _lock.
        self.cancel_pending_load()
        self.wait_for_load_to_settle()
        while True:
            if not self._lock.acquire(blocking = wait):
                return
            try:
                # Recheck under the lock: transcribe claims _active_requests while holding it.
                if not self._holds_expected_model(expected_model):
                    return
                busy = bool(self._active_requests)
                if busy and not wait:
                    return
                if not busy or time.monotonic() >= drain_deadline:
                    if busy:
                        logger.warning(
                            "mtmd STT still had %d active request(s) after %.0fs; "
                            "releasing anyway",
                            self._active_requests,
                            _ACTIVE_REQUEST_DRAIN_TIMEOUT,
                        )
                    self._release_locked()
                    return
            finally:
                self._lock.release()
            # Drain outside the lock (draining under it would deadlock the request), then retry.
            self._drain_active_requests(drain_deadline)

    def cancel_pending_load(self) -> bool:
        """Startup runs outside _lock and _process is set only once /health answers, so use an event."""
        if not self._loading:
            return False
        event = self._load_cancel_event
        if event is None:
            return False
        event.set()
        process = self._starting_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        return True

    def _cancel_owned_load(self, owner: threading.Event) -> bool:
        """Cancel startup only when it belongs to this transcription."""
        with self._lock:
            event = self._load_cancel_event
            if not self._loading or event is None or self._load_owner_cancel_event is not owner:
                return False
            event.set()
            process = self._starting_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        return True

    def wait_for_load_to_settle(self) -> None:
        """Taking _start_lock is the wait: load holds it across startup and the cleanup of a
        cancelled one."""
        with self._start_lock:
            pass

    def _raise_if_update_in_progress(self) -> None:
        if self._update_in_progress:
            raise SttUnavailableError(
                "The local transcription runtime is being updated. Try dictation again shortly."
            )

    @contextmanager
    def update_maintenance(self) -> Iterator[bool]:
        """On Windows a live llama-server blocks the swap; the guard is published before waiting on
        the locks."""
        self._update_in_progress = True
        try:
            with self._start_lock, self._lock:
                model_was_active = self._process_alive()
                self._release_locked()
                yield model_was_active
        finally:
            self._update_in_progress = False

    @staticmethod
    def _reserve_free_port() -> tuple[socket.socket, int]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        return sock, sock.getsockname()[1]

    def _ensure_model_downloaded(self, model_id: str) -> tuple[str, str]:
        paths = _cached_model_paths(model_id)
        if paths is None:
            raise SttModelNotDownloadedError(f"STT model '{model_id}' is not downloaded yet.")
        return paths

    def load(
        self,
        model: Optional[str] = None,
        request_cancel_event: Optional[threading.Event] = None,
        device: Optional[str] = None,
    ) -> None:
        """``device`` is the user's audio device preference; ``cpu`` starts
        llama-server at ``-ngl 0``, the same offload training already forces."""
        from core.inference.audio_device import audio_device_forces_cpu

        if request_cancel_event is not None and request_cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        self._raise_if_update_in_progress()
        model_id = resolve_mtmd_model_id(model)
        from utils.llama_cpp_path_settings import custom_llama_cpp_path_revision

        path_revision = custom_llama_cpp_path_revision()
        binary = ensure_engine_available()
        # startup happens outside _lock, so this keeps two callers from each spawning a server and orphaning the first
        with self._start_lock:
            if request_cancel_event is not None and request_cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            self._raise_if_update_in_progress()
            self._load_locked(
                model_id,
                binary,
                request_cancel_event,
                path_revision = path_revision,
                device = device,
            )

    def _load_locked(
        self,
        model_id: str,
        binary: str,
        request_cancel_event: Optional[threading.Event] = None,
        *,
        path_revision: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        from core.inference.audio_device import audio_device_forces_cpu

        if path_revision is None:
            from utils.llama_cpp_path_settings import custom_llama_cpp_path_revision
            path_revision = custom_llama_cpp_path_revision()
        with self._lock:
            # None is no opinion: a caller sending none cannot move the server.
            if device is None:
                forced_cpu = (
                    self._forced_cpu if self._process_alive() else audio_device_forces_cpu(None)
                )
            else:
                forced_cpu = audio_device_forces_cpu(device)
            # Both mean "no offload", which _gpu_disabled already tracks.
            training = _training_active() or forced_cpu
            if (
                self._process_alive()
                and self._model_id == model_id
                and self._binary_path_revision == path_revision
            ):
                # Never restart a busy server just to move it back to the GPU; the next idle load does it.
                if self._gpu_disabled == training or self._active_requests:
                    # Recorded though nothing restarts: dropping it sends the next device-less load to the GPU.
                    self._forced_cpu = forced_cpu
                    self._schedule_idle_unload_locked()
                    return
            # Announce before the slow probe: is_loading() is read lock-free by training admission.
            cancel_event = (
                request_cancel_event if request_cancel_event is not None else threading.Event()
            )
            self._load_cancel_event = cancel_event
            self._load_owner_cancel_event = request_cancel_event
            self._loading = True
            released = False
            try:
                if cancel_event.is_set():
                    raise SttLoadCancelledError("Dictation model loading was cancelled.")
                # Before the release: a 409 for an undownloaded model must not cost the current server.
                model_path, mmproj_path = self._ensure_model_downloaded(model_id)
                # Only guard a live server: a request against one that already died must not block recovery.
                if self._active_requests and self._process_alive():
                    raise SttModelBusyError(
                        "A transcription is still running on the current dictation model. "
                        "Try again in a moment."
                    )
                self._release_locked()
                released = True
            finally:
                # Nothing started, so take the announcement back; past here the startup owns it.
                if not released:
                    self._loading = False
                    self._load_cancel_event = None
                    self._load_owner_cancel_event = None
            # Re-read last: training may have started while the old server was reaped.
            training = _training_active() or forced_cpu
        try:
            sock, port = self._reserve_free_port()
            cmd = [
                binary,
                "-m",
                model_path,
                "--mmproj",
                mmproj_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                # one short request at a time, so one slot keeps the footprint down next to a loaded chat model
                "--parallel",
                "1",
                # keep off the accelerator during training so a dictation load cannot reclaim VRAM training just freed
                "-ngl",
                "0" if training else "99",
            ]
            if training:
                # -ngl 0 does not cover the projector; clip.cpp offloads it separately.
                cmd.append("--no-mmproj-offload")
            sock.close()
            # Refuse to spawn once shutdown has begun; no shutdown step stops this sidecar.
            if is_process_shutting_down():
                raise SttLoadCancelledError(
                    "Unsloth is shutting down; not starting the MTMD server."
                )
            from core.inference.llama_cpp import (
                LlamaCppBackend,
                _llama_server_api_key_enabled,
                _llama_server_key_launch,
            )

            api_key = key_file = None
            key_env: dict = {}
            # A custom build without --api-key-file would exit on the flag; it runs keyless as before.
            if _llama_server_api_key_enabled() and LlamaCppBackend.probe_server_capabilities(
                binary
            ).get("supports_api_key_file", True):
                # Per-launch key against permissive CORS; read once at startup, so deleted once ready.
                api_key = secrets.token_urlsafe(32)
                key_argv, key_env, key_file = _llama_server_key_launch(api_key)
                # Right after the binary: --no-mmproj-offload has to stay last.
                cmd[1:1] = key_argv
            process = subprocess.Popen(
                cmd,
                # An undrained pipe blocks llama-server once its log output fills the buffer.
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                stdin = subprocess.DEVNULL,
                # Like chat: bundled and pip CUDA libs on the loader path; secrets scrubbed.
                env = {**_llama_server_child_env(binary), **key_env},
                # Die with Unsloth, so a crash never orphans a server on the GPU.
                **child_popen_kwargs(),
            )
            # Published before the wait so training can preempt a startup that is already allocating.
            with self._lock:
                self._starting_process = process
            adopt_pid(process.pid)  # terminate_all backstop for graceful exits
            # Recheck once the pid is recorded (closes gate-to-record gap); _reap leaves nothing half-tracked.
            if is_process_shutting_down():
                _reap(process)
                raise SttLoadCancelledError(
                    "Unsloth is shutting down; not starting the MTMD server."
                )
            try:
                ready = self._wait_for_server(process, port, cancel_event)
            finally:
                if key_file is not None:
                    with suppress(OSError):
                        key_file.unlink()
            if not ready:
                # Reap here: _process is unset, so unload() misses a child ignoring SIGTERM.
                _reap(process)
                if cancel_event.is_set():
                    # 409 via the route, like other sidecars: expected preemption, not a broken or missing runtime (501).
                    raise SttLoadCancelledError(
                        "Dictation model loading was cancelled so training could start."
                    )
                raise SttUnavailableError(f"llama-server did not become ready for '{model_id}'.")
            with self._lock:
                self._process = process
                self._port = port
                self._api_key = api_key
                self._model_id = model_id
                self._binary_path_revision = path_revision
                self._gpu_disabled = training
                self._forced_cpu = forced_cpu
                self._generation += 1
                self._schedule_idle_unload_locked()
        finally:
            with self._lock:
                self._loading = False
                self._load_cancel_event = None
                self._load_owner_cancel_event = None
                self._starting_process = None

    @staticmethod
    def _wait_for_server(
        process: subprocess.Popen,
        port: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        from core.inference.llama_cpp import LlamaCppBackend

        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return False
            if process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout = 2) as response:
                    if response.status == 200:
                        return LlamaCppBackend._health_listener_is_ours(process, port)
            except Exception:
                pass
            # Outside the except: a non-200 2xx would otherwise spin this loop with no delay.
            time.sleep(0.25)
        return False

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
        cancel_event: Optional[threading.Event] = None,
        on_progress = None,
    ) -> dict:
        """Transcribe encoded audio bytes, as the other sidecars do.

        ``fast`` has no effect: decoding is greedy. A supported ``language``
        prefills the assistant turn; otherwise the model auto-detects it.
        """
        ensure_engine_available()
        model_id = resolve_mtmd_model_id(model)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        # No training guard: load() already pins the server to CPU while training runs.
        self._ensure_model_downloaded(model_id)
        decoded_audio = _decode_audio_bounded(audio, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        audio_seconds = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        self.load(model_id, request_cancel_event = cancel_event)
        with self._lock:
            port = self._port
            api_key = self._api_key
            if port is None or not self._process_alive():
                raise SttUnavailableError("The dictation server is not running.")
            # Another client may have switched models since load returned; refuse rather than use it.
            if self._model_id != model_id:
                raise SttModelBusyError(
                    "The dictation model changed while this recording was being "
                    "prepared. Try again."
                )
            # Disarm the idle timer: long audio can outlast the keep-alive.
            self._active_requests += 1
            self._cancel_idle_unload_locked()
        try:
            # Outside the lock so unload (including training's) is not blocked for the whole request.
            text = self._post_transcribe(
                port,
                model_id,
                wav_bytes,
                audio_seconds,
                language = language,
                cancel_event = cancel_event,
                **({"api_key": api_key} if api_key else {}),
                **({"on_progress": on_progress} if on_progress is not None else {}),
            )
            if cancel_event is not None and cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
        except Exception:
            if cancel_event is not None and cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            raise
        finally:
            with self._lock:
                self._active_requests -= 1
                self._schedule_idle_unload_locked()
        return {
            "text": text,
            "language": normalize_whisper_language(language),
            "duration": audio_seconds,
            "model": model_id,
        }

    def cancel_transcription(self, cancel_event: threading.Event) -> bool:
        already_cancelled = cancel_event.is_set()
        cancel_event.set()
        return self._cancel_owned_load(cancel_event) or not already_cancelled

    def _post_transcribe(
        self,
        port: int,
        model_id: str,
        wav_bytes: bytes,
        audio_seconds: Optional[float] = None,
        *,
        language: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        on_progress = None,
        api_key: Optional[str] = None,
    ) -> str:
        spec = MTMD_STT_MODELS[model_id]
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(wav_bytes).decode("ascii"),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": _TRANSCRIBE_PROMPT},
                    ],
                }
            ],
            # Greedy: a transcript, not a sampled paraphrase.
            "temperature": 0,
            "max_tokens": _transcript_token_budget(audio_seconds),
        }
        forced_language = QWEN3_ASR_LANGUAGE_NAMES.get(normalize_whisper_language(language) or "")
        assistant_prefix = ""
        if forced_language:
            assistant_prefix = f"language {forced_language}<asr_text>"
            payload["messages"].append({"role": "assistant", "content": assistant_prefix})
            # Explicit, so a server-wide --no-prefill-assistant cannot close the prefix as a finished turn.
            payload.update({"continue_final_message": True, "add_generation_prompt": False})
        if on_progress is not None:
            payload["stream"] = True
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout = _TRANSCRIBE_TIMEOUT_SECONDS
        )
        cancel_done = threading.Event()
        if cancel_event is not None:
            threading.Thread(
                target = _close_connection_on_cancel,
                args = (connection, cancel_event, cancel_done),
                daemon = True,
            ).start()
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            connection.request(
                "POST",
                "/v1/chat/completions",
                body = json.dumps(payload).encode("utf-8"),
                headers = {
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                },
            )
            with connection.getresponse() as response:
                if on_progress is not None and 200 <= response.status < 300:
                    # llama-server does not echo the prefill, so seed it to publish progress from the first token.
                    text = assistant_prefix
                    finished = False
                    for line in response:
                        if cancel_event is not None and cancel_event.is_set():
                            raise SttTranscriptionCancelledError("Transcription cancelled.")
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            finished = True
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError(
                                "The transcription server could not complete this recording."
                            )
                        choices = event.get("choices") or []
                        if choices:
                            finished = finished or choices[0].get("finish_reason") is not None
                            text += choices[0].get("delta", {}).get("content") or ""
                            if not spec.transcript_marker or spec.transcript_marker in text:
                                on_progress(
                                    {"text": _clean_transcript(text, spec.transcript_marker)}
                                )
                    if not finished:
                        raise RuntimeError(
                            "The transcription server disconnected before finishing."
                        )
                    return _clean_transcript(text, spec.transcript_marker)
                response_body = response.read()
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"MTMD transcription server returned HTTP {response.status}."
                    )
                body = json.loads(response_body.decode("utf-8"))
        finally:
            cancel_done.set()
            connection.close()
        choices = body.get("choices") or []
        text = (choices[0].get("message", {}).get("content") or "") if choices else ""
        return _clean_transcript(text, spec.transcript_marker)


def _clean_transcript(text: str, marker: Optional[str]) -> str:
    """Drop the model's leading metadata, keeping only the transcript."""
    if marker and marker in text:
        text = text.rsplit(marker, 1)[1]
    return text.strip()


_sidecar: Optional[MtmdSttSidecar] = None
_sidecar_lock = threading.Lock()


def get_mtmd_stt_sidecar() -> MtmdSttSidecar:
    global _sidecar
    with _sidecar_lock:
        if _sidecar is None:
            _sidecar = MtmdSttSidecar()
        return _sidecar
