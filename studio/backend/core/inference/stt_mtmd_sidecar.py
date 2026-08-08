# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp multimodal (mtmd) speech-to-text sidecar for Studio dictation.

whisper.cpp loads only the Whisper architecture, so newer ASR models run through
llama.cpp instead: a text model plus an audio mmproj, served by `llama-server`
and driven by the OpenAI audio content part.

Owns one `llama-server` on an ephemeral 127.0.0.1 port, loaded on demand and
unloaded after the same keep-alive as the other sidecars. Each model is two GGUF
files, so downloads run the shared worker twice.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from loggers import get_logger

from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid

from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttLoadCancelledError,
    SttModelBusyError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    _SelectedHubFile,
    _capture_stt_hub_cache,
    _claim_stt_repository,
    _decode_audio_bounded,
    _downloaded_file_bytes,
    _fallback_revisions,
    _HF_COMMIT_SHA,
    _prepare_stt_cache_for_http,
    _read_revision_record,
    _TARGET_SAMPLE_RATE,
    _training_active,
    _write_revision_record,
    normalize_whisper_language,
)

logger = get_logger(__name__)


@dataclass(frozen = True)
class MtmdSttModel:
    repo: str
    model_file: str
    mmproj_file: str
    label: str
    # Qwen3-ASR emits "language English<asr_text>" before the transcript.
    transcript_marker: Optional[str] = None


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
# Voxtral Mini is left out: in chat mode it answers the audio instead of
# transcribing it, and drops sentences when it complies. Parakeet and Nemotron
# ASR are too: llama.cpp has the audio graphs but not the text architectures.

_TRANSCRIBE_PROMPT = "Transcribe the audio."
# Output cap per second of audio. Speech runs about 3 tokens a second in
# English and more in scripts with no word boundaries, so this is deliberately
# generous: generation stops at EOS long before it, and the cap only exists so
# a looping model cannot run to the request timeout.
_TRANSCRIPT_TOKENS_PER_SECOND = 30
_MIN_TRANSCRIPT_TOKENS = 512
# Well under any of these models' trained context, which also has to hold the
# audio. llama-server is left on its default context (loaded from the model).
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
        # No PyAV means every transcription 501s on decode, so offering a
        # multi-gigabyte download here would be a waste.
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
    """Stop a child and wait for it, so its port and VRAM are actually free.

    terminate() alone returns before the process has gone, and a child that
    ignores SIGTERM would hold both until Studio exits.
    """
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
        # The PID is dead, so drop it before it can be reused by something else
        # that terminate_all would then signal.
        forget_pid(process.pid)


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


# The panel polls every model every 750ms, and each answer is two hf_hub_download()
# calls that stat the snapshot even local-only, so memoise the boolean briefly.
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


class _MtmdDownloadState:
    """Tracks the two-file download of one mtmd dictation model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._selected_files: tuple[_SelectedHubFile, ...] = ()
        self._revision: Optional[str] = None
        self._hub_cache: Optional[Path] = None
        self._cancelled = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            model_id = self._model_id
            snapshot = {
                "downloading": downloading,
                "model": model_id if downloading else None,
                "error": self._error,
                "cancelled": self._cancelled,
                "bytes_total": self._total_bytes if downloading else None,
            }
        # Outside the lock: _downloaded_bytes() stats the cache, and a cancel must not queue.
        snapshot["bytes_done"] = self._downloaded_bytes(model_id) if downloading else None
        return snapshot

    def cancel(self) -> bool:
        """Stop an in-flight download. False when none was running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            from core.inference.stt_download_worker import terminate_download
            terminate_download(process)
        return True

    def _downloaded_bytes(self, model_id: Optional[str] = None) -> Optional[int]:
        """Count only the two selected files in their three cache forms.

        model_id lets status() call this without holding the lock.
        """
        try:
            model_id = model_id or self._model_id
            hub_cache = self._hub_cache
            selected_files = self._selected_files
            if not model_id or hub_cache is None or not selected_files:
                return None
            repo = MTMD_STT_MODELS[model_id].repo
            done = sum(
                _downloaded_file_bytes(
                    hub_cache = hub_cache,
                    repo = repo,
                    filename = selected.path,
                    size = selected.size,
                    blob_key = selected.blob_key,
                    revision = self._revision,
                )
                for selected in selected_files
            )
            total = self._total_bytes
            return min(done, total) if total is not None else done
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = resolve_mtmd_model_id(model_id)
        hub_cache = _capture_stt_hub_cache()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    # A cancelled run keeps _cancelled set, so joining it would
                    # silently download nothing. Ask for a retry instead.
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

    def _run(
        self,
        model_id: str,
        hf_token: Optional[str],
        hub_cache: Optional[Path] = None,
    ) -> None:
        if hub_cache is None:
            from core.inference.stt_sidecar import _active_hf_hub_cache
            hub_cache = self._hub_cache or _active_hf_hub_cache()
        spec = MTMD_STT_MODELS[model_id]
        registry = None
        owner = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            selected: list[_SelectedHubFile] = []
            revision: Optional[str] = None
            for filename in (spec.model_file, spec.mmproj_file):
                meta = get_hf_file_metadata(
                    hf_hub_url(spec.repo, filename, revision = revision),
                    token = hf_token or None,
                )
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
            # A cancel during metadata has no child to stop, so check it here:
            # claiming and preparing the cache would mutate it after the user
            # was told the download stopped.
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
                if (
                    _cached_model_paths(
                        model_id,
                        hub_cache = hub_cache,
                        revision = revision,
                    )
                    is None
                ):
                    raise RuntimeError("downloaded mtmd files are missing from the captured cache")
                _write_revision_record(spec.repo, revision)
                return
            with self._lock:
                if cancelled or process.returncode < 0:
                    self._cancelled = True
                    return
            detail = stderr.decode("utf-8", "replace").strip()
            logger.warning("mtmd STT download failed for %s: %s", model_id, detail)
            with self._lock:
                self._error = f"Download failed for '{model_id}'."
        except Exception as exc:
            with self._lock:
                if not self._cancelled:
                    logger.warning("mtmd STT download failed for %s: %s", model_id, exc)
                    self._error = f"Download failed for '{model_id}'."
        finally:
            # Release first: it is the half that would wedge the repository.
            if registry is not None and owner is not None:
                registry.release_repository_owner(spec.repo, owner)
            # The memo is stale however this ended; dropping it now lets the next poll settle.
            _forget_downloaded_probe(model_id)


_download_state = _MtmdDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> None:
    _download_state.start(resolve_mtmd_model_id(model), hf_token)


def download_status() -> dict:
    return _download_state.status()


def cancel_model_download() -> bool:
    return _download_state.cancel()


class MtmdSttSidecar:
    """One llama-server process serving the selected mtmd dictation model."""

    def __init__(self, keep_alive_seconds: float = STT_KEEP_ALIVE_SECONDS) -> None:
        self._lock = threading.RLock()
        # Held across a whole startup; _lock is released while llama-server boots.
        self._start_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_id: Optional[str] = None
        self._loading = False
        # Published as soon as Popen returns, so a startup can be preempted
        # before _process is assigned (readiness takes up to three minutes).
        self._starting_process: Optional[subprocess.Popen] = None
        # Whether the resident server was launched with the GPU pinned off for
        # training. Kept so a dictation after the run does not stay on CPU.
        self._gpu_disabled = False
        self._load_cancel_event: Optional[threading.Event] = None
        self._update_in_progress = False
        self._keep_alive_seconds = keep_alive_seconds
        self._idle_timer: Optional[threading.Timer] = None
        self._generation = 0
        # Transcription runs outside _lock and can outlast the keep-alive, so
        # the idle timer stays disarmed while any request is in flight.
        self._active_requests = 0

    @property
    def loaded_model(self) -> Optional[str]:
        # Lock-free: _lock is held across reaps and llama.cpp installs, but the status
        # route reads this on the event loop. _process_alive() snapshots _process before
        # poll(), so a concurrent unload is safe.
        return self._model_id if self._process_alive() else None

    @property
    def device(self) -> Optional[str]:
        # Derived, not a second probe: two probes can straddle the publish and
        # report a device with no model.
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
            # Reap before clearing: a lock-free reader mid-reap must still see the
            # model, or training starts while the dying server still holds its VRAM.
            _reap(process)
        finally:
            self._process = None
            self._port = None
            self._model_id = None

    def unload(self) -> None:
        # A startup has not assigned _process yet, so releasing alone would let
        # it finish and republish the model that was just unloaded. Cancel and
        # settle outside _lock: load() holds _start_lock across startup and
        # takes _lock inside it, so holding _lock here would invert them.
        self.cancel_pending_load()
        self.wait_for_load_to_settle()
        with self._lock:
            self._release_locked()

    def cancel_pending_load(self) -> bool:
        """Preempt a starting llama-server so training is not raced for VRAM.

        _process is only assigned once the server answers /health, so unload()
        cannot reach a child that is still allocating. Startup runs outside
        _lock, so this acts without it: the event makes _wait_for_server give
        up, and load() then reaps the child.
        """
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

    def wait_for_load_to_settle(self) -> None:
        """Block until a cancelled startup has been reaped and its VRAM freed.

        load() holds _start_lock across startup and its cleanup, so taking it
        is the wait.
        """
        with self._start_lock:
            pass

    def _raise_if_update_in_progress(self) -> None:
        if self._update_in_progress:
            raise SttUnavailableError(
                "The local transcription runtime is being updated. Try dictation again shortly."
            )

    @contextmanager
    def update_maintenance(self) -> Iterator[bool]:
        """Block new loads while the llama.cpp tree this binary lives in is
        replaced. The chat backend coordinates its own server; this sidecar runs
        the same executable, so on Windows a live one blocks the swap.

        The guard is published before waiting for the locks, so a load already
        past its own check still cannot start a process against a half-swapped
        tree. Yields whether a warm server had to be unloaded.
        """
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
            raise SttModelNotDownloadedError(
                f"STT model '{model_id}' is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            )
        return paths

    def load(self, model: Optional[str] = None) -> None:
        self._raise_if_update_in_progress()
        model_id = resolve_mtmd_model_id(model)
        binary = ensure_engine_available()
        # Startup happens outside _lock (it is slow), so this keeps two callers
        # from each spawning a server and orphaning the first.
        with self._start_lock:
            self._raise_if_update_in_progress()
            self._load_locked(model_id, binary)

    def _load_locked(self, model_id: str, binary: str) -> None:
        with self._lock:
            training = _training_active()
            if self._process_alive() and self._model_id == model_id:
                # Same model, so only the offload mode can differ: a server
                # started at -ngl 0 during training would otherwise serve every
                # later dictation on CPU. Restarting for that is an
                # optimisation, never worth killing a running transcription
                # for, so an in-flight request keeps the server it has and the
                # next idle load picks the GPU back up.
                if self._gpu_disabled == training or self._active_requests:
                    self._schedule_idle_unload_locked()
                    return
            # Announced before the slow probe and reap: is_loading() is read lock-free,
            # so a training start would otherwise see False and wait out the startup in
            # unload() instead of cancelling this load.
            cancel_event = threading.Event()
            self._load_cancel_event = cancel_event
            self._loading = True
            released = False
            try:
                # Before the release: a 409 for a model that is not downloaded
                # must not cost the user the server they were already using.
                model_path, mmproj_path = self._ensure_model_downloaded(model_id)
                # Only when there is a live server to protect: a request against
                # a server that already died must not block recovery.
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
            # Re-read last: _release_locked() reaps the old server, which can
            # take seconds, and training admission that already passed its own
            # check cannot come back to cancel this load. Publishing _loading
            # first covers the other order, so between them every training start
            # either cancels this load or is seen by it.
            training = _training_active()
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
                # One short request at a time, so one slot keeps the footprint
                # down next to a loaded chat model.
                "--parallel",
                "1",
                # Keep off the accelerator during training (as whisper.cpp does)
                # so a dictation load cannot reclaim VRAM training just freed.
                "-ngl",
                "0" if training else "99",
            ]
            if training:
                # -ngl 0 covers the main model only. clip.cpp offloads the
                # projector on its own flag, which is what the chat backend's
                # _cmd_has_gpu_companion() treats as a GPU companion whatever
                # --gpu-layers says.
                cmd.append("--no-mmproj-offload")
            sock.close()
            process = subprocess.Popen(
                cmd,
                # Nothing reads these, and an undrained pipe blocks llama-server
                # mid-startup once its logs fill the buffer.
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                stdin = subprocess.DEVNULL,
                # Bundled libs and pip CUDA runtimes on the loader path, secrets
                # scrubbed, as the chat backend spawns the same binary.
                env = _llama_server_child_env(binary),
                # Die with Studio, so a crash never orphans a server on the GPU.
                **child_popen_kwargs(),
            )
            # Published before the wait, so training can preempt a startup that
            # is already allocating; _process is not set for another 180s.
            with self._lock:
                self._starting_process = process
            adopt_pid(process.pid)  # terminate_all backstop for graceful exits
            if not self._wait_for_server(process, port, cancel_event):
                # Reap it here: _process was never assigned, so unload() cannot
                # reach a child that ignores SIGTERM and keeps port and VRAM.
                _reap(process)
                if cancel_event.is_set():
                    # 409 through the route, like the other sidecars: expected
                    # preemption, not a broken or missing runtime (501).
                    raise SttLoadCancelledError(
                        "Dictation model loading was cancelled so training could start."
                    )
                raise SttUnavailableError(f"llama-server did not become ready for '{model_id}'.")
            with self._lock:
                self._process = process
                self._port = port
                self._model_id = model_id
                self._gpu_disabled = training
                self._generation += 1
                self._schedule_idle_unload_locked()
        finally:
            with self._lock:
                self._loading = False
                self._load_cancel_event = None
                self._starting_process = None

    @staticmethod
    def _wait_for_server(
        process: subprocess.Popen,
        port: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
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
                        return True
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
    ) -> dict:
        """Transcribe encoded audio bytes, as the other sidecars do.

        ``fast`` has no effect: decoding is greedy and the model picks the
        language itself.
        """
        ensure_engine_available()
        model_id = resolve_mtmd_model_id(model)
        # No training guard here on purpose: load() starts the server with
        # -ngl 0 --no-mmproj-offload while a run is active, so this transcribes
        # on CPU exactly as whisper.cpp and Transformers do. Refusing after a
        # preload that succeeded only discarded the user's recording.
        # Reject a missing model before decoding, matching the other sidecars.
        self._ensure_model_downloaded(model_id)
        decoded_audio = _decode_audio_bounded(audio)
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        audio_seconds = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        self.load(model_id)
        with self._lock:
            port = self._port
            if port is None or not self._process_alive():
                raise SttUnavailableError("The dictation server is not running.")
            # Another client can switch models in the gap between that load
            # returning and this lock, and the port read here would then be its
            # server. Refuse rather than transcribe on the wrong model.
            if self._model_id != model_id:
                raise SttModelBusyError(
                    "The dictation model changed while this recording was being "
                    "prepared. Try again."
                )
            # Long audio can outlast the keep-alive, and _post_transcribe runs
            # outside the lock, so disarm the timer rather than let it kill
            # llama-server mid-request and throw the dictation away.
            self._active_requests += 1
            self._cancel_idle_unload_locked()
        try:
            # Outside the lock: a held lock would block unload, including the
            # one a training run performs, for the whole request timeout.
            text = self._post_transcribe(port, model_id, wav_bytes, audio_seconds)
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

    def _post_transcribe(
        self,
        port: int,
        model_id: str,
        wav_bytes: bytes,
        audio_seconds: Optional[float] = None,
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
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data = json.dumps(payload).encode("utf-8"),
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout = _TRANSCRIBE_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
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
