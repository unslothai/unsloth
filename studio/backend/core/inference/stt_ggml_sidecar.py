# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp (GGML/GGUF) speech-to-text sidecar for Unsloth dictation.

Runs the same curated Whisper checkpoints as the Transformers sidecar
(stt_sidecar.py) through whisper.cpp's `whisper-server`, ~2.5x faster at
identical quality on Apple Silicon and CPU because its Metal/CPU kernels run
the weights in f16 where PyTorch MPS requires fp32.

Owns a single `whisper-server` subprocess bound to 127.0.0.1 on an ephemeral
port; the model loads on demand, stays warm between dictations, and unloads
after the same keep-alive as the Transformers sidecar. Curated GGML checkpoints
are single files from `unslothai/whisper-*-GGUF`, downloaded directly rather
than through the Model Hub (whose variant planner only handles `.gguf` chat
layouts).

Binary discovery mirrors `_find_llama_server_binary`: env override, then managed
Unsloth home, then PATH. With no binary the engine is unavailable and dictation
falls back to the Transformers sidecar; `scripts/build_whisper_cpp.sh` installs
the binary.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from loggers import get_logger

from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttAudioDecodeError,
    SttLanguageError,
    SttLoadCancelledError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    SttTranscriptionCancelledError,
    _capture_stt_hub_cache,
    _claim_stt_repository,
    _close_connection_on_cancel,
    _decode_audio_bounded,
    _downloaded_file_bytes,
    _fallback_revisions,
    _HF_COMMIT_SHA,
    _known_whisper_languages,
    _prepare_stt_cache_for_http,
    _read_revision_record,
    _TARGET_SAMPLE_RATE,
    _training_active,
    _write_revision_record,
    normalize_whisper_language,
)
from utils.prebuilt.child_env import isolate_home, scrub_env, wsl_system_rocm_lib_dirs
from utils.prebuilt.runtime_libs import dedupe_existing_dirs
from utils.prebuilt.whisper_layout import lookup_marker
from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid

logger = get_logger(__name__)

# Curated GGML checkpoints, one repo per model. Keys match the Transformers
# sidecar's ids so the frontend reuses one picker; values are the single file
# inside each repo.
GGML_STT_REPOS: dict[str, str] = {
    "tiny": "unslothai/whisper-tiny-GGUF",
    "base": "unslothai/whisper-base-GGUF",
    "small": "unslothai/whisper-small-GGUF",
    "large-v3-turbo": "unslothai/whisper-large-v3-turbo-GGUF",
    "large-v3": "unslothai/whisper-large-v3-GGUF",
}
GGML_STT_MODELS: dict[str, str] = {
    "tiny": "whisper-tiny.bin",
    "base": "whisper-base.bin",
    "small": "whisper-small.bin",
    "large-v3-turbo": "whisper-large-v3-turbo.bin",
    "large-v3": "whisper-large-v3.bin",
}
DEFAULT_GGML_STT_MODEL = "small"

_SERVER_START_TIMEOUT_SECONDS = 120.0
_TRANSCRIBE_TIMEOUT_SECONDS = 600.0


class SttEngineUnavailableError(SttUnavailableError):
    """whisper-server is not installed; the GGUF dictation engine is off."""


def resolve_ggml_model_id(model: Optional[str]) -> str:
    """Validate a curated GGML model id. Custom repos are not supported here."""
    if model is None or not str(model).strip():
        return DEFAULT_GGML_STT_MODEL
    normalized = str(model).strip()
    if normalized in GGML_STT_MODELS:
        return normalized
    raise SttModelIdError(
        f"STT model '{model}' is not a curated GGUF dictation model. "
        f"Choose one of: {', '.join(GGML_STT_MODELS)}."
    )


def _managed_whisper_cpp_dir() -> Path:
    """`<STUDIO_HOME>/whisper.cpp` in custom mode, else `~/.unsloth/whisper.cpp`.

    Mirrors `managed_node_dir` / `_find_llama_server_binary` so managed runtimes
    share one parent directory.
    """
    legacy = Path.home() / ".unsloth" / "whisper.cpp"
    try:
        from utils.paths.storage_roots import studio_root

        resolved = studio_root()
        legacy_studio = Path.home() / ".unsloth" / "studio"
        try:
            is_legacy = resolved.resolve() == legacy_studio.resolve()
        except (OSError, ValueError):
            is_legacy = resolved == legacy_studio
        return legacy if is_legacy else (resolved / "whisper.cpp")
    except (ImportError, OSError, ValueError):
        override = (
            os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
        ).strip()
        if override:
            try:
                return Path(override).expanduser().resolve() / "whisper.cpp"
            except (OSError, ValueError):
                return Path(override).expanduser() / "whisper.cpp"
        return legacy


def find_whisper_server_binary() -> Optional[str]:
    """Locate the whisper-server binary.

    Search order:
    1. WHISPER_SERVER_PATH environment variable (direct path to binary)
    2. UNSLOTH_WHISPER_CPP_PATH env var (custom whisper.cpp install dir)
    3. managed dir: <STUDIO_HOME or ~/.unsloth>/whisper.cpp/{,build/bin/}whisper-server
    4. whisper-server on PATH
    """
    binary_name = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"

    def _layout_candidates(d: Path) -> list[Path]:
        cands = [d / binary_name, d / "build" / "bin" / binary_name]
        if sys.platform == "win32":
            cands.append(d / "build" / "bin" / "Release" / binary_name)
        return cands

    env_path = os.environ.get("WHISPER_SERVER_PATH")
    if env_path:
        p = Path(env_path)
        if _is_runnable(p):
            return str(p)

    custom_dir = os.environ.get("UNSLOTH_WHISPER_CPP_PATH")
    if custom_dir:
        for p in _layout_candidates(Path(custom_dir)):
            if _is_runnable(p):
                return str(p)

    for p in _layout_candidates(_managed_whisper_cpp_dir()):
        if _is_runnable(p):
            return str(p)

    return shutil.which(binary_name)


def _is_runnable(p: Path) -> bool:
    """A real whisper-server is an executable file. On Windows os.access(X_OK) is
    effectively an existence check; on Unix it rejects a non-executable stub so a
    half-written or wrong-mode file isn't mistaken for the server."""
    try:
        return p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK))
    except OSError:
        # is_file() propagates EACCES: an unreadable install dir must read as
        # engine-unavailable, like a missing one, never a 500 out of stt/status.
        return False


def _whisper_install_marker(binary: str) -> Optional[dict]:
    """The prebuilt install marker above ``binary``, or None (source/custom builds)."""
    return lookup_marker(binary).marker


def slim_runtime_intact(binary: str) -> bool:
    """True unless the marker says slim and the linked ggml runtime is missing
    beside the server. New markers record the exact wired filenames
    (linked_libraries), all of which must be present; legacy markers without the
    field fall back to the per-OS core ggml name globs. A broken slim install
    reads as engine-unavailable (reinstall via `unsloth studio update`), never a
    crash at load."""
    lookup = lookup_marker(binary)
    marker = lookup.marker
    if lookup.invalid or marker is None:
        return not lookup.slim_collision
    if not marker or marker.get("install_kind") != "slim":
        return True
    if lookup.authoritative:
        valid = marker.get("component") == "whisper.cpp"
        valid = valid and isinstance(marker.get("schema_version"), int)
        valid = valid and all(
            isinstance(marker.get(key), str) and marker[key]
            for key in ("release_tag", "backend", "paired_llama_tag")
        )
        valid = valid and isinstance(marker.get("linked_libraries"), list)
        valid = valid and bool(marker.get("linked_libraries"))
        valid = valid and all(
            isinstance(name, str) and name and Path(name).name == name
            for name in marker["linked_libraries"]
        )
        if not valid:
            return False
    bin_dir = Path(binary).parent
    linked = marker.get("linked_libraries")
    if isinstance(linked, list) and linked and all(isinstance(name, str) for name in linked):
        intact = all((bin_dir / name).is_file() for name in linked)
    else:
        if sys.platform == "win32":
            required = ("ggml.dll", "ggml-base.dll")
        elif sys.platform == "darwin":
            required = ("libggml*.dylib", "libggml-base*.dylib")
        else:
            required = ("libggml.so*", "libggml-base.so*")
        intact = all(any(p.is_file() for p in bin_dir.glob(pattern)) for pattern in required)
    runtime_dirs = marker.get("linked_runtime_directories")
    if intact and isinstance(runtime_dirs, list) and runtime_dirs:
        intact = all(
            isinstance(name, str)
            and name
            and (bin_dir / name).is_dir()
            and any(path.is_file() for path in (bin_dir / name).rglob("*"))
            for name in runtime_dirs
        )
    if intact and marker.get("backend") == "rocm":
        # Membership plus required, not equality: hipBLASLt builds no Tensile
        # kernels for gfx1030 and the rest of RDNA2, so llama's ROCm bundle for
        # those ships libhipblaslt with no hipblaslt/ catalog, and the installer
        # wires only the catalogs the bundle has; demanding both read a correct
        # install as broken (#8364). rocblas stays mandatory (the backend module
        # links librocblas directly) and a name outside the pair still means
        # stale wiring. Windows overlays wire no catalogs, so both sets are
        # empty there and this reduces to the old equality.
        known_runtime_dirs = set() if sys.platform == "win32" else {"hipblaslt", "rocblas"}
        required_runtime_dirs = set() if sys.platform == "win32" else {"rocblas"}
        # Any wiring from version 2 on records linked_runtime_directories, so
        # pin the floor, not one version, or an installer bump strands installs.
        wiring_version = marker.get("runtime_wiring_version")
        intact = (
            isinstance(wiring_version, int)
            and wiring_version >= 2
            and isinstance(runtime_dirs, list)
            and set(runtime_dirs) <= known_runtime_dirs
            and required_runtime_dirs <= set(runtime_dirs)
        )
    if not intact:
        logger.warning(
            "slim whisper install is missing its linked ggml runtime at "
            f"{bin_dir}; run `unsloth studio update` to reinstall it"
        )
    return intact


# A runtime that starts, answers GET /, and then dies on the first actual inference.
# Reported on Windows with ROCm on gfx1200, where rocBLAS is missing its TensileLibrary:
# the marker and the linked libraries are all present, so slim_runtime_intact() is happy
# and is_available() said yes, which meant _resolve_serving_stt_engine never fell back and
# every recording 501'd while the UI showed the model as loaded. Only inference can prove
# this, so it is recorded when inference fails and cleared when one succeeds. Process
# lifetime by design: a reinstall restarts Unsloth.
_runtime_inference_failure: Optional[str] = None
_runtime_failure_lock = threading.Lock()


def note_runtime_inference_failure(reason: str) -> None:
    global _runtime_inference_failure
    with _runtime_failure_lock:
        if _runtime_inference_failure is None:
            logger.warning(
                "whisper.cpp runtime failed to serve a transcription (%s); "
                "treating the engine as unavailable and using Transformers instead",
                reason,
            )
        _runtime_inference_failure = reason


def clear_runtime_inference_failure() -> None:
    global _runtime_inference_failure
    with _runtime_failure_lock:
        _runtime_inference_failure = None


def runtime_inference_failure() -> Optional[str]:
    with _runtime_failure_lock:
        return _runtime_inference_failure


def is_available() -> bool:
    binary = find_whisper_server_binary()
    if binary is None:
        return False
    if not slim_runtime_intact(binary):
        return False
    if runtime_inference_failure() is not None:
        return False
    try:
        import av  # noqa: F401
    except Exception:
        # No PyAV means every transcription 501s on decode.
        return False
    return True


def ensure_engine_available() -> str:
    binary = find_whisper_server_binary()
    if binary is None:
        raise SttEngineUnavailableError(
            "The local transcription runtime is not installed. Run "
            "`unsloth studio update` to install it."
        )
    if not slim_runtime_intact(binary):
        raise SttEngineUnavailableError(
            "The local transcription runtime is missing its paired ggml "
            "libraries. Run `unsloth studio update` to reinstall it."
        )
    return binary


# ---------------------------------------------------------------------------
# whisper-server child-process environment
# ---------------------------------------------------------------------------
# Build the whisper-server env: prepend the binary dir (co-located libs win, and
# a backstop where the loader ignores the rpath) and scrub secret-bearing vars the
# binary never needs. On WSL2 ROCm the system HIP libs go first, since a bundle's
# bare-metal HIP cannot drive /dev/dxg. A CUDA bundle ships libggml-cuda.so but not
# libcudart/libcublas (paired with the user's PyTorch), so add the
# CUDA-from-PyTorch runtime dirs the selection gated on, else the backend cannot
# resolve a runtime that lives only in wheels. Mirrors llama's binary_env(); the
# scrub/WSL/dedupe helpers live in utils.prebuilt.

# Module-level aliases keep the historical patch points for tests and callers.
_wsl_system_rocm_lib_dirs = wsl_system_rocm_lib_dirs
_dedupe_existing_dirs = dedupe_existing_dirs


def _whisper_server_child_env(binary: str) -> dict[str, str]:
    """Env for the whisper-server subprocess: secrets scrubbed, home/profile vars
    repointed at a managed scratch dir (a downloaded binary must not see the real
    home's token caches), co-located libs on the loader path, WSL system HIP first
    on WSL2 ROCm."""
    env = scrub_env(os.environ)
    isolate_home(env, str(_managed_whisper_cpp_dir() / ".child_home"))
    bin_dir = str(Path(binary).parent)
    # A CUDA bundle needs the CUDA-from-PyTorch wheel dirs so libcudart/libcublas
    # resolve at launch when they live only in site-packages/nvidia/*/lib. Placed
    # after bin_dir so co-located libs still win; empty for other bundles.
    cuda_runtime_dirs: list[str] = []
    bundle_dir = Path(bin_dir)
    has_cuda_module = any(
        path.is_file()
        for pattern in ("libggml-cuda.so*", "ggml-cuda*.dll")
        for path in bundle_dir.glob(pattern)
    )
    if has_cuda_module:
        try:
            from utils.prebuilt.runtime_libs import python_runtime_dirs
            cuda_runtime_dirs = python_runtime_dirs()
        except Exception:
            cuda_runtime_dirs = []
    if sys.platform == "win32":
        var, lead = "PATH", [bin_dir, *cuda_runtime_dirs]
    elif sys.platform == "darwin":
        var, lead = "DYLD_LIBRARY_PATH", [bin_dir]
    else:
        var, lead = "LD_LIBRARY_PATH", [bin_dir, *cuda_runtime_dirs]
        wsl_rocm = _wsl_system_rocm_lib_dirs()
        if wsl_rocm:
            lead = [*wsl_rocm, bin_dir, *cuda_runtime_dirs]
            env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    existing = [p for p in env.get(var, "").split(os.pathsep) if p]
    env[var] = os.pathsep.join(_dedupe_existing_dirs([*lead, *existing]))
    return env


# ---------------------------------------------------------------------------
# Model file download (single files; deliberately outside the Model Hub flow)
# ---------------------------------------------------------------------------


def _cached_model_path(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Path of a fully downloaded GGML file in the shared HF cache, else None."""
    from huggingface_hub import hf_hub_download

    from core.inference.stt_sidecar import _active_hf_hub_cache

    repo_id = GGML_STT_REPOS[model_id]
    root = hub_cache if hub_cache is not None else _active_hf_hub_cache()

    def cached_at(candidate_revision: str) -> Optional[str]:
        try:
            return hf_hub_download(
                repo_id = repo_id,
                filename = GGML_STT_MODELS[model_id],
                revision = candidate_revision,
                local_files_only = True,
                cache_dir = str(root),
            )
        except Exception:
            return None

    # Explicit revisions keep verification on the downloaded commit.
    if revision is not None:
        return cached_at(revision)

    recorded = _read_revision_record(repo_id)
    if recorded:
        cached = cached_at(recorded)
        if cached is not None:
            return cached

    for candidate in _fallback_revisions(repo_id, hub_cache = root):
        cached = cached_at(candidate)
        if cached is not None:
            _write_revision_record(repo_id, candidate)
            return cached
    return None


class _GgmlDownloadState:
    """Tracks one background hf_hub_download of a curated GGML file."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._etag: Optional[str] = None
        self._revision: Optional[str] = None
        self._hub_cache: Optional[Path] = None
        self._cancelled = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            snapshot = {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "cancelled": self._cancelled,
                # Which model the cancel applies to. "model" goes None once the worker
                # thread stops, so a settled cancellation was indistinguishable from an
                # unrelated one and a deferred load restarted the whole download.
                "cancelled_model": self._model_id if self._cancelled else None,
                "bytes_total": self._total_bytes if downloading else None,
            }
            captured = (
                self._model_id,
                self._etag,
                self._total_bytes,
                self._hub_cache,
                self._revision,
            )
        # Outside the lock: _downloaded_bytes() stats the cache, and a cancel must not queue.
        snapshot["bytes_done"] = self._downloaded_bytes(*captured) if downloading else None
        return snapshot

    def cancel(self) -> bool:
        """Stop an in-flight download. False when none was running.

        The partial blob stays cached, so a restart resumes from it.
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
        model_id: Optional[str] = None,
        etag: Optional[str] = None,
        total: Optional[int] = None,
        hub_cache: Optional[Path] = None,
        revision: Optional[str] = None,
    ) -> Optional[int]:
        """Count this file across partial, finalized, and snapshot locations.

        status() captures these under the lock and passes them in: reading them
        here would let a run that starts mid-probe pair its bytes with the total
        of the run that just ended.
        """
        try:
            model_id = model_id if model_id is not None else self._model_id
            etag = etag if etag is not None else self._etag
            total = total if total is not None else self._total_bytes
            hub_cache = hub_cache if hub_cache is not None else self._hub_cache
            revision = revision if revision is not None else self._revision
            if not model_id or not etag or total is None or hub_cache is None:
                return None
            return _downloaded_file_bytes(
                hub_cache = hub_cache,
                repo = GGML_STT_REPOS[model_id],
                filename = GGML_STT_MODELS[model_id],
                size = total,
                blob_key = etag,
                revision = revision,
            )
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = resolve_ggml_model_id(model_id)
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
                    f"Another GGUF dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._error = None
            self._total_bytes = None
            self._etag = None
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
        repo_id = GGML_STT_REPOS[model_id]
        filename = GGML_STT_MODELS[model_id]
        registry = None
        owner = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            try:
                meta = get_hf_file_metadata(hf_hub_url(repo_id, filename), token = hf_token or None)
                total_bytes = int(meta.size or 0)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError("could not resolve GGML download metadata") from exc
            revision = meta.commit_hash
            etag = meta.etag
            if not isinstance(revision, str) or not _HF_COMMIT_SHA.fullmatch(revision):
                raise RuntimeError("could not resolve an immutable GGML revision")
            if not isinstance(etag, str) or not etag:
                raise RuntimeError("could not resolve the GGML blob identity")
            if total_bytes <= 0:
                raise RuntimeError("could not resolve the GGML file size")
            # A cancel during metadata has no child to stop. Without these the
            # run still reserves the repo and rewrites the cache after the stop.
            with self._lock:
                if self._cancelled:
                    return
            registry, owner = _claim_stt_repository(repo_id)
            with self._lock:
                if self._cancelled:
                    return
            _prepare_stt_cache_for_http(repo_id, hub_cache)
            with self._lock:
                self._total_bytes = total_bytes
                self._etag = etag
                self._revision = revision
            # Out of process so cancel() can terminate it; a thread blocked in
            # hf_hub_download could not be interrupted.
            from core.inference.stt_download_worker import (
                reap_download,
                spawn_download,
                terminate_download,
            )

            args = [
                "--repo-id",
                repo_id,
                "--revision",
                revision,
                "--filename",
                filename,
            ]
            process = spawn_download(
                args,
                hf_token = hf_token or None,
                hub_cache = hub_cache,
            )
            with self._lock:
                if self._cancelled:
                    # cancel() landed between start() and the spawn.
                    terminate_download(process)
                self._process = process
            # reap_download(), not communicate(): only it drops the adopted PID,
            # which could otherwise be reused and then signalled by terminate_all.
            stderr = reap_download(process)
            with self._lock:
                if self._process is process:
                    self._process = None
                cancelled = self._cancelled
            if process.returncode == 0 and not cancelled:
                if (
                    _cached_model_path(
                        model_id,
                        hub_cache = hub_cache,
                        revision = revision,
                    )
                    is None
                ):
                    raise RuntimeError("downloaded file is missing from the captured cache")
                _write_revision_record(repo_id, revision)
                return
            with self._lock:
                if cancelled or process.returncode < 0:
                    self._cancelled = True
                    return
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            logger.warning("GGUF STT download failed for %s: %s", model_id, detail)
            with self._lock:
                self._error = f"Download failed for '{model_id}'."
        except Exception as exc:
            with self._lock:
                if not self._cancelled:
                    logger.warning("GGUF STT download failed for %s: %s", model_id, exc)
                    self._error = f"Download failed for '{model_id}'."
        finally:
            if registry is not None and owner is not None:
                registry.release_repository_owner(repo_id, owner)


_download_state = _GgmlDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> None:
    _download_state.start(resolve_ggml_model_id(model), hf_token)


def download_status() -> dict:
    return _download_state.status()


def cancel_model_download() -> bool:
    return _download_state.cancel()


# ---------------------------------------------------------------------------
# WAV packaging
# ---------------------------------------------------------------------------


def _pcm_to_wav_bytes(decoded_audio) -> bytes:
    """Wrap decoded float32 mono 16 kHz PCM into an in-memory 16-bit WAV."""
    import numpy as np

    clipped = np.clip(decoded_audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_TARGET_SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


class GgmlSttSidecar:
    """Owns one whisper-server subprocess and proxies dictation to it."""

    def __init__(self, keep_alive_seconds: float = STT_KEEP_ALIVE_SECONDS) -> None:
        self._lock = threading.RLock()
        self._load_state_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_id: Optional[str] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0
        self._keep_alive_seconds = keep_alive_seconds
        # Set while whisper-server starts so training admission can account for
        # the accelerator memory it is about to bind. Read without the lock.
        self._loading = False
        # A still-starting whisper-server is cancellable so training can preempt
        # it before it binds accelerator memory. Assigned inside self._lock but
        # acted on without it: cancel_pending_load() runs while load() holds the
        # lock, so the event is the source of truth and terminating the process
        # is a best-effort fast path.
        self._load_cancel_event: Optional[threading.Event] = None
        self._load_owner_cancel_event: Optional[threading.Event] = None
        self._starting_process: Optional[subprocess.Popen] = None
        # Set before the updater waits for _lock, then kept set while it owns
        # the lock and atomically replaces the managed install tree. New loads
        # fail fast instead of starting a process from files being swapped.
        self._update_in_progress = False

    @property
    def loaded_model(self) -> Optional[str]:
        # Lock-free status read (like stt_sidecar.py): transcribe() holds
        # self._lock for the whole inference call (up to
        # _TRANSCRIBE_TIMEOUT_SECONDS), and status polls plus training admission
        # must not block behind it. _process_alive() snapshots self._process
        # before poll(), which subprocess guards with _waitpid_lock, so a
        # concurrent unload is safe.
        return self._model_id if self._process_alive() else None

    @property
    def device(self) -> Optional[str]:
        return "whisper.cpp" if self._process_alive() else None

    def is_loading(self) -> bool:
        # True only while whisper-server is starting (seconds to bind its GPU
        # backend); load() sets and clears the flag around that window.
        with self._load_state_lock:
            return self._loading

    @property
    def keep_alive_seconds(self) -> float:
        return self._keep_alive_seconds

    def _process_alive(self) -> bool:
        # Snapshot self._process once: a concurrent unload() nulls it under the
        # lock, so lock-free readers would otherwise re-read None between the
        # truthiness check and .poll().
        process = self._process
        return process is not None and process.poll() is None

    # -- idle unload ------------------------------------------------------

    def _cancel_idle_unload_locked(self) -> None:
        self._idle_generation += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if not self._process_alive():
            return
        generation = self._idle_generation
        timer = threading.Timer(self._keep_alive_seconds, self._idle_unload, args = (generation,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        with self._lock:
            if generation != self._idle_generation:
                return
            logger.info("Unloading idle GGUF STT model %s", self._model_id)
            self._release_locked()

    # -- process lifecycle -------------------------------------------------

    def _release_locked(self) -> None:
        self._cancel_idle_unload_locked()
        process = self._process
        self._process = None
        self._port = None
        self._model_id = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout = 10)
        if process is not None:
            forget_pid(process.pid)

    def _holds_expected_model(self, expected: Optional[str]) -> bool:
        """Whether the resident model is the one the caller claimed. Call under ``_lock``.

        A caller that owns a specific model must not release whatever happens to be
        resident: another surface can switch the engine between the ownership check and
        the request reaching the sidecar.
        """
        if expected is None:
            return True
        current = self._model_id
        if current is None:
            return False
        if current == expected:
            return True
        try:
            return current == resolve_ggml_model_id(expected)
        except Exception:  # noqa: BLE001 - an unresolvable name is not this model
            return False

    def unload(
        self,
        wait: bool = True,
        expected_model: Optional[str] = None,
    ) -> None:
        """Release the resident model. ``wait=False`` skips a sidecar mid-request.

        `transcribe` holds ``_lock`` across the whole round trip, so a caller releasing
        engines it does not own must not block behind one. ``expected_model`` scopes the
        release to one model, compared under the lock.
        """
        if not self._lock.acquire(blocking = wait):
            return
        try:
            if not self._holds_expected_model(expected_model):
                return
            self._release_locked()
        finally:
            self._lock.release()

    def _raise_if_update_in_progress(self) -> None:
        if self._update_in_progress:
            raise SttEngineUnavailableError(
                "The local transcription runtime is being updated. Try dictation again shortly."
            )

    @contextmanager
    def update_maintenance(self) -> Iterator[bool]:
        """Block new loads while the managed whisper.cpp tree is replaced.

        The flag is published before waiting for an existing transcription to
        release ``_lock``. Holding that lock across the yielded installer phase
        prevents Windows from relocking the executable and prevents every host
        from starting a process against a partially swapped tree. The yielded
        value records whether a warm model had to be unloaded.
        """
        self._update_in_progress = True
        try:
            with self._lock:
                model_was_active = self._process_alive()
                self._release_locked()
                yield model_was_active
        finally:
            self._update_in_progress = False

    def cancel_pending_load(self) -> bool:
        # Preempt a starting whisper-server so training does not launch while it
        # binds accelerator memory. load() holds self._lock for the whole startup,
        # so act without the lock: signal abort and terminate the starting
        # process. _wait_for_server observes the event and raises, then load()
        # reaps the process and releases the lock.
        with self._load_state_lock:
            event = self._load_cancel_event
            if not self._loading or event is None:
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
        with self._load_state_lock:
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
        # load() holds self._lock across startup and cancel cleanup, so acquiring
        # it blocks until a cancelled server is killed, reaped, and its
        # accelerator memory released.
        with self._lock:
            pass

    @staticmethod
    def _reserve_free_port() -> tuple[socket.socket, int]:
        """Bind an ephemeral port and keep the socket held.

        The caller closes the reservation immediately before spawning
        whisper-server, shrinking the window in which another local process
        could bind the port. SO_REUSEADDR lets the child rebind right after.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s, s.getsockname()[1]

    def _ensure_model_downloaded(self, model_id: str) -> str:
        path = _cached_model_path(model_id)
        if path is None:
            raise SttModelNotDownloadedError(
                f"STT model '{model_id}' (GGUF) is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            )
        return path

    def load(
        self,
        model: Optional[str] = None,
        request_cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Start (or switch) whisper-server for the requested curated model."""
        if request_cancel_event is not None and request_cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        self._raise_if_update_in_progress()
        model_id = resolve_ggml_model_id(model)
        with self._lock:
            if request_cancel_event is not None and request_cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            self._raise_if_update_in_progress()
            binary = ensure_engine_available()
            if self._process_alive() and self._model_id == model_id:
                self._schedule_idle_unload_locked()
                return
            model_path = self._ensure_model_downloaded(model_id)
            reservation, port = self._reserve_free_port()
            command = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port)]
            marker = _whisper_install_marker(binary)
            if _training_active():
                # Keep whisper.cpp off the accelerator during training (like the
                # Transformers sidecar's CPU choice) so a mid-training dictation
                # cannot reclaim the VRAM training just freed.
                command.append("--no-gpu")
            elif marker is not None and marker.get("backend") == "cpu":
                # A deliberate CPU install must stay CPU: the slim wiring links
                # every llama ggml backend (including CUDA/ROCm), so without
                # this flag a cpu-selected install would still grab the GPU.
                command.append("--no-gpu")
            logger.info(
                "Starting whisper-server for STT model %s on 127.0.0.1:%s",
                model_id,
                port,
            )
            cancel_event = (
                request_cancel_event if request_cancel_event is not None else threading.Event()
            )
            with self._load_state_lock:
                self._load_cancel_event = cancel_event
                self._load_owner_cancel_event = request_cancel_event
                self._loading = True
            try:
                if cancel_event.is_set():
                    raise SttLoadCancelledError("GGUF STT model loading was cancelled.")
                self._release_locked()
                # Release the reservation as late as possible: whisper-server
                # binds the port moments after this close.
                reservation.close()
                process = subprocess.Popen(
                    command,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                    stdin = subprocess.DEVNULL,
                    # Co-located GPU libs on the loader path (WSL system HIP first),
                    # secrets scrubbed from the downloaded binary's env.
                    env = _whisper_server_child_env(binary),
                    # Die with Unsloth (Linux PDEATHSIG, Windows job) so a crash
                    # never orphans a server holding the model.
                    **child_popen_kwargs(),
                )
                with self._load_state_lock:
                    self._starting_process = process
                adopt_pid(process.pid)  # terminate_all backstop for graceful exits
                try:
                    self._wait_for_server(process, port, cancel_event)
                except Exception:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout = 10)
                    forget_pid(process.pid)
                    raise
                self._process = process
                self._port = port
                self._model_id = model_id
                self._schedule_idle_unload_locked()
            finally:
                reservation.close()  # no-op when already released before spawn
                with self._load_state_lock:
                    self._loading = False
                    self._load_cancel_event = None
                    self._load_owner_cancel_event = None
                    self._starting_process = None

    @staticmethod
    def _wait_for_server(
        process: subprocess.Popen,
        port: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise SttLoadCancelledError(
                    "GGUF STT model loading was cancelled so training could start."
                )
            if process.poll() is not None:
                raise SttEngineUnavailableError(
                    "The local transcription runtime exited before becoming "
                    "ready; the model file may be corrupt or unsupported."
                )
            # Require a whisper-server-specific response twice, with the managed
            # child alive around each probe. An arbitrary local process that won
            # the bind race would otherwise be mistaken for the sidecar and
            # receive the user's microphone audio.
            if GgmlSttSidecar._probe_is_whisper_server(process, port) and (
                GgmlSttSidecar._probe_is_whisper_server(process, port)
            ):
                return
            time.sleep(0.2)
        raise SttEngineUnavailableError("The local transcription runtime did not start in time.")

    @staticmethod
    def _probe_is_whisper_server(process: subprocess.Popen, port: int) -> bool:
        """One readiness probe: our child is alive and the responder looks like
        whisper.cpp's server (its index page and errors identify whisper)."""
        if process.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", method = "GET")
            with urllib.request.urlopen(req, timeout = 2) as response:
                body = response.read(65536)
        except Exception:
            return False
        if process.poll() is not None:
            return False
        return b"whisper" in body.lower()

    # -- transcription ------------------------------------------------------

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Transcribe encoded audio bytes via whisper-server.

        Accepts any container PyAV can decode (same validation and caps as the
        Transformers sidecar). Returns {text, language, duration, model}.
        """
        self._raise_if_update_in_progress()
        ensure_engine_available()
        model_id = resolve_ggml_model_id(model)
        lang = normalize_whisper_language(language)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        known_languages = _known_whisper_languages()
        if lang is not None and known_languages is not None and lang not in known_languages:
            raise SttLanguageError(
                f"Language '{language}' is not supported by STT model '{model_id}'."
            )
        # Reject a missing model before decoding so a long clip does not burn CPU
        # only to 409 (matches the Transformers sidecar's preflight).
        self._ensure_model_downloaded(model_id)
        decoded_audio = _decode_audio_bounded(audio, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        with self._lock:
            try:
                if cancel_event is None:
                    self.load(model_id)
                else:
                    self.load(model_id, request_cancel_event = cancel_event)
                text = self._post_inference(wav_bytes, lang, fast, cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
            except Exception:
                if cancel_event is not None and cancel_event.is_set():
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
                raise
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": model_id,
        }

    def cancel_transcription(self, cancel_event: threading.Event) -> bool:
        already_cancelled = cancel_event.is_set()
        cancel_event.set()
        return self._cancel_owned_load(cancel_event) or not already_cancelled

    def _post_inference(
        self,
        wav_bytes: bytes,
        lang: Optional[str],
        fast: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        boundary = uuid.uuid4().hex
        fields = {
            "temperature": "0.0",
            "response_format": "json",
            # Match the Transformers sidecar: 5-way beam search, greedy for fast.
            "beam_size": "1" if fast else "5",
            "language": lang or "auto",
        }
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                (
                    f"--{boundary}\r\nContent-Disposition: form-data; "
                    f'name="{name}"\r\n\r\n{value}\r\n'
                ).encode()
            )
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                'name="file"; filename="dictation.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + wav_bytes
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        # http.client, not urllib: a cancel needs the socket, and urlopen exposes none.
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout = _TRANSCRIBE_TIMEOUT_SECONDS
        )
        cancel_done = threading.Event()
        if cancel_event is not None:
            threading.Thread(
                target = _close_connection_on_cancel,
                args = (connection, cancel_event, cancel_done),
                daemon = True,
            ).start()
        try:
            connection.request(
                "POST",
                "/inference",
                body = body,
                headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with connection.getresponse() as response:
                if not 200 <= response.status < 300:
                    raise SttEngineUnavailableError(
                        f"The local transcription runtime returned HTTP {response.status}."
                    )
                payload = json.loads(response.read().decode("utf-8"))
        except (SttAudioDecodeError, SttEngineUnavailableError):
            raise
        except Exception as exc:
            # A cancel closes this socket deliberately, so it is not evidence of a broken
            # runtime and must not disable the engine.
            if cancel_event is None or not cancel_event.is_set():
                note_runtime_inference_failure(f"{type(exc).__name__}: {exc}")
            raise SttEngineUnavailableError(
                "The local transcription runtime did not answer the request. "
                "Transcription will use the Transformers engine from now on."
            ) from exc
        finally:
            cancel_done.set()
            connection.close()
        text = payload.get("text")
        if not isinstance(text, str):
            raise SttAudioDecodeError("Could not decode the audio.")
        # It served a transcription, so whatever failed earlier was transient.
        clear_runtime_inference_failure()
        # whisper.cpp joins segments with newlines; dictation wants one line.
        return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


_sidecar: Optional[GgmlSttSidecar] = None


def get_ggml_stt_sidecar() -> GgmlSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = GgmlSttSidecar()
    return _sidecar
