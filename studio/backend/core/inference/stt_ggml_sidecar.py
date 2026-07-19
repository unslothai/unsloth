# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp (GGML/GGUF) speech-to-text sidecar for Studio dictation.

Runs the same curated Whisper checkpoints as the Transformers sidecar
(stt_sidecar.py) through whisper.cpp's `whisper-server`, which is ~2.5x faster
at identical transcription quality on Apple Silicon and CPU hosts because its
Metal/CPU kernels run the weights in f16 where PyTorch MPS requires fp32.

The sidecar owns a single `whisper-server` subprocess bound to 127.0.0.1 on an
ephemeral port. The model loads on demand, stays warm between dictations, and
unloads after the same keep-alive as the Transformers sidecar. Curated GGML
checkpoints are single files from the Unsloth-hosted `unslothai/whisper-*-GGUF`
repositories; they are downloaded directly, which is why this engine does not
go through the Model Hub flow -- the Hub's variant planner only handles
`.gguf` chat-model layouts.

Binary discovery mirrors `_find_llama_server_binary`: an explicit env override
first, then the managed Studio home, then PATH. When no binary is found the
engine reports unavailable and dictation falls back to the Transformers
sidecar; `scripts/build_whisper_cpp.sh` builds and installs the binary.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Optional

from loggers import get_logger

from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttAudioDecodeError,
    SttLanguageError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    _decode_audio_bounded,
    _known_whisper_languages,
    _TARGET_SAMPLE_RATE,
    _training_active,
    normalize_whisper_language,
)
from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid

logger = get_logger(__name__)

# Unsloth-hosted GGML checkpoints, one repository per curated model. Keys are
# the same stable ids the Transformers sidecar curates so the frontend can
# reuse one model picker; values are the single file inside each repository.
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
    """`<STUDIO_HOME>/whisper.cpp` in custom mode, else legacy `~/.unsloth/whisper.cpp`.

    Mirrors `managed_node_dir` / `_find_llama_server_binary` so every managed
    runtime shares one parent directory.
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
    half-written or wrong-mode file is not mistaken for the server."""
    return p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK))


def is_available() -> bool:
    if find_whisper_server_binary() is None:
        return False
    try:
        import av  # noqa: F401
    except Exception:
        # Decoding uploads needs PyAV; without it every transcription 501s.
        return False
    return True


def ensure_engine_available() -> str:
    binary = find_whisper_server_binary()
    if binary is None:
        raise SttEngineUnavailableError(
            "The local transcription runtime is not installed. Run "
            "`unsloth studio update` to install it."
        )
    return binary


# ---------------------------------------------------------------------------
# whisper-server child-process environment
# ---------------------------------------------------------------------------
# A prebuilt whisper-server co-locates its shared libs (libwhisper, libggml-*,
# and for GPU bundles the HIP/Vulkan backends) beside the binary under an
# $ORIGIN / @loader_path rpath. We still prepend the binary dir to the loader
# path as a backstop (and for hosts whose loader ignores the rpath), and scrub
# secret-bearing vars the downloaded binary never needs (models load via -m; the
# parent process keeps its own env for HF downloads). On WSL2 ROCm the system
# HIP libs go first: a bundle's bare-metal HIP cannot drive /dev/dxg and
# segfaults, so the WSL-capable libamdhip64/librocdxg must win while the bundle
# still supplies libggml-hip/librocblas. Mirrors install_llama_prebuilt.py's
# binary_env(); kept local so the sidecar need not import the installer CLI.
# (The CUDA-from-PyTorch runtime_line lib discovery lands when CUDA bundles ship;
# CPU/Metal P0 bundles are static and ROCm/Vulkan bundles are self-contained.)

_STT_SECRET_ENV_EXACT = frozenset({
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
    "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_SECRET", "KUBECONFIG", "SSH_AUTH_SOCK",
})
# Case-insensitive substring markers for names we do not enumerate (no bare "KEY").
_STT_SECRET_ENV_MARKERS = (
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASSPHRASE", "CREDENTIAL",
    "PRIVATE_KEY", "API_KEY",
)
# Proxy / index URLs embed creds in their value; the offline server never needs them.
_STT_SECRET_ENV_URL_NAMES = frozenset({
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "RSYNC_PROXY",
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
})
# Also drop values with URL userinfo creds (scheme://user:secret@host).
_STT_URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def _stt_is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return (
        upper in _STT_SECRET_ENV_EXACT
        or upper in _STT_SECRET_ENV_URL_NAMES
        or any(marker in upper for marker in _STT_SECRET_ENV_MARKERS)
    )


def _wsl_system_rocm_lib_dirs() -> list[str]:
    """System ROCm lib dir(s) to load before a bundle's HIP on WSL2. Strict no-op
    off WSL (needs /dev/dxg, a "microsoft" /proc/version, and a librocdxg)."""
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except OSError:
        return []
    dirs: list[str] = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            dirs.append(d)
    return dirs


def _dedupe_existing_dirs(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            p = Path(raw).expanduser()
            if not p.is_dir():
                continue
            resolved = str(p.resolve())
        except (OSError, ValueError):
            continue
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _whisper_server_child_env(binary: str) -> dict[str, str]:
    """Env for the whisper-server subprocess: secrets scrubbed, co-located libs on
    the loader path, WSL system HIP first on WSL2 ROCm."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not _stt_is_secret_env_name(k) and not _STT_URL_USERINFO_RE.search(v or "")
    }
    bin_dir = str(Path(binary).parent)
    if sys.platform == "win32":
        var, lead = "PATH", [bin_dir]
    elif sys.platform == "darwin":
        var, lead = "DYLD_LIBRARY_PATH", [bin_dir]
    else:
        var, lead = "LD_LIBRARY_PATH", [bin_dir]
        wsl_rocm = _wsl_system_rocm_lib_dirs()
        if wsl_rocm:
            lead = [*wsl_rocm, bin_dir]
            env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    existing = [p for p in env.get(var, "").split(os.pathsep) if p]
    env[var] = os.pathsep.join(_dedupe_existing_dirs([*lead, *existing]))
    return env


# ---------------------------------------------------------------------------
# Model file download (single files; deliberately outside the Model Hub flow)
# ---------------------------------------------------------------------------


def _cached_model_path(model_id: str) -> Optional[str]:
    """Path of a fully downloaded GGML file in the shared HF cache, else None."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(
            repo_id = GGML_STT_REPOS[model_id],
            filename = GGML_STT_MODELS[model_id],
            local_files_only = True,
        )
    except Exception:
        return None


class _GgmlDownloadState:
    """Tracks one background hf_hub_download of a curated GGML file."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._model_id: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._etag: Optional[str] = None

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            return {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "bytes_total": self._total_bytes if downloading else None,
                "bytes_done": self._incomplete_bytes() if downloading else None,
            }

    def _incomplete_bytes(self) -> Optional[int]:
        """Best-effort progress: size of the in-flight blob in the HF cache.

        hf_hub_download writes to ``blobs/<etag>.incomplete``; prefer the blob
        for this file's etag, falling back to the largest in-flight blob.
        """
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Callers may already hold self._lock (non-reentrant); bare
            # attribute reads are safe without it.
            model_id = self._model_id
            if not model_id:
                return None
            repo_dir = (
                Path(HF_HUB_CACHE)
                / f"models--{GGML_STT_REPOS[model_id].replace('/', '--')}"
                / "blobs"
            )
            if not repo_dir.is_dir():
                return None
            etag = self._etag
            if etag:
                target = repo_dir / f"{etag}.incomplete"
                if target.is_file():
                    return target.stat().st_size
            sizes = [p.stat().st_size for p in repo_dir.glob("*.incomplete") if p.is_file()]
            return max(sizes) if sizes else None
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = resolve_ggml_model_id(model_id)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    return
                raise SttModelIdError(
                    f"Another GGUF dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._error = None
            self._total_bytes = None
            self._etag = None
            thread = threading.Thread(target = self._run, args = (model_id, hf_token), daemon = True)
            self._thread = thread
            thread.start()

    def _run(self, model_id: str, hf_token: Optional[str]) -> None:
        repo_id = GGML_STT_REPOS[model_id]
        filename = GGML_STT_MODELS[model_id]
        try:
            from huggingface_hub import (
                get_hf_file_metadata,
                hf_hub_download,
                hf_hub_url,
            )
            try:
                # One HEAD request for the progress total and etag.
                meta = get_hf_file_metadata(hf_hub_url(repo_id, filename), token = hf_token or None)
                with self._lock:
                    self._total_bytes = meta.size
                    self._etag = meta.etag
            except Exception:
                pass
            hf_hub_download(
                repo_id = repo_id,
                filename = filename,
                token = hf_token or None,
            )
        except Exception as exc:
            logger.warning("GGUF STT download failed for %s: %s", model_id, exc)
            with self._lock:
                self._error = f"Download failed for '{model_id}'."


_download_state = _GgmlDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> None:
    _download_state.start(resolve_ggml_model_id(model), hf_token)


def download_status() -> dict:
    return _download_state.status()


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
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_id: Optional[str] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0
        self._keep_alive_seconds = keep_alive_seconds
        # Set while whisper-server is starting so training admission can account
        # for the accelerator memory it is about to bind. Read without the lock.
        self._loading = False

    @property
    def loaded_model(self) -> Optional[str]:
        with self._lock:
            return self._model_id if self._process_alive() else None

    @property
    def device(self) -> Optional[str]:
        with self._lock:
            return "whisper.cpp" if self._process_alive() else None

    def is_loading(self) -> bool:
        # True only while whisper-server is starting (may take seconds to bind
        # its GPU backend); load() sets and clears the flag around that window.
        return self._loading

    @property
    def keep_alive_seconds(self) -> float:
        return self._keep_alive_seconds

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

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

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def cancel_pending_load(self) -> bool:
        # No async load phase to cancel; unloading is enough for training.
        return False

    def wait_for_load_to_settle(self) -> None:
        return None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _ensure_model_downloaded(self, model_id: str) -> str:
        path = _cached_model_path(model_id)
        if path is None:
            raise SttModelNotDownloadedError(
                f"STT model '{model_id}' (GGUF) is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            )
        return path

    def load(self, model: Optional[str] = None) -> None:
        """Start (or switch) whisper-server for the requested curated model."""
        model_id = resolve_ggml_model_id(model)
        with self._lock:
            binary = ensure_engine_available()
            if self._process_alive() and self._model_id == model_id:
                self._schedule_idle_unload_locked()
                return
            model_path = self._ensure_model_downloaded(model_id)
            self._release_locked()
            port = self._find_free_port()
            command = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port)]
            if _training_active():
                # Keep whisper.cpp off the accelerator during training, matching
                # the Transformers sidecar's CPU device choice, so a mid-training
                # dictation cannot reclaim the VRAM training just freed.
                command.append("--no-gpu")
            logger.info(
                "Starting whisper-server for STT model %s on 127.0.0.1:%s",
                model_id,
                port,
            )
            self._loading = True
            try:
                process = subprocess.Popen(
                    command,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                    stdin = subprocess.DEVNULL,
                    # Co-located GPU libs on the loader path (WSL system HIP first),
                    # secrets scrubbed from the downloaded binary's env.
                    env = _whisper_server_child_env(binary),
                    # Die with Studio (Linux PDEATHSIG, Windows job) so a crash
                    # never orphans a server still holding the model.
                    **child_popen_kwargs(),
                )
                adopt_pid(process.pid)  # terminate_all backstop for graceful exits
                try:
                    self._wait_for_server(process, port)
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
                self._loading = False

    @staticmethod
    def _wait_for_server(process: subprocess.Popen, port: int) -> None:
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SttEngineUnavailableError(
                    "The local transcription runtime exited before becoming "
                    "ready; the model file may be corrupt or unsupported."
                )
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/", method = "GET")
                with urllib.request.urlopen(req, timeout = 2):
                    return
            except Exception:
                time.sleep(0.2)
        raise SttEngineUnavailableError("The local transcription runtime did not start in time.")

    # -- transcription ------------------------------------------------------

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
    ) -> dict:
        """Transcribe encoded audio bytes via whisper-server.

        Accepts any container PyAV can decode (same validation and caps as the
        Transformers sidecar). Returns {text, language, duration, model}.
        """
        ensure_engine_available()
        model_id = resolve_ggml_model_id(model)
        lang = normalize_whisper_language(language)
        known_languages = _known_whisper_languages()
        if lang is not None and known_languages is not None and lang not in known_languages:
            raise SttLanguageError(
                f"Language '{language}' is not supported by STT model '{model_id}'."
            )
        # Reject a missing model before decoding so a long clip cannot burn CPU
        # only to 409, matching the Transformers sidecar's download preflight.
        self._ensure_model_downloaded(model_id)
        decoded_audio = _decode_audio_bounded(audio)
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        with self._lock:
            try:
                self.load(model_id)
                text = self._post_inference(wav_bytes, lang, fast)
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": model_id,
        }

    def _post_inference(self, wav_bytes: bytes, lang: Optional[str], fast: bool) -> str:
        boundary = uuid.uuid4().hex
        fields = {
            "temperature": "0.0",
            "response_format": "json",
            # Match the Transformers sidecar decoding policy: five-way beam
            # search by default, single-candidate greedy for fast dictation.
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
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/inference",
            data = body,
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout = _TRANSCRIBE_TIMEOUT_SECONDS) as resp:
                payload = json.load(resp)
        except SttAudioDecodeError:
            raise
        except Exception as exc:
            raise SttEngineUnavailableError(
                "The local transcription runtime did not answer the request."
            ) from exc
        text = payload.get("text")
        if not isinstance(text, str):
            raise SttAudioDecodeError("Could not decode the audio.")
        # whisper.cpp joins segments with newlines; dictation wants one line.
        return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


_sidecar: Optional[GgmlSttSidecar] = None


def get_ggml_stt_sidecar() -> GgmlSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = GgmlSttSidecar()
    return _sidecar
