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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loggers import get_logger

from utils.process_lifetime import adopt_pid, child_popen_kwargs

from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    _decode_audio_bounded,
    _TARGET_SAMPLE_RATE,
    _training_active,
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
    """True when llama-server is installed; the engine needs nothing else."""
    return find_llama_server_binary() is not None


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


def _cached_main_revision(repo: str) -> Optional[str]:
    """The commit `main` currently points at in the cache, if it is recorded."""
    from core.inference.stt_sidecar import _repo_cache_dir

    try:
        revision = (_repo_cache_dir(repo) / "refs" / "main").read_text("utf-8").strip()
    except OSError:
        return None
    return revision or None


def _cached_file(model_id: str, filename: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(
            repo_id = MTMD_STT_MODELS[model_id].repo,
            filename = filename,
            local_files_only = True,
        )
    except Exception:
        return None


def _cached_model_paths(model_id: str) -> Optional[tuple[str, str]]:
    """Both cached files, or None when either is missing."""
    spec = MTMD_STT_MODELS[model_id]
    model = _cached_file(model_id, spec.model_file)
    mmproj = _cached_file(model_id, spec.mmproj_file)
    if model is None or mmproj is None:
        return None
    return model, mmproj


def is_model_downloaded(model_id: str) -> bool:
    return model_id in MTMD_STT_MODELS and _cached_model_paths(model_id) is not None


class _MtmdDownloadState:
    """Tracks the two-file download of one mtmd dictation model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._cancelled = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            return {
                "downloading": downloading,
                "model": self._model_id if downloading else None,
                "error": self._error,
                "cancelled": self._cancelled,
                "bytes_total": self._total_bytes if downloading else None,
                "bytes_done": self._cache_bytes() if downloading else None,
            }

    def cancel(self) -> bool:
        """Stop an in-flight download. False when none was running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        return True

    def _cache_bytes(self) -> Optional[int]:
        """Best-effort progress: bytes in this repo's cache blobs."""
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            model_id = self._model_id
            if not model_id:
                return None
            repo = MTMD_STT_MODELS[model_id].repo.replace("/", "--")
            blobs = Path(HF_HUB_CACHE) / f"models--{repo}" / "blobs"
            if not blobs.is_dir():
                return 0
            return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file())
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = resolve_mtmd_model_id(model_id)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    return
                raise SttModelIdError(
                    f"Another dictation model ('{self._model_id}') is still "
                    "downloading; wait for it to finish."
                )
            self._model_id = model_id
            self._error = None
            self._total_bytes = None
            self._cancelled = False
            self._process = None
            thread = threading.Thread(target = self._run, args = (model_id, hf_token), daemon = True)
            self._thread = thread
            thread.start()

    def _run(self, model_id: str, hf_token: Optional[str]) -> None:
        spec = MTMD_STT_MODELS[model_id]
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            total = 0
            for filename in (spec.model_file, spec.mmproj_file):
                try:
                    meta = get_hf_file_metadata(
                        hf_hub_url(spec.repo, filename), token = hf_token or None
                    )
                    total += int(meta.size or 0)
                except Exception:
                    pass
            with self._lock:
                self._total_bytes = total or None

            from core.inference.stt_download_worker import spawn_download

            # One worker per file. Both are required, so a cancel between them
            # leaves the model not downloaded. The first resolves "main" and the
            # second is pinned to whatever it landed on, so a repo update
            # mid-download cannot mix two commits. The first stays unpinned
            # because hf_hub_download only writes refs/main for a named
            # revision, and _cached_file() resolves through that ref.
            revision: Optional[str] = None
            for filename in (spec.model_file, spec.mmproj_file):
                process = spawn_download(
                    ["--repo-id", spec.repo, "--filename", filename]
                    + (["--revision", revision] if revision else []),
                    hf_token = hf_token or None,
                )
                with self._lock:
                    if self._cancelled:
                        process.terminate()
                    self._process = process
                _, stderr = process.communicate()
                if process.returncode == 0:
                    revision = revision or _cached_main_revision(spec.repo)
                    continue
                with self._lock:
                    if self._cancelled or process.returncode < 0:
                        self._cancelled = True
                        return
                detail = (stderr or b"").decode("utf-8", "replace").strip()
                logger.warning("mtmd STT download failed for %s: %s", model_id, detail)
                with self._lock:
                    self._error = f"Download failed for '{model_id}'."
                return
        except Exception as exc:
            logger.warning("mtmd STT download failed for %s: %s", model_id, exc)
            with self._lock:
                self._error = f"Download failed for '{model_id}'."


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
        self._keep_alive_seconds = keep_alive_seconds
        self._idle_timer: Optional[threading.Timer] = None
        self._generation = 0

    @property
    def loaded_model(self) -> Optional[str]:
        with self._lock:
            return self._model_id if self._process_alive() else None

    @property
    def device(self) -> Optional[str]:
        return "llama.cpp" if self.loaded_model else None

    def is_loading(self) -> bool:
        with self._lock:
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
        if self._keep_alive_seconds <= 0:
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
        self._process = None
        self._port = None
        self._model_id = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout = 10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

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
        model_id = resolve_mtmd_model_id(model)
        binary = ensure_engine_available()
        # Startup happens outside _lock (it is slow), so this keeps two callers
        # from each spawning a server and orphaning the first.
        with self._start_lock:
            self._load_locked(model_id, binary)

    def _load_locked(self, model_id: str, binary: str) -> None:
        with self._lock:
            if self._process_alive() and self._model_id == model_id:
                self._schedule_idle_unload_locked()
                return
            self._release_locked()
            model_path, mmproj_path = self._ensure_model_downloaded(model_id)
            self._loading = True
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
                "-ngl",
                "99",
            ]
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
            adopt_pid(process.pid)  # terminate_all backstop for graceful exits
            if not self._wait_for_server(process, port):
                try:
                    process.terminate()
                except Exception:
                    pass
                raise SttUnavailableError(f"llama-server did not become ready for '{model_id}'.")
            with self._lock:
                self._process = process
                self._port = port
                self._model_id = model_id
                self._generation += 1
                self._schedule_idle_unload_locked()
        finally:
            with self._lock:
                self._loading = False

    @staticmethod
    def _wait_for_server(process: subprocess.Popen, port: int) -> bool:
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout = 2) as response:
                    if response.status == 200:
                        return True
            except Exception:
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
        if _training_active():
            raise SttUnavailableError("Dictation is paused while a training run is using the GPU.")
        # Reject a missing model before decoding, matching the other sidecars.
        self._ensure_model_downloaded(model_id)
        decoded_audio = _decode_audio_bounded(audio)
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        with self._lock:
            try:
                self.load(model_id)
                port = self._port
                if port is None:
                    raise SttUnavailableError("The dictation server is not running.")
                text = self._post_transcribe(port, model_id, wav_bytes)
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": normalize_whisper_language(language),
            "duration": duration,
            "model": model_id,
        }

    def _post_transcribe(self, port: int, model_id: str, wav_bytes: bytes) -> str:
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
            "max_tokens": 2048,
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
