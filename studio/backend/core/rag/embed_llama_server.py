# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF embedder over the bundled llama.cpp, served via HTTP (no torch).

Opt-in (``RAG_EMBED_BACKEND=llama-server``). Runs a dedicated
``llama-server --embedding`` subprocess on its own port and calls its
OpenAI-style ``/v1/embeddings`` + ``/tokenize``. Fully isolated from the chat
backend: separate process, no new Studio route, command built here directly.

Device is ``auto`` (GPU when present -- NVIDIA/ROCm or Apple Metal -- else CPU,
falling back to CPU if a GPU start fails); ``RAG_EMBED_DEVICE=cpu``/``gpu``
forces it. GPU detection reuses llama_cpp's static probes, so it needs no torch.

We call only llama_cpp's *static* helpers; the instance-coupled bits (download,
health, kill) are small local copies, since constructing a ``LlamaCppBackend``
runs its ``__init__`` reaper and binds chat state. That reaper kills any Studio
llama-server, so each request re-spawns ours if it died (self-heal).
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

from . import config

logger = logging.getLogger(__name__)

# httpx transport errors meaning "the server is gone" -> trigger a respawn.
_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)


class LlamaServerBackend:
    """Manages a llama.cpp embedding subprocess and talks to it over HTTP."""

    def __init__(self) -> None:
        # Lifecycle (spawn/restart/kill) is serialized; HTTP requests are not
        # (httpx.Client is thread-safe and llama-server batches concurrently).
        self._lifecycle_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._stdout_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._dim: int | None = None
        self._dim_lock = threading.Lock()
        self._model_path: str | None = None
        self._binary: str | None = None
        # Sticky after an auto GPU start fails: later spawns stay on CPU.
        self._force_cpu = False
        # Pooled client; requests pass full URLs, so a respawn's new port needs
        # no rebuild.
        self._client = httpx.Client(timeout = config.EMBED_REQUEST_TIMEOUT_S)
        atexit.register(self._shutdown)

    # ── HTTP base ────────────────────────────────────────────────

    @property
    def _base_url(self) -> str:
        return f"http://{config.EMBED_HOST}:{self._port}"

    # ── Binary / model resolution ────────────────────────────────

    def _resolve_binary(self) -> str:
        """Find llama-server (via the chat backend's discovery), verify it
        supports embeddings, cache it. Raises if missing/unsupported."""
        if self._binary is not None:
            return self._binary
        from core.inference.llama_cpp import LlamaCppBackend

        binary = LlamaCppBackend._find_llama_server_binary()
        if not binary:
            raise RuntimeError(
                "llama-server binary not found; cannot use RAG_EMBED_BACKEND="
                "llama-server. Install llama.cpp or set LLAMA_SERVER_PATH / "
                "UNSLOTH_LLAMA_CPP_PATH."
            )
        self._assert_embedding_support(binary)
        self._binary = binary
        return binary

    @staticmethod
    @lru_cache(maxsize = 8)
    def _help_text(binary: str) -> str:
        """`llama-server --help`, cached. Capture both streams, ignore exit code
        (some builds exit non-zero on --help)."""
        try:
            proc = subprocess.run(
                [binary, "--help"],
                capture_output = True,
                text = True,
                timeout = 30,
                **windows_hidden_subprocess_kwargs(),
            )
            return (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("could not run `llama-server --help`: %s", e)
            return ""

    def _assert_embedding_support(self, binary: str) -> None:
        help_text = self._help_text(binary)
        # Empty help -> assume capable (a real missing flag still fails at spawn).
        if help_text and "--embedding" not in help_text:
            raise RuntimeError(
                "the bundled llama-server build lacks --embedding support; "
                "RAG_EMBED_BACKEND=llama-server requires an embeddings-capable build"
            )

    def _resolve_model_path(self) -> str:
        """Download (or cache-hit) the GGUF embedder, returning its local path.
        Picks the variant-matching, non-mmproj .gguf in the repo."""
        if self._model_path is not None:
            return self._model_path
        from huggingface_hub import hf_hub_download, list_repo_files

        repo = config.EMBED_GGUF_REPO
        token = os.environ.get("HF_TOKEN") or None
        files = [
            f for f in list_repo_files(repo, token = token) if f.lower().endswith(".gguf")
        ]
        files = [f for f in files if "mmproj" not in f.lower()]
        if not files:
            raise RuntimeError(f"no .gguf file found in embedder repo {repo!r}")
        variant = config.EMBED_GGUF_VARIANT.lower()
        match = [f for f in files if variant in f.lower()] or files
        filename = sorted(match, key = len)[0]
        logger.info("resolving GGUF embedder %s/%s", repo, filename)
        self._model_path = hf_hub_download(repo_id = repo, filename = filename, token = token)
        return self._model_path

    # ── Device selection ─────────────────────────────────────────

    # Min free VRAM (MiB) for the embedder; below this, auto stays on CPU.
    _MIN_GPU_FREE_MIB = 1024

    def _use_gpu(self) -> bool:
        """``RAG_EMBED_DEVICE``: ``gpu``/``cpu`` force it; ``auto`` uses a GPU when
        present. A sticky CPU fallback (after an auto GPU start fails) wins."""
        dev = config.EMBED_DEVICE.lower()
        if dev == "gpu":
            return True
        if dev == "cpu" or self._force_cpu:
            return False
        return self._gpu_available()  # auto

    @staticmethod
    def _gpu_available() -> bool:
        """Apple Metal, or an NVIDIA/ROCm GPU with enough free VRAM. Reuses
        llama_cpp's static ``_get_gpu_free_memory`` (nvidia-smi first, so the
        common path needs no torch)."""
        from utils.hardware import is_apple_silicon

        if is_apple_silicon():
            return True  # bundled mac build offloads to Metal
        from core.inference.llama_cpp import LlamaCppBackend

        gpus = LlamaCppBackend._get_gpu_free_memory()  # [(idx, free_mib)], honors CVD
        return any(free >= LlamaServerBackend._MIN_GPU_FREE_MIB for _, free in gpus)

    # ── Subprocess command / env ─────────────────────────────────

    def _build_cmd(
        self, binary: str, model_path: str, port: int, *, use_gpu: bool
    ) -> list[str]:
        # No --embd-normalize (not in every build; we normalize in Python to
        # match the ST path). --fit off: don't auto-resize ctx/offload to fill
        # device memory.
        cmd = [
            binary,
            "-m",
            model_path,
            "--host",
            config.EMBED_HOST,
            "--port",
            str(port),
            "--embedding",
            "--pooling",
            "cls",
            "--fit",
            "off",
        ]
        # -1 offloads every layer (matches the chat server); 0 keeps it on CPU.
        cmd += ["-ngl", "-1" if use_gpu else "0"]
        return cmd

    def _build_env(self, binary: str, *, use_gpu: bool) -> dict[str, str]:
        env = child_env_without_native_path_secret()
        env["LLAMA_SET_ROWS"] = "1"  # ggml set_rows fast path
        if use_gpu:
            # Inherit Studio's CUDA_VISIBLE_DEVICES; put CUDA libs on the path.
            self._add_linux_cuda_libs(env, str(Path(binary).parent))
        else:
            # Blank devices so a CUDA build stays on CPU and reserves no VRAM.
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    @staticmethod
    def _add_linux_cuda_libs(env: dict[str, str], binary_dir: str) -> None:
        """Best-effort LD_LIBRARY_PATH so the prebuilt binary finds CUDA libs."""
        import glob
        import platform
        import sys

        if sys.platform == "win32":
            return  # Windows resolves CUDA via PATH already in the inherited env.
        arch = platform.machine()
        lib_dirs = [binary_dir]
        for pattern in (
            os.path.join(
                sys.prefix, "lib", "python*", "site-packages", "nvidia", "cu*", "lib"
            ),
            os.path.join(
                sys.prefix, "lib", "python*", "site-packages", "nvidia", "cudnn", "lib"
            ),
        ):
            lib_dirs.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
        for cuda_lib in (
            "/usr/local/cuda/lib64",
            f"/usr/local/cuda/targets/{arch}-linux/lib",
            "/usr/local/cuda-12/lib64",
        ):
            if os.path.isdir(cuda_lib):
                lib_dirs.append(cuda_lib)
        existing = env.get("LD_LIBRARY_PATH", "")
        joined = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{joined}:{existing}" if existing else joined

    # ── Subprocess lifecycle ─────────────────────────────────────

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        """Drain the child's stdout so its pipe buffer never deadlocks; keep the
        tail for crash diagnostics."""
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)
                    if len(self._stdout_lines) > 200:
                        del self._stdout_lines[:-200]
                    logger.debug("[llama-embed] %s", line)
        except Exception:  # noqa: BLE001 - drain thread must never raise
            pass

    def _spawn(self) -> None:
        """Start the embed server (caller holds the lifecycle lock). On ``auto``,
        a failed GPU start falls back to CPU once; explicit ``gpu``/``cpu`` do
        not."""
        use_gpu = self._use_gpu()
        try:
            self._spawn_once(use_gpu)
        except RuntimeError:
            auto = config.EMBED_DEVICE.lower() not in ("gpu", "cpu")
            if use_gpu and auto:
                logger.warning("embed server GPU start failed; falling back to CPU")
                self._force_cpu = True
                self._spawn_once(False)
            else:
                raise

    def _spawn_once(self, use_gpu: bool) -> None:
        binary = self._resolve_binary()
        model_path = self._resolve_model_path()
        port = config.EMBED_PORT or self._find_free_port()
        env = self._build_env(binary, use_gpu = use_gpu)
        cmd = self._build_cmd(binary, model_path, port, use_gpu = use_gpu)
        logger.info(
            "starting llama-server embedder (%s): %s",
            "gpu" if use_gpu else "cpu",
            " ".join(cmd),
        )
        self._stdout_lines = []
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env = env,
            **windows_hidden_subprocess_kwargs(),
        )
        self._process = proc
        self._port = port
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout,
            args = (proc,),
            daemon = True,
            name = "llama-embed-stdout",
        )
        self._stdout_thread.start()
        if not self._wait_for_health(config.EMBED_STARTUP_TIMEOUT_S):
            tail = "\n".join(self._stdout_lines[-30:])
            self._kill_process()
            raise RuntimeError(
                "llama-server embedder failed to become healthy. "
                f"Last output:\n{tail[:2000]}"
            )

    @staticmethod
    def _find_free_port() -> int:
        from core.inference.llama_cpp import LlamaCppBackend

        return LlamaCppBackend._find_free_port()

    def _wait_for_health(self, timeout: float, interval: float = 0.5) -> bool:
        """Poll /health until 200, bailing early if the process exits."""
        deadline = time.monotonic() + timeout
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            if not self._process_alive():
                code = None if self._process is None else self._process.returncode
                logger.error("llama-server embedder exited early (code %s)", code)
                return False
            try:
                if httpx.get(url, timeout = 2.0).status_code == 200:
                    return True
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                pass
            time.sleep(interval)
        logger.error("llama-server embedder health check timed out after %ss", timeout)
        return False

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _ensure_ready(self) -> None:
        """Guarantee a live server, (re)spawning if needed. Double-checked so the
        alive path takes no lock; self-heals after the chat reaper kills us."""
        if self._process_alive():
            return
        with self._lifecycle_lock:
            if self._process_alive():
                return
            self._kill_process()
            self._spawn()

    def _restart(self) -> None:
        with self._lifecycle_lock:
            self._kill_process()
            self._spawn()

    def _kill_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server embedder did not exit on SIGTERM; killing")
            proc.kill()
            try:
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.warning("error killing llama-server embedder: %s", e)
        finally:
            self._process = None
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
                self._stdout_thread = None

    def _shutdown(self) -> None:
        try:
            self._kill_process()
        finally:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass

    # ── HTTP requests (self-healing on a dropped connection) ─────

    def _post(self, path: str, payload: dict) -> dict:
        """POST to the server, restarting once and retrying on a dropped
        connection (the reaper may have killed us between the liveness check and
        the request) or a timeout (the bundled embedding build occasionally
        wedges a request); a fresh server unsticks both."""
        last_exc: Exception | None = None
        for attempt in range(2):
            self._ensure_ready()
            try:
                resp = self._client.post(f"{self._base_url}{path}", json = payload)
                resp.raise_for_status()
                return resp.json()
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as e:
                last_exc = e
                if attempt == 0:
                    self._restart()
                    continue
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                raise RuntimeError(
                    f"llama-server embedder POST {path} -> {e.response.status_code}: {body}"
                ) from e
        raise RuntimeError(
            f"llama-server embedder POST {path} failed after retry"
        ) from last_exc

    # ── EmbeddingBackend API ─────────────────────────────────────

    def encode(self, texts, *, model_name = None, normalize = True):
        """Embed texts -> (N, dim) float32. ``model_name`` is ignored (the GGUF
        is fixed by config). Normalizes in Python to match the ST backend."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim()), dtype = np.float32)
        rows: list[list[float]] = []
        batch = max(1, config.EMBED_BATCH)
        for start in range(0, n, batch):
            chunk = list(texts[start : start + batch])
            data = self._post(
                "/v1/embeddings",
                {"input": chunk, "model": "embedding", "encoding_format": "float"},
            )
            items = data.get("data", [])
            if len(items) != len(chunk):
                raise RuntimeError(
                    f"embedder returned {len(items)} vectors for {len(chunk)} inputs"
                )
            # OpenAI spec lets the server reorder; sort back by index.
            items = sorted(items, key = lambda d: d.get("index", 0))
            rows.extend(d["embedding"] for d in items)
        arr = np.asarray(rows, dtype = np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"embedder returned ragged vectors: shape {arr.shape}")
        if normalize:
            norms = np.linalg.norm(arr, axis = 1, keepdims = True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def dim(self, *, model_name = None) -> int:
        """Embedding width, probed once via a 1-text encode and cached."""
        if self._dim is not None:
            return self._dim
        with self._dim_lock:
            if self._dim is None:
                vec = self.encode(["x"], normalize = False)
                self._dim = int(vec.shape[1])
        return self._dim

    def warm(self, *, model_name = None) -> None:
        """Start the server and probe dim off the request path."""
        self._ensure_ready()
        self.dim()

    def token_counter(self, *, model_name = None):
        """Callable counting tokens with the GGUF's own tokenizer via /tokenize,
        so chunk sizing matches the embedder. Cached per text."""

        @lru_cache(maxsize = 4096)
        def _count(text: str) -> int:
            data = self._post("/tokenize", {"content": text, "add_special": False})
            return len(data.get("tokens", []))

        return _count
