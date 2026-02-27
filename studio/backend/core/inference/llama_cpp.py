"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""
import atexit
import json
import logging
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import httpx

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """
    Manages a llama-server subprocess for GGUF model inference.

    Lifecycle:
        1. load_model()  — starts llama-server with the GGUF file
        2. generate_chat_completion() — proxies to /v1/chat/completions, streams back
        3. unload_model() — terminates llama-server subprocess
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_identifier: Optional[str] = None
        self._gguf_path: Optional[str] = None
        self._hf_repo: Optional[str] = None
        self._hf_variant: Optional[str] = None
        self._is_vision: bool = False
        self._healthy = False
        self._lock = threading.Lock()

        atexit.register(self._cleanup)

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._healthy

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return self._is_vision

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _find_llama_server_binary() -> Optional[str]:
        """
        Locate the llama-server binary.

        Search order:
        1. LLAMA_SERVER_PATH environment variable
        2. ./llama.cpp/build/bin/llama-server  (built by setup.sh in-tree)
        3. llama-server on PATH  (system install)
        4. ./bin/llama-server  (legacy: extracted binary)
        """
        import os

        # 1. Env var
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path and Path(env_path).is_file():
            return env_path

        # Project root: llama_cpp.py → inference/ → core/ → backend/ → studio/ → root
        project_root = Path(__file__).resolve().parents[4]

        # 2. In-tree llama.cpp build (setup.sh builds here)
        build_path = project_root / "llama.cpp" / "build" / "bin" / "llama-server"
        if build_path.is_file():
            return str(build_path)

        # 3. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 4. Legacy: extracted to bin/
        bin_path = project_root / "bin" / "llama-server"
        if bin_path.is_file():
            return str(bin_path)

        return None

    # ── Port allocation ───────────────────────────────────────────

    @staticmethod
    def _find_free_port() -> int:
        """Find an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    # ── Lifecycle ─────────────────────────────────────────────────

    def load_model(
        self,
        *,
        # Local mode: pass a path to a .gguf file
        gguf_path: Optional[str] = None,
        # HF mode: let llama-server download via -hf "repo:quant"
        hf_repo: Optional[str] = None,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        # Common
        model_identifier: str,
        is_vision: bool = False,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
    ) -> bool:
        """
        Start llama-server with a GGUF model.

        Two modes:
        - Local: ``gguf_path="/path/to/model.gguf"`` → uses ``-m``
        - HF:    ``hf_repo="unsloth/gemma-3-4b-it-GGUF", hf_variant="Q4_K_M"`` → uses ``-hf``

        In HF mode, llama-server handles downloading, caching, and
        auto-loading mmproj files for vision models.

        Returns True if server started and health check passed.
        """
        with self._lock:
            self._kill_process()

            binary = self._find_llama_server_binary()
            if not binary:
                raise RuntimeError(
                    "llama-server binary not found. "
                    "Run setup.sh to build it, install llama.cpp, "
                    "or set LLAMA_SERVER_PATH environment variable."
                )

            self._port = self._find_free_port()

            # Build command based on mode
            if hf_repo:
                hf_spec = f"{hf_repo}:{hf_variant}" if hf_variant else hf_repo
                cmd = [
                    binary,
                    "-hf", hf_spec,
                    "--port", str(self._port),
                    "-c", str(n_ctx),
                    "-ngl", str(n_gpu_layers),
                ]
                if hf_token:
                    cmd.extend(["--hf-token", hf_token])
            elif gguf_path:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
                cmd = [
                    binary,
                    "-m", gguf_path,
                    "--port", str(self._port),
                    "-c", str(n_ctx),
                    "-ngl", str(n_gpu_layers),
                ]
            else:
                raise ValueError("Either gguf_path or hf_repo must be provided")

            if n_threads is not None:
                cmd.extend(["--threads", str(n_threads)])

            logger.info(f"Starting llama-server: {' '.join(cmd)}")

            # Set LD_LIBRARY_PATH so llama-server can find its shared libs
            # (libmtmd.so, libllama.so, etc.) which live next to the binary
            import os
            env = os.environ.copy()
            binary_dir = str(Path(binary).parent)
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{binary_dir}:{existing_ld}" if existing_ld else binary_dir

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            self._gguf_path = gguf_path
            self._hf_repo = hf_repo
            self._hf_variant = hf_variant
            self._is_vision = is_vision
            self._model_identifier = model_identifier

            # HF mode: llama-server downloads before becoming healthy — need longer timeout
            timeout = 600.0 if hf_repo else 120.0
            if not self._wait_for_health(timeout=timeout):
                self._kill_process()
                raise RuntimeError(
                    "llama-server failed to start. "
                    "Check that the GGUF file is valid and you have enough memory."
                )

            self._healthy = True

            logger.info(
                f"llama-server ready on port {self._port} "
                f"for model '{model_identifier}'"
            )
            return True

    def unload_model(self) -> bool:
        """Terminate the llama-server subprocess and clean up state."""
        with self._lock:
            self._kill_process()
            logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._hf_repo = None
            self._hf_variant = None
            self._is_vision = False
            self._port = None
            self._healthy = False
            return True

    def _kill_process(self):
        """Terminate the subprocess if running."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit on SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            self._process = None

    def _cleanup(self):
        """atexit handler to ensure llama-server is terminated."""
        self._kill_process()

    def _wait_for_health(self, timeout: float = 120.0, interval: float = 0.5) -> bool:
        """
        Poll llama-server's /health endpoint until it responds 200.

        Also monitors subprocess for early exit/crash.
        """
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self._port}/health"

        while time.monotonic() < deadline:
            # Check if process crashed
            if self._process.poll() is not None:
                # Read remaining output for error info
                output = self._process.stdout.read() if self._process.stdout else ""
                logger.error(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Output: {output[:2000]}"
                )
                return False

            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(interval)

        logger.error(f"llama-server health check timed out after {timeout}s")
        return False

    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _build_openai_messages(
        messages: list[dict],
        image_b64: Optional[str] = None,
    ) -> list[dict]:
        """
        Build OpenAI-format messages, optionally injecting an image_url
        content part into the last user message for vision models.

        If no image is provided, returns messages as-is.
        """
        if not image_b64:
            return messages

        # Find the last user message and convert to multimodal content parts
        result = [msg.copy() for msg in messages]
        last_user_idx = None
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx is not None:
            text_content = result[last_user_idx].get("content", "")
            result[last_user_idx]["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]

        return result

    # ── Generation (proxy to llama-server) ────────────────────────

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.1,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """
        Send a chat completion request to llama-server and stream tokens back.

        Uses /v1/chat/completions — llama-server handles chat template
        application and vision (multimodal image_url parts) natively.

        Yields cumulative text (matching InferenceBackend's convention).
        """
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        openai_messages = self._build_openai_messages(messages, image_b64)

        payload = {
            "messages": openai_messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repeat_penalty": repetition_penalty,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""

        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    for raw_chunk in response.iter_text():
                        if cancel_event is not None and cancel_event.is_set():
                            break

                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                return
                            if not line.startswith("data: "):
                                continue

                            try:
                                data = json.loads(line[6:])
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    token = delta.get("content", "")
                                    if token:
                                        cumulative += token
                                        yield cumulative
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise
