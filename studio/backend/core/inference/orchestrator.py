# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference orchestrator — subprocess-based.

Provides the same API as InferenceBackend, but delegates all ML work
to a persistent subprocess. The subprocess is spawned on first model load
and stays alive for subsequent requests.

When switching between models that need different transformers versions
(e.g. GLM-4.7-Flash needs 5.x, Qwen needs 4.57.x), the old subprocess
is killed and a new one is spawned with the correct version.

Pattern follows core/training/training.py.
"""

import atexit
import base64
import structlog
from loggers import get_logger
import multiprocessing as mp
import queue
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Generator, Optional, Tuple, Union

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")


class InferenceOrchestrator:
    """
    Inference backend orchestrator — subprocess-based.

    Exposes the same API surface as InferenceBackend so routes/inference.py
    needs minimal changes. Internally, all heavy ML operations happen in
    a persistent subprocess.
    """

    def __init__(self):
        # Subprocess state
        self._proc: Optional[mp.Process] = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None
        self._cancel_event: Any = None  # mp.Event — set to cancel generation instantly
        self._lock = threading.Lock()
        self._gen_lock = (
            threading.Lock()
        )  # Serializes generation — one request at a time

        # Local state mirrors (updated from subprocess responses)
        self.active_model_name: Optional[str] = None
        self.models: dict = {}
        self.loading_models: set = set()
        self.loaded_local_models: list = []
        self.default_models = [
            "unsloth/Qwen3-4B-Instruct-2507",
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
            "unsloth/Phi-3.5-mini-instruct",
            "unsloth/Gemma-3-4B-it",
            "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        ]

        # Version tracking for subprocess reuse
        self._current_transformers_major: Optional[str] = None  # "4" or "5"

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _spawn_subprocess(self, config: dict) -> None:
        """Spawn a new inference subprocess."""
        from .worker import run_inference_process

        self._cmd_queue = _CTX.Queue()
        self._resp_queue = _CTX.Queue()
        self._cancel_event = _CTX.Event()

        self._proc = _CTX.Process(
            target = run_inference_process,
            kwargs = {
                "cmd_queue": self._cmd_queue,
                "resp_queue": self._resp_queue,
                "cancel_event": self._cancel_event,
                "config": config,
            },
            daemon = True,
        )
        self._proc.start()
        logger.info("Inference subprocess started (pid=%s)", self._proc.pid)

    def _cancel_generation(self) -> None:
        """Cancel any ongoing generation in the subprocess (instant)."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _shutdown_subprocess(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the inference subprocess."""
        if self._proc is None or not self._proc.is_alive():
            self._proc = None
            return

        # 1. Cancel any ongoing generation first (instant via mp.Event)
        self._cancel_generation()
        time.sleep(0.5)  # Brief wait for generation to stop

        # 2. Drain stale responses from queue
        self._drain_queue()

        # 3. Send shutdown command
        try:
            self._cmd_queue.put({"type": "shutdown"})
        except (OSError, ValueError):
            pass

        # 4. Wait for graceful shutdown
        try:
            self._proc.join(timeout = timeout)
        except Exception:
            pass

        # 5. Force kill if still alive
        if self._proc is not None and self._proc.is_alive():
            logger.warning("Inference subprocess did not exit gracefully, terminating")
            try:
                self._proc.terminate()
                self._proc.join(timeout = 5)
            except Exception:
                pass
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Subprocess still alive after terminate, killing")
                try:
                    self._proc.kill()
                    self._proc.join(timeout = 3)
                except Exception:
                    pass

        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        self._cancel_event = None
        logger.info("Inference subprocess shut down")

    def _cleanup(self):
        """atexit handler."""
        self._shutdown_subprocess(timeout = 5.0)

    def _ensure_subprocess_alive(self) -> bool:
        """Check if subprocess is alive."""
        return self._proc is not None and self._proc.is_alive()

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: dict) -> None:
        """Send a command to the subprocess."""
        if self._cmd_queue is None:
            raise RuntimeError("No inference subprocess running")
        try:
            self._cmd_queue.put(cmd)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to send command to subprocess: {exc}")

    def _read_resp(self, timeout: float = 1.0) -> Optional[dict]:
        """Read a response from the subprocess (non-blocking with timeout)."""
        if self._resp_queue is None:
            return None
        try:
            return self._resp_queue.get(timeout = timeout)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    def _wait_response(self, expected_type: str, timeout: float = 120.0) -> dict:
        """Block until a response of the expected type arrives.

        Also handles 'status' and 'error' events during the wait.
        Returns the matching response dict.
        Raises RuntimeError on timeout or subprocess crash.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resp = self._read_resp(timeout = min(remaining, 1.0))

            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    raise RuntimeError("Inference subprocess crashed during wait")
                continue

            rtype = resp.get("type", "")

            if rtype == expected_type:
                return resp

            if rtype == "error":
                error_msg = resp.get("error", "Unknown error")
                raise RuntimeError(f"Subprocess error: {error_msg}")

            if rtype == "status":
                logger.info("Subprocess status: %s", resp.get("message", ""))
                continue

            # Other response types during wait — skip
            logger.debug(
                "Skipping response type '%s' while waiting for '%s'",
                rtype,
                expected_type,
            )

        raise RuntimeError(
            f"Timeout waiting for '{expected_type}' response after {timeout}s"
        )

    def _drain_queue(self) -> list:
        """Drain all pending responses."""
        events = []
        if self._resp_queue is None:
            return events
        while True:
            try:
                events.append(self._resp_queue.get_nowait())
            except queue.Empty:
                return events
            except (EOFError, OSError, ValueError):
                return events

    def _drain_until_gen_done(self, timeout: float = 5.0) -> None:
        """Consume resp_queue events until gen_done/gen_error, discarding them.

        Called after cancel to ensure stale tokens from the cancelled
        generation don't leak into the next request.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self._read_resp(timeout = min(0.5, deadline - time.monotonic()))
            if resp is None:
                if not self._ensure_subprocess_alive():
                    return
                continue
            rtype = resp.get("type", "")
            if rtype in ("gen_done", "gen_error"):
                return
        logger.warning("Timed out waiting for gen_done after cancel")

    # ------------------------------------------------------------------
    # Public API — same interface as InferenceBackend
    # ------------------------------------------------------------------

    def load_model(
        self,
        config,  # ModelConfig
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> bool:
        """Load a model for inference.

        Always spawns a fresh subprocess for each model load. This ensures
        a clean Python interpreter — no stale unsloth patches, torch.compile
        caches, or inspect.getsource() failures from a previous model.
        """
        from utils.transformers_version import needs_transformers_5

        model_name = config.identifier
        self.loading_models.add(model_name)

        try:
            needed_major = "5" if needs_transformers_5(model_name) else "4"

            # Build config dict for subprocess
            sub_config = {
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "load_in_4bit": load_in_4bit,
                "hf_token": hf_token or "",
                "gguf_variant": getattr(config, "gguf_variant", None),
                "trust_remote_code": trust_remote_code,
            }

            # Always kill existing subprocess and spawn fresh.
            # Reusing a subprocess after unsloth patches torch internals
            # causes inspect.getsource() failures on the next model load.
            if self._ensure_subprocess_alive():
                self._cancel_generation()
                time.sleep(0.3)
                self._shutdown_subprocess()

            elif self._proc is not None:
                # Dead subprocess — clean up
                self._shutdown_subprocess(timeout = 2)

            logger.info(
                "Spawning fresh inference subprocess for '%s' (transformers %s.x)",
                model_name,
                needed_major,
            )
            self._spawn_subprocess(sub_config)
            resp = self._wait_response("loaded", timeout = 180)

            # Update local state from response
            if resp.get("success"):
                self._current_transformers_major = needed_major
                model_info = resp.get("model_info", {})
                self.active_model_name = model_info.get("identifier", model_name)
                self.models[self.active_model_name] = {
                    "is_vision": model_info.get("is_vision", False),
                    "is_lora": model_info.get("is_lora", False),
                    "display_name": model_info.get("display_name", model_name),
                    "is_audio": model_info.get("is_audio", False),
                    "audio_type": model_info.get("audio_type"),
                    "has_audio_input": model_info.get("has_audio_input", False),
                }
                self.loading_models.discard(model_name)
                logger.info("Model '%s' loaded successfully in subprocess", model_name)
                return True
            else:
                error = resp.get("error", "Failed to load model")
                self.loading_models.discard(model_name)
                self.active_model_name = None
                self.models.clear()
                raise Exception(error)

        except Exception:
            self.loading_models.discard(model_name)
            self.active_model_name = None
            self.models.clear()
            raise

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from the subprocess."""
        if not self._ensure_subprocess_alive():
            # No subprocess — just clear local state
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None
            return True

        try:
            self._send_cmd(
                {
                    "type": "unload",
                    "model_name": model_name,
                }
            )
            resp = self._wait_response("unloaded", timeout = 30)

            # Update local state
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None

            logger.info("Model '%s' unloaded from subprocess", model_name)
            return True

        except Exception as exc:
            logger.error("Error unloading model '%s': %s", model_name, exc)
            # Clear local state anyway
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None
            return False

    def generate_chat_response(
        self,
        messages: list,
        system_prompt: str = "",
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Generate response, streaming tokens from subprocess."""
        yield from self._generate_inner(
            messages = messages,
            system_prompt = system_prompt,
            image = image,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            cancel_event = cancel_event,
            use_adapter = None,
        )

    def generate_with_adapter_control(
        self,
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        """Generate with adapter control, streaming tokens from subprocess."""
        yield from self._generate_inner(
            use_adapter = use_adapter,
            cancel_event = cancel_event,
            **gen_kwargs,
        )

    def _generate_inner(
        self,
        messages: list = None,
        system_prompt: str = "",
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        cancel_event = None,
        use_adapter = None,
    ) -> Generator[str, None, None]:
        """Inner generation logic — sends command to subprocess, yields tokens.

        Serialized by _gen_lock: only one generation runs at a time.
        This prevents concurrent readers from consuming each other's
        tokens off the shared resp_queue.
        """
        if not self._ensure_subprocess_alive():
            yield "Error: Inference subprocess is not running"
            return

        if not self.active_model_name:
            yield "Error: No active model"
            return

        # Serialize generation — single GPU, one generation at a time.
        # Without this lock, two concurrent readers on the same resp_queue
        # can consume and drop each other's token events.
        with self._gen_lock:
            yield from self._generate_locked(
                messages = messages,
                system_prompt = system_prompt,
                image = image,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_new_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                cancel_event = cancel_event,
                use_adapter = use_adapter,
            )

    def _generate_locked(
        self,
        messages: list = None,
        system_prompt: str = "",
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        cancel_event = None,
        use_adapter = None,
    ) -> Generator[str, None, None]:
        """Actual generation logic — must be called under _gen_lock."""
        request_id = str(uuid.uuid4())

        # Convert PIL Image to base64 if needed
        image_b64 = None
        if image is not None:
            image_b64 = self._pil_to_base64(image)

        cmd = {
            "type": "generate",
            "request_id": request_id,
            "messages": messages or [],
            "system_prompt": system_prompt,
            "image_base64": image_b64,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }

        if use_adapter is not None:
            cmd["use_adapter"] = use_adapter

        try:
            self._send_cmd(cmd)
        except RuntimeError as exc:
            yield f"Error: {exc}"
            return

        # Yield tokens from response queue — we are the only reader
        # because _gen_lock is held.
        while True:
            resp = self._read_resp(timeout = 30.0)

            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    yield "Error: Inference subprocess crashed during generation"
                    return
                continue

            rtype = resp.get("type", "")

            # Status messages — skip
            if rtype == "status":
                continue

            # Error without request_id = subprocess-level error
            resp_rid = resp.get("request_id")
            if rtype == "error" and not resp_rid:
                yield f"Error: {resp.get('error', 'Unknown error')}"
                return

            if rtype == "token":
                # Check cancel from route (e.g. SSE connection closed)
                if cancel_event is not None and cancel_event.is_set():
                    self._cancel_generation()
                    # Wait for the subprocess to acknowledge cancellation
                    # (gen_done/gen_error) so stale events don't leak into
                    # the next generation request.
                    self._drain_until_gen_done(timeout = 5.0)
                    return
                yield resp.get("text", "")

            elif rtype == "gen_done":
                return

            elif rtype == "gen_error":
                yield f"Error: {resp.get('error', 'Unknown error')}"
                return

    def reset_generation_state(self):
        """Cancel any ongoing generation and reset state."""
        self._cancel_generation()
        if not self._ensure_subprocess_alive():
            return
        try:
            self._send_cmd({"type": "reset"})
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Audio generation — TTS, ASR, audio input
    # ------------------------------------------------------------------

    def generate_audio_response(
        self,
        text: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        use_adapter: Optional[Union[bool, str]] = None,
    ) -> Tuple[bytes, int]:
        """Generate TTS audio. Returns (wav_bytes, sample_rate).

        Blocking — sends command and waits for the complete audio response.
        """
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")
        if not self.active_model_name:
            raise RuntimeError("No active model")

        import uuid

        request_id = str(uuid.uuid4())

        cmd = {
            "type": "generate_audio",
            "request_id": request_id,
            "text": text,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if use_adapter is not None:
            cmd["use_adapter"] = use_adapter

        self._send_cmd(cmd)

        # Wait for audio_done or audio_error
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resp = self._read_resp(timeout = min(remaining, 1.0))

            if resp is None:
                if not self._ensure_subprocess_alive():
                    raise RuntimeError(
                        "Inference subprocess crashed during audio generation"
                    )
                continue

            rtype = resp.get("type", "")

            if rtype == "audio_done":
                wav_bytes = base64.b64decode(resp["wav_base64"])
                sample_rate = resp["sample_rate"]
                return wav_bytes, sample_rate

            if rtype == "audio_error":
                raise RuntimeError(resp.get("error", "Audio generation failed"))

            if rtype == "error":
                raise RuntimeError(resp.get("error", "Unknown error"))

            if rtype == "status":
                continue

        raise RuntimeError("Timeout waiting for audio generation (120s)")

    def generate_whisper_response(
        self,
        audio_array,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Whisper ASR — sends audio to subprocess, yields text."""
        yield from self._generate_audio_input_inner(
            audio_array = audio_array,
            audio_type = "whisper",
            messages = [],
            system_prompt = "",
            cancel_event = cancel_event,
        )

    def generate_audio_input_response(
        self,
        messages,
        system_prompt,
        audio_array,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Audio input generation (e.g. Gemma 3n) — streams text tokens."""
        yield from self._generate_audio_input_inner(
            audio_array = audio_array,
            audio_type = None,  # worker will use generate_audio_input_response
            messages = messages,
            system_prompt = system_prompt,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            cancel_event = cancel_event,
        )

    def _generate_audio_input_inner(
        self,
        audio_array,
        audio_type: Optional[str] = None,
        messages: list = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Shared inner logic for audio input generation (Whisper + ASR)."""
        if not self._ensure_subprocess_alive():
            yield "Error: Inference subprocess is not running"
            return
        if not self.active_model_name:
            yield "Error: No active model"
            return

        with self._gen_lock:
            import uuid

            request_id = str(uuid.uuid4())

            # Convert numpy array to list for mp.Queue serialization
            audio_data = (
                audio_array.tolist()
                if hasattr(audio_array, "tolist")
                else list(audio_array)
            )

            cmd = {
                "type": "generate_audio_input",
                "request_id": request_id,
                "audio_data": audio_data,
                "audio_type": audio_type,
                "messages": messages or [],
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
            }

            try:
                self._send_cmd(cmd)
            except RuntimeError as exc:
                yield f"Error: {exc}"
                return

            # Yield tokens — same pattern as _generate_locked
            while True:
                resp = self._read_resp(timeout = 30.0)

                if resp is None:
                    if not self._ensure_subprocess_alive():
                        yield "Error: Inference subprocess crashed during audio input generation"
                        return
                    continue

                rtype = resp.get("type", "")

                if rtype == "status":
                    continue

                if rtype == "error" and not resp.get("request_id"):
                    yield f"Error: {resp.get('error', 'Unknown error')}"
                    return

                if rtype == "token":
                    if cancel_event is not None and cancel_event.is_set():
                        self._cancel_generation()
                        self._drain_until_gen_done(timeout = 5.0)
                        return
                    yield resp.get("text", "")

                elif rtype == "gen_done":
                    return

                elif rtype == "gen_error":
                    yield f"Error: {resp.get('error', 'Unknown error')}"
                    return

    # ------------------------------------------------------------------
    # Local helpers (no subprocess needed)
    # ------------------------------------------------------------------

    def resize_image(self, img, max_size: int = 800):
        """Resize image while maintaining aspect ratio.
        No ML imports needed — runs locally in parent process.
        """
        if img is None:
            return None
        if img.size[0] > max_size or img.size[1] > max_size:
            from PIL import Image

            ratio = min(max_size / img.size[0], max_size / img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    @staticmethod
    def _pil_to_base64(img) -> str:
        """Convert a PIL Image to base64 string for IPC."""
        buf = BytesIO()
        img.save(buf, format = "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def get_current_model(self) -> Optional[str]:
        """Get currently active model name."""
        return self.active_model_name

    def is_model_loading(self) -> bool:
        """Check if any model is currently loading."""
        return len(self.loading_models) > 0

    def get_loading_model(self) -> Optional[str]:
        """Get name of currently loading model."""
        return next(iter(self.loading_models)) if self.loading_models else None

    def check_vision_model_compatibility(self) -> bool:
        """Check if current model supports vision."""
        if self.active_model_name and self.active_model_name in self.models:
            return self.models[self.active_model_name].get("is_vision", False)
        return False


# ========== GLOBAL INSTANCE ==========
_inference_backend = None


def get_inference_backend() -> InferenceOrchestrator:
    """Get global inference backend instance (orchestrator)."""
    global _inference_backend
    if _inference_backend is None:
        _inference_backend = InferenceOrchestrator()
    return _inference_backend
