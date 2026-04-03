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
import os
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
from utils.hardware import prepare_gpu_selection

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")


class DownloadStallError(RuntimeError):
    """Raised when the worker reports no download progress for too long."""


# Dispatcher timeout constants (seconds)
_DISPATCH_READ_TIMEOUT = 30.0
_DISPATCH_POLL_INTERVAL = 0.5
_DISPATCH_STOP_TIMEOUT = 5.0
_DISPATCH_IDLE_TIMEOUT = 30.0
_DISPATCH_DRAIN_TIMEOUT = 5.0


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

        # Dispatcher state — for compare mode (adapter-controlled requests).
        # Instead of serializing via _gen_lock, adapter-controlled requests
        # send commands directly to the subprocess and read from per-request
        # mailboxes. A dispatcher thread routes resp_queue events by request_id.
        self._mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()  # Protects _mailboxes dict
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()

        # Local state mirrors (updated from subprocess responses)
        self.active_model_name: Optional[str] = None
        self.models: dict = {}
        self.loading_models: set = set()
        self.loaded_local_models: list = []
        from core.inference.defaults import get_default_models

        self._static_models = get_default_models()
        self._top_gguf_cache: Optional[list[str]] = None
        self._top_hub_cache: Optional[list[str]] = None
        self._top_models_ready = threading.Event()

        # Version tracking for subprocess reuse
        self._current_transformers_major: Optional[str] = None  # "4" or "5"

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

        # Kick off background fetch of top models from HF
        threading.Thread(
            target = self._fetch_top_models, daemon = True, name = "top-models"
        ).start()

    # ------------------------------------------------------------------
    # Default models (top GGUFs fetched dynamically from HF)
    # ------------------------------------------------------------------

    @property
    def default_models(self) -> list[str]:
        # Wait up to 5s for background HF fetch to finish
        self._top_models_ready.wait(timeout = 5)
        top_gguf = self._top_gguf_cache or []
        top_hub = self._top_hub_cache or []
        # Curated static defaults first (editorial picks like new models),
        # then HF download-ranked models to backfill.
        # Send extras so the frontend still has 4 per category
        # after removing already-downloaded models.
        result: list[str] = []
        seen: set[str] = set()
        for m in self._static_models + top_gguf + top_hub:
            if m not in seen:
                result.append(m)
                seen.add(m)
        return result

    def _fetch_top_models(self) -> None:
        """Fetch top GGUF and non-GGUF repos from unsloth by downloads."""
        try:
            import httpx

            resp = httpx.get(
                "https://huggingface.co/api/models",
                params = {
                    "author": "unsloth",
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": "80",
                },
                timeout = 15,
            )
            if resp.status_code == 200:
                models = resp.json()
                # Top 40 GGUFs - frontend pages through them on-demand via
                # infinite scroll, so we send a deep pool.
                gguf_ids = [
                    m["id"] for m in models if m.get("id", "").upper().endswith("-GGUF")
                ][:40]
                # Top 40 non-GGUF hub models
                hub_ids = [
                    m["id"]
                    for m in models
                    if not m.get("id", "").upper().endswith("-GGUF")
                ][:40]
                if gguf_ids:
                    self._top_gguf_cache = gguf_ids
                    logger.info("Top GGUF models: %s", gguf_ids)
                if hub_ids:
                    self._top_hub_cache = hub_ids
                    logger.info("Top hub models: %s", hub_ids)
        except Exception as e:
            logger.warning("Failed to fetch top models: %s", e)
        finally:
            self._top_models_ready.set()

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
        self._stop_dispatcher()  # Stop dispatcher before killing subprocess
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

    def _wait_response(self, expected_type: str, timeout: float = 300.0) -> dict:
        """Block until a response of the expected type arrives.

        Also handles 'status' and 'error' events during the wait.
        Returns the matching response dict.
        Raises RuntimeError on timeout or subprocess crash.

        The *timeout* is an **inactivity** timeout: it resets whenever the
        subprocess sends a status message, so long-running operations (large
        downloads, slow model loads) won't be killed as long as the subprocess
        keeps reporting progress.
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
                # Reset deadline — subprocess is still alive and working
                deadline = time.monotonic() + timeout
                continue

            if rtype == "stall":
                msg = resp.get("message", "Download stalled")
                logger.warning("Subprocess reported stall: %s", msg)
                raise DownloadStallError(msg)

            # Other response types during wait — skip
            logger.debug(
                "Skipping response type '%s' while waiting for '%s'",
                rtype,
                expected_type,
            )

        raise RuntimeError(
            f"Timeout waiting for '{expected_type}' response "
            f"(no activity for {timeout}s)"
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
    # Dispatcher — per-request mailbox routing for compare mode
    # ------------------------------------------------------------------

    def _start_dispatcher(self) -> None:
        """Start the dispatcher thread if not already running.

        The dispatcher reads from the shared resp_queue and routes
        responses to per-request mailbox queues. This allows multiple
        adapter-controlled (compare) requests to be in-flight without
        holding _gen_lock.
        """
        if self._dispatcher_thread is not None and self._dispatcher_thread.is_alive():
            return

        self._dispatcher_stop.clear()
        self._dispatcher_thread = threading.Thread(
            target = self._dispatcher_loop,
            daemon = True,
            name = "inference-dispatcher",
        )
        self._dispatcher_thread.start()
        logger.debug("Dispatcher thread started")

    def _stop_dispatcher(self) -> None:
        """Signal the dispatcher to stop and wait for it."""
        if self._dispatcher_thread is None:
            return
        self._dispatcher_stop.set()
        self._dispatcher_thread.join(timeout = _DISPATCH_STOP_TIMEOUT)
        self._dispatcher_thread = None
        logger.debug("Dispatcher thread stopped")

    def _dispatcher_loop(self) -> None:
        """Background loop: read resp_queue → route to mailboxes by request_id."""
        while not self._dispatcher_stop.is_set():
            if self._resp_queue is None:
                break

            try:
                resp = self._resp_queue.get(timeout = _DISPATCH_POLL_INTERVAL)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                break

            rid = resp.get("request_id")
            rtype = resp.get("type", "")

            # Status messages — log and skip
            if rtype == "status":
                logger.info("Subprocess status: %s", resp.get("message", ""))
                continue

            # Route to mailbox if a matching request_id exists
            if rid:
                with self._mailbox_lock:
                    mbox = self._mailboxes.get(rid)
                if mbox is not None:
                    mbox.put(resp)
                    continue

            # No matching mailbox — might be for a _gen_lock reader or orphaned
            # Push it back so _read_resp can pick it up. But we can't un-get
            # from mp.Queue, so log a warning.
            if rtype not in ("status",):
                logger.debug(
                    "Dispatcher: no mailbox for request_id=%s type=%s, dropping",
                    rid,
                    rtype,
                )

    def _generate_dispatched(
        self,
        messages: list = None,
        system_prompt: str = "",
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        use_adapter = None,
    ) -> Generator[str, None, None]:
        """Dispatched generation — sends command without holding _gen_lock.

        Uses a per-request mailbox to receive tokens. This allows two
        compare-mode requests to be queued in the subprocess simultaneously,
        eliminating the inter-generation round-trip overhead.

        The subprocess processes commands sequentially from its cmd_queue,
        so generation is still serialized at the GPU level — we just avoid
        the orchestrator-level lock contention.
        """
        if not self._ensure_subprocess_alive():
            yield "Error: Inference subprocess is not running"
            return

        if not self.active_model_name:
            yield "Error: No active model"
            return

        # Ensure dispatcher is running
        self._start_dispatcher()

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

        # Create mailbox BEFORE sending command
        mailbox: queue.Queue = queue.Queue()
        with self._mailbox_lock:
            self._mailboxes[request_id] = mailbox

        try:
            self._send_cmd(cmd)
        except RuntimeError as exc:
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)
            yield f"Error: {exc}"
            return

        # Read tokens from our private mailbox
        try:
            while True:
                try:
                    resp = mailbox.get(timeout = _DISPATCH_READ_TIMEOUT)
                except queue.Empty:
                    # Timeout — check subprocess health
                    if not self._ensure_subprocess_alive():
                        yield "Error: Inference subprocess crashed during generation"
                        return
                    continue

                rtype = resp.get("type", "")

                if rtype == "token":
                    # Check cancel from route (e.g. SSE connection closed)
                    if cancel_event is not None and cancel_event.is_set():
                        self._cancel_generation()
                        # Drain remaining events for this request
                        self._drain_mailbox(mailbox, timeout = 5.0)
                        return
                    yield resp.get("text", "")

                elif rtype == "gen_done":
                    return

                elif rtype == "gen_error":
                    yield f"Error: {resp.get('error', 'Unknown error')}"
                    return
        finally:
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)

    def _drain_mailbox(self, mailbox: queue.Queue, timeout: float = 5.0) -> None:
        """Drain a mailbox until gen_done/gen_error, discarding tokens."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = mailbox.get(
                    timeout = min(_DISPATCH_POLL_INTERVAL, deadline - time.monotonic())
                )
            except queue.Empty:
                continue
            rtype = resp.get("type", "")
            if rtype in ("gen_done", "gen_error"):
                return
        logger.warning("Timed out draining mailbox after cancel")

    def _wait_dispatcher_idle(self) -> None:
        """Wait for all dispatched requests to complete, then stop dispatcher.

        Called by _generate_inner before using the _gen_lock path, to ensure
        the dispatcher thread isn't competing for resp_queue reads.
        """
        if self._dispatcher_thread is None or not self._dispatcher_thread.is_alive():
            return

        # Wait for all mailboxes to be emptied (dispatched requests complete)
        deadline = time.monotonic() + _DISPATCH_IDLE_TIMEOUT
        while time.monotonic() < deadline:
            with self._mailbox_lock:
                if not self._mailboxes:
                    break
            time.sleep(0.1)

        # Only stop dispatcher if all mailboxes drained.  If compare
        # requests are still active, leave the dispatcher running so
        # their token routing isn't killed mid-stream.
        with self._mailbox_lock:
            still_active = bool(self._mailboxes)
        if still_active:
            logger.warning(
                "Dispatcher still has %d active mailbox(es); "
                "leaving dispatcher running for compare requests",
                len(self._mailboxes),
            )
        else:
            self._stop_dispatcher()

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
        gpu_ids: Optional[list[int]] = None,
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
                "gpu_ids": gpu_ids,
            }
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
                gpu_ids,
                model_name = model_name,
                hf_token = hf_token,
                load_in_4bit = load_in_4bit,
            )
            sub_config["resolved_gpu_ids"] = resolved_gpu_ids
            sub_config["gpu_selection"] = gpu_selection

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

            disable_xet = sub_config.get("disable_xet", False) or (
                os.environ.get("HF_HUB_DISABLE_XET") == "1"
            )

            for attempt in range(2):
                logger.info(
                    "Spawning fresh inference subprocess for '%s' "
                    "(transformers %s.x, attempt %d/2%s)",
                    model_name,
                    needed_major,
                    attempt + 1,
                    ", xet disabled" if disable_xet else "",
                )
                sub_config["disable_xet"] = disable_xet
                self._spawn_subprocess(sub_config)

                try:
                    resp = self._wait_response("loaded")
                except DownloadStallError:
                    # First stall and Xet was enabled -> retry with Xet disabled
                    if attempt == 0 and not disable_xet:
                        logger.warning(
                            "Download stalled for '%s' -- retrying with "
                            "HF_HUB_DISABLE_XET=1",
                            model_name,
                        )
                        self._shutdown_subprocess(timeout = 5)
                        disable_xet = True
                        continue
                    # Second stall (or already had xet disabled) -> give up
                    self._shutdown_subprocess(timeout = 5)
                    raise RuntimeError(
                        f"Download stalled for '{model_name}' even with "
                        f"HF_HUB_DISABLE_XET=1 -- check your network connection"
                    )

                # Got a response — check success
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
                    logger.info(
                        "Model '%s' loaded successfully in subprocess", model_name
                    )
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
            resp = self._wait_response("unloaded")

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
        repetition_penalty: float = 1.0,
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
        """Generate with adapter control, streaming tokens from subprocess.

        Uses the dispatcher path (no _gen_lock) so that compare-mode
        requests don't block each other. The subprocess naturally
        serializes them via its sequential command loop.
        """
        yield from self._generate_dispatched(
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
        repetition_penalty: float = 1.0,
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

        # If the dispatcher is running (from a previous compare-mode request),
        # wait for all dispatched requests to finish, then stop the dispatcher
        # so we can safely read from resp_queue directly.
        self._wait_dispatcher_idle()

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
        repetition_penalty: float = 1.0,
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
        repetition_penalty: float = 1.0,
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
        repetition_penalty: float = 1.0,
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
        repetition_penalty: float = 1.0,
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
