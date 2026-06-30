# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference orchestrator — subprocess-based.

Same API as InferenceBackend, but delegates all ML work to a persistent
subprocess spawned on first model load and reused for later requests.

When switching between models needing different transformers versions
(e.g. GLM-4.7-Flash needs 5.x, Qwen needs 4.57.x), the old subprocess is
killed and a new one spawned with the correct version.

Pattern follows core/training/training.py.
"""

import atexit
import base64
import os
import signal
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

# Re-exported from the shared helper so GGUF, training, and inference share one
# type; kept importable here for backwards compatibility.
from utils.hf_xet_fallback import DownloadStallError

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")


# Dispatcher timeout constants (seconds)
_DISPATCH_READ_TIMEOUT = 30.0
_DISPATCH_POLL_INTERVAL = 0.5
_DISPATCH_STOP_TIMEOUT = 5.0
_DISPATCH_IDLE_TIMEOUT = 30.0
_DISPATCH_DRAIN_TIMEOUT = 5.0


class InferenceOrchestrator:
    """
    Inference backend orchestrator — subprocess-based.

    Same API surface as InferenceBackend (so routes/inference.py needs
    minimal changes); all heavy ML work happens in a persistent subprocess.
    """

    def __init__(self):
        # Subprocess state
        self._proc: Optional[mp.Process] = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None
        self._cancel_event: Any = None  # mp.Event — set to cancel generation
        self._gen_lock = threading.Lock()  # Serializes generation

        # Dispatcher state for compare mode (adapter-controlled requests):
        # bypass _gen_lock, send commands directly, read from per-request
        # mailboxes routed by a dispatcher thread on request_id.
        self._mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()

        # Local state mirrors (updated from subprocess responses)
        self.active_model_name: Optional[str] = None
        self.models: dict = {}
        self.loading_models: set = set()
        from core.inference.defaults import get_default_models

        self._static_models = get_default_models()
        self._top_gguf_cache: Optional[list[str]] = None
        self._top_hub_cache: Optional[list[str]] = None
        self._top_models_ready = threading.Event()

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

        threading.Thread(target = self._fetch_top_models, daemon = True, name = "top-models").start()

    # ------------------------------------------------------------------
    # Default models (top GGUFs fetched dynamically from HF)
    # ------------------------------------------------------------------

    @property
    def default_models(self) -> list[str]:
        # Wait up to 5s for background HF fetch
        self._top_models_ready.wait(timeout = 5)
        top_gguf = self._top_gguf_cache or []
        top_hub = self._top_hub_cache or []
        # Curated static defaults first, then HF download-ranked to backfill.
        # Send extras so the frontend keeps 4 per category after removing
        # downloaded ones.
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
                # Top 40 GGUFs (deep pool for frontend infinite scroll)
                gguf_ids = [m["id"] for m in models if m.get("id", "").upper().endswith("-GGUF")][
                    :40
                ]
                # Top 40 non-GGUF hub models
                hub_ids = [
                    m["id"] for m in models if not m.get("id", "").upper().endswith("-GGUF")
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
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )

        from .worker import run_inference_process

        with native_path_secret_removed_for_child_start():
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            self._cancel_event = _CTX.Event()

            self._proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = (run_inference_process,),
                kwargs = {
                    "cmd_queue": self._cmd_queue,
                    "resp_queue": self._resp_queue,
                    "cancel_event": self._cancel_event,
                    "config": config,
                },
                daemon = True,
            )
            self._proc.start()
        from utils.process_lifetime import adopt_pid

        adopt_pid(self._proc.pid)  # bind to parent lifetime (Windows job / sweep)
        logger.info("Inference subprocess started (pid=%s)", self._proc.pid)

    def _cancel_generation(self) -> None:
        """Cancel any ongoing generation in the subprocess (instant)."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _shutdown_subprocess(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the inference subprocess."""
        self._stop_dispatcher()  # before killing subprocess
        if self._proc is None or not self._proc.is_alive():
            self._proc = None
            return

        # 1. Cancel any ongoing generation first (instant via mp.Event)
        self._cancel_generation()
        time.sleep(0.5)

        # 2. Drain stale responses
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
        """True if the subprocess is alive."""
        return self._proc is not None and self._proc.is_alive()

    def _subprocess_crash_message(self, context: str) -> str:
        """Return a user-facing crash message with the worker exit status."""
        context_label = {
            "wait": "loading the model",
            "generation": "generating a response",
            "audio generation": "generating audio",
            "audio input generation": "processing audio input",
        }.get(context, context)
        message = f"The inference worker stopped unexpectedly while {context_label}."

        if self._proc is None:
            return f"{message} Details: process missing."

        exitcode = self._proc.exitcode
        pid = self._proc.pid
        if exitcode is None:
            return f"{message} Details: pid={pid}."

        if exitcode < 0:
            signum = -exitcode
            try:
                sig_name = signal.Signals(signum).name
            except ValueError:
                sig_name = f"SIG{signum}"

            suffix = ""
            if sig_name == "SIGKILL":
                suffix = (
                    " This usually means the system killed it under memory pressure. "
                    "Try a smaller model, lower context length, or close other GPU-heavy apps."
                )
            return (
                f"{message}{suffix} " f"Details: pid={pid}, signal={sig_name}, exitcode={exitcode}."
            )

        return f"{message} Details: pid={pid}, exitcode={exitcode}."

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

    def _wait_response(
        self,
        expected_type: str,
        timeout: float = 300.0,
    ) -> dict:
        """Block until a response of the expected type arrives.

        Also handles 'status' and 'error' events during the wait. Returns the
        matching response dict; raises RuntimeError on timeout or crash.

        *timeout* is an **inactivity** timeout: it resets on each status
        message, so long-running operations (large downloads, slow loads)
        survive as long as the subprocess keeps reporting progress.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resp = self._read_resp(timeout = min(remaining, 1.0))

            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    raise RuntimeError(self._subprocess_crash_message("wait"))
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
            f"Timeout waiting for '{expected_type}' response " f"(no activity for {timeout}s)"
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

        Called after cancel so stale tokens from the cancelled generation
        don't leak into the next request.
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
    # Generation command + token-stream helpers (shared by all paths)
    # ------------------------------------------------------------------

    def _build_generate_cmd(
        self,
        request_id: str,
        image_b64: Optional[str],
        *,
        messages: list = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        use_adapter = None,
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> dict:
        """Build the 'generate' command shared by the locked and dispatched paths."""
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
        # Only forward template kwargs the caller set, for older worker compat.
        if use_adapter is not None:
            cmd["use_adapter"] = use_adapter
        if tools is not None:
            cmd["tools"] = tools
        if enable_thinking is not None:
            cmd["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            cmd["reasoning_effort"] = reasoning_effort
        if preserve_thinking is not None:
            cmd["preserve_thinking"] = preserve_thinking
        return cmd

    def _consume_token_stream(
        self,
        read_one,
        drain_on_cancel,
        *,
        crash_context: str,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        read_timeout: float = 30.0,
    ) -> Generator[str, None, None]:
        """Yield tokens from a response stream until gen_done/gen_error.

        ``read_one(timeout)`` returns the next response (or None on timeout) and
        owns the queue choice — the shared resp_queue under _gen_lock, or a
        per-request mailbox on the dispatcher path — so this loop stays agnostic
        of which queue is read. On cancel, ``drain_on_cancel()`` consumes the
        cancel ack from that same source so stale events don't leak into the
        next request.
        """
        while True:
            resp = read_one(read_timeout)
            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    yield f"Error: {self._subprocess_crash_message(crash_context)}"
                    return
                continue

            rtype = resp.get("type", "")
            if rtype == "status":
                continue
            # Subprocess-level error (no request_id); request-scoped failures
            # arrive as gen_error below.
            if rtype == "error" and not resp.get("request_id"):
                yield f"Error: {resp.get('error', 'Unknown error')}"
                return

            if rtype == "token":
                # Cancel from route (e.g. SSE connection closed).
                if cancel_event is not None and cancel_event.is_set():
                    self._cancel_generation()
                    drain_on_cancel()
                    return
                yield resp.get("text", "")
            elif rtype == "gen_done":
                if stats_holder is not None:
                    stats_holder["stats"] = resp.get("stats")
                return
            elif rtype == "gen_error":
                yield f"Error: {resp.get('error', 'Unknown error')}"
                return

    # ------------------------------------------------------------------
    # Dispatcher — per-request mailbox routing for compare mode
    # ------------------------------------------------------------------

    def _start_dispatcher(self) -> None:
        """Start the dispatcher thread if not already running.

        The dispatcher reads the shared resp_queue and routes responses to
        per-request mailbox queues, letting multiple adapter-controlled
        (compare) requests be in-flight without holding _gen_lock.
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

            # Sole consumer of the response queue; if it died every in-flight
            # stream would hang, so never let routing kill the dispatcher.
            try:
                rid = resp.get("request_id")
                rtype = resp.get("type", "")

                # Status messages: log and skip
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

                # No matching mailbox; can't un-get from mp.Queue, so just log.
                logger.debug(
                    "Dispatcher: no mailbox for request_id=%s type=%s, dropping",
                    rid,
                    rtype,
                )
            except Exception:
                logger.exception("Inference dispatcher: failed to route a response; continuing")
                continue

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
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Dispatched generation — sends command without holding _gen_lock.

        Uses a per-request mailbox for tokens so two compare-mode requests can
        be queued at once. The subprocess still runs commands sequentially, so
        GPU work stays serialized; this only avoids orchestrator lock contention.
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

        cmd = self._build_generate_cmd(
            request_id,
            image_b64,
            messages = messages,
            system_prompt = system_prompt,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            use_adapter = use_adapter,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )

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

        def read_mailbox(timeout):
            try:
                return mailbox.get(timeout = timeout)
            except queue.Empty:
                return None

        # Read tokens from our private mailbox (the dispatcher owns resp_queue).
        try:
            yield from self._consume_token_stream(
                read_mailbox,
                lambda: self._drain_mailbox(mailbox, timeout = 5.0),
                crash_context = "generation",
                cancel_event = cancel_event,
                stats_holder = stats_holder,
                read_timeout = _DISPATCH_READ_TIMEOUT,
            )
        finally:
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)

    def _drain_mailbox(
        self,
        mailbox: queue.Queue,
        timeout: float = 5.0,
    ) -> None:
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

        Called by _generate_inner before the _gen_lock path so the dispatcher
        thread isn't competing for resp_queue reads.
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

        # Only stop dispatcher if all mailboxes drained. If compare requests
        # are still active, leave it running so their token routing isn't
        # killed mid-stream.
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

    def share_distributed_object(
        self,
        obj,
        timeout: Optional[float] = 300.0,
    ):
        """Share a small object through the worker's MLX distributed group."""
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")

        self._wait_dispatcher_idle()
        request_id = str(uuid.uuid4())
        cmd = {
            "type": "share_object",
            "request_id": request_id,
            "object": obj,
        }

        with self._gen_lock:
            self._send_cmd(cmd)
            deadline = None if timeout is None else time.monotonic() + timeout
            while deadline is None or time.monotonic() < deadline:
                remaining = 1.0 if deadline is None else max(0.1, deadline - time.monotonic())
                resp = self._read_resp(timeout = min(remaining, 1.0))
                if resp is None:
                    if not self._ensure_subprocess_alive():
                        raise RuntimeError(self._subprocess_crash_message("sharing chat turn"))
                    continue

                rtype = resp.get("type", "")
                rid = resp.get("request_id")
                if rid and rid != request_id:
                    logger.debug(
                        "Skipping response for request_id=%s while sharing request_id=%s",
                        rid,
                        request_id,
                    )
                    continue
                if rtype == "shared":
                    return resp.get("object")
                if rtype == "share_error":
                    raise RuntimeError(resp.get("error", "Failed to share object"))
                if rtype == "error":
                    raise RuntimeError(resp.get("error", "Subprocess error"))
                if rtype == "status":
                    continue

            raise RuntimeError("Timeout waiting for distributed object share")

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
        approved_remote_code_fingerprint: Optional[str] = None,
        gpu_ids: Optional[list[int]] = None,
        subject: Optional[str] = None,
        tensor_parallel: bool = False,
        mlx_distributed: bool = False,
    ) -> bool:
        """Load a model for inference.

        Always spawns a fresh subprocess per load for a clean interpreter (no
        stale unsloth patches, torch.compile caches, or getsource failures).
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
                "approved_remote_code_fingerprint": approved_remote_code_fingerprint,
                "subject": subject,
                "gpu_ids": gpu_ids,
                "tensor_parallel": bool(tensor_parallel),
                "mlx_distributed": bool(mlx_distributed),
                "mlx_parallel_mode": ("tensor" if tensor_parallel else "pipeline")
                if mlx_distributed
                else None,
            }
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
                gpu_ids,
                model_name = model_name,
                hf_token = hf_token,
                load_in_4bit = load_in_4bit,
            )
            sub_config["resolved_gpu_ids"] = resolved_gpu_ids
            sub_config["gpu_selection"] = gpu_selection

            # Always kill the existing subprocess and spawn fresh: reusing one
            # after unsloth patches torch internals breaks getsource on reload.
            if self._ensure_subprocess_alive():
                self._cancel_generation()
                time.sleep(0.3)
                self._shutdown_subprocess()

            elif self._proc is not None:
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
                    # First stall with Xet on -> retry with Xet disabled
                    if attempt == 0 and not disable_xet:
                        logger.warning(
                            "Download stalled for '%s' -- retrying with HF_HUB_DISABLE_XET=1",
                            model_name,
                        )
                        self._shutdown_subprocess(timeout = 5)
                        disable_xet = True
                        continue
                    # Second stall (or xet already off) -> give up
                    self._shutdown_subprocess(timeout = 5)
                    raise RuntimeError(
                        f"Download stalled for '{model_name}' even with "
                        f"HF_HUB_DISABLE_XET=1 -- check your network connection"
                    )

                if resp.get("success"):
                    model_info = resp.get("model_info", {})
                    self.active_model_name = model_info.get("identifier", model_name)
                    self.models[self.active_model_name] = {
                        "is_vision": model_info.get("is_vision", False),
                        "is_lora": model_info.get("is_lora", False),
                        "is_mlx": model_info.get("is_mlx", False),
                        "display_name": model_info.get("display_name", model_name),
                        "is_audio": model_info.get("is_audio", False),
                        "audio_type": model_info.get("audio_type"),
                        "has_audio_input": model_info.get("has_audio_input", False),
                        "context_length": model_info.get("context_length"),
                    }
                    # Mirror chat_template_info so routes can classify caps
                    # without re-entering the subprocess.
                    _tpl_info = model_info.get("chat_template_info")
                    if isinstance(_tpl_info, dict):
                        self.models[self.active_model_name]["chat_template_info"] = _tpl_info
                    self.loading_models.discard(model_name)
                    logger.info("Model '%s' loaded successfully in subprocess", model_name)
                    return True
                else:
                    # Worker reports failures (consent gate included) under "message".
                    error = resp.get("message") or resp.get("error") or "Failed to load model"
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
        if model_name in self.loading_models:
            logger.info(
                "Cancelling in-flight load for model '%s' by terminating subprocess",
                model_name,
            )
            self._shutdown_subprocess(timeout = 0.5)
            self.loading_models.discard(model_name)
            self.active_model_name = None
            self.models.clear()
            return True

        if not self._ensure_subprocess_alive():
            # No subprocess — clear local state
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
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Generate response, streaming tokens from subprocess.

        ``tools`` / ``enable_thinking`` / ``reasoning_effort`` /
        ``preserve_thinking`` are forwarded so the template can render tool
        schemas and reasoning controls.

        ``stats_holder``: caller-owned dict; on gen_done its "stats" key gets
        the worker's usage/timings. Request-scoped to avoid cross-stream reads.
        """
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
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            stats_holder = stats_holder,
        )

    def generate_chat_completion_with_tools(
        self,
        messages: list,
        tools: list,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        max_tool_iterations: int = 25,
        auto_heal_tool_calls: bool = True,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
        rag_scope: Optional[dict] = None,
        confirm_tool_calls: bool = False,
        bypass_permissions: bool = False,
        use_adapter: Optional[Union[bool, str]] = None,
        stats_holder: Optional[dict] = None,
        **_unused,
    ):
        """Run the safetensors agentic tool loop in the parent process,
        calling the worker for each turn.

        Yields the same event dicts as the GGUF tool loop so the route layer
        can stream both backends through one helper.
        """
        from core.inference.safetensors_agentic import run_safetensors_tool_loop
        from core.inference.tools import execute_tool

        max_new_tokens = max_tokens if max_tokens and max_tokens > 0 else 2048

        def _single_turn(conv: list, *, active_tools: Optional[list[dict]] = None):
            # ``conv`` already carries any system message. ``active_tools`` lets
            # run_safetensors_tool_loop drop one-shot tools (e.g. render_html) from
            # later same-response prompts.
            turn_tools = active_tools if active_tools is not None else tools
            common_kwargs = dict(
                messages = conv,
                system_prompt = "",
                image = None,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_new_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                cancel_event = cancel_event,
                tools = turn_tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                # last turn wins, like the GGUF tool loop
                stats_holder = stats_holder,
            )
            if use_adapter is not None:
                yield from self.generate_with_adapter_control(
                    use_adapter = use_adapter,
                    **common_kwargs,
                )
            else:
                yield from self.generate_chat_response(**common_kwargs)

        initial = list(messages)
        if system_prompt:
            initial = [{"role": "system", "content": system_prompt}] + initial

        yield from run_safetensors_tool_loop(
            single_turn = _single_turn,
            messages = initial,
            tools = tools,
            execute_tool = execute_tool,
            cancel_event = cancel_event,
            auto_heal_tool_calls = auto_heal_tool_calls,
            max_tool_iterations = max_tool_iterations,
            tool_call_timeout = tool_call_timeout,
            session_id = session_id,
            rag_scope = rag_scope,
            confirm_tool_calls = confirm_tool_calls,
            bypass_permissions = bypass_permissions,
        )

    def generate_with_adapter_control(
        self,
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        """Generate with adapter control, streaming tokens from subprocess.

        Uses the dispatcher path (no _gen_lock) so compare-mode requests
        don't block each other; the subprocess serializes them via its
        sequential command loop.
        """
        yield from self._generate_dispatched(
            use_adapter = use_adapter,
            cancel_event = cancel_event,
            stats_holder = stats_holder,
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
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Inner generation logic — sends command to subprocess, yields tokens.

        Serialized by _gen_lock (one generation at a time) so concurrent
        readers don't consume each other's tokens off the shared resp_queue.
        """
        if not self._ensure_subprocess_alive():
            yield "Error: Inference subprocess is not running"
            return

        if not self.active_model_name:
            yield "Error: No active model"
            return

        # Drain any prior compare-mode dispatcher so we can read resp_queue.
        self._wait_dispatcher_idle()

        # Serialize generation: two concurrent readers on resp_queue would
        # consume and drop each other's token events. Hold _gen_lock across the
        # cmd build + send + whole stream so we stay the sole resp_queue reader.
        with self._gen_lock:
            request_id = str(uuid.uuid4())
            image_b64 = self._pil_to_base64(image) if image is not None else None
            cmd = self._build_generate_cmd(
                request_id,
                image_b64,
                messages = messages,
                system_prompt = system_prompt,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_new_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                use_adapter = use_adapter,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
            )

            try:
                self._send_cmd(cmd)
            except RuntimeError as exc:
                yield f"Error: {exc}"
                return

            yield from self._consume_token_stream(
                self._read_resp,
                lambda: self._drain_until_gen_done(timeout = 5.0),
                crash_context = "generation",
                cancel_event = cancel_event,
                stats_holder = stats_holder,
            )

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

        Blocking — sends command and waits for the full audio response.
        """
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")
        if not self.active_model_name:
            raise RuntimeError("No active model")

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
                    raise RuntimeError(self._subprocess_crash_message("audio generation"))
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
            request_id = str(uuid.uuid4())

            # numpy array -> list for mp.Queue serialization
            audio_data = (
                audio_array.tolist() if hasattr(audio_array, "tolist") else list(audio_array)
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

            yield from self._consume_token_stream(
                self._read_resp,
                lambda: self._drain_until_gen_done(timeout = 5.0),
                crash_context = "audio input generation",
                cancel_event = cancel_event,
            )

    # ------------------------------------------------------------------
    # Local helpers (no subprocess needed)
    # ------------------------------------------------------------------

    def resize_image(
        self,
        img,
        max_size: int = 800,
    ):
        """Resize image preserving aspect ratio (runs locally, no ML imports)."""
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
        """Currently active model name."""
        return self.active_model_name

    def is_model_loading(self) -> bool:
        """True if any model is loading."""
        return len(self.loading_models) > 0

    def get_loading_model(self) -> Optional[str]:
        """Name of the currently loading model."""
        return next(iter(self.loading_models)) if self.loading_models else None

    def check_vision_model_compatibility(self) -> bool:
        """True if the current model supports vision."""
        if self.active_model_name and self.active_model_name in self.models:
            return self.models[self.active_model_name].get("is_vision", False)
        return False

    def _is_gpt_oss_model(self, model_name: str = None) -> bool:
        """Parent-side gpt-oss detection so the route avoids an IPC round-trip."""
        from utils.datasets import is_gpt_oss_model_name
        return is_gpt_oss_model_name(model_name or self.active_model_name or "")


# ========== GLOBAL INSTANCE ==========
_inference_backend = None


def get_inference_backend() -> InferenceOrchestrator:
    """Global inference backend instance (orchestrator)."""
    global _inference_backend
    if _inference_backend is None:
        _inference_backend = InferenceOrchestrator()
    return _inference_backend
