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

# Max wait for a cancelled generation to release _gen_lock before unload_model
# tears the subprocess down. Only bounds a wedged worker.
_UNLOAD_GEN_LOCK_TIMEOUT = 15.0


class GenStreamError(str):
    """A stream chunk carrying a real backend/generation error, not model text.

    Subclasses str so existing display/logging consumers are unaffected, while
    callers can distinguish a real error from model output whose visible text
    starts with "Error:" by checking isinstance(chunk, GenStreamError).
    """

    __slots__ = ("public",)

    def __new__(
        cls,
        value,
        *,
        public: bool = False,
    ):
        obj = str.__new__(cls, value)
        obj.public = bool(public)
        return obj


class GenStreamErrorRaised(RuntimeError):
    """Internal exception form of ``GenStreamError`` for generator boundaries."""

    __slots__ = ("public",)

    def __init__(
        self,
        value,
        *,
        public: bool = False,
    ):
        super().__init__(value)
        self.public = bool(public)


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
        # Set for the whole unload; the worker never clears it (unlike _cancel_event),
        # so a generate queued behind the cancelled one is skipped, not run.
        self._drain_event: Any = None
        self._gen_lock = threading.Lock()  # Serializes generation
        # Set during a switch so a generation winning the _gen_lock handoff bails
        # instead of starting on the outgoing model.
        self._unload_pending = False

        # Dispatcher state for compare mode (adapter-controlled requests):
        # bypass _gen_lock, send commands directly, read from per-request
        # mailboxes routed by a dispatcher thread on request_id.
        self._mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()
        # Serializes dispatcher start/stop. _generate_dispatched (compare mode) bypasses
        # _gen_lock, so two concurrent compare requests can both reach _start_dispatcher;
        # without this lock both could observe no live dispatcher and each spawn one,
        # orphaning the extra thread (self._dispatcher_thread tracks only the last). The
        # orphan later steals the "unloaded" reply off resp_queue and hangs unload_model.
        self._dispatcher_lifecycle_lock = threading.Lock()

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
        top_gguf = self._top_gguf_cache or []
        top_hub = self._top_hub_cache or []
        # Never wait for the remote Hugging Face ranking during startup. Chat's
        # first /api/models/list needs curated defaults immediately; the
        # background fetch backfills extra choices on later calls.
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
        # Same recheck as the training/export spawns, REPAIR reservations only: a
        # repair swaps without holding the lifecycle gate this load's caller owns,
        # while an install cannot swap until this gate is released (and then its
        # queued-load snapshot aborts it), so tolerating installs here lets the
        # load win instead of failing both sides. Also covers the OpenAI
        # auto-switch path, which enters _load_model_impl without route guards.
        from utils.transformers_version import (
            SidecarSwapInProgress,
            sidecar_swap_kind,
        )

        if sidecar_swap_kind() == "repair":
            raise SidecarSwapInProgress(
                "A transformers repair is replacing the latest sidecar; retry when it completes."
            )
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )

        from .worker import run_inference_process

        with native_path_secret_removed_for_child_start():
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            self._cancel_event = _CTX.Event()
            self._drain_event = _CTX.Event()

            self._proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = (run_inference_process,),
                kwargs = {
                    "cmd_queue": self._cmd_queue,
                    "resp_queue": self._resp_queue,
                    "cancel_event": self._cancel_event,
                    "drain_event": self._drain_event,
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

    def is_worker_alive(self) -> bool:
        """True while the inference subprocess is running, even with no model
        active (a failed load can leave a live worker holding sidecar modules)."""
        proc = self._proc
        return proc is not None and proc.is_alive()

    def _shutdown_subprocess(self, timeout: float = 10.0) -> bool:
        """Gracefully shut down the inference subprocess.

        Returns True only once the worker is confirmed dead. If it survives
        terminate/kill (e.g. wedged in an uninterruptible CUDA syscall that outlives
        SIGKILL) the live handle is KEPT, not nulled, so is_worker_alive() and the
        pre-swap liveness guard can still observe the survivor instead of a cleared
        handle and refuse the destructive sidecar swap."""
        self._stop_dispatcher()  # before killing subprocess
        if self._proc is None or not self._proc.is_alive():
            self._proc = None
            return True

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

        if self._proc is not None and self._proc.is_alive():
            # Survived SIGKILL (uninterruptible syscall): keep the handle so callers
            # and the pre-swap guard see a live worker rather than a nulled one.
            logger.error(
                "Inference subprocess still alive after terminate/kill; "
                "preserving its handle for the pre-swap liveness check"
            )
            return False

        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        self._cancel_event = None
        self._drain_event = None
        logger.info("Inference subprocess shut down")
        return True

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
        presence_penalty: float = 0.0,
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
            "presence_penalty": presence_penalty,
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
        # Latch this stream's subprocess/queue: if a wedged worker is torn down and a
        # later load spawns a fresh one, bail rather than re-block on the new queue
        # under _gen_lock (deadlock).
        initial_proc = self._proc
        initial_resp_queue = self._resp_queue
        while True:
            if self._proc is not initial_proc or self._resp_queue is not initial_resp_queue:
                yield GenStreamError(
                    f"Error: {self._subprocess_crash_message(crash_context)}",
                    public = True,
                )
                return
            resp = read_one(read_timeout)
            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    yield GenStreamError(
                        f"Error: {self._subprocess_crash_message(crash_context)}",
                        public = True,
                    )
                    return
                continue

            rtype = resp.get("type", "")
            if rtype == "status":
                continue
            # Subprocess-level error (no request_id); request-scoped failures
            # arrive as gen_error below.
            if rtype == "error" and not resp.get("request_id"):
                yield GenStreamError(f"Error: {resp.get('error', 'Unknown error')}")
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
                yield GenStreamError(f"Error: {resp.get('error', 'Unknown error')}")
                return

    # ------------------------------------------------------------------
    # Dispatcher — per-request mailbox routing for compare mode
    # ------------------------------------------------------------------

    def _start_dispatcher(self) -> bool:
        """Start the dispatcher thread if not already running.

        The dispatcher reads the shared resp_queue and routes responses to
        per-request mailbox queues, letting multiple adapter-controlled
        (compare) requests be in-flight without holding _gen_lock.

        The whole check-then-spawn runs under _dispatcher_lifecycle_lock so
        concurrent compare requests (which bypass _gen_lock) can't both observe
        no live dispatcher and each spawn one. Returns True only for the caller
        that actually started a new thread; False if one was already alive.
        """
        with self._dispatcher_lifecycle_lock:
            # Refuse to start while an unload is in progress. unload_model sets
            # _unload_pending under this same lock before it stops the idle
            # dispatcher, so a start queued behind that stop observes the unload
            # here and bails. Without this a fresh dispatcher would be spawned
            # after the stop, become the resp_queue reader, and consume the
            # worker's "unloaded" reply (unroutable, so dropped) before
            # unload_model's _wait_response sees it -- hanging the unload 300s.
            if self._unload_pending:
                return False
            if self._dispatcher_thread is not None and self._dispatcher_thread.is_alive():
                return False

            self._dispatcher_stop.clear()
            self._dispatcher_thread = threading.Thread(
                target = self._dispatcher_loop,
                daemon = True,
                name = "inference-dispatcher",
            )
            self._dispatcher_thread.start()
            logger.debug("Dispatcher thread started")
            return True

    def _stop_dispatcher(self) -> None:
        """Signal the dispatcher to stop and wait for it.

        Runs under _dispatcher_lifecycle_lock (paired with _start_dispatcher) so
        a stop can't interleave with a concurrent start. Callers must NOT hold
        _mailbox_lock here: this joins the dispatcher, and the dispatcher loop
        takes _mailbox_lock, so holding it would deadlock the join.
        """
        with self._dispatcher_lifecycle_lock:
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
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Dispatched generation — sends command without holding _gen_lock.

        Uses a per-request mailbox for tokens so two compare-mode requests can
        be queued at once. The subprocess still runs commands sequentially, so
        GPU work stays serialized; this only avoids orchestrator lock contention.
        """
        if not self._ensure_subprocess_alive():
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return

        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        # Latch the target model so the recheck below can detect a switch that completed
        # between _start_dispatcher and mailbox registration (mirrors the locked path's
        # expected_model check).
        expected_model = self.active_model_name

        # Switch in flight (unload waiting on _gen_lock). This path bypasses the lock,
        # so without this early-out a compare request would enqueue a generate on the
        # outgoing model and delay the switch.
        if self._unload_pending:
            yield GenStreamError("Error: model is being unloaded", public = True)
            return

        # Ensure the dispatcher runs. _start_dispatcher serializes concurrent starters under
        # _dispatcher_lifecycle_lock and returns True only for the caller that actually spawned
        # the thread, so at most one dispatcher ever exists even when two compare requests race
        # here. Derive dispatcher_preexisting from that atomic result (not a separate unlocked
        # is_alive() read): if THIS call started the dispatcher and then bails on a racing
        # unload, it must stop it again (see the unloading bail below).
        started = self._start_dispatcher()
        dispatcher_preexisting = not started

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
            presence_penalty = presence_penalty,
            use_adapter = use_adapter,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )

        # Create the mailbox BEFORE sending, rechecking _unload_pending under
        # _mailbox_lock: an unload sets _unload_pending before _wait_dispatcher_idle
        # reads _mailboxes under the same lock, so either the idle check sees this
        # mailbox (and tears the dispatcher down) or we see the unload and bail.
        # Registering after would orphan the mailbox and hang the compare stream forever.
        mailbox: queue.Queue = queue.Queue()
        with self._mailbox_lock:
            # _unload_pending alone is not enough: an unload that ran fully since
            # _start_dispatcher clears it in its finally and stops the dispatcher, so it
            # reads False here though the dispatcher is gone and the model swapped. Also
            # bail when the active model changed or the dispatcher died: a mailbox with no
            # dispatcher to route gen_done/gen_error hangs the compare stream.
            dispatcher_alive = (
                self._dispatcher_thread is not None and self._dispatcher_thread.is_alive()
            )
            unloading = (
                self._unload_pending
                or self.active_model_name != expected_model
                or not dispatcher_alive
            )
            if not unloading:
                self._mailboxes[request_id] = mailbox
            # When bailing without a mailbox, note whether any OTHER compare request still
            # routes through the dispatcher; if none and this call started it, stop it below.
            orphaned_dispatcher = unloading and not dispatcher_preexisting and not self._mailboxes
        if unloading:
            # A racing unload can pass its _wait_dispatcher_idle() while the dispatcher was
            # stopped, then set _unload_pending. The one we just started would otherwise
            # linger with no mailboxes, race unload_model's _wait_response for the "unloaded"
            # reply off resp_queue, and drop it as unroutable -- hanging the unload 300s. Stop
            # it here so the unload stays the sole resp_queue reader. Outside _mailbox_lock:
            # _stop_dispatcher joins the dispatcher, which itself takes that lock.
            if orphaned_dispatcher:
                self._stop_dispatcher()
            yield GenStreamError("Error: model is being unloaded", public = True)
            return

        try:
            self._send_cmd(cmd)
        except RuntimeError as exc:
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)
            yield GenStreamError(f"Error: {exc}")
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

    def _wait_dispatcher_idle(self) -> bool:
        """Wait for all dispatched requests to complete, then stop dispatcher.

        Returns True if the dispatcher was stopped (all mailboxes drained, or no
        dispatcher was running), and False if it was left running because compare
        requests were still active after _DISPATCH_IDLE_TIMEOUT.

        Called before the _gen_lock path so the dispatcher thread isn't competing
        for resp_queue reads.
        """
        if self._dispatcher_thread is None or not self._dispatcher_thread.is_alive():
            return True

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
            return False
        self._stop_dispatcher()
        return True

    def share_distributed_object(
        self,
        obj,
        timeout: Optional[float] = 300.0,
    ):
        """Share a small object through the worker's MLX distributed group."""
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")

        self._wait_dispatcher_idle()
        with self._mailbox_lock:
            if self._mailboxes:
                raise RuntimeError(
                    "Cannot share distributed objects while compare requests are active"
                )
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

    # Monotonic count of PUBLISHED loads; lets the install route detect a load
    # (including a same-model reload) that completed while it waited on the gate.
    # Bumped when the load result is published, not at load start: a start-time
    # bump is already visible when the installer snapshots mid-load, so the
    # completed reload would look unchanged and get unloaded by the swap.
    load_generation: int = 0

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

            # Recheck the sidecar reservation BEFORE tearing the old worker down,
            # for REPAIRS only: an install holds this same lifecycle gate, so it
            # cannot swap while this load runs, and its queued-load snapshot
            # aborts it after this load publishes -- the load wins cleanly.
            # Raising here (repair) keeps the current model loaded.
            from utils.transformers_version import (
                SidecarSwapInProgress,
                sidecar_swap_kind,
            )

            if sidecar_swap_kind() == "repair":
                raise SidecarSwapInProgress(
                    "A transformers repair is replacing the latest sidecar; "
                    "retry when it completes."
                )

            # Always kill the existing subprocess and spawn fresh: reusing one
            # after unsloth patches torch internals breaks getsource on reload.
            if self._ensure_subprocess_alive():
                self._cancel_generation()
                time.sleep(0.3)
                if self._shutdown_subprocess() is False:
                    # The worker survived terminate/kill (e.g. a wedged CUDA syscall that
                    # outlives SIGKILL). Its handle is kept, so is_worker_alive() and the
                    # pre-swap guard still see it; do not spawn a second worker over one
                    # still holding GPU memory. Fail so the load can retry once it exits.
                    raise RuntimeError(
                        "The current inference worker did not exit and still holds GPU "
                        "memory; not starting a new model over it. Retry shortly."
                    )
            elif self._proc is not None:
                self._shutdown_subprocess(timeout = 2)

            disable_xet = sub_config.get("disable_xet", False) or (
                os.environ.get("HF_HUB_DISABLE_XET") == "1"
            )

            for attempt in range(2):
                # Stop-loading (/unload -> cancel_load) aborts a load by discarding this
                # model's loading marker. cancel_load only kills a live child; if the cancel
                # lands before any child exists (GPU placement, or between retries) there is
                # nothing to kill, and without this check the loop would spawn a worker and
                # load the model after /unload reported it unloaded. Observe removal and stop.
                if model_name not in self.loading_models:
                    logger.info(
                        "Load for '%s' was cancelled before spawn; not starting a worker",
                        model_name,
                    )
                    self.active_model_name = None
                    self.models.clear()
                    return False
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

                # A cancel can land after the pre-spawn recheck but while _spawn_subprocess
                # is still creating the queues/process. cancel_load runs off the lifecycle
                # gate, so its _shutdown_subprocess can see _proc still None and no-op,
                # orphaning this fresh worker; the load would then wait for "loaded" and
                # publish a model /unload reported unloaded, over a live subprocess nothing
                # reaps. Recheck now the child exists and tear it down before publishing.
                if model_name not in self.loading_models:
                    logger.info(
                        "Load for '%s' was cancelled during spawn; tearing the worker down",
                        model_name,
                    )
                    self._shutdown_subprocess(timeout = 5)
                    self.active_model_name = None
                    self.models.clear()
                    return False

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
                    # A cancel can land while we were parked in _wait_response above.
                    # cancel_load (off the lifecycle gate) discards this model's loading
                    # marker BEFORE its teardown, so a Stop-loading that fired after the
                    # worker queued "loaded" (which we can still consume during cancel_load's
                    # shutdown window) shows up here only as the marker's removal. Without
                    # this recheck we would publish active_model_name/models for a model
                    # /unload reported cancelled, over a subprocess cancel_load just killed;
                    # its post-teardown re-clear cannot undo a publish that lands after it
                    # returns. Observe the removal and abort; cancel_load owns teardown.
                    if model_name not in self.loading_models:
                        logger.info(
                            "Load for '%s' was cancelled while waiting for 'loaded'; "
                            "not publishing the cancelled model",
                            model_name,
                        )
                        self.active_model_name = None
                        self.models.clear()
                        return False
                    model_info = resp.get("model_info", {})
                    self.active_model_name = model_info.get("identifier", model_name)
                    self.load_generation += 1
                    # A load always spawns a fresh subprocess holding only this model, so
                    # mirror that. A lingering stale name would pass unload_model's "not in
                    # self.models" guard, and the worker's absent-name fallback would unload
                    # its *active* model, not the already-gone one.
                    self.models = {}
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

        except Exception as exc:
            self.loading_models.discard(model_name)
            from utils.transformers_version import SidecarSwapInProgress

            if isinstance(exc, SidecarSwapInProgress) and self._ensure_subprocess_alive():
                # Raised before the old worker was torn down: the previous model
                # is still live, so keep the mirrors (clearing them would let the
                # installer treat the worker as inactive and kill it unreported).
                raise
            self.active_model_name = None
            self.models.clear()
            raise

    def cancel_load(self, model_name: str) -> bool:
        """Abort an in-flight load by terminating its subprocess.

        Returns True if a load for ``model_name`` (matched case-insensitively) was
        cancelled, False if nothing was loading under that name. This only tears the
        loading subprocess down -- it sends no command to a worker -- so, unlike the
        rest of ``unload_model``, it is safe to run WITHOUT the inference lifecycle
        gate. ``/unload`` calls it off-gate so the "stop loading" button can interrupt
        a safetensors load that holds the gate for its whole (multi-minute) duration;
        a gated cancel could never preempt that load.
        """
        target = model_name
        if target not in self.loading_models:
            target = next(
                (m for m in self.loading_models if m.lower() == model_name.lower()),
                model_name,
            )
        if target not in self.loading_models:
            return False
        logger.info(
            "Cancelling in-flight load for model '%s' by terminating subprocess",
            target,
        )
        # Discard the loading marker (and clear local state) BEFORE the teardown, not
        # after. cancel_load runs off the lifecycle gate, alongside a load_model that
        # rechecks this marker before each spawn. But _shutdown_subprocess can block (~1s
        # tearing a live child down and joining the dispatcher), so clearing only after
        # leaves a window where load_model reads the marker still set, passes its pre-spawn
        # recheck, and loads the model after /unload reported it cancelled. Clear first.
        self.loading_models.discard(target)
        self.active_model_name = None
        self.models.clear()
        self._shutdown_subprocess(timeout = 0.5)
        # Clear the local mirrors again AFTER the teardown. A racing off-gate load_model
        # may still be parked in _wait_response("loaded"): its worker already queued a
        # "loaded" reply, so during the shutdown window above (the 0.5s settle before the
        # response queue is drained and nulled) that thread can consume it and repopulate
        # active_model_name/models, undoing the pre-teardown clear. _shutdown_subprocess
        # nulls the queue but not the mirrors, so without this second clear /unload reports
        # success while the backend still advertises a killed model. The nulled queue lets
        # no further "loaded" through, so re-clearing here wipes any repopulation.
        self.active_model_name = None
        self.models.clear()
        return True

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from the subprocess."""
        # active_model_name can differ in case from the client's raw /unload name (the
        # load path canonicalizes casing). Match case-insensitively and use the canonical
        # spelling so the guard, unload command, and cleanup below hit the loaded model.
        if (
            self.active_model_name is not None
            and model_name != self.active_model_name
            and model_name.lower() == self.active_model_name.lower()
        ):
            model_name = self.active_model_name
        # In-flight load: tear its subprocess down (shared loading-cancel logic; no
        # worker command sent).
        if self.cancel_load(model_name):
            return True

        if not self._ensure_subprocess_alive():
            # No subprocess — clear local state
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None
            return True

        # Nothing loaded under this name: don't unload a stale model. The worker falls
        # back to unloading its *active* model when the name is absent, so a stale unload
        # (lost a race to a concurrent load) would hit the wrong one.
        if model_name != self.active_model_name and model_name not in self.models:
            self.models.pop(model_name, None)
            return True

        # The subprocess runs commands sequentially, so a bare unload queues behind a
        # running generate (a 2-3 min hang). Cancel first (via the mp.Event the worker
        # polls each token), then take _gen_lock as sole resp_queue reader (like GGUF).
        #
        # Set _unload_pending under _dispatcher_lifecycle_lock so it is ordered ahead of
        # the dispatcher stop that _wait_dispatcher_idle runs under the same lock: a
        # compare request's _start_dispatcher queued behind that stop then observes the
        # unload and refuses to spawn a fresh dispatcher that would eat the "unloaded"
        # reply off resp_queue. This is a standalone acquisition (no _gen_lock held yet),
        # so it keeps the _gen_lock -> _dispatcher_lifecycle_lock order and can't deadlock.
        with self._dispatcher_lifecycle_lock:
            self._unload_pending = True
        # Cancelling only the running generation isn't enough: the worker clears
        # cancel_event at each generate start, so a queued one would clear it and run the
        # outgoing model to completion. drain_event, never cleared, makes any generate
        # dequeued during the unload skip.
        if self._drain_event is not None:
            self._drain_event.set()
        try:
            self._cancel_generation()
            acquired = self._gen_lock.acquire(timeout = _UNLOAD_GEN_LOCK_TIMEOUT)
            if not acquired:
                # Wedged worker: tear the subprocess down to free the GPU (next load respawns).
                logger.warning(
                    "Unload: generation did not yield %.1fs after cancel; "
                    "shutting the inference subprocess down to free the model",
                    _UNLOAD_GEN_LOCK_TIMEOUT,
                )
                self._shutdown_subprocess(timeout = 5)
                self.models.pop(model_name, None)
                if self.active_model_name == model_name:
                    self.active_model_name = None
                return True

            try:
                # Stop the compare-mode dispatcher so it can't consume the "unloaded" reply
                # off resp_queue before we do. A dispatched generation bypasses _gen_lock, so
                # a wedged one slips past the acquire above; if the dispatcher is still active
                # it owns resp_queue and the queued unload hangs _wait_response behind the
                # stuck generate. Mirror the wedged locked path: tear the subprocess down.
                if not self._wait_dispatcher_idle():
                    logger.warning(
                        "Unload: compare-mode dispatcher still active after idle "
                        "wait; shutting the inference subprocess down to free the model"
                    )
                    self._shutdown_subprocess(timeout = 5)
                    self.models.pop(model_name, None)
                    if self.active_model_name == model_name:
                        self.active_model_name = None
                    return True
                # Drop stale tokens so they can't be read as the unload reply.
                self._drain_queue()
                self._send_cmd(
                    {
                        "type": "unload",
                        "model_name": model_name,
                    }
                )
                self._wait_response("unloaded")

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
            finally:
                self._gen_lock.release()
        finally:
            self._unload_pending = False
            if self._drain_event is not None:
                self._drain_event.clear()

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
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Generate response, streaming tokens from subprocess.

        ``tools`` / ``enable_thinking`` / ``reasoning_effort`` /
        ``preserve_thinking`` are forwarded so the template can render tool
        schemas and reasoning controls.

        ``stats_holder``: caller-owned dict; on gen_done its "stats" key gets
        the worker's usage/timings. Request-scoped to avoid cross-stream reads.

        ``presence_penalty`` matches the GGUF sampling path (0 disables it).
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
            presence_penalty = presence_penalty,
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
        nudge_tool_calls: Optional[bool] = None,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        rag_scope: Optional[dict] = None,
        confirm_tool_calls: bool = False,
        bypass_permissions: bool = False,
        permission_mode: Optional[str] = None,
        use_adapter: Optional[Union[bool, str]] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        reasoning_prefilled: bool = False,
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
                presence_penalty = presence_penalty,
            )
            if use_adapter is not None:
                stream = self.generate_with_adapter_control(
                    use_adapter = use_adapter,
                    **common_kwargs,
                )
            else:
                stream = self.generate_chat_response(**common_kwargs)
            close_stream = False
            try:
                for chunk in stream:
                    if isinstance(chunk, GenStreamError):
                        close_stream = True
                        raise GenStreamErrorRaised(str(chunk), public = chunk.public)
                    yield chunk
            finally:
                if close_stream:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug("failed to close errored generation stream", exc_info = True)

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
            nudge_tool_calls = nudge_tool_calls,
            max_tool_iterations = max_tool_iterations,
            tool_call_timeout = tool_call_timeout,
            session_id = session_id,
            thread_id = thread_id,
            rag_scope = rag_scope,
            confirm_tool_calls = confirm_tool_calls,
            bypass_permissions = bypass_permissions,
            permission_mode = permission_mode,
            reasoning_prefilled = reasoning_prefilled,
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
        sequential command loop. Backend failures raise instead of becoming
        assistant text.
        """
        stream = self._generate_dispatched(
            use_adapter = use_adapter,
            cancel_event = cancel_event,
            stats_holder = stats_holder,
            **gen_kwargs,
        )
        try:
            for chunk in stream:
                if isinstance(chunk, GenStreamError):
                    # Preserve the public/operational flag so the route can surface
                    # the real message (e.g. "model is being unloaded") instead of a
                    # generic error. Mirrors the safetensors tool loop's _single_turn.
                    raise GenStreamErrorRaised(str(chunk), public = chunk.public)
                yield chunk
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

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
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Inner generation logic — sends command to subprocess, yields tokens.

        Serialized by _gen_lock (one generation at a time) so concurrent
        readers don't consume each other's tokens off the shared resp_queue.
        """
        if not self._ensure_subprocess_alive():
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return

        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        expected_model = self.active_model_name

        # Drain any prior compare-mode dispatcher so we can read resp_queue.
        self._wait_dispatcher_idle()

        # Serialize generation: two concurrent readers on resp_queue would
        # consume and drop each other's token events. Hold _gen_lock across the
        # cmd build + send + whole stream so we stay the sole resp_queue reader.
        with self._gen_lock:
            # Recheck under the lock: an unload we raced may have cleared/swapped the model.
            # _unload_pending resets after the lock releases, so it can read False by now;
            # the active-model check catches that handoff and a reload that swapped models,
            # so we never generate on the wrong one.
            if self._unload_pending or self.active_model_name != expected_model:
                # Won the lock handoff during a switch; don't start on the outgoing model.
                yield GenStreamError("Error: model is being unloaded", public = True)
                return
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
                presence_penalty = presence_penalty,
                use_adapter = use_adapter,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
            )

            try:
                self._send_cmd(cmd)
            except RuntimeError as exc:
                yield GenStreamError(f"Error: {exc}")
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
        expected_model = self.active_model_name

        # Serialize under _gen_lock (sole resp_queue reader) and refuse to start on the
        # outgoing model once an unload is pending, like the text and audio-input paths.
        # Without this a concurrent /audio/generate could run TTS on a model being switched.
        with self._gen_lock:
            # Recheck under the lock (see _generate_inner): a raced unload/switch may have
            # cleared or swapped the model while we waited.
            if self._unload_pending or self.active_model_name != expected_model:
                raise RuntimeError("model is being unloaded")

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
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return
        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        expected_model = self.active_model_name

        with self._gen_lock:
            # Recheck under the lock (see _generate_inner): a raced unload/switch may have
            # cleared or swapped the model while we waited.
            if self._unload_pending or self.active_model_name != expected_model:
                # Won the lock handoff during a switch; don't start on the outgoing model.
                yield GenStreamError("Error: model is being unloaded", public = True)
                return
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
                yield GenStreamError(f"Error: {exc}")
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
