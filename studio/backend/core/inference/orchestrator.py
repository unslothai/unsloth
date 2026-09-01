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
from typing import Any, Generator, Optional, Sequence, Tuple, Union
from core.inference.audio_errors import (
    AUDIO_UNSUPPORTED_CODE,
    AudioBackendUnsupportedError,
    AudioGenerationCancelledError,
)
from utils.hardware import get_device, prepare_gpu_selection
from utils.utils import hf_env_offline

# Re-exported from the shared helper so GGUF, training and inference share one type.
# Via PEP 562, not a module-level import: resolving the name imports unsloth_zoo, hence
# torch, and routes/inference.py imports this module at startup only for GenStream*.
DownloadStallError: type


def __getattr__(name: str):
    if name == "DownloadStallError":
        from utils.hf_xet_fallback import DownloadStallError as _exc
        return _exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = get_logger(__name__)


class _LoadCancelled(Exception):
    """Internal control flow for a caller-cancelled model load."""


_CTX = mp.get_context("spawn")


# Dispatcher timeout constants (seconds)
_DISPATCH_READ_TIMEOUT = 30.0
_DISPATCH_POLL_INTERVAL = 0.5
_DISPATCH_STOP_TIMEOUT = 5.0
_DISPATCH_IDLE_TIMEOUT = 30.0
_DISPATCH_DRAIN_TIMEOUT = 5.0

# Only bounds the Transformers subprocess path; llama.cpp TTS never reaches here.
# 120s was tuned against GGUF speeds and killed real work: a safetensors LoRA on a
# mid-range GPU needs minutes for the same clip a GGUF returns in seconds. A dead
# worker is already caught every second by _ensure_subprocess_alive, so this only
# has to bound one that is alive and wedged, and a generous value costs nothing.
_AUDIO_GENERATION_TIMEOUT = 900.0
_AUDIO_GENERATION_BASE_TOKENS = 2048
AUDIO_GENERATION_MAX_TOKENS = 8192
MOSS_TTS_MAX_FRAMES = 32768
MINIMAX_MUSIC_MAX_FRAMES = 9000
_AUDIO_CANCEL_DRAIN_TIMEOUT = 5.0
# Before audio_started there is nobody to receive the cancel, and a prefill pass (a 3B TTS
# model on CPU, or OuteTTS's per-token Python repetition penalty) routinely outlasts the
# drain window. Tearing down on that budget unloads the model the user just loaded.
_AUDIO_CANCEL_TEARDOWN_TIMEOUT = 30.0

# Max wait for a cancelled generation to release _gen_lock before unload_model
# tears the subprocess down. Only bounds a wedged worker.
_UNLOAD_GEN_LOCK_TIMEOUT = 15.0


def _audio_generation_timeout(
    max_new_tokens: int,
    base: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> float:
    """Scale a floor by the requested token count.

    ``base`` differs per backend by an order of magnitude: the Transformers subprocess
    needs minutes for a clip llama.cpp returns in seconds, so llama_cpp.py passes its
    own. Sharing one base silently gave every GGUF read the Transformers budget, which
    holds other_inference_request_count() up and blocks idle auto-unload for that long.

    Resolved at call time, not bound as a default: a default is evaluated once at import,
    so reassigning the module constant afterwards had no effect.
    """
    if base is None:
        base = _AUDIO_GENERATION_TIMEOUT
    if max_tokens is None:
        max_tokens = AUDIO_GENERATION_MAX_TOKENS
    max_new_tokens = min(max(1, int(max_tokens)), max(1, int(max_new_tokens)))
    token_scale = max(1.0, max_new_tokens / _AUDIO_GENERATION_BASE_TOKENS)
    return base * token_scale


_MLX_RUNTIME_MIRROR_FIELDS = (
    "mlx_kv_bits",
    "mlx_kv_bits_requested",
    "mlx_kv_quant_eligibility",
    "mlx_kv_quant_reason",
    "mlx_kv_quant_note",
    "chat_template_override_requested",
    "chat_template_override_reason",
)


def _mlx_runtime_mirror_fields(model_info: dict) -> dict:
    """MLX runtime state the parent mirrors, omitting what was not reported.

    Only the MLX backend sends these. Creating the keys for every backend would
    make the reload comparison see a None the backend never stored, and reload
    on every identical request.
    """
    return {key: model_info[key] for key in _MLX_RUNTIME_MIRROR_FIELDS if key in model_info}


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


def _summed_tool_loop_stats(total, turn):
    """Fold one tool-loop turn's report into the loop's running total.

    Every turn spends its tokens on the same request, so the reply reports their
    sum, as the llama.cpp tool loop does; reporting only the last turn hides the
    tokens that produced the tool call. The prompt count is the last turn's to
    report one, which already contains the tool results the earlier turns produced.
    """
    if not isinstance(turn, dict):
        return total
    if not isinstance(total, dict):
        return turn
    prior_usage = total.get("usage") or {}
    usage = dict(turn.get("usage") or {})
    completion = (usage.get("completion_tokens") or 0) + (prior_usage.get("completion_tokens") or 0)
    usage["completion_tokens"] = completion
    # The prompt is the loop's, not one turn's, so a turn that ended before
    # reporting keeps the last count that arrived. Its details describe that same
    # count and move with it, or cached tokens could outnumber prompt tokens.
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = prior_usage.get("prompt_tokens") or 0
        usage.pop("prompt_tokens_details", None)
        if prior_usage.get("prompt_tokens_details") is not None:
            usage["prompt_tokens_details"] = prior_usage["prompt_tokens_details"]
    usage["total_tokens"] = usage["prompt_tokens"] + completion
    # Details describe the completion, so they sum with it rather than describing
    # one turn against every turn's tokens.
    details = dict(prior_usage.get("completion_tokens_details") or {})
    for field, value in (usage.get("completion_tokens_details") or {}).items():
        details[field] = (details.get(field) or 0) + (value or 0)
    if details:
        usage["completion_tokens_details"] = details
    summed = dict(turn)
    summed["usage"] = usage
    timings = dict(turn.get("timings") or {})
    prior = total.get("timings") or {}
    # Seeded from the turn but folded unconditionally, as the llama.cpp loop does:
    # a turn reporting no timings must not take the loop's totals with it.
    if timings or prior:
        for field in ("predicted_ms", "predicted_n"):
            timings[field] = (timings.get(field) or 0) + (prior.get(field) or 0)
        # Rates describe the totals above, not the turn they arrived with: leaving
        # the last turn's would report a speed the summed counts contradict.
        predicted_ms = timings.get("predicted_ms") or 0
        predicted_n = timings.get("predicted_n") or 0
        timings["predicted_per_token_ms"] = (predicted_ms / predicted_n) if predicted_n else 0.0
        timings["predicted_per_second"] = (
            (predicted_n / (predicted_ms / 1000.0)) if predicted_ms else 0.0
        )
        summed["timings"] = timings
    return summed


def _mirrored_model_entry(model_info: dict, model_name: str) -> dict:
    """The parent's view of a model the worker holds.

    Measured or classified in the subprocess and unrecoverable once the model lives
    there, so a field the worker sends and this does not copy is one the API can never
    report.
    """
    return {
        "is_vision": model_info.get("is_vision", False),
        "is_lora": model_info.get("is_lora", False),
        "is_mlx": model_info.get("is_mlx", False),
        "display_name": model_info.get("display_name", model_name),
        "is_audio": model_info.get("is_audio", False),
        "audio_type": model_info.get("audio_type"),
        "has_audio_input": model_info.get("has_audio_input", False),
        "context_length": model_info.get("context_length"),
        "native_context_length": model_info.get("native_context_length"),
        "max_context_length": model_info.get("max_context_length"),
        "requested_context_length": model_info.get("requested_context_length"),
        "context_length_enforced": model_info.get("context_length_enforced"),
    }


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
        self._subprocess_shutdown_lock = threading.Lock()
        self._cancel_event: Any = None  # mp.Event — set to cancel generation
        # Set for the whole unload; the worker never clears it (unlike _cancel_event),
        # so a generate queued behind the cancelled one is skipped, not run.
        self._drain_event: Any = None
        self._gen_lock = threading.Lock()  # Serializes generation
        # Cancel event of the request holding _gen_lock: lets a Stop tell whether it owns the
        # running generation or is queued behind it (the worker's event is shared).
        self._active_cancel_events: list = []
        self._executing_cancel_events: list = []
        self._active_cancel_lock = threading.Lock()
        # Held across claim + _send_cmd so claim order matches the subprocess dequeue order,
        # which _owns_worker relies on.
        self._send_order_lock = threading.Lock()
        # Set during a switch so a generation winning the _gen_lock handoff bails
        # instead of starting on the outgoing model.
        self._unload_pending = False
        # Blocking TTS has no token response that can safely establish worker ownership
        # while it is queued behind compare-mode work. Reserve the dispatcher admission
        # lane for the whole TTS request so its shared cancel event cannot hit a sibling.
        # Mutated under _dispatcher_lifecycle_lock while holding _gen_lock.
        self._exclusive_tts_pending = False
        # A live VRAM probe is another exclusive worker command.  Compare-mode
        # requests bypass _gen_lock, so reserve their admission lane while the
        # parent asks the idle worker for allocator-owned bytes.
        self._exclusive_vram_probe_pending = False

        # Dispatcher state for compare mode (adapter-controlled requests):
        # bypass _gen_lock, send commands directly, read from per-request
        # mailboxes routed by a dispatcher thread on request_id.
        self._mailboxes: dict[str, queue.Queue] = {}
        # request_id -> cancel event, so the dispatcher can move worker ownership as it routes.
        # Consumers read their mailbox whenever they get to it, so only the dispatcher sees
        # responses in the order the worker produced them.
        self._request_cancel_events: dict[str, object] = {}
        # Mailboxes for the _gen_lock generations. Kept apart from _mailboxes because that map
        # means "compare requests are in flight" to the unload and distributed paths.
        self._direct_mailboxes: dict[str, queue.Queue] = {}
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

        # The list depends on detection (chat-only hosts get the GGUF set) and the MLX
        # self-heal re-detects, so unchecked a repaired Mac serves the chat-only list
        # forever. Stamp read BEFORE the list, or a re-detect tags the old list as new.
        import utils.hardware.hardware as _hw_mod

        self._static_models_generation = _hw_mod.DETECTION_GENERATION
        self._static_models = get_default_models()
        # Own lock for the stamp/value pair; the construction lock is held across a
        # build that waits on hardware detection.
        self._static_models_lock = threading.Lock()
        self._top_gguf_cache: Optional[list[str]] = None
        self._top_hub_cache: Optional[list[str]] = None
        self._top_models_ready = threading.Event()

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

        # Deliberately NOT started here: construction now runs on the startup warm thread, so
        # fetching from __init__ would call huggingface.co on every boot. First reader starts it.
        self._top_models_started = False

    # ------------------------------------------------------------------
    # Default models (top GGUFs fetched dynamically from HF)
    # ------------------------------------------------------------------

    def _refresh_static_models_if_stale(self) -> None:
        """Recompute the curated defaults if hardware was re-detected since."""
        import utils.hardware.hardware as _hw_mod

        generation = _hw_mod.DETECTION_GENERATION
        if generation == self._static_models_generation:
            return
        from core.inference.defaults import get_default_models

        # Built outside the lock so readers do not queue behind the torch import.
        models = get_default_models()
        with self._static_models_lock:
            # Commit only while still the newest: a slow reader storing its older list
            # under a newer stamp would look current for the life of the process.
            if generation != _hw_mod.DETECTION_GENERATION:
                return
            if generation <= self._static_models_generation:
                return
            self._static_models = models
            self._static_models_generation = generation
        logger.info("hardware was re-detected; curated default models refreshed")

    def _start_top_models_fetch(self) -> None:
        """Kick the remote ranking fetch once, on first read of the model list.

        Guarded by the construction lock, so two concurrent first-readers cannot each put up
        a thread. Skipped when the host asked for no outbound calls: the fetch is a raw
        httpx.get, so HF_HUB_OFFLINE does not reach it on its own. Via hf_env_offline(), not
        a literal "1" test, since HF_HUB_OFFLINE=true/on and TRANSFORMERS_OFFLINE count too.
        """
        if self._top_models_started:
            return
        # Checked before the latch: claiming it while offline would retire the fetch for the
        # process, so an offline boot or a temporary force_hf_offline() could never recover.
        if hf_env_offline():
            logger.info("offline mode requested; skipping the remote top-models ranking")
            return
        with _inference_backend_lock:
            if self._top_models_started:
                return
            self._top_models_started = True
        threading.Thread(target = self._fetch_top_models, daemon = True, name = "top-models").start()

    @property
    def default_models(self) -> list[str]:
        self._refresh_static_models_if_stale()
        self._start_top_models_fetch()
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
                # Counts at info, ids at debug: two lists of 40 repo names, one line each,
                # cost ~1.5 KB of every boot to say the catalog fetch worked.
                if gguf_ids:
                    self._top_gguf_cache = gguf_ids
                    logger.info("Fetched %d top GGUF models", len(gguf_ids))
                    logger.debug("Top GGUF models: %s", gguf_ids)
                if hub_ids:
                    self._top_hub_cache = hub_ids
                    logger.info("Fetched %d top hub models", len(hub_ids))
                    logger.debug("Top hub models: %s", hub_ids)
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
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths

        cache_env = get_hf_cache_paths().child_env({})

        with (
            child_environment_for_spawn(cache_env),
            native_path_secret_removed_for_child_start(),
        ):
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            self._cancel_event = _CTX.Event()
            self._drain_event = _CTX.Event()

            self._proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = ("core.inference.worker", "run_inference_process", cache_env),
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

    def post_handoff_gpu_availability_gb(
        self,
    ) -> Optional[tuple[dict[int, float], dict[int, float], dict[int, float]]]:
        """Atomically snapshot live free, total and disposable-worker VRAM.

        The worker is queried only while idle.  This avoids retaining a load-time
        allocator value after reset_generation_state() releases cache, and it keeps
        compare-mode's queue reader from consuming the probe response.  The send
        lock spans the system and worker reads, so a reset cannot make the same
        cache bytes appear in both values.
        """
        proc = self._proc
        active = self.active_model_name
        if proc is None or not proc.is_alive() or not active:
            return None

        with self._dispatcher_lifecycle_lock:
            if self._exclusive_tts_pending or getattr(self, "_exclusive_vram_probe_pending", False):
                return None
            self._exclusive_vram_probe_pending = True
        acquired = False
        try:
            if not self._wait_dispatcher_idle():
                return None
            acquired = self._gen_lock.acquire(timeout = 5.0)
            if not acquired:
                return None
            with self._active_cancel_lock:
                if self._active_cancel_events:
                    return None
            if self._proc is not proc or not proc.is_alive() or self.active_model_name != active:
                return None
            request_id = str(uuid.uuid4())
            with self._send_order_lock:
                from utils.hardware import get_visible_gpu_utilization

                live_free: dict[int, float] = {}
                total_by_index: dict[int, float] = {}
                for device in get_visible_gpu_utilization().get("devices", []):
                    try:
                        index = int(device["index"])
                        total = float(device["vram_total_gb"])
                        used = float(device["vram_used_gb"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    live_free[index] = max(total - used, 0.0)
                    total_by_index[index] = total
                if not live_free:
                    return None
                self._send_cmd({"type": "gpu_memory", "request_id": request_id})
                response = self._wait_response(
                    "gpu_memory",
                    timeout = 5.0,
                    expected_request_id = request_id,
                )
            if self._proc is not proc or not proc.is_alive() or self.active_model_name != active:
                return None
            reported = response.get("reclaimable_gpu_gb")
            if not isinstance(reported, dict):
                return None
            try:
                reclaimable = {
                    int(index): max(float(value), 0.0) for index, value in reported.items()
                }
                return live_free, total_by_index, reclaimable
            except (TypeError, ValueError):
                return None
        except Exception as exc:
            logger.warning("Could not query inference worker GPU memory: %s", exc)
            return None
        finally:
            if acquired:
                self._gen_lock.release()
            with self._dispatcher_lifecycle_lock:
                self._exclusive_vram_probe_pending = False

    def _shutdown_subprocess(self, timeout: float = 10.0) -> bool:
        with self._subprocess_shutdown_lock:
            return self._shutdown_subprocess_locked(timeout)

    def _shutdown_subprocess_locked(self, timeout: float) -> bool:
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
                from utils.process_lifetime import terminate_pid
                terminate_pid(self._proc.pid, timeout = 5)
                self._proc.join(timeout = 5)
            except Exception:
                pass
            if self._proc is not None and self._proc.is_alive():
                logger.warning("Process-tree shutdown failed, terminating the worker directly")
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
        self._reset_worker_scoped_state()
        logger.info("Inference subprocess shut down")
        return True

    @staticmethod
    def _wait_for_worker_vram_settle(
        since_kill: float,
        *,
        expected_free_gb: Optional[dict[int, float]] = None,
        max_wait: float = 2.0,
        interval: float = 0.25,
        tolerance_mib: int = 256,
    ) -> bool:
        """Poll cross-platform live telemetry until dead-worker VRAM stabilises."""
        from utils.hardware import get_visible_gpu_utilization

        if since_kill <= 0:
            return not expected_free_gb
        deadline = time.monotonic() + max(max_wait, 0.0)

        def _probe() -> Optional[dict[int, int]]:
            if time.monotonic() >= deadline:
                return None
            try:
                result: dict[int, int] = {}
                for device in get_visible_gpu_utilization().get("devices", []):
                    index = int(device["index"])
                    total = float(device["vram_total_gb"])
                    used = float(device["vram_used_gb"])
                    result[index] = max(int((total - used) * 1024), 0)
                return result or None
            except (KeyError, TypeError, ValueError, OverflowError):
                return None

        previous = _probe()
        if not previous:
            return not expected_free_gb
        expected_mib = {
            int(index): max(int(float(value) * 1024), 0)
            for index, value in (expected_free_gb or {}).items()
        }

        def _threshold_reached(sample: dict[int, int]) -> bool:
            return bool(expected_mib) and all(
                sample.get(index, -1) >= required for index, required in expected_mib.items()
            )

        if _threshold_reached(previous):
            return True
        observed_reclaim = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))
            current = _probe()
            if not current or current.keys() != previous.keys():
                return not expected_free_gb
            if any(
                current[index] - previous[index]
                >= max(tolerance_mib, int(max(current[index], previous[index]) * 0.02))
                for index in current
            ):
                observed_reclaim = True
            if _threshold_reached(current):
                return True
            stable = all(
                abs(current[index] - previous[index])
                < max(tolerance_mib, int(max(current[index], previous[index]) * 0.02))
                for index in current
            )
            # Two unchanged low samples can precede delayed driver reclaim.
            # Stability is meaningful only after an upward release was observed;
            # otherwise consume the full bounded window.
            if stable and observed_reclaim and not expected_mib:
                return True
            previous = current
        return not expected_free_gb

    def _reset_worker_scoped_state(self) -> None:
        """Drop bookkeeping that only means anything for the worker that just died.

        Ownership is scoped by cancel-event identity alone, so a consumer still blocked
        on its mailbox when the process was replaced stayed recorded as the executor. A
        generation on the fresh worker then failed _owns_worker and could not be stopped.
        Mailboxes go too: nothing will ever route to them, and a stale one reads as
        compare activity to the unload path.
        """
        with self._active_cancel_lock:
            self._active_cancel_events.clear()
            self._executing_cancel_events.clear()
        with self._mailbox_lock:
            self._mailboxes.clear()
            self._direct_mailboxes.clear()
            self._request_cancel_events.clear()

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
            return f"{message}{suffix} Details: pid={pid}, signal={sig_name}, exitcode={exitcode}."

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
        expected_request_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Block until a response of the expected type arrives.

        Also handles 'status' and 'error' events during the wait. Returns the
        matching response dict; raises RuntimeError on timeout or crash.

        *timeout* is an **inactivity** timeout: it resets on each status
        message, so long-running operations (large downloads, slow loads)
        survive as long as the subprocess keeps reporting progress.
        """
        # Local: resolving this name runs the shim's lazy unsloth_zoo load, which pulls torch.
        # The shim caches its pick, so this site and load_model()'s `except` see one class.
        from utils.hf_xet_fallback import DownloadStallError

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise _LoadCancelled()
            remaining = max(0.1, deadline - time.monotonic())
            poll_seconds = 0.1 if cancel_event is not None else 1.0
            resp = self._read_resp(timeout = min(remaining, poll_seconds))

            if resp is None:
                # Check subprocess health
                if not self._ensure_subprocess_alive():
                    raise RuntimeError(self._subprocess_crash_message("wait"))
                continue

            rtype = resp.get("type", "")

            if rtype == expected_type and (
                expected_request_id is None or resp.get("request_id") == expected_request_id
            ):
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
            f"Timeout waiting for '{expected_type}' response (no activity for {timeout}s)"
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

    def _direct_reader(self, request_id: str):
        """Response reader for a _gen_lock generation, safe once compare exists.

        The dispatcher and this reader would otherwise both consume _resp_queue. A
        dispatcher started mid-stream took our responses and dropped them as
        unaddressed (truncating or hanging the chat), and this reader, already blocked
        on the queue, could take a compare request's response before that dispatcher
        saw it. Registering a mailbox fixes the first; handing foreign responses to
        their own mailbox fixes the second.

        Returns (read_one, drain, release).
        """
        mailbox: queue.Queue = queue.Queue()
        with self._mailbox_lock:
            self._direct_mailboxes[request_id] = mailbox

        def read_one(timeout: float = 1.0):
            try:
                return mailbox.get_nowait()
            except queue.Empty:
                pass
            thread = self._dispatcher_thread
            if thread is not None and thread.is_alive():
                # It owns the queue now, and it routes to us.
                try:
                    return mailbox.get(timeout = timeout)
                except queue.Empty:
                    return None
            resp = self._read_resp(timeout = timeout)
            if resp is None:
                return None
            rid = resp.get("request_id")
            if rid and rid != request_id:
                with self._mailbox_lock:
                    other = self._mailboxes.get(rid) or self._direct_mailboxes.get(rid)
                    owner = self._request_cancel_events.get(rid)
                if other is not None:
                    # We beat the dispatcher to this response, so make its ownership move here
                    # too. The compare consumer opts out of marking, so nothing else promotes
                    # or retires that request: skipping it left this one recorded as the
                    # executor, ignoring its Stop and letting a late reset cancel it.
                    if owner is not None:
                        if resp.get("type", "") in ("gen_done", "gen_error"):
                            self._release_worker(owner)
                        else:
                            self._mark_worker_started(owner)
                    other.put(resp)
                    return None
            return resp

        def drain(timeout: float = 5.0) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                resp = read_one(timeout = min(0.5, deadline - time.monotonic()))
                if resp is None:
                    if not self._ensure_subprocess_alive():
                        return True
                    continue
                if resp.get("type", "") in (
                    "gen_done",
                    "gen_error",
                    "audio_done",
                    "audio_error",
                ):
                    return True
            logger.warning("Timed out waiting for terminal response after cancel")
            return False

        def release() -> None:
            with self._mailbox_lock:
                self._direct_mailboxes.pop(request_id, None)

        return read_one, drain, release

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
        continue_final_message: bool = False,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
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
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
        }
        if seed is not None:
            cmd["seed"] = seed
        if stop:
            cmd["stop"] = stop
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
        if continue_final_message:
            cmd["continue_final_message"] = True
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
        mark_started: bool = True,
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
            # The worker is answering THIS request, so it is the one executing: only now may its
            # cancel event speak for the shared worker one. The dispatched path opts out: its
            # dispatcher already did this in worker order, which a mailbox read can lag behind.
            if mark_started:
                self._mark_worker_started(cancel_event)
            # Subprocess-level error (no request_id); request-scoped failures
            # arrive as gen_error below.
            if rtype == "error" and not resp.get("request_id"):
                yield GenStreamError(f"Error: {resp.get('error', 'Unknown error')}")
                return

            if rtype == "token":
                # Cancel from route (e.g. SSE connection closed).
                if cancel_event is not None and cancel_event.is_set():
                    # Same rule as reset_generation_state: the shared worker event may only be set by
                    # the generation the worker is running. A dispatched request can still be draining
                    # stale mailbox tokens after the dispatcher retired it, and signalling from here
                    # would end the next one instead. Tearing this stream down is always safe, so the
                    # local drain happens either way.
                    if self._owns_worker(cancel_event):
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
            if (
                self._unload_pending
                or self._exclusive_tts_pending
                or getattr(self, "_exclusive_vram_probe_pending", False)
            ):
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
                        mbox = self._mailboxes.get(rid) or self._direct_mailboxes.get(rid)
                        owner = self._request_cancel_events.get(rid)
                    if mbox is not None:
                        # Worker order, not consumer order: retire a request the moment its last response
                        # is routed. Waiting for the consumer's finally left it owning the worker after
                        # the worker moved on, so a late Stop for it cancelled whichever request started next.
                        if owner is not None:
                            if rtype in ("gen_done", "gen_error"):
                                self._release_worker(owner)
                            else:
                                self._mark_worker_started(owner)
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
        continue_final_message: bool = False,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
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
        if self._exclusive_tts_pending:
            yield GenStreamError("Error: audio generation is in progress", public = True)
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
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
            use_adapter = use_adapter,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
            seed = seed,
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
            tts_reserved = self._exclusive_tts_pending
            probe_reserved = getattr(self, "_exclusive_vram_probe_pending", False)
            blocked = unloading or tts_reserved or probe_reserved
            if not blocked:
                self._mailboxes[request_id] = mailbox
                if cancel_event is not None:
                    self._request_cancel_events[request_id] = cancel_event
            # When bailing without a mailbox, note whether any OTHER compare request still
            # routes through the dispatcher; if none and this call started it, stop it below.
            orphaned_dispatcher = blocked and not dispatcher_preexisting and not self._mailboxes
        if blocked:
            # A racing unload can pass its _wait_dispatcher_idle() while the dispatcher was
            # stopped, then set _unload_pending. The one we just started would otherwise
            # linger with no mailboxes, race unload_model's _wait_response for the "unloaded"
            # reply off resp_queue, and drop it as unroutable -- hanging the unload 300s. Stop
            # it here so the unload stays the sole resp_queue reader. Outside _mailbox_lock:
            # _stop_dispatcher joins the dispatcher, which itself takes that lock.
            if orphaned_dispatcher:
                self._stop_dispatcher()
            detail = (
                "Error: audio generation is in progress"
                if tts_reserved
                else "Error: model switch is checking GPU memory"
                if probe_reserved
                else "Error: model is being unloaded"
            )
            yield GenStreamError(detail, public = True)
            return

        # Claim before sending, like the locked path: dispatched runs are concurrent by design,
        # so without this a Stop on one saw no owner and reset the worker, ending its siblings.
        # Claim and enqueue under one lock, or two dispatcher threads interleave and claim order
        # stops matching the subprocess's command order, which _owns_worker reads.
        try:
            with self._send_order_lock:
                self._claim_worker(cancel_event)
                self._send_cmd(cmd)
        except RuntimeError as exc:
            self._release_worker(cancel_event)
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)
                self._request_cancel_events.pop(request_id, None)
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
                mark_started = False,
            )
        finally:
            # Normally already retired by the dispatcher at gen_done; this covers streams that
            # end without one (cancel, disconnect, a dead subprocess).
            self._release_worker(cancel_event)
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)
                self._request_cancel_events.pop(request_id, None)

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

    def _wait_dispatcher_idle(self, cancel_event = None) -> bool:
        """Wait for all dispatched requests to complete, then stop dispatcher.

        Returns True if the dispatcher was stopped (all mailboxes drained, or no
        dispatcher was running), and False if it was left running because compare
        requests were still active after _DISPATCH_IDLE_TIMEOUT.

        Callers must reserve admission before using this as an exclusive handoff;
        TTS does so under _gen_lock, while older direct paths call it before the lock.
        """
        if self._dispatcher_thread is None or not self._dispatcher_thread.is_alive():
            return True

        # Wait for all mailboxes to be emptied (dispatched requests complete)
        deadline = time.monotonic() + _DISPATCH_IDLE_TIMEOUT
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return False
            with self._mailbox_lock:
                if not self._mailboxes:
                    break
            time.sleep(0.1)

        if cancel_event is not None and cancel_event.is_set():
            return False

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
        mlx_kv_bits: Optional[int] = None,
        chat_template_override: Optional[str] = None,
        load_cancel_event: Optional[threading.Event] = None,
        post_handoff_expected_free_gb: Optional[dict[int, float]] = None,
    ) -> bool:
        """Load a model for inference.

        Always spawns a fresh subprocess per load for a clean interpreter (no
        stale unsloth patches, torch.compile caches, or getsource failures).
        """
        from utils.transformers_version import needs_transformers_5

        # Same lazy-shim reason as _wait_response(); see the note there.
        from utils.hf_xet_fallback import DownloadStallError

        model_name = config.identifier
        self.loading_models.add(model_name)
        if load_cancel_event is not None and load_cancel_event.is_set():
            self.loading_models.discard(model_name)
            logger.info("Load cancelled before worker start: %s", model_name)
            return False

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
                "mlx_kv_bits": mlx_kv_bits,
                "chat_template_override": chat_template_override,
            }
            resolved_gpu_ids, gpu_selection = prepare_gpu_selection(
                gpu_ids,
                model_name = model_name,
                hf_token = hf_token,
                load_in_4bit = load_in_4bit,
            )
            sub_config["resolved_gpu_ids"] = resolved_gpu_ids
            sub_config["gpu_selection"] = gpu_selection
            # Parent-detected backend for the worker's apply_gpu_ids().
            sub_config["device_backend"] = get_device().value

            if load_cancel_event is not None and load_cancel_event.is_set():
                self.loading_models.discard(model_name)
                logger.info("Load cancelled before worker teardown: %s", model_name)
                return False

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
            had_worker_handle = self._proc is not None
            worker_shutdown_at = 0.0
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
                worker_shutdown_at = time.monotonic()
            elif self._proc is not None:
                self._shutdown_subprocess(timeout = 2)
                if self._proc is None:
                    worker_shutdown_at = time.monotonic()

            if had_worker_handle or post_handoff_expected_free_gb:
                expected_free_gb = post_handoff_expected_free_gb
                if (
                    expected_free_gb is None
                    and len(resolved_gpu_ids or ()) == 1
                    and isinstance(gpu_selection, dict)
                ):
                    required_gb = gpu_selection.get("required_gb")
                    if required_gb is not None:
                        expected_free_gb = {int(resolved_gpu_ids[0]): float(required_gb)}
                settled = self._wait_for_worker_vram_settle(
                    worker_shutdown_at or time.monotonic(),
                    expected_free_gb = expected_free_gb,
                )
                if expected_free_gb is not None and not settled:
                    raise RuntimeError(
                        "GPU memory from the previous inference worker was not released; "
                        "not starting the replacement model. Retry shortly."
                    )

            disable_xet = sub_config.get("disable_xet", False) or (
                os.environ.get("HF_HUB_DISABLE_XET") == "1"
            )

            for attempt in range(2):
                # Stop-loading (/unload -> cancel_load) aborts a load by discarding this
                # model's loading marker. cancel_load only kills a live child; if the cancel
                # lands before any child exists (GPU placement, or between retries) there is
                # nothing to kill, and without this check the loop would spawn a worker and
                # load the model after /unload reported it unloaded. Observe removal and stop.
                if model_name not in self.loading_models or (
                    load_cancel_event is not None and load_cancel_event.is_set()
                ):
                    self.loading_models.discard(model_name)
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
                if model_name not in self.loading_models or (
                    load_cancel_event is not None and load_cancel_event.is_set()
                ):
                    self.loading_models.discard(model_name)
                    logger.info(
                        "Load for '%s' was cancelled during spawn; tearing the worker down",
                        model_name,
                    )
                    self._shutdown_subprocess(timeout = 5)
                    self.active_model_name = None
                    self.models.clear()
                    return False

                try:
                    if load_cancel_event is None:
                        resp = self._wait_response("loaded")
                    else:
                        resp = self._wait_response("loaded", cancel_event = load_cancel_event)
                except _LoadCancelled:
                    logger.info(
                        "Load for '%s' was cancelled while waiting for 'loaded'",
                        model_name,
                    )
                    self.loading_models.discard(model_name)
                    self._shutdown_subprocess(timeout = 5)
                    self.active_model_name = None
                    self.models.clear()
                    return False
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
                    if model_name not in self.loading_models or (
                        load_cancel_event is not None and load_cancel_event.is_set()
                    ):
                        cancelled_by_event = (
                            load_cancel_event is not None and load_cancel_event.is_set()
                        )
                        self.loading_models.discard(model_name)
                        logger.info(
                            "Load for '%s' was cancelled while waiting for 'loaded'; "
                            "not publishing the cancelled model",
                            model_name,
                        )
                        if cancelled_by_event:
                            self._shutdown_subprocess(timeout = 5)
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
                    self.models[self.active_model_name] = _mirrored_model_entry(
                        model_info, model_name
                    )
                    self.models[self.active_model_name].update(
                        _mlx_runtime_mirror_fields(model_info)
                    )
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
            # Reap workers after any failed load, including inactivity timeouts
            # that leave installs and GPU memory alive (#9398).
            try:
                self._shutdown_subprocess(timeout = 5)
            except Exception as teardown_exc:
                logger.warning("Could not shut the failed load's worker down: %s", teardown_exc)
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

    # --- Dictation models -------------------------------------------------
    # These run in the STT sidecars (whisper-server, llama-server, and the
    # Transformers spawn child), not the chat worker. Their lifecycle goes
    # through here all the same, so one object knows everything that is
    # resident and Voice settings and Model Hub cannot report different things
    # about one model.

    def load_stt_model(
        self,
        model: Optional[str],
        engine: str,
        request_cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Make a dictation model resident on its sidecar."""
        from core.inference import stt_registry
        stt_registry.load(model, engine, request_cancel_event)

    def unload_stt_model(
        self,
        engines: Optional[Sequence[str]] = None,
        expected_model: Optional[str] = None,
    ) -> list:
        """Release dictation models (all engines by default); returns refusals.

        ``expected_model`` scopes the release to a sidecar still holding that model.
        """
        from core.inference import stt_registry
        return stt_registry.unload(engines, expected_model = expected_model)

    def resident_stt_model(self) -> dict:
        """What dictation holds, alongside active_model_name for chat."""
        from core.inference import stt_registry
        return stt_registry.resident()

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
                # empty_cache in the child cannot return the accelerator context, so an
                # idle worker keeps its high-water mark -- VRAM the GGUF backend cannot
                # see and gpu_arbiter never evicts (both are chat-owned). Nothing left
                # to serve, so drop it; load_model respawns a fresh worker regardless.
                if not self.models and not self.loading_models:
                    logger.info("No models left resident; shutting the inference subprocess down")
                    # The unload already succeeded: a failed teardown must not report it failed.
                    try:
                        self._shutdown_subprocess(timeout = 5)
                    except Exception as exc:
                        logger.warning("Could not shut the idle inference subprocess down: %s", exc)
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

    def count_chat_tokens(
        self,
        messages: list,
        system_prompt: str = "",
        *,
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        timeout: float = 30.0,
    ) -> tuple[int, Optional[str]]:
        """Prompt tokens the loaded model would receive, and whose tokenizer counted them.

        Reads through an addressed mailbox, as generations do: compare mode bypasses the
        generation lock and leaves a dispatcher owning the response queue, which would
        route this reply nowhere.
        """
        if not self._gen_lock.acquire(blocking = False):
            raise RuntimeError("Cannot count tokens while a generation is in progress")
        request_id = str(uuid.uuid4())
        read_one, _drain, release_mailbox = self._direct_reader(request_id)
        try:
            self._send_cmd(
                {
                    "type": "count_tokens",
                    "request_id": request_id,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "tools": tools,
                    "enable_thinking": enable_thinking,
                    "reasoning_effort": reasoning_effort,
                    "preserve_thinking": preserve_thinking,
                }
            )
            deadline = time.monotonic() + timeout
            resp = None
            while time.monotonic() < deadline:
                candidate = read_one(timeout = min(1.0, deadline - time.monotonic()))
                if candidate is None:
                    if not self._ensure_subprocess_alive():
                        raise RuntimeError(self._subprocess_crash_message("count"))
                    continue
                # A reply whose own mailbox is gone -- an earlier count that timed out
                # while the worker still held its command -- is handed back to whoever is
                # reading, and the type alone cannot tell it from this one's.
                if (
                    candidate.get("type") == "count_tokens_response"
                    and candidate.get("request_id") == request_id
                ):
                    resp = candidate
                    break
            if resp is None:
                raise RuntimeError("Timed out counting tokens")
        finally:
            release_mailbox()
            self._gen_lock.release()
        error = resp.get("error")
        if error:
            raise RuntimeError(error)
        return int(resp["input_tokens"]), resp.get("model")

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
        continue_final_message: bool = False,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """Generate response, streaming tokens from subprocess.

        ``tools`` / ``enable_thinking`` / ``reasoning_effort`` /
        ``preserve_thinking`` are forwarded so the template can render tool
        schemas and reasoning controls.

        ``stats_holder``: caller-owned dict; on gen_done its "stats" key gets
        the worker's usage, timings and terminal reason. Request-scoped to avoid
        cross-stream reads.

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
            continue_final_message = continue_final_message,
            stats_holder = stats_holder,
            presence_penalty = presence_penalty,
            seed = seed,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
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
        continue_final_message: bool = False,
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
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        reasoning_prefilled: bool = False,
        seed: Optional[int] = None,
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
            turn_stats: dict = {}
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
                # Self-limiting: after a tool call the conversation ends on a tool
                # result, so later turns render as ordinary new turns.
                continue_final_message = continue_final_message,
                # Reported per turn and summed below, since the whole loop answers
                # one request.
                stats_holder = turn_stats,
                presence_penalty = presence_penalty,
                seed = seed,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
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
                # A turn that never reported -- one a cancel interrupted -- folds in
                # as nothing, leaving the turns that did.
                if stats_holder is not None:
                    stats_holder["stats"] = _summed_tool_loop_stats(
                        stats_holder.get("stats"), turn_stats.get("stats")
                    )

        initial = list(messages)
        if system_prompt:
            initial = [{"role": "system", "content": system_prompt}] + initial

        # Same profile the renderer uses, so the controller never drops a tool over a
        # marker this model does not treat as structure. The controller is also given the
        # catalog safe under every template this turn could select, because the
        # native-template fallback renders with a different profile (#7066).
        from core.inference.chat_template_helpers import (
            mapped_chat_template,
            markup_for_tokenizer,
            renderable_tool_catalog,
        )

        _model_info = self.models.get(self.active_model_name) or {}
        # Resolved BEFORE the profile: the mapper installs its template during the render.
        _mapped_tpl = mapped_chat_template(_model_info, self.active_model_name)

        yield from run_safetensors_tool_loop(
            markup = markup_for_tokenizer(_model_info.get("tokenizer"), tools, _mapped_tpl),
            renderable_tools = renderable_tool_catalog(
                tools,
                _model_info.get("tokenizer"),
                _model_info,
                active_model_name = self.active_model_name,
                template = _mapped_tpl,
            ),
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
            continue_final_message = continue_final_message,
            # So a conversation search can be sized against what this model can hold.
            context_length = _model_info.get("context_length"),
            max_tokens = max_new_tokens,
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
        continue_final_message: bool = False,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
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
            if cancel_event is not None and cancel_event.is_set():
                # Stopped while queued on the lock. Sending anyway occupied the worker with a
                # run the user ended: the cancel is only seen on a token, so a long prefill
                # (or a generation that reaches gen_done without one) held up its siblings.
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
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                use_adapter = use_adapter,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                seed = seed,
            )

            # Claim the worker BEFORE sending, so a Stop on some OTHER chat -- still queued on the
            # lock above, having generated nothing -- cannot reset the generation this is starting.
            # Claiming after the send left the command running unclaimed. Released in the finally.
            # Own mailbox: a compare request can start the dispatcher while this is streaming,
            # and it would otherwise consume our responses and drop them.
            read_one, drain, release_mailbox = self._direct_reader(request_id)
            try:
                try:
                    with self._send_order_lock:
                        self._claim_worker(cancel_event)
                        self._send_cmd(cmd)
                except RuntimeError as exc:
                    yield GenStreamError(f"Error: {exc}")
                    return

                yield from self._consume_token_stream(
                    read_one,
                    lambda: drain(timeout = 5.0),
                    crash_context = "generation",
                    cancel_event = cancel_event,
                    stats_holder = stats_holder,
                )
            finally:
                self._release_worker(cancel_event)
                release_mailbox()

    def _claim_worker(self, cancel_event) -> None:
        """Record this request as one the worker will run.

        Admission only. The subprocess executes generations one at a time, so a
        dispatched request sitting behind another in the command queue is claimed
        but not executing, and must not be able to signal the shared cancel event
        (that would end whichever request IS executing). _mark_worker_started
        promotes it once the worker answers it.
        """
        with self._active_cancel_lock:
            self._active_cancel_events.append(cancel_event)

    def _mark_worker_started(self, cancel_event) -> None:
        """Promote a claimed request to executing, on its first worker response.

        Sole executor: the subprocess runs one generation at a time, so answering
        this one means it has left the previous one behind.
        """
        if cancel_event is None:
            return
        with self._active_cancel_lock:
            if self._executing_cancel_events[:1] != [cancel_event]:
                self._executing_cancel_events[:] = [cancel_event]

    def _release_worker(self, cancel_event) -> None:
        with self._active_cancel_lock:
            for bucket in (self._active_cancel_events, self._executing_cancel_events):
                try:
                    bucket.remove(cancel_event)
                except ValueError:
                    pass

    def _owns_worker(self, cancel_event) -> bool:
        """Whether a reset from this request may signal the shared cancel event.

        True when it is one of the EXECUTING generations, and when nothing is in
        flight at all: an error path that resets before anything started has no
        one else to interrupt, so it must not become a silent no-op. Claimed but
        queued does not count, or a Stop on a queued request would end the
        running one, including during the prefill before any response arrives.
        """
        with self._active_cancel_lock:
            if not self._active_cancel_events:
                # Nothing in flight at all, so there is no one to protect.
                return True
            if self._executing_cancel_events:
                return any(ev is cancel_event for ev in self._executing_cancel_events)
            # Claimed but nothing has answered yet (A is in prefill). The worker takes commands
            # in order, so the oldest claim is the executor; anyone else here is queued behind it.
            return self._active_cancel_events[0] is cancel_event

    def reset_generation_state(self, caller_cancel_event = None):
        """Cancel any ongoing generation and reset state.

        ``caller_cancel_event`` scopes the reset to one request. The worker has a
        single cancel event and generation is serialized on _gen_lock, so a chat
        that is still queued has no generation of its own to reset: calling this
        from its Stop handler would kill whichever chat currently holds the lock.
        Pass the request's own event and the reset is dropped unless that request
        is the one running. Omit it for genuinely global resets (unload, switch).
        """
        if caller_cancel_event is not None and not self._owns_worker(caller_cancel_event):
            return
        self._cancel_generation()
        if not self._ensure_subprocess_alive():
            return
        try:
            with self._send_order_lock:
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
        cancel_event = None,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        """Generate TTS audio. Returns (wav_bytes, sample_rate).

        Blocking — sends command and waits for the full audio response.
        """
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")
        if not self.active_model_name:
            raise RuntimeError("No active model")
        expected_model = self.active_model_name

        # Serialize under _gen_lock and reserve dispatcher admission before waiting for
        # compare work to drain. A bare idle wait is racy: a compare request can register
        # between the wait and this command, leaving TTS queued without safe ownership of
        # the worker's single shared cancel event.
        with self._gen_lock:
            with self._dispatcher_lifecycle_lock:
                self._exclusive_tts_pending = True
            try:
                dispatcher_idle = self._wait_dispatcher_idle(cancel_event = cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise AudioGenerationCancelledError("Audio generation cancelled")
                if not dispatcher_idle:
                    raise RuntimeError(
                        "Cannot start audio generation while compare requests are active"
                    )

                # Recheck after the dispatcher wait: unload can set its flag without
                # _gen_lock, and a switch may have completed while this call was queued.
                if self._unload_pending or self.active_model_name != expected_model:
                    raise AudioGenerationCancelledError("model is being unloaded")

                # Bound public API integers before either enqueuing work or
                # calculating the floating-point watchdog deadline.
                model_info = self.models.get(expected_model, {})
                audio_type = model_info.get("audio_type")
                max_token_ceiling = AUDIO_GENERATION_MAX_TOKENS
                if audio_type in ("moss_tts_local", "moss_tts_nano"):
                    try:
                        detected_context = int(model_info.get("context_length") or 0)
                    except (TypeError, ValueError):
                        detected_context = 0
                    max_token_ceiling = detected_context or MOSS_TTS_MAX_FRAMES
                elif audio_type == "minimax_music3":
                    max_token_ceiling = MINIMAX_MUSIC_MAX_FRAMES
                max_new_tokens = min(
                    max_token_ceiling,
                    max(1, int(max_new_tokens)),
                )
                generation_timeout = _audio_generation_timeout(
                    max_new_tokens,
                    max_tokens = max_token_ceiling,
                )
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
                if instructions is not None:
                    cmd["instructions"] = instructions
                if language is not None:
                    cmd["language"] = language
                if seed is not None:
                    cmd["seed"] = int(seed)

                # Same shared-queue hazard as _generate_inner: see _direct_reader.
                read_one, _drain, release_mailbox = self._direct_reader(request_id)
                try:
                    # Claim before enqueueing so request-scoped reset ownership follows the same
                    # discipline as text and audio-input generation.
                    with self._send_order_lock:
                        self._claim_worker(cancel_event)
                        self._send_cmd(cmd)

                    deadline = time.monotonic() + generation_timeout
                    cancel_signalled = False
                    cancel_deadline = None
                    worker_started = False
                    while time.monotonic() < deadline:
                        if (
                            cancel_event is not None
                            and cancel_event.is_set()
                            and cancel_deadline is None
                        ):
                            cancel_deadline = time.monotonic() + _AUDIO_CANCEL_TEARDOWN_TIMEOUT
                            deadline = min(deadline, cancel_deadline)
                        if (
                            worker_started
                            and cancel_event is not None
                            and cancel_event.is_set()
                            and not cancel_signalled
                            and self._owns_worker(cancel_event)
                        ):
                            # audio_started is emitted after the worker clears stale state,
                            # so this signal cannot be erased or hit an earlier request.
                            self._cancel_generation()
                            cancel_signalled = True
                            # The cancel is delivered now, so hold the worker to the drain
                            # window from here rather than from when the caller asked.
                            cancel_deadline = time.monotonic() + _AUDIO_CANCEL_DRAIN_TIMEOUT
                            deadline = min(deadline, cancel_deadline)
                        remaining = max(0.1, deadline - time.monotonic())
                        resp = read_one(timeout = min(remaining, 1.0))

                        if resp is None:
                            if not self._ensure_subprocess_alive():
                                raise RuntimeError(
                                    self._subprocess_crash_message("audio generation")
                                )
                            continue

                        rtype = resp.get("type", "")
                        if rtype == "audio_started":
                            self._mark_worker_started(cancel_event)
                            worker_started = True
                            continue

                        if rtype == "audio_done":
                            if cancel_event is not None and cancel_event.is_set():
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            wav_bytes = base64.b64decode(resp["wav_base64"])
                            sample_rate = resp["sample_rate"]
                            return wav_bytes, sample_rate

                        if rtype == "audio_error":
                            if resp.get("cancelled") or (
                                cancel_event is not None and cancel_event.is_set()
                            ):
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            # Tagged code = no path for this task, not a failure.
                            if resp.get("code") == AUDIO_UNSUPPORTED_CODE:
                                raise AudioBackendUnsupportedError(
                                    resp.get("error", "This backend cannot generate audio."),
                                    hint = resp.get("hint"),
                                )
                            raise RuntimeError(resp.get("error", "Audio generation failed"))

                        if rtype == "error":
                            if cancel_event is not None and cancel_event.is_set():
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            raise RuntimeError(resp.get("error", "Unknown error"))

                        if rtype == "status":
                            continue

                    # A caller cancellation already spent the drain window polling this
                    # request's mailbox. Tear down an unresponsive worker now instead of
                    # waiting out the much longer generation watchdog or draining twice.
                    if cancel_deadline is not None:
                        if self._shutdown_subprocess(timeout = _AUDIO_CANCEL_DRAIN_TIMEOUT):
                            self.active_model_name = None
                            self.models.clear()
                        raise AudioGenerationCancelledError("Audio generation cancelled")

                    # Do not release worker ownership or dispatcher exclusivity over a
                    # command that may still be generating. Cancel, consume its terminal
                    # response, and tear down a worker that does not acknowledge promptly.
                    self._cancel_generation()
                    if not _drain(timeout = _AUDIO_CANCEL_DRAIN_TIMEOUT):
                        if self._shutdown_subprocess(timeout = _AUDIO_CANCEL_DRAIN_TIMEOUT):
                            self.active_model_name = None
                            self.models.clear()
                    raise RuntimeError(
                        f"Timeout waiting for audio generation ({generation_timeout:g}s)"
                    )
                finally:
                    self._release_worker(cancel_event)
                    release_mailbox()
            finally:
                with self._dispatcher_lifecycle_lock:
                    self._exclusive_tts_pending = False

    def generate_whisper_response(
        self,
        audio_array,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Whisper ASR — sends audio to subprocess, yields text."""
        yield from self._generate_audio_input_inner(
            audio_array = audio_array,
            audio_type = "whisper",
            messages = [],
            system_prompt = "",
            cancel_event = cancel_event,
            stats_holder = stats_holder,
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
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        stop = None,
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
            use_adapter = use_adapter,
            cancel_event = cancel_event,
            stats_holder = stats_holder,
            stop = stop,
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
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        stop = None,
    ) -> Generator[str, None, None]:
        """Shared inner logic for audio input generation (Whisper + ASR).

        ``stats_holder``: as in generate_chat_response — caller-owned, filled on
        gen_done with the worker's usage / budget report.
        """
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
            if cancel_event is not None and cancel_event.is_set():
                # Stopped while queued on the lock, same as _generate_inner.
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
            # As in the text path: key stays absent unless the caller selected one.
            if use_adapter is not None:
                cmd["use_adapter"] = use_adapter
            if stop:
                cmd["stop"] = stop

            # Same shared-queue hazard as _generate_inner: see _direct_reader.
            read_one, drain, release_mailbox = self._direct_reader(request_id)
            try:
                try:
                    # Claim under the send lock, like _generate_inner: unclaimed, a compare request queued
                    # behind this looked like the oldest owner, so stopping it killed this one.
                    with self._send_order_lock:
                        self._claim_worker(cancel_event)
                        self._send_cmd(cmd)
                except RuntimeError as exc:
                    yield GenStreamError(f"Error: {exc}")
                    return

                yield from self._consume_token_stream(
                    read_one,
                    lambda: drain(timeout = 5.0),
                    crash_context = "audio input generation",
                    cancel_event = cancel_event,
                    stats_holder = stats_holder,
                )
            finally:
                self._release_worker(cancel_event)
                release_mailbox()

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
# Guards the lazy construction below. The first build runs hardware detection, seconds cold,
# and first-paint routes call this getter from executor threads. Unlocked, several would see
# None and each build an orchestrator, orphaning all but the last plus any load on them.
_inference_backend_lock = threading.Lock()


def peek_inference_backend() -> Optional["InferenceOrchestrator"]:
    """The orchestrator if one exists, else None. Never constructs one.

    For callers that only describe what is already loaded: constructing reaches
    get_default_models() -> get_device(), which blocks on the torch import during the warm.
    """
    return _inference_backend


def get_inference_backend() -> InferenceOrchestrator:
    """Global inference backend instance (orchestrator)."""
    global _inference_backend
    # Double-checked: the cheap read keeps the hot path lock-free, the recheck picks a builder.
    if _inference_backend is None:
        with _inference_backend_lock:
            if _inference_backend is None:
                _inference_backend = InferenceOrchestrator()
    return _inference_backend
