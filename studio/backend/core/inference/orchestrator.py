# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference orchestrator, subprocess-based. Same API as InferenceBackend, but delegates all ML work
to a persistent subprocess spawned on first model load and reused for later requests. When
switching between models needing different transformers versions (e.g. GLM-4.7-Flash needs 5.x,
Qwen needs 4.57.x), the old subprocess is killed and a new one spawned with the correct version.
Pattern follows core/training/training.py."""

import atexit
import base64
import contextlib
import os
import signal
from loggers import get_logger
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
import multiprocessing as mp
import queue
import re
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Generator, Mapping, Optional, Sequence, Tuple, Union
from core.inference.audio_device import audio_device_forces_cpu
from core.inference.context_refusal import ContextBudgetExceeded
from core.inference.native_audio import NATIVE_AUDIO_TYPES, is_native_audio_model
from core.inference.audio_errors import (
    AUDIO_UNSUPPORTED_CODE,
    AudioBackendUnsupportedError,
    AudioGenerationCancelledError,
)
from core.inference.worker import PendingTeardowns, StopLedger
from utils.hardware import get_device, prepare_gpu_selection
from utils.utils import hf_env_offline, is_metal_queue_dead

# Re-exported from the shared helper so GGUF, training and inference share one type. Via PEP 562, not a module-level
# import: resolving the name imports unsloth_zoo, hence torch, and routes/inference.py imports this module at startup
# only for GenStream*.
DownloadStallError: type


def __getattr__(name: str):
    if name == "DownloadStallError":
        from utils.hf_xet_fallback import DownloadStallError as _exc
        return _exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = get_logger(__name__)

# Delimited, not a whitelist: a whitelist stopped at the apostrophe in `/home/o'connor/`.
_PATH_COMPONENT = r"[^\s\\/](?:(?:(?![A-Za-z]:[\\/])[^\\/\n\",;])*[^\s\\/])?"
# Second alternative: the root-level case (`/model.gguf`, `\\server\share`) has no trailing separator.
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:/])(?:\\\\[^\\/\s]+[\\/]|[A-Za-z]:[\\/]|/)"
    r"(?:(?:" + _PATH_COMPONENT + r"[\\/])+[^\s\\/]*|[^\s\\/\",;]+[\\/]?)"
)


def _shorten_path(match: "re.Match[str]") -> str:
    text = match.group(0)
    tail = text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return f".../{tail}" if tail else "..."


# A line written by a logger; `decode_bicodec` logs 500 characters of generated text this way.
_LOG_RECORD_RE = re.compile(
    r"""^\s*(?:
        \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}   # a leading ISO timestamp
      | \[?(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b
      | [\w]+(?:\.[\w]+)+\s*:                     # a dotted logger name before the colon
    )""",
    re.VERBOSE,
)

# Only the writer can tell a logged traceback from a dying process's; the bytes are identical. See `utils.worker_stderr`.
_LOG_CONTINUATION_PREFIX = "    | "
_LOG_RECORD_START_MARK = "\x1f"


def _looks_like_a_log_record(line: str) -> bool:
    """The mark is the reliable half; the patterns are the fallback for a producer that bypassed that handler."""
    if line.startswith(_LOG_CONTINUATION_PREFIX) or line.startswith(_LOG_RECORD_START_MARK):
        return True
    return bool(_LOG_RECORD_RE.match(line))


# Errs towards dropping: a missed diagnostic costs a detail, a kept content line leaves the host.
_DIAGNOSTIC_START_RE = re.compile(
    r"""^(?:
        Traceback\ \(most\ recent\ call\ last\):
      | terminate\ called
      | what\(\):
      | Fatal\ Python\ error:
      | (?:Current\ )?[Tt]hread\ 0x
      | Stack\ \(most\ recent\ call\ first\):
      | Segmentation\ fault | Bus\ error | Illegal\ instruction
      | Floating\ point\ exception | Aborted | Killed | Trace/breakpoint\ trap
      | \*\*\*                                     # *** stack smashing detected ***
      | double\ free | free\(\) | malloc\(\) | munmap_chunk | corrupted\ (?:size|double-linked)
      | std::(?:bad_alloc|terminate) | libc\+\+abi
      | GGML_ASSERT | CUDA\ error | HIP\ error | cudaError
      | [A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Abort|Fault|Signal)\s*:
    )""",
    re.VERBOSE,
)


_TRACEBACK_HEADER = "Traceback (most recent call last):"


def _starts_a_new_diagnostic(line: str) -> bool:
    return bool(_DIAGNOSTIC_START_RE.match(line))


def _diagnostic_lines_only(lines: "list[str]") -> "list[str]":
    """Line by line is not a filter, because a log record is not always one line."""
    kept: "list[str]" = []
    inside_a_diagnostic = False
    for line in lines:
        if line.startswith(_LOG_CONTINUATION_PREFIX):
            # Another thread logging mid-abort does not end the abort.
            continue
        if _looks_like_a_log_record(line):
            inside_a_diagnostic = False
            continue
        if _starts_a_new_diagnostic(line):
            inside_a_diagnostic = True
            kept.append(line)
            continue
        if inside_a_diagnostic and (not line.strip() or line[:1] in (" ", "\t")):
            kept.append(line)
            continue
        inside_a_diagnostic = False
    return kept


def _crash_lines(lines: "list[str]") -> "list[str]":
    """The last traceback, unless a diagnostic follows it: `logger.exception` for a RECOVERED failure leaves one too."""
    starts = [
        index for index, line in enumerate(lines) if line.lstrip().startswith(_TRACEBACK_HEADER)
    ]
    if not starts:
        return lines
    header = starts[-1]
    end = len(lines)
    for index in range(header + 1, len(lines)):
        line = lines[index]
        if line.strip() and line[:1] not in (" ", "\t"):
            end = index + 1
            break
    after = lines[end:]
    return after if after else lines[header:]


def _redact_worker_output(text: str) -> str:
    """Three redactors because none subsumes another: native path leases, token shapes, filesystem layout."""
    if not text:
        return ""
    redacted = text
    try:
        from utils.native_path_leases import redact_native_paths
        redacted = redact_native_paths(redacted)
    except Exception:  # noqa: BLE001 -- a redactor that cannot run must not lose the others
        pass
    try:
        from hub.utils.download_registry import scrub_secrets
        redacted = scrub_secrets(redacted)
    except Exception:  # noqa: BLE001
        pass
    try:
        # `redact_log_text` is idempotent, so it runs after `scrub_secrets`.
        from utils.log_redaction import redact_log_text
        redacted = redact_log_text(redacted)
    except Exception:  # noqa: BLE001
        pass
    return _ABSOLUTE_PATH_RE.sub(_shorten_path, redacted)


class _WorkerMailbox(queue.Queue):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker


class _LoadCancelled(Exception):
    """Internal control flow for a caller-cancelled model load."""


_CTX = mp.get_context("spawn")


_DISPATCH_READ_TIMEOUT = 30.0

_STOP_NOTICE_INTERVAL = 0.1
_DISPATCH_POLL_INTERVAL = 0.5
_DISPATCH_STOP_TIMEOUT = 5.0
_DISPATCH_IDLE_TIMEOUT = 30.0
_DISPATCH_DRAIN_TIMEOUT = 5.0
_CANCELLED_ROWS_GRACE = 5.0

# Only bounds the Transformers subprocess path; llama.cpp TTS never reaches here. 120s was tuned against GGUF speeds
# and killed real work: a safetensors LoRA on a mid-range GPU needs minutes for the same clip a GGUF returns in
# seconds. A dead worker is already caught every second by _ensure_subprocess_alive, so this only has to bound one
# that is alive and wedged, and a generous value costs nothing.
_AUDIO_GENERATION_TIMEOUT = 900.0
_AUDIO_GENERATION_BASE_TOKENS = 2048
AUDIO_GENERATION_MAX_TOKENS = 8192
MOSS_TTS_MAX_FRAMES = 32768
MINIMAX_MUSIC_MAX_FRAMES = 9000
_AUDIO_CANCEL_DRAIN_TIMEOUT = 5.0
# Before audio_started there is nobody to receive the cancel, and a prefill pass (a 3B TTS model on CPU, or OuteTTS's
# per-token Python repetition penalty) routinely outlasts the drain window. Tearing down on that budget unloads the
# model the user just loaded.
_AUDIO_CANCEL_TEARDOWN_TIMEOUT = 30.0

# Max wait for a cancelled generation to release _gen_lock before unload_model tears the subprocess down. Only bounds
# a wedged worker.
_UNLOAD_GEN_LOCK_TIMEOUT = 15.0


def _audio_generation_timeout(
    max_new_tokens: int,
    base: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> float:
    """Scale a floor by the requested token count. ``base`` differs per backend by an order of
    magnitude: the Transformers subprocess needs minutes for a clip llama.cpp returns in seconds,
    so llama_cpp.py passes its own; sharing one base silently gave every GGUF read the
    Transformers budget, which holds other_inference_request_count() up and blocks idle
    auto-unload for that long. Resolved at call time, not bound as a default: a default is
    evaluated once at import, so reassigning the module constant afterwards had no effect."""
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
    "mlx_kv_quant",
    "mlx_kv_quant_requested",
    "mlx_kv_quant_eligibility",
    "mlx_kv_quant_reason",
    "mlx_kv_quant_note",
    "chat_template_override_requested",
    "chat_template_override_reason",
)


def _mlx_runtime_mirror_fields(model_info: dict) -> dict:
    """MLX runtime state the parent mirrors, omitting what was not reported. Only the MLX backend
    sends these. Creating the keys for every backend would make the reload comparison see a None
    the backend never stored, and reload on every identical request."""
    return {key: model_info[key] for key in _MLX_RUNTIME_MIRROR_FIELDS if key in model_info}


class GenStreamError(str):
    """A stream chunk carrying a real backend/generation error, not model text. Subclasses str so
    existing display/logging consumers are unaffected, while callers can distinguish a real error
    from model output whose visible text starts with "Error:" by checking isinstance(chunk,
    GenStreamError)."""

    __slots__ = ("public", "openai_param")

    def __new__(
        cls,
        value,
        *,
        public: bool = False,
        openai_param: Optional[str] = None,
    ):
        obj = str.__new__(cls, value)
        obj.public = bool(public)
        # Set for a refusal about one request field, so the caller answers 400 not 500.
        obj.openai_param = openai_param
        return obj


class GenStreamErrorRaised(RuntimeError):
    """Internal exception form of ``GenStreamError`` for generator boundaries."""

    __slots__ = ("public", "openai_param")

    def __init__(
        self,
        value,
        *,
        public: bool = False,
        openai_param: Optional[str] = None,
    ):
        super().__init__(value)
        self.public = bool(public)
        self.openai_param = openai_param

    @classmethod
    def from_chunk(cls, chunk: "GenStreamError") -> "GenStreamErrorRaised":
        """Keeps public/openai_param so a refusal is not answered 500."""
        return cls(str(chunk), public = chunk.public, openai_param = chunk.openai_param)


def _summed_tool_loop_stats(total, turn):
    """Fold one tool-loop turn's report into the loop's running total. Every turn spends its tokens
    on the same request, so the reply reports their sum, as the llama.cpp tool loop does;
    reporting only the last turn hides the tokens that produced the tool call. The prompt count
    is the last turn's to report one, which already contains the tool results the earlier turns
    produced."""
    if not isinstance(turn, dict):
        return total
    if not isinstance(total, dict):
        return turn
    prior_usage = total.get("usage") or {}
    usage = dict(turn.get("usage") or {})
    completion = (usage.get("completion_tokens") or 0) + (prior_usage.get("completion_tokens") or 0)
    usage["completion_tokens"] = completion
    # The prompt is the loop's, not one turn's, so a turn that ended before reporting keeps the last count that
    # arrived. Its details describe that same count and move with it, or cached tokens could outnumber prompt tokens.
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = prior_usage.get("prompt_tokens") or 0
        usage.pop("prompt_tokens_details", None)
        if prior_usage.get("prompt_tokens_details") is not None:
            usage["prompt_tokens_details"] = prior_usage["prompt_tokens_details"]
    usage["total_tokens"] = usage["prompt_tokens"] + completion
    # Details describe the completion, so they sum with it rather than describing one turn against every turn's tokens
    details = dict(prior_usage.get("completion_tokens_details") or {})
    for field, value in (usage.get("completion_tokens_details") or {}).items():
        details[field] = (details.get(field) or 0) + (value or 0)
    if details:
        usage["completion_tokens_details"] = details
    summed = dict(turn)
    summed["usage"] = usage
    timings = dict(turn.get("timings") or {})
    prior = total.get("timings") or {}
    # Seeded from the turn but folded unconditionally, as the llama.cpp loop does: a turn reporting no timings must
    # not take the loop's totals with it.
    if timings or prior:
        for field in ("predicted_ms", "predicted_n"):
            timings[field] = (timings.get(field) or 0) + (prior.get(field) or 0)
        # Rates describe the totals above, not the turn they arrived with: leaving the last turn's would report a
        # speed the summed counts contradict.
        predicted_ms = timings.get("predicted_ms") or 0
        predicted_n = timings.get("predicted_n") or 0
        timings["predicted_per_token_ms"] = (predicted_ms / predicted_n) if predicted_n else 0.0
        timings["predicted_per_second"] = (
            (predicted_n / (predicted_ms / 1000.0)) if predicted_ms else 0.0
        )
        summed["timings"] = timings
    return summed


def _encoded_images(images, to_base64) -> list:
    """Replayed MCP pictures arrive already PNG-encoded; a caller's decoded list does not."""
    return [one if isinstance(one, str) else to_base64(one) for one in images or ()]


def _mirrored_model_entry(model_info: dict, model_name: str) -> dict:
    """The parent's view of a model the worker holds. Measured or classified in the subprocess and
    unrecoverable once the model lives there, so a field the worker sends and this does not copy
    is one the API can never report."""
    return {
        "is_vision": model_info.get("is_vision", False),
        "is_lora": model_info.get("is_lora", False),
        "is_mlx": model_info.get("is_mlx", False),
        "display_name": model_info.get("display_name", model_name),
        "is_audio": model_info.get("is_audio", False),
        "audio_type": model_info.get("audio_type"),
        "has_audio_input": model_info.get("has_audio_input", False),
        "has_video_input": model_info.get("has_video_input", False),
        "context_length": model_info.get("context_length"),
        "native_context_length": model_info.get("native_context_length"),
        "max_context_length": model_info.get("max_context_length"),
        "requested_context_length": model_info.get("requested_context_length"),
        "context_length_enforced": model_info.get("context_length_enforced"),
        "context_length_fitted": model_info.get("context_length_fitted"),
        "context_unbounded_when_batched": model_info.get("context_unbounded_when_batched"),
        "mlx_context_budget": model_info.get("mlx_context_budget"),
    }


class InferenceOrchestrator:
    """Inference backend orchestrator, subprocess-based. Same API surface as InferenceBackend (so
    routes/inference.py needs minimal changes); all heavy ML work happens in a persistent
    subprocess."""

    def __init__(self):
        self._proc: Optional[mp.Process] = None
        # Retired when the next worker is spawned; read long after _proc has been cleared.
        self._stderr_capture: Any = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None
        self._subprocess_shutdown_lock = threading.Lock()
        self._cancel_event: Any = None  # mp.Event - set to cancel generation
        # Set for the whole unload; the worker never clears it (unlike _cancel_event), so a generate queued behind the
        # cancelled one is skipped, not run.
        self._drain_event: Any = None
        self._stop_ledger: Any = None
        self._pending_teardowns: Any = None
        self._gen_lock = threading.Lock()  # Serializes generation
        # Cancel event of the request holding _gen_lock: lets a Stop tell whether it owns the running generation or is
        # queued behind it (the worker's event is shared).
        self._active_cancel_events: list = []
        self._executing_cancel_events: list = []
        self._active_cancel_lock = threading.Lock()
        # Held across claim + _send_cmd so claim order matches the subprocess dequeue order, which _owns_worker relies
        self._send_order_lock = threading.RLock()
        # Set during a switch so a generation winning the _gen_lock handoff bails instead of starting on the outgoing
        # model
        self._unload_pending = False
        self._worker_reserved_for: Optional[str] = None

        # Dispatcher state for compare mode (adapter-controlled requests): bypass _gen_lock, send commands directly,
        # read from per-request mailboxes routed by a dispatcher thread on request_id.
        self._mailboxes: dict[str, queue.Queue] = {}
        # request_id -> cancel event, so the dispatcher can move worker ownership as it routes. Consumers read their
        # mailbox whenever they get to it, so only the dispatcher sees responses in the order the worker produced
        # them.
        self._request_cancel_events: dict[str, object] = {}
        # Mailboxes for the _gen_lock generations. Kept apart from _mailboxes because that map means "compare requests
        # are in flight" to the unload and distributed paths.
        self._direct_mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()
        # Serializes dispatcher start/stop. _generate_dispatched (compare mode) bypasses _gen_lock, so two concurrent
        # compare requests can both reach _start_dispatcher; without this lock both could observe no live dispatcher
        # and each spawn one, orphaning the extra thread (self._dispatcher_thread tracks only the last). The orphan
        # later steals the "unloaded" reply off resp_queue and hangs unload_model.
        self._dispatcher_lifecycle_lock = threading.Lock()
        self._worker_released = threading.Condition(self._dispatcher_lifecycle_lock)

        # Local state mirrors (updated from subprocess responses)
        self.active_model_name: Optional[str] = None
        self.models: dict = {}
        self.loading_models: set = set()
        from core.inference.defaults import get_default_models

        # The list depends on detection (chat-only hosts get the GGUF set) and the MLX self-heal re-detects, so
        # unchecked a repaired Mac serves the chat-only list forever. Stamp read BEFORE the list, or a re-detect tags
        # the old list as new.
        import utils.hardware.hardware as _hw_mod

        self._static_models_generation = _hw_mod.DETECTION_GENERATION
        self._static_models = get_default_models()
        # Own lock for the stamp/value pair; the construction lock is held across a build that waits on hardware
        # detection
        self._static_models_lock = threading.Lock()
        self._top_gguf_cache: Optional[list[str]] = None
        self._top_hub_cache: Optional[list[str]] = None
        self._top_models_ready = threading.Event()

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

        # Deliberately NOT started here: construction now runs on the startup warm thread, so fetching from __init__
        # would call huggingface.co on every boot. First reader starts it.
        self._top_models_started = False

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
            # Commit only while still the newest: a slow reader storing its older list under a newer stamp would look
            # current for the life of the process
            if generation != _hw_mod.DETECTION_GENERATION:
                return
            if generation <= self._static_models_generation:
                return
            self._static_models = models
            self._static_models_generation = generation
        logger.info("hardware was re-detected; curated default models refreshed")

    def _start_top_models_fetch(self) -> None:
        """Kick the remote ranking fetch once, on first read of the model list. Guarded by the
        construction lock, so two concurrent first-readers cannot each put up a thread. Skipped
        when the host asked for no outbound calls: the fetch is a raw httpx.get, so
        HF_HUB_OFFLINE does not reach it on its own. Via hf_env_offline(), not a literal "1"
        test, since HF_HUB_OFFLINE=true/on and TRANSFORMERS_OFFLINE count too."""
        if self._top_models_started:
            return
        # Checked before the latch: claiming it while offline would retire the fetch for the process, so an offline
        # boot or a temporary force_hf_offline() could never recover.
        if hf_env_offline():
            logger.info("offline mode requested; skipping the remote top-models ranking")
            return
        with _inference_backend_lock:
            if self._top_models_started:
                return
            self._top_models_started = True
        threading.Thread(target = self._fetch_top_models, daemon = True, name = "top-models").start()

    def set_parallel_slots(self, n_parallel) -> int:
        from core.inference.llama_server_args import clamp_parallel_slots

        slots = clamp_parallel_slots(n_parallel)
        entry = self.models.get(self.active_model_name or "")
        if entry is not None:
            entry["parallel_slots"] = slots
        return slots

    @property
    def effective_parallel_slots(self) -> int:
        from core.inference.llama_server_args import PARALLEL_DEFAULT

        entry = self.models.get(self.active_model_name or "") or {}
        slots = entry.get("parallel_slots")
        return slots if isinstance(slots, int) and slots > 0 else PARALLEL_DEFAULT

    @property
    def default_models(self) -> list[str]:
        self._refresh_static_models_if_stale()
        self._start_top_models_fetch()
        top_gguf = self._top_gguf_cache or []
        top_hub = self._top_hub_cache or []
        # Use detected hardware here: discovery runs on the event loop.
        from core.inference.defaults import suggestions_for_host
        import utils.hardware.hardware as _hw_mod

        # A chat-only Mac never reaches the MLX loader, so its ranking is left as fetched.
        device = None if _hw_mod.CHAT_ONLY else _hw_mod.DEVICE
        fetched = suggestions_for_host(top_gguf + top_hub, device)
        # Never wait for the remote Hugging Face ranking during startup. Chat's
        # first /api/models/list needs curated defaults immediately; the
        # background fetch backfills extra choices on later calls.
        result: list[str] = []
        seen: set[str] = set()
        for m in self._static_models + fetched:
            if m not in seen:
                result.append(m)
                seen.add(m)
        return result

    def _fetch_top_models(self) -> None:
        """Fetch top GGUF and non-GGUF repos from unsloth by downloads."""
        try:
            import httpx
            from utils.hf_endpoint import get_hf_endpoint

            resp = httpx.get(
                f"{get_hf_endpoint()}/api/models",
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
                # Counts at info, ids at debug: two lists of 40 repo names cost ~1.5 KB of every boot to say the
                # catalog fetch worked
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

    def _spawn_subprocess(
        self,
        config: dict,
        cache_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        from utils.transformers_version import (
            SidecarSwapInProgress,
            sidecar_swap_kind,
        )

        if sidecar_swap_kind() == "repair":
            raise SidecarSwapInProgress(
                "A transformers repair is replacing the latest sidecar; retry when it completes."
            )
        # Last gate before Popen. A preview or auto-switch load is not a
        # _ScopedLoadAttempt, so the route's shutdown sweep cannot cancel it; it can
        # clear the load's own checks and only then reach here, after the shutdown
        # already stopped this subprocess. Checked at the spawn itself so the answer
        # cannot go stale between the check and the child.
        from utils.process_lifetime import is_process_shutting_down

        if is_process_shutting_down():
            raise RuntimeError("Unsloth is shutting down; not starting an inference subprocess")
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths

        cache_env = (
            dict(cache_environment)
            if cache_environment is not None
            else get_hf_cache_paths().child_env({})
        )

        # Retired here, not at shutdown: a crash message outlives _shutdown_subprocess clearing _proc.
        self._retire_stderr_capture()
        self._worker_stopped_deliberately = False
        try:
            from utils.worker_stderr import WorkerStderrCapture
            self._stderr_capture = WorkerStderrCapture(prefix = "unsloth-inference-worker-")
        except Exception as exc:
            # No sink is the old behaviour; never a failed spawn.
            logger.debug("Could not open a worker stderr mirror: %s", exc)
            self._stderr_capture = None

        with (
            child_environment_for_spawn(cache_env),
            native_path_secret_removed_for_child_start(),
        ):
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            self._cancel_event = _CTX.Event()
            self._drain_event = _CTX.Event()
            self._stop_ledger = StopLedger(_CTX)
            self._pending_teardowns = PendingTeardowns(_CTX)

            # Built into a local FIRST, and started through that local. A shutdown can
            # observe a not-yet-alive child, clear self._proc and finish its sweep while
            # start() is still returning, so the attribute is not a handle this code can
            # rely on from here on: snapshotting it after start() would capture the None
            # and lose the only reference to a live child, which is the orphan this
            # change exists to prevent.
            _child_kwargs: dict = {
                "cmd_queue": self._cmd_queue,
                "resp_queue": self._resp_queue,
                "cancel_event": self._cancel_event,
                "drain_event": self._drain_event,
                "stop_ledger": self._stop_ledger,
                "pending_teardowns": self._pending_teardowns,
                "config": config,
            }
            if self._stderr_capture is not None:
                from utils.native_path_leases import STDERR_MIRROR_KWARG

                # Consumed by run_without_native_path_secret; it never reaches the entrypoint.
                _child_kwargs[STDERR_MIRROR_KWARG] = self._stderr_capture.path
            _spawned_proc = _CTX.Process(
                target = run_without_native_path_secret,
                args = ("core.inference.worker", "run_inference_process", cache_env),
                kwargs = _child_kwargs,
                daemon = True,
            )
            self._proc = _spawned_proc
            _spawned_proc.start()
        from utils.process_lifetime import adopt_pid

        adopt_pid(_spawned_proc.pid)  # bind to parent lifetime (Windows job / sweep)

        # The gate above is 30-odd lines and a process start away from here, so a
        # shutdown can begin in between, see no live _proc, and finish its sweep
        # while this child is still being born. Recheck now it exists and reap it,
        # the same shape as the cancel_load recheck below the caller's spawn.
        # A lock across the spawn would close it too, but _shutdown_subprocess holds
        # that lock for its whole teardown, so quitting would then queue behind a
        # spawn it is about to undo. adopt_pid runs first either way: a child that
        # dies here must still be in the sweep record.
        if is_process_shutting_down() or self._proc is not _spawned_proc:
            logger.info("Shutdown began during spawn; tearing the new inference worker down")
            self._shutdown_subprocess(timeout = 5)
            # If shutdown already dropped the mirror, that call cannot see this child.
            # The local handle is the only one left, so reap it here -- and escalate the
            # way _shutdown_subprocess_locked does, rather than abandoning a worker that
            # ignores SIGTERM. The step-7 snapshot has already been taken by this point,
            # so a child left alive here survives until a later startup reaps its record.
            try:
                if _spawned_proc.is_alive():
                    _spawned_proc.terminate()
                    _spawned_proc.join(5)
                if _spawned_proc.is_alive():
                    from utils.process_lifetime import terminate_pid
                    terminate_pid(_spawned_proc.pid, timeout = 5)
                    _spawned_proc.join(5)
                if _spawned_proc.is_alive():
                    _spawned_proc.kill()
                    _spawned_proc.join(5)
                if _spawned_proc.is_alive():
                    logger.warning(
                        "Raced inference worker (pid=%s) survived terminate and kill; "
                        "leaving it in the lifetime record for the startup sweep",
                        _spawned_proc.pid,
                    )
            except Exception as exc:
                logger.debug("Could not reap the raced inference worker: %s", exc)
            raise RuntimeError("Unsloth is shutting down; not starting an inference subprocess")
        logger.info("Inference subprocess started (pid=%s)", _spawned_proc.pid)

    def _cancel_generation(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _cancel_every_generation(self) -> None:
        self._teardown_going_out()
        self._cancel_generation()
        try:
            self._send_cmd({"type": "cancel"})
        except Exception:
            self._teardown_not_sent()
            logger.debug("Could not tell the worker to let go of its batch", exc_info = True)

    def is_worker_alive(self) -> bool:
        """True while the inference subprocess is running, even with no model active (a failed load
        can leave a live worker holding sidecar modules)."""
        proc = self._proc
        return proc is not None and proc.is_alive()

    def post_handoff_gpu_availability_gb(
        self,
    ) -> Optional[tuple[dict[int, float], dict[int, float], dict[int, float]]]:
        """Atomically snapshot live free, total and disposable-worker VRAM. The worker is queried
        only while idle, which avoids retaining a load-time allocator value after
        reset_generation_state() releases cache and keeps compare-mode's queue reader from
        consuming the probe response. The send lock spans the system and worker reads, so a reset
        cannot make the same cache bytes appear in both values."""
        proc = self._proc
        active = self.active_model_name
        if proc is None or not proc.is_alive() or not active:
            return None

        with self._dispatcher_lifecycle_lock:
            if self._worker_reserved_for is not None:
                return None
            self._worker_reserved_for = "model switch is checking GPU memory"
        acquired = False
        try:
            if not self._wait_worker_idle():
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
                from utils.hardware import get_visible_gpu_utilization, gpu_query

                live_free: dict[int, float] = {}
                total_by_index: dict[int, float] = {}
                with gpu_query.fresh_reads():
                    _live_devices = get_visible_gpu_utilization().get("devices", [])
                for device in _live_devices:
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
                self._worker_reserved_for = None

    def _shutdown_subprocess(self, timeout: float = 10.0) -> bool:
        with self._subprocess_shutdown_lock:
            return self._shutdown_subprocess_locked(timeout)

    def _shutdown_subprocess_locked(self, timeout: float) -> bool:
        """Gracefully shut down the inference subprocess. Returns True only once the worker is
        confirmed dead. If it survives terminate/kill (e.g. wedged in an uninterruptible CUDA
        syscall that outlives SIGKILL) the live handle is KEPT, not nulled, so is_worker_alive()
        and the pre-swap liveness guard can still observe the survivor instead of a cleared
        handle and refuse the destructive sidecar swap."""
        self._stop_dispatcher()  # before killing subprocess
        if self._proc is None or not self._proc.is_alive():
            # Already gone: a nonzero status is an unwaited crash and keeps its replay.
            exitcode = getattr(self._proc, "exitcode", 0) if self._proc is not None else 0
            self._worker_stopped_deliberately = exitcode == 0
            self._proc = None
            return True

        self._teardown_going_out()
        self._cancel_generation()
        time.sleep(0.5)

        self._drain_queue()

        try:
            self._cmd_queue.put({"type": "shutdown"})
        except (OSError, ValueError):
            self._teardown_not_sent()

        try:
            self._proc.join(timeout = timeout)
        except Exception:
            pass

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
            # Survived SIGKILL (uninterruptible syscall): keep the handle so callers and the pre-swap guard see a live
            # worker rather than a nulled one.
            logger.error(
                "Inference subprocess still alive after terminate/kill; "
                "preserving its handle for the pre-swap liveness check"
            )
            return False

        # Without this flag every model switch replayed a healthy worker's stderr at ERROR.
        self._worker_stopped_deliberately = True
        self._proc = None
        self._cmd_queue = None
        self._resp_queue = None
        self._cancel_event = None
        self._drain_event = None
        self._stop_ledger = None
        self._pending_teardowns = None
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
                from utils.hardware import gpu_query

                result: dict[int, int] = {}
                # Consecutive samples are compared: a cached one would read as settled.
                with gpu_query.fresh_reads():
                    devices = get_visible_gpu_utilization().get("devices", [])
                for device in devices:
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
            # Two unchanged low samples can precede delayed driver reclaim. Stability is meaningful only after an
            # upward release was observed; otherwise consume the full bounded window.
            if stable and observed_reclaim and not expected_mib:
                return True
            previous = current
        return not expected_free_gb

    def _reset_worker_scoped_state(self) -> None:
        """Drop bookkeeping that only means anything for the worker that just died."""
        with self._active_cancel_lock:
            self._active_cancel_events.clear()
            self._executing_cancel_events.clear()
        with self._mailbox_lock:
            self._mailboxes.clear()
            self._direct_mailboxes.clear()
            self._request_cancel_events.clear()

    def _cleanup(self):
        self._shutdown_subprocess(timeout = 5.0)
        self._retire_stderr_capture()

    def _retire_stderr_capture(self) -> None:
        """A native fault BETWEEN requests has no waiter, so nothing reaches `_subprocess_crash_message`."""
        capture = getattr(self, "_stderr_capture", None)
        if capture is None:
            return
        proc = getattr(self, "_proc", None)
        try:
            worker_is_gone = proc is None or not proc.is_alive()
        except Exception:  # noqa: BLE001 -- a handle in teardown; treat it as gone
            worker_is_gone = True
        # `_shutdown_subprocess_locked` has already cleared `_proc`, so the flag is all that still knows.
        if worker_is_gone and not getattr(self, "_worker_stopped_deliberately", False):
            self._log_worker_stderr_once(
                getattr(proc, "pid", None),
                getattr(proc, "exitcode", None),
            )
        try:
            capture.close()
        except Exception:
            pass
        self._stderr_capture = None

    def _public_worker_stderr_tail(self) -> str:
        """Spans the worker's WHOLE lifetime and goes out verbatim through `GenStreamError(public = True)`."""
        raw = self._worker_stderr_tail()
        if not raw:
            return ""
        # Filtered FIRST, then searched: searching raw selected logger-written traceback headers.
        kept = _crash_lines(_diagnostic_lines_only(raw.splitlines()))
        block = "\n".join(kept).strip()
        if not block:
            return ""
        return _redact_worker_output(block)

    def _log_worker_stderr_once(
        self,
        pid,
        exitcode,
        *,
        worker_exited: bool = True,
    ) -> None:
        """The RAW tail, at most once per worker; *worker_exited* is False for a still-live replacement."""
        capture = getattr(self, "_stderr_capture", None)
        if capture is None:
            return
        logged = getattr(self, "_stderr_tail_logged", None)
        if isinstance(logged, tuple) and logged[0] is capture:
            # Already replayed for a real exit, or this is a second non-terminal call.
            if logged[1] or not worker_exited:
                return
        # Marked before the read, so a failure here cannot become a log line per call.
        self._stderr_tail_logged = (capture, bool(worker_exited))
        raw = self._worker_stderr_tail()
        if not raw:
            return
        raw = raw.replace(_LOG_RECORD_START_MARK, "")
        logger.error(
            "Inference worker stderr (pid=%s, exitcode=%s):\n%s",
            pid,
            exitcode,
            raw,
        )

    def _worker_stderr_tail(self) -> str:
        capture = getattr(self, "_stderr_capture", None)
        if capture is None:
            return ""
        try:
            return capture.tail()
        except Exception:
            return ""

    def _teardown_going_out(self) -> None:
        if self._pending_teardowns is not None:
            self._pending_teardowns.sending()

    def _teardown_not_sent(self) -> None:
        if self._pending_teardowns is not None:
            self._pending_teardowns.unsent()

    def _ensure_subprocess_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _observe_response(self, resp, worker):
        """Retire ``worker`` if its Metal queue is dead; nothing else reaps it."""
        detail = resp.get("error") or ""
        if worker is None or not is_metal_queue_dead(detail):
            return resp
        with self._subprocess_shutdown_lock:
            if self._proc is not worker:
                return resp
            logger.error("Retiring the inference worker: its GPU queue is dead (%s)", detail)
            if self._shutdown_subprocess_locked(5):  # a survivor still holds the model
                self.active_model_name = None
                self.models.clear()
        return resp

    def _observe_off_thread(self, resp, worker) -> None:
        """Off the dispatcher thread because retiring joins it."""
        if worker is None or not is_metal_queue_dead(resp.get("error") or ""):
            return
        threading.Thread(
            target = self._observe_response,
            args = (resp, worker),
            daemon = True,
            name = "inference-retire-worker",
        ).start()

    def _subprocess_crash_message(
        self,
        context: str,
        *,
        with_worker_output: bool = False,
    ) -> str:
        """``with_worker_output`` defaults to FALSE: the opposite default discloses a traceback to a merely QUEUED request."""
        context_label = {
            "wait": "loading the model",
            "generation": "generating a response",
            "audio generation": "generating audio",
            "audio input generation": "processing audio input",
        }.get(context, context)
        message = f"The inference worker stopped unexpectedly while {context_label}."

        proc = self._proc
        if proc is None:
            # A concurrent teardown can clear `_proc`; the bytes outlive the handle.
            self._log_worker_stderr_once(None, None)
            return f"{message} Details: process missing."

        exitcode = proc.exitcode
        pid = proc.pid
        if exitcode is None:
            # NOT terminal: the capture belongs to the live replacement, whose crash must stay replayable.
            self._log_worker_stderr_once(pid, None, worker_exited = False)
            return f"{message} Details: pid={pid}."

        # What the worker said before it went (#7843), narrowed to what a client may see.
        tail = self._public_worker_stderr_tail() if with_worker_output else ""
        details = f"\n\nWorker error output:\n{tail}" if tail else ""
        # The operator's unredacted copy: the worker's forwarding daemon thread can die first.
        self._log_worker_stderr_once(pid, exitcode)

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
                f"{message}{suffix} Details: pid={pid}, signal={sig_name}, "
                f"exitcode={exitcode}.{details}"
            )

        return f"{message} Details: pid={pid}, exitcode={exitcode}.{details}"

    def _send_cmd(self, cmd: dict) -> None:
        if self._cmd_queue is None:
            raise RuntimeError("No inference subprocess running")
        try:
            self._cmd_queue.put(cmd)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to send command to subprocess: {exc}")

    def _read_mailbox(
        self,
        mailbox: _WorkerMailbox,
        timeout: Optional[float] = None,
    ):
        resp = mailbox.get_nowait() if timeout is None else mailbox.get(timeout = timeout)
        return self._observe_response(resp, mailbox.worker)

    def _read_resp(
        self,
        timeout: float = 1.0,
        observe: bool = True,
    ) -> Optional[dict]:
        # Handle before queue, else a reload between them blames the replacement.
        worker = self._proc
        resp_queue = self._resp_queue
        if resp_queue is None:
            return None
        try:
            resp = resp_queue.get(timeout = timeout)
            return self._observe_response(resp, worker) if observe else resp
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
        """Block until a response of the expected type arrives. Also handles 'status' and 'error'
        events during the wait. Returns the matching response dict; raises RuntimeError on
        timeout or crash. *timeout* is an **inactivity** timeout: it resets on each status
        message, so long-running operations (large downloads, slow loads) survive as long as the
        subprocess keeps reporting progress."""
        # Local: resolving this name runs the shim's lazy unsloth_zoo load, which pulls torch. The shim caches its
        # pick, so this site and load_model()'s `except` see one class.
        from utils.hf_xet_fallback import DownloadStallError

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise _LoadCancelled()
            remaining = max(0.1, deadline - time.monotonic())
            poll_seconds = 0.1 if cancel_event is not None else 1.0
            resp = self._read_resp(timeout = min(remaining, poll_seconds))

            if resp is None:
                if not self._ensure_subprocess_alive():
                    raise RuntimeError(
                        self._subprocess_crash_message(
                            "wait", with_worker_output = self._owns_worker(cancel_event)
                        )
                    )
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
                deadline = time.monotonic() + timeout
                continue

            if rtype == "stall":
                msg = resp.get("message", "Download stalled")
                logger.warning("Subprocess reported stall: %s", msg)
                raise DownloadStallError(msg)

            logger.debug(
                "Skipping response type '%s' while waiting for '%s'",
                rtype,
                expected_type,
            )

        raise RuntimeError(
            f"Timeout waiting for '{expected_type}' response (no activity for {timeout}s)"
        )

    def _drain_queue(self) -> list:
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

    def _direct_reader(
        self,
        request_id: str,
        cancel_event = None,
    ):
        """Response reader for a _gen_lock generation, safe once compare exists.

        The dispatcher and this reader would otherwise both consume _resp_queue. A dispatcher
        started mid-stream took our responses and dropped them as unaddressed (truncating or hanging
        the chat), and this reader, already blocked on the queue, could take a compare request's
        response before that dispatcher saw it. Registering a mailbox fixes the first; handing
        foreign responses to their own mailbox fixes the second.

        Returns (read_one, drain, release).
        """
        mailbox = _WorkerMailbox(self._proc)
        with self._mailbox_lock:
            self._direct_mailboxes[request_id] = mailbox
            if cancel_event is not None:
                self._request_cancel_events[request_id] = cancel_event

        def read_one(timeout: float = 1.0):
            try:
                return self._read_mailbox(mailbox)
            except queue.Empty:
                pass
            thread = self._dispatcher_thread
            if thread is not None and thread.is_alive():
                try:
                    return self._read_mailbox(mailbox, timeout)
                except queue.Empty:
                    return None
            worker = self._proc  # handle before queue, as in _read_resp
            resp = self._read_resp(timeout = timeout, observe = False)
            if resp is None:
                return None
            rid = resp.get("request_id")
            if rid and rid != request_id:
                with self._mailbox_lock:
                    other = self._mailboxes.get(rid) or self._direct_mailboxes.get(rid)
                    owner = self._request_cancel_events.get(rid)
                if other is not None:
                    if owner is not None:
                        if resp.get("type", "") in ("gen_done", "gen_error"):
                            self._release_worker(owner)
                        else:
                            self._mark_worker_started(owner)
                    other.put(resp)
                # Observe only after hand-over: retiring clears the registry.
                self._observe_response(resp, worker)
                # Outside the mailbox check on purpose: a released request's late frames go to nobody.
                return None
            return self._observe_response(resp, worker)

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
                if cancel_event is not None:
                    self._request_cancel_events.pop(request_id, None)

        return read_one, drain, release

    def _drain_until_gen_done(self, timeout: float = 5.0) -> None:
        """Consume resp_queue events until gen_done/gen_error, discarding them. Called after cancel
        so stale tokens from the cancelled generation don't leak into the next request."""
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

    def _build_generate_cmd(
        self,
        request_id: str,
        image_b64: Optional[str],
        *,
        images_b64: Optional[list] = None,
        image_ordinal: Optional[int] = None,
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
        tool_protocol_active: Optional[bool] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        rows: Optional[list] = None,
        video_b64: Optional[str] = None,
        response_format: Optional[dict] = None,
        reasoning_is_extracted: bool = False,
    ) -> dict:
        """Build the 'generate' command shared by the locked and dispatched paths."""
        cmd = {
            "type": "generate",
            "request_id": request_id,
            "messages": messages or [],
            "system_prompt": system_prompt,
            "image_base64": image_b64,
            "images_base64": images_b64 or None,
            "image_ordinal": image_ordinal,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "parallel_slots": self.effective_parallel_slots,
        }
        if seed is not None:
            cmd["seed"] = seed
        if stop:
            cmd["stop"] = stop
        if response_format is not None:
            cmd["response_format"] = response_format
            cmd["reasoning_is_extracted"] = bool(reasoning_is_extracted)
        if video_b64:
            cmd["video_base64"] = video_b64
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
        if rows:
            cmd["rows"] = rows
        if tool_protocol_active is not None:
            cmd["tool_protocol_active"] = tool_protocol_active
        return cmd

    def _consume_token_stream(
        self,
        read_one,
        drain_on_cancel,
        *,
        crash_context: str,
        request_id: str = "",
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        read_timeout: float = 30.0,
        mark_started: bool = True,
        rows: Optional[int] = None,
    ) -> Generator[Any, None, None]:
        """Yield tokens from a response stream until gen_done/gen_error."""
        # Latch this stream's subprocess/queue: if a wedged worker is torn down and a later load spawns a fresh one,
        # bail rather than re-block on the new queue under _gen_lock (deadlock).
        initial_proc = self._proc
        initial_resp_queue = self._resp_queue
        reading_on_until = None
        stop_recorded = False
        stop_sent = False
        while True:
            if self._proc is not initial_proc or self._resp_queue is not initial_resp_queue:
                if stop_sent:
                    return
                # No tail here whatever the lists say: the capture is the REPLACEMENT's.
                detail = self._subprocess_crash_message(crash_context)
                yield GenStreamError(f"Error: {detail}", public = True)
                return
            if not stop_sent and cancel_event is not None:
                if cancel_event.is_set():
                    stop_recorded, stop_sent = self._stop_and_signal(
                        request_id,
                        cancel_event,
                        stop_recorded,
                        may_signal = False,
                    )
                    if stop_sent and rows is not None and reading_on_until is None:
                        reading_on_until = time.monotonic() + _CANCELLED_ROWS_GRACE
            timeout = read_timeout
            if not stop_recorded and cancel_event is not None:
                timeout = min(timeout, _STOP_NOTICE_INTERVAL)
            if reading_on_until is not None:
                remaining = reading_on_until - time.monotonic()
                if remaining <= 0:
                    drain_on_cancel()
                    return
                timeout = min(timeout, remaining)
            resp = read_one(timeout)
            if resp is None:
                if not self._ensure_subprocess_alive():
                    if stop_sent:
                        return
                    # Only the request the worker was RUNNING gets its last words; the rest were queued behind it.
                    detail = self._subprocess_crash_message(
                        crash_context, with_worker_output = self._owns_worker(cancel_event)
                    )
                    yield GenStreamError(f"Error: {detail}", public = True)
                    return
                continue

            rtype = resp.get("type", "")
            if rtype == "status":
                continue
            if mark_started and not stop_sent:
                self._mark_worker_started(cancel_event)
            # Subprocess-level error (no request_id); request-scoped failures arrive as gen_error below
            if rtype == "error" and not resp.get("request_id"):
                if stop_sent:
                    return
                yield GenStreamError(f"Error: {resp.get('error', 'Unknown error')}")
                return

            if rtype == "token" and cancel_event is not None and cancel_event.is_set():
                if not stop_sent:
                    stop_recorded, stop_sent = self._stop_and_signal(
                        request_id,
                        cancel_event,
                        stop_recorded,
                    )
                    if stop_sent and rows is not None and reading_on_until is None:
                        reading_on_until = time.monotonic() + _CANCELLED_ROWS_GRACE
                if rows is None:
                    drain_on_cancel()
                    return
                # Several replies read on for their completions, but a token drawn
                # after the Stop is one the same reply alone would never have shown.
                continue

            if rtype == "row_done" and rows is not None:
                if stats_holder is not None:
                    reported = stats_holder.setdefault("stats", [None] * rows)
                    row = int(resp.get("row", 0))
                    if 0 <= row < len(reported):
                        reported[row] = resp.get("stats")
                yield int(resp.get("row", 0)), None
            elif rtype == "token":
                if rows is not None:
                    yield int(resp.get("row", 0)), resp.get("text", "")
                else:
                    yield resp.get("text", "")
            elif rtype == "gen_done":
                if stats_holder is not None and rows is None:
                    stats_holder["stats"] = resp.get("stats")
                self._forget_request(request_id, cancel_event)
                return
            elif rtype == "gen_error":
                self._forget_request(request_id, cancel_event)
                if stop_sent:
                    return
                _budget = resp.get("context_budget")
                if _budget:
                    # Rebuilt rather than yielded as text: the route arms match on the type.
                    raise ContextBudgetExceeded(
                        _budget["request_tokens"], _budget["context_tokens"]
                    )
                yield GenStreamError(
                    f"Error: {resp.get('error', 'Unknown error')}",
                    public = bool(resp.get("public", False)),
                    openai_param = resp.get("openai_param"),
                )
                return

    def _start_dispatcher(self) -> Optional[threading.Thread]:
        """Start the dispatcher thread if not already running."""
        with self._dispatcher_lifecycle_lock:
            if self._unload_pending or self._worker_reserved_for:
                return None
            if self._dispatcher_thread is not None and self._dispatcher_thread.is_alive():
                return None

            self._dispatcher_stop.clear()
            self._dispatcher_thread = threading.Thread(
                target = self._dispatcher_loop,
                daemon = True,
                name = "inference-dispatcher",
            )
            self._dispatcher_thread.start()
            logger.debug("Dispatcher thread started")
            return self._dispatcher_thread

    def _stop_dispatcher(self, thread: Optional[threading.Thread] = None) -> None:
        """Signal the dispatcher to stop and wait for it."""
        with self._dispatcher_lifecycle_lock:
            if self._dispatcher_thread is None:
                return
            if thread is not None and self._dispatcher_thread is not thread:
                return
            self._dispatcher_stop.set()
            self._dispatcher_thread.join(timeout = _DISPATCH_STOP_TIMEOUT)
            self._dispatcher_thread = None
            logger.debug("Dispatcher thread stopped")

    def _dispatcher_loop(self) -> None:
        """Background loop: read resp_queue and route to mailboxes by request_id."""
        while not self._dispatcher_stop.is_set():
            if self._resp_queue is None:
                break

            worker = self._proc  # handle before queue, as in _read_resp
            try:
                resp = self._resp_queue.get(timeout = _DISPATCH_POLL_INTERVAL)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                break

            # Sole consumer of the response queue; if it died every in-flight stream would hang, so never let routing
            # kill the dispatcher.
            try:
                rid = resp.get("request_id")
                rtype = resp.get("type", "")

                if rtype == "status":
                    logger.info("Subprocess status: %s", resp.get("message", ""))
                    continue

                # Route to mailbox if a matching request_id exists
                delivered = False
                if rid:
                    with self._mailbox_lock:
                        mbox = self._mailboxes.get(rid) or self._direct_mailboxes.get(rid)
                        owner = self._request_cancel_events.get(rid)
                    if mbox is not None:
                        # Worker order, not consumer order: retire a request the moment its last response is routed.
                        # Waiting for the consumer's finally left it owning the worker after the worker moved on, so a
                        # late Stop for it cancelled whichever request started next.
                        if owner is not None:
                            if rtype in ("gen_done", "gen_error"):
                                self._release_worker(owner)
                            else:
                                self._mark_worker_started(owner)
                        mbox.put(resp)
                        delivered = True

                if not delivered:
                    logger.debug(
                        "Dispatcher: no mailbox for request_id=%s type=%s, dropping",
                        rid,
                        rtype,
                    )
                # Every response: an abandoned mailbox is never read.
                self._observe_off_thread(resp, worker)
            except Exception:
                logger.exception("Inference dispatcher: failed to route a response; continuing")
                continue

    def _generate_dispatched(
        self,
        messages: list = None,
        system_prompt: str = "",
        image = None,
        images: Optional[list] = None,
        image_ordinal: Optional[int] = None,
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
        tool_protocol_active: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        rows: Optional[list] = None,
        expected_model: Optional[str] = None,
        video: Optional[str] = None,
        response_format: Optional[dict] = None,
        reasoning_is_extracted: bool = False,
    ) -> Generator[Any, None, None]:
        """Dispatched generation, sending the command without holding _gen_lock. Uses a per-request
        mailbox for tokens so two compare-mode requests can be queued at once. The subprocess
        still runs commands sequentially, so GPU work stays serialized; this only avoids
        orchestrator lock contention."""
        if not self._ensure_subprocess_alive():
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return

        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        # Latch the target model so the recheck below can detect a switch that completed between _start_dispatcher and
        # mailbox registration (mirrors the locked path's expected_model check).
        if expected_model is None:
            expected_model = self.active_model_name

        # Switch in flight (unload waiting on _gen_lock). This path bypasses the lock, so without this early-out a
        # compare request would enqueue a generate on the outgoing model and delay the switch.
        if self._unload_pending:
            yield GenStreamError("Error: model is being unloaded", public = True)
            return
        request_id = str(uuid.uuid4())

        image_b64 = self._pil_to_base64(image) if image is not None else None
        images_b64 = _encoded_images(images, self._pil_to_base64)

        cmd = self._build_generate_cmd(
            request_id,
            image_b64,
            images_b64 = images_b64,
            image_ordinal = image_ordinal,
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
            response_format = response_format,
            reasoning_is_extracted = reasoning_is_extracted,
            use_adapter = use_adapter,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
            tool_protocol_active = tool_protocol_active,
            seed = seed,
            rows = rows,
            video_b64 = video,
        )

        mailbox = _WorkerMailbox(self._proc)
        admission_deadline = time.monotonic() + _DISPATCH_IDLE_TIMEOUT
        while True:
            self._wait_while_worker_held(cancel_event, deadline = admission_deadline)
            if cancel_event is not None and cancel_event.is_set():
                return

            started = self._start_dispatcher()

            with self._mailbox_lock:
                dispatcher_alive = (
                    self._dispatcher_thread is not None
                    and self._dispatcher_thread.is_alive()
                    and not self._dispatcher_stop.is_set()
                )
                unloading = self._unload_pending or self.active_model_name != expected_model
                reserved_for = self._worker_reserved_for
                blocked = unloading or bool(reserved_for) or not dispatcher_alive
                if not blocked:
                    self._mailboxes[request_id] = mailbox
                    if cancel_event is not None:
                        self._request_cancel_events[request_id] = cancel_event
                orphaned_dispatcher = unloading and started is not None and not self._mailboxes
                if orphaned_dispatcher:
                    self._dispatcher_stop.set()
            if not blocked:
                break
            # _stop_dispatcher joins the dispatcher, which itself takes that lock.
            if orphaned_dispatcher:
                self._stop_dispatcher(started)
            if not unloading and time.monotonic() < admission_deadline:
                time.sleep(_STOP_NOTICE_INTERVAL)
                continue
            if reserved_for:
                detail = f"Error: {reserved_for}"
            elif unloading:
                detail = "Error: model is being unloaded"
            else:
                detail = "Error: nothing is routing replies right now; try again"
            yield GenStreamError(detail, public = True)
            return

        def read_mailbox(timeout):
            try:
                return self._read_mailbox(mailbox, timeout)
            except queue.Empty:
                return None

        send_error = None
        try:
            try:
                with self._send_order_lock:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    self._claim_worker(cancel_event)
                    self._send_cmd(cmd)
            except RuntimeError as exc:
                send_error = exc

            if send_error is None:
                yield from self._consume_token_stream(
                    read_mailbox,
                    lambda: self._drain_mailbox(mailbox, timeout = 5.0),
                    crash_context = "generation",
                    request_id = request_id,
                    cancel_event = cancel_event,
                    stats_holder = stats_holder,
                    read_timeout = _DISPATCH_READ_TIMEOUT,
                    mark_started = False,
                    rows = len(rows) if rows else None,
                )
        finally:
            # Covers streams that end without a gen_done (never sent, cancel, disconnect, dead worker).
            self._release_worker(cancel_event)
            with self._mailbox_lock:
                self._mailboxes.pop(request_id, None)
                self._request_cancel_events.pop(request_id, None)
        if send_error is not None:
            yield GenStreamError(f"Error: {send_error}")

    def _drain_mailbox(
        self,
        mailbox: _WorkerMailbox,
        timeout: float = 5.0,
    ) -> None:
        """Drain a mailbox until gen_done/gen_error, discarding tokens."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = self._read_mailbox(
                    mailbox, min(_DISPATCH_POLL_INTERVAL, deadline - time.monotonic())
                )
            except queue.Empty:
                continue
            rtype = resp.get("type", "")
            if rtype in ("gen_done", "gen_error"):
                return
        logger.warning("Timed out draining mailbox after cancel")

    @contextlib.contextmanager
    def _reserve_worker(self, reason: str):
        """Hold the worker for one caller, answering everything else with ``reason``."""
        with self._worker_released:
            if self._worker_reserved_for is not None:
                raise RuntimeError(f"The worker is already held: {self._worker_reserved_for}")
            self._worker_reserved_for = reason
        try:
            yield
        finally:
            with self._worker_released:
                self._worker_reserved_for = None
                self._worker_released.notify_all()

    def _wait_while_worker_held(
        self,
        cancel_event = None,
        deadline = None,
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + _DISPATCH_IDLE_TIMEOUT
        with self._worker_released:
            while self._worker_reserved_for is not None:
                if cancel_event is not None and cancel_event.is_set():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._worker_released.wait(min(remaining, _STOP_NOTICE_INTERVAL))

    def _replies_in_flight(self) -> int:
        with self._mailbox_lock:
            return len(self._mailboxes) + len(self._direct_mailboxes)

    def _wait_worker_idle(
        self,
        cancel_event = None,
        timeout: float = _DISPATCH_IDLE_TIMEOUT,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return False
            if not self._replies_in_flight():
                break
            time.sleep(0.1)

        if cancel_event is not None and cancel_event.is_set():
            return False

        remaining = self._replies_in_flight()
        if remaining:
            logger.warning(
                "%d repl(ies) still in flight after the idle wait; leaving the dispatcher "
                "running for them",
                remaining,
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

        with self._gen_lock:
            with self._reserve_worker("a distributed share is in progress"):
                if not self._wait_worker_idle():
                    raise RuntimeError(
                        "Cannot share distributed objects while a reply is still generating"
                    )
                request_id = str(uuid.uuid4())
                cmd = {
                    "type": "share_object",
                    "request_id": request_id,
                    "object": obj,
                }

                self._send_cmd(cmd)
                deadline = None if timeout is None else time.monotonic() + timeout
                while deadline is None or time.monotonic() < deadline:
                    remaining = 1.0 if deadline is None else max(0.1, deadline - time.monotonic())
                    resp = self._read_resp(timeout = min(remaining, 1.0))
                    if resp is None:
                        if not self._ensure_subprocess_alive():
                            raise RuntimeError(
                                self._subprocess_crash_message(
                                    "sharing chat turn",
                                    with_worker_output = self._owns_worker(None),
                                )
                            )
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

    # Monotonic count of PUBLISHED loads; lets the install route detect a load (including a same-model reload) that
    # completed while it waited on the gate. Bumped when the load result is published, not at load start: a start-time
    # bump is already visible when the installer snapshots mid-load, so the completed reload would look unchanged and
    # get unloaded by the swap.
    load_generation: int = 0

    @_invalidates_gpu_memory("inference load")
    def load_model(
        self,
        config,
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
        mlx_kv_quant: Optional[str] = None,
        chat_template_override: Optional[str] = None,
        load_cancel_event: Optional[threading.Event] = None,
        post_handoff_expected_free_gb: Optional[dict[int, float]] = None,
        audio_device: Optional[str] = None,
        on_prior_worker_released: Optional[Callable[[], None]] = None,
        cache_environment: Optional[Mapping[str, str]] = None,
        anonymous_hf_access: bool = False,
        audio_codec_path: Optional[str] = None,
        n_parallel: Optional[int] = None,
    ) -> bool:
        """Load a model for inference."""
        from core.inference.llama_server_args import clamp_parallel_slots

        parallel_slots = clamp_parallel_slots(n_parallel)
        from utils.transformers_version import needs_transformers_5

        from utils.hf_xet_fallback import DownloadStallError

        model_name = config.identifier
        self.loading_models.add(model_name)
        if load_cancel_event is not None and load_cancel_event.is_set():
            self.loading_models.discard(model_name)
            logger.info("Load cancelled before worker start: %s", model_name)
            return False

        try:
            needed_major = "5" if needs_transformers_5(model_name) else "4"

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
                "mlx_kv_quant": mlx_kv_quant,
                "chat_template_override": chat_template_override,
                # Read in the worker, which hides the accelerators before detection.
                "audio_device": audio_device,
            }
            if anonymous_hf_access:
                sub_config["anonymous_hf_access"] = True
            if audio_codec_path is not None:
                sub_config["audio_codec_path"] = audio_codec_path
            if audio_device_forces_cpu(audio_device) and is_native_audio_model(model_name):
                # Choosing a card for a load that takes none harms it twice: several GPUs are rejected as unsupported
                # sharding, and required_gb becomes expected_free_gb, so the settle wait raises on a busy card.
                resolved_gpu_ids, gpu_selection = None, {"selection_mode": "cpu_audio"}
            else:
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

            # Recheck the sidecar reservation BEFORE tearing the old worker down, for REPAIRS only: an install holds
            # this same lifecycle gate, so it cannot swap while this load runs, and its queued-load snapshot aborts it
            # after this load publishes -- the load wins cleanly. Raising here (repair) keeps the current model
            # loaded.
            from utils.transformers_version import (
                SidecarSwapInProgress,
                sidecar_swap_kind,
            )

            if sidecar_swap_kind() == "repair":
                raise SidecarSwapInProgress(
                    "A transformers repair is replacing the latest sidecar; "
                    "retry when it completes."
                )

            # Always kill the existing subprocess and spawn fresh: reusing one after unsloth patches torch internals
            # breaks getsource on reload.
            had_worker_handle = self._proc is not None
            worker_shutdown_at = 0.0
            if self._ensure_subprocess_alive():
                self._cancel_generation()
                time.sleep(0.3)
                if self._shutdown_subprocess() is False:
                    # The worker survived terminate/kill (e.g. a wedged CUDA syscall that outlives SIGKILL). Its
                    # handle is kept, so is_worker_alive() and the pre-swap guard still see it; do not spawn a second
                    # worker over one still holding GPU memory. Fail so the load can retry once it exits.
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

            # Previous worker gone, VRAM back: the last moment before a long download.
            if on_prior_worker_released is not None:
                on_prior_worker_released()

            disable_xet = sub_config.get("disable_xet", False) or (
                os.environ.get("HF_HUB_DISABLE_XET") == "1"
            )

            for attempt in range(2):
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
                if cache_environment is None:
                    self._spawn_subprocess(sub_config)
                else:
                    self._spawn_subprocess(sub_config, cache_environment)

                # A cancel can land after the pre-spawn recheck but while _spawn_subprocess is still creating the
                # queues/process. cancel_load runs off the lifecycle gate, so its _shutdown_subprocess can see _proc
                # still None and no-op, orphaning this fresh worker; the load would then wait for "loaded" and publish
                # a model /unload reported unloaded, over a live subprocess nothing reaps. Recheck now the child
                # exists and tear it down before publishing.
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
                    self._shutdown_subprocess(timeout = 5)
                    raise RuntimeError(
                        f"Download stalled for '{model_name}' even with "
                        f"HF_HUB_DISABLE_XET=1 -- check your network connection"
                    )

                if resp.get("success"):
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
                    from utils.process_lifetime import is_process_shutting_down

                    model_info = resp.get("model_info", {})
                    # A "loaded" reply dequeued just as shutdown kills the worker would
                    # otherwise be published here, and active_model_name is what the
                    # already-loaded fast path trusts without testing liveness. Held
                    # under the lock shutdown kills with, so the check and the
                    # publication are one step rather than a race.
                    with self._subprocess_shutdown_lock:
                        if is_process_shutting_down():
                            logger.info(
                                "Shutdown overtook the load of '%s'; not publishing it as resident",
                                model_name,
                            )
                            self.loading_models.discard(model_name)
                            self.active_model_name = None
                            self.models.clear()
                            return False
                        self.active_model_name = model_info.get("identifier", model_name)
                        self.load_generation += 1
                        # A load always spawns a fresh subprocess holding only this model, so mirror that. A lingering stale
                        # name would pass unload_model's "not in self.models" guard, and the worker's absent-name fallback
                        # would unload its *active* model, not the already-gone one.
                        self.models = {}
                        self.models[self.active_model_name] = _mirrored_model_entry(
                            model_info, model_name
                        )
                        self.models[self.active_model_name]["parallel_slots"] = parallel_slots
                        self.models[self.active_model_name]["can_batch"] = model_info.get(
                            "can_batch"
                        )
                        # Lets the already-loaded shortcut tell a CPU request from the GPU
                        # model it would otherwise report as satisfied. Native audio only:
                        # marking anything else tells training a GPU model holds no VRAM.
                        self.models[self.active_model_name]["audio_cpu"] = model_info.get(
                            "audio_type"
                        ) in NATIVE_AUDIO_TYPES and audio_device_forces_cpu(audio_device)
                        self.models[self.active_model_name].update(
                            _mlx_runtime_mirror_fields(model_info)
                        )
                        # Mirror chat_template_info so routes can classify caps without re-entering the subprocess
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
                # Raised before the old worker was torn down: the previous model is still live, so keep the mirrors
                # (clearing them would let the installer treat the worker as inactive and kill it unreported).
                raise
            self.active_model_name = None
            self.models.clear()
            # Reap workers after any failed load, including inactivity timeouts that leave installs and GPU memory
            # alive (#9398)
            try:
                self._shutdown_subprocess(timeout = 5)
            except Exception as teardown_exc:
                logger.warning("Could not shut the failed load's worker down: %s", teardown_exc)
            raise

    def cancel_load(self, model_name: str) -> bool:
        """Abort an in-flight load by terminating its subprocess. Returns True if a load for
        ``model_name`` (matched case-insensitively) was cancelled, False if nothing was loading
        under that name. This only tears the loading subprocess down -- it sends no command to a
        worker -- so, unlike the rest of ``unload_model``, it is safe to run WITHOUT the
        inference lifecycle gate. ``/unload`` calls it off-gate so the "stop loading" button can
        interrupt a safetensors load that holds the gate for its whole (multi-minute) duration; a
        gated cancel could never preempt that load."""
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
        # Discard the loading marker (and clear local state) BEFORE the teardown, not after. cancel_load runs off the
        # lifecycle gate, alongside a load_model that rechecks this marker before each spawn. But _shutdown_subprocess
        # can block (~1s tearing a live child down and joining the dispatcher), so clearing only after leaves a window
        # where load_model reads the marker still set, passes its pre-spawn recheck, and loads the model after /unload
        # reported it cancelled. Clear first.
        self.loading_models.discard(target)
        self.active_model_name = None
        self.models.clear()
        self._shutdown_subprocess(timeout = 0.5)
        # Clear the local mirrors again AFTER the teardown. A racing off-gate load_model may still be parked in
        # _wait_response("loaded"): its worker already queued a "loaded" reply, so during the shutdown window above
        # (the 0.5s settle before the response queue is drained and nulled) that thread can consume it and repopulate
        # active_model_name/models, undoing the pre-teardown clear. _shutdown_subprocess nulls the queue but not the
        # mirrors, so without this second clear /unload reports success while the backend still advertises a killed
        # model. The nulled queue lets no further "loaded" through, so re-clearing here wipes any repopulation.
        self.active_model_name = None
        self.models.clear()
        return True

    # Dictation models run in the STT sidecars (whisper-server, llama-server, and the Transformers spawn child), not
    # the chat worker. Their lifecycle goes through here all the same, so one object knows everything that is resident
    # and Voice settings and Model Hub cannot report different things about one model.
    def load_stt_model(
        self,
        model: Optional[str],
        engine: str,
        request_cancel_event: Optional[threading.Event] = None,
        device: Optional[str] = None,
    ) -> None:
        """Make a dictation model resident on its sidecar. ``device`` is the user's audio device
        preference (``auto``/``cpu``/``gpu``)."""
        from core.inference import stt_registry
        stt_registry.load(model, engine, request_cancel_event, device = device)

    def unload_stt_model(
        self,
        engines: Optional[Sequence[str]] = None,
        expected_model: Optional[str] = None,
        wait: bool = True,
    ) -> list:
        """Release dictation models (all engines by default); returns refusals. ``expected_model``
        scopes the release to a sidecar still holding that model. ``wait=False`` leaves a sidecar
        that is mid-request alone, for a caller freeing memory opportunistically rather than to
        reclaim it now."""
        from core.inference import stt_registry
        return stt_registry.unload(engines, wait = wait, expected_model = expected_model)

    def resident_stt_model(self) -> dict:
        """What dictation holds, alongside active_model_name for chat."""
        from core.inference import stt_registry
        return stt_registry.resident()

    @_invalidates_gpu_memory("inference unload")
    def unload_model(self, model_name: str) -> bool:
        # active_model_name can differ in case from the client's raw /unload name (the load path canonicalizes
        # casing). Match case-insensitively and use the canonical spelling so the guard, unload command, and cleanup
        # below hit the loaded model.
        if (
            self.active_model_name is not None
            and model_name != self.active_model_name
            and model_name.lower() == self.active_model_name.lower()
        ):
            model_name = self.active_model_name
        # In-flight load: tear its subprocess down (shared loading-cancel logic; no worker command sent)
        if self.cancel_load(model_name):
            return True

        if not self._ensure_subprocess_alive():
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None
            return True

        # Nothing loaded under this name: don't unload a stale model. The worker falls back to unloading its *active*
        # model when the name is absent, so a stale unload (lost a race to a concurrent load) would hit the wrong one.
        if model_name != self.active_model_name and model_name not in self.models:
            self.models.pop(model_name, None)
            return True

        with self._dispatcher_lifecycle_lock:
            self._unload_pending = True
        # Cancelling only the running generation isn't enough: the worker clears cancel_event at each generate start,
        # so a queued one would clear it and run the outgoing model to completion. drain_event, never cleared, makes
        # any generate dequeued during the unload skip.
        if self._drain_event is not None:
            self._drain_event.set()
        try:
            self._cancel_every_generation()
            acquired = self._gen_lock.acquire(timeout = _UNLOAD_GEN_LOCK_TIMEOUT)
            if not acquired:
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
                if not self._wait_worker_idle(timeout = _DISPATCH_IDLE_TIMEOUT):
                    logger.warning(
                        "Unload: compare-mode dispatcher still active after idle "
                        "wait; shutting the inference subprocess down to free the model"
                    )
                    self._shutdown_subprocess(timeout = 5)
                    self.models.pop(model_name, None)
                    if self.active_model_name == model_name:
                        self.active_model_name = None
                    return True
                self._drain_queue()
                self._teardown_going_out()
                try:
                    self._send_cmd(
                        {
                            "type": "unload",
                            "model_name": model_name,
                        }
                    )
                except Exception:
                    self._teardown_not_sent()
                    raise
                self._wait_response("unloaded")

                self.models.pop(model_name, None)
                if self.active_model_name == model_name:
                    self.active_model_name = None

                logger.info("Model '%s' unloaded from subprocess", model_name)
                # empty_cache in the child cannot return the accelerator context, so an idle worker keeps its
                # high-water mark -- VRAM the GGUF backend cannot see and gpu_arbiter never evicts (both are
                # chat-owned). Nothing left to serve, so drop it; load_model respawns a fresh worker regardless.
                if not self.models and not self.loading_models:
                    logger.info("No models left resident; shutting the inference subprocess down")
                    try:
                        self._shutdown_subprocess(timeout = 5)
                    except Exception as exc:
                        logger.warning("Could not shut the idle inference subprocess down: %s", exc)
                return True

            except Exception as exc:
                logger.error("Error unloading model '%s': %s", model_name, exc)
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
        """Prompt tokens the loaded model would receive, and whose tokenizer counted them. Reads
        through an addressed mailbox, as generations do: compare mode bypasses the generation
        lock and leaves a dispatcher owning the response queue, which would route this reply
        nowhere."""
        if not self._gen_lock.acquire(blocking = False):
            raise RuntimeError("Cannot count tokens while a generation is in progress")
        with self._mailbox_lock:
            dispatched = bool(self._mailboxes)
        if dispatched:
            self._gen_lock.release()
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
                        # A count takes no part in the claim bookkeeping.
                        raise RuntimeError(
                            self._subprocess_crash_message(
                                "count", with_worker_output = self._owns_worker(None)
                            )
                        )
                    continue
                # _direct_reader already drops a reply whose mailbox is gone; this is the backstop.
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
        images: Optional[list] = None,
        image_ordinal: Optional[int] = None,
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
        tool_protocol_active: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        video: Optional[str] = None,
        response_format: Optional[dict] = None,
        reasoning_is_extracted: bool = False,
    ) -> Generator[str, None, None]:
        """Generate response, streaming tokens from subprocess. ``tools`` / ``enable_thinking`` /
        ``reasoning_effort`` / ``preserve_thinking`` are forwarded so the template can render
        tool schemas and reasoning controls. ``stats_holder`` is a caller-owned dict whose
        "stats" key gets the worker's usage, timings and terminal reason on gen_done;
        request-scoped to avoid cross-stream reads. ``presence_penalty`` matches the GGUF
        sampling path (0 disables it)."""
        yield from self._generate_inner(
            messages = messages,
            system_prompt = system_prompt,
            image = image,
            images = images,
            image_ordinal = image_ordinal,
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
            tool_protocol_active = tool_protocol_active,
            stats_holder = stats_holder,
            presence_penalty = presence_penalty,
            seed = seed,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
            response_format = response_format,
            reasoning_is_extracted = reasoning_is_extracted,
            video = video,
        )

    def generate_chat_batch(
        self,
        rows: list,
        messages: list,
        system_prompt: str = "",
        image = None,
        images: Optional[list] = None,
        image_ordinal: Optional[int] = None,
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
        tool_protocol_active: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        video: Optional[str] = None,
        response_format: Optional[dict] = None,
        reasoning_is_extracted: bool = False,
    ) -> Generator[Any, None, None]:
        """Ask for several replies to one prompt in a single command."""
        yield from self._generate_inner(
            messages = messages,
            system_prompt = system_prompt,
            image = image,
            images = images,
            image_ordinal = image_ordinal,
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
            tool_protocol_active = tool_protocol_active,
            stats_holder = stats_holder,
            presence_penalty = presence_penalty,
            seed = seed,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
            rows = rows,
            video = video,
            response_format = response_format,
            reasoning_is_extracted = reasoning_is_extracted,
        )

    def generate_chat_completion_with_tools(
        self,
        messages: list,
        tools: list,
        system_prompt: str = "",
        images: Optional[list] = None,
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
        caller_image_indexes: "tuple[int, ...]" = (),
        **_unused,
    ):
        """Run the safetensors agentic tool loop in the parent process, calling the worker for each
        turn. Yields the same event dicts as the GGUF tool loop so the route layer can stream
        both backends through one helper."""
        from core.inference.safetensors_agentic import run_safetensors_tool_loop
        from core.inference.tools import execute_tool

        # None lets the backend size an unset limit once it has counted the prompt.
        max_new_tokens = max_tokens if max_tokens and max_tokens > 0 else None
        # Only a model that reads images gets a sink; the loop leaves MCP pictures
        # out of the prompt without one.
        loop_images: Optional[list] = (
            list(images or [])
            if self.models.get(self.active_model_name, {}).get("is_vision")
            else None
        )

        # The worker's usage for the LATEST turn only. Hoisted out of the turn so the loop can size a conversation
        # search against a real prompt count, and cleared on the way in rather than on each way out, so a turn that
        # failed or was cancelled leaves it empty instead of handing the loop an earlier turn's number.
        turn_stats: dict = {}

        def _single_turn(
            conv: list,
            *,
            active_tools: Optional[list[dict]] = None,
            tool_protocol_active: Optional[bool] = None,
        ):
            # ``conv`` already carries any system message. ``active_tools`` lets run_safetensors_tool_loop drop
            # one-shot tools (e.g. render_html) from later same-response prompts.
            turn_tools = active_tools if active_tools is not None else tools
            turn_stats.clear()
            common_kwargs = dict(
                messages = conv,
                system_prompt = "",
                image = None,
                images = list(loop_images) if loop_images else None,
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
                # Self-limiting: after a tool call the conversation ends on a tool result, so later turns render as
                # ordinary new turns.
                continue_final_message = continue_final_message,
                tool_protocol_active = tool_protocol_active,
                # Reported per turn and summed below, since the whole loop answers one request.
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
                        raise GenStreamErrorRaised.from_chunk(chunk)
                    yield chunk
            finally:
                if close_stream:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug("failed to close errored generation stream", exc_info = True)
                # A turn that never reported (one a cancel interrupted) folds in as nothing, leaving the turns that
                # did
                if stats_holder is not None:
                    stats_holder["stats"] = _summed_tool_loop_stats(
                        stats_holder.get("stats"), turn_stats.get("stats")
                    )

        initial = list(messages)
        if system_prompt:
            initial = [{"role": "system", "content": system_prompt}] + initial

        # Same profile the renderer uses, so the controller never drops a tool over a marker this model does not treat
        # as structure. The controller is also given the catalog safe under every template this turn could select,
        # because the native-template fallback renders with a different profile (#7066).
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
            generation_stats_holder = turn_stats,
            images_sink = loop_images,
            # Which sink entries are the caller's own attachment, so the loop's cap
            # never evicts it. Empty when the model reads no images, since there is
            # then no sink to protect anything in.
            caller_image_indexes = tuple(caller_image_indexes) if loop_images else (),
        )

    def generate_with_adapter_control(
        self,
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        """Generate with adapter control, streaming tokens from subprocess. Uses the dispatcher path
        (no _gen_lock) so compare-mode requests don't block each other; the subprocess serializes
        them via its sequential command loop. Backend failures raise instead of becoming
        assistant text."""
        stream = self._generate_dispatched(
            use_adapter = use_adapter,
            cancel_event = cancel_event,
            stats_holder = stats_holder,
            **gen_kwargs,
        )
        try:
            for chunk in stream:
                if isinstance(chunk, GenStreamError):
                    raise GenStreamErrorRaised.from_chunk(chunk)
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
        images: Optional[list] = None,
        image_ordinal: Optional[int] = None,
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
        tool_protocol_active: Optional[bool] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        rows: Optional[list] = None,
        video: Optional[str] = None,
        response_format: Optional[dict] = None,
        reasoning_is_extracted: bool = False,
    ) -> Generator[Any, None, None]:
        """Inner generation logic: sends the command to the subprocess and yields tokens. Serialized
        by _gen_lock (one generation at a time) so concurrent readers don't consume each other's
        tokens off the shared resp_queue."""
        if not self._ensure_subprocess_alive():
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return

        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        expected_model = self.active_model_name

        if self._stop_ledger is not None and self._stop_ledger.read_by_worker():
            yield from self._generate_dispatched(
                messages = messages,
                system_prompt = system_prompt,
                image = image,
                images = images,
                image_ordinal = image_ordinal,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_new_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                cancel_event = cancel_event,
                use_adapter = use_adapter,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                tool_protocol_active = tool_protocol_active,
                stats_holder = stats_holder,
                presence_penalty = presence_penalty,
                seed = seed,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                rows = rows,
                expected_model = expected_model,
                video = video,
                response_format = response_format,
                reasoning_is_extracted = reasoning_is_extracted,
            )
            return

        with self._gen_lock:
            if self._unload_pending or self.active_model_name != expected_model:
                yield GenStreamError("Error: model is being unloaded", public = True)
                return
            if cancel_event is not None and cancel_event.is_set():
                return
            request_id = str(uuid.uuid4())
            image_b64 = self._pil_to_base64(image) if image is not None else None
            images_b64 = _encoded_images(images, self._pil_to_base64)
            cmd = self._build_generate_cmd(
                request_id,
                image_b64,
                images_b64 = images_b64,
                image_ordinal = image_ordinal,
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
                response_format = response_format,
                reasoning_is_extracted = reasoning_is_extracted,
                use_adapter = use_adapter,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                tool_protocol_active = tool_protocol_active,
                seed = seed,
                rows = rows,
                video_b64 = video,
            )

            read_one, drain, release_mailbox = self._direct_reader(request_id, cancel_event)
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
                    request_id = request_id,
                    cancel_event = cancel_event,
                    stats_holder = stats_holder,
                    rows = len(rows) if rows else None,
                )
            finally:
                self._release_worker(cancel_event)
                release_mailbox()

    def _claim_worker(self, cancel_event) -> None:
        """Record this request as one the worker will run."""
        with self._active_cancel_lock:
            self._active_cancel_events.append(cancel_event)

    def _mark_worker_started(self, cancel_event) -> None:
        if cancel_event is None:
            return
        with self._active_cancel_lock:
            if not any(ev is cancel_event for ev in self._active_cancel_events):
                return
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
        """Whether a reset from this request may signal the shared cancel event. True when it is one
        of the EXECUTING generations, and when nothing is in flight at all: an error path that
        resets before anything started has no one else to interrupt, so it must not become a
        silent no-op. Claimed but queued does not count, or a Stop on a queued request would end
        the running one, including during the prefill before any response arrives."""
        with self._active_cancel_lock:
            if not self._active_cancel_events:
                return True
            if self._executing_cancel_events:
                return any(ev is cancel_event for ev in self._executing_cancel_events)
            # Claimed but nothing has answered yet (A is in prefill). The worker takes commands in order, so the
            # oldest claim is the executor; anyone else here is queued behind it.
            return self._active_cancel_events[0] is cancel_event

    def _stop_and_signal(
        self,
        request_id: str,
        cancel_event,
        recorded: bool = False,
        *,
        may_signal: bool = True,
    ) -> tuple[bool, bool]:
        """End one request: by name where the worker reads stops, else via the shared event.
        Reports (recorded, sent)."""
        ledger = self._stop_ledger
        sent = False
        if ledger is not None:
            if not recorded:
                recorded = bool(
                    request_id and self._ensure_subprocess_alive() and ledger.stop(request_id)
                )
            sent = bool(recorded and ledger.read_by_worker())
        if not sent and may_signal:
            with self._active_cancel_lock:
                answered = any(ev is cancel_event for ev in self._executing_cancel_events)
            if answered:
                self._cancel_generation()
                sent = True
        if sent:
            self._release_worker(cancel_event)
        return recorded, sent

    def _forget_request(self, request_id: str, cancel_event) -> None:
        """Retire a reply as it ends: a cancel event outliving its request stops a later one."""
        if request_id:
            with self._mailbox_lock:
                self._request_cancel_events.pop(request_id, None)
        self._release_worker(cancel_event)

    def _request_of(self, cancel_event) -> Optional[str]:
        if cancel_event is None:
            return None
        with self._mailbox_lock:
            for request_id, event in self._request_cancel_events.items():
                if event is cancel_event:
                    return request_id
        return None

    def reset_generation_state(self, caller_cancel_event = None):
        """Cancel any ongoing generation and reset state."""
        if caller_cancel_event is not None:
            request_id = self._request_of(caller_cancel_event)
            if request_id is not None:
                self._stop_and_signal(request_id, caller_cancel_event)
                return
            with self._send_order_lock:
                if not self._owns_worker(caller_cancel_event):
                    return
                self._reset_worker()
            return
        self._reset_worker()

    def _reset_worker(self) -> None:
        self._teardown_going_out()
        self._cancel_generation()
        if not self._ensure_subprocess_alive():
            self._teardown_not_sent()
            return
        try:
            with self._send_order_lock:
                self._send_cmd({"type": "reset"})
        except RuntimeError:
            self._teardown_not_sent()

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
        """Generate TTS audio. Returns (wav_bytes, sample_rate). Blocking: sends the command and
        waits for the full audio response."""
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")
        if not self.active_model_name:
            raise RuntimeError("No active model")
        expected_model = self.active_model_name

        # Serialize under _gen_lock and reserve dispatcher admission before waiting for compare work to drain. A bare
        # idle wait is racy: a compare request can register between the wait and this command, leaving TTS queued
        # without safe ownership of the worker's single shared cancel event.
        with self._gen_lock:
            with self._reserve_worker("audio generation is in progress"):
                idle = self._wait_worker_idle(cancel_event = cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise AudioGenerationCancelledError("Audio generation cancelled")
                if not idle:
                    raise RuntimeError(
                        "Cannot start audio generation while a reply is still generating"
                    )

                # Recheck after the dispatcher wait: unload can set its flag without _gen_lock, and a switch may have
                # completed while this call was queued.
                if self._unload_pending or self.active_model_name != expected_model:
                    raise AudioGenerationCancelledError("model is being unloaded")

                # Bound public API integers before either enqueuing work or calculating the floating-point watchdog
                # deadline
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
                read_one, _drain, release_mailbox = self._direct_reader(request_id, cancel_event)
                try:
                    # Claim before enqueueing so request-scoped reset ownership follows the same discipline as text
                    # and audio-input generation
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
                            # audio_started is emitted after the worker clears stale state, so this signal cannot be
                            # erased or hit an earlier request.
                            self._cancel_generation()
                            cancel_signalled = True
                            # The cancel is delivered now, so hold the worker to the drain window from here rather
                            # than from when the caller asked
                            cancel_deadline = time.monotonic() + _AUDIO_CANCEL_DRAIN_TIMEOUT
                            deadline = min(deadline, cancel_deadline)
                        remaining = max(0.1, deadline - time.monotonic())
                        resp = read_one(timeout = min(remaining, 1.0))

                        if resp is None:
                            if not self._ensure_subprocess_alive():
                                raise RuntimeError(
                                    self._subprocess_crash_message(
                                        "audio generation",
                                        with_worker_output = self._owns_worker(cancel_event),
                                    )
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

                    # A caller cancellation already spent the drain window polling this request's mailbox. Tear down
                    # an unresponsive worker now instead of waiting out the much longer generation watchdog or
                    # draining twice.
                    if cancel_deadline is not None:
                        if self._shutdown_subprocess(timeout = _AUDIO_CANCEL_DRAIN_TIMEOUT):
                            self.active_model_name = None
                            self.models.clear()
                        raise AudioGenerationCancelledError("Audio generation cancelled")

                    # Do not release worker ownership or dispatcher exclusivity over a command that may still be
                    # generating. Cancel, consume its terminal response, and tear down a worker that does not
                    # acknowledge promptly.
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

    def generate_whisper_response(
        self,
        audio_array,
        cancel_event = None,
        stats_holder: Optional[dict] = None,
        extra_audio_arrays: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """Whisper ASR: sends audio to the subprocess and yields text."""
        yield from self._generate_audio_input_inner(
            audio_array = audio_array,
            audio_type = "whisper",
            messages = [],
            system_prompt = "",
            cancel_event = cancel_event,
            stats_holder = stats_holder,
            extra_audio_arrays = extra_audio_arrays,
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
        extra_audio_arrays: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """Audio input generation (e.g. Gemma 3n): streams text tokens."""
        yield from self._generate_audio_input_inner(
            audio_array = audio_array,
            audio_type = None,  # worker uses generate_audio_input_response
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
            extra_audio_arrays = extra_audio_arrays,
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
        extra_audio_arrays: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """Shared inner logic for audio input generation (Whisper + ASR). ``stats_holder``: as in
        generate_chat_response, caller-owned and filled on gen_done with the worker's usage /
        budget report."""
        if not self._ensure_subprocess_alive():
            yield GenStreamError("Error: Inference subprocess is not running", public = True)
            return
        if not self.active_model_name:
            yield GenStreamError("Error: No active model", public = True)
            return
        expected_model = self.active_model_name

        with self._gen_lock:
            # Recheck under the lock (see _generate_inner): a raced unload/switch may have cleared or swapped the
            # model while we waited.
            if self._unload_pending or self.active_model_name != expected_model:
                yield GenStreamError("Error: model is being unloaded", public = True)
                return
            if cancel_event is not None and cancel_event.is_set():
                return
            request_id = str(uuid.uuid4())

            import numpy as np

            # Raw float32 bytes per clip; far cheaper to pickle than tolist().
            clips = [audio_array, *(extra_audio_arrays or [])]
            audio_clips = [np.asarray(clip, dtype = np.float32).tobytes() for clip in clips]

            cmd = {
                "type": "generate_audio_input",
                "request_id": request_id,
                "audio_clips": audio_clips,
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
            if use_adapter is not None:
                cmd["use_adapter"] = use_adapter
            if stop:
                cmd["stop"] = stop

            read_one, drain, release_mailbox = self._direct_reader(request_id, cancel_event)
            try:
                try:
                    # Claim under the send lock, like _generate_inner: unclaimed, a compare request queued behind this
                    # looked like the oldest owner, so stopping it killed this one.
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
                    request_id = request_id,
                    cancel_event = cancel_event,
                    stats_holder = stats_holder,
                )
            finally:
                self._release_worker(cancel_event)
                release_mailbox()

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
        return len(self.loading_models) > 0

    def get_loading_model(self) -> Optional[str]:
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


_inference_backend = None
# Guards the lazy construction below. The first build runs hardware detection, seconds cold, and first-paint routes
# call this getter from executor threads. Unlocked, several would see None and each build an orchestrator, orphaning
# all but the last plus any load on them.
_inference_backend_lock = threading.Lock()


def peek_inference_backend() -> Optional["InferenceOrchestrator"]:
    """The orchestrator if one exists, else None. Never constructs one. For callers that only
    describe what is already loaded: constructing reaches get_default_models() -> get_device(),
    which blocks on the torch import during the warm."""
    return _inference_backend


def get_inference_backend() -> InferenceOrchestrator:
    """Global inference backend instance (orchestrator)."""
    global _inference_backend
    # Double-checked: the cheap read keeps the hot path lock-free, the recheck picks a builder
    if _inference_backend is None:
        with _inference_backend_lock:
            if _inference_backend is None:
                _inference_backend = InferenceOrchestrator()
    return _inference_backend
