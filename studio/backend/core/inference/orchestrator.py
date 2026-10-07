# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference orchestrator, subprocess-based. Same API as InferenceBackend, but delegates all ML work
to a persistent subprocess spawned on first model load and reused for later requests. When
switching between models needing different transformers versions (e.g. GLM-4.7-Flash needs 5.x,
Qwen needs 4.57.x), the old subprocess is killed and a new one spawned with the correct version.
Pattern follows core/training/training.py."""

import atexit
import contextvars
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
from core.inference.audio_device import audio_device_forces_cpu, audio_load_runs_on_cpu
from core.inference.context_refusal import ContextBudgetExceeded
from core.inference.native_audio import NATIVE_AUDIO_TYPES, is_native_audio_model
from core.inference.audio_errors import (
    AUDIO_RUNTIME_ERROR_CODE,
    AUDIO_UNSUPPORTED_CODE,
    AudioBackendUnsupportedError,
    AudioGenerationCancelledError,
    AudioRuntimeError,
)
from core.inference.worker import PendingTeardowns, StopLedger
from utils.hardware import get_device, prepare_gpu_selection
from utils.utils import hf_env_offline, is_metal_queue_dead

# PEP 562 lazy export: resolving it imports unsloth_zoo (torch), too heavy at startup.
DownloadStallError: type


def __getattr__(name: str):
    if name == "DownloadStallError":
        from utils.hf_xet_fallback import DownloadStallError as _exc
        return _exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = get_logger(__name__)

# Delimited, not a whitelist: a whitelist stopped at the apostrophe in `/home/o'connor/`.
_PATH_COMPONENT = r"[^\s\\/](?:(?:(?![A-Za-z]:[\\/])[^\\/\n\",;])*[^\s\\/])?"
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

# Transformers TTS only; generous since _ensure_subprocess_alive already catches dead workers.
_AUDIO_GENERATION_TIMEOUT = 900.0
_AUDIO_GENERATION_BASE_TOKENS = 2048
AUDIO_GENERATION_MAX_TOKENS = 8192
MOSS_TTS_MAX_FRAMES = 32768
MINIMAX_MUSIC_MAX_FRAMES = 9000
_AUDIO_CANCEL_DRAIN_TIMEOUT = 5.0
# Prefill can outlast the drain window; tearing down early unloads the just-loaded model.
_AUDIO_CANCEL_TEARDOWN_TIMEOUT = 30.0

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
    "mlx_int8_prefill",
    "mlx_int8_prefill_requested",
    "mlx_int8_prefill_reason",
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
    # Prompt count is the loop's; details move with it so cached never exceeds prompt tokens.
    if not usage.get("prompt_tokens"):
        usage["prompt_tokens"] = prior_usage.get("prompt_tokens") or 0
        usage.pop("prompt_tokens_details", None)
        if prior_usage.get("prompt_tokens_details") is not None:
            usage["prompt_tokens_details"] = prior_usage["prompt_tokens_details"]
    usage["total_tokens"] = usage["prompt_tokens"] + completion
    details = dict(prior_usage.get("completion_tokens_details") or {})
    for field, value in (usage.get("completion_tokens_details") or {}).items():
        details[field] = (details.get(field) or 0) + (value or 0)
    if details:
        usage["completion_tokens_details"] = details
    summed = dict(turn)
    summed["usage"] = usage
    timings = dict(turn.get("timings") or {})
    prior = total.get("timings") or {}
    if timings or prior:
        for field in ("predicted_ms", "predicted_n"):
            timings[field] = (timings.get(field) or 0) + (prior.get(field) or 0)
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
        "audio_family": model_info.get("audio_family"),
        "audio_options": model_info.get("audio_options"),
        "gguf_variant": model_info.get("gguf_variant"),
        "audio_workflows": model_info.get("audio_workflows"),
        "audio_reference_text": model_info.get("audio_reference_text"),
        "audio_required_inputs": model_info.get("audio_required_inputs"),
        "audio_clone": model_info.get("audio_clone"),
        "audio_options_by_workflow": model_info.get("audio_options_by_workflow"),
        "audio_workflow_tasks": model_info.get("audio_workflow_tasks"),
        "audio_server_task": model_info.get("audio_server_task"),
        "audio_convert": model_info.get("audio_convert"),
        "audio_convert_route": model_info.get("audio_convert_route"),
        "audio_convert_rules": model_info.get("audio_convert_rules"),
        "audio_edit": model_info.get("audio_edit"),
        "audio_music": model_info.get("audio_music"),
        "audio_cpp_backend": model_info.get("audio_cpp_backend"),
    }


class InferenceOrchestrator:
    """Inference backend orchestrator, subprocess-based. Same API surface as InferenceBackend (so
    routes/inference.py needs minimal changes); all heavy ML work happens in a persistent
    subprocess."""

    _load_download_keys: Sequence[str] = ()

    def __init__(self):
        self._managed_engine = None
        self._proc: Optional[mp.Process] = None
        self._stderr_capture: Any = None
        self._cmd_queue: Any = None
        self._resp_queue: Any = None
        self._subprocess_shutdown_lock = threading.RLock()
        self._cancel_event: Any = None
        # Never cleared by the worker, so a generate queued behind the cancelled one is skipped.
        self._drain_event: Any = None
        self._stop_ledger: Any = None
        self._pending_teardowns: Any = None
        self._gen_lock = threading.Lock()
        self._active_cancel_events: list = []
        self._executing_cancel_events: list = []
        self._active_cancel_lock = threading.Lock()
        # Held across claim + _send_cmd so claim order matches subprocess dequeue order (_owns_worker).
        self._send_order_lock = threading.RLock()
        self._unload_pending = False
        self._worker_reserved_for: Optional[str] = None

        self._mailboxes: dict[str, queue.Queue] = {}
        # Only the dispatcher sees responses in worker order, so it moves ownership.
        self._request_cancel_events: dict[str, object] = {}
        # Separate from _mailboxes, which means "compare requests in flight" to unload.
        self._direct_mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._dispatcher_stop = threading.Event()
        # Prevents two compare requests spawning duplicate dispatchers that steal the unload reply.
        self._dispatcher_lifecycle_lock = threading.Lock()
        self._worker_released = threading.Condition(self._dispatcher_lifecycle_lock)

        self.active_model_name: Optional[str] = None
        self.models: dict = {}
        self.loading_models: set = set()
        from core.inference.defaults import get_default_models

        # Read the detection stamp BEFORE the list, or a re-detect tags the old list as new.
        import utils.hardware.hardware as _hw_mod

        self._static_models_generation = _hw_mod.DETECTION_GENERATION
        self._static_models = get_default_models()
        self._static_models_lock = threading.Lock()
        self._top_gguf_cache: Optional[list[str]] = None
        self._top_hub_cache: Optional[list[str]] = None
        self._top_models_ready = threading.Event()

        atexit.register(self._cleanup)
        logger.info("InferenceOrchestrator initialized (subprocess mode)")

        # Not started here: would hit huggingface.co on every boot.
        self._top_models_started = False

    def _refresh_static_models_if_stale(self) -> None:
        """Recompute the curated defaults if hardware was re-detected since."""
        import utils.hardware.hardware as _hw_mod

        generation = _hw_mod.DETECTION_GENERATION
        if generation == self._static_models_generation:
            return
        from core.inference.defaults import get_default_models

        models = get_default_models()
        with self._static_models_lock:
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
        # Check offline before the latch, or an offline boot could never retry the fetch.
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

        if getattr(self, "_managed_engine", None) is not None:
            return 1
        entry = self.models.get(self.active_model_name or "") or {}
        slots = entry.get("parallel_slots")
        return slots if isinstance(slots, int) and slots > 0 else PARALLEL_DEFAULT

    @property
    def default_models(self) -> list[str]:
        self._refresh_static_models_if_stale()
        self._start_top_models_fetch()
        top_gguf = self._top_gguf_cache or []
        top_hub = self._top_hub_cache or []
        from core.inference.defaults import suggestions_for_host
        import utils.hardware.hardware as _hw_mod

        device = None if _hw_mod.CHAT_ONLY else _hw_mod.DEVICE
        fetched = suggestions_for_host(top_gguf + top_hub, device)
        # Never wait for the Hub ranking at startup; the background fetch backfills.
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
                gguf_ids = [m["id"] for m in models if m.get("id", "").upper().endswith("-GGUF")][
                    :40
                ]
                hub_ids = [
                    m["id"] for m in models if not m.get("id", "").upper().endswith("-GGUF")
                ][:40]
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
        # Last gate before Popen: preview/auto-switch loads are not cancellable by the shutdown sweep.
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

            # Build and start via a local: shutdown may clear self._proc while start() returns (orphan).
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

        adopt_pid(_spawned_proc.pid)

        # Recheck after spawn: shutdown may have swept while the child was born. adopt_pid runs first.
        # No lock across the spawn: _shutdown_subprocess holds it for its whole teardown.
        if is_process_shutting_down() or self._proc is not _spawned_proc:
            logger.info("Shutdown began during spawn; tearing the new inference worker down")
            self._shutdown_subprocess(timeout = 5)
            # If shutdown already dropped _proc it cannot see this child; reap and escalate like
            # _shutdown_subprocess_locked.
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
        managed = getattr(self, "_managed_engine", None)
        if managed is not None:
            return managed.alive()
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
            managed = getattr(self, "_managed_engine", None)
            if managed is not None:
                if not managed.stop():
                    return False
                self._managed_engine = None
                self.active_model_name = None
                self.models.clear()
            return self._shutdown_subprocess_locked(timeout)

    def _shutdown_subprocess_locked(self, timeout: float) -> bool:
        """Gracefully shut down the inference subprocess. Returns True only once the worker is
        confirmed dead. If it survives terminate/kill (e.g. wedged in an uninterruptible CUDA
        syscall that outlives SIGKILL) the live handle is KEPT, not nulled, so is_worker_alive()
        and the pre-swap liveness guard can still observe the survivor instead of a cleared
        handle and refuse the destructive sidecar swap."""
        self._stop_dispatcher()
        if self._proc is None or not self._proc.is_alive():
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
            # Survived SIGKILL (uninterruptible syscall): keep the handle so guards see a live worker.
            logger.error(
                "Inference subprocess still alive after terminate/kill; "
                "preserving its handle for the pre-swap liveness check"
            )
            return False

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
            # Stable low samples can precede delayed driver reclaim; trust stability only after a release.
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
            if self._shutdown_subprocess_locked(5):
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
            self._log_worker_stderr_once(None, None)
            return f"{message} Details: process missing."

        exitcode = proc.exitcode
        pid = proc.pid
        if exitcode is None:
            # NOT terminal: the capture belongs to the live replacement.
            self._log_worker_stderr_once(pid, None, worker_exited = False)
            return f"{message} Details: pid={pid}."

        tail = self._public_worker_stderr_tail() if with_worker_output else ""
        details = f"\n\nWorker error output:\n{tail}" if tail else ""
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
        # Local import: resolving this pulls unsloth_zoo and torch via the shim.
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

            if rtype == "downloads":
                self._claim_load_downloads(resp)
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
        # Latch the subprocess: if a fresh worker replaces it, bail instead of deadlocking under _gen_lock.
        initial_proc = self._proc
        initial_resp_queue = self._resp_queue
        reading_on_until = None
        stop_recorded = False
        stop_sent = False
        while True:
            if self._proc is not initial_proc or self._resp_queue is not initial_resp_queue:
                if stop_sent:
                    return
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

            # Sole consumer of the response queue; never let routing kill the dispatcher.
            try:
                rid = resp.get("request_id")
                rtype = resp.get("type", "")

                if rtype == "status":
                    logger.info("Subprocess status: %s", resp.get("message", ""))
                    continue

                delivered = False
                if rid:
                    with self._mailbox_lock:
                        mbox = self._mailboxes.get(rid) or self._direct_mailboxes.get(rid)
                        owner = self._request_cancel_events.get(rid)
                    if mbox is not None:
                        # Retire in worker order, else a late Stop cancels whichever request started next.
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
        if expected_model is None:
            expected_model = self.active_model_name

        # Switch in flight; this path bypasses _gen_lock, so bail instead of hitting the outgoing model.
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

    # Bumped at publish, not load start, so a same-model reload mid-install is detected.
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
        mlx_int8_prefill: bool = False,
        chat_template_override: Optional[str] = None,
        load_cancel_event: Optional[threading.Event] = None,
        post_handoff_expected_free_gb: Optional[dict[int, float]] = None,
        audio_device: Optional[str] = None,
        on_prior_worker_released: Optional[Callable[[], None]] = None,
        cache_environment: Optional[Mapping[str, str]] = None,
        anonymous_hf_access: bool = False,
        audio_codec_path: Optional[str] = None,
        engine: str = "auto",
        engine_options = None,
        n_parallel: Optional[int] = None,
    ) -> bool:
        """Load a model for inference. Always spawns a fresh subprocess per load for a clean
        interpreter (no stale unsloth patches, torch.compile caches, or getsource failures)."""
        if engine != "auto":
            return self._load_managed_engine(
                engine,
                config,
                max_seq_length,
                gpu_ids,
                hf_token,
                load_cancel_event,
                cache_environment,
                anonymous_hf_access,
                engine_options,
                trust_remote_code,
                approved_remote_code_fingerprint,
                subject,
            )
        if getattr(self, "_managed_engine", None) is not None:
            if not self._shutdown_subprocess():
                raise RuntimeError("Previous inference engine has not stopped")
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
                "mlx_int8_prefill": bool(mlx_int8_prefill),
                "chat_template_override": chat_template_override,
                "audio_device": audio_device,
            }
            if anonymous_hf_access:
                sub_config["anonymous_hf_access"] = True
            if audio_codec_path is not None:
                sub_config["audio_codec_path"] = audio_codec_path
            audio_cpp_model = getattr(config, "audio_cpp", None) is not None
            if audio_cpp_model:
                sub_config["audio_cpp"] = True
                if audio_load_runs_on_cpu(getattr(config, "audio_type", None), audio_device):
                    audio_device = "cpu"
                    sub_config["audio_device"] = audio_device
            if audio_device_forces_cpu(audio_device) and (
                audio_cpp_model or is_native_audio_model(model_name)
            ):
                # No card for CPU audio: multiple GPUs are rejected and the settle wait raises on a busy card.
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
            sub_config["device_backend"] = get_device().value

            if load_cancel_event is not None and load_cancel_event.is_set():
                self.loading_models.discard(model_name)
                logger.info("Load cancelled before worker teardown: %s", model_name)
                return False

            # Recheck the sidecar reservation before teardown (repairs only), keeping the current model.
            from utils.transformers_version import (
                SidecarSwapInProgress,
                sidecar_swap_kind,
            )

            if sidecar_swap_kind() == "repair":
                raise SidecarSwapInProgress(
                    "A transformers repair is replacing the latest sidecar; "
                    "retry when it completes."
                )

            # Always spawn fresh: unsloth patches torch internals, breaking getsource on reuse.
            had_worker_handle = self._proc is not None
            worker_shutdown_at = 0.0
            if self._ensure_subprocess_alive():
                self._cancel_generation()
                time.sleep(0.3)
                if self._shutdown_subprocess() is False:
                    # Worker survived kill (wedged CUDA syscall); do not spawn a second over its GPU memory.
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

                # cancel_load may have no-oped while _proc was None; recheck and tear down before publishing.
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
                    # Under the shutdown lock so a "loaded" reply cannot be published for a killed worker.
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
                        # Fresh subprocess holds only this model; a stale name would unload the wrong one.
                        self.models = {}
                        self.models[self.active_model_name] = _mirrored_model_entry(
                            model_info, model_name
                        )
                        self.models[self.active_model_name]["parallel_slots"] = parallel_slots
                        self.models[self.active_model_name]["can_batch"] = model_info.get(
                            "can_batch"
                        )
                        # Lets the loaded shortcut tell a CPU request from a GPU model. Native audio only:
                        # marking anything else tells training a GPU model holds no VRAM.
                        _audio_type = model_info.get("audio_type")
                        self.models[self.active_model_name]["audio_cpu"] = (
                            _audio_type in NATIVE_AUDIO_TYPES
                            and audio_load_runs_on_cpu(_audio_type, audio_device)
                        )
                        self.models[self.active_model_name].update(
                            _mlx_runtime_mirror_fields(model_info)
                        )
                        _tpl_info = model_info.get("chat_template_info")
                        if isinstance(_tpl_info, dict):
                            self.models[self.active_model_name]["chat_template_info"] = _tpl_info
                    self.loading_models.discard(model_name)
                    logger.info("Model '%s' loaded successfully in subprocess", model_name)
                    return True
                else:
                    error = resp.get("message") or resp.get("error") or "Failed to load model"
                    self.active_model_name = None
                    self.models.clear()
                    raise Exception(error)

        except Exception as exc:
            self.loading_models.discard(model_name)
            from utils.transformers_version import SidecarSwapInProgress

            if isinstance(exc, SidecarSwapInProgress) and self._ensure_subprocess_alive():
                # Old worker still live: keep the mirrors so the installer does not kill it unreported.
                raise
            self.active_model_name = None
            self.models.clear()
            try:
                self._shutdown_subprocess(timeout = 5)
            except Exception as teardown_exc:
                logger.warning("Could not shut the failed load's worker down: %s", teardown_exc)
            raise
        finally:
            self._release_load_downloads()

    def _claim_load_downloads(self, resp: dict) -> None:
        from hub.services.load_downloads import claim_load_downloads
        self._release_load_downloads()
        try:
            self._load_download_keys = claim_load_downloads(
                resp.get("repo_ids") or [],
                xet_disabled = bool(resp.get("xet_disabled")),
                hub_cache = resp.get("hub_cache"),
            )
        except Exception as exc:
            logger.warning("Could not register the load's downloads: %s", exc)

    def _release_load_downloads(self) -> None:
        keys, self._load_download_keys = self._load_download_keys, []
        if not keys:
            return
        from hub.services.load_downloads import release_load_downloads

        try:
            release_load_downloads(keys)
        except Exception as exc:
            logger.warning("Could not release the load's downloads: %s", exc)

    def reap_dead_managed_engine(self) -> bool:
        """True when a crashed engine was cleared, so the caller can drop its residency."""
        with self._subprocess_shutdown_lock:
            managed = getattr(self, "_managed_engine", None)
            settled = self.active_model_name or not self.loading_models
            if managed is not None and settled and not managed.alive():
                self._shutdown_subprocess()
                return getattr(self, "_managed_engine", None) is None
            return False

    def _load_managed_engine(
        self,
        engine,
        config,
        context,
        gpu_ids,
        hf_token,
        cancel,
        cache_environment,
        anonymous,
        options = None,
        trust_remote_code = False,
        approved_remote_code_fingerprint = None,
        subject = None,
    ):
        from types import SimpleNamespace

        from core.inference.managed_engine import ManagedEngine
        from utils.hf_cache_settings import get_hf_cache_paths
        from hub.utils.hf_tokens import apply_token_to_child_env

        if not self._shutdown_subprocess():
            raise RuntimeError("Previous inference process has not stopped")
        model = config.identifier
        with self._subprocess_shutdown_lock:
            self.active_model_name = None
            self.models.clear()
            self.loading_models.add(model)
            managed = ManagedEngine(engine)
            self._managed_engine = managed
        try:
            # The engine loads the checkpoint itself, so the worker's malware and consent gates run here.
            from core.inference.worker import _run_security_gates

            replies = []
            if not _run_security_gates(
                [config.path if getattr(config, "is_local", False) else model],
                trust_remote_code = bool(trust_remote_code),
                hf_token = None if anonymous else hf_token,
                approved_fingerprint = approved_remote_code_fingerprint,
                resp_queue = SimpleNamespace(put = replies.append),
                compute_subdirs = False,
                subject = subject,
            ):
                raise RuntimeError(
                    (replies[-1].get("message") if replies else None)
                    or "The model was blocked by the security scan."
                )
            env = get_hf_cache_paths().child_env()
            if cache_environment:
                env.update(cache_environment)
            apply_token_to_child_env(env, False if anonymous else hf_token)
            managed.start(
                model,
                context,
                gpu_ids,
                env,
                cancel,
                options,
                trust_remote_code,
                # Validation read config.path; a WSL drive path only resolves in that form.
                model_path = config.path if config.is_local else None,
            )
            with self._subprocess_shutdown_lock:
                if (
                    self._managed_engine is not managed
                    or model not in self.loading_models
                    or (cancel is not None and cancel.is_set())
                    or not managed.alive()
                ):
                    raise RuntimeError("Model load cancelled")
                self.models[model] = {
                    "engine": engine,
                    "engine_parallelism": (options or {}).get("parallelism", "tensor"),
                    "engine_precision": (options or {}).get("precision", "auto"),
                    "is_vision": (options or {}).get("is_vision", config.is_vision),
                    "chat_template_info": {
                        "accepts_multiple_images": bool(
                            (options or {}).get("is_vision", config.is_vision)
                        )
                    },
                    "is_audio": False,
                    "is_lora": False,
                    "context_length": managed.context,
                    "max_context_length": managed.context,
                    "context_length_enforced": True,
                    "requested_context_length": context,
                    "max_seq_length_requested": context,
                    "load_in_4bit_requested": False,
                    "gpu_ids_requested": gpu_ids,
                    "gpu_ids": list(gpu_ids or [0]),
                    "tensor_parallel": len(gpu_ids or [0]) > 1
                    and (options or {}).get("parallelism", "tensor") == "tensor",
                    "supports_tools": bool((options or {}).get("tool_parser")),
                }
                self.active_model_name = model
                self.load_generation += 1
                return True
        except Exception:
            self._shutdown_subprocess()
            raise
        finally:
            self.loading_models.discard(model)

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
        # Clear the marker BEFORE teardown, or an off-gate load_model passes its recheck and loads.
        self.loading_models.discard(target)
        self.active_model_name = None
        self.models.clear()
        managed = getattr(self, "_managed_engine", None) is not None
        stopped = self._shutdown_subprocess(timeout = 0.5)
        # Clear again AFTER teardown: a racing load may consume a queued "loaded" and repopulate.
        self.active_model_name = None
        self.models.clear()
        if managed and stopped is False:
            raise RuntimeError("The inference engine did not stop.")
        return True

    def load_stt_model(
        self,
        model: Optional[str],
        engine: str,
        request_cancel_event: Optional[threading.Event] = None,
        device: Optional[str] = None,
        **options,
    ) -> None:
        """Make a dictation model resident on its sidecar. ``device`` is the user's audio device
        preference (``auto``/``cpu``/``gpu``)."""
        from core.inference import stt_registry
        stt_registry.load(model, engine, request_cancel_event, device = device, **options)

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
        # Load path canonicalizes casing, so match the /unload name case-insensitively.
        if (
            self.active_model_name is not None
            and model_name != self.active_model_name
            and model_name.lower() == self.active_model_name.lower()
        ):
            model_name = self.active_model_name
        if self.cancel_load(model_name):
            return True

        managed = getattr(self, "_managed_engine", None)
        if managed is not None:
            if model_name != self.active_model_name:
                return True
            return self._shutdown_subprocess()
        if not self._ensure_subprocess_alive():
            self.models.pop(model_name, None)
            if self.active_model_name == model_name:
                self.active_model_name = None
            return True

        # Worker unloads its active model when the name is absent, so refuse stale names.
        if model_name != self.active_model_name and model_name not in self.models:
            self.models.pop(model_name, None)
            return True

        with self._dispatcher_lifecycle_lock:
            self._unload_pending = True
        # The worker clears cancel_event per generate; drain_event (never cleared) skips queued ones.
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
                # An idle worker keeps its VRAM high-water mark that no arbiter evicts, so drop it.
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
        managed = getattr(self, "_managed_engine", None)
        if managed is not None:
            return managed.count_tokens(messages, system_prompt, tools = tools)
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
                        raise RuntimeError(
                            self._subprocess_crash_message(
                                "count", with_worker_output = self._owns_worker(None)
                            )
                        )
                    continue
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

    def compact_chat_context(
        self,
        messages: list,
        *,
        system_prompt: str = "",
        tools: Optional[list] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thread_id: Optional[str] = None,
        cancel_event = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        continue_final_message: bool = False,
        tool_loop: bool = False,
        recall_reachable: bool = False,
        anchor_ids = None,
        replay_boundary: bool = True,
        recall_done: bool = False,
        request_branch: Optional[list] = None,
        live_branch: Optional[list] = None,
    ) -> dict:
        """Fit one MLX prompt into the served window under the policy GGUF uses.

        A ``tool_loop`` turn carrying tools may reset the epoch; a plain turn only when a later
        one can search (``recall_reachable``). ``request_branch`` is the client's transcript and
        ``live_branch`` that plus the loop's replies and tool results; both default to the prompt.
        Never raises: a failed fit returns the request unchanged.
        """
        unchanged = {
            "messages": list(messages),
            "system_prompt": system_prompt,
            "events": [],
            "recalled": False,
            "anchored": [],
        }
        model_info = self.models.get(self.active_model_name) or {}
        context_length = int(model_info.get("context_length") or 0)
        if (
            context_overflow != "truncate_oldest"
            or not model_info.get("is_mlx")
            or context_length <= 1
        ):
            return unchanged

        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend(messages)

        try:
            from core.inference.chat_template_helpers import trailing_assistant_resume_kind
            from core.inference.context_window import (
                messages_without_unpriced_media,
                retrieval_budget,
            )
            from core.inference.llama_cpp import (
                _archive_and_recall,
                _boundary_metadata,
                _can_reset_epoch,
                _compaction_fit_kwargs,
                _conversation_recall_reserve,
                _fit_with_instruction_pins,
                _keeps_compaction_boundary,
                _memory_tool_withheld,
                _records_boundary,
                _sticky_compaction_state,
            )

            if messages_without_unpriced_media(conversation) is not conversation:
                return unchanged
            if cancel_event is not None and cancel_event.is_set():
                return unchanged
            if continue_final_message and trailing_assistant_resume_kind(conversation):
                return unchanged

            request_branch = request_branch or conversation
            calls_tools = tool_loop and bool(tools)
            recall_offered = tool_loop and any(
                isinstance(tool, dict)
                and (tool.get("function") or {}).get("name") == "search_conversation"
                for tool in tools or ()
            )

            def _count(fitted):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Context compaction cancelled")
                return self.count_chat_tokens(
                    fitted,
                    "",
                    tools = tools,
                    enable_thinking = enable_thinking,
                    reasoning_effort = reasoning_effort,
                    preserve_thinking = preserve_thinking,
                )[0]

            can_reset = _can_reset_epoch(
                thread_id,
                calls_tools if tool_loop else recall_reachable,
                # A plain turn carries no catalogue to read; the route answered for it.
                tools_withheld = tool_loop and _memory_tool_withheld(thread_id, tools),
            )
            sticky, sticky_is_checkpoint = (
                _sticky_compaction_state(
                    thread_id,
                    request_branch,
                    context_policy = context_policy,
                    can_reset = can_reset,
                    compaction_headroom_ratio = compaction_headroom_ratio,
                )
                if replay_boundary
                else (0, False)
            )
            fitted, truncation = _fit_with_instruction_pins(
                conversation,
                context_length = context_length,
                max_tokens = max_tokens,
                count_tokens = _count,
                anchor_ids = anchor_ids,
                keeps_boundary = _keeps_compaction_boundary(thread_id),
                can_reset = can_reset,
                recall_offered = recall_offered,
                reserve_tokens = _conversation_recall_reserve(thread_id),
                sticky_dropped = sticky,
                sticky_is_checkpoint = sticky_is_checkpoint,
                **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
            )
            if not truncation:
                return {**unchanged, "boundary_applied": True}

            recall = _archive_and_recall(
                fitted,
                conversation,
                branch_messages = live_branch or conversation,
                thread_id = thread_id,
                # A forged tool exchange is only safe when the request advertises the tool.
                style = "tool" if recall_offered else "inline",
                force_recall = bool(truncation.get("checkpoint_started", True)),
                recall_done = recall_done or not truncation["fits"],
                recall_budget_tokens = retrieval_budget(
                    context_length,
                    max_tokens,
                    truncation.get("prompt_tokens_after") or 0,
                    reply_returns = calls_tools,
                ),
                count_tokens = _count,
            )
            fitted = recall["conversation"]
            truncation = {**truncation, **recall["counts"]}
            if _records_boundary(truncation):
                truncation = {
                    **truncation,
                    **_boundary_metadata(fitted, request_branch, compaction_headroom_ratio),
                }
            return {
                "messages": fitted,
                "system_prompt": "",
                "events": [*recall["events"], {"type": "context_truncated", **truncation}],
                "recalled": bool(recall["recalled"]),
                "anchored": list(recall["anchored"]),
                "boundary_applied": True,
            }
        except Exception as exc:
            logger.warning("Could not preflight the MLX context window: %s", exc)
            return unchanged

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
        sandbox_level: Optional[str] = None,
        use_adapter: Optional[Union[bool, str]] = None,
        stats_holder: Optional[dict] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list] = None,
        reasoning_prefilled: bool = False,
        seed: Optional[int] = None,
        caller_image_indexes: "tuple[int, ...]" = (),
        mcp_image = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        **_unused,
    ):
        """Run the safetensors agentic tool loop in the parent process, calling the worker for each
        turn. Yields the same event dicts as the GGUF tool loop so the route layer can stream
        both backends through one helper."""
        from core.inference.safetensors_agentic import run_safetensors_tool_loop
        from core.inference.tools import execute_tool

        max_new_tokens = max_tokens if max_tokens and max_tokens > 0 else None
        loop_images: Optional[list] = (
            list(images or [])
            if self.models.get(self.active_model_name, {}).get("is_vision")
            else None
        )

        # Latest turn only; cleared on entry so a failed turn cannot leak an earlier count.
        turn_stats: dict = {}

        def _single_turn(
            conv: list,
            *,
            active_tools: Optional[list[dict]] = None,
            tool_protocol_active: Optional[bool] = None,
        ):
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
                continue_final_message = continue_final_message,
                tool_protocol_active = tool_protocol_active,
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
                if stats_holder is not None:
                    stats_holder["stats"] = _summed_tool_loop_stats(
                        stats_holder.get("stats"), turn_stats.get("stats")
                    )

        initial = list(messages)
        if system_prompt:
            initial = [{"role": "system", "content": system_prompt}] + initial

        # Same profile as the renderer, and a catalog safe under every possible template.
        from core.inference.chat_template_helpers import (
            mapped_chat_template,
            markup_for_tokenizer,
            renderable_tool_catalog,
        )

        _model_info = self.models.get(self.active_model_name) or {}
        # Resolved BEFORE the profile: the mapper installs its template during the render.
        _mapped_tpl = mapped_chat_template(_model_info, self.active_model_name)

        _request_branch = list(initial)
        _request_ids = {id(message) for message in initial}
        _sticky_boundary_applied = False
        _conversation_recall_done = False
        _rolling_anchor_ids: set[int] = set()
        for message in reversed(initial):
            if message.get("role") == "user":
                _rolling_anchor_ids.add(id(message))
                break

        def _fit_iteration(conversation: list, active_tools: list, live_branch: list) -> dict:
            nonlocal _sticky_boundary_applied, _conversation_recall_done
            if loop_images:
                return {}
            # Pin the newest tool result and its user turn for this fit only, or results fill the window.
            pinned = _rolling_anchor_ids
            for index in range(len(conversation) - 1, -1, -1):
                message = conversation[index]
                if message.get("role") == "tool" and id(message) not in _request_ids | pinned:
                    asked = [id(m) for m in conversation[:index] if m.get("role") == "user"]
                    pinned = pinned | {id(message), *asked[-1:]}
                    break
            result = self.compact_chat_context(
                conversation,
                tools = active_tools,
                context_overflow = context_overflow,
                context_policy = context_policy,
                compaction_headroom_ratio = compaction_headroom_ratio,
                max_tokens = max_new_tokens,
                thread_id = thread_id,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                tool_loop = True,
                anchor_ids = pinned,
                replay_boundary = not _sticky_boundary_applied,
                recall_done = _conversation_recall_done,
                request_branch = _request_branch,
                live_branch = live_branch,
            )
            # The saved boundary applies once, in the first fit that runs (not resumed or failed).
            if result.get("boundary_applied"):
                _sticky_boundary_applied = True
            if result.get("recalled"):
                _conversation_recall_done = True
                for message in result.get("anchored") or ():
                    _rolling_anchor_ids.add(id(message))
            return result

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
            mcp_image = mcp_image,
            bypass_permissions = bypass_permissions,
            permission_mode = permission_mode,
            sandbox_level = sandbox_level,
            reasoning_prefilled = reasoning_prefilled,
            continue_final_message = continue_final_message,
            context_length = _model_info.get("context_length"),
            max_tokens = max_new_tokens,
            generation_stats_holder = turn_stats,
            images_sink = loop_images,
            caller_image_indexes = tuple(caller_image_indexes) if loop_images else (),
            context_fitter = _fit_iteration,
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
        if getattr(self, "_managed_engine", None) is not None:
            raise GenStreamErrorRaised(
                "Adapter comparisons are unavailable for this engine.", public = True
            )
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
        managed = getattr(self, "_managed_engine", None)
        if managed is not None:
            kwargs = dict(locals())
            for key in ("self", "managed", "kwargs"):
                kwargs.pop(key, None)
            from .engine_transport import EngineHTTPError

            try:
                cumulative = ""
                for delta in managed.generate(**kwargs):
                    cumulative += delta
                    yield cumulative
            except EngineHTTPError:
                raise
            except Exception as exc:
                yield GenStreamError(str(exc), public = True)
            return
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
            # The worker takes commands in order, so the oldest claim is the executor.
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

    def separate_audio_response(
        self,
        source_path: str,
        output_dir: str,
        audio_options: Optional[dict] = None,
        cancel_event = None,
    ) -> list[dict]:
        """Split a prepared 44.1 kHz WAV into stems under ``output_dir``; the full token budget
        gives the watchdog the same hour the runtime request has."""
        outputs, _sample_rate = self.generate_audio_response(
            text = "",
            max_new_tokens = AUDIO_GENERATION_MAX_TOKENS,
            cancel_event = cancel_event,
            audio_options = audio_options,
            workflow = "separate",
            audio_inputs = {"source": source_path},
            output_dir = output_dir,
        )
        return outputs

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
        audio_options: Optional[dict] = None,
        workflow: Optional[str] = None,
        audio_inputs: Optional[dict[str, str]] = None,
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
        convert: Optional[dict] = None,
        edit: Optional[dict] = None,
        music: Optional[dict] = None,
        output_dir: Optional[str] = None,
        stats_holder: Optional[dict] = None,
    ) -> Tuple[bytes, int]:
        """Generate TTS audio. Returns (wav_bytes, sample_rate). Blocking: sends the command and
        waits for the full audio response. ``audio_inputs`` maps a role (reference, emotion,
        source, target) to a server-local WAV path; audio bytes never cross the queue. A separation
        (``output_dir`` set) returns (outputs, sample_rate) instead, see ``separate_audio_response``."""
        if not self._ensure_subprocess_alive():
            raise RuntimeError("Inference subprocess is not running")
        if not self.active_model_name:
            raise RuntimeError("No active model")
        expected_model = self.active_model_name

        # Reserve dispatcher admission under _gen_lock; a bare idle wait races compare requests.
        with self._gen_lock:
            with self._reserve_worker("audio generation is in progress"):
                idle = self._wait_worker_idle(cancel_event = cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise AudioGenerationCancelledError("Audio generation cancelled")
                if not idle:
                    raise RuntimeError(
                        "Cannot start audio generation while a reply is still generating"
                    )

                if self._unload_pending or self.active_model_name != expected_model:
                    raise AudioGenerationCancelledError("model is being unloaded")

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
                if audio_options:
                    cmd["audio_options"] = dict(audio_options)
                if workflow is not None:
                    cmd["workflow"] = workflow
                if audio_inputs is not None:
                    cmd["audio_inputs"] = {str(k): str(v) for k, v in audio_inputs.items()}
                if reference_text is not None:
                    cmd["reference_text"] = reference_text
                if speed is not None:
                    cmd["speed"] = float(speed)
                if convert is not None:
                    cmd["convert"] = dict(convert)
                if edit is not None:
                    cmd["edit"] = dict(edit)
                if music is not None:
                    cmd["music"] = dict(music)
                    try:
                        music_wait = float(music.get("timeout_s") or 0.0)
                    except (TypeError, ValueError):
                        music_wait = 0.0
                    # Outlast the worker's wait so its error, not the watchdog, reaches the caller.
                    generation_timeout = max(generation_timeout, music_wait + 60.0)
                if output_dir is not None:
                    cmd["output_dir"] = str(output_dir)

                read_one, _drain, release_mailbox = self._direct_reader(request_id, cancel_event)
                try:
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
                            self._cancel_generation()
                            cancel_signalled = True
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

                        if rtype in ("audio_done", "audio_error") and isinstance(
                            resp.get("audio_runtime"), dict
                        ):
                            entry = self.models.get(expected_model)
                            if entry is not None:
                                entry.update(resp["audio_runtime"])

                        if rtype == "audio_done":
                            if cancel_event is not None and cancel_event.is_set():
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            if resp.get("outputs") is not None:
                                return resp["outputs"], int(resp.get("sample_rate") or 0)
                            wav_bytes = base64.b64decode(resp["wav_base64"])
                            sample_rate = resp["sample_rate"]
                            status_patch = resp.get("status_patch")
                            if isinstance(status_patch, dict):
                                live = self.models.get(expected_model)
                                if live is not None and "audio_music" in status_patch:
                                    live["audio_music"] = status_patch["audio_music"]
                            if stats_holder is not None:
                                stats_holder["stats"] = resp.get("stats")
                            return wav_bytes, sample_rate

                        if rtype == "audio_error":
                            if resp.get("cancelled") or (
                                cancel_event is not None and cancel_event.is_set()
                            ):
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            if resp.get("code") == AUDIO_UNSUPPORTED_CODE:
                                raise AudioBackendUnsupportedError(
                                    resp.get("error", "This backend cannot generate audio."),
                                    hint = resp.get("hint"),
                                )
                            if resp.get("code") == AUDIO_RUNTIME_ERROR_CODE:
                                raise AudioRuntimeError(
                                    resp.get("error", "Audio generation failed"),
                                    status = resp.get("status"),
                                )
                            raise RuntimeError(resp.get("error", "Audio generation failed"))

                        if rtype == "error":
                            if cancel_event is not None and cancel_event.is_set():
                                raise AudioGenerationCancelledError("Audio generation cancelled")
                            raise RuntimeError(resp.get("error", "Unknown error"))

                        if rtype == "status":
                            continue

                    if cancel_deadline is not None:
                        if self._shutdown_subprocess(timeout = _AUDIO_CANCEL_DRAIN_TIMEOUT):
                            self.active_model_name = None
                            self.models.clear()
                        raise AudioGenerationCancelledError("Audio generation cancelled")

                    # Keep ownership until the command acknowledges; tear down a worker that does not.
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
        use_adapter: Optional[Union[bool, str]] = None,
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
            use_adapter = use_adapter,
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
            audio_type = None,
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
            if self._unload_pending or self.active_model_name != expected_model:
                yield GenStreamError("Error: model is being unloaded", public = True)
                return
            if cancel_event is not None and cancel_event.is_set():
                return
            request_id = str(uuid.uuid4())

            import numpy as np

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
                    # Claim under the send lock, else stopping a queued compare request kills this one.
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
# Lazy build is slow and called from executor threads; lock to avoid duplicate orchestrators.
_inference_backend_lock = threading.Lock()


routed_slot: contextvars.ContextVar = contextvars.ContextVar("routed_slot", default = None)


def peek_inference_backend() -> Optional["InferenceOrchestrator"]:
    """The orchestrator if one exists, else None. Never constructs one. For callers that only
    describe what is already loaded: constructing reaches get_default_models() -> get_device(),
    which blocks on the torch import during the warm."""
    slot = routed_slot.get()
    return slot.orchestrator if slot is not None else _inference_backend


def get_inference_backend() -> InferenceOrchestrator:
    """Global inference backend instance (orchestrator)."""
    global _inference_backend
    slot = routed_slot.get()
    if slot is not None:
        return slot.orchestrator
    if _inference_backend is None:
        with _inference_backend_lock:
            if _inference_backend is None:
                _inference_backend = InferenceOrchestrator()
    return _inference_backend
