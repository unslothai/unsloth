# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions through its
OpenAI-compatible /v1/chat/completions endpoint.
"""

import ast
import atexit
import contextlib
import functools
import json
import logging
import math
import os
import re
import struct
from loggers import get_logger
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    NoReturn,
    Optional,
    Union,
)

import httpx

from core.inference.llama_server_args import (
    _LAYER_OFFLOAD_FLAGS,
    _effective_tensor_parallel,
    _flag_name,
    _tensor_parallel_matches_loaded,
    extra_args_disable_mmproj,
    parse_cache_override,
    parse_cache_override_per_axis,
    parse_ctx_override,
    parse_gpu_layers_override,
    parse_split_mode_override,
    resolve_requested_ctx,
    strip_shadowing_flags,
    strip_split_mode_only,
)

# Share strip / signal constants with the multi-format parser so BUFFERING also
# catches Llama-3 / Mistral / Gemma 4 (legacy helper only knew <tool_call> / <function=).
from core.inference.tool_call_parser import (
    _GEMMA_BARE_TC_PREFIX_RE,
    _GEMMA_BARE_TC_RE,
    _TOOL_ALL_PATS as _PARSER_TOOL_ALL_PATS,
    _TOOL_CLOSED_PATS as _PARSER_TOOL_CLOSED_PATS,
    _balanced_brace_end,
    _strip_function_xml_calls,
    _strip_gemma_wrapperless_calls,
    _strip_glm_calls,
    _strip_mistral_closed_calls,
    TOOL_XML_SIGNALS as _SHARED_TOOL_XML_SIGNALS,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    parse_tool_calls_from_text as _shared_parse_tool_calls_from_text,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup as _shared_strip_tool_markup,
)

# The healer owns the bracket-tag + rehearsal strip helpers and their name-gated
# pattern lists, so the GGUF streaming strip stays aligned with the parser.
from core.tool_healing import (
    _REHEARSAL_TAIL_STRIP_RE,
    _strip_bracket_tag_calls,
    apply_tool_strip_patterns,
    strip_outside_think,
)
from utils.native_path_leases import child_env_without_native_path_secret
from utils.child_stdio import utf8_child_env
from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)
from utils.process_lifetime import child_popen_kwargs as _child_popen_kwargs
from core.inference.tool_call_parser import (
    MAX_ACT_REPROMPTS as _MAX_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS as _NUDGE_TOOL_CALLS_STATUS,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
    is_reprompt_repeat as _is_reprompt_repeat,
    is_reprompt_restatement as _is_reprompt_restatement,
    is_short_intent_without_action as _is_short_intent_without_action,
    reprompt_to_act_message as _reprompt_to_act_message,
)
from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    awaiting_approval_status,
    tool_event_provenance,
)
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)

logger = get_logger(__name__)


class LlamaServerNotFoundError(RuntimeError):
    """GGUF model needs the llama.cpp runtime but no llama-server is installed.
    Subclasses RuntimeError so existing handlers still catch it."""


class _LlamaStreamCancelled(Exception):
    """Internal signal for an expected client/request cancellation."""


@dataclass(frozen = True)
class GgufLoadIntent:
    """Immutable caller intent replayed by retries and recovery."""

    model_identifier: str
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    mtp_draft_path: Optional[str] = None
    dspark_draft_path: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_variant: Optional[str] = None
    hf_token: Optional[str] = None
    is_vision: bool = False
    n_ctx: int = 4096
    chat_template_override: Optional[str] = None
    cache_type_kv: Optional[str] = None
    speculative_type: Optional[str] = None
    spec_draft_n_max: Optional[int] = None
    tensor_parallel: bool = False
    gpu_memory_mode: Literal["auto", "manual"] = "auto"
    gpu_layers: int = -1
    n_cpu_moe: int = 0
    tensor_split: Optional[tuple[float, ...]] = None
    gpu_ids: Optional[tuple[int, ...]] = None
    gpu_ids_are_vulkan_ordinals: Optional[bool] = None
    n_threads: Optional[int] = None
    n_parallel: int = 1
    extra_args: Optional[tuple[str, ...]] = None
    # The route materialises inherited extras from the live server, so the list alone
    # cannot say whether the caller named it. Duplicate-load checks need the difference:
    # a launch-time rewrite makes the launched and requested lists diverge.
    extra_args_inherited: bool = False
    preserve_multi_gpu_on_layer: bool = False
    compare_mtp_draft: bool = False
    # Internal state that keeps recovered auto-Vulkan respawns CPU-isolated.
    cpu_fallback: bool = False

    def __post_init__(self):
        for key in ("tensor_split", "gpu_ids", "extra_args"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, tuple(value))
        if self.gpu_ids == ():
            object.__setattr__(self, "gpu_ids", None)


# Shared so the from_identifier preflight and the load-time raise stay in sync.
LLAMA_SERVER_NOT_FOUND_DETAIL = (
    "This is a GGUF model, but the llama.cpp runtime (llama-server) is not "
    "installed. Run `unsloth studio setup` to download the prebuilt runtime, "
    "then try again. (Advanced: set LLAMA_SERVER_PATH to an existing binary.)"
)

# Shared by the pre-teardown and post-metadata rejections (#7205).
# Every env var that can put a drafter or spec type into the child. Pre-b8955
# spellings included: the new names arrived 2026-04-28 and the launchable floor is
# 2025-08-30, so a build in that window reads the old ones and would keep the drafter.
_SPEC_ENV_VARS = (
    "LLAMA_ARG_SPEC_DRAFT_MODEL",
    "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
    "LLAMA_ARG_SPEC_TYPE",
    "LLAMA_ARG_MODEL_DRAFT",
    "LLAMA_ARG_HFD_REPO",
)

_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR = (
    "This Mac's Metal device is virtualised, where offloaded layers can return "
    "corrupt output, and the installed unsloth_zoo diffusion shim has no --ngl, "
    "so the zero-layer CPU pin cannot be applied and the runner would use Metal "
    "anyway. Run 'unsloth studio update' to get a shim with --ngl, or set "
    "UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU on a VM you have verified."
)

_VULKAN_DIFFUSION_GPU_IDS_ERROR = (
    "GPU selection (gpu_ids) is not supported for a DiffusionGemma "
    "GGUF on a Vulkan llama.cpp build: the diffusion runner selects "
    "its device by CUDA physical index, which has no defined mapping "
    "to ggml Vulkan device ordinals. Omit gpu_ids to use the default "
    "device."
)


# llama-server can serve HTTP 200 while running a model entirely on CPU when a
# GPU backend fails to init (#5807 / #5106 / #5830). Classify the startup log so
# Unsloth can warn. Priority: explicit "offloaded N/M layers to GPU" counts
# (authoritative), then GPU "model buffer size" lines (host-pinned _Host
# excluded), then the "device_info:" device table (disconfirm only).
_GPU_OFFLOAD_MARKERS = (
    "CUDA",
    "ROCm",
    "ROCM",
    "HIP",
    "Metal",
    "Vulkan",
    "OpenCL",
    "SYCL",
    "MUSA",
    "CANN",
)
_OFFLOADED_LAYERS_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+gpu", re.IGNORECASE
)
_DEVICE_ROW_RE = re.compile(
    r"-\s*(CUDA|ROCm|ROCM|HIP|Metal|Vulkan|SYCL|OpenCL|MUSA|CANN|CPU)\w*\s*:",
    re.IGNORECASE,
)
_GPU_DEVICE_PREFIXES = (
    "cuda",
    "rocm",
    "hip",
    "metal",
    "vulkan",
    "sycl",
    "opencl",
    "musa",
    "cann",
)


def classify_gpu_offload_lines(lines: "list[str]") -> Optional[bool]:
    """True if the model landed on a GPU, False if it stayed on CPU despite GPU
    intent, None when the log has no usable signal."""
    # Counted offload is authoritative, keyed on the model with the most layers.
    # A separate MTP/draft model logs its own (much smaller) "offloaded N/M"
    # line, so decide on the largest-M line: a drafter that fits on GPU must not
    # mask a main model running on CPU. N>0 on that model is True, 0 is False.
    max_total = -1
    offloaded_at_max = 0
    for line in lines:
        match = _OFFLOADED_LAYERS_RE.search(line)
        if not match:
            continue
        offloaded, total = int(match.group(1)), int(match.group(2))
        if total > max_total or (total == max_total and offloaded > offloaded_at_max):
            max_total, offloaded_at_max = total, offloaded
    if max_total >= 0:
        return offloaded_at_max > 0

    # GPU marker on a *model* buffer; _Host buffers are CPU-pinned, not offload.
    # Buffer lines are authoritative: present but none on a GPU means CPU-only,
    # so do not let the device table below override that.
    saw_model_buffer = False
    for line in lines:
        if "model buffer size" not in line:
            continue
        saw_model_buffer = True
        if "_Host" not in line and any(m in line for m in _GPU_OFFLOAD_MARKERS):
            return True
    if saw_model_buffer:
        return False

    # device_info: lists *available* devices (printed whenever a GPU backend is
    # visible), not where the model loaded, so it can only disconfirm: an
    # all-CPU table means no usable GPU. A visible GPU device is not proof the
    # model used it, so it does not return True. Rows after the header only.
    after_header = False
    saw_device_row = False
    saw_gpu_device = False
    for line in lines:
        if "device_info:" in line:
            after_header = True
            continue
        if not after_header:
            continue
        match = _DEVICE_ROW_RE.search(line)
        if not match:
            continue
        saw_device_row = True
        if match.group(1).lower().startswith(_GPU_DEVICE_PREFIXES):
            saw_gpu_device = True
    if saw_device_row and not saw_gpu_device:
        return False
    return None


def _wsl_system_rocm_lib_dirs() -> "list[str]":
    """System ROCm lib dir(s) to load before a prebuilt's bundled HIP, on WSL.

    The bundled bare-metal HIP can't drive WSL's /dev/dxg and segfaults on the
    first GPU call; the system ROCm libs (libamdhip64 + librocdxg) can, while
    the bundle still supplies libggml-hip / librocblas (gfx1151 kernels).
    Mirrors install_llama_prebuilt._wsl_system_rocm_lib_dirs so a prebuilt that
    passed install validation runs the same at serve time. No-op off a ROCDXG
    WSL host (needs /dev/dxg, "microsoft" /proc/version, librocdxg in /opt/rocm).
    """
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except (OSError, UnicodeDecodeError):
        return []
    out: "list[str]" = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            out.append(d)
    return out


def _bundled_hip_present(binary_dir: str) -> bool:
    """True when a prebuilt bundle ships its own HIP backend library."""
    if not binary_dir:
        return False
    try:
        # Glob the version suffix (libggml-hip.so, .so.0, .so.0.11.1) the same
        # way the installer's runtime health check matches libggml-hip.so*.
        return any(Path(str(binary_dir)).glob("libggml-hip.so*"))
    except OSError:
        return False


def _native_linux_system_rocm_lib_dirs(binary_dir: str = "") -> "list[str]":
    """System ROCm lib dir(s) to prepend before a prebuilt's bundled HIP, on native Linux.

    The bundled bare-metal HIP runtime can mismatch the host amdkfd driver and crash
    in hsa_init(); prepending the whole system ROCm lib dir loads a driver-matched,
    version-consistent stack (libhsa-runtime64 / libamdhip64 / librocblas) ahead of it.
    The whole dir is deliberate: mixing the bundle's rocBLAS with a different-version
    system HIP/ROCR risks missing symbols. UNSLOTH_LLAMA_NO_SYSTEM_ROCM=1 keeps the pure
    bundle (for a host whose system ROCm lacks this arch); no-op on WSL / non-Linux.
    """
    if os.environ.get("UNSLOTH_LLAMA_NO_SYSTEM_ROCM") == "1":
        return []
    if sys.platform != "linux" or os.path.exists("/dev/dxg"):
        return []
    if not os.path.exists("/dev/kfd"):
        return []
    if not _bundled_hip_present(binary_dir):
        return []
    # Env-configured ROCm root first; /opt/rocm only as a fallback so a stale
    # /opt/rocm doesn't shadow the driver-matching install these vars point at.
    candidates = []
    for var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        val = os.environ.get(var)
        if val:
            candidates.append(val)
    candidates.append("/opt/rocm")
    out: "list[str]" = []
    seen: "set[str]" = set()
    for base in candidates:
        for lib_sub in ("lib", "lib64"):
            d = os.path.join(base, lib_sub)
            if d in seen:
                continue
            seen.add(d)
            if os.path.exists(os.path.join(d, "libhsa-runtime64.so")) or os.path.exists(
                os.path.join(d, "libhsa-runtime64.so.1")
            ):
                out.append(d)
                # ROCm keeps LLVM's versioned runtime under <root>/lib/llvm, so a
                # lib64 host still finds it under lib. Probe both and keep them
                # ahead of the bundle, else system libamd_comgr binds to the
                # bundle's incompatible libLLVM.so.*.
                for _sub in (lib_sub, "lib"):
                    llvm_lib = os.path.join(base, _sub, "llvm", "lib")
                    if llvm_lib not in seen and os.path.isdir(llvm_lib):
                        seen.add(llvm_lib)
                        out.append(llvm_lib)
    return out


# Plan-without-action re-prompt state now lives in tool_call_parser (imported above).

# Default max_tokens to the effective context when known. The floor is high
# enough for reasoning-heavy GGUFs and max_tokens-omitting API clients.
_DEFAULT_MAX_TOKENS_FLOOR = 32768
_DEFAULT_FIRST_TOKEN_TIMEOUT_S = 1200.0  # 20 min
# A transport error can arrive before the child is reapable; a request path cannot
# afford the 5s the background MTP reload spends on the same race.
_RESPAWN_REAP_GRACE_S = 1.0


def _finalize_reasoning_only_cumulative(
    cumulative: str, reasoning_text: str, finish_reason: Optional[str], promote_reasoning_only: bool
) -> str:
    """Close a live thinking block and promote it only after a clean stop.

    Local inference streams cumulative snapshots. Replacing ``<think>...`` with
    bare reasoning at EOF makes the final snapshot shorter, so suffix-based
    route consumers drop the intended fallback. Keep the snapshot append-only.
    A length-truncated thought is not a final answer, so close it without
    promotion and let the client surface the ``length`` terminal state. Raw
    consumers that do not split reasoning from visible content can disable the
    fallback to avoid returning the same reasoning twice.
    """
    visible_fallback = (
        reasoning_text if promote_reasoning_only and finish_reason != "length" else ""
    )
    return cumulative + "</think>" + visible_fallback


# Only large streamed tool payloads get an early provisional card; render_html
# is exempt because it needs immediate artifact feedback.
_PROVISIONAL_ARGS_MIN_CHARS = 256
_DEFAULT_STREAM_STALL_TIMEOUT_S = 120.0  # 2 min
# Cap tool calls from a single TEXTUAL-fallback turn (mirrors the safetensors
# loop). Structured delta.tool_calls are grammar-bounded by llama-server; text
# parsed from content is not, so one runaway turn could fan out unbounded.
_MAX_TOOL_CALLS_PER_TURN = 8
# Obligation phrasing INTENT_SIGNAL leaves alone ("I need to call ..."), paired with
# an action verb. Sentence-anchored: mid-sentence the same words are prose that names
# a tool ("The API I should invoke is foo() because ..."), and suppressing that loses
# a real answer. "should"/"must" sit outside the need|have|ought group because they
# take a bare infinitive. "invoke"/"query" stay out of the verb list: they read as
# technical prose far more often than as a stall.
_FORCED_PLAN_INTENT = re.compile(
    r"(?:^|[.!?]\s+)\s*"
    r"(?:i\s+(?:(?:need|have|ought)\s+to|should|must)|need\s+to|going\s+to|must|should)"
    r"\s+(?:\w+\s+){0,2}?(?:call|use|run|search|fetch|render)\b",
    re.I | re.M,
)
# "the answer is not in the context" announces a *missing* answer, so the negated
# forms are excluded or the plan behind them would ship as the final response.
_FINAL_ANSWER_SIGNAL = re.compile(
    r"\b(?:final\s+answer|answer\s*:|here\s+is|here's|in\s+summary|result\s*:"
    r"|(?:the\s+)?answer\s+is(?!\s+(?:not|unavailable|unknown|unclear|missing)\b))\b",
    re.I,
)
# A plan that pivots ("I should call web_search, but Tokyo is the capital") has an
# answer attached, so the turn must survive. Leaking a plan sentence is cosmetic;
# dropping an answer is not, so the doubtful case keeps the output. The pivot has to
# carry text of its own: "I should call web_search, though." answers nothing.
_ANSWER_PIVOT = re.compile(
    r"\b(?:but|however|although|though|that\s+said|in\s+the\s+meantime|meanwhile)\b"
    r"[\W_]*(?:\w+[\W_]+){1,}\w",
    re.I,
)


def _gguf_active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in (active_tools or [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


# Rehearsal NAME chars (word + hyphen, matching the parser); the lookbehind excludes the
# Mistral [CALL_ID]...[ARGS] shape.
_GGUF_REHEARSAL_ARGS_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]")


def _gguf_rehearsal_signal_pos(text: str, active_tools: list[dict]) -> int:
    """Index of the first ``NAME[ARGS]`` whose NAME is an active tool, else -1. A
    bare/inactive-name ``foo[ARGS]`` in prose is not a call; mirrors the safetensors
    ``_earliest_tool_signal`` name-gating (no unrestricted GGUF mode)."""
    active = set(_gguf_active_tool_names(active_tools))
    if not active:
        return -1
    for m in _GGUF_REHEARSAL_ARGS_RE.finditer(text):
        if m.group(1) in active:
            return m.start()
    return -1


def _gguf_has_genuine_tool_signal(text: str, signals, active_tools: list[dict]) -> bool:
    """True when ``text`` holds a genuine tool-call boundary for one of ``signals``.

    Unambiguous markers (``<tool_call>``, ``[TOOL_CALLS]``, ``<function=``) count on a
    plain substring hit; an ``[ARGS]`` hit is genuine only when an active tool name
    precedes it, so inactive-name prose is neither drained nor parsed."""
    for sig in signals:
        if sig == "[ARGS]":
            if _gguf_rehearsal_signal_pos(text, active_tools) >= 0:
                return True
            continue
        if sig in text:
            return True
    return False


_TEXT_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w.\-]+)"')
_TEXT_TOOL_GEMMA_RE = re.compile(r"\s*call:([\w.\-]+)")
_TEXT_TOOL_REHEARSAL_RE = re.compile(r"\s*([\w.\-]+)\s*\[ARGS\]")


def _sniff_text_tool_name(text: str, enabled_names: set) -> str:
    """Best-effort tool name from a partially drained TEXT tool call, gated on
    enabled names so prose can never spawn a card. Used only to open the live
    argument pane early; the authoritative parse still happens at stream end."""
    m = _TEXT_TOOL_NAME_RE.search(text[:4096])
    if m and m.group(1) in enabled_names:
        return m.group(1)
    m = _TEXT_TOOL_GEMMA_RE.match(text[:256])
    if m and m.group(1) in enabled_names:
        return m.group(1)
    m = _TEXT_TOOL_REHEARSAL_RE.match(text[:256])
    if m and m.group(1) in enabled_names:
        return m.group(1)
    return ""


def _is_rehearsal_prefix(stripped: str, active_tools: list[dict]) -> bool:
    """True if ``stripped`` is a (possibly partial) prefix of ``NAME[ARGS]`` for an
    active tool -- the bare tool name arriving in its own chunk before ``[ARGS]{...}``.
    Mirrors the safetensors loop so the split rehearsal call is not streamed."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    for name in _gguf_active_tool_names(active_tools):
        if stripped == name or f"{name}[ARGS]".startswith(stripped):
            return True
    return False


def _held_rehearsal_tail_len(text: str, active_tools: list[dict]) -> int:
    """Length of a trailing bare tool-name token that may be a split rehearsal call
    (``...web_search`` with ``[ARGS]{...}`` still to arrive), so STREAMING can hold it
    instead of leaking the name. Returns 0 for ordinary prose. Mirrors safetensors."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    return len(tail) if tail and _is_rehearsal_prefix(tail, active_tools) else 0


def _should_suppress_forced_no_tool_output(text: str, previous: str = "") -> bool:
    """Suppress only repeated forced-turn planning text, not final answers.

    ``previous`` is the stall text that triggered the nudge, so a retry that
    moved on can be told from one that just said the same thing again.
    """
    stripped = text.strip()
    if not stripped or len(stripped) >= _REPROMPT_MAX_CHARS:
        return False
    if _FINAL_ANSWER_SIGNAL.search(stripped):
        return False
    plan = _FORCED_PLAN_INTENT.search(stripped)
    if plan is not None:
        # Only the plan itself is safe to drop; anything the turn pivots to after it
        # is the answer the user is waiting for.
        return _ANSWER_PIVOT.search(stripped[plan.end() :]) is None
    if not _is_short_intent_without_action(stripped):
        return False
    # INTENT_SIGNAL also fires on lead-ins to a real answer ("Now I have the results.
    # The capital is Tokyo."), so a bare intent match is a stall only when the retry
    # adds nothing. No ``previous`` keeps the standalone "is this a stall?" contract.
    return not previous or _is_reprompt_restatement(stripped, previous)


# ── Pre-compiled patterns for GGUF shard detection ───────────
_SHARD_FULL_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


# ── Sliding-window-pattern resolver ───────────────────────────
# Resolves the per-layer SWA mask when a GGUF reports a sliding window but
# no `sliding_window_pattern` field. Tier order in `_resolve_swa_pattern`:
# GGUF metadata, on-disk cache, bootstrap dict below, transformers
# introspection, HF Hub config.json, legacy 1/4 fallback. Period N means
# layer i is SWA iff `(i + 1) % N != 0`, matching transformers. Skipped on
# purpose: phi3 (no key/val length in GGUF, window >= ctx anyway), qwen2
# family (converter strips sliding_window when use_sliding_window=False),
# mistral v0.1/v0.2 (all-SWA can't be a period).
_BOOTSTRAP_SWA_DEFAULTS: dict[str, int] = {
    "gemma2": 2,  # Gemma2Config.sliding_window_pattern
    "gemma3": 6,  # Gemma3TextConfig.sliding_window_pattern
    "gemma3n": 5,  # text_config.layer_types: SWA*4 + FULL
    "gpt_oss": 2,  # text_config.layer_types: alternating
    "cohere2": 4,  # Cohere2Config.sliding_window_pattern
}

# Process-wide cache backed by JSON on disk. Values are int period or
# list[bool] mask. Lazy-loaded.
_SWA_CACHE: Optional[dict] = None
_SWA_CACHE_LOCK = threading.Lock()


def _hf_endpoint_host() -> str:
    """Host of the configured hub endpoint (HF_ENDPOINT aware)."""
    from utils.utils import hf_endpoint_host
    return hf_endpoint_host()


def _probe_dns_dead(host: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Quick DNS check on a daemon thread, so concurrent sockets aren't
    affected by socket.setdefaulttimeout. Defaults to the configured endpoint's host."""
    from utils.utils import dns_host_dead, hf_endpoint_host
    return dns_host_dead(host or hf_endpoint_host(), timeout)


def _hf_env_offline() -> bool:
    """True when an HF offline env var is set to any truthy value (1/true/yes/on).

    Mirrors utils.models.model_config._env_offline so a user-set HF_HUB_OFFLINE=true
    (not just "1") still routes through the local-cache reuse path below.
    """
    try:
        from utils.models.model_config import _env_offline
        return _env_offline()
    except Exception:
        return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_unreachable() -> bool:
    """True when the Hub can't be reached: dead DNS, or DNS that resolves with no egress.

    DNS alone misses the common shapes (WAN down behind a live router, captive portal,
    stale cache), leaving every hub call to burn its retry backoff. The DNS shortcut stands
    down when a proxy is configured, since the proxy resolves the host.
    """
    try:
        from utils.utils import hf_dns_dead, hf_reachability_memo, hf_unreachable
    except Exception:
        return False
    # A fresh verdict already answers this. One request opens several guards, and the DNS
    # shortcut is only cheap when DNS is fast: a slow resolver costs its full timeout per
    # guard and leaves a stuck thread behind each time. Peek only, never record: a dead
    # lookup fails fast anyway, and recording it would make recovery sticky for the TTL.
    memo = hf_reachability_memo()
    if memo is not None:
        return memo
    if hf_dns_dead():
        return True
    return hf_unreachable()


@contextlib.contextmanager
def _hf_offline_if_unreachable():
    """Force HF offline for this block only when the Hub is unreachable; restores on exit
    so a transient hiccup can't quarantine the process. No-op if the USER set the env var.

    Env vars alone are too late mid-process (the offline constants are read at import), so
    this holds a refcounted force_hf_offline window. Overlapping requests each take their
    own reference: no-opping would let another guard's exit restore the network
    mid-operation and drop this request back onto the retry path.
    """
    try:
        from utils.utils import force_hf_offline, force_hf_offline_state, hf_env_offline
    except Exception:
        yield False
        return

    ours, hf_hub_env_present = force_hf_offline_state()
    # Membership preserves HF_HUB_OFFLINE=0 as an explicit stay-online override.
    # Read it with the refcount so a concurrent guard cannot look user-owned.
    if hf_hub_env_present and not ours:
        yield False
        return
    # TRANSFORMERS_OFFLINE alone still asks for offline, and the hub does not read it.
    # Honour it directly: probing would put network traffic in an explicitly offline process.
    if not ours and hf_env_offline():
        with force_hf_offline():
            yield True
        return
    # Already forced by an in-flight guard: reuse that verdict and hold a reference.
    if not ours and not _hf_unreachable():
        yield False
        return

    if not ours:
        logger.warning("Hugging Face endpoint unreachable; using local HF cache for this load.")
    with force_hf_offline():
        yield True


def _hf_offline_if_unreachable_for(model_name):
    """Guard, but only for remote repo ids: a local path is served from disk and never
    reaches the hub, so probing would add seconds per request and prevent no retry."""
    try:
        from utils.paths import is_local_path
        if isinstance(model_name, str) and is_local_path(model_name):
            return contextlib.nullcontext()
    except Exception:
        pass
    return _hf_offline_if_unreachable()


try:
    _SLOT_SAVE_MAX_BYTES = int(os.environ.get("UNSLOTH_SLOT_SAVE_MAX_BYTES") or (10 << 30))
except ValueError:
    _SLOT_SAVE_MAX_BYTES = 10 << 30

# The idle loop holds the lifecycle gate across a slot save, so a newly arriving
# request waits on the in-flight save's HTTP call. Bound it (was 120s) so a slow
# or stuck save can't stall the next request for minutes; best-effort save just
# falls back to a plain unload. Override with UNSLOTH_SLOT_SAVE_TIMEOUT (seconds).
try:
    _SLOT_SAVE_HTTP_TIMEOUT = float(os.environ.get("UNSLOTH_SLOT_SAVE_TIMEOUT") or 30.0)
except ValueError:
    _SLOT_SAVE_HTTP_TIMEOUT = 30.0
if _SLOT_SAVE_HTTP_TIMEOUT <= 0:
    _SLOT_SAVE_HTTP_TIMEOUT = 30.0


def _swa_cache_path() -> Path:
    home = os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME")
    base = Path(home) if home else Path.home() / ".unsloth" / "studio"
    return base / "swa_cache.json"


def _load_swa_cache() -> dict:
    global _SWA_CACHE
    with _SWA_CACHE_LOCK:
        if _SWA_CACHE is not None:
            return _SWA_CACHE
        try:
            with open(_swa_cache_path(), encoding = "utf-8-sig") as f:
                _SWA_CACHE = json.load(f)
                if not isinstance(_SWA_CACHE, dict):
                    _SWA_CACHE = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
            _SWA_CACHE = {}
        return _SWA_CACHE


def _save_swa_cache(cache: dict) -> None:
    try:
        path = _swa_cache_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(cache, f, indent = 2, sort_keys = True)
        tmp.replace(path)
    except (OSError, UnicodeDecodeError):
        pass


def _period_from_layer_types(layer_types: list) -> Optional[int]:
    """Smallest period N where `(i+1) % N != 0` matches the SWA mask, else None."""
    if not layer_types:
        return None
    is_swa = ["full" not in str(t).lower() for t in layer_types]
    n = len(is_swa)
    for N in range(1, n + 1):
        if all(((i + 1) % N != 0) == is_swa[i] for i in range(n)):
            return N
    return None


def _swa_entry_from_layer_types(lt) -> Optional[object]:
    """Period int, or per-layer bool mask, from a transformers ``layer_types`` list."""
    if isinstance(lt, list) and lt:
        return _period_from_layer_types(lt) or ["full" not in str(t).lower() for t in lt]
    return None


def _fetch_swa_entry_from_hf(repo_id: str) -> Optional[object]:
    try:
        from huggingface_hub import hf_hub_download
        from utils.hf_cache_settings import active_hf_hub_cache

        cfg_path = hf_hub_download(
            repo_id,
            "config.json",
            repo_type = "model",
            cache_dir = active_hf_hub_cache(),
        )
        with open(cfg_path, encoding = "utf-8-sig") as f:
            cfg = json.load(f)
    except Exception:
        return None

    src = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    period = src.get("sliding_window_pattern")
    if isinstance(period, int) and period > 0:
        return period
    return _swa_entry_from_layer_types(src.get("layer_types"))


def _arch_aliases(arch: str) -> tuple:
    # GGUF emits `falcon-h1`; HF model_type is `falcon_h1`. Normalise both ways.
    seen = []
    for a in (arch, arch.replace("-", "_"), arch.replace("_", "-")):
        if a and a not in seen:
            seen.append(a)
    return tuple(seen)


def _swa_entry_from_config_obj(cfg) -> Optional[object]:
    src = getattr(cfg, "text_config", None) or cfg
    period = getattr(src, "sliding_window_pattern", None)
    if isinstance(period, int) and period > 0:
        return period
    return _swa_entry_from_layer_types(getattr(src, "layer_types", None))


_SWA_PATTERN_SOURCE_RE = re.compile(r"sliding_window_pattern\s*(?::\s*[\w\[\], ]*)?\s*=\s*(\d+)")


def _resolve_swa_entry_from_transformers(arch: str) -> Optional[object]:
    """Default-instantiate the matching Config; on failure, regex-parse its
    source for `sliding_window_pattern = N`."""
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )
    except Exception:
        return None

    cfg_class = None
    for alias in _arch_aliases(arch):
        if alias in CONFIG_MAPPING_NAMES:
            try:
                cfg_class = CONFIG_MAPPING[alias]
                break
            except Exception:
                cfg_class = None
    if cfg_class is None:
        return None

    try:
        if (entry := _swa_entry_from_config_obj(cfg_class())) is not None:
            return entry
    except Exception:
        pass

    import inspect

    candidates = [cfg_class]
    text_cfg_class = getattr(cfg_class, "sub_configs", {}).get("text_config")
    if text_cfg_class is not None:
        candidates.append(text_cfg_class)
    for cls in candidates:
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        if m := _SWA_PATTERN_SOURCE_RE.search(src):
            period = int(m.group(1))
            if period > 0:
                return period
    return None


def _resolve_swa_pattern(
    arch: Optional[str],
    n_layers: Optional[int],
    source_repo_candidates: tuple = (),
    *,
    allow_network: Optional[bool] = None,
) -> Optional[list]:
    if not arch or not n_layers:
        return None
    if allow_network is None:
        allow_network = os.environ.get("UNSLOTH_STUDIO_OFFLINE", "0") not in (
            "1",
            "true",
            "True",
            "yes",
        )

    cache = _load_swa_cache()

    def _entry_to_mask(entry):
        if isinstance(entry, int) and entry > 0:
            return [(i + 1) % entry != 0 for i in range(n_layers)]
        if isinstance(entry, list) and entry:
            return [bool(entry[i % len(entry)]) for i in range(n_layers)]
        return None

    def _persist(entry):
        with _SWA_CACHE_LOCK:
            cache[arch] = entry
        _save_swa_cache(cache)

    if (entry := cache.get(arch)) is not None:
        if (mask := _entry_to_mask(entry)) is not None:
            return mask

    if (entry := _BOOTSTRAP_SWA_DEFAULTS.get(arch)) is not None:
        return _entry_to_mask(entry)

    entry = _resolve_swa_entry_from_transformers(arch)
    if entry is not None:
        _persist(entry)
        return _entry_to_mask(entry)

    # Tier 3: live HF fetch (result persistently cached)
    if allow_network:
        for repo_id in source_repo_candidates:
            if not repo_id:
                continue
            entry = _fetch_swa_entry_from_hf(repo_id)
            if entry is not None:
                _persist(entry)
                return _entry_to_mask(entry)

    return None


def _hf_repo_from_url(url: Optional[str]) -> Optional[str]:
    """Strip `https://huggingface.co/owner/name(/...)` -> `owner/name`."""
    if not url or "huggingface.co/" not in url:
        return None
    tail = url.split("huggingface.co/", 1)[1].rstrip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


# Lazy import to avoid pulling transformers in at module level.
def _extract_model_size_b(model_id: str):
    from utils.models import extract_model_size_b
    return extract_model_size_b(model_id)


_TOOL_TEMPLATE_MARKERS = (
    "{%- if tools %}",
    "{%- if tools -%}",
    "{% if tools %}",
    "{% if tools -%}",
    # Defensive templates guard with `tools is defined` before truth-testing
    # (e.g. Inkling: `{%- if tools is defined and tools -%}`).
    "{%- if tools is defined",
    "{% if tools is defined",
    '"role" == "tool"',
    "'role' == 'tool'",
    'message.role == "tool"',
    "message.role == 'tool'",
    # DeepSeek: no top-level ``{% if tools %}`` block; it gates emission on
    # ``message['role'] == 'tool'`` plus ``message['tool_calls'] is defined``.
    "message['role'] == 'tool'",
    'message["role"] == "tool"',
    "message['tool_calls']",
    'message["tool_calls"]',
    "tool_calls is defined",
)


# Canonical reasoning_effort levels, weakest -> strongest. Used to read the
# discrete set a template branches on (e.g. GLM-5.2 uses 'high' | 'max', Inkling
# uses the full 'none'..'max' ladder) so we only ever offer levels the template
# actually understands.
_REASONING_EFFORT_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _extract_reasoning_effort_levels(chat_template: str) -> list:
    """Return the reasoning_effort levels a template references, in canonical
    (weakest -> strongest) order.

    Looks for the quoted literals (e.g. ``'high'`` / ``"max"``) the template
    compares ``reasoning_effort`` against, so we surface exactly the levels it
    branches on and nothing else.
    """
    return [
        level
        for level in _REASONING_EFFORT_SCALE
        if f"'{level}'" in chat_template or f'"{level}"' in chat_template
    ]


def detect_reasoning_flags(
    chat_template: Optional[str],
    model_identifier: Optional[str] = None,
    *,
    log_source: Optional[str] = None,
) -> dict:
    """Classify a chat template's reasoning and tool-calling capabilities.

    Returns the same six keys as the GGUF sniffer: ``supports_reasoning``,
    ``reasoning_style`` (``"enable_thinking"`` | ``"reasoning_effort"`` |
    ``"enable_thinking_effort"``), ``reasoning_always_on``,
    ``reasoning_effort_levels``, ``supports_preserve_thinking``,
    ``supports_tools``. A falsy ``chat_template`` yields the all-default dict.
    Used by both the llama-server backend at load time and the
    safetensors/transformers paths in ``routes/inference`` so they agree on
    what the frontend sees.
    """
    flags = {
        "supports_reasoning": False,
        "reasoning_style": "enable_thinking",
        "reasoning_always_on": False,
        "reasoning_effort_levels": [],
        "supports_preserve_thinking": False,
        "supports_tools": False,
    }
    if not chat_template:
        return flags
    tpl = chat_template
    prefix = f"{log_source}: " if log_source else ""

    # 'none' is this style's disable sentinel, not an effort level: the off
    # switch is enable_thinking=false, and the Think menu already hides it.
    # Left in, it is the weakest level, so a stored effort the model does not
    # offer clamps onto it and silently disables thinking (Kimi-K3).
    effort_levels = (
        [lv for lv in _extract_reasoning_effort_levels(tpl) if lv != "none"]
        if ("reasoning_effort" in tpl and "enable_thinking" in tpl)
        else []
    )
    if effort_levels:
        # DeepSeek-V4's encoder accepts reasoning_effort {'high', 'max'} but its
        # template only branches on 'max', so the literal scan misses 'high'. Add it
        # (matched on whole repo-name segments, so 'deepseek-v40' won't false-match)
        # to expose the full high/max ladder instead of max alone.
        segments = re.split(r"[-_.]", (model_identifier or "").lower().split("/")[-1])
        is_dsv4 = "deepseek4" in segments or any(
            a == "deepseek" and b == "v4" for a, b in zip(segments, segments[1:])
        )
        if is_dsv4 and "high" not in effort_levels:
            effort_levels = sorted(set(effort_levels) | {"high"}, key = _REASONING_EFFORT_SCALE.index)
        # GLM-5.2-style: an enable_thinking on/off gate PLUS a reasoning_effort
        # level among a discrete set (e.g. 'high' | 'max'). Distinct from
        # gpt-oss (reasoning_effort only, no on/off gate) and Qwen
        # (enable_thinking only). Disabling is enable_thinking=false; the levels
        # are the quoted effort literals the template actually branches on.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking_effort"
        flags["reasoning_effort_levels"] = effort_levels
        logger.info(
            f"{prefix}model supports reasoning "
            f"(enable_thinking + reasoning_effort: {effort_levels})"
        )
    elif "enable_thinking" in tpl:
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking"
        logger.info(f"{prefix}model supports reasoning (enable_thinking)")
    elif "reasoning_effort" in tpl:
        # gpt-oss / Harmony use reasoning_effort
        # ("low" | "medium" | "high"), not a boolean. Inkling maps a wider
        # named ladder ('none'..'max'); surface the levels the template
        # actually branches on so the Think menu offers exactly those and
        # nothing more ('none' renders as thinking off: effort 0). Guard:
        # trust the scan only when it includes the core 'low' and 'high'
        # literals, so a template that quotes e.g. 'none' for an unrelated
        # comparison keeps the default low/medium/high set.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "reasoning_effort"
        scanned = _extract_reasoning_effort_levels(tpl)
        if "low" in scanned and "high" in scanned:
            flags["reasoning_effort_levels"] = scanned
        logger.info(
            f"{prefix}model supports reasoning "
            f"(reasoning_effort: {flags['reasoning_effort_levels'] or 'default levels'})"
        )
    elif "thinking" in tpl:
        # DeepSeek uses 'thinking', not 'enable_thinking'
        normalized_id = (model_identifier or "").lower()
        if "deepseek" in normalized_id:
            flags["supports_reasoning"] = True
            logger.info(f"{prefix}model supports reasoning (DeepSeek thinking)")

    # Hardcoded <think> tags or reasoning_content in the template mean
    # thinking is always on (no toggle).
    if not flags["supports_reasoning"]:
        if ("<think>" in tpl and "</think>" in tpl) or "reasoning_content" in tpl:
            flags["supports_reasoning"] = True
            flags["reasoning_always_on"] = True
            logger.info(f"{prefix}model always reasons (<think> tags in template)")

    # preserve_thinking: independent kwarg on some Qwen templates that
    # keeps historical <think> blocks in prior assistant turns.
    if "preserve_thinking" in tpl:
        flags["supports_preserve_thinking"] = True
        logger.info(f"{prefix}model supports preserve_thinking")

    if any(marker in tpl for marker in _TOOL_TEMPLATE_MARKERS):
        flags["supports_tools"] = True
        logger.info(f"{prefix}model supports tool calling")

    return flags


# Gemma 4 ships MTP as a separate drafter (no "-mtp" in the name). Gemma 3n
# ships no drafter, so it is excluded -- it takes the normal non-MTP path.
_GEMMA_MTP_FAMILY_RE = re.compile(r"gemma[-_]?4[-_]", re.IGNORECASE)


def _is_gemma_mtp_family(name: Optional[str]) -> bool:
    """Match Gemma 4 by name."""
    return bool(name) and bool(_GEMMA_MTP_FAMILY_RE.search(name))


def _is_gemma_mtp_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Match Gemma 4 by id or GGUF filename."""
    return _is_gemma_mtp_family(model_identifier) or _is_gemma_mtp_family(
        Path(gguf_path).name if gguf_path else None
    )


def _is_mtp_model_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Name-based MTP detector. Fallback for the metadata signal."""
    for cand in (model_identifier, Path(gguf_path).name if gguf_path else None):
        if cand and "-mtp" in cand.lower():
            return True
        # Recognise Gemma 4 too, so a failed drafter download surfaces a
        # fallback reason instead of silently defaulting.
        if cand and _is_gemma_mtp_family(cand):
            return True
    return False


# Mirrors hub.utils.gguf._DRAFTER_KINDS / _DRAFTER_DIR_KINDS.
_DRAFTER_KINDS = ("mtp", "dspark", "dflash")
_DRAFTER_DIR_KINDS = ("mtp", "dspark")


def _drafter_path_kind(path: str) -> Optional[str]:
    """Drafter kind naming *path*: basename prefix, or exact parent dir for
    ``_DRAFTER_DIR_KINDS``. Never a substring: the kind names double as family
    names, so ``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`` IS the model, as is anything
    in a user's ``dflash/`` folder."""
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return None
    parts = [segment for segment in p.split("/") if segment]
    name, parents = parts[-1], parts[:-1]
    for kind in _DRAFTER_KINDS:
        if name.startswith(f"{kind}-"):
            return kind
    for kind in _DRAFTER_DIR_KINDS:
        if kind in parents:
            return kind
    return None


def _is_companion_gguf_path(path: str) -> bool:
    """True for a non-main GGUF: vision mmproj, or a separate drafter (repo-root
    ``mtp-*.gguf``, the ``MTP/`` subdir copies for Gemma 4, the ``dspark/``
    drafters for DeepSeek V4 Flash). Mirrors hub.utils.gguf so variant resolution
    never picks a companion as the main model -- a Gemma ``Q8_0`` request must not
    resolve to ``MTP/...-Q8_0-MTP.gguf``, which sorts ahead of the real weight.

    EXCLUSION ONLY. Use ``_is_mtp_only_drafter_path`` to pick a drafter to launch.
    """
    p = path.lower()
    if not p.endswith(".gguf"):
        return False
    if "mmproj" in p:
        return True
    return _drafter_path_kind(path) is not None


def _is_mtp_only_drafter_path(path: str) -> bool:
    """MTP only, not the other companion GGUFs. Each drafter kind needs its own
    ``--spec-type``, so discovery narrows to one kind rather than reusing the
    broad companion predicate, which exists to EXCLUDE companions from menus."""
    return _drafter_path_kind(path) == "mtp"


def _is_dspark_drafter_path(path: str) -> bool:
    """True for a DSpark sidecar, excluding the separate DFlash method.

    Broader than model_config.detect_dspark_file, which additionally requires a
    published drafter NAME: a repo's ``dspark/`` directory holds only sidecars,
    while a user's local folder may hold anything. Only ever used against repo
    listings and cache snapshots for that reason.
    """
    return _drafter_path_kind(path) == "dspark"


_BIG_ENDIAN_GGUF_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)
_GGUF_KNOWN_QUANT_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|Q[0-9]+_[0-9]+"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)


def _is_big_endian_gguf_path(path: str, variant_key: str = "") -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0].lower()
    variant_key = variant_key.strip().lower()
    variant_index = stem.find(variant_key) if variant_key else -1
    parent = normalized.rsplit("/", 1)[0].lower() if "/" in normalized else ""
    variant_in_parent_only = (
        bool(parent)
        and variant_index < 0
        and (
            (variant_key and variant_key in parent)
            or (not variant_key and _GGUF_KNOWN_QUANT_RE.search(parent) is not None)
        )
    )
    for match in _BIG_ENDIAN_GGUF_FILENAME_RE.finditer(stem):
        if variant_index >= 0 and variant_index < match.start():
            return True
        tail = stem[match.end() :].lstrip("._-")
        if not tail or _GGUF_KNOWN_QUANT_RE.search(tail) is None:
            return not variant_in_parent_only
    return False


def _gguf_snapshot_files(snapshot: Path) -> list[str]:
    return [
        p.relative_to(snapshot).as_posix()
        for p in snapshot.rglob("*")
        if p.is_file() and p.name.lower().endswith(".gguf")
    ]


def _cached_hf_snapshot_file(
    repo_id: str,
    filename: str,
    *,
    expected_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """Return a cached snapshot file even when HF's current-ref probe misses it."""
    if not filename:
        return None
    parts = [part for part in filename.replace("\\", "/").split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        return None
    try:
        if cache_dir is None:
            from utils.models.model_config import _iter_hf_cache_snapshots
            snapshots = _iter_hf_cache_snapshots(repo_id)
        else:
            from hub.utils.hf_cache_state import iter_active_repo_cache_dirs
            snapshots = (
                snapshot
                for repo_dir in iter_active_repo_cache_dirs(
                    "model",
                    repo_id,
                    root = Path(cache_dir),
                )
                for snapshot in (repo_dir / "snapshots").glob("*")
                if snapshot.is_dir()
            )
        for snap in snapshots:
            candidate = snap.joinpath(*parts)
            if not candidate.is_file():
                continue
            if expected_size:
                try:
                    if candidate.stat().st_size < expected_size:
                        continue
                except OSError:
                    continue
            return str(candidate)
    except Exception as e:
        logger.debug("Snapshot cache lookup failed for %s/%s: %s", repo_id, filename, e)
    return None


def _snapshot_has_all_shards(
    main_path: str, main_filename: str, shards: Iterable[str], expected_sizes: dict[str, int]
) -> bool:
    """True when every shard sits beside ``main_path`` in the same cache snapshot.

    llama.cpp loads a split GGUF by resolving its siblings from the main shard's
    directory, so a cached main shard is only safe to reuse when the rest of the
    set is co-located; otherwise the caller must fetch the whole set together.
    """
    root = Path(main_path)
    for _ in [part for part in main_filename.replace("\\", "/").split("/") if part]:
        root = root.parent
    for shard in shards:
        parts = [part for part in shard.replace("\\", "/").split("/") if part]
        if not parts or any(part in (".", "..") for part in parts):
            return False
        sibling = root.joinpath(*parts)
        try:
            if not sibling.is_file():
                return False
            expected = expected_sizes.get(shard)
            if expected and sibling.stat().st_size < expected:
                return False
        except OSError:
            return False
    return True


def _resolve_repo_id_casing(hf_repo: str) -> str:
    """Map a requested repo id to its cached canonical casing, or return it unchanged.

    A case-variant request (for example a lowercased id) resolves to the
    canonical-cased cache directory so the main GGUF and its companions
    (mmproj / MTP drafter) all read the same cache entry. Returns ``hf_repo``
    unchanged when resolution is unavailable or errors.
    """
    try:
        from utils.paths import resolve_cached_repo_id_case
        return resolve_cached_repo_id_case(hf_repo)
    except Exception:
        return hf_repo


def _cached_colocated_split_main(
    repo_id: str, main_filename: str, shards: Iterable[str], expected_sizes: dict[str, int]
) -> Optional[str]:
    """Main-shard path from a cache snapshot that also holds every sibling shard.

    A newer snapshot may hold only the first shard while an older snapshot has the
    complete split set. ``_cached_hf_snapshot_file`` would return that newer partial
    main and the co-location check would then force a refetch, so scan snapshots for
    one where the whole set is present and return that main path instead. None when
    no snapshot holds the full set.
    """
    main_parts = [part for part in main_filename.replace("\\", "/").split("/") if part]
    if not main_parts or any(part in (".", "..") for part in main_parts):
        return None
    try:
        from utils.models.model_config import _iter_hf_cache_snapshots
        for snap in _iter_hf_cache_snapshots(repo_id):
            main_path = snap.joinpath(*main_parts)
            if not main_path.is_file():
                continue
            expected_main = expected_sizes.get(main_filename)
            try:
                if expected_main and main_path.stat().st_size < expected_main:
                    continue
            except OSError:
                continue
            if _snapshot_has_all_shards(str(main_path), main_filename, shards, expected_sizes):
                return str(main_path)
    except Exception as e:
        logger.debug("Co-located split snapshot lookup failed for %s: %s", repo_id, e)
    return None


def _cached_variant_resolution(repo_id: str, hf_variant: str) -> tuple[Optional[str], list[str]]:
    """Find a cached main GGUF and its shards for a variant."""
    candidate = next(_cached_variant_candidates(repo_id, hf_variant), None)
    if candidate is None:
        return None, []
    _, main, shards, _ = candidate
    return main, shards


def _cached_variant_candidates(
    repo_id: str,
    hf_variant: str,
    *,
    require_mmproj: bool = False,
) -> Generator[tuple[str, str, list[str], Path], None, None]:
    """Yield complete cached variant copies in snapshot preference order."""
    try:
        from utils.models.model_config import _iter_hf_cache_snapshots
        for snap in _iter_hf_cache_snapshots(repo_id):
            cached_files = _gguf_snapshot_files(snap)
            matches = _gguf_files_for_variant(cached_files, hf_variant)
            if not matches:
                continue
            main = matches[0]
            shards = _gguf_extra_shards(matches, main)
            split = _SHARD_FULL_RE.match(main)
            if split:
                numbers = {
                    int(match.group(2))
                    for path in [main, *shards]
                    if (match := _SHARD_FULL_RE.match(path))
                }
                if numbers != set(range(1, int(split.group(3)) + 1)):
                    continue
            main_path = snap.joinpath(*main.replace("\\", "/").split("/"))
            if not main_path.is_file() or not _snapshot_has_all_shards(
                str(main_path), main, shards, {}
            ):
                continue
            if require_mmproj and not _pick_mmproj(cached_files):
                continue
            yield str(main_path), main, shards, snap
    except Exception as e:
        logger.debug(f"Cache lookup for variant failed: {e}")


def _cached_candidate_matches_revision_size(
    repo_id: str, candidate: tuple[str, str, list[str], Path], hf_token: Optional[str]
) -> bool:
    """Check cached byte sizes against the snapshot's own Hub revision.

    A snapshot pointer is normally published only after its blob is complete.
    When the old revision is still queryable, also compare every weight file's
    size so a manually truncated cache entry is not treated as reusable. If
    metadata cannot be reached, retain the cache's normal offline semantics.
    """
    main_path, main, shards, snap = candidate
    paths = [main, *shards]
    try:
        from huggingface_hub import get_paths_info
        infos = list(
            get_paths_info(
                repo_id,
                paths,
                revision = snap.name,
                token = hf_token,
            )
        )
    except Exception as e:
        logger.debug(
            "Could not size-check cached GGUF %s at revision %s: %s",
            repo_id,
            snap.name,
            e,
        )
        return True

    if not infos:
        # The Hub answers an unknown (e.g. force-pushed away) revision with an
        # empty result, not an error; treat it like unreachable metadata.
        return True
    expected_sizes = {info.path: info.size for info in infos if info.size is not None}
    if any(path not in expected_sizes for path in paths):
        return False
    try:
        if os.path.getsize(main_path) < expected_sizes[main]:
            return False
    except OSError:
        return False
    return _snapshot_has_all_shards(main_path, main, shards, expected_sizes)


def _cached_complete_candidate(
    repo_id: str, gguf_filename: Optional[str], shards: list[str]
) -> Optional[tuple[str, str, list[str], Path]]:
    """Return one complete exact-filename cache candidate with snapshot context."""
    if not gguf_filename:
        return None
    if shards:
        main_path = _cached_colocated_split_main(repo_id, gguf_filename, shards, {})
    else:
        m = _SHARD_FULL_RE.match(gguf_filename)
        if m and int(m.group(3)) > 1:
            return None
        main_path = _cached_hf_snapshot_file(repo_id, gguf_filename)
    if main_path is None:
        return None
    snap = _snapshot_dir_of(main_path)
    if snap is None:
        return None
    return main_path, gguf_filename, shards, snap


def cached_gguf_for_load(
    hf_repo: str,
    hf_variant: Optional[str],
    *,
    require_mmproj: bool = False,
    verify_sizes: bool = False,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Return a cached GGUF that can be loaded without downloading."""
    if not hf_variant:
        return None
    hf_repo = _resolve_repo_id_casing(hf_repo)
    for candidate in _cached_variant_candidates(
        hf_repo,
        hf_variant,
        require_mmproj = require_mmproj,
    ):
        if verify_sizes and not _cached_candidate_matches_revision_size(
            hf_repo, candidate, hf_token
        ):
            continue
        return candidate[0]
    return None


def _snapshot_dir_of(path: str) -> Optional[Path]:
    """Return the HF cache snapshot containing path, if any."""
    try:
        p = Path(os.path.abspath(path))
    except OSError:
        return None
    for ancestor in p.parents:
        if ancestor.parent.name == "snapshots":
            return ancestor
    return None


def _hub_cache_dir_for_snapshot_path(path: Optional[str]) -> Optional[str]:
    """Return the HF Hub cache root that owns a snapshot-contained path."""
    if not path:
        return None
    snapshot = _snapshot_dir_of(path)
    if snapshot is None or snapshot.parent.name != "snapshots":
        return None
    return str(snapshot.parent.parent.parent)


def _companion_snapshot_sibling(
    near_path: str, pick: Callable[[list[str]], Optional[str]]
) -> Optional[str]:
    """Find a companion in the same snapshot as near_path."""
    snap = _snapshot_dir_of(near_path)
    if snap is None:
        return None
    try:
        sibling = pick(_gguf_snapshot_files(snap))
    except Exception:
        return None
    if not sibling:
        return None
    candidate = snap / sibling
    return str(candidate) if candidate.is_file() else None


def _pick_mmproj(candidates: list[str]) -> Optional[str]:
    mmproj_files = sorted(
        f for f in candidates if f.lower().endswith(".gguf") and "mmproj" in Path(f).name.lower()
    )
    if not mmproj_files:
        return None
    return next((f for f in mmproj_files if f.lower().endswith("-f16.gguf")), mmproj_files[0])


def _hub_download_in_flight(hf_repo: str) -> bool:
    try:
        from hub.utils.download_registry import get_models_registry
        return bool(get_models_registry().active_job_refs(hf_repo))
    except Exception:
        return False


def _hub_download_blocks_gguf_load(
    hf_repo: str,
    hf_variant: Optional[str],
    *,
    require_mmproj: bool = False,
    hf_token: Optional[str] = None,
) -> bool:
    """Whether an active Hub job makes this GGUF load unsafe.

    Same-variant jobs can reclaim the stale snapshot a load would reuse, so
    they always block. Other jobs block only when this load lacks a complete
    cached copy and would write to the shared cache itself.
    """
    try:
        from hub.utils.download_registry import get_models_registry

        registry = get_models_registry()
        if not registry.active_job_refs(hf_repo):
            return False
        if registry.has_active_variant(hf_repo, hf_variant):
            return True
    except Exception:
        return False
    return (
        cached_gguf_for_load(
            hf_repo,
            hf_variant,
            require_mmproj = require_mmproj,
            verify_sizes = True,
            hf_token = hf_token,
        )
        is None
    )


# Active GGUF loads by normalized repo ID.
_LOADS_IN_FLIGHT: dict[str, int] = {}
_LOADS_IN_FLIGHT_LOCK = threading.Lock()


@contextlib.contextmanager
def gguf_load_in_flight(hf_repo: Optional[str]):
    """Track an HF GGUF load until the context exits."""
    key = (hf_repo or "").strip().lower()
    if not key:
        yield
        return
    with _LOADS_IN_FLIGHT_LOCK:
        _LOADS_IN_FLIGHT[key] = _LOADS_IN_FLIGHT.get(key, 0) + 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            remaining = _LOADS_IN_FLIGHT.get(key, 1) - 1
            if remaining <= 0:
                _LOADS_IN_FLIGHT.pop(key, None)
            else:
                _LOADS_IN_FLIGHT[key] = remaining


# Chat loads in flight, repo-agnostic (local paths and safetensors included). The repo-keyed counter above answers the download manager; this one answers the GPU arbiter, which must know before llama-server spawns.
_CHAT_LOADS_IN_FLIGHT = 0


@contextlib.contextmanager
def chat_load_in_flight():
    """Track a chat load from before its download until the load call returns."""
    global _CHAT_LOADS_IN_FLIGHT
    with _LOADS_IN_FLIGHT_LOCK:
        _CHAT_LOADS_IN_FLIGHT += 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            _CHAT_LOADS_IN_FLIGHT = max(0, _CHAT_LOADS_IN_FLIGHT - 1)


def chat_load_active() -> bool:
    """Whether a chat load is in flight, spawned or not.

    ``is_active`` only covers a live llama-server process, which an HF load does not have
    until its GGUF finished downloading -- minutes during which the arbiter's evictor would
    find nothing to cancel and grant the GPU to a second workload.
    """
    with _LOADS_IN_FLIGHT_LOCK:
        return _CHAT_LOADS_IN_FLIGHT > 0


def zero_vram_chat_load(
    gpu_memory_mode: str,
    gpu_layers: int,
    extra_args: Optional[list] = None,
    needs_mmproj: bool = False,
    speculative_type: Optional[str] = None,
) -> bool:
    """Whether a GGUF load will hold no VRAM at all, decided from the request.

    A deliberate manual zero-offload load runs on the CPU and is launched with the GPUs hidden
    from the child (see ``_cpu_only_zero_offload``), so it neither needs the GPU arbiter nor
    should evict an image/video pipeline for it. This mirrors that launch-time mask so both
    agree: an mmproj, a GPU drafter, a device pin or a surviving tensor mode all keep the GPUs
    visible, and the mask is skipped on Vulkan. ``--mmproj`` and ``--model-draft`` are added by
    the backend rather than being present in ``extra_args``, so their intent is passed in.
    """
    if gpu_memory_mode != "manual" or gpu_layers != 0:
        return False
    # Any speculative mode may launch a GPU drafter and only the request's own knobs are known here, so treat every selection as GPU-bearing except
    # "off". Canonicalize first: the UI sends the literal "off", which a bare truthiness test read as "speculation requested".
    spec_mode = _canonicalize_spec_mode(speculative_type)
    if needs_mmproj or spec_mode not in (None, "off"):
        return False
    if LlamaCppBackend._is_vulkan_backend():
        return False
    return not LlamaCppBackend._zero_offload_keeps_gpu_visible(
        [str(arg) for arg in (extra_args or [])], os.environ
    )


def hf_gguf_load_in_flight(hf_repo: str) -> bool:
    """Return whether a GGUF load is active for hf_repo."""
    key = (hf_repo or "").strip().lower()
    if not key:
        return False
    with _LOADS_IN_FLIGHT_LOCK:
        return _LOADS_IN_FLIGHT.get(key, 0) > 0


def _with_gguf_load_marker(load: Callable):
    """Keep an HF repo marked for the full synchronous load call."""

    @functools.wraps(load)
    def wrapped(self, intent: GgufLoadIntent):
        hf_repo = intent.hf_repo
        with gguf_load_in_flight(hf_repo):
            if hf_repo and _hub_download_blocks_gguf_load(
                hf_repo,
                intent.hf_variant,
                require_mmproj = bool(
                    intent.is_vision and not extra_args_disable_mmproj(intent.extra_args)
                ),
                hf_token = intent.hf_token,
            ):
                raise RuntimeError(
                    f"'{hf_repo}' is currently being downloaded by the download manager"
                )
            return load(self, intent)

    return wrapped


def _gguf_extra_shards(files: Iterable[str], first_shard: str) -> list[str]:
    m = _SHARD_FULL_RE.match(first_shard)
    if not m:
        return []
    prefix = m.group(1)
    total = m.group(3)
    sibling_pat = re.compile(
        r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(total) + r"\.gguf$",
        re.IGNORECASE,
    )
    return sorted(f for f in files if f != first_shard and sibling_pat.match(f))


def _gguf_files_for_variant(files: Iterable[str], variant: str) -> list[str]:
    """Return main GGUF files matching a requested variant.

    Prefer exact quant-label matches over loose substring matches so a request
    for ``stories260K`` does not resolve to ``stories260K-be.gguf``.
    """
    variant_key = variant.strip().lower()
    main_files = [
        f
        for f in files
        if f.lower().endswith(".gguf")
        and not _is_companion_gguf_path(f)
        and not _is_big_endian_gguf_path(f, variant_key)
    ]
    if not variant_key:
        return sorted(main_files)

    try:
        from utils.models.model_config import _extract_quant_label
    except Exception:
        _extract_quant_label = None

    if _extract_quant_label is not None:
        try:
            exact = sorted(f for f in main_files if _extract_quant_label(f).lower() == variant_key)
            if exact:
                return exact
        except Exception as e:
            logger.warning("Failed to extract GGUF quant labels: %s", e)

    boundary = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(variant_key) + r"(?![a-zA-Z0-9])")
    return sorted(f for f in main_files if boundary.search(f.lower()))


# Below this many B params, draft-mtp regresses vs spec-off (bench in
# _build_speculative_flags); auto mode drops MTP under it.
_MTP_MIN_SIZE_B = 3.0

# Cap total GPU occupancy at this fraction of the card. The fit reserves an
# absolute (1 - frac) * total per GPU when total VRAM is known, else a fraction
# of free (see _fit_context_to_vram), plus a byte-accurate MTP draft reserve.
# 3%: the context-linear compute buffer is now modelled (_compute_buffer_ctx_bytes),
# so this cushion no longer covers it - only fragmentation, the per-device CUDA
# context on a multi-GPU split, and MoE routing, which measure ~2-3% (Qwen3.5-397B on
# 3 GPUs under-predicts by 2.7%). Below 3% one fragmentation spike overflows to CPU.
_CTX_FIT_VRAM_FRACTION = 0.97

# Apple unified memory is shared with the OS, so tighter than VRAM. Matches the
# 0.85 MLX uses in mlx_inference.py (_configure_memory_limits); not kept in sync.
_APPLE_UNIFIED_MEMORY_FRACTION = 0.85

# Flat MTP reserve, used only when GGUF dims are too sparse for the byte-accurate
# reserve (_estimate_mtp_overhead_bytes). Applied to both the fit budget and pin.
_MTP_VRAM_RESERVE_FRAC = 0.05


def _kv_bytes_per_elem(cache_type: Optional[str]) -> float:
    """Bytes per KV-cache element for a llama.cpp cache type (f16 default)."""
    return {
        "f32": 4.0,
        "f16": 2.0,
        "bf16": 2.0,
        "q8_0": 34 / 32,
        "q5_1": 0.75,
        "q5_0": 0.6875,
        "q4_1": 0.625,
        "q4_0": 0.5625,
        "iq4_nl": 0.5625,
    }.get((cache_type or "f16").strip().lower(), 2.0)


def _pad_kv_cells(cells: int) -> int:
    return ((cells + 255) // 256) * 256


def _kv_cache_cell_layout(n_ctx: int, n_parallel: int, kv_unified: bool) -> tuple[int, int, int]:
    """Return llama.cpp's slot count, stream count, and cells per stream."""
    slots = max(1, n_parallel)
    padded_ctx = _pad_kv_cells(n_ctx)
    streams = 1 if kv_unified else slots
    if padded_ctx <= 0:
        return slots, streams, 0
    cells_per_stream = padded_ctx if kv_unified else _pad_kv_cells(padded_ctx // slots)
    return slots, streams, cells_per_stream


def _env_main_cache_type_for_budget(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Heavier of the inherited LLAMA_ARG_CACHE_TYPE_K/_V env types when it
    exceeds the f16 default, else None. Unsloth emits --cache-type only for the
    param/extras path, so a heavier env (f32) would otherwise reach the child
    unbudgeted; quantized env types stay over-reserved by f16 (-> None)."""
    e = os.environ if env is None else env
    f16_bpe = _kv_bytes_per_elem("f16")
    heaviest: Optional[str] = None
    heaviest_bpe = f16_bpe
    for var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"):
        raw = (e.get(var) or "").strip().lower()
        if not raw:
            continue
        bpe = _kv_bytes_per_elem(raw)
        if bpe > heaviest_bpe:
            heaviest, heaviest_bpe = raw, bpe
    return heaviest


def _extra_args_main_cache_type_for_budget(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Heavier (max bytes/elem) of the explicit --cache-type-k/-v extras, or None.

    Extras are appended last and win per axis, so an asymmetric K=f32,V=f16 must be
    budgeted by its heavier axis. resolve_cache_type_kv returns only the last-wins
    single type, which under-reserves the heavier axis when the lighter one is last."""
    k, v = parse_cache_override_per_axis(extra_args)
    candidates = [c for c in (k, v) if c]
    if not candidates:
        return None
    return max(candidates, key = _kv_bytes_per_elem)


def _effective_main_cache_types(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[str, str]:
    """Effective main K/V cache types after environment and CLI precedence."""
    source_env = os.environ if env is None else env
    env_k = (source_env.get("LLAMA_ARG_CACHE_TYPE_K") or "f16").strip().lower()
    env_v = (source_env.get("LLAMA_ARG_CACHE_TYPE_V") or "f16").strip().lower()
    arg_k, arg_v = parse_cache_override_per_axis(args)
    return (
        (arg_k or env_k).strip().lower(),
        (arg_v or env_v).strip().lower(),
    )


def _planned_main_cache_types(
    cache_type_kv: Optional[str],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """Main K/V types the loader's managed flags and user extras will produce."""
    args = list(extra_args or ())
    emitted_type = _extra_args_main_cache_type_for_budget(args) or cache_type_kv
    if emitted_type:
        args = [
            "--cache-type-k",
            emitted_type,
            "--cache-type-v",
            emitted_type,
            *args,
        ]
    return _effective_main_cache_types(args, env)


def _auto_mode_drops_mtp(
    req_mode: Optional[str],
    size_b: Optional[float],
    *,
    has_separate_drafter: bool = False,
) -> bool:
    """Auto mode drops MTP below _MTP_MIN_SIZE_B for an embedded draft head
    (its per-token cost regresses there); a separate drafter (Gemma) is a tiny
    standalone model that still speeds up below 3B, so it never drops. Forced
    mtp / mtp+ngram engage regardless of size."""
    if has_separate_drafter:
        return False
    return req_mode == "auto" and size_b is not None and size_b < _MTP_MIN_SIZE_B


def _mla_mtp_auto_enabled() -> bool:
    """Whether Auto may pick embedded MTP for an MLA model (GLM-5.2/DeepSeek/Kimi).

    Off by default: llama.cpp's MLA/DSA MTP path keeps a duplicated full target-KV
    context and recomputes the sparse-attention indexer every draft step, so it runs
    ~2x slower than no speculation (GLM-5.2 bench: 27 vs 45 tok/s, flat across draft
    depth and 96-100% acceptance) -- the opposite of the vLLM/SGLang speedup on the
    same model. Set UNSLOTH_MLA_MTP_ENABLED=1 to let Auto promote MLA MTP again once
    that path is optimized upstream. Forced mtp / mtp+ngram ignore this gate."""
    return os.environ.get("UNSLOTH_MLA_MTP_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _extra_args_set_spec_type(extra_args: Optional[Iterable[str]]) -> bool:
    """User passed --spec-type / --spec-default? llama-server takes one
    --spec-type (comma-separated to chain), so suppress auto-emit."""
    return _extra_args_set_any_flag(extra_args, {"--spec-type", "--spec-default"})


# Layer-offload override detection. Single-sourced from llama_server_args, which
# also strips these (plus the MoE flags) from inherited extras; sharing the layer
# set keeps detection and stripping from drifting.
_GPU_OFFLOAD_OVERRIDE_FLAGS = _LAYER_OFFLOAD_FLAGS
_THREAD_OVERRIDE_FLAGS = frozenset({"-t", "--threads"})
# common_params defaults in the bundled llama.cpp runtime.
_DEFAULT_LLAMA_N_BATCH = 2048
_DEFAULT_LLAMA_N_UBATCH = 512
_LLAMA_ARG_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_LLAMA_ARG_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})
_LLAMA_ARG_AUTO_VALUES = frozenset({"auto", "-1"})
_LLAMA_ARG_TRUE_OR_AUTO_VALUES = _LLAMA_ARG_TRUE_VALUES | _LLAMA_ARG_AUTO_VALUES
_LLAMA_ARG_TRUE_FALSE_AUTO_VALUES = _LLAMA_ARG_TRUE_OR_AUTO_VALUES | _LLAMA_ARG_FALSE_VALUES


def _extra_args_set_any_flag(extra_args: Optional[Iterable[str]], flags: Collection[str]) -> bool:
    if not extra_args:
        return False
    for raw in extra_args:
        flag = _flag_name(str(raw))
        if flag in flags:
            return True
    return False


def _diffusion_manual_ngl(gpu_memory_mode: str, gpu_layers: int) -> Optional[int]:
    """The --ngl the diffusion runner would launch with, or None for its default.

    Only an explicit manual count reaches the child; Auto (-1) and Unsloth mode defer.
    Shared with both dedup guards, so an inert manual preference cannot loop.
    """
    return int(gpu_layers) if (gpu_memory_mode == "manual" and int(gpu_layers) >= 0) else None


def _scale_diffusion_required_gb(required_gb: float, ngl: int, n_layers: Optional[int]) -> float:
    """GPU share of a partially offloaded diffusion GGUF, for the training guard.

    Only the ``ngl`` resident layers compete for VRAM; sizing the whole file 409s a
    split that fits. An unknown layer count keeps the full estimate."""
    if not n_layers or n_layers <= 0 or ngl >= n_layers:
        return required_gb
    return required_gb * (ngl / n_layers)


def _source_declares_option(source: str, option: str) -> bool:
    """True iff ``source`` really registers ``option`` on an argparse parser.

    Parsed, not grepped: a substring test matches the flag in a comment and misses it
    when quoted differently. Falls back to a substring scan when the file will not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return option in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == option:
                return True
    return False


def _shim_supports_ngl(shim_cmd: List[str]) -> bool:
    """Whether the resolved diffusion shim accepts --ngl.

    The package floor still allows an unsloth_zoo without the flag and argparse exits
    on an unknown option, so emitting it blindly breaks the launch. Read the source
    rather than spawning a probe: the launch path is already slow.
    """
    # _find_diffusion_assets emits a file override ([python, <path>]) or the installed
    # package ([python, -m, <module>]). Key on the shape, not the suffix (an override may
    # be extensionless, .pyw or SHIM.PY), and probe that exact file: a sibling shim.py
    # answers for a different shim than argparse will run.
    if len(shim_cmd) >= 2 and "-m" not in shim_cmd:
        candidates = [Path(shim_cmd[-1])]
    else:
        candidates = []
        try:
            import importlib.util
            spec = importlib.util.find_spec("unsloth_zoo")
            if spec is not None and spec.submodule_search_locations:
                pkg_dir = Path(list(spec.submodule_search_locations)[0])
                candidates.append(pkg_dir / "diffusion_studio" / "shim.py")
        except Exception:
            return False
    for path in candidates:
        try:
            source = path.read_text(encoding = "utf-8", errors = "ignore")
        except OSError:
            continue
        if _source_declares_option(source, "--ngl"):
            return True
    return False


def _swa_full_from_args_or_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether llama.cpp receives the enable-only full-size SWA option."""
    if _extra_args_set_any_flag(extra_args, {"--swa-full"}):
        return True
    value = (os.environ if env is None else env).get("LLAMA_ARG_SWA_FULL")
    return value in _LLAMA_ARG_TRUE_VALUES


def _kv_offload_from_args(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Resolve llama.cpp's environment and last-wins KV-offload flags (default on):
    LLAMA_ARG_KV_OFFLOAD parses before argv, and ``-kvo`` / ``--kv-offload`` is the
    positive form, so a CLI re-enable beats a false env."""
    enabled = True
    value = (os.environ if env is None else env).get("LLAMA_ARG_KV_OFFLOAD")
    if value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    elif value in _LLAMA_ARG_TRUE_VALUES:
        enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvo", "--kv-offload"}:
            enabled = True
        elif flag in {"-nkvo", "--no-kv-offload"}:
            enabled = False
    return enabled


def _is_positive_int(value: Optional[str]) -> bool:
    """Whether ``value`` parses as an int > 0 (llama.cpp's std::stoi)."""
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def _pipeline_parallel_disabled_by_args(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_layers: Optional[int] = None,
) -> bool:
    """Whether the launch turns OFF llama.cpp's pipeline parallelism, i.e. the 4x in
    ``_CTX_COMPUTE_SPLIT_MULT``. llama-context.cpp requires ``-sm layer``, KV offload
    on and an empty tensor-override list (among other things); anything here is back to
    1x, and charging the multiplier anyway would waste context. Studio passes none of
    it, but extra_args and the environment can.

    ``-cmoe`` / ``-ncmoe`` count: both push into ``tensor_buft_overrides`` exactly like
    ``-ot`` (common/arg.cpp). ``-otd`` does not - it targets the draft model.
    LLAMA_ARG_SPLIT_MODE is not read here: the launch pops a non-layer inherited value
    on the layer path, so it never reaches the child. Ambiguous input resolves to
    "still pipelined" - over-reserving only costs context, under-reserving OOMs the
    split. ``n_layers`` is the GGUF block count for the ``-ngl`` check below; None/0
    means unknown."""
    source_env = os.environ if env is None else env
    if source_env.get("LLAMA_ARG_OVERRIDE_TENSOR"):
        return True
    if source_env.get("LLAMA_ARG_CPU_MOE") in _LLAMA_ARG_TRUE_VALUES:
        return True
    if _is_positive_int(source_env.get("LLAMA_ARG_N_CPU_MOE")):
        return True
    if not _kv_offload_from_args(extra_args, env = env):
        return True
    # -sm none/row keep the layer path and pass through, so they really do disable
    # it. -sm tensor does not: the layer branch is only reached after a downgrade,
    # which strips the flag (strip_split_mode_only) and leaves the child on layer.
    split_mode = parse_split_mode_override(extra_args)
    if split_mode is not None and split_mode.strip().lower() not in {"layer", "tensor"}:
        return True
    # Pipelining needs n_gpu_layers > n_layer_all, so a user -ngl (appended after
    # Studio's -ngl -1, last-wins) that cannot exceed the count loads a prefix and
    # turns it off. Negative (all layers) or above the count keeps the step, even
    # in between: n_layers is block_count and can undercount n_layer_all (MTP layers).
    if n_layers:
        try:
            gpu_layers = parse_gpu_layers_override(extra_args)
        except ValueError:
            gpu_layers = None  # malformed: unknown, so keep the step
        if gpu_layers is not None and 0 <= gpu_layers <= n_layers:
            return True
    values = [str(raw) for raw in extra_args or ()]
    for i, raw in enumerate(values):
        flag = _flag_name(raw)
        # Any -ot counts even if it matches nothing: the list need only be non-empty.
        if flag in {"-ot", "--override-tensor", "-cmoe", "--cpu-moe"}:
            return True
        if flag in {"-ncmoe", "--n-cpu-moe"}:
            # -ncmoe 0 pushes no override, so it leaves pipelining on.
            _, eq, inline = raw.partition("=")
            if _is_positive_int(inline if eq else (values[i + 1] if i + 1 < len(values) else "")):
                return True
    return False


def _kv_unified_from_args(
    extra_args: Optional[Iterable[str]],
    default: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins unified KV flags."""
    enabled = False
    value = (os.environ if env is None else env).get("LLAMA_ARG_KV_UNIFIED")
    if value in _LLAMA_ARG_TRUE_VALUES:
        enabled = True
    elif value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    if default:
        # Studio's managed --kv-unified flag is appended after environment
        # parsing and before user extras.
        enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvu", "--kv-unified"}:
            enabled = True
        elif flag in {"-no-kvu", "--no-kv-unified"}:
            enabled = False
    return enabled


def _flash_attn_enabled_from_args(
    args: Optional[Iterable[str]],
    default: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins flash-attention settings."""
    enabled = default
    # llama.cpp applies LLAMA_ARG_FLASH_ATTN before parsing argv (arg.cpp set_env),
    # so the CLI still wins. --flash-attn has no args_neg, so no LLAMA_ARG_NO_ twin.
    value = (os.environ if env is None else env).get("LLAMA_ARG_FLASH_ATTN")
    if value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
        enabled = True
    values = [str(arg) for arg in args] if args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fa", "--flash-attn"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values) and values[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
            value = values[i + 1]
        if value in _LLAMA_ARG_FALSE_VALUES:
            enabled = False
        elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
            enabled = True
    return enabled


def _effective_spec_type(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """The last --spec-type label a launch carries: final CLI --spec-type (or
    --spec-default, which resolves non-MTP), else LLAMA_ARG_SPEC_TYPE. A display and
    comparison label only, NOT what llama.cpp enables: types accumulate there, so ask
    _extra_args_requests_mtp whether MTP is on. None if neither sets it."""
    args = [str(a) for a in extra_args] if extra_args else []
    cli_present = False
    cli_value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag == "--spec-default":
            cli_present = True
            cli_value = "default"
            continue
        if flag != "--spec-type":
            continue
        cli_present = True
        cli_value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    if cli_present:
        return cli_value
    return (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")


def _child_spec_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Mapping[str, str]:
    """The spec env the child will actually see: empty unless extras name a
    --spec-type, because the launch scrubs LLAMA_ARG_SPEC_* whenever Unsloth owns the
    spec block. Predict from this, not os.environ, or you describe a server that is
    not the one being started."""
    if not _extra_args_set_spec_type(extra_args):
        return {}
    return os.environ if env is None else env


def _accumulated_spec_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> set:
    """Every speculative type a launch carries, lowercased. Types accumulate rather
    than replace: llama.cpp applies the env first through the same handler and every
    --spec-type inserts at the end, so a later flag cannot clear an earlier one (even
    --spec-type none only appends NONE). Look at all of them, not just the last."""
    values = []
    args = [str(a) for a in extra_args] if extra_args else []
    for i, raw in enumerate(args):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) == "--spec-type":
            values.append(inline if eq else (args[i + 1] if i + 1 < len(args) else ""))
    env_value = (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")
    if env_value:
        values.append(env_value)
    return {p.strip().lower() for v in values for p in v.split(",") if p.strip()}


def _extra_args_requests_mtp(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if MTP lands in llama.cpp's spec-type vector, so the budget must reserve
    for it and the slots must clamp. Per _accumulated_spec_types, ANY source counts."""
    return bool(_accumulated_spec_types(extra_args, env) & {"mtp", "draft-mtp"})


def _extra_args_requests_dspark(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if DSpark lands in llama.cpp's accumulated spec-type vector."""
    return "draft-dspark" in _accumulated_spec_types(extra_args, env)


@functools.lru_cache(maxsize = 1)
def _metal_device_is_paravirtual() -> bool:
    """True when Metal is a virtualised Apple GPU, whose offload can corrupt output.

    Seen on our own macos-14/15 runners: offload returned an ordered walk through the
    character set while gpu_layers=0 on the same model, quant and binary was coherent,
    and MLX stayed correct. It does not reproduce on every build or quant (gemma-3-270m
    Q4_K_M on b9000 and b10090 was byte identical at gpu_layers 0 and 99), so
    UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 keeps the GPU on a VM known good. Physical Apple
    Silicon never matches.

    Three probes, cheapest first. MLX names the device outright (~40 ms) but is not on
    every Mac. hw.model then answers the headless case that matters most: measured on
    macos-14/15, SPDisplaysDataType returns zero bytes on a VM with no display so it
    cannot see a cloud or CI Mac at all, while hw.model still reports VirtualMac2,1 or
    VMM-x86_64 against Mac<n>,<n> on real hardware. SPDisplaysDataType stays last for
    desktop VMs, which do have a display and do name the paravirtual chipset.
    """
    if sys.platform != "darwin":
        return False
    if os.environ.get("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("UNSLOTH_ALLOW_PARAVIRTUAL_METAL set: keeping Metal offload.")
        return False
    name = ""
    try:
        import mlx.core as mx
        name = str(mx.device_info().get("device_name") or "")
    except Exception:
        name = ""
    if "paravirtual" not in name.lower():
        for _probe_cmd, _match in (
            # VMM-x86_64 is the Intel-guest identifier and contains neither
            # "virtual" nor "paravirtual"; upstream ships Metal on macOS x64 too.
            (["sysctl", "-n", "hw.model"], ("virtual", "vmm-")),
            (["system_profiler", "SPDisplaysDataType"], ("paravirtual",)),
        ):
            try:
                probe = subprocess.run(
                    _probe_cmd,
                    capture_output = True,
                    text = True,
                    timeout = 30,
                    encoding = "utf-8",
                    errors = "replace",
                )
            except Exception:
                continue
            _out = (probe.stdout or "").lower()
            if any(m in _out for m in _match):
                name = f"{name} {probe.stdout}".strip()
                break
    _low = name.lower()
    if "paravirtual" in _low or "virtual" in _low or "vmm-" in _low:
        logger.warning(
            "Metal device looks virtualised (%s). Offload can return corrupt output on "
            "paravirtual Apple GPUs, so GGUF inference will run on CPU. MLX is "
            "unaffected. Set UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU.",
            name.strip()[:120] or "unknown",
        )
        return True
    return False


def _paravirtual_probe_answered(server_caps: Mapping[str, object]) -> bool:
    """False when the --help probe failed, so every capability read as absent.

    Both companion flags predate the oldest build Studio can launch (--no-mmproj-offload
    is b5178, --gpu-layers-draft is from 2023, and the base argv already requires b6325),
    so "capability missing" means the probe did not answer, not that the build lacks it:
    one malformed inherited LLAMA_ARG_* makes llama-server --help exit non-zero and the
    all-false result is cached for the process. Dropping the projector or the drafter on
    that would be a self-inflicted outage, so drops require a conclusive probe and the
    pins cover the unanswered case.
    """
    return not server_caps.get("mtp_probe_inconclusive")


def _paravirtual_draft_ngl_flag(server_caps: Mapping[str, object]) -> Optional[str]:
    """The draft-layer flag to pin with, or None only when the build truly lacks one."""
    flag = server_caps.get("spec_draft_ngl_flag")
    if flag:
        return str(flag)
    # Unanswered probe: use the spelling every launchable build has had since 2023
    # rather than treating the drafter as unpinnable.
    if not _paravirtual_probe_answered(server_caps):
        return "--gpu-layers-draft"
    return None


def _paravirtual_mmproj_pinnable(server_caps: Mapping[str, object]) -> bool:
    """True when --no-mmproj-offload can be emitted, including on an unanswered probe."""
    return bool(
        server_caps.get("supports_no_mmproj_offload")
        or not _paravirtual_probe_answered(server_caps)
    )


_OVERRIDE_TENSOR_FLAGS: frozenset = frozenset(
    {
        "-ot",
        "--override-tensor",
        "-otd",
        "--override-tensor-draft",
        "--spec-draft-override-tensor",
    }
)


def _paravirtual_strip_gpu_overrides(
    extra_args: Optional[Iterable[str]], *, log_dropped: bool = True
) -> Optional[List[str]]:
    """Drop --override-tensor entries that place weights back on the corrupt device.

    ``-ot`` is the one placement flag ``--gpu-layers 0`` cannot answer: llama.cpp applies
    the override while selecting each weight's buffer type, before any layer is assigned
    to a device, so ``-ot ".*=Metal"`` puts weights on the virtualised GPU whatever the
    layer count says. It also accumulates rather than replaces, so a later flag cannot
    neutralise it; the entry has to go. ``-otd`` does the same for the drafter and would
    defeat the draft-layer pin.

    Only non-CPU targets are dropped: ``-ot exps=CPU`` moves weights the same way this
    fallback does, so stripping it would slow the very load it rescues.

    ``log_dropped=False`` for the duplicate-load comparators, which only inspect a
    request and must not narrate a drop no launch is performing, on every /load.
    """
    if not extra_args:
        return extra_args if extra_args is None else list(extra_args)
    args = [str(a) for a in extra_args]
    out: List[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        # _flag_name mirrors llama.cpp's underscore folding (arg.cpp does
        # std::replace(_ -> -) before matching), so --override_tensor is the same flag.
        base, _, inline = tok.partition("=")
        if _flag_name(base) in _OVERRIDE_TENSOR_FLAGS:
            value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
            targets = [
                part.rsplit("=", 1)[-1].strip().lower() for part in value.split(",") if "=" in part
            ]
            if targets and all(t == "cpu" for t in targets):
                out.append(tok)
                if not inline and i + 1 < len(args):
                    out.append(args[i + 1])
            elif log_dropped:
                logger.warning(
                    "Dropping %s %s: this Mac's Metal device is virtualised, so an "
                    "override that keeps weights on it would return corrupt output.",
                    base,
                    value,
                )
            i += 1 if inline else 2
            continue
        out.append(tok)
        i += 1
    return out


class ParavirtualPlacement(NamedTuple):
    """A load request's GPU placement after the virtualised-Metal CPU pin."""

    gpu_memory_mode: Literal["auto", "manual"]
    gpu_layers: int
    tensor_parallel: bool
    tensor_split: Optional[List[float]]
    n_cpu_moe: int
    extra_args: Optional[List[str]]


def paravirtual_normalized_request(
    *,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
    gpu_layers: int = -1,
    tensor_parallel: bool = False,
    tensor_split: Optional[List[float]] = None,
    n_cpu_moe: int = 0,
    extra_args: Optional[Iterable[str]] = None,
    log_dropped: bool = True,
) -> ParavirtualPlacement:
    """Map a requested placement to the one a virtualised Metal device launches.

    Single source of truth for the CPU pin, shared by the launch (load_model) and by
    BOTH duplicate-load comparators (``_already_in_target_state`` and the route's
    ``_request_matches_loaded_settings``). The launch records what it normalized, so a
    comparator judging the RAW request would mismatch its own state and tear down a
    healthy CPU server on every repeat Apply -- and, with an unbounded reload loop, 409
    or cancel a live generation.

    Pure, unconditional (callers gate on ``_metal_device_is_paravirtual()``) and
    idempotent, so re-applying it to a normalized request is a no-op.

    Deliberately NOT keyed on the request already being manual/0: that places the main
    model on CPU, but the extras are the caller's, and ``-ot ".*=Metal"`` is applied
    while each weight's buffer type is chosen, before any layer is assigned, so it puts
    weights back on the corrupt device -- and the route strips only the ``-ngl`` family.

    ``log_dropped=False`` for the comparators, which only inspect a request and must not
    narrate a rewrite no launch is performing.
    """
    normalized = extra_args if extra_args is None else list(extra_args)
    if normalized:
        # Extras are appended AFTER the managed --gpu-layers 0 and llama.cpp is
        # last-wins, so an `-ngl 99` would re-enable the corrupt offload. The route
        # strips these only in manual mode, so an Auto request still carries them.
        normalized = strip_shadowing_flags(
            normalized,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_offload = True,
        )
        # strip_offload covers only the layer-count family; an --override-tensor aimed
        # at Metal survives it and places weights before any layer assignment.
        normalized = _paravirtual_strip_gpu_overrides(normalized, log_dropped = log_dropped)
    return ParavirtualPlacement(
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        # Nothing is left on the GPU to split, and the launch drops these anyway;
        # normalize here too so the recorded state and the next request agree.
        tensor_parallel = False,
        tensor_split = None,
        n_cpu_moe = 0,
        extra_args = normalized,
    )


def _paravirtual_split_mode_pin(extra_args: Optional[Iterable[str]]) -> List[str]:
    """Flags that neutralise a pass-through ``--split-mode`` on a virtualised Mac.

    A user ``--split-mode`` is deliberately allowed (the Tensor Parallelism toggle
    cannot express row/none), but the paravirtual fallback has already pinned this load
    to CPU, and the flag is NOT inert at ``--gpu-layers 0``: llama.cpp builds a
    buffer-type list for every device in ``model->devices`` before any layer is
    assigned, and n_gpu_layers never reaches that step. So ``-sm row`` throws "device
    <X> does not support split buffers" on every backend except SYCL
    (``make_gpu_buft_list``, llama-model.cpp, since ggml-org/llama.cpp#24216 removed
    CUDA's split buffers), and ``-sm tensor`` throws "LLAMA_SPLIT_MODE_TENSOR not
    implemented for architecture" for every arch ``llm_arch_supports_sm_tensor``
    excludes. Either turns the CPU rescue into a server that refuses to start, and the
    zero-offload mask cannot save it: it writes only CUDA/HIP visibility vars, so the
    Metal device stays in the list on the one platform this fallback fires on.

    Overridden rather than stripped: llama.cpp is last-wins, so ``layer`` (the default,
    and a provable no-op at ``--gpu-layers 0`` -- llama-model.cpp computes the split
    points but never indexes them once ``act_gpu_layers`` is 0) neutralises the mode
    without rewriting ``extra_args``. The route's duplicate-load comparator compares
    stored extras verbatim and the UI does not round-trip the extras box, so stripping
    would leave the two permanently unequal and turn every later Apply into a real model
    swap. ``--tensor-split`` needs neither treatment for the same never-indexed reason.

    Returns [] when extras set no mode or already set ``layer``.
    """
    mode = (parse_split_mode_override(extra_args) or "").strip().lower()
    if not mode or mode == "layer":
        return []
    logger.warning(
        "Overriding --split-mode %s with the default layer split: this Mac's "
        "Metal device is virtualised, so the model runs on CPU and a GPU split "
        "mode would only fail the load.",
        mode,
    )
    return ["--split-mode", "layer"]


def _extra_args_requests_separate_draft(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if a non-MTP model draft mode (draft-simple/draft-eagle3) is selected, which
    loads a separate draft model the budget must reserve (draft-mtp ->
    _extra_args_requests_mtp; ngram-* load no model). Accumulating for the same reason as
    _extra_args_requests_mtp: a later --spec-default or --spec-type cannot clear an
    inherited draft-simple, so reading only the last under-reserves the model it loads."""
    return bool(
        _accumulated_spec_types(extra_args, env) & {"draft-simple", "draft-eagle3", "draft-dspark"}
    )


def _extra_args_spec_draft_n_max(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Draft depth from extras (``--spec-draft-n-max`` or legacy ``--draft-max``), else None."""
    if not extra_args:
        return None
    args = [str(a) for a in extra_args]
    found: Optional[int] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in ("--spec-draft-n-max", "--draft-max"):
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            found = int(value)
        except (TypeError, ValueError):
            continue
    return found


def _extra_args_mtp_draft_path(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Separate drafter path from extras (local --model-draft/-md or HF
    --spec-draft-hf/-hfd/...), else the LLAMA_ARG_SPEC_DRAFT_MODEL/_HF_REPO env,
    else None. An HF repo isn't a local file, so the budget can't size it (falls
    back to the flat reserve), but recognizing it avoids sizing the wrong one."""
    flags = {
        "--model-draft",
        "--spec-draft-model",
        "-md",
        "--spec-draft-hf",
        "-hfd",
        "-hfrd",
        "--hf-repo-draft",
    }
    args = [str(a) for a in extra_args] if extra_args else []
    found: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in flags:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if value and not value.startswith("-"):
            found = value
    if found is not None:
        return found
    e = os.environ if env is None else env
    return (
        e.get("LLAMA_ARG_SPEC_DRAFT_MODEL")
        or e.get("LLAMA_ARG_SPEC_DRAFT_HF_REPO")
        # Pre-b8955 spellings, live between the launchable floor (2025-08-30) and the
        # rename (2026-04-28).
        or e.get("LLAMA_ARG_MODEL_DRAFT")
        or e.get("LLAMA_ARG_HFD_REPO")
        or None
    )


def _extra_args_draft_cache_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[Optional[str], Optional[str]]:
    """Draft KV cache types (k_type, v_type), each from extras else the
    LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K/_V env, else None (f16). K and V are
    independent: a one-sided override must not apply to both."""
    args = [str(a) for a in extra_args] if extra_args else []
    k_flags = {"--cache-type-k-draft", "--spec-draft-type-k", "-ctkd"}
    v_flags = {"--cache-type-v-draft", "--spec-draft-type-v", "-ctvd"}
    k_type: Optional[str] = None
    v_type: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in k_flags and flag not in v_flags:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if not value or value.startswith("-"):
            continue
        if flag in k_flags:
            k_type = value
        else:
            v_type = value
    e = os.environ if env is None else env
    if k_type is None:
        k_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K") or None
    if v_type is None:
        v_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V") or None
    return k_type, v_type


def _extra_args_draft_offloaded_to_cpu(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if the SEPARATE draft model is on CPU (so the budget must not charge
    its weights+KV): --spec-draft-ngl 0, or --spec-draft-device naming only
    cpu/none, else the LLAMA_ARG_N_GPU_LAYERS_DRAFT env the child honors (the
    device flag has no env). An embedded MTP head follows the main -ngl, so these
    draft-only flags don't move it. Last-wins, so only each flag's final value counts."""
    ngl_flags = {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}
    args = [str(a) for a in extra_args] if extra_args else []
    last_ngl: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if flag in ngl_flags:
            last_ngl = value
    if last_ngl is None:
        last_ngl = (os.environ if env is None else env).get("LLAMA_ARG_N_GPU_LAYERS_DRAFT")
    if last_ngl is not None:
        try:
            if int(last_ngl) == 0:
                return True
        except (TypeError, ValueError):
            pass
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is not None:
        devs = [d.strip().lower() for d in last_dev.split(",") if d.strip()]
        if devs and all(d in ("cpu", "none") for d in devs):
            return True
    return False


def _extra_args_device(
    extra_args: Optional[Iterable[str]], flags: Collection[str]
) -> Optional[str]:
    """Return the last value for any device flag in ``flags``."""
    args = [str(a) for a in extra_args] if extra_args else []
    value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag in flags:
            value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    return value


def _extra_args_main_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit main-device value, if any."""
    return _extra_args_device(extra_args, {"--device", "-dev"})


def _extra_args_draft_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit draft-device value, if any."""
    return _extra_args_device(extra_args, {"--spec-draft-device", "-devd", "--device-draft"})


def _extra_args_draft_device_pin(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return a GPU draft-device override; cpu/none do not conflict with a pin."""
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is None:
        return None
    devs = [d.strip() for d in last_dev.split(",") if d.strip()]
    if not devs or all(d.lower() in ("cpu", "none") for d in devs):
        return None
    return last_dev


def _extra_args_n_ubatch(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_ctx: Optional[int] = None,
) -> Optional[int]:
    """Effective ubatch after llama.cpp normalizes it, or None at defaults."""
    values = {
        "batch": _DEFAULT_LLAMA_N_BATCH,
        "ubatch": _DEFAULT_LLAMA_N_UBATCH,
    }
    source_env = os.environ if env is None else env
    overridden = False
    for key, env_name in (
        ("batch", "LLAMA_ARG_BATCH"),
        ("ubatch", "LLAMA_ARG_UBATCH"),
    ):
        raw = source_env.get(env_name)
        if raw:
            try:
                values[key] = int(raw)
                overridden = True
            except (TypeError, ValueError):
                pass

    args = [str(a) for a in extra_args] if extra_args else []
    flags = {
        "-b": "batch",
        "--batch-size": "batch",
        "-ub": "ubatch",
        "--ubatch-size": "ubatch",
    }
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        key = flags.get(flag)
        if key is None:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            values[key] = int(value)
            overridden = True
        except (TypeError, ValueError):
            continue
    if not overridden:
        return None

    # common_params stores signed values, then llama_context_params converts
    # them to uint32_t. A zero ubatch means "use batch"; the context then caps
    # ubatch at batch size.
    batch = values["batch"] & 0xFFFFFFFF
    raw_ubatch = values["ubatch"]
    ubatch = batch if raw_ubatch == 0 else raw_ubatch & 0xFFFFFFFF
    effective = min(batch, ubatch)
    if n_ctx is not None and n_ctx > 0:
        effective = min(effective, n_ctx)
    return effective


def _build_ngram_mod_flags(
    caps: Optional[dict],
    n_match: int = 24,
    n_min: int = 48,
    n_max: int = 64,
) -> list[str]:
    """Emit the right ngram-mod knob flags for the running llama-server.

    Post-rename builds expose ``--spec-ngram-mod-n-{match,min,max}``;
    pre-rename builds expose legacy ``--spec-ngram-size-n`` /
    ``--draft-min`` / ``--draft-max``. ``caps`` comes from
    ``probe_server_capabilities``; ``ngram_mod_flavor`` says which set is
    real (vs a removal-stub). Returns ``[]`` when neither is available so
    the caller can drop ngram-mod entirely.
    """
    flavor = caps.get("ngram_mod_flavor") if caps else None
    if flavor == "new":
        return [
            "--spec-ngram-mod-n-match",
            str(n_match),
            "--spec-ngram-mod-n-min",
            str(n_min),
            "--spec-ngram-mod-n-max",
            str(n_max),
        ]
    if flavor == "legacy":
        # Pre-rename llama.cpp: same knobs lived under --spec-ngram-size-n
        # (lookup length) and generic --draft-min / --draft-max (N range).
        return [
            "--spec-ngram-size-n",
            str(n_match),
            "--draft-min",
            str(n_min),
            "--draft-max",
            str(n_max),
        ]
    return []


# Canonical Speculative Decoding modes exposed by the Unsloth chat UI.
# Dropdown renders six (auto, mtp, dspark, ngram, mtp+ngram, off); the load API
# also accepts legacy values the original Switch and external callers emit
# (default, draft-mtp, ngram-mod, ngram-simple).
_CANONICAL_SPEC_MODES = {"auto", "mtp", "dspark", "ngram", "mtp+ngram", "off", "ngram-simple"}
_LEGACY_SPEC_MODE_MAP = {
    "default": "auto",
    "draft-mtp": "mtp",
    "draft-dspark": "dspark",
    "ngram-mod": "ngram",
}


def _canonicalize_spec_mode(value):
    """Map any accepted ``speculative_type`` input onto a canonical mode.

    Returns ``auto``, ``mtp``, ``dspark``, ``ngram``, ``mtp+ngram``, ``off``,
    ``ngram-simple``, or ``None`` (callers treat ``None`` as ``auto``).
    Unknown strings collapse to ``auto`` so a stale UI value or typo falls
    back to the safe platform-aware path.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if stripped in _CANONICAL_SPEC_MODES:
        return stripped
    if stripped in _LEGACY_SPEC_MODE_MAP:
        return _LEGACY_SPEC_MODE_MAP[stripped]
    # Old persisted state emits llama.cpp comma-chains e.g.
    # "ngram-mod,draft-mtp"; collapse the most common one explicitly.
    pieces = [p.strip() for p in stripped.split(",") if p.strip()]
    has_mtp = any(p in ("mtp", "draft-mtp") for p in pieces)
    has_ngram = any(p in ("ngram", "ngram-mod") for p in pieces)
    if has_mtp and has_ngram:
        return "mtp+ngram"
    if has_mtp:
        return "mtp"
    if has_ngram:
        return "ngram"
    return "auto"


def _backfill_usage_from_timings(usage, timings):
    """Synthesize ``usage`` from llama-server's ``timings`` when the
    OpenAI-style usage block is missing or reports zero tokens.

    The Unsloth chat UI computes generation t/s from
    ``meta.usage.completion_tokens / totalStreamTime``. llama-server always
    populates ``timings.predicted_n`` (true decoded count) and
    ``timings.prompt_n``, but the final SSE chunk's ``usage`` can be absent
    or zero on some server builds / streaming configs, making the UI fall
    back to wall-clock t/s and dilute speculative-decoding speedups.
    """
    if not timings:
        return usage
    if usage and usage.get("completion_tokens"):
        return usage
    predicted_n = timings.get("predicted_n")
    prompt_n = timings.get("prompt_n")
    if predicted_n is None and prompt_n is None:
        return usage
    out = dict(usage or {})
    if not out.get("completion_tokens") and predicted_n is not None:
        out["completion_tokens"] = predicted_n
    if not out.get("prompt_tokens") and prompt_n is not None:
        out["prompt_tokens"] = prompt_n
    out["total_tokens"] = int(out.get("prompt_tokens") or 0) + int(
        out.get("completion_tokens") or 0
    )
    return out


# Host RAM to leave free on an integrated GPU, matching llama.cpp's own --fit
# margin (default 1024 MiB per device). ggml reports an iGPU's "VRAM" as shared
# system RAM, so hold back the same margin rather than inventing a larger one.
_IGPU_HOST_RESERVE_MIB = 1024


def _apply_igpu_host_reserve_mib(free_mib: int, is_igpu: bool) -> int:
    """Reserve host headroom on an integrated (shared-memory) Vulkan GPU.

    An iGPU's reported free "VRAM" is really free system RAM, so sizing
    context/offload against all of it would push the host into swap or the OOM
    killer. Leave the same margin llama.cpp's --fit uses. ``is_igpu`` comes from
    ggml's device type, so a discrete card is never touched; only ever reduces.
    """
    if not is_igpu:
        return free_mib
    return max(0, free_mib - _IGPU_HOST_RESERVE_MIB)


def _resolve_llama_binary(binary: str) -> Path:
    """Resolve a managed symlink or shell entrypoint to the real server."""
    # Managed entrypoints are symlinks or shell wrappers into build/bin.
    resolved = Path(binary).resolve()
    try:
        with open(resolved, "rb") as _f:
            _head = _f.read(256)
        if _head.startswith(b"#!"):
            _m = re.search(r'exec "\$\(dirname "\$0"\)/([^"]+)"', _head.decode("utf-8", "ignore"))
            if _m:
                return (resolved.parent / _m.group(1)).resolve()
    except OSError:
        pass
    return resolved


def _llama_lib_dir(binary: str) -> Path:
    return _resolve_llama_binary(binary).parent


def _lib_dir_has_ggml_backend(lib_dir: Path, backend: str) -> bool:
    """Match an exact or versioned ggml backend library soname."""
    stem = f"ggml-{backend}.dll" if sys.platform == "win32" else f"libggml-{backend}.so"
    try:
        return any(
            f.name == stem or f.name.startswith(stem + ".")
            for f in lib_dir.iterdir()
            if f.is_file()
        )
    except OSError:
        return False


def _is_external_link(path: Path) -> bool:
    """True when ``path`` is a --with-llama-cpp-dir local link: a POSIX symlink
    or a Windows directory junction / reparse point. Such a link resolves into
    the user's own llama.cpp checkout, which Unsloth does not own."""
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    if os.name == "nt":
        try:
            import stat
            attrs = os.lstat(path).st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, AttributeError):
            return False
    return False


# Inkling's template takes a numeric thinking-effort dial (0..0.99) and its
# float() coercion turns unrecognized named levels into 0, i.e. no thinking.
# Map OpenAI-style names to the values the model was trained on. Module-level
# so duck-typed engine stand-ins in tests do not need the attribute.
_INKLING_REASONING_EFFORT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}


def _coerce_reasoning_effort(architecture, kwargs: dict) -> dict:
    if architecture == "inkling":
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str):
            mapped = _INKLING_REASONING_EFFORT.get(effort.strip().lower())
            if mapped is not None:
                kwargs["reasoning_effort"] = mapped
    return kwargs


class CountAborted(Exception):
    """A token count stood down mid-flight because its answer stopped mattering."""


class LlamaCppBackend:
    """Manages a llama-server subprocess for GGUF model inference.

    Lifecycle:
        1. load_model(): start llama-server with the GGUF file
        2. generate_chat_completion(): proxy to /v1/chat/completions, stream back
        3. unload_model(): terminate the subprocess
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_identifier: Optional[str] = None
        self._gguf_path: Optional[str] = None
        self._hf_repo: Optional[str] = None
        # Separate MTP drafter launched with the current model; reload-dedup
        # key so a drafter that appears next to the weights forces a reload.
        self._mtp_draft_path: Optional[str] = None
        # Drafter this load resolved then deliberately suppressed (virtualised Metal
        # with no draft-layer flag). The dedup keys on the drafter the caller asked
        # for, still on disk, so without this the detected path never matches the
        # launched None and every repeat Apply tears down a healthy server.
        self._mtp_draft_suppressed_path: Optional[str] = None
        # Why MTP was disabled on the last load that asked for it (auto on an
        # MTP model, or forced mtp / mtp+ngram), else None. Drives the "update
        # llama.cpp" hint in the UI. "binary_no_mtp" / "binary_outdated" ->
        # a newer prebuilt would help; "runtime_error" -> it may not.
        self._spec_fallback_reason: Optional[str] = None
        self._spec_drafter_kind: Optional[str] = None
        self._dspark_sidecar_absent: bool = False
        # Set after an auto-Vulkan crash recovers with all devices disabled.
        self._cpu_fallback_reason: Optional[str] = None
        self._cpu_fallback_runtime = None
        self._cpu_fallback_runtime_source: Optional[str] = None
        self._hf_variant: Optional[str] = None
        self._is_vision: bool = False
        # Block-diffusion model (e.g. DiffusionGemma): served by the diffusion
        # runner, not llama-server. Set from the GGUF architecture at load.
        self._architecture: Optional[str] = None
        self._is_diffusion: bool = False
        self._diffusion_visual_bin: Optional[str] = None
        self._healthy = False
        self._load_rss_hwm = (None, 0)  # (pid, peak VmRSS) for load_progress
        self._stats_logger = None  # vLLM-style engine-stats poller, set on load
        # Set by _classify_gpu_offload after _wait_for_health.
        self._gpu_offload_active: Optional[bool] = None
        # Diffusion only: the split the running child was asked for.
        self._diffusion_requested_ngl: Optional[int] = None
        self._context_length: Optional[int] = None
        self._effective_context_length: Optional[int] = None
        self._max_context_length: Optional[int] = None
        self._effective_parallel_slots: int = 1
        # --parallel the last load asked for, before any fit-time reduction.
        self._requested_n_parallel: int = 1
        self._chat_template: Optional[str] = None
        self._markup_tokens: list = []
        self._markup_profile = None
        self._chat_template_override: Optional[str] = None
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_style: str = "enable_thinking"
        self._reasoning_effort_levels: list = []
        self._supports_preserve_thinking: bool = False
        self._supports_tools: bool = False
        self._cache_type_kv: Optional[str] = None
        # Whether --split-mode tensor was applied on the active load.
        self._tensor_parallel: bool = False
        # GPU memory strategy applied on the active load ("auto"/"manual").
        self._gpu_memory_mode: str = "auto"
        # Manual-mode load options (echoed back so the UI round-trips them).
        self._gpu_layers: int = -1
        # MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none.
        self._n_cpu_moe: int = 0
        # Relative model share per GPU (--tensor-split), in GPU order; None =
        # default (llama.cpp splits by free VRAM).
        self._tensor_split: Optional[List[float]] = None
        # User-picked physical GPU indices (None = automatic selection).
        self._gpu_ids: Optional[List[int]] = None
        # RAW requested GPU pin, before the fit narrowed it. self._gpu_ids records the
        # EFFECTIVE (fit-narrowed) pin for /status; dedupe compares this raw value so a
        # [0, 1] narrowed to [0] and re-sent as [0, 1] still matches (#7239).
        self._requested_gpu_ids: Optional[List[int]] = None
        # Layer load kept multi-GPU only to honor a downgraded tensor request, so a
        # later explicit tensor-off reloads instead of deduping to it (#6659).
        self._layer_preserves_tensor_intent: bool = False
        self._reasoning_default: bool = True
        self._speculative_type: Optional[str] = None
        # Canonical UI-facing mode the user requested
        # (auto/mtp/ngram/mtp+ngram/off/ngram-simple). Round-tripped through the
        # status API so the dropdown reflects the picked mode, not the resolved
        # flag set (auto on a 27B MTP GGUF resolves to draft-mtp but reads "Auto").
        self._requested_spec_mode: Optional[str] = None
        # User --spec-draft-n-max override (None = platform default).
        self._spec_draft_n_max: Optional[int] = None
        # KV-cache estimation fields (populated by _read_gguf_metadata)
        self._n_layers: Optional[int] = None
        # MoE metadata (populated by _read_gguf_metadata): expert count (>0 =
        # MoE) and leading dense-layer count (offsets --n-cpu-moe, which counts
        # from layer 0). See the n_moe_layers property.
        self._n_experts: Optional[int] = None
        self._leading_dense_block_count: Optional[int] = None
        self._n_kv_heads: Optional[int] = None
        self._n_kv_heads_by_layer: Optional[list[int]] = None
        self._n_heads: Optional[int] = None
        self._embedding_length: Optional[int] = None
        # For the compute-graph buffer estimate; vocab from the tokens array len.
        self._feed_forward_length: Optional[int] = None
        self._vocab_size: Optional[int] = None
        # Architecture-aware KV fields for 5-path estimation
        self._kv_key_length: Optional[int] = None
        self._kv_value_length: Optional[int] = None
        self._sliding_window: Optional[int] = None
        self._sliding_window_pattern: Optional[list[bool]] = None
        self._full_attention_interval: Optional[int] = None
        self._kv_lora_rank: Optional[int] = None
        self._key_length_mla: Optional[int] = None
        self._kv_key_length_swa: Optional[int] = None
        self._kv_value_length_swa: Optional[int] = None
        self._ssm_inner_size: Optional[int] = None
        self._ssm_state_size: Optional[int] = None
        self._ssm_conv_kernel: Optional[int] = None
        self._kda_head_dim: Optional[int] = None
        # Last N layers reuse earlier layers' KV and don't allocate their own
        # cache (Gemma 3n / Gemma 4: <arch>.attention.shared_kv_layers).
        self._shared_kv_layers: Optional[int] = None
        # MTP head count (llama.cpp #22673); >0 enables --spec-type draft-mtp.
        self._nextn_predict_layers: Optional[int] = None
        self._lock = threading.Lock()
        # Wraps load_model() end-to-end so concurrent loads serialise and never
        # coexist as two llama-server processes (#5401). RLock so MTP-crash
        # recovery can re-acquire it for its nested load_model.
        self._serial_load_lock = threading.RLock()
        # Serialises mid-session respawns so many generations hitting a killed
        # server trigger at most one reload (see _respawn_if_dead).
        self._respawn_lock = threading.Lock()
        # Bumped by every unload. load_model clears _cancel_event, so a respawn that
        # raced an unload needs a signal that survives the clear (see _respawn_if_dead).
        self._unload_epoch = 0
        # Set by the in-app updater while it swaps prebuilt binaries; load_model()
        # rejects fast so no server starts from a half-swapped binary.
        self._llama_update_in_progress = False
        # Last extra_args / requested n_ctx, preserved across unload so the chat
        # UI's /unload+/load Apply path can inherit them (#5401).
        # ``_extra_args_source`` records the (model_identifier, hf_variant) the
        # stored args came from so the route can refuse cross-model inheritance.
        self._extra_args: Optional[List[str]] = None
        # Extras as REQUESTED, before a launch-time rewrite (the virtualised-Metal
        # drafter drop strips the spec group), so the duplicate-load comparators can
        # compare requested-vs-requested like _requested_n_ctx does.
        self._requested_extra_args: Optional[List[str]] = None
        self._extra_args_source: Optional[tuple[str, Optional[str]]] = None
        self._requested_n_ctx: int = 0
        # Last healthy caller intent for crash recovery. Memory-only and never logged.
        self._last_load_intent: Optional[GgufLoadIntent] = None
        self._mtp_runtime_fallback_lock = threading.Lock()
        self._mtp_runtime_fallback_in_progress = False
        # Background watchdog so an MTP+tensor crash recovers even when no request
        # observes it (direct proxy endpoints, or nothing in flight).
        self._mtp_watchdog_thread: Optional[threading.Thread] = None
        self._mtp_watchdog_stop = threading.Event()
        # True when the launch actually runs MTP+tensor (Unsloth- or user/env-driven);
        # gates the probe, watchdog, and recovery so pass-through MTP is covered.
        self._mtp_runtime_fallback_active = False
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        # llama-server tee log (see _drain_stdout / _kill_process).
        self._llama_log_fh = None
        self._llama_log_path: Optional[Path] = None
        self._cancel_event = threading.Event()
        self._api_key: Optional[str] = None
        self._slot_save_dir: Optional[str] = None
        self._slot_save_binary: Optional[tuple[str, int]] = None
        # (gguf_identity, launch_fingerprint) snapshotted at load, so a later slot
        # save can tell whether the model files were swapped on disk since load.
        self._slot_loaded_identity: Optional[tuple] = None
        self._prompt_cache_disabled: bool = False
        self._swa_full: bool = False
        self._kv_cache_unified: bool = False
        self._n_ubatch: int = self._DEFAULT_N_UBATCH
        self._flash_attn_enabled: bool = True
        self._effective_cache_types: tuple[str, str] = ("f16", "f16")
        # Total KV allocation context across all slots. _effective_context_length
        # becomes the per-slot request limit after /props reconciliation.
        self._kv_cache_context_total: Optional[int] = None
        # True once a probe has completed; cleared on transient failure.
        self._is_audio: bool = False
        self._audio_type: Optional[str] = None
        self._audio_probed: bool = False
        # Audio INPUT capability (distinct from _is_audio, which is TTS output).
        self._has_audio_input: bool = False
        self._mmproj_has_audio: bool = False  # clip.has_audio_encoder, set at load
        # Monotonic timestamp set in _kill_process; read by load_model
        # to decide whether to wait for the VRAM reclaim to finish.
        self._last_kill_monotonic: float = 0.0

        _reaped = self._kill_orphaned_servers()
        if _reaped:
            # Reaped VRAM frees lazily; arm the settle wait so the first load
            # waits before ranking GPUs by free memory.
            self._last_kill_monotonic = time.monotonic()
        atexit.register(self._cleanup)

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._healthy

    @property
    def is_active(self) -> bool:
        """True if a llama-server process exists (loading or loaded)."""
        return self._process is not None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def _auth_headers(self) -> "Optional[dict[str, str]]":
        """Bearer header matching the --api-key direct-stream mode uses, else
        None (so unauthenticated llama-server calls don't get a spurious 401)."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return self._is_vision

    @property
    def is_diffusion(self) -> bool:
        """True when the loaded GGUF is a block-diffusion model (DiffusionGemma)."""
        return self._is_diffusion

    @property
    def swa_full(self) -> bool:
        """Whether the active llama-server received full-size SWA mode."""
        return self._swa_full

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def gguf_path(self) -> Optional[str]:
        return self._gguf_path

    @property
    def hf_repo(self) -> Optional[str]:
        """HF repo of the loaded model, or None for local/native file loads."""
        return self._hf_repo

    @property
    def mtp_draft_path(self) -> Optional[str]:
        return self._mtp_draft_path

    @property
    def mtp_draft_suppressed_path(self) -> Optional[str]:
        """Drafter the last load resolved and then dropped on purpose, else None.
        Duplicate-load checks accept it alongside mtp_draft_path so a drafter still on
        disk does not read as a settings change."""
        return self._mtp_draft_suppressed_path

    @property
    def spec_fallback_reason(self) -> Optional[str]:
        """Why MTP was disabled on the last MTP-requesting load, else None."""
        return self._spec_fallback_reason

    @property
    def extra_args(self) -> Optional[List[str]]:
        """Extra llama-server flags from the last load (a copy). None =
        never set, [] = explicitly cleared. Used by the route for
        inheritance."""
        return list(self._extra_args) if self._extra_args is not None else None

    @property
    def requested_extra_args(self) -> Optional[List[str]]:
        """Extras the last load was INVOKED with (a copy), before any launch-time
        rewrite; falls back to the launched list when nothing was rewritten. Keeps a
        rewrite the caller cannot see (the virtualised-Metal drafter drop) from reading
        as a settings change in the duplicate-load comparators."""
        stored = (
            self._requested_extra_args
            if self._requested_extra_args is not None
            else self._extra_args
        )
        return list(stored) if stored is not None else None

    @property
    def requested_n_ctx(self) -> int:
        """n_ctx the last load was invoked with (not the effective cap).
        0 means Auto. Used by the route to detect Auto-vs-explicit flips."""
        return self._requested_n_ctx

    @property
    def extra_args_source(self) -> Optional[tuple[str, Optional[str]]]:
        """(model_identifier, hf_variant) the stored extra_args came from.
        ``None`` if no extras have ever been recorded. Used by the route
        to refuse cross-model inheritance (#5401)."""
        return self._extra_args_source

    @property
    def context_length(self) -> Optional[int]:
        """Return the effective context length the server is running at."""
        return self._effective_context_length or self._context_length

    @property
    def effective_parallel_slots(self) -> int:
        """Return the serving-slot count the active llama-server actually uses."""
        try:
            slots = int(getattr(self, "_effective_parallel_slots", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def requested_parallel_slots(self) -> int:
        """--parallel the last load asked for, before any fit-time reduction.
        The reload dedupe compares requested-vs-requested (like requested_n_ctx);
        the effective count would reload forever after a fitter reduction."""
        try:
            slots = int(getattr(self, "_requested_n_parallel", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def max_context_length(self) -> Optional[int]:
        """Return the largest context that fits on this hardware at load time.

        The UI's "safe zone" warning threshold: the ``_fit_context_to_vram``
        binary-search cap for the best GPU subset, or the 4096 fallback if the
        weights exceed 90% of every subset. The slider ceiling is
        ``native_context_length``; dragging above this triggers the warning.
        """
        return self._max_context_length or self._context_length

    @property
    def native_context_length(self) -> Optional[int]:
        """Return the model's native context length from GGUF metadata."""
        return self._context_length

    def _commit_effective_parallel_slots(self, n_parallel: int) -> None:
        try:
            slots = int(n_parallel)
        except (TypeError, ValueError):
            slots = 1
        self._effective_parallel_slots = max(1, slots)

    def _reset_effective_parallel_slots(self) -> None:
        self._effective_parallel_slots = 1
        # Cleared with the effective count so a stale value can't skew the dedupe.
        self._requested_n_parallel = 1

    @staticmethod
    def _read_rss_bytes(pid: int) -> Optional[int]:
        """Resident set size of ``pid`` in bytes, from /proc/<pid>/status (Linux).
        0 when the status has no VmRSS line (zombie / kernel thread); None where
        /proc is unavailable (macOS/Windows) or the value is unreadable."""
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # IndexError guards a "VmRSS:" line with no value column.
                        return int(line.split()[1]) * 1024  # kB -> bytes
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return 0  # readable but no VmRSS line

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        During warm-up llama-server mmaps weight shards into page cache before
        pushing layers to VRAM, a window where status only reports ``loading``
        and the UI spinner looks stuck for minutes on large MoEs. Samples
        ``/proc/<pid>/status VmRSS`` against the sum of GGUF shard sizes for a
        real progress bar. Returns ``None`` when no load is in flight.

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # VmRSS of the llama-server
                "bytes_total":  int,   # sum of shard file sizes
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }

        Linux-only; returns ``None`` where ``/proc/<pid>/status`` is unavailable.
        """
        proc = self._process
        if proc is None:
            return None
        pid = proc.pid
        if pid is None:
            return None

        # Sum shard sizes (primary + any extras alongside).
        bytes_total = 0
        gguf_path = self._gguf_path
        if gguf_path:
            primary = Path(gguf_path)
            try:
                if primary.is_file():
                    bytes_total += primary.stat().st_size
            except OSError:
                pass
            # Extra shards share the primary's prefix before the shard index.
            try:
                parent = primary.parent
                stem = primary.name
                m = _SHARD_RE.match(stem)
                prefix = m.group(1) if m else None
                if prefix and parent.is_dir():
                    prefix_lower = prefix.lower()
                    for sibling in parent.iterdir():
                        if (
                            sibling.is_file()
                            and sibling.name.lower().startswith(prefix_lower)
                            and sibling.name != stem
                            and sibling.suffix.lower() == ".gguf"
                        ):
                            try:
                                bytes_total += sibling.stat().st_size
                            except OSError:
                                pass
            except OSError:
                pass

        # VmRSS of the llama-server; None where /proc is unavailable.
        bytes_loaded = LlamaCppBackend._read_rss_bytes(pid)
        if bytes_loaded is None:
            return None

        # RSS climbs as weights page in, then drops once -ngl offloads them to
        # VRAM and the mmap pages are freed. Hold a per-process high-water mark
        # so the bar never regresses to ~8% mid-load (#5740).
        hwm_pid, hwm = getattr(self, "_load_rss_hwm", (None, 0))
        hwm = bytes_loaded if hwm_pid != pid else max(hwm, bytes_loaded)
        self._load_rss_hwm = (pid, hwm)
        bytes_loaded = hwm

        phase = "ready" if self._healthy else "mmap"
        fraction = 0.0
        if bytes_total > 0:
            fraction = min(1.0, bytes_loaded / bytes_total)
        # Once llama-server is healthy the load is complete by definition. With
        # layers offloaded to VRAM (-ngl) the process releases the mmap'd weight
        # pages, so VmRSS sinks back well below the shard total; the raw RSS
        # fraction would then report a partial (~8%) load indefinitely and freeze
        # a fraction-driven progress bar even though the model is ready (#5740).
        if self._healthy:
            if bytes_total > 0:
                bytes_loaded = bytes_total
            fraction = 1.0
        return {
            "phase": phase,
            "bytes_loaded": bytes_loaded,
            "bytes_total": bytes_total,
            "fraction": round(fraction, 4),
        }

    @property
    def chat_template(self) -> Optional[str]:
        return self._chat_template

    @property
    def markup_profile(self):
        """This model's own structural markers, from its GGUF template and vocabulary.

        None when neither could be read, which falls back to the curated patterns: a model
        we cannot profile stays fully swept rather than unprotected (#7066)."""
        # getattr, not attribute access: a lightweight backend double in a test never runs
        # __init__, and an unprofiled model must fall back rather than raise.
        cached = getattr(self, "_markup_profile", None)
        if cached is None:
            from core.inference.chat_template_helpers import model_markup
            cached = (
                model_markup(
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                    getattr(self, "_markup_tokens", None),
                ),
            )
            try:
                self._markup_profile = cached
            except AttributeError:
                pass
        return cached[0]

    @property
    def chat_template_override(self) -> Optional[str]:
        return self._chat_template_override

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_always_on(self) -> bool:
        return self._reasoning_always_on

    @property
    def reasoning_style(self) -> str:
        return self._reasoning_style

    @property
    def reasoning_effort_levels(self) -> list:
        """Discrete reasoning_effort levels the template offers (e.g. GLM-5.2's
        ['high', 'max']). Empty unless reasoning_style == 'enable_thinking_effort'."""
        return self._reasoning_effort_levels

    @property
    def supports_preserve_thinking(self) -> bool:
        return self._supports_preserve_thinking

    @property
    def reasoning_default(self) -> bool:
        return self._reasoning_default

    def _reasoning_kwargs(self, enable_thinking: bool) -> dict:
        if self._reasoning_style == "enable_thinking_effort":
            # GLM-5.2-style: enable_thinking is the on/off gate; when on, leave
            # the template's default effort (max) in place.
            return {"enable_thinking": enable_thinking}
        if self._reasoning_style == "reasoning_effort":
            return _coerce_reasoning_effort(
                getattr(self, "_architecture", None),
                {"reasoning_effort": "high" if enable_thinking else "low"},
            )
        return {"enable_thinking": enable_thinking}

    def _request_reasoning_kwargs(
        self,
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> Optional[dict]:
        """Build chat_template_kwargs from per-request reasoning fields.

        Merges the active model's reasoning style (``enable_thinking`` or
        ``reasoning_effort``) plus the independent ``preserve_thinking``
        kwarg when the template supports it.
        """
        kwargs: dict = {}
        # Always-on reasoning models hardcode <think> tags and don't consume
        # enable_thinking / reasoning_effort -- skip.
        if self._supports_reasoning and not self._reasoning_always_on:
            if self._reasoning_style == "enable_thinking_effort":
                # GLM-5.2-style: enable_thinking gates thinking on/off, and the
                # reasoning_effort level (e.g. 'high' | 'max') is only meaningful
                # while thinking is on. Disabling is enable_thinking=false; a raw
                # API caller can also disable via the OpenAI-style
                # reasoning_effort="none" sentinel. We never coerce off into a
                # 'low' effort the way gpt-oss does (those models genuinely
                # cannot disable).
                thinking_off = enable_thinking is False or reasoning_effort == "none"
                # A named effort level implies thinking on, so emit enable_thinking
                # even if the caller sent only reasoning_effort (else the template
                # defaults it off and the requested level never renders).
                effort_on = reasoning_effort in self._reasoning_effort_levels
                if enable_thinking is not None or reasoning_effort == "none" or effort_on:
                    kwargs["enable_thinking"] = not thinking_off
                if not thinking_off and effort_on:
                    kwargs["reasoning_effort"] = reasoning_effort
            elif self._reasoning_style == "reasoning_effort":
                if reasoning_effort in ("none", "low", "medium", "high"):
                    kwargs["reasoning_effort"] = reasoning_effort
                elif reasoning_effort == "minimal":
                    kwargs["reasoning_effort"] = "low"
                elif enable_thinking is not None:
                    kwargs["reasoning_effort"] = "high" if enable_thinking else "low"
            else:
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
        if self._supports_preserve_thinking and preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        _coerce_reasoning_effort(getattr(self, "_architecture", None), kwargs)
        return kwargs or None

    @property
    def supports_tools(self) -> bool:
        # DiffusionGemma serves via the visual runner, whose live per-step canvas
        # frames are dropped by the agentic tool loop; never route it through tools.
        if self._is_diffusion:
            return False
        return self._supports_tools

    @property
    def supports_tool_passthrough(self) -> bool:
        # supports_tools is forced off for DiffusionGemma (its agentic loop drops the
        # per-step canvas frames), but client passthrough skips that loop, so it uses
        # the real _supports_tools.
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        return self._cache_type_kv

    @property
    def tensor_parallel(self) -> bool:
        """Whether --split-mode tensor is active on the loaded server."""
        return self._tensor_parallel

    @property
    def gpu_memory_mode(self) -> str:
        """Active GPU memory strategy: 'auto' or 'manual' (gpu_layers < 0 = Auto/--fit, >= 0 = pinned)."""
        return self._gpu_memory_mode

    @property
    def gpu_layers(self) -> int:
        """Requested --gpu-layers for manual mode (-1 when not manual)."""
        return self._gpu_layers

    @property
    def holds_no_vram(self) -> bool:
        """Whether the resident server is a confirmed zero-VRAM launch.

        True only for a deliberate manual zero-offload load whose launched argv carried no GPU
        companion, pin or tensor mode, so the child was started with the GPUs hidden. The GPU
        arbiter uses this to leave an image/video pipeline alone when re-asserting ownership for
        such a model. ``_gpu_offload_active`` is None when no GPU was detected at all (nothing to
        arbitrate) and True when something still reached the GPU, so both keep the normal path.
        """
        return (
            self._gpu_memory_mode == "manual"
            and self._gpu_layers == 0
            and self._gpu_offload_active is False
        )

    @property
    def diffusion_requested_ngl(self) -> Optional[int]:
        """The diffusion split the live runner was ASKED for, or None for the default.

        Differs from ``gpu_layers`` only when the shim could not take the split: that
        reports what is running, this keeps the ask so a repeat still dedupes."""
        return getattr(self, "_diffusion_requested_ngl", None)

    @property
    def n_cpu_moe(self) -> int:
        """MoE expert layers manual mode kept on CPU (--n-cpu-moe); 0 = none."""
        return self._n_cpu_moe

    @property
    def tensor_split(self) -> Optional[List[float]]:
        """Manual-mode relative model share per GPU (--tensor-split); None =
        default (split by free VRAM)."""
        return self._tensor_split

    @property
    def gpu_ids(self) -> Optional[List[int]]:
        """User-picked physical GPU indices, or None for automatic selection."""
        return self._gpu_ids

    @property
    def requested_gpu_ids(self) -> Optional[List[int]]:
        """RAW requested GPU pin (before the fit narrowed it), or None for auto.
        gpu_ids echoes the EFFECTIVE pin for /status."""
        return self._requested_gpu_ids

    def matches_gpu_ids(self, gpu_ids: Optional[List[int]]) -> bool:
        """Whether a requested pin is already satisfied by the active runner.

        A regular GGUF load may narrow the requested placement pool to the
        smallest fitting subset. Accept both the original request and the
        effective status-echoed subset so either can round-trip without a
        needless reload. Diffusion drives one device and keeps its existing
        lowest-device normalization.
        """
        if self._is_diffusion:
            requested = [sorted(int(x) for x in gpu_ids)[0]] if gpu_ids else None
            # Raw pick accepted too: a forced-CPU runner records no effective pin, so
            # without it re-sending the same selection would reload every time.
            return requested in ((self._gpu_ids or None), (self._requested_gpu_ids or None))

        requested = sorted(int(x) for x in gpu_ids) if gpu_ids else None
        raw = self._requested_gpu_ids or None
        effective = self._gpu_ids or None
        return requested == raw or requested == effective

    def _record_matching_gpu_request(self, gpu_ids: Optional[List[int]]) -> None:
        """Adopt the caller's explicit pool after a full already-loaded match.

        Matching an effective subset avoids a reload, but the incoming request
        is still the user's latest placement intent. Record it so status and a
        later reload do not restore GPUs the user just removed.
        """
        if self._is_diffusion:
            self._requested_gpu_ids = [sorted(int(x) for x in gpu_ids)[0]] if gpu_ids else None
        else:
            self._requested_gpu_ids = sorted(int(x) for x in gpu_ids) if gpu_ids else None
        if self._last_load_intent is not None:
            self._last_load_intent = replace(
                self._last_load_intent,
                gpu_ids = tuple(self._requested_gpu_ids) if self._requested_gpu_ids else None,
            )

    @property
    def last_load_intent(self) -> Optional[GgufLoadIntent]:
        return self._last_load_intent

    def _runtime_matches_intent(
        self, intent: GgufLoadIntent, effective_extra_args: Optional[list[str]]
    ) -> bool:
        """Whether active runtime settings satisfy one resolved caller intent."""
        if self._requested_n_ctx != int(intent.n_ctx):
            return False
        if not self._is_diffusion and self._requested_n_parallel != max(1, int(intent.n_parallel)):
            return False

        def _norm(value):
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip().lower()
                return value or None
            return value

        if _norm(self._cache_type_kv) != _norm(intent.cache_type_kv):
            return False

        extra_args = list(effective_extra_args) if effective_extra_args is not None else None
        # A request omitting the extras field inherits the LAUNCHED list; anything else is
        # the caller's own. A launch-time rewrite (drafter drop, MTP crash replay) makes
        # the two differ, and the equality below wants what the load was INVOKED with.
        # Judged on the flag, not on value equality: a caller that deliberately sends the
        # stripped list is clearing the failed drafter and must not read as inheriting it.
        _invoked_extras = self.requested_extra_args if intent.extra_args_inherited else extra_args
        # A virtualised Metal device never launches a tensor split (CPU-pinned, and
        # --split-mode is overridden with the default layer split), so judge the
        # normalized request there: an extras `-sm tensor`, which the pin deliberately
        # leaves in place, would otherwise never match.
        if _metal_device_is_paravirtual():
            if self._tensor_parallel:
                return False
        elif not _tensor_parallel_matches_loaded(
            extra_args, intent.tensor_parallel, self._tensor_parallel
        ):
            return False
        if (
            self._layer_preserves_tensor_intent
            and not _effective_tensor_parallel(extra_args, intent.tensor_parallel)
            and not intent.preserve_multi_gpu_on_layer
        ):
            return False

        if self._is_diffusion:
            requested_ngl = _diffusion_manual_ngl(intent.gpu_memory_mode, intent.gpu_layers)
            if requested_ngl != self.diffusion_requested_ngl:
                return False
            # A split dropped by an older shim still matches until the shim gains
            # --ngl support, at which point the same request should apply it.
            if (
                requested_ngl is not None
                and self._gpu_layers != requested_ngl
                and self.diffusion_split_supported()
            ):
                return False
        else:
            if self._swa_full != _swa_full_from_args_or_env(extra_args):
                return False
            if self._gpu_memory_mode != intent.gpu_memory_mode:
                return False
            if intent.gpu_memory_mode == "manual" and (
                self._gpu_layers != intent.gpu_layers
                or (
                    intent.gpu_layers >= 0
                    and (
                        self._n_cpu_moe != intent.n_cpu_moe
                        or (tuple(self._tensor_split) if self._tensor_split else None)
                        != (intent.tensor_split or None)
                    )
                )
            ):
                return False

        if not self.matches_gpu_ids(list(intent.gpu_ids) if intent.gpu_ids else None):
            return False

        spec_owned_by_extra_args = _extra_args_set_spec_type(extra_args)
        speculative_type = (
            None
            if spec_owned_by_extra_args
            else _canonicalize_spec_mode(intent.speculative_type) or "auto"
        )
        backend_spec = self._requested_spec_mode
        if spec_owned_by_extra_args:
            if backend_spec is not None:
                return False
        elif speculative_type != (backend_spec or "auto"):
            return False
        # Reload so the next Apply retries the drafter fetch. Worth it when the
        # repo publishes a drafter and the fetch failed; not when it publishes
        # none, which for DSpark is the permanent state of every repo but one and
        # would relaunch an identical server forever.
        if (
            intent.gguf_path is None
            and self._spec_fallback_reason == "drafter_not_found"
            and speculative_type in ("auto", "mtp", "mtp+ngram", "dspark")
            and not (self._spec_drafter_kind == "dspark" and self._dspark_sidecar_absent)
            and not spec_owned_by_extra_args
        ):
            return False
        compared_draft_n_max = self._spec_draft_n_max
        if self._spec_fallback_reason == "runtime_error" and self._last_load_intent is not None:
            # The MTP-free recovery clears the runtime value but retains the
            # user's prior value in its intent. Only a changed value should retry MTP.
            compared_draft_n_max = self._last_load_intent.spec_draft_n_max
        if (
            (
                self._speculative_type in ("draft-mtp", "draft-dspark")
                or self._spec_fallback_reason == "runtime_error"
            )
            and intent.spec_draft_n_max is not None
            and intent.spec_draft_n_max != (compared_draft_n_max or 0)
        ):
            return False
        if (self._chat_template_override or None) != (intent.chat_template_override or None):
            return False

        if tuple(_invoked_extras or ()) != tuple(self.requested_extra_args or ()):
            return False
        if (intent.gguf_path is not None or intent.compare_mtp_draft) and speculative_type in (
            "auto",
            "mtp",
            "mtp+ngram",
            "dspark",
        ):
            try:
                # Auto counts as dspark once it resolved that way: the launch stored
                # the DSpark sidecar, so comparing the MTP field (None for these
                # repos) against it would reload a healthy server on every Apply.
                _compare_dspark = speculative_type == "dspark" or (
                    speculative_type == "auto" and self._speculative_type == "draft-dspark"
                )
                intent_draft = (
                    intent.dspark_draft_path if _compare_dspark else intent.mtp_draft_path
                )
                requested_draft = Path(intent_draft).resolve() if intent_draft else None
                # A drafter the last load dropped on purpose counts as launched here:
                # the file is still there, so comparing it against the stored None
                # would reload the same drafter-free server forever.
                _loaded = self._mtp_draft_path or self._mtp_draft_suppressed_path
                loaded_draft = Path(_loaded).resolve() if _loaded else None
            except OSError:
                return False
            if requested_draft != loaded_draft:
                return False
        return True

    @property
    def n_layers(self) -> Optional[int]:
        """Model layer count (GGUF block_count), or None if unknown."""
        return self._n_layers

    @property
    def n_moe_layers(self) -> int:
        """Number of MoE expert layers (the --n-cpu-moe ceiling), 0 if not MoE.

        block_count minus the leading dense layers (which carry no experts):
        --n-cpu-moe counts from layer 0, so those dense layers are no-ops.
        """
        if not self._n_experts or not self._n_layers:
            return 0
        return max(0, self._n_layers - (self._leading_dense_block_count or 0))

    @staticmethod
    def _resolve_cpu_moe_flag(
        n_cpu_moe: int, n_moe_layers: int, leading_dense: int
    ) -> Optional[int]:
        """The --n-cpu-moe value (absolute first-N layers), or None to omit it.

        Clamps the requested count to the model's MoE layers, then offsets past
        the leading dense layers (--n-cpu-moe counts from layer 0). Returns None
        for nothing-to-offload (0 requested) or a non-MoE model.
        """
        if n_cpu_moe <= 0 or n_moe_layers <= 0:
            return None
        return leading_dense + min(n_cpu_moe, n_moe_layers)

    @staticmethod
    def _sanitize_tensor_split(tensor_split: Optional[List[float]]) -> List[float]:
        """Per-GPU shares with negative and non-finite entries clamped to 0.

        A direct caller's negative entry would launch a placement different
        from the ratio the UI showed, and inf would pass a plain ``> 0`` total
        gate and emit ``--tensor-split inf,...``. Returns [] for input that
        can't be read as floats (the length gate at the call site then drops
        the split).
        """
        try:
            return [
                x if math.isfinite(x) and x > 0.0 else 0.0 for x in (float(v) for v in tensor_split)
            ]
        except (TypeError, ValueError, OverflowError):
            return []

    @property
    def layer_preserves_tensor_intent(self) -> bool:
        """True when a downgraded tensor request kept this layer load multi-GPU."""
        return self._layer_preserves_tensor_intent

    @property
    def speculative_type(self) -> Optional[str]:
        return self._speculative_type

    @property
    def spec_drafter_kind(self) -> Optional[str]:
        """Which drafter the resolution was about, ``mtp`` or ``dspark``.

        Distinct from ``requested_spec_mode``, which stays ``auto`` when Auto
        resolves the kind itself, and from ``speculative_type``, which reads
        ``default`` once a fallback fires. The UI needs the kind to name the
        right recovery in a fallback notice.
        """
        return self._spec_drafter_kind

    @property
    def requested_spec_mode(self) -> Optional[str]:
        """Canonical UI-facing mode the user requested (see field doc)."""
        return self._requested_spec_mode

    @property
    def spec_draft_n_max(self) -> Optional[int]:
        """User --spec-draft-n-max override active on the load, or None when
        the platform default (6 GPU / 3 CPU) is in effect."""
        return self._spec_draft_n_max

    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _resolved_studio_root_and_is_legacy() -> "tuple[Optional[Path], bool]":
        """Resolve the Unsloth install root and classify it as the legacy
        ~/.unsloth/studio root vs. a custom (env/venv-inferred) root.

        Returns (resolved_root, is_legacy). On any import/resolution failure the
        root is treated as legacy and resolved_root is None -- callers must read
        resolved_root only when is_legacy is False. Shared by
        _find_llama_server_binary (discovery) and _kill_orphaned_servers
        (cleanup) so the two never disagree on which root is legacy.
        """
        try:
            from utils.paths.storage_roots import studio_root as _sr  # noqa: WPS433

            resolved = _sr()
            legacy_studio = Path.home() / ".unsloth" / "studio"
            try:
                is_legacy = resolved.resolve() == legacy_studio.resolve()
            except (OSError, ValueError):
                is_legacy = resolved == legacy_studio
            return (None if is_legacy else resolved), is_legacy
        except (ImportError, OSError, ValueError):
            return None, True

    @staticmethod
    def _find_llama_server_binary(*, include_denied: bool = False) -> Optional[str]:
        """
        Locate the llama-server binary.

        Search order:
        1.  LLAMA_SERVER_PATH environment variable (direct path to binary)
        1b. UNSLOTH_LLAMA_CPP_PATH env var (custom llama.cpp install dir)
        2.  ~/.unsloth/llama.cpp/llama-server        (make build, root dir)
        3.  ~/.unsloth/llama.cpp/build/bin/llama-server  (cmake build, Linux)
        4.  ~/.unsloth/llama.cpp/build/bin/Release/llama-server.exe  (cmake build, Windows)
        5.  ./llama.cpp/llama-server                 (legacy: make build, root dir)
        6.  ./llama.cpp/build/bin/llama-server        (legacy: cmake in-tree build)
        7.  llama-server on PATH                     (system install)
        8.  ./bin/llama-server                       (legacy: extracted binary)
        """
        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        def _file_status(p: Path) -> str:
            # "file", "absent", or "denied" (exists but stays access-denied
            # across a short retry: Windows AV/ACL or an install replace in
            # flight). is_file() raises PermissionError (WinError 5) instead of
            # returning False for the locked case, so never treat it as missing.
            for _ in range(5):
                try:
                    return "file" if p.is_file() else "absent"
                except PermissionError:
                    time.sleep(0.2)
                except OSError:
                    return "absent"
            return "denied"

        def _is_file(p: Path) -> bool:
            return _file_status(p) == "file"

        def _layout_candidates(d: Path) -> list:
            # build layouts probed under a llama.cpp dir, highest priority first
            cands = [d / binary_name, d / "build" / "bin" / binary_name]
            if sys.platform == "win32":
                cands.append(d / "build" / "bin" / "Release" / binary_name)
            return cands

        def _unavailable(p: object) -> None:
            # a pinned or managed binary that exists but is access-denied: report
            # it instead of silently downgrading to a lower-priority llama-server
            logger.warning(
                f"llama-server at {p} exists but is access-denied (antivirus or "
                "an in-flight install); not falling back to another binary, "
                "retry once it is released"
            )
            return None

        def _scan_pinned(paths: list):
            # first existing candidate wins -> (path, None); a present-but-denied
            # one -> (None, denied_path) so the caller reports it rather than
            # skipping to a lower-priority location. include_denied returns the
            # locked path instead: diffusion asset lookup only needs its dir.
            for p in paths:
                st = _file_status(p)
                if st == "file":
                    return str(p), None
                if st == "denied":
                    return (str(p), None) if include_denied else (None, p)
            return None, None

        # 1. Env var: direct path to binary
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path:
            hit, locked = _scan_pinned([Path(env_path)])
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 1b. UNSLOTH_LLAMA_CPP_PATH: custom llama.cpp install dir
        custom_llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
        if custom_llama_cpp:
            hit, locked = _scan_pinned(_layout_candidates(Path(custom_llama_cpp)))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 2-4. Match installer layout: env-mode -> $STUDIO_HOME/llama.cpp;
        # default/HOME-redirect -> ~/.unsloth/llama.cpp (sibling of studio).
        legacy_llama = Path.home() / ".unsloth" / "llama.cpp"
        _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
        if _is_legacy:
            search_roots = [legacy_llama]
        else:
            # _kill_orphaned_servers excludes the legacy root in custom mode;
            # discovery must match so we never spawn a server we then refuse to
            # clean up. UNSLOTH_LLAMA_CPP_PATH (handled earlier) is the explicit
            # way to share a build across roots.
            search_roots = [_resolved_sr / "llama.cpp"]
        for unsloth_home in search_roots:
            hit, locked = _scan_pinned(_layout_candidates(unsloth_home))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 5-6. Legacy: in-tree build (older setup.sh / setup.ps1). A fallback,
        # so a denied candidate here just continues (no no-fallback halt).
        project_root = Path(__file__).resolve().parents[4]
        for p in _layout_candidates(project_root / "llama.cpp"):
            if _is_file(p):
                return str(p)

        # 7. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 8. Legacy: extracted to bin/
        bin_path = project_root / "bin" / binary_name
        if _is_file(bin_path):
            return str(bin_path)

        return None

    # ── llama-server capability probe ─────────────────────────────

    # Prebuilts based on llama.cpp b10259..b10268 advertise draft-dspark but land between
    # the reshape regression (ggml-org/llama.cpp#26531) and its fix (#26577), so a DSpark
    # launch aborts on load. Matched on the base build number, not one exact tag: every
    # release in that window is affected. Source builds carry no install marker, so only
    # managed prebuilts are gated.
    _BROKEN_DSPARK_BUILDS = range(10259, 10269)

    @classmethod
    def _dspark_release_is_broken(cls, release_tag: Optional[str]) -> bool:
        match = re.match(r"b(\d+)", str(release_tag or ""))
        return bool(match) and int(match.group(1)) in cls._BROKEN_DSPARK_BUILDS

    # Cached on (path, mtime); `unsloth studio update` bumps mtime.
    _capability_cache: dict[tuple[str, int], dict[str, object]] = {}

    @classmethod
    def probe_server_capabilities(cls, binary: Optional[str] = None) -> dict[str, object]:
        """Parse `llama-server --help` for feature flags. Returns
        {found, mtp_token, supports_mtp, supports_dspark, ngram_mod_flavor,
        supports_ngram_mod, spec_draft_n_max_flag, cache flag support}.

        ``ngram_mod_flavor``: ``"new"`` when the post-rename
        ``--spec-ngram-mod-n-match / -n-min / -n-max`` are real args;
        ``"legacy"`` when only the pre-rename
        ``--spec-ngram-size-n / --draft-min / --draft-max`` are real (the
        rename ships stub removal entries for legacy names, told apart by
        the "argument has been removed" description); ``None`` if neither
        set is usable.

        ``spec_draft_n_max_flag``: the flag the binary accepts --
        ``--spec-draft-n-max`` post-rename, ``--draft-max`` on legacy.
        ``None`` means n_max cannot be set.
        """
        bin_path = binary or cls._find_llama_server_binary()
        if not bin_path or not Path(bin_path).is_file():
            return {
                "found": False,
                "mtp_token": None,
                "supports_mtp": False,
                "supports_dspark": False,
                "mtp_probe_inconclusive": True,
                "ngram_mod_flavor": None,
                "supports_ngram_mod": False,
                "spec_draft_n_max_flag": None,
                "supports_kv_unified": False,
                "supports_fit_ctx": False,
                "supports_fit_target": False,
                "supports_cache_ram": False,
                "supports_ctx_checkpoints": False,
                "supports_no_cache_prompt": False,
                "supports_metrics": False,
                "supports_slot_save": False,
                "supports_no_mmproj_offload": False,
                "spec_draft_ngl_flag": None,
            }
        try:
            mtime = int(Path(bin_path).stat().st_mtime)
        except OSError:
            mtime = 0
        cache_key = (bin_path, mtime)
        cached = cls._capability_cache.get(cache_key)
        if cached is not None:
            return cached

        mtp_token: Optional[str] = None
        supports_dspark = False
        ngram_mod_flavor: Optional[str] = None
        spec_draft_n_max_flag: Optional[str] = None
        supports_kv_unified = False
        supports_fit_ctx = False
        supports_fit_target = False
        supports_cache_ram = False
        supports_ctx_checkpoints = False
        supports_no_cache_prompt = False
        supports_metrics = False
        supports_slot_save = False
        supports_no_mmproj_offload = False
        spec_draft_ngl_flag = None
        saw_spec_type = False
        probe_ok = False
        help_text = ""
        try:
            probe_env = cls._llama_server_env_for_binary(bin_path)
            result = subprocess.run(
                [bin_path, "--help"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = probe_env,
            )
            probe_ok = result.returncode == 0
            help_text = (result.stdout or "") + "\n" + (result.stderr or "")
            # Split into per-flag blocks (each --flag line + its indented
            # continuation), so the "argument has been removed" description
            # sits with its flag.
            blocks: dict[str, str] = {}
            current_flags: list[str] = []
            current_desc: list[str] = []
            for line in help_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("-") and not line.startswith(" "):
                    # New flag line; flush previous.
                    if current_flags:
                        desc = " ".join(current_desc)
                        for f in current_flags:
                            blocks[f] = desc
                    current_flags = []
                    current_desc = [stripped]
                    # Extract long-form flag tokens from the DECLARATION
                    # prefix only (comma-separated aliases). Stop at the
                    # first non-flag token so flag references inside
                    # descriptions are ignored.
                    for tok in re.split(r"[,\s]+", stripped):
                        if tok.startswith("--") and re.match(r"--[A-Za-z][A-Za-z0-9_-]*$", tok):
                            current_flags.append(tok)
                        elif tok.startswith("-") and len(tok) > 1:
                            # short alias like -fa; keep scanning aliases.
                            continue
                        else:
                            # First non-flag token marks end of decl.
                            break
                else:
                    current_desc.append(stripped)
            if current_flags:
                desc = " ".join(current_desc)
                for f in current_flags:
                    blocks[f] = desc

            def _is_real(flag: str) -> bool:
                """True if the flag exists AND is not a removal stub."""
                desc = blocks.get(flag)
                if desc is None:
                    return False
                return "argument has been removed" not in desc

            # MTP token from the full --spec-type help block (decl + indented
            # continuation). First-line-only probing missed builds putting the
            # enum on the next line (#7302). Prefer draft-mtp (PR #22673) over mtp.
            spec_help = blocks.get("--spec-type") or ""
            if not spec_help:
                # Fallback: join --spec-type lines, avoiding incidental "mtp" in --help.
                spec_help = "\n".join(
                    line for line in help_text.splitlines() if "--spec-type" in line
                )
            mtp_token = cls._mtp_token_from_spec_help(spec_help)
            supports_dspark = bool(
                re.search(
                    r"(?<![a-z0-9_-])draft-dspark(?![a-z0-9_-])",
                    spec_help.lower(),
                )
            )
            # Only a resolved --spec-type block confirms missing MTP; empty/crash
            # leaves saw_spec_type False so supports_mtp fails open.
            saw_spec_type = bool(spec_help.strip()) and "--spec-type" in spec_help

            # ngram-mod flag flavor. Post-rename builds advertise both new
            # args (real) and legacy ones (stubs); pre-rename builds only
            # have legacy ones as real.
            new_ngram_real = (
                _is_real("--spec-ngram-mod-n-match")
                and _is_real("--spec-ngram-mod-n-min")
                and _is_real("--spec-ngram-mod-n-max")
            )
            legacy_ngram_real = (
                _is_real("--spec-ngram-size-n")
                and _is_real("--draft-max")
                and _is_real("--draft-min")
            )
            if new_ngram_real:
                ngram_mod_flavor = "new"
            elif legacy_ngram_real:
                ngram_mod_flavor = "legacy"

            # n_max flag: prefer post-rename, fall back to legacy.
            if _is_real("--spec-draft-n-max"):
                spec_draft_n_max_flag = "--spec-draft-n-max"
            elif _is_real("--draft-max"):
                spec_draft_n_max_flag = "--draft-max"

            supports_kv_unified = _is_real("--kv-unified")
            supports_fit_ctx = _is_real("--fit-ctx")
            supports_fit_target = _is_real("--fit-target")
            supports_cache_ram = _is_real("--cache-ram")
            supports_ctx_checkpoints = _is_real("--ctx-checkpoints")
            supports_no_cache_prompt = _is_real("--no-cache-prompt")
            supports_metrics = _is_real("--metrics")
            supports_slot_save = _is_real("--slot-save-path")
            supports_no_mmproj_offload = _is_real("--no-mmproj-offload")
            # Record WHICH alias this build has: --spec-draft-ngl only landed in
            # b8955, and a build exposing only --gpu-layers-draft would refuse to
            # start on the newer name. Long forms only, since the block parser above
            # skips short aliases, so "-ngld" could never have matched.
            for _alias in ("--spec-draft-ngl", "--gpu-layers-draft", "--n-gpu-layers-draft"):
                if _is_real(_alias):
                    spec_draft_ngl_flag = _alias
                    break
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug(f"llama-server --help probe failed: {exc}")
            saw_spec_type = False
            probe_ok = False
            help_text = ""

        help_nonempty = bool(help_text.strip())
        # Confirmed only when a successful --help lists a --spec-type block with
        # mtp/draft-mtp; nonempty --help without it is a definitive pre-spec
        # binary; failed/empty probes stay inconclusive (#7302).
        if saw_spec_type and probe_ok:
            supports_mtp = mtp_token is not None
            mtp_probe_inconclusive = False
        elif help_nonempty and probe_ok:
            supports_mtp = False
            mtp_probe_inconclusive = False
        else:
            supports_mtp = False
            mtp_probe_inconclusive = True

        if supports_dspark:
            try:
                from utils.llama_cpp_freshness import read_install_marker

                marker = read_install_marker(bin_path) or {}
                release_tag = marker.get("release_tag") or marker.get("tag")
                if cls._dspark_release_is_broken(release_tag):
                    supports_dspark = False
                    logger.info(
                        "Disabling DSpark for known-broken llama.cpp prebuilt %s",
                        release_tag,
                    )
            except Exception as exc:
                logger.debug("Could not inspect llama.cpp prebuilt identity: %s", exc)

        info = {
            "found": True,
            "mtp_token": mtp_token,
            "supports_mtp": supports_mtp,
            "supports_dspark": bool(supports_dspark and saw_spec_type and probe_ok),
            "mtp_probe_inconclusive": mtp_probe_inconclusive,
            "ngram_mod_flavor": ngram_mod_flavor,
            "supports_ngram_mod": ngram_mod_flavor is not None,
            "spec_draft_n_max_flag": spec_draft_n_max_flag,
            "supports_kv_unified": supports_kv_unified,
            "supports_fit_ctx": supports_fit_ctx,
            "supports_fit_target": supports_fit_target,
            "supports_cache_ram": supports_cache_ram,
            "supports_ctx_checkpoints": supports_ctx_checkpoints,
            "supports_no_cache_prompt": supports_no_cache_prompt,
            "supports_metrics": supports_metrics,
            "supports_slot_save": supports_slot_save,
            "supports_no_mmproj_offload": supports_no_mmproj_offload,
            "spec_draft_ngl_flag": spec_draft_ngl_flag,
        }
        cls._capability_cache[cache_key] = info
        return info

    @staticmethod
    def _mtp_token_from_spec_help(spec_help: str) -> Optional[str]:
        """Extract ``draft-mtp`` / ``mtp`` from a ``--spec-type`` help snippet.

        Prefers ``draft-mtp`` (llama.cpp PR #22673) over the later bare ``mtp``
        rename. Returns ``None`` when neither token appears as an enum value.
        """
        text = spec_help or ""
        if "draft-mtp" in text:
            return "draft-mtp"
        # Bare `mtp` enum token (`|mtp|`, `,mtp,`, ...), not a substring.
        if re.search(r"(?<![A-Za-z0-9_-])mtp(?![A-Za-z0-9_-])", text):
            return "mtp"
        return None

    # ── GPU allocation ────────────────────────────────────────────

    @staticmethod
    def _get_gguf_size_bytes(model_path: str) -> int:
        """Total GGUF size in bytes, including split shards."""
        main = Path(model_path)
        total = main.stat().st_size

        # Check for split shards (e.g. model-00001-of-00003.gguf)
        m = _SHARD_FULL_RE.match(main.name)
        if m:
            prefix, _, num_total = m.group(1), m.group(2), m.group(3)
            sibling_pat = re.compile(
                r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(num_total) + r"\.gguf$",
                re.IGNORECASE,
            )
            for sibling in main.parent.iterdir():
                if sibling != main and sibling_pat.match(sibling.name):
                    total += sibling.stat().st_size

        return total

    @staticmethod
    def _is_vulkan_backend(binary: Optional[str] = None) -> bool:
        """True if the installed llama.cpp build is Vulkan-only.

        The official prebuilts are single-backend, so the Vulkan ggml lib next
        to llama-server identifies a Vulkan build. Keeps the free-memory probe
        and GPU pin in ggml's Vulkan device-index space. For a custom
        multi-backend build with a CUDA or HIP ggml lib alongside Vulkan, defer
        to that backend (torch-usable, better-understood probe/pin).
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        lib_dir = _llama_lib_dir(binary)
        if not _lib_dir_has_ggml_backend(lib_dir, "vulkan"):
            return False
        for _backend in ("cuda", "hip"):
            if _lib_dir_has_ggml_backend(lib_dir, _backend):
                return False
        return True

    @staticmethod
    def _resolve_visible_physical_ids() -> Optional[list[int]]:
        """Physical GPU ids behind the active visibility mask (HIP/ROCR/CUDA on
        ROCm, CUDA otherwise). None when no mask is set; empty list for an empty
        mask. Shared by the APU / datacenter / free-memory probes so they agree
        on the ordinal->physical mapping."""
        try:
            import torch

            # Same ROCm detection as _emit_child_gpu_visibility: AMD SDK wheels
            # leave version.hip unset but encode "rocm" in __version__. The two
            # must agree, else an inherited ROCR mask reads back as "no mask",
            # ordinal 0 is labelled physical 0, and the child's new ROCR pin
            # re-exposes the GPU the inherited mask was hiding.
            is_rocm = (
                getattr(torch.version, "hip", None) is not None
                or "rocm" in getattr(torch, "__version__", "").lower()
            )
        except Exception:
            is_rocm = False
        if is_rocm:
            hip_v = os.environ.get("HIP_VISIBLE_DEVICES")
            # ROCR_VISIBLE_DEVICES is a Linux ROCr variable; Windows HIP has no
            # ROCr layer, so a stray ROCR var there does not mask the runtime and
            # must not be read as the ordinal->physical mapping (mirrors the
            # Windows gate in _emit_child_gpu_visibility).
            rocr_v = None if sys.platform == "win32" else os.environ.get("ROCR_VISIBLE_DEVICES")
            cvd = (
                hip_v
                if hip_v is not None
                else rocr_v
                if rocr_v is not None
                else os.environ.get("CUDA_VISIBLE_DEVICES")
            )
        else:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            return None
        try:
            return [int(x.strip()) for x in cvd.split(",") if x.strip()]
        except ValueError:
            return None

    @staticmethod
    def _emit_child_gpu_visibility(
        env: dict,
        pinned: str,
        *,
        prefer_rocr: bool = False,
    ) -> None:
        """Write the child's GPU visibility mask: CUDA, plus a ROCm mirror on AMD
        (masking only CUDA_VISIBLE_DEVICES leaves an AMD child seeing every GPU).

        Default: HIP_VISIBLE_DEVICES, clearing any inherited ROCR mask so the two
        can't stack (ROCR re-indexes from 0, then a non-zero HIP pin points out of
        range, HIP sees 0 devices, and llama.cpp falls back to CPU).

        prefer_rocr masks at the ROCr/HSA layer instead (clearing HIP). A HIP mask
        filters only AFTER the HSA runtime enumerates every agent, and that
        enumeration segfaults at startup on a GPU the build has no kernels for
        (e.g. a gfx1036 iGPU under a gfx103X prebuilt: that bundle maps only
        gfx1030/1031/1032/1034), before llama-server logs a line. ROCR drops the
        device at the driver layer, consuming physical ids.
        The CPU-only sentinel ("-1") has no portable ROCR spelling, so it keeps
        the HIP mask. Windows keeps the HIP mask too: ROCR_VISIBLE_DEVICES is a
        Linux ROCr variable (Windows HIP has no ROCr layer), so the ROCR pin
        would be dead there while the cleared HIP mask stops selecting."""
        env["CUDA_VISIBLE_DEVICES"] = pinned
        try:
            import torch as _torch

            # torch.version.hip is set on ROCm, None on CUDA; AMD SDK wheels may
            # leave it unset but encode "rocm" in __version__ (mirrors detect_hardware).
            if (
                getattr(_torch.version, "hip", None) is not None
                or "rocm" in getattr(_torch, "__version__", "").lower()
            ):
                if prefer_rocr and pinned != "-1" and sys.platform != "win32":
                    env["ROCR_VISIBLE_DEVICES"] = pinned
                    env.pop("HIP_VISIBLE_DEVICES", None)
                    # ROCR re-indexes the visible agents from 0, and with HIP
                    # cleared HIP honours CUDA_VISIBLE_DEVICES -- so it must carry
                    # the post-ROCR ordinals (0..N-1), not the physical ids, else a
                    # non-zero pick points out of range and HIP sees 0 devices (the
                    # same stacking the default path avoids by clearing ROCR).
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(
                        str(i) for i in range(len(pinned.split(",")))
                    )
                else:
                    env["HIP_VISIBLE_DEVICES"] = pinned
                    env.pop("ROCR_VISIBLE_DEVICES", None)
        except Exception as e:
            logger.debug("Failed to set ROCm visibility env vars for child: %s", e)

    @staticmethod
    def _pin_visible_gpu_order_for_split(env: dict) -> None:
        """Pin the child's GPU enumeration to the picker's order for a manual
        ``--tensor-split`` across the whole visible set. CUDA's default
        FASTEST_FIRST enumeration applies the shares to the wrong cards on
        heterogeneous hosts (#5025), and CUDA_DEVICE_ORDER only fixes the
        numbering base: an inherited numeric visibility mask ALSO defines
        enumeration order, so a reordered parent mask (CUDA_VISIBLE_DEVICES=3,1)
        would still hand the shares to the wrong cards. The UI built the split
        positionally over get_backend_visible_gpu_info's device list (ascending
        physical via nvidia-smi, inherited mask order on the torch fallback), so
        re-emit the same set in that report order -- not an assumed ascending
        sort. The visible set itself never changes. No mask, an empty mask, or a
        UUID/MIG mask (which resolves to None) is left alone -- the multi-GPU
        controls are hidden for the latter."""
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        inherited = LlamaCppBackend._resolve_visible_physical_ids()
        if not inherited:
            return
        order = None
        try:
            from utils.hardware import get_backend_visible_gpu_info
            info = get_backend_visible_gpu_info()
            if info.get("available") and info.get("index_kind") == "physical":
                reported = [d["index"] for d in info.get("devices", [])]
                if sorted(reported) == sorted(inherited):
                    order = reported
        except Exception as e:
            logger.debug("Could not read reported GPU order for split pin: %s", e)
        if order is None:
            order = sorted(inherited)
        # Re-emit at the layer that produced the mapping. A parent masked only
        # via ROCR_VISIBLE_DEVICES hides agents at the driver layer, and the
        # default HIP re-emission clears that mask -- HSA then enumerates every
        # agent again and can segfault at startup on an unsupported GPU the
        # parent was hiding (the crash prefer_rocr exists to avoid). Linux-only,
        # mirroring _resolve_visible_physical_ids: on Windows a stray ROCR var
        # is dead and was not the mapping's source.
        prefer_rocr = (
            sys.platform != "win32"
            and env.get("HIP_VISIBLE_DEVICES") is None
            and env.get("ROCR_VISIBLE_DEVICES") is not None
        )
        LlamaCppBackend._emit_child_gpu_visibility(
            env, ",".join(str(i) for i in order), prefer_rocr = prefer_rocr
        )

    @staticmethod
    def _amd_apu_wants_unified_memory(gpu_indices = None) -> bool:
        """True only for AMD unified-memory APUs (gfx1150/gfx1151/gfx1152), where
        GGML_CUDA_ENABLE_UNIFIED_MEMORY lets llama.cpp use shared system RAM (it
        hurts discrete GPUs). gpu_indices (PHYSICAL ids) scopes the check to the
        selected GPUs, so a dGPU on a mixed host is not treated as unified-memory;
        None means every visible GPU."""
        try:
            import torch

            if getattr(torch.version, "hip", None) is None:
                return False
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            # Map visible ordinal -> physical id via the active ROCm mask (HIP,
            # then ROCR, then CUDA), mirroring _get_gpu_memory's ROCm branch.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            arch_by_id: dict[int, str] = {}
            for ordinal in range(torch.cuda.device_count()):
                try:
                    _arch = (
                        getattr(torch.cuda.get_device_properties(ordinal), "gcnArchName", "") or ""
                    )
                except Exception:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                arch_by_id[pid] = _arch.split(":")[0].strip().lower()
            for _i in list(gpu_indices) if gpu_indices is not None else list(arch_by_id):
                # gfx1152 is Krackan Point (Radeon 860M/840M), the third RDNA 3.5
                # APU: same shared GPU/system-RAM pool as Strix Point/Halo.
                if arch_by_id.get(_i) in {"gfx1150", "gfx1151", "gfx1152"}:
                    return True
        except Exception:
            return False
        return False

    # Datacenter / professional NVIDIA parts that benefit from the llama.cpp
    # FP32-accum / P2P tunings. Whole-word (\b) so short markers don't match
    # workstation parts as substrings: "a100" must not fire on "RTX A1000".
    _DATACENTER_GPU_RE = re.compile(
        r"\b(?:a100|a30|h100|h200|h800|gh200|b200|b100|b300|gb200|gb300|"
        r"l40s?|l4|rtx pro 6000|rtx 6000 ada)\b"
    )

    @staticmethod
    def _is_datacenter_gpu(gpu_indices = None) -> bool:
        """True iff every selected NVIDIA GPU is a datacenter/professional part.
        NVIDIA-only, fails open to False (consumer GeForce, ROCm, CPU and errors
        are left untouched); a mixed DC+consumer selection counts as non-DC.

        gpu_indices are PHYSICAL ids (see _get_gpu_free_memory), but
        get_device_properties wants mask-relative ordinals, so we rebuild the
        ordinal->physical map from CUDA_VISIBLE_DEVICES and key names by physical
        id. Otherwise a masked host (CUDA_VISIBLE_DEVICES=4,5,6,7, selection [4,5])
        would drop the tuning or probe the wrong GPU."""
        try:
            import torch

            if getattr(torch.version, "hip", None) is not None:
                return False  # ROCm reuses torch.cuda.*; not a CUDA part
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            count = torch.cuda.device_count()

            # Mirror _get_gpu_free_memory: map visible ordinal -> physical id via
            # CUDA_VISIBLE_DEVICES; unset/unparsable leaves physical id == ordinal.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()

            pattern = LlamaCppBackend._DATACENTER_GPU_RE
            names_by_id: dict[int, str] = {}
            for ordinal in range(count):
                try:
                    name = (torch.cuda.get_device_properties(ordinal).name or "").lower()
                except Exception:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                names_by_id[pid] = name

            indices = list(gpu_indices) if gpu_indices else list(names_by_id)
            saw = False
            for _i in indices:
                name = names_by_id.get(_i)
                if name is None:
                    continue  # not visible -> skip (fail conservative)
                saw = True
                if not pattern.search(name):
                    return False
            return saw
        except Exception:
            return False

    @staticmethod
    def _effective_gpu_count(gpu_indices = None) -> int:
        """GPUs llama-server will use: len(selection), else the visible CUDA
        device count (None = every visible GPU). 0 on error so multi-GPU tuning
        stays off when the count is unknown."""
        if gpu_indices is not None:
            return len(gpu_indices)
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            return 0
        return 0

    @staticmethod
    def _apply_datacenter_env(env: dict, gpu_indices = None) -> bool:
        """Inject DC llama.cpp tuning into env in place via setdefault (user
        values win); return whether the box qualified. Opt out with
        UNSLOTH_DISABLE_DC_TUNING=1; only datacenter NVIDIA parts qualify
        (consumer/ROCm/CPU/error are a no-op). Sets GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F
        for any qualifying GPU (FP32 accum: ~0% cost on B200, real cost on GeForce),
        plus GGML_CUDA_P2P + CUDA_SCALE_LAUNCH_QUEUES=4x for multi-GPU (+33-51% pp
        tensor-split, +8-16% pipeline split on B200)."""
        if os.environ.get("UNSLOTH_DISABLE_DC_TUNING") == "1":
            return False
        if not LlamaCppBackend._is_datacenter_gpu(gpu_indices):
            return False
        env.setdefault("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1")
        if LlamaCppBackend._effective_gpu_count(gpu_indices) > 1:
            env.setdefault("GGML_CUDA_P2P", "1")
            env.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        return True

    @staticmethod
    def _visible_devices_mask(env_name: str) -> Optional[set[int]]:
        """Physical indices a ``*_VISIBLE_DEVICES`` mask permits, or None if unset.

        ``if x.strip()`` filters trailing-comma masks ("0,1,"); an empty mask
        ("") yields an empty set (all devices hidden), distinct from an unset
        var (None, no mask). Used by the nvidia-smi probe.
        """
        raw = os.environ.get(env_name)
        if raw is None:
            return None
        try:
            return set(int(x.strip()) for x in raw.split(",") if x.strip())
        except ValueError:
            return None

    @staticmethod
    def _vulkan_pin_args(gpu_indices: Optional[Iterable[int]]) -> list[str]:
        """``--device Vulkan<i>,...`` to pin a Vulkan launch to selected GPUs.

        The indices are ggml's compact Vulkan ordinals (as _get_gpu_free_memory
        reports and the registry names ``Vulkan<i>``). Pin by that name, NOT via
        GGML_VK_VISIBLE_DEVICES: ggml parses that env var in the raw
        vkEnumeratePhysicalDevices space (before dropping CPU/llvmpipe devices
        and deduplicating ICDs), so a compact ordinal there could select a
        different physical device or the CPU rasterizer.
        """
        if not gpu_indices:
            return []
        return ["--device", ",".join(f"Vulkan{i}" for i in gpu_indices)]

    @staticmethod
    def _backend_lacks_gpu_lib(binary: Optional[str] = None) -> bool:
        """Detect a split-library build proven to be CPU-only.

        Unknown and static layouts return False to avoid rejecting custom GPU builds.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        lib_dir = _llama_lib_dir(binary)
        if not lib_dir or not lib_dir.is_dir():
            return False
        if any(_lib_dir_has_ggml_backend(lib_dir, b) for b in ("vulkan", "cuda", "hip")):
            return False
        return any(_lib_dir_has_ggml_backend(lib_dir, b) for b in ("cpu", "base"))

    def is_vulkan_build(self) -> bool:
        """Whether the resolved llama-server uses Vulkan ordinals."""
        try:
            return self._is_vulkan_backend(self._find_llama_server_binary())
        except Exception:
            return False

    @staticmethod
    def _strip_device_extra_args(extra_args):
        """Drop device and main-GPU flags so gpu_ids remains authoritative."""
        return strip_shadowing_flags(
            extra_args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_device = True,
        )

    @staticmethod
    def _get_gpu_free_memory(binary: Optional[str] = None) -> list[tuple[int, int]]:
        """Query free memory per GPU. Returns ``(gpu_index, free_mib)`` sorted by
        index; empty if no supported GPU is reachable. Thin wrapper over
        ``_get_gpu_memory`` for callers that only need free VRAM."""
        return [(idx, free) for idx, free, _total in LlamaCppBackend._get_gpu_memory(binary)]

    @staticmethod
    def _apple_metal_memory_budget_bytes() -> int:
        """Unified-memory budget for GGUF context fitting on Apple Silicon.

        No GPU is enumerated on Metal, so the context would default to native and
        over-commit unified memory ("Compute error." at decode, #5118/#6529). Use a
        fraction of MLX's Metal working-set, else total RAM; 0 off Apple Silicon or
        when unresolvable, so callers skip the cap.
        """
        from utils.hardware import is_apple_silicon

        if not is_apple_silicon():
            return 0
        rec_bytes = 0
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                rec_bytes = int(mx.device_info().get("max_recommended_working_set_size") or 0)
        except Exception:
            rec_bytes = 0
        if rec_bytes <= 0:
            try:
                import psutil
                rec_bytes = int(psutil.virtual_memory().total)
            except Exception:
                return 0
        return int(rec_bytes * _APPLE_UNIFIED_MEMORY_FRACTION)

    @staticmethod
    def _get_gpu_memory(binary: Optional[str] = None) -> list[tuple[int, int, int]]:
        """Query free AND total memory per GPU.

        Order:
          1. ``nvidia-smi`` (NVIDIA CUDA hosts) -- respects
             ``CUDA_VISIBLE_DEVICES``.
          2. ``torch.cuda.mem_get_info`` -- universal fallback that works
             on AMD ROCm too (HIP runtime reuses the ``torch.cuda.*``
             namespace). Covers the AMD case for issue #5106 (nvidia-smi
             probe returned [] on AMD) and NVIDIA hosts missing
             ``nvidia-smi`` from PATH.

        On a Vulkan build the ggml Vulkan probe is authoritative, so the indices
        are ggml's compact Vulkan ordinals (the space the pin selects via
        ``--device Vulkan<i>``). It reports ``total`` for discrete cards and 0
        for an iGPU (shared RAM) so the fit falls back to free*frac there.
        Otherwise nvidia-smi / torch cover NVIDIA + AMD ROCm.

        Returns (gpu_index, free_mib, total_mib) sorted by index; empty if no
        supported GPU is reachable.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if LlamaCppBackend._is_vulkan_backend(binary):
            return LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
        # ── NVIDIA via nvidia-smi ────────────────────────────────────
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
                gpus: list[tuple[int, int, int]] = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    # Index and free required; skip a bad line rather than abandon
                    # the probe to the torch fallback.
                    try:
                        idx = int(parts[0])
                        free_mib = int(parts[1])
                    except ValueError:
                        continue
                    # Total parsed separately: a two-column line or a non-integer
                    # total ("N/A" on MIG/vGPU) keeps the GPU at total 0 (fit uses
                    # the free*frac fallback) instead of dropping it.
                    total_mib = 0
                    if len(parts) >= 3 and parts[2]:
                        try:
                            total_mib = int(parts[2])
                        except ValueError:
                            total_mib = 0
                    if allowed is not None and idx not in allowed:
                        continue
                    gpus.append((idx, free_mib, total_mib))
                # Match the docstring's sort-by-id guarantee (driver order isn't).
                gpus.sort(key = lambda g: g[0])
                if gpus:
                    return gpus
        except Exception as e:
            logger.debug(f"nvidia-smi probe failed: {e}")

        # ── Torch fallback (covers AMD ROCm and missing nvidia-smi) ──
        try:
            import torch

            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return []
            if not hasattr(torch.cuda, "mem_get_info"):
                return []
            # torch.cuda enumerates GPUs RELATIVE to the visibility mask. We
            # feed these IDs back into the subprocess as CVD, so visible ordinals
            # must be translated to physical indices first; otherwise CVD=2,3
            # gets rewritten to 0,1 and targets the wrong GPUs.
            # Match utils/hardware/hardware.py::_get_parent_visible_gpu_spec:
            # treat an empty mask (HIP_VISIBLE_DEVICES="") as "no GPUs" rather
            # than falling through. ``or`` would coerce "" to the wrong source.
            # Empty mask (CVD="") yields an empty list -> no GPUs, consistent
            # with the nvidia-smi path.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            gpus = []
            for ordinal in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(ordinal)
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                gpus.append((idx, free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)))
            # Match the nvidia-smi path's docstring guarantee of sorted-by-id.
            return sorted(gpus, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"torch GPU probe failed: {e}")
            return []

    @staticmethod
    def _vulkan_auto_gpu_memory(gpus: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Prefer discrete devices for automatic Vulkan placement."""
        discrete = [gpu for gpu in gpus if gpu[2] > 0]
        return discrete or gpus

    @staticmethod
    def _run_vulkan_probe(binary: Optional[str] = None) -> list[dict]:
        """Run ``_vulkan_probe.py`` and parse its per-device lines.

        Returns raw (uncapped) rows sorted by index:
        ``{"index", "free_mib", "total_mib", "is_igpu", "name"}``. The index is
        ggml's compact Vulkan ordinal -- the one the registry names
        ``Vulkan<index>`` and load_model pins with ``--device``, NOT the raw
        ``GGML_VK_VISIBLE_DEVICES`` space. A user-set ``GGML_VK_VISIBLE_DEVICES``
        is honored by ggml (passed through), so the list already reflects it.
        ``name`` is ggml's device description; "" from an older 4-column probe.
        [] when no Vulkan build or device is reachable.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return []
        binary_dir = _llama_lib_dir(binary)
        if not _lib_dir_has_ggml_backend(binary_dir, "vulkan"):
            return []

        env = child_env_without_native_path_secret()
        # Let ggml apply GGML_VK_VISIBLE_DEVICES; Python sees compact ordinals.
        if sys.platform != "win32":
            # Let the loader resolve sibling ggml libs next to the binary.
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{binary_dir}:{existing_ld}" if existing_ld else str(binary_dir)
            )
        probe_script = Path(__file__).with_name("_vulkan_probe.py")
        try:
            # UTF-8 to match the probe's stdout reconfigure: device names can be
            # non-ASCII, and the platform-default decode (cp1252) could throw.
            result = subprocess.run(
                [sys.executable, str(probe_script), str(binary_dir)],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = utf8_child_env(env),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                logger.debug(
                    f"vulkan GPU probe exited {result.returncode}: {result.stderr.strip()}"
                )
                return []
        except Exception as e:
            logger.debug(f"vulkan GPU probe failed: {e}")
            return []

        rows: list[dict] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            # 4 columns from an older probe (no name); 5 with the name column.
            if len(parts) not in (4, 5):
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "free_mib": int(parts[1]) // (1024 * 1024),
                        "is_igpu": parts[2] == "1",
                        "total_mib": int(parts[3]) // (1024 * 1024),
                        "name": parts[4].strip() if len(parts) == 5 else "",
                    }
                )
            except ValueError:
                continue
        rows.sort(key = lambda r: r["index"])
        return rows

    @staticmethod
    def vulkan_device_inventory(binary: Optional[str] = None) -> list[dict]:
        """UI-facing Vulkan device list: the devices llama-server will actually
        use, with real totals (an iGPU keeps its shared-RAM total here -- the
        caller labels it, unlike the fit which zeroes it). Same rows as
        ``_run_vulkan_probe``; names fall back to ``Vulkan<i>``.
        """
        rows = LlamaCppBackend._run_vulkan_probe(binary)
        for row in rows:
            if not row["name"]:
                row["name"] = f"Vulkan{row['index']}"
        return rows

    @staticmethod
    def _get_gpu_free_memory_vulkan(binary: Optional[str] = None) -> list[tuple[int, int, int]]:
        """Query free (and total) VRAM per device via the bundled ggml Vulkan backend.

        Fit-oriented view of ``_run_vulkan_probe``: returns (device_index,
        free_mib, total_mib) sorted by index. iGPUs leave a host-RAM margin (see
        ``_apply_igpu_host_reserve_mib``) and report total 0; discrete cards pass
        their real total through. [] when no Vulkan build or device is reachable.
        """
        gpus: list[tuple[int, int, int]] = []
        for row in LlamaCppBackend._run_vulkan_probe(binary):
            idx, free_mib, is_igpu = row["index"], row["free_mib"], row["is_igpu"]
            # iGPU "total" is shared RAM, not a VRAM budget -> keep 0 so the
            # fit stays on free*frac (the host reserve below is its
            # headroom); a discrete card passes its real total through.
            total_mib = 0 if is_igpu else row["total_mib"]
            capped = _apply_igpu_host_reserve_mib(free_mib, is_igpu)
            if capped < free_mib:
                logger.info(
                    f"Vulkan device VK{idx} is an integrated GPU sharing system "
                    f"RAM; reserving {free_mib - capped}MiB host headroom "
                    f"({free_mib}->{capped}MiB usable)"
                )
            gpus.append((idx, capped, total_mib))
        if gpus:
            logger.info(
                "Vulkan GPU memory detected: "
                + ", ".join(f"VK{idx}={free}MiB" for idx, free, _total in gpus)
            )
        return gpus

    @staticmethod
    def _available_system_memory_mib() -> Optional[int]:
        """Available system RAM in MiB (psutil, then /proc/meminfo), or None if
        neither is readable. On a unified-memory APU this, not the ROCm-reported
        VRAM, is the real ceiling: the weights load into shared system RAM."""
        try:
            import psutil
            return int(psutil.virtual_memory().available // (1024 * 1024))
        except Exception:
            pass
        try:
            with open("/proc/meminfo", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024  # kB -> MiB
        except Exception:
            pass
        return None

    @staticmethod
    def _apu_ram_shortfall_message(
        model_size_bytes: int,
        avail_mib: Optional[int],
        headroom_mib: int = 2048,
    ) -> Optional[str]:
        """On a unified-memory APU, return a user-facing refusal when the weights
        cannot fit in available system RAM (else None). Weights only: KV/context
        auto-reduce, so counting them too would refuse loads that would succeed.
        None avail (unknown RAM) never refuses."""
        if avail_mib is None:
            return None
        need_mib = model_size_bytes / (1024 * 1024)
        if need_mib <= avail_mib - headroom_mib:
            return None
        return (
            f"This model needs about {need_mib / 1024:.0f} GB but only about "
            f"{avail_mib / 1024:.0f} GB of memory is available. On a unified-memory "
            "APU the weights load into system RAM, so a larger model is stopped by "
            "the OS mid-load. Use a smaller or more quantized GGUF, or free memory "
            "(on WSL, raise the memory limit in .wslconfig)."
        )

    # Skip the wait when the last kill is older than this; the driver has
    # already reclaimed the prior process's allocations.
    _VRAM_SETTLE_WINDOW_S: float = 15.0

    @staticmethod
    def _wait_for_vram_settle(
        max_wait: float = 2.0,
        interval: float = 0.25,
        tolerance_mib: int = 256,
        since_kill: float = 0.0,
    ) -> None:
        """Poll ``_get_gpu_free_memory`` until free VRAM stabilises.

        The driver reclaims a dead process's allocations asynchronously, so
        sampling free memory in the kill-to-spawn window reads artificially low
        and pushes GPU selection toward needless CPU offload (the Apply-reload
        OOM bare-shell launches never see).

        Short-circuits on cold start, stale kill (older than
        ``_VRAM_SETTLE_WINDOW_S``), CPU-only hosts, probe exceptions, and GPU-set
        changes. ``max_wait`` bounds wall-clock time so a wedged ``nvidia-smi``
        can't extend the reload.
        """
        now = time.monotonic()
        if since_kill <= 0.0:
            return
        if now - since_kill > LlamaCppBackend._VRAM_SETTLE_WINDOW_S:
            return
        deadline = now + max_wait

        def _probe_or_none():
            if time.monotonic() >= deadline:
                return None
            try:
                return LlamaCppBackend._get_gpu_free_memory()
            except Exception:
                return None

        prev = _probe_or_none()
        if prev is None or not prev:
            return
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # Clip the nap so a near-zero ``max_wait`` is respected.
            time.sleep(min(interval, remaining))
            curr = _probe_or_none()
            if curr is None or not curr or len(curr) != len(prev):
                return
            prev_map = dict(prev)
            stable = True
            for idx, free in curr:
                if idx not in prev_map:
                    stable = False
                    break
                prev_free = prev_map[idx]
                # Adaptive: 2% of the larger sample dominates the 256 MiB floor.
                per_gpu_tol = max(tolerance_mib, int(max(free, prev_free) * 0.02))
                if abs(free - prev_free) >= per_gpu_tol:
                    stable = False
                    break
            if stable:
                return
            prev = curr

    # Free-VRAM fraction at which Unsloth pins the GPU directly instead of
    # deferring to ``--fit on``. 3% headroom: the compute buffer is now modelled in
    # the fit, so this only guards fragmentation + multi-GPU per-device CUDA context
    # (~2-3%); kept >= 3% as a floor (0.90 dropped 91-94% fits to CPU offload, #5106).
    _GPU_PIN_VRAM_FRACTION = 0.97

    # Fallback per-device tensor-mode compute buffer (MiB), used only when GGUF
    # dims are unavailable so _estimate_compute_buffer_bytes (the primary, derived
    # path) returns 0.
    _TENSOR_PARALLEL_BUFFER_RESERVE_MIB = 5120

    # Fixed per-device overhead on every GPU of a LAYER split (CUDA context +
    # scratch), beyond the conserved slot-scaling buffer. ~0.9 GB/device measured
    # (Qwen3.6-27B, b9625), independent of --parallel; reserved per extra GPU so a
    # tight layer split can't advertise a context that OOMs at load.
    _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024

    # KV cache types llama.cpp accepts in tensor mode. A quantized KV cache
    # aborts a --split-mode tensor load, so it's dropped for the tensor attempt.
    _TENSOR_PARALLEL_KV_TYPES = frozenset({"f16", "bf16", "f32"})

    # V cache types that llama.cpp can run WITHOUT flash attention. Only the V
    # axis has the dependency: a quantized V cache (q8_0/q4_0/q4_1/q5_0/q5_1/
    # iq4_nl) aborts init with "V cache quantization requires flash_attn", while
    # a quantized K cache runs fine without FA. So the flash-attn-off crash-
    # recovery fallback must reset a quantized V cache to f16 before it can
    # launch (and leaves K alone). These three are the only non-quantized types.
    _NON_QUANTIZED_KV_TYPES = frozenset({"f16", "bf16", "f32"})

    # Main-model placement settings that Manual mode owns. They must not leak
    # from Studio's parent environment into llama-server and silently override
    # the command assembled from the current request. Draft-model placement is
    # intentionally separate and remains available to speculative decoding.
    _MANUAL_PLACEMENT_ENV_VARS = (
        "LLAMA_ARG_CPU_MOE",
        "LLAMA_ARG_N_CPU_MOE",
        "LLAMA_ARG_N_GPU_LAYERS",
        "LLAMA_ARG_TENSOR_SPLIT",
        "LLAMA_ARG_FIT",
        "LLAMA_ARG_FIT_TARGET",
        "LLAMA_ARG_FIT_CTX",
    )

    # Eligibility must reject every inherited placement value replay removes.
    _CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS = (
        *_MANUAL_PLACEMENT_ENV_VARS,
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
        "LLAMA_ARG_SPLIT_MODE",
        "LLAMA_ARG_OVERRIDE_TENSOR",
    )

    # (binary, mtime, model) that aborted on --split-mode tensor this process (#6415
    # geometry limit, e.g. MQA n_head_kv=1). Model-keyed so one model's abort doesn't
    # skip tensor for others; tensor is tried by default, recorded only on a real abort.
    _tensor_split_abort_keys: set[tuple[str, int, str]] = set()

    @classmethod
    def _tensor_split_cache_key(
        cls, binary: Optional[str], model: Optional[str]
    ) -> Optional[tuple[str, int, str]]:
        """(path, mtime_ns, model) key; ns mtime re-probes a same-second binary swap."""
        if not binary or not model:
            return None
        try:
            mtime = Path(binary).stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (binary, mtime, model)

    @classmethod
    def _tensor_split_aborts(cls, binary: Optional[str], model: Optional[str]) -> bool:
        """True if (binary, model) aborted on --split-mode tensor this session."""
        key = cls._tensor_split_cache_key(binary, model)
        return key is not None and key in cls._tensor_split_abort_keys

    @classmethod
    def _record_tensor_split_abort(cls, binary: Optional[str], model: Optional[str]) -> None:
        """Remember a (binary, model) that aborts on --split-mode tensor."""
        key = cls._tensor_split_cache_key(binary, model)
        if key is not None:
            cls._tensor_split_abort_keys.add(key)

    @staticmethod
    def _windows_pip_nvidia_dll_dirs(prefix: str) -> list[str]:
        """Return DLL dirs from pip-installed CUDA wheels under
        ``<prefix>/Lib/site-packages/`` so llama-server.exe can load
        ``cudart64_X.dll`` / ``cublas64_X.dll`` without a system CUDA toolkit.
        Mirrors the Linux ``nvidia/cu*/lib`` LD_LIBRARY_PATH block, covering the
        Windows wheel layouts seen in the wild:
          * ``nvidia/<pkg>/bin`` -- legacy modular wheels.
          * ``nvidia/<pkg>/bin/x86_64`` and ``.../bin/x64`` -- CUDA 13 layout
            for unsuffixed packages (#5106).
          * ``nvidia/<pkg>/Library/bin`` (and arch subdirs) -- conda repacks.
          * ``torch/lib`` -- PyTorch's CUDA-bundled wheel can ship
            ``cudart64_*.dll`` here; mirrors install_llama_prebuilt.py.

        Walks with ``Path.iterdir`` not ``glob.glob`` so it's safe against
        Windows paths containing ``[`` or ``]`` (valid in usernames)."""
        site_packages = Path(prefix) / "Lib" / "site-packages"
        out: list[str] = []
        seen: set[str] = set()

        def _add(path: Path) -> None:
            if not path.is_dir():
                return
            key = os.path.normcase(os.path.abspath(str(path)))
            if key in seen:
                return
            seen.add(key)
            out.append(str(path))

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for pkg_dir in nvidia_root.iterdir():
                if not pkg_dir.is_dir():
                    continue
                # Arch-specific subdirs first so the explicit cudart64_X.dll
                # location wins over an empty sibling ``bin``.
                for sub in (
                    pkg_dir / "bin" / "x86_64",
                    pkg_dir / "bin" / "x64",
                    pkg_dir / "bin",
                    pkg_dir / "Library" / "bin" / "x86_64",
                    pkg_dir / "Library" / "bin" / "x64",
                    pkg_dir / "Library" / "bin",
                ):
                    _add(sub)
        _add(site_packages / "torch" / "lib")
        return out

    @staticmethod
    def _build_windows_path_dirs(binary_dir: str, prefix: str, cuda_path: str) -> list[str]:
        """Ordered PATH entries prepended so llama-server.exe resolves cudart /
        cublas DLLs: binary_dir, pip nvidia wheels, CUDA_PATH/bin, .../bin/x64.
        Extracted so test_windows_gpu_detection_mock tests the real logic. #5106."""
        path_dirs = [binary_dir]
        path_dirs.extend(LlamaCppBackend._windows_pip_nvidia_dll_dirs(prefix))
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.isdir(cuda_bin):
                path_dirs.append(cuda_bin)
            cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
            if os.path.isdir(cuda_bin_x64):
                path_dirs.append(cuda_bin_x64)
        return path_dirs

    @staticmethod
    def _llama_server_env_for_binary(binary: str) -> dict[str, str]:
        """Build a subprocess env that lets llama-server resolve native libs."""
        env = child_env_without_native_path_secret()
        # _llama_lib_dir resolves the llama-server symlink to the real build/bin.
        binary_dir = str(_llama_lib_dir(binary))

        if sys.platform == "win32":
            # Ordering: see _build_windows_path_dirs. #5106.
            path_dirs = LlamaCppBackend._build_windows_path_dirs(
                binary_dir,
                sys.prefix,
                os.environ.get("CUDA_PATH", ""),
            )
            existing_path = env.get("PATH", "")
            env["PATH"] = ";".join(path_dirs) + ";" + existing_path

            # ROCm: the prebuilt bundles rocblas.dll but NOT the Tensile
            # kernel files (rocblas/library/*.dat + *.hsaco); the DLL searches
            # <binary_dir>/rocblas/library/ which doesn't exist.
            _hip_path = os.environ.get("HIP_PATH", os.environ.get("ROCM_PATH", ""))
            if _hip_path:
                _rocblas_lib = os.path.join(_hip_path, "bin", "rocblas", "library")
                if os.path.isdir(_rocblas_lib):
                    env.setdefault("ROCBLAS_TENSILE_LIBPATH", _rocblas_lib)
        else:
            # Linux: LD_LIBRARY_PATH for shared libs next to the binary plus
            # CUDA runtime libs (libcudart, libcublas, etc.)
            import platform

            lib_dirs = []
            # WSL: system HIP before the bundle's (which segfaults on /dev/dxg).
            lib_dirs.extend(_wsl_system_rocm_lib_dirs())
            if lib_dirs:
                env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
            # Native Linux AMD: system ROCm libs before the bundle's HIP runtime,
            # which can be incompatible with the host amdkfd driver.
            lib_dirs.extend(_native_linux_system_rocm_lib_dirs(binary_dir))
            lib_dirs.append(binary_dir)
            _arch = platform.machine()  # x86_64, aarch64, etc.

            # Pip-installed nvidia CUDA runtime libs. The prebuilt binary links
            # libcudart.so.13 / libcublas.so.13 which live here, not in
            # /usr/local/cuda.
            import glob as _glob

            for _nv_pattern in [
                os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", _sub, "lib")
                for _sub in ("cu*", "cudnn", "nvjitlink")
            ]:
                for _nv_dir in _glob.glob(_nv_pattern):
                    if os.path.isdir(_nv_dir):
                        lib_dirs.append(_nv_dir)

            for cuda_lib in [
                "/usr/local/cuda/lib64",
                f"/usr/local/cuda/targets/{_arch}-linux/lib",
                # Fallback CUDA compat paths (e.g. binary built with CUDA 12
                # where default /usr/local/cuda is CUDA 13+).
                "/usr/local/cuda-12/lib64",
                "/usr/local/cuda-12.8/lib64",
                f"/usr/local/cuda-12/targets/{_arch}-linux/lib",
                f"/usr/local/cuda-12.8/targets/{_arch}-linux/lib",
            ]:
                if os.path.isdir(cuda_lib):
                    lib_dirs.append(cuda_lib)
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            new_ld = ":".join(lib_dirs)
            env["LD_LIBRARY_PATH"] = f"{new_ld}:{existing_ld}" if existing_ld else new_ld

        return env

    @classmethod
    def _clear_manual_placement_env(cls, env: dict[str, str]) -> None:
        """Remove inherited main-model placement owned by Manual mode."""
        for name in cls._MANUAL_PLACEMENT_ENV_VARS:
            env.pop(name, None)

    @staticmethod
    def _select_gpus(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
        usable_fraction: Optional[float] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        per_device_overhead_bytes: int = 0,
        min_gpus: int = 1,
    ) -> tuple[Optional[list[int]], bool]:
        """Pick GPU(s) for a model from estimated VRAM and free memory.

        ``min_gpus`` (default 1, capped at ``len(gpus)``) keeps a downgraded
        tensor/multi-GPU request spread instead of collapsing to one card.

        ``model_size_bytes`` should include weights and estimated KV cache.
        ``usable_fraction`` (default ``_GPU_PIN_VRAM_FRACTION``) provides
        headroom for compute buffers, CUDA context, and other runtime
        overhead; callers lower it when MTP reserves VRAM for a draft model.
        ``total_by_idx`` (index -> total MiB) makes the headroom an ABSOLUTE
        ``(1 - fraction) * total`` per GPU instead of a fraction of free.
        ``per_device_overhead_bytes`` is the fixed layer-split cost per GPU beyond
        the first; a k-GPU pin must hold ``model + (k-1) * overhead`` or it can OOM
        a device after -ngl -1 (no --fit fallback). Single-GPU adds none.

        Returns (gpu_indices, use_fit):
          - ([1], False)       fits on 1 GPU at the headroom threshold
          - ([1, 2], False)    needs 2 GPUs
          - (None, True)       too large, let --fit handle it
        """
        if not gpus:
            return None, True

        min_gpus = max(1, min(min_gpus, len(gpus)))
        model_size_mib = model_size_bytes / (1024 * 1024)
        if usable_fraction is None:
            usable_fraction = LlamaCppBackend._GPU_PIN_VRAM_FRACTION
        overhead_mib = per_device_overhead_bytes / (1024 * 1024)

        # Per-GPU usable budget: free - (1-frac)*total when total is known, else
        # the legacy free*frac (also covers a total-0 two-column probe).
        def _usable(idx: int, free_mib: int) -> float:
            t = total_by_idx.get(idx, 0) if total_by_idx else 0
            if t > 0:
                return max(0.0, free_mib - (1.0 - usable_fraction) * t)
            return free_mib * usable_fraction

        # Rank by usable budget (free - reserve), not raw free: a more-used large
        # card can have less usable room than a less-used small one.
        ranked = sorted(gpus, key = lambda g: _usable(g[0], g[1]), reverse = True)

        # Cap a downgraded multi-GPU request to the usable count so it doesn't pull
        # in a near-full card to hit min_gpus. No-op for the default min_gpus == 1.
        usable_count = sum(1 for idx, free_mib in ranked if _usable(idx, free_mib) > overhead_mib)
        min_gpus = max(1, min(min_gpus, usable_count or 1))

        # Try 1 GPU at the usable-VRAM threshold (only when one device is allowed).
        if min_gpus <= 1 and _usable(ranked[0][0], ranked[0][1]) >= model_size_mib:
            return [ranked[0][0]], False

        # Try N GPUs (most-free first); each past the first adds per-device overhead.
        # Require at least min_gpus devices before accepting a fit.
        cumulative = 0.0
        selected = []
        for idx, free_mib in ranked:
            selected.append(idx)
            cumulative += _usable(idx, free_mib)
            if (
                len(selected) >= min_gpus
                and cumulative >= model_size_mib + (len(selected) - 1) * overhead_mib
            ):
                return sorted(selected), False

        # Too large even for all GPUs; let --fit handle it
        logger.debug(
            "Model does not fit in available GPU memory, falling back to --fit",
            model_size_mib = round(model_size_mib, 2),
            ranked_gpus = ranked,
        )
        return None, True

    @staticmethod
    def _select_gpus_split_aware(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
        usable_fraction: Optional[float] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        per_device_overhead_bytes: int = 0,
        min_gpus: int = 1,
        split_extra_bytes: int = 0,
    ) -> tuple[Optional[list[int]], bool]:
        """``_select_gpus``, re-checked at the multi-device context-compute rate.

        The device count comes out of the footprint, so the first pass must price the
        compute buffer at the single-device rate -- but a layer split replicates a BIGGER
        one per card (see ``_CTX_COMPUTE_SPLIT_MULT``). ``split_extra_bytes`` is that
        per-device step (0 when llama.cpp declines pipeline parallelism): charge it and
        re-select once the count is known to be > 1, exact in one retry since the step is
        count-independent. The bigger overhead can cut ``_select_gpus``'s usable-card
        count to one, and a lone card is not a split, so a retry landing below two
        devices is re-priced without the delta. Still no fit returns (None, True) ->
        --fit on, i.e. CPU offload rather than a pin that OOMs at launch."""
        gpu_indices, use_fit = LlamaCppBackend._select_gpus(
            model_size_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes,
            min_gpus = min_gpus,
        )
        if use_fit or split_extra_bytes <= 0 or not gpu_indices or len(gpu_indices) < 2:
            return gpu_indices, use_fit
        retry_indices, retry_fit = LlamaCppBackend._select_gpus(
            model_size_bytes + split_extra_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes + split_extra_bytes,
            min_gpus = min_gpus,
        )
        if not retry_fit and retry_indices and len(retry_indices) >= 2:
            return retry_indices, retry_fit
        single, single_fit = LlamaCppBackend._select_gpus(
            model_size_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes,
            min_gpus = 1,
        )
        if not single_fit and single is not None and len(single) == 1:
            return single, False
        return retry_indices, retry_fit

    @staticmethod
    def _every_gpu_holds_reserve(usable_mib: Iterable[float], reserve_bytes: int) -> bool:
        """Whether every card in a layer-split subset can hold the reserve it replicates
        (flat per-device overhead + its own context-compute copy). A pooled budget cannot
        see this: in the sum a roomy card's spare VRAM covers a nearly full card's copy,
        and the small card OOMs at launch. Bites when the subset loop cannot start at
        one GPU (``_auto_min_gpus`` >= 2, set by every tensor -> layer downgrade);
        starting at one, the failed n-1 subset already bounds the smallest card from
        below by the reserve."""
        reserve_mib = reserve_bytes / (1024 * 1024)
        values = list(usable_mib)
        return bool(values) and all(usable >= reserve_mib for usable in values)

    @staticmethod
    def _cap_ctx_to_per_device_reserve(
        ctx: int,
        usable_mib: Iterable[float],
        reserve_bytes_fn: Callable[[int], int],
        min_ctx: int = 4096,
    ) -> int:
        """Largest context in [``min_ctx``, ``ctx``] whose replicated per-device reserve
        still fits the SMALLEST card of a layer-split subset; 0 when even ``min_ctx``
        does not. The reserve shrinks with the context, so a subset
        ``_every_gpu_holds_reserve`` turns down at the pooled context usually serves a
        lower one: 40000 + 2500 MiB free holds 31488, not the 4096 a drop would leave.
        Bisects the reserve, which is monotone in context on every branch but
        proportional on only some: deepseek4's flat indexer term makes a closed form
        answer 43341 on a 4000 MiB card, whose reserve there is 6048 MiB. The MiB
        comparison mirrors ``_every_gpu_holds_reserve``, so that gate accepts what this
        returns."""
        values = list(usable_mib)
        if not values or ctx < min_ctx:
            return 0
        smallest = min(values)
        lo, hi, best = min_ctx, ctx, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if reserve_bytes_fn(mid) / (1024 * 1024) <= smallest:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        best = (best // 256) * 256  # same alignment _fit_context_to_vram uses
        return best if best >= min_ctx else 0

    # ── KV cache VRAM estimation ─────────────────────────────────────

    def _can_estimate_kv(self) -> bool:
        """True if we have enough GGUF metadata to estimate KV cache size."""
        if self._n_layers is None:
            return False
        # MLA: kv_lora_rank suffices (K-only cache).
        if self._kv_lora_rank is not None:
            return True
        # New-style: need explicit key AND value dimensions.
        if self._kv_key_length is not None and self._kv_value_length is not None:
            return True
        # Legacy: need embedding_length + a head count (scalar or per-layer).
        return self._embedding_length is not None and (
            self._n_kv_heads is not None
            or self._n_heads is not None
            or self._n_kv_heads_by_layer is not None
        )

    def _kv_heads_for_layer(self, layer_idx: int, fallback: int) -> int:
        if self._n_kv_heads_by_layer is not None and layer_idx < len(self._n_kv_heads_by_layer):
            return self._n_kv_heads_by_layer[layer_idx]
        return fallback

    def _recurrent_state_bytes(self, n_parallel: int = 1) -> int:
        """VRAM for the conv + recurrent state of linear-attention layers.

        Hybrids pair an MLA/attention cache with llama_memory_recurrent, which
        allocates per recurrent layer and per sequence, always f32, offloaded
        beside the KV cache. Unlike KV this does not scale with context, so it
        is the whole reservation at short contexts and the bisection in
        _fit_context_to_vram cannot shrink it away.

        Sizes follow llama_hparams::n_embd_r/n_embd_s. Kimi KDA only: Mamba
        needs ssm.group_count, which is not parsed, so those models keep the
        previous behaviour rather than get a wrong number.
        """
        if not self._n_kv_heads_by_layer or not self._n_layers or not self._kda_head_dim:
            return 0
        n_recurrent = sum(1 for i in range(self._n_layers) if self._kv_heads_for_layer(i, 1) == 0)
        if n_recurrent == 0:
            return 0
        head_dim, n_head = self._kda_head_dim, self._n_heads or 1
        d_conv = self._ssm_conv_kernel or 4
        n_embd_r = 3 * max(0, d_conv - 1) * n_head * head_dim
        n_embd_s = head_dim * head_dim * n_head
        return int(n_recurrent * (n_embd_r + n_embd_s) * 4 * max(1, n_parallel))

    def _legacy_head_dim(self) -> int:
        """Head-dim fallback for GGUFs without explicit key/value dims. Reached
        only via the legacy branch of _can_estimate_kv(), so _embedding_length
        is non-None here."""
        return self._embedding_length // self._n_heads if self._n_heads else 128  # type: ignore[operator]

    def _max_kv_value_width(
        self,
        default_len: int,
        swa_len: Optional[int] = None,
    ) -> int:
        """llama.cpp's hparams.n_embd_v_gqa_max() over every model layer."""
        n_layers = self._n_layers or 1
        n_kv = self._n_kv_heads or self._n_heads or 1
        if self._sliding_window_pattern is None:
            max_len = max(default_len, swa_len or default_len)
            return max(
                self._kv_heads_for_layer(layer_idx, n_kv) * max_len for layer_idx in range(n_layers)
            )
        return max(
            self._kv_heads_for_layer(layer_idx, n_kv)
            * (
                (swa_len or default_len)
                if (
                    layer_idx < len(self._sliding_window_pattern)
                    and self._sliding_window_pattern[layer_idx]
                )
                else default_len
            )
            for layer_idx in range(n_layers)
        )

    def _estimate_kv_cache_bytes(
        self,
        n_ctx: int,
        cache_type_kv: Optional[str] = None,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        ctx_checkpoints: int = 0,
        flash_attn: bool = True,
    ) -> int:
        """Estimate KV cache VRAM for a given context length.

        5-path architecture-aware estimation:
          1. MLA      -- compressed KV latent + RoPE, K-only (no separate V)
          2. Hybrid   -- only attention layers need KV (Mamba layers don't)
          3. SWA      -- sliding-window layers use compact or full cache cells
          4. GQA      -- standard full KV with explicit key/value dimensions
          5. Legacy   -- fallback using embed // n_heads

        Server-flag knobs (mirror llama-server's CLI):
          swa_full        -- --swa-full: SWA layers cache full n_ctx (path 3->4).
          n_parallel      -- --parallel slots: controls per-slot stream padding.
          kv_unified      -- --kv-unified: one shared stream vs one per slot.
          n_ubatch        -- --ubatch-size: SWA cache's processing headroom.
          ctx_checkpoints -- --ctx-checkpoints: N SWA snapshots per slot.
          flash_attn      -- False pads variable-width V tensors to the model max.

        Returns 0 if metadata is insufficient.
        """
        if not self._can_estimate_kv() or n_ctx <= 0:
            return 0

        n_layers = self._n_layers  # type: ignore[assignment]
        # Gemma 3n / Gemma 4 reuse earlier KV in the last ``shared_kv_layers``
        # blocks (no cache). Floor at 1 so a bad GGUF can't zero out KV.
        shared = self._shared_kv_layers or 0
        n_layers_kv = max(1, n_layers - shared)
        n_kv = self._n_kv_heads or self._n_heads or 1  # type: ignore[assignment]

        # Bytes per element depends on KV cache quantization
        bpe_k = _kv_bytes_per_elem(cache_type_kv)
        # The automatic FA-off retry rewrites an invalid quantized V cache to
        # f16. Pricing that viable retry here avoids under-reserving it.
        bpe_v = bpe_k if flash_attn else max(bpe_k, _kv_bytes_per_elem("f16"))

        slots, streams, cells_per_stream = _kv_cache_cell_layout(n_ctx, n_parallel, kv_unified)
        total_cells = cells_per_stream * streams
        ubatch = max(
            0,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )

        # Path 1: MLA (DeepSeek-V2/V3, GLM-4.7, GLM-5, Kimi-K2.5)
        # One compressed KV latent per token/layer (shared across heads); V is
        # reconstructed from it, no separate V cache. key_length = kv_lora_rank
        # + rope_dim. MLA GGUFs set head_count_kv=1; default to 1 if absent to
        # avoid falling back to n_heads (e.g. 128 for DeepSeek-V3) which 128x's.
        if self._kv_lora_rank is not None:
            n_kv_mla = self._n_kv_heads or 1
            rope_dim = self._key_length_mla or 64
            key_len = self._kv_key_length or (self._kv_lora_rank + rope_dim)
            # Hybrid MLA (Kimi-K3): head_count_kv is a per-layer array whose KDA
            # linear-attention layers are 0 and hold a constant-size state instead
            # of a growing cache. Only the non-zero layers allocate KV, so counting
            # every block overstates it by the hybrid ratio (93 blocks vs 24 real
            # ones is 3.9x, or 78 GiB of phantom cache at 1M context) and shrinks
            # the auto-fit context for no reason. Uniform MLA keeps n_layers_kv.
            n_mla_layers = n_layers_kv
            if self._n_kv_heads_by_layer:
                attn = sum(
                    1 for i in range(n_layers_kv) if self._kv_heads_for_layer(i, n_kv_mla) > 0
                )
                n_mla_layers = max(1, attn)
            return int(
                n_mla_layers * total_cells * n_kv_mla * key_len * bpe_k
            ) + self._recurrent_state_bytes(n_parallel)

        key_len = self._kv_key_length
        val_len = self._kv_value_length

        # Path 2: Hybrid Mamba/Attention (Qwen3.5-27B, Qwen3.5-35B-A3B)
        # Only 1 in N layers is attention; the rest are Mamba (no KV cache).
        if self._ssm_inner_size is not None and self._full_attention_interval is not None:
            fai = self._full_attention_interval
            n_attn = -(-n_layers // fai) if fai > 0 else n_layers  # ceiling division
            if key_len is not None and val_len is not None:
                v_width = n_kv * val_len if flash_attn else self._max_kv_value_width(val_len)
                return int(n_attn * total_cells * (n_kv * key_len * bpe_k + v_width * bpe_v))
            head_dim = self._legacy_head_dim()
            return int(n_attn * total_cells * n_kv * 2 * head_dim * bpe_k)

        # Path 3: Sliding window (Gemma 2/3/3n/4, gpt-oss, Cohere2 ...). Pattern
        # from the resolver; if absent, falls through to the legacy 1/4-global
        # heuristic. --parallel N accounting (verified against llama-server):
        # non-SWA cells total n_ctx across streams. Compact SWA adds one processing
        # micro-batch to the window allowance and pads to 256 cells; unified mode
        # holds all slots in one stream, while non-unified mode has one stream per
        # slot. --swa-full expands SWA to each stream's full context.
        if (
            self._sliding_window is not None
            and self._sliding_window > 0
            and key_len is not None
            and val_len is not None
        ):
            swa = self._sliding_window
            if swa_full:
                swa_cells_total = total_cells
            else:
                swa_limit = swa * (slots if kv_unified else 1) + ubatch
                swa_cells_per_stream = min(cells_per_stream, swa_limit)
                swa_cells_per_stream = _pad_kv_cells(swa_cells_per_stream)
                swa_cells_total = swa_cells_per_stream * streams
            key_len_swa = self._kv_key_length_swa or key_len
            val_len_swa = self._kv_value_length_swa or val_len
            padded_v_width = None if flash_attn else self._max_kv_value_width(val_len, val_len_swa)
            if self._sliding_window_pattern is not None:
                global_bytes = 0.0
                swa_bytes = 0.0
                checkpoint_extra_per_slot = 0.0
                # Only layers that allocate their own KV; trailing shared layers
                # reuse earlier caches.
                for layer_idx in range(n_layers_kv):
                    layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
                    is_swa = (
                        layer_idx < len(self._sliding_window_pattern)
                        and self._sliding_window_pattern[layer_idx]
                    )
                    layer_key_bytes = layer_n_kv * (key_len_swa if is_swa else key_len) * bpe_k
                    layer_value_bytes = (
                        layer_n_kv * (val_len_swa if is_swa else val_len)
                        if padded_v_width is None
                        else padded_v_width
                    ) * bpe_v
                    layer_kv_bytes = layer_key_bytes + layer_value_bytes
                    if is_swa:
                        swa_bytes += swa_cells_total * layer_kv_bytes
                        if ctx_checkpoints > 0 and not swa_full:
                            checkpoint_extra_per_slot += ctx_checkpoints * swa * layer_kv_bytes
                    else:
                        global_bytes += total_cells * layer_kv_bytes
                return int(global_bytes + swa_bytes + slots * checkpoint_extra_per_slot)
            n_global = max(1, n_layers_kv // 4)
            n_swa = n_layers_kv - n_global
            global_v_width = n_kv * val_len if padded_v_width is None else padded_v_width
            swa_v_width = n_kv * val_len_swa if padded_v_width is None else padded_v_width
            kv_per_token = n_kv * key_len * bpe_k + global_v_width * bpe_v
            kv_per_token_swa = n_kv * key_len_swa * bpe_k + swa_v_width * bpe_v
            global_bytes = n_global * total_cells * kv_per_token
            swa_bytes = n_swa * swa_cells_total * kv_per_token_swa
            checkpoint_extra_per_slot = (
                ctx_checkpoints * n_swa * swa * kv_per_token_swa
                if ctx_checkpoints > 0 and not swa_full
                else 0.0
            )
            return int(global_bytes + swa_bytes + slots * checkpoint_extra_per_slot)

        # Path 4: Standard GQA with explicit key/value dimensions
        if key_len is not None and val_len is not None:
            padded_v_width = None if flash_attn else self._max_kv_value_width(val_len)
            bytes_per_cell = 0.0
            for layer_idx in range(n_layers_kv):
                layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
                v_width = layer_n_kv * val_len if padded_v_width is None else padded_v_width
                bytes_per_cell += layer_n_kv * key_len * bpe_k + v_width * bpe_v
            return int(total_cells * bytes_per_cell)

        # Path 5: Legacy fallback (old GGUFs without explicit dimensions)
        head_dim = self._legacy_head_dim()
        return int(2 * n_kv * head_dim * n_layers_kv * total_cells * bpe_k)

    def _draft_backend_for(self, drafter_path: str) -> Optional["LlamaCppBackend"]:
        """Lightweight backend with a drafter GGUF's metadata, to size its own KV
        via _estimate_kv_cache_bytes. Cached per path; None if unreadable."""
        cache = getattr(self, "_draft_backend_cache", None)
        if cache is not None and cache[0] == drafter_path:
            return cache[1]
        db: Optional[LlamaCppBackend] = None
        try:
            db = LlamaCppBackend.__new__(LlamaCppBackend)
            for attr in (
                "_context_length",
                "_n_layers",
                "_n_kv_heads",
                "_n_heads",
                "_embedding_length",
                "_kv_key_length",
                "_kv_value_length",
                "_kv_lora_rank",
                "_sliding_window",
                "_sliding_window_pattern",
                "_ssm_inner_size",
                "_full_attention_interval",
                "_key_length_mla",
                "_n_kv_heads_by_layer",
                "_kv_key_length_swa",
                "_kv_value_length_swa",
                "_shared_kv_layers",
                "_nextn_predict_layers",
            ):
                setattr(db, attr, None)
            db._model_identifier = "mtp-draft"
            db._read_gguf_metadata(drafter_path)
        except Exception as e:  # unreadable drafter -> caller falls back
            logger.debug(f"Could not read drafter GGUF for MTP budget: {e}")
            db = None
        self._draft_backend_cache = (drafter_path, db)
        return db

    def _mtp_draft_kv_bytes(
        self,
        n_ctx: int,
        *,
        drafter_path: Optional[str] = None,
        draft_cache_type_k: Optional[str] = None,
        draft_cache_type_v: Optional[str] = None,
        n_parallel: int = 1,
        swa_full: bool = False,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """Draft KV cache bytes at n_ctx, sized from GGUF dims (K and V types are
        independent). Separate drafter (Gemma): its own KV via _estimate_kv_cache_bytes
        at the heavier type. Embedded head (Qwen): nextn_predict_layers attention
        layers from the main dims. None when dims are missing (flat fallback)."""
        if n_ctx <= 0:
            return None
        bpe_k = _kv_bytes_per_elem(draft_cache_type_k)
        bpe_v = _kv_bytes_per_elem(draft_cache_type_v)
        if drafter_path:
            db = self._draft_backend_for(drafter_path)
            if db is None or not db._can_estimate_kv():
                return None
            # Gemma 4 assistant layers share the target context's final global
            # and SWA KV tensors, so only the drafter weights add memory.
            if getattr(db, "_architecture", None) == "gemma4-assistant":
                return 0
            heavier = draft_cache_type_k if bpe_k >= bpe_v else draft_cache_type_v
            # The drafter uses the main model's slot and stream layout, so its
            # compact SWA and per-stream padding must follow the same settings.
            kv = db._estimate_kv_cache_bytes(
                n_ctx,
                heavier,
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )
            return kv if kv > 0 else None
        nextn = self._nextn_predict_layers or 0
        n_kv = self._n_kv_heads or self._n_heads
        k_len = self._kv_key_length
        v_len = self._kv_value_length
        if not (nextn and n_kv and k_len and v_len):
            return None
        # The embedded MTP head is one draft layer, so a quantized draft KV can't
        # amortize its overhead and fits *less* context than f16 (llama.cpp#24102).
        # Floor it at f16: a quantized override is priced as f16, f32 keeps its 4
        # bytes. The separate-drafter branch is multi-layer, so it keeps its type.
        f16_bpe = _kv_bytes_per_elem("f16")
        bpe_k = max(bpe_k, f16_bpe)
        bpe_v = max(bpe_v, f16_bpe)
        _, streams, cells_per_stream = _kv_cache_cell_layout(n_ctx, n_parallel, kv_unified)
        v_width = n_kv * v_len
        if not flash_attn:
            v_width = self._max_kv_value_width(
                v_len,
                self._kv_value_length_swa,
            )
        return int(nextn * (n_kv * k_len * bpe_k + v_width * bpe_v) * cells_per_stream * streams)

    def _estimate_mtp_overhead_bytes(
        self,
        n_ctx: int,
        *,
        spec_draft_n_max: int = 0,
        draft_cache_type_k: Optional[str] = None,
        draft_cache_type_v: Optional[str] = None,
        drafter_path: Optional[str] = None,
        draft_weights_bytes: int = 0,
        n_parallel: int = 1,
        mtp_keeps_target_ctx: bool = True,
        swa_full: bool = False,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """MTP draft reserve at ``n_ctx`` = draft KV (grows with ctx) + separate-
        drafter weights + (MTP + MLA only) a duplicated target KV context. The
        verify buffer rides in the ctx-fit headroom (no tuned constant). None when
        the draft KV can't be sized (caller keeps the flat fallback).
        ``draft_weights_bytes`` is the drafter file size (0 for embedded).
        ``mtp_keeps_target_ctx`` is True for MTP draft modes (which keep the
        duplicated target context) and False for separate-drafter spec modes
        (draft-simple/draft-eagle3), which do not."""
        draft_kv = self._mtp_draft_kv_bytes(
            n_ctx,
            drafter_path = drafter_path,
            draft_cache_type_k = draft_cache_type_k,
            draft_cache_type_v = draft_cache_type_v,
            n_parallel = n_parallel,
            swa_full = swa_full,
            kv_unified = kv_unified,
            n_ubatch = n_ubatch,
            flash_attn = flash_attn,
        )
        weights = max(0, draft_weights_bytes)
        # MLA models (GLM-5.x, DeepSeek, Kimi-K2) under MTP keep a *second* full copy
        # of the target model's KV context for draft verification -- llama.cpp's
        # `ctx_tgt=yes` -- allocated at f16 regardless of the main cache type. It is
        # ~the main KV again and dwarfs the embedded draft head (GLM-5.2 @ 1M ctx:
        # a ~2 GiB head next to a ~89 GiB target copy), so omitting it lets auto-fit
        # pick a context that fits on paper but OOMs cublasCreate at the first
        # decode. Gated on both MLA (kv_lora_rank present) and the engaged mode
        # actually being MTP: non-MLA MTP (Qwen/Gemma) keeps no such copy, and the
        # separate-drafter spec modes (draft-simple/draft-eagle3) load a small
        # distinct drafter with its own KV -- already counted in draft_kv/weights --
        # rather than duplicating the target, so they must not be charged for it.
        target_ctx_copy = 0
        if mtp_keeps_target_ctx and self._kv_lora_rank is not None:
            target_ctx_copy = self._estimate_kv_cache_bytes(
                n_ctx,
                "f16",
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )
        if draft_kv is None:
            # KV unsized (exotic/remote drafter): still reserve known weights + any
            # MLA target copy so a large config can't launch over budget (the small
            # unsized draft KV rides in the cushion). Nothing known -> None, so the
            # caller keeps the flat fallback.
            total = weights + target_ctx_copy
            return total if total > 0 else None
        return draft_kv + weights + target_ctx_copy

    _DEFAULT_N_UBATCH = _DEFAULT_LLAMA_N_UBATCH
    _COMPUTE_BUFFER_SAFETY = 1.15  # upper-bound margin on the compute-buffer estimate
    # Soft VRAM the modeled terms omit; charged to the fit budget on tight tiers (#6682).
    _CUDA_CONTEXT_RESERVE_BYTES = 320 * 1024 * 1024  # CUDA ctx + cuBLAS workspace (~330 MiB)
    _MMPROJ_VRAM_SAFETY = 1.4  # mmproj worst-case buffer vs file size (runtime ~1.3x)
    _MTP_DRAFT_COMPUTE_BYTES = 224 * 1024 * 1024  # MTP draft decode graph beyond its KV
    # The flash-attn KQ mask + attention scratch grow ~linearly with context; the flat
    # _estimate_compute_buffer_bytes term only covers ctx -> 0. The per-token rate
    # depends on the KV cache type: a QUANTIZED cache (q8_0/q5/q4/iq4) needs a
    # context-sized dequant scratch that scales with n_embd, measured at 0.74-2.02 x
    # n_embd across Qwen3.5/3.6 (2B/4B/9B/27B) and Gemma-4 (12B/31B) at q8_0; an
    # f16/bf16/f32 cache skips the dequant and pays only the KQ mask, a flat n_ubatch*2
    # bytes per context token regardless of n_embd (measured 1024 B/tok on Qwen-9B and
    # Gemma-31B alike). So Qwen3.5-4B at 256k is 1.30 GiB at q8_0 vs 0.31 GiB at f16.
    # 2.25 covers the worst quantized case (Qwen3.5-4B, ~2.0x) plus the under-modeled
    # flat base; the mask safety covers the f16 base gap. Without this term, tight tiers
    # at extreme context over-pin and spill to CPU (the 3% cushion is only ~0.25 GiB on
    # an 8 GB card, far below the ~1-2.4 GiB quantized buffer at 256k): e.g. Qwen3.5-4B
    # Q4 at 256k needs ~8.5 GiB on a real 8 GB card (weights 2.4 + KV 4.3 + compute 1.3
    # + CUDA ctx) -> CPU spill; with this reserve the auto context caps to ~210k, fits.
    _CTX_COMPUTE_BYTES_PER_EMBD = 2.25  # quantized KV, regular attention (dequant scratch)
    _CTX_COMPUTE_BYTES_PER_EMBD_MLA = 1.25  # quantized KV, MLA (compressed attn: measured 0.94x)
    _CTX_COMPUTE_F16_MASK_SAFETY = 1.5  # f16/bf16/f32 KV: KQ mask only (n_ubatch*2 B/tok)
    # Per-device f16 ctx-linear rate steps 4x once the model spans >1 device under
    # `-sm layer`: pipeline parallelism sets the ggml scheduler's n_copies to
    # GGML_SCHED_MAX_COPIES == 4, and the [n_kv, n_ubatch] f16 KQ mask is a
    # GGML_TENSOR_FLAG_INPUT, so 1 live copy becomes 4. Hence a step on "is split", not
    # a ramp in device count: measured 2.00 B/tok/ubatch on 1 GPU and 8.00 on 2-8 GPUs,
    # architecture- and --parallel-independent. Charging 1x on a split under-reserves
    # 2.67x (= 8/(2*1.5)). llama.cpp declines it in _pipeline_parallel_disabled_by_args.
    _CTX_COMPUTE_SPLIT_MULT = 4
    # DeepSeek-V4 (deepseek4): its lightning indexer + sparse attention reserve a large
    # context-scaling compute buffer the rates above miss (present even with an f16
    # cache). Measured on UD-Q4_K_XL (ub=512): ~2 GiB at 16k -> ~65.5 GiB at 1M. Without
    # it auto-fit commits the full 1M train context, OOMs the reserve, and spills to CPU.
    _DSV4_CTX_COMPUTE_FLAT_BYTES = 2 * 1024**3  # ctx-independent indexer scratch
    _DSV4_CTX_COMPUTE_BYTES_PER_TOK = 72000  # per token at ub=512 (~72 GiB at 1M)

    # Inkling (inkling): with the reserve fix in the bundled llama.cpp (full-cache
    # reserve context reports the whole cache as position-contiguous, so the
    # worst-case graph is the banded flash path instead of the dense-bias
    # fallback), the ctx-linear compute term is just the banded KQ-mask cont:
    # measured on UD-IQ1_S (ub=512) 64K -> 1M total-VRAM slope of 50.6 KiB/tok,
    # of which 45,056 B/tok is KV -> ~5.6 KiB/tok compute. 8192 adds ~1.5x
    # headroom. (Pre-fix builds reserved the dense fallback at ~402 KiB/tok and
    # could not load large contexts at all.)
    _INKLING_CTX_COMPUTE_BYTES_PER_TOK = 8192  # per token at ub=512
    # Dense relative-bias fallback rate (quantized KV cache disables the banded
    # path): measured 402.5 GiB reserve at 1M ctx pre-fix, ~402 KiB per token.
    _INKLING_CTX_COMPUTE_DENSE_BYTES_PER_TOK = 402470  # per token at ub=512

    def _estimate_compute_buffer_bytes(
        self,
        *,
        n_ubatch: Optional[int] = None,
        n_parallel: int = 1,
        per_device_tensor: bool = False,
    ) -> int:
        """Per-device compute-graph buffer (bytes) from GGUF dims: a vocab-width
        output buffer + activation scratch. Context-independent; scales with
        ``--parallel`` (serving slots). Tensor mode materializes it on every device.
        A slight upper bound over measured allocations; 0 when dims are missing."""
        n_vocab = self._vocab_size or 0
        n_embd = self._embedding_length or 0
        if n_vocab <= 0 or n_embd <= 0:
            return 0
        ub = max(
            1,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )
        par = max(1, int(n_parallel))
        out_buffer = n_vocab * ub * 4  # f32 output/logits buffer
        act_scratch = 4 * n_embd * ub * 4  # a few resident hidden-width buffers
        if per_device_tensor:
            # Output + comm/staging materialized on every device, every slot.
            compute = 2 * act_scratch + out_buffer * par
        else:
            # Each extra concurrent slot adds one output buffer (chat decode sizes
            # ~one logit row per slot; would under-count embeddings/--logits-all,
            # not run here). Matches measured {1:36,2:492,4:1388,8:3220} MiB.
            compute = act_scratch + out_buffer * max(0, par - 1)
        return int(compute * self._COMPUTE_BUFFER_SAFETY)

    def _compute_buffer_ctx_bytes(
        self,
        n_ctx: int,
        n_ubatch: Optional[int] = None,
        cache_type_kv: Optional[str] = None,
        *,
        layer_split: bool = False,
    ) -> int:
        """Context-linear growth of the per-device compute buffer (bytes), charged
        on top of the flat ``_estimate_compute_buffer_bytes``. The flash-attn KQ
        mask + attention scratch scale ~linearly with context and with the micro-
        batch; the flat term only covers ctx -> 0. A quantized KV cache adds a
        context-sized dequant scratch that scales with n_embd; f16/bf16/f32 pays only
        the KQ mask, a flat n_ubatch*2 bytes per context token. ``cache_type_kv`` None
        -> f16 (llama.cpp's default; an env-set quantized cache is budgeted as f16 on
        the KV side, whose over-reservation absorbs the dequant scratch). Returns 0
        when dims are missing or ``n_ctx`` <= 0.

        ``layer_split`` adds the extra KQ-mask copies a multi-device split pays (see
        ``_CTX_COMPUTE_SPLIT_MULT``) on every architecture except deepseek4, whose own
        rate already subsumes them. Tensor mode passes False, having been measured
        separately at the single-device rate (see ``_plan_tensor_parallel``)."""
        n_embd = self._embedding_length or 0
        if n_embd <= 0 or n_ctx <= 0:
            return 0
        ub = max(
            1,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )
        # One KQ mask copy ([n_kv, n_ubatch] f16), n_embd-independent. Every rate below
        # is a single-GPU total containing exactly one, so a split adds the extra copies.
        mask_rate = ub * 2 * self._CTX_COMPUTE_F16_MASK_SAFETY
        split_extra = (self._CTX_COMPUTE_SPLIT_MULT - 1) * mask_rate if layer_split else 0.0
        if getattr(self, "_architecture", None) == "deepseek4":
            # DSV4 indexer/CSA buffer (see constants): flat + linear, ub-scaled. Fires
            # for any KV type -- the indexer scratch is present even with an f16 cache.
            # No split_extra, unlike every other arch: 72000 already clears the measured
            # 65.5 KiB/tok by 4928 B/tok, more than the 4608 the mask copies cost.
            ub_scale = ub / self._DEFAULT_N_UBATCH
            return int(
                self._DSV4_CTX_COMPUTE_FLAT_BYTES
                + self._DSV4_CTX_COMPUTE_BYTES_PER_TOK * n_ctx * ub_scale
            )
        if getattr(self, "_architecture", None) == "inkling":
            ub_scale = ub / self._DEFAULT_N_UBATCH
            # The fused banded path requires an f32/f16/bf16 KV cache. A quantized
            # cache forces the dense relative-bias fallback, whose compute buffer is
            # [n_kv, ub, n_head] f32 (~402 KiB per context token at ub=512); size the
            # fit for that so a q8_0 cache gets a small honest context instead of an
            # unloadable one that crash-loops the server.
            if cache_type_kv and _kv_bytes_per_elem(cache_type_kv) < 2.0:
                rate = self._INKLING_CTX_COMPUTE_DENSE_BYTES_PER_TOK
            else:
                # Banded flash path (see constants): linear, ub-scaled.
                rate = self._INKLING_CTX_COMPUTE_BYTES_PER_TOK
            # Both paths allocate the mask, and 8192 has only ~1.5x headroom over the
            # 5.6 KiB/tok banded measurement -- a split alone would overrun it.
            return int(rate * n_ctx * ub_scale + split_extra * n_ctx)
        if _kv_bytes_per_elem(cache_type_kv) < 2.0:
            # Quantized cache: the dequant scratch dominates and scales with n_embd.
            # MLA (compressed KV) needs far less of it: measured 0.94 x n_embd on
            # GLM-5.2 and Kimi-K2.7 vs up to 2.02x on regular attention.
            ub_scale = ub / self._DEFAULT_N_UBATCH
            rate = (
                self._CTX_COMPUTE_BYTES_PER_EMBD_MLA
                if self._key_length_mla
                else self._CTX_COMPUTE_BYTES_PER_EMBD
            )
            per_tok = rate * n_embd * ub_scale
        else:
            # f16/bf16/f32: only the KQ mask.
            per_tok = mask_rate
        # Only the mask goes 1x -> 4x, not the dequant scratch, so ADD the extra copies:
        # max() would under-reserve the quantized path by the whole dequant scratch.
        return int((per_tok + split_extra) * n_ctx)

    def _slots_that_fit_on_gpu(
        self,
        n_parallel: int,
        effective_ctx: int,
        gpus: list[tuple[int, int]],
        total_by_idx: Optional[dict[int, int]],
        base_footprint_bytes: int,
        cache_type_kv: Optional[str],
        pin_fraction: float,
        per_device_overhead_bytes: int,
        min_gpus: int,
        n_ubatch: Optional[int] = None,
        swa_full: bool = False,
        kv_unified: bool = True,
        flash_attn: bool = True,
        split_extra_bytes: int = 0,
    ) -> tuple[Optional[list[int]], bool, int]:
        """Largest serving-slot count in [1, n_parallel) whose fully-on-GPU footprint fits,
        so Unsloth keeps the model on GPU (-ngl -1) instead of --fit on, which offloads layers
        to host and collapses decode ~3x (oobabooga #6718). ``base_footprint_bytes`` is the
        slot-independent footprint (weights + soft overhead + MTP + context-linear compute,
        minus the folded compute buffer); each candidate re-adds the slot-sized compute buffer
        and KV, then re-selects GPUs like the explicit-context path -- ``split_extra_bytes``
        included, so a multi-GPU candidate is re-checked at the split rate. Returns
        (gpu_indices, use_fit=False, slots) for the largest fitting count, else (None, True,
        n_parallel). Only ever reduces; unit-testable with synthetic VRAM maps."""
        for slots in range(n_parallel - 1, 0, -1):
            cb = self._estimate_compute_buffer_bytes(
                n_ubatch = n_ubatch, n_parallel = slots, per_device_tensor = False
            )
            if cb <= 0:
                cb = self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
            total = (
                base_footprint_bytes
                + cb
                + self._estimate_kv_cache_bytes(
                    effective_ctx,
                    cache_type_kv,
                    n_parallel = slots,
                    swa_full = swa_full,
                    kv_unified = kv_unified,
                    n_ubatch = n_ubatch,
                    flash_attn = flash_attn,
                )
            )
            gpu_indices, use_fit = self._select_gpus_split_aware(
                total,
                gpus,
                usable_fraction = pin_fraction,
                total_by_idx = total_by_idx,
                per_device_overhead_bytes = per_device_overhead_bytes,
                min_gpus = min_gpus,
                split_extra_bytes = split_extra_bytes,
            )
            if not use_fit:
                return gpu_indices, False, slots
        return None, True, n_parallel

    def _fit_context_to_vram(
        self,
        requested_ctx: int,
        available_mib: int,
        model_size_bytes: int,
        cache_type_kv: Optional[str] = None,
        min_ctx: int = 4096,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        ctx_checkpoints: int = 0,
        flash_attn: bool = True,
        kv_on_gpu: bool = True,
        mtp_engaged: bool = False,
        mtp_overhead_fn: Optional[Callable[[int], int]] = None,
        compute_ctx_bytes_fn: Optional[Callable[[int], int]] = None,
        budget_frac: Optional[float] = None,
        total_mib: Optional[int] = None,
    ) -> int:
        """Return the largest context length that fits in GPU VRAM.

        Budget caps occupancy at ``_CTX_FIT_VRAM_FRACTION`` of the card: an
        absolute ``free - (1 - frac) * total`` when ``total_mib`` is given, else
        ``free * frac``. Weights alone over budget returns ``requested_ctx``.

        ``kv_on_gpu`` mirrors ``--kv-offload`` (default on); when False the KV
        cache lives in CPU RAM and the requested context is honored verbatim.
        Other keyword args mirror ``_estimate_kv_cache_bytes``.

        ``mtp_engaged`` reserves extra VRAM for the MTP draft model's KV cache +
        compute buffers, else tight tiers (e.g. 32 GB) spill to a slower path.
        """
        if not self._can_estimate_kv():
            logger.debug(
                "Skipping context fit because KV cache metadata is unavailable",
                requested_ctx = requested_ctx,
                available_mib = available_mib,
            )
            return requested_ctx

        # KV lives off-GPU: no VRAM accounting needed for the cache itself.
        if not kv_on_gpu:
            return requested_ctx

        kv_kwargs = dict(
            swa_full = swa_full,
            n_parallel = n_parallel,
            kv_unified = kv_unified,
            n_ubatch = n_ubatch,
            ctx_checkpoints = ctx_checkpoints,
            flash_attn = flash_attn,
        )

        # byte-accurate mtp_overhead_fn supersedes the flat fraction (the fallback
        # when dims can't size the draft KV); callers may override budget_frac.
        if budget_frac is None:
            flat_mtp = mtp_engaged and mtp_overhead_fn is None
            budget_frac = _CTX_FIT_VRAM_FRACTION - (_MTP_VRAM_RESERVE_FRAC if flat_mtp else 0.0)
        # Absolute reserve off total when known, else fraction-of-free; clamp >=0.
        if total_mib is not None and total_mib > 0:
            budget_mib = max(0.0, available_mib - (1.0 - budget_frac) * total_mib)
        else:
            budget_mib = available_mib * budget_frac
        budget_bytes = budget_mib * 1024 * 1024
        model_footprint = model_size_bytes

        def _mtp_at(ctx: int) -> int:
            return mtp_overhead_fn(ctx) if mtp_overhead_fn is not None else 0

        def _cc_at(ctx: int) -> int:
            # Context-linear compute-buffer growth (flash-attn KQ mask + scratch);
            # the flat term in model_footprint only covers ctx -> 0.
            return compute_ctx_bytes_fn(ctx) if compute_ctx_bytes_fn is not None else 0

        # Already fits?
        kv = self._estimate_kv_cache_bytes(requested_ctx, cache_type_kv, **kv_kwargs)
        if model_footprint + kv + _mtp_at(requested_ctx) + _cc_at(requested_ctx) <= budget_bytes:
            return requested_ctx

        # Weights + compute buffer alone exceed budget -- reducing ctx can't help.
        if model_footprint >= budget_bytes:
            logger.debug(
                "Model footprint exceeds GPU budget before KV cache",
                requested_ctx = requested_ctx,
                available_mib = available_mib,
                model_size_gb = round(model_footprint / (1024**3), 2),
            )
            return requested_ctx

        # Binary search for max context that fits (KV + MTP draft reserve at that ctx)
        remaining = budget_bytes - model_footprint
        effective_min = min(min_ctx, requested_ctx)
        lo, hi = effective_min, requested_ctx
        best = effective_min
        while lo <= hi:
            mid = (lo + hi) // 2
            kv = self._estimate_kv_cache_bytes(mid, cache_type_kv, **kv_kwargs)
            if kv + _mtp_at(mid) + _cc_at(mid) <= remaining:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Round down to nearest 256 for alignment, never above requested_ctx
        best = (best // 256) * 256
        best = max(effective_min, best)
        best = min(best, requested_ctx)
        return best

    # ── Variant fallback ────────────────────────────────────────────

    @staticmethod
    def _find_smallest_fitting_variant(
        hf_repo: str,
        free_bytes: int,
        hf_token: Optional[str] = None,
    ) -> Optional[tuple[str, int, list[str]]]:
        """Find the smallest GGUF variant (including all shards) that fits.

        Groups split shards by variant prefix and sums their sizes (e.g.
        UD-Q4_K_XL with 9 shards of 50 GB each = 450 GB total).

        Returns (first_shard_filename, total_size_bytes, extra_shards) or None.
        """
        try:
            from huggingface_hub import get_paths_info, list_repo_files

            files = list_repo_files(hf_repo, token = hf_token)
            gguf_files = [
                f
                for f in files
                if f.lower().endswith(".gguf")
                and not _is_companion_gguf_path(f)
                and not _is_big_endian_gguf_path(f)
            ]
            if not gguf_files:
                return None

            # Sizes for all GGUF files
            path_infos = list(get_paths_info(hf_repo, gguf_files, token = hf_token))
            size_map = {p.path: (p.size or 0) for p in path_infos}

            # Group by variant: shards share a prefix before -NNNNN-of-NNNNN
            variants: dict[str, list[str]] = {}
            for f in gguf_files:
                m = _SHARD_RE.match(f)
                key = m.group(1) if m else f
                variants.setdefault(key, []).append(f)

            # Sum shard sizes per variant, track the first shard (for download)
            variant_sizes: list[tuple[str, int, list[str]]] = []
            for key, shard_files in variants.items():
                total = sum(size_map.get(f, 0) for f in shard_files)
                first = sorted(shard_files)[0]
                variant_sizes.append((first, total, shard_files))

            # Smallest that fits
            variant_sizes.sort(key = lambda x: x[1])
            for first_file, total_size, shard_files in variant_sizes:
                if total_size > 0 and total_size <= free_bytes:
                    return (
                        first_file,
                        total_size,
                        [path for path in sorted(shard_files) if path != first_file],
                    )

            return None
        except Exception:
            return None

    # ── Port allocation ───────────────────────────────────────────

    @staticmethod
    def _find_free_port() -> int:
        """Find an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # ── Stdout drain (prevents pipe deadlock on Windows) ─────────

    def _drain_stdout(self):
        """Read subprocess stdout lines in a background thread.

        Prevents a pipe-buffer deadlock on Windows (~4 KB buffer): without
        draining, llama-server blocks on writes and never becomes healthy.
        Each line is also teed to ``self._llama_log_fh`` when set, so a
        post-mortem has the full output even if the crash predates the
        drain-thread join in ``_wait_for_health``.
        """
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)
                    logger.debug(f"[llama-server] {line}")
                    fh = getattr(self, "_llama_log_fh", None)
                    if fh is not None:
                        try:
                            fh.write(line + "\n")
                            fh.flush()
                        except (ValueError, OSError):
                            # Log file closed under us; tee silently.
                            pass
        except Exception:
            # Never let the drain thread die: a full stdout pipe can deadlock
            # llama-server (Windows). Pipe-closed on exit is the common case.
            logger.debug("llama-server stdout drain stopped", exc_info = True)

    # GGUF KV type sizes for fast skipping
    _GGUF_TYPE_SIZE = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    @staticmethod
    def _gguf_skip_value(f, vtype: int) -> None:
        """Skip a GGUF KV value without reading it."""
        sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(vtype)
        if sz is not None:
            f.seek(sz, 1)
        elif vtype == 8:  # STRING
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif vtype == 9:  # ARRAY
            atype = struct.unpack("<I", f.read(4))[0]
            alen = struct.unpack("<Q", f.read(8))[0]
            elem_sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(atype)
            if elem_sz is not None:
                f.seek(elem_sz * alen, 1)
            elif atype == 8:
                for _ in range(alen):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    f.seek(slen, 1)
            else:
                for _ in range(alen):
                    LlamaCppBackend._gguf_skip_value(f, atype)

    @staticmethod
    def _gguf_read_array_value(f, atype: int, alen: int) -> Optional[list]:
        if atype == 4:  # UINT32
            return [struct.unpack("<I", f.read(4))[0] for _ in range(alen)]
        if atype == 5:  # INT32
            return [struct.unpack("<i", f.read(4))[0] for _ in range(alen)]
        if atype == 7:  # BOOL
            return [struct.unpack("<?", f.read(1))[0] for _ in range(alen)]

        if atype == 8:  # STRING: kept only for the delimiter-shaped entries (#7066).
            from core.inference.chat_template_helpers import delimiter_shaped_tokens

            kept: list = []
            for _ in range(alen):
                n = struct.unpack("<Q", f.read(8))[0]
                raw = f.read(n)
                # A vocabulary holds arbitrary bytes; a marker is text, so undecodable
                # entries are simply not markers.
                try:
                    kept.append(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    continue
            return delimiter_shaped_tokens(kept)

        for _ in range(alen):
            LlamaCppBackend._gguf_skip_value(f, atype)
        return None

    @classmethod
    def _gguf_path_is_diffusion(cls, gguf_path: str, model_identifier: str) -> bool:
        """Classify a downloaded GGUF without mutating the active backend."""
        probe = object.__new__(cls)
        probe._model_identifier = model_identifier
        probe._read_gguf_metadata(gguf_path)
        return probe._is_diffusion

    def _reject_vulkan_diffusion_gpu_ids_before_teardown(
        self, gguf_path: str, model_identifier: str
    ) -> None:
        """Reject Vulkan + gpu_ids for diffusion GGUFs before Phase 1 teardown."""
        if self._gguf_path_is_diffusion(gguf_path, model_identifier):
            raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)

    def _read_gguf_metadata(self, gguf_path: str) -> None:
        """Read context_length, architecture params, and chat_template from a GGUF header.

        Parses only the KV pairs we need (~30ms even for multi-GB files).
        For split GGUFs, metadata is always in shard 1.
        """
        # Reset metadata so stale flags (e.g. _supports_reasoning) don't
        # carry over when switching models.
        self._context_length = None
        self._chat_template = None
        self._markup_tokens = []
        self._markup_profile = None
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_style = "enable_thinking"
        self._reasoning_effort_levels = []
        self._reasoning_default = True
        self._supports_preserve_thinking = False
        self._supports_tools = False
        self._n_layers = None
        self._n_experts = None
        self._leading_dense_block_count = None
        self._n_kv_heads = None
        self._n_kv_heads_by_layer = None
        self._n_heads = None
        self._embedding_length = None
        self._feed_forward_length = None
        self._vocab_size = None
        self._kv_key_length = None
        self._kv_value_length = None
        self._sliding_window = None
        self._sliding_window_pattern = None
        self._full_attention_interval = None
        self._kv_lora_rank = None
        self._key_length_mla = None
        self._kv_key_length_swa = None
        self._kv_value_length_swa = None
        self._ssm_inner_size = None
        self._ssm_state_size = None
        self._ssm_conv_kernel = None
        self._kda_head_dim = None
        self._shared_kv_layers = None
        self._nextn_predict_layers = None
        self._architecture = None
        self._is_diffusion = False

        try:
            canvas_seen = False
            WANTED = {
                "general.architecture",
                "tokenizer.chat_template",
                # Vocab size = tokens array length (no vocab_size key in many GGUFs).
                "tokenizer.ggml.tokens",
                # Block-diffusion marker (DiffusionGemma); routes to the diffusion runner.
                "diffusion.canvas_length",
                # Source-repo hints for the SWA resolver's HF fallback.
                "general.source.huggingface.repository",
                "general.source.url",
                "general.source.repo_url",
                "general.base_model.0.repo_url",
                "general.base_model.0.organization",
                "general.base_model.0.name",
                "general.basename",
                "general.organization",
                "general.size_label",
                "general.finetune",
            }
            # Arch-specific keys added dynamically once we know the arch.
            arch_keys: dict[str, str] = {}  # gguf_key -> attribute name
            arch = None
            sliding_window_pattern_period: Optional[int] = None
            general: dict[str, str] = {}

            with open(gguf_path, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                if magic != 0x46554747:  # b"GGUF" as little-endian u32
                    return
                _version = struct.unpack("<I", f.read(4))[0]
                _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

                for _ in range(kv_count):
                    # Tolerate truncated input (e.g. a partial header from an
                    # HTTP byte-range fetch): bail out so the resolver
                    # fallback runs on whatever we parsed.
                    try:
                        key_len_bytes = f.read(8)
                        if len(key_len_bytes) < 8:
                            break
                        key_len = struct.unpack("<Q", key_len_bytes)[0]
                        key_bytes = f.read(key_len)
                        if len(key_bytes) < key_len:
                            break
                        key = key_bytes.decode("utf-8")
                        vtype_bytes = f.read(4)
                        if len(vtype_bytes) < 4:
                            break
                        vtype = struct.unpack("<I", vtype_bytes)[0]
                    except (struct.error, UnicodeDecodeError):
                        break

                    try:
                        if key in WANTED or key in arch_keys:
                            if vtype == 8:  # STRING
                                slen = struct.unpack("<Q", f.read(8))[0]
                                val_s = f.read(slen).decode("utf-8")
                                if key.startswith("general.") and key != "general.architecture":
                                    general[key] = val_s
                                if key == "general.architecture":
                                    arch = val_s
                                    self._architecture = val_s
                                    arch_keys = {
                                        f"{arch}.context_length": "context_length",
                                        f"{arch}.block_count": "n_layers",
                                        f"{arch}.expert_count": "n_experts",
                                        f"{arch}.leading_dense_block_count": "leading_dense_block_count",
                                        f"{arch}.attention.head_count_kv": "n_kv_heads",
                                        f"{arch}.attention.head_count": "n_heads",
                                        f"{arch}.embedding_length": "embedding_length",
                                        f"{arch}.feed_forward_length": "feed_forward_length",
                                        f"{arch}.attention.key_length": "kv_key_length",
                                        f"{arch}.attention.value_length": "kv_value_length",
                                        f"{arch}.attention.sliding_window": "sliding_window",
                                        f"{arch}.attention.sliding_window_pattern": "sliding_window_pattern",
                                        f"{arch}.full_attention_interval": "full_attention_interval",
                                        f"{arch}.attention.kv_lora_rank": "kv_lora_rank",
                                        f"{arch}.attention.key_length_mla": "key_length_mla",
                                        f"{arch}.attention.key_length_swa": "kv_key_length_swa",
                                        f"{arch}.attention.value_length_swa": "kv_value_length_swa",
                                        f"{arch}.attention.shared_kv_layers": "shared_kv_layers",
                                        f"{arch}.ssm.inner_size": "ssm_inner_size",
                                        f"{arch}.ssm.state_size": "ssm_state_size",
                                        f"{arch}.ssm.conv_kernel": "ssm_conv_kernel",
                                        f"{arch}.kda.head_dim": "kda_head_dim",
                                        f"{arch}.nextn_predict_layers": "nextn_predict_layers",
                                    }
                                elif key == "tokenizer.chat_template":
                                    self._chat_template = val_s
                            elif vtype in (4, 10):  # UINT32 or UINT64
                                val_i = (
                                    struct.unpack("<I", f.read(4))[0]
                                    if vtype == 4
                                    else struct.unpack("<Q", f.read(8))[0]
                                )
                                if key == "diffusion.canvas_length":
                                    canvas_seen = True
                                attr = arch_keys.get(key)
                                if attr:
                                    if attr == "sliding_window_pattern":
                                        sliding_window_pattern_period = val_i
                                    else:
                                        setattr(self, f"_{attr}", val_i)
                            elif vtype == 9:  # ARRAY
                                atype = struct.unpack("<I", f.read(4))[0]
                                alen = struct.unpack("<Q", f.read(8))[0]
                                # Vocab size = token count; keep the length, not the strings.
                                if key == "tokenizer.ggml.tokens":
                                    self._vocab_size = int(alen)
                                val_a = self._gguf_read_array_value(f, atype, alen)
                                if key == "tokenizer.ggml.tokens" and val_a is not None:
                                    # Only the delimiter-shaped entries are kept, a few
                                    # hundred out of a six-figure vocab, so the sweep can
                                    # tell this model's own sentinels from another family's
                                    # without holding the whole vocabulary (#7066).
                                    from core.inference.chat_template_helpers import (
                                        delimiter_shaped_tokens,
                                    )
                                    self._markup_tokens = delimiter_shaped_tokens(val_a)
                                attr = arch_keys.get(key)
                                if attr == "n_kv_heads" and val_a is not None:
                                    self._n_kv_heads_by_layer = [int(x) for x in val_a]
                                    if self._n_kv_heads is None and self._n_kv_heads_by_layer:
                                        self._n_kv_heads = max(self._n_kv_heads_by_layer)
                                elif attr == "sliding_window_pattern" and val_a is not None:
                                    self._sliding_window_pattern = [bool(x) for x in val_a]
                                    sliding_window_pattern_period = None
                            else:
                                self._gguf_skip_value(f, vtype)
                        else:
                            self._gguf_skip_value(f, vtype)
                    except (struct.error, UnicodeDecodeError):
                        # Truncated input (e.g. HTTP byte-range header
                        # fetch); break so the resolver fallback runs on
                        # what we have.
                        break

            # Decide diffusion routing before the SWA resolver below: it can raise on an arch transformers
            # does not know, which would otherwise drop a DiffusionGemma model to plain llama-server.
            self._is_diffusion = bool(
                (arch and arch.lower().startswith("diffusion")) or canvas_seen
            )
            if self._is_diffusion:
                logger.info(
                    f"GGUF metadata: diffusion model detected (architecture={arch}); "
                    "will serve via the diffusion runner"
                )

            # Expand a scalar period straight from the GGUF first.
            if (
                self._sliding_window_pattern is None
                and sliding_window_pattern_period
                and self._n_layers
            ):
                self._sliding_window_pattern = [
                    (i + 1) % sliding_window_pattern_period != 0 for i in range(self._n_layers)
                ]

            # Otherwise hand off to the resolver (cache / bootstrap / transformers / HF). Diffusion models
            # skip it: they do not use Unsloth's SWA pattern and the resolver can raise for them.
            if (
                self._sliding_window_pattern is None
                and self._sliding_window
                and self._n_layers
                and not self._is_diffusion
            ):
                hf_repo_candidates = (
                    general.get("general.source.huggingface.repository"),
                    _hf_repo_from_url(general.get("general.source.url")),
                    _hf_repo_from_url(general.get("general.source.repo_url")),
                    _hf_repo_from_url(general.get("general.base_model.0.repo_url")),
                    (
                        f"{general['general.base_model.0.organization']}/"
                        f"{general['general.base_model.0.name']}".replace(" ", "-")
                        if general.get("general.base_model.0.organization")
                        and general.get("general.base_model.0.name")
                        else None
                    ),
                    (
                        f"{general['general.organization']}/{general['general.basename']}".replace(
                            " ", "-"
                        )
                        if general.get("general.organization") and general.get("general.basename")
                        else None
                    ),
                )
                self._sliding_window_pattern = _resolve_swa_pattern(
                    arch,
                    self._n_layers,
                    hf_repo_candidates,
                )

            if self._context_length:
                logger.info(f"GGUF metadata: context_length={self._context_length}")
            if self._chat_template:
                logger.info(f"GGUF metadata: chat_template={len(self._chat_template)} chars")
                # Detect thinking/reasoning support from chat template.
                flags = detect_reasoning_flags(
                    self._chat_template,
                    self._model_identifier,
                    log_source = "GGUF metadata",
                )
                self._supports_reasoning = flags["supports_reasoning"]
                self._reasoning_style = flags["reasoning_style"]
                self._reasoning_effort_levels = flags.get("reasoning_effort_levels", [])
                self._reasoning_always_on = flags["reasoning_always_on"]
                self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                self._supports_tools = flags["supports_tools"]
        except Exception as e:
            logger.warning(f"Failed to read GGUF metadata: {e}")

    # ── Diffusion runner (DiffusionGemma) ──

    def _find_diffusion_assets(self) -> Optional[tuple[list, str, Optional[str]]]:
        """Resolve how to launch the DiffusionGemma runner: (shim argv prefix,
        visual-server binary, optional extra PYTHONPATH dir for the file override).

        Shim: UNSLOTH_DG_SHIM (a .py file) first, else the installed
        unsloth_zoo.diffusion_studio.shim. Binary: DG_VISUAL_BIN first, else
        alongside llama-server. Returns None if neither can be found.
        """
        import importlib.util

        # Visual-server binary: env override, else next to llama-server or in the
        # install's build/bin (where the prebuilt/installer puts it). .exe on Windows.
        visual_bin = os.environ.get("DG_VISUAL_BIN")
        if not visual_bin:
            name = "llama-diffusion-gemma-visual-server" + (".exe" if os.name == "nt" else "")
            # include_denied: a transiently locked llama-server still pins the
            # install dir so the adjacent visual-server can be found
            base = self._find_llama_server_binary(include_denied = True)
            if base:
                base_dir = Path(base).parent
                for cand in (
                    base_dir / name,
                    base_dir / "build" / "bin" / name,
                    base_dir / "build" / "bin" / "Release" / name,
                ):
                    if cand.is_file():
                        visual_bin = str(cand)
                        break
        if not (visual_bin and Path(visual_bin).is_file()):
            return None

        # Shim: a file override (its dir goes on PYTHONPATH), else the zoo package via -m.
        shim_file = os.environ.get("UNSLOTH_DG_SHIM")
        if shim_file and Path(shim_file).is_file():
            return ([sys.executable, shim_file], visual_bin, str(Path(shim_file).parent))

        # Find the installed shim without importing the heavy unsloth_zoo package
        # (find_spec on the top-level package does not run its __init__).
        try:
            spec = importlib.util.find_spec("unsloth_zoo")
        except Exception:
            spec = None
        if spec is not None and spec.submodule_search_locations:
            pkg_dir = Path(list(spec.submodule_search_locations)[0])
            if (pkg_dir / "diffusion_studio" / "shim.py").is_file():
                return (
                    [sys.executable, "-m", "unsloth_zoo.diffusion_studio.shim"],
                    visual_bin,
                    None,
                )

        return None

    def diffusion_split_supported(self) -> bool:
        """Whether a diffusion launch right now would honour --ngl.

        Mirrors the launch gate for the training guard: a split the launcher will drop
        must be guarded as GPU-resident. False when no shim resolves or the probe fails,
        matching the launcher's own fallback."""
        try:
            assets = self._find_diffusion_assets()
        except Exception:
            return False
        if assets is None:
            return False
        shim_cmd, _visual_bin, _extra_pythonpath = assets
        return _shim_supports_ngl(shim_cmd)

    @staticmethod
    def _diffusion_gpu_arg(
        gpu_ids: Optional[List[int]],
        *,
        cpu_only: bool = False,
        force_cpu: bool = False,
    ) -> str:
        """Device token passed to the diffusion visual-server child.

        The visual engine replaces its child's CUDA visibility mask with this
        token, so an unpinned load must carry forward the first token from the
        parent's mask rather than turning a parent-relative ordinal into a new
        physical selection.
        """
        # force_cpu alone outranks the picker: zero GPU layers place nothing, and a real
        # device would let the child hold a context while _gpu_offload_active reports
        # False, making the training VRAM coordinator skip the unload. cpu_only stays
        # *below* the picker: an explicit pick wins on a host whose GPU torch cannot see.
        if force_cpu:
            return ""
        if gpu_ids:
            return str(sorted(gpu_ids)[0])
        if cpu_only:
            return ""
        if "DG_GPU" in os.environ:
            return os.environ["DG_GPU"]
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_mask:
            first = next((token.strip() for token in parent_mask.split(",") if token.strip()), "")
            if first and first != "-1":
                return first
        return "0"

    def _start_diffusion_server(
        self,
        *,
        model_path: str,
        gguf_path: Optional[str],
        hf_repo: Optional[str],
        hf_variant: Optional[str],
        model_identifier: str,
        n_ctx: int,
        extra_args: Optional[List[str]],
        gpu_ids: Optional[List[int]] = None,
        gpu_memory_mode: Literal["auto", "manual"] = "auto",
        gpu_layers: int = -1,
    ) -> bool:
        """Launch the OpenAI-compat diffusion shim (which drives the on-device
        visual decoder) and wait for health. Presents the same /v1 + /health
        interface as llama-server, so the rest of Unsloth is unchanged.

        Manual mode forwards gpu_layers as --ngl so a GGUF larger than VRAM can be
        split across GPU and RAM; auto leaves the runner's all-layers default."""
        assets = self._find_diffusion_assets()
        if assets is None:
            raise RuntimeError(
                "DiffusionGemma runner not found. Install unsloth_zoo (which ships "
                "unsloth_zoo.diffusion_studio.shim) or set UNSLOTH_DG_SHIM to a shim "
                "file, and provide the visual-server binary via DG_VISUAL_BIN or next "
                "to llama-server in the install tree."
            )
        shim_cmd, visual_bin, extra_pythonpath = assets
        self._diffusion_visual_bin = visual_bin

        self._kill_process()
        self._port = self._find_free_port()
        # Auto-size (0): the visual server probes the largest context that fits this GPU's VRAM
        # (capped at the training context). An explicit in-range n_ctx overrides it.
        maxtok = n_ctx if (n_ctx and 0 < n_ctx <= 65536) else 0
        # Manual mode: honour the picked split. Without it the runner pins every layer
        # to GPU and a GGUF larger than VRAM dies in cudaMalloc even at 0 layers
        # (#7574). An older shim has no --ngl, so drop it rather than die in argparse.
        manual_ngl = _diffusion_manual_ngl(gpu_memory_mode, gpu_layers)
        if manual_ngl is not None and not _shim_supports_ngl(shim_cmd):
            # Dropping a zero-layer split on virtualised Metal undoes the CPU pin:
            # cpu_only below is torch.cuda only, so it reads 0 on a Mac and the empty
            # --gpu token still lets the runner use Metal. Only the split keeps weights
            # off it, so refuse rather than serve possibly corrupt output.
            if manual_ngl == 0 and _metal_device_is_paravirtual():
                raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)
            logger.warning(
                "DiffusionGemma: ignoring the %d GPU-layer split; the installed "
                "unsloth_zoo shim has no --ngl. Upgrade unsloth_zoo to set the split.",
                manual_ngl,
            )
            manual_ngl = None
        # No visible CUDA GPU: a genuine CPU host, or a GPU host masked with
        # CUDA_VISIBLE_DEVICES="" to force CPU serving. Keep the visual-server child
        # CPU-masked (empty --gpu) so the shim does not re-expose GPU 0 via its default.
        cpu_only = self._effective_gpu_count() == 0
        # Stronger than cpu_only and deliberately not derived from it: _effective_gpu_count
        # is torch.cuda only, so it reads 0 on Metal, Vulkan, Windows-HIP or Intel XPU hosts
        # the visual server will happily use. Only an explicit zero-layer split keeps the
        # weights off every backend, and a real --gpu token there would let the child hold a
        # CUDA context while _gpu_offload_active reports False, which the coordinator trusts.
        holds_no_gpu = manual_ngl == 0
        # Honor the GPU picker first: the diffusion runner takes a single device,
        # so use the lowest selected GPU (matches the sorted set recorded below, so
        # the device used == the echoed gpu_ids[0]). With no pick, fall back to the
        # CPU-only mask, else DG_GPU / 0.
        gpu = self._diffusion_gpu_arg(gpu_ids, cpu_only = cpu_only, force_cpu = holds_no_gpu)

        cmd = list(shim_cmd) + [
            "--gguf",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--gpu",
            gpu,
            "--maxtok",
            str(maxtok),
        ]
        if manual_ngl is not None:
            cmd += ["--ngl", str(manual_ngl)]
            # Block diffusion is compute-bound, so CPU-side layers cost far more than on an
            # autoregressive model (~6 tok/s vs ~31 for non-diffusion Gemma). Make the
            # tradeoff visible instead of swapping an OOM for a slow model.
            logger.warning(
                "DiffusionGemma: %d GPU layers requested. Layers left on CPU are much "
                "slower for block diffusion than for a regular model; prefer a smaller "
                "quant over a large CPU split if throughput matters.",
                manual_ngl,
            )

        env = child_env_without_native_path_secret()
        # `python -m unsloth_zoo.diffusion_studio.shim` imports unsloth_zoo, which
        # refuses to load unless UNSLOTH_IS_PRESENT is set (normally by `import
        # unsloth`). The shim never imports unsloth, so set it here as unsloth does.
        env["UNSLOTH_IS_PRESENT"] = "1"
        # The shim's `import unsloth_zoo` aborts in get_device_type() ("needs a GPU")
        # when no accelerator is visible, even though it only drives the CPU
        # visual-server binary and does no torch GPU work. Allow the CPU device so the
        # runner starts; the visual server still runs on the CPU llama.cpp build.
        if cpu_only:
            env.setdefault("UNSLOTH_ALLOW_CPU", "1")
        env["DG_VISUAL_BIN"] = visual_bin
        env["DG_GPU"] = gpu
        if gpu_ids:
            # The visual server remasks via CUDA_VISIBLE_DEVICES=<gpu>; pin PCI
            # order (as the llama-server path does) so the picked physical id maps
            # to the GPU the picker showed, not CUDA's default fastest-first order.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The file-override shim imports its sibling visual_engine; put its dir on PYTHONPATH.
        # (The zoo-package shim is an installed module and needs no PYTHONPATH change.)
        if extra_pythonpath:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                (extra_pythonpath + os.pathsep + existing) if existing else extra_pythonpath
            )

        logger.info(f"Starting DiffusionGemma runner: {' '.join(cmd)}")
        self._stdout_lines = []
        self._llama_log_fh = None
        self._llama_log_path = None
        try:
            log_dir = _swa_cache_path().parent / "logs" / "diffusion-server"
            log_dir.mkdir(parents = True, exist_ok = True)
            self._llama_log_path = log_dir / f"diffusion-{int(time.time())}-port-{self._port}.log"
            self._llama_log_fh = open(self._llama_log_path, "w", encoding = "utf-8", buffering = 1)
            logger.info(f"diffusion runner stdout/stderr -> {self._llama_log_path}")
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Could not open diffusion runner log file: {e}")

        # The shim (and its visual server) die with this backend process, so a
        # Unsloth crash/restart never orphans a GPU process.
        self._process = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = utf8_child_env(env),
            **_windows_hidden_subprocess_kwargs(),
            **_child_popen_kwargs(),
        )
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout, daemon = True, name = "diffusion-stdout"
        )
        self._stdout_thread.start()

        # Publish state before the health wait (mirrors the llama-server path).
        self._gguf_path = model_path
        self._hf_repo = hf_repo
        self._is_vision = False
        self._is_audio = False  # clear any prior TTS/audio model's routing flag
        self._model_identifier = model_identifier
        self._cache_type_kv = None
        self._swa_full = False
        self._kv_cache_unified = False
        self._n_ubatch = self._DEFAULT_N_UBATCH
        self._flash_attn_enabled = True
        self._effective_cache_types = ("f16", "f16")
        self._kv_cache_context_total = None
        # False means "confirmed to hold no VRAM" and makes the training coordinator skip
        # the unload, so it is claimed only for a zero-layer split.
        self._gpu_offload_active = not holds_no_gpu
        # Diffusion uses only the layer split of the llama.cpp GPU-memory knobs: record what
        # was applied (so /load, /status and reload dedup agree with the child) and reset the
        # rest so a previous GGUF's manual settings are not reported.
        self._gpu_memory_mode = gpu_memory_mode if manual_ngl is not None else "auto"
        self._gpu_layers = manual_ngl if manual_ngl is not None else -1
        # What was ASKED for, not always what was applied (no --ngl downgrades it above).
        # /status reports reality via _gpu_layers, but the dedup guards must compare like
        # with like or the same request reloads on every /load.
        self._diffusion_requested_ngl = _diffusion_manual_ngl(gpu_memory_mode, gpu_layers)
        self._n_cpu_moe = 0
        self._tensor_split = None
        # Diffusion is never tensor-parallel; clear any state left by a prior TP
        # chat load (load_model phase 1 only kills the process, it doesn't run
        # the unload reset) so /status doesn't misreport TP and an identical
        # re-Apply doesn't reload against stale tensor-parallel state.
        self._tensor_parallel = False
        # The single-device runner records only the lowest selected GPU (chosen
        # above), not the whole pick, and clears any explicit pin from a prior
        # chat load; a multi-GPU list would misreport placement and mis-dedup.
        # A zero-layer split on virtualised Metal masks the child's devices outright, so
        # it pins nothing: record no effective pin, or /status names a device the runner
        # never used and clearing that pick reloads the same CPU runner.
        _pv_diffusion_cpu = manual_ngl == 0 and _metal_device_is_paravirtual()
        self._gpu_ids = [sorted(gpu_ids)[0]] if gpu_ids and not _pv_diffusion_cpu else None
        # The frontend prefers requested_gpu_ids when hydrating the picker.
        # Diffusion uses only one device, so echo the collapsed effective pin,
        # not unused members of the original request.
        self._requested_gpu_ids = [sorted(gpu_ids)[0]] if gpu_ids else None
        if hf_variant:
            self._hf_variant = hf_variant
        elif gguf_path:
            try:
                from utils.models.model_config import _extract_quant_label
                self._hf_variant = _extract_quant_label(gguf_path)
            except Exception:
                self._hf_variant = None
        else:
            self._hf_variant = None
        # Provisional until the server reports the budget it resolved (auto-size picks it from VRAM).
        self._effective_context_length = maxtok or self._context_length
        self._max_context_length = self._context_length or maxtok or None

        healthy = self._wait_for_health(timeout = 600.0)
        if healthy:
            self._healthy = True
            self._gpu_offload_active = not holds_no_gpu
            if extra_args is not None:
                self._extra_args = list(extra_args)
                # No spec block to drop here, so launched IS requested; rebind so a
                # prior GGUF load's record cannot outlive it.
                self._requested_extra_args = list(extra_args)
                self._extra_args_source = (model_identifier, hf_variant)
            # The visual server logs "MAXTOK=<N>" with the context budget it actually resolved
            # (auto-sized to VRAM). Read it back so the UI context bar shows the real budget.
            chosen = maxtok
            try:
                for _ln in reversed(self._stdout_lines):
                    _m = re.search(r"MAXTOK=(\d+)", _ln)
                    if _m:
                        chosen = int(_m.group(1))
                        break
            except Exception:
                pass
            if chosen and chosen > 0:
                self._effective_context_length = chosen
                self._max_context_length = chosen
            self._requested_n_ctx = int(n_ctx)
        else:
            self._healthy = False
            logger.error("DiffusionGemma runner failed to become healthy")
        return healthy

    # ── HF download (no lock held) ───────────────────────────────

    def _download_gguf(
        self,
        *,
        hf_repo: str,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        force: bool = False,
        allow_smaller_fallback: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Download GGUF file(s) from HuggingFace. Returns local path.

        Runs WITHOUT self._lock so unload_model() can set _cancel_event at
        any time; checks it between each shard download.

        ``force`` re-fetches even when a (possibly stale) blob is cached.
        ``allow_smaller_fallback=False`` raises on low disk instead of silently
        switching to a smaller quant. ``cancel_event`` overrides
        ``self._cancel_event`` so an update can use a private event without
        touching the shared one; defaults to the shared event.
        """
        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        from utils.hf_cache_settings import get_hf_cache_paths

        download_cache_dir = str(get_hf_cache_paths().hub_cache)
        try:
            import huggingface_hub  # noqa: F401 -- presence check only
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for HF model loading. "
                "Install it with: pip install huggingface_hub"
            )

        resolved_hf_repo = _resolve_repo_id_casing(hf_repo)
        if resolved_hf_repo != hf_repo:
            logger.info(
                "Using cached repo_id casing '%s' for requested '%s'",
                resolved_hf_repo,
                hf_repo,
            )
            hf_repo = resolved_hf_repo

        # Resolve the filename from the variant
        gguf_filename = None
        gguf_extra_shards: list[str] = []
        if hf_variant:
            try:
                from huggingface_hub import list_repo_files

                files = list_repo_files(hf_repo, token = hf_token)
                gguf_files = _gguf_files_for_variant(files, hf_variant)
                if gguf_files:
                    gguf_filename = gguf_files[0]
                    gguf_extra_shards = _gguf_extra_shards(gguf_files, gguf_filename)
            except Exception as e:
                logger.warning(f"Could not list repo files: {e}")

            # Fall back to the local cache when the repo listing is unavailable.
            if not gguf_filename:
                cached_name, cached_shards = _cached_variant_resolution(hf_repo, hf_variant)
                if cached_name:
                    gguf_filename = cached_name
                    gguf_extra_shards = cached_shards
                    logger.info(
                        "Resolved variant %s -> %s from local HF cache",
                        hf_variant,
                        gguf_filename,
                    )

            if not gguf_filename:
                repo_name = hf_repo.split("/")[-1].replace("-GGUF", "")
                gguf_filename = f"{repo_name}-{hf_variant}.gguf"

        # Prefer the existing model. Updates use force=True to fetch a new revision.
        if not force:
            if hf_variant:
                # Resolve by variant so a newer revision's filename does not hide
                # the complete older copy. Size-check against that older snapshot's
                # own revision when its metadata remains available.
                cached_main = cached_gguf_for_load(
                    hf_repo,
                    hf_variant,
                    verify_sizes = True,
                    hf_token = hf_token,
                )
            else:
                candidate = _cached_complete_candidate(hf_repo, gguf_filename, gguf_extra_shards)
                cached_main = (
                    candidate[0]
                    if candidate is not None
                    and _cached_candidate_matches_revision_size(hf_repo, candidate, hf_token)
                    else None
                )
            if cached_main is not None:
                logger.info(f"Reusing cached GGUF: {cached_main}")
                return cached_main

        # Check disk space; fall back to a smaller variant if needed
        all_gguf_files = [gguf_filename] + gguf_extra_shards
        try:
            from huggingface_hub import get_paths_info, try_to_load_from_cache

            path_infos = list(get_paths_info(hf_repo, all_gguf_files, token = hf_token))
            total_bytes = sum((p.size or 0) for p in path_infos)

            # Subtract bytes already in the HF cache so we only preflight
            # against what we must download. Without this, re-loading a
            # cached large model (e.g. MiniMax-M2.7-GGUF at 131 GB) fails
            # cold whenever free disk is below the full weight footprint,
            # even though nothing needs downloading.
            already_cached_bytes = 0
            # Count only files that can resume this download.
            offline = _hf_env_offline()
            # Offline split sets are reusable only when every shard shares a snapshot.
            split_needs_refetch = bool(offline and not force and gguf_extra_shards)
            if not force and not split_needs_refetch:
                for p in path_infos:
                    if not p.size:
                        continue
                    try:
                        cached_path = try_to_load_from_cache(
                            hf_repo,
                            p.path,
                            cache_dir = download_cache_dir,
                        )
                    except Exception:
                        cached_path = None
                    if (
                        not (isinstance(cached_path, str) and os.path.exists(cached_path))
                        and offline
                    ):
                        cached_path = _cached_hf_snapshot_file(
                            hf_repo,
                            p.path,
                            expected_size = p.size,
                            cache_dir = download_cache_dir,
                        )
                    if isinstance(cached_path, str) and os.path.exists(cached_path):
                        try:
                            on_disk = os.path.getsize(cached_path)
                        except OSError:
                            on_disk = 0
                        # Satisfied only when the full blob is present.
                        if on_disk >= p.size:
                            already_cached_bytes += p.size

            total_download_bytes = max(0, total_bytes - already_cached_bytes)

            if total_download_bytes > 0:
                Path(download_cache_dir).mkdir(parents = True, exist_ok = True)
                free_bytes = shutil.disk_usage(download_cache_dir).free

                total_gb = total_download_bytes / (1024**3)
                free_gb = free_bytes / (1024**3)
                cached_gb = already_cached_bytes / (1024**3)

                logger.info(
                    f"GGUF download: {total_gb:.1f} GB needed "
                    f"({cached_gb:.1f} GB already cached), "
                    f"{free_gb:.1f} GB free on disk"
                )

                if total_download_bytes > free_bytes:
                    if not allow_smaller_fallback:
                        # Update path: never silently switch to a smaller quant;
                        # surface the disk shortfall for the requested variant.
                        raise RuntimeError(
                            f"Not enough disk space to download {gguf_filename}. "
                            f"Only {free_gb:.1f} GB free in {download_cache_dir}"
                        )
                    smaller = self._find_smallest_fitting_variant(
                        hf_repo,
                        free_bytes,
                        hf_token,
                    )
                    if smaller:
                        fallback_file, fallback_size, fallback_shards = smaller
                        logger.info(
                            f"Selected variant too large ({total_gb:.1f} GB), "
                            f"falling back to {fallback_file} ({fallback_size / (1024**3):.1f} GB)"
                        )
                        gguf_filename = fallback_file
                        gguf_extra_shards = fallback_shards

                        # The selected fallback is a new load target. Apply the
                        # same any-revision reuse policy before starting a fetch.
                        fallback_candidate = _cached_complete_candidate(
                            hf_repo, gguf_filename, gguf_extra_shards
                        )
                        if fallback_candidate is not None and (
                            _cached_candidate_matches_revision_size(
                                hf_repo, fallback_candidate, hf_token
                            )
                        ):
                            logger.info(f"Reusing cached fallback GGUF: {fallback_candidate[0]}")
                            return fallback_candidate[0]
                    else:
                        raise RuntimeError(
                            f"Not enough disk space to download any variant. "
                            f"Only {free_gb:.1f} GB free in {download_cache_dir}"
                        )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        gguf_label = f"{hf_repo}/{gguf_filename}" + (
            f" (+{len(gguf_extra_shards)} shards)" if gguf_extra_shards else ""
        )
        logger.info(f"Resolving GGUF: {gguf_label}")
        try:
            if cancel_event.is_set():
                raise RuntimeError("Cancelled")
            dl_start = time.monotonic()
            # Xet primary, HTTP fallback on stall; per-file so finished shards stay cached.
            local_path = hf_hub_download_with_xet_fallback(
                hf_repo,
                gguf_filename,
                hf_token,
                cancel_event = cancel_event,
                on_status = lambda m: logger.info(m),
                force_download = force,
                cache_dir = download_cache_dir,
            )
            for shard in gguf_extra_shards:
                if cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                logger.info(f"Resolving GGUF shard: {shard}")
                hf_hub_download_with_xet_fallback(
                    hf_repo,
                    shard,
                    hf_token,
                    cancel_event = cancel_event,
                    force_download = force,
                    cache_dir = download_cache_dir,
                )
        except Exception as e:
            if isinstance(e, RuntimeError) and "Cancelled" in str(e):
                raise
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
            )

        dl_elapsed = time.monotonic() - dl_start
        if dl_elapsed < 2.0:
            logger.info(f"GGUF resolved from cache: {local_path}")
        else:
            logger.info(f"GGUF downloaded in {dl_elapsed:.1f}s: {local_path}")
        return local_path

    def _download_companion_gguf(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str],
        pick: Callable[[list[str]], Optional[str]],
        label: str,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
        outcome: Optional[dict] = None,
    ) -> Optional[str]:
        """Resolve and fetch a companion GGUF (mmproj / MTP drafter) by name.

        Prefers a companion co-located with ``near_path``'s cache snapshot,
        then tries the live repo file list, then the local HF cache snapshots
        (offline, same fallback as _download_gguf), then hf_hub_download.
        Runs WITHOUT self._lock (like _download_gguf); honors _cancel_event so
        an /unload between the main download and here skips the fetch.
        ``cancel_event`` overrides ``self._cancel_event`` (defaults to it).
        """
        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            return None

        # Keep companion files in the main GGUF's snapshot.
        if near_path:
            cached = _companion_snapshot_sibling(near_path, pick)
            if cached:
                logger.info("Reusing cached %s: %s", label, cached)
                return cached

        from utils.hf_cache_settings import get_hf_cache_paths

        companion_cache_dir = _hub_cache_dir_for_snapshot_path(near_path) or str(
            get_hf_cache_paths().hub_cache
        )

        if _hub_download_in_flight(hf_repo):
            logger.info("Skipping %s download while a hub download is active", label)
            return None

        target: Optional[str] = None
        from huggingface_hub import list_repo_files

        # Retry a transient listing blip; permanent repo/auth errors and offline
        # mode are not retried (offline raises at once -> fall through to cache).
        for attempt in range(3):
            if cancel_event.is_set():
                return None
            try:
                target = pick(list_repo_files(hf_repo, token = hf_token))
                break
            except Exception as e:
                if type(e).__name__ in (
                    "RepositoryNotFoundError",
                    "GatedRepoError",
                    "RevisionNotFoundError",
                    "EntryNotFoundError",
                    "OfflineModeIsEnabled",
                ):
                    logger.debug(f"Could not list repo files for {label}: {e}")
                    break
                logger.debug(
                    f"Could not list repo files for {label} (attempt {attempt + 1}/3): {e}"
                )
                if attempt < 2:
                    cancel_event.wait(2**attempt)

        if target is None:
            try:
                from utils.models.model_config import _iter_hf_cache_snapshots
                for snap in _iter_hf_cache_snapshots(hf_repo, companion_cache_dir):
                    rel_files = _gguf_snapshot_files(snap)
                    target = pick(rel_files)
                    if target is not None:
                        logger.info("Resolved %s %s from local HF cache", label, target)
                        break
            except Exception as e:
                logger.debug(f"Offline cache lookup for {label} failed: {e}")

        # "The repo publishes none" and "the fetch failed" both return None, but
        # only the second is worth retrying on the next Apply. Left unset on the
        # early returns above, where the answer is genuinely unknown.
        if outcome is not None and not cancel_event.is_set():
            outcome["listed"] = target is not None
        if target is None or cancel_event.is_set():
            return None

        # Offline, resolve the companion straight from the cache snapshot that
        # holds it. resolve_cached_repo_id_case can return a partial lower-case
        # spelling when any dir exists under the requested casing, so calling
        # hf_hub_download with hf_repo would miss the canonical file and silently
        # drop the companion. _cached_hf_snapshot_file scans every case variant.
        if _hf_env_offline():
            cached = _cached_hf_snapshot_file(
                hf_repo,
                target,
                cache_dir = companion_cache_dir,
            )
            if cached:
                logger.info("Resolved %s from local HF cache: %s", label, cached)
                return cached

        try:
            logger.info(f"Downloading {label}: {hf_repo}/{target}")
            # Same policy; companions are best-effort (caller below swallows failures to None).
            return hf_hub_download_with_xet_fallback(
                hf_repo,
                target,
                hf_token,
                cancel_event = cancel_event,
                cache_dir = companion_cache_dir,
            )
        except Exception as e:
            logger.warning(f"Could not download {label}: {e}")
            return None

    def _download_mmproj(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
    ) -> Optional[str]:
        """Download the mmproj (vision projection) file from a GGUF repo.

        Prefers mmproj-F16.gguf, else any mmproj*.gguf. Returns the local
        path, or None if none exists. ``cancel_event`` overrides
        ``self._cancel_event`` (defaults to it). ``near_path`` prefers a
        copy co-located with the main GGUF's cache snapshot.
        """

        return self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = _pick_mmproj,
            label = "mmproj",
            cancel_event = cancel_event,
            near_path = near_path,
        )

    def _cached_repo_mtp_drafter(
        self,
        hf_repo: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Optional[str]:
        """A drafter already in this repo's local HF cache, reused offline when a
        fresh copy can't be fetched. Prefers a repo-root ``mtp-*.gguf`` across all
        cached snapshots; else an existing ``MTP/`` copy (any precision -- the
        target verifies every drafted token). None if none is cached."""
        try:
            from utils.models.model_config import _iter_hf_cache_snapshots

            roots: list[Path] = []
            subdirs: list[Path] = []
            snapshots = (
                _iter_hf_cache_snapshots(hf_repo)
                if cache_dir is None
                else _iter_hf_cache_snapshots(hf_repo, cache_dir)
            )
            for snap in snapshots:  # newest first
                for f in sorted(_gguf_snapshot_files(snap)):
                    # MTP only: a DSpark drafter needs --spec-type draft-dspark,
                    # so it must never be launched as an MTP one.
                    if _is_mtp_only_drafter_path(f):
                        (roots if "/" not in f else subdirs).append(snap / f)
            # Keep snapshot order (newest first), root before any MTP/ copy, so a
            # newer main GGUF pairs with the newest cached drafter, not a stale one.
            for cand in roots + subdirs:
                if cand.is_file():
                    return str(cand)
        except Exception as e:
            logger.debug("Cached MTP drafter lookup failed for %s: %s", hf_repo, e)
        return None

    def _download_mtp(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        near_path: Optional[str] = None,
    ) -> Optional[str]:
        """Download the separate MTP drafter (speculative head) from a GGUF repo.

        Targets the repo-root ``mtp-*.gguf`` companion -- the Q8_0 drafter
        unsloth mirrors there for llama.cpp ``-hf`` auto-discovery (smallest,
        recommended for speculation). Repos that bake the MTP head into the
        main GGUF (e.g. Qwen) ship no such sibling and this returns None. The
        higher-precision copies under ``MTP/`` are for explicit selection and
        are intentionally skipped. Returns the local path, or None.
        """

        def _pick_mtp(candidates: list[str]) -> Optional[str]:
            # Root-level only: MTP/ subdir copies now share the mtp- prefix but
            # are explicit-selection, not auto-fetch (they'd sort ahead of root).
            mtp_files = sorted(
                f
                for f in candidates
                if f.lower().endswith(".gguf")
                and "/" not in f
                and Path(f).name.lower().startswith("mtp-")
            )
            return mtp_files[0] if mtp_files else None

        if near_path:
            cached = _companion_snapshot_sibling(near_path, _pick_mtp)
            if cached:
                logger.info("Reusing cached MTP drafter: %s", cached)
                return cached

        # Offline, reuse any drafter already on disk (a fresh copy can't be
        # fetched). Online, _download_companion_gguf/hf_hub_download reuse the
        # current cached file and refetch a changed one, so skip the probe here
        # rather than pair new weights with a stale draft.
        if _hf_env_offline():
            cached = self._cached_repo_mtp_drafter(
                hf_repo,
                cache_dir = _hub_cache_dir_for_snapshot_path(near_path),
            )
            if cached:
                logger.info(f"Reusing cached MTP drafter (offline): {cached}")
                return cached

        return self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = _pick_mtp,
            label = "MTP drafter",
            near_path = near_path,
        )

    def _cached_repo_dspark_drafter(
        self,
        hf_repo: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Optional[str]:
        """The preferred already-cached DSpark sidecar for a repo, Q8_0 first
        (dspark_preference_key), so an offline reuse picks the same file the
        online download would have fetched."""
        try:
            from utils.models.model_config import (
                _iter_hf_cache_snapshots,
                dspark_preference_key,
            )

            snapshots = (
                _iter_hf_cache_snapshots(hf_repo)
                if cache_dir is None
                else _iter_hf_cache_snapshots(hf_repo, cache_dir)
            )
            candidates: list[Path] = []
            for snap in snapshots:
                candidates.extend(
                    snap / name
                    for name in _gguf_snapshot_files(snap)
                    if _is_dspark_drafter_path(name)
                )
            for candidate in sorted(candidates, key = lambda p: dspark_preference_key(p.name)):
                if candidate.is_file():
                    return str(candidate)
        except Exception as exc:
            logger.debug("Cached DSpark drafter lookup failed for %s: %s", hf_repo, exc)
        return None

    def _download_dspark(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        near_path: Optional[str] = None,
        binary: Optional[str] = None,
    ) -> Optional[str]:
        """Download the published DSpark sidecar, preferring its Q8_0 copy.

        The ~11 GB fetch is gated on ``supports_dspark``: a binary that cannot run
        ``draft-dspark`` (every prebuilt in the known-broken window included) falls
        back, so the download would never be opened. A raised probe still fetches,
        since launch re-probes and may yet engage.
        """

        def _pick_dspark(candidates: list[str]) -> Optional[str]:
            from utils.models.model_config import dspark_preference_key
            files = sorted(
                (name for name in candidates if _is_dspark_drafter_path(name)),
                key = dspark_preference_key,
            )
            return files[0] if files else None

        cached = _companion_snapshot_sibling(near_path, _pick_dspark) if near_path else None
        if not cached and _hf_env_offline():
            cached = self._cached_repo_dspark_drafter(
                hf_repo,
                cache_dir = _hub_cache_dir_for_snapshot_path(near_path),
            )
        try:
            if not self.probe_server_capabilities(binary).get("supports_dspark"):
                logger.warning(
                    "Skipping the DSpark sidecar download: llama-server has no usable "
                    "--spec-type draft-dspark, so this load falls back to no speculative "
                    "decoding. Run `unsloth studio update`, then reload."
                )
                # A sidecar already on disk is still reported: the route rediscovers it
                # on every Apply, so answering None would make the reuse check compare
                # it against a launched None and reload the same server each time.
                # _build_speculative_flags re-checks the capability and still falls back.
                return cached
        except Exception as exc:
            logger.debug("DSpark capability probe failed before the sidecar fetch: %s", exc)

        if cached:
            logger.info("Reusing cached DSpark drafter: %s", cached)
            return cached
        outcome: dict = {}
        found = self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = _pick_dspark,
            label = "DSpark drafter",
            near_path = near_path,
            outcome = outcome,
        )
        # Distinguishes a repo that ships no sidecar (the permanent state for
        # every repo but one) from a fetch that failed and could yet succeed.
        self._dspark_sidecar_absent = outcome.get("listed") is False
        return found

    def _resolve_launch_mmproj_path(
        self, *, model_path: str, mmproj_path: Optional[str]
    ) -> Optional[str]:
        """Return mmproj_path iff it exists on disk AND matches the model family.

        None if mmproj_path is None, missing, or family-mismatched.
        """
        if not mmproj_path:
            return None

        mmproj = Path(mmproj_path)
        if not mmproj.is_file():
            logger.warning(f"mmproj file not found: {mmproj_path}")
            return None

        from utils.models.model_config import mmproj_matches_model_family

        if not mmproj_matches_model_family(model_path, str(mmproj)):
            logger.warning(
                f"mmproj does not match model family: model={Path(model_path).name} "
                f"mmproj={mmproj.name}"
            )
            return None

        return str(mmproj)

    def _mmproj_vram_bytes(self, launch_mmproj_path: Optional[str]) -> int:
        """Return resolved mmproj VRAM bytes, or 0 when absent/unreadable."""
        if not launch_mmproj_path:
            return 0
        try:
            return self._get_gguf_size_bytes(launch_mmproj_path)
        except OSError as e:
            logger.debug(f"Could not size mmproj {launch_mmproj_path}: {e}")
            return 0

    def _resolve_launch_mtp_path(
        self,
        *,
        mtp_draft_path: Optional[str],
        drafter_label: str = "MTP",
    ) -> Optional[str]:
        """Return mtp_draft_path iff it exists on disk, else None.

        No family check needed: the drafter is only ever auto-resolved from
        the same repo as the main GGUF (see _download_mtp).
        """
        if not mtp_draft_path:
            return None
        if not Path(mtp_draft_path).is_file():
            logger.warning(f"{drafter_label} drafter file not found: {mtp_draft_path}")
            return None
        return str(mtp_draft_path)

    # ── Lifecycle ─────────────────────────────────────────────────

    # GGUF ``general.architecture`` values for diffusion / image models.
    # llama.cpp has no such architectures, so loading one as a chat model dies
    # with "unknown model architecture: '<arch>'". These match the patched
    # stable-diffusion.cpp / ComfyUI-GGUF enums. Unsloth publishes FLUX and
    # Qwen-Image GGUFs under
    # https://huggingface.co/collections/unsloth/unsloth-diffusion-ggufs.
    # Matched exactly (not a substring) so a chat arch containing "wan"/"sd1"
    # (e.g. "taiwan") isn't misrouted to Images.
    _DIFFUSION_ARCHES = frozenset(
        (
            "qwen_image",
            "flux",
            "sd1",
            "sdxl",
            "sd3",
            "aura",
            "hidream",
            "cosmos",
            "ltxv",
            "hyvid",
            "wan",
            "lumina2",
        )
    )

    # Distro package names for the shared libraries the Linux prebuilt links,
    # so the loader failure below can say what to install.
    _SHARED_LIB_PACKAGES = {
        "libgomp.so.1": " (Debian/Ubuntu: libgomp1, Fedora/RHEL: libgomp)",
    }

    # Shared objects Unsloth ships itself, next to llama-server in build/bin
    # (see runtime_payload_health_groups in install_llama_prebuilt.py, and
    # _llama_server_env_for_binary which puts that dir on LD_LIBRARY_PATH). No
    # distro packages these, so a loader failure naming one means the managed
    # runtime is incomplete or mismatched, not that something must be installed.
    _BUNDLED_LIB_PREFIXES = ("libllama", "libggml", "libmtmd")

    @staticmethod
    def _is_bundled_llama_library(lib: str) -> bool:
        # The loader prints a bare soname when the file is absent but a full
        # path when it found and rejected it, so compare on the basename.
        return os.path.basename((lib or "").strip()).startswith(
            LlamaCppBackend._BUNDLED_LIB_PREFIXES
        )

    @staticmethod
    def _is_unsloth_managed_binary(binary: Optional[str]) -> bool:
        """Would `unsloth studio update` actually replace ``binary``?

        A pinned LLAMA_SERVER_PATH, or a llama-server found on PATH, is
        explicitly unmanaged (update_flow.managed_install_root returns None),
        so telling the user to update Unsloth's runtime cannot repair it.
        Unknown binary (callers that pass nothing) keeps the managed default.
        """
        if not binary:
            return True
        try:
            from utils.llama_cpp_update import _llama_install_root
            return _llama_install_root(binary) is not None
        except Exception:
            return True

    @staticmethod
    def _is_library_path(lib: str) -> bool:
        # glibc's own discriminator is `strchr (name, '/') == NULL`
        # (elf/dl-load.c): a slash anywhere means the name is a pathname and no
        # search happens, so a relative DT_NEEDED is as exact as an absolute
        # one. os.sep would be platform-bound and this string comes from
        # whatever host ran llama-server, so match both separators.
        lib = (lib or "").strip()
        return "/" in lib or "\\" in lib

    @staticmethod
    def _missing_library_message(lib: str, binary: Optional[str] = None) -> str:
        """The loader could not FIND ``lib`` ("cannot open shared object file")."""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib} is part of Unsloth's own "
                    "llama.cpp runtime and is missing from the install. Run "
                    "`unsloth studio update` to reinstall it, then load the model "
                    "again."
                )
            return (
                f"llama-server could not start: {lib} is part of the llama.cpp "
                "runtime that the llama-server binary in use was built with. That "
                "binary is a custom install Unsloth does not manage, so reinstall "
                "or rebuild that llama.cpp, then load the model again."
            )
        # A bare soname means the loader searched the standard directories,
        # which is what a package populates. Any name carrying a separator is a
        # pathname the loader opened directly (e.g. a vendor .so under /opt), so
        # one exact file is absent and no package will put it there.
        if LlamaCppBackend._is_library_path(lib):
            _remedy = (
                "run `unsloth studio update`"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                f"llama-server could not start: {lib} is missing from that exact "
                f"location. Restore it, or {_remedy}, then load the model again."
            )
        return (
            f"llama-server could not start: the system library "
            f"{lib or 'it needs'} is missing. Install it with your package "
            f"manager{LlamaCppBackend._SHARED_LIB_PACKAGES.get(lib, '')}, "
            "then load the model again."
        )

    @staticmethod
    def _unloadable_library_message(
        lib: str,
        reason: str,
        binary: Optional[str] = None,
    ) -> str:
        """The loader FOUND ``lib`` and refused it: the same "error while
        loading shared libraries" line also carries "file too short", "invalid
        ELF header", "wrong ELF class", "cannot open shared object file:
        Permission denied", ... Installing a package is the wrong advice there
        -- the file is already present."""
        detail = f" ({reason})" if reason else ""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib}, part of Unsloth's own "
                    f"llama.cpp runtime, could not be loaded{detail}. The install "
                    "is incomplete or mismatched. Run `unsloth studio update` to "
                    "reinstall it, then load the model again."
                )
            return (
                f"llama-server could not start: {lib}, part of the llama.cpp "
                "runtime that the llama-server binary in use was built with, "
                f"could not be loaded{detail}. That binary is a custom install "
                "Unsloth does not manage, so reinstall or rebuild that "
                "llama.cpp, then load the model again."
            )
        return (
            f"llama-server could not start: the library {lib or 'it needs'} "
            f"could not be loaded{detail}. Reinstall or update whatever "
            "provides it, then load the model again."
        )

    @staticmethod
    def _classify_llama_start_failure(
        output: str,
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        returncode: Optional[int] = None,
        binary: Optional[str] = None,
    ) -> str:
        """Explain *why* llama-server failed to start, from its output.

        Several distinct failures otherwise collapse into the same opaque
        "invalid GGUF or out of memory" message. Worst case: a diffusion GGUF
        loaded as a chat model -- valid file, plenty of memory, but llama.cpp
        has no such architecture, so the user is told to free memory that was
        never the problem (#5842). Pick the most specific message we can.
        """
        lowered = (output or "").lower()

        # The dynamic loader kills llama-server before main(), so nothing below
        # matches and the fallback blames the file or memory instead. The Linux
        # prebuilt links libgomp.so.1, which a stock container does not ship.
        # glibc prints "<prog>: error while loading shared libraries: <object>:
        # <diagnostic>[: <strerror>]" (elf/dl-catch.c fatal_error). <object> is
        # echoed verbatim, so an absolute dependency under a directory with
        # spaces keeps them ("/opt/My Runtime/libfoo.so: file too short"):
        # split on the ": " that introduces the diagnostic, not on whitespace.
        missing_lib = re.search(
            r"error while loading shared libraries:[ \t]*([^\r\n]+?)"
            r"(?::[ \t]+([^\r\n]*))?[ \t]*\r?$",
            output or "",
            re.MULTILINE,
        )
        if missing_lib:
            _lib = missing_lib.group(1).strip()
            _reason = (missing_lib.group(2) or "").strip()
            # glibc omits the object name entirely on its own allocation
            # failures ("...shared libraries: cannot create search path array"),
            # so the first word is prose, not a library. Report it unnamed.
            if not (
                "/" in _lib or "\\" in _lib or any(_e in _lib for _e in (".so", ".dylib", ".dll"))
            ):
                return LlamaCppBackend._unloadable_library_message(
                    "", f"{_lib} {_reason}".strip(), binary
                )
            # Only "cannot open shared object file" means absent; the same line
            # reports corrupt/incompatible libraries too, which are present.
            # glibc appends strerror(errno) to that diagnostic, and only ENOENT
            # ("No such file or directory") means the file is not there: EACCES
            # ("Permission denied", seen with an absolute DT_NEEDED path, an
            # unreadable parent directory, SELinux mislabels or container
            # sandboxing) means it exists and cannot be opened, where installing
            # a package is the wrong advice. A truncated tail keeps the
            # missing-library reading.
            _reason_l = _reason.lower()
            _errno_text = ""
            if "cannot open shared object file:" in _reason_l:
                _errno_text = _reason_l.split("cannot open shared object file:", 1)[1].strip()
            if not _reason_l or (
                "cannot open" in _reason_l and (not _errno_text or "no such file" in _errno_text)
            ):
                return LlamaCppBackend._missing_library_message(_lib, binary)
            return LlamaCppBackend._unloadable_library_message(_lib, _reason, binary)

        # Tensor parallelism (--split-mode tensor) is arch-gated in llama.cpp;
        # unsupported architectures abort the load with this marker. Point the
        # user at the toggle instead of a generic invalid-GGUF/OOM message.
        if "split_mode_tensor not implemented" in lowered:
            return (
                "Tensor parallelism is not supported for this model's "
                "architecture. Turn off Tensor Parallelism in the model "
                "settings and reload."
            )

        # Detect Ollama source up front so the arch branch can keep the
        # Ollama hint instead of the generic "unsupported arch" message.
        gguf = gguf_path or ""
        is_ollama = (
            ".studio_links" in gguf
            or os.sep + "ollama_links" + os.sep in gguf
            or os.sep + ".cache" + os.sep + "ollama" + os.sep in gguf
            or (model_identifier or "").startswith("ollama/")
        )

        # "unknown model architecture: '<arch>'": diffusion -> Images page,
        # Ollama -> Ollama hint, else a precise "unsupported" message. Exact
        # match so chat archs aren't misrouted.
        arch_match = re.search(r"unknown model architecture:\s*'([^']+)'", lowered)
        if arch_match:
            arch = arch_match.group(1)
            if arch in LlamaCppBackend._DIFFUSION_ARCHES:
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. Use "
                    "Unsloth's Images page to generate with local diffusion "
                    "GGUFs such as FLUX and Qwen-Image."
                )
            if is_ollama:
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )
            return (
                f"llama.cpp does not support this GGUF's model architecture "
                f"('{arch}'). The file is valid, but this model type cannot "
                "be run with llama-server."
            )

        # Other Ollama compat failures that don't name an arch. Only when
        # the output shows a GGUF compat issue, not OOM / missing binaries.
        if is_ollama:
            gguf_compat_hints = (
                "key not found",
                "unknown model architecture",
                "failed to load model",
            )
            if any(h in lowered for h in gguf_compat_hints):
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )

        # 127 without any loader line above is not specific to a library: it is
        # also what a shell wrapper entrypoint reports when its exec target is
        # gone (_llama_lib_dir supports those, as does a custom
        # LLAMA_SERVER_PATH), and what a "symbol lookup error" from a mismatched
        # runtime exits with. Name both causes instead of blaming a distro
        # package -- but still keep it off the GGUF and off memory.
        if returncode == 127:
            # Same provenance split as the library branches: the updater cannot
            # touch a pinned binary, so do not send its owner there.
            _remedy = (
                "run `unsloth studio update` to reinstall the llama.cpp runtime"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                "llama-server exited immediately (status 127): either the "
                "llama-server executable could not be found or run, or one of "
                f"the shared libraries it needs could not be loaded. Check the "
                f"llama-server log, and {_remedy}."
            )

        # SIGKILL with no diagnostic output is the OOM killer (e.g. a model too
        # large for the WSL VM's RAM cap); name it actionably.
        if returncode == -9:
            return (
                "llama-server was stopped by the operating system (signal 9), "
                "most likely out of memory. Try a smaller or more quantized "
                "GGUF, lower the context length, or free memory (on WSL, raise "
                "the memory limit in .wslconfig)."
            )
        # SIGTERM is also how an unload/cancel or a supervisor stops the server,
        # so report it neutrally rather than blaming memory.
        if returncode == -15:
            return (
                "llama-server was terminated (signal 15) before it became "
                "healthy. If you cancelled or unloaded the model this is "
                "expected; otherwise check the llama-server log for the cause."
            )

        # A live server that never answered 200 on /health is not a bad GGUF:
        # the load is too large for VRAM/context, or a local proxy/VPN grabbed
        # the loopback probe (#5740).
        if "health check timed out" in lowered:
            return (
                "llama-server started but never became healthy on its local "
                "/health endpoint. Try a smaller context length or a more "
                "quantized GGUF, and if you use a VPN or HTTP proxy make sure "
                "localhost bypasses it (NO_PROXY=127.0.0.1,localhost)."
            )

        # Fallback: genuinely unknown failure (OOM, missing binary ...).
        return (
            "llama-server failed to start. "
            "Check that the GGUF file is valid and you have enough memory."
        )

    def _plan_tensor_parallel(
        self,
        gpus: list[tuple[int, int]],
        model_size: int,
        target_ctx: int,
        cache_type_kv: Optional[str] = None,
        n_parallel: int = 1,
        mtp_engaged: bool = False,
        mtp_overhead_fn: Optional[Callable[[int], int]] = None,
        mtp_flat_reserve_bytes: int = 0,
        max_target_ctx: Optional[int] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        n_ubatch: Optional[int] = None,
        soft_overhead_bytes: int = 0,
        swa_full: bool = False,
        kv_unified: bool = True,
        flash_attn: bool = True,
    ) -> tuple[int, int, list[int], Optional[list[int]]]:
        """Plan a ``--split-mode tensor`` load. Pure: no model or GPU needed.

        ``gpus`` is a list of ``(gpu_index, free_mib)``; ``model_size`` is the
        weight size in bytes; ``target_ctx`` is the context to fit (the explicit
        request, or the model's native length for auto). ``max_target_ctx`` is
        the native/hardware ceiling used only for the UI bound (defaults to
        ``target_ctx``). Returns
        ``(effective_ctx, max_available_ctx, gpu_indices, tensor_split)``.

        Policy (assumes >= 2 GPUs; the caller drops the toggle below that):
        - Cap context to the KV that fits the pooled VRAM after the weights, one
          per-device flat compute-graph buffer (``_estimate_compute_buffer_bytes``,
          deterministic from dims; flat fallback when dims are unavailable), and the
          per-device context-linear compute growth (``_compute_buffer_ctx_bytes``,
          replicated on every device in tensor mode, so summed over the split).
          llama.cpp's ``--fit`` is a no-op in tensor mode, so this is the only
          cap, honored even for an explicit ``-c``. It is more accurate than the
          0.80 whole-pool heuristic, which over-reserves and leaves VRAM unused.
        - ``tensor_split`` is None (llama.cpp's even default, safe for every arch
          incl. Gemma 3n which GGML_ASSERTs on a weighted split) when an even
          share fits the smallest GPU; otherwise it is weighted by usable budget
          so the roomier GPU absorbs more weight and the smallest keeps room for KV.
        ``total_by_idx`` enables the total-based occupancy cap; ``n_ubatch`` sizes
        the compute buffer. ``soft_overhead_bytes`` is the CUDA-context / mmproj /
        MTP-draft-graph reserve the layer path folds into ``model_size_fit``;
        charged against the pooled budget so tensor mode reserves the same overhead.
        """

        # Per-GPU usable budget: free - (1-frac)*total, else (unknown total, e.g. a
        # two-column probe) the legacy free*frac. Mirrors _select_gpus and
        # _gpu_usable so the 5% cushion is kept on every path, not dropped here.
        def _usable(idx: int, free_mib: int) -> float:
            t = total_by_idx.get(idx, 0) if total_by_idx else 0
            if t > 0:
                return max(0.0, free_mib - (1.0 - _CTX_FIT_VRAM_FRACTION) * t)
            return max(0.0, free_mib * _CTX_FIT_VRAM_FRACTION)

        # Drop GPUs whose usable budget can't hold the per-device compute-graph
        # buffer; they'd OOM in tensor mode. Admitting on raw free would let a
        # partly-used big card in with no budget left. Defense-in-depth (load_model
        # gates too). Derived per-device reserve; flat fallback.
        _reserve_bytes = self._estimate_compute_buffer_bytes(
            n_ubatch = n_ubatch, n_parallel = n_parallel, per_device_tensor = True
        )
        reserve_mib = (
            _reserve_bytes // (1024 * 1024)
            if _reserve_bytes > 0
            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
        )
        usable_gpus = [g for g in gpus if _usable(g[0], g[1]) >= reserve_mib]
        gpu_indices = sorted(idx for idx, _ in usable_gpus)
        if len(gpu_indices) < 2:
            # Tensor parallelism is meaningless on <2 GPUs (the caller drops the
            # toggle before this); be defensive and never emit a split here.
            return (
                target_ctx if target_ctx > 0 else 4096,
                target_ctx if target_ctx > 0 else 4096,
                gpu_indices,
                None,
            )
        free_by_idx = {idx: free for idx, free in usable_gpus}
        usable_by_idx = {idx: _usable(idx, free_by_idx[idx]) for idx in gpu_indices}
        pool_mib = sum(usable_by_idx.values())
        # MTP reserve: byte-accurate per-ctx inside _fit_ctx (mtp_overhead_fn) plus
        # a flat cushion that the byte fn can't size -- 2 GiB when dims are wholly
        # unavailable (no fn), or mtp_flat_reserve_bytes when the fn is weights-only
        # because the draft KV couldn't be sized (_mtp_kv_unsized). Without this the
        # binary search spends the unsized-KV cushion on main context and OOMs.
        flat_mtp_bytes = max(0, mtp_flat_reserve_bytes)
        if mtp_engaged and mtp_overhead_fn is None:
            flat_mtp_bytes = max(flat_mtp_bytes, 2 * 1024**3)
        # soft_overhead_bytes is the CUDA-context / mmproj / MTP-draft-graph reserve
        # the layer path folds into model_size_fit. Tensor mode has no --fit valve, so
        # an unreserved overshoot OOMs at startup rather than offloading; charge it here
        # too. Once (pooled), mirroring the layer path -- the per-device CUDA context is
        # a known slight under-charge, left for real multi-GPU data.
        kv_budget_b = (
            (pool_mib - len(gpu_indices) * reserve_mib) * 1024 * 1024
            - model_size
            - flat_mtp_bytes
            - max(0, soft_overhead_bytes)
        )

        def _mtp_at(ctx: int) -> int:
            return mtp_overhead_fn(ctx) if mtp_overhead_fn is not None else 0

        def _kv_at(ctx: int) -> int:
            return self._estimate_kv_cache_bytes(
                ctx,
                cache_type_kv,
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )

        # Context-linear compute buffer, summed over the split. Tensor mode
        # replicates the compute graph on EVERY device (measured: the per-device
        # buffer grows a flat n_ubatch*2 bytes/token, ~1024 B/tok on Qwen3.5-9B at
        # f16, independent of n_embd), so the growth is n_dev x the per-device
        # term. cache_type_kv here is always non-quantized (tensor forces f16), so
        # _compute_buffer_ctx_bytes returns the light KQ-mask term, not the heavy
        # quantized dequant scratch. The flat reserve_mib above only covers ctx->0;
        # without this the fit over-pins and OOMs at high context on a tight pool
        # (0.5-4 GiB unreserved at 262k-1M across 2-4 GPUs), the tensor-mode analog
        # of the layer-split compute bug.
        n_dev = len(gpu_indices)

        def _cc_ctx(ctx: int) -> int:
            return n_dev * self._compute_buffer_ctx_bytes(ctx, n_ubatch, cache_type_kv)

        def _fit_ctx(ctx: int) -> int:
            # Largest context whose KV (+ MTP draft reserve + context-linear
            # compute) fits the pooled budget. Floors small, but never raises an
            # explicit ctx above asked.
            if self._can_estimate_kv() and ctx > 0:
                ctx_floor = min(2048, ctx)
                if kv_budget_b <= 0:
                    # Weights + buffers exceed the pool -> floor; the load then
                    # falls back to layer split.
                    return ctx_floor

                def _consumer(c: int) -> int:
                    return _kv_at(c) + _mtp_at(c) + _cc_ctx(c)

                if _consumer(ctx) <= kv_budget_b:
                    return ctx
                lo, hi, best = ctx_floor, ctx, ctx_floor
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if _consumer(mid) <= kv_budget_b:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best
            # KV size unknown -> can't prove a safe cap; floor.
            return min(4096, ctx) if ctx > 0 else 4096

        # max_available_ctx is the hardware ceiling for the UI bound, sized from
        # the native context independent of an explicit small -c (which only
        # caps effective_ctx).
        max_ctx_target = max_target_ctx if (max_target_ctx and max_target_ctx > 0) else target_ctx
        max_available_ctx = _fit_ctx(max_ctx_target)
        effective_ctx = min(_fit_ctx(target_ctx), max_available_ctx)

        min_usable_mib = min(usable_by_idx.values())
        kv_bytes = _kv_at(effective_ctx) if (self._can_estimate_kv() and effective_ctx > 0) else 0
        # The MTP reserve also has to fit the even split (mirror the pooled budget):
        # byte-accurate per-ctx (0 when no fn) plus the same flat cushion as above.
        mtp_bytes = (_mtp_at(effective_ctx) if effective_ctx > 0 else 0) + flat_mtp_bytes
        # Context-linear compute is replicated per device; charge the whole split so
        # the weighted ratio reflects it (mirrors kv_budget_b's per-device reserve).
        cc_bytes = _cc_ctx(effective_ctx) if effective_ctx > 0 else 0
        even_share_mib = (
            (model_size + kv_bytes + mtp_bytes + cc_bytes) / len(gpu_indices) / (1024 * 1024)
        )
        tensor_split: Optional[list[int]] = None
        if even_share_mib > (min_usable_mib - reserve_mib):
            # Each device also holds its replicated share of the context-linear
            # compute (cc_bytes/n_dev) on top of the flat reserve. The even-share
            # gate above charges cc_bytes; the split weights must subtract it too, or
            # the smaller card is weighted above its real usable budget and OOMs (the
            # per-device analog of the layer path's per-GPU overhead in _select_gpus).
            cc_per_dev_mib = (cc_bytes // len(gpu_indices)) // (1024 * 1024) if cc_bytes else 0
            adj = [
                max(0, int(usable_by_idx[i] - reserve_mib - cc_per_dev_mib)) for i in gpu_indices
            ]
            if sum(adj) > 0:
                tensor_split = adj
        return effective_ctx, max_available_ctx, gpu_indices, tensor_split

    @staticmethod
    def _is_projector_incompatibility(output: str) -> bool:
        """True when llama-server aborted because it cannot load the model's
        vision/audio projector (mmproj), typically an installed llama.cpp
        that predates the projector format. Conservative: only matches
        projector-format errors so unrelated failures (OOM, bad GGUF, port
        bind, ...) keep their own handling, and a bare 'clip'/'mmproj'
        mention in a normal startup log does not match.
        """
        text = (output or "").lower()
        if any(
            m in text
            for m in (
                "unknown projector type",
                "unsupported projector",
                "unsupported mmproj",
            )
        ):
            return True
        # Builds that phrase it via clip.cpp without the exact words above.
        return (
            "clip" in text
            and "projector" in text
            and ("unknown" in text or "unsupported" in text or "not supported" in text)
        )

    @staticmethod
    def _mmproj_retry_failure_message(*, projector_confirmed: bool, detail: str) -> str:
        """User-facing error when the text-only --mmproj strip retry also fails.

        Confirmed projector-format mismatches keep the historical wording.
        Bare signal crashes (common on some ROCm/driver paths) must not be
        reported as "Vision projector incompatible" — that misled #7302.
        """
        if projector_confirmed:
            return (
                "Vision projector incompatible with this llama.cpp "
                "build, and the text-only retry also failed: " + detail
            )
        return (
            "Vision model failed to start (llama-server crashed with "
            "--mmproj), and the text-only retry also failed: " + detail
        )

    @staticmethod
    def _output_has_nonprojector_diagnostic(output: str) -> bool:
        """True when the output already names a concrete non-projector cause (out
        of memory, an unsupported architecture, a tensor-parallel limit). A hard
        crash carrying such a marker must surface that error, not be silently
        retried text-only as if the vision projector were at fault; a bare crash
        with no marker still gets the text-only retry.
        """
        text = (output or "").lower()
        return any(
            m in text
            for m in (
                "out of memory",
                "failed to allocate",
                "unknown model architecture",
                "split_mode_tensor not implemented",
            )
        )

    @staticmethod
    def _is_tensor_split_assert(output: str) -> bool:
        """True only for the #6415 split-axis warmup assert (GGML_BACKEND_SPLIT_AXIS_*),
        not any ggml assert/abort, so an unrelated invariant isn't cached. stderr is
        merged into output."""
        text = (output or "").lower()
        if "ggml_assert" not in text and "ggml_abort" not in text:
            return False
        # the split-axis enum token, unique to this assert (not the source file).
        return "split_axis" in text

    @staticmethod
    def _is_signal_crash(returncode: Optional[int]) -> bool:
        """True only on a hard fault (SIGSEGV/SIGABRT/SIGILL/SIGFPE/SIGBUS or a
        Windows 0xC0000000+ status), not SIGKILL/SIGTERM/SIGINT (OOM killer /
        unload) nor a clean exit or still-running (None) process.
        """
        if returncode is None:
            return False
        if returncode >= 0xC0000000:  # Windows access violation / illegal instruction
            return True
        return -returncode in (4, 6, 7, 8, 11)  # SIGILL SIGABRT SIGBUS SIGFPE SIGSEGV

    @staticmethod
    def _is_abort_exit(returncode: Optional[int]) -> bool:
        """Windows CRT abort() exit code (3) from GGML_ASSERT on MSVC -- not a POSIX
        signal or 0xC0000000+ NTSTATUS."""
        return returncode == 3

    @classmethod
    def _should_record_tensor_split_abort(cls, returncode: Optional[int], output: str) -> bool:
        """The #6415 split-axis abort: the marker plus a hard crash (POSIX signal or
        Windows abort exit). Marker required so a generic crash isn't cached."""
        return cls._is_tensor_split_assert(output) and (
            cls._is_signal_crash(returncode) or cls._is_abort_exit(returncode)
        )

    @staticmethod
    def _with_flash_attn_off(cmd: list[str]) -> Optional[list[str]]:
        """Return cmd with flash attention forced off, or None when its effective
        (last-wins) value is already off/absent so there is nothing to retry. FA
        kernels hard-crash at startup on some ROCm builds; disabling FA keeps
        vision and MTP, the least destructive rung. A bare --flash-attn/-fa reads
        as on, so it counts toward the effective value and is neutralised too;
        every form is flipped in place (length preserved for downstream slices)."""
        out = list(cmd)

        def explicit(i):
            nxt = out[i + 1] if i + 1 < len(out) else None
            return nxt if nxt in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES else None

        effective = None
        for i, tok in enumerate(out):
            name = _flag_name(tok)
            if name in ("--flash-attn", "-fa") and "=" in tok:
                effective = tok.partition("=")[2]
            elif name in ("--flash-attn", "-fa"):
                effective = explicit(i) or "on"
        if effective not in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
            return None
        for i, tok in enumerate(out):
            name = _flag_name(tok)
            if name in ("--flash-attn", "-fa") and "=" in tok:
                flag, _, value = tok.partition("=")
                if value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
                    out[i] = f"{flag}=off"
            elif name in ("--flash-attn", "-fa"):
                if explicit(i) in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
                    out[i + 1] = "off"
                elif explicit(i) is None:  # bare flag (reads as on) -> explicit off
                    out[i] = f"{tok}=off"

        # A quantized V cache requires flash attention in llama.cpp: the init
        # aborts with "V cache quantization requires flash_attn". A quantized K
        # cache has no such requirement and runs fine without FA, so it is left
        # untouched -- resetting it would needlessly enlarge the K cache and can
        # OOM a memory-constrained config. Studio launches with FA on, so a
        # quantized --cache-type-v is legal at launch but would make THIS FA-off
        # retry crash on init instead of recovering. Reset a quantized V cache --
        # main and draft (the draft context shares the global --flash-attn flag,
        # so its V cache aborts too) -- to f16 (the llama.cpp default);
        # non-quantized types -- f16/bf16/f32 -- run fine without FA and are left
        # untouched. The value is rewritten in place so the list length is
        # preserved for downstream slices, matching the flash-attn flip above.
        _v_cache_flags = (
            "--cache-type-v",
            "-ctv",
            "--cache-type-v-draft",
            "--spec-draft-type-v",
            "-ctvd",
        )
        _cache_reset = False
        for i, tok in enumerate(out):
            # llama.cpp rewrites '_' to '-' for any argv token starting with
            # '--' before matching, so a legal pass-through spelling such as
            # --cache_type_v parses as --cache-type-v and still enables a
            # quantized V cache. Canonicalize the flag name the same way so the
            # reset recognizes the underscore aliases too; short flags (-ctv)
            # and the type value are left untouched.
            name = _flag_name(tok)
            if name not in _v_cache_flags:
                continue
            if "=" in tok:
                flag, _, value = tok.partition("=")
                if value.strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i] = f"{flag}=f16"
                    _cache_reset = True
            elif i + 1 < len(out):
                if out[i + 1].strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i + 1] = "f16"
                    _cache_reset = True
        if _cache_reset:
            logger.info(
                "V cache dtype reset to f16 because flash attention was disabled "
                "by the crash-recovery fallback (quantized V cache requires flash "
                "attention in llama.cpp; the K cache is left untouched)."
            )
        return out

    @staticmethod
    def _drop_env_quantized_v_cache(env: MutableMapping[str, str]) -> bool:
        """Drop an inherited quantized V-cache env var (main or draft) in place
        before a flash-attn-off retry, returning True if anything was removed.

        The argv rewrite in ``_with_flash_attn_off`` only reaches flags on the
        command line. Studio deliberately lets an env-only cache type reach the
        child untouched (an asymmetric K/V env must survive), so a quantized V
        cache set purely through ``LLAMA_ARG_CACHE_TYPE_V`` (or the draft
        ``LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V``) would still abort the FA-off retry
        with "V cache quantization requires flash_attn". Dropping it lets
        llama.cpp fall back to the f16 default. Only V is dropped: a quantized K
        cache runs fine without flash attention, so its env var is preserved.
        """
        dropped = False
        for var in ("LLAMA_ARG_CACHE_TYPE_V", "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"):
            value = (env.get(var) or "").strip().lower()
            if value and value not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                env.pop(var, None)
                dropped = True
        return dropped

    @staticmethod
    def _strip_mmproj_args(cmd: list[str]) -> list[str]:
        """Return cmd without the '--mmproj <path>' pair (text-only retry).
        Every other flag is preserved; a no-op when --mmproj is absent.
        """
        out: list[str] = []
        skip_value = False
        for tok in cmd:
            if skip_value:
                skip_value = False
                continue
            if tok == "--mmproj":
                skip_value = True
                continue
            out.append(tok)
        return out

    @staticmethod
    def _installed_gpu_backend(binary: Optional[str] = None) -> Optional[str]:
        """Return the installed GPU backend used for user-facing advice.

        Metal is not probed by the .so/.dll matcher, so macOS gets neutral advice.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return None
        lib_dir = _llama_lib_dir(binary)
        for backend in ("cuda", "hip", "vulkan"):
            if _lib_dir_has_ggml_backend(lib_dir, backend):
                return backend
        return None

    @staticmethod
    def _vulkan_prebuilt_was_auto_selected(binary: Optional[str]) -> bool:
        """Whether setup chose this managed Vulkan bundle without an override.

        Explicit selections set an override or marker. The managed marker also
        excludes custom and PATH binaries.
        """
        if not binary:
            return False
        if os.environ.get("UNSLOTH_FORCE_VULKAN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        if (os.environ.get("UNSLOTH_LLAMA_CPP_BACKEND") or "").strip().lower() in {
            "cpu",
            "vulkan",
            "hip",
            "rocm",
        }:
            return False
        try:
            from utils.llama_cpp_update import _llama_install_root

            install_root = _llama_install_root(binary)
            if install_root is None:
                return False
            marker = json.loads(
                (install_root / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8")
            )
        except (OSError, ValueError, TypeError):
            return False
        return (
            isinstance(marker, dict)
            and not bool(marker.get("force_cpu"))
            and marker.get("llama_backend") in (None, "", "auto")
        )

    @staticmethod
    def _strip_cpu_fallback_main_placement(args: Iterable[str]) -> list[str]:
        """Remove every raw main-model placement choice the CPU replay replaces."""
        stripped = strip_shadowing_flags(
            args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = True,
            strip_offload = True,
            strip_device = True,
        )
        return _paravirtual_strip_gpu_overrides(stripped, log_dropped = False) or []

    @classmethod
    def _auto_vulkan_cpu_fallback_eligible(
        cls,
        binary: Optional[str],
        intent: GgufLoadIntent,
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Whether a signal may replace automatic Vulkan placement with CPU.

        Explicit placement choices must remain failures instead of being discarded.
        """
        args = [str(arg) for arg in (extra_args or ())]
        companion_placement_flags = {
            *_OVERRIDE_TENSOR_FLAGS,
            "--spec-draft-device",
            "-devd",
            "--device-draft",
            "--spec-draft-ngl",
            "-ngld",
            "--gpu-layers-draft",
            "--n-gpu-layers-draft",
            "--mmproj-offload",
            "--no-mmproj-offload",
        }
        child_env = os.environ if env is None else env
        return bool(
            cls._is_vulkan_backend(binary)
            and cls._vulkan_prebuilt_was_auto_selected(binary)
            and intent.gpu_memory_mode == "auto"
            and not intent.gpu_ids
            and not intent.tensor_parallel
            and not intent.tensor_split
            and intent.n_cpu_moe == 0
            and cls._strip_cpu_fallback_main_placement(args) == args
            and not any(_flag_name(arg) in companion_placement_flags for arg in args)
            and not any(
                str(child_env.get(name) or "").strip()
                for name in (
                    *cls._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS,
                    "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
                    "LLAMA_ARG_MMPROJ_OFFLOAD",
                    "GGML_BACKEND_PATH",
                )
            )
        )

    @classmethod
    def _cpu_isolated_replay(
        cls,
        cmd: Iterable[str],
        env: dict[str, str],
        server_caps: Mapping[str, object],
    ) -> Optional[list[str]]:
        """Build a replay that cannot initialize or schedule on a GPU.

        Unlike ``-ngl 0``, ``--device none`` removes GPU devices. Refuse companions
        that this binary cannot pin to CPU.
        """
        original = [str(arg) for arg in cmd]
        has_mmproj = "--mmproj" in original
        if has_mmproj and not _paravirtual_mmproj_pinnable(server_caps):
            return None
        has_draft = _extra_args_mtp_draft_path(original, env = env) is not None
        draft_ngl_flag = _paravirtual_draft_ngl_flag(server_caps) if has_draft else None
        if has_draft and draft_ngl_flag is None:
            return None

        replay = cls._strip_cpu_fallback_main_placement(original)
        replay.extend(["--gpu-layers", "0", "--fit", "off", "--device", "none"])
        if has_mmproj:
            replay.append("--no-mmproj-offload")
        if has_draft:
            replay.extend([str(draft_ngl_flag), "0", "--device-draft", "none"])

        for name in (*cls._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS, "GGML_BACKEND_PATH"):
            env.pop(name, None)
        return replay

    def _cpu_isolated_binary(self, binary: Optional[str]) -> Optional[str]:
        """Stage managed llama.cpp without GPU backend libraries.

        Backend loading precedes ``--device none``, so isolation prevents the
        Vulkan plugin from loading first.
        """
        if not binary or not self._is_unsloth_managed_binary(binary):
            return None
        source_binary = _resolve_llama_binary(binary)
        if self._cpu_fallback_runtime_source == str(source_binary):
            staged = Path(self._cpu_fallback_runtime.name) / source_binary.name
            if staged.is_file():
                return str(staged)
        self._cleanup_cpu_fallback_runtime()
        staged_runtime = None
        try:
            runtime_root = _swa_cache_path().parent / "runtime"
            runtime_root.mkdir(parents = True, exist_ok = True)
            staged_runtime = tempfile.TemporaryDirectory(
                prefix = "llama-cpu-",
                dir = runtime_root,
            )
            staged_dir = Path(staged_runtime.name)
            lib_dir = _llama_lib_dir(str(source_binary))
            gpu_backend = re.compile(
                r"^(?:lib)?ggml-(?:cuda|hip|vulkan|metal|sycl|opencl|musa|cann|virtgpu)"
            )
            candidates = list(lib_dir.iterdir())
            if source_binary.parent != lib_dir:
                candidates.append(source_binary)
            for source in candidates:
                try:
                    resolved = source.resolve(strict = True)
                except OSError:
                    continue
                name = source.name
                is_shared = name.endswith((".dll", ".dylib")) or ".so" in name
                if resolved != source_binary and not is_shared:
                    continue
                if gpu_backend.match(name.lower()):
                    continue
                destination = staged_dir / name
                if destination.exists():
                    continue
                try:
                    os.link(resolved, destination)
                except OSError:
                    shutil.copy2(resolved, destination)
            staged_binary = staged_dir / source_binary.name
            cpu_library_present = any(
                entry.is_file()
                and entry.name.lower().startswith(("libggml-cpu", "ggml-cpu"))
                for entry in staged_dir.iterdir()
            )
            if not staged_binary.is_file() or not cpu_library_present:
                staged_runtime.cleanup()
                return None
        except (OSError, ValueError):
            if staged_runtime is not None:
                with contextlib.suppress(Exception):
                    staged_runtime.cleanup()
            return None
        self._cpu_fallback_runtime = staged_runtime
        self._cpu_fallback_runtime_source = str(source_binary)
        return str(staged_binary)

    def _cleanup_cpu_fallback_runtime(self) -> None:
        runtime = getattr(self, "_cpu_fallback_runtime", None)
        self._cpu_fallback_runtime = None
        self._cpu_fallback_runtime_source = None
        if runtime is not None:
            with contextlib.suppress(Exception):
                runtime.cleanup()

    def _cleanup_failed_cpu_fallback(self) -> None:
        """Remove every process and staged artifact from a terminal CPU replay."""
        self._kill_process()
        self._cpu_fallback_reason = None
        self._cleanup_cpu_fallback_runtime()

    @staticmethod
    def _gpu_init_crash_message(binary: Optional[str] = None) -> str:
        """Return backend-specific advice for a GPU initialization crash."""
        base = (
            "llama-server crashed at startup on both the vision and text-only "
            "attempts -- a GPU driver/runtime initialization crash, not a model "
            "or vision-projector problem. "
        )
        backend = LlamaCppBackend._installed_gpu_backend(binary)
        if backend == "vulkan":
            return base + (
                "This build uses Vulkan, which Unsloth selects when it detects an "
                "Intel or AMD GPU. Older integrated GPUs and their drivers can fault "
                "inside Vulkan device initialization. Update your Vulkan driver (Mesa "
                "on Linux), or reinstall llama.cpp on CPU with "
                "UNSLOTH_LLAMA_CPP_BACKEND=cpu unsloth studio update."
            )
        if backend == "hip":
            return base + (
                "This often means an unsupported secondary GPU; on AMD/ROCm, hide it "
                "with ROCR_VISIBLE_DEVICES (e.g. ROCR_VISIBLE_DEVICES=0 exposes only "
                "the first GPU) before launching Unsloth Studio."
            )
        if backend == "cuda":
            return base + (
                "This often means an unsupported secondary GPU or a driver/CUDA "
                "runtime mismatch; hide the extra GPU with CUDA_VISIBLE_DEVICES (e.g. "
                "CUDA_VISIBLE_DEVICES=0 exposes only the first GPU) before launching "
                "Unsloth Studio, or update the NVIDIA driver."
            )
        return base + (
            "Check the llama-server log and your GPU driver, and make sure the "
            "installed llama.cpp build matches this machine's GPU."
        )

    @staticmethod
    def _redacted_cmd_for_log(cmd: "list[str]") -> "list[str]":
        """Copy of cmd with the value after --api-key replaced by <redacted>."""
        out = list(cmd)
        if "--api-key" in out:
            ki = out.index("--api-key") + 1
            if ki < len(out):
                out[ki] = "<redacted>"
        return out

    def _start_llama_process(self, cmd: list[str], env: dict) -> None:
        """Spawn llama-server from cmd and start draining its output.

        Caller holds self._lock. Resets the stdout buffer, opens a fresh
        per-attempt tee log, launches the process, and starts the drain
        thread. Used for the initial start and the text-only mmproj retry.
        """
        # Defensive kill: if a concurrent load slipped past Phase 1
        # (because its `self._process` was None at the time) and already
        # stored a Popen handle here, drop that orphan before we overwrite
        # the reference. See issue #5161.
        self._kill_process()

        self._stdout_lines = []
        # Tee llama-server output to a dedicated log file so a post-mortem
        # in CI (or after a remote-debug session) has the full subprocess
        # trail even when the parent only stored the last 50 lines.
        self._llama_log_fh = None
        try:
            log_dir = _swa_cache_path().parent / "logs" / "llama-server"
            log_dir.mkdir(parents = True, exist_ok = True)
            self._llama_log_path = log_dir / f"llama-{int(time.time())}-port-{self._port}.log"
            self._llama_log_fh = open(
                self._llama_log_path,
                "w",
                encoding = "utf-8",
                buffering = 1,
            )
            logger.info(f"llama-server stdout/stderr -> {self._llama_log_path}")
        except (OSError, UnicodeDecodeError) as e:
            # Best-effort; never block the load on logging.
            logger.debug(f"Could not open llama-server log file: {e}")
            self._llama_log_path = None

        # Log the argv per attempt (the text-only mmproj retry re-enters here
        # with --mmproj stripped), redacting the API key.
        logger.info(f"Starting llama-server: {' '.join(self._redacted_cmd_for_log(cmd))}")

        self._process = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            **_windows_hidden_subprocess_kwargs(),
            **_child_popen_kwargs(),
        )
        # Cross-session backstop: record the PID so a later startup can reap this
        # server if parent-death cleanup did not run (macOS / best-effort failure).
        self._record_server_pid(self._process.pid)

        # Start background thread to drain stdout and prevent pipe deadlock
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout, daemon = True, name = "llama-stdout"
        )
        self._stdout_thread.start()

    @_with_gguf_load_marker
    def load_model(self, intent: GgufLoadIntent) -> bool:
        """Start llama-server from one immutable load intent."""
        gguf_path = intent.gguf_path
        mmproj_path = intent.mmproj_path
        mtp_draft_path = intent.mtp_draft_path
        dspark_draft_path = intent.dspark_draft_path
        hf_repo = intent.hf_repo
        hf_variant = intent.hf_variant
        hf_token = intent.hf_token
        model_identifier = intent.model_identifier
        is_vision = intent.is_vision
        n_ctx = intent.n_ctx
        chat_template_override = intent.chat_template_override
        cache_type_kv = intent.cache_type_kv
        speculative_type = intent.speculative_type
        spec_draft_n_max = intent.spec_draft_n_max
        tensor_parallel = intent.tensor_parallel
        gpu_memory_mode = intent.gpu_memory_mode
        gpu_layers = intent.gpu_layers
        n_cpu_moe = intent.n_cpu_moe
        tensor_split = list(intent.tensor_split) if intent.tensor_split is not None else None
        gpu_ids = list(intent.gpu_ids) if intent.gpu_ids is not None else None
        gpu_ids_are_vulkan_ordinals = intent.gpu_ids_are_vulkan_ordinals
        n_threads = intent.n_threads
        n_parallel = intent.n_parallel
        extra_args = list(intent.extra_args) if intent.extra_args is not None else None
        preserve_multi_gpu_on_layer = intent.preserve_multi_gpu_on_layer
        # Serialise the whole load so concurrent /load calls never leave two
        # llama-server processes alive (#5401 / #5161). Doesn't block /unload.
        with self._serial_load_lock:
            # In-app update swapping binaries: refuse fast (set under this lock,
            # so any in-flight load has drained) instead of using a half-swapped one.
            if getattr(self, "_llama_update_in_progress", False):
                raise RuntimeError("llama.cpp is updating; try again in a moment.")

            # A virtualised Apple GPU corrupts every offloaded layer, so pin GGUF
            # inference to CPU there; physical Apple Silicon is untouched. Settled
            # ABOVE the duplicate-load check (and the KV estimates) so the compared
            # placement is the one that launches: the launch records the normalized
            # values, so comparing the raw request would miss the fast path and tear
            # down a healthy CPU server on every repeat Auto /load.
            # Keyed on the hardware, not the request, and applied whole rather than
            # only when the request asks for offload: Manual + 0 layers already places
            # the main model on CPU, but the companions below (mmproj, a separate
            # drafter) do not read --gpu-layers, and an `-ot ".*=Metal"` the route never
            # strips would put weights straight back on the corrupt device.
            # paravirtual_normalized_request is the one definition of that rewrite,
            # shared with both duplicate-load comparators.
            _paravirtual_cpu_forced = _metal_device_is_paravirtual()
            if _paravirtual_cpu_forced:
                _pv_placement = paravirtual_normalized_request(
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    tensor_parallel = tensor_parallel,
                    tensor_split = tensor_split,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = extra_args,
                )
                if (gpu_memory_mode, gpu_layers) != (
                    _pv_placement.gpu_memory_mode,
                    _pv_placement.gpu_layers,
                ):
                    logger.warning(
                        "Forcing gpu_layers=0 for %s: this Mac's Metal device is "
                        "virtualised and offloaded layers produce corrupt output.",
                        model_identifier or gguf_path,
                    )
                if extra_args and _pv_placement.extra_args != list(extra_args):
                    logger.warning(
                        "Dropping offload flags from extra args on a "
                        "virtualised Metal device: %s -> %s",
                        list(extra_args),
                        _pv_placement.extra_args,
                    )
                if tensor_parallel or tensor_split:
                    logger.info(
                        "Virtualised Metal device: dropping tensor split/parallel "
                        "flags (nothing to split on the GPU)"
                    )
                gpu_memory_mode = _pv_placement.gpu_memory_mode
                gpu_layers = _pv_placement.gpu_layers
                n_cpu_moe = _pv_placement.n_cpu_moe
                extra_args = _pv_placement.extra_args
                # Spelled out (not read off _pv_placement, which returns exactly these)
                # so tests/test_tp_vision_regression.py's TP-drop allowlist sees this site.
                tensor_parallel = False
                tensor_split = None
                # The intent is what the duplicate-load check compares and what a retry
                # replays, so carry the rewrite into it rather than the raw request.
                intent = replace(
                    intent,
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = tuple(extra_args) if extra_args is not None else None,
                    tensor_parallel = False,
                    tensor_split = None,
                )

            # Duplicate /load that raced past the route check: do nothing if the
            # live server already satisfies this request.
            if self.adopt_load_intent_if_matched(intent):
                logger.info(
                    f"load_model: backend already in target state for "
                    f"'{model_identifier}', skipping reload"
                )
                # Retry probe only if a prior attempt didn't finish.
                if not self._audio_probed:
                    try:
                        detected = self._detect_audio_type_strict()
                        self._audio_probed = True
                    except Exception as exc:
                        logger.debug("Fast-path audio probe failed: %s", exc)
                        detected = None
                    if not self._apply_detected_audio(detected):
                        return False
                if not self._healthy:
                    return False
                return True

            self._cancel_event.clear()

            # Resolve llama-server now but defer a not-found error: a block-diffusion
            # GGUF uses the diffusion runner, and its arch is only known after the header.
            binary = self._find_llama_server_binary()
            is_vulkan_backend = self._is_vulkan_backend(binary)
            _vulkan_ordinal_pin = (
                is_vulkan_backend and bool(gpu_ids) and gpu_ids_are_vulkan_ordinals is not False
            )
            _cpu_only_pin = (
                bool(gpu_ids) and not is_vulkan_backend and self._backend_lacks_gpu_lib(binary)
            )

            # Without --kv-unified an explicit --parallel N splits -c into windows of -c/N, so on a
            # build lacking the flag the default of 4 would quarter every context window for a
            # feature it cannot serve: fall back to one slot. Ahead of the KV estimates so the
            # fit matches what launches.
            if (
                n_parallel > 1
                and binary
                and not self.probe_server_capabilities(binary).get("supports_kv_unified")
            ):
                logger.warning(
                    "llama-server at %s has no --kv-unified, so %d parallel slots would "
                    "split the context window %d ways. Using 1 slot instead; update "
                    "llama.cpp to run chats in parallel.",
                    binary,
                    n_parallel,
                    n_parallel,
                )
                n_parallel = 1

            # MTP is documented as single-slot: the model cards ship "-np 1" and say
            # "-np > 1 [...] not yet supported with MTP". The MTP state is per sequence,
            # so what breaks at more slots is the draft: unequal per-slot token counts
            # make split_equal reorder the ubatch while the nextn hidden states are
            # still read by raw token index, so slots read each other's rows, acceptance
            # collapses and MTP ends up slower than no speculation. Clamped beside the
            # --kv-unified clamp so the KV fit matches.
            # The env counts: llama.cpp appends spec types rather than replacing them
            # (--spec-type inserts, --spec-default push_backs, enablement is a find over
            # the vector), so an inherited LLAMA_ARG_SPEC_TYPE=draft-mtp really does
            # launch MTP and no later flag can clear it.
            _pv_extras_clamped_slots = 0
            if n_parallel > 1 and _extra_args_requests_mtp(
                extra_args, env = _child_spec_env(extra_args)
            ):
                # Kept so the paravirtual drafter drop can hand the slots back if it
                # strips the very spec group this clamped for.
                _pv_extras_clamped_slots = n_parallel
                logger.warning(
                    "MTP speculative decoding (--spec-type draft-mtp) does not support "
                    "%d parallel slots; using 1. Load without MTP to serve chats in "
                    "parallel.",
                    n_parallel,
                )
                n_parallel = 1

            # ── Vulkan-ordinal preflight (BEFORE the Phase 1 kill) ────────
            # An explicit Vulkan pin the ggml probe never enumerated cannot be honored.
            # Validate it ABOVE the kill so an invalid selection leaves the live model
            # untouched: CUDA ids are range-checked at the route, but Vulkan ordinals are
            # not, so a stale gpu_ids=[99] used to kill the server then 400, leaving
            # nothing running (#7239). _get_gpu_memory needs only the binary (safe pre-
            # download) and reuses the later fit's issubset logic. Guarded on a found
            # Vulkan build + a pin so a deferred not-found stays deferred for diffusion,
            # and on the ordinals being unconfirmed: a caller-confirmed physical CUDA pin
            # is not in this namespace and must not be range-checked against it.
            if _vulkan_ordinal_pin and binary:
                _pf_wanted = {int(x) for x in gpu_ids}
                _pf_probed = {g[0] for g in self._get_gpu_memory(binary)}
                if not _pf_wanted.issubset(_pf_probed):
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {sorted(_pf_wanted)} not "
                        f"present. Available Vulkan devices: {sorted(_pf_probed)}."
                    )

            # Classify before killing the healthy server (#7205); Phase 2 reuses this
            # path. Diffusion changes the meaning or validity of requested placement,
            # so both the remote and the local branch settle it above the teardown.
            # A zero-layer diffusion pin this shim cannot apply is refused; probe
            # once here so the HF case can be settled above the teardown as well.
            _pv_diffusion_unpinnable = bool(
                _paravirtual_cpu_forced and not self.diffusion_split_supported()
            )
            _preflight_model_path = None
            if hf_repo and (_vulkan_ordinal_pin or _cpu_only_pin or _pv_diffusion_unpinnable):
                _resolved_repo = _resolve_repo_id_casing(hf_repo)
                if _resolved_repo != hf_repo:
                    logger.info(
                        "Using cached repo_id casing '%s' for requested '%s'",
                        _resolved_repo,
                        hf_repo,
                    )
                    hf_repo = _resolved_repo
                with _hf_offline_if_unreachable():
                    _preflight_model_path = self._download_gguf(
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        hf_token = hf_token,
                    )
                _preflight_is_diffusion = self._gguf_path_is_diffusion(
                    _preflight_model_path, model_identifier
                )
                if _preflight_is_diffusion:
                    if _vulkan_ordinal_pin:
                        raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                    if _pv_diffusion_unpinnable:
                        raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)
                elif _cpu_only_pin:
                    raise ValueError(
                        f"Requested gpu_ids {list(gpu_ids)} but the llama.cpp build has "
                        "no GPU backend (CPU-only build); it would ignore the pin and run "
                        "on CPU. Omit gpu_ids to run on CPU."
                    )
            elif is_vulkan_backend and gpu_ids and gguf_path and not hf_repo:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

            # Only an ordinal pin is ambiguous for the diffusion runner; a
            # caller-confirmed physical pin keeps its CUDA placement path.
            if _vulkan_ordinal_pin and gguf_path and not hf_repo:
                self._reject_vulkan_diffusion_gpu_ids_before_teardown(
                    gguf_path,
                    model_identifier,
                )
            # Same for the unpinnable diffusion split: _start_diffusion_server enforces
            # it everywhere, but only after the teardown, so settle the local case here
            # and leave the running server alone.
            if (
                _pv_diffusion_unpinnable
                and gguf_path
                and not hf_repo
                and self._gguf_path_is_diffusion(gguf_path, model_identifier)
            ):
                raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)

            # ── Phase 1: kill old process (under lock, fast) ──────────
            _replaying_cpu_fallback = bool(
                self._cpu_fallback_reason and self._last_load_intent == intent
            )
            with self._lock:
                self._kill_process()
                if not _replaying_cpu_fallback:
                    self._cpu_fallback_reason = None
                    self._cleanup_cpu_fallback_runtime()

            # ── Phase 2: download (NO lock held, so cancel can proceed) ──
            # mtp_draft_path arrives set for local Gemma loads (detected
            # sibling); for -hf loads it's None here and resolved just below.
            # The canonical mode drives which drafter is downloaded, sized and
            # launched, so resolve it once before either branch can use it.
            _spec_canon = _canonicalize_spec_mode(speculative_type) or "auto"
            # Scope HF_HUB_OFFLINE to the download block only when DNS is
            # dead; cleanup runs even on exception so a transient hiccup
            # can't quarantine future loads.
            if hf_repo:
                # Resolve the requested repo id to its cached canonical casing once,
                # up front, so the main GGUF and its companions (mmproj / MTP drafter)
                # all resolve from the same cache entry. Otherwise a case-variant
                # request resolves the main file from the canonical cache dir while the
                # companions keep the requested casing and miss the cached files.
                _resolved_repo = _resolve_repo_id_casing(hf_repo)
                if _resolved_repo != hf_repo:
                    logger.info(
                        "Using cached repo_id casing '%s' for requested '%s'",
                        _resolved_repo,
                        hf_repo,
                    )
                    hf_repo = _resolved_repo
                with _hf_offline_if_unreachable():
                    model_path = _preflight_model_path or self._download_gguf(
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        hf_token = hf_token,
                    )
                    # Auto-download mmproj for vision models unless opted out.
                    if is_vision and not mmproj_path and not extra_args_disable_mmproj(extra_args):
                        mmproj_path = self._download_mmproj(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            near_path = model_path,
                        )
                    # Auto-download the separate MTP drafter (e.g. Gemma) when
                    # the requested spec mode can use it. Repos with the head
                    # baked into the main GGUF (Qwen) have no mtp- sibling and
                    # this no-ops, so the size gate stays out of it: a separate
                    # drafter speeds up even sub-3B (Gemma E2B), and the resolver
                    # below decides the final emission. Skipped only when the
                    # user disabled MTP or drives --spec-type manually.
                    if (
                        not mtp_draft_path
                        and _spec_canon in ("auto", "mtp", "mtp+ngram")
                        and not _extra_args_set_spec_type(extra_args)
                    ):
                        mtp_draft_path = self._download_mtp(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            near_path = model_path,
                        )
                    # "auto" is included: DSpark is the default whenever the repo
                    # ships a sidecar. Repos without one no-op, exactly like the
                    # MTP fetch above, so this costs one listing and no download.
                    if (
                        not dspark_draft_path
                        and _spec_canon in ("auto", "dspark")
                        and not _extra_args_set_spec_type(extra_args)
                    ):
                        dspark_draft_path = self._download_dspark(
                            hf_repo = hf_repo,
                            hf_token = hf_token,
                            near_path = model_path,
                            binary = binary,
                        )
            elif gguf_path:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
                model_path = gguf_path
            else:
                raise ValueError("Either gguf_path or hf_repo must be provided")

            # Set identifier early so _read_gguf_metadata can use it (DeepSeek).
            self._model_identifier = model_identifier

            # Auto resolves to DSpark whenever a sidecar is available and this
            # binary can run it: 1.84x decode on 4x B200 and 1.91x on one, against
            # the ngram-mod fallback this architecture would otherwise get. Gated
            # on the capability because _download_dspark also reports a cached
            # sidecar an incapable binary cannot launch, and promoting there would
            # turn Auto's fallback into no speculative decoding at all.
            if (
                _spec_canon == "auto"
                and dspark_draft_path
                and not _extra_args_set_spec_type(extra_args)
            ):
                try:
                    if self.probe_server_capabilities(binary).get("supports_dspark"):
                        _spec_canon = "dspark"
                        logger.info("Auto: DSpark sidecar available, using draft-dspark.")
                except Exception as exc:
                    logger.debug("DSpark capability probe failed during Auto: %s", exc)

            # MTP and DSpark are mutually exclusive, and the drafter sizing and
            # lifecycle path below is architecture agnostic, so it carries the
            # one drafter the mode selected. From here mtp_draft_path means
            # "the drafter this load will launch", whichever kind that is;
            # _spec_canon is the only thing that says which.
            if _spec_canon == "dspark":
                mtp_draft_path = dspark_draft_path

            # Read GGUF metadata (context_length, chat_template); header-only.
            self._read_gguf_metadata(model_path)

            if self._cancel_event.is_set():
                logger.info("Load cancelled after download phase")
                return False

            # Block-diffusion GGUFs (DiffusionGemma) cannot run on llama-server;
            # serve them with the diffusion runner (same OpenAI-compat interface).
            if self._is_diffusion:
                # Final defense: route and pre-teardown preflights reject before Phase 1.
                # The diffusion runner cannot translate Vulkan ordinals to CUDA IDs, so
                # this fires unless the caller confirmed the pin is physical CUDA IDs.
                if is_vulkan_backend and gpu_ids and gpu_ids_are_vulkan_ordinals is not False:
                    raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                # Not a tensor/layer GGUF: clear any preserved-fallback flag from a
                # prior load (this path skips the command builder that clears it).
                self._layer_preserves_tensor_intent = False
                # Same reason: a diffusion server never carries a drafter, only unload
                # clears these, and the dedupe compares the pair, so a stale one
                # reloads a healthy diffusion server on every Apply.
                self._mtp_draft_path = None
                self._mtp_draft_suppressed_path = None
                with self._lock:
                    if self._cancel_event.is_set():
                        logger.info("Load cancelled before diffusion server start")
                        return False
                    return self._start_diffusion_server(
                        model_path = model_path,
                        gguf_path = gguf_path,
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        model_identifier = model_identifier,
                        n_ctx = n_ctx,
                        extra_args = extra_args,
                        gpu_ids = gpu_ids,
                        gpu_memory_mode = gpu_memory_mode,
                        gpu_layers = gpu_layers,
                    )

            if not binary:
                # distinguish a transiently locked binary (antivirus / in-flight
                # install) from a missing one so the user retries, not reinstalls
                locked = self._find_llama_server_binary(include_denied = True)
                if locked:
                    raise RuntimeError(
                        f"llama-server at {locked} is temporarily unavailable "
                        "(access-denied; antivirus or an in-flight install). "
                        "Retry the load once it is released."
                    )
                # Reached only after the diffusion early-return above, so this is a
                # genuine llama-server-backed GGUF with no runtime. Raise the typed
                # error so /load returns the actionable 400 (not a generic 500), the
                # same message remote validation already shows.
                raise LlamaServerNotFoundError(LLAMA_SERVER_NOT_FOUND_DETAIL)

            server_caps = self.probe_server_capabilities(binary)

            # Outside ``self._lock`` so /unload, /cancel, /status aren't
            # blocked. ``unload_model`` also records the kill, so the
            # frontend /unload+/load Apply path engages the wait here even
            # without an in-process kill.
            self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)

            # ── Phase 3: start llama-server (under lock) ──────────────
            with self._lock:
                # Re-check cancel inside lock
                if self._cancel_event.is_set():
                    logger.info("Load cancelled before server start")
                    return False

                self._port = self._find_free_port()

                # Select GPU(s) from model size + estimated KV cache. Seed
                # safe defaults before probing so the except path has valid
                # state to publish.
                ctx_override = parse_ctx_override(extra_args)
                requested_ctx = resolve_requested_ctx(extra_args, n_ctx)
                swa_full = _swa_full_from_args_or_env(extra_args)
                _effective_ubatch = _extra_args_n_ubatch(
                    extra_args,
                    n_ctx = (requested_ctx if requested_ctx > 0 else self._context_length),
                )
                planned_kv_unified = _kv_unified_from_args(
                    extra_args,
                    default = n_parallel > 1 and server_caps.get("supports_kv_unified", False),
                )
                _pipeline_parallel_off = _pipeline_parallel_disabled_by_args(
                    extra_args, n_layers = self._n_layers
                )
                # A hard-crash recovery may relaunch this same plan with FA off.
                # Size that larger cache up front so the recovery cannot OOM.
                planned_flash_attn = False
                cache_override = parse_cache_override(extra_args)
                # Budget the heavier of asymmetric --cache-type-k/-v extras (they
                # win per axis at launch, appended last); resolve_cache_type_kv only
                # returns the last-wins type, which under-reserves the heavier axis.
                # The user's extras still set the real (possibly asymmetric) child
                # cache, so this only affects the reserve, not the emitted command.
                _extras_cache = _extra_args_main_cache_type_for_budget(extra_args)
                cache_type_kv = _extras_cache if _extras_cache is not None else cache_type_kv
                _cache_type_from_env = False
                if cache_type_kv is None:
                    # Param/extras set nothing, so the child inherits
                    # LLAMA_ARG_CACHE_TYPE_K/_V. Adopt a heavier env type (f32) for
                    # the reserve only; the launch does NOT re-emit it (that would
                    # rewrite an asymmetric K=f32,V=f16 env into symmetric flags),
                    # so _cache_type_from_env keeps it out of the emitted flags.
                    cache_type_kv = _env_main_cache_type_for_budget()
                    _cache_type_from_env = cache_type_kv is not None
                # A user --split-mode in extras last-wins-overrides the toggle, and
                # an inherited tensor LLAMA_ARG_SPLIT_MODE flips it on (the child
                # would run tensor unbudgeted otherwise). The duplicate-load matchers
                # use the same helper so a healthy env-driven tensor server matches.
                split_mode_override = parse_split_mode_override(extra_args)
                tensor_parallel = _effective_tensor_parallel(extra_args, tensor_parallel)
                # gpu_layers=0 leaves nothing to split, yet --split-mode tensor or
                # a per-GPU ratio still launches tensor mode -- and under the
                # CPU-only mask below (no visible devices) that aborts the server
                # instead of loading on CPU. Drop both here (nothing to split).
                if gpu_memory_mode == "manual" and gpu_layers == 0:
                    if tensor_parallel or tensor_split:
                        logger.info(
                            "Manual gpu_layers=0: dropping tensor split/parallel "
                            "flags (nothing to split on the GPU)"
                        )
                    tensor_parallel = False
                    tensor_split = None
                # Record the requested strategy for /status and the load
                # response. 'manual' has no fallback, so the request value is the
                # value actually applied.
                self._gpu_memory_mode = gpu_memory_mode
                # The layer/MoE/split knobs apply only with an explicit offload
                # (manual + gpu_layers >= 0); else record defaults so /status and
                # /load don't report knobs the server never applied.
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    self._gpu_layers = gpu_layers
                    self._n_cpu_moe = n_cpu_moe
                    self._tensor_split = tensor_split
                else:
                    self._gpu_layers = -1
                    self._n_cpu_moe = 0
                    self._tensor_split = None
                self._requested_gpu_ids = sorted(gpu_ids) if gpu_ids else None
                self._gpu_ids = list(self._requested_gpu_ids) if self._requested_gpu_ids else None
                # Manual offload skips the TP planner but still emits --split-mode
                # tensor at launch; drop it when fewer than 2 GPUs are in use --
                # tensor split is a no-op there and aborts on some architectures.
                # Done before the cache-drop below so a quantized KV survives.
                if (
                    tensor_parallel
                    and gpu_memory_mode == "manual"
                    and gpu_layers >= 0
                    and self._effective_gpu_count(sorted(gpu_ids) if gpu_ids else None) < 2
                ):
                    logger.info(
                        "Tensor parallelism requested in manual mode but fewer "
                        "than 2 GPUs are in use; ignoring (needs >= 2)."
                    )
                    tensor_parallel = False
                # Drop TP for manual + Auto layers before the cache-drop below (like
                # the <2-GPU guard above), so a requested quantized KV survives into
                # the --fit load rather than being stripped for a tensor attempt.
                if tensor_parallel and gpu_memory_mode == "manual" and gpu_layers < 0:
                    logger.info(
                        "Manual mode with Auto layers hands memory management to "
                        "llama.cpp --fit, which is incompatible with tensor "
                        "parallelism; ignoring the tensor split."
                    )
                    tensor_parallel = False
                # Tensor mode aborts on a quantized KV cache, so drop it for the
                # tensor attempt (and strip any inherited/explicit --cache-type
                # that would re-impose it when appended last). Layer split does
                # support it, so remember the dropped type and the original extras
                # to restore (verbatim, incl. an asymmetric K/V) if we later fall
                # back to layer split below.
                _tensor_dropped_cache_type_kv: Optional[str] = None
                _tensor_dropped_extra_args: Optional[list] = None
                # Tensor mode rejects any quantized axis. cache_type_kv is the
                # heavier-by-bytes budget type, which can mask a quantized axis (an
                # f16 budget hides a paired q4_0), so also test each explicit
                # --cache-type-k/-v extra, not just the budget type.
                _ck_extra, _cv_extra = parse_cache_override_per_axis(extra_args)
                _cache_non_tensor_safe = any(
                    c and c.strip().lower() not in self._TENSOR_PARALLEL_KV_TYPES
                    for c in (cache_type_kv, _ck_extra, _cv_extra)
                )
                if tensor_parallel and _cache_non_tensor_safe:
                    logger.info(
                        "Tensor parallelism requires a non-quantized KV cache; "
                        "ignoring cache type %s for the tensor attempt.",
                        cache_type_kv,
                    )
                    _tensor_dropped_cache_type_kv = cache_type_kv
                    cache_type_kv = None
                    if extra_args:
                        # Keep the originals so a layer downgrade restores the real
                        # (possibly asymmetric) --cache-type-k/-v the layer path
                        # supports, not just the scalar heavier type.
                        _tensor_dropped_extra_args = list(extra_args)
                        extra_args = strip_shadowing_flags(
                            extra_args,
                            strip_context = False,
                            strip_cache = True,
                            strip_spec = False,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                    # The launch keeps an inherited tensor-safe env cache type (the
                    # env cleanup only pops quantized ones), so re-adopt a heavier
                    # env type (f32) for the budget here too -- mirrors the initial
                    # adoption, which was skipped because the param/extras set the
                    # (now-dropped) quantized type. Else the child allocates f32 KV
                    # against an f16 budget.
                    _env_tensor_cache = _env_main_cache_type_for_budget()
                    if _env_tensor_cache is not None:
                        cache_type_kv = _env_tensor_cache
                        _cache_type_from_env = True
                if ctx_override is not None and ctx_override > 0:
                    logger.info(f"User --ctx-size {ctx_override} honored; skipping auto-reduce")
                if cache_override is not None:
                    _ck, _cv = parse_cache_override_per_axis(extra_args)
                    logger.info(
                        f"User --cache-type-k/-v (k={_ck}, v={_cv}) honored; "
                        "KV estimate budgets the heavier axis"
                    )
                if split_mode_override is not None:
                    logger.info(
                        f"User --split-mode {split_mode_override} honored; "
                        "reconciled into tensor_parallel state"
                    )
                effective_ctx = requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                max_available_ctx = self._context_length or effective_ctx
                gpus: list[tuple[int, int]] = []
                # Keep fit-budget and launch-flag mmproj resolution in sync.
                launch_mmproj_path = None
                if not extra_args_disable_mmproj(extra_args):
                    launch_mmproj_path = self._resolve_launch_mmproj_path(
                        model_path = model_path,
                        mmproj_path = mmproj_path,
                    )
                # A virtualised Metal device corrupts the vision encoder too, and
                # --no-mmproj-offload is the only way to keep it off there (clip.cpp
                # never reads --gpu-layers). Without that flag the choice is a projector
                # on the corrupt device or no projector, so drop it: silently encoding
                # every image from garbage is the failure this fallback exists to stop.
                # Only when the probe answered: --no-mmproj-offload is from b5178 and
                # the base argv already needs b6325, so a build that can start here
                # always has it, and a false capability means the --help probe failed
                # (see _paravirtual_probe_answered). Dropping vision on that would be a
                # self-inflicted outage; the pin below covers the unanswered case.
                _pv_mmproj_unpinnable = bool(
                    launch_mmproj_path
                    and _paravirtual_cpu_forced
                    and _paravirtual_probe_answered(server_caps)
                    and not server_caps.get("supports_no_mmproj_offload")
                )
                if _pv_mmproj_unpinnable:
                    logger.warning(
                        "Disabling image/audio input for this session: this Mac's "
                        "Metal device is virtualised and this llama-server build has "
                        "no --no-mmproj-offload, so the vision projector would keep "
                        "running on the device whose output is corrupt. Run 'unsloth "
                        "studio update' to install a build that supports it."
                    )
                    launch_mmproj_path = None
                # Need both a resolved mmproj AND the config vision flag; a stray
                # mmproj passing the family-name heuristic must not flip a non-VLM
                # GGUF into vision mode.
                effective_is_vision = bool(launch_mmproj_path) and bool(is_vision)
                if is_vision and not effective_is_vision and not _pv_mmproj_unpinnable:
                    logger.warning(
                        "Vision-capable GGUF loaded without a usable mmproj; "
                        "image input will be disabled for this session"
                    )
                # Seed before the try: the except (GPU-selection failure ->
                # --fit on) falls through to the launch which reads this, and the
                # probe that assigns it may throw first. Captured before manual
                # empty `gpus` so the speculative defaults stay GPU-aware and the
                # CPU-fallback check still knows GPUs were present.
                _detected_gpus: list[tuple[int, int]] = []
                total_by_idx: dict[int, int] = {}
                model_size = None  # set in the fit try; used by the APU RAM guard
                # Layer-fallback min GPUs; raised below on a tensor downgrade. Bound
                # before the try so the --fit-on except path still has it (no UnboundLocal).
                _layer_min_gpus = 1
                # An explicit Vulkan ordinal absent from the ggml probe cannot be
                # honored; flag it in the fit and reject after the try (raising inside
                # would be swallowed into the --fit-on fallback). Bound before the try.
                _vulkan_explicit_unmatched = False
                _vulkan_requested_ids: list[int] = []
                _vulkan_available_ordinals: list[int] = []
                try:
                    gguf_size = self._get_gguf_size_bytes(model_path)
                    # Include GPU-loaded mmproj in the fit budget (#5825).
                    mmproj_size = (
                        self._mmproj_vram_bytes(launch_mmproj_path) if effective_is_vision else 0
                    )
                    model_size = gguf_size + mmproj_size
                    # 2-tuple gpus for existing logic + a total map for the absolute
                    # per-GPU headroom (correct when the GPU is already partly used).
                    # Pass binary so a Vulkan build probes ggml's Vulkan ordinals.
                    _gpu_mem = self._get_gpu_memory(binary)
                    if is_vulkan_backend and not gpu_ids and gpu_memory_mode != "manual":
                        _gpu_mem = self._vulkan_auto_gpu_memory(_gpu_mem)
                    gpus = [(idx, free) for idx, free, _t in _gpu_mem]
                    # Restrict the fit (and thus the layer plan + pin env) to the
                    # selected GPUs; fail-open if none match so a stale UI choice
                    # can't strand the load on CPU (issue #7164).
                    if gpu_ids:
                        # A Vulkan build indexes by ggml ordinal. An explicit ordinal
                        # absent from the probe can't be pinned, so reject after the try
                        # rather than fail-open onto a device the user didn't pick.
                        _wanted_ids = {int(x) for x in gpu_ids}
                        # Reject if ANY requested ordinal is absent, not only when none
                        # match: [0, 99] against {0, 1} silently drops 99. Comparing the
                        # full requested set (before filter narrows) still lets the fitter
                        # pick a valid subset later -- that is narrowing, not absence.
                        _probed_ordinals = {g[0] for g in gpus}
                        if is_vulkan_backend and not _wanted_ids.issubset(_probed_ordinals):
                            _vulkan_explicit_unmatched = True
                            _vulkan_requested_ids = sorted(_wanted_ids)
                            _vulkan_available_ordinals = sorted(_probed_ordinals)
                        # Restrict the probed pool to the selection; fail-open (keep the
                        # full pool) if none match so a stale UI choice can't strand the
                        # load on CPU (issue #7164).
                        _sel_gpus = [g for g in gpus if g[0] in _wanted_ids]
                        gpus = _sel_gpus if _sel_gpus else gpus
                    total_by_idx = {idx: total for idx, _f, total in _gpu_mem}
                    # GPU picker: restrict every mode to the chosen devices, so
                    # auto selection only considers them and manual mask to
                    # them (the env block below pins CUDA/HIP_VISIBLE_DEVICES).
                    if gpu_ids:
                        _picked = set(gpu_ids)
                        gpus = [g for g in gpus if g[0] in _picked]

                    # GPUs the model will run on -- captured before manual
                    # empty `gpus` to bypass the planner. bool() drives the
                    # GPU-aware speculative defaults; the list feeds the
                    # CPU-fallback check.
                    _detected_gpus = list(gpus)

                    def _gpu_usable(g, frac = _CTX_FIT_VRAM_FRACTION):
                        # Per-GPU usable budget for ranking: free - (1-frac)*total.
                        # Callers pass the ACTIVE fraction so the ranking matches the
                        # budget the fit then tests (else mixed totals mis-order).
                        idx, free = g
                        t = total_by_idx.get(idx, 0)
                        if t > 0:
                            return free - (1.0 - frac) * t
                        return free * frac

                    def _pool_budget_mib(subset, frac):
                        # Sum each GPU's own usable budget. Pooling free and total
                        # separately would let an unknown-total GPU (MIG/vGPU/N/A)
                        # add full free with no cushion among known-total GPUs.
                        return sum(max(0.0, _gpu_usable(g, frac)) for g in subset)

                    # Resolve effective context: 0 means let llama-server use
                    # the model's native length. Only expand to a known native
                    # length if metadata exists; else keep 0 as a sentinel.
                    if requested_ctx > 0:
                        effective_ctx = requested_ctx
                    elif self._context_length is not None:
                        effective_ctx = self._context_length
                    else:
                        effective_ctx = 0
                    original_ctx = effective_ctx
                    # Default UI ceiling to the native context length;
                    # GPU/VRAM-fit logic below may shrink it on limited HW.
                    max_available_ctx = self._context_length or effective_ctx

                    # Manual + Auto layers (the Manual default): hand memory
                    # management to llama.cpp's --fit. Emptying the probed GPU set
                    # no-ops the selection/TP planning below, leaving gpu_indices
                    # None (an explicit gpu_ids pick still pins below) and use_fit
                    # True. An explicit context is honored (--fit optimizes around
                    # it); 0 lets --fit size it.
                    if gpu_memory_mode == "manual" and gpu_layers < 0:
                        # Tensor parallelism was already dropped above (before the
                        # cache-drop), so a quantized KV survives into this --fit load.
                        gpus = []
                        effective_ctx = requested_ctx if requested_ctx > 0 else 0
                        original_ctx = effective_ctx
                        # --fit aborts under --split-mode tensor; a raw extras
                        # --split-mode/--tensor-split (appended last) would
                        # otherwise reach llama-server. Strip it like the TP
                        # downgrade does.
                        extra_args = strip_split_mode_only(extra_args)
                    elif gpu_memory_mode == "manual":
                        # Manual offload (--gpu-layers + --fit off): no automatic
                        # device masking (a gpu_ids pick still pins below) or
                        # context cap -- the user owns both. tensor_parallel is
                        # honored but skips the memory-based planner (gpus = []);
                        # the toggle just emits --split-mode tensor (split by free
                        # VRAM, or by the Split ratio if set).
                        gpus = []
                        effective_ctx = (
                            requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                        )
                        original_ctx = effective_ctx
                        # Strip the user --split-mode when the toggle owns the split
                        # (TP engaged -> Studio emits --split-mode tensor) or when the
                        # user asked for tensor (which aborts on a single GPU even if
                        # the manual <2-GPU guard downgraded TP). Otherwise keep their
                        # non-tensor mode (row/none/layer) -- the toggle can't express
                        # those.
                        if tensor_parallel or split_mode_override == "tensor":
                            extra_args = strip_split_mode_only(extra_args)

                    # Will MTP engage? If so, auto-fit reserves draft-model VRAM.
                    # Mirrors _build_speculative_flags: forced mtp/mtp+ngram always
                    # engage; auto only on an MTP model >= 3B; ngram/off never. A
                    # separate drafter (Gemma) counts as an MTP model.
                    # _spec_canon, not the raw request: Auto has already resolved to
                    # dspark above when a sidecar is available, and the reserve below
                    # differs by kind (no duplicated target-KV copy, depth 3).
                    _mtp_effective = _spec_canon
                    _mtp_size_for_fit = _extract_model_size_b(model_identifier)
                    # Sub-3B drops MTP only for an embedded head; a separate
                    # drafter (Gemma) engages and needs its VRAM reserved.
                    _mtp_sub_3b_for_fit = (
                        _mtp_size_for_fit is not None
                        and _mtp_size_for_fit < _MTP_MIN_SIZE_B
                        and not bool(mtp_draft_path)
                    )
                    # The inherited spec env reaches the child only when the extras own
                    # --spec-type; the launch scrubs it otherwise. Reserve against the
                    # env in that case only, where flags and env accumulate and launch.
                    _spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                    # Extras can run MTP even when Unsloth suppresses its own emission.
                    _user_mtp_via_extras = _extra_args_requests_mtp(extra_args, env = _spec_env)
                    # A non-MTP model-based draft mode (draft-simple/draft-eagle3) in
                    # extras also loads a separate draft model that needs reserving;
                    # engage only when extras actually name a drafter for it.
                    _user_draft_via_extras = _extra_args_requests_separate_draft(
                        extra_args, env = _spec_env
                    ) and bool(_extra_args_mtp_draft_path(extra_args))
                    # Mirror _build_speculative_flags: reserve only for MTP the launch
                    # resolver will actually emit (needs a head/drafter and a binary
                    # that supports --spec-type mtp).
                    _mtp_model_for_fit = bool(
                        self._nextn_predict_layers
                        or _is_mtp_model_name(model_identifier, model_path)
                        or bool(mtp_draft_path)
                    ) and not (
                        # Drafterless Gemma falls back to ngram-mod; reserve no
                        # drafter VRAM for it (mirrors the launch resolver).
                        _is_gemma_mtp_name(model_identifier, model_path)
                        and not mtp_draft_path
                        and not self._nextn_predict_layers
                    )
                    _mtp_binary_ok = True
                    _mtp_probe_raised = False
                    if not _user_mtp_via_extras:
                        try:
                            _fit_caps = self.probe_server_capabilities(binary) or {}
                            _mtp_binary_ok = bool(
                                _fit_caps.get("supports_dspark")
                                if _mtp_effective == "dspark"
                                else _fit_caps.get("mtp_token")
                            )
                        except Exception:
                            _mtp_binary_ok = False
                            _mtp_probe_raised = True
                    _auto_studio_mtp = (
                        not _extra_args_set_spec_type(extra_args)
                        and _mtp_model_for_fit
                        and (
                            _mtp_effective in ("mtp", "mtp+ngram", "dspark")
                            or (_mtp_effective == "auto" and not _mtp_sub_3b_for_fit)
                        )
                        and (
                            _mtp_binary_ok
                            # Reserve on a raised (uncached) probe too: it re-probes in
                            # _build_speculative_flags and may still engage MTP (embedded
                            # head or separate drafter -- _mtp_model_for_fit covers both).
                            or _mtp_probe_raised
                        )
                    )
                    _mtp_will_engage = bool(
                        _user_mtp_via_extras or _user_draft_via_extras or _auto_studio_mtp
                    )
                    # The duplicated full target-KV copy (ctx_tgt) is an MTP-only
                    # cost: the MTP head runs a second context over the target
                    # model's own KV geometry. Every separate-drafter mode loads a
                    # distinct drafter with its own KV and keeps no such copy, so
                    # only charge it when the engaged mode is truly MTP. Those modes
                    # arrive by two routes, hence the two conditions: draft-simple
                    # and draft-eagle3 via _user_draft_via_extras (already excluded,
                    # they never set _user_mtp_via_extras), and DSpark via Studio's
                    # own resolution, which does set _auto_studio_mtp.
                    _engaged_is_mtp = bool(
                        _user_mtp_via_extras or (_auto_studio_mtp and _mtp_effective != "dspark")
                    )

                    # Effective draft depth: extras win (last-wins at launch), else
                    # the field, else the platform default (2 GPU / 3 CPU).
                    _extra_n_max = _extra_args_spec_draft_n_max(extra_args)
                    _mtp_eff_n_max = _extra_n_max if _extra_n_max is not None else spec_draft_n_max
                    if _mtp_eff_n_max is None:
                        # _detected_gpus (not gpus) so manual -- which empty
                        # gpus to bypass the planner -- keep the GPU draft depth the
                        # launch flags also use, instead of the CPU default.
                        _mtp_eff_n_max = (
                            3 if _mtp_effective == "dspark" else (2 if _detected_gpus else 3)
                        )
                    # Separate-drafter weights live on GPU (an embedded head is
                    # already in model_size). Size the drafter the launch loads, by
                    # precedence: extras --model-draft (last-wins), else Unsloth's
                    # emitted mtp_draft_path, else the env drafter. Sizing the wrong
                    # one would under-reserve and OOM.
                    _cli_draft_for_budget = _extra_args_mtp_draft_path(extra_args, env = {})
                    _studio_draft_for_budget = (
                        mtp_draft_path
                        if (
                            _mtp_will_engage
                            and mtp_draft_path
                            and not _extra_args_set_spec_type(extra_args)
                        )
                        else None
                    )
                    # Same scrubbed view: reserving for an inherited drafter the child
                    # never loads shrinks the context or rejects a placement that fits.
                    _env_draft_for_budget = _extra_args_mtp_draft_path([], env = _spec_env)
                    _mtp_draft_for_budget = (
                        _cli_draft_for_budget or _studio_draft_for_budget or _env_draft_for_budget
                    )
                    # Drafter offloaded to CPU keeps its weights+KV off the GPU, so
                    # drop it from the budget (an embedded head stays in the model).
                    # Consult the env too: the child honors LLAMA_ARG_N_GPU_LAYERS_DRAFT.
                    _draft_on_cpu = _extra_args_draft_offloaded_to_cpu(extra_args, env = os.environ)
                    if _draft_on_cpu:
                        _mtp_draft_for_budget = None
                    _mtp_draft_weights = 0
                    if _mtp_draft_for_budget:
                        try:
                            _mtp_draft_weights = self._get_gguf_size_bytes(_mtp_draft_for_budget)
                        except Exception:
                            _mtp_draft_weights = 0
                    # Draft K/V types (f16 by default; independent extras overrides).
                    _mtp_draft_ck, _mtp_draft_cv = _extra_args_draft_cache_types(extra_args)

                    # Byte-accurate reserve when dims allow, else None -> flat fallback.
                    mtp_overhead_fn: Optional[Callable[[int], int]] = None
                    # True when the byte reserve is the drafter weights ONLY because
                    # its KV couldn't be sized; the flat fraction must then stay on
                    # as the cushion for that unsized draft KV (it is not covered by
                    # the weights-only mtp_overhead_fn).
                    _mtp_kv_unsized = False
                    if _mtp_will_engage:
                        _probe_ctx = self._context_length or (
                            effective_ctx if effective_ctx > 0 else 4096
                        )
                        _draft_kv_probe = self._mtp_draft_kv_bytes(
                            _probe_ctx,
                            drafter_path = _mtp_draft_for_budget,
                            draft_cache_type_k = _mtp_draft_ck,
                            draft_cache_type_v = _mtp_draft_cv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                        )
                        if (
                            self._estimate_mtp_overhead_bytes(
                                _probe_ctx,
                                spec_draft_n_max = _mtp_eff_n_max,
                                draft_cache_type_k = _mtp_draft_ck,
                                draft_cache_type_v = _mtp_draft_cv,
                                drafter_path = _mtp_draft_for_budget,
                                draft_weights_bytes = _mtp_draft_weights,
                                n_parallel = n_parallel,
                                mtp_keeps_target_ctx = _engaged_is_mtp,
                                swa_full = swa_full,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                            )
                            is not None
                        ):
                            # Reserve is weights-only when the draft KV is unsizable.
                            _mtp_kv_unsized = _draft_kv_probe is None

                            # Closure binding this load's draft params; ctx varies.
                            def mtp_overhead_fn(
                                ctx: int,
                                _n: int = _mtp_eff_n_max,
                                _ck: Optional[str] = _mtp_draft_ck,
                                _cv: Optional[str] = _mtp_draft_cv,
                                _dp: Optional[str] = _mtp_draft_for_budget,
                                _w: int = _mtp_draft_weights,
                                _np: int = n_parallel,
                                _mtp: bool = _engaged_is_mtp,
                                _swa_full: bool = swa_full,
                                _kv_unified: bool = planned_kv_unified,
                                _n_ubatch: Optional[int] = _effective_ubatch,
                                _flash_attn: bool = planned_flash_attn,
                            ) -> int:
                                v = self._estimate_mtp_overhead_bytes(
                                    ctx,
                                    spec_draft_n_max = _n,
                                    draft_cache_type_k = _ck,
                                    draft_cache_type_v = _cv,
                                    drafter_path = _dp,
                                    draft_weights_bytes = _w,
                                    n_parallel = _np,
                                    mtp_keeps_target_ctx = _mtp,
                                    swa_full = _swa_full,
                                    kv_unified = _kv_unified,
                                    n_ubatch = _n_ubatch,
                                    flash_attn = _flash_attn,
                                )
                                return v if v is not None else 0

                    def _mtp_bytes(ctx: int) -> int:
                        return mtp_overhead_fn(ctx) if mtp_overhead_fn is not None else 0

                    def _kv_bytes(ctx: int) -> int:
                        return self._estimate_kv_cache_bytes(
                            ctx,
                            cache_type_kv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                        )

                    def _cc_bytes(ctx: int, n_gpus: int = 1) -> int:
                        # Context-linear compute-buffer growth (flash-attn KQ mask +
                        # attention scratch); the flat _compute_buffer_pipeline folded
                        # into model_size_fit only covers ctx -> 0. Charged per
                        # candidate context so the fit can't over-pin and spill. The
                        # rate depends on the KV cache type (quantized adds a dequant
                        # scratch), so pass it through. In a layer split this buffer is
                        # replicated on EVERY device (measured ~equal per GPU), so scale
                        # by the device count; a large model at high context otherwise
                        # under-reserves ~(n-1)x it (e.g. Qwen3.5-397B on 3 GPUs). The
                        # per-device rate also steps up once split, unless llama.cpp
                        # declines the pipeline parallelism that causes the step.
                        return max(1, n_gpus) * self._compute_buffer_ctx_bytes(
                            ctx,
                            _effective_ubatch,
                            cache_type_kv,
                            layer_split = n_gpus > 1 and not _pipeline_parallel_off,
                        )

                    def _cc_split_extra(ctx: int) -> int:
                        # Per-device step from the single-device rate to the split one,
                        # for the paths that must select GPUs before they know the
                        # count. 0 when llama.cpp declines pipeline parallelism.
                        return max(0, _cc_bytes(ctx, 2) // 2 - _cc_bytes(ctx))

                    # Layer-split compute buffer (one lump; tensor mode reserves it
                    # per device in _plan_tensor_parallel). Context-independent, so
                    # fold it into the model footprint for the branches below. Falls
                    # back to the flat reserve when dims are missing (returns 0), a
                    # safe upper bound since the tensor buffer >= the layer one.
                    _compute_buffer_pipeline = self._estimate_compute_buffer_bytes(
                        n_ubatch = _effective_ubatch,
                        n_parallel = n_parallel,
                        per_device_tensor = False,
                    )
                    if _compute_buffer_pipeline <= 0:
                        _compute_buffer_pipeline = (
                            self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Layer split adds a fixed per-device overhead on every GPU. The
                    # folded buffer covers one device; reserve the extra devices'
                    # share so a k-GPU split can't pin a context that OOMs a device
                    # (k=1 adds nothing).
                    _pipeline_overhead_bytes = self._PIPELINE_PER_DEVICE_OVERHEAD_MIB * 1024 * 1024

                    # Auto-cap context to fit VRAM and select GPUs. Explicit n_ctx:
                    # honor it, cap only if it fits no combination. Auto (native):
                    # prefer fewer GPUs with reduced context (multi-GPU is slower).
                    gpu_indices, use_fit = None, True
                    # Per-GPU weight proportions for tensor mode (None lets
                    # llama.cpp split by free VRAM).
                    tp_tensor_split: Optional[list[int]] = None
                    explicit_ctx = requested_ctx > 0
                    # Flat MTP reserve fraction: used only as the fallback when the
                    # byte-accurate mtp_overhead_fn can't size the draft KV (dims
                    # unavailable, or _mtp_kv_unsized = weights-only). A separate
                    # drafter on CPU uses no GPU (no reserve); an embedded head is on
                    # GPU regardless of draft-offload flags (keep its reserve).
                    _flat_mtp_engages = _mtp_will_engage and (
                        mtp_overhead_fn is None or _mtp_kv_unsized
                    )
                    _draft_cpu_no_embedded = _draft_on_cpu and not self._nextn_predict_layers
                    # MTP reserves GPU VRAM unless its only drafter is a separate
                    # CPU-offloaded one (an embedded head stays on GPU). The tensor
                    # path reserves like the layer path; gate both on this.
                    _mtp_reserves_gpu = _mtp_will_engage and not _draft_cpu_no_embedded
                    _flat_mtp_reserve = (
                        _MTP_VRAM_RESERVE_FRAC
                        if (_flat_mtp_engages and not _draft_cpu_no_embedded)
                        else 0.0
                    )
                    _pin_fraction = self._GPU_PIN_VRAM_FRACTION - _flat_mtp_reserve

                    # Charge the soft overhead _CTX_FIT_VRAM_FRACTION under-covers on tight
                    # tiers, gated so plain dense loads (#5106) only pay the CUDA-ctx base.
                    # CUDA/cuBLAS context is discrete-GPU only (not Metal); the mmproj and
                    # MTP draft-graph buffers exist on every backend.
                    _soft_overhead = self._CUDA_CONTEXT_RESERVE_BYTES if gpus else 0
                    if effective_is_vision and mmproj_size > 0:
                        _soft_overhead += int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                    if _mtp_reserves_gpu:
                        _soft_overhead += self._MTP_DRAFT_COMPUTE_BYTES
                    model_size_fit = model_size + _compute_buffer_pipeline + _soft_overhead

                    def _subset_model_size(n_gpus: int) -> int:
                        return model_size_fit + max(0, n_gpus - 1) * _pipeline_overhead_bytes

                    # Unified-memory budget (0 off Apple Silicon) for the no-GPU Metal cap below.
                    _apple_budget_mib = self._apple_metal_memory_budget_bytes() // (1024 * 1024)

                    def _restore_after_tensor_downgrade():
                        # Restore the quantized KV + extras tensor dropped (layer
                        # split supports them), minus --split-mode.
                        nonlocal cache_type_kv, _cache_type_from_env, extra_args
                        if _tensor_dropped_cache_type_kv is not None:
                            cache_type_kv = _tensor_dropped_cache_type_kv
                            _cache_type_from_env = False
                        extra_args = strip_split_mode_only(
                            _tensor_dropped_extra_args
                            if _tensor_dropped_extra_args is not None
                            else extra_args
                        )

                    # The route fallback retry is tensor-off; keep it multi-GPU.
                    if preserve_multi_gpu_on_layer:
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))

                    if tensor_parallel and self._tensor_split_aborts(binary, model_identifier):
                        # Aborted on tensor for this model this session (#6415); skip
                        # tensor upfront, layer split serves it.
                        logger.info(
                            "Tensor parallelism skipped: this llama.cpp build aborted "
                            "on --split-mode tensor for this model earlier this "
                            "session; using layer split across %d GPU(s).",
                            len(gpus),
                        )
                        tensor_parallel = False
                        # Keep the multi-GPU request (gated on it, not the cache).
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        _restore_after_tensor_downgrade()

                    # Tensor mode replicates a compute buffer on every GPU, so drop
                    # GPUs below that reserve from the set up front (gpu_indices
                    # becomes the CUDA_VISIBLE_DEVICES mask, fully excluding them).
                    tp_gpus = gpus
                    # Manual mode owns the layer count and context, so it skips
                    # the memory-based planner; its toggle still emits
                    # --split-mode tensor below (split by free VRAM, or by the
                    # Split ratio if set). auto plans here.
                    plan_tp = tensor_parallel and gpu_memory_mode != "manual"
                    if plan_tp:
                        # Deterministic per-device compute buffer (replicated on
                        # every device in tensor mode); flat fallback when dims
                        # are unavailable. _plan_tensor_parallel uses the same.
                        _tp_reserve_bytes = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = True,
                        )
                        reserve_mib = (
                            _tp_reserve_bytes // (1024 * 1024)
                            if _tp_reserve_bytes > 0
                            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
                        )
                        # Admit by usable budget (free - (1-frac)*total), not raw
                        # free: a partly-used big card can clear the reserve on raw
                        # free yet have no budget left.
                        tp_gpus = [g for g in gpus if _gpu_usable(g) >= reserve_mib]

                    if plan_tp and len(tp_gpus) < 2:
                        # Tensor parallelism needs >= 2 usable GPUs. On a single
                        # GPU --split-mode tensor is a no-op; with 0 GPUs (CPU-only
                        # or probe failed) it must not reach llama-server; and a
                        # GPU below the buffer reserve can't participate. Drop the
                        # flag and fall through to normal layer/CPU allocation.
                        logger.info(
                            "Tensor parallelism requested but only %d of %d GPU(s) "
                            "have enough free VRAM for the compute buffer; "
                            "ignoring (needs >= 2).",
                            len(tp_gpus),
                            len(gpus),
                        )
                        tensor_parallel = False
                        # GPUs below tensor's compute-buffer reserve can still do layer
                        # split, so keep multi-GPU (mirrors the budget/geometry drops);
                        # _select_gpus caps unusable cards.
                        if len(gpus) >= 2:
                            _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        # Layer split supports a quantized KV the tensor attempt
                        # dropped; restore the original cache type + extras (minus
                        # --split-mode) so the layer launch re-emits them.
                        _restore_after_tensor_downgrade()

                    if tensor_parallel and tp_gpus:
                        # Pooled usable budget (after each device's compute buffer)
                        # must hold the non-shrinkable footprint: weights + the MTP
                        # reserve. The planner can shrink ctx/KV, not these.
                        _tp_weight_budget_mib = (
                            sum(_gpu_usable(g) for g in tp_gpus) - len(tp_gpus) * reserve_mib
                        )
                        _tp_flat_mtp = 2 * 1024**3  # flat reserve when dims unavailable
                        if not _mtp_reserves_gpu:
                            # No MTP, or its only drafter is CPU-offloaded (no GPU).
                            _tp_mtp_floor = 0
                        elif mtp_overhead_fn is not None and not _mtp_kv_unsized:
                            _tp_mtp_floor = _mtp_bytes(
                                min(2048, effective_ctx) if effective_ctx > 0 else 2048
                            )
                        else:
                            # Dims unavailable / weights-only: tensor mode has no
                            # --fit valve, so keep the flat reserve as the unsized-KV
                            # cushion, never below the known byte reserve.
                            _tp_mtp_floor = max(
                                _tp_flat_mtp,
                                _mtp_bytes(min(2048, effective_ctx) if effective_ctx > 0 else 2048),
                            )
                        _tp_required_mib = (model_size + _tp_mtp_floor + _soft_overhead) / (
                            1024 * 1024
                        )
                        if _tp_weight_budget_mib <= _tp_required_mib:
                            logger.info(
                                "Tensor parallelism requested but the pooled VRAM "
                                "budget cannot hold the weights, MTP reserve, and "
                                "per-device compute buffers; falling back to layer split."
                            )
                            tensor_parallel = False
                            # Weights needed >1 card, so keep multi-GPU across the
                            # usable tensor GPUs.
                            if len(tp_gpus) >= 2:
                                _layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))
                            # Restore the dropped quantized KV + cache extras (minus
                            # --split-mode); layer split supports them.
                            _restore_after_tensor_downgrade()

                    if tensor_parallel and tp_gpus:
                        # Tensor-parallel allocation; see _plan_tensor_parallel.
                        target_ctx = (
                            effective_ctx
                            if explicit_ctx
                            else (self._context_length or effective_ctx)
                        )
                        # When the draft KV couldn't be sized (weights-only reserve),
                        # the planner's mtp_overhead_fn is non-None but covers only
                        # weights, so pass the flat cushion for the unsized KV (else
                        # the binary search spends it on context).
                        _tp_unsized_mtp_reserve = (
                            2 * 1024**3 if (_mtp_reserves_gpu and _mtp_kv_unsized) else 0
                        )
                        (
                            effective_ctx,
                            max_available_ctx,
                            gpu_indices,
                            tp_tensor_split,
                        ) = self._plan_tensor_parallel(
                            tp_gpus,
                            model_size,
                            target_ctx,
                            cache_type_kv = cache_type_kv,
                            n_parallel = n_parallel,
                            mtp_engaged = _mtp_reserves_gpu,
                            mtp_overhead_fn = mtp_overhead_fn,
                            mtp_flat_reserve_bytes = _tp_unsized_mtp_reserve,
                            # Report the UI ceiling from native ctx, not the
                            # explicit small request.
                            max_target_ctx = self._context_length or target_ctx,
                            total_by_idx = total_by_idx,
                            n_ubatch = _effective_ubatch,
                            soft_overhead_bytes = _soft_overhead,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                        )
                        use_fit = False
                    elif gpus and self._can_estimate_kv() and effective_ctx > 0:
                        # Compute the largest hardware-aware cap from the model's
                        # native context across all usable GPU subsets (for UI
                        # bounds), independent of the currently requested context.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        if native_ctx_for_cap > 0:
                            ranked_for_cap = sorted(
                                gpus,
                                key = lambda g: _gpu_usable(
                                    g, _CTX_FIT_VRAM_FRACTION - _flat_mtp_reserve
                                ),
                                reverse = True,
                            )
                            best_cap = 0
                            _cap_fraction = _CTX_FIT_VRAM_FRACTION - _flat_mtp_reserve
                            for n_gpus in range(1, len(ranked_for_cap) + 1):
                                subset = ranked_for_cap[:n_gpus]
                                # Per-GPU-consistent pool budget (fixes mixed
                                # known/unknown totals); pass it as an absolute
                                # budget so the fit and the check below agree.
                                pool_budget = _pool_budget_mib(subset, _cap_fraction)
                                _ms = _subset_model_size(n_gpus)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    native_ctx_for_cap,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    total_mib = None,
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib <= pool_budget:
                                    best_cap = max(best_cap, capped)
                            if best_cap > 0:
                                max_available_ctx = best_cap
                            else:
                                # Weights exceed 90% of every GPU subset, so no
                                # context fits. Anchor the UI "safe zone" at 4096
                                # so the slider warns above the fallback.
                                max_available_ctx = min(4096, native_ctx_for_cap)

                        if explicit_ctx:
                            # Honor the requested context verbatim. If it fits,
                            # pin GPUs and skip --fit; else ship -c <ctx> --fit
                            # on and let llama-server flex -ngl (CPU offload).
                            requested_total = (
                                model_size_fit
                                + _kv_bytes(effective_ctx)
                                + _mtp_bytes(effective_ctx)
                                + _cc_bytes(effective_ctx)
                            )
                            # The compute buffer is replicated on every device in a
                            # layer split; fold it into the per-device reserve so a
                            # multi-GPU pin sizes each card for its own copy. Priced
                            # single-device (the count comes out of this figure), then
                            # re-checked at the split rate, falling back to --fit on.
                            gpu_indices, use_fit = self._select_gpus_split_aware(
                                requested_total,
                                gpus,
                                usable_fraction = _pin_fraction,
                                total_by_idx = total_by_idx,
                                per_device_overhead_bytes = _pipeline_overhead_bytes
                                + _cc_bytes(effective_ctx),
                                min_gpus = _layer_min_gpus,
                                split_extra_bytes = _cc_split_extra(effective_ctx),
                            )
                            # No silent shrink: effective_ctx stays == requested_ctx.
                        else:
                            # Auto context: prefer fewer GPUs, cap to fit. Same
                            # headroom threshold as _select_gpus (#5106). Rank by the
                            # active pin fraction so the order matches the fit budget.
                            pin_fraction = _pin_fraction
                            ranked = sorted(
                                gpus, key = lambda g: _gpu_usable(g, pin_fraction), reverse = True
                            )
                            # Skips _select_gpus, so apply its cap: count only cards
                            # whose usable VRAM clears the per-device layer overhead.
                            _pipeline_overhead_mib = _pipeline_overhead_bytes / (1024 * 1024)
                            _auto_min_gpus = max(
                                1,
                                min(
                                    _layer_min_gpus,
                                    sum(
                                        1
                                        for g in ranked
                                        if _gpu_usable(g, pin_fraction) > _pipeline_overhead_mib
                                    )
                                    or 1,
                                ),
                            )
                            for n_gpus in range(_auto_min_gpus, len(ranked) + 1):
                                subset = ranked[:n_gpus]
                                pool_budget = _pool_budget_mib(subset, pin_fraction)
                                _ms = _subset_model_size(n_gpus)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    effective_ctx,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    total_mib = None,
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib > pool_budget:
                                    continue
                                # Pooling admits a subset whose per-device reserve
                                # does not fit the smallest card, which then OOMs at
                                # launch; only knowable once `capped` is chosen.
                                _usable_mib = [_gpu_usable(g, pin_fraction) for g in subset]
                                _reserve_at = lambda c, n = n_gpus: (
                                    (_pipeline_overhead_bytes if n > 1 else 0)
                                    + _cc_bytes(c, n) // n
                                )
                                if not self._every_gpu_holds_reserve(
                                    _usable_mib, _reserve_at(capped)
                                ):
                                    # Keep the largest context the smallest card holds
                                    # rather than lose the subset to the 4096 drop below.
                                    # Costs no context: subsets are prefixes of a
                                    # usable-descending ranking and the reserve is the
                                    # same function of context for every n > 1, so any
                                    # later subset clears the gate no higher than this.
                                    capped = self._cap_ctx_to_per_device_reserve(
                                        capped, _usable_mib, _reserve_at
                                    )
                                    if capped <= 0:
                                        continue
                                    # Pooled terms only shrink with the context, so this
                                    # cannot newly fail; re-price rather than lean on that
                                    # across the KV and MTP branches.
                                    kv = _kv_bytes(capped)
                                    footprint_mib = (
                                        _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                    ) / (1024 * 1024)
                                    if footprint_mib > pool_budget:
                                        continue
                                effective_ctx = capped
                                gpu_indices = sorted(idx for idx, _ in subset)
                                use_fit = False
                                break
                            else:
                                # Native ctx doesn't fit. Drop to 4096 and
                                # re-check before --fit on: a model overflowing
                                # at 131k may pin fine with a 4096 KV (#5106).
                                effective_ctx = min(4096, effective_ctx)
                                if effective_ctx > 0:
                                    for n_gpus in range(_auto_min_gpus, len(ranked) + 1):
                                        subset = ranked[:n_gpus]
                                        kv = _kv_bytes(effective_ctx)
                                        footprint_mib = (
                                            _subset_model_size(n_gpus)
                                            + kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, n_gpus)
                                        ) / (1024 * 1024)
                                        if footprint_mib > _pool_budget_mib(subset, pin_fraction):
                                            continue
                                        # Pooling hides the per-device reserve here too.
                                        if not self._every_gpu_holds_reserve(
                                            (_gpu_usable(g, pin_fraction) for g in subset),
                                            (_pipeline_overhead_bytes if n_gpus > 1 else 0)
                                            + _cc_bytes(effective_ctx, n_gpus) // n_gpus,
                                        ):
                                            continue
                                        gpu_indices = sorted(idx for idx, _ in subset)
                                        use_fit = False
                                        break

                    elif gpus:
                        # Can't estimate KV -- file-size-only check; keep the
                        # ceiling at native context (already the default).
                        logger.debug(
                            "Falling back to file-size-only GPU selection",
                            model_size_gb = round(model_size / (1024**3), 2),
                        )
                        # Add the byte-accurate MTP reserve here too when it is
                        # available; otherwise _pin_fraction carries the flat
                        # fallback (the two are mutually exclusive by design).
                        _fs_total = model_size_fit + _mtp_bytes(
                            self._context_length or effective_ctx or 4096
                        )
                        gpu_indices, use_fit = self._select_gpus(
                            _fs_total,
                            gpus,
                            usable_fraction = _pin_fraction,
                            total_by_idx = total_by_idx,
                            per_device_overhead_bytes = _pipeline_overhead_bytes,
                            min_gpus = _layer_min_gpus,
                        )
                        if use_fit and not explicit_ctx:
                            # Weights don't fit on any subset; default UI to 4096
                            # so the slider isn't on an unusable native ctx.
                            effective_ctx = min(4096, effective_ctx) if effective_ctx > 0 else 4096

                    elif _apple_budget_mib > 0 and effective_ctx > 0:
                        # No GPU on Metal: the branches above are skipped and the context
                        # stays at native, over-committing unified memory (#5118, #6529).
                        # Cap with the same fit math (--fit on stays as a backstop); only
                        # auto context shrinks, explicit is honored.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        # Reserve the flat MTP fraction up front like the discrete
                        # _pin_fraction, so an unsized MTP draft (e.g. Qwen3.6-MTP, #6529)
                        # can't over-commit. No-op when MTP is off; exclusive with the
                        # byte-accurate _mtp_bytes reserve.
                        _apple_fit_budget_mib = int(
                            _apple_budget_mib * max(0.0, 1.0 - _flat_mtp_reserve)
                        )
                        if self._can_estimate_kv():
                            cap = self._fit_context_to_vram(
                                native_ctx_for_cap,
                                _apple_fit_budget_mib,
                                model_size_fit,
                                cache_type_kv,
                                swa_full = swa_full,
                                n_parallel = n_parallel,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                                mtp_engaged = _mtp_reserves_gpu,
                                mtp_overhead_fn = mtp_overhead_fn,
                                compute_ctx_bytes_fn = _cc_bytes,
                                budget_frac = 1.0,
                                total_mib = None,
                            )
                            _cap_footprint_mib = (
                                model_size_fit + _kv_bytes(cap) + _mtp_bytes(cap) + _cc_bytes(cap)
                            ) / (1024 * 1024)
                            # Fit returns the request unchanged when it fits OR weights
                            # exceed budget; only the latter over-commits, so floor to 4096.
                            max_available_ctx = (
                                cap
                                if _cap_footprint_mib <= _apple_fit_budget_mib
                                else min(4096, native_ctx_for_cap)
                            )
                        else:
                            # No KV estimate: mirror the discrete file-size-only fallback
                            # and floor to 4096 rather than launch at native and over-commit.
                            max_available_ctx = min(4096, native_ctx_for_cap)
                        if not explicit_ctx:
                            effective_ctx = max_available_ctx

                    # Prefer fewer serving slots on GPU over --fit on offload: when the extra
                    # --parallel slots push the footprint past the pin budget, llama-server
                    # offloads layers to host and decode collapses ~3x (#6718). Retry the fit
                    # at fewer slots, keeping the largest count that stays fully on GPU and the
                    # chosen context. Skips tensor mode / Metal / KV-inestimable paths.
                    if (
                        use_fit
                        and n_parallel > 1
                        and gpus
                        and self._can_estimate_kv()
                        and effective_ctx > 0
                    ):
                        # Slot-independent footprint (folded compute buffer swapped out so the
                        # helper re-adds a slot-sized one per candidate).
                        _base_footprint = (
                            model_size_fit
                            - _compute_buffer_pipeline
                            + _mtp_bytes(effective_ctx)
                            + _cc_bytes(effective_ctx)
                        )
                        _gi_slots, _uf_slots, _slots = self._slots_that_fit_on_gpu(
                            n_parallel,
                            effective_ctx,
                            gpus,
                            total_by_idx,
                            _base_footprint,
                            cache_type_kv,
                            _pin_fraction,
                            _pipeline_overhead_bytes + _cc_bytes(effective_ctx),
                            _layer_min_gpus,
                            _effective_ubatch,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                            split_extra_bytes = _cc_split_extra(effective_ctx),
                        )
                        if not _uf_slots:
                            logger.info(
                                "Serving slots reduced %d -> %d to keep the model on GPU "
                                "(avoid --fit offload) at context %d.",
                                n_parallel,
                                _slots,
                                effective_ctx,
                            )
                            gpu_indices, use_fit, n_parallel = _gi_slots, False, _slots

                    # MTP reserve at the final context, for the logs below.
                    _mtp_reserve_bytes = _mtp_bytes(effective_ctx) if _mtp_will_engage else 0
                    if _mtp_will_engage:
                        _mtp_note = (
                            f"MTP reserve: {_mtp_reserve_bytes / (1024**3):.2f} GB "
                            f"(draft KV @ {effective_ctx} + verify n_max={_mtp_eff_n_max}"
                            + (", flat-frac fallback" if mtp_overhead_fn is None else "")
                            + "), "
                        )
                    else:
                        _mtp_note = ""

                    if effective_ctx < original_ctx:
                        kv_est = _kv_bytes(effective_ctx)
                        logger.info(
                            f"Context auto-reduced: {original_ctx} -> {effective_ctx} "
                            f"(model: {model_size / (1024**3):.1f} GB, "
                            f"est. KV cache: {kv_est / (1024**3):.1f} GB, "
                            f"{_mtp_note}".rstrip(", ")
                            + ")"
                        )

                    kv_cache_bytes = _kv_bytes(effective_ctx)
                    mmproj_note = (
                        f"mmproj: {mmproj_size / (1024**3):.1f} GB, " if mmproj_size else ""
                    )
                    logger.info(
                        f"GGUF size: {gguf_size / (1024**3):.1f} GB, "
                        f"{mmproj_note}"
                        f"est. KV cache: {kv_cache_bytes / (1024**3):.1f} GB, "
                        f"{_mtp_note}"
                        f"context: {effective_ctx}, "
                        # --fit flag state, not "does it fit": off means this subset provably fits.
                        f"GPUs free: {gpus}, selected: {gpu_indices}, --fit: {'on' if use_fit else 'off'}"
                    )
                except Exception as e:
                    logger.warning(f"GPU selection failed ({e}), using --fit on")
                    gpu_indices, use_fit = None, True
                    tp_tensor_split = None
                    effective_ctx = requested_ctx  # fall back to original

                # An unenumerated explicit Vulkan ordinal can't be pinned; fail loudly
                # instead of fitting onto an unselected device. Clear the raw selection
                # the early state-publish recorded so it never leaks into gpu_ids (#7239).
                if _vulkan_explicit_unmatched:
                    self._gpu_ids = None
                    self._requested_gpu_ids = None
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {_vulkan_requested_ids} not "
                        f"present. Available Vulkan devices: {_vulkan_available_ordinals}."
                    )

                # GPU picker: when no narrower subset was chosen (manual, or
                # a failed/file-size selection), pin the whole picked set so the
                # model can't spill onto an unpicked GPU.
                if gpu_ids and gpu_indices is None:
                    gpu_indices = sorted(gpu_ids)
                # Auto Vulkan fit prefers discrete GPUs and keeps that pool pinned.
                elif (
                    is_vulkan_backend
                    and gpu_memory_mode != "manual"
                    and use_fit
                    and gpu_indices is None
                    and _detected_gpus
                ):
                    gpu_indices = sorted(idx for idx, _free in _detected_gpus)

                # Status reports the explicit subset the launch actually uses,
                # while reload dedupe compares requested_gpu_ids so repeating a
                # wider request that fit narrowed does not restart the server.
                # A forced-CPU launch carries --device none, so it pins no GPU at all:
                # echoing the request as effective would report devices the runtime never
                # touched, and clearing an ineffective pick would then reload a CPU server
                # that is already in the target state. requested_gpu_ids still holds the
                # pick, so repeating it keeps matching.
                if gpu_ids and not _paravirtual_cpu_forced:
                    effective_pin = gpu_indices if gpu_indices is not None else gpu_ids
                    self._gpu_ids = sorted(int(idx) for idx in effective_pin)
                else:
                    self._gpu_ids = None

                # Unified-memory APUs load weights into system RAM (under WSL the VM
                # cap, not the ROCm-reported VRAM, is the real ceiling); refuse an
                # oversize load the OS would otherwise kill mid-flight. Base model
                # only: an optional MTP drafter is dropped by the MTP-drop fallback.
                # CUDA/ROCm ids only; a Vulkan build's gpu_indices are ggml ordinals.
                if (
                    model_size is not None
                    and not is_vulkan_backend
                    and self._amd_apu_wants_unified_memory(gpu_indices)
                ):
                    _ram_msg = self._apu_ram_shortfall_message(
                        model_size, self._available_system_memory_mib()
                    )
                    if _ram_msg:
                        raise RuntimeError(_ram_msg)

                # Audio input straight from the mmproj (clip.has_audio_encoder),
                # independent of token names.
                self._mmproj_has_audio = False
                if launch_mmproj_path:
                    try:
                        from utils.models.gguf_metadata import (
                            read_mmproj_audio_capability,
                        )
                        self._mmproj_has_audio = bool(
                            read_mmproj_audio_capability(launch_mmproj_path)
                        )
                    except Exception as e:
                        logger.debug(f"mmproj audio-capability read failed: {e}")

                cmd = [
                    binary,
                    "-m",
                    model_path,
                    "--port",
                    str(self._port),
                    "--parallel",
                    str(n_parallel),
                    "--flash-attn",
                    "on",  # Force flash attention for speed
                    # Error out at n_ctx instead of silently rotating the KV cache; frontend catches it and points the user at "Context Length".
                    "--no-context-shift",
                ]
                # A positive context is always passed (in auto-fit, --fit then
                # optimizes the gpu-layer offload around it). When auto-fit has
                # no explicit context, omit -c so --fit sizes it to fit VRAM:
                # "-c 0" would instead pin the FULL native context (llama.cpp's
                # -c handler sets fit_params_min_ctx = UINT32_MAX on value 0,
                # disabling --fit's reduction). See gpu_memory_mode.
                auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0
                if effective_ctx > 0:
                    cmd.extend(["-c", str(effective_ctx)])
                elif not auto_fit:
                    cmd.extend(["-c", "0"])

                server_caps = self.probe_server_capabilities(binary)

                # Report a clean public model id (matching GET /v1/models) rather
                # than the raw -m path in llama-server's own /v1/models and the
                # "model" field of its chat/completions responses.
                from core.inference.model_ids import public_model_id

                _alias = public_model_id(self._model_identifier or model_path)
                if _alias:
                    cmd.extend(["--alias", _alias])

                fully_gpu_offloaded = False
                # Set when a positional --tensor-split is emitted, so the env block
                # can pin CUDA to PCI order even without a GPU subset (see below).
                manual_tensor_split_emitted = False
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    # Pin the user's layer count and disable auto-fit. --fit off
                    # also means _ctx_integrity_flags must not add --fit-ctx.
                    use_fit = False
                    cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])
                    # Keep the first n_cpu_moe MoE layers' experts on CPU.
                    moe_flag = self._resolve_cpu_moe_flag(
                        n_cpu_moe,
                        self.n_moe_layers,
                        self._leading_dense_block_count or 0,
                    )
                    if moe_flag is not None:
                        cmd.extend(["--n-cpu-moe", str(moe_flag)])
                    elif n_cpu_moe:
                        # Requested on a dense model: nothing was emitted, so
                        # don't report a count llama-server never received.
                        self._n_cpu_moe = 0
                    # Distribute the model across GPUs by the user's per-GPU shares
                    # (--tensor-split). Works with layer split and tensor
                    # parallelism; --fit off means no fit/tensor abort. Only emit
                    # when >1 GPU is in use AND the list length matches that count:
                    # the field is hidden (not cleared) when the picker narrows to
                    # one, and a direct caller can send a stale ratio for a different
                    # GPU set. Studio drops any mismatch to the free-VRAM default
                    # (llama.cpp would silently zero-pad a short list, or abort past
                    # its 16-device cap).
                    _split_gpus = self._effective_gpu_count(gpu_indices)
                    if tensor_split and _split_gpus > 1:
                        # An all-zero/non-positive sanitized split assigns nothing
                        # anywhere, so fall through to the free-VRAM default in
                        # that case.
                        _sanitized_split = self._sanitize_tensor_split(tensor_split)
                        _split_total = sum(_sanitized_split)
                        if len(_sanitized_split) == _split_gpus and _split_total > 0:
                            cmd.extend(
                                ["--tensor-split", ",".join(f"{x:g}" for x in _sanitized_split)]
                            )
                            self._tensor_split = _sanitized_split
                            manual_tensor_split_emitted = True
                        else:
                            logger.warning(
                                "Dropping manual --tensor-split (%d entries for "
                                "%d GPUs, sanitized total %s); llama.cpp's "
                                "free-VRAM split applies instead",
                                len(tensor_split),
                                _split_gpus,
                                _split_total,
                            )
                            self._tensor_split = None
                    elif tensor_split:
                        # Single effective GPU: the split is never emitted, so
                        # don't report it as active via /status and /load.
                        self._tensor_split = None
                elif use_fit:
                    cmd.extend(["--fit", "on"])
                elif gpu_indices is not None:
                    # Fits on selected GPU(s) -- force all layers on GPU. --fit off is
                    # required: without it llama.cpp's default --fit on second-guesses
                    # and offloads ~1 GB at --parallel 4 even though the model fits.
                    cmd.extend(["-ngl", "-1", "--fit", "off"])
                    fully_gpu_offloaded = True

                # Expose Prometheus /metrics for the engine-stats logger, only
                # when the binary advertises it (older/custom binaries may not).
                if server_caps.get("supports_metrics"):
                    cmd.append("--metrics")
                self._slot_save_dir = None
                self._slot_save_binary = None
                self._prompt_cache_disabled = False
                if server_caps.get("supports_slot_save"):
                    try:
                        from utils.paths.storage_roots import (  # noqa: WPS433
                            llama_slot_cache_root,
                        )

                        slot_dir = llama_slot_cache_root()
                        slot_dir.mkdir(parents = True, exist_ok = True)
                        # Saved KV encodes chat content; keep it from other local users.
                        with contextlib.suppress(OSError):
                            os.chmod(slot_dir, 0o700)
                        cmd.extend(["--slot-save-path", str(slot_dir)])
                        self._slot_save_dir = str(slot_dir)
                        self._slot_save_binary = (binary, Path(binary).stat().st_mtime_ns)
                    except OSError:
                        self._slot_save_dir = None
                        self._slot_save_binary = None
                cmd.extend(
                    self._ctx_integrity_flags(
                        n_parallel,
                        use_fit,
                        auto_fit,
                        requested_ctx,
                        effective_ctx,
                        server_caps,
                    )
                )
                offload_overridden = _extra_args_set_any_flag(
                    extra_args, _GPU_OFFLOAD_OVERRIDE_FLAGS
                )
                threads_overridden = _extra_args_set_any_flag(extra_args, _THREAD_OVERRIDE_FLAGS)
                full_offload_tuning_active = fully_gpu_offloaded and not offload_overridden

                # Thread count: an unset --threads makes llama.cpp pick physical
                # cores (common_cpu_get_num_math), but an explicit --threads -1
                # resolves to hardware_concurrency() (every hyperthread), which
                # contends on the memory bus and slows CPU / hybrid decode. So
                # omit the flag when unset and only pin it for an explicit
                # override or the Windows full-offload OpenMP cap. Pass-through
                # thread flags in extra_args still win (appended last). #5692
                if (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and not threads_overridden
                ):
                    cmd.extend(["--threads", "2"])
                elif n_threads is not None and n_threads > 0:
                    cmd.extend(["--threads", str(n_threads)])

                # Enable Jinja chat template rendering
                cmd.extend(["--jinja"])

                # KV cache data type
                _valid_cache_types = {
                    "f16",
                    "bf16",
                    "q8_0",
                    "q4_0",
                    "q4_1",
                    "q5_0",
                    "q5_1",
                    "iq4_nl",
                    "f32",
                }
                # Normalize like the budget does (_planned_main_cache_types): a
                # case-sensitive match drops "Q8_0", emitting no flag, so llama.cpp
                # runs f16 while the estimate priced q8_0. Emit the normalized
                # spelling; kv_cache_type_from_str is case-sensitive.
                cache_type_kv = cache_type_kv.strip().lower() if cache_type_kv else cache_type_kv
                if (
                    cache_type_kv
                    and cache_type_kv in _valid_cache_types
                    and not _cache_type_from_env
                ):
                    cmd.extend(
                        [
                            "--cache-type-k",
                            cache_type_kv,
                            "--cache-type-v",
                            cache_type_kv,
                        ]
                    )
                    self._cache_type_kv = cache_type_kv
                    logger.info(f"KV cache type: {cache_type_kv}")
                else:
                    # An env-only type is left inherited (untouched) so an
                    # asymmetric K/V env reaches the child as set.
                    self._cache_type_kv = None

                # Tensor parallelism: split the model across GPUs by tensor
                # rather than by layer. The UI only offers it on multi-GPU; a
                # direct single-GPU caller is redundant (supported archs no-op,
                # unsupported ones abort and the /load path retries layer split).
                # Default (layer split) is left implicit by omitting the flag.
                # See llama.cpp --split-mode.
                if tensor_parallel:
                    cmd.extend(["--split-mode", "tensor"])
                    if tp_tensor_split and len(tp_tensor_split) > 1:
                        cmd.extend(
                            [
                                "--tensor-split",
                                ",".join(str(int(x)) for x in tp_tensor_split),
                            ]
                        )
                    self._tensor_parallel = True
                    self._layer_preserves_tensor_intent = False
                    logger.info(
                        "Tensor parallelism: --split-mode tensor, --tensor-split %s",
                        tp_tensor_split,
                    )
                else:
                    self._tensor_parallel = False
                    # > 1 only when a tensor request was downgraded but kept multi-GPU.
                    self._layer_preserves_tensor_intent = _layer_min_gpus > 1

                # Speculative decoding. See _build_speculative_flags for the
                # mode resolution, benchmarks, and llama.cpp references.
                _vulkan_pin_ids = gpu_indices if gpu_indices is not None else (gpu_ids or None)
                _draft_device = _extra_args_main_device(extra_args) if gpu_ids is None else None
                if _draft_device is None and is_vulkan_backend and _vulkan_pin_ids:
                    _draft_device = ",".join(f"Vulkan{i}" for i in _vulkan_pin_ids)
                launch_mtp_draft_path = self._resolve_launch_mtp_path(
                    mtp_draft_path = mtp_draft_path,
                    drafter_label = "DSpark" if _spec_canon == "dspark" else "MTP",
                )
                _pv_suppressed_draft_path: Optional[str] = None
                _pv_suppressed_spec_extra_args: Optional[List[str]] = None
                # Same shape as the projector drop above: the CPU pin below needs a
                # draft-layer flag from the probe, and without one the drafter keeps its
                # automatic placement on the virtualised device. Unlike the projector
                # this is throughput, not correctness -- a token is emitted only when the
                # CPU-side target samples it itself (common_sampler_sample_and_accept_n
                # pushes the target's own draw and stops at the first mismatch), so a
                # corrupt drafter can only get its proposals rejected, costing a forward
                # pass per step for nothing. So drop it, from both sources: Unsloth's
                # resolved drafter and a user/env one in the extras. The rest of the spec
                # group goes with it, since a --spec-type needing a draft model
                # (draft-simple, eagle3) aborts llama-server once the model is gone;
                # _build_speculative_flags re-derives the drafter-free modes below. A
                # user who already pinned the drafter (-ngld 0 / --spec-draft-device cpu)
                # keeps it; the probe only decides whether Unsloth can emit the flag.
                # And as with the projector, --gpu-layers-draft has existed since 2023,
                # so a missing flag means the probe failed: on an unanswered probe pin
                # with the legacy spelling rather than drop.
                # Keyed on the drafter the launch really loads, not one that merely
                # exists: a user-owned --spec-type returns early from
                # _build_speculative_flags, so Unsloth's resolved sibling never becomes
                # --model-draft and keying on it would strip a drafter-free mode
                # (--spec-type ngram-mod and its knobs) for a drafter that never
                # launches. The inherited env counts only where it survives to the child
                # (the launch scrubs LLAMA_ARG_SPEC_DRAFT_* unless the extras own
                # --spec-type), or os.environ would drop a drafter that never loads and
                # take the caller's spec tuning with it.
                _pv_draft_unpinnable = bool(
                    _paravirtual_cpu_forced
                    and not _paravirtual_draft_ngl_flag(server_caps)
                    and (
                        _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                        or (launch_mtp_draft_path and not _extra_args_set_spec_type(extra_args))
                    )
                    and not _extra_args_draft_offloaded_to_cpu(extra_args)
                )
                if _pv_draft_unpinnable:
                    logger.warning(
                        "Dropping the speculative drafter for this session: this "
                        "Mac's Metal device is virtualised and this llama-server "
                        "build advertises no draft-layer flag, so the drafter would "
                        "keep running on the device whose output is corrupt. Output "
                        "stays correct either way; this only costs speed. Run "
                        "'unsloth studio update' to install a build that supports it."
                    )
                    # Remember what was suppressed: the drafter stays on disk, so caller
                    # and route keep detecting it, and comparing that against the
                    # launched None would reload a healthy server on every repeat Apply.
                    _pv_suppressed_draft_path = launch_mtp_draft_path
                    launch_mtp_draft_path = None
                    if extra_args:
                        # Same reason: record the extras as REQUESTED next to the
                        # launched ones. Both comparators key on the caller's list,
                        # which keeps arriving unchanged, so comparing it against this
                        # rewrite would reload a healthy server on every repeat Apply.
                        _pv_suppressed_spec_extra_args = list(extra_args)
                        extra_args = strip_shadowing_flags(
                            extra_args,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                        # The slots were clamped for an MTP that just went away, so give
                        # them back. Restored before the spec flags are rebuilt, so the
                        # backstop below re-clamps if Unsloth's own resolution is MTP.
                        if _pv_extras_clamped_slots > 1 and not _extra_args_requests_mtp(
                            extra_args, env = _child_spec_env(extra_args)
                        ):
                            n_parallel = _pv_extras_clamped_slots
                            if "--parallel" in cmd:
                                cmd[cmd.index("--parallel") + 1] = str(n_parallel)
                            logger.info(
                                "Restoring %d parallel slots: the drafter drop left a "
                                "non-MTP server.",
                                n_parallel,
                            )
                spec_flags = self._build_speculative_flags(
                    speculative_type = speculative_type,
                    spec_draft_n_max = spec_draft_n_max,
                    extra_args = extra_args,
                    model_identifier = model_identifier,
                    model_path = model_path,
                    gpus = bool(_detected_gpus),
                    binary = binary,
                    mtp_draft_path = (None if _spec_canon == "dspark" else launch_mtp_draft_path),
                    dspark_draft_path = (launch_mtp_draft_path if _spec_canon == "dspark" else None),
                    dspark_fit_sized = not use_fit,
                    draft_device = _draft_device,
                )
                # _build_speculative_flags judged the stripped list, so a user
                # --spec-type the drop removed left the requested mode reading "auto"
                # while the caller's extras still mean "the user owns it" (None).
                # Restore the requested view, or the spec-mode compare never matches.
                if _extra_args_set_spec_type(_pv_suppressed_spec_extra_args):
                    self._requested_spec_mode = None
                # Remember where the spec block sits so a drafter-load failure
                # can be retried with these flags swapped out (see below).
                _spec_start = len(cmd)
                cmd.extend(spec_flags)

                # A *separate* drafter does not inherit the main --gpu-layers:
                # common_base_params_to_speculative overwrites n_gpu_layers with
                # params.speculative.draft.n_gpu_layers, whose default -1 means auto, so
                # the drafter would run the corrupt path the CPU pin exists to avoid. An
                # embedded MTP head needs nothing here: with no draft model that override
                # is skipped (has_draft is false) and the head follows the main
                # placement. Probe-gated like the other optional flags so the name
                # matches the build (-ngld predates --spec-draft-ngl) and an unknown
                # argument never refuses the start; a build advertising none of them had
                # its drafter dropped above. extra_args is inspected alongside spec_flags
                # because a user-owned --spec-type makes _build_speculative_flags return
                # nothing, leaving their --model-draft unpinned. Emitted after the
                # pass-through extras below, not here: the parser is last-wins and the
                # paravirtual strip covers only the *main* offload flags, so a user
                # -ngld/--spec-draft-ngl would otherwise undo this.
                _pv_draft_cpu_pin: List[str] = []
                _pv_mmproj_cpu_pin: List[str] = []
                _pv_split_mode_pin: List[str] = []
                # --gpu-layers 0 alone does not keep work off the device: with op_offload
                # on (the default) ggml_backend_sched still sends ops whose weights sit
                # in host buffers to a higher-priority backend, and Metal accepts them at
                # batch >= 32. --device none drops the device from the list outright.
                _pv_device_pin: List[str] = ["--device", "none"] if _paravirtual_cpu_forced else []
                if (
                    _paravirtual_cpu_forced
                    # Same env the child will see: when the extras own --spec-type the
                    # launch keeps LLAMA_ARG_SPEC_DRAFT_*, so a drafter named only there
                    # still loads and would take the corrupt device unpinned.
                    and _extra_args_mtp_draft_path(
                        [*spec_flags, *(extra_args or [])], env = _child_spec_env(extra_args)
                    )
                    is not None
                    and _paravirtual_draft_ngl_flag(server_caps)
                ):
                    # The layer count alone is not enough, for the same reason
                    # --gpu-layers 0 was not: common_base_params_to_speculative replaces
                    # the draft context's device list with the draft one, so the main
                    # --device none never reaches it and an empty draft list leaves every
                    # device visible. --device-draft is spelled the same across the
                    # whole supported range.
                    _pv_draft_cpu_pin = [
                        str(_paravirtual_draft_ngl_flag(server_caps)),
                        "0",
                        "--device-draft",
                        "none",
                    ]
                    logger.warning(
                        "Pinning the speculative drafter to CPU: this Mac's Metal "
                        "device is virtualised."
                    )

                # Emitted after the pass-through extras too, same last-wins reason.
                if _paravirtual_cpu_forced:
                    _pv_split_mode_pin = _paravirtual_split_mode_pin(extra_args)

                # Backstop for the MTP clamp above, which only sees an explicit
                # --spec-type in extra_args; speculative_type="auto"/"mtp" is not
                # resolved until _build_speculative_flags runs, here. The KV fit was
                # sized for more slots, so this over-reserves, never under-reserves.
                # Skipped when the user's extras own --spec-type: _build_speculative_flags
                # emits nothing then, and judging the empty flag list would fall through
                # to LLAMA_ARG_SPEC_TYPE and clamp a launch that is not MTP at all.
                _mtp_clamped_slots = 0
                if (
                    n_parallel > 1
                    and not _extra_args_set_spec_type(extra_args)
                    and _extra_args_requests_mtp(spec_flags, env = _child_spec_env(extra_args))
                ):
                    try:
                        _np_at = cmd.index("--parallel")
                    except ValueError:
                        _np_at = -1
                    if _np_at >= 0:
                        logger.warning(
                            "%s resolved to MTP speculative decoding, which does not "
                            "support %d parallel slots; using 1.",
                            model_identifier,
                            n_parallel,
                        )
                        cmd[_np_at + 1] = "1"
                        # _commit_effective_parallel_slots reports what launched, so
                        # rebind rather than only patching cmd.
                        _mtp_clamped_slots = n_parallel
                        n_parallel = 1

                # Apply custom chat template override if provided.
                self._chat_template_override = chat_template_override
                if chat_template_override:
                    import tempfile

                    flags = detect_reasoning_flags(
                        chat_template_override,
                        self._model_identifier,
                        log_source = "GGUF chat template override",
                    )
                    self._supports_reasoning = flags["supports_reasoning"]
                    self._reasoning_style = flags["reasoning_style"]
                    self._reasoning_effort_levels = flags.get("reasoning_effort_levels", [])
                    self._reasoning_always_on = flags["reasoning_always_on"]
                    self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                    self._supports_tools = flags["supports_tools"]

                    self._chat_template_file = tempfile.NamedTemporaryFile(
                        mode = "w",
                        encoding = "utf-8",
                        suffix = ".jinja",
                        delete = False,
                        prefix = "unsloth_chat_template_",
                    )
                    self._chat_template_file.write(chat_template_override)
                    self._chat_template_file.close()
                    cmd.extend(["--chat-template-file", self._chat_template_file.name])
                    logger.info(f"Using custom chat template file: {self._chat_template_file.name}")

                # Default thinking mode for reasoning models. The Qwen3.5 Small tier
                # (0.8B, 2B, 4B, 9B) ships with reasoning off by default and the rest of
                # the family has it on; the templates are byte-identical across sizes, so
                # this tier split only exists as a launch flag. Always-on templates ignore
                # the kwarg, so skip.
                if self._supports_reasoning and not self._reasoning_always_on:
                    thinking_default = True
                    mid = (model_identifier or "").lower()
                    if "qwen3.5" in mid or "qwen3.6" in mid:
                        # extract_model_size_b prefers MoE active params, reading
                        # 35B-A3B as 3B. Match total params instead: scan segments
                        # right to left so the size nearest the leaf beats a size-like
                        # parent dir, and keep its \s* / M-suffix spellings.
                        size_re = r"(?:^|[-_.])(\d+\.?\d*)\s*([bm])(?:$|[-_.])"
                        size_match = None
                        for seg in reversed(mid.replace("\\", "/").split("/")):
                            size_match = re.search(size_re, seg)
                            if size_match:
                                break
                        if size_match:
                            size_b = float(size_match.group(1))
                            if size_match.group(2) == "m":
                                size_b /= 1000.0
                            # <= 9, not < 9: 9B is the top of the Small tier, so it is off too.
                            if size_b <= 9:
                                thinking_default = False
                    self._reasoning_default = thinking_default
                    reasoning_kw = self._reasoning_kwargs(thinking_default)
                    # preserve_thinking is an independent kwarg. Default it OFF
                    # at launch so direct OpenAI-compatible callers that omit the
                    # field match the UI's default-off behavior (the bundled
                    # gemma-4 template also defaults it false; the frontend sends
                    # preserve_thinking per request once toggled on).
                    if self._supports_preserve_thinking:
                        reasoning_kw["preserve_thinking"] = False
                    cmd.extend(
                        [
                            "--chat-template-kwargs",
                            json.dumps(reasoning_kw),
                        ]
                    )
                    logger.info(f"Reasoning model: {reasoning_kw} by default")

                if launch_mmproj_path and effective_is_vision:
                    cmd.extend(["--mmproj", launch_mmproj_path])
                    logger.info(f"Using mmproj for vision: {launch_mmproj_path}")
                    # The projector never reads --gpu-layers: clip.cpp picks a GPU
                    # backend from the boolean mmproj_use_gpu, which defaults true, so
                    # the vision encoder would keep running the corrupt path the CPU pin
                    # exists to avoid and encode every image from garbage. A build that
                    # conclusively lacks the flag never gets here (the projector is
                    # dropped up front); an inconclusive probe still pins, since every
                    # build that can start here has had the flag since b5178. Deferred
                    # past the pass-through extras like the drafter pin, since
                    # --mmproj-offload is a real flag a user can pass and llama.cpp is
                    # last-wins.
                    if _paravirtual_cpu_forced and _paravirtual_mmproj_pinnable(server_caps):
                        _pv_mmproj_cpu_pin = ["--no-mmproj-offload"]
                        logger.warning(
                            "Disabling mmproj GPU offload: this Mac's Metal device "
                            "is virtualised."
                        )

                # Option C: --api-key for direct client access when enabled
                import secrets as _secrets

                if os.getenv("UNSLOTH_DIRECT_STREAM", "0") == "1":
                    self._api_key = _secrets.token_urlsafe(32)
                    cmd.extend(["--api-key", self._api_key])
                    logger.info("llama-server started with --api-key for direct streaming")
                else:
                    self._api_key = None

                # Windows + full offload: drop the host-RAM KV checkpoints that cause
                # WDDM/PCI-E overhead, but keep prompt caching (in-VRAM prefix reuse) so
                # a repeated prompt is not re-prefilled on every request. #5692.
                if sys.platform == "win32" and full_offload_tuning_active:
                    unsupported_cache_flags: list[str] = []
                    if server_caps.get("supports_cache_ram"):
                        cmd.extend(["--cache-ram", "0"])
                    else:
                        unsupported_cache_flags.append("--cache-ram")
                    if server_caps.get("supports_ctx_checkpoints"):
                        cmd.extend(["--ctx-checkpoints", "0"])
                    else:
                        unsupported_cache_flags.append("--ctx-checkpoints")
                    if unsupported_cache_flags:
                        logger.info(
                            "Skipping unsupported Windows cache flags for llama-server: %s",
                            ", ".join(unsupported_cache_flags),
                        )

                # Record the pin actually applied (fit-narrowed gpu_indices, else the raw
                # request) for the keep-warm loop, dedupe, and /status, so an explicit
                # [0, 1] narrowed to [0] records [0] and /status never echoes an ordinal
                # the child never saw. Auto selection (no gpu_ids) stays None (#7239).
                if _paravirtual_cpu_forced:
                    # --device none pins nothing, so record no effective pin here either.
                    # Ahead of the backend branches: this is the authoritative record and
                    # would otherwise re-echo the device the child never saw.
                    self._gpu_ids = None
                elif is_vulkan_backend:
                    # Only record an EXPLICIT Vulkan pin: an auto pick still narrows +
                    # pins below, but recording it would misreport an explicit pin and
                    # make dedupe miss the loaded server; mirrors the CUDA/ROCm branch.
                    self._gpu_ids = (
                        sorted(int(x) for x in _vulkan_pin_ids)
                        if (gpu_ids and _vulkan_pin_ids)
                        else None
                    )
                elif gpu_ids:
                    # Physical pin: the fit-selected subset when the fit ran, else the raw
                    # user selection so an explicit choice is honoured even when the fit
                    # could not size the model.
                    _effective_pin_ids = (
                        [int(x) for x in gpu_indices]
                        if gpu_indices is not None
                        else [int(x) for x in gpu_ids]
                    )
                    self._gpu_ids = (
                        sorted(int(x) for x in _effective_pin_ids) if _effective_pin_ids else None
                    )
                else:
                    self._gpu_ids = None

                # Also record the RAW requested pin (before the fit narrowed it). Load
                # dedupe compares this so a [0, 1] narrowed to [0] and re-sent as [0, 1]
                # still matches, while /status keeps echoing the effective pin (#7239).
                self._requested_gpu_ids = sorted(int(x) for x in gpu_ids) if gpu_ids else None

                if is_vulkan_backend and _vulkan_pin_ids is not None:
                    cmd += LlamaCppBackend._vulkan_pin_args(_vulkan_pin_ids)

                # User pass-through args go last. Placement flags are removed
                # below when the Studio picker owns the GPU selection.
                if extra_args:
                    _emit_extra_args = list(extra_args)
                    if gpu_ids is not None:
                        # gpu_ids owns placement, so remove competing device flags.
                        _before_device_strip = list(_emit_extra_args)
                        _emit_extra_args = self._strip_device_extra_args(_emit_extra_args)
                        if _emit_extra_args != _before_device_strip:
                            logger.info(
                                "Dropped user device placement flags from extra args: "
                                "explicit gpu_ids owns placement."
                            )
                    cmd.extend(str(a) for a in _emit_extra_args)
                    logger.info(
                        f"Appending user extra args to llama-server: {list(_emit_extra_args)}"
                    )

                # Last so it wins: the drafter CPU pin on a virtualised Metal device
                # is a correctness fix, not a preference, and llama.cpp is last-wins.
                if _pv_draft_cpu_pin:
                    cmd.extend(_pv_draft_cpu_pin)
                if _pv_mmproj_cpu_pin:
                    cmd.extend(_pv_mmproj_cpu_pin)
                # Also last: _zero_offload_keeps_gpu_visible reads the finished cmd, so
                # the override must be in it before the env block below judges this a
                # zero-VRAM server.
                if _pv_split_mode_pin:
                    cmd.extend(_pv_split_mode_pin)
                if _pv_device_pin:
                    cmd.extend(_pv_device_pin)

                kv_cache_unified = _kv_unified_from_args(cmd)

                logger.info(f"Starting llama-server: {' '.join(self._redacted_cmd_for_log(cmd))}")

                # Library paths so llama-server finds its shared libs and CUDA DLLs.
                env = self._llama_server_env_for_binary(binary)
                # Record placement values before planner reconciliation removes them.
                _inherited_main_placement_requested = any(
                    str(env.get(name) or "").strip()
                    for name in self._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS
                )
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(env)
                # Omitting --threads relies on llama.cpp's physical-core default, so
                # drop an inherited LLAMA_ARG_THREADS that would otherwise feed the
                # arg handler and silently force hardware_concurrency(). #5692
                if "--threads" not in cmd:
                    env.pop("LLAMA_ARG_THREADS", None)

                # Reconcile the inherited LLAMA_ARG_* env with Unsloth's final
                # decision: stripping CLI extras on a tensor->layer downgrade
                # can't remove env vars, so the child could run a mode/KV Unsloth
                # didn't budget.
                if not tensor_parallel:
                    # Layer split: clear a non-layer inherited split mode (and any
                    # paired tensor-split) so the child can't override the layer plan.
                    _inherited_sm = (env.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
                    if _inherited_sm and _inherited_sm != "layer":
                        env.pop("LLAMA_ARG_SPLIT_MODE", None)
                        env.pop("LLAMA_ARG_TENSOR_SPLIT", None)
                else:
                    # Unsloth owns the tensor split: it emits --tensor-split when it
                    # picks an uneven one (CLI wins) and nothing when an even split
                    # is safe. Clear any inherited LLAMA_ARG_TENSOR_SPLIT so the even
                    # case can't be overridden by a stale env (the layer branch above
                    # clears it too).
                    env.pop("LLAMA_ARG_TENSOR_SPLIT", None)
                    # Tensor split aborts on a quantized KV; clear an inherited
                    # quantized cache type so the child uses the tensor-safe default.
                    for _ct_var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"):
                        _ct_raw = (env.get(_ct_var) or "").strip().lower()
                        if _ct_raw and _ct_raw not in self._TENSOR_PARALLEL_KV_TYPES:
                            env.pop(_ct_var, None)

                # A gpu_ids pin also owns inherited device placement.
                if gpu_ids is not None:
                    env.pop("LLAMA_ARG_DEVICE", None)
                    env.pop("LLAMA_ARG_MAIN_GPU", None)

                # The unpinnable-drafter drop above rewrites argv, which cannot reach the
                # drafter the child picks up from its own env. The spec type goes too,
                # for the same reason the CLI one did: llama.cpp appends types rather
                # than replacing them, so an inherited draft-simple would outlive the
                # model it needs. Unsloth owns the spec block whenever the extras do not
                # name a --spec-type, and nothing _build_speculative_flags emits can undo
                # an inherited LLAMA_ARG_SPEC_TYPE (env applied first, then appended), so
                # a managed non-MTP launch would still start MTP, a crash-recovery replay
                # could not drop it, and the fit never budgeted the drafter the env adds.
                # Clearing it makes the emitted flags final. A user-owned --spec-type is
                # left alone: their flags and the env accumulate, which is what the slot
                # clamp above reads.
                if _pv_draft_unpinnable or not _extra_args_set_spec_type(extra_args):
                    for _pv_spec_var in _SPEC_ENV_VARS:
                        env.pop(_pv_spec_var, None)

                # The projector guard has a worse blind spot: it only clears Unsloth's
                # resolved path, so an inherited LLAMA_ARG_MMPROJ loads a projector the
                # guard believes it dropped, unpinned and independent of --gpu-layers 0.
                # LLAMA_ARG_MMPROJ_URL goes too: its download overwrites mmproj.path, so
                # it outranks even the --mmproj Unsloth emits. Unsloth always passes its
                # own projector on the command line (with --no-mmproj-offload), so an
                # inherited one is only ever the corrupt path.
                if _paravirtual_cpu_forced:
                    for _pv_mmproj_var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
                        env.pop(_pv_mmproj_var, None)
                    # LLAMA_ARG_OVERRIDE_TENSOR is the one placement var --gpu-layers 0
                    # cannot answer: applied during weight-buft selection, before any
                    # layer is assigned, so it puts weights back on the corrupt Metal
                    # buffer on its own. Same CPU-target rule as the argv strip, so an
                    # inherited exps=CPU override keeps working and only GPU-bound goes.
                    _pv_env_ot = env.get("LLAMA_ARG_OVERRIDE_TENSOR")
                    if _pv_env_ot and not _paravirtual_strip_gpu_overrides(
                        ["-ot", str(_pv_env_ot)]
                    ):
                        env.pop("LLAMA_ARG_OVERRIDE_TENSOR", None)

                # Windows + full offload: PASSIVE OMP + 2 threads stop
                # spin-wait burning CPU. CPU/partial offload keeps default
                # OMP parallelism. #5692.
                if sys.platform == "win32" and full_offload_tuning_active:
                    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
                    if not threads_overridden:
                        env.setdefault("OMP_NUM_THREADS", "2")

                # AMD unified-memory APUs (gfx1150/gfx1151): let llama.cpp use
                # shared system RAM. setdefault so a user value wins. Not on Vulkan
                # (nor DC below): gpu_indices are ggml ordinals, not CUDA/ROCm ids.
                if not is_vulkan_backend and self._amd_apu_wants_unified_memory(gpu_indices):
                    env.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
                    logger.info("AMD unified-memory APU: set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1")

                # DC NVIDIA GPUs: FP32 accum (+ P2P / launch queues for multi-GPU).
                # See _apply_datacenter_env; opt out with UNSLOTH_DISABLE_DC_TUNING=1.
                if not is_vulkan_backend and self._apply_datacenter_env(env, gpu_indices):
                    multi_gpu = self._effective_gpu_count(gpu_indices) > 1
                    logger.info(
                        f"Data-center GPU detected: applied DC llama.cpp env tuning (multi_gpu={multi_gpu})"
                    )

                # Pin to selected GPU(s) (issue #7164; resolved above into gpu_indices).
                # On ROCm, narrowing only CUDA_VISIBLE_DEVICES leaves the AMD child
                # seeing the full set, so set HIP_VISIBLE_DEVICES too. Vulkan is pinned
                # via --device (above), not here.
                # A deliberate zero-offload load with no GPU companions runs
                # entirely on CPU, yet a visible CUDA device still costs the child
                # ~0.5 GB (context + compute scratch) that the CPU-only
                # classification below reports as free. Hide the GPUs so the load
                # is exactly what it claims: zero VRAM (verified: GPU stays at idle
                # baseline and generation runs). Companion loads keep the normal
                # masking, and a user device pin (in extras or an inherited
                # LLAMA_ARG_DEVICE) keeps control of its own devices -- the child
                # aborts on a pin it can't see. The draft-device forms count too:
                # llama-server parses them even with no drafter loaded.
                _cpu_only_zero_offload = (
                    gpu_memory_mode == "manual"
                    and gpu_layers == 0
                    and not is_vulkan_backend
                    and not self._zero_offload_keeps_gpu_visible(cmd, env)
                )
                if _cpu_only_zero_offload:
                    self._emit_child_gpu_visibility(env, "-1")
                elif gpu_indices is not None and not is_vulkan_backend:
                    # When the user picked GPUs by index, align CUDA's ordering
                    # with the PCI-bus order the picker enumerated (nvidia-smi),
                    # so "GPU 1" in the UI is GPU 1 to llama.cpp -- not CUDA's
                    # default FASTEST_FIRST order (#5025).
                    if gpu_ids:
                        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    # Mask on AMD at the ROCr/HSA layer: HIP-only masking still
                    # enumerates every agent first, which segfaults on a deselected
                    # unsupported GPU (e.g. gfx1036 iGPU under a gfx103X prebuilt).
                    self._emit_child_gpu_visibility(
                        env, ",".join(str(i) for i in gpu_indices), prefer_rocr = True
                    )
                elif manual_tensor_split_emitted and not is_vulkan_backend:
                    # A manual per-GPU ratio across ALL GPUs (no explicit pick, so
                    # no CUDA_VISIBLE_DEVICES mask above): the UI built the
                    # --tensor-split list in ascending physical/PCI index order,
                    # so pin the child's enumeration to that order too. The whole
                    # visible set stays in use; only its ordering is fixed.
                    self._pin_visible_gpu_order_for_split(env)

                # Captured before any text-only fallback strips it from cmd.
                launched_with_mmproj = "--mmproj" in cmd

                # One-shot --fit off retry: recent llama.cpp runs a "fitting
                # params to device memory" step by default (--fit defaults to
                # 'on') even when -ngl is explicit. That step has aborted on
                # some ROCm hosts (ggml-cuda.cu ROCm error during worst-case
                # estimation, e.g. MTP + mmproj models on gfx1151). When
                # Unsloth's own VRAM math already placed the model
                # (use_fit=False), the step is redundant second-guessing --
                # retry once with --fit off before declaring the load failed.
                # Never retry when fit was requested (use_fit) or the caller
                # passed an explicit fit flag via extra args.
                # Argv actually launched (post --fit off / MTP); text-only retry strips this.
                _last_spawn_cmd = list(cmd)
                _spawn_cwd: Optional[str] = None

                def _spawn_and_wait(run_cmd, *, label = ""):
                    """Start llama-server with run_cmd and wait for health.

                    Retries once with --fit off when the first attempt
                    crashes during startup and run_cmd is eligible (see
                    _fit_off_retry_eligible).
                    """
                    nonlocal _last_spawn_cmd
                    _fit_retry_allowed = self._fit_off_retry_eligible(run_cmd, use_fit)
                    for _spawn_attempt in (0, 1):
                        # Defensive kill: drop an orphan Popen a concurrent load may
                        # have stored before we overwrite the reference (#5161).
                        # Also reaps the crashed first attempt on the retry pass.
                        self._kill_process()

                        self._stdout_lines = []
                        # Tee llama-server output to a dedicated log file so a
                        # post-mortem has the full trail even when the parent only
                        # kept the last 50 lines. Path is under the studio home.
                        # ``label`` (MTP fallback) and the attempt index (--fit
                        # off retry) keep a respawn within the same epoch second
                        # from truncating the crash log a retry warning just
                        # pointed the user at.
                        self._llama_log_fh = None
                        try:
                            log_dir = _swa_cache_path().parent / "logs" / "llama-server"
                            log_dir.mkdir(parents = True, exist_ok = True)
                            self._llama_log_path = log_dir / (
                                f"llama-{int(time.time())}{label}-port-{self._port}"
                                f"-try{_spawn_attempt}.log"
                            )
                            self._llama_log_fh = open(
                                self._llama_log_path,
                                "w",
                                encoding = "utf-8",
                                buffering = 1,
                            )
                            logger.info(f"llama-server stdout/stderr -> {self._llama_log_path}")
                        except (OSError, UnicodeDecodeError) as e:
                            # Best-effort; never block the load on logging.
                            logger.debug(f"Could not open llama-server log file: {e}")
                            self._llama_log_path = None
                        _last_spawn_cmd = list(run_cmd)
                        self._process = subprocess.Popen(
                            run_cmd,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.STDOUT,
                            text = True,
                            encoding = "utf-8",
                            errors = "replace",
                            env = env,
                            cwd = _spawn_cwd,
                            **_windows_hidden_subprocess_kwargs(),
                            **_child_popen_kwargs(),
                        )
                        self._record_server_pid(self._process.pid)

                        # Background thread to drain stdout (prevents pipe deadlock)
                        self._stdout_thread = threading.Thread(
                            target = self._drain_stdout, daemon = True, name = "llama-stdout"
                        )
                        self._stdout_thread.start()
                        if self._wait_for_health(timeout = 600.0):
                            return True
                        _startup_crashed = (
                            self._process.poll() is not None and self._process.returncode != 0
                        )
                        # A split-axis abort (#6415) is fit-independent: skip the
                        # --fit off retry and let the caller latch it.
                        _split_axis_crash = self._is_tensor_split_assert(
                            "\n".join(self._stdout_lines[-50:])
                        )
                        if (
                            _spawn_attempt == 0
                            and fully_gpu_offloaded
                            and _startup_crashed
                            and not _split_axis_crash
                        ):
                            # We forced --fit off because Unsloth's (conservative) VRAM
                            # math placed the model fully on GPU. A startup crash here
                            # means that estimate was optimistic, so fall back to --fit
                            # on and let llama.cpp offload rather than fail the load.
                            # DSpark takes this path too: fitting only skips its
                            # unmeasurable sidecar reserve, the drafter still loads.
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with forced --fit off; the fit estimate was optimistic, "
                                "retrying once with --fit on so it can offload. "
                                "Crash log: %s",
                                self._process.returncode,
                                self._llama_log_path,
                            )
                            # Flip Unsloth's own --fit off (added first, before any
                            # user extra args) to on; a user's later --fit still wins
                            # by last-arg. Defensive: if absent, the default is already
                            # --fit on, so leave it.
                            _run = list(run_cmd)
                            if "--fit" in _run:
                                _run[_run.index("--fit") + 1] = "on"
                            run_cmd = _run
                            continue
                        if (
                            _spawn_attempt == 0
                            and _fit_retry_allowed
                            and _startup_crashed
                            and not _split_axis_crash
                        ):
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with the default memory-fit step enabled; Unsloth "
                                "already verified the model fits, retrying once "
                                "with --fit off. Crash log: %s",
                                self._process.returncode,
                                self._llama_log_path,
                            )
                            run_cmd = [*run_cmd, "--fit", "off"]
                            continue
                        return False

                _requested_intent_for_cpu_fallback = intent
                _launched_mmproj_has_audio = self._mmproj_has_audio

                def _raise_terminal_load_failure(detail: str) -> NoReturn:
                    if intent.cpu_fallback:
                        self._cleanup_failed_cpu_fallback()
                    raise RuntimeError(detail)

                def _try_auto_vulkan_cpu_fallback(
                    failed_cmd: list[str],
                    failed_env: dict[str, str],
                    failed_rc: Optional[int],
                ) -> tuple[bool, bool, Optional[str]]:
                    """Replay a signal crash without devices.

                    CPU health confirms a GPU-init fault. Report a CPU failure directly.
                    """
                    nonlocal cmd, intent, gpu_memory_mode, gpu_layers, n_cpu_moe
                    nonlocal tensor_parallel, tensor_split, gpu_indices, use_fit, _spawn_cwd
                    if not self._is_signal_crash(failed_rc):
                        return False, False, None
                    if _inherited_main_placement_requested:
                        return False, False, None
                    if not self._auto_vulkan_cpu_fallback_eligible(
                        binary,
                        _requested_intent_for_cpu_fallback,
                        extra_args,
                        failed_env,
                    ):
                        return False, False, None
                    replay = self._cpu_isolated_replay(failed_cmd, failed_env, server_caps)
                    if replay is None:
                        return False, False, None
                    cpu_binary = self._cpu_isolated_binary(binary)
                    if cpu_binary is None:
                        return False, False, None
                    replay[0] = cpu_binary
                    _spawn_cwd = str(Path(cpu_binary).parent)

                    logger.warning(
                        "The auto-selected Vulkan backend hard-crashed during "
                        "startup; retrying once with llama.cpp devices disabled."
                    )
                    if not _spawn_and_wait(replay, label = "-cpu"):
                        cpu_rc = self._process.poll() if self._process is not None else None
                        detail = self._classify_llama_start_failure(
                            "\n".join(self._stdout_lines[-50:]),
                            gguf_path,
                            self._model_identifier,
                            cpu_rc,
                            binary,
                        )
                        self._cleanup_failed_cpu_fallback()
                        return True, False, detail

                    cmd = replay
                    gpu_memory_mode = "manual"
                    gpu_layers = 0
                    n_cpu_moe = 0
                    tensor_parallel = False
                    tensor_split = None
                    gpu_indices = None
                    use_fit = False
                    intent = replace(
                        intent,
                        gpu_memory_mode = "manual",
                        gpu_layers = 0,
                        n_cpu_moe = 0,
                        tensor_parallel = False,
                        tensor_split = None,
                        gpu_ids = None,
                        cpu_fallback = True,
                    )
                    self._gpu_memory_mode = "manual"
                    self._gpu_layers = 0
                    self._n_cpu_moe = 0
                    self._tensor_split = None
                    self._tensor_parallel = False
                    self._gpu_ids = None
                    self._requested_gpu_ids = None
                    self._layer_preserves_tensor_intent = False
                    self._is_vision = launched_with_mmproj
                    self._mmproj_has_audio = _launched_mmproj_has_audio
                    self._cpu_fallback_reason = "vulkan_startup_crash"
                    logger.warning(
                        "llama-server loaded successfully on CPU after the "
                        "auto-selected Vulkan backend crashed. GPU acceleration "
                        "is disabled for this model session."
                    )
                    return True, True, None

                # Store the resolved on-disk path, not the caller's kwarg: in
                # HF mode gguf_path is None and ``model_path`` is what
                # llama-server mmap's, which downstream consumers need. Must be
                # set BEFORE the spawn: load_progress() reads _gguf_path for
                # the mmap progress total while the health wait runs.
                self._gguf_path = model_path
                self._hf_repo = hf_repo
                self._mtp_draft_path = launch_mtp_draft_path
                self._mtp_draft_suppressed_path = _pv_suppressed_draft_path
                # For local GGUF files, extract variant from filename if absent
                if hf_variant:
                    self._hf_variant = hf_variant
                elif gguf_path:
                    try:
                        from utils.models.model_config import _extract_quant_label
                        self._hf_variant = _extract_quant_label(gguf_path)
                    except Exception:
                        self._hf_variant = None
                else:
                    self._hf_variant = None
                self._is_vision = effective_is_vision
                self._model_identifier = model_identifier

                # Store the effective (possibly capped) context separately; do
                # NOT overwrite _context_length (the native length for display).
                self._effective_context_length = (
                    effective_ctx if effective_ctx > 0 else self._context_length
                )
                self._max_context_length = (
                    max_available_ctx if max_available_ctx > 0 else self._effective_context_length
                )

                if intent.cpu_fallback:
                    replay = self._cpu_isolated_replay(cmd, env, server_caps)
                    cpu_binary = self._cpu_isolated_binary(binary)
                    if replay is None or cpu_binary is None:
                        _raise_terminal_load_failure(
                            "The prior Vulkan CPU fallback could not be reconstructed; "
                            "run `unsloth studio update` to repair the managed llama.cpp runtime."
                        )
                    replay[0] = cpu_binary
                    _spawn_cwd = str(Path(cpu_binary).parent)
                    cmd = replay
                    self._cpu_fallback_reason = "vulkan_startup_crash"
                healthy = _spawn_and_wait(cmd)
                # #6415 split-mode tensor warmup abort. Latch it on THIS first spawn:
                # the flash-attn-off retry below can't run tensor (needs flash_attn),
                # so its output drops the marker and recording later would miss it,
                # looping every load. Record and raise to the route's layer fallback,
                # skipping the futile flash-attn/MTP retries.
                if not healthy and self._tensor_parallel and not self._cancel_event.is_set():
                    _ts_out = "\n".join(self._stdout_lines[-50:])
                    _ts_rc = self._process.poll() if self._process is not None else None
                    if self._should_record_tensor_split_abort(_ts_rc, _ts_out):
                        LlamaCppBackend._record_tensor_split_abort(binary, model_identifier)
                        self._kill_process()
                        raise RuntimeError(
                            "llama-server aborted on --split-mode tensor "
                            "(split-axis geometry); retrying with layer split."
                        )
                # Flash-attention kernels hard-crash at startup on some ROCm/GPU
                # builds (frequently inside the vision tower). Disabling FA keeps
                # both vision and MTP, so retry that way before dropping either.
                # Only on a hard fault with FA on; a cancel/unload stops respawn.
                if not healthy and not self._cancel_event.is_set():
                    _fa_rc = self._process.poll() if self._process is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(_last_spawn_cmd)
                        if self._is_signal_crash(_fa_rc)
                        else None
                    )
                    if _fa_cmd is not None:
                        logger.warning(
                            "llama-server hard-crashed at startup (exit %s) with "
                            "flash attention on; retrying once with --flash-attn "
                            "off (keeps vision and MTP).",
                            _fa_rc,
                        )
                        self._kill_process()
                        # The argv rewrite can't reach an env-only quantized V
                        # cache; drop it so the FA-off child doesn't abort on it.
                        if self._drop_env_quantized_v_cache(env):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        cmd = _fa_cmd
                        healthy = _spawn_and_wait(_fa_cmd, label = "-noflash")

                # MTP from Unsloth's spec flags or the user's (extra_args
                # --spec-type / LLAMA_ARG_SPEC_TYPE). The env survives to the child only
                # when the extras own --spec-type (the launch scrubs it otherwise), so
                # anything else would judge a server it is not starting.
                _launch_spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                _spec_requested_mtp = any(
                    "mtp" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_mtp(extra_args, env = _launch_spec_env)
                _spec_requested_dspark = any(
                    "draft-dspark" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_dspark(extra_args, env = _launch_spec_env)
                # Is the launched server actually running MTP+tensor? Gates the
                # probe/watchdog/recovery; cleared if the MTP-drop fallback wins.
                _mtp_active_for_launched_server = bool(
                    self._tensor_parallel and _spec_requested_mtp
                )
                # MTP can pass /health then crash the flash-attn kernel on the
                # first decode under tensor; probe one generation so the fallback
                # catches that too. Tensor-only, so ordinary MTP stays probe-free.
                if (
                    healthy
                    and self._tensor_parallel
                    and _spec_requested_mtp
                    and not self._cancel_event.is_set()
                    and not self._probe_mtp_decode()
                ):
                    # A first-decode hard fault is usually the FA kernel: retry
                    # FA-off (keeps MTP) before dropping speculative decoding below.
                    _probe_rc = self._process.poll() if self._process is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(_last_spawn_cmd)
                        if self._is_signal_crash(_probe_rc)
                        else None
                    )
                    healthy = False
                    if _fa_cmd is not None:
                        logger.warning(
                            "MTP first-decode hard-crashed (exit %s) with flash "
                            "attention on; retrying with --flash-attn off.",
                            _probe_rc,
                        )
                        self._kill_process()
                        # The argv rewrite can't reach an env-only quantized V
                        # cache; drop it so the FA-off child doesn't abort on it.
                        if self._drop_env_quantized_v_cache(env):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        cmd = _fa_cmd
                        healthy = (
                            _spawn_and_wait(_fa_cmd, label = "-noflash-mtp")
                            and self._probe_mtp_decode()
                        )
                    if not healthy:
                        logger.warning(
                            "MTP speculative decoding crashed on the first decode "
                            "under tensor parallelism; retrying without it."
                        )
                # Any MTP request can abort the server: a separate drafter
                # (Gemma) on a binary that predates its arch, or an embedded
                # head (Qwen) the binary cannot build. Retry once with the
                # spec slice replaced by --spec-default so the main model still
                # loads. Gate on the spec block (not the drafter path, which
                # off/ngram local loads also carry) and keep
                # _requested_spec_mode so a duplicate /load doesn't thrash. The
                # cancel check stops an /unload-killed attempt respawning. A
                # decode-probe failure above also routes here.
                if (
                    not healthy
                    and (_spec_requested_mtp or _spec_requested_dspark)
                    and not self._cancel_event.is_set()
                ):
                    # Blame the binary only when the output shows MTP itself
                    # failing (unknown arch / draft or context build); an
                    # unrelated crash (e.g. OOM) gets a neutral message.
                    _lo = "\n".join(self._stdout_lines).lower()
                    # Only an unknown architecture proves the prebuilt predates
                    # this MTP model (an update fixes it). The memory/context
                    # build failures are generic (VRAM / ctx pressure), where an
                    # update may not help, so classify those as runtime_error.
                    _arch_unsupported = "unknown model architecture" in _lo
                    if (
                        _arch_unsupported
                        or "failed to measure draft model memory" in _lo
                        or "failed to measure mtp context memory" in _lo
                        or "failed to create llama_context" in _lo
                    ):
                        _retry_reason = (
                            "the prebuilt may predate it; retrying without "
                            "speculative decoding -- run `unsloth studio "
                            "update` for this drafter"
                        )
                        self._spec_fallback_reason = (
                            "binary_outdated" if _arch_unsupported else "runtime_error"
                        )
                    else:
                        _retry_reason = (
                            "retrying without speculative decoding in case the drafter is the cause"
                        )
                        self._spec_fallback_reason = "runtime_error"
                    _drafter = (
                        Path(launch_mtp_draft_path).name
                        if launch_mtp_draft_path
                        else "embedded head"
                    )
                    logger.warning(
                        "llama-server failed to start with speculative drafter (%s); %s.",
                        _drafter,
                        _retry_reason,
                    )
                    self._kill_process()
                    _fb_tail = cmd[_spec_start + len(spec_flags) :]
                    _fb_stripped_extras: Optional[List[str]] = None
                    # Appending --spec-default cannot clear MTP the extras or the env
                    # carry: types accumulate, so it would retry the mode that just
                    # failed and lose a main model that loads fine without it. The
                    # tail loses its spec group and the env goes with it, the same way
                    # the crash replay does it.
                    if _extra_args_requests_mtp(
                        extra_args, env = _launch_spec_env
                    ) or _extra_args_requests_dspark(extra_args, env = _launch_spec_env):
                        _fb_tail = strip_shadowing_flags(
                            _fb_tail,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                        for _fb_spec_var in _SPEC_ENV_VARS:
                            env.pop(_fb_spec_var, None)
                        _fb_stripped_extras = strip_shadowing_flags(
                            extra_args or [],
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                        # The extras owned --spec-type, so the clamp that took the slots
                        # was the extras one and _mtp_clamped_slots stayed 0. The strip
                        # above makes this retry genuinely non-MTP, so hand those back
                        # too, or --parallel 1 sticks and the requested-vs-requested
                        # dedupe keeps every later load on the same one-slot server.
                        # Only where no GPU had to admit them: that clamp runs before the
                        # slot fit, so on a GPU box the count was never budgeted and the
                        # retry would size buffers nothing approved.
                        if not _detected_gpus:
                            _mtp_clamped_slots = max(_mtp_clamped_slots, _pv_extras_clamped_slots)
                    fallback_cmd = cmd[:_spec_start] + ["--spec-default"] + _fb_tail
                    # fallback_cmd inherits the MTP slot clamp, but this retry is not
                    # MTP: hand back the slots the KV fit was sized for, or --parallel N
                    # silently serves one chat at a time (and the requested-vs-requested
                    # dedupe makes that stick). Later retries derive from this argv, so
                    # rebind n_parallel now.
                    if _mtp_clamped_slots > 1 and "--parallel" in fallback_cmd:
                        fallback_cmd[fallback_cmd.index("--parallel") + 1] = str(_mtp_clamped_slots)
                        n_parallel = _mtp_clamped_slots
                    healthy = _spawn_and_wait(fallback_cmd, label = "-retry")
                    if healthy:
                        self._speculative_type = "default"
                        _mtp_active_for_launched_server = False
                        # The record below stores extra_args as what LAUNCHED, but this
                        # server launched without the spec group. Leaving the MTP list
                        # there would have the next inheriting Apply pick it back up and
                        # repeat the crash. Keep it as the requested state only, the way
                        # the runtime-crash replay does.
                        if _fb_stripped_extras is not None:
                            _pv_suppressed_spec_extra_args = (
                                _pv_suppressed_spec_extra_args
                                if _pv_suppressed_spec_extra_args is not None
                                else list(extra_args or [])
                            )
                            extra_args = _fb_stripped_extras

                # A too-old llama.cpp can reject a model's --mmproj projector
                # (format message or a bare SIGSEGV); retry once text-only.
                if not healthy:
                    out = "\n".join(self._stdout_lines[-50:])
                    # Read the crash code before _kill_process() clears _process.
                    _crash_rc = self._process.poll() if self._process is not None else None
                    self._kill_process()
                    # The #6415 split-axis abort is latched earlier (first spawn).
                    # Skip if a cancel/unload is pending (mirrors the MTP guard).
                    _projector_msg = self._is_projector_incompatibility(out)
                    _signal_mmproj_guess = self._is_signal_crash(
                        _crash_rc
                    ) and not self._output_has_nonprojector_diagnostic(out)
                    if (
                        launched_with_mmproj
                        and not self._cancel_event.is_set()
                        and (_projector_msg or _signal_mmproj_guess)
                    ):
                        _vision_cpu_replay_cmd = list(_last_spawn_cmd)
                        if _projector_msg:
                            logger.warning(
                                "llama-server could not load this model's vision "
                                "projector (--mmproj). The installed llama.cpp build is "
                                "likely too old for it. Loading text-only for this "
                                "session; run 'unsloth studio update' to enable vision."
                            )
                        else:
                            logger.warning(
                                "llama-server crashed while loading this model's vision "
                                "projector (--mmproj). Retrying text-only for this "
                                "session; if this persists, run 'unsloth studio update' "
                                "or check GPU/driver logs."
                            )
                        cmd = self._strip_mmproj_args(_last_spawn_cmd)
                        # This retry bypasses _spawn_and_wait, so refresh the
                        # launched-argv snapshot itself -- the zero-offload
                        # classification below must not see the stripped --mmproj.
                        _last_spawn_cmd = list(cmd)
                        self._is_vision = False
                        self._mmproj_has_audio = False
                        self._start_llama_process(cmd, env)
                        if not self._wait_for_health(timeout = 600.0):
                            # Read the exit code before _kill_process() clears it, so
                            # an OS-killed text-only retry still gets the OOM message.
                            _retry_rc = self._process.poll() if self._process is not None else None
                            self._kill_process()
                            # A confirmed projector error is not evidence of a GPU fault.
                            if self._is_signal_crash(_retry_rc) and not _projector_msg:
                                _cpu_attempted, _cpu_healthy, _cpu_detail = (
                                    _try_auto_vulkan_cpu_fallback(
                                        _vision_cpu_replay_cmd,
                                        env,
                                        _retry_rc,
                                    )
                                )
                                if _cpu_healthy:
                                    healthy = True
                                elif _cpu_attempted:
                                    _raise_terminal_load_failure(str(_cpu_detail))
                                else:
                                    _raise_terminal_load_failure(
                                        self._gpu_init_crash_message(binary)
                                    )
                            if healthy:
                                pass
                            else:
                                _retry_detail = self._classify_llama_start_failure(
                                    "\n".join(self._stdout_lines[-50:]),
                                    gguf_path,
                                    self._model_identifier,
                                    _retry_rc,
                                    binary,
                                )
                                _raise_terminal_load_failure(
                                    self._mmproj_retry_failure_message(
                                        projector_confirmed = _projector_msg,
                                        detail = _retry_detail,
                                    )
                                )
                    else:
                        _cpu_attempted, _cpu_healthy, _cpu_detail = (
                            _try_auto_vulkan_cpu_fallback(_last_spawn_cmd, env, _crash_rc)
                        )
                        if _cpu_healthy:
                            healthy = True
                        elif _cpu_attempted:
                            _raise_terminal_load_failure(str(_cpu_detail))
                        else:
                            _raise_terminal_load_failure(
                                self._classify_llama_start_failure(
                                    out,
                                    gguf_path,
                                    self._model_identifier,
                                    _crash_rc,
                                    binary,
                                )
                            )

                self._healthy = True
                self._commit_effective_parallel_slots(n_parallel)
                self._swa_full = swa_full
                self._kv_cache_unified = kv_cache_unified
                self._n_ubatch = max(
                    0,
                    int(self._DEFAULT_N_UBATCH if _effective_ubatch is None else _effective_ubatch),
                )
                self._flash_attn_enabled = (
                    _flash_attn_enabled_from_args(_last_spawn_cmd, env = env)
                    and self._architecture != "grok"
                )
                self._effective_cache_types = _effective_main_cache_types(
                    _last_spawn_cmd,
                    env,
                )
                self._kv_cache_context_total = effective_ctx if effective_ctx > 0 else None

                # Server is up: adopt the real per-request context it allocated
                # -- the length --fit chose, or a --parallel slot split -- so the
                # reported context_length matches reality. (Querying /props
                # before the spawn above always failed; the seeded value was the
                # requested/native length.)
                self._reconcile_effective_ctx_with_server()
                if self._kv_cache_context_total is not None:
                    self._n_ubatch = min(
                        self._n_ubatch,
                        self._kv_cache_context_total,
                    )

                # Commit caller intent only after _healthy=True so a failed start
                # can't poison the next inheritance check. None keeps prior, []
                # clears, list sets. Source records hf_variant for the route's
                # same_source check.
                if extra_args is not None:
                    # Persist the same device-stripped extras the command used when gpu_ids
                    # owns placement, so a later inheriting reload (after gpu_ids clears)
                    # can't resurrect the dropped --device (#7188).
                    self._extra_args = (
                        self._strip_device_extra_args(extra_args)
                        if gpu_ids is not None
                        else list(extra_args)
                    )
                    # Device-stripped the same way, so both comparator sides share a rule.
                    _pv_requested = (
                        _pv_suppressed_spec_extra_args
                        if _pv_suppressed_spec_extra_args is not None
                        else extra_args
                    )
                    self._requested_extra_args = (
                        self._strip_device_extra_args(_pv_requested)
                        if gpu_ids is not None
                        else list(_pv_requested)
                    )
                    self._extra_args_source = (model_identifier, hf_variant)
                self._requested_n_ctx = int(n_ctx)
                # Local n_parallel may have been reduced above; the snapshot has the ask.
                self._requested_n_parallel = max(1, int(intent.n_parallel))
                # Commit the known-good snapshot + whether MTP+tensor is live, then
                # watch this load for a mid-generation crash.
                self._last_load_intent = intent
                self._mtp_runtime_fallback_active = _mtp_active_for_launched_server
                self._start_mtp_crash_watchdog()

                # Catch silent CPU fallback when GPU was intended (#5106). Manual
                # offload (no picker) leaves gpu_indices None and use_fit False, so
                # include its GPU-layer intent; use the preserved probe since
                # auto-layers/manual empty `gpus`. A deliberate zero-offload load
                # classifies by its launched argv instead: the main model is
                # CPU-only by construction and must read False (not None), or
                # training needlessly unloads a server holding no VRAM.
                _deliberate_cpu_only = gpu_memory_mode == "manual" and gpu_layers == 0
                if _deliberate_cpu_only:
                    self._gpu_offload_active = self._zero_offload_gpu_flag(
                        _last_spawn_cmd, _detected_gpus, env
                    )
                else:
                    self._gpu_offload_active = self._classify_gpu_offload(
                        gpu_indices is not None or use_fit or gpu_memory_mode == "manual",
                        _detected_gpus,
                    )
                if self._gpu_offload_active is False and not _deliberate_cpu_only:
                    logger.warning(
                        "llama-server appears to have loaded the model entirely "
                        "on CPU even though Unsloth detected at least one GPU. "
                        "This usually means the prebuilt binary's GPU backend "
                        "failed to load -- on Windows, cudart64_X.dll / "
                        "cublas64_X.dll could not be resolved. Reinstall the "
                        "Unsloth llama.cpp prebuilt or install a matching CUDA "
                        "toolkit (issue unslothai/unsloth#5106).",
                    )

                logger.info(
                    f"llama-server ready on port {self._port} for model '{model_identifier}'"
                )
                # Poll llama-server /metrics -> vLLM-style engine_stats logs
                # (only when the binary exposes /metrics).
                if server_caps.get("supports_metrics"):
                    try:
                        from core.inference.llama_stats import maybe_start_stats_logger
                        if self._stats_logger is not None:
                            self._stats_logger.stop()
                        self._stats_logger = maybe_start_stats_logger(self.base_url, logger)
                    except Exception as e:
                        logger.debug(f"engine-stats logger not started: {e}")
                else:
                    self._stats_logger = None

            # Probe outside _lock (interruptible by /unload); init inside.
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            try:
                detected = self._detect_audio_type_strict()
                self._audio_probed = True
            except Exception as exc:
                logger.debug("Audio probe failed: %s", exc)
                detected = None
            if not self._apply_detected_audio(detected):
                return False

            if not self._healthy:
                return False
            # Snapshot the files the server actually loaded. If a GGUF shard or a
            # LoRA/control-vector sidecar is swapped on disk afterwards while the
            # old weights stay mapped, save_slots_for_resume() compares against
            # this and refuses to persist KV that a reload could misapply.
            if self._slot_save_dir:
                self._slot_loaded_identity = (
                    self._gguf_file_identity(self._gguf_path),
                    self._slot_launch_fingerprint(),
                )
            return True

    def _build_speculative_flags(
        self,
        *,
        speculative_type: Optional[str],
        spec_draft_n_max: Optional[int],
        extra_args: Optional[List[str]],
        model_identifier: str,
        model_path: Optional[str],
        gpus: bool,
        binary: Optional[str],
        mtp_draft_path: Optional[str] = None,
        dspark_draft_path: Optional[str] = None,
        dspark_fit_sized: bool = True,
        draft_device: Optional[str] = None,
    ) -> List[str]:
        """Return the llama-server flag list for the requested spec mode.

        Side effects: sets ``self._speculative_type`` (resolved internal
        emit), ``self._requested_spec_mode`` (canonical UI mode for the
        status round-trip), and ``self._spec_draft_n_max`` (user override
        only; None when the platform default applies).

        Speculative decoding (n-gram self-speculation, zero VRAM):
        ngram-mod uses a ~16 MB shared hash pool, constant memory /
        complexity, variable draft lengths. Helps most when the model
        repeats existing text (code refactor, summarisation, reasoning);
        for low-repetition chat, overhead is ~5 ms.

        Benchmarks from upstream llama.cpp speculative-decoding PRs:
          Scenario                        | Without | With    | Speedup
          gpt-oss-120b code refactor      | 181 t/s | 446 t/s | 2.5x
          Qwen3-235B offloaded            |  12 t/s |  21 t/s | 1.8x
          gpt-oss-120b repeat (92% accept)| 181 t/s | 814 t/s | 4.5x
        Refs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
              https://github.com/ggml-org/llama.cpp/pull/19164
              https://github.com/ggml-org/llama.cpp/pull/18471
              MTP guide: unsloth.ai/docs/models/qwen3.6#mtp-guide

        Sub-3B dense MTP regresses vs spec-off when the head is baked into the
        main GGUF (Qwen): the draft head's per-token cost exceeds the
        acceptance savings at this scale. Q4_K_XL clean bench (each prompt once
        after an unrelated warmup) on B200 + x86 CPU:
          0.8B GPU: draft-mtp n=2 = 0.58x vs OFF; ngram-only = 1.10x
          2B   GPU: draft-mtp n=2 = 0.82x vs OFF; OFF or ngram = 1.00x
          0.8B CPU: chained n=2   = 0.86x vs OFF; ngram-only = 1.19x
          2B   CPU: chained n=2   = 0.83x vs OFF; ngram-only = 1.01x
          4B+ GPU/CPU: spec on is a net win (1.08x-1.46x).
        A separate drafter (Gemma's root mtp-*.gguf) is a different, cheaper
        mechanism that wins even below 3B, so it is exempt from the sub-3B drop
        (``mtp_draft_path`` set -> not too small). B200 Q4_K_XL bench, draft-mtp
        n=2 vs OFF: gemma-4-E2B (2B) = 1.21x, accept ~0.65 (vs ngram = 1.00x);
        gemma-4-E4B (4B) and 12B engage as usual.
        Auto falls back to ngram-mod (zero-VRAM, near-zero idle cost on
        diverse content) for an embedded sub-3B head; forced MTP on a model
        with no head/drafter defaults back (mtp -> spec-default, mtp+ngram ->
        ngram-mod) since llama-server aborts otherwise; a drafter the binary
        cannot build (older prebuilt, or a CUDA kernel limit) aborts the spawn
        and the load retries once without speculative decoding.
        """
        flags: List[str] = []
        # Reset; emit branches re-set on the resolved emission.
        self._spec_draft_n_max = None
        self._speculative_type = None
        self._spec_fallback_reason = None

        # Canonical UI-facing requested mode (legacy values mapped via
        # _canonicalize_spec_mode).
        canonical_mode = _canonicalize_spec_mode(speculative_type)
        # MTP signals: head baked into the main GGUF (Qwen, via metadata or
        # name), or a separate drafter resolved from the repo (Gemma).
        is_mtp_model = (
            bool(self._nextn_predict_layers)
            or _is_mtp_model_name(model_identifier, model_path)
            or bool(mtp_draft_path)
        )
        user_owns_spec_type = _extra_args_set_spec_type(extra_args)
        _mtp_size_b = _extract_model_size_b(model_identifier)
        # The sub-3B regression is an embedded-head cost; a separate drafter
        # (Gemma) is a cheap standalone model that wins below 3B, so exempt it.
        _mtp_too_small = (
            _mtp_size_b is not None and _mtp_size_b < _MTP_MIN_SIZE_B and not bool(mtp_draft_path)
        )
        # Drafterless Gemma (name-only MTP, no embedded head): emitting MTP
        # would abort llama-server, so every mode below falls back instead.
        _mtp_drafter_missing = (
            _is_gemma_mtp_name(model_identifier, model_path)
            and not mtp_draft_path
            and not self._nextn_predict_layers
        )
        # Embedded MTP head on an MLA model (GLM-5.2/DeepSeek/Kimi, detected by
        # kv_lora_rank): llama.cpp's MLA/DSA MTP path is ~2x slower than no spec,
        # so Auto drops it (override via the Settings dropdown / forced mtp, or
        # UNSLOTH_MLA_MTP_ENABLED=1). Separate drafters (Gemma, mtp_draft_path) and
        # non-MLA embedded heads (Qwen, no kv_lora_rank) are unaffected.
        _auto_mla_embedded_mtp = (
            bool(self._nextn_predict_layers)
            and self._kv_lora_rank is not None
            and not bool(mtp_draft_path)
            and not _mla_mtp_auto_enabled()
        )

        if user_owns_spec_type:
            # User --spec-type wins outright; suppress auto-emit to avoid a
            # duplicate spec block.
            self._requested_spec_mode = None
            return flags

        effective_mode = canonical_mode or "auto"
        self._requested_spec_mode = effective_mode
        if effective_mode == "off":
            return flags
        if effective_mode == "ngram-simple":
            flags.extend(["--spec-type", "ngram-simple"])
            self._speculative_type = "ngram-simple"
            return flags
        caps = self.probe_server_capabilities(binary)

        # Which drafter every branch below is about. Set once here, before any
        # fallback can erase the evidence: the UI names the recovery from this,
        # and under Auto neither the requested mode nor the resolved
        # _speculative_type ("default" after a fallback) still carries the kind.
        self._spec_drafter_kind = (
            "dspark"
            if canonical_mode == "dspark"
            or (
                (canonical_mode or "auto") == "auto"
                and dspark_draft_path
                and caps.get("supports_dspark")
            )
            else "mtp"
        )

        def _resolved_draft_n_max() -> int:
            # User override wins; else platform default (the B200 / x86
            # clean-sweep sweet spot from PR #5582 is n=2 GPU, n=3 CPU;
            # past 3 regresses on essay-style low-acceptance prompts).
            if spec_draft_n_max is not None:
                n = int(spec_draft_n_max)
                self._spec_draft_n_max = n
                return n
            return 2 if gpus else 3

        def _emit_dspark() -> None:
            """Append --model-draft <sidecar> --spec-type draft-dspark + n-max.

            Callers have already established the sidecar and the capability; Auto
            reaches this too, since DSpark is the default when both hold.
            """
            if not dspark_fit_sized:
                # Not a blocker: the sidecar ships no token_embd/output and borrows
                # the target's, so llama.cpp's fit step cannot build a standalone
                # draft context to measure it (llama-context.cpp:153-160) and skips
                # the reserve (server-context.cpp:1190-1193). The load still works;
                # only the ~11 GB is missing from the fit budget.
                logger.info(
                    "DSpark under --fit on: llama.cpp cannot size the sidecar during "
                    "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
                    "Manual mode if the load runs out of VRAM.",
                    (self._get_gguf_size_bytes(dspark_draft_path) or 0) / 1e9,
                )
            draft_n_max = int(spec_draft_n_max) if spec_draft_n_max is not None else 3
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = draft_n_max
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    dspark_draft_path,
                    "--spec-type",
                    "draft-dspark",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            self._speculative_type = "draft-dspark"
            logger.info("Spec decoding: draft-dspark using %s", dspark_draft_path)

        if effective_mode == "dspark":
            # Capability first: the fetch is gated on the same answer, so a missing
            # sidecar here is usually the binary's fault. Blaming the file instead
            # would tell the user to place one, and reload on every Apply.
            if not caps.get("supports_dspark"):
                logger.warning(
                    "DSpark requested but llama-server lacks --spec-type "
                    "draft-dspark; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not dspark_draft_path:
                logger.warning(
                    "DSpark requested but no matching dspark-*.gguf sidecar was found; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            _emit_dspark()
            return flags

        def _emit_mtp(*, chain_ngram: bool) -> bool:
            """Append --spec-type mtp[/draft-mtp][,ngram-mod] + n-max."""
            mtp_token = caps.get("mtp_token") if caps else None
            if not mtp_token:
                inconclusive = bool(caps.get("mtp_probe_inconclusive")) if caps else True
                if inconclusive:
                    logger.info(
                        "Requested MTP speculative decoding but llama-server MTP "
                        "capability probe was inconclusive; loading without "
                        "speculative decoding."
                    )
                else:
                    logger.warning(
                        "Requested MTP speculative decoding but "
                        "llama-server lacks --spec-type mtp/draft-mtp; "
                        "run `unsloth studio update`. Loading without "
                        "speculative decoding."
                    )
                # Override an inherited LLAMA_ARG_SPEC_TYPE=draft-mtp (CLI wins
                # over env) so the child matches the binary-capability gate and
                # the no-MTP budget, like the sibling no-head/non-MTP fallbacks.
                flags.append("--spec-default")
                self._speculative_type = "default"
                if inconclusive:
                    self._spec_fallback_reason = None
                else:
                    self._spec_fallback_reason = "binary_no_mtp"
                return False
            draft_n_max = _resolved_draft_n_max()
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            # Separate-file drafter (Gemma): point llama-server at it. Baked-in
            # heads (Qwen) pass no path -- llama-server reads them from the
            # main GGUF.
            if mtp_draft_path:
                flags.extend(["--model-draft", mtp_draft_path])
                if draft_device:
                    flags.extend(["--spec-draft-device", draft_device])
                logger.info(f"Using separate MTP drafter: {mtp_draft_path}")
            spec_value = mtp_token
            ngram_knobs: list[str] = []
            if chain_ngram:
                ngram_knobs = _build_ngram_mod_flags(caps)
                if ngram_knobs:
                    spec_value = f"ngram-mod,{mtp_token}"
                else:
                    logger.warning(
                        "llama-server lacks ngram-mod tuning "
                        "flags; loading MTP only (no ngram chain)"
                    )
            flags.extend(["--spec-type", spec_value, n_max_flag, str(draft_n_max)])
            flags.extend(ngram_knobs)
            self._speculative_type = "draft-mtp"
            chain_label = "chained ngram-mod" if chain_ngram else "MTP-only"
            logger.info(f"Spec decoding: {mtp_token} ({chain_label})")
            return True

        def _emit_ngram_mod() -> bool:
            """Append --spec-type ngram-mod + flag-set knobs."""
            ngram_knobs = _build_ngram_mod_flags(caps)
            flags.extend(["--spec-type", "ngram-mod"])
            if not ngram_knobs:
                logger.warning(
                    "llama-server lacks ngram-mod tuning "
                    "flags; loading without --spec-ngram-mod-* knobs"
                )
            flags.extend(ngram_knobs)
            self._speculative_type = "ngram-mod"
            logger.info("Spec decoding: ngram-mod")
            return True

        def _fallback_drafter_not_found() -> None:
            """Drafterless Gemma: use ngram-mod (or spec-default) and record why."""
            logger.warning(
                "Model %s is MTP-capable but no drafter or head was found; "
                "falling back. Check network or run `unsloth studio update`.",
                model_identifier,
            )
            if caps.get("supports_ngram_mod"):
                _emit_ngram_mod()
            else:
                flags.append("--spec-default")
                self._speculative_type = "default"
            self._spec_fallback_reason = "drafter_not_found"

        if effective_mode == "ngram":
            _emit_ngram_mod()
            return flags
        if effective_mode in ("mtp", "mtp+ngram"):
            chain_ngram = effective_mode == "mtp+ngram"
            if not is_mtp_model:
                if chain_ngram:
                    logger.warning(
                        "MTP+Ngram requested but this GGUF has no MTP head or "
                        "drafter; loading ngram-mod only."
                    )
                    _emit_ngram_mod()
                else:
                    logger.warning(
                        "MTP requested but this GGUF has no MTP head or drafter; "
                        "loading without speculative decoding."
                    )
                    flags.append("--spec-default")
                    self._speculative_type = "default"
                return flags
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
                return flags
            if _mtp_too_small:
                logger.warning(
                    f"Forcing {effective_mode.upper()} on a {_mtp_size_b:.1f}B model; "
                    "the bench shows MTP regresses below 3B. "
                    "Engaging anyway (user override)."
                )
            _emit_mtp(chain_ngram = chain_ngram)
            return flags

        # effective_mode == "auto": the promotion path. llama.cpp #22673:
        # MTP is compatible with mmproj, so there's no vision gate.
        if dspark_draft_path and caps.get("supports_dspark"):
            # DSpark first: load_model only hands a sidecar down once it has one
            # this binary can launch, and it beats every other Auto outcome for
            # this architecture (1.84x on 4x B200, 1.91x on one). Without it these
            # models fall through to --spec-default, i.e. no drafter at all.
            _emit_dspark()
        elif _auto_mla_embedded_mtp:
            # MLA embedded-MTP (GLM-5.2 et al.): the MTP path regresses vs spec-off
            # on llama.cpp today, so Auto drops it and falls back to ngram-mod (or
            # spec-off if unsupported), mirroring the sub-3B branch. Forced mtp /
            # mtp+ngram (handled above) still engage; UNSLOTH_MLA_MTP_ENABLED=1
            # re-enables this promotion once upstream optimizes the path.
            self._spec_fallback_reason = "mla_mtp_disabled"
            if caps.get("supports_ngram_mod"):
                logger.info(
                    "Auto: MLA embedded-MTP model detected; llama.cpp's MLA/DSA "
                    "MTP path is slower than no speculation, so using ngram-mod "
                    "instead. Override via the Unsloth Speculative Decoding "
                    "dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    "Auto: MLA embedded-MTP model detected; disabling speculative "
                    "decoding (this llama-server does not advertise ngram-mod). "
                    "Override via the dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                # spec-off: emit nothing, mirroring the sub-3B no-ngram path.
        elif is_mtp_model and not _mtp_too_small:
            if _mtp_drafter_missing:
                # Name-only MTP, drafter did not resolve (download failed/absent).
                _fallback_drafter_not_found()
            else:
                # GPU: MTP-only. CPU/Mac: chain ngram-mod + MTP.
                _emit_mtp(chain_ngram = not gpus)
        elif is_mtp_model and _mtp_too_small:
            # Sub-3B fallback: drop the MTP draft head, keep ngram-mod when
            # the binary supports it.
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
            elif caps.get("supports_ngram_mod"):
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold; using ngram-mod "
                    "only (zero-VRAM, no draft head). Override via "
                    "--spec-type or the Unsloth Speculative Decoding "
                    "dropdown."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold and the bundled "
                    "llama-server does not advertise ngram-mod; "
                    "auto-disabling speculative decoding."
                )
        else:
            # Non-MTP model: let llama-server choose its default strategy.
            flags.append("--spec-default")
            self._speculative_type = "default"
        return flags

    def adopt_load_intent_if_matched(self, intent: GgufLoadIntent) -> bool:
        """Match live state and adopt the caller's compatible GPU placement."""
        if not self.matches_load_source(intent):
            return False
        # Match repeated Auto requests against the recovered resident CPU state.
        if self._cpu_fallback_reason and self._auto_vulkan_cpu_fallback_eligible(
            self._find_llama_server_binary(),
            intent,
            list(intent.extra_args) if intent.extra_args is not None else None,
        ):
            intent = replace(
                intent,
                gpu_memory_mode = "manual",
                gpu_layers = 0,
                tensor_parallel = False,
                tensor_split = None,
                n_cpu_moe = 0,
                gpu_ids = None,
            )
        # The stored state is what LAUNCHED, and on a virtualised Metal device that is
        # the CPU-pinned rewrite, not the request. Apply the same rewrite here
        # (idempotent, so load_model's own call is a no-op) or a client that keeps
        # sending Auto mismatches on every call.
        if _metal_device_is_paravirtual():
            (
                _pv_mode,
                _pv_layers,
                _pv_tp,
                _pv_split,
                _pv_cpu_moe,
                _pv_extras,
            ) = paravirtual_normalized_request(
                gpu_memory_mode = intent.gpu_memory_mode,
                gpu_layers = intent.gpu_layers,
                tensor_parallel = intent.tensor_parallel,
                tensor_split = list(intent.tensor_split) if intent.tensor_split else None,
                n_cpu_moe = intent.n_cpu_moe,
                extra_args = list(intent.extra_args) if intent.extra_args is not None else None,
                log_dropped = False,
            )
            intent = replace(
                intent,
                gpu_memory_mode = _pv_mode,
                gpu_layers = _pv_layers,
                tensor_parallel = _pv_tp,
                tensor_split = tuple(_pv_split) if _pv_split else None,
                n_cpu_moe = _pv_cpu_moe,
                extra_args = tuple(_pv_extras) if _pv_extras is not None else None,
            )
        effective_extra_args = (
            list(intent.extra_args) if intent.extra_args is not None else self._extra_args
        )
        candidate_extra_args = list(effective_extra_args) if effective_extra_args else []
        if intent.extra_args is not None and intent.gpu_ids is not None:
            candidate_extra_args = self._strip_device_extra_args(candidate_extra_args)
        if not self._runtime_matches_intent(intent, candidate_extra_args):
            return False
        self._record_matching_gpu_request(
            list(intent.gpu_ids) if intent.gpu_ids is not None else None
        )
        return True

    def matches_load_source(self, intent: GgufLoadIntent) -> bool:
        """Whether the resident model has the intent's identity and weights."""
        if not self.is_loaded or (self._model_identifier or "").lower() != (
            intent.model_identifier.lower()
        ):
            return False
        if intent.gguf_path is not None and self._gguf_path:
            try:
                return Path(self._gguf_path).resolve() == Path(intent.gguf_path).resolve()
            except OSError:
                return False
        return (self._hf_variant or "").lower() == (intent.hf_variant or "").lower()

    def _classify_gpu_offload(
        self, expected_gpu: bool, detected_gpus: list[tuple[int, int]]
    ) -> Optional[bool]:
        """True if the model landed on a GPU, False if only CPU buffers landed
        despite GPU intent, None when there's no signal. Delegates to the shared
        classifier so it tracks current llama.cpp logs (offloaded-layer counts /
        device_info), not just the older "model buffer size" lines."""
        if not detected_gpus or not expected_gpu:
            return None
        return classify_gpu_offload_lines(self._stdout_lines)

    @staticmethod
    def _cmd_has_gpu_companion(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """True when the argv/env carries a GPU companion: any --mmproj form, or
        a drafter (Studio's --model-draft, the extras aliases, or the
        LLAMA_ARG_SPEC_DRAFT_* env) -- these offload to the GPU regardless of
        the main ``--gpu-layers``. A drafter explicitly forced to CPU
        (--spec-draft-ngl 0 / --spec-draft-device cpu) doesn't count."""
        _args = [str(a) for a in cmd]
        if any(a.startswith("--mmproj") for a in _args):
            # --no-mmproj-offload clears mmproj_use_gpu and clip.cpp gates the whole
            # GPU backend on it, so the projector holds no VRAM. Last flag wins.
            _off = [a for a in _args if a in ("--mmproj-offload", "--no-mmproj-offload")]
            if not _off or _off[-1] != "--no-mmproj-offload":
                return True
        if _extra_args_mtp_draft_path(cmd, env) is None:
            return False
        return not _extra_args_draft_offloaded_to_cpu(cmd, env)

    @staticmethod
    def _zero_offload_keeps_gpu_visible(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """Whether a zero-layer launch still has a reason to use visible GPUs.

        Keep this shared by child masking and post-launch residency bookkeeping:
        a device pin, surviving tensor mode, mmproj, or GPU drafter prevents the
        launch from being a confirmed zero-VRAM server.
        """
        return (
            LlamaCppBackend._cmd_has_gpu_device_pin(cmd, env)
            or _effective_tensor_parallel(cmd, False, env)
            or LlamaCppBackend._cmd_has_gpu_companion(cmd, env)
        )

    @staticmethod
    def _cmd_has_gpu_device_pin(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """True when the effective main or draft ``--device`` pin names a GPU."""
        main_flags = {"--device", "-dev"}
        draft_flags = {"--spec-draft-device", "-devd", "--device-draft"}
        last_main: Optional[str] = None
        last_draft: Optional[str] = None
        args = [str(arg) for arg in cmd]
        for index, raw in enumerate(args):
            flag = _flag_name(raw)
            _, equals, inline = raw.partition("=")
            if flag not in main_flags and flag not in draft_flags:
                continue
            value = inline if equals else (args[index + 1] if index + 1 < len(args) else "")
            if flag in main_flags:
                last_main = value
            else:
                last_draft = value
        if last_main is None:
            last_main = (env or {}).get("LLAMA_ARG_DEVICE")

        def _names_gpu(value: Optional[str]) -> bool:
            if value is None:
                return False
            devices = [item.strip().lower() for item in value.split(",") if item.strip()]
            return not devices or any(item not in ("cpu", "none") for item in devices)

        return _names_gpu(last_main) or _names_gpu(last_draft)

    @staticmethod
    def _zero_offload_gpu_flag(
        spawn_cmd: list,
        detected_gpus: list,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[bool]:
        """GPU-residency flag for a deliberate manual zero-offload load. The
        main model is CPU-only by construction, but device pins, tensor mode,
        mmproj, and GPU drafters can still make the server hold VRAM. The counted
        offload classifier cannot see those allocations. This uses the same
        predicate as the launch-time zero-VRAM mask; None means no GPU signal."""
        if LlamaCppBackend._is_vulkan_backend():
            # Unlike -ngl 0, the recovery's last-wins --device none removes Vulkan.
            main_device = (_extra_args_main_device(spawn_cmd) or "").strip().lower()
            if main_device in ("cpu", "none"):
                return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)
            if detected_gpus:
                return True
        if not detected_gpus:
            return None
        return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)

    def load_cancelled(self) -> bool:
        """True if a load was cancelled (e.g. via unload/_cancel_event) and not
        yet consumed by the next load_model. Lets the tensor->layer fallback
        avoid restarting a load the user just cancelled."""
        return self._cancel_event.is_set()

    def unload_model(self) -> bool:
        """Terminate the subprocess and cancel any in-flight download."""
        self._cancel_event.set()
        with self._lock:
            self._unload_epoch += 1
            self._kill_process()
            self._cleanup_cpu_fallback_runtime()
            logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._hf_repo = None
            self._mtp_draft_path = None
            self._mtp_draft_suppressed_path = None
            self._spec_fallback_reason = None
            self._spec_drafter_kind = None
            self._dspark_sidecar_absent = False
            self._cpu_fallback_reason = None
            self._last_load_intent = None
            self._mtp_runtime_fallback_active = False
            self._hf_variant = None
            self._is_vision = False
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            self._mmproj_has_audio = False
            self._port = None
            self._healthy = False
            self._context_length = None
            self._effective_context_length = None
            self._max_context_length = None
            self._reset_effective_parallel_slots()
            self._slot_save_dir = None
            self._slot_save_binary = None
            self._slot_loaded_identity = None
            self._prompt_cache_disabled = False
            self._swa_full = False
            self._kv_cache_unified = False
            self._n_ubatch = self._DEFAULT_N_UBATCH
            self._flash_attn_enabled = True
            self._effective_cache_types = ("f16", "f16")
            self._kv_cache_context_total = None
            self._chat_template = None
            self._markup_tokens = []
            self._markup_profile = None
            self._chat_template_override = None
            self._supports_reasoning = False
            self._reasoning_always_on = False
            self._reasoning_style = "enable_thinking"
            self._reasoning_effort_levels = []
            self._reasoning_default = True
            self._supports_preserve_thinking = False
            self._supports_tools = False
            self._cache_type_kv = None
            # GPU-pin state describes the active runner only; clear it so an explicit
            # pin never leaks into the next (or diffusion) runner.
            self._gpu_ids = None
            self._requested_gpu_ids = None
            self._tensor_parallel = False
            self._gpu_memory_mode = "auto"
            self._gpu_layers = -1
            self._n_cpu_moe = 0
            self._tensor_split = None
            self._layer_preserves_tensor_intent = False
            self._speculative_type = None
            self._requested_spec_mode = None
            self._spec_draft_n_max = None
            self._n_layers = None
            self._n_experts = None
            self._leading_dense_block_count = None
            self._n_kv_heads = None
            self._n_kv_heads_by_layer = None
            self._n_heads = None
            self._embedding_length = None
            self._kv_key_length = None
            self._kv_value_length = None
            self._sliding_window = None
            self._sliding_window_pattern = None
            self._full_attention_interval = None
            self._kv_lora_rank = None
            self._key_length_mla = None
            self._kv_key_length_swa = None
            self._kv_value_length_swa = None
            self._ssm_inner_size = None
            self._ssm_state_size = None
            self._ssm_conv_kernel = None
            self._kda_head_dim = None
            self._shared_kv_layers = None
            self._nextn_predict_layers = None
            # Clean up temp chat template file.
            if hasattr(self, "_chat_template_file") and self._chat_template_file:
                try:
                    os.unlink(self._chat_template_file.name)
                except Exception:
                    pass
                self._chat_template_file = None
            # Free audio codec GPU memory.
            if LlamaCppBackend._codec_mgr is not None:
                LlamaCppBackend._codec_mgr.unload()
                LlamaCppBackend._codec_mgr = None
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return True

    def _kill_process(self):
        """Terminate the subprocess if running."""
        # Stop the watchdog before a deliberate kill so a planned reload/unload
        # isn't seen as a crash; a real crash never routes through here.
        self._stop_mtp_crash_watchdog()
        self._reset_effective_parallel_slots()
        # Cleared before the early return, not in the finally below, so a stale split never
        # outlives its runner even when there was no process to kill.
        self._diffusion_requested_ngl = None
        if self._process is None:
            return
        # Not every _process is a Popen: tests stand one in to mean "a server is
        # loaded" without spawning anything. Terminating what cannot be terminated
        # is not an error worth reporting, but everything else still has to happen,
        # so this skips only the signalling and falls through to the finally rather
        # than returning early and duplicating part of it.
        terminable = hasattr(self._process, "terminate")
        if not terminable:
            logger.debug("no terminable llama-server process to kill; clearing state")
        try:
            if terminable:
                self._process.terminate()
                self._process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            # Signal first, report after. logger here is a structlog PrintLogger
            # writing straight to stdout, so a closed stream raises out of the
            # warning; doing it first meant SIGKILL was skipped and the server
            # left running with the reference about to be dropped below.
            self._process.kill()
            # Between the two, deliberately. After kill() so the signal never
            # depends on a log write, and before this wait() because a second
            # TimeoutExpired raised in here is not caught by the handler it is
            # raised from -- it escapes, and a truly unkillable server would then
            # be reported by nothing at all.
            logger.warning("llama-server did not exit on SIGTERM, sent SIGKILL")
            self._process.wait(timeout = 5)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            # getattr: teardown must tolerate a partially-built backend (failed
            # __init__ or a __new__-built instance), as with _llama_log_fh below.
            if getattr(self, "_stats_logger", None) is not None:
                self._stats_logger.stop()
                self._stats_logger = None
            self._process = None
            self._clear_server_pid()
            # Clear healthy so a /load during the replacement's warm-up can't
            # short-circuit against the previous server's health (#5401).
            self._healthy = False
            # Reset to unknown so the training guard treats the next (still
            # loading) server as VRAM-resident rather than reading the killed
            # server's stale zero-offload flag until the health probe reclassifies.
            self._gpu_offload_active = None
            # Drives _wait_for_vram_settle in the next load_model; set in finally
            # so both in-process and frontend Apply paths record the kill.
            self._last_kill_monotonic = time.monotonic()
            stdout_thread = getattr(self, "_stdout_thread", None)
            if stdout_thread is not None:
                stdout_thread.join(timeout = 2)
                self._stdout_thread = None
            fh = getattr(self, "_llama_log_fh", None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                self._llama_log_fh = None

    @staticmethod
    def _server_pidfile_path() -> Optional[Path]:
        """Pidfile recording the live llama-server PID, under the active studio root
        (per-root, so concurrent Studios with distinct UNSLOTH_STUDIO_HOME stay
        isolated, mirroring the reaper's custom-root isolation)."""
        try:
            from utils.paths.storage_roots import studio_root  # noqa: WPS433
            return studio_root() / "llama-server.pid"
        except Exception:
            return None

    @classmethod
    def _record_server_pid(cls, pid: int) -> None:
        """Best-effort record of the spawned llama-server PID for orphan reaping.

        Stores ``pid:starttime`` so a later startup can reject a PID that has
        since been recycled to a different process (see ``_pid_start_identity``).
        A bare ``pid`` (no identity) is still accepted on read for compatibility.
        """
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(f"{pid}:{cls._pid_start_identity(pid)}", encoding = "utf-8")
        except Exception as e:
            logger.debug(f"Could not write llama-server pidfile: {e}")

    @classmethod
    def _clear_server_pid(cls) -> None:
        """Best-effort removal of the llama-server pidfile."""
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.unlink(missing_ok = True)
        except Exception as e:
            logger.debug(f"Could not remove llama-server pidfile: {e}")

    @staticmethod
    def _pid_is_llama_server(pid: int) -> bool:
        """True only if pid is a live process whose binary is a llama-server. Guards
        against PID reuse before killing a recorded orphan; returns False on any
        uncertainty so an unrelated process is never killed."""
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                if (proc.name() or "").lower().startswith("llama-server"):
                    return True
                return Path(proc.exe() or "").name.lower().startswith("llama-server")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return False
        except ImportError:
            pass
        if sys.platform != "linux":
            return False
        try:
            if Path(os.readlink(f"/proc/{pid}/exe")).name.lower().startswith("llama-server"):
                return True
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                tokens = fh.read().split(b"\x00")
            first = tokens[0].decode("utf-8", "replace") if tokens else ""
            return Path(first).name.lower().startswith("llama-server")
        except OSError:
            return False

    @staticmethod
    def _pid_start_identity(pid: int) -> str:
        """Stable per-PID identity (process start time) guarding against PID reuse.

        Returns a token string, or "" when it cannot be determined (the caller
        then falls back to the llama-server name check only)."""
        try:
            import psutil
            try:
                return str(psutil.Process(pid).create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return ""
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                # field 22 (starttime), counted from after the ")" that closes comm.
                return data[data.rfind(b")") + 2 :].split()[19].decode()
            except (OSError, IndexError):
                return ""
        return ""

    @staticmethod
    def _pid_parent_is_alive(pid: int) -> bool:
        """True if the recorded server's parent is still running, i.e. the server is
        NOT orphaned. Lets the cross-session reap kill only a true orphan (parent
        gone) and never a live server owned by a running Unsloth, regardless of which
        process performs the sweep. Biased toward "alive" on uncertainty so a live
        server is never mistakenly reaped."""
        try:
            import psutil

            try:
                ppid = psutil.Process(pid).ppid()
            except psutil.NoSuchProcess:
                return False  # the recorded server itself is gone
            except psutil.Error:
                return True  # cannot tell -- never risk killing a live server
            if ppid <= 1:
                return False  # reparented to init -> orphan
            return psutil.pid_exists(ppid)
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                ppid = int(data[data.rfind(b")") + 2 :].split()[1])
            except (OSError, IndexError, ValueError):
                return False
            if ppid <= 1:
                return False
            return Path(f"/proc/{ppid}").exists()
        return False

    @staticmethod
    def _unlink_pidfile(path: Path) -> None:
        """Best-effort removal of a resolved pidfile path."""
        try:
            path.unlink(missing_ok = True)
        except Exception:
            pass

    @classmethod
    def _reap_recorded_pid(cls) -> int:
        """Kill the exact llama-server PID recorded at spawn, but only when it is a
        genuine orphan -- its parent (the Unsloth that spawned it) is gone. This is
        the cross-session backstop the parent-death reaper (Job Object /
        PR_SET_PDEATHSIG) cannot cover: an orphan left by an already-dead Unsloth
        (macOS, a best-effort failure, or a pre-existing orphan). Path-independent,
        so it also catches an orphan the install-root match would miss.

        A live server whose parent is still running is never reaped, so constructing
        a second backend in-process (the helper / advisor paths each build a
        LlamaCppBackend) cannot kill the active chat server. A recorded PID that has
        been recycled to a different process is rejected by the start-time identity
        and the llama-server name check, so unrelated user processes are never
        touched. SIGKILL falls back to SIGTERM on Windows, where os.kill maps it to
        TerminateProcess and SIGKILL is undefined."""
        path = cls._server_pidfile_path()
        if path is None or not path.exists():
            return 0

        pid = -1
        identity = ""
        try:
            pid_str, _, identity = path.read_text(encoding = "utf-8").strip().partition(":")
            pid = int(pid_str)
        except Exception:
            pid = -1

        if pid <= 0:
            cls._unlink_pidfile(path)  # garbage record
            return 0
        if pid == os.getpid():
            return 0  # never our own pid; leave the record alone

        if cls._pid_parent_is_alive(pid):
            # Live server with a running parent -> not an orphan; keep the record so
            # a later startup can still reap it if that parent later dies abnormally.
            return 0

        # Parent is gone: candidate orphan. Reject a PID recycled to something else.
        if identity and cls._pid_start_identity(pid) != identity:
            cls._unlink_pidfile(path)
            return 0

        killed = 0
        if cls._pid_is_llama_server(pid):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                killed = 1
                logger.info(f"Killed orphaned llama-server from pidfile (pid={pid})")
            except (ProcessLookupError, PermissionError):
                pass
            except Exception as e:
                logger.debug(f"Could not kill recorded llama-server pid {pid}: {e}")
        cls._unlink_pidfile(path)
        return killed

    @staticmethod
    def _kill_orphaned_servers() -> int:
        """Kill orphaned llama-server processes started by studio.

        Only kills processes whose resolved binary lives under a known
        Unsloth install dir (or matches an exact env-var override), to avoid
        terminating unrelated llama-server instances. Mirrors every location
        _find_llama_server_binary() can return, so orphans from any
        supported install path are cleaned up.

        Uses psutil for cross-platform support (Linux, macOS, Windows);
        falls back to pgrep + /proc/<pid>/exe on Linux when psutil is
        absent.

        Returns the count of processes killed; callers arm the VRAM-settle
        wait on a positive count.
        """
        # Cross-session backstop first: reap the exact PID we recorded at spawn,
        # but only if it is a true orphan whose parent is gone (so a helper backend
        # built while a chat server is live can never kill it). The root-gated
        # enumeration below stays as a fallback.
        killed = LlamaCppBackend._reap_recorded_pid()
        try:
            # -- Build the ownership allowlist --------------------------------
            # exact_binaries -- env var overrides (exact path match).
            # install_roots  -- Unsloth-owned dir trees (binary must be under one).
            install_roots: list[Path] = []

            # Env-mode custom root (mirrors _find_llama_server_binary).
            _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
            _is_custom_root = not _is_legacy
            if _is_custom_root:
                install_roots.append(_resolved_sr / "llama.cpp")

            # Primary install dir (default mode only). Env-mode skips this so a
            # custom-root Unsloth can't kill a default-install Unsloth's server.
            if not _is_custom_root:
                install_roots.append(Path.home() / ".unsloth" / "llama.cpp")

            # Legacy in-tree build dirs (older setup.sh)
            project_root = Path(__file__).resolve().parents[4]
            install_roots.append(project_root / "llama.cpp")

            # Legacy: extracted binary
            install_roots.append(project_root / "bin")

            # UNSLOTH_LLAMA_CPP_PATH env var (custom install dir)
            custom_dir = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
            if custom_dir:
                install_roots.append(Path(custom_dir))

            # LLAMA_SERVER_PATH env var (exact binary path)
            exact_binaries: list[Path] = []
            env_binary = os.environ.get("LLAMA_SERVER_PATH")
            if env_binary:
                try:
                    exact_binaries.append(Path(env_binary).resolve())
                except OSError:
                    pass

            # Resolve all roots so is_relative_to works reliably.
            resolved_roots: list[Path] = []
            for root in install_roots:
                try:
                    # A --with-llama-cpp-dir local link (symlink/junction)
                    # resolves into the user's own checkout. Adding it would let
                    # us treat the user's externally-launched llama-server as our
                    # orphan and kill it, so leave such roots out of the
                    # allowlist (we forgo orphan-reaping for local-link installs).
                    if _is_external_link(root):
                        continue
                    resolved_roots.append(root.resolve())
                except OSError:
                    pass

            my_pid = os.getpid()

            # -- Enumerate processes -------------------------------------------
            # Prefer psutil (cross-platform); fall back to pgrep + /proc on
            # Linux when psutil is absent.
            try:
                import psutil
                has_psutil = True
            except ImportError:
                has_psutil = False

            if has_psutil:
                for proc in psutil.process_iter(["pid", "name", "exe"]):
                    try:
                        if proc.info["pid"] == my_pid:
                            continue

                        name = proc.info.get("name") or ""
                        if not name.lower().startswith("llama-server"):
                            continue

                        exe = proc.info.get("exe")
                        if not exe:
                            continue

                        exe_path = Path(exe).resolve()

                        # Ownership: exact match OR binary under a known root.
                        is_ours = exe_path in exact_binaries or any(
                            exe_path.is_relative_to(root) for root in resolved_roots
                        )
                        if not is_ours:
                            continue

                        # A live parent means a running Unsloth (or the user's
                        # shell) still owns it -- not an orphan.
                        if LlamaCppBackend._pid_parent_is_alive(proc.info["pid"]):
                            continue

                        proc.kill()
                        killed += 1
                        logger.info(
                            f"Killed orphaned llama-server process (pid={proc.info['pid']})"
                        )
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        pass
            else:
                # -- Fallback: pgrep + /proc/<pid>/exe (Linux only) -----------
                if sys.platform != "linux":
                    return killed
                result = subprocess.run(
                    ["pgrep", "-a", "-f", "llama-server"],
                    capture_output = True,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                    timeout = 5,
                    env = child_env_without_native_path_secret(),
                )
                if result.returncode != 0:
                    return killed

                for line in result.stdout.strip().splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) < 2:
                        continue
                    pid = int(parts[0])
                    if pid == my_pid:
                        continue

                    # /proc/<pid>/exe symlinks the real binary, avoiding
                    # cmdline-parsing ambiguities; fall back to the first
                    # cmdline token when /proc is unavailable.
                    proc_exe = Path(f"/proc/{pid}/exe")
                    try:
                        binary = proc_exe.resolve(strict = True)
                    except (OSError, ValueError):
                        cmdline = parts[1]
                        token = cmdline.split()[0] if cmdline.strip() else ""
                        if not token:
                            continue
                        binary = Path(token).resolve(strict = False)

                    owned = binary in exact_binaries or any(
                        binary.is_relative_to(root) for root in resolved_roots
                    )
                    if not owned:
                        continue

                    if LlamaCppBackend._pid_parent_is_alive(pid):
                        continue

                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                        logger.info(f"Killed orphaned llama-server process (pid={pid})")
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        pass
        except Exception:
            logger.warning("Error during orphan server cleanup", exc_info = True)
        return killed

    def _cleanup(self):
        """atexit handler to ensure llama-server is terminated.

        Nothing here may report a failure through the logging machinery. By the
        time atexit runs, the streams the handlers write to can already be closed,
        and a write then fails -- which is how this was found, as several unrelated
        tracebacks printed after the test summary.

        Two different mechanisms, because there are two kinds of logger in play.
        This module's own logger is a structlog PrintLogger writing straight to
        stdout (loggers/config.py), which has no error handling of its own and does
        not consult raiseExceptions at all, so only the except below covers it.
        raiseExceptions covers the stdlib loggers other libraries install, which
        report a broken handler by printing their own traceback instead of raising.
        Neither one alone is enough.
        """
        raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            self._kill_process()
        except Exception:
            # atexit swallows this anyway, and there is nowhere left to report it.
            pass
        finally:
            logging.raiseExceptions = raise_exceptions

    @staticmethod
    def _fit_off_retry_eligible(cmd: "list[str]", use_fit: bool) -> bool:
        """Whether a llama-server startup crash may be retried with --fit off.

        Only when Unsloth's own VRAM math placed the model (use_fit=False)
        and nothing on the command line set the fit mode explicitly
        (-fit / --fit, space- or equals-form). --fit-ctx / --fit-target /
        -fitc / -fitt tune the fit step but do not select the mode, so
        they do not block the retry.
        """
        if use_fit:
            return False
        for a in cmd:
            if a in ("-fit", "--fit") or a.startswith(("-fit=", "--fit=")):
                return False
        return True

    def _probe_mtp_decode(self, timeout: float = 60.0) -> bool:
        """One tiny /completion to confirm MTP survives the first decode.

        MTP-draft can pass /health yet crash the flash-attn kernel only once
        tokens generate (e.g. under --split-mode tensor). False on any error so
        the caller can drop MTP and retry.
        """
        url = f"{self.base_url}/completion"
        payload = {"prompt": "Hi", "n_predict": 4, "temperature": 0.0, "stream": False}
        try:
            resp = httpx.post(
                url,
                json = payload,
                timeout = timeout,
                headers = self._auth_headers,
                trust_env = False,
            )
        except Exception as e:
            logger.debug(f"MTP decode probe failed: {e}")
            return False
        if resp.status_code != 200:
            logger.debug(f"MTP decode probe returned HTTP {resp.status_code}")
            return False
        # A crash can drop the connection or kill the process right after a reply.
        if self._process is not None and self._process.poll() is not None:
            return False
        return True

    def _slot_launch_fingerprint(self) -> tuple:
        # KV validity keys on extra args, stat'd sidecar weights, effective ctx.
        sidecars = []
        for path in self._sidecar_weight_files():
            try:
                st = os.stat(path)
                sidecars.append((path, st.st_size, st.st_mtime_ns))
            except OSError:
                sidecars.append((path, None, None))
        return (
            tuple(self._extra_args or ()),
            tuple(sidecars),
            self._requested_n_ctx,
            self._effective_context_length,
            self._effective_cache_types,
            self.effective_parallel_slots,
            self._swa_full,
            self._kv_cache_unified,
            self._n_ubatch,
            self._flash_attn_enabled,
        )

    def _gguf_file_identity(self, path) -> Optional[tuple]:
        # (size, mtime_ns) per shard: a split GGUF keys KV validity on every sibling.
        p = Path(path)
        paths = [p]
        m = _SHARD_FULL_RE.match(p.name)
        if m:
            prefix, _first, total = m.groups()
            paths = [
                p.with_name(f"{prefix}-{i:05d}-of-{total}{p.suffix}")
                for i in range(1, int(total) + 1)
            ]
        try:
            return tuple((sp.stat().st_size, sp.stat().st_mtime_ns) for sp in paths)
        except OSError:
            return None

    _SIDECAR_WEIGHT_FLAGS = (
        "--lora",
        "--lora-scaled",
        "--control-vector",
        "--control-vector-scaled",
    )

    def _sidecar_weight_files(self) -> list[str]:
        # llama.cpp: comma-separated paths, FNAME:SCALE on -scaled (older builds: FNAME SCALE).
        args = [str(a).strip() for a in (self._extra_args or ())]
        files: list[str] = []
        for i, arg in enumerate(args):
            flag = _flag_name(arg)
            _, sep, inline = arg.partition("=")
            if flag not in self._SIDECAR_WEIGHT_FLAGS:
                continue
            operand = inline if sep else (args[i + 1] if i + 1 < len(args) else "")
            if not operand:
                continue
            candidates = [operand]
            pieces = [p for p in operand.split(",") if p]
            if len(pieces) > 1:
                candidates.extend(pieces)
            if flag.endswith("-scaled"):
                for item in list(candidates):
                    # ":<number>" tail is a scale; rpartition spares drive letters.
                    head, colon, tail = item.rpartition(":")
                    if not (colon and head):
                        continue
                    try:
                        float(tail)
                    except ValueError:
                        continue
                    candidates.append(head)
            for cand in candidates:
                if cand not in files:
                    files.append(cand)
        return files

    def _prompt_cache_off(self) -> bool:
        # Caching off makes restores useless; last prompt-cache flag wins, env only when unset.
        last = None
        for arg in self._extra_args or ():
            flag = arg.strip().split("=", 1)[0]
            if flag in ("--cache-prompt", "--no-cache-prompt"):
                last = flag
        if last is not None:
            return last == "--no-cache-prompt"
        if self._prompt_cache_disabled:
            return True
        if os.environ.get("LLAMA_ARG_NO_CACHE_PROMPT") is not None:
            return True
        env = (os.environ.get("LLAMA_ARG_CACHE_PROMPT") or "").strip().lower()
        return env in _LLAMA_ARG_FALSE_VALUES

    def save_slots_for_resume(
        self, should_abort: Optional[Callable[[], bool]] = None
    ) -> Optional[dict]:
        if (
            not self.is_loaded
            or not self._slot_save_dir
            or not self._gguf_path
            or self._prompt_cache_off()
        ):
            return None
        # Same predicate as the estimator's SWA path: a window alone is not enough.
        # phi3 GGUFs carry attention.sliding_window but no key/value length, and
        # llama.cpp forces them back to a non-SWA cache, so their slots do restore.
        if (
            (self._sliding_window or 0) > 0
            and self._kv_key_length is not None
            and self._kv_value_length is not None
            and not self._swa_full
        ):
            logger.debug("Skipping slot save: compact SWA cache cannot be reused after restart")
            return None
        save_dir = Path(self._slot_save_dir)
        gguf_stat = self._gguf_file_identity(self._gguf_path)
        if gguf_stat is None:
            return None
        launch = self._slot_launch_fingerprint()
        # If the GGUF or a sidecar was swapped on disk while the original weights
        # stayed mapped, the live KV belongs to the old weights but a reload would
        # load the new file. Persisting it would let restore misapply stale KV.
        if self._slot_loaded_identity is not None and self._slot_loaded_identity != (
            gguf_stat,
            launch,
        ):
            logger.debug("Skipping slot save: model files changed on disk since load")
            return None
        try:
            estimate = self._estimate_kv_cache_bytes(
                self._kv_cache_context_total
                or self._effective_context_length
                or self._context_length
                or 0,
                max(self._effective_cache_types, key = _kv_bytes_per_elem),
                n_parallel = self.effective_parallel_slots,
                swa_full = self._swa_full,
                kv_unified = self._kv_cache_unified,
                n_ubatch = self._n_ubatch,
                flash_attn = self._flash_attn_enabled,
            )
            # Skip before writing anything when the estimate alone blows the cap,
            # rather than fully writing a slot and discarding it afterwards.
            if estimate > _SLOT_SAVE_MAX_BYTES:
                logger.debug(
                    "Skipping slot save: estimated %d bytes exceeds cap %d",
                    estimate,
                    _SLOT_SAVE_MAX_BYTES,
                )
                return None
            # A 0 estimate means metadata was insufficient, not a zero-byte cache:
            # a slot can still be many GiB, so demand room for the whole cap before
            # trusting the post-write check.
            required = (estimate if estimate > 0 else _SLOT_SAVE_MAX_BYTES) + (1 << 30)
            if shutil.disk_usage(save_dir).free < required:
                logger.debug("Skipping slot save: insufficient free disk")
                return None
        except Exception:
            pass
        token = uuid.uuid4().hex[:8]
        entries: list[dict] = []
        total_bytes = 0
        for slot in range(self.effective_parallel_slots):
            # A request pending mid-save waits on the gate; stop wasting its time.
            if should_abort is not None and should_abort():
                break
            filename = f"resume-{token}-slot{slot}.bin"
            path = save_dir / filename
            try:
                resp = httpx.post(
                    f"{self.base_url}/slots/{slot}",
                    params = {"action": "save"},
                    json = {"filename": filename},
                    headers = self._auth_headers,
                    timeout = _SLOT_SAVE_HTTP_TIMEOUT,
                    trust_env = False,
                )
            except Exception as e:
                logger.debug(f"slot {slot} save failed: {e}")
                with contextlib.suppress(OSError):
                    path.unlink()
                break
            if resp.status_code != 200:
                logger.debug(f"slot {slot} save returned HTTP {resp.status_code}")
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            try:
                body = resp.json()
                if not isinstance(body, dict):
                    raise ValueError("slot save response was not a JSON object")
                n_saved = int(body.get("n_saved") or 0)
            except Exception as e:
                # A 200 that still wrote a file but returns a malformed body must
                # clean up like the transport/HTTP error paths above, or the file
                # (which holds chat KV) is orphaned until the next startup sweep.
                logger.debug(f"slot {slot} save returned an invalid response: {e}")
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            if n_saved <= 0:
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            # Account by the bytes actually on disk, not the server-reported
            # count, so the cap holds even if a custom binary under-reports.
            try:
                n_written = path.stat().st_size
            except OSError:
                n_written = 0
            total_bytes += n_written
            entries.append({"id": slot, "filename": filename, "n_saved": n_saved})
            if total_bytes > _SLOT_SAVE_MAX_BYTES:
                break  # already over the cap; the discard below cleans up
        if not entries:
            return None
        if total_bytes > _SLOT_SAVE_MAX_BYTES:
            logger.debug(
                "Discarding slot save: %d bytes exceeds cap %d",
                total_bytes,
                _SLOT_SAVE_MAX_BYTES,
            )
            for entry in entries:
                with contextlib.suppress(OSError):
                    (save_dir / entry["filename"]).unlink()
            return None
        return {
            "dir": self._slot_save_dir,
            "binary": self._slot_save_binary,
            "gguf": str(self._gguf_path),
            "gguf_stat": gguf_stat,
            "launch": launch,
            "slots": entries,
        }

    def restore_slots_for_resume(self, manifest: dict) -> None:
        if not self.is_loaded or not self._slot_save_dir:
            return
        for entry in manifest.get("slots") or []:
            try:
                resp = httpx.post(
                    f"{self.base_url}/slots/{int(entry['id'])}",
                    params = {"action": "restore"},
                    json = {"filename": str(entry["filename"])},
                    headers = self._auth_headers,
                    timeout = _SLOT_SAVE_HTTP_TIMEOUT,
                    trust_env = False,
                )
            except Exception as e:
                logger.debug(f"slot restore failed: {e}")
                break
            if resp.status_code != 200:
                logger.debug(f"slot {entry.get('id')} restore returned HTTP {resp.status_code}")

    def _maybe_recover_from_mtp_crash(self, exc: Optional[BaseException] = None) -> bool:
        """Schedule one background reload without MTP after a mid-generation death.

        MTP+tensor can crash the flash-attn kernel on a later request, after
        load_model returned, past the load-time fallback and decode probe. Not a
        persistent ban: a fresh load re-tries MTP. Returns True if scheduled.
        """
        # Cheap async-safe gate: only our live MTP+tensor launch, not cancelled,
        # with a snapshot to replay.
        if self._cancel_event.is_set():
            return False
        if not self._mtp_runtime_fallback_active:
            return False
        # Read before claiming: a raise after the claim strands the flag, and nothing
        # else clears it, blocking every later respawn.
        snapshot = self._last_load_intent
        proc = self._process
        if snapshot is None or proc is None:
            return False
        # Single-flight: the first failure claims the reload.
        with self._mtp_runtime_fallback_lock:
            if self._mtp_runtime_fallback_in_progress:
                return False
            self._mtp_runtime_fallback_in_progress = True

        def _recover():
            try:
                # Confirm the process really exited (the error can arrive a beat
                # early) so a transient stream error can't disable MTP.
                deadline = time.monotonic() + 5.0
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if proc.poll() is None:
                    logger.debug("Generation error but llama-server is alive; keeping MTP.")
                    return
                logger.warning(
                    "llama-server exited mid-generation with MTP under tensor "
                    "parallelism (%s); reloading without speculative decoding.",
                    type(exc).__name__ if exc is not None else "server exited",
                )
                # Re-check under the load lock (RLock allows the nested
                # load_model) so a newer load isn't clobbered by this stale replay.
                requested_mode = snapshot.speculative_type
                with self._serial_load_lock:
                    if self._cancel_event.is_set():
                        logger.info("MTP-crash reload skipped: load was cancelled/unloaded.")
                        return
                    if self._process is not proc:
                        logger.info("MTP-crash reload skipped: a newer load is already active.")
                        return
                    if self._last_load_intent != snapshot:
                        logger.info("MTP-crash reload skipped: load settings changed.")
                        return
                    # Appending --spec-default cannot drop MTP: llama.cpp appends spec
                    # types rather than replacing, so the extras have to lose theirs.
                    # Once they do, Unsloth owns the block and the launch scrubs the
                    # inherited env for the replay too.
                    _ea = list(snapshot.extra_args or ())
                    fallback_extra_args = snapshot.extra_args
                    if _extra_args_requests_mtp(_ea, env = _child_spec_env(_ea)):
                        fallback_extra_args = tuple(
                            strip_shadowing_flags(
                                _ea,
                                strip_context = False,
                                strip_cache = False,
                                strip_spec = True,
                                strip_template = False,
                                strip_split_mode = False,
                            )
                        )
                    fallback = replace(
                        snapshot,
                        speculative_type = "off",
                        extra_args = fallback_extra_args,
                    )
                    self.load_model(fallback)
                    # Same reason the mode is restored below: the replay launched a
                    # stripped list but the caller keeps sending the original, so a
                    # comparator seeing only the stripped one would reload the very MTP
                    # configuration that just crashed.
                    if fallback_extra_args is not snapshot.extra_args:
                        self._requested_extra_args = (
                            self._strip_device_extra_args(_ea)
                            if snapshot.gpu_ids is not None
                            else list(_ea)
                        )
                    # Restore the requested mode + reason load_model("off") cleared,
                    # so /status shows the user's mode + note (like the startup fallback).
                    self._requested_spec_mode = _canonicalize_spec_mode(requested_mode)
                    self._spec_fallback_reason = "runtime_error"
                logger.info("Reloaded without MTP after the tensor-parallel crash.")
            except Exception as e:
                logger.error(f"Reload without MTP failed: {e}")
            finally:
                with self._mtp_runtime_fallback_lock:
                    self._mtp_runtime_fallback_in_progress = False

        try:
            threading.Thread(target = _recover, daemon = True, name = "mtp-crash-reload").start()
        except RuntimeError as exc:
            # Release the claim: a reload that never started would block respawn forever.
            with self._mtp_runtime_fallback_lock:
                self._mtp_runtime_fallback_in_progress = False
            logger.error(f"Could not start the MTP-crash reload: {exc}")
            return False
        return True

    def _start_mtp_crash_watchdog(self) -> None:
        """Background poll that recovers on an MTP+tensor crash even when no
        request observes it (direct proxy endpoints, or nothing in flight).

        Armed only for a live MTP+tensor launch; the no-MTP reload disarms it, so
        it can't loop.
        """
        if not self._mtp_runtime_fallback_active:
            return
        proc = self._process
        if proc is None:
            return
        # Replace any prior watchdog (loads are serialised, so at most one).
        self._stop_mtp_crash_watchdog()
        stop = threading.Event()
        self._mtp_watchdog_stop = stop

        def _watch():
            # Exit on stop or process death. _kill_process sets stop before
            # terminating, so re-check it: only a real crash (stop unset) recovers.
            while not stop.wait(1.0):
                if proc.poll() is not None:
                    if not stop.is_set():
                        self._maybe_recover_from_mtp_crash()
                    return

        t = threading.Thread(target = _watch, daemon = True, name = "mtp-crash-watchdog")
        self._mtp_watchdog_thread = t
        t.start()

    def _stop_mtp_crash_watchdog(self) -> None:
        """Signal the crash watchdog to exit; called before any deliberate kill."""
        stop = getattr(self, "_mtp_watchdog_stop", None)
        if stop is not None:
            stop.set()
        self._mtp_watchdog_thread = None

    def _wait_for_health(
        self,
        timeout: float = 120.0,
        interval: float = 0.5,
    ) -> bool:
        """Poll llama-server's /health until 200; also detect early exit/crash."""
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/health"

        while time.monotonic() < deadline:
            # Process crashed?
            if self._process.poll() is not None:
                # Let the drain thread collect final output.
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout = 2)
                output = "\n".join(self._stdout_lines[-50:])
                # Keep the TAIL: crash details (abort reason, ROCm/CUDA error
                # text) print last, after the long startup banner. Head
                # truncation has cut off exactly the diagnostic line before.
                _log_hint = (
                    f" Full log: {self._llama_log_path}"
                    if getattr(self, "_llama_log_path", None)
                    else ""
                )
                logger.error(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Output (tail): {output[-2000:]}{_log_hint}"
                )
                return False

            try:
                # trust_env=False: skip ambient HTTP(S)_PROXY, which if it 503s
                # for 127.0.0.1 loops the probe until timeout and hangs load.
                resp = httpx.get(url, timeout = 2.0, trust_env = False)
                if resp.status_code == 200:
                    return True
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                # ReadError covers TCP RST mid-read while still binding the port
                # (Windows: WinError 10054); the crash branch catches real exits.
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.WriteError,
            ):
                pass

            time.sleep(interval)

        # Leave a marker so _classify_llama_start_failure tells a live but
        # never-healthy load (too large, or a proxy hijacking the loopback
        # probe) apart from a bad GGUF (#5740).
        self._stdout_lines.append(f"llama-server health check timed out after {timeout}s")
        logger.error(f"llama-server health check timed out after {timeout}s")
        return False

    @staticmethod
    def _ctx_integrity_flags(
        n_parallel: int,
        use_fit: bool,
        auto_fit: bool,
        requested_ctx: int,
        effective_ctx: int,
        caps: dict,
    ) -> list[str]:
        """Flags that keep the per-request window equal to the advertised ctx.

        Explicit ``--parallel`` disables llama-server's auto-slots
        ``--kv-unified`` default, silently splitting ``-c`` into per-slot
        windows of ``-c / N``; restore the shared pool so one request can use
        the full context. With ``--fit on``, ``--fit-ctx`` floors the fit step
        at an explicitly requested ctx so it offloads or fails instead of
        silently shrinking the window. The 8192 auto-floor and tighter
        ``--fit-target`` margin apply only under Manual + Auto (``auto_fit``),
        which omits ``-c``: on the legacy auto path ``-c 0`` already pins the
        native window and ``--fit-ctx 8192`` would override it down to 8192.
        """
        flags: list[str] = []
        if n_parallel > 1 and caps.get("supports_kv_unified"):
            flags.append("--kv-unified")
        if use_fit and caps.get("supports_fit_ctx"):
            if requested_ctx > 0 and effective_ctx > 0:
                # Floor the fit step at the explicitly requested ctx.
                flags.extend(["--fit-ctx", str(effective_ctx)])
            elif auto_fit:
                # Manual + Auto omits -c, so floor at 8192 so --fit doesn't
                # shrink the window below a usable size.
                flags.extend(["--fit-ctx", "8192"])
        if use_fit and auto_fit and caps.get("supports_fit_target"):
            # Leave 512 MiB free per device so llama.cpp packs more of the
            # model onto the GPU before spilling to system RAM.
            flags.extend(["--fit-target", "512"])
        return flags

    def _query_server_n_ctx(self) -> Optional[int]:
        """Per-slot context llama-server actually allocated, from ``/props``.

        The memory-fit step or ``--parallel`` slot split can leave this below
        the requested ``-c``; requests are validated against this value.
        """
        url = f"{self.base_url}/props"
        try:
            resp = httpx.get(url, timeout = 5.0, trust_env = False)
            if resp.status_code != 200:
                return None
            settings = resp.json().get("default_generation_settings") or {}
            n_ctx = settings.get("n_ctx")
            return int(n_ctx) if n_ctx else None
        except Exception:
            return None

    def _reconcile_effective_ctx_with_server(self) -> None:
        """Adopt the server's real ``n_ctx`` when it is below Unsloth's value.

        Keeps ``context_length`` (load response, status route, passthrough
        ``max_tokens`` ceiling) honest; clients sized to the requested value
        would otherwise hit ``exceed_context_size_error`` 400s early.
        """
        actual_n_ctx = self._query_server_n_ctx()
        if not actual_n_ctx or actual_n_ctx <= 0:
            return
        slots = 1 if self._kv_cache_unified else self.effective_parallel_slots
        self._kv_cache_context_total = actual_n_ctx * slots
        if self._effective_context_length and actual_n_ctx < self._effective_context_length:
            logger.warning(
                "llama-server allocated a smaller per-request context than "
                f"requested ({self._effective_context_length} -> {actual_n_ctx}; "
                "memory fit or --parallel slot split); clients must treat "
                f"{actual_n_ctx} as the real context window."
            )
            self._effective_context_length = actual_n_ctx
        elif not self._effective_context_length:
            self._effective_context_length = actual_n_ctx

    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _parse_tool_calls_from_text(
        content: str,
        *,
        allow_incomplete: bool = True,
        enabled_tool_names: Optional[set] = None,
    ) -> list[dict]:
        """Wrapper around the shared parser; ``enabled_tool_names`` gates the markerless bare-JSON form."""
        return _shared_parse_tool_calls_from_text(
            content,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    @staticmethod
    def _build_openai_messages(messages: list[dict], image_b64: Optional[str] = None) -> list[dict]:
        """Build OpenAI-format messages, optionally injecting an image_url part
        into the last user message for vision models. As-is if no image."""
        if not image_b64:
            return messages

        # Convert the last user message to multimodal content parts
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

    @contextlib.contextmanager
    def _open_stream(self, url: str, payload: dict, cancel_event):
        """Open a streaming POST to llama-server, retrying through prefill, and
        yield ``(response, first_token_deadline)`` once a 200 lands. Owns the
        httpx.Client + auth headers for the stream's lifetime; raises
        RuntimeError on a non-200. Shared scaffold for the streaming consumers,
        which differ only in how they parse the SSE body."""
        stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
        with httpx.Client(
            timeout = stream_timeout,
            limits = httpx.Limits(max_keepalive_connections = 0),
            trust_env = False,
        ) as client:
            first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
            with self._stream_with_retry(
                client,
                url,
                payload,
                cancel_event,
                headers = self._auth_headers,
                first_token_deadline = first_token_deadline,
            ) as response:
                if response.status_code != 200:
                    error_body = response.read().decode()
                    raise RuntimeError(
                        f"llama-server returned {response.status_code}: {error_body}"
                    )
                yield response, first_token_deadline

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
        stall_timeout_s: float = _DEFAULT_STREAM_STALL_TIMEOUT_S,
        first_token_deadline: Optional[float] = None,
        post_first_chunk_read_timeout_s: Optional[float] = _DEFAULT_STREAM_STALL_TIMEOUT_S,
    ) -> Generator[str, None, None]:
        """Iterate a stream while polling cancel and stall timeouts."""
        text_iter = response.iter_text()
        if first_token_deadline is None:
            first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
        last_chunk_at: Optional[float] = None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                if last_chunk_at is None:
                    remaining_s = first_token_deadline - time.monotonic()
                    if remaining_s <= 0:
                        raise httpx.ReadTimeout("The model did not produce a first token in time.")
                    LlamaCppBackend._set_stream_read_timeout(response, remaining_s)
                chunk = next(text_iter)
                if chunk:
                    if last_chunk_at is None and post_first_chunk_read_timeout_s is not None:
                        LlamaCppBackend._set_stream_read_timeout(
                            response,
                            post_first_chunk_read_timeout_s,
                        )
                    last_chunk_at = time.monotonic()
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                now = time.monotonic()
                if last_chunk_at is None:
                    if now >= first_token_deadline:
                        raise
                elif now - last_chunk_at >= stall_timeout_s:
                    raise httpx.ReadTimeout("The model stopped producing tokens mid-response.")
                continue

    @staticmethod
    def _set_stream_read_timeout(response: "httpx.Response", read_timeout_s: float) -> None:
        """Lower only post-header stream reads; keep prefill timeout long."""
        try:
            timeout_ext = response.request.extensions.get("timeout")
            if isinstance(timeout_ext, dict):
                timeout_ext["read"] = read_timeout_s
        except Exception:
            logger.debug("Could not lower response read timeout", exc_info = True)

    @staticmethod
    def _shutdown_active_httpx_sockets(client: "httpx.Client") -> None:
        """Best-effort interrupt for a sync httpx read blocked in recv(), whether
        parked before headers (prefill) or mid-stream."""
        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            connections = list(getattr(pool, "_connections", []) or [])
            for connection in connections:
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        except Exception:
            logger.debug("Could not shutdown active httpx socket", exc_info = True)
        try:
            client.close()
        except Exception:
            logger.debug("Could not close httpx client", exc_info = True)

    @staticmethod
    def _install_cancel_aware_read(
        client: "httpx.Client",
        cancel_event: threading.Event,
        response: Optional["httpx.Response"] = None,
        poll_s: float = 0.2,
    ) -> None:
        """Wrap the httpcore stream so the reader interrupts its own blocked recv() on cancel.

        A cross-thread socket shutdown wakes a parked recv() on POSIX but not on
        Windows (Winsock), so read in short slices and poll cancel_event between them
        (plain or TLS); slice timeouts are swallowed so a slow-but-alive stream survives.
        httpcore snapshots request.extensions["timeout"]["read"] once at body start, so
        given ``response`` we re-read the live value per call to honor the post-first-token
        stall timeout instead of the long prefill timeout."""
        import httpcore

        def _live_read_timeout() -> Optional[float]:
            if response is None:
                return None
            try:
                ext = response.request.extensions.get("timeout")
                if isinstance(ext, dict):
                    value = ext.get("read")
                    if isinstance(value, (int, float)):
                        return float(value)
            except Exception:
                pass
            return None

        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            for connection in list(getattr(pool, "_connections", []) or []):
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                if stream is None or getattr(stream, "_unsloth_cancel_wrapped", False):
                    continue
                orig_read = stream.read

                def read(
                    max_bytes,
                    timeout = None,
                    _orig = orig_read,
                ):
                    live = _live_read_timeout()
                    effective = live if live is not None else timeout
                    deadline = None if effective is None else time.monotonic() + effective
                    while True:
                        if cancel_event.is_set():
                            raise httpcore.ReadError("stream cancelled by user")
                        if deadline is None:
                            step = poll_s
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise httpcore.ReadTimeout("read operation timed out")
                            step = min(poll_s, remaining)
                        try:
                            return _orig(max_bytes, timeout = step)
                        except httpcore.ReadTimeout:
                            if deadline is not None and time.monotonic() >= deadline:
                                raise
                            continue  # slow but alive: keep reading

                stream.read = read
                stream._unsloth_cancel_wrapped = True
        except Exception:
            logger.debug("Could not install cancel-aware read", exc_info = True)

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
        headers: Optional[dict] = None,
        first_token_deadline: Optional[float] = None,
    ):
        """Open one streaming POST and let cancel interrupt prefill or reads."""
        if cancel_event is not None and cancel_event.is_set():
            raise _LlamaStreamCancelled

        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        try:
                            # response.close() can't wake a read already blocked in
                            # recv(); only a socket shutdown does, so shut down first.
                            LlamaCppBackend._shutdown_active_httpx_sockets(client)
                            if r is not None:
                                r.close()
                            return
                        except Exception as e:
                            logger.debug(f"Error closing request in cancel watcher: {e}")
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(target = _cancel_watcher, daemon = True, name = "prefill-cancel")
            watcher.start()

        try:
            if first_token_deadline is None:
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
            prefill_read_timeout = max(0.1, first_token_deadline - time.monotonic())
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = prefill_read_timeout,
                write = 10,
                pool = 10,
            )
            with client.stream(
                "POST",
                url,
                json = payload,
                timeout = prefill_timeout,
                headers = headers,
            ) as response:
                _response_ref[0] = response
                if cancel_event is not None:
                    # Portable mid-stream cancel: the reader polls cancel itself, so
                    # Stop interrupts a stalled read where the watcher's Windows socket
                    # shutdown does not. Pass response to honor the live stall timeout.
                    LlamaCppBackend._install_cancel_aware_read(client, cancel_event, response)
                if cancel_event is not None and cancel_event.is_set():
                    raise _LlamaStreamCancelled
                yield response
                return
        except (httpx.RequestError, RuntimeError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise _LlamaStreamCancelled
            raise
        finally:
            _cancel_closed.set()

    def _server_socket_is_open(self, timeout_s: float = 0.15) -> bool:
        """True if anything still accepts on the server port.

        The listening socket dies with the process, so this tells a live server
        from a dead one without waiting for the child to become reapable.
        """
        port = self._port
        if not port:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = timeout_s):
                return True
        except OSError:
            return False

    def _respawn_if_dead(self) -> bool:
        """Relaunch the llama-server if its process has exited.

        A loaded chat model can be SIGKILL'd mid-session (usually GPU/RAM pressure
        from a training run on the same box), leaving a defunct process while
        ``is_loaded`` still reads True. Replay the last ``load_model`` call to
        recover, returning True once healthy. Serialised on ``_respawn_lock`` so
        many generations hitting the dead server trigger at most one reload.
        """
        # Read outside the lock so a queued caller can tell the replacement from the child
        # its own error came from; otherwise each burns the grace wait below, and that
        # sleep is held under the lock, so the waits serialise.
        served_by = self._process
        with self._respawn_lock:
            proc = self._process
            if proc is None:
                return False
            if self._cancel_event.is_set():
                # unload_model sets this before it kills, so the child can still be
                # accepting. Reporting it healthy would aim the retry at a server
                # that is deliberately going away.
                return False
            if proc is not served_by:
                # Replaced while we queued: this child never served our request.
                return self._healthy
            if proc.poll() is None:
                # Still serving, so the error was transient. Charging it the grace below
                # would cost a second per caller, serialised under this lock.
                if self._server_socket_is_open():
                    return self._healthy
                # A closing server can beat its own exit status: calling it alive returns
                # the stale _healthy and spends the retry on the corpse.
                deadline = time.monotonic() + _RESPAWN_REAP_GRACE_S
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            if proc.poll() is None:
                # Alive: either a concurrent caller already respawned it (healthy), or
                # this connection error wasn't a dead server.
                return self._healthy
            with self._mtp_runtime_fallback_lock:
                if self._mtp_runtime_fallback_in_progress:
                    # An MTP-free reload owns this corpse; replaying the old kwargs
                    # restarts the crashing config and aborts that reload.
                    logger.info("Respawn skipped: an MTP-free reload is already recovering.")
                    return False
            # The RLock lets the load_model below re-enter it.
            with self._serial_load_lock:
                if self._process is not proc:
                    logger.info("Respawn skipped: a newer load is already active.")
                    return self._healthy
                # Snapshot under _lock, the one unload_model holds, so a teardown is
                # either wholly before us (flag set) or wholly after (epoch bumped).
                # _serial_load_lock alone would not exclude it: unload never takes it.
                with self._lock:
                    if self._cancel_event.is_set():
                        logger.info("Respawn skipped: the model was unloaded.")
                        return False
                    intent = self._last_load_intent
                    if intent is None:
                        return False
                    epoch = self._unload_epoch
                    self._healthy = False
                logger.warning(
                    f"llama-server for '{self._model_identifier}' exited "
                    f"(code {proc.returncode}); respawning to recover the session"
                )
                try:
                    started = bool(self.load_model(intent))
                except Exception as exc:
                    logger.error(f"Failed to respawn llama-server: {exc}")
                    return False
                if started and self._unload_epoch != epoch:
                    # An unload landed mid-reload. load_model cleared _cancel_event on
                    # the way in, so the epoch is the only surviving evidence; undo the
                    # replacement rather than leave a model the user stopped running.
                    logger.info("Respawn undone: the model was unloaded during the reload.")
                    self.unload_model()
                    return False
                return started

    @contextlib.contextmanager
    def _open_chat_stream_with_respawn_retry(self, payload: dict, cancel_event):
        """Open a chat stream, respawning a dead llama-server once before streaming.

        Retry only when opening the response fails: once it is open a consumer may
        already have emitted content or tool events, so a replay could duplicate
        output and side effects. ``base_url`` is resolved per attempt because a
        respawn may pick a new port. The budget is one retry per model request, not
        per chat turn, so a long tool loop never discards a completed tool.

        A child dying after the accept but before the headers surfaces as
        ReadError/WriteError/RemoteProtocolError rather than ConnectError, and which
        one differs per OS. llama-server flushes its 200 at slot start, so that window
        is an upload still in flight or a request behind busy slots; a death during
        decode arrives with the response open and is not replayed. Timeouts are
        excluded: the server is slow, not dead, and a replay would spend the
        first-token budget twice.
        """
        for attempt in range(2):
            response_opened = False
            try:
                url = f"{self.base_url}/v1/chat/completions"
                with self._open_stream(url, payload, cancel_event) as opened:
                    response_opened = True
                    yield opened
                    return
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if response_opened:
                    raise
                if self._maybe_recover_from_mtp_crash(exc):
                    raise RuntimeError("Lost connection to llama-server") from exc
                if attempt == 0 and self._respawn_if_dead():
                    logger.warning(
                        "llama-server was unreachable; respawned it and retrying the generation"
                    )
                    continue
                raise

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        seed: Optional[int] = None,
        promote_reasoning_only: bool = True,
        _allow_respawn_retry: bool = True,
    ) -> Generator[Union[str, dict], None, None]:
        """
        Send a chat completion to llama-server and stream tokens back.

        Uses /v1/chat/completions -- llama-server applies the chat template
        and handles vision (multimodal image_url parts) natively.

        Yields cumulative text (matching InferenceBackend's convention).
        """
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

        openai_messages = self._build_openai_messages(messages, image_b64)

        payload = {
            # llama-server applies the chat template itself (#7066).
            "messages": neutralize_control_markup_in_messages(
                openai_messages, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        # Per-request enable_thinking / reasoning_effort / preserve_thinking
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        # Default cap to the model context when known.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        payload["stream_options"] = {"include_usage": True}

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None

        try:
            with self._open_stream(url, payload, cancel_event) as (
                response,
                first_token_deadline,
            ):
                buffer = ""
                has_content_tokens = False
                reasoning_text = ""
                for raw_chunk in self._iter_text_cancellable(
                    response,
                    cancel_event,
                    first_token_deadline = first_token_deadline,
                ):
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if line == "data: [DONE]":
                            if in_thinking:
                                if has_content_tokens:
                                    # Real thinking + content: close the tag
                                    cumulative += "</think>"
                                    yield cumulative
                                else:
                                    # Only reasoning_content, no content:
                                    # model put its whole reply in reasoning
                                    # (e.g. Qwen3 always-think). Show it as
                                    # the main response, not a thinking block.
                                    cumulative = _finalize_reasoning_only_cumulative(
                                        cumulative,
                                        reasoning_text,
                                        _metadata_finish_reason,
                                        promote_reasoning_only,
                                    )
                                    yield cumulative
                            _stream_done = True
                            break  # exit inner while
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                            # Diffusion frame (per-step canvas) from the shim: forward untouched so
                            # the frontend renders it in place. No assistant text, so it never enters
                            # the cumulative content.
                            if data.get("type") == "diffusion_frame":
                                yield data
                                continue
                            # Capture server timings/usage from final chunks.
                            _chunk_timings = data.get("timings")
                            if _chunk_timings:
                                _metadata_timings = _chunk_timings
                            _chunk_usage = data.get("usage")
                            if _chunk_usage:
                                _metadata_usage = _chunk_usage
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _metadata_finish_reason = _fr

                                # Reasoning/thinking tokens: llama-server
                                # sends these as "reasoning_content"; wrap
                                # in <think> tags for the frontend parser.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_text += reasoning
                                    if not in_thinking:
                                        cumulative += "<think>"
                                        in_thinking = True
                                    cumulative += reasoning
                                    yield cumulative

                                token = delta.get("content", "")
                                if token:
                                    has_content_tokens = True
                                    if in_thinking:
                                        cumulative += "</think>"
                                        in_thinking = False
                                    cumulative += token
                                    yield cumulative
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                    if _stream_done:
                        break  # exit outer for
                if _metadata_usage or _metadata_timings or _metadata_finish_reason:
                    _metadata_usage = _backfill_usage_from_timings(
                        _metadata_usage, _metadata_timings
                    )
                    yield {
                        "type": "metadata",
                        # Never None: a finish-only metadata event (no usage,
                        # no timings) would otherwise crash consumers that do
                        # usage.get(...) on the non-streaming paths.
                        "usage": _metadata_usage or {},
                        "timings": _metadata_timings,
                        "finish_reason": _metadata_finish_reason,
                    }

        except _LlamaStreamCancelled:
            return
        except httpx.ConnectError as e:
            # Server already down. If this was an MTP+tensor crash, recover by
            # reloading without MTP (scheduled in the background) and fail this
            # request. Otherwise the server was likely SIGKILL'd by GPU pressure
            # from a concurrent training run: respawn the same config and retry the
            # generation once (bounded by the private flag, no duplicate output).
            if self._maybe_recover_from_mtp_crash(e):
                raise RuntimeError("Lost connection to llama-server")
            if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                logger.warning(
                    "llama-server was unreachable; respawned it and retrying the generation"
                )
                yield from self.generate_chat_completion(
                    messages,
                    image_b64 = image_b64,
                    temperature = temperature,
                    top_p = top_p,
                    top_k = top_k,
                    min_p = min_p,
                    max_tokens = max_tokens,
                    repetition_penalty = repetition_penalty,
                    presence_penalty = presence_penalty,
                    stop = stop,
                    cancel_event = cancel_event,
                    enable_thinking = enable_thinking,
                    reasoning_effort = reasoning_effort,
                    preserve_thinking = preserve_thinking,
                    seed = seed,
                    promote_reasoning_only = promote_reasoning_only,
                    _allow_respawn_retry = False,
                )
                return
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            # Died mid-generation: recover MTP, re-raise unchanged for this request.
            self._maybe_recover_from_mtp_crash(e)
            raise

    # ── Tool-calling agentic loop ──────────────────────────────

    def generate_chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
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
        seed: Optional[int] = None,
        disable_parallel_tool_use: bool = False,
        confirm_tool_calls: bool = False,
        bypass_permissions: bool = False,
        permission_mode: Optional[str] = None,
        promote_reasoning_only: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        permission_mode: "ask" confirms every call (with confirm_tool_calls),
        "auto" only pauses calls detected as potentially unsafe, "off" never
        pauses (sandbox stays on), "full" is the same as bypass_permissions.
        Unset/unknown behaves as "ask".

        Yields dicts:
          {"type": "status", "text": "Searching: ..."/"Reading: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        from core.inference.tool_stream_exec import accepts_output_callback, stream_tool_execution
        from core.inference.tools import (
            build_rag_autoinject,
            execute_tool,
            has_text_only_provisional_card,
            is_always_safe_tool,
            is_high_risk_tool_call,
        )

        # "full" and bypass_permissions are the same switch, whichever arrives
        # first wins. "off" keeps the sandbox but never prompts. Unset defaults to
        # "auto"; unknown falls back to the stricter "ask". An explicit
        # confirm_tool_calls=True with no mode is already resolved to "ask" at the
        # request layer, so it never arrives here as an ambiguous unset.
        if permission_mode == "full":
            bypass_permissions = True
        elif bypass_permissions:
            permission_mode = "full"
        elif permission_mode is None:
            permission_mode = "auto"
        elif permission_mode not in ("ask", "auto", "off"):
            permission_mode = "ask"

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        conversation = list(messages)

        # Forced first-pass RAG so a doc question doesn't lose to web_search. Emits
        # the same tool card + citations a real call would. Skip it only when a
        # retrieval call would actually prompt (ask mode); auto never gates the
        # safe search_knowledge_base tool, so retrieval must still run there.
        # off never prompts either, so it also keeps first-pass retrieval.
        _skip_autoinject = (
            confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
        )
        _auto = None if _skip_autoinject else build_rag_autoinject(conversation, rag_scope)
        if _auto:
            for _ev in _auto["events"]:
                yield _ev
            conversation.extend(_auto["messages"])

        _accumulated_completion_tokens = 0
        _accumulated_predicted_ms = 0.0
        _accumulated_predicted_n = 0
        # GGUF buffers reasoning; emit server-side timing before answer text.
        _reasoning_started_at: Optional[float] = None
        _reasoning_summary_emitted = False

        # Gate telling a genuine NAME[ARGS] rehearsal from inactive-name prose; built from the
        # ORIGINAL tools list so a spent one-shot still reads as a tool name. None = no gate.
        _enabled_names_gate = set(_gguf_active_tool_names(tools)) if tools else None
        # Detection must see the same names as the strip gate (ORIGINAL list, incl. a spent
        # one-shot), else its repeat is stripped but never drained and the turn ends blank.
        _detect_tools = list(tools or [])

        def _reasoning_summary_event(started_at: float) -> dict:
            return {
                "type": "reasoning_summary",
                "duration_ms": round((time.monotonic() - started_at) * 1000.0),
            }

        # Enabled-name gate for the markerless Gemma strip (disabled/example
        # names stay visible). Set per iteration; None = pre-loop name-agnostic.
        _enabled_tool_names = None

        def _strip_tool_markup(
            text: str,
            *,
            final: bool = False,
            force: bool = False,
        ) -> str:
            if not (auto_heal_tool_calls or force):
                return text
            # Delegate to the shared parser-side strip so the GGUF cleanup covers every family the
            # parser promotes (Llama <|python_tag|>, Mistral [TOOL_CALLS], bare rehearsal, function
            # XML, Gemma) and stays aligned with detection; tool_healing's strip omits the loop-only
            # forms (python_tag / Mistral name) and would leak them into display.
            return _shared_strip_tool_markup(
                text, final = final, enabled_tool_names = _enabled_names_gate
            )

        def _strip_tool_markup_streaming(text: str, *, force: bool = False) -> str:
            if not (auto_heal_tool_calls or force):
                return text

            def _seg(segment: str, is_last: bool) -> str:
                # Same scan order as the parser's _strip_segment (seg_final -> is_last): balanced
                # strips first (nested JSON removed whole; literal markup inside a value is that
                # call's data), then the guarded function-XML / GLM scans, then the regex arms
                # (DeepSeek / Kimi / closed forms). EOS-anchored tail arms run only on the last
                # segment (a bare ``foo[ARGS]`` before <think> is prose). Rehearsal + markerless
                # strips are name-gated on the ORIGINAL list (strip/detect aligned).
                seg = _strip_mistral_closed_calls(segment)
                seg = _strip_bracket_tag_calls(seg, enabled_tool_names = _enabled_names_gate)
                if is_last:
                    seg = _strip_gemma_wrapperless_calls(seg, _enabled_names_gate)
                seg = _strip_function_xml_calls(seg, final = is_last)
                seg = _strip_glm_calls(seg, final = is_last)
                pats = _PARSER_TOOL_ALL_PATS if is_last else _PARSER_TOOL_CLOSED_PATS
                for pat in pats:
                    seg = pat.sub("", seg)
                if is_last:
                    seg = apply_tool_strip_patterns(
                        seg, [_REHEARSAL_TAIL_STRIP_RE], enabled_tool_names = _enabled_names_gate
                    )
                return seg

            # Preserve think blocks verbatim (a rehearsed call inside one must not be deleted).
            return strip_outside_think(text, _seg)

        def _build_metadata_event(usage, timings, finish_reason):
            """Final usage+timings metadata event for the given pass, merging its
            usage/timings with the running cross-iteration accumulators. None when
            there is nothing to report."""
            _fu = _backfill_usage_from_timings(usage, timings) or {}
            _fp = _fu.get("prompt_tokens", 0)
            _tc = _fu.get("completion_tokens", 0) + _accumulated_completion_tokens
            if not (usage or timings or _accumulated_completion_tokens or finish_reason):
                return None
            _mt = dict(timings) if timings else {}
            if _accumulated_predicted_ms or _accumulated_predicted_n:
                _mt["predicted_ms"] = _mt.get("predicted_ms", 0) + _accumulated_predicted_ms
                _mt["predicted_n"] = _mt.get("predicted_n", 0) + _accumulated_predicted_n
                if _mt["predicted_ms"] > 0:
                    _mt["predicted_per_second"] = _mt["predicted_n"] / (
                        _mt["predicted_ms"] / 1000.0
                    )
            _usage = {
                "prompt_tokens": _fp,
                "completion_tokens": _tc,
                "total_tokens": _fp + _tc,
            }
            # Preserve KV-cache hit details (cached_tokens) so the tool path
            # reports them like the standard non-tool path does, not always 0.
            if _fu.get("prompt_tokens_details"):
                _usage["prompt_tokens_details"] = _fu["prompt_tokens_details"]
            return {
                "type": "metadata",
                "usage": _usage,
                "timings": _mt,
                "finish_reason": finish_reason,
            }

        def _flush_reasoning_and_buffer():
            """Close a live-streamed <think> block (or emit the buffered reasoning
            as one block if it never streamed), then append the held
            content_buffer to the cumulative display text."""
            nonlocal cumulative_display, in_thinking
            if in_thinking:
                cumulative_display += "</think>"
                in_thinking = False
            elif reasoning_accum:
                cumulative_display += "<think>" + reasoning_accum + "</think>"
            cumulative_display += content_buffer

        def _close_streamed_think() -> bool:
            """Close a live-streamed <think> before a tool call drains, so
            consumers without a reasoning extractor (Anthropic) get a balanced
            block. Returns True when the caller should yield the result."""
            nonlocal cumulative_display, in_thinking, _last_emitted
            if not in_thinking:
                return False
            cumulative_display += "</think>"
            in_thinking = False
            if len(cumulative_display) > len(_last_emitted) and not _suppress_visible_output:
                _last_emitted = cumulative_display
                return True
            return False

        def _looks_like_enabled_bare_json(text: str, enabled_tool_names: set) -> bool:
            """True when ``text`` opens with an ENABLED markerless bare-JSON call; an ordinary JSON answer returns False."""
            probe = strip_llama3_leading_sentinels(text.lstrip())
            if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
                return False
            return strip_leading_bare_json_call(probe, enabled_tool_names) != probe

        # Sanitized at construction, not at each gate: the controller authorizes execution,
        # and llama-server's structured delta.tool_calls path reaches prepare_call without
        # _enabled_tool_names at all. A tool dropped for unsafe markup is absent from the
        # prompt, so it must leave the controller too or it stays executable by name (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_tool_descriptions as _neutralize_tool_descriptions,
        )

        tool_controller = ToolLoopController(
            tools = _neutralize_tool_descriptions(tools, None, self.markup_profile),
            auto_heal_tool_calls = auto_heal_tool_calls,
        )

        def _tool_succeeded(tool_name: str) -> bool:
            key_prefix = f"{tool_name}:"
            return any(
                record.executed and not record.is_error and record.key.startswith(key_prefix)
                for record in tool_controller.history
            )

        _MAX_BUFFER_CHARS = 32
        # Hold a leading ``{`` well past the 32-char XML cap until it balances (mirrors safetensors).
        _MAX_BARE_JSON_BUFFER = 16384
        _append_budget_exhausted_nudge = True
        # RAG: cap knowledge-base searches per assistant turn. The controller is
        # tool-agnostic, so this gate stays in the loop.
        _kb_search_count = 0

        # ── Re-prompt on plan-without-action ─────────────────
        # Model describes intent without calling a tool: re-prompt once. A
        # direct answer ("4", "Hello!") won't match. Pattern shared with the
        # safetensors loop (tool_call_parser.INTENT_SIGNAL).
        _reprompt_count = 0
        # Budgeted apart from _reprompt_count so a pre-tool nudge can't spend it.
        _post_tool_reprompts = 0
        # Text that triggered the last nudge; if the retry restates it, stop.
        _last_reprompt_text = ""
        # Gates ``max_tool_iterations`` on real tool turns (not the enlarged range) so reserved
        # re-prompt slots don't extend the budget. Mirrors the safetensors guard.
        _tool_iters_done = 0
        _forced_tool_call_pending = False

        # Reserve extra iterations for re-prompts so they don't consume the
        # caller's tool-call budget; only when tool iterations are allowed.
        _extra = _MAX_REPROMPTS + 1 if max_tool_iterations > 0 else 0
        # Owned by this request and dropped with it: the sweep below re-reads every earlier
        # turn on each iteration, and the rewrite is a function of the text alone (#7066).
        from core.inference.chat_template_helpers import sweep_cache as _sweep_cache

        _markup_cache = _sweep_cache()
        for iteration in range(max_tool_iterations + _extra):
            if cancel_event is not None and cancel_event.is_set():
                return
            # Whether this turn ran a tool; a no-op-only turn stays False and doesn't consume budget.
            _turn_executed_real_tool = False

            active_tools = tool_controller.active_tools()
            if not active_tools:
                _append_budget_exhausted_nudge = False
                break
            from core.inference.chat_template_helpers import neutralize_tool_descriptions

            # An MCP server's description and inputSchema are remote text the template renders
            # into the system turn (#7066). Computed above the gate: a tool dropped for unsafe
            # markup is absent from the prompt and must be absent from what we EXECUTE too,
            # else the model names it and the raw gate lets the call through.
            safe_tools = neutralize_tool_descriptions(
                active_tools, _markup_cache, self.markup_profile
            )
            # Gate the markerless bare-JSON form on enabled names so an ordinary JSON answer isn't misread as a call.
            _enabled_tool_names = {
                (tool.get("function") or {}).get("name")
                for tool in safe_tools
                if (tool.get("function") or {}).get("name")
            }
            # Shared signal tuple so GGUF BUFFERING wakes on every format the parser knows (like safetensors).
            _tool_xml_signals = _SHARED_TOOL_XML_SIGNALS

            # Build payload -- stream: True so we detect tool signals
            # in the first 1-2 chunks without a non-streaming penalty.
            from core.inference.chat_template_helpers import (
                neutralize_control_markup_in_messages,
            )

            payload = {
                # Re-run every iteration: tool results land in ``conversation`` as the
                # loop goes, and a forged turn in one would render for real (#7066).
                "messages": neutralize_control_markup_in_messages(
                    conversation, _markup_cache, self.markup_profile
                ),
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
            }
            # As in the passthrough builder: if every name carried markup the catalog is
            # now empty, and "tools": [] would still advertise tool use.
            if safe_tools:
                payload["tools"] = safe_tools
                payload["tool_choice"] = "auto"
            _reasoning_kw = self._request_reasoning_kwargs(
                enable_thinking, reasoning_effort, preserve_thinking
            )
            if _reasoning_kw is not None:
                payload["chat_template_kwargs"] = _reasoning_kw
            payload["max_tokens"] = (
                max_tokens
                if max_tokens is not None
                else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
            )
            if stop:
                payload["stop"] = stop
            if seed is not None:
                payload["seed"] = seed

            try:
                # ── Speculative buffer state machine ──────────────────
                # BUFFERING: accumulate content, check for tool signals
                # STREAMING: no tool detected, yield tokens to caller
                # DRAINING:  tool signal found, silently consume rest
                _S_BUFFERING = 0
                _S_STREAMING = 1
                _S_DRAINING = 2

                detect_state = _S_BUFFERING
                content_buffer = ""  # Raw content held during BUFFERING
                content_accum = ""  # All content tokens (for tool parsing)
                reasoning_accum = ""
                # Time each reasoning pass so final answers can replace tool timing.
                _reasoning_started_at = None
                _reasoning_summary_emitted = False
                _deferred_reasoning_summary = None
                cumulative_display = ""  # Cumulative yielded text (with <think>)
                in_thinking = False
                has_content_tokens = False
                tool_calls_acc = {}  # Structured delta.tool_calls fragments
                has_structured_tc = False
                _iter_usage = None
                _iter_timings = None
                _iter_finish_reason = None
                _stream_done = False
                _last_emitted = ""
                # Provisional tool_start cards already shown, keyed by tool_call_id.
                provisional_started_tool_calls: dict[str, str] = {}
                resolved_provisional_tool_call_ids: set[str] = set()
                _suppress_visible_output = _forced_tool_call_pending
                # Cards that already got their first tool_args event; later
                # fragments stream individually so a big payload isn't a dead spinner.
                arg_streamed_tool_call_ids: set[str] = set()
                # TEXT tool-call path: a committed call's raw text streams to a
                # provisional card under the parser's first-call id ("call_0"), so
                # the final tool_start reconciles in place.
                _text_args_call_start = -1
                _text_args_streamed_upto = -1
                _text_args_id = ""
                _text_args_name = ""
                _confirm_gated_iteration = bool(confirm_tool_calls) and not bypass_permissions

                with self._open_chat_stream_with_respawn_retry(payload, cancel_event) as (
                    response,
                    first_token_deadline,
                ):
                    raw_buf = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response,
                        cancel_event,
                        first_token_deadline = first_token_deadline,
                    ):
                        raw_buf += raw_chunk
                        while "\n" in raw_buf:
                            line, raw_buf = raw_buf.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                # Flush thinking state for STREAMING
                                if detect_state == _S_STREAMING and in_thinking:
                                    if has_content_tokens:
                                        cumulative_display += "</think>"
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": _strip_tool_markup(
                                                    cumulative_display,
                                                    final = True,
                                                ),
                                            }
                                    else:
                                        cumulative_display = _finalize_reasoning_only_cumulative(
                                            cumulative_display,
                                            reasoning_accum,
                                            _iter_finish_reason,
                                            promote_reasoning_only,
                                        )
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                _ct = chunk_data.get("timings")
                                if _ct:
                                    _iter_timings = _ct
                                _cu = chunk_data.get("usage")
                                if _cu:
                                    _iter_usage = _cu

                                choices = chunk_data.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _iter_finish_reason = _fr

                                # ── Structured tool_calls ──
                                tc_deltas = delta.get("tool_calls")
                                if tc_deltas:
                                    # Preserve any visible preface before draining
                                    # the structured tool call.
                                    has_structured_tc = True
                                    detect_state = _S_DRAINING
                                    # Close the reasoning prefix before the tool card
                                    # (mirrors the is_match path).
                                    if _close_streamed_think():
                                        yield {"type": "content", "text": cumulative_display}
                                    for tc_d in tc_deltas:
                                        idx = tc_d.get("index", 0)
                                        if idx not in tool_calls_acc:
                                            tool_calls_acc[idx] = {
                                                "id": tc_d.get("id", f"call_{idx}"),
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                        elif tc_d.get("id"):
                                            # Update ID if a real one
                                            # arrives on a later delta.
                                            tool_calls_acc[idx]["id"] = tc_d["id"]
                                        func = tc_d.get("function", {})
                                        if func.get("name"):
                                            tool_calls_acc[idx]["function"]["name"] += func["name"]
                                        if func.get("arguments"):
                                            tool_calls_acc[idx]["function"]["arguments"] += func[
                                                "arguments"
                                            ]
                                        current_name = tool_calls_acc[idx]["function"].get(
                                            "name", ""
                                        )
                                        fallback_id = f"call_{idx}"
                                        current_id = tool_calls_acc[idx].get("id", fallback_id)
                                        already_started = (
                                            current_id in provisional_started_tool_calls
                                        )
                                        # Empty/synthetic ids cannot reconcile with real starts.
                                        has_real_id = bool(current_id) and current_id != fallback_id
                                        # Show one early card per eligible streamed tool call.
                                        _is_completed_one_shot = (
                                            current_name == "render_html"
                                            and _tool_succeeded("render_html")
                                        )
                                        # render_html is one-shot.
                                        _one_shot_already_provisional = (
                                            current_name == "render_html"
                                            and "render_html"
                                            in provisional_started_tool_calls.values()
                                        )
                                        # Later parallel cards only reconcile when parallel use is enabled.
                                        # In auto mode an always-safe tool (render_html) never
                                        # prompts, so it must stream its early card too; mirror
                                        # that here instead of gating on the raw confirm flag.
                                        _confirm_gated = (
                                            confirm_tool_calls
                                            and not bypass_permissions
                                            and not (
                                                permission_mode == "auto"
                                                and is_always_safe_tool(current_name)
                                            )
                                            # A text-preview card still streams while gated;
                                            # hiding it blanks the chat.
                                            and not has_text_only_provisional_card(current_name)
                                        )
                                        # Keep small-argument tools on the normal path.
                                        _args_len = len(
                                            tool_calls_acc[idx]["function"].get("arguments", "")
                                        )
                                        _payload_is_large = (
                                            current_name == "render_html"
                                            or _args_len >= _PROVISIONAL_ARGS_MIN_CHARS
                                        )
                                        if (
                                            current_name
                                            and (idx == 0 or not disable_parallel_tool_use)
                                            and has_real_id
                                            and not already_started
                                            and not _is_completed_one_shot
                                            and not _one_shot_already_provisional
                                            and not _confirm_gated
                                            and _payload_is_large
                                            and any(
                                                (tool.get("function") or {}).get("name")
                                                == current_name
                                                for tool in active_tools
                                            )
                                        ):
                                            provisional_started_tool_calls[current_id] = (
                                                current_name
                                            )
                                            yield {
                                                "type": "tool_start",
                                                "tool_name": current_name,
                                                "tool_call_id": current_id,
                                                "arguments": {},
                                                "provenance": tool_event_provenance(
                                                    provisional = True,
                                                ),
                                            }
                                        # Stream argument text so the UI shows the code being
                                        # written: first event the backlog, later the fragment.
                                        # Display only; accumulator untouched.
                                        if current_id in provisional_started_tool_calls:
                                            if current_id not in arg_streamed_tool_call_ids:
                                                arg_streamed_tool_call_ids.add(current_id)
                                                _args_backlog = tool_calls_acc[idx]["function"].get(
                                                    "arguments", ""
                                                )
                                                if _args_backlog:
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": current_id,
                                                        "tool_name": current_name,
                                                        "text": _args_backlog,
                                                    }
                                            elif func.get("arguments"):
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": current_id,
                                                    "tool_name": current_name,
                                                    "text": func["arguments"],
                                                }
                                    continue

                                # ── Reasoning tokens ──
                                # Stream live except while DRAINING: reasoning is
                                # orthogonal to tool detection (content_buffer
                                # only), and the route resets prev_text on
                                # tool_start, so the <think> block stays a
                                # monotonic prefix like the no-tool path.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if _reasoning_started_at is None:
                                        _reasoning_started_at = time.monotonic()
                                    reasoning_accum += reasoning
                                    if detect_state != _S_DRAINING:
                                        if not in_thinking:
                                            cumulative_display += "<think>"
                                            in_thinking = True
                                        cumulative_display += reasoning
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }

                                # ── Content tokens ──
                                token = delta.get("content", "")
                                if token:
                                    # First answer token ends reasoning.
                                    if (
                                        _reasoning_started_at is not None
                                        and not _reasoning_summary_emitted
                                    ):
                                        _reasoning_summary_emitted = True
                                        _summary = _reasoning_summary_event(_reasoning_started_at)
                                        if _suppress_visible_output:
                                            _deferred_reasoning_summary = _summary
                                        else:
                                            yield _summary
                                    has_content_tokens = True
                                    content_accum += token

                                    if detect_state == _S_DRAINING:
                                        # Accumulate silently for parsing, but stream the drained
                                        # TEXT call to a provisional card. Gated on an enabled-name
                                        # sniff + size floor so prose/small calls spawn no pane; id
                                        # matches the first call so the final tool_start reconciles.
                                        if not has_structured_tc and _text_args_call_start >= 0:
                                            if not _text_args_id:
                                                _call_text = content_accum[_text_args_call_start:]
                                                _sniffed = _sniff_text_tool_name(
                                                    _call_text, _enabled_tool_names
                                                )
                                                # Structured-path rule: gated calls
                                                # stream only from a text-preview card.
                                                if (
                                                    _sniffed
                                                    and not (
                                                        _confirm_gated_iteration
                                                        and not has_text_only_provisional_card(
                                                            _sniffed
                                                        )
                                                    )
                                                    and (
                                                        _sniffed == "render_html"
                                                        or len(_call_text)
                                                        >= _PROVISIONAL_ARGS_MIN_CHARS
                                                    )
                                                ):
                                                    _text_args_id = "call_0"
                                                    _text_args_name = _sniffed
                                                    if (
                                                        _text_args_id
                                                        not in provisional_started_tool_calls
                                                    ):
                                                        provisional_started_tool_calls[
                                                            _text_args_id
                                                        ] = _sniffed
                                                        yield {
                                                            "type": "tool_start",
                                                            "tool_name": _sniffed,
                                                            "tool_call_id": _text_args_id,
                                                            "arguments": {},
                                                            "provenance": tool_event_provenance(
                                                                provisional = True,
                                                            ),
                                                        }
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": _text_args_id,
                                                        "tool_name": _sniffed,
                                                        "text": _call_text,
                                                    }
                                                    _text_args_streamed_upto = len(content_accum)
                                            elif len(content_accum) > _text_args_streamed_upto:
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": _text_args_id,
                                                    "tool_name": _text_args_name,
                                                    "text": content_accum[
                                                        _text_args_streamed_upto:
                                                    ],
                                                }
                                                _text_args_streamed_upto = len(content_accum)

                                    elif detect_state == _S_STREAMING:
                                        if in_thinking:
                                            cumulative_display += "</think>"
                                            in_thinking = False
                                        cumulative_display += token
                                        cleaned = _strip_tool_markup_streaming(cumulative_display)
                                        # Hold a trailing bare active-tool-name (split rehearsal)
                                        # until [ARGS] arrives; released by later prose or stream end.
                                        _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                        _emit = (
                                            cleaned[: len(cleaned) - _hold] if _hold else cleaned
                                        )
                                        if len(_emit) > len(_last_emitted):
                                            _last_emitted = _emit
                                            if not _suppress_visible_output:
                                                yield {
                                                    "type": "content",
                                                    "text": _emit,
                                                }

                                    elif detect_state == _S_BUFFERING:
                                        content_buffer += token
                                        stripped_buf = content_buffer.lstrip()
                                        if not stripped_buf:
                                            continue

                                        # Bracket tags arrive mid-buffer, so substring-check too;
                                        # ``[ARGS]`` counts only as a regex-matched NAME[ARGS].
                                        is_prefix = False
                                        is_match = False
                                        for sig in _tool_xml_signals:
                                            if stripped_buf.startswith(sig):
                                                is_match = True
                                                break
                                            if sig.startswith(stripped_buf):
                                                is_prefix = True
                                                break
                                            if sig == "[ARGS]":
                                                # Active NAME[ARGS] only; inactive-name prose
                                                # is gated out, not drained/parsed.
                                                if (
                                                    _gguf_rehearsal_signal_pos(
                                                        stripped_buf, _detect_tools
                                                    )
                                                    >= 0
                                                ):
                                                    is_match = True
                                                    break
                                            elif sig.startswith("[") and sig in stripped_buf:
                                                is_match = True
                                                break

                                        # Split rehearsal: hold the bare name until
                                        # its [ARGS] arrives and matches above.
                                        is_rehearsal_prefix = False
                                        if (
                                            not is_match
                                            and not is_prefix
                                            and _is_rehearsal_prefix(stripped_buf, _detect_tools)
                                        ):
                                            is_prefix = True
                                            is_rehearsal_prefix = True

                                        # Signal-less call shapes (mirror the safetensors
                                        # loop): Llama-3.2 bare {"name":..} and Gemma
                                        # call:NAME{...} would otherwise stream raw.
                                        _hold_buffer = False
                                        # Whole buffer is the call (no visible prefix) -- drain silently.
                                        _drain_silently = False
                                        if not is_match and not is_prefix:
                                            _bare = strip_llama3_leading_sentinels(stripped_buf)
                                            if _bare.startswith("{"):
                                                if _balanced_brace_end(_bare, 0) is None:
                                                    if len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    elif _looks_like_enabled_bare_json(
                                                        _bare, _enabled_tool_names
                                                    ):
                                                        # Oversized still-open enabled call: drain
                                                        # rather than leak; a giant ordinary JSON
                                                        # answer still streams.
                                                        _drain_silently = True
                                                elif self._parse_tool_calls_from_text(
                                                    content_buffer,
                                                    allow_incomplete = auto_heal_tool_calls,
                                                    enabled_tool_names = _enabled_tool_names,
                                                ):
                                                    _drain_silently = True
                                            elif (
                                                "call:".startswith(stripped_buf)
                                                or _GEMMA_BARE_TC_PREFIX_RE.match(stripped_buf)
                                                is not None
                                                or _GEMMA_BARE_TC_RE.match(stripped_buf) is not None
                                            ):
                                                # Whitespace-tolerant like the parser.
                                                if _GEMMA_BARE_TC_RE.match(stripped_buf):
                                                    _drain_silently = True
                                                elif len(stripped_buf) < _MAX_BUFFER_CHARS:
                                                    _hold_buffer = True

                                        if _drain_silently:
                                            # The buffered content IS the call; drain it
                                            # without yielding. A live <think> prefix is
                                            # separate from it -- close that.
                                            detect_state = _S_DRAINING
                                            # Call text begins at the held buffer
                                            # (live arg display only; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                            if _close_streamed_think():
                                                yield {
                                                    "type": "content",
                                                    "text": cumulative_display,
                                                }
                                        elif is_match:
                                            # Tool signal -- flush any visible
                                            # prefix before DRAINING so the
                                            # route sends it before tool_start.
                                            # Use the final strip (all families incl. Llama
                                            # <|python_tag|> / Mistral name): the buffer holds
                                            # the whole call, so a streaming closed-only strip
                                            # would leak its open-ended markup as display text.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                                final = True,
                                                force = True,
                                            )
                                            if len(cleaned) > len(_last_emitted):
                                                _last_emitted = cleaned
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": cleaned,
                                                    }
                                            detect_state = _S_DRAINING
                                            # Live-arg display starts at the held buffer
                                            # (visible prefix flushed above; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                        elif _hold_buffer or (
                                            is_prefix
                                            and (
                                                is_rehearsal_prefix
                                                or len(stripped_buf) < _MAX_BUFFER_CHARS
                                            )
                                        ):
                                            # A rehearsal prefix is self-bounded; the buffer
                                            # cap must not cut long MCP names short.
                                            pass  # keep buffering
                                        else:
                                            # Not a tool -- flush buffer
                                            detect_state = _S_STREAMING
                                            # Flush reasoning accumulated
                                            # during BUFFERING.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                            )
                                            # Same trailing-name hold as STREAMING for this
                                            # first flush out of BUFFERING.
                                            _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                            _emit = (
                                                cleaned[: len(cleaned) - _hold]
                                                if _hold
                                                else cleaned
                                            )
                                            if len(_emit) > len(_last_emitted):
                                                _last_emitted = _emit
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": _emit,
                                                    }

                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                        if _stream_done:
                            break  # exit outer for

                # ── Resolve BUFFERING at stream end ──
                if detect_state == _S_BUFFERING:
                    stripped_buf = content_buffer.lstrip()
                    # A held bare-JSON fragment has no XML signal; route it to DRAINING (the signal-only
                    # gate below would flush the raw JSON to the user).
                    _bare_eos = strip_llama3_leading_sentinels(stripped_buf)
                    # Gate on enabled names so an ordinary JSON answer isn't routed to DRAINING and dropped.
                    _is_bare_tc = bool(active_tools) and _looks_like_enabled_bare_json(
                        _bare_eos, _enabled_tool_names
                    )
                    if stripped_buf and _gguf_has_genuine_tool_signal(
                        stripped_buf, _tool_xml_signals, _detect_tools
                    ):
                        detect_state = _S_DRAINING
                    elif _is_bare_tc:
                        detect_state = _S_DRAINING
                    elif content_accum or reasoning_accum:
                        detect_state = _S_STREAMING
                        if content_buffer:
                            # Flush reasoning first.
                            _flush_reasoning_and_buffer()
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    ),
                                }
                        elif reasoning_accum and not has_content_tokens:
                            # Reasoning-only reply: show it as the main response,
                            # not a thinking block (mirrors the no-tool path; the
                            # route's extractor closes the streamed <think>).
                            if _reasoning_started_at is not None and not _reasoning_summary_emitted:
                                _reasoning_summary_emitted = True
                                _summary = _reasoning_summary_event(_reasoning_started_at)
                                if _suppress_visible_output:
                                    _deferred_reasoning_summary = _summary
                                else:
                                    yield _summary
                            cumulative_display = _finalize_reasoning_only_cumulative(
                                cumulative_display,
                                reasoning_accum,
                                _iter_finish_reason,
                                promote_reasoning_only,
                            )
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": cumulative_display,
                                }
                    else:
                        # Held buffer was no tool signal and no enabled bare-JSON call: a leading ``{`` is an
                        # ordinary JSON answer and must be shown; any other partial-markup prefix is dropped.
                        _held = strip_llama3_leading_sentinels(content_buffer.lstrip())
                        if _held.startswith("{") and not _suppress_visible_output:
                            yield {"type": "content", "text": _held}
                        return

                # ── STREAMING path: no tool call ──
                if detect_state == _S_STREAMING:
                    # Safety net: re-parse the full content for tool calls. The
                    # route layer resets prev_text on tool_start, so post-tool
                    # synthesis streams correctly even if content was emitted
                    # before the tool XML.
                    # Unconditional (not gated on _tool_xml_signals): bare-JSON and Gemma wrapper-less
                    # calls carry no XML signal, so a signal gate would let them slip past.
                    _safety_tc = self._parse_tool_calls_from_text(
                        content_accum,
                        allow_incomplete = auto_heal_tool_calls,
                        enabled_tool_names = _enabled_tool_names,
                    )
                    if not _safety_tc:
                        # ── Re-prompt on plan-without-action ──
                        # Intent described without a tool call: nudge it to act. Up
                        # to _MAX_REPROMPTS times, only on short responses with intent
                        # signals -- "4" or "Hello!" won't trigger it. Uses content,
                        # else reasoning text (reasoning-only stalls).
                        _stripped = content_accum.strip()
                        if not _stripped:
                            _stripped = reasoning_accum.strip()
                        _render_html_already_done_intent = _tool_succeeded(
                            "render_html"
                        ) and re.search(
                            r"(?i)\brender[_\s-]?html\b",
                            _stripped,
                        )
                        # A post-tool stall still deserves a nudge, but each retry
                        # re-runs tools, so allow only one. RAG autoinject never lands
                        # in history, so _auto keeps a doc-grounded turn from reading
                        # as pre-tool (mirrors safetensors rag_autoinjected).
                        _already_acted = bool(_auto) or any(
                            record.executed for record in tool_controller.history
                        )
                        if _already_acted:
                            _reprompt_used, _reprompt_cap = _post_tool_reprompts, 1
                        else:
                            _reprompt_used, _reprompt_cap = _reprompt_count, _MAX_REPROMPTS
                        # None keeps the default-on re-prompt; False disables it.
                        if (
                            auto_heal_tool_calls
                            and (nudge_tool_calls is None or nudge_tool_calls)
                            and active_tools
                            and not _render_html_already_done_intent
                            and _reprompt_used < _reprompt_cap
                            and not _is_reprompt_repeat(_stripped, _last_reprompt_text)
                            and _is_short_intent_without_action(_stripped)
                        ):
                            _reprompt_count += 1
                            if _already_acted:
                                _post_tool_reprompts += 1
                            _last_reprompt_text = _stripped
                            logger.info(
                                f"Re-prompt {_reprompt_used + 1}/{_reprompt_cap}: "
                                f"model responded without calling tools "
                                f"({len(_stripped)} chars)"
                            )
                            conversation.append(
                                {
                                    "role": "assistant",
                                    "content": _stripped,
                                }
                            )
                            available_tool_names = [
                                (tool.get("function") or {}).get("name")
                                for tool in active_tools
                                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                            ]
                            available_tool_names = [name for name in available_tool_names if name]
                            tool_hint = " or ".join(available_tool_names) or "an available tool"
                            _forced_tool_call_pending = True
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": _reprompt_to_act_message(tool_hint),
                                }
                            )
                            # Accumulate tokens and timing from this iteration.
                            _fu_r = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _accumulated_completion_tokens += _fu_r.get("completion_tokens", 0)
                            _it_r = _iter_timings or {}
                            _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                            # Blank first (the route resets its text cursor only on an
                            # empty status), then the badge so the retry is not a hang.
                            yield {"type": "status", "text": ""}
                            yield {"type": "status", "text": _NUDGE_TOOL_CALLS_STATUS}
                            continue

                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                            if not _should_suppress_forced_no_tool_output(
                                _stripped,
                                _last_reprompt_text,
                            ):
                                if cumulative_display:
                                    forced_visible_text = _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    )
                                elif content_accum:
                                    forced_visible_text = _strip_tool_markup(
                                        content_accum,
                                        final = True,
                                    )
                                else:
                                    forced_visible_text = reasoning_accum
                                if forced_visible_text:
                                    yield {
                                        "type": "content",
                                        "text": forced_visible_text,
                                    }
                                    if _deferred_reasoning_summary is not None:
                                        yield _deferred_reasoning_summary
                        elif not _suppress_visible_output:
                            # Turn ended as a plain answer (no [ARGS] followed): the held
                            # rehearsal tail is real prose, release it.
                            _final_clean = _strip_tool_markup_streaming(cumulative_display)
                            if len(_final_clean) > len(_last_emitted):
                                yield {"type": "content", "text": _final_clean}

                        # Content was already streamed.  Yield metadata.
                        yield {"type": "status", "text": ""}
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                    # Safety net caught tool XML -- treat as tool call.
                    tool_calls = _safety_tc
                    content_text = _strip_tool_markup(
                        content_accum,
                        final = True,
                        force = True,
                    )
                    logger.info(
                        f"Safety net: parsed {len(tool_calls)} tool call(s) from streamed content"
                    )
                else:
                    # ── DRAINING path: assemble tool_calls ──
                    tool_calls = None
                    content_text = content_accum
                    if has_structured_tc:
                        # Drop incomplete fragments (e.g. from max_tokens
                        # truncation or disconnect).
                        tool_calls = [
                            tool_calls_acc[i]
                            for i in sorted(tool_calls_acc)
                            if (tool_calls_acc[i].get("function", {}).get("name", "").strip())
                        ] or None
                    if not tool_calls:
                        # Unconditional re-parse: we only reach DRAINING when the buffer looked like a
                        # call, and bare-JSON / Gemma wrapper-less calls carry no XML signal to gate on.
                        tool_calls = self._parse_tool_calls_from_text(
                            content_accum,
                            allow_incomplete = auto_heal_tool_calls,
                            enabled_tool_names = _enabled_tool_names,
                        )
                    if tool_calls and not has_structured_tc:
                        content_text = _strip_tool_markup(
                            content_text,
                            final = True,
                            force = True,
                        )
                        # ``_strip_tool_markup`` only knows XML; also drop a leading bare-JSON call so the
                        # executed call isn't replayed as text or next-turn history.
                        content_text = strip_leading_bare_json_call(
                            content_text, _enabled_tool_names
                        )
                    if tool_calls:
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s) from "
                            f"{'structured delta' if has_structured_tc else 'content text'}"
                        )
                    if not tool_calls:
                        # DRAINING but no tool calls (false positive): close any
                        # provisional cards (a sniff can open one whose call never
                        # parses); without a tool_end the card spins forever.
                        for _pid, _pname in provisional_started_tool_calls.items():
                            if _pid not in resolved_provisional_tool_call_ids:
                                resolved_provisional_tool_call_ids.add(_pid)
                                yield {
                                    "type": "tool_end",
                                    "tool_name": _pname,
                                    "tool_call_id": _pid,
                                    "result": "",
                                    "provenance": tool_event_provenance(provisional = True),
                                }
                        # Merge metrics from prior tool iterations so they aren't dropped.
                        yield {"type": "status", "text": ""}
                        if content_accum:
                            # Strip leaked tool-call XML before yielding.
                            content_accum = _strip_tool_markup(content_accum, final = True)
                        # A truncated bare-JSON call has no XML markup to strip and didn't parse. With
                        # Auto-Heal on, drop a leading ENABLED-tool fragment (ordinary JSON answers untouched);
                        # off keeps it visible per the strict contract.
                        if content_accum and active_tools and auto_heal_tool_calls:
                            content_accum = strip_leading_bare_json_call(
                                content_accum, _enabled_tool_names
                            )
                        if content_accum:
                            yield {"type": "content", "text": content_accum}
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                # ── Execute tool calls ──
                _accumulated_completion_tokens += (
                    _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                ).get("completion_tokens", 0)
                _it = _iter_timings or {}
                _accumulated_predicted_ms += _it.get("predicted_ms", 0)
                _accumulated_predicted_n += _it.get("predicted_n", 0)

                # Collapse exact-duplicate calls and cap the count for the TEXTUAL
                # fallback (mirrors the safetensors loop; see _MAX_TOOL_CALLS_PER_TURN).
                if tool_calls and not has_structured_tc and len(tool_calls) > 1:
                    _seen_keys: set = set()
                    _deduped: list = []
                    for _tc in tool_calls:
                        _fn = _tc.get("function", {}) or {}
                        _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                        if _key in _seen_keys:
                            continue
                        _seen_keys.add(_key)
                        _deduped.append(_tc)
                        if len(_deduped) >= _MAX_TOOL_CALLS_PER_TURN:
                            break
                    if len(_deduped) != len(tool_calls):
                        logger.info(
                            "GGUF textual fallback: collapsed %d repeated tool call(s) "
                            "in one turn to %d",
                            len(tool_calls),
                            len(_deduped),
                        )
                    tool_calls = _deduped

                # disable_parallel_tool_use: execute only the first tool call
                # this turn. Truncate before building assistant_msg so the
                # conversation stays consistent and extra calls are never executed.
                if disable_parallel_tool_use and tool_calls and len(tool_calls) > 1:
                    tool_calls = tool_calls[:1]

                assistant_msg: dict = {"role": "assistant", "content": content_text}
                assistant_appended = False
                # Collect no-op nudges and flush them after the batch, so a no-op
                # doesn't abort it and drop the parallel calls that follow.
                deferred_noop_msgs: list = []

                # The text-path provisional card uses the parser's default id ("call_0");
                # a Mistral-style call carries its own id and would open a duplicate. Reuse
                # the card's id for the first matching call (same tool name) to reconcile.
                _text_provisional_id = _text_args_id if not has_structured_tc else ""

                for tc in tool_calls or []:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    if (
                        _text_provisional_id
                        and _text_provisional_id in provisional_started_tool_calls
                        and _text_provisional_id not in resolved_provisional_tool_call_ids
                        and tc.get("id") not in provisional_started_tool_calls
                        and provisional_started_tool_calls[_text_provisional_id] == tool_name
                    ):
                        tc = {**tc, "id": _text_provisional_id}
                    provisional_match = tc.get("id") in provisional_started_tool_calls
                    decision = tool_controller.prepare_call(
                        tc,
                        forced = _forced_tool_call_pending,
                        provisional = provisional_match,
                    )

                    if not decision.should_execute:
                        if content_text and not assistant_appended:
                            conversation.append(assistant_msg)
                            assistant_appended = True
                        if provisional_match:
                            # A provisional tool card is already on screen for this
                            # id; close it so it never dangles when the controller
                            # turns the call into an internal no-op (duplicate /
                            # disabled / render_html_repeat).
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": "",
                                "provenance": decision.provenance,
                            }
                        completion = tool_controller.record_noop(decision)
                        deferred_noop_msgs.append(completion.model_message())
                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                        logger.info(
                            "Suppressed local GGUF tool call as internal no-op: "
                            f"action={decision.action} tool={decision.tool_name}"
                        )
                        continue

                    if not assistant_appended:
                        assistant_msg["tool_calls"] = [decision.as_assistant_tool_call()]
                        conversation.append(assistant_msg)
                        assistant_appended = True
                    else:
                        assistant_msg.setdefault("tool_calls", []).append(
                            decision.as_assistant_tool_call()
                        )

                    # Bypass wins here too, so a direct internal caller with both
                    # flags never prompts. "auto" pauses only high-risk calls;
                    # "off" never prompts (sandbox stays on).
                    needs_confirm = (
                        bool(confirm_tool_calls)
                        and not bypass_permissions
                        and permission_mode != "off"
                    )
                    if needs_confirm and permission_mode == "auto":
                        needs_confirm = is_high_risk_tool_call(
                            decision.tool_name, decision.arguments
                        )
                    approval_id = new_approval_id() if needs_confirm else ""
                    decision_slot = (
                        begin_tool_decision(session_id, approval_id) if needs_confirm else None
                    )
                    start_event = decision.tool_start_event()
                    start_event["approval_id"] = approval_id
                    start_event["awaiting_confirmation"] = needs_confirm

                    try:
                        # Gated calls are not running yet; a "Running ..." badge
                        # counting up while it waits on a human reads as a hang.
                        yield {
                            "type": "status",
                            "text": (
                                awaiting_approval_status(decision.tool_name)
                                if needs_confirm
                                else decision.status_text
                            ),
                        }
                        yield start_event

                        _decision = (
                            wait_tool_decision(
                                decision_slot,
                                approval_id,
                                cancel_event = cancel_event,
                            )
                            if decision_slot is not None
                            else None
                        )
                        if _decision is not None and _decision != "deny":
                            # Approved: now it really is running.
                            yield {"type": "status", "text": decision.status_text}
                        if _decision == "deny":
                            decision_slot = None
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": TOOL_REJECTED_MESSAGE,
                                "provenance": decision.provenance,
                            }
                            denied_message = {
                                "role": "tool",
                                "name": decision.tool_name,
                                "content": TOOL_REJECTED_MESSAGE,
                            }
                            if decision.tool_call_id:
                                denied_message["tool_call_id"] = decision.tool_call_id
                            conversation.append(denied_message)
                            if _forced_tool_call_pending:
                                _forced_tool_call_pending = False
                            continue
                        decision_slot = None
                    finally:
                        if decision_slot is not None:
                            abort_tool_decision(decision_slot, approval_id)

                    _effective_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
                    # RAG: cap paraphrased KB re-searches that slip past the dup guard.
                    if (
                        decision.tool_name == "search_knowledge_base"
                        and _kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
                    ):
                        result = RAG_SEARCH_CAP_NUDGE
                    else:
                        # Execute in a worker thread so live stdout chunks and heartbeats
                        # stream while the tool blocks (the SSE route turns heartbeats into
                        # keepalives). Result is byte-identical to a direct call.
                        def _invoke_tool(_output_callback, _decision = decision):
                            # execute_tool is injectable and may be monkey-patched with the
                            # pre-PR signature; forward output_callback only if it's accepted.
                            kwargs = dict(
                                cancel_event = cancel_event,
                                timeout = _effective_timeout,
                                session_id = session_id,
                                thread_id = thread_id,
                                rag_scope = rag_scope,
                                disable_sandbox = bypass_permissions,
                            )
                            if accepts_output_callback(execute_tool):
                                kwargs["output_callback"] = _output_callback
                            return execute_tool(
                                _decision.tool_name,
                                _decision.arguments,
                                **kwargs,
                            )

                        result = yield from stream_tool_execution(
                            _invoke_tool,
                            tool_name = decision.tool_name,
                            tool_call_id = decision.tool_call_id,
                            cancel_event = cancel_event,
                        )
                        if decision.tool_name == "search_knowledge_base":
                            _kb_search_count += 1
                    completion = tool_controller.record_result(decision, result)
                    resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                    # A real execution opens the post-tool phase; carrying the pre-tool
                    # stall text over would read the same sentence as a repeat and
                    # swallow the one post-tool nudge.
                    _last_reprompt_text = ""
                    # A tool ran this turn, so it counts against the caller's budget.
                    _turn_executed_real_tool = True
                    yield completion.tool_end_event()
                    conversation.append(completion.tool_message())

                    if _forced_tool_call_pending:
                        _forced_tool_call_pending = False

                append_deferred_nudges(conversation, deferred_noop_msgs)

                # Close provisional cards not resolved by execution/no-op handling.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "",
                            "provenance": tool_event_provenance(provisional = True),
                        }

                # Clear tool status badge before next generation/final pass.
                yield {"type": "status", "text": ""}
                if tool_controller.force_final_answer or not tool_controller.active_tools():
                    _append_budget_exhausted_nudge = False
                    break
                # Count only real tool turns against the cap so reserved re-prompt slots can't become
                # extra tool rounds; a no-op correction turn doesn't consume budget (GGUF parity).
                if _turn_executed_real_tool:
                    _tool_iters_done += 1
                    if _tool_iters_done >= max_tool_iterations:
                        break
                continue

            except _LlamaStreamCancelled:
                return
            except httpx.ConnectError:
                # Mark unresolved provisional cards as failed before raising.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: lost connection to llama-server before the tool call completed.",
                            "provenance": tool_event_provenance(provisional = True),
                        }
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                # Same cleanup for other mid-iteration failures.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: the tool call was interrupted before it completed.",
                            "provenance": tool_event_provenance(provisional = True),
                        }
                raise

        # ── Tool iteration cap reached -- synthesize final answer ──
        # The model used all iterations without a final text response. Nudge
        # the final streaming pass to produce a useful answer instead of
        # continuing to request tools.
        if max_tool_iterations > 0 and _append_budget_exhausted_nudge:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available tool calls. Based on "
                        "everything you have found so far, provide your final "
                        "answer now. Do not call any more tools."
                    ),
                }
            )

        # Clear status.
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context.
        from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

        stream_payload = {
            "messages": neutralize_control_markup_in_messages(
                conversation, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            stream_payload["chat_template_kwargs"] = _reasoning_kw
        stream_payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        if stop:
            stream_payload["stop"] = stop
        if seed is not None:
            stream_payload["seed"] = seed
        stream_payload["stream_options"] = {"include_usage": True}

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""
        _final_reasoning_started_at: Optional[float] = None
        _final_reasoning_summary_emitted = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None
        _stream_done = False

        try:
            with self._open_chat_stream_with_respawn_retry(stream_payload, cancel_event) as (
                response,
                first_token_deadline,
            ):
                buffer = ""
                for raw_chunk in self._iter_text_cancellable(
                    response,
                    cancel_event,
                    first_token_deadline = first_token_deadline,
                ):
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if line == "data: [DONE]":
                            if in_thinking:
                                if (
                                    _final_reasoning_started_at is not None
                                    and not _final_reasoning_summary_emitted
                                ):
                                    _final_reasoning_summary_emitted = True
                                    yield _reasoning_summary_event(_final_reasoning_started_at)
                                if has_content_tokens:
                                    cumulative += "</think>"
                                    yield {
                                        "type": "content",
                                        "text": _strip_tool_markup(cumulative, final = True),
                                    }
                                else:
                                    cumulative = _finalize_reasoning_only_cumulative(
                                        cumulative,
                                        reasoning_text,
                                        _metadata_finish_reason,
                                        promote_reasoning_only,
                                    )
                                    yield {"type": "content", "text": cumulative}
                            _stream_done = True
                            break  # exit inner while
                        if not line.startswith("data: "):
                            continue

                        try:
                            chunk_data = json.loads(line[6:])
                            # Capture server timings/usage from final chunks.
                            _chunk_timings = chunk_data.get("timings")
                            if _chunk_timings:
                                _metadata_timings = _chunk_timings
                            _chunk_usage = chunk_data.get("usage")
                            if _chunk_usage:
                                _metadata_usage = _chunk_usage
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _metadata_finish_reason = _fr

                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if _final_reasoning_started_at is None:
                                        _final_reasoning_started_at = time.monotonic()
                                    reasoning_text += reasoning
                                    if not in_thinking:
                                        cumulative += "<think>"
                                        in_thinking = True
                                    cumulative += reasoning
                                    yield {"type": "content", "text": cumulative}

                                token = delta.get("content", "")
                                if token:
                                    if (
                                        _final_reasoning_started_at is not None
                                        and not _final_reasoning_summary_emitted
                                    ):
                                        _final_reasoning_summary_emitted = True
                                        yield _reasoning_summary_event(_final_reasoning_started_at)
                                    has_content_tokens = True
                                    if in_thinking:
                                        cumulative += "</think>"
                                        in_thinking = False
                                    cumulative += token
                                    cleaned = _strip_tool_markup(cumulative)
                                    # Emit only when cleaned text grows (monotonic).
                                    if len(cleaned) > len(_last_emitted):
                                        _last_emitted = cleaned
                                        yield {"type": "content", "text": cleaned}
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                    if _stream_done:
                        break  # exit outer for
                _meta = _build_metadata_event(
                    _metadata_usage, _metadata_timings, _metadata_finish_reason
                )
                if _meta is not None:
                    yield _meta

        except _LlamaStreamCancelled:
            return
        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise

    # ── Prompt token counting ──────────────────────────────────

    def count_chat_tokens(
        self,
        messages,
        system = None,
        tools = None,
        strict: bool = False,
        chat_template_kwargs = None,
        should_abort = None,
    ) -> int:
        """Count prompt tokens for a chat request via llama-server.

        Non-strict callers keep the historical best-effort behavior and get 0 when a
        count cannot be determined. Strict callers (public count_tokens endpoints)
        raise instead: the text-only fallback is an approximation, not a count.

        ``chat_template_kwargs`` reaches /apply-template unchanged, so a caller can
        price a request rendered in a non-default reasoning mode.

        ``should_abort`` is polled between the two llama-server calls. Admission is the
        caller's job; this only stops a count that was admitted while idle from spending its
        second round trip once the answer stopped mattering. Raises when it fires.
        """
        if not self.is_loaded:
            if strict:
                raise RuntimeError("llama-server is not loaded")
            return 0

        def _block_text(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str):
                            parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return ""

        # Normalize system into a leading message / plain text.
        system_text = ""
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = _block_text(system)

        # Count the neutralized prompt: raw text budgets a prompt generation never sends (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_control_markup,
            neutralize_control_markup_in_messages,
            neutralize_tool_descriptions,
        )

        _profile = self.markup_profile
        messages = neutralize_control_markup_in_messages(messages, None, _profile)
        # The model's own profile, like its two neighbours. The curated sweep here made the
        # count describe a different prompt from the one generation sends: a Llama profile
        # has no "</think>", so generation leaves that text byte-exact while the count path
        # was turning it into "< /think>" before /apply-template (#7066).
        system_text = (
            neutralize_control_markup(system_text)
            if _profile is None
            else _profile.rewrite_control(system_text)
        )
        tools = neutralize_tool_descriptions(tools, None, _profile)

        try:
            with httpx.Client(timeout = 10, headers = self._auth_headers, trust_env = False) as client:

                def _tokenize(text: str) -> int:
                    r = client.post(
                        f"{self.base_url}/tokenize",
                        json = {"content": text, "add_special": True},
                    )
                    if r.status_code != 200:
                        if strict:
                            raise RuntimeError("llama-server tokenizer failed")
                        return 0
                    tokens = r.json().get("tokens", [])
                    if not isinstance(tokens, list):
                        if strict:
                            raise RuntimeError("llama-server tokenizer returned invalid tokens")
                        return 0
                    return len(tokens)

                # 1. Try /apply-template to render the real chat prompt.
                template_messages = list(messages) if messages else []
                if system_text:
                    template_messages = [
                        {"role": "system", "content": system_text}
                    ] + template_messages
                # An empty list is passed through as-is: llama-server renders through minja, not
                # jinja2, so messages[0] yields undefined rather than raising and the template
                # returns its bare preamble. A placeholder turn would add a system block instead.
                apply_template_failed = False
                try:
                    # llama-server's /apply-template renders tool declarations
                    # into the prompt when ``tools`` is supplied, so pass them
                    # through, otherwise tool-schema tokens go uncounted.
                    template_body = {"messages": template_messages}
                    if tools:
                        template_body["tools"] = tools
                    # Layered over the load-time --chat-template-kwargs: only keys sent here move.
                    if chat_template_kwargs:
                        template_body["chat_template_kwargs"] = chat_template_kwargs
                    resp = client.post(
                        f"{self.base_url}/apply-template",
                        json = template_body,
                    )
                    if resp.status_code == 200:
                        prompt = resp.json().get("prompt", "")
                        if isinstance(prompt, str):
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                            return _tokenize(prompt)
                    apply_template_failed = True
                except CountAborted:
                    # Not a template failure: swallowed, the text fallback tokenizes anyway,
                    # which is the work being declined. Must precede the generic except.
                    raise
                except Exception:
                    apply_template_failed = True

                # The fallback drops role markers, special tokens and tool schemas (~30% of a
                # six-turn two-tool prompt), so strict callers error rather than undercount.
                if strict and apply_template_failed:
                    raise RuntimeError("llama-server could not render the chat template")

                # 2. Fallback: concatenate plain text and tokenize. Append a
                # serialized form of the tools so they still contribute to the
                # count when /apply-template is unavailable.
                parts = []
                if system_text:
                    parts.append(system_text)
                for msg in messages or []:
                    if isinstance(msg, dict):
                        parts.append(_block_text(msg.get("content", "")))
                if tools:
                    try:
                        parts.append(json.dumps(tools, ensure_ascii = False))
                    except Exception:
                        pass
                return _tokenize("\n".join(p for p in parts if p))
        except Exception:
            if strict:
                raise
            return 0

    # ── TTS support ────────────────────────────────────────────

    def detect_audio_type(self) -> Optional[str]:
        """Detect audio/TTS codec; swallows errors (use _strict to distinguish)."""
        try:
            return self._detect_audio_type_strict()
        except Exception as e:
            logger.debug(f"Audio type detection failed: {e}")
            return None

    def _apply_detected_audio(self, detected: Optional[str]) -> bool:
        """Apply a probed audio codec under self._lock. Returns True to continue
        the load (codec inited OK, or nothing to init), False to abort (server
        unhealthy or codec init failed). Shared by the fast-path retry and the
        main load path."""
        if detected in ("snac", "bicodec", "dac"):
            with self._lock:
                if not self._healthy:
                    return False
                try:
                    self.init_audio_codec(detected)
                    self._is_audio = True
                    self._audio_type = detected
                except Exception as exc:
                    # Surface as HTTP 500 (matches pre-PR contract).
                    logger.warning("Failed to init audio codec '%s': %s", detected, exc)
                    self._audio_probed = False
                    return False
        elif detected:
            # csm / whisper / audio_vlm: track type but keep _is_audio False --
            # GGUF TTS routing only fires for snac/bicodec/dac.
            with self._lock:
                if not self._healthy:
                    return False
                self._audio_type = detected
        # Audio input = token probe (audio_vlm/whisper) OR mmproj encoder.
        from utils.models.model_config import is_audio_input_type

        self._has_audio_input = bool(is_audio_input_type(self._audio_type)) or bool(
            self._mmproj_has_audio
        )
        return True

    def _detect_audio_type_strict(self) -> Optional[str]:
        """Codec name on match, None on non-audio, raises on transport/JSON errors."""
        if not self.is_loaded:
            return None
        with httpx.Client(timeout = 10, headers = self._auth_headers, trust_env = False) as client:

            def _detok(tid: int) -> str:
                # Non-200 means "marker not in vocab" -- keep probing.
                # Transport / JSON errors still raise.
                r = client.post(f"{self.base_url}/detokenize", json = {"tokens": [tid]})
                if r.status_code != 200:
                    return ""
                return r.json().get("content", "")

            def _tok(text: str) -> list[int]:
                r = client.post(
                    f"{self.base_url}/tokenize",
                    json = {"content": text, "add_special": False},
                )
                if r.status_code != 200:
                    return []
                return r.json().get("tokens", [])

            # Codec-specific tokens (not generic ones that non-audio models may have)
            if "<custom_token_" in _detok(128258) and "<custom_token_" in _detok(128259):
                return "snac"
            if len(_tok("<|AUDIO|>")) == 1 and len(_tok("<|audio_eos|>")) == 1:
                return "csm"
            if len(_tok("<|startoftranscript|>")) == 1:
                return "whisper"
            # Gemma 3n: <audio_soft_token>; Gemma 4: <|audio|> (not csm's <|AUDIO|>).
            if len(_tok("<audio_soft_token>")) == 1 or len(_tok("<|audio|>")) == 1:
                return "audio_vlm"
            if len(_tok("<|bicodec_semantic_0|>")) == 1 and len(_tok("<|bicodec_global_0|>")) == 1:
                return "bicodec"
            if len(_tok("<|c1_0|>")) == 1 and len(_tok("<|c2_0|>")) == 1:
                return "dac"
        return None

    # Prompt format per codec: (template, stop_tokens, needs_token_ids).
    # Matches InferenceBackend._generate_snac/bicodec/dac.
    _TTS_PROMPTS = {
        "snac": (
            "<custom_token_3>{text}<|eot_id|><custom_token_4>",
            ["<custom_token_2>"],
            True,
        ),
        "bicodec": (
            "<|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>",
            ["<|im_end|>", "</s>"],
            False,
        ),
        "dac": (
            "<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|><|global_features_start|>\n",
            ["<|im_end|>", "<|audio_end|>"],
            False,
        ),
    }

    _codec_mgr = None  # Shared AudioCodecManager instance

    def init_audio_codec(self, audio_type: str) -> None:
        """Load the audio codec at model load time (mirrors the non-GGUF path)."""
        import torch
        from core.inference.audio_codecs import AudioCodecManager

        if LlamaCppBackend._codec_mgr is None:
            LlamaCppBackend._codec_mgr = AudioCodecManager()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_repo_path = None

        # BiCodec needs a repo with BiCodec/ weights -- download canonical SparkTTS
        if audio_type == "bicodec":
            from huggingface_hub import snapshot_download
            import os

            repo_path = snapshot_download("unsloth/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B")
            model_repo_path = os.path.abspath(repo_path)

        LlamaCppBackend._codec_mgr.load_codec(audio_type, device, model_repo_path = model_repo_path)
        logger.info(f"Loaded audio codec for GGUF TTS: {audio_type}")

    def generate_audio_response(
        self,
        text: str,
        audio_type: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple:
        """
        Generate TTS audio via llama-server /completion + codec decode.
        Returns (wav_bytes, sample_rate).

        ``cancel_event`` lets a Stop or a forced model swap end the request: the
        decode is one blocking POST, so a watcher closes the client out from under
        it rather than polling. Raises RuntimeError once cancelled.
        """
        if audio_type not in self._TTS_PROMPTS:
            raise RuntimeError(f"GGUF TTS does not support '{audio_type}' codec.")

        tpl, stop, need_ids = self._TTS_PROMPTS[audio_type]
        # Raw prompt, not messages: codec delimiters are the only structure to break (#7066).
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text

        payload: dict = {
            "prompt": tpl.format(text = neutralize_tts_prompt_text(text, audio_type)),
            "stream": False,
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
        }
        if stop:
            payload["stop"] = stop
        if need_ids:
            payload["n_probs"] = 1

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Audio generation cancelled")

        with httpx.Client(
            timeout = httpx.Timeout(300, connect = 10),
            headers = self._auth_headers,
            trust_env = False,
        ) as client:
            finished = threading.Event()
            watcher: Optional[threading.Thread] = None
            if cancel_event is not None:

                def _close_when_cancelled() -> None:
                    while not finished.wait(0.05):
                        if cancel_event.is_set():
                            # Closing mid-request makes the blocking post raise
                            # httpx.RequestError, the only way out of it.
                            with contextlib.suppress(Exception):
                                client.close()
                            return

                watcher = threading.Thread(target = _close_when_cancelled, daemon = True)
                watcher.start()
            try:
                resp = client.post(f"{self.base_url}/completion", json = payload)
            except httpx.RequestError:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Audio generation cancelled") from None
                raise
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join(timeout = 0.5)
            if resp.status_code != 200:
                raise RuntimeError(f"llama-server returned {resp.status_code}: {resp.text}")

        # The codec decode below is GPU work with no interruption point, so check here:
        # cancelling after this only wastes the decode it cannot stop.
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Audio generation cancelled")

        data = resp.json()
        token_ids = (
            [p["id"] for p in data.get("completion_probabilities", []) if "id" in p]
            if need_ids
            else None
        )

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LlamaCppBackend._codec_mgr.decode(
            audio_type, device, token_ids = token_ids, text = data.get("content", "")
        )
