# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""

import atexit
import contextlib
import json
import os
import re
import struct
import structlog
from loggers import get_logger
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator, List, Optional
from urllib.parse import urlparse

import httpx

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


# ── Pre-compiled patterns for plan-without-action re-prompt ──
# Forward-looking intent signals that indicate the model is
# describing what it *will* do rather than giving a final answer.
_INTENT_SIGNAL = re.compile(
    r"(?i)("
    # Direct intent: "I'll ...", "I will ...", "Let me ...", "I am going to ..."
    # Handles both straight and curly apostrophes.
    # Excludes "I can", "I should", "I want to", "let's" which
    # appear frequently in direct answers / explanations.
    r"\b(i['\u2019](ll|m going to|m gonna)|i am (going to|gonna)|i will|i shall|let me|allow me)\b"
    r"|"
    # Step/plan framing: "First ...", "Step 1:", "Here's my plan"
    r"\b(?:first\b|step \d+:?|here['\u2019]?s (?:my |the |a )?(?:plan|approach))"
    r"|"
    # "Now I" / "Next I" patterns
    r"\b(?:now i|next i)\b"
    r")"
)
_MAX_REPROMPTS = 3

# Without max_tokens, llama-server defaults to n_predict = n_ctx (up to
# 262144 for Qwen3.5), producing many-minute zombie decodes when cancel
# fails. t_max_predict_ms is a wall-clock backstop applied unconditionally,
# but the llama.cpp README notes it ONLY fires after a newline has been
# generated -- a model stuck in a long unbroken non-newline sequence is
# unbounded by it. So we still want a token cap as the front-line limiter.
#
# The cap is the model's effective context length when we know it,
# falling back to a generous floor when metadata is unavailable. 4096 was
# too low: Qwen3 / gpt-oss reasoning traces routinely exceed it, and any
# OpenAI-API caller that omits max_tokens (langchain, llama-index, raw
# curl) sees responses silently truncated mid-sentence.
_DEFAULT_MAX_TOKENS_FLOOR = 32768
_DEFAULT_T_MAX_PREDICT_MS = 600_000  # 10 min
_REPROMPT_MAX_CHARS = 2000

# ── Pre-compiled patterns for GGUF shard detection ───────────
_SHARD_FULL_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$")


# ── Sliding-window-pattern resolver ───────────────────────────
# Resolves the per-layer SWA mask when a GGUF reports a sliding window
# but no `sliding_window_pattern` field. Tier order in
# `_resolve_swa_pattern`: GGUF metadata, on-disk cache, bootstrap dict
# below, transformers introspection, HF Hub config.json, legacy 1/4
# fallback. Period N means layer i is SWA iff `(i + 1) % N != 0`,
# matching transformers. Skipped on purpose: phi3 (no key/val length
# in GGUF, window >= ctx anyway), qwen2 family (converter strips
# sliding_window when use_sliding_window=False), mistral v0.1/v0.2
# (all-SWA can't be expressed as a period).
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
            with open(_swa_cache_path()) as f:
                _SWA_CACHE = json.load(f)
                if not isinstance(_SWA_CACHE, dict):
                    _SWA_CACHE = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            _SWA_CACHE = {}
        return _SWA_CACHE


def _save_swa_cache(cache: dict) -> None:
    try:
        path = _swa_cache_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent = 2, sort_keys = True)
        tmp.replace(path)
    except OSError:
        pass


def _period_from_layer_types(layer_types: list) -> Optional[int]:
    """Smallest period N where `(i+1) % N != 0` matches the SWA mask,
    or None if no fixed period fits."""
    if not layer_types:
        return None
    is_swa = ["full" not in str(t).lower() for t in layer_types]
    n = len(is_swa)
    for N in range(1, n + 1):
        if all(((i + 1) % N != 0) == is_swa[i] for i in range(n)):
            return N
    return None


def _fetch_swa_entry_from_hf(repo_id: str) -> Optional[object]:
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id, "config.json", repo_type = "model")
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None

    src = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    period = src.get("sliding_window_pattern")
    if isinstance(period, int) and period > 0:
        return period
    lt = src.get("layer_types")
    if isinstance(lt, list) and lt:
        return _period_from_layer_types(lt) or [
            "full" not in str(t).lower() for t in lt
        ]
    return None


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
    lt = getattr(src, "layer_types", None)
    if isinstance(lt, list) and lt:
        return _period_from_layer_types(lt) or [
            "full" not in str(t).lower() for t in lt
        ]
    return None


_SWA_PATTERN_SOURCE_RE = re.compile(
    r"sliding_window_pattern\s*(?::\s*[\w\[\], ]*)?\s*=\s*(\d+)"
)


def _resolve_swa_entry_from_transformers(arch: str) -> Optional[object]:
    """Default-instantiate the matching Config; on failure, regex-parse
    its source for `sliding_window_pattern = N`."""
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

    # Tier 3: live HF fetch (with persistent caching of the result)
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
    """Strip `https://huggingface.co/owner/name(/...)` to `owner/name`."""
    if not url or "huggingface.co/" not in url:
        return None
    tail = url.split("huggingface.co/", 1)[1].rstrip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


# Model size extraction — lazy import to avoid pulling in transformers
# at module level.  See PR description for the full explanation.
def _extract_model_size_b(model_id: str):
    from utils.models import extract_model_size_b

    return extract_model_size_b(model_id)


# ── Pre-compiled patterns for tool XML stripping ─────────────
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
]

# ── Pre-compiled patterns for tool-call XML parsing ──────────
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")


_TOOL_TEMPLATE_MARKERS = (
    "{%- if tools %}",
    "{%- if tools -%}",
    "{% if tools %}",
    "{% if tools -%}",
    '"role" == "tool"',
    "'role' == 'tool'",
    'message.role == "tool"',
    "message.role == 'tool'",
)


def detect_reasoning_flags(
    chat_template: Optional[str],
    model_identifier: Optional[str] = None,
    *,
    log_source: Optional[str] = None,
) -> dict:
    """Classify a chat template's reasoning and tool-calling capabilities.

    Returns a dict with the same five keys populated by the GGUF sniffer:
    ``supports_reasoning``, ``reasoning_style``
    (``"enable_thinking"`` | ``"reasoning_effort"``),
    ``reasoning_always_on``, ``supports_preserve_thinking``, and
    ``supports_tools``. Used by both the llama-server backend at load
    time and the safetensors/transformers paths in ``routes/inference``
    so the two agree on what the frontend will see.
    """
    flags = {
        "supports_reasoning": False,
        "reasoning_style": "enable_thinking",
        "reasoning_always_on": False,
        "supports_preserve_thinking": False,
        "supports_tools": False,
    }
    if not chat_template:
        return flags
    tpl = chat_template
    prefix = f"{log_source}: " if log_source else ""

    if "enable_thinking" in tpl:
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking"
        logger.info(f"{prefix}model supports reasoning (enable_thinking)")
    elif "reasoning_effort" in tpl:
        # gpt-oss / Harmony templates use reasoning_effort
        # ("low" | "medium" | "high") instead of a boolean.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "reasoning_effort"
        logger.info(f"{prefix}model supports reasoning (reasoning_effort)")
    elif "thinking" in tpl:
        # DeepSeek uses 'thinking' instead of 'enable_thinking'
        normalized_id = (model_identifier or "").lower()
        if "deepseek" in normalized_id:
            flags["supports_reasoning"] = True
            logger.info(f"{prefix}model supports reasoning (DeepSeek thinking)")

    # Hardcoded <think> tags or reasoning_content in the template mean
    # thinking is always on (no toggle to disable it).
    if not flags["supports_reasoning"]:
        if ("<think>" in tpl and "</think>" in tpl) or "reasoning_content" in tpl:
            flags["supports_reasoning"] = True
            flags["reasoning_always_on"] = True
            logger.info(f"{prefix}model always reasons (<think> tags in template)")

    # preserve_thinking is an independent kwarg on some Qwen templates
    # that keeps historical <think> blocks in prior assistant turns.
    if "preserve_thinking" in tpl:
        flags["supports_preserve_thinking"] = True
        logger.info(f"{prefix}model supports preserve_thinking")

    if any(marker in tpl for marker in _TOOL_TEMPLATE_MARKERS):
        flags["supports_tools"] = True
        logger.info(f"{prefix}model supports tool calling")

    return flags


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
        self._context_length: Optional[int] = None
        self._effective_context_length: Optional[int] = None
        self._max_context_length: Optional[int] = None
        self._chat_template: Optional[str] = None
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_style: str = "enable_thinking"
        self._supports_preserve_thinking: bool = False
        self._supports_tools: bool = False
        self._cache_type_kv: Optional[str] = None
        self._reasoning_default: bool = True
        self._speculative_type: Optional[str] = None
        # KV-cache estimation fields (populated by _read_gguf_metadata)
        self._n_layers: Optional[int] = None
        self._n_kv_heads: Optional[int] = None
        self._n_kv_heads_by_layer: Optional[list[int]] = None
        self._n_heads: Optional[int] = None
        self._embedding_length: Optional[int] = None
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
        # Last N layers reuse KV from earlier layers and don't allocate
        # their own cache (Gemma 3n / Gemma 4: <arch>.attention.shared_kv_layers).
        self._shared_kv_layers: Optional[int] = None
        self._lock = threading.Lock()
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._api_key: Optional[str] = None

        self._kill_orphaned_servers()
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
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return self._is_vision

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def context_length(self) -> Optional[int]:
        """Return the effective context length the server is running at."""
        return self._effective_context_length or self._context_length

    @property
    def max_context_length(self) -> Optional[int]:
        """Return the largest context that fits on this hardware at load time.

        This is the "safe zone" threshold the UI renders warnings
        against. For a model whose weights fit on some GPU subset, it
        is the binary-search cap from ``_fit_context_to_vram`` for that
        subset. For a model whose weights exceed 90% of every GPU
        subset, it is the 4096 fallback -- the spec's default when the
        model will not fit. The UI slider ceiling is
        ``native_context_length``; dragging above ``max_context_length``
        triggers the "might be slower" warning.
        """
        return self._max_context_length or self._context_length

    @property
    def native_context_length(self) -> Optional[int]:
        """Return the model's native context length from GGUF metadata."""
        return self._context_length

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        While llama-server is warming up, its process is typically in
        kernel state D (disk sleep) mmap'ing the weight shards into
        page cache before pushing layers to VRAM. During that window
        ``/api/inference/status`` only reports ``loading``, which gives
        the UI nothing to display besides a spinner that looks stuck
        for minutes on large MoE models.

        This method samples ``/proc/<pid>/status VmRSS`` against the
        sum of the GGUF shard sizes so the UI can render a real bar
        and compute rate / ETA. Returns ``None`` when no load is in
        flight (no process, or process already healthy).

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # VmRSS of the llama-server
                "bytes_total":  int,   # sum of shard file sizes
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }

        Linux-only in the current implementation. On macOS/Windows the
        equivalent would be a different API; this returns ``None`` on
        platforms where ``/proc/<pid>/status`` is unavailable.
        """
        proc = self._process
        if proc is None:
            return None
        pid = proc.pid
        if pid is None:
            return None

        # Sum up shard sizes (primary + any extras sitting alongside).
        bytes_total = 0
        gguf_path = self._gguf_path
        if gguf_path:
            primary = Path(gguf_path)
            try:
                if primary.is_file():
                    bytes_total += primary.stat().st_size
            except OSError:
                pass
            # Extra shards live alongside the primary with the same prefix
            # before the shard index (e.g. ``-00001-of-00004.gguf``).
            try:
                parent = primary.parent
                stem = primary.name
                m = _SHARD_RE.match(stem)
                prefix = m.group(1) if m else None
                if prefix and parent.is_dir():
                    for sibling in parent.iterdir():
                        if (
                            sibling.is_file()
                            and sibling.name.startswith(prefix)
                            and sibling.name != stem
                            and sibling.suffix == ".gguf"
                        ):
                            try:
                                bytes_total += sibling.stat().st_size
                            except OSError:
                                pass
            except OSError:
                pass

        # Read VmRSS from /proc/<pid>/status. Kilobytes on Linux.
        bytes_loaded = 0
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        bytes_loaded = kb * 1024
                        break
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return None

        phase = "ready" if self._healthy else "mmap"
        fraction = 0.0
        if bytes_total > 0:
            fraction = min(1.0, bytes_loaded / bytes_total)
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
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_always_on(self) -> bool:
        return self._reasoning_always_on

    @property
    def reasoning_style(self) -> str:
        return self._reasoning_style

    @property
    def supports_preserve_thinking(self) -> bool:
        return self._supports_preserve_thinking

    @property
    def reasoning_default(self) -> bool:
        return self._reasoning_default

    def _reasoning_kwargs(self, enable_thinking: bool) -> dict:
        if self._reasoning_style == "reasoning_effort":
            return {"reasoning_effort": "high" if enable_thinking else "low"}
        return {"enable_thinking": enable_thinking}

    def _request_reasoning_kwargs(
        self,
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> Optional[dict]:
        """Build chat_template_kwargs from per-request reasoning fields.

        Produces a merged dict covering the active model's reasoning style
        (``enable_thinking`` or ``reasoning_effort``) plus the independent
        ``preserve_thinking`` kwarg when the template supports it.
        """
        kwargs: dict = {}
        # Always-on reasoning models hardcode <think> tags in their template
        # and do not consume enable_thinking / reasoning_effort -- skip.
        if self._supports_reasoning and not self._reasoning_always_on:
            if self._reasoning_style == "reasoning_effort":
                if reasoning_effort in ("low", "medium", "high"):
                    kwargs["reasoning_effort"] = reasoning_effort
                elif enable_thinking is not None:
                    kwargs["reasoning_effort"] = "high" if enable_thinking else "low"
            else:
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
        if self._supports_preserve_thinking and preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        return kwargs or None

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        return self._cache_type_kv

    @property
    def speculative_type(self) -> Optional[str]:
        return self._speculative_type

    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _find_llama_server_binary() -> Optional[str]:
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
        import os
        import sys

        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        # 1. Env var — direct path to binary
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path and Path(env_path).is_file():
            return env_path

        # 1b. UNSLOTH_LLAMA_CPP_PATH — custom llama.cpp install directory
        custom_llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
        if custom_llama_cpp:
            custom_dir = Path(custom_llama_cpp)
            # Root dir (make builds)
            root_bin = custom_dir / binary_name
            if root_bin.is_file():
                return str(root_bin)
            # build/bin/ (cmake builds on Linux)
            cmake_bin = custom_dir / "build" / "bin" / binary_name
            if cmake_bin.is_file():
                return str(cmake_bin)
            # build/bin/Release/ (cmake builds on Windows)
            if sys.platform == "win32":
                win_bin = custom_dir / "build" / "bin" / "Release" / binary_name
                if win_bin.is_file():
                    return str(win_bin)

        # 2-4. Match installer layout: env-mode -> $STUDIO_HOME/llama.cpp;
        # default/HOME-redirect -> ~/.unsloth/llama.cpp (sibling of studio).
        legacy_llama = Path.home() / ".unsloth" / "llama.cpp"
        try:
            from utils.paths.storage_roots import studio_root as _sr  # noqa: WPS433

            _resolved_sr = _sr()
            _legacy_studio = Path.home() / ".unsloth" / "studio"
            try:
                _is_legacy = _resolved_sr.resolve() == _legacy_studio.resolve()
            except (OSError, ValueError):
                _is_legacy = _resolved_sr == _legacy_studio
            if _is_legacy:
                search_roots = [legacy_llama]
            else:
                # why: _kill_orphaned_servers excludes the legacy root in custom
                # mode; discovery must match so we never spawn a server we then
                # refuse to clean up. UNSLOTH_LLAMA_CPP_PATH (handled earlier)
                # is the explicit way to share a build across roots.
                search_roots = [_resolved_sr / "llama.cpp"]
        except (ImportError, OSError, ValueError):
            search_roots = [legacy_llama]
        _seen_roots: set[str] = set()
        _unique_roots: list[Path] = []
        for r in search_roots:
            k = str(r)
            if k not in _seen_roots:
                _seen_roots.add(k)
                _unique_roots.append(r)
        for unsloth_home in _unique_roots:
            home_root = unsloth_home / binary_name
            if home_root.is_file():
                return str(home_root)
            home_linux = unsloth_home / "build" / "bin" / binary_name
            if home_linux.is_file():
                return str(home_linux)
            if sys.platform == "win32":
                home_win = unsloth_home / "build" / "bin" / "Release" / binary_name
                if home_win.is_file():
                    return str(home_win)

        # 5–6. Legacy: in-tree build (older setup.sh / setup.ps1 versions)
        project_root = Path(__file__).resolve().parents[4]
        # Root dir (make builds)
        root_path = project_root / "llama.cpp" / binary_name
        if root_path.is_file():
            return str(root_path)
        # build/bin/ (cmake builds)
        build_path = project_root / "llama.cpp" / "build" / "bin" / binary_name
        if build_path.is_file():
            return str(build_path)
        if sys.platform == "win32":
            win_path = (
                project_root / "llama.cpp" / "build" / "bin" / "Release" / binary_name
            )
            if win_path.is_file():
                return str(win_path)

        # 7. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 8. Legacy: extracted to bin/
        bin_path = project_root / "bin" / binary_name
        if bin_path.is_file():
            return str(bin_path)

        return None

    # ── GPU allocation ────────────────────────────────────────────

    @staticmethod
    def _get_gguf_size_bytes(model_path: str) -> int:
        """Get total GGUF size in bytes, including split shards."""
        main = Path(model_path)
        total = main.stat().st_size

        # Check for split shards (e.g., model-00001-of-00003.gguf)
        m = _SHARD_FULL_RE.match(main.name)
        if m:
            prefix, _, num_total = m.group(1), m.group(2), m.group(3)
            sibling_pat = re.compile(
                r"^"
                + re.escape(prefix)
                + r"-\d{5}-of-"
                + re.escape(num_total)
                + r"\.gguf$"
            )
            for sibling in main.parent.iterdir():
                if sibling != main and sibling_pat.match(sibling.name):
                    total += sibling.stat().st_size

        return total

    @staticmethod
    def _get_gpu_free_memory() -> list[tuple[int, int]]:
        """Query free memory per GPU.

        Order:
          1. ``nvidia-smi`` (NVIDIA CUDA hosts) -- respects
             ``CUDA_VISIBLE_DEVICES``.
          2. ``torch.cuda.mem_get_info`` -- universal fallback that
             works on AMD ROCm too because the HIP runtime
             reuses the entire ``torch.cuda.*`` namespace. Covers the
             AMD case for issue #5106 (nvidia-smi-only probe silently
             returned [] on AMD hosts) and also rescues NVIDIA hosts
             where ``nvidia-smi`` is missing from PATH.

        Returns list of (gpu_index, free_mib) sorted by index. Empty
        list if no supported GPU is reachable.
        """
        import os

        # ── NVIDIA via nvidia-smi ────────────────────────────────────
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                allowed: Optional[set[int]] = None
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cvd is not None:
                    try:
                        # `if x.strip()` filters trailing-comma masks like
                        # "0,1," which would otherwise raise ValueError on
                        # an empty token. An explicitly empty mask (CVD="")
                        # yields an empty `allowed` set so all GPUs are
                        # filtered out, matching the codebase convention.
                        allowed = set(
                            int(x.strip()) for x in cvd.split(",") if x.strip()
                        )
                    except ValueError:
                        pass
                gpus: list[tuple[int, int]] = []
                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        idx = int(parts[0].strip())
                        free_mib = int(parts[1].strip())
                        if allowed is not None and idx not in allowed:
                            continue
                        gpus.append((idx, free_mib))
                # Match the docstring's sort-by-id guarantee. nvidia-smi
                # almost always returns sorted output, but driver order
                # is not formally guaranteed.
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
            # torch.cuda enumerates GPUs RELATIVE to the visibility mask.
            # On NVIDIA builds the mask is CUDA_VISIBLE_DEVICES; on AMD
            # ROCm builds it is HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES
            # if HIP is unset). Downstream we feed these IDs back into the
            # llama-server subprocess as CVD, so we must translate visible
            # ordinals back to physical indices first; otherwise launching
            # with ``CUDA_VISIBLE_DEVICES=2,3`` would get rewritten to
            # ``CUDA_VISIBLE_DEVICES=0,1`` and target the wrong GPUs.
            physical_ids: Optional[list[int]] = None
            # Match the codebase convention in
            # ``utils/hardware/hardware.py::_get_parent_visible_gpu_spec``:
            # treat an explicitly empty mask (``HIP_VISIBLE_DEVICES=""``)
            # as "set to no GPUs" rather than falling through to the next
            # var. ``or`` would coerce empty string to falsy and silently
            # promote the wrong source.
            if getattr(torch.version, "hip", None) is not None:
                hip_v = os.environ.get("HIP_VISIBLE_DEVICES")
                rocr_v = os.environ.get("ROCR_VISIBLE_DEVICES")
                cvd = (
                    hip_v
                    if hip_v is not None
                    else rocr_v
                    if rocr_v is not None
                    else os.environ.get("CUDA_VISIBLE_DEVICES")
                )
            else:
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None:
                try:
                    # Empty mask (CVD="") yields an empty list so the
                    # below loop produces no GPUs, consistent with the
                    # nvidia-smi path and utils/hardware/hardware.py.
                    physical_ids = [int(x.strip()) for x in cvd.split(",") if x.strip()]
                except ValueError:
                    physical_ids = None
            gpus = []
            for ordinal in range(torch.cuda.device_count()):
                free_bytes, _total_bytes = torch.cuda.mem_get_info(ordinal)
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                gpus.append((idx, free_bytes // (1024 * 1024)))
            # Match the nvidia-smi path's docstring guarantee of sorted-by-id.
            return sorted(gpus, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"torch GPU probe failed: {e}")
            return []

    @staticmethod
    def _select_gpus(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
    ) -> tuple[Optional[list[int]], bool]:
        """Pick GPU(s) for a model based on estimated VRAM and free memory.

        ``model_size_bytes`` should include both model weights and estimated
        KV cache.  The 90% threshold provides headroom for compute buffers,
        CUDA context, and other runtime overhead.

        Returns (gpu_indices, use_fit):
          - ([1], False)       model fits on 1 GPU at 90% of free
          - ([1, 2], False)    model needs 2 GPUs
          - (None, True)       model too large, let --fit handle it
        """
        if not gpus:
            return None, True

        model_size_mib = model_size_bytes / (1024 * 1024)

        # Sort GPUs by free memory descending
        ranked = sorted(gpus, key = lambda g: g[1], reverse = True)

        # Try fitting on 1 GPU (90% of free memory threshold)
        if ranked[0][1] * 0.90 >= model_size_mib:
            return [ranked[0][0]], False

        # Try fitting on N GPUs (accumulate free memory from most-free)
        cumulative = 0
        selected = []
        for idx, free_mib in ranked:
            selected.append(idx)
            cumulative += free_mib * 0.90
            if cumulative >= model_size_mib:
                return sorted(selected), False

        # Model is too large even for all GPUs, let --fit handle it
        logger.debug(
            "Model does not fit in available GPU memory, falling back to --fit",
            model_size_mib = round(model_size_mib, 2),
            ranked_gpus = ranked,
        )
        return None, True

    # ── KV cache VRAM estimation ─────────────────────────────────────

    def _can_estimate_kv(self) -> bool:
        """True if we have enough GGUF metadata to estimate KV cache size."""
        if self._n_layers is None:
            return False
        # MLA: kv_lora_rank is sufficient (K-only cache)
        if self._kv_lora_rank is not None:
            return True
        # New-style: need both explicit key AND value dimensions
        if self._kv_key_length is not None and self._kv_value_length is not None:
            return True
        # Legacy: need embedding_length + a head count (scalar or per-layer).
        return self._embedding_length is not None and (
            self._n_kv_heads is not None
            or self._n_heads is not None
            or self._n_kv_heads_by_layer is not None
        )

    def _kv_heads_for_layer(self, layer_idx: int, fallback: int) -> int:
        if self._n_kv_heads_by_layer is not None and layer_idx < len(
            self._n_kv_heads_by_layer
        ):
            return self._n_kv_heads_by_layer[layer_idx]
        return fallback

    def _estimate_kv_cache_bytes(
        self,
        n_ctx: int,
        cache_type_kv: Optional[str] = None,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        ctx_checkpoints: int = 0,
    ) -> int:
        """Estimate KV cache VRAM for a given context length.

        Uses 5-path architecture-aware estimation:
          1. MLA      -- compressed KV latent + RoPE, K-only (no separate V)
          2. Hybrid   -- only attention layers need KV (Mamba layers don't)
          3. SWA      -- sliding-window layers cache min(ctx, window) tokens
          4. GQA      -- standard full KV with explicit key/value dimensions
          5. Legacy   -- fallback using embed // n_heads

        Server-flag knobs (mirror llama-server's CLI):
          swa_full        -- ``--swa-full``: force SWA layers to cache the
                             full ``n_ctx`` (collapses path 3 to path 4
                             sizing for the SWA layers).
          n_parallel      -- ``--parallel``: number of server slots.
                             Verified empirically against llama-server:
                             non-SWA layers stay constant (cells split
                             across slots), SWA layers scale linearly
                             (per-slot window).
          kv_unified      -- ``--kv-unified`` (default on): retained for
                             API forward-compat. Currently a no-op for
                             memory math because the unified buffer total
                             matches per-slot buffers in measured cases.
          ctx_checkpoints -- ``--ctx-checkpoints``: SWA snapshot count per
                             slot (PR #15293). Each snapshot stores one
                             sliding-window of state per SWA layer.

        Returns 0 if metadata is insufficient for estimation.
        """
        if not self._can_estimate_kv() or n_ctx <= 0:
            return 0

        n_layers = self._n_layers  # type: ignore[assignment]
        # Gemma 3n / Gemma 4 reuse KV from earlier layers in the last
        # ``shared_kv_layers`` blocks -- those don't allocate their own
        # cache.  Floor at 1 so a misconfigured GGUF can't zero out KV.
        shared = self._shared_kv_layers or 0
        n_layers_kv = max(1, n_layers - shared)
        n_kv = self._n_kv_heads or self._n_heads or 1  # type: ignore[assignment]

        # Bytes per element depends on KV cache quantization
        bpe = {
            "f32": 4.0,
            "f16": 2.0,
            "bf16": 2.0,
            "q8_0": 34 / 32,
            "q5_1": 0.75,
            "q5_0": 0.6875,
            "q4_1": 0.625,
            "q4_0": 0.5625,
            "iq4_nl": 0.5625,
        }.get(cache_type_kv or "f16", 2.0)

        slots = max(1, n_parallel)

        # Path 1: MLA (DeepSeek-V2/V3, GLM-4.7, GLM-5, Kimi-K2.5)
        # MLA stores one compressed KV latent per token/layer (shared across heads).
        # V is reconstructed from the latent on the fly -- no separate V cache.
        # key_length = kv_lora_rank + rope_dim (the full compressed representation).
        # MLA GGUFs set head_count_kv=1; default to 1 if absent to avoid
        # falling back to n_heads (e.g., 128 for DeepSeek-V3) which would 128x.
        if self._kv_lora_rank is not None:
            n_kv_mla = self._n_kv_heads or 1
            rope_dim = self._key_length_mla or 64
            key_len = self._kv_key_length or (self._kv_lora_rank + rope_dim)
            return int(n_layers_kv * n_ctx * n_kv_mla * key_len * bpe)

        key_len = self._kv_key_length
        val_len = self._kv_value_length

        # Path 2: Hybrid Mamba/Attention (Qwen3.5-27B, Qwen3.5-35B-A3B)
        # Only 1 in N layers is attention; the rest are Mamba (no KV cache).
        if (
            self._ssm_inner_size is not None
            and self._full_attention_interval is not None
        ):
            fai = self._full_attention_interval
            n_attn = -(-n_layers // fai) if fai > 0 else n_layers  # ceiling division
            if key_len is not None and val_len is not None:
                return int(n_attn * n_ctx * n_kv * (key_len + val_len) * bpe)
            head_dim = self._embedding_length // self._n_heads if self._n_heads else 128  # type: ignore[operator]
            return int(n_attn * n_ctx * n_kv * 2 * head_dim * bpe)

        # Path 3: Sliding window (Gemma 2/3/3n/4, gpt-oss, Cohere2 ...).
        # Pattern is filled in by the resolver at parse time; if absent,
        # falls through to the legacy 1/4-global heuristic below.
        # Per-layer-type ``--parallel N`` accounting (verified empirically
        # against ``llama-server``):
        #   * non-SWA layers:    total cells = n_ctx, partitioned across
        #                         slots -> total memory CONSTANT in slots.
        #   * SWA layers:         per-slot cells = 2 * sliding_window
        #                         (capped at n_ctx and at per_slot_ctx
        #                         when ctx is split among many slots) ->
        #                         total memory grows LINEARLY in slots.
        # ``--swa-full`` forces full n_ctx for SWA layers instead.
        # ``--ctx-checkpoints N`` adds N snapshots per SWA layer per slot.
        if (
            self._sliding_window is not None
            and self._sliding_window > 0
            and key_len is not None
            and val_len is not None
        ):
            swa = self._sliding_window
            per_slot_ctx = max(1, n_ctx // slots)
            # ``--swa-full`` makes SWA layers cache the full context just
            # like non-SWA: cells get partitioned across slots, so per-slot
            # cells = per_slot_ctx and the slots*per-slot product collapses
            # back to the constant ``n_ctx`` total.  Otherwise SWA caches
            # 2*sliding_window per slot, clamped at the per-slot ctx.
            swa_cells_per_slot = (
                per_slot_ctx if swa_full else min(n_ctx, 2 * swa, per_slot_ctx)
            )
            key_len_swa = self._kv_key_length_swa or key_len
            val_len_swa = self._kv_value_length_swa or val_len
            if self._sliding_window_pattern is not None:
                global_bytes = 0.0  # constant across slots
                swa_bytes_per_slot = 0.0  # multiplied by slots
                checkpoint_extra_per_slot = 0.0
                # Iterate only over layers that allocate their own KV;
                # the trailing ``shared`` layers reuse earlier caches.
                for layer_idx in range(n_layers_kv):
                    layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
                    is_swa = (
                        layer_idx < len(self._sliding_window_pattern)
                        and self._sliding_window_pattern[layer_idx]
                    )
                    if is_swa:
                        swa_bytes_per_slot += (
                            swa_cells_per_slot
                            * layer_n_kv
                            * (key_len_swa + val_len_swa)
                            * bpe
                        )
                        if ctx_checkpoints > 0 and not swa_full:
                            checkpoint_extra_per_slot += (
                                ctx_checkpoints
                                * swa
                                * layer_n_kv
                                * (key_len_swa + val_len_swa)
                                * bpe
                            )
                    else:
                        global_bytes += n_ctx * layer_n_kv * (key_len + val_len) * bpe
                return int(
                    global_bytes
                    + slots * (swa_bytes_per_slot + checkpoint_extra_per_slot)
                )
            n_global = max(1, n_layers_kv // 4)
            n_swa = n_layers_kv - n_global
            kv_per_token = n_kv * (key_len + val_len) * bpe
            kv_per_token_swa = n_kv * (key_len_swa + val_len_swa) * bpe
            global_bytes = n_global * n_ctx * kv_per_token
            swa_bytes_per_slot = n_swa * swa_cells_per_slot * kv_per_token_swa
            checkpoint_extra_per_slot = (
                ctx_checkpoints * n_swa * swa * kv_per_token_swa
                if ctx_checkpoints > 0 and not swa_full
                else 0.0
            )
            return int(
                global_bytes + slots * (swa_bytes_per_slot + checkpoint_extra_per_slot)
            )

        # Path 4: Standard GQA with explicit key/value dimensions
        if key_len is not None and val_len is not None:
            return int(n_layers_kv * n_ctx * n_kv * (key_len + val_len) * bpe)

        # Path 5: Legacy fallback (old GGUFs without explicit dimensions)
        head_dim = self._embedding_length // self._n_heads if self._n_heads else 128  # type: ignore[operator]
        return int(2 * n_kv * head_dim * n_layers_kv * n_ctx * bpe)

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
        ctx_checkpoints: int = 0,
        kv_on_gpu: bool = True,
    ) -> int:
        """Return the largest context length that fits in GPU VRAM.

        Uses 90% of available VRAM as the budget (matching _select_gpus
        threshold -- 10% reserved for compute buffers, CUDA context,
        scratch space, flash-attn workspace, etc.).
        If the model weights alone don't fit, returns min_ctx unchanged.

        ``kv_on_gpu`` mirrors ``--kv-offload`` (default on). When False
        the KV cache lives in CPU RAM and doesn't compete with weights
        for VRAM; the requested context is honored verbatim. The other
        keyword args mirror ``_estimate_kv_cache_bytes``.
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
            ctx_checkpoints = ctx_checkpoints,
        )

        budget_bytes = available_mib * 1024 * 1024 * 0.90
        model_footprint = model_size_bytes

        # Check if requested context already fits
        kv = self._estimate_kv_cache_bytes(requested_ctx, cache_type_kv, **kv_kwargs)
        if model_footprint + kv <= budget_bytes:
            return requested_ctx

        # Model weights alone exceed budget -- can't help by reducing ctx.
        # Return requested_ctx unchanged; --fit will handle VRAM management.
        if model_footprint >= budget_bytes:
            logger.debug(
                "Model footprint exceeds GPU budget before KV cache",
                requested_ctx = requested_ctx,
                available_mib = available_mib,
                model_size_gb = round(model_footprint / (1024**3), 2),
            )
            return requested_ctx

        # Binary search for max context that fits
        remaining = budget_bytes - model_footprint
        effective_min = min(min_ctx, requested_ctx)
        lo, hi = effective_min, requested_ctx
        best = effective_min
        while lo <= hi:
            mid = (lo + hi) // 2
            kv = self._estimate_kv_cache_bytes(mid, cache_type_kv, **kv_kwargs)
            if kv <= remaining:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Round down to nearest 256 for alignment, but never exceed requested_ctx
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
    ) -> Optional[tuple[str, int]]:
        """Find the smallest GGUF variant (including all shards) that fits.

        Groups split shards by variant prefix and sums their sizes.
        For example, UD-Q4_K_XL with 9 shards of 50 GB each = 450 GB total.

        Returns (first_shard_filename, total_size_bytes) or None if nothing fits.
        """
        try:
            from huggingface_hub import get_paths_info, list_repo_files

            files = list_repo_files(hf_repo, token = hf_token)
            gguf_files = [
                f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()
            ]
            if not gguf_files:
                return None

            # Get sizes for all GGUF files
            path_infos = list(get_paths_info(hf_repo, gguf_files, token = hf_token))
            size_map = {p.path: (p.size or 0) for p in path_infos}

            # Group files by variant: shards share a prefix before -NNNNN-of-NNNNN
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

            # Sort by total size ascending and pick the smallest that fits
            variant_sizes.sort(key = lambda x: x[1])
            for first_file, total_size, _ in variant_sizes:
                if total_size > 0 and total_size <= free_bytes:
                    return first_file, total_size

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
        """
        Read lines from the subprocess stdout in a background thread.

        This prevents a pipe-buffer deadlock on Windows where the default
        pipe buffer is only ~4 KB.  Without draining, llama-server blocks
        on writes and never becomes healthy.
        """
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)
                    logger.debug(f"[llama-server] {line}")
        except (ValueError, OSError):
            # Pipe closed — process is terminating
            pass

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

        for _ in range(alen):
            LlamaCppBackend._gguf_skip_value(f, atype)
        return None

    def _read_gguf_metadata(self, gguf_path: str) -> None:
        """Read context_length, architecture params, and chat_template from a GGUF header.

        Parses only the KV pairs we need (~30ms even for multi-GB files).
        For split GGUFs, metadata is always in shard 1.
        """
        # Reset metadata from any previously loaded model so stale flags
        # (eg _supports_reasoning) do not carry over when switching models.
        self._context_length = None
        self._chat_template = None
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_style = "enable_thinking"
        self._reasoning_default = True
        self._supports_preserve_thinking = False
        self._supports_tools = False
        self._n_layers = None
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
        self._shared_kv_layers = None

        try:
            WANTED = {
                "general.architecture",
                "tokenizer.chat_template",
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
            # Additional arch-specific keys are added dynamically once
            # we know the architecture name.
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
                    # Tolerate truncated input (e.g., a partial header
                    # fetched via HTTP byte-range): bail out gracefully
                    # so the resolver fallback still runs on whatever
                    # we did manage to parse.
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
                                if (
                                    key.startswith("general.")
                                    and key != "general.architecture"
                                ):
                                    general[key] = val_s
                                if key == "general.architecture":
                                    arch = val_s
                                    arch_keys = {
                                        f"{arch}.context_length": "context_length",
                                        f"{arch}.block_count": "n_layers",
                                        f"{arch}.attention.head_count_kv": "n_kv_heads",
                                        f"{arch}.attention.head_count": "n_heads",
                                        f"{arch}.embedding_length": "embedding_length",
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
                                    }
                                elif key == "tokenizer.chat_template":
                                    self._chat_template = val_s
                            elif vtype in (4, 10):  # UINT32 or UINT64
                                val_i = (
                                    struct.unpack("<I", f.read(4))[0]
                                    if vtype == 4
                                    else struct.unpack("<Q", f.read(8))[0]
                                )
                                attr = arch_keys.get(key)
                                if attr:
                                    if attr == "sliding_window_pattern":
                                        sliding_window_pattern_period = val_i
                                    else:
                                        setattr(self, f"_{attr}", val_i)
                            elif vtype == 9:  # ARRAY
                                atype = struct.unpack("<I", f.read(4))[0]
                                alen = struct.unpack("<Q", f.read(8))[0]
                                val_a = self._gguf_read_array_value(f, atype, alen)
                                attr = arch_keys.get(key)
                                if attr == "n_kv_heads" and val_a is not None:
                                    self._n_kv_heads_by_layer = [int(x) for x in val_a]
                                    if self._n_kv_heads is None and val_a:
                                        self._n_kv_heads = max(int(x) for x in val_a)
                                elif (
                                    attr == "sliding_window_pattern"
                                    and val_a is not None
                                ):
                                    self._sliding_window_pattern = [
                                        bool(x) for x in val_a
                                    ]
                                    sliding_window_pattern_period = None
                            else:
                                self._gguf_skip_value(f, vtype)
                        else:
                            self._gguf_skip_value(f, vtype)
                    except (struct.error, UnicodeDecodeError):
                        # Truncated input (e.g., HTTP byte-range fetch
                        # of just the GGUF header); break so the
                        # resolver fallback still runs on what we have.
                        break

            # Expand a scalar period straight from the GGUF first.
            if (
                self._sliding_window_pattern is None
                and sliding_window_pattern_period
                and self._n_layers
            ):
                self._sliding_window_pattern = [
                    (i + 1) % sliding_window_pattern_period != 0
                    for i in range(self._n_layers)
                ]

            # Otherwise hand off to the resolver (cache / bootstrap /
            # transformers / HF). See `_resolve_swa_pattern`.
            if (
                self._sliding_window_pattern is None
                and self._sliding_window
                and self._n_layers
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
                        f"{general['general.organization']}/"
                        f"{general['general.basename']}".replace(" ", "-")
                        if general.get("general.organization")
                        and general.get("general.basename")
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
                logger.info(
                    f"GGUF metadata: chat_template={len(self._chat_template)} chars"
                )
                # Detect thinking/reasoning support from chat template
                flags = detect_reasoning_flags(
                    self._chat_template,
                    self._model_identifier,
                    log_source = "GGUF metadata",
                )
                self._supports_reasoning = flags["supports_reasoning"]
                self._reasoning_style = flags["reasoning_style"]
                self._reasoning_always_on = flags["reasoning_always_on"]
                self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                self._supports_tools = flags["supports_tools"]
        except Exception as e:
            logger.warning(f"Failed to read GGUF metadata: {e}")

    # ── HF download (no lock held) ───────────────────────────────

    def _download_gguf(
        self,
        *,
        hf_repo: str,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """Download GGUF file(s) from HuggingFace. Returns local path.

        Runs WITHOUT self._lock so that unload_model() can set
        _cancel_event at any time. Checks _cancel_event between
        each shard download.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for HF model loading. "
                "Install it with: pip install huggingface_hub"
            )

        # Determine the filename from the variant
        gguf_filename = None
        gguf_extra_shards: list[str] = []
        if hf_variant:
            try:
                from huggingface_hub import list_repo_files

                files = list_repo_files(hf_repo, token = hf_token)
                variant_lower = hf_variant.lower()
                boundary = re.compile(
                    r"(?<![a-zA-Z0-9])" + re.escape(variant_lower) + r"(?![a-zA-Z0-9])"
                )
                gguf_files = sorted(
                    f
                    for f in files
                    if f.endswith(".gguf") and boundary.search(f.lower())
                )
                if gguf_files:
                    gguf_filename = gguf_files[0]
                    m = _SHARD_FULL_RE.match(gguf_filename)
                    if m:
                        prefix = m.group(1)
                        total = m.group(3)
                        sibling_pat = re.compile(
                            r"^"
                            + re.escape(prefix)
                            + r"-\d{5}-of-"
                            + re.escape(total)
                            + r"\.gguf$"
                        )
                        gguf_extra_shards = [
                            f for f in gguf_files[1:] if sibling_pat.match(f)
                        ]
            except Exception as e:
                logger.warning(f"Could not list repo files: {e}")

            if not gguf_filename:
                repo_name = hf_repo.split("/")[-1].replace("-GGUF", "")
                gguf_filename = f"{repo_name}-{hf_variant}.gguf"

        # Check disk space and fall back to a smaller variant if needed
        all_gguf_files = [gguf_filename] + gguf_extra_shards
        try:
            import os

            from huggingface_hub import get_paths_info, try_to_load_from_cache

            path_infos = list(get_paths_info(hf_repo, all_gguf_files, token = hf_token))
            total_bytes = sum((p.size or 0) for p in path_infos)

            # Subtract bytes already present in the HF cache so we only
            # preflight against what we actually have to download. Without
            # this, re-loading a cached large model (e.g. MiniMax-M2.7-GGUF
            # at 131 GB) fails cold whenever free disk is below the full
            # weight footprint, even though nothing needs downloading.
            already_cached_bytes = 0
            for p in path_infos:
                if not p.size:
                    continue
                try:
                    cached_path = try_to_load_from_cache(hf_repo, p.path)
                except Exception:
                    cached_path = None
                if isinstance(cached_path, str) and os.path.exists(cached_path):
                    try:
                        on_disk = os.path.getsize(cached_path)
                    except OSError:
                        on_disk = 0
                    # Count as satisfied only when the full blob is present.
                    if on_disk >= p.size:
                        already_cached_bytes += p.size

            total_download_bytes = max(0, total_bytes - already_cached_bytes)

            if total_download_bytes > 0:
                cache_dir = os.environ.get(
                    "HF_HUB_CACHE",
                    str(Path.home() / ".cache" / "huggingface" / "hub"),
                )
                Path(cache_dir).mkdir(parents = True, exist_ok = True)
                free_bytes = shutil.disk_usage(cache_dir).free

                total_gb = total_download_bytes / (1024**3)
                free_gb = free_bytes / (1024**3)
                cached_gb = already_cached_bytes / (1024**3)

                logger.info(
                    f"GGUF download: {total_gb:.1f} GB needed "
                    f"({cached_gb:.1f} GB already cached), "
                    f"{free_gb:.1f} GB free on disk"
                )

                if total_download_bytes > free_bytes:
                    smaller = self._find_smallest_fitting_variant(
                        hf_repo,
                        free_bytes,
                        hf_token,
                    )
                    if smaller:
                        fallback_file, fallback_size = smaller
                        logger.info(
                            f"Selected variant too large ({total_gb:.1f} GB), "
                            f"falling back to {fallback_file} ({fallback_size / (1024**3):.1f} GB)"
                        )
                        gguf_filename = fallback_file
                        _m = _SHARD_RE.match(gguf_filename)
                        _prefix = _m.group(1) if _m else None
                        if _prefix:
                            gguf_extra_shards = sorted(
                                f
                                for f in all_gguf_files
                                if f.startswith(_prefix)
                                and f != gguf_filename
                                and "mmproj" not in f.lower()
                            )
                        else:
                            gguf_extra_shards = []
                    else:
                        raise RuntimeError(
                            f"Not enough disk space to download any variant. "
                            f"Only {free_gb:.1f} GB free in {cache_dir}"
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
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            dl_start = time.monotonic()
            local_path = hf_hub_download(
                repo_id = hf_repo,
                filename = gguf_filename,
                token = hf_token,
            )
            for shard in gguf_extra_shards:
                if self._cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                logger.info(f"Resolving GGUF shard: {shard}")
                hf_hub_download(
                    repo_id = hf_repo,
                    filename = shard,
                    token = hf_token,
                )
        except RuntimeError as e:
            if "Cancelled" in str(e):
                raise
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
            )

        dl_elapsed = time.monotonic() - dl_start
        if dl_elapsed < 2.0:
            logger.info(f"GGUF resolved from cache: {local_path}")
        else:
            logger.info(f"GGUF downloaded in {dl_elapsed:.1f}s: {local_path}")
        return local_path

    def _download_mmproj(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
    ) -> Optional[str]:
        """Download the mmproj (vision projection) file from a GGUF repo.

        Prefers mmproj-F16.gguf, falls back to any mmproj*.gguf file.
        Returns the local path, or None if no mmproj file exists.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(hf_repo, token = hf_token)
            mmproj_files = sorted(
                f for f in files if f.endswith(".gguf") and "mmproj" in f.lower()
            )
            if not mmproj_files:
                return None

            # Prefer F16 variant
            target = None
            for f in mmproj_files:
                if f.lower().endswith("-f16.gguf"):
                    target = f
                    break
            if target is None:
                target = mmproj_files[0]

            logger.info(f"Downloading mmproj: {hf_repo}/{target}")
            local_path = hf_hub_download(
                repo_id = hf_repo,
                filename = target,
                token = hf_token,
            )
            return local_path
        except Exception as e:
            logger.warning(f"Could not download mmproj: {e}")
            return None

    # ── Lifecycle ─────────────────────────────────────────────────

    def load_model(
        self,
        *,
        # Local mode: pass a path to a .gguf file
        gguf_path: Optional[str] = None,
        # Vision projection (mmproj) for local vision models
        mmproj_path: Optional[str] = None,
        # HF mode: let llama-server download via -hf "repo:quant"
        hf_repo: Optional[str] = None,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        # Common
        model_identifier: str,
        is_vision: bool = False,
        n_ctx: int = 4096,
        chat_template_override: Optional[str] = None,
        cache_type_kv: Optional[str] = None,
        speculative_type: Optional[str] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,  # Accepted for caller compat, unused
        n_parallel: int = 1,
        extra_args: Optional[List[str]] = None,
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
        self._cancel_event.clear()

        # ── Phase 1: kill old process (under lock, fast) ──────────
        with self._lock:
            self._kill_process()

        binary = self._find_llama_server_binary()
        if not binary:
            raise RuntimeError(
                "llama-server binary not found. "
                "Run setup.sh to build it, install llama.cpp, "
                "or set LLAMA_SERVER_PATH environment variable."
            )

        # ── Phase 2: download (NO lock held, so cancel can proceed) ──
        if hf_repo:
            model_path = self._download_gguf(
                hf_repo = hf_repo,
                hf_variant = hf_variant,
                hf_token = hf_token,
            )
            # Auto-download mmproj for vision models
            if is_vision and not mmproj_path:
                mmproj_path = self._download_mmproj(
                    hf_repo = hf_repo,
                    hf_token = hf_token,
                )
        elif gguf_path:
            if not Path(gguf_path).is_file():
                raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
            model_path = gguf_path
        else:
            raise ValueError("Either gguf_path or hf_repo must be provided")

        # Set identifier early so _read_gguf_metadata can use it for DeepSeek detection
        self._model_identifier = model_identifier

        # Read GGUF metadata (context_length, chat_template) -- fast, header only
        self._read_gguf_metadata(model_path)

        # Check cancel after download
        if self._cancel_event.is_set():
            logger.info("Load cancelled after download phase")
            return False

        # ── Phase 3: start llama-server (under lock) ──────────────
        with self._lock:
            # Re-check cancel inside lock
            if self._cancel_event.is_set():
                logger.info("Load cancelled before server start")
                return False

            self._port = self._find_free_port()

            # Select GPU(s) based on model size + estimated KV cache.
            # Seed safe defaults before GPU probing so the except path
            # still has valid state to publish.
            effective_ctx = n_ctx if n_ctx > 0 else (self._context_length or 0)
            max_available_ctx = self._context_length or effective_ctx
            try:
                model_size = self._get_gguf_size_bytes(model_path)
                gpus = self._get_gpu_free_memory()

                # Resolve effective context: 0 means let llama-server use the
                # model's native length.  Only expand to a known native length
                # if metadata is available; otherwise preserve 0 as a sentinel.
                if n_ctx > 0:
                    effective_ctx = n_ctx
                elif self._context_length is not None:
                    effective_ctx = self._context_length
                else:
                    effective_ctx = 0
                original_ctx = effective_ctx
                # Default UI ceiling to the model's native context length.
                # GPU/VRAM-fit logic below may shrink this if hardware is limited.
                max_available_ctx = self._context_length or effective_ctx

                # Auto-cap context to fit in GPU VRAM and select GPUs.
                #
                # Two policies depending on whether the user set n_ctx:
                #
                # Explicit n_ctx (user chose a context length):
                #   Honor it. Try the full requested context with _select_gpus
                #   (which uses as many GPUs as needed). Only cap if it doesn't
                #   fit on any GPU combination.
                #
                # Auto n_ctx=0 (model's native context):
                #   Prefer fewer GPUs with reduced context over more GPUs,
                #   since multi-GPU is slower and the user didn't ask for a
                #   specific context length.
                gpu_indices, use_fit = None, True
                explicit_ctx = n_ctx > 0

                if gpus and self._can_estimate_kv() and effective_ctx > 0:
                    # Compute the largest hardware-aware cap from the model's
                    # native context across all usable GPU subsets (for UI
                    # bounds), independent of the currently requested context.
                    native_ctx_for_cap = self._context_length or effective_ctx
                    if native_ctx_for_cap > 0:
                        ranked_for_cap = sorted(gpus, key = lambda g: g[1], reverse = True)
                        best_cap = 0
                        for n_gpus in range(1, len(ranked_for_cap) + 1):
                            subset = ranked_for_cap[:n_gpus]
                            pool_mib = sum(free for _, free in subset)
                            capped = self._fit_context_to_vram(
                                native_ctx_for_cap,
                                pool_mib,
                                model_size,
                                cache_type_kv,
                                n_parallel = n_parallel,
                            )
                            kv = self._estimate_kv_cache_bytes(
                                capped, cache_type_kv, n_parallel = n_parallel
                            )
                            total_mib = (model_size + kv) / (1024 * 1024)
                            if total_mib <= pool_mib * 0.90:
                                best_cap = max(best_cap, capped)
                        if best_cap > 0:
                            max_available_ctx = best_cap
                        else:
                            # Weights exceed 90% of every GPU subset's free
                            # memory, so there is no fitting context. Anchor
                            # the UI's "safe zone" threshold at 4096 (the
                            # spec's default when the model cannot fit) so
                            # the ctx slider shows the "might be slower"
                            # warning as soon as the user drags above the
                            # fallback default instead of never.
                            max_available_ctx = min(4096, native_ctx_for_cap)

                    if explicit_ctx:
                        # Honor the user's requested context verbatim. If it
                        # fits, pin GPUs and skip --fit; if it doesn't, ship
                        # -c <user_ctx> --fit on and let llama-server flex
                        # -ngl (CPU layer offload). The UI is expected to
                        # have surfaced the "might be slower" warning before
                        # the user submitted a ctx above the fit ceiling.
                        requested_total = model_size + self._estimate_kv_cache_bytes(
                            effective_ctx, cache_type_kv, n_parallel = n_parallel
                        )
                        gpu_indices, use_fit = self._select_gpus(requested_total, gpus)
                        # No silent shrink: effective_ctx stays == n_ctx.
                    else:
                        # Auto context: prefer fewer GPUs, cap context to fit.
                        ranked = sorted(gpus, key = lambda g: g[1], reverse = True)
                        for n_gpus in range(1, len(ranked) + 1):
                            subset = ranked[:n_gpus]
                            pool_mib = sum(free for _, free in subset)
                            capped = self._fit_context_to_vram(
                                effective_ctx,
                                pool_mib,
                                model_size,
                                cache_type_kv,
                                n_parallel = n_parallel,
                            )
                            kv = self._estimate_kv_cache_bytes(
                                capped, cache_type_kv, n_parallel = n_parallel
                            )
                            total_mib = (model_size + kv) / (1024 * 1024)
                            if total_mib <= pool_mib * 0.90:
                                effective_ctx = capped
                                gpu_indices = sorted(idx for idx, _ in subset)
                                use_fit = False
                                break
                        else:
                            # No subset can host the weights (weights alone
                            # exceed 90% of every pool). Per spec, default
                            # the UI-visible context to 4096 and let
                            # --fit on flex -ngl so llama-server offloads
                            # layers to CPU RAM.
                            effective_ctx = min(4096, effective_ctx)

                elif gpus:
                    # Can't estimate KV -- fall back to file-size-only check.
                    # Without KV estimation we cannot prove a hardware cap, so
                    # keep the ceiling at the native context (already the default).
                    logger.debug(
                        "Falling back to file-size-only GPU selection",
                        model_size_gb = round(model_size / (1024**3), 2),
                    )
                    gpu_indices, use_fit = self._select_gpus(model_size, gpus)
                    if use_fit and not explicit_ctx:
                        # Weights don't fit on any subset. Default the UI to
                        # 4096 so the slider doesn't land on an unusable native
                        # context. --fit on will flex -ngl at runtime.
                        effective_ctx = (
                            min(4096, effective_ctx) if effective_ctx > 0 else 4096
                        )

                if effective_ctx < original_ctx:
                    kv_est = self._estimate_kv_cache_bytes(
                        effective_ctx, cache_type_kv, n_parallel = n_parallel
                    )
                    logger.info(
                        f"Context auto-reduced: {original_ctx} -> {effective_ctx} "
                        f"(model: {model_size / (1024**3):.1f} GB, "
                        f"est. KV cache: {kv_est / (1024**3):.1f} GB)"
                    )

                kv_cache_bytes = self._estimate_kv_cache_bytes(
                    effective_ctx, cache_type_kv, n_parallel = n_parallel
                )
                logger.info(
                    f"GGUF size: {model_size / (1024**3):.1f} GB, "
                    f"est. KV cache: {kv_cache_bytes / (1024**3):.1f} GB, "
                    f"context: {effective_ctx}, "
                    f"GPUs free: {gpus}, selected: {gpu_indices}, fit: {use_fit}"
                )
            except Exception as e:
                logger.warning(f"GPU selection failed ({e}), using --fit on")
                gpu_indices, use_fit = None, True
                effective_ctx = n_ctx  # fall back to original

            cmd = [
                binary,
                "-m",
                model_path,
                "--port",
                str(self._port),
                "-c",
                str(effective_ctx) if effective_ctx > 0 else "0",
                "--parallel",
                str(n_parallel),
                "--flash-attn",
                "on",  # Force flash attention for speed
                # Error out at n_ctx instead of silently rotating the KV cache; frontend catches it and points the user at "Context Length".
                "--no-context-shift",
            ]

            if use_fit:
                cmd.extend(["--fit", "on"])
            elif gpu_indices is not None:
                # Model fits on selected GPU(s) -- offload all layers
                cmd.extend(["-ngl", "-1"])

            # -1 = llama.cpp auto-detect (physical cores). Pass explicitly so we
            # do not inherit llama-server's internal default, which has historically
            # varied (hardware concurrency incl. hyperthreads on some builds).
            cmd.extend(["--threads", str(n_threads if n_threads is not None else -1)])

            # Always enable Jinja chat template rendering for proper template support
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
            if cache_type_kv and cache_type_kv in _valid_cache_types:
                cmd.extend(
                    ["--cache-type-k", cache_type_kv, "--cache-type-v", cache_type_kv]
                )
                self._cache_type_kv = cache_type_kv
                logger.info(f"KV cache type: {cache_type_kv}")
            else:
                self._cache_type_kv = None

            # Speculative decoding (n-gram self-speculation, zero VRAM cost)
            # ngram-mod: ~16 MB shared hash pool, constant memory/complexity,
            # variable draft lengths.  Helps most when the model repeats
            # existing text (code refactoring, summarization, reasoning).
            # For general chat with low repetition, overhead is ~5 ms.
            #
            # Benchmarks from upstream llama.cpp speculative-decoding PRs:
            #   Scenario                        | Without | With    | Speedup
            #   gpt-oss-120b code refactor      | 181 t/s | 446 t/s | 2.5x
            #   Qwen3-235B offloaded            |  12 t/s |  21 t/s | 1.8x
            #   gpt-oss-120b repeat (92% accept)| 181 t/s | 814 t/s | 4.5x
            #
            # Params from llama.cpp docs (docs/speculative.md):
            #   --spec-ngram-size-n 24  (small n not recommended)
            #   --draft-min 48 --draft-max 64 (MoEs need long drafts;
            #     dense models can reduce these)
            # ref: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
            # ref: https://github.com/ggml-org/llama.cpp/pull/19164
            # ref: https://github.com/ggml-org/llama.cpp/pull/18471
            # ``"default"`` -> let llama-server pick a sensible spec
            # config via ``--spec-default``. Explicit type names are
            # passed through with the manual draft tuning we've shipped
            # historically so power users keep their overrides.
            _valid_spec_types = {"ngram-simple", "ngram-mod"}
            normalized_spec = (
                speculative_type.lower().strip() if speculative_type else None
            )
            if normalized_spec and normalized_spec != "off" and not is_vision:
                if normalized_spec == "default":
                    cmd.append("--spec-default")
                    self._speculative_type = "default"
                elif normalized_spec in _valid_spec_types:
                    cmd.extend(["--spec-type", normalized_spec])
                    if normalized_spec == "ngram-mod":
                        cmd.extend(
                            [
                                "--spec-ngram-size-n",
                                "24",
                                "--draft-min",
                                "48",
                                "--draft-max",
                                "64",
                            ]
                        )
                    self._speculative_type = normalized_spec
                else:
                    self._speculative_type = None
            else:
                self._speculative_type = None

            # Apply custom chat template override if provided
            if chat_template_override:
                import tempfile

                self._chat_template = chat_template_override
                flags = detect_reasoning_flags(
                    self._chat_template,
                    self._model_identifier,
                    log_source = "GGUF chat template override",
                )
                self._supports_reasoning = flags["supports_reasoning"]
                self._reasoning_style = flags["reasoning_style"]
                self._reasoning_always_on = flags["reasoning_always_on"]
                self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                self._supports_tools = flags["supports_tools"]

                self._chat_template_file = tempfile.NamedTemporaryFile(
                    mode = "w",
                    suffix = ".jinja",
                    delete = False,
                    prefix = "unsloth_chat_template_",
                )
                self._chat_template_file.write(chat_template_override)
                self._chat_template_file.close()
                cmd.extend(["--chat-template-file", self._chat_template_file.name])
                logger.info(
                    f"Using custom chat template file: {self._chat_template_file.name}"
                )

            # For reasoning models, set default thinking mode.
            # Qwen3.5/3.6 models below 9B (0.8B, 2B, 4B) disable thinking by default.
            # Only 9B and larger enable thinking.
            # Always-on templates ignore the kwarg entirely, so skip.
            if self._supports_reasoning and not self._reasoning_always_on:
                thinking_default = True
                mid = (model_identifier or "").lower()
                if "qwen3.5" in mid or "qwen3.6" in mid:
                    size_val = _extract_model_size_b(mid)
                    if size_val is not None and size_val < 9:
                        thinking_default = False
                self._reasoning_default = thinking_default
                reasoning_kw = self._reasoning_kwargs(thinking_default)
                cmd.extend(
                    [
                        "--chat-template-kwargs",
                        json.dumps(reasoning_kw),
                    ]
                )
                logger.info(f"Reasoning model: {reasoning_kw} by default")

            if mmproj_path:
                if not Path(mmproj_path).is_file():
                    logger.warning(f"mmproj file not found: {mmproj_path}")
                else:
                    cmd.extend(["--mmproj", mmproj_path])
                    logger.info(f"Using mmproj for vision: {mmproj_path}")

            # Option C: add --api-key for direct client access when enabled
            import os as _os
            import secrets as _secrets

            if _os.getenv("UNSLOTH_DIRECT_STREAM", "0") == "1":
                self._api_key = _secrets.token_urlsafe(32)
                cmd.extend(["--api-key", self._api_key])
                logger.info("llama-server started with --api-key for direct streaming")
            else:
                self._api_key = None

            # User-supplied pass-through args go last so llama.cpp's
            # last-wins flag parsing lets the user override Studio's
            # auto-set tier-2 flags (e.g. --cache-type-k, --spec-type).
            # The route layer has already validated this list against
            # the managed-flag denylist via validate_extra_args().
            if extra_args:
                cmd.extend(str(a) for a in extra_args)
                logger.info(
                    f"Appending user extra args to llama-server: {list(extra_args)}"
                )

            _log_cmd = list(cmd)
            if "--api-key" in _log_cmd:
                _ki = _log_cmd.index("--api-key") + 1
                if _ki < len(_log_cmd):
                    _log_cmd[_ki] = "<redacted>"
            logger.info(f"Starting llama-server: {' '.join(_log_cmd)}")

            # Set library paths so llama-server can find its shared libs and CUDA DLLs
            import os
            import sys

            env = child_env_without_native_path_secret()
            binary_dir = str(Path(binary).parent)

            if sys.platform == "win32":
                # On Windows, CUDA DLLs (cublas64_12.dll, cudart64_12.dll, etc.)
                # must be on PATH. Add CUDA_PATH\bin if available.
                path_dirs = [binary_dir]
                cuda_path = os.environ.get("CUDA_PATH", "")
                if cuda_path:
                    cuda_bin = os.path.join(cuda_path, "bin")
                    if os.path.isdir(cuda_bin):
                        path_dirs.append(cuda_bin)
                    # Some CUDA installs put DLLs in bin\x64
                    cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
                    if os.path.isdir(cuda_bin_x64):
                        path_dirs.append(cuda_bin_x64)
                existing_path = env.get("PATH", "")
                env["PATH"] = ";".join(path_dirs) + ";" + existing_path
            else:
                # Linux: set LD_LIBRARY_PATH for shared libs next to the binary
                # and CUDA runtime libs (libcudart, libcublas, etc.)
                import platform

                lib_dirs = [binary_dir]
                _arch = platform.machine()  # x86_64, aarch64, etc.

                # Pip-installed nvidia CUDA runtime libs (e.g. torch's
                # bundled cuda-bindings).  The prebuilt llama.cpp binary
                # links against libcudart.so.13 / libcublas.so.13 which
                # live here, not in /usr/local/cuda.
                import glob as _glob

                for _nv_pattern in [
                    os.path.join(
                        sys.prefix,
                        "lib",
                        "python*",
                        "site-packages",
                        "nvidia",
                        "cu*",
                        "lib",
                    ),
                    os.path.join(
                        sys.prefix,
                        "lib",
                        "python*",
                        "site-packages",
                        "nvidia",
                        "cudnn",
                        "lib",
                    ),
                    os.path.join(
                        sys.prefix,
                        "lib",
                        "python*",
                        "site-packages",
                        "nvidia",
                        "nvjitlink",
                        "lib",
                    ),
                ]:
                    for _nv_dir in _glob.glob(_nv_pattern):
                        if os.path.isdir(_nv_dir):
                            lib_dirs.append(_nv_dir)

                for cuda_lib in [
                    "/usr/local/cuda/lib64",
                    f"/usr/local/cuda/targets/{_arch}-linux/lib",
                    # Fallback CUDA compat paths (e.g. binary built with
                    # CUDA 12 on a system where default /usr/local/cuda
                    # points to CUDA 13+).
                    "/usr/local/cuda-12/lib64",
                    "/usr/local/cuda-12.8/lib64",
                    f"/usr/local/cuda-12/targets/{_arch}-linux/lib",
                    f"/usr/local/cuda-12.8/targets/{_arch}-linux/lib",
                ]:
                    if os.path.isdir(cuda_lib):
                        lib_dirs.append(cuda_lib)
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                new_ld = ":".join(lib_dirs)
                env["LD_LIBRARY_PATH"] = (
                    f"{new_ld}:{existing_ld}" if existing_ld else new_ld
                )

            # Pin to selected GPU(s). On ROCm, llama-server (and any torch
            # in the subprocess) honors HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES;
            # narrowing only CUDA_VISIBLE_DEVICES leaves an AMD child seeing
            # the full HIP/ROCR set the parent inherited.
            if gpu_indices is not None:
                pinned = ",".join(str(i) for i in gpu_indices)
                env["CUDA_VISIBLE_DEVICES"] = pinned
                try:
                    import torch as _torch

                    if getattr(_torch.version, "hip", None) is not None:
                        env["HIP_VISIBLE_DEVICES"] = pinned
                        env["ROCR_VISIBLE_DEVICES"] = pinned
                except Exception as e:
                    logger.debug(
                        "Failed to set ROCm visibility env vars for child: %s", e
                    )

            # Defensive kill: if a concurrent load slipped past Phase 1
            # (because its `self._process` was None at the time) and
            # already stored a Popen handle here, drop that orphan
            # before we overwrite the reference. See issue #5161.
            self._kill_process()

            self._stdout_lines = []
            self._process = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )

            # Start background thread to drain stdout and prevent pipe deadlock
            self._stdout_thread = threading.Thread(
                target = self._drain_stdout, daemon = True, name = "llama-stdout"
            )
            self._stdout_thread.start()

            # Store the resolved on-disk path, not the caller's kwarg. In
            # HF mode the caller passes gguf_path=None and the real path
            # (``model_path``) is what llama-server is actually mmap'ing.
            # Downstream consumers (load_progress, log lines, etc.) need
            # the path that exists on disk.
            self._gguf_path = model_path
            self._hf_repo = hf_repo
            # For local GGUF files, extract variant from filename if not provided
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
            self._is_vision = is_vision
            self._model_identifier = model_identifier

            # Store the effective (possibly capped) context separately.
            # Do NOT overwrite _context_length -- it holds the model's native
            # context length from GGUF metadata and is used for display/info.
            self._effective_context_length = (
                effective_ctx if effective_ctx > 0 else self._context_length
            )
            self._max_context_length = (
                max_available_ctx
                if max_available_ctx > 0
                else self._effective_context_length
            )

            # Wait for llama-server to become healthy
            if not self._wait_for_health(timeout = 600.0):
                self._kill_process()
                _gguf = gguf_path or ""
                _is_ollama = (
                    ".studio_links" in _gguf
                    or os.sep + "ollama_links" + os.sep in _gguf
                    or os.sep + ".cache" + os.sep + "ollama" + os.sep in _gguf
                    or (self._model_identifier or "").startswith("ollama/")
                )
                # Only show the Ollama-specific message when the server
                # output indicates a GGUF compatibility issue, not for
                # unrelated failures like OOM or missing binaries.
                if _is_ollama:
                    _output = "\n".join(self._stdout_lines[-50:]).lower()
                    _gguf_compat_hints = (
                        "key not found",
                        "unknown model architecture",
                        "failed to load model",
                    )
                    if any(h in _output for h in _gguf_compat_hints):
                        raise RuntimeError(
                            "Some Ollama models do not work with llama.cpp. "
                            "Try a different model, or use this model directly through Ollama instead."
                        )
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
        """Terminate the llama-server subprocess and cancel any in-flight download."""
        self._cancel_event.set()
        with self._lock:
            self._kill_process()
            logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._hf_repo = None
            self._hf_variant = None
            self._is_vision = False
            self._is_audio = False
            self._audio_type = None
            self._port = None
            self._healthy = False
            self._context_length = None
            self._effective_context_length = None
            self._max_context_length = None
            self._chat_template = None
            self._supports_reasoning = False
            self._reasoning_always_on = False
            self._reasoning_style = "enable_thinking"
            self._reasoning_default = True
            self._supports_preserve_thinking = False
            self._supports_tools = False
            self._cache_type_kv = None
            self._speculative_type = None
            self._n_layers = None
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
            self._shared_kv_layers = None
            # Clean up temp chat template file
            if hasattr(self, "_chat_template_file") and self._chat_template_file:
                try:
                    import os

                    os.unlink(self._chat_template_file.name)
                except Exception:
                    pass
                self._chat_template_file = None
            # Free audio codec GPU memory
            if LlamaCppBackend._codec_mgr is not None:
                LlamaCppBackend._codec_mgr.unload()
                LlamaCppBackend._codec_mgr = None
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return True

    def _kill_process(self):
        """Terminate the subprocess if running."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit on SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout = 5)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            self._process = None
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
                self._stdout_thread = None

    @staticmethod
    def _kill_orphaned_servers():
        """Kill orphaned llama-server processes started by studio.

        Only kills processes whose resolved binary lives under a known
        Studio install directory (or matches an exact env-var override)
        to avoid terminating unrelated llama-server instances.

        Mirrors every location that _find_llama_server_binary() can
        return from so that orphans from any supported install path
        are still cleaned up.

        Uses psutil for cross-platform support (Linux, macOS, Windows).
        Falls back to pgrep + /proc/<pid>/exe on Linux when psutil is
        not installed.
        """
        import os
        import signal
        import sys

        try:
            # -- Build the ownership allowlist --------------------------------
            # Two kinds of matches:
            #   exact_binaries  -- env var overrides (exact path match only)
            #   install_roots   -- directory trees that are Studio-owned
            #                      (binary must be *under* one of these)
            install_roots: list[Path] = []

            # Env-mode custom root (mirrors _find_llama_server_binary).
            _is_custom_root = False
            try:
                from utils.paths.storage_roots import studio_root as _sr  # noqa: WPS433

                _resolved_sr = _sr()
                _legacy_studio = Path.home() / ".unsloth" / "studio"
                try:
                    _is_custom_root = _resolved_sr.resolve() != _legacy_studio.resolve()
                except (OSError, ValueError):
                    _is_custom_root = _resolved_sr != _legacy_studio
                if _is_custom_root:
                    install_roots.append(_resolved_sr / "llama.cpp")
            except (ImportError, OSError, ValueError):
                pass

            # Primary install dir (default mode only). Env-mode skips this so
            # a custom-root Studio cannot kill a concurrent default-install
            # Studio's llama-server (same OS user, different install).
            if not _is_custom_root:
                install_roots.append(Path.home() / ".unsloth" / "llama.cpp")

            # Legacy in-tree build dirs (older setup.sh versions)
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

            # Resolve all roots so is_relative_to works reliably
            resolved_roots: list[Path] = []
            for root in install_roots:
                try:
                    resolved_roots.append(root.resolve())
                except OSError:
                    pass

            my_pid = os.getpid()

            # -- Enumerate processes -------------------------------------------
            # Prefer psutil (cross-platform).  Fall back to pgrep + /proc on
            # Linux when psutil is not installed.
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

                        # Check ownership: exact binary match OR binary is
                        # under a known install root (proper ancestry, not
                        # substring).
                        is_ours = exe_path in exact_binaries or any(
                            exe_path.is_relative_to(root) for root in resolved_roots
                        )
                        if not is_ours:
                            continue

                        proc.kill()
                        logger.info(
                            f"Killed orphaned llama-server process "
                            f"(pid={proc.info['pid']})"
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
                    return
                result = subprocess.run(
                    ["pgrep", "-a", "-f", "llama-server"],
                    capture_output = True,
                    text = True,
                    timeout = 5,
                    env = child_env_without_native_path_secret(),
                )
                if result.returncode != 0:
                    return

                for line in result.stdout.strip().splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) < 2:
                        continue
                    pid = int(parts[0])
                    if pid == my_pid:
                        continue

                    # Resolve the actual executable.  /proc/<pid>/exe is a
                    # symlink to the real binary and avoids all cmdline-
                    # parsing ambiguities (spaces in paths, argv rewriting).
                    # Fall back to the first cmdline token when /proc is
                    # unavailable.
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

                    try:
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"Killed orphaned llama-server process (pid={pid})")
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        pass
        except Exception:
            logger.warning("Error during orphan server cleanup", exc_info = True)

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
                # Give the drain thread a moment to collect final output
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout = 2)
                output = "\n".join(self._stdout_lines[-50:])
                logger.error(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Output: {output[:2000]}"
                )
                return False

            try:
                resp = httpx.get(url, timeout = 2.0)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(interval)

        logger.error(f"llama-server health check timed out after {timeout}s")
        return False

    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _parse_tool_calls_from_text(content: str) -> list[dict]:
        """
        Parse tool calls from XML markup in content text.

        Handles formats like:
          <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
          <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
        Closing tags (</tool_call>, </function>, </parameter>) are all optional
        since models frequently omit them.
        """
        tool_calls = []

        # Pattern 1: JSON inside <tool_call> tags.
        # Use balanced-brace extraction that skips braces inside JSON strings.
        for m in _TC_JSON_START_RE.finditer(content):
            brace_start = m.end() - 1  # position of the opening {
            depth, i = 0, brace_start
            in_string = False
            while i < len(content):
                ch = content[i]
                if in_string:
                    if ch == "\\" and i + 1 < len(content):
                        i += 2  # skip escaped character
                        continue
                    if ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                json_str = content[brace_start : i + 1]
                try:
                    obj = json.loads(json_str)
                    tc = {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": obj.get("name", ""),
                            "arguments": obj.get("arguments", {}),
                        },
                    }
                    if isinstance(tc["function"]["arguments"], dict):
                        tc["function"]["arguments"] = json.dumps(
                            tc["function"]["arguments"]
                        )
                    tool_calls.append(tc)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Pattern 2: XML-style <function=name><parameter=key>value</parameter></function>
        # All closing tags optional -- models frequently omit </parameter>,
        # </function>, and/or </tool_call>.
        if not tool_calls:
            # Step 1: Find all <function=name> positions and extract their bodies.
            # Body boundary: use only </tool_call> or next <function= as hard
            # boundaries.  We avoid using </function> as a boundary because
            # code parameter values can contain that literal string.
            # After extracting, we trim a trailing </function> if present.
            func_starts = list(_TC_FUNC_START_RE.finditer(content))
            for idx, fm in enumerate(func_starts):
                func_name = fm.group(1)
                body_start = fm.end()
                # Hard boundaries: next <function= tag or </tool_call>
                next_func = (
                    func_starts[idx + 1].start()
                    if idx + 1 < len(func_starts)
                    else len(content)
                )
                end_tag = _TC_END_TAG_RE.search(content[body_start:])
                if end_tag:
                    body_end = body_start + end_tag.start()
                else:
                    body_end = len(content)
                body_end = min(body_end, next_func)
                body = content[body_start:body_end]
                # Trim trailing </function> if present (it's the real closing tag)
                body = _TC_FUNC_CLOSE_RE.sub("", body)

                # Step 2: Extract parameters from body.
                # For single-parameter functions (the common case: code, command,
                # query), use body end as the only boundary to avoid false matches
                # on </parameter> inside code strings.
                arguments = {}
                param_starts = list(_TC_PARAM_START_RE.finditer(body))
                if len(param_starts) == 1:
                    # Single parameter: value is everything from after the tag
                    # to end of body, trimming any trailing </parameter>.
                    pm = param_starts[0]
                    val = body[pm.end() :]
                    val = _TC_PARAM_CLOSE_RE.sub("", val)
                    arguments[pm.group(1)] = val.strip()
                else:
                    for pidx, pm in enumerate(param_starts):
                        param_name = pm.group(1)
                        val_start = pm.end()
                        # Value ends at next <parameter= or end of body
                        next_param = (
                            param_starts[pidx + 1].start()
                            if pidx + 1 < len(param_starts)
                            else len(body)
                        )
                        val = body[val_start:next_param]
                        # Trim trailing </parameter> if present
                        val = _TC_PARAM_CLOSE_RE.sub("", val)
                        arguments[param_name] = val.strip()

                tc = {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments),
                    },
                }
                tool_calls.append(tc)

        return tool_calls

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

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Iterate over an httpx streaming response with cancel support.

        Checks cancel_event between chunks and on ReadTimeout.  The
        cancel watcher in _stream_with_retry also calls response.close()
        on cancel, which unblocks iter_text() once the response exists.
        During normal streaming llama-server sends tokens frequently,
        so the cancel check between chunks is the primary mechanism.
        """
        text_iter = response.iter_text()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                chunk = next(text_iter)
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                # No data within the timeout window -- just loop back
                # and re-check cancel_event.
                continue

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
        headers: Optional[dict] = None,
    ):
        """Open an httpx streaming POST with cancel support.

        Sends the request once with a long read timeout (120 s) so
        prompt processing (prefill) can finish without triggering a
        retry storm.  The previous 0.5 s timeout caused duplicate POST
        requests every half second, forcing llama-server to restart
        processing each time.

        A background watcher thread provides cancel by closing the
        response when cancel_event is set.  Limitation: httpx does not
        allow interrupting a blocked read from another thread before
        the response object exists, so cancel during the initial
        header wait (prefill phase) only takes effect once headers
        arrive.  After that, response.close() unblocks reads promptly.
        In practice llama-server prefill is 1-5 s for typical prompts,
        during which cancel is deferred -- still much better than the
        old retry storm which made prefill slower.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise GeneratorExit

        # Background watcher: close the response if cancel is requested.
        # Only effective after response headers arrive (httpx limitation).
        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    # Cancel requested. Keep polling until the response object
                    # exists so we can close it, or until the main thread
                    # finishes on its own (_cancel_closed is set in finally).
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        if r is not None:
                            try:
                                r.close()
                                return
                            except Exception as e:
                                logger.debug(
                                    f"Error closing response in cancel watcher: {e}"
                                )
                        # Response not created yet -- wait briefly and retry
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, daemon = True, name = "prefill-cancel"
            )
            watcher.start()

        try:
            # Long read timeout so prefill (prompt processing) can finish
            # without triggering a retry storm.  Cancel during both
            # prefill and streaming is handled by the watcher thread
            # which closes the response, unblocking any httpx read.
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = 120.0,
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
                if cancel_event is not None and cancel_event.is_set():
                    raise GeneratorExit
                yield response
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.CloseError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise GeneratorExit
            raise
        finally:
            _cancel_closed.set()

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
    ) -> Generator[str | dict, None, None]:
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
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        # Pass enable_thinking / reasoning_effort / preserve_thinking per-request
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        # Default cap to the model's effective context length when known,
        # otherwise the conservative floor. The wall-clock backstop below
        # keeps a stuck model from running indefinitely either way.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
        if stop:
            payload["stop"] = stop
        payload["stream_options"] = {"include_usage": True}

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None

        try:
            # _stream_with_retry uses a 120 s read timeout so prefill
            # can finish.  Cancel during streaming is handled by the
            # watcher thread (closes the response on cancel_event).
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            _auth_headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            )
            with httpx.Client(
                timeout = stream_timeout, limits = httpx.Limits(max_keepalive_connections = 0)
            ) as client:
                with self._stream_with_retry(
                    client,
                    url,
                    payload,
                    cancel_event,
                    headers = _auth_headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    has_content_tokens = False
                    reasoning_text = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
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
                                        # Only reasoning_content, no content tokens:
                                        # the model put its entire reply in reasoning
                                        # (e.g. Qwen3 always-think mode). Show it
                                        # as the main response, not as a thinking block.
                                        cumulative = reasoning_text
                                        yield cumulative
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                data = json.loads(line[6:])
                                # Capture server timings/usage from final chunks
                                _chunk_timings = data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    # Handle reasoning/thinking tokens
                                    # llama-server sends these as "reasoning_content"
                                    # Wrap in <think> tags for the frontend parser
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
                                logger.debug(
                                    f"Skipping malformed SSE line: {line[:100]}"
                                )
                        if _stream_done:
                            break  # exit outer for
                    if _metadata_usage or _metadata_timings:
                        yield {
                            "type": "metadata",
                            "usage": _metadata_usage,
                            "timings": _metadata_timings,
                        }

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
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
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        Yields dicts with:
          {"type": "status", "text": "Searching: ..."/"Reading: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        from core.inference.tools import execute_tool

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        conversation = list(messages)
        url = f"{self.base_url}/v1/chat/completions"
        _accumulated_completion_tokens = 0
        _accumulated_predicted_ms = 0.0
        _accumulated_predicted_n = 0

        def _strip_tool_markup(text: str, *, final: bool = False) -> str:
            if not auto_heal_tool_calls:
                return text
            patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
            for pat in patterns:
                text = pat.sub("", text)
            return text.strip() if final else text

        # XML prefixes that signal a tool call in content.
        # Empty when auto_heal is disabled so the buffer never
        # speculatively holds content for XML detection.
        _TOOL_XML_SIGNALS = (
            ("<tool_call>", "<function=") if auto_heal_tool_calls else ()
        )
        _MAX_BUFFER_CHARS = 32

        # ── Duplicate tool-call detection ────────────────────────
        # Track recent (tool_name, arguments) hashes to detect loops
        # where the model repeats the exact same call.  Retries after
        # a transient failure are allowed (only block when the previous
        # identical call succeeded).
        _tool_call_history: list[tuple[str, bool]] = []  # (key, failed)

        # ── Re-prompt on plan-without-action ─────────────────
        # When the model describes what it intends to do (forward-looking
        # language) without actually calling a tool, re-prompt once.
        # Only triggers on responses that signal intent/planning -- a
        # direct answer like "4" or "Hello!" will not match.
        # Pattern is compiled once at module level (_INTENT_SIGNAL).
        _reprompt_count = 0

        # Reserve extra iterations for re-prompts so they don't
        # consume the caller's tool-call budget.  Only add the
        # extra slot when tool iterations are actually allowed.
        _extra = _MAX_REPROMPTS if max_tool_iterations > 0 else 0
        for iteration in range(max_tool_iterations + _extra):
            if cancel_event is not None and cancel_event.is_set():
                return

            # Build payload -- stream: True so we detect tool signals
            # in the first 1-2 chunks without a non-streaming penalty.
            payload = {
                "messages": conversation,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "tools": tools,
                "tool_choice": "auto",
            }
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
            payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
            if stop:
                payload["stop"] = stop

            try:
                _auth_headers = (
                    {"Authorization": f"Bearer {self._api_key}"}
                    if self._api_key
                    else None
                )

                # ── Speculative buffer state machine ──────────────────
                # BUFFERING: accumulating content, checking for tool signals
                # STREAMING: no tool detected, yielding tokens to caller
                # DRAINING:  tool signal found, silently consuming rest
                _S_BUFFERING = 0
                _S_STREAMING = 1
                _S_DRAINING = 2

                detect_state = _S_BUFFERING
                content_buffer = ""  # Raw content held during BUFFERING
                content_accum = ""  # All content tokens (for tool parsing)
                reasoning_accum = ""
                cumulative_display = ""  # Cumulative text yielded (with <think>)
                in_thinking = False
                has_content_tokens = False
                tool_calls_acc = {}  # Structured delta.tool_calls fragments
                has_structured_tc = False
                _iter_usage = None
                _iter_timings = None
                _stream_done = False
                _last_emitted = ""

                stream_timeout = httpx.Timeout(
                    connect = 10,
                    read = 0.5,
                    write = 10,
                    pool = 10,
                )
                with httpx.Client(
                    timeout = stream_timeout,
                    limits = httpx.Limits(max_keepalive_connections = 0),
                ) as client:
                    with self._stream_with_retry(
                        client,
                        url,
                        payload,
                        cancel_event,
                        headers = _auth_headers,
                    ) as response:
                        if response.status_code != 200:
                            error_body = response.read().decode()
                            raise RuntimeError(
                                f"llama-server returned {response.status_code}: "
                                f"{error_body}"
                            )

                        raw_buf = ""
                        for raw_chunk in self._iter_text_cancellable(
                            response,
                            cancel_event,
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
                                            yield {
                                                "type": "content",
                                                "text": _strip_tool_markup(
                                                    cumulative_display,
                                                    final = True,
                                                ),
                                            }
                                        else:
                                            cumulative_display = reasoning_accum
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

                                    # ── Structured tool_calls ──
                                    tc_deltas = delta.get("tool_calls")
                                    if tc_deltas:
                                        # Once visible content has been
                                        # emitted, do not reclassify this
                                        # turn as a tool call.
                                        if _last_emitted:
                                            continue
                                        has_structured_tc = True
                                        detect_state = _S_DRAINING
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
                                                # Update ID if real one
                                                # arrives on a later delta
                                                tool_calls_acc[idx]["id"] = tc_d["id"]
                                            func = tc_d.get("function", {})
                                            if func.get("name"):
                                                tool_calls_acc[idx]["function"][
                                                    "name"
                                                ] += func["name"]
                                            if func.get("arguments"):
                                                tool_calls_acc[idx]["function"][
                                                    "arguments"
                                                ] += func["arguments"]
                                        continue

                                    # ── Reasoning tokens ──
                                    # Only yield in STREAMING state. In BUFFERING
                                    # and DRAINING, accumulate silently so we don't
                                    # corrupt the consumer's prev_text tracker
                                    # (routes/inference.py never resets prev_text
                                    # between tool iterations).
                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_accum += reasoning
                                        if detect_state == _S_STREAMING:
                                            if not in_thinking:
                                                cumulative_display += "<think>"
                                                in_thinking = True
                                            cumulative_display += reasoning
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }

                                    # ── Content tokens ──
                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        content_accum += token

                                        if detect_state == _S_DRAINING:
                                            pass  # accumulate silently

                                        elif detect_state == _S_STREAMING:
                                            if in_thinking:
                                                cumulative_display += "</think>"
                                                in_thinking = False
                                            cumulative_display += token
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                            )
                                            if len(cleaned) > len(_last_emitted):
                                                _last_emitted = cleaned
                                                yield {
                                                    "type": "content",
                                                    "text": cleaned,
                                                }

                                        elif detect_state == _S_BUFFERING:
                                            content_buffer += token
                                            stripped_buf = content_buffer.lstrip()
                                            if not stripped_buf:
                                                continue

                                            # Check tool signal prefixes
                                            is_prefix = False
                                            is_match = False
                                            for sig in _TOOL_XML_SIGNALS:
                                                if stripped_buf.startswith(sig):
                                                    is_match = True
                                                    break
                                                if sig.startswith(stripped_buf):
                                                    is_prefix = True
                                                    break

                                            if is_match:
                                                detect_state = _S_DRAINING
                                            elif (
                                                is_prefix
                                                and len(stripped_buf)
                                                < _MAX_BUFFER_CHARS
                                            ):
                                                pass  # keep buffering
                                            else:
                                                # Not a tool -- flush buffer
                                                detect_state = _S_STREAMING
                                                # Flush any reasoning accumulated
                                                # during BUFFERING phase
                                                if reasoning_accum:
                                                    cumulative_display += "<think>"
                                                    cumulative_display += (
                                                        reasoning_accum
                                                    )
                                                    cumulative_display += "</think>"
                                                cumulative_display += content_buffer
                                                cleaned = _strip_tool_markup(
                                                    cumulative_display,
                                                )
                                                if len(cleaned) > len(_last_emitted):
                                                    _last_emitted = cleaned
                                                    yield {
                                                        "type": "content",
                                                        "text": cleaned,
                                                    }

                                except json.JSONDecodeError:
                                    logger.debug(
                                        f"Skipping malformed SSE line: " f"{line[:100]}"
                                    )
                            if _stream_done:
                                break  # exit outer for

                # ── Resolve BUFFERING at stream end ──
                if detect_state == _S_BUFFERING:
                    stripped_buf = content_buffer.lstrip()
                    if (
                        stripped_buf
                        and auto_heal_tool_calls
                        and any(s in stripped_buf for s in _TOOL_XML_SIGNALS)
                    ):
                        detect_state = _S_DRAINING
                    elif content_accum or reasoning_accum:
                        detect_state = _S_STREAMING
                        if content_buffer:
                            # Flush any reasoning accumulated first
                            if reasoning_accum:
                                cumulative_display += "<think>"
                                cumulative_display += reasoning_accum
                                cumulative_display += "</think>"
                            cumulative_display += content_buffer
                            yield {
                                "type": "content",
                                "text": _strip_tool_markup(
                                    cumulative_display,
                                    final = True,
                                ),
                            }
                        elif reasoning_accum and not has_content_tokens:
                            # Reasoning-only response (no content tokens):
                            # show reasoning as plain text, matching
                            # the final streaming pass behavior for
                            # models that put everything in reasoning.
                            cumulative_display = reasoning_accum
                            yield {
                                "type": "content",
                                "text": cumulative_display,
                            }
                    else:
                        return

                # ── STREAMING path: no tool call ──
                if detect_state == _S_STREAMING:
                    # Safety net: check for XML tool signals in content.
                    # The route layer resets prev_text on tool_start, so
                    # post-tool synthesis streams correctly even if
                    # content was already emitted before the tool XML.
                    _safety_tc = None
                    if auto_heal_tool_calls and any(
                        s in content_accum for s in _TOOL_XML_SIGNALS
                    ):
                        _safety_tc = self._parse_tool_calls_from_text(
                            content_accum,
                        )
                    if not _safety_tc:
                        # ── Re-prompt on plan-without-action ──
                        # If the model described what it intends to do
                        # (forward-looking language) without calling any
                        # tool, nudge it to act.  Only fires once per
                        # request and only on short responses that
                        # contain intent signals -- a direct answer
                        # like "4" or "Hello!" won't trigger this.
                        # Use content if available, otherwise fall back
                        # to reasoning text (reasoning-only stalls).
                        _stripped = content_accum.strip()
                        if not _stripped:
                            _stripped = reasoning_accum.strip()
                        if (
                            tools
                            and _reprompt_count < _MAX_REPROMPTS
                            and 0 < len(_stripped) < _REPROMPT_MAX_CHARS
                            and _INTENT_SIGNAL.search(_stripped)
                        ):
                            _reprompt_count += 1
                            logger.info(
                                f"Re-prompt {_reprompt_count}/{_MAX_REPROMPTS}: "
                                f"model responded without calling tools "
                                f"({len(_stripped)} chars)"
                            )
                            conversation.append(
                                {
                                    "role": "assistant",
                                    "content": _stripped,
                                }
                            )
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "STOP. Do NOT write code or explain. "
                                        "You MUST call a tool NOW. "
                                        "Call web_search or python immediately."
                                    ),
                                }
                            )
                            # Accumulate tokens and timing from this iteration
                            _fu_r = _iter_usage or {}
                            _accumulated_completion_tokens += _fu_r.get(
                                "completion_tokens", 0
                            )
                            _it_r = _iter_timings or {}
                            _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                            yield {"type": "status", "text": ""}
                            continue

                        # Content was already streamed.  Yield metadata.
                        yield {"type": "status", "text": ""}
                        _fu = _iter_usage or {}
                        _fc = _fu.get("completion_tokens", 0)
                        _fp = _fu.get("prompt_tokens", 0)
                        _tc = _fc + _accumulated_completion_tokens
                        if (
                            _iter_usage
                            or _iter_timings
                            or _accumulated_completion_tokens
                        ):
                            _mt = dict(_iter_timings) if _iter_timings else {}
                            if _accumulated_predicted_ms or _accumulated_predicted_n:
                                _mt["predicted_ms"] = (
                                    _mt.get("predicted_ms", 0)
                                    + _accumulated_predicted_ms
                                )
                                _tn = (
                                    _mt.get("predicted_n", 0) + _accumulated_predicted_n
                                )
                                _mt["predicted_n"] = _tn
                                _tms = _mt["predicted_ms"]
                                if _tms > 0:
                                    _mt["predicted_per_second"] = _tn / (_tms / 1000.0)
                            yield {
                                "type": "metadata",
                                "usage": {
                                    "prompt_tokens": _fp,
                                    "completion_tokens": _tc,
                                    "total_tokens": _fp + _tc,
                                },
                                "timings": _mt,
                            }
                        return

                    # Safety net caught tool XML -- treat as tool call
                    tool_calls = _safety_tc
                    content_text = _strip_tool_markup(
                        content_accum,
                        final = True,
                    )
                    logger.info(
                        f"Safety net: parsed {len(tool_calls)} tool call(s) "
                        f"from streamed content"
                    )
                else:
                    # ── DRAINING path: assemble tool_calls ──
                    tool_calls = None
                    content_text = content_accum
                    if has_structured_tc:
                        # Filter out incomplete fragments (e.g. from
                        # truncation by max_tokens or disconnect).
                        tool_calls = [
                            tool_calls_acc[i]
                            for i in sorted(tool_calls_acc)
                            if (
                                tool_calls_acc[i]
                                .get("function", {})
                                .get("name", "")
                                .strip()
                            )
                        ] or None
                    if (
                        not tool_calls
                        and auto_heal_tool_calls
                        and any(s in content_accum for s in _TOOL_XML_SIGNALS)
                    ):
                        tool_calls = self._parse_tool_calls_from_text(
                            content_accum,
                        )
                    if tool_calls and not has_structured_tc:
                        content_text = _strip_tool_markup(
                            content_text,
                            final = True,
                        )
                    if tool_calls:
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s) from "
                            f"{'structured delta' if has_structured_tc else 'content text'}"
                        )
                    if not tool_calls:
                        # DRAINING but no tool calls (false positive).
                        # Merge accumulated metrics from prior tool
                        # iterations so they are not silently dropped.
                        yield {"type": "status", "text": ""}
                        if content_accum:
                            # Strip leaked tool-call XML before yielding
                            content_accum = _strip_tool_markup(
                                content_accum, final = True
                            )
                        if content_accum:
                            yield {"type": "content", "text": content_accum}
                        _fu = _iter_usage or {}
                        _fc = _fu.get("completion_tokens", 0)
                        _fp = _fu.get("prompt_tokens", 0)
                        _tc = _fc + _accumulated_completion_tokens
                        if (
                            _iter_usage
                            or _iter_timings
                            or _accumulated_completion_tokens
                        ):
                            _mt = dict(_iter_timings) if _iter_timings else {}
                            if _accumulated_predicted_ms or _accumulated_predicted_n:
                                _mt["predicted_ms"] = (
                                    _mt.get("predicted_ms", 0)
                                    + _accumulated_predicted_ms
                                )
                                _tn = (
                                    _mt.get("predicted_n", 0) + _accumulated_predicted_n
                                )
                                _mt["predicted_n"] = _tn
                                _tms = _mt["predicted_ms"]
                                if _tms > 0:
                                    _mt["predicted_per_second"] = _tn / (_tms / 1000.0)
                            yield {
                                "type": "metadata",
                                "usage": {
                                    "prompt_tokens": _fp,
                                    "completion_tokens": _tc,
                                    "total_tokens": _fp + _tc,
                                },
                                "timings": _mt,
                            }
                        return

                # ── Execute tool calls ──
                _accumulated_completion_tokens += (_iter_usage or {}).get(
                    "completion_tokens", 0
                )
                _it = _iter_timings or {}
                _accumulated_predicted_ms += _it.get("predicted_ms", 0)
                _accumulated_predicted_n += _it.get("predicted_n", 0)

                assistant_msg = {"role": "assistant", "content": content_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                conversation.append(assistant_msg)

                for tc in tool_calls or []:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    raw_args = func.get("arguments", {})

                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            if auto_heal_tool_calls:
                                arguments = {"query": raw_args}
                            else:
                                arguments = {"raw": raw_args}
                    else:
                        arguments = raw_args

                    if tool_name == "web_search":
                        _ws_url = (arguments.get("url") or "").strip()
                        if _ws_url:
                            _parsed = urlparse(_ws_url)
                            if _parsed.scheme in ("http", "https") and _parsed.hostname:
                                _ws_host = _parsed.hostname
                                if _ws_host.startswith("www."):
                                    _ws_host = _ws_host[4:]
                                status_text = f"Reading: {_ws_host}"
                            else:
                                status_text = "Reading page..."
                        else:
                            status_text = f"Searching: {arguments.get('query', '')}"
                    elif tool_name == "python":
                        preview = (
                            (arguments.get("code") or "").strip().split("\n")[0][:60]
                        )
                        status_text = (
                            f"Running Python: {preview}"
                            if preview
                            else "Running Python..."
                        )
                    elif tool_name == "terminal":
                        cmd_preview = (arguments.get("command") or "")[:60]
                        status_text = (
                            f"Running: {cmd_preview}"
                            if cmd_preview
                            else "Running command..."
                        )
                    else:
                        status_text = f"Calling: {tool_name}"
                    yield {"type": "status", "text": status_text}

                    yield {
                        "type": "tool_start",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "arguments": arguments,
                    }

                    # ── Duplicate call detection ──────────────
                    # str(dict) is stable here: arguments always comes from
                    # json.loads on the same model output within one request,
                    # so insertion order is deterministic (Python 3.7+).
                    _tc_key = tool_name + str(arguments)
                    _prev = _tool_call_history[-1] if _tool_call_history else None
                    if _prev and _prev[0] == _tc_key and not _prev[1]:
                        result = (
                            "You already made this exact call. "
                            "Do not repeat the same tool call. "
                            "Try a different approach: fetch a URL "
                            "from previous results, use Python to "
                            "process data you already have, or "
                            "provide your final answer now."
                        )
                    else:
                        _effective_timeout = (
                            None if tool_call_timeout >= 9999 else tool_call_timeout
                        )
                        result = execute_tool(
                            tool_name,
                            arguments,
                            cancel_event = cancel_event,
                            timeout = _effective_timeout,
                            session_id = session_id,
                        )

                    yield {
                        "type": "tool_end",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "result": result,
                    }

                    # Nudge model to try a different approach on errors
                    _error_prefixes = (
                        "Error",
                        "Search failed",
                        "Execution error",
                        "Blocked:",
                        "Exit code",
                        "Failed to fetch",
                        "Failed to resolve",
                        "No query provided",
                    )
                    _is_error = isinstance(result, str) and result.lstrip().startswith(
                        _error_prefixes
                    )
                    _tool_call_history.append((_tc_key, _is_error))
                    # Strip image sentinel before feeding result to the LLM
                    # (the full result with sentinel is still yielded via
                    # tool_end so the frontend can extract image paths).
                    _result_content = result
                    if "\n__IMAGES__:" in _result_content:
                        _result_content = _result_content.rsplit("\n__IMAGES__:", 1)[0]
                    if _is_error:
                        _result_content = (
                            _result_content + "\n\nThe tool call encountered an issue. "
                            "Please try a different approach or rephrase your request."
                        )

                    tool_msg = {
                        "role": "tool",
                        "name": tool_name,
                        "content": _result_content,
                    }
                    tool_call_id = tc.get("id")
                    if tool_call_id:
                        tool_msg["tool_call_id"] = tool_call_id
                    conversation.append(tool_msg)

                # Clear tool status badge before next generation iteration
                yield {"type": "status", "text": ""}
                # Continue the loop to let model respond with context
                continue

            except httpx.ConnectError:
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                raise

        # ── Tool iteration cap reached -- synthesize final answer ──
        # The model used all iterations without producing a final text
        # response. Inject a nudge so the final streaming pass produces
        # a useful answer instead of continuing to request tools.
        if max_tool_iterations > 0:
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

        # Clear status
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context
        stream_payload = {
            "messages": conversation,
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
        stream_payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
        if stop:
            stream_payload["stop"] = stop
        stream_payload["stream_options"] = {"include_usage": True}

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""
        _metadata_usage = None
        _metadata_timings = None
        _stream_done = False

        try:
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            _auth_headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            )
            with httpx.Client(
                timeout = stream_timeout, limits = httpx.Limits(max_keepalive_connections = 0)
            ) as client:
                with self._stream_with_retry(
                    client,
                    url,
                    stream_payload,
                    cancel_event,
                    headers = _auth_headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
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
                                        cumulative += "</think>"
                                        yield {
                                            "type": "content",
                                            "text": _strip_tool_markup(
                                                cumulative, final = True
                                            ),
                                        }
                                    else:
                                        cumulative = reasoning_text
                                        yield {"type": "content", "text": cumulative}
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                # Capture server timings/usage from final chunks
                                _chunk_timings = chunk_data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = chunk_data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield {"type": "content", "text": cumulative}

                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                        cumulative += token
                                        cleaned = _strip_tool_markup(cumulative)
                                        # Only emit when cleaned text grows (monotonic).
                                        if len(cleaned) > len(_last_emitted):
                                            _last_emitted = cleaned
                                            yield {"type": "content", "text": cleaned}
                            except json.JSONDecodeError:
                                logger.debug(
                                    f"Skipping malformed SSE line: {line[:100]}"
                                )
                        if _stream_done:
                            break  # exit outer for
                    _final_usage = _metadata_usage or {}
                    _final_completion = _final_usage.get("completion_tokens", 0)
                    _final_prompt = _final_usage.get("prompt_tokens", 0)
                    _total_completion = (
                        _final_completion + _accumulated_completion_tokens
                    )
                    if _metadata_usage or _metadata_timings:
                        _merged_timings = (
                            dict(_metadata_timings) if _metadata_timings else {}
                        )
                        if _accumulated_predicted_ms or _accumulated_predicted_n:
                            _merged_timings["predicted_ms"] = (
                                _merged_timings.get("predicted_ms", 0)
                                + _accumulated_predicted_ms
                            )
                            _total_predicted_n = (
                                _merged_timings.get("predicted_n", 0)
                                + _accumulated_predicted_n
                            )
                            _merged_timings["predicted_n"] = _total_predicted_n
                            _total_predicted_ms = _merged_timings["predicted_ms"]
                            if _total_predicted_ms > 0:
                                _merged_timings["predicted_per_second"] = (
                                    _total_predicted_n / (_total_predicted_ms / 1000.0)
                                )
                        yield {
                            "type": "metadata",
                            "usage": {
                                "prompt_tokens": _final_prompt,
                                "completion_tokens": _total_completion,
                                "total_tokens": _final_prompt + _total_completion,
                            },
                            "timings": _merged_timings,
                        }

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise

    # ── TTS support ────────────────────────────────────────────

    def detect_audio_type(self) -> Optional[str]:
        """Detect audio/TTS codec by probing the loaded model's vocabulary."""
        if not self.is_loaded:
            return None
        try:
            _auth_headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            )
            with httpx.Client(timeout = 10, headers = _auth_headers) as client:

                def _detok(tid: int) -> str:
                    r = client.post(
                        f"{self.base_url}/detokenize", json = {"tokens": [tid]}
                    )
                    return r.json().get("content", "") if r.status_code == 200 else ""

                def _tok(text: str) -> list[int]:
                    r = client.post(
                        f"{self.base_url}/tokenize",
                        json = {"content": text, "add_special": False},
                    )
                    return r.json().get("tokens", []) if r.status_code == 200 else []

                # Check codec-specific tokens (not generic ones that may exist in non-audio models)
                if "<custom_token_" in _detok(128258) and "<custom_token_" in _detok(
                    128259
                ):
                    return "snac"
                if len(_tok("<|AUDIO|>")) == 1 and len(_tok("<|audio_eos|>")) == 1:
                    return "csm"
                if len(_tok("<|startoftranscript|>")) == 1:
                    return "whisper"
                if (
                    len(_tok("<|bicodec_semantic_0|>")) == 1
                    and len(_tok("<|bicodec_global_0|>")) == 1
                ):
                    return "bicodec"
                if len(_tok("<|c1_0|>")) == 1 and len(_tok("<|c2_0|>")) == 1:
                    return "dac"
        except Exception as e:
            logger.debug(f"Audio type detection failed: {e}")
        return None

    # Prompt format per codec: (template, stop_tokens, needs_token_ids)
    # Matches prompts in InferenceBackend._generate_snac/bicodec/dac
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
        """Load the audio codec at model load time (mirrors non-GGUF path)."""
        import torch
        from core.inference.audio_codecs import AudioCodecManager

        if LlamaCppBackend._codec_mgr is None:
            LlamaCppBackend._codec_mgr = AudioCodecManager()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_repo_path = None

        # BiCodec needs a repo with BiCodec/ weights — download canonical SparkTTS
        if audio_type == "bicodec":
            from huggingface_hub import snapshot_download
            import os

            repo_path = snapshot_download(
                "unsloth/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B"
            )
            model_repo_path = os.path.abspath(repo_path)

        LlamaCppBackend._codec_mgr.load_codec(
            audio_type, device, model_repo_path = model_repo_path
        )
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
    ) -> tuple:
        """
        Generate TTS audio via llama-server /completion + codec decoding.
        Returns (wav_bytes, sample_rate).
        """
        if audio_type not in self._TTS_PROMPTS:
            raise RuntimeError(f"GGUF TTS does not support '{audio_type}' codec.")

        tpl, stop, need_ids = self._TTS_PROMPTS[audio_type]

        payload: dict = {
            "prompt": tpl.format(text = text),
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

        _auth_headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        )
        with httpx.Client(
            timeout = httpx.Timeout(300, connect = 10), headers = _auth_headers
        ) as client:
            resp = client.post(f"{self.base_url}/completion", json = payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"llama-server returned {resp.status_code}: {resp.text}"
                )

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
