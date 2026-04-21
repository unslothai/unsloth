# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the ``max_context_length`` warning-threshold semantics.

``/api/inference/status.max_context_length`` is what the ctx slider in
the chat settings sheet reads to decide when to render the "Exceeds
estimated VRAM capacity. The model may use system RAM." warning:

    ctxDisplayValue > ggufMaxContextLength → show warning

For models whose weights fit on some GPU subset, the warning threshold
is the largest ctx that fits fully in VRAM (the binary-search cap from
``_fit_context_to_vram``). For models whose weights exceed 90% of every
GPU subset's free memory, the warning must fire as soon as the user
drags above the 4096 spec default (otherwise a user loading e.g.
MiniMax-M2.7 on a 97 GB GPU sees a slider up to 196608 with no
indication that any value above 4096 will trigger ``--fit on`` and
degrade performance).

These tests pin both cases. No GPU probing, no subprocess, no GGUF I/O.
Cross-platform: Linux, macOS, Windows, WSL.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test.  Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GIB = 1024**3


def _make_backend(native_ctx = 131072):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._context_length = native_ctx
    inst._n_layers = 80
    inst._n_kv_heads = 8
    inst._n_heads = 64
    inst._embedding_length = 8192
    inst._kv_key_length = 128
    inst._kv_value_length = 128
    inst._kv_lora_rank = None
    inst._sliding_window = None
    inst._ssm_inner_size = None
    inst._full_attention_interval = None
    inst._key_length_mla = None
    return inst


def _compute_max_available_ctx(native_ctx, model_gib, gpus, kv_per_token_bytes = 325_000):
    """Run the ceiling-probe block from load_model and return the final
    ``max_available_ctx`` value the backend would assign to
    ``_max_context_length``.
    """
    inst = _make_backend(native_ctx = native_ctx)
    model_size = int(model_gib * GIB)

    inst._estimate_kv_cache_bytes = (
        lambda n, _t = None: 0 if n <= 0 else n * kv_per_token_bytes
    )
    inst._can_estimate_kv = lambda: True

    context_length = inst._context_length
    effective_ctx = context_length
    max_available_ctx = context_length

    cache_type_kv = None
    native_ctx_for_cap = context_length

    ranked_for_cap = sorted(gpus, key = lambda g: g[1], reverse = True)
    best_cap = 0
    for n_gpus in range(1, len(ranked_for_cap) + 1):
        subset = ranked_for_cap[:n_gpus]
        pool_mib = sum(free for _, free in subset)
        capped = inst._fit_context_to_vram(
            native_ctx_for_cap,
            pool_mib,
            model_size,
            cache_type_kv,
        )
        kv = inst._estimate_kv_cache_bytes(capped, cache_type_kv)
        total_mib = (model_size + kv) / (1024 * 1024)
        if total_mib <= pool_mib * 0.90:
            best_cap = max(best_cap, capped)
    if best_cap > 0:
        max_available_ctx = best_cap
    else:
        max_available_ctx = min(4096, native_ctx_for_cap)

    return max_available_ctx


# ---------------------------------------------------------------------------
# Weights exceed every GPU subset's VRAM  (MiniMax-M2.7-like)
# ---------------------------------------------------------------------------


class TestMaxContextLengthForWeightsExceedVRAM:
    """The UI ``max_context_length`` threshold must fall back to 4096 so
    the warning fires as soon as the user drags above the spec default.
    """

    def test_minimax_like(self):
        """131 GB weights, single 97 GB GPU, native ctx 196608."""
        got = _compute_max_available_ctx(
            native_ctx = 196608,
            model_gib = 131,
            gpus = [(0, 97_000)],
        )
        assert got == 4096

    def test_multi_gpu_all_subsets_fail(self):
        """400 GB weights across a 4x80 GB pool (320 GB total, still too small)."""
        got = _compute_max_available_ctx(
            native_ctx = 131072,
            model_gib = 400,
            gpus = [(0, 80_000), (1, 80_000), (2, 80_000), (3, 80_000)],
        )
        assert got == 4096

    def test_native_below_fallback_is_preserved(self):
        """If the model's native ctx is itself smaller than 4096, do not
        advertise a larger value than the model supports."""
        got = _compute_max_available_ctx(
            native_ctx = 2048,
            model_gib = 200,
            gpus = [(0, 80_000)],
        )
        assert got == 2048


# ---------------------------------------------------------------------------
# Fittable models (regression guard)
# ---------------------------------------------------------------------------


class TestMaxContextLengthForFittableModels:
    """The existing best-cap behaviour must be unchanged."""

    def test_small_model_fits_easily(self):
        """8 GB model on 24 GB GPU: should auto-pick a large ctx."""
        got = _compute_max_available_ctx(
            native_ctx = 131072,
            model_gib = 8,
            gpus = [(0, 24_000)],
            kv_per_token_bytes = 8192,
        )
        assert got > 4096
        assert got <= 131072

    def test_medium_model_multi_gpu(self):
        """60 GB model split across 2 GPUs: picks a fitting ctx."""
        got = _compute_max_available_ctx(
            native_ctx = 131072,
            model_gib = 60,
            gpus = [(0, 40_000), (1, 40_000)],
            kv_per_token_bytes = 8192,
        )
        assert got > 4096

    def test_tiny_model_on_huge_gpu_near_native(self):
        """2 GB model, 80 GB GPU, negligible KV: should approach native."""
        got = _compute_max_available_ctx(
            native_ctx = 131072,
            model_gib = 2,
            gpus = [(0, 80_000)],
            kv_per_token_bytes = 64,
        )
        assert got >= 131072 - 256  # rounded to 256 boundary


# ---------------------------------------------------------------------------
# Property plumbing
# ---------------------------------------------------------------------------


class TestMaxContextLengthProperty:
    def test_falls_back_to_native_when_unset(self):
        inst = _make_backend(native_ctx = 131072)
        inst._max_context_length = None
        assert inst.max_context_length == 131072

    def test_returns_stored_value_when_set(self):
        inst = _make_backend(native_ctx = 131072)
        inst._max_context_length = 4096
        assert inst.max_context_length == 4096
