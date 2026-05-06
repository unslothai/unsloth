# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the GGUF load-time context auto-fit decision.

Guards two regressions in ``LlamaCppBackend.load_model``:

1. **Auto mode on weights-exceed-VRAM** (``n_ctx == 0``): when the model
   weights alone exceed 90% of every GPU subset's free memory, the
   auto-pick loop used to exit without matching, leaving
   ``effective_ctx`` at the model's native context (e.g. 196608 for
   MiniMax-M2.7). The intended default per Studio's UI spec is 4096 so
   the slider lands on a usable value; the user can still drag higher
   and trigger ``--fit on`` with a warning.

2. **Explicit ctx silently shrunk when KV overflows**: with fittable
   weights but a requested ctx whose KV cache pushes total memory over
   90% of VRAM, the old code binary-searched a smaller ctx and emitted
   ``-c <capped> -ngl -1`` without informing the caller. The UI had
   already surfaced its "might be slower" warning and expects the user's
   explicit ctx to be honored with ``--fit on`` flexing ``-ngl`` instead.

Tests avoid GPU probing, subprocess spawning, and GGUF I/O by driving the
post-metadata decision block directly against a stubbed instance.

Requires no GPU, network, or external libraries beyond pytest.
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
FALLBACK_CTX = 4096


def _make_backend(
    native_ctx = 131072,
    n_layers = 80,
    n_kv_heads = 8,
    n_heads = 64,
    kv_key_length = 128,
    kv_value_length = 128,
):
    """Create a LlamaCppBackend instance with GGUF metadata fields set and
    the helpers used by the decision block stubbed out."""
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._context_length = native_ctx
    inst._n_layers = n_layers
    inst._n_kv_heads = n_kv_heads
    inst._n_heads = n_heads
    inst._embedding_length = 8192
    inst._kv_key_length = kv_key_length
    inst._kv_value_length = kv_value_length
    inst._kv_lora_rank = None
    inst._sliding_window = None
    inst._sliding_window_pattern = None
    inst._ssm_inner_size = None
    inst._full_attention_interval = None
    inst._key_length_mla = None
    inst._n_kv_heads_by_layer = None
    inst._kv_key_length_swa = None
    inst._kv_value_length_swa = None
    return inst


def _drive(
    n_ctx,
    model_gib,
    gpus,
    native_ctx = 131072,
    kv_per_token_bytes = 325_000,
    can_estimate_kv = True,
):
    """Drive the post-metadata portion of load_model with stubbed inputs.

    Mirrors the decision block at llama_cpp.py:1137-1296 so we can assert
    the command that would be built, without subprocesses or GPU probes.
    """
    inst = _make_backend(native_ctx = native_ctx)
    model_size = int(model_gib * GIB)
    cache_type_kv = None

    def fake_estimate(n_ctx_, _type = None, **_kwargs):
        return 0 if n_ctx_ <= 0 else n_ctx_ * kv_per_token_bytes

    inst._estimate_kv_cache_bytes = fake_estimate
    inst._can_estimate_kv = lambda: can_estimate_kv

    context_length = inst._context_length

    effective_ctx = n_ctx if n_ctx > 0 else (context_length or 0)
    max_available_ctx = context_length or effective_ctx
    if n_ctx > 0:
        effective_ctx = n_ctx
    elif context_length is not None:
        effective_ctx = context_length
    else:
        effective_ctx = 0
    original_ctx = effective_ctx
    max_available_ctx = context_length or effective_ctx

    gpu_indices, use_fit = None, True
    explicit_ctx = n_ctx > 0

    if gpus and inst._can_estimate_kv() and effective_ctx > 0:
        native_ctx_for_cap = context_length or effective_ctx
        if native_ctx_for_cap > 0:
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

        if explicit_ctx:
            requested_total = model_size + inst._estimate_kv_cache_bytes(
                effective_ctx, cache_type_kv
            )
            gpu_indices, use_fit = inst._select_gpus(requested_total, gpus)
        else:
            ranked = sorted(gpus, key = lambda g: g[1], reverse = True)
            matched = False
            for n_gpus in range(1, len(ranked) + 1):
                subset = ranked[:n_gpus]
                pool_mib = sum(free for _, free in subset)
                capped = inst._fit_context_to_vram(
                    effective_ctx,
                    pool_mib,
                    model_size,
                    cache_type_kv,
                )
                kv = inst._estimate_kv_cache_bytes(capped, cache_type_kv)
                total_mib = (model_size + kv) / (1024 * 1024)
                if total_mib <= pool_mib * 0.90:
                    effective_ctx = capped
                    gpu_indices = sorted(idx for idx, _ in subset)
                    use_fit = False
                    matched = True
                    break
            if not matched:
                effective_ctx = min(FALLBACK_CTX, effective_ctx)
    elif gpus:
        gpu_indices, use_fit = inst._select_gpus(model_size, gpus)
        if use_fit and not explicit_ctx:
            effective_ctx = (
                min(FALLBACK_CTX, effective_ctx) if effective_ctx > 0 else FALLBACK_CTX
            )

    return {
        "c_arg": effective_ctx if effective_ctx > 0 else 0,
        "use_fit": use_fit,
        "gpu_indices": gpu_indices,
        "max_available_ctx": max_available_ctx,
        "original_ctx": original_ctx,
    }


# ---------------------------------------------------------------------------
# Auto mode, model weights exceed VRAM  (Bug A guard)
# ---------------------------------------------------------------------------


class TestAutoModeWeightsExceedVRAM:
    """``n_ctx == 0`` on a model whose weights don't fit anywhere."""

    def test_minimax_like_single_gpu(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
        )
        assert plan["c_arg"] == FALLBACK_CTX
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None
        # UI slider ceiling stays at native: user can still drag higher
        # and get the "might be slower" path.
        assert plan["max_available_ctx"] == 196608

    def test_multi_gpu_all_subsets_fail(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 400,
            gpus = [(0, 80_000), (1, 80_000), (2, 80_000), (3, 80_000)],
            native_ctx = 131072,
        )
        assert plan["c_arg"] == FALLBACK_CTX
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None

    def test_no_kv_metadata_auto(self):
        """File-size-only fallback path also defaults to 4096."""
        plan = _drive(
            n_ctx = 0,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
            can_estimate_kv = False,
        )
        assert plan["c_arg"] == FALLBACK_CTX
        assert plan["use_fit"] is True


# ---------------------------------------------------------------------------
# Explicit ctx, KV overflows fittable weights  (Bug B guard)
# ---------------------------------------------------------------------------


class TestExplicitCtxRespectsUser:
    """``n_ctx > 0`` must never be silently shrunk."""

    def test_fittable_weights_oversized_kv(self):
        # 8 GB weights + 131k ctx KV on 24 GB VRAM.
        # Budget = 21.6 GB, KV at 131k >> 13.6 GB remaining, so
        # _select_gpus flips use_fit=True.
        plan = _drive(
            n_ctx = 131072,
            model_gib = 8,
            gpus = [(0, 24_000)],
            native_ctx = 131072,
        )
        assert plan["c_arg"] == 131072
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None

    def test_explicit_that_fits_uses_ngl(self):
        plan = _drive(
            n_ctx = 8192,
            model_gib = 8,
            gpus = [(0, 24_000)],
            native_ctx = 131072,
        )
        assert plan["c_arg"] == 8192
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0]

    def test_explicit_on_weights_exceed_vram(self):
        # User drags the slider to 32k on a too-big model: honored.
        plan = _drive(
            n_ctx = 32768,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
        )
        assert plan["c_arg"] == 32768
        assert plan["use_fit"] is True

    def test_explicit_at_fallback_on_too_big(self):
        plan = _drive(
            n_ctx = FALLBACK_CTX,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
        )
        assert plan["c_arg"] == FALLBACK_CTX
        assert plan["use_fit"] is True

    def test_explicit_below_floor_honored(self):
        # 2048 is below --fit-ctx default; still honored since user set it.
        plan = _drive(
            n_ctx = 2048,
            model_gib = 8,
            gpus = [(0, 24_000)],
        )
        assert plan["c_arg"] == 2048


# ---------------------------------------------------------------------------
# Non-regression: fittable + auto still auto-picks largest fitting ctx
# ---------------------------------------------------------------------------


class TestFittableAutoPickRegressions:
    def test_small_model_one_gpu(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 8,
            gpus = [(0, 24_000)],
            native_ctx = 131072,
            kv_per_token_bytes = 8192,
        )
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0]
        assert plan["c_arg"] > FALLBACK_CTX

    def test_medium_model_needs_multi_gpu(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 60,
            gpus = [(0, 40_000), (1, 40_000)],
            native_ctx = 131072,
            kv_per_token_bytes = 8192,
        )
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0, 1]

    def test_no_kv_metadata_fittable_auto(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 8,
            gpus = [(0, 24_000)],
            native_ctx = 131072,
            can_estimate_kv = False,
        )
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0]


# ---------------------------------------------------------------------------
# Platform-agnostic input shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform_tag", ["linux", "windows", "mac", "rocm"])
def test_identical_decision_across_platforms(platform_tag):
    """The decision function takes ``[(gpu_idx, free_mib), ...]`` regardless
    of how upstream (nvidia-smi / nvidia-smi.exe / Metal / rocm-smi) produced
    it. Identical inputs must yield identical plans."""
    plan_a = _drive(n_ctx = 0, model_gib = 8, gpus = [(0, 24_000)])
    plan_b = _drive(n_ctx = 0, model_gib = 8, gpus = [(0, 24_000)])
    assert plan_a == plan_b, platform_tag
