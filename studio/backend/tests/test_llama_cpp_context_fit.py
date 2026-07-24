# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the GGUF load-time context auto-fit decision.

Guards two regressions in ``LlamaCppBackend.load_model``:

1. Auto mode (``n_ctx == 0``) when weights exceed every GPU subset's free
   memory: auto-pick should fall back to 4096 (a usable slider value) rather
   than leaving native ctx. User can still drag higher onto ``--fit on``.
2. Explicit ctx must never be silently shrunk: when KV overflows fittable
   weights, honor the explicit ctx with ``--fit on`` flexing ``-ngl``.

Drives the post-metadata decision block against a stubbed instance: no GPU,
network, subprocess, or GGUF I/O. Cross-platform.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy/unavailable deps before importing the module under test.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

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

from core.inference.llama_cpp import (
    _APPLE_UNIFIED_MEMORY_FRACTION,
    _CTX_FIT_VRAM_FRACTION,
    LlamaCppBackend,
    classify_gpu_offload_lines,
)
from core.inference.llama_server_args import parse_ctx_override, resolve_requested_ctx


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
    """LlamaCppBackend with GGUF metadata set and decision helpers stubbed."""
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
    extra_args = None,
    apple_budget_mib = 0,
    flat_mtp_reserve = 0.0,
):
    """Drive the post-metadata portion of load_model with stubbed inputs.

    Mirrors llama_cpp.py:1137-1296 to assert the built command, without
    subprocesses or GPU probes.
    """
    inst = _make_backend(native_ctx = native_ctx)
    model_size = int(model_gib * GIB)
    cache_type_kv = None

    def fake_estimate(
        n_ctx_,
        _type = None,
        **_kwargs,
    ):
        return 0 if n_ctx_ <= 0 else n_ctx_ * kv_per_token_bytes

    inst._estimate_kv_cache_bytes = fake_estimate
    inst._can_estimate_kv = lambda: can_estimate_kv

    context_length = inst._context_length
    # Use the production helper, not a reimplementation, to avoid testing our own logic.
    ctx_override = parse_ctx_override(extra_args)
    requested_ctx = resolve_requested_ctx(extra_args, n_ctx)

    effective_ctx = requested_ctx if requested_ctx > 0 else (context_length or 0)
    max_available_ctx = context_length or effective_ctx
    if requested_ctx > 0:
        effective_ctx = requested_ctx
    elif context_length is not None:
        effective_ctx = context_length
    else:
        effective_ctx = 0
    original_ctx = effective_ctx
    max_available_ctx = context_length or effective_ctx

    gpu_indices, use_fit = None, True
    explicit_ctx = requested_ctx > 0

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
                if total_mib <= pool_mib * _CTX_FIT_VRAM_FRACTION:
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
            pin_fraction = LlamaCppBackend._GPU_PIN_VRAM_FRACTION
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
                if total_mib <= pool_mib * pin_fraction:
                    effective_ctx = capped
                    gpu_indices = sorted(idx for idx, _ in subset)
                    use_fit = False
                    matched = True
                    break
            if not matched:
                effective_ctx = min(FALLBACK_CTX, effective_ctx)
                # Mirror llama_cpp.py: re-check fit at FALLBACK_CTX.
                if effective_ctx > 0:
                    for n_gpus in range(1, len(ranked) + 1):
                        subset = ranked[:n_gpus]
                        pool_mib = sum(free for _, free in subset)
                        kv = inst._estimate_kv_cache_bytes(effective_ctx, cache_type_kv)
                        total_mib = (model_size + kv) / (1024 * 1024)
                        if total_mib <= pool_mib * pin_fraction:
                            gpu_indices = sorted(idx for idx, _ in subset)
                            use_fit = False
                            break
    elif gpus:
        gpu_indices, use_fit = inst._select_gpus(model_size, gpus)
        if use_fit and not explicit_ctx:
            effective_ctx = min(FALLBACK_CTX, effective_ctx) if effective_ctx > 0 else FALLBACK_CTX
    elif apple_budget_mib > 0 and effective_ctx > 0:
        # Mirrors the Apple unified-memory branch in load_model: flat MTP reserve
        # off the budget up front (no-op at 0), sparse-KV floors to FALLBACK_CTX,
        # only auto context shrinks.
        native_ctx_for_cap = context_length or effective_ctx
        apple_fit_budget_mib = int(apple_budget_mib * max(0.0, 1.0 - flat_mtp_reserve))
        if inst._can_estimate_kv():
            cap = inst._fit_context_to_vram(
                native_ctx_for_cap,
                apple_fit_budget_mib,
                model_size,
                cache_type_kv,
                budget_frac = 1.0,
            )
            cap_footprint_mib = (model_size + inst._estimate_kv_cache_bytes(cap, cache_type_kv)) / (
                1024 * 1024
            )
            max_available_ctx = (
                cap
                if cap_footprint_mib <= apple_fit_budget_mib
                else min(FALLBACK_CTX, native_ctx_for_cap)
            )
        else:
            max_available_ctx = min(FALLBACK_CTX, native_ctx_for_cap)
        if not explicit_ctx:
            effective_ctx = max_available_ctx

    return {
        "c_arg": effective_ctx if effective_ctx > 0 else 0,
        "use_fit": use_fit,
        "gpu_indices": gpu_indices,
        "max_available_ctx": max_available_ctx,
        "original_ctx": original_ctx,
        "ctx_override": ctx_override,
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
        # UI slider ceiling stays at native: user can drag higher and get
        # the "might be slower" path.
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
        # 8 GB weights + 131k ctx KV on 24 GB VRAM. Budget = 21.6 GB, KV
        # at 131k >> 13.6 GB remaining, so _select_gpus flips use_fit=True.
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
        # 2048 is below --fit-ctx default; honored since user set it.
        plan = _drive(
            n_ctx = 2048,
            model_gib = 8,
            gpus = [(0, 24_000)],
        )
        assert plan["c_arg"] == 2048


# ---------------------------------------------------------------------------
# Pass-through --ctx-size participates in context fit (#5676).
# ---------------------------------------------------------------------------


class TestExtraArgsCtxOverride:
    def test_ctx_size_extra_honored_over_auto(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
            extra_args = ["--ctx-size", "128000"],
        )
        assert plan["ctx_override"] == 128000
        assert plan["original_ctx"] == 128000
        assert plan["c_arg"] == 128000
        assert plan["use_fit"] is True

    def test_ctx_size_short_alias_honored_over_auto(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 131,
            gpus = [(0, 97_000)],
            native_ctx = 196608,
            extra_args = ["-c", "128000"],
        )
        assert plan["c_arg"] == 128000
        assert plan["use_fit"] is True

    def test_ctx_size_extra_wins_over_first_class_field(self):
        plan = _drive(
            n_ctx = 4096,
            model_gib = 8,
            gpus = [(0, 24_000)],
            native_ctx = 131072,
            extra_args = ["--ctx-size", "128000"],
        )
        assert plan["original_ctx"] == 128000
        assert plan["c_arg"] == 128000


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
# #5106 regression: 91-95% utilization must still pin GPU.
# ---------------------------------------------------------------------------


class TestTightFitPinsToGPU:
    """Models that fit at 91-95% of free VRAM must use the GPU."""

    def test_rtx_4090_qwen_24gb_class(self):
        # noahterbest's #5106 log: 20.8 GB model on 22805 MiB free GPU,
        # ctx=4096 -> ~94% utilization, ~1.4 GiB headroom.
        plan = _drive(
            n_ctx = 0,
            model_gib = 20.8,
            gpus = [(0, 22_805)],
            native_ctx = 131072,
            kv_per_token_bytes = 25_000,
        )
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0]

    def test_explicit_ctx_at_94_pct_pins_to_gpu(self):
        # Explicit-ctx branch must agree with auto-ctx on headroom.
        plan = _drive(
            n_ctx = 4096,
            model_gib = 20.8,
            gpus = [(0, 22_805)],
            native_ctx = 131072,
            kv_per_token_bytes = 25_000,
        )
        assert plan["use_fit"] is False
        assert plan["gpu_indices"] == [0]

    def test_genuine_overflow_still_uses_fit(self):
        # Beyond 95% must still defer to --fit on.
        plan = _drive(
            n_ctx = 4096,
            model_gib = 23,
            gpus = [(0, 22_000)],
            native_ctx = 131072,
            kv_per_token_bytes = 25_000,
        )
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None


# ---------------------------------------------------------------------------
# Platform-agnostic input shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform_tag", ["linux", "windows", "mac", "rocm"])
def test_identical_decision_across_platforms(platform_tag):
    """Decision takes ``[(gpu_idx, free_mib), ...]`` regardless of source;
    identical inputs must yield identical plans."""
    plan_a = _drive(n_ctx = 0, model_gib = 8, gpus = [(0, 24_000)])
    plan_b = _drive(n_ctx = 0, model_gib = 8, gpus = [(0, 24_000)])
    assert plan_a == plan_b, platform_tag


# ---------------------------------------------------------------------------
# _classify_gpu_offload: detect silent CPU fallback (#5106).
# ---------------------------------------------------------------------------


class TestClassifyGpuOffload:
    def _backend(self, stdout_lines):
        inst = LlamaCppBackend.__new__(LlamaCppBackend)
        inst._stdout_lines = list(stdout_lines)
        return inst

    def test_cuda_buffer_present_returns_true(self):
        inst = self._backend(
            [
                "load_tensors: offloaded 33/33 layers to GPU",
                "load_tensors:        CUDA0 model buffer size = 21000.0 MiB",
                "load_tensors:   CPU_Mapped model buffer size =     0.6 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is True

    def test_cpu_only_buffer_returns_false(self):
        # Buffer lines printed but only CPU buffers -- the silent CPU
        # fallback symptom we want to catch.
        inst = self._backend(
            [
                "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB",
                "load_tensors:          CPU model buffer size =     0.6 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_no_buffer_lines_returns_none(self):
        # If we can't see buffer-allocation lines at all, don't guess.
        inst = self._backend(
            [
                "INFO [main] starting server",
                "load_tensors: file format = GGUF V3",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is None

    def test_no_gpus_detected_returns_none(self):
        # CPU-only systems are valid; suppress the warning entirely.
        inst = self._backend(
            [
                "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(False, []) is None

    def test_user_did_not_intend_gpu_returns_none(self):
        # Unsloth called start_llama_server without expecting GPU; don't warn.
        inst = self._backend(
            [
                "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(False, [(0, 22805)]) is None

    def test_rocm_buffer_marker_returns_true(self):
        inst = self._backend(
            [
                "load_tensors:        ROCm0 model buffer size = 21000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is True

    def test_metal_buffer_marker_returns_true(self):
        inst = self._backend(
            [
                "load_tensors:       Metal model buffer size = 8000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is True

    def test_offloaded_zero_count_returns_false(self):
        # Authoritative count overrides any GPU-looking buffer line.
        inst = self._backend(
            [
                "load_tensors: offloaded 0/33 layers to GPU",
                "load_tensors:        CUDA0 model buffer size = 21000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_offloaded_draft_then_main_returns_true(self):
        # A small draft model (0/2) does not mask the main model (33/33).
        inst = self._backend(
            [
                "load_tensors: offloaded 0/2 layers to GPU",
                "load_tensors: offloaded 33/33 layers to GPU",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is True

    def test_main_on_cpu_with_draft_on_gpu_returns_false(self):
        # MTP: the small drafter fits on GPU (1/1) but the main model is on CPU
        # (0/33). Decide on the largest model, so the warning still fires.
        inst = self._backend(
            [
                "load_tensors: offloaded 0/33 layers to GPU",
                "load_tensors: offloaded 1/1 layers to GPU",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_main_on_gpu_with_draft_on_cpu_returns_true(self):
        # Reverse: main model on GPU (33/33), drafter on CPU (0/1) -> no warning.
        inst = self._backend(
            [
                "load_tensors: offloaded 33/33 layers to GPU",
                "load_tensors: offloaded 0/1 layers to GPU",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is True

    def test_cuda_host_buffer_excluded_returns_false(self):
        # CUDA_Host is CPU-pinned memory, not a model offload.
        inst = self._backend(
            [
                "load_tensors:    CUDA_Host model buffer size =   500.0 MiB",
                "load_tensors:          CPU model buffer size = 21000.0 MiB",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_device_info_gpu_row_alone_is_inconclusive(self):
        # device_info lists available devices, not where the model loaded, so a
        # GPU row alone is not proof of offload.
        inst = self._backend(
            [
                "print_info: device_info:",
                "  - CUDA0 : 24564 MiB free",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is None

    def test_cpu_buffers_with_gpu_device_row_returns_false(self):
        # Definite CPU-only buffers must win over a GPU device-inventory row.
        inst = self._backend(
            [
                "load_tensors:          CPU model buffer size = 21000.0 MiB",
                "print_info: device_info:",
                "  - CUDA0 : 24564 MiB free",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_device_info_cpu_only_returns_false(self):
        inst = self._backend(
            [
                "print_info: device_info:",
                "  - CPU : 64000 MiB free",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    def test_system_info_cuda_before_device_info_does_not_count(self):
        # A compiled-in backend named in system_info is not proof of offload;
        # only the device_info table (here CPU only) decides.
        inst = self._backend(
            [
                "system_info: CUDA : ARCHS = 890 | n_threads = 8",
                "print_info: device_info:",
                "  - CPU : 64000 MiB free",
            ]
        )
        assert inst._classify_gpu_offload(True, [(0, 22805)]) is False

    @pytest.mark.parametrize(
        "marker",
        ["CUDA0", "ROCm0", "HIP0", "Metal", "Vulkan0", "OpenCL0", "SYCL0", "MUSA0", "CANN0"],
    )
    def test_all_gpu_buffer_markers_return_true(self, marker):
        assert (
            classify_gpu_offload_lines([f"load_tensors: {marker} model buffer size = 8000.0 MiB"])
            is True
        )

    def test_module_level_no_signal_returns_none(self):
        assert classify_gpu_offload_lines(["INFO starting server"]) is None


def test_select_gpus_ranks_by_usable_not_raw_free():
    # 80 GB card (30 GB free -> 25.9 GB usable) vs 32 GB card (29 GB free -> 27.4
    # GB usable). A 27 GB model fits the 32 GB card alone; raw-free ranking would
    # try the 80 GB card first and split across both. Usable ranking picks [1].
    gpus = [(0, 30000), (1, 29000)]
    totals = {0: 81920, 1: 32607}
    model = int(27000 * 1024 * 1024)
    idxs, use_fit = LlamaCppBackend._select_gpus(model, gpus, total_by_idx = totals)
    assert idxs == [1] and use_fit is False


def test_select_gpus_reserves_per_device_overhead():
    # Two 16 GB cards, ~15181 MiB usable each at 0.95 -> 30362 MiB pooled. A 30000
    # MiB model fits the pool with no per-device overhead, but a layer split also
    # pays ~1 GiB/extra-GPU; that pushes the 2-GPU need to 31024 MiB > pool, so a
    # pin would OOM -> must fall back to --fit. Single-GPU fits add no overhead
    # (Finding F1, the explicit/file-size multi-GPU pin gap).
    gpus = [(0, 16000), (1, 16000)]
    totals = {0: 16384, 1: 16384}
    gib = 1024 * 1024 * 1024
    model = int(30000 * 1024 * 1024)
    idxs, use_fit = LlamaCppBackend._select_gpus(model, gpus, total_by_idx = totals)
    assert idxs == [0, 1] and use_fit is False  # fits 2 GPUs without overhead
    idxs2, use_fit2 = LlamaCppBackend._select_gpus(
        model, gpus, total_by_idx = totals, per_device_overhead_bytes = gib
    )
    assert idxs2 is None and use_fit2 is True  # overhead tips it past the pool
    # A single-GPU fit is unchanged by the overhead (k=1 adds nothing).
    small = int(15000 * 1024 * 1024)
    a, _ = LlamaCppBackend._select_gpus(small, gpus, total_by_idx = totals)
    b, _ = LlamaCppBackend._select_gpus(
        small, gpus, total_by_idx = totals, per_device_overhead_bytes = gib
    )
    assert a == [0] and b == [0]


# ---------------------------------------------------------------------------
# Apple Silicon unified-memory context cap (#5118, #6529): no discrete GPU on
# Metal, so the auto context defaulted to native and over-committed unified
# memory. The fix budgets and caps the auto context (explicit stays verbatim).
# ---------------------------------------------------------------------------


def _force_apple(monkeypatch):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_platform, "machine", lambda: "arm64")


def _install_fake_mlx(monkeypatch, working_set_bytes):
    """Minimal mlx.core stub exposing metal.is_available() and device_info()."""
    mlx = _types.ModuleType("mlx")
    mlx_core = _types.ModuleType("mlx.core")
    mlx_core.metal = _types.SimpleNamespace(is_available = lambda: True)
    mlx_core.device_info = lambda: {"max_recommended_working_set_size": working_set_bytes}
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)


class TestAppleUnifiedMemoryBudget:
    def test_zero_off_apple_silicon(self, monkeypatch):
        import platform as _platform

        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        assert LlamaCppBackend._apple_metal_memory_budget_bytes() == 0

    def test_uses_metal_working_set(self, monkeypatch):
        _force_apple(monkeypatch)
        ws = 27 * GIB  # ~recommended working set on a 36 GB Mac
        _install_fake_mlx(monkeypatch, ws)
        assert LlamaCppBackend._apple_metal_memory_budget_bytes() == int(
            ws * _APPLE_UNIFIED_MEMORY_FRACTION
        )

    def test_falls_back_to_total_ram_without_mlx(self, monkeypatch):
        _force_apple(monkeypatch)
        monkeypatch.setitem(sys.modules, "mlx", None)  # import mlx.core -> ImportError
        fake_psutil = _types.ModuleType("psutil")
        fake_psutil.virtual_memory = lambda: _types.SimpleNamespace(total = 36 * GIB)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert LlamaCppBackend._apple_metal_memory_budget_bytes() == int(
            36 * GIB * _APPLE_UNIFIED_MEMORY_FRACTION
        )

    def test_zero_when_no_budget_resolvable(self, monkeypatch):
        _force_apple(monkeypatch)
        monkeypatch.setitem(sys.modules, "mlx", None)
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert LlamaCppBackend._apple_metal_memory_budget_bytes() == 0


class TestAppleContextCap:
    """The real ``_fit_context_to_vram`` against the reporter's M3 Pro case."""

    def test_caps_native_context_into_unified_budget(self):
        # ~15.7 GB weights at native 262144 (~16 GB KV) -> ~32 GB on a 36 GB M3
        # Pro (~23 GB budget); the fit must reduce the context to fit.
        inst = _make_backend(native_ctx = 262144)
        inst._can_estimate_kv = lambda: True
        inst._estimate_kv_cache_bytes = (
            lambda n, *a, **k: 0 if n <= 0 else int(n * 64_000)  # ~16 GB @ 262144
        )
        model_size_fit = int(15.7 * GIB)
        budget_mib = int(27 * GIB * _APPLE_UNIFIED_MEMORY_FRACTION) // (1024 * 1024)

        # The native footprint over-commits the budget -- this is the bug.
        native_footprint_mib = (model_size_fit + inst._estimate_kv_cache_bytes(262144)) // (
            1024 * 1024
        )
        assert native_footprint_mib > budget_mib

        capped = inst._fit_context_to_vram(
            262144, budget_mib, model_size_fit, None, budget_frac = 1.0
        )
        assert capped < 262144
        capped_footprint_mib = (model_size_fit + inst._estimate_kv_cache_bytes(capped)) // (
            1024 * 1024
        )
        assert capped_footprint_mib <= budget_mib


class TestAppleBranchEndToEnd:
    """Drive the Apple elif glue (cap / floor / explicit) via _drive, no GPU."""

    def test_auto_context_capped_below_native(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            kv_per_token_bytes = 64_000,
            apple_budget_mib = 23_000,  # ~22 GB: weights fit, native KV doesn't
        )
        assert 0 < plan["c_arg"] < 262144
        assert plan["use_fit"] is True  # --fit on still ships as a backstop
        assert plan["gpu_indices"] is None  # no CUDA device pinning on Metal
        assert plan["max_available_ctx"] == plan["c_arg"]

    def test_floors_to_fallback_when_weights_exceed_budget(self):
        # Weights alone exceed budget: ctx can't help, so floor to 4096.
        plan = _drive(
            n_ctx = 0,
            model_gib = 100,
            gpus = [],
            native_ctx = 262144,
            apple_budget_mib = 20_000,
        )
        assert plan["c_arg"] == FALLBACK_CTX
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None

    def test_explicit_context_honored_verbatim(self):
        # Explicit context is never shrunk, but the UI ceiling still tightens.
        plan = _drive(
            n_ctx = 200_000,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            kv_per_token_bytes = 64_000,
            apple_budget_mib = 23_000,
        )
        assert plan["c_arg"] == 200_000  # launch context honored verbatim
        assert plan["use_fit"] is True
        # Ceiling reflects the budget so the over-budget warning still fires.
        assert plan["max_available_ctx"] < 262144


class TestAppleMtpFlatReserve:
    """Apple cap reserves the flat MTP fraction up front (like _pin_fraction) so
    an unsized MTP draft (Qwen3.6-MTP, #6529) can't over-commit."""

    def test_flat_reserve_keeps_draft_within_budget(self):
        # No reserve -> cap fills the budget, leaving nothing for the ~5% draft.
        kw = dict(
            n_ctx = 0,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            kv_per_token_bytes = 64_000,
            apple_budget_mib = 23_000,
        )
        no_reserve = _drive(**kw, flat_mtp_reserve = 0.0)
        with_reserve = _drive(**kw, flat_mtp_reserve = 0.05)

        def footprint_mib(ctx):
            return (15.7 * GIB + ctx * 64_000) / (1024 * 1024)

        # No reserve: main footprint + 5% draft exceeds the budget.
        assert footprint_mib(no_reserve["c_arg"]) + 0.05 * 23_000 > 23_000
        # With reserve: the cap is smaller and the full footprint fits.
        assert with_reserve["c_arg"] < no_reserve["c_arg"]
        assert footprint_mib(with_reserve["c_arg"]) + 0.05 * 23_000 <= 23_000

    def test_no_reserve_is_a_noop_when_mtp_absent(self):
        # flat_mtp_reserve == 0 (the common, non-MTP case) must not change the cap.
        kw = dict(
            n_ctx = 0,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            kv_per_token_bytes = 64_000,
            apple_budget_mib = 23_000,
        )
        assert _drive(**kw, flat_mtp_reserve = 0.0) == _drive(**kw)


class TestAppleNoKvMetadataFloor:
    """Sparse KV metadata floors the auto context to FALLBACK_CTX (like the
    discrete file-size-only fallback) instead of launching at native."""

    def test_sparse_kv_floors_auto_context(self):
        plan = _drive(
            n_ctx = 0,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            can_estimate_kv = False,
            apple_budget_mib = 23_000,
        )
        assert plan["c_arg"] == FALLBACK_CTX  # not native 262144
        assert plan["use_fit"] is True
        assert plan["gpu_indices"] is None

    def test_sparse_kv_still_honors_explicit_context(self):
        plan = _drive(
            n_ctx = 100_000,
            model_gib = 15.7,
            gpus = [],
            native_ctx = 262144,
            can_estimate_kv = False,
            apple_budget_mib = 23_000,
        )
        assert plan["c_arg"] == 100_000  # explicit honored even without KV sizing
