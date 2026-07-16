# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``_estimate_compute_buffer_bytes``: it scales with ``--parallel``,
tensor exceeds pipeline, and it is a safe upper bound on the allocations measured
on real hardware (Qwen3.6-27B-MTP: parallel 1/2/4/8 -> 36/492/1388/3220 MiB single
GPU, ~600 MiB/device tensor). No GPU, subprocess, or GGUF I/O."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)
# httpx -- only stub when the real library is missing. Unconditional stubbing
# shadows HTTPError/Response that huggingface_hub.errors imports at load time,
# silently breaking the transformers introspection tier in tests collected after
# this one (the stub leaks via sys.modules for the whole session).
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

from core.inference.llama_cpp import LlamaCppBackend

MIB = 1024 * 1024


def _backend(
    vocab = 248320,
    embd = 5120,
    mla = None,
    arch = None,
):
    """Backend with just the dims the compute-buffer estimate reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._vocab_size = vocab
    b._embedding_length = embd
    b._key_length_mla = mla  # non-None -> MLA (compressed attention)
    b._architecture = arch  # GGUF general.architecture (e.g. 'deepseek4')
    return b


# Measured ground truth (MiB) the estimate must upper-bound.
_PIPELINE_MEASURED = {1: 36, 2: 492, 4: 1388, 8: 3220}
_TENSOR_MEASURED_PER_DEVICE = 600


class TestSafeUpperBound:
    """The estimate must be >= every measured allocation (never under-reserve)."""

    @pytest.mark.parametrize("parallel,measured", sorted(_PIPELINE_MEASURED.items()))
    def test_pipeline_upper_bounds_measured(self, parallel, measured):
        est = _backend()._estimate_compute_buffer_bytes(n_parallel = parallel) / MIB
        assert est >= measured, f"under-reserved at parallel={parallel}: {est:.0f} < {measured}"

    @pytest.mark.parametrize("parallel,measured", sorted(_PIPELINE_MEASURED.items()))
    def test_pipeline_not_wildly_over(self, parallel, measured):
        # Stay within ~2x of measured so we don't waste context (the point of
        # replacing the flat reserve). parallel=1 is tiny in absolute terms.
        est = _backend()._estimate_compute_buffer_bytes(n_parallel = parallel) / MIB
        assert est <= max(measured * 2.0, 128)

    def test_tensor_upper_bounds_measured(self):
        est = _backend()._estimate_compute_buffer_bytes(n_parallel = 1, per_device_tensor = True) / MIB
        assert est >= _TENSOR_MEASURED_PER_DEVICE

    def test_tensor_far_below_old_flat_reserve(self):
        # The whole point: deterministic estimate << flat 5120 for this model.
        est = _backend()._estimate_compute_buffer_bytes(n_parallel = 1, per_device_tensor = True) / MIB
        assert est < LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB


class TestScaling:
    def test_grows_with_serving_slots(self):
        b = _backend()
        vals = [b._estimate_compute_buffer_bytes(n_parallel = p) for p in (1, 2, 4, 8)]
        assert vals == sorted(vals) and vals[0] < vals[-1]

    def test_parallel_1_is_small(self):
        # Single-token decode: a few tens of MiB, not gigabytes.
        est = _backend()._estimate_compute_buffer_bytes(n_parallel = 1) / MIB
        assert est < 128

    def test_tensor_exceeds_pipeline_at_same_parallel(self):
        b = _backend()
        pipe = b._estimate_compute_buffer_bytes(n_parallel = 1)
        tens = b._estimate_compute_buffer_bytes(n_parallel = 1, per_device_tensor = True)
        assert tens > pipe

    def test_scales_with_vocab(self):
        small = _backend(vocab = 32000)._estimate_compute_buffer_bytes(n_parallel = 4)
        big = _backend(vocab = 256000)._estimate_compute_buffer_bytes(n_parallel = 4)
        assert big > small

    def test_scales_with_ubatch(self):
        b = _backend()
        lo = b._estimate_compute_buffer_bytes(n_parallel = 4, n_ubatch = 256)
        hi = b._estimate_compute_buffer_bytes(n_parallel = 4, n_ubatch = 1024)
        assert hi > lo


class TestFallback:
    def test_zero_when_vocab_missing(self):
        assert _backend(vocab = None)._estimate_compute_buffer_bytes(n_parallel = 4) == 0

    def test_zero_when_embd_missing(self):
        assert _backend(embd = None)._estimate_compute_buffer_bytes(n_parallel = 4) == 0

    def test_zero_lets_tensor_plan_use_flat_fallback(self):
        # When dims are missing, _plan_tensor_parallel must fall back to the flat
        # reserve (defense-in-depth) rather than reserving 0 and OOMing.
        b = _backend(vocab = None, embd = None)
        b._n_layers = None  # can't estimate KV -> floors ctx, still returns a plan
        ec, mac, gi, ts = b._plan_tensor_parallel([(0, 48000), (1, 48000)], 8 * 1024**3, 8192)
        assert gi == [0, 1]  # both GPUs usable under the flat fallback


class TestParallel1Default:
    """At Studio's default --parallel 1 the buffer is negligible in pipeline."""

    def test_default_n_parallel(self):
        est = _backend()._estimate_compute_buffer_bytes() / MIB
        assert est < 128


class TestContextLinearBuffer:
    """``_compute_buffer_ctx_bytes``: the flash-attn KQ-mask + attention scratch
    grow ~linearly with context; the flat estimate above only covers ctx -> 0.
    Measured slope (q8_0 KV, ubatch 512) was 0.74-2.02 x n_embd; 2 x n_embd is the
    worst-case upper bound the term must hold to."""

    # (model, n_embd, ctx, measured CUDA0 compute buffer MiB at that ctx, q8_0/ub512)
    _MEASURED = [
        ("Qwen3.5-2B", 2048, 262144, 796),
        ("Qwen3.5-4B", 2560, 262144, 1330),  # worst slope, 2.02 x n_embd
        ("Qwen3.5-9B", 4096, 262144, 1336),
        ("Qwen3.6-27B", 5120, 262144, 1360),
        ("Gemma-4-31B", 5376, 262144, 2392),
    ]

    def test_zero_by_default(self):
        # Omitted/zero ctx -> no term (keeps the flat callers unchanged).
        assert _backend()._compute_buffer_ctx_bytes(0) == 0

    def test_zero_when_embd_missing(self):
        assert _backend(embd = None)._compute_buffer_ctx_bytes(262144) == 0

    def test_grows_linearly_with_context(self):
        b = _backend(embd = 4096)
        a = b._compute_buffer_ctx_bytes(65536)
        d = b._compute_buffer_ctx_bytes(131072)
        assert d == pytest.approx(2 * a, rel = 1e-6)

    def test_scales_with_embd(self):
        # The quantized (dequant-scratch) rate scales with n_embd; f16 (mask) does not.
        small = _backend(embd = 2048)._compute_buffer_ctx_bytes(131072, cache_type_kv = "q8_0")
        big = _backend(embd = 5120)._compute_buffer_ctx_bytes(131072, cache_type_kv = "q8_0")
        assert big > small

    def test_scales_with_ubatch(self):
        b = _backend(embd = 4096)
        lo = b._compute_buffer_ctx_bytes(131072, n_ubatch = 256)
        hi = b._compute_buffer_ctx_bytes(131072, n_ubatch = 1024)
        assert hi > lo

    @pytest.mark.parametrize("name,embd,ctx,measured", _MEASURED)
    def test_upper_bounds_measured_compute_growth(self, name, embd, ctx, measured):
        # flat term + context-linear term must cover the real (q8_0) buffer at full ctx.
        b = _backend(embd = embd)
        flat = b._estimate_compute_buffer_bytes(n_parallel = 1)
        total = (flat + b._compute_buffer_ctx_bytes(ctx, cache_type_kv = "q8_0")) / MIB
        assert total >= measured, f"{name}: under-reserved {total:.0f} < {measured}"

    def test_worst_case_rate_covers_two_x_embd(self):
        # >= 2 x n_embd bytes per context token at the default micro-batch (the worst
        # measured quantized slope, Qwen3.5-4B), so flat + term upper-bounds the buffer.
        embd = 4096
        b = _backend(embd = embd)
        per_tok = b._compute_buffer_ctx_bytes(100000, cache_type_kv = "q8_0") / 100000
        assert per_tok >= 2 * embd


class TestContextBufferKVQuant:
    """The context-linear rate depends on the KV cache type: a quantized cache adds a
    context-sized dequant scratch (heavy); f16/bf16/f32 only pays the KQ mask (light).
    Measured Qwen3.5-4B at 256k: 1.30 GiB (q8_0) vs 0.31 GiB (f16)."""

    def test_quantized_heavier_than_f16(self):
        b = _backend(embd = 4096)
        q = b._compute_buffer_ctx_bytes(131072, cache_type_kv = "q8_0")
        f = b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        assert q > f

    def test_none_cache_type_is_f16(self):
        # None -> f16 (llama.cpp's default); the env-quantized case is covered by the
        # KV budget's f16 over-reservation, so we take the lighter mask-only rate.
        b = _backend(embd = 4096)
        assert b._compute_buffer_ctx_bytes(
            131072, cache_type_kv = None
        ) == b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")

    @pytest.mark.parametrize("ct", ["f16", "bf16", "f32"])
    def test_unquantized_uses_mask_only_rate(self, ct):
        # f16/bf16/f32: KQ mask only, n_ubatch*2 B/tok, independent of n_embd.
        b_small = _backend(embd = 2048)
        b_big = _backend(embd = 8192)
        per_small = b_small._compute_buffer_ctx_bytes(100000, cache_type_kv = ct) / 100000
        per_big = b_big._compute_buffer_ctx_bytes(100000, cache_type_kv = ct) / 100000
        assert per_small == per_big  # no n_embd scaling on the f16 path
        expected = 512 * 2 * LlamaCppBackend._CTX_COMPUTE_F16_MASK_SAFETY  # ubatch 512
        assert per_small == pytest.approx(expected, rel = 1e-6)

    @pytest.mark.parametrize("ct", ["q8_0", "q5_1", "q4_0", "iq4_nl"])
    def test_quantized_types_use_heavy_rate(self, ct):
        embd = 4096
        b = _backend(embd = embd)
        per_tok = b._compute_buffer_ctx_bytes(100000, cache_type_kv = ct) / 100000
        assert per_tok == pytest.approx(
            LlamaCppBackend._CTX_COMPUTE_BYTES_PER_EMBD * embd, rel = 1e-6
        )

    def test_f16_covers_measured_mask(self):
        # f16 buffer is ~mask only (~n_ubatch*2 B/tok); 0.5 x n_embd must cover the
        # measured Qwen3.5-4B f16 slope (~0.4 x n_embd = 0.31 GiB at 256k).
        b = _backend(embd = 2560)  # Qwen3.5-4B
        est = b._compute_buffer_ctx_bytes(262144, cache_type_kv = "f16") / MIB
        assert est >= 320  # measured 0.31 GiB growth


class TestContextBufferMLA:
    """MLA (compressed attention) needs a smaller quantized dequant scratch than
    regular attention: measured 0.94 x n_embd on GLM-5.2 and Kimi-K2.7 vs up to
    2.02x on Qwen/Gemma. Charging the regular rate would badly over-reserve a tight
    multi-GPU MLA pin (per-device scaling multiplies the error)."""

    def test_mla_lighter_than_regular(self):
        reg = _backend(embd = 6144, mla = None)._compute_buffer_ctx_bytes(262144, cache_type_kv = "q8_0")
        mla = _backend(embd = 6144, mla = 256)._compute_buffer_ctx_bytes(262144, cache_type_kv = "q8_0")
        assert mla < reg

    @pytest.mark.parametrize(
        "name,embd,ctx,measured",
        [
            ("GLM-5.2", 6144, 754688, 4141),  # per-device compute MiB at q8_0
            ("Kimi-K2.7", 7168, 262144, 1690),
        ],
    )
    def test_mla_rate_covers_measured(self, name, embd, ctx, measured):
        b = _backend(embd = embd, mla = 256)
        est = b._compute_buffer_ctx_bytes(ctx, cache_type_kv = "q8_0") / MIB
        assert est >= measured, f"{name}: MLA under-reserved {est:.0f} < {measured}"

    def test_mla_not_wildly_over(self):
        # 1.25 x n_embd should stay within ~1.6x of the measured 0.94x (not 2.4x like
        # the regular 2.25 rate would), so a multi-GPU MLA pin keeps its context.
        b = _backend(embd = 6144, mla = 256)
        est = b._compute_buffer_ctx_bytes(754688, cache_type_kv = "q8_0") / MIB
        assert est <= 4141 * 1.7


class TestContextBufferDSV4:
    """DeepSeek-V4 (deepseek4) reserves a large lightning-indexer / sparse-attention
    compute buffer the KQ-mask and MLA rates miss (present even with an f16 cache).
    Measured on UD-Q4_K_XL (ub=512): ~2 GiB at 16k ctx, ~65.5 GiB at 1M. The auto-fit
    must see this so it does not commit the full 1M train context and OOM (spilling
    to CPU at ~4 tok/s)."""

    _MEASURED_1M_GIB = 65.5  # 70353790464 B compute-graph reserve that OOM'd at 1M ctx
    GIB = 1024**3

    def test_covers_measured_1m_buffer(self):
        b = _backend(embd = 4096, arch = "deepseek4")
        gib = b._compute_buffer_ctx_bytes(1048576, cache_type_kv = "f16") / self.GIB
        assert gib >= self._MEASURED_1M_GIB, f"under-reserved {gib:.1f} < {self._MEASURED_1M_GIB}"

    def test_not_wildly_over_at_1m(self):
        # Within ~1.3x of measured so the fit still grants a large (~256k) context.
        b = _backend(embd = 4096, arch = "deepseek4")
        gib = b._compute_buffer_ctx_bytes(1048576, cache_type_kv = "f16") / self.GIB
        assert gib <= self._MEASURED_1M_GIB * 1.3

    def test_fires_for_f16_cache(self):
        # The bug: an f16 (default) cache took the tiny mask-only path. DSV4 must
        # reserve GiB, not the ~MiB a non-DSV4 model reserves at the same ctx.
        dsv4 = _backend(embd = 4096, arch = "deepseek4")._compute_buffer_ctx_bytes(
            262144, cache_type_kv = "f16"
        )
        other = _backend(embd = 4096, arch = "qwen3")._compute_buffer_ctx_bytes(
            262144, cache_type_kv = "f16"
        )
        assert dsv4 > 40 * other

    def test_cache_type_independent(self):
        # Indexer scratch is present for an f16 and a quantized cache alike.
        b = _backend(embd = 4096, arch = "deepseek4")
        assert b._compute_buffer_ctx_bytes(
            262144, cache_type_kv = "f16"
        ) == b._compute_buffer_ctx_bytes(262144, cache_type_kv = "q8_0")

    def test_flat_floor_at_small_ctx(self):
        # ~2 GiB indexer scratch present even at tiny ctx (covers the measured 16k ~2 GiB).
        b = _backend(embd = 4096, arch = "deepseek4")
        assert b._compute_buffer_ctx_bytes(16384, cache_type_kv = "f16") / self.GIB >= 2.0

    def test_scales_with_context_and_ubatch(self):
        b = _backend(embd = 4096, arch = "deepseek4")
        assert b._compute_buffer_ctx_bytes(131072) > b._compute_buffer_ctx_bytes(65536)
        assert b._compute_buffer_ctx_bytes(131072, n_ubatch = 1024) > b._compute_buffer_ctx_bytes(
            131072, n_ubatch = 256
        )

    def test_non_dsv4_unchanged(self):
        # Regression guard: a non-deepseek4 model keeps the mask-only f16 rate.
        b = _backend(embd = 4096, arch = "llama")
        per_tok = b._compute_buffer_ctx_bytes(100000, cache_type_kv = "f16") / 100000
        expected = 512 * 2 * LlamaCppBackend._CTX_COMPUTE_F16_MASK_SAFETY
        assert per_tok == pytest.approx(expected, rel = 1e-6)
