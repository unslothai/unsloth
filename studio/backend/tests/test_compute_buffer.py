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
):
    """Backend with just the dims the compute-buffer estimate reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._vocab_size = vocab
    b._embedding_length = embd
    b._key_length_mla = mla  # non-None -> MLA (compressed attention)
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


GIB = 1024**3


def _ws_backend(embd = 5120, layers = 64):
    """Backend with the dims _estimate_decode_workspace_bytes reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._embedding_length = embd
    b._n_layers = layers
    return b


class TestDecodeWorkspace:
    """Per-slot decode-workspace headroom used to cap serving slots (#6718)."""

    def test_single_slot_is_zero(self):
        # A single stream is covered by the fit cushion; no extra headroom needed.
        assert _ws_backend()._estimate_decode_workspace_bytes(1) == 0
        assert _ws_backend()._estimate_decode_workspace_bytes(0) == 0

    def test_scales_with_slots(self):
        b = _ws_backend()
        assert b._estimate_decode_workspace_bytes(4) > b._estimate_decode_workspace_bytes(2) > 0

    def test_scales_with_model_dims(self):
        big = _ws_backend(5120, 64)._estimate_decode_workspace_bytes(4)
        small = _ws_backend(4096, 32)._estimate_decode_workspace_bytes(4)
        assert big > small  # bigger forward compute -> more workspace per slot

    def test_mtp_inflates(self):
        b = _ws_backend()
        assert b._estimate_decode_workspace_bytes(
            4, mtp_active = True
        ) > b._estimate_decode_workspace_bytes(4)

    def test_27b_per_slot_matches_calibration(self):
        # Calibration anchor: Qwen3.6-27B (5120 x 64) ~2 GiB/slot non-MTP, so --parallel 4
        # needs ~8 GiB free after load (measured recovery point).
        per_slot = _ws_backend(5120, 64)._estimate_decode_workspace_bytes(4) / 4 / GIB
        assert 1.7 <= per_slot <= 2.4

    def test_missing_dims_uses_floor(self):
        b = LlamaCppBackend.__new__(LlamaCppBackend)
        b._embedding_length = 0
        b._n_layers = 0
        assert b._estimate_decode_workspace_bytes(4) > 0


def _cap_backend(embd = 5120, layers = 64):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    # Full arch attribute set (mirrors _draft_backend_for's None defaults) so the
    # KV estimate reaches a real path instead of AttributeError; the cap's
    # try/except would otherwise silently swallow it and return n_parallel.
    for attr in (
        "_kv_lora_rank",
        "_kv_key_length",
        "_kv_value_length",
        "_n_kv_heads_by_layer",
        "_shared_kv_layers",
        "_ssm_inner_size",
        "_full_attention_interval",
        "_sliding_window",
        "_sliding_window_pattern",
        "_kv_key_length_swa",
        "_kv_value_length_swa",
    ):
        setattr(b, attr, None)
    b._n_layers = layers
    b._embedding_length = embd
    b._vocab_size = 248320
    b._n_heads = 40  # Qwen3.6-27B: head_dim = 5120/40 = 128 (matches real GQA cache)
    b._n_kv_heads = 8
    b._context_length = 262144
    b._key_length_mla = None
    return b


def _cap(
    b,
    free_gb,
    total_gb,
    *,
    ndev = 1,
    ctx = 90624,
    model_gib = 17.0,
    mtp = False,
    n_par = 4,
):
    gpus = [(i, int(free_gb * 1024)) for i in range(ndev)]
    tbi = {i: int(total_gb * 1024) for i in range(ndev)}
    fn = (lambda c: int(0.5 * GIB)) if mtp else None
    return b._cap_serving_slots(
        n_par, ctx, list(range(ndev)), gpus, tbi, int(model_gib * GIB), 0, "q8_0", fn, mtp, 512
    )


class TestCapServingSlots:
    """Cap --parallel to what the card can decode at full speed (keeps context)."""

    def test_tight_24gb_27b_caps_to_one(self):
        # The reviewer's case: 24 GB card + 27B + MTP at high ctx -> 1 slot (full speed).
        assert _cap(_cap_backend(), 24, 24, mtp = True) == 1

    def test_roomy_card_keeps_all_slots(self):
        assert _cap(_cap_backend(), 183, 183, mtp = True) == 4

    def test_mid_card_reduces_partially(self):
        assert 1 <= _cap(_cap_backend(), 32, 32, mtp = True) <= 3

    def test_only_reduces_never_raises(self):
        # A 2-slot request on a tight card is at most 2 (never bumped up).
        assert _cap(_cap_backend(), 24, 24, n_par = 2, mtp = True) <= 2

    def test_floor_is_one(self):
        assert _cap(_cap_backend(), 8, 8, mtp = True) == 1

    def test_single_slot_request_unchanged(self):
        assert _cap(_cap_backend(), 24, 24, n_par = 1, mtp = True) == 1

    def test_small_model_not_over_capped(self):
        # A small model on a modest card keeps its slots (dim-scaled workspace is tiny).
        assert _cap(_cap_backend(2048, 28), 16, 16, model_gib = 2.0, mtp = False) == 4

    def test_multi_gpu_gated_by_tightest_device(self):
        # One roomy (80 GB) + one tight (20 GB) device: the tight one constrains slots.
        b = _cap_backend()
        gpus = [(0, 80 * 1024), (1, 20 * 1024)]
        tbi = {0: 80 * 1024, 1: 24 * 1024}
        k = b._cap_serving_slots(
            4,
            90624,
            [0, 1],
            gpus,
            tbi,
            int(34 * GIB),
            0,
            "q8_0",
            (lambda c: int(0.5 * GIB)),
            True,
            512,
        )
        assert k < 4

    def test_no_gpu_pin_unchanged(self):
        # No GPU indices (CPU / --fit path): slot count untouched.
        assert (
            _cap_backend()._cap_serving_slots(
                4, 90624, None, [], {}, int(17 * GIB), 0, "q8_0", None, False, 512
            )
            == 4
        )
