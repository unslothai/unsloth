# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compute-buffer estimates checked against measured allocations and synthetic layouts."""

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

from core.inference.llama_cpp import (
    _FIT_MIN_CTX,
    _kv_bytes_per_elem,
    _planned_main_cache_types,
    _planned_scratch_cache_type,
    LlamaCppBackend,
)

MIB = 1024 * 1024
GIB = 1024 * MIB


def _backend(
    vocab = 248320,
    embd = 5120,
    mla = None,
    arch = None,
    pooling_type = None,
):
    """Backend with just the dims the compute-buffer estimate reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._vocab_size = vocab
    b._embedding_length = embd
    b._key_length_mla = mla  # non-None -> MLA (compressed attention)
    b._architecture = arch  # GGUF general.architecture (e.g. 'deepseek4')
    b._pooling_type = pooling_type
    return b


def _pad(cells):
    return ((cells + 255) // 256) * 256


def _hidden_copies(embd, ub = 512):
    """The hidden-state copies a layer-split device holds, one per scheduler copy."""
    return LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT * embd * ub * 4


def _backend_from_gguf_local(
    n_layers = 80,
    embd = 8192,
    n_kv_heads = 8,
    head_dim = 128,
):
    """Backend that can also estimate KV bytes, for the planner cells below.

    A real ``__init__`` (not ``__new__`` like ``_backend`` above) so every
    attribute the KV estimator reads exists at its default; only the handful of
    dims these cells vary are overridden. These tests are about which cache type
    is priced, not about GGUF parsing.
    """
    b = LlamaCppBackend()
    b._vocab_size = 248_320
    b._embedding_length = embd
    b._n_layers = n_layers
    b._n_kv_heads = n_kv_heads
    b._kv_key_length = head_dim
    b._kv_value_length = head_dim
    b._context_length = 262_144
    assert b._can_estimate_kv()
    return b


_GEMMA4_12B_PATTERN = ([True] * 5 + [False]) * 8

# GGUF dims of models whose buffers were measured on the bundled llama.cpp b10909
# (RTX 6000 Ada, --flash-attn on, --kv-unified above one slot).
_DIMS = {
    "Llama-3.2-1B": dict(
        _vocab_size = 128256,
        _embedding_length = 2048,
        _feed_forward_length = 8192,
        _n_layers = 16,
        _n_heads = 32,
        _n_kv_heads = 8,
        _kv_key_length = 64,
        _kv_value_length = 64,
    ),
    "Qwen3-8B": dict(
        _vocab_size = 151936,
        _embedding_length = 4096,
        _feed_forward_length = 12288,
        _n_layers = 36,
        _n_heads = 32,
        _n_kv_heads = 8,
        _kv_key_length = 128,
        _kv_value_length = 128,
    ),
    "Qwen3.5-4B": dict(
        _vocab_size = 248320,
        _embedding_length = 2560,
        _feed_forward_length = 9216,
        _n_layers = 32,
        _n_heads = 16,
        _n_kv_heads = 4,
        _kv_key_length = 256,
        _kv_value_length = 256,
        _ssm_inner_size = 4096,
        _ssm_state_size = 128,
        _ssm_group_count = 16,
        _full_attention_interval = 4,
    ),
    "Qwen3.6-35B-A3B": dict(
        _vocab_size = 248320,
        _embedding_length = 2048,
        _expert_used_count = 8,
        _expert_feed_forward_length = 512,
        _expert_shared_feed_forward_length = 512,
        _n_layers = 40,
        _n_heads = 16,
        _n_kv_heads = 2,
        _kv_key_length = 256,
        _kv_value_length = 256,
        _ssm_inner_size = 4096,
        _ssm_state_size = 128,
        _ssm_group_count = 16,
        _full_attention_interval = 4,
    ),
    "Gemma-4-12B": dict(
        _vocab_size = 262144,
        _embedding_length = 3840,
        _feed_forward_length = 15360,
        _n_layers = 48,
        _n_heads = 16,
        _n_kv_heads = 8,
        _n_kv_heads_by_layer = [1 if not swa else 8 for swa in _GEMMA4_12B_PATTERN],
        _kv_key_length = 512,
        _kv_value_length = 512,
        _kv_key_length_swa = 256,
        _kv_value_length_swa = 256,
        _sliding_window = 1024,
        _sliding_window_pattern = _GEMMA4_12B_PATTERN,
    ),
    "Gemma-4-E4B": dict(
        _vocab_size = 262144,
        _embedding_length = 2560,
        _feed_forward_length = 10240,
        _embedding_length_per_layer_input = 256,
        _n_layers = 42,
        _n_heads = 8,
        _n_kv_heads = 2,
        _kv_key_length = 512,
        _kv_value_length = 512,
        _kv_key_length_swa = 256,
        _kv_value_length_swa = 256,
        _sliding_window = 512,
        _sliding_window_pattern = ([True] * 5 + [False]) * 7,
    ),
    "GLM-4.7-Flash": dict(
        _vocab_size = 154880,
        _embedding_length = 2048,
        _feed_forward_length = 10240,
        _expert_used_count = 4,
        _expert_feed_forward_length = 1536,
        _expert_shared_count = 1,
        _n_layers = 47,
        _n_heads = 20,
        _n_kv_heads = 1,
        _kv_key_length = 576,
        _kv_value_length = 512,
        _kv_lora_rank = 512,
        _key_length_mla = 256,
    ),
}


def _measured(name):
    b = LlamaCppBackend()
    for attr, value in _DIMS[name].items():
        setattr(b, attr, value)
    return b


# llama-server's "CUDA0 compute buffer size" (MiB): (model, ubatch, slots, n_ctx, KV type).
_MEASURED_COMPUTE = [
    ("Llama-3.2-1B", 256, 4, 32768, "f16", 46.01),
    ("Llama-3.2-1B", 2048, 4, 32768, "f16", 368.04),
    ("Llama-3.2-1B", 512, 1, 32768, "f16", 92.01),
    ("Llama-3.2-1B", 512, 8, 32768, "f16", 92.01),
    ("Llama-3.2-1B", 512, 4, 131072, "f16", 188.01),
    ("Llama-3.2-1B", 512, 4, 131072, "q8_0", 400.04),
    ("Llama-3.2-1B", 2048, 4, 131072, "q8_0", 832.07),
    ("Qwen3-8B", 512, 4, 32768, "f16", 128.01),
    ("Qwen3-8B", 2048, 4, 131072, "f16", 896.04),
    ("Qwen3-8B", 512, 4, 131072, "q8_0", 672.09),
    ("Qwen3.5-4B", 512, 1, 32768, "f16", 96.02),
    ("Qwen3.5-4B", 512, 8, 32768, "f16", 115.52),
    ("Qwen3.5-4B", 2048, 4, 131072, "f16", 768.06),
    ("Qwen3.5-4B", 2048, 4, 131072, "q8_0", 1224.34),
    ("Qwen3.6-35B-A3B", 256, 4, 32768, "f16", 66.02),
    ("Qwen3.6-35B-A3B", 2048, 1, 32768, "f16", 528.2),
    ("Qwen3.6-35B-A3B", 512, 4, 131072, "q8_0", 436.31),
    ("Gemma-4-12B", 512, 1, 32768, "f16", 154.52),
    ("Gemma-4-12B", 1120, 4, 32768, "f16", 346.21),
    ("Gemma-4-12B", 2048, 8, 32768, "f16", 652.07),
    ("Gemma-4-12B", 2048, 4, 131072, "f16", 1020.07),
    ("Gemma-4-12B", 512, 4, 131072, "q8_0", 436.8),
    ("Gemma-4-E4B", 512, 4, 32768, "f16", 135.52),
    ("Gemma-4-E4B", 2048, 4, 131072, "q8_0", 1317.35),
    ("GLM-4.7-Flash", 512, 4, 32768, "f16", 110.01),
    ("GLM-4.7-Flash", 2048, 4, 131072, "f16", 824.04),
    ("GLM-4.7-Flash", 512, 4, 131072, "q8_0", 358.9),
]


def _priced_mib(b, ub, slots, ctx, kv):
    flat = b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
    return (flat + b._compute_buffer_ctx_bytes(ctx, ub, kv)) / MIB


class TestMeasuredComputeBuffer:
    """Flat plus context-linear must cover llama-server's own reserve without
    inflating it: the vocab-wide rows the old formula charged per slot came to 16 GiB
    for Gemma 4 12B at 8 slots and ubatch 2048, which reserves 652 MiB."""

    @pytest.mark.parametrize("name,ub,slots,ctx,kv,measured", _MEASURED_COMPUTE)
    def test_covers_the_measured_buffer(self, name, ub, slots, ctx, kv, measured):
        est = _priced_mib(_measured(name), ub, slots, ctx, kv)
        assert est >= measured, f"{name} ub{ub} p{slots} {ctx} {kv}: {est:.0f} < {measured}"

    @pytest.mark.parametrize("name,ub,slots,ctx,kv,measured", _MEASURED_COMPUTE)
    def test_stays_close_to_the_measured_buffer(self, name, ub, slots, ctx, kv, measured):
        # The quantized scratch and the activations peak at different points of the
        # graph but are charged together, the widest gap at a short context.
        est = _priced_mib(_measured(name), ub, slots, ctx, kv)
        assert est <= measured * (1.35 if kv == "f16" else 1.3), f"{name}: {est:.0f} vs {measured}"

    def test_the_reported_gemma4_vision_raise(self):
        # The micro-batch raise for the Gemma 4 projector, 512 -> 1120 at 4 slots and
        # 262144 cells: llama-server's buffer grows 455 MiB, the old estimate 2.5 GiB.
        b = _measured("Gemma-4-12B")
        raise_mib = _priced_mib(b, 1120, 4, 262144, "f16") - _priced_mib(b, 512, 4, 262144, "f16")
        assert 455 <= raise_mib <= 455 * 1.15


class TestFlatBufferShape:
    def test_slots_add_output_rows_not_activations(self):
        b = _measured("Gemma-4-12B")
        one = b._estimate_compute_buffer_bytes(n_ubatch = 2048, n_parallel = 1)
        eight = b._estimate_compute_buffer_bytes(n_ubatch = 2048, n_parallel = 8)
        rows = 7 * 262144 * 4
        swa = (_pad(1024 * 8 + 2048) - _pad(1024 + 2048)) * 2048 * 2
        assert eight - one == pytest.approx(
            rows * LlamaCppBackend._COMPUTE_BUFFER_SAFETY + swa, rel = 1e-6
        )

    def test_scales_linearly_with_ubatch(self):
        b = _measured("Qwen3-8B")
        lo = b._estimate_compute_buffer_bytes(n_ubatch = 512, n_parallel = 1)
        hi = b._estimate_compute_buffer_bytes(n_ubatch = 1024, n_parallel = 1)
        rows = 151936 * 4 * LlamaCppBackend._COMPUTE_BUFFER_SAFETY
        assert hi - rows == pytest.approx(2 * (lo - rows), rel = 1e-6)

    def test_a_tensor_device_holds_the_single_device_buffer(self):
        # Qwen3 8B and Gemma 4 12B split over two GPUs reserve their single-GPU
        # buffer on each device, at ubatch 512 and 2048 alike.
        for name in ("Qwen3-8B", "Gemma-4-12B"):
            b = _measured(name)
            assert b._estimate_compute_buffer_bytes(
                n_ubatch = 2048, n_parallel = 4, per_device_tensor = True
            ) == b._estimate_compute_buffer_bytes(n_ubatch = 2048, n_parallel = 4)

    def test_embedding_rows_are_pooled_not_vocab_wide(self):
        b = _backend(vocab = 30522, embd = 384, pooling_type = 1)
        chat = _backend(vocab = 30522, embd = 384)
        assert b._estimate_compute_buffer_bytes(n_ubatch = 512, n_parallel = 4) > (
            chat._estimate_compute_buffer_bytes(n_ubatch = 512, n_parallel = 4)
        )
        # 512 pooled rows of 384 floats, far below 512 vocabulary rows.
        assert b._estimate_compute_buffer_bytes(n_ubatch = 512, n_parallel = 4) < 30522 * 512 * 4

    def test_widest_block_sets_the_width(self):
        # Dense FFN, routed experts, and the delta-net convolution each can be widest.
        assert _measured("Qwen3-8B")._compute_activation_width() == 4 * 12288
        assert _measured("Qwen3.6-35B-A3B")._compute_activation_width() == (
            8 * (2 * 2048 + 3 * 512) + 3 * 512
        )
        ssm = _backend(embd = 2560)
        ssm._ssm_inner_size, ssm._ssm_state_size, ssm._ssm_group_count = 4096, 128, 16
        assert ssm._compute_activation_width() == 4 * (4096 + 2 * 16 * 128)
        # Per-layer input embeddings ride on top of the widest block.
        assert _measured("Gemma-4-E4B")._compute_activation_width() == 4 * 10240 + 256 * 42

    def test_header_without_ffn_dims_uses_the_embedding_floor(self):
        assert _backend(embd = 4096)._compute_activation_width() == 12 * 4096

    def test_expert_width_falls_back_to_the_ffn_length(self):
        # Granite 4.0 H writes its 512-wide experts as feed_forward_length.
        b = _backend(vocab = 100352, embd = 1536)
        b._n_experts, b._expert_used_count = 64, 6
        b._feed_forward_length, b._expert_shared_feed_forward_length = 512, 1024
        assert b._compute_activation_width() == 6 * (2 * 1536 + 3 * 512) + 3 * 1024

    def test_granite_hybrid_is_covered(self):
        # Held out from the fit: llama-server reserved 80.0 MiB at one slot and 95.3
        # MiB at eight (ubatch 512, 32768 cells), the recurrent state per slot.
        b = _backend(vocab = 100352, embd = 1536)
        b._n_experts, b._expert_used_count = 64, 6
        b._feed_forward_length, b._expert_shared_feed_forward_length = 512, 1024
        b._ssm_inner_size, b._ssm_state_size, b._ssm_group_count = 3072, 128, 1
        for slots, measured in ((1, 80.0), (8, 95.3)):
            est = _priced_mib(b, 512, slots, 32768, "f16")
            assert measured <= est <= measured * 1.3

    def test_a_window_wider_than_the_context_does_not_slide(self):
        # Phi-4-mini declares a 262144 window over a 131072 context and no pattern;
        # pricing it as a sliding mask charged 12x its 92 MiB buffer.
        b = _backend(vocab = 200064, embd = 3072)
        b._feed_forward_length, b._context_length = 8192, 131072
        b._sliding_window, b._sliding_window_pattern = 262144, None
        assert b._effective_sliding_window() == 0
        assert 92.0 <= _priced_mib(b, 512, 4, 32768, "f16") <= 92.0 * 1.3
        b._sliding_window_pattern = [False] * 32
        assert b._effective_sliding_window() == 0
        b._sliding_window_pattern = [True, False] * 16
        assert b._effective_sliding_window() == 262144


class TestFallback:
    def test_zero_when_vocab_missing(self):
        assert _backend(vocab = None)._estimate_compute_buffer_bytes(n_parallel = 4) == 0

    def test_vocab_ceiling_prices_a_header_without_a_token_array(self):
        b = _backend(vocab = None)
        assert b._estimate_compute_buffer_bytes(n_parallel = 4, vocab_ceiling = 262144) == (
            _backend(vocab = 262144)._estimate_compute_buffer_bytes(n_parallel = 4)
        )

    def test_zero_when_embd_missing(self):
        assert _backend(embd = None)._estimate_compute_buffer_bytes(n_parallel = 4) == 0

    def test_zero_lets_tensor_plan_use_flat_fallback(self):
        # When dims are missing, _plan_tensor_parallel must fall back to the flat
        # reserve (defense-in-depth) rather than reserving 0 and OOMing.
        b = _backend(vocab = None, embd = None)
        b._n_layers = None  # can't estimate KV -> floors ctx, still returns a plan
        ec, mac, gi, ts = b._plan_tensor_parallel([(0, 48000), (1, 48000)], 8 * 1024**3, 8192)
        assert gi == [0, 1]  # both GPUs usable under the flat fallback


class TestContextLinearBuffer:
    """``_compute_buffer_ctx_bytes``: the KQ mask, n_ubatch * 2 bytes per cell, plus
    for a quantized cache an f16 copy of the widest global layer's K and V."""

    # (model, n_embd, ctx, measured CUDA0 compute buffer MiB at that ctx, q8_0/ub512).
    # From an earlier build; kept as upper bounds on a header with only n_embd.
    _MEASURED = [
        ("Qwen3.5-2B", 2048, 262144, 796),
        ("Qwen3.5-4B", 2560, 262144, 1330),
        ("Qwen3.5-9B", 4096, 262144, 1336),
        ("Qwen3.6-27B", 5120, 262144, 1360),
        ("Gemma-4-31B", 5376, 262144, 2392),
    ]

    def test_zero_by_default(self):
        assert _backend()._compute_buffer_ctx_bytes(0) == 0

    def test_zero_when_embd_missing(self):
        assert _backend(embd = None)._compute_buffer_ctx_bytes(262144) == 0

    def test_grows_linearly_with_context(self):
        b = _measured("Qwen3-8B")
        for kv in ("f16", "q8_0"):
            a = b._compute_buffer_ctx_bytes(65536, cache_type_kv = kv)
            d = b._compute_buffer_ctx_bytes(131072, cache_type_kv = kv)
            assert d == pytest.approx(2 * a, rel = 1e-6)

    @pytest.mark.parametrize("name,embd,ctx,measured", _MEASURED)
    def test_upper_bounds_measured_compute_growth(self, name, embd, ctx, measured):
        b = _backend(embd = embd)
        flat = b._estimate_compute_buffer_bytes(n_parallel = 1)
        total = (flat + b._compute_buffer_ctx_bytes(ctx, cache_type_kv = "q8_0")) / MIB
        assert total >= measured, f"{name}: under-reserved {total:.0f} < {measured}"


class TestContextBufferKVQuant:
    def test_quantized_heavier_than_f16(self):
        b = _measured("Qwen3-8B")
        assert b._compute_buffer_ctx_bytes(131072, cache_type_kv = "q8_0") > (
            b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        )

    def test_none_cache_type_is_f16(self):
        b = _measured("Qwen3-8B")
        assert b._compute_buffer_ctx_bytes(131072, cache_type_kv = None) == (
            b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        )

    @pytest.mark.parametrize("ct", ["f16", "bf16", "f32"])
    def test_unquantized_pays_exactly_the_mask(self, ct):
        # n_ubatch * 2 bytes per cell whatever the model, as measured on every family.
        for name in _DIMS:
            b = _measured(name)
            for ub in (512, 2048):
                assert b._compute_buffer_ctx_bytes(100000, ub, ct) == ub * 2 * 100000

    @pytest.mark.parametrize("ct", ["q8_0", "q5_1", "q4_0", "iq4_nl"])
    def test_quantized_adds_the_widest_kv_copy_whatever_the_ubatch(self, ct):
        b = _measured("Qwen3-8B")
        for ub in (512, 2048):
            extra = b._compute_buffer_ctx_bytes(100000, ub, ct) - ub * 2 * 100000
            assert extra == 8 * (128 + 128) * 2 * 100000

    def test_sliding_window_layers_do_not_set_the_rate(self):
        # Gemma 4 12B: 8 KV heads on the sliding layers, 1 on the global ones.
        assert _measured("Gemma-4-12B")._attention_dequant_width() == 1 * (512 + 512)

    def test_mla_copies_the_compressed_key_only(self):
        assert _measured("GLM-4.7-Flash")._attention_dequant_width() == 576

    def test_header_without_head_dims_falls_back_to_embd(self):
        assert _backend(embd = 4096)._attention_dequant_width() == 4096


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
        assert per_tok == pytest.approx(512 * 2, rel = 1e-6)


class TestContextBufferLayerSplit:
    """The per-device f16 (KQ-mask) rate steps 4x once the model spans >1 device under
    ``-sm layer``: a step on "is split", not a ramp in device count. ``_MEASURED`` is
    from llama.cpp's memory breakdown: compute-column ctx slope / n_ctx / n_ubatch."""

    _RATE_SINGLE = 2.0  # B/tok/ubatch, per device
    _RATE_SPLIT = 8.0
    # (model, n_gpus, n_ubatch, measured B/tok/ubatch/device)
    _MEASURED = [
        ("Qwen3.5-9B-MTP", 1, 512, 2.0),
        ("Qwen3.5-9B-MTP", 2, 512, 8.0),
        ("Qwen3.6-27B-MTP", 1, 512, 2.0),
        ("Qwen3.6-27B-MTP", 4, 512, 8.0),
        ("Qwen3.6-35B-A3B", 1, 2048, 2.0),
        ("Qwen3.6-35B-A3B", 2, 2048, 8.0),
        ("gemma-4-12b-it", 1, 512, 2.0),
        ("gemma-4-12b-it", 3, 512, 8.0),
        ("gemma-4-12b-it", 5, 512, 8.0),
        ("gemma-4-12b-it", 6, 512, 8.0),
        ("gemma-4-12b-it", 8, 512, 8.0),
        ("gemma-4-26B-A4B-it", 1, 512, 2.0),
        ("gemma-4-26B-A4B-it", 2, 512, 8.0),
        ("Kimi-K3", 4, 512, 8.0),
    ]

    def test_default_is_single_device(self):
        # Existing callers are unaffected: the flag defaults off.
        b = _backend(embd = 4096)
        assert b._compute_buffer_ctx_bytes(
            131072, cache_type_kv = "f16"
        ) == b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16", layer_split = False)

    def test_split_is_the_multiplier_plus_the_hidden_state_copies(self):
        b = _backend(embd = 4096)
        one = b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        many = b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16", layer_split = True)
        assert many == one * LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT + _hidden_copies(4096)

    def test_matches_the_measured_two_gpu_split(self):
        # Qwen3 8B over an RTX 6000 Ada and an RTX 3090: 256.04 MiB on each device at
        # 32768 cells and ubatch 512, 2560.16 MiB at 131072 cells and ubatch 2048.
        b = _measured("Qwen3-8B")
        for ctx, ub, measured in ((32768, 512, 256.04), (131072, 2048, 2560.16)):
            per_device = (
                b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = 1)
                + b._compute_buffer_ctx_bytes(ctx, ub, "f16", layer_split = True)
            ) / MIB
            assert measured <= per_device <= measured * 1.1

    @pytest.mark.parametrize("name,n_gpus,ub,measured", _MEASURED)
    def test_upper_bounds_measured_per_device_rate(self, name, n_gpus, ub, measured):
        # Never under-reserve.
        b = _backend(embd = 4096)
        per_tok = (
            b._compute_buffer_ctx_bytes(
                100000, n_ubatch = ub, cache_type_kv = "f16", layer_split = n_gpus > 1
            )
            / 100000
        )
        assert per_tok >= measured * ub, f"{name} n={n_gpus}: {per_tok:.0f} < {measured * ub:.0f}"

    @pytest.mark.parametrize("name,n_gpus,ub,measured", _MEASURED)
    def test_charges_exactly_the_measured_per_device_rate(self, name, n_gpus, ub, measured):
        # The mask is priced exactly; only the split's hidden-state copies sit on top.
        b = _backend(embd = 4096)
        hidden = _hidden_copies(4096, ub) if n_gpus > 1 else 0
        charged = b._compute_buffer_ctx_bytes(
            100000, n_ubatch = ub, cache_type_kv = "f16", layer_split = n_gpus > 1
        )
        assert charged - hidden == pytest.approx(measured * ub * 100000, rel = 1e-6)

    def test_pre_fix_split_reserve_was_short(self):
        # The bug: without the step a split reserved a quarter of its masks.
        b = _backend(embd = 4096)
        one = b._compute_buffer_ctx_bytes(1048576, cache_type_kv = "f16")
        measured = self._RATE_SPLIT * 512 * 1048576
        assert one < measured
        assert (
            b._compute_buffer_ctx_bytes(1048576, cache_type_kv = "f16", layer_split = True) >= measured
        )

    def test_kimi_k3_1m_four_gpu_reserve(self):
        # The reported case: Kimi-K3 UD-IQ1_M, 1M ctx, 4 GPUs, ub 512. llama.cpp
        # allocated 4.0 GiB per device; Unsloth reserved 1.5 GiB.
        b = _backend(embd = 7168, mla = 576)
        gib = b._compute_buffer_ctx_bytes(1048576, cache_type_kv = "f16", layer_split = True) / (
            1024**3
        )
        assert 4.0 <= gib <= 6.5

    @pytest.mark.parametrize("ct", ["q8_0", "q4_0"])
    @pytest.mark.parametrize("embd,mla", [(2048, None), (8192, None), (7168, 576)])
    def test_quantized_adds_the_mask_delta(self, ct, embd, mla):
        # The quantized rate is a single-GPU TOTAL holding mask*1 + dequant scratch.
        # Only the mask replicates, so a split adds exactly 3 more masks.
        b = _backend(embd = embd, mla = mla)
        mask = b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        single = b._compute_buffer_ctx_bytes(131072, cache_type_kv = ct)
        split = b._compute_buffer_ctx_bytes(131072, cache_type_kv = ct, layer_split = True)
        delta = (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT - 1) * mask + _hidden_copies(embd)
        assert split == pytest.approx(single + delta, rel = 1e-9)

    @pytest.mark.parametrize("embd,mla", [(2048, None), (2560, None), (8192, None), (7168, 576)])
    def test_quantized_split_beats_the_old_max_floor(self, embd, mla):
        # The pre-fix floor max(quantized, 4x mask) treated the dequant scratch and
        # the enlarged mask as alternatives, leaving the smaller one unbudgeted.
        b = _backend(embd = embd, mla = mla)
        old_floor = max(
            b._compute_buffer_ctx_bytes(262144, cache_type_kv = "q8_0"),
            b._compute_buffer_ctx_bytes(262144, cache_type_kv = "f16", layer_split = True),
        )
        assert b._compute_buffer_ctx_bytes(262144, cache_type_kv = "q8_0", layer_split = True) > (
            old_floor
        )

    def test_quantized_split_covers_measured_plus_mask(self):
        # Qwen3.5-4B (n_embd 2560) at 256k q8_0: 1330 MiB measured single-device,
        # + 3 replicated [n_kv, ub] f16 masks (768 MiB) = 2098 MiB per device. The
        # old floor reserved 1536 MiB and left ~560 MiB/device unbudgeted.
        b = _backend(embd = 2560)
        ctx, ub = 262144, 512
        mask_mib = 3 * ub * 2 * ctx / MIB
        assert mask_mib == pytest.approx(768.0, rel = 1e-6)
        split_mib = b._compute_buffer_ctx_bytes(ctx, ub, "q8_0", layer_split = True) / MIB
        assert split_mib >= 1330 + mask_mib

    def test_deepseek4_rate_unchanged(self):
        # Its own rate already carries the mask copies: 72000 - 65.5 KiB/tok measured
        # = 4928 B/tok of margin against the 4608 a split adds.
        b = _backend(embd = 4096, arch = "deepseek4")
        assert b._compute_buffer_ctx_bytes(
            131072, cache_type_kv = "f16", layer_split = True
        ) == b._compute_buffer_ctx_bytes(131072, cache_type_kv = "f16")
        measured = TestContextBufferDSV4._MEASURED_1M_GIB * 1024**3 / 1048576
        split_masks = (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT - 1) * 512 * 2
        assert LlamaCppBackend._DSV4_CTX_COMPUTE_BYTES_PER_TOK - measured >= split_masks


class TestContextBufferInklingSplit:
    """Inkling's rates are single-device totals too, and the banded 8192 B/tok has only
    ~1.5x headroom over its 5.6 KiB/tok measurement, too little for a split's masks."""

    _MEASURED_BANDED = 5734  # ~5.6 KiB/tok compute at ub 512 (see the constant)
    _CTX = 1048576
    _UB = 512

    def _rate(
        self,
        ct,
        layer_split,
        ub = 512,
    ):
        b = _backend(embd = 4096, arch = "inkling")
        return (
            b._compute_buffer_ctx_bytes(self._CTX, ub, cache_type_kv = ct, layer_split = layer_split)
            / self._CTX
        )

    def test_pre_fix_banded_split_reserve_was_short(self):
        # The bug: the banded rate alone does not cover measured + 3 more masks.
        measured_split = self._MEASURED_BANDED + 3 * self._UB * 2
        assert LlamaCppBackend._INKLING_CTX_COMPUTE_BYTES_PER_TOK < measured_split
        assert self._rate("f16", True) >= measured_split

    @pytest.mark.parametrize("ct", ["f16", None, "q8_0"])
    def test_split_adds_exactly_the_extra_mask_copies(self, ct):
        delta = (LlamaCppBackend._CTX_COMPUTE_SPLIT_MULT - 1) * self._UB * 2
        assert self._rate(ct, True) == pytest.approx(self._rate(ct, False) + delta, rel = 1e-9)

    def test_dense_fallback_delta_is_present_but_tiny(self):
        # ~402 KiB/tok dwarfs the 4608 B/tok of masks, yet the masks are allocated on
        # that path too, so charge them rather than argue about the margin.
        single, split = self._rate("q8_0", False), self._rate("q8_0", True)
        assert single < split <= single * 1.02

    def test_split_delta_scales_with_ubatch(self):
        assert self._rate("f16", True, ub = 1024) - self._rate("f16", False, ub = 1024) == (
            pytest.approx(2 * (self._rate("f16", True) - self._rate("f16", False)), rel = 1e-9)
        )

    def test_single_device_rates_unchanged(self):
        # No context is lost on a single GPU: the flag defaults off.
        for ct in ("f16", "q8_0"):
            assert self._rate(ct, False) == pytest.approx(
                (
                    LlamaCppBackend._INKLING_CTX_COMPUTE_DENSE_BYTES_PER_TOK
                    if ct == "q8_0"
                    else LlamaCppBackend._INKLING_CTX_COMPUTE_BYTES_PER_TOK
                ),
                rel = 1e-6,
            )


class TestLayerSplitWiring:
    """``_cc_bytes`` in ``load_model`` learns the device count, so it is the one place
    that can set ``layer_split``. Source-level: the closure needs a real load."""

    def _cc_bytes_source(self):
        import inspect

        src = inspect.getsource(LlamaCppBackend.load_model).splitlines()
        start = next(i for i, l in enumerate(src) if "def _cc_bytes(" in l)
        indent = len(src[start]) - len(src[start].lstrip())
        body = [src[start]]
        for line in src[start + 1 :]:
            # A one-per-line signature closes with ") -> int:" at the def's indent.
            closes_signature = line.lstrip().startswith(")")
            if line.strip() and len(line) - len(line.lstrip()) <= indent and not closes_signature:
                break
            body.append(line)
        return "\n".join(body)

    def test_forwards_layer_split_from_device_count(self):
        assert "layer_split = n_gpus > 1" in self._cc_bytes_source()

    def test_still_scales_by_device_count(self):
        # Per device on top of the replication, not instead of it: n x the split rate.
        assert "max(1, n_gpus) * self._compute_buffer_ctx_bytes" in self._cc_bytes_source()


class TestPipelineParallelPredicate:
    """The 4x step is llama.cpp's pipeline parallelism (ggml n_copies == 4), which
    llama-context.cpp declines unless the split mode is layer, KV offload is on and the
    tensor-override list is empty; charging it then wastes context. Causal check: on 2
    GPUs a -ot matching nothing changes no placement yet takes the rate 8.00 -> 2.00."""

    def _off(
        self,
        args = None,
        env = None,
        n_layers = None,
    ):
        from core.inference.llama_cpp import _pipeline_parallel_disabled_by_args
        return _pipeline_parallel_disabled_by_args(args, env = env or {}, n_layers = n_layers)

    def test_plain_launch_keeps_the_step(self):
        assert self._off([]) is False
        assert self._off(None) is False
        assert self._off(["-c", "131072", "--parallel", "4"]) is False

    @pytest.mark.parametrize("flag", ["-ot", "--override-tensor"])
    def test_any_tensor_override_disables(self, flag):
        # Even a pattern matching nothing: has_tensor_overrides only checks non-empty.
        assert self._off([flag, "zzz_matches_nothing=CUDA0"]) is True

    @pytest.mark.parametrize("flag", ["-nkvo", "--no-kv-offload"])
    def test_kv_offload_off_disables(self, flag):
        assert self._off([flag]) is True

    def test_env_tensor_override_disables(self):
        assert self._off([], env = {"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CPU"}) is True
        assert self._off([], env = {"LLAMA_ARG_OVERRIDE_TENSOR": ""}) is False

    @pytest.mark.parametrize("val", ["off", "0", "false", "disabled"])
    def test_env_kv_offload_off_disables(self, val):
        assert self._off([], env = {"LLAMA_ARG_KV_OFFLOAD": val}) is True

    def test_env_kv_offload_on_keeps_the_step(self):
        assert self._off([], env = {"LLAMA_ARG_KV_OFFLOAD": "1"}) is False

    def test_unrelated_flags_do_not_disable(self):
        assert self._off(["--kv-unified", "-fa", "on", "-ngl", "-1"]) is False

    @pytest.mark.parametrize(
        "flag", ["-otd", "--override-tensor-draft", "--spec-draft-override-tensor"]
    )
    def test_draft_override_does_not_disable(self, flag):
        # -otd targets the draft model, not the main model's tensor_buft_overrides.
        assert self._off([flag, "exps=CPU"]) is False

    # -- KV offload is last-wins across env then CLI (arg.cpp parses env first) --

    @pytest.mark.parametrize("flag", ["-kvo", "--kv-offload"])
    def test_cli_kv_offload_reenable_beats_a_false_env(self, flag):
        # The positive form exists, so this launch pipelines: 1x here would OOM it.
        assert self._off([flag], env = {"LLAMA_ARG_KV_OFFLOAD": "0"}) is False

    def test_last_kv_offload_flag_wins(self):
        assert self._off(["-kvo", "-nkvo"]) is True
        assert self._off(["-nkvo", "-kvo"]) is False

    def test_kv_offload_env_junk_value_keeps_the_default(self):
        assert self._off([], env = {"LLAMA_ARG_KV_OFFLOAD": "maybe"}) is False

    # -- pipeline parallelism requires LLAMA_SPLIT_MODE_LAYER --

    @pytest.mark.parametrize("mode", ["none", "row", "NONE", " row "])
    def test_non_layer_split_mode_disables(self, mode):
        assert self._off(["-sm", mode]) is True
        assert self._off([f"--split-mode={mode}"]) is True

    def test_explicit_layer_split_mode_keeps_the_step(self):
        assert self._off(["-sm", "layer"]) is False
        assert self._off(["-sm", "row", "-sm", "layer"]) is False  # last-wins

    def test_tensor_split_mode_keeps_the_step(self):
        # The layer branch is an elif on tensor_parallel, so it is only reached
        # after a downgrade -- which strips -sm and leaves the child pipelined.
        assert self._off(["-sm", "tensor"]) is False

    def test_layer_branch_is_only_reached_after_the_flag_is_stripped(self):
        # Guards the assumption above.
        import inspect

        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "iftensor_parallelandtp_gpus:" in compact
        assert "extra_args=strip_split_mode_only(" in compact
        assert "elifgpusandself._can_estimate_kv()andeffective_ctx>0:" in compact

    def test_env_split_mode_is_ignored(self):
        # load_model pops a non-layer inherited LLAMA_ARG_SPLIT_MODE on the layer path,
        # so the child always runs -sm layer; honoring it would reserve 1x for a split.
        assert self._off([], env = {"LLAMA_ARG_SPLIT_MODE": "row"}) is False

    def test_layer_path_scrubs_a_non_layer_split_mode_env(self):
        # Guards the assumption the test above rests on.
        import inspect

        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert 'if_inherited_smand_inherited_sm!="layer":' in compact
        assert 'env.pop("LLAMA_ARG_SPLIT_MODE",None)' in compact

    # -- -cmoe / -ncmoe set tensor_buft_overrides exactly like -ot --

    @pytest.mark.parametrize("flag", ["-cmoe", "--cpu-moe"])
    def test_cpu_moe_disables(self, flag):
        assert self._off([flag]) is True

    @pytest.mark.parametrize("flag", ["-ncmoe", "--n-cpu-moe"])
    def test_n_cpu_moe_disables(self, flag):
        assert self._off([flag, "8"]) is True
        assert self._off([f"{flag}=8"]) is True

    @pytest.mark.parametrize("flag", ["-ncmoe", "--n-cpu-moe"])
    def test_n_cpu_moe_zero_keeps_the_step(self, flag):
        # The handler loops N times, so 0 pushes no override at all.
        assert self._off([flag, "0"]) is False

    def test_env_cpu_moe_disables(self):
        assert self._off([], env = {"LLAMA_ARG_CPU_MOE": "1"}) is True
        # handler_void only fires on a truthy env value.
        assert self._off([], env = {"LLAMA_ARG_CPU_MOE": "0"}) is False
        assert self._off([], env = {"LLAMA_ARG_CPU_MOE": ""}) is False

    def test_env_n_cpu_moe_disables(self):
        assert self._off([], env = {"LLAMA_ARG_N_CPU_MOE": "4"}) is True
        assert self._off([], env = {"LLAMA_ARG_N_CPU_MOE": "0"}) is False
        assert self._off([], env = {"LLAMA_ARG_N_CPU_MOE": "not-a-number"}) is False

    # -- a finite -ngl override loads a layer prefix, so pipelining is off --

    @pytest.mark.parametrize("flag", ["-ngl", "--gpu-layers", "--n-gpu-layers"])
    def test_finite_gpu_layers_below_the_count_disables(self, flag):
        # User extras land after Unsloth's -ngl -1, so this last-wins.
        assert self._off([flag, "1"], n_layers = 93) is True
        assert self._off([f"{flag}=1"], n_layers = 93) is True

    def test_all_layers_keeps_the_step(self):
        # n_gpu_layers() is n_layer_all + 1 for any negative value.
        assert self._off(["-ngl", "-1"], n_layers = 93) is False

    def test_gpu_layers_above_the_count_keeps_the_step(self):
        # 999 > n_layer_all, so llama.cpp still pipelines.
        assert self._off(["-ngl", "999"], n_layers = 93) is False

    def test_gpu_layers_at_the_boundary(self):
        # Pipelining needs n_gpu_layers > n_layer_all, so equal is off; one above
        # keeps the step because block_count can undercount n_layer_all.
        assert self._off(["-ngl", "93"], n_layers = 93) is True
        assert self._off(["-ngl", "94"], n_layers = 93) is False

    def test_zero_gpu_layers_disables(self):
        assert self._off(["-ngl", "0"], n_layers = 93) is True

    def test_unknown_layer_count_keeps_the_step(self):
        assert self._off(["-ngl", "1"]) is False
        assert self._off(["-ngl", "1"], n_layers = 0) is False

    def test_last_gpu_layers_flag_wins(self):
        assert self._off(["-ngl", "1", "--gpu-layers", "-1"], n_layers = 93) is False
        assert self._off(["-ngl", "-1", "--gpu-layers", "1"], n_layers = 93) is True

    def test_malformed_gpu_layers_keeps_the_step(self):
        # validate_extra_args rejects these upstream; ambiguous here means keep.
        assert self._off(["-ngl", "abc"], n_layers = 93) is False
        assert self._off(["-ngl", "-2"], n_layers = 93) is False
        assert self._off(["-ngl"], n_layers = 93) is False

    def test_wired_into_the_fit(self):
        # The flag has to reach _cc_bytes, else the predicate is dead code.
        import inspect

        src = inspect.getsource(LlamaCppBackend.load_model)
        compact = "".join(src.split())
        assert "_pipeline_parallel_disabled_by_args(extra_args,n_layers=self._n_layers)" in compact
        assert "layer_split = n_gpus > 1 and not _pipeline_parallel_off" in src
        # The count is only real if the GGUF header was parsed first.
        assert src.index("_read_gguf_metadata(model_path)") < src.index(
            "_pipeline_parallel_disabled_by_args("
        )


class TestPerDeviceSplitReserve:
    """The auto-context loop admits a GPU subset on the POOLED budget, but the per-device
    reserve (flat layer overhead + this PR's enlarged context-compute copy) is replicated
    on every card: in the sum a roomy card's spare VRAM covers a nearly full card's copy,
    and the small card OOMs at launch. Exposed when the loop cannot start at one GPU
    (``_auto_min_gpus >= 2``, set by every tensor -> layer downgrade); starting at one,
    the pooled test charges n copies where the per-card test charges one, so size n is
    reached only after n-1 failed, which bounds the smallest card below by the reserve -
    the check is then provably redundant, homogeneous or not."""

    _OH = LlamaCppBackend._PIPELINE_PER_DEVICE_OVERHEAD_MIB * MIB
    _UB = 2048
    # 48 GB / 24 GB cards, the small one mostly occupied. Its usable budget still
    # clears the flat overhead alone, so _auto_min_gpus keeps counting it.
    _HETEROGENEOUS = ([(0, 40_000), (1, 2_500)], {0: 49_152, 1: 24_576})
    _HOMOGENEOUS = ([(0, 40_000), (1, 40_000)], {0: 49_152, 1: 49_152})

    def _fit_backend(self, kv_per_tok = 20_000):
        b = _backend(embd = 8192)
        b._can_estimate_kv = lambda: True
        b._estimate_kv_cache_bytes = lambda ctx, ct = None, **kw: max(0, ctx) * kv_per_tok
        return b

    def _drive(
        self,
        b,
        gpus,
        totals,
        model_mib,
        native_ctx,
        min_gpus = 2,
        enforce = True,
        cap = True,
    ):
        """Mirror of load_model's auto-context subset loop over the production fit
        and reserve helpers. Returns (gpu_indices, chosen_ctx)."""
        frac = LlamaCppBackend._GPU_PIN_VRAM_FRACTION
        model_bytes = model_mib * MIB

        def usable(g):
            return g[1] - (1.0 - frac) * totals[g[0]]

        ranked = sorted(gpus, key = usable, reverse = True)
        for n in range(min_gpus, len(ranked) + 1):
            subset = ranked[:n]
            pool = sum(max(0.0, usable(g)) for g in subset)
            ms = model_bytes + (n - 1) * self._OH
            cc = lambda c, n = n: (
                n * b._compute_buffer_ctx_bytes(c, self._UB, "f16", layer_split = n > 1)
            )
            capped = b._fit_context_to_vram(
                native_ctx,
                pool,
                ms,
                "f16",
                n_ubatch = self._UB,
                compute_ctx_bytes_fn = cc,
                budget_frac = 1.0,
                total_mib = None,
            )
            if (ms + b._estimate_kv_cache_bytes(capped) + cc(capped)) / MIB > pool:
                continue
            usable_mib = [usable(g) for g in subset]
            reserve_at = lambda c, n = n: (self._OH if n > 1 else 0) + cc(c, n) // n
            if enforce and not LlamaCppBackend._every_gpu_holds_reserve(
                usable_mib, reserve_at(capped)
            ):
                if not cap:
                    continue
                capped = LlamaCppBackend._cap_ctx_to_per_device_reserve(
                    capped, usable_mib, reserve_at
                )
                if capped <= 0:
                    continue
                if (ms + b._estimate_kv_cache_bytes(capped) + cc(capped)) / MIB > pool:
                    continue
            return sorted(idx for idx, _ in subset), capped
        return None, 0

    def _drive_reduced(
        self,
        b,
        gpus,
        totals,
        model_mib,
        min_gpus = 2,
        enforce = True,
    ):
        """Mirror of the Auto offload loop the native loop falls through to. Same
        pooled admission, so it needs the same per-device gate.

        Driven at the fit search floor rather than at ``_AUTO_OFFLOAD_CTX``: what
        this exercises is the per-device reserve gate, and the floor is the lowest
        context the loop above can still be admitted at, so it is the value that
        makes the gate decide anything.
        """
        frac = LlamaCppBackend._GPU_PIN_VRAM_FRACTION
        ctx = _FIT_MIN_CTX

        def usable(g):
            return g[1] - (1.0 - frac) * totals[g[0]]

        ranked = sorted(gpus, key = usable, reverse = True)
        for n in range(min_gpus, len(ranked) + 1):
            subset = ranked[:n]
            pool = sum(max(0.0, usable(g)) for g in subset)
            cc = n * b._compute_buffer_ctx_bytes(ctx, self._UB, "f16", layer_split = n > 1)
            ms = model_mib * MIB + (n - 1) * self._OH
            if (ms + b._estimate_kv_cache_bytes(ctx) + cc) / MIB > pool:
                continue
            if enforce and not LlamaCppBackend._every_gpu_holds_reserve(
                (usable(g) for g in subset),
                (self._OH if n > 1 else 0) + cc // n,
            ):
                continue
            return sorted(idx for idx, _ in subset), ctx
        return None, 0

    def _card_at_floor_reserve(self, b, total_mib, margin_mib):
        """Free VRAM leaving card 1 exactly ``margin_mib`` from the reserve it
        replicates AT THE FIT FLOOR, which is the context ``_drive_reduced`` runs.

        Derived rather than written out: the whole point of these two cases is a
        card sized one MiB either side of that reserve, so the number has to follow
        the floor. Spelled 4096 it silently stopped straddling anything when the
        floor moved to 8192 -- the card was built against reserve(4096) == 1120 MiB
        while the driver charged reserve(8192) == 1216.
        """
        reserve_mib = (
            self._OH + b._compute_buffer_ctx_bytes(_FIT_MIN_CTX, self._UB, "f16", layer_split = True)
        ) / MIB
        return round(reserve_mib + margin_mib + 0.03 * total_mib)

    def test_reduced_context_fallback_enforces_the_same_reserve(self):
        # Card 1 sized one MiB under the reserve it replicates at the floor: the
        # pooled budget still admits the pair, so dropping to the floor pinned a
        # card that OOMs.
        b = self._fit_backend()
        totals = {0: 49_152, 1: 24_576}
        gpus = [(0, 40_000), (1, self._card_at_floor_reserve(b, totals[1], -1))]
        assert self._drive_reduced(b, gpus, totals, 20_480, enforce = False) == (
            [0, 1],
            _FIT_MIN_CTX,
        )
        assert self._drive_reduced(b, gpus, totals, 20_480) == (None, 0)

    def test_reduced_context_fallback_keeps_a_card_that_holds_it(self):
        b = self._fit_backend()
        totals = {0: 49_152, 1: 24_576}
        gpus = [(0, 40_000), (1, self._card_at_floor_reserve(b, totals[1], +1))]
        assert self._drive_reduced(b, gpus, totals, 20_480) == ([0, 1], _FIT_MIN_CTX)

    def test_pooled_budget_hides_the_small_cards_shortfall(self):
        # Pre-fix: the pair is admitted at native context even though card 1 has
        # ~1.7 GiB usable and owes 1 GiB overhead + 6 GiB of replicated KQ mask.
        b = self._fit_backend()
        gpus, totals = self._HETEROGENEOUS
        gpu_indices, ctx = self._drive(b, gpus, totals, 20_480, 262144, enforce = False)
        assert gpu_indices == [0, 1] and ctx == 262144
        reserve_mib = (
            self._OH + b._compute_buffer_ctx_bytes(ctx, self._UB, "f16", layer_split = True)
        ) / MIB
        card1_usable = 2_500 - 0.03 * 24_576
        assert card1_usable < reserve_mib  # would OOM card 1 at load

    def test_subset_is_capped_not_rejected_when_a_card_cannot_hold_its_reserve(self):
        # 1762.72 MiB usable on card 1 holds 30720, not the pooled 262144. Rejecting
        # the subset instead drops auto to the 4096 fallback for no reason.
        b = self._fit_backend()
        gpus, totals = self._HETEROGENEOUS
        assert self._drive(b, gpus, totals, 20_480, 262144, cap = False) == (None, 0)
        assert self._drive(b, gpus, totals, 20_480, 262144) == ([0, 1], 30_720)

    def test_homogeneous_gpus_are_unaffected(self):
        b = self._fit_backend()
        gpus, totals = self._HOMOGENEOUS
        with_gate = self._drive(b, gpus, totals, 20_480, 262144)
        without = self._drive(b, gpus, totals, 20_480, 262144, enforce = False)
        assert with_gate == without == ([0, 1], 262144)

    @pytest.mark.parametrize("model_mib", [8_192, 20_480, 30_720])
    def test_no_op_when_the_loop_may_start_at_one_gpu(self, model_mib):
        # _auto_min_gpus == 1: the n-1 subset having failed already bounds the
        # smallest card below by the reserve, so the gate changes nothing.
        b = self._fit_backend()
        for gpus, totals in (self._HETEROGENEOUS, self._HOMOGENEOUS):
            with_gate = self._drive(b, gpus, totals, model_mib, 262144, min_gpus = 1)
            without = self._drive(b, gpus, totals, model_mib, 262144, min_gpus = 1, enforce = False)
            assert with_gate == without
            assert with_gate[0] is not None

    def test_reserve_check_rejects_only_the_short_card(self):
        reserve = 3072 * MIB  # bytes in, MiB compared
        assert LlamaCppBackend._every_gpu_holds_reserve([4000.0, 3072.0], reserve) is True
        assert LlamaCppBackend._every_gpu_holds_reserve([40000.0, 3071.0], reserve) is False
        assert LlamaCppBackend._every_gpu_holds_reserve([-10.0], reserve) is False
        assert LlamaCppBackend._every_gpu_holds_reserve([], reserve) is False

    def test_wired_into_the_auto_context_loop(self):
        import inspect

        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        # Native-context loop, the reduced-to-4096 fallback below it, and the
        # Auto drafter-drop probe above them, which caps to the same reserve so
        # it cannot price a drafter at a context the weakest card never holds.
        assert compact.count("ifnotself._every_gpu_holds_reserve(") == 3
        # Gated on the chosen context, and only reachable after the pooled test.
        assert "_usable_mib=[_gpu_usable(g,pin_fraction)forginsubset]" in compact
        assert "(_gpu_usable(g,pin_fraction)forginsubset)," in compact
        assert "+_cc_bytes(c,n)//n)" in compact
        assert "+_cc_bytes(effective_ctx,n_gpus)//n_gpus," in compact
        # The cap runs only on gate failure, and the pooled price is redone after it.
        assert (
            compact.count(
                "capped=self._cap_ctx_to_per_device_reserve("
                "capped,_usable_mib,_reserve_at)ifcapped<=0:continue"
            )
            == 1
        )
        assert "kv=_kv_bytes(capped)footprint_mib=(_ms+kv+_mtp_bytes(capped)" in compact


class TestPerDeviceReserveCap:
    """Rejecting a subset outright costs context the smallest card could have held.
    Reuses the gate class's fixtures; the cap only changes what "reject" means."""

    _OH = TestPerDeviceSplitReserve._OH
    _UB = TestPerDeviceSplitReserve._UB
    _HETEROGENEOUS = TestPerDeviceSplitReserve._HETEROGENEOUS
    _HOMOGENEOUS = TestPerDeviceSplitReserve._HOMOGENEOUS
    _fit_backend = TestPerDeviceSplitReserve._fit_backend
    _drive = TestPerDeviceSplitReserve._drive

    def test_cap_is_exact_at_the_256_boundary(self):
        b = self._fit_backend()
        usable = 2_500 - 0.03 * 24_576  # 1762.72 MiB
        reserve = lambda c: (
            (self._OH + b._compute_buffer_ctx_bytes(c, self._UB, "f16", layer_split = True)) / MIB
        )
        # 1024 MiB overhead + 256 MiB of hidden-state copies + 4 masks at 1/64 MiB a cell.
        assert reserve(30_720) == 1760.0 <= usable
        assert reserve(30_976) == 1764.0 > usable

    def test_capped_context_still_fits_the_pooled_budget(self):
        b = self._fit_backend()
        gpus, totals = self._HETEROGENEOUS
        idx, ctx = self._drive(b, gpus, totals, 20_480, 262144)
        assert idx == [0, 1] and ctx == 30_720
        pool = sum(g[1] - 0.03 * totals[g[0]] for g in gpus)
        cc = 2 * b._compute_buffer_ctx_bytes(ctx, self._UB, "f16", layer_split = True)
        footprint = (20_480 * MIB + self._OH + b._estimate_kv_cache_bytes(ctx) + cc) / MIB
        assert footprint <= pool

    def test_homogeneous_gpus_are_unaffected_by_the_cap(self):
        b = self._fit_backend()
        gpus, totals = self._HOMOGENEOUS
        assert self._drive(b, gpus, totals, 20_480, 262144) == ([0, 1], 262144)

    _card_at_floor_reserve = TestPerDeviceSplitReserve._card_at_floor_reserve

    def test_cap_floors_at_the_fit_minimum_and_still_rejects_below_it(self):
        # Card 1 one MiB under reserve(_FIT_MIN_CTX): the cap has nothing to salvage,
        # so it returns 0 rather than handing back the floor it could not price.
        b = self._fit_backend()
        totals = {0: 49_152, 1: 24_576}
        gpus = [(0, 40_000), (1, self._card_at_floor_reserve(b, totals[1], -1))]
        assert self._drive(b, gpus, totals, 20_480, 262144) == (None, 0)

    def test_cap_keeps_a_card_that_holds_exactly_the_fit_minimum(self):
        b = self._fit_backend()
        totals = {0: 49_152, 1: 24_576}
        gpus = [(0, 40_000), (1, self._card_at_floor_reserve(b, totals[1], +1))]
        assert self._drive(b, gpus, totals, 20_480, 262144) == ([0, 1], _FIT_MIN_CTX)

    def test_flat_arch_term_is_not_rate_inverted(self):
        # deepseek4 carries a flat indexer term, so inverting the per-token rate
        # answers 43341, whose reserve is 6048 MiB on a 4000 MiB card.
        b = _backend(embd = 7168, arch = "deepseek4")
        reserve = lambda c: self._OH + b._compute_buffer_ctx_bytes(c, 512, "q8_0", layer_split = True)
        cap = LlamaCppBackend._cap_ctx_to_per_device_reserve(200_000, [4000.0], reserve)
        assert cap == 13_312
        assert reserve(cap) / MIB <= 4000.0 < reserve(cap + 256) / MIB
        assert reserve(43_341) / MIB > 4000.0

    def test_flat_arch_below_the_floor_is_infeasible(self):
        b = _backend(embd = 7168, arch = "deepseek4")
        reserve = lambda c: self._OH + b._compute_buffer_ctx_bytes(c, 512, "q8_0", layer_split = True)
        assert LlamaCppBackend._cap_ctx_to_per_device_reserve(200_000, [2500.0], reserve) == 0

    @pytest.mark.parametrize("model_mib", [8_192, 16_384, 20_480, 30_720])
    def test_capping_never_loses_to_continuing_with_more_gpus(self, model_mib):
        # A third small card cannot rescue the subset: it stays in the ranking, so
        # the reserve there is no smaller. Capping at 2 beats falling through.
        b = self._fit_backend()
        gpus = [(0, 40_000), (1, 2_500), (2, 2_400)]
        totals = {0: 49_152, 1: 24_576, 2: 24_576}
        assert self._drive(b, gpus, totals, model_mib, 262144, cap = False) == (None, 0)
        idx, ctx = self._drive(b, gpus, totals, model_mib, 262144)
        assert idx == [0, 1] and ctx == 30_720

    def test_helper_edges(self):
        cap = LlamaCppBackend._cap_ctx_to_per_device_reserve
        assert cap(262144, [], lambda c: 0) == 0
        assert cap(1024, [4000.0], lambda c: 0) == 0  # ctx below the 4096 floor
        assert cap(262144, [4000.0], lambda c: 0) == 262144  # free reserve, no cap
        linear = lambda c: c * 16384  # 16 KiB per ctx token, so 2000 MiB buys 128000
        best = cap(262144, [2000.0], linear)
        assert best == 128_000 and best % 256 == 0
        assert linear(best) / MIB <= 2000.0 < linear(best + 256) / MIB


class TestSplitRateRecheckAfterSelection:
    """``_select_gpus`` derives the device count FROM the footprint, so its callers can
    only price the context-compute buffer at the single-device rate. A multi-GPU answer
    then has to be re-priced at the split rate before it is pinned with
    ``use_fit = False``, or a high-context explicit request OOMs at launch. Synthetic
    VRAM maps over the production helper: 24 GB cards at 0.97 (usable 23838 MiB each),
    ctx 1M at ub 512 f16 -> 1536 MiB of compute per device single-device, 6144 MiB split,
    so each card in a split owes 4608 MiB more than the first pass charged it."""

    _OH = LlamaCppBackend._PIPELINE_PER_DEVICE_OVERHEAD_MIB * MIB
    _UB = 512
    _CTX = 1048576
    _FRAC = LlamaCppBackend._GPU_PIN_VRAM_FRACTION
    _CARD = 24_576

    def _cards(self, n):
        return [(i, self._CARD) for i in range(n)], {i: self._CARD for i in range(n)}

    def _pin(
        self,
        total_mib,
        n_cards,
        recheck = True,
        min_gpus = 1,
        ctx = None,
    ):
        """Mirror of the explicit-context branch: total_mib is its weights + KV + MTP
        footprint, to which each path adds one single-device compute copy."""
        b = _backend(embd = 4096)
        ctx = ctx or self._CTX
        cc1 = b._compute_buffer_ctx_bytes(ctx, self._UB, "f16")
        cc_split = b._compute_buffer_ctx_bytes(ctx, self._UB, "f16", layer_split = True)
        gpus, totals = self._cards(n_cards)
        return LlamaCppBackend._select_gpus_split_aware(
            total_mib * MIB + cc1,
            gpus,
            usable_fraction = self._FRAC,
            total_by_idx = totals,
            per_device_overhead_bytes = self._OH + cc1,
            min_gpus = min_gpus,
            split_extra_bytes = (cc_split - cc1) if recheck else 0,
        )

    def test_pre_fix_pinned_a_pair_that_cannot_hold_the_split_rate(self):
        # The bug, over the plain selector this branch used to call: the pair needs
        # 44096 MiB of its 47677 MiB pool at the single-device rate, but 53312 at the
        # split rate -- pinned ~5.5 GiB short, with no --fit fallback after -ngl -1.
        b = _backend(embd = 4096)
        cc1 = b._compute_buffer_ctx_bytes(self._CTX, self._UB, "f16")
        ccs = b._compute_buffer_ctx_bytes(self._CTX, self._UB, "f16", layer_split = True)
        gpus, totals = self._cards(2)
        assert LlamaCppBackend._select_gpus(
            40_000 * MIB + cc1,
            gpus,
            usable_fraction = self._FRAC,
            total_by_idx = totals,
            per_device_overhead_bytes = self._OH + cc1,
        ) == ([0, 1], False)
        pool = 2 * (self._CARD - (1.0 - self._FRAC) * self._CARD)
        assert (40_000 * MIB + ccs + self._OH + ccs) / MIB > pool

    def test_recheck_falls_back_to_fit_when_no_subset_holds_it(self):
        # Honest failure: --fit on degrades to CPU offload, matching this branch's
        # documented behaviour, instead of pinning a launch that OOMs.
        assert self._pin(40_000, 2) == (None, True)

    def test_recheck_widens_the_subset_when_a_card_is_spare(self):
        # Three cards: the first pass still answers 2, the re-check takes all 3.
        assert self._pin(40_000, 3, recheck = False) == ([0, 1], False)
        assert self._pin(40_000, 3) == ([0, 1, 2], False)

    def test_single_gpu_pin_is_untouched(self):
        assert self._pin(20_000, 2) == ([0], False)
        assert self._pin(20_000, 2, recheck = False) == ([0], False)

    def test_equal_cards_never_collapse_to_one_gpu(self):
        # Every card clears the enlarged overhead, so the retry keeps min_gpus.
        for total in range(24_000, 46_000, 2_000):
            gi, use_fit = self._pin(total, 4)
            assert use_fit or (gi is not None and len(gi) >= 2)

    def test_collapse_to_one_gpu_is_repriced_without_the_delta(self):
        # Unequal cards: only the big one clears overhead + delta, so _select_gpus cuts
        # its usable-card count to one. A lone card is not a split and pays no delta,
        # so charging it there sent a load that fits alone to --fit on (CPU offload).
        gpus = [(0, 16 * 1024), (1, 3 * 1024)]
        kw = dict(usable_fraction = 1.0, per_device_overhead_bytes = int(2.5 * GIB))
        assert LlamaCppBackend._select_gpus(int(14 * GIB), gpus, min_gpus = 2, **kw) == (
            [0, 1],
            False,
        )
        assert LlamaCppBackend._select_gpus_split_aware(
            int(14 * GIB), gpus, min_gpus = 2, split_extra_bytes = int(4.5 * GIB), **kw
        ) == ([0], False)
        # Exactly the plain single-device answer, not a relaxed split.
        assert LlamaCppBackend._select_gpus(int(14 * GIB), gpus, min_gpus = 1, **kw) == (
            [0],
            False,
        )

    def test_reprice_still_reports_fit_when_no_single_card_holds_it(self):
        # 30 GiB over two 20 GiB cards: the split no longer fits and neither does one
        # card, so the honest answer stays --fit on.
        assert LlamaCppBackend._select_gpus_split_aware(
            int(30 * GIB),
            [(0, 20 * 1024), (1, 20 * 1024)],
            usable_fraction = 1.0,
            per_device_overhead_bytes = int(1 * GIB),
            min_gpus = 2,
            split_extra_bytes = int(15 * GIB),
        ) == (None, True)

    def test_zero_step_reduces_to_plain_selection(self):
        # llama.cpp declining pipeline parallelism makes the step 0 (_cc_split_extra
        # reads the same layer_split gate), and the helper is then a pass-through.
        b = _backend(embd = 4096)
        cc1 = b._compute_buffer_ctx_bytes(self._CTX, self._UB, "f16")
        gpus, totals = self._cards(3)
        for total_mib in (20_000, 40_000, 90_000):
            assert self._pin(total_mib, 3, recheck = False) == LlamaCppBackend._select_gpus(
                total_mib * MIB + cc1,
                gpus,
                usable_fraction = self._FRAC,
                total_by_idx = totals,
                per_device_overhead_bytes = self._OH + cc1,
            )

    def test_small_context_is_unaffected(self):
        # 4096 ctx: the 18 MiB of extra masks changes no decision.
        assert self._pin(40_000, 2, ctx = 4096) == self._pin(40_000, 2, recheck = False, ctx = 4096)

    def test_wired_into_every_call_site(self):
        import ast
        import inspect
        import re
        import textwrap

        source = inspect.getsource(LlamaCppBackend.load_model)
        load = "".join(source.split())
        # Read the call sites out of the parse tree. Counting spellings said the same
        # thing while it lasted, but it also reddened on a rename that changed nothing
        # about the rule: the three sites price at three different contexts and the
        # names they use for them are not the contract.
        wired = [
            ast.unparse(keyword.value)
            for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg in ("split_extra_bytes", "split_extra_for_slots")
        ]
        # Projector floor pin, explicit-context pin, reduced-slot retry, per-candidate
        # re-fit, and the overcommit notice's q8_0 what-if, which prices the requested
        # context at q8_0 KV. A sixth has to come here and say which context it prices at.
        #
        # Four, not the three the counting version asserted. It counted two spellings,
        # `_cc_split_extra(effective_ctx)` and `_cc_split_extra(ctx),`, and the
        # projector-floor site spells its context `_mm_floor_ctx`, so it was invisible
        # to the check that claimed to cover every call site. It has been wired
        # correctly the whole time; nothing was holding it there.
        assert len(wired) == 5, wired
        # Each passes the step at a context of its own, so none is exempt and none
        # hardcodes one: `_cc_split_extra(4096)` would not match.
        # The reduced-slot search re-prices the step for each candidate slot count.
        for expression in wired:
            assert re.fullmatch(
                r"_cc_split_extra\(\w+\)|lambda s: _cc_split_extra\(\w+, s\)"
                r"|_cc_split_extra\(\w+, cache_type='q8_0'\)",
                expression,
            ), expression
        assert "gpu_indices,use_fit=self._select_gpus_split_aware(" in load
        # The step rides _cc_bytes' pipelining gate, so it is 0 when llama.cpp declines.
        assert (
            "returnmax(0,_cc_bytes(ctx,2,slots,cache_type)//2-_cc_bytes(ctx,1,slots,cache_type),)"
            in load
        )
        slots = "".join(inspect.getsource(LlamaCppBackend._slots_that_fit_on_gpu).split())
        assert "self._select_gpus_split_aware(" in slots
        assert (
            "split_extra_bytes=(split_extra_for_slots(slots)ifsplit_extra_for_slotselsesplit_extra_bytes),"
            in slots
        )


# ── The scratch rate keys off the LIGHTER axis ───────────────────────────────
#
# Since ggml-org/llama.cpp#23792 Unsloth no longer rewrites the requested type for the
# tensor attempt, so an asymmetric pair is reachable in the one mode with no --fit
# valve. The budget resolves ONE scalar, the heavier axis, for KV bytes; handing that
# to _compute_buffer_ctx_bytes prices a q4_0 K cache as if nothing were quantized,
# because the dequant branch gates on bytes/elem < 2.0.


class TestScratchTakesTheLighterAxis:
    """_planned_scratch_cache_type is the seam that keeps the two terms honest:
    heavier axis for KV bytes, lighter axis for the dequant scratch."""

    @pytest.mark.parametrize(
        "k,v,expected",
        [
            ("f16", "f16", "f16"),
            ("q8_0", "q8_0", "q8_0"),
            ("q4_0", "f16", "q4_0"),  # the shape the heavier scalar hides
            ("f16", "q4_0", "q4_0"),  # and with the axes swapped
            ("q8_0", "q4_0", "q4_0"),
            ("f32", "q8_0", "q8_0"),
        ],
    )
    def test_it_picks_the_quantized_axis_whichever_side_it_is_on(self, k, v, expected):
        extras = ["--cache-type-k", k, "--cache-type-v", v]
        assert _planned_scratch_cache_type(None, extras) == expected
        # And the budget still takes the heavier one, so the two disagree exactly
        # when they should.
        heavier = max(_planned_main_cache_types(None, extras), key = _kv_bytes_per_elem)
        assert (heavier != expected) == (_kv_bytes_per_elem(k) != _kv_bytes_per_elem(v))

    def test_a_managed_symmetric_request_leaves_both_terms_equal(self):
        """Unsloth emits one type on both axes, so nothing changes for the common
        case -- this fix must not move the fit for a plain q8_0 load."""
        for kv in ("f16", "q8_0", "q4_0", "iq4_nl"):
            assert _planned_scratch_cache_type(kv, None) == kv

    def test_extras_beat_the_managed_field_per_axis(self):
        """Extras are appended last and win per axis, so the scratch must follow
        them, not the field the UI sent."""
        assert _planned_scratch_cache_type("f16", ["--cache-type-k", "q4_0"]) == "q4_0"

    def test_it_reads_the_inherited_env_when_nothing_else_sets_a_type(self):
        env = {"LLAMA_ARG_CACHE_TYPE_K": "q4_0", "LLAMA_ARG_CACHE_TYPE_V": "f16"}
        assert _planned_scratch_cache_type(None, None, env) == "q4_0"

    @pytest.mark.parametrize("k,v", [("q4_0", "f16"), ("f16", "q4_0")])
    def test_the_asymmetric_pair_selects_the_dequant_rate(self, k, v):
        """The point of the seam: at the same context the asymmetric pair must
        cost what the quantized axis really allocates, not the KQ-mask floor."""
        b = _backend()
        extras = ["--cache-type-k", k, "--cache-type-v", v]
        heavier = max(_planned_main_cache_types(None, extras), key = _kv_bytes_per_elem)
        lighter = _planned_scratch_cache_type(None, extras)

        heavy_rate = b._compute_buffer_ctx_bytes(131_072, 2048, heavier)
        light_rate = b._compute_buffer_ctx_bytes(131_072, 2048, lighter)

        assert light_rate > heavy_rate, (light_rate, heavy_rate)
        # Same answer as a symmetric quantized cache: the scratch is per-tensor
        # work on the quantized axis, not something the f16 axis discounts.
        assert light_rate == b._compute_buffer_ctx_bytes(131_072, 2048, "q4_0")


class TestTensorFitPricesTheQuantizedAxis:
    """End to end through the planner: the advertised context must not exceed
    what an honest per-axis price allows. Tensor mode has no --fit valve, so an
    optimistic context OOMs at startup instead of spilling."""

    @staticmethod
    def _plan(
        b,
        cache_type,
        scratch_type,
        ub,
        ngpu = 4,
        per_gpu = 48_000,
    ):
        return b._plan_tensor_parallel(
            gpus = [(i, per_gpu) for i in range(ngpu)],
            model_size = 60 * 1024**3,
            target_ctx = 262_144,
            max_target_ctx = 262_144,
            total_by_idx = {i: per_gpu for i in range(ngpu)},
            cache_type_kv = cache_type,
            scratch_cache_type_kv = scratch_type,
            n_parallel = 1,
            swa_full = False,
            kv_unified = True,
            n_ubatch = ub,
            flash_attn = False,
        )

    def test_an_asymmetric_pair_is_capped_like_its_quantized_axis(self):
        """-ctk q4_0 -ctv f16 at a raised micro-batch. Priced from the heavier
        axis alone the planner advertises the full 262144; the quantized axis
        cannot hold it."""
        b = _backend_from_gguf_local()
        # 36 GB cards, so the KV cache rather than the native length bounds the context.
        optimistic = self._plan(b, "f16", None, 2048, per_gpu = 36_000)[0]
        honest = self._plan(b, "f16", "q4_0", 2048, per_gpu = 36_000)[0]

        assert honest < optimistic, (honest, optimistic)
        # Still a real context, not the 2048 floor: the fix must not collapse the
        # fit, only stop it over-advertising.
        assert honest > 2048, honest
        # It comes out BELOW a symmetric q4_0 load, which is right and worth
        # pinning: the asymmetric pair pays f16 KV bytes on both axes (the heavier
        # axis budgets storage) AND the full quantized dequant scratch. Both terms
        # conservative is the point; neither one alone describes this launch.
        assert honest < self._plan(b, "q4_0", "q4_0", 2048, per_gpu = 36_000)[0]

    def test_a_symmetric_request_is_unchanged(self):
        """Default and explicit scratch type agree when both axes match, so no
        existing load moves."""
        b = _backend_from_gguf_local()
        for kv in ("f16", "q8_0", "q4_0"):
            for ub in (512, 2048):
                assert self._plan(b, kv, None, ub) == self._plan(b, kv, kv, ub)


class TestSpeculativeComputeBuffers:
    """Draft buffers must cover growth beyond the 224 MiB floor.
    Measurements below are from Qwen3.8 27B's embedded head."""

    def _qwen38(self):
        b = _backend(vocab = 248320, embd = 5120)
        b._feed_forward_length = 17408
        return b

    @pytest.mark.parametrize(
        "ub,ctx,measured", [(512, 32768, 130.02), (1120, 32768, 284.41), (1120, 131072, 428.78)]
    )
    def test_floor_plus_growth_covers_the_draft_context(self, ub, ctx, measured):
        b = self._qwen38()
        charged = (
            LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES + b._mtp_draft_compute_bytes(ctx, n_ubatch = ub)
        ) / MIB
        assert charged >= measured

    def test_short_context_and_micro_batch_stay_on_the_floor(self):
        assert self._qwen38()._mtp_draft_compute_bytes(4096, n_ubatch = 256) == 0

    def test_growth_follows_the_mask_once_past_the_floor(self):
        b = self._qwen38()
        lo = b._mtp_draft_compute_bytes(131072, n_ubatch = 1120)
        hi = b._mtp_draft_compute_bytes(262144, n_ubatch = 1120)
        assert hi - lo == 1120 * 2 * 131072

    def test_verify_rows_scale_with_slots_and_depth(self):
        b = self._qwen38()
        base = b._mtp_draft_compute_bytes(4096, n_ubatch = 256)
        rows = b._mtp_draft_compute_bytes(4096, n_ubatch = 256, n_parallel = 4, spec_draft_n_max = 2)
        assert rows - base == 4 * 2 * LlamaCppBackend._SPEC_VERIFY_ROW_COPIES * 248320 * 4

    def test_the_reserve_carries_it(self):
        b = self._qwen38()
        b._n_layers, b._n_kv_heads, b._kv_key_length, b._kv_value_length = 65, 4, 256, 256
        b._nextn_predict_layers = 1
        kwargs = dict(n_parallel = 1, target_rollback = False, mtp_keeps_target_ctx = False)
        short = b._estimate_mtp_overhead_bytes(4096, n_ubatch = 256, **kwargs)
        long = b._estimate_mtp_overhead_bytes(262144, n_ubatch = 1120, **kwargs)
        draft_kv = b._mtp_draft_kv_bytes(262144, n_ubatch = 1120) - b._mtp_draft_kv_bytes(
            4096, n_ubatch = 256
        )
        assert long - short == draft_kv + b._mtp_draft_compute_bytes(262144, n_ubatch = 1120)


class TestFlashAttentionOff:
    """Without flash attention llama.cpp materialises [n_kv, n_ubatch, n_head] float32
    attention scores and builds the KQ mask as float32. Measured on b10909 (RTX 6000
    Ada), f16 cache, llama-server's CUDA0 compute buffer in MiB."""

    _MEASURED = [
        ("Llama-3.2-1B", 512, 4, 32768, 2130.01),
        ("Llama-3.2-1B", 2048, 4, 32768, 8520.02),
        ("Llama-3.2-1B", 512, 4, 65536, 4242.01),
        ("Qwen3-8B", 1120, 4, 32768, 4698.76),
        ("Qwen3.5-4B", 512, 4, 65536, 2214.01),
        ("Qwen3.6-35B-A3B", 2048, 4, 32768, 4504.18),
        ("Gemma-4-12B", 1120, 4, 32768, 2540.8),
        ("Gemma-4-12B", 512, 4, 65536, 2248.01),
        ("Gemma-4-E4B", 512, 4, 65536, 1217.51),
        ("GLM-4.7-Flash", 1120, 4, 32768, 3086.3),
        ("GLM-4.7-Flash", 512, 4, 65536, 2818.9),
    ]

    @pytest.mark.parametrize("name,ub,slots,ctx,measured", _MEASURED)
    def test_covers_the_measured_buffer(self, name, ub, slots, ctx, measured):
        b = _measured(name)
        est = (
            b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
            + b._compute_buffer_ctx_bytes(ctx, ub, "f16", flash_attn = False)
        ) / MIB
        assert measured <= est <= measured * 1.1, f"{name}: {est:.0f} vs {measured}"

    def test_the_rate_is_the_score_matrix_plus_a_float32_mask(self):
        b = _measured("Gemma-4-12B")
        on = b._compute_buffer_ctx_bytes(100000, 1120, "f16")
        off = b._compute_buffer_ctx_bytes(100000, 1120, "f16", flash_attn = False)
        assert on == 1120 * 2 * 100000
        assert off == 1120 * (4 + 16 * 4) * 100000


class TestLayerSplitSlidingWindowSlots:
    """The replicated sliding-window masks of a layer split hold the window for every
    slot, as the single-device mask does. Measured per GPU across an RTX 6000 Ada and an
    RTX 3090, Gemma 4 12B at ubatch 1120."""

    @pytest.mark.parametrize("slots,ctx,measured", [(8, 65536, 971), (16, 131072, 1601)])
    def test_covers_the_measured_split(self, slots, ctx, measured):
        b = _measured("Gemma-4-12B")
        per_device = (
            b._estimate_compute_buffer_bytes(n_ubatch = 1120, n_parallel = slots)
            + b._compute_buffer_ctx_bytes(ctx, 1120, "f16", layer_split = True, n_parallel = slots)
        ) / MIB
        one_slot_window = (
            b._estimate_compute_buffer_bytes(n_ubatch = 1120, n_parallel = slots)
            + b._compute_buffer_ctx_bytes(ctx, 1120, "f16", layer_split = True)
        ) / MIB
        assert one_slot_window < measured <= per_device <= measured * 1.1


class TestOutputRowsByBuild:
    """Before ggml-org/llama.cpp#23861 (b9460) a context reserved an output row for every
    micro-batch token, once, whatever the slot count: on Vulkan, Gemma 4 E2B at ubatch
    512 reserved 515 MiB on b9415 and 137 MiB on b10909, the same at one slot or four."""

    @pytest.mark.parametrize(
        "output,build",
        [
            ("version: 0.4.0-dev (build 10909, commit 329b6160f)\nbuilt with GNU", 10909),
            ("version: 9415 (06d26dfdf)\nbuilt with GNU 11.4.0", 9415),
            ("version: 1 (unknown)\n", None),
            ("no version here", None),
        ],
    )
    def test_the_build_is_read_from_either_version_format(
        self, tmp_path, monkeypatch, output, build
    ):
        import subprocess
        import types

        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        monkeypatch.setattr(LlamaCppBackend, "_build_number_cache", {})
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(stdout = output, stderr = "", returncode = 0),
        )
        assert LlamaCppBackend.probe_build_number(str(binary)) == build

    @pytest.mark.parametrize(
        "build,legacy", [(9415, True), (9459, True), (9460, False), (None, False)]
    )
    def test_only_a_known_older_build_reserves_micro_batch_rows(self, monkeypatch, build, legacy):
        monkeypatch.setattr(
            LlamaCppBackend, "probe_build_number", classmethod(lambda cls, binary = None: build)
        )
        assert LlamaCppBackend.reserves_micro_batch_outputs() is legacy

    def test_an_older_build_is_charged_a_row_per_micro_batch_token(self):
        # Measured b9415 Vulkan compute buffers (MiB) at 32768 cells, one and four slots.
        cases = [
            (dict(vocab = 128256, embd = 2048, ff = 8192), 512, 254.50),
            (dict(vocab = 151936, embd = 4096, ff = 12288), 2048, 1667.05),
            (dict(vocab = 262144, embd = 1536, ff = 12288, ple = 256 * 35), 2048, 2704.08),
        ]
        for dims, ub, measured in cases:
            b = _backend(vocab = dims["vocab"], embd = dims["embd"])
            b._feed_forward_length = dims["ff"]
            b._embedding_length_per_layer_input = 0
            b._n_layers = 1
            if "ple" in dims:
                b._embedding_length_per_layer_input, b._n_layers = 256, 35
            b._reserves_micro_batch_outputs = True
            for slots in (1, 4):
                est = (
                    b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
                    + b._compute_buffer_ctx_bytes(32768, ub, "f16")
                ) / MIB
                assert measured <= est <= measured * 1.5, (dims, slots, est)
            b._reserves_micro_batch_outputs = False
            assert b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = 1) < (
                dims["vocab"] * ub * 4
            )
