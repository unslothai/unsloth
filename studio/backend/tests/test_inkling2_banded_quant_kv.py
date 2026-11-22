# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 compute per llama.cpp build: #399 runs quantized K/V banded, older go dense (#18833)."""

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

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import (
    _INKLING2_BANDED_LEGACY,
    _Inkling2BandedSupport,
    LlamaCppBackend,
)

MIB = 1024 * 1024
OLD_CUDA = _Inkling2BandedSupport("cuda", False)
NEW_CUDA = _Inkling2BandedSupport("cuda", True)
NEW_HIP = _Inkling2BandedSupport("hip", True)

# Registration strings of the banded MMA kernel: #399 instantiates it per K/V tile type.
NEW_KERNEL = b"_Z25flash_attn_ext_banded_mmaILi128ELi1EL9ggml_type8EEvPKcS2_S2_S2_PKhS2_PKfPfS7_fil"
OLD_KERNEL = b"_Z25flash_attn_ext_banded_mmaILi128ELi1EEvPKcS1_S1_S1_PKhS1_PKfPfS6_fil"


def _inkling2_mini(support = _INKLING2_BANDED_LEGACY) -> LlamaCppBackend:
    """Inkling-2-Mini as _read_gguf_metadata leaves it (inkling2, 48 cache slots)."""
    b = LlamaCppBackend()
    b._architecture = "inkling2"
    b._context_length = 1048576
    b._n_layers = 48
    b._n_weight_layers = 24
    b._n_experts = 128
    b._n_kv_heads = 8
    b._n_kv_heads_by_layer = [8] * 48
    b._n_heads_by_layer = [24] * 48
    b._attn_res_block_size = 6
    b._embedding_length = 2304
    b._feed_forward_length = 1152
    b._expert_used_count = 8
    b._expert_feed_forward_length = 1152
    b._vocab_size = 200058
    b._kv_key_length = 128
    b._kv_value_length = 128
    b._kv_key_length_swa = 128
    b._kv_value_length_swa = 128
    b._sliding_window = 1024
    b._sliding_window_pattern = ([True] * 5 + [False]) * 8
    b._inkling2_banded_support = support
    return b


def _compute(
    b,
    ctx,
    kv,
    ub = 1024,
    slots = 4,
    fa = True,
):
    return b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots) + (
        b._compute_buffer_ctx_bytes(ctx, ub, kv, flash_attn = fa, n_parallel = slots)
    )


def _total(
    b,
    ctx,
    kv,
    ub = 1024,
    slots = 4,
):
    kv_bytes = b._estimate_kv_cache_bytes(
        ctx, kv, n_parallel = slots, kv_unified = True, n_ubatch = ub, flash_attn = True
    )
    return kv_bytes + _compute(b, ctx, kv, ub, slots)


class TestBuildDecidesTheQuantizedPath:
    @pytest.mark.parametrize("ctx", [32768, 131072, 262144])
    @pytest.mark.parametrize("kv", ["q8_0", "q4_0"])
    def test_old_build_keeps_the_dense_estimate(self, ctx, kv):
        dense_rate = 1024 * (8 * 24 + 4 + 2)
        for support in (_INKLING2_BANDED_LEGACY, OLD_CUDA, _Inkling2BandedSupport("cpu", False)):
            b = _inkling2_mini(support)
            assert b._compute_buffer_ctx_bytes(ctx, 1024, kv) == dense_rate * ctx

    @pytest.mark.parametrize("ctx", [32768, 131072, 262144])
    @pytest.mark.parametrize("kv", ["q8_0", "q4_0"])
    def test_new_cuda_build_puts_a_quantized_cache_below_f16(self, ctx, kv):
        b = _inkling2_mini(NEW_CUDA)
        assert _total(b, ctx, kv) < _total(b, ctx, "f16")
        assert _compute(b, ctx, kv) == _compute(b, ctx, "f16")
        assert _total(b, ctx, kv) < _total(_inkling2_mini(OLD_CUDA), ctx, kv)

    @pytest.mark.parametrize("ctx", [32768, 131072, 262144])
    def test_old_build_reports_q8_above_f16(self, ctx):
        # The reported shape (#18833): what the shipped build really does, so it must stay.
        b = _inkling2_mini(OLD_CUDA)
        assert _total(b, ctx, "q8_0") > _total(b, ctx, "f16")

    @pytest.mark.parametrize("backend", ["metal", "vulkan", "sycl"])
    @pytest.mark.parametrize("kv", ["f16", "q8_0"])
    def test_backends_without_a_banded_kernel_take_the_dense_bias(self, backend, kv):
        for quant_kv in (False, True):
            b = _inkling2_mini(_Inkling2BandedSupport(backend, quant_kv))
            assert b._compute_buffer_ctx_bytes(65536, 1024, kv) == 1024 * (8 * 24 + 4 + 2) * 65536

    def test_hip_reads_an_f16_copy_of_the_cache(self):
        cuda = _inkling2_mini(NEW_CUDA)._compute_buffer_ctx_bytes(131072, 1024, "q8_0")
        hip_b = _inkling2_mini(NEW_HIP)
        hip = hip_b._compute_buffer_ctx_bytes(131072, 1024, "q8_0")
        assert hip - cuda == hip_b._attention_dequant_width() * 2 * 131072
        assert _total(hip_b, 131072, "q8_0") < _total(hip_b, 131072, "f16")

    @pytest.mark.parametrize("kv,copy", [("q5_0", True), ("q4_1", True), ("q8_0", False)])
    def test_types_without_a_tile_dequant_pay_the_copy(self, kv, copy):
        b = _inkling2_mini(NEW_CUDA)
        extra = b._compute_buffer_ctx_bytes(65536, 1024, kv) - b._compute_buffer_ctx_bytes(
            65536, 1024, "f16"
        )
        assert extra == (b._attention_dequant_width() * 2 * 65536 if copy else 0)

    def test_iq4_nl_has_no_f16_converter_so_stays_dense(self):
        b = _inkling2_mini(NEW_CUDA)
        assert b._compute_buffer_ctx_bytes(65536, 1024, "iq4_nl") == 1024 * 198 * 65536

    def test_shared_mask_shrinks_the_f16_rate(self):
        for support, rate in ((OLD_CUDA, 18), (NEW_CUDA, 4)):
            b = _inkling2_mini(support)
            slope = (
                b._compute_buffer_ctx_bytes(2_000_000, 1024, "f16")
                - b._compute_buffer_ctx_bytes(1_000_000, 1024, "f16")
            ) / 1_000_000
            assert slope == rate * 1024


class TestRuntimeComputeBuffer:
    """CUDA0 MiB at exit, Inkling-2-Mini UD-IQ1_S, -np 4 --kv-unified, B200; old = b12602, new = #399."""

    _MEASURED = [
        # (support, ub, n_ctx, kv, reserve MiB, runtime MiB)
        (OLD_CUDA, 1024, 32768, "f16", 1117.0, 1722.3),
        (OLD_CUDA, 1024, 131072, "f16", 2749.0, 3093.3),
        (NEW_CUDA, 1024, 32768, "f16", 513.6, 1594.1),
        (NEW_CUDA, 1024, 32768, "q8_0", 513.6, 1594.1),
        (NEW_CUDA, 1024, 32768, "q4_0", 513.6, 1594.1),
        (NEW_CUDA, 1024, 131072, "f16", 717.6, 1768.4),
        (NEW_CUDA, 512, 32768, "f16", 257.6, 747.6),
        (NEW_CUDA, 512, 131072, "q8_0", 540.8, 837.3),
        (OLD_CUDA, 512, 32768, "f16", 541.1, 811.7),
        # Studio's own 24 GiB launch for #18757 (ub 512, 162048 cells), slot 0 at 77K / 150K.
        (OLD_CUDA, 512, 162048, "f16", 1629.0, 1629.0),
        (NEW_CUDA, 512, 162048, "f16", 418.5, 885.0),
    ]
    # Long context, slot 0 filled to only 0.125 n_ctx: the load-time reserve, a floor.
    _RESERVE = [
        (NEW_CUDA, 512, 524288, "f16", 1078.0),
        (NEW_CUDA, 512, 1048576, "f16", 2102.0),
        (NEW_CUDA, 1024, 1048576, "f16", 4205.6),
    ]

    @pytest.mark.parametrize("support,ub,ctx,kv,reserve,runtime", _MEASURED)
    def test_estimate_covers_the_reallocated_buffer(self, support, ub, ctx, kv, reserve, runtime):
        est = _compute(_inkling2_mini(support), ctx, kv, ub = ub) / MIB
        assert runtime <= est <= runtime * 1.2

    @pytest.mark.parametrize("support,ub,ctx,kv,reserve", _RESERVE)
    def test_long_context_estimate_covers_the_reserve(self, support, ub, ctx, kv, reserve):
        est = _compute(_inkling2_mini(support), ctx, kv, ub = ub) / MIB
        assert reserve <= est <= reserve * 1.25


class TestProbe:
    def _install(self, tmp_path, libs: dict) -> str:
        server = tmp_path / "llama-server"
        server.write_bytes(b"\x7fELF server")
        for name, data in libs.items():
            (tmp_path / name).write_bytes(data)
        return str(server)

    @pytest.fixture(autouse = True)
    def _no_metal(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_build_has_metal", staticmethod(lambda binary = None: False)
        )
        llama_cpp._lib_matches.cache_clear()

    def test_new_cuda_library(self, tmp_path):
        binary = self._install(
            tmp_path, {"libggml-base.so.1": b"", "libggml-cuda.so.1.0.0": b"x" + NEW_KERNEL + b"y"}
        )
        assert LlamaCppBackend.inkling2_banded_support(binary) == NEW_CUDA

    def test_old_cuda_library(self, tmp_path):
        binary = self._install(tmp_path, {"libggml-cuda.so": OLD_KERNEL})
        assert LlamaCppBackend.inkling2_banded_support(binary) == OLD_CUDA

    def test_hip_library(self, tmp_path):
        binary = self._install(tmp_path, {"libggml-hip.so": NEW_KERNEL})
        assert LlamaCppBackend.inkling2_banded_support(binary) == NEW_HIP

    def test_vulkan_build_never_claims_the_quantized_path(self, tmp_path):
        binary = self._install(tmp_path, {"libggml-vulkan.so": NEW_KERNEL})
        assert LlamaCppBackend.inkling2_banded_support(binary) == ("vulkan", False)

    def test_metal_build(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_build_has_metal", staticmethod(lambda binary = None: True)
        )
        binary = self._install(tmp_path, {"libggml-metal.dylib": b""})
        assert LlamaCppBackend.inkling2_banded_support(binary) == ("metal", False)

    def test_static_build_is_unknown(self, tmp_path):
        assert LlamaCppBackend.inkling2_banded_support(self._install(tmp_path, {})) == (
            _INKLING2_BANDED_LEGACY
        )

    def test_missing_binary_is_legacy(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda *a, **k: None)
        )
        assert LlamaCppBackend.inkling2_banded_support() == _INKLING2_BANDED_LEGACY

    def test_unset_support_resolves_the_installed_build_once(self, monkeypatch):
        calls = []

        def _probe(cls, binary = None):
            calls.append(binary)
            return NEW_CUDA

        monkeypatch.setattr(LlamaCppBackend, "inkling2_banded_support", classmethod(_probe))
        b = _inkling2_mini(None)
        first = b._compute_buffer_ctx_bytes(65536, 1024, "q8_0")
        assert first == b._compute_buffer_ctx_bytes(65536, 1024, "q8_0")
        assert calls == [None]
        assert first < _inkling2_mini(OLD_CUDA)._compute_buffer_ctx_bytes(65536, 1024, "q8_0")

    def test_a_backend_built_without_init_prices_as_legacy(self):
        b = LlamaCppBackend.__new__(LlamaCppBackend)
        assert b._inkling2_banded() == _INKLING2_BANDED_LEGACY
