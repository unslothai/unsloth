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
NEW_CUDA = _Inkling2BandedSupport("cuda", True, True)
# A #399 build on a pre-Turing (or unknown) card: no banded MMA, so no in-tile dequant.
NEW_CUDA_PRE_TURING = _Inkling2BandedSupport("cuda", True, False)
NEW_HIP = _Inkling2BandedSupport("hip", True)
# unslothai/llama.cpp b12649-mix on: several slots reserve the dense relative bias.
MULTI_SEQ_CUDA = _Inkling2BandedSupport("cuda", True, True, True, True)
# The same reserve assumed from an unreadable build number: priced, not identified.
MULTI_SEQ_CUDA_UNKNOWN = _Inkling2BandedSupport("cuda", True, True, True, False)
LOOP_MARKER = b"inkling2_loop: %zu cache slots exceed the maximum of %d"
# The banded-position fix to that reserve: several slots and short ubatches run banded.
BANDED_POS_CUDA = _Inkling2BandedSupport("cuda", True, True, True, True, True)
BANDED_POS_CUDA_UNKNOWN = _Inkling2BandedSupport("cuda", True, True, True, False, True)
BANDED_POS_MARKERS = b"\x00attn_inp_k_pos_swa\x00attn_inp_rel_shift_swa\x00"

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

    @pytest.mark.parametrize("kv", ["q8_0", "q4_0"])
    def test_pre_turing_cuda_reads_an_f16_copy_of_the_cache(self, kv):
        b = _inkling2_mini(NEW_CUDA_PRE_TURING)
        extra = b._compute_buffer_ctx_bytes(262144, 1024, kv) - _inkling2_mini(
            NEW_CUDA
        )._compute_buffer_ctx_bytes(262144, 1024, kv)
        assert extra == b._attention_dequant_width() * 2 * 262144 > 0

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
        self._caps(monkeypatch, {0: 90})

    @staticmethod
    def _caps(
        monkeypatch,
        caps,
        sms = None,
    ):
        monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: caps))
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_cuda_sms", staticmethod(lambda binary = None: sms)
        )

    @pytest.mark.parametrize(
        "caps,sms,tile",
        [
            ({0: 80}, None, True),
            ({0: 75, 1: 120}, None, True),
            ({0: 61}, None, False),  # Pascal P40 / GTX 10xx
            ({0: 70}, None, False),  # Volta: no Turing MMA either
            ({0: 89, 1: 61}, None, False),  # one old card is enough
            ({}, None, False),  # unknown fails closed
            ({0: 86}, frozenset({50, 61}), False),  # PTX-only bundle runs sm_61 code
            ({0: 86}, frozenset({61, 75, 86}), True),
        ],
    )
    def test_in_tile_dequant_follows_the_compute_capability(
        self, tmp_path, monkeypatch, caps, sms, tile
    ):
        self._caps(monkeypatch, caps, sms)
        binary = self._install(tmp_path, {"libggml-cuda.so": NEW_KERNEL})
        assert LlamaCppBackend.inkling2_banded_support(binary) == (
            "cuda",
            True,
            tile,
            False,
            False,
            False,
        )

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
        assert LlamaCppBackend.inkling2_banded_support(binary) == (
            "vulkan",
            False,
            False,
            False,
            False,
            False,
        )

    def test_metal_build(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_build_has_metal", staticmethod(lambda binary = None: True)
        )
        binary = self._install(tmp_path, {"libggml-metal.dylib": b""})
        assert LlamaCppBackend.inkling2_banded_support(binary) == (
            "metal",
            False,
            False,
            False,
            False,
            False,
        )

    @pytest.mark.parametrize(
        "build,dense", [(12649, True), (12700, True), (12631, False), (None, True)]
    )
    def test_loop_loader_and_build_mark_the_dense_reserve(
        self, tmp_path, monkeypatch, build, dense
    ):
        # b12631-mix has the loader but still reserves the banded graph.
        monkeypatch.setattr(
            LlamaCppBackend, "probe_build_number", classmethod(lambda cls, binary = None: build)
        )
        binary = self._install(
            tmp_path,
            {"libggml-cuda.so": NEW_KERNEL, "libllama.so.1.0.0": b"x" + LOOP_MARKER + b"y"},
        )
        assert LlamaCppBackend.inkling2_banded_support(binary) == (
            "cuda",
            True,
            True,
            dense,
            build is not None,
            False,
        )

    @pytest.mark.parametrize(
        "build,lib,expected",
        [
            # The fix as built from ink-AB-combined: build 13010 (4f96ac056).
            (13010, LOOP_MARKER + BANDED_POS_MARKERS, BANDED_POS_CUDA),
            # Its markers identify it without a build number.
            (None, LOOP_MARKER + BANDED_POS_MARKERS, BANDED_POS_CUDA_UNKNOWN),
            # b12649-mix and ded02bdeb: the loader string alone.
            (12649, LOOP_MARKER, MULTI_SEQ_CUDA),
            # Both inputs are needed: one alone is not the fix this prices.
            (13010, LOOP_MARKER + b"\x00attn_inp_k_pos_swa\x00", MULTI_SEQ_CUDA),
            (13010, LOOP_MARKER + b"\x00attn_inp_rel_shift_swa\x00", MULTI_SEQ_CUDA),
            # A longer name is another input, not the marker.
            (
                13010,
                LOOP_MARKER + b"\x00attn_inp_k_pos_swa_x\x00attn_inp_rel_shift_swa\x00",
                MULTI_SEQ_CUDA,
            ),
            # b12631-mix's build gates the dense reserve, and the fix with it.
            (12631, LOOP_MARKER + BANDED_POS_MARKERS, NEW_CUDA._replace(build_known = True)),
        ],
    )
    def test_banded_position_inputs_mark_the_fixed_reserve(
        self, tmp_path, monkeypatch, build, lib, expected
    ):
        monkeypatch.setattr(
            LlamaCppBackend, "probe_build_number", classmethod(lambda cls, binary = None: build)
        )
        binary = self._install(
            tmp_path, {"libggml-cuda.so": NEW_KERNEL, "libllama.so.1.0.0": b"x" + lib + b"y"}
        )
        if build is None:
            expected = expected._replace(build_known = False)
        assert LlamaCppBackend.inkling2_banded_support(binary) == expected

    def test_libllama_without_the_loader_keeps_the_banded_reserve(self, tmp_path):
        binary = self._install(tmp_path, {"libggml-cuda.so": NEW_KERNEL, "libllama.so": b"llama"})
        assert LlamaCppBackend.inkling2_banded_support(binary) == NEW_CUDA

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


def _llama_cpp_chunk_rows(
    n_ctx,
    ub,
    n_head = 24,
):
    """inkling2.cpp's query rows per dense-bias chunk (single stream, full cache)."""
    n_kv = (n_ctx + 255) // 256 * 256
    budget = (1 << 29) // (n_kv * n_head) // 64 * 64
    tiles = -(-ub // 64)
    return min(ub, max(64, budget, -(-tiles // 64) * 64))


class TestMultiSlotDenseReserve:
    """b12649-mix dense reserve. Measured CUDA0 MiB, Inkling-2-Mini UD-Q4_K_XL, --kv-unified,
    flash attention, RTX 6000 Ada."""

    _MEASURED = [
        # (n_ctx, ub, slots, cache type, --swa-full, MiB)
        (8192, 512, 4, "f16", False, 880.53),
        (16384, 512, 4, "f16", False, 1480.53),
        (32768, 512, 4, "f16", False, 2680.53),
        (65536, 512, 4, "f16", False, 3798.03),
        (131072, 512, 4, "f16", False, 3990.03),
        (241664, 512, 4, "f16", False, 5106.03),
        (262144, 512, 4, "f16", False, 5526.03),
        (524288, 512, 4, "f16", False, 10902.03),
        (32768, 2048, 4, "f16", False, 5033.19),
        (131072, 2048, 4, "f16", False, 8297.19),
        (8192, 4096, 4, "f16", False, 8162.0),
        (65536, 512, 2, "f16", False, 3791.87),
        (65536, 512, 4, "q8_0", False, 3801.78),
        (262144, 512, 4, "q8_0", False, 5526.03),
        # The sliding layers chunk over their own cache: --swa-full or enough slots to fill it.
        (65536, 512, 4, "f16", True, 6461.28),
        (65536, 512, 4, "q8_0", True, 6465.03),
        (131072, 512, 4, "q8_0", True, 9340.53),
        (65536, 512, 16, "f16", False, 3834.94),
        (262144, 512, 16, "f16", False, 5562.94),
        (65536, 512, 64, "f16", False, 6465.85),
        (131072, 512, 64, "f16", False, 7150.35),
        (65536, 512, 64, "q8_0", False, 6469.60),
        (98304, 512, 4, "f16", True, 7899.78),
        (196608, 512, 4, "q4_0", True, 12219.03),
        (274688, 512, 4, "q4_0", True, 17022.78),
        # llama.cpp's allocation-free graph sizing on the same build.
        (524288, 512, 4, "q4_0", True, 32379.0),
        (196608, 512, 64, "q4_0", False, 7794.10),
        (274688, 512, 4, "q4_0", False, 5783.28),
    ]

    @pytest.mark.parametrize("ctx,ub,slots,kv,swa_full,measured", _MEASURED)
    def test_estimate_covers_the_reserve(self, ctx, ub, slots, kv, swa_full, measured):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        est = (
            b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
            + b._compute_buffer_ctx_bytes(ctx, ub, kv, n_parallel = slots, swa_full = swa_full)
        ) / MIB
        # Highest at 8K ub=4096, where the sliding cache is the whole context: 1.77x.
        assert measured <= est <= measured * 1.8

    def test_a_sliding_cache_that_needs_chunks_adds_its_own_bias(self):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        compact = b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 4)
        assert b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 16) == compact
        assert b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 64) > compact + 2600 * MIB
        assert b._compute_buffer_ctx_bytes(
            65536, 512, "f16", n_parallel = 4, swa_full = True
        ) == b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 64)

    @pytest.mark.parametrize("ub", [512, 2048, 4096])
    def test_bias_bound_covers_every_chunk_and_never_falls(self, ub):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        prev = [0, 0, 0]
        for ctx in range(4096, 1048577, 256):
            elems = b._inkling2_dense_bias_elems(ctx, ub, 24)
            assert elems >= ((ctx + 255) // 256 * 256) * _llama_cpp_chunk_rows(ctx, ub)
            cost = [
                b._compute_buffer_ctx_bytes(ctx, ub, "f16", n_parallel = s, swa_full = f)
                for s, f in ((4, False), (64, False), (4, True))
            ]
            assert all(c >= p for c, p in zip(cost, prev)), ctx
            prev = cost

    @pytest.mark.parametrize(
        "ctx,ub,rows",
        [
            (32768, 512, 512),  # 682 rows fit the budget: one call
            (65536, 512, 320),
            (131072, 512, 128),
            (262144, 512, 64),
            (1048576, 512, 64),  # the 64-row floor
            (131072, 8192, 128),  # at most 64 chunks: 8192 / 64
        ],
    )
    def test_reference_chunk_rows(self, ctx, ub, rows):
        assert _llama_cpp_chunk_rows(ctx, ub) == rows

    # One slot: the reserve holds a small ubatch's dense bias; 256K ub=512 adds the prompt's pool.
    _ONE_SLOT = [
        # (n_ctx, ub, cache type, MiB)
        (65536, 256, "f16", 357.02),
        (262144, 256, "f16", 1353.02),
        (65536, 512, "f16", 414.30),
        (131072, 512, "f16", 769.71),
        (262144, 512, "f16", 1497.71 + 112),
        (524288, 512, "f16", 2953.71),
        (655360, 512, "f16", 3681.71),
        (65536, 1024, "f16", 867.52),
        (262144, 1024, "f16", 1792.33),
        (65536, 2048, "f16", 1980.96),
        (262144, 2048, "f16", 2844.58),
        (262144, 512, "bf16", 2522.08),
        (262144, 512, "q8_0", 2522.08),
        (262144, 512, "q4_0", 2522.08),
        (262144, 2048, "q8_0", 3420.96),
    ]

    # One slot with --swa-full: the sliding layers' per-cell inputs span the full cache.
    _ONE_SLOT_SWA_FULL = [
        (131072, 512, "q4_0", 1515.58),
        (262144, 512, "q4_0", 3011.58),
        (415488, 512, "q4_0", 4761.78),
        (131072, 512, "f16", 1009.20),
        (131072, 2048, "q4_0", 2340.96),
        (262144, 2048, "q4_0", 4604.96),
        (415488, 2048, "q4_0", 7253.66),
        (131072, 2048, "f16", 2172.58),
    ]

    @pytest.mark.parametrize("ctx,ub,kv,measured", _ONE_SLOT_SWA_FULL)
    def test_one_slot_swa_full_covers_the_reserve(self, ctx, ub, kv, measured):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        est = (
            b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = 1)
            + b._compute_buffer_ctx_bytes(ctx, ub, kv, n_parallel = 1, swa_full = True)
        ) / MIB
        # Highest at 131K ub=2048, where the per-ubatch terms dominate: 2.1x.
        assert measured <= est <= measured * 2.2

    @pytest.mark.parametrize("ctx,ub,kv,measured", _ONE_SLOT)
    def test_one_slot_estimate_covers_the_reserve(self, ctx, ub, kv, measured):
        est = _compute(_inkling2_mini(MULTI_SEQ_CUDA), ctx, kv, ub = ub, slots = 1) / MIB
        assert measured <= est <= measured * 1.8

    @pytest.mark.parametrize(
        "kv,per_cell", [("f16", 6144), ("q8_0", 6144 + 4096), ("bf16", 6144 + 4096)]
    )
    def test_one_slot_reserve_holds_the_small_ubatch_dense_bias(self, kv, per_cell):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        n = 1 << 20
        grown = b._inkling2_reserve_bytes(2 * n, 512, kv) - b._inkling2_reserve_bytes(n, 512, kv)
        assert grown == per_cell * n

    def test_older_builds_keep_the_banded_price_for_several_slots(self):
        b = _inkling2_mini(NEW_CUDA)
        assert b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 4) == (
            b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = 1)
        )

    @pytest.mark.parametrize("slots", [1, 4])
    def test_fit_margin_is_only_the_pool_once_the_reserve_holds_the_worst_case(self, slots):
        # 256 KiB per ub token.
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        assert b._inkling2_fit_regrowth_mib(262144, 512, "q8_0", n_parallel = slots) == 128

    def test_an_unknown_build_keeps_the_older_fit_margin(self):
        # b12631-mix has the loader but the banded reserve, so it keeps the growth allowance.
        unknown = _inkling2_mini(MULTI_SEQ_CUDA_UNKNOWN)
        known = _inkling2_mini(MULTI_SEQ_CUDA)
        assert unknown._inkling2_fit_regrowth_mib(32768, 2048, "f16", n_parallel = 4) > 512
        assert known._inkling2_fit_regrowth_mib(32768, 2048, "f16", n_parallel = 4) == 512
        assert unknown._compute_buffer_ctx_bytes(32768, 2048, "f16", n_parallel = 4) == (
            known._compute_buffer_ctx_bytes(32768, 2048, "f16", n_parallel = 4)
        )

    @pytest.mark.parametrize(
        "ctx,ub,kv,slots",
        [(262144, 512, "q8_0", 4), (32768, 2048, "f16", 4), (131072, 512, "q4_0", 1)],
    )
    def test_an_unknown_build_prices_the_banded_build_margin(self, ctx, ub, kv, slots):
        unknown = _inkling2_mini(MULTI_SEQ_CUDA_UNKNOWN)
        banded = _inkling2_mini(NEW_CUDA)
        margin = unknown._inkling2_fit_regrowth_mib(ctx, ub, kv, n_parallel = slots)
        assert margin == banded._inkling2_fit_regrowth_mib(ctx, ub, kv, n_parallel = slots)
        if (ctx, slots) == (262144, 4):
            # An RTX 3090 load with a build-1 llama-server reserved 5957 MiB here.
            assert margin == 600

    def test_banded_builds_keep_the_regrowth_margin_for_several_slots(self):
        b = _inkling2_mini(NEW_CUDA)
        assert b._inkling2_fit_regrowth_mib(32768, 2048, "f16", n_parallel = 4) > 0

    def test_flash_attention_off_keeps_the_dense_upper_bound(self):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        assert b._compute_buffer_ctx_bytes(
            65536, 512, "f16", flash_attn = False, n_parallel = 4
        ) == b._compute_buffer_ctx_bytes(65536, 512, "f16", flash_attn = False, n_parallel = 1)

    def test_slot_retry_prices_each_candidate_with_its_own_compute(self, monkeypatch):
        b = _inkling2_mini(MULTI_SEQ_CUDA)
        seen = []

        def _select(total, gpus, **kw):
            seen.append(kw["per_device_overhead_bytes"])
            return [0], True  # nothing fits, so every candidate is priced

        monkeypatch.setattr(b, "_select_gpus_split_aware", _select)

        def cc(slots):
            return b._compute_buffer_ctx_bytes(65536, 512, "f16", n_parallel = slots)

        b._slots_that_fit_on_gpu(
            4,
            65536,
            [(0, 48_000)],
            {0: 49_140},
            0,
            "f16",
            1.0,
            0,
            1,
            512,
            ctx_compute_for_slots = cc,
        )
        assert seen == [cc(3), cc(2), cc(1)]
        assert cc(1) * 3 < cc(2)


class TestBandedPositionReserve:
    """The banded-position fix to #386 (ink-AB-combined, build 13010): the reserve holds the f16
    KQ masks and a per-ubatch remainder, whatever the slots or K/V type. Measured CUDA0 MiB at load,
    llama-server, Inkling-2-Mini UD-IQ1_S, flash attention, -b 2048, --kv-unified, B200."""

    _MEASURED = [
        # (n_ctx, ub, slots, cache type, --swa-full, MiB)
        (4096, 512, 1, "f16", False, 224.23),
        (8192, 512, 1, "f16", False, 228.24),
        (16384, 512, 1, "f16", False, 236.28),
        (65536, 512, 1, "f16", False, 284.46),
        (524288, 512, 1, "f16", False, 734.21),
        (4096, 2048, 1, "f16", False, 902.63),
        (16384, 2048, 1, "f16", False, 950.67),
        (65536, 2048, 1, "f16", False, 1142.86),
        (524288, 2048, 1, "f16", False, 2936.61),
        (32768, 256, 1, "f16", False, 126.15),
        (262144, 256, 1, "f16", False, 239.02),
        (32768, 1024, 1, "f16", False, 505.47),
        (262144, 1024, 1, "f16", False, 954.35),
        (262144, 512, 1, "bf16", False, 477.21),
        (1048576, 512, 1, "q4_0", False, 1248.21),
        (32768, 512, 1, "f16", True, 282.96),
        (131072, 512, 1, "f16", True, 475.71),
        (131072, 2048, 1, "f16", True, 1899.60),
        (262144, 512, 1, "q4_0", True, 732.71),
        (415488, 512, 1, "q4_0", True, 1033.38),
        (415488, 2048, 1, "q4_0", True, 4123.77),
        (65536, 512, 2, "f16", False, 285.54),
        (262144, 2048, 2, "f16", False, 1915.68),
        (4096, 512, 4, "f16", False, 226.97),
        (8192, 512, 4, "f16", False, 231.48),
        (32768, 512, 4, "f16", False, 255.58),
        (65536, 512, 4, "f16", False, 287.70),
        (131072, 512, 4, "q8_0", False, 351.95),
        (262144, 512, 4, "q5_1", False, 480.45),
        (524288, 512, 4, "f16", False, 737.45),
        (1048576, 512, 4, "q4_1", False, 1251.45),
        (32768, 2048, 4, "q5_0", False, 1026.98),
        (131072, 2048, 4, "f16", False, 1411.35),
        (262144, 2048, 4, "q4_0", False, 1923.85),
        (1048576, 2048, 4, "q8_0", False, 4998.85),
        (131072, 1024, 4, "f16", False, 704.09),
        (32768, 256, 4, "f16", False, 127.89),
        (65536, 512, 4, "f16", True, 347.44),
        (131072, 512, 4, "q8_0", True, 475.95),
        (131072, 2048, 4, "f16", True, 1899.82),
        (274688, 512, 4, "q4_0", True, 757.53),
        (274688, 512, 4, "q4_0", False, 492.78),
        (65536, 512, 8, "f16", False, 292.02),
        (65536, 512, 16, "f16", False, 300.66),
        (262144, 512, 16, "f16", False, 493.36),
        (65536, 512, 64, "f16", False, 352.02),
        (131072, 512, 64, "f16", False, 416.80),
        (196608, 512, 64, "q4_0", False, 480.98),
    ]

    @pytest.mark.parametrize("ctx,ub,slots,kv,swa_full,measured", _MEASURED)
    def test_estimate_covers_the_reserve(self, ctx, ub, slots, kv, swa_full, measured):
        b = _inkling2_mini(BANDED_POS_CUDA)
        est = (
            b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
            + b._compute_buffer_ctx_bytes(ctx, ub, kv, n_parallel = slots, swa_full = swa_full)
        ) / MIB
        # Highest with 64 slots at 64K, where the flat estimate's output rows lead: 1.25x.
        assert measured <= est <= measured * 1.26

    @pytest.mark.parametrize(
        "ctx,ub,measured",
        [
            # q8_0 K with f16 V dequantizes one more copy: 48 KiB per ub token.
            (524288, 512, 758.21),
            (1048576, 512, 1275.45),
            (1048576, 2048, 5094.85),
            (32768, 2048, 1122.98),
        ],
    )
    def test_mixed_kv_types_stay_covered(self, ctx, ub, measured):
        b = _inkling2_mini(BANDED_POS_CUDA)
        for slots in (1, 4):
            est = (
                b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = slots)
                + b._compute_buffer_ctx_bytes(ctx, ub, "q8_0", n_parallel = slots)
            ) / MIB
            assert measured <= est

    @pytest.mark.parametrize("ub", [256, 512, 2048])
    def test_slots_and_kv_types_cost_nothing_and_context_never_lowers_it(self, ub):
        b = _inkling2_mini(BANDED_POS_CUDA)
        prev = 0
        for ctx in range(4096, 1048577, 4096):
            costs = {
                b._compute_buffer_ctx_bytes(ctx, ub, kv, n_parallel = s)
                for kv in ("f16", "q8_0", "q4_0")
                for s in (1, 4, 64)
            }
            assert len(costs) == 1
            cost = costs.pop()
            assert cost >= prev
            prev = cost

    @pytest.mark.parametrize(
        "ctx,ub,slots,kv,swa_full",
        [
            (32768, 512, 4, "f16", False),
            (262144, 512, 4, "q8_0", False),
            (1048576, 512, 1, "q4_0", False),
            (274688, 512, 4, "q4_0", True),
            (65536, 512, 64, "f16", False),
        ],
    )
    def test_far_below_the_dense_reserve(self, ctx, ub, slots, kv, swa_full):
        fixed = _inkling2_mini(BANDED_POS_CUDA)._compute_buffer_ctx_bytes(
            ctx, ub, kv, n_parallel = slots, swa_full = swa_full
        )
        dense = _inkling2_mini(MULTI_SEQ_CUDA)._compute_buffer_ctx_bytes(
            ctx, ub, kv, n_parallel = slots, swa_full = swa_full
        )
        assert fixed * 3 < dense

    def test_a_layer_split_adds_the_mask_copies(self):
        b = _inkling2_mini(BANDED_POS_CUDA)
        one = b._compute_buffer_ctx_bytes(32768, 512, "f16", n_parallel = 4)
        split = b._compute_buffer_ctx_bytes(32768, 512, "f16", n_parallel = 4, layer_split = True)
        assert split - one == 3 * 2 * 512 * 32768

    @pytest.mark.parametrize("support", [BANDED_POS_CUDA, BANDED_POS_CUDA_UNKNOWN])
    @pytest.mark.parametrize("ub", [512, 2048])
    def test_fit_margin_is_only_the_pool(self, support, ub):
        # 256 KiB per ub token; the markers identify the fix without a build number.
        b = _inkling2_mini(support)
        for ctx, slots in ((32768, 4), (262144, 1), (1048576, 4)):
            assert b._inkling2_fit_regrowth_mib(ctx, ub, "q8_0", n_parallel = slots) == ub // 4

    def test_flash_attention_off_and_small_ubatches_keep_the_dense_upper_bound(self):
        fixed = _inkling2_mini(BANDED_POS_CUDA)
        dense = _inkling2_mini(MULTI_SEQ_CUDA)
        for kw in ({"flash_attn": False}, {}):
            ub = 512 if kw else 32
            assert fixed._compute_buffer_ctx_bytes(65536, ub, "f16", n_parallel = 4, **kw) == (
                dense._compute_buffer_ctx_bytes(65536, ub, "f16", n_parallel = 4, **kw)
            )
