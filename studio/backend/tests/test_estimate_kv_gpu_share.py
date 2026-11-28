# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF estimate's KV share on the GPU under a manual -ngl: llama.cpp offloads the last
layers (i_gpu_start = n_weight_layers + 1 - ngl), and SWA layers, shared-KV layers and layers
without attention hold fewer cache cells than full-attention ones."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# gemma-3 layout: five SWA layers then one full-attention layer.
_GEMMA3_PATTERN = [(i + 1) % 6 != 0 for i in range(18)]


def _probe(
    pattern,
    *,
    n_layers = None,
    swa = 512,
    **attrs,
) -> LlamaCppBackend:
    probe = LlamaCppBackend(manages_processes = False)
    probe._n_layers = n_layers or len(pattern)
    probe._sliding_window_pattern = pattern
    probe._sliding_window = swa
    probe._n_kv_heads = 1
    probe._n_heads = 4
    probe._kv_key_length = 256
    probe._kv_value_length = 256
    for key, value in attrs.items():
        setattr(probe, key, value)
    return probe


def _runtime(n_ctx = 4096):
    return SimpleNamespace(
        n_ctx = n_ctx,
        swa_full = False,
        n_parallel = 1,
        kv_unified = True,
        n_ubatch = 512,
        cache_type_kv = None,
        flash_attn = True,
    )


class TestKvGpuShare:
    def test_mixed_swa_layers_weight_by_cells(self):
        # ctx 4096, SWA 512 + ubatch 512 -> 1024 cells vs 4096 for a full layer.
        probe = _probe(_GEMMA3_PATTERN)
        assert probe._kv_gpu_share(0, 4096) == 0.0
        # -ngl 1 is the output layer only; -ngl 19 (and beyond) is every repeating layer.
        assert probe._kv_gpu_share(1, 4096) == 0.0
        assert probe._kv_gpu_share(19, 4096) == 1.0
        assert probe._kv_gpu_share(999, 4096) == 1.0
        # -ngl 9: layers 10..17 on the GPU, 6 SWA (1 unit) + 2 full (4 units) of 15 + 12.
        assert probe._kv_gpu_share(9, 4096) == pytest.approx(14 / 27)

    def test_shared_kv_tail_holds_no_cache(self):
        # Last 6 layers reuse earlier KV: offloading only them puts no cache on the GPU.
        probe = _probe([False] * 12, swa = 1024, _shared_kv_layers = 6)
        probe._sliding_window_pattern = [True] + [False] * 11
        assert probe._kv_gpu_share(7, 4096) == 0.0
        assert probe._kv_gpu_share(13, 4096) == 1.0

    def test_layers_without_attention_hold_no_cache(self):
        # Hybrid header: no SWA weights, per-layer KV heads decide.
        probe = _probe(None, n_layers = 4, _n_kv_heads_by_layer = [0, 8, 0, 8])
        assert probe._kv_gpu_share(2, 4096) == 0.5
        probe._n_kv_heads_by_layer = [8, 8, 0, 0]
        assert probe._kv_gpu_share(3, 4096) == 0.0

    def test_loop_slots_sit_with_their_physical_layer(self):
        # 4 physical layers, core [1, 3) run twice: slots 4, 5 reuse layers 1, 2.
        pattern = [True, True, False, True, True, False]
        probe = _probe(pattern, _n_weight_layers = 4, _loop_start = 1, _loop_end = 3)
        probe._architecture = "inkling2_loop"
        # -ngl 3: physical layers 2, 3 on the GPU, so slots 2, 3 and 5 (full, swa, full).
        assert probe._kv_gpu_share(3, 4096) == pytest.approx(9 / 12)

    def test_scalar_heads_shared_kv_tail(self):
        # No SWA vector: the last 6 of 12 layers share KV, and the estimator floors at one layer.
        probe = _probe(None, n_layers = 12, _shared_kv_layers = 6)
        assert probe._kv_gpu_share(7, 4096) == 0.0
        assert probe._kv_gpu_share(8, 4096) == pytest.approx(1 / 6)
        probe._shared_kv_layers = 12
        assert probe._kv_gpu_share(12, 4096) == 0.0
        assert probe._kv_gpu_share(13, 4096) == 1.0

    def test_target_kv_leaves_the_nextn_block_out(self):
        # Hybrid recurrent + one MTP block: the target cache skips it, so -ngl 2 holds none.
        probe = _probe(
            None,
            n_layers = 5,
            _n_kv_heads_by_layer = [0, 0, 0, 8, 8],
            _ssm_inner_size = 64,
            _full_attention_interval = 4,
            _nextn_predict_layers = 1,
        )
        assert probe._kv_gpu_share(2, 4096) == 0.0
        assert probe._kv_gpu_share(3, 4096) == 1.0

    def test_recurrent_state_follows_its_own_layers(self, monkeypatch):
        probe = _probe(None, n_layers = 4, _n_kv_heads_by_layer = [0, 0, 0, 8])
        monkeypatch.setattr(probe, "_rollback_state_bytes", lambda n_parallel = 1: 100)
        # -ngl 2: the attention layer is on the GPU, all three recurrent layers on the CPU.
        assert probe._kv_gpu_share(2, 4096, kv_bytes = 400) == pytest.approx(0.75)
        assert probe._kv_gpu_share(3, 4096, kv_bytes = 400) == pytest.approx((300 + 100 / 3) / 400)
        # Without the total, attention cells only.
        assert probe._kv_gpu_share(2, 4096) == 1.0

    def test_per_head_widths_pad_v_without_flash_attn(self):
        # Heads [8, 2], K = V = 256, f16: FA off pads layer 1's V to 8 heads, as the total does.
        probe = _probe(None, n_layers = 2, swa = 0, _n_kv_heads_by_layer = [8, 2])
        assert probe._kv_gpu_share(2, 4096) == pytest.approx(2 / 10)
        assert probe._kv_gpu_share(2, 4096, flash_attn = False) == pytest.approx(10 / 26)
        # The vector sums to the total the estimator prices.
        widths = probe._kv_head_layer_widths([8, 2], flash_attn = False)
        cells = 4096
        assert sum(widths) * cells == probe._estimate_kv_cache_bytes(4096, flash_attn = False)

    def test_nemotron_ffn_only_layers_are_not_recurrent(self, monkeypatch):
        probe = _probe(
            None,
            n_layers = 4,
            _n_kv_heads_by_layer = [0, 0, 8, 0],
            _feed_forward_length_by_layer = [0, 21504, 0, 0],
            _architecture = "nemotron_h_moe",
        )
        monkeypatch.setattr(probe, "_rollback_state_bytes", lambda n_parallel = 1: 100)
        # -ngl 2 offloads layer 3 only: one of the two Mamba layers (0, 3), no attention.
        assert probe._kv_gpu_share(2, 4096, kv_bytes = 400) == pytest.approx(0.5 * 100 / 400)
        # Elsewhere a zero-head layer stays recurrent: layers 0, 1, 3, one on the GPU.
        probe._architecture = "qwen3"
        assert probe._kv_gpu_share(2, 4096, kv_bytes = 400) == pytest.approx(100 / 3 / 400)

    def test_scalar_head_hybrid_uses_the_attention_interval(self):
        probe = _probe(None, n_layers = 8, _ssm_inner_size = 64, _full_attention_interval = 4)
        # Attention at layers 3 and 7; -ngl 5 offloads layers 4..7.
        assert probe._kv_gpu_share(5, 4096) == 0.5

    def test_unweighable_header_is_none(self):
        # Uniform attention, no per-layer heads: nothing to weigh, the caller falls back.
        probe = _probe(None, n_layers = 12)
        assert probe._kv_gpu_share(6, 4096) is None


class TestRouteKvShare:
    def test_weighted_share_replaces_the_layer_fraction(self):
        import routes.inference as ri

        probe = _probe(_GEMMA3_PATTERN)
        fraction = ri._gguf_offloaded_layer_fraction("manual", 9, 18, device_pin_governs = True)
        assert fraction == pytest.approx(9 / 19)
        share = ri._gguf_gpu_kv_share(probe, fraction, 18, _runtime(), 1)
        assert share == pytest.approx(14 / 27)

    def test_flash_attn_state_reaches_the_split(self):
        import routes.inference as ri

        seen = {}

        class _Probe:
            def _kv_gpu_share(self, gpu_layers, n_ctx, **kw):
                seen.update(kw, gpu_layers = gpu_layers)
                return 0.25

        runtime = _runtime()
        runtime.flash_attn = False
        assert ri._gguf_gpu_kv_share(_Probe(), 9 / 19, 18, runtime, 123) == 0.25
        assert seen["flash_attn"] is False and seen["kv_bytes"] == 123 and seen["gpu_layers"] == 9

    def test_falls_back_to_the_layer_fraction(self):
        import routes.inference as ri

        fraction = 6 / 13
        assert (
            ri._gguf_gpu_kv_share(_probe(None, n_layers = 12), fraction, 12, _runtime(), 1)
            == fraction
        )

        class _Broken:
            def _kv_gpu_share(self, *a, **k):
                raise RuntimeError("unread header")

        assert ri._gguf_gpu_kv_share(_Broken(), fraction, 12, _runtime(), 1) == fraction
        # Zero and full offload never ask the header.
        assert ri._gguf_gpu_kv_share(_Broken(), 0.0, 12, _runtime(), 1) == 0.0
        assert ri._gguf_gpu_kv_share(_Broken(), 1.0, 12, _runtime(), 1) == 1.0
