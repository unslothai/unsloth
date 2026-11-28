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
        share = ri._gguf_gpu_kv_share(probe, fraction, 18, _runtime())
        assert share == pytest.approx(14 / 27)

    def test_falls_back_to_the_layer_fraction(self):
        import routes.inference as ri

        fraction = 6 / 13
        assert (
            ri._gguf_gpu_kv_share(_probe(None, n_layers = 12), fraction, 12, _runtime()) == fraction
        )

        class _Broken:
            def _kv_gpu_share(self, *a, **k):
                raise RuntimeError("unread header")

        assert ri._gguf_gpu_kv_share(_Broken(), fraction, 12, _runtime()) == fraction
        # Zero and full offload never ask the header.
        assert ri._gguf_gpu_kv_share(_Broken(), 0.0, 12, _runtime()) == 0.0
        assert ri._gguf_gpu_kv_share(_Broken(), 1.0, 12, _runtime()) == 1.0
