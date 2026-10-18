# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The hybrid cache's per-layer heads and V width, lazy-mode emission on a default or unread
selection, the populated window under --swa-full, one layer-count bound at the reader, and the
paging watch on a ROCm host."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference.llama_cpp import LlamaCppBackend, _kv_bytes_per_elem  # noqa: E402
from core.inference.offload_layout import LLAMA_MAX_LAYERS  # noqa: E402
from core.inference.offload_planner import _fit_fallback_placement  # noqa: E402
from test_offload_cost_gate import GIB, gated, swa_split_layout  # noqa: E402

MIB = 1024**2


def _hybrid(
    heads,
    *,
    key_len = 128,
    val_len = 128,
) -> LlamaCppBackend:
    """Jamba-shaped: the zeros of head_count_kv ARE the recurrent rows."""
    b = LlamaCppBackend()
    for key, value in dict(
        _architecture = "jamba",
        _n_layers = len(heads),
        _n_heads = 32,
        _n_kv_heads = max(heads),
        _n_kv_heads_by_layer = list(heads),
        _embedding_length = 4096,
        _kv_key_length = key_len,
        _kv_value_length = val_len,
        _ssm_inner_size = 8192,
        _ssm_state_size = 16,
        _ssm_conv_kernel = 4,
    ).items():
        setattr(b, key, value)
    return b


def _per_cell(b, **kw) -> float:
    """Cache bytes per cell: the recurrent state does not grow with the context."""
    lo = b._estimate_kv_cache_bytes(8192, **kw)
    hi = b._estimate_kv_cache_bytes(16384, **kw)
    return (hi - lo) / 8192


# ------------------------------------------------ findings 1 / 3: the hybrid attention rows


def test_a_hybrid_cache_sums_each_attention_rows_own_heads():
    """[0,0,0,8,0,0,0,4] is 12 heads of cache; charging both rows the widest one priced 16."""
    b = _hybrid([0, 0, 0, 8, 0, 0, 0, 4])
    assert b._hybrid_layer_split(8) is not None
    assert _per_cell(b) == 12 * (128 + 128) * 2


def test_a_legacy_hybrid_prices_v_at_its_own_element_size():
    """Without flash attention a quantized V is rewritten to f16 (bpe_v), and the legacy branch
    priced V at K's quantized width: an under-reservation."""
    b = _hybrid([0, 0, 0, 8, 0, 0, 0, 8], key_len = None, val_len = None)
    head_dim = b._legacy_head_dim()
    bpe_k = _kv_bytes_per_elem("q8_0")
    got = _per_cell(b, cache_type_kv = "q8_0", flash_attn = False)
    assert abs(got - 16 * head_dim * (bpe_k + 2)) < 1e-6
    # Flash on keeps both at q8_0.
    got_fa = _per_cell(b, cache_type_kv = "q8_0", flash_attn = True)
    assert abs(got_fa - 16 * head_dim * 2 * bpe_k) < 1e-6


# ------------------------------------------------ findings 2 / 5: lazy-mode emission


def _lazy(
    monkeypatch,
    *,
    vulkan_rows = None,
    unified = (),
    integrated = (),
):
    b = LlamaCppBackend()
    b._architecture = "gemma4"
    b._tensor_spill_layout = lambda *a, **k: type("L", (), {"per_layer_embd_bytes": 8 * GIB})()
    monkeypatch.setattr(
        LlamaCppBackend, "_run_vulkan_probe", staticmethod(lambda _binary = None: vulkan_rows)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set(unified))
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: set(integrated))
    )
    return b


def _emit(
    b,
    gpu_indices,
    *,
    vulkan = False,
):
    return b._studio_lazy_mode(
        model_path = "/m.gguf",
        server_caps = {"supports_lazy_mode": True, "lazy_mode_flag": "--lazy-mode"},
        gpu_indices = gpu_indices,
        binary = "/fake/llama-server",
        is_vulkan_backend = vulkan,
        extra_args = [],
        env = {},
    )


def test_a_default_hip_selection_beside_an_apu_is_left_to_auto(monkeypatch):
    """The child's default list drops the APU beside a discrete card (src/llama.cpp adds
    integrated devices only when no discrete one was found), so auto pages the table; "off" made
    it resident while the planner priced it paged."""
    b = _lazy(monkeypatch, unified = {1})
    assert _emit(b, None) is None
    # A mask holding both cards: the child still drops the APU.
    assert _emit(b, [0, 1]) == "on"
    # Only the APU: off, as auto would resolve it.
    assert _emit(b, [1]) == "off"
    assert _emit(b, [0]) == "on"


def test_an_unread_vulkan_type_is_left_to_auto(monkeypatch):
    rows = [{"index": 0, "is_igpu": False, "type_known": False}]
    b = _lazy(monkeypatch, vulkan_rows = rows)
    assert _emit(b, [0], vulkan = True) is None
    known = [{"index": 0, "is_igpu": True, "type_known": True}]
    assert _emit(_lazy(monkeypatch, vulkan_rows = known), [0], vulkan = True) == "off"


# ------------------------------------------------ finding 6: --swa-full traffic


def test_a_full_size_window_is_charged_its_populated_prefix():
    """--swa-full reserves the window layers at the full context (llama-kv-cache-iswa.cpp:
    size_swa = size_base), but a 2048-token request has populated 2048 cells of them."""
    layout = swa_split_layout()
    n_ctx = 131072
    per_tok = 5 * GIB // n_ctx  # every layer full-context under --swa-full

    def full(ctx, slots):
        return per_tok * ctx

    shape = dict(workload_prompt_tokens = 2048, workload_generated_tokens = 0)
    args = dict(quantised = False, kv_bytes_floor = 5 * GIB, kv_on_host = False)
    placed = _fit_fallback_placement(
        layout,
        gated(kv_bytes_at = full, kv_swa_bytes_floor = 4 * GIB, **shape),
        8 * GIB,
        n_ctx,
        **args,
    )
    assert placed is not None and placed.kv_host_bytes > 0
    # The moved share can never exceed what the request populated across EVERY layer.
    assert placed.kv_host_bytes <= full(2048, 1), placed.kv_host_bytes / MIB


# ------------------------------------------------ findings 4 / 8 / 9: one layer-count bound


def test_a_layer_count_past_llamas_cap_is_absent_metadata():
    from test_kv_cache_estimation import _backend_from_gguf

    fields = {"attention.head_count": 32, "attention.head_count_kv": 8, "embedding_length": 4096}
    ok = _backend_from_gguf("llama", dict(block_count = LLAMA_MAX_LAYERS, **fields))
    assert ok._n_layers == LLAMA_MAX_LAYERS and ok._can_estimate_kv()
    over = _backend_from_gguf("llama", dict(block_count = LLAMA_MAX_LAYERS + 1, **fields))
    assert over._n_layers is None
    assert not over._can_estimate_kv()
    assert over._estimate_kv_cache_bytes(8192) == 0
    # Set by any other route, the estimator still refuses it.
    ok._n_layers = 1 << 40
    assert not ok._can_estimate_kv()


# ------------------------------------------------ finding 10: the paging watch on ROCm


def test_the_paging_watch_stands_down_on_a_rocm_host(monkeypatch):
    from core.inference import llama_cpp as llama_mod

    monkeypatch.setattr(llama_mod.sys, "platform", "win32")
    sampled = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_nvidia_smi_free_total_mib",
        staticmethod(lambda gpu_indices = None: sampled.append(gpu_indices) or [(0, 1000, 24000)]),
    )
    monkeypatch.setattr(LlamaCppBackend, "_nvidia_link_gib_s", staticmethod(lambda *a, **k: 25.0))
    b = LlamaCppBackend()
    kw = dict(gpu_indices = [0], model_bytes = 8 * GIB, fully_gpu_offloaded = True)
    monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False))
    assert b._windows_sysmem_fallback_watch(**kw) is not None, "the CUDA control must watch"
    sampled.clear()
    monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True))
    assert b._windows_sysmem_fallback_watch(**kw) is None
    assert not sampled
