# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Both GGUF layouts of a looped Inkling 2 read as the same model: the physical-layer layout
(`inkling2_loop`, block_count 24 + loop keys) expands to the cache slots llama.cpp allocates,
matching the visit-indexed layout (`inkling2`, block_count 48)."""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import LlamaCppBackend

N_PHYS, START, END, COUNT = 24, 6, 18, 3
N_SLOT = N_PHYS + (COUNT - 1) * (END - START)
# Physical sliding pattern: every 6th layer is full attention.
PHYS_SWA = [(i + 1) % 6 != 0 for i in range(N_PHYS)]


def _visit_layers():
    """Physical layer of each visit: prelude, three passes over the core, coda."""
    core = list(range(START, END))
    return list(range(START)) + core * COUNT + list(range(END, N_PHYS))


def _write(path, arch, layers, loop):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch)
    w.add_context_length(1048576)
    w.add_block_count(len(layers))
    w.add_embedding_length(2304)
    w.add_feed_forward_length(1152)
    w.add_head_count([24] * len(layers))
    w.add_head_count_kv([8] * len(layers))
    w.add_key_length(128)
    w.add_value_length(128)
    w.add_sliding_window(1024)
    w.add_array(f"{arch}.attention.sliding_window_pattern", [PHYS_SWA[p] for p in layers])
    w.add_expert_count(128)
    w.add_expert_used_count(8)
    w.add_uint32(f"{arch}.attn_res.block_size", 6)
    for key, value in loop.items():
        w.add_uint32(f"{arch}.{key}", value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _read(path):
    b = LlamaCppBackend()
    b._read_gguf_metadata(str(path))
    return b


@pytest.fixture
def layouts(tmp_path):
    legacy = _write(
        tmp_path / "legacy.gguf",
        "inkling2",
        _visit_layers(),
        {"loop.core_start": START, "loop.core_end": END, "num_loops": COUNT, "loop.gate_rank": 64},
    )
    mainline = _write(
        tmp_path / "mainline.gguf",
        "inkling2_loop",
        list(range(N_PHYS)),
        {"loop.start": START, "loop.end": END, "loop.count": COUNT, "loop.gate_rank": 64},
    )
    return _read(legacy), _read(mainline)


def test_the_physical_layout_expands_to_the_cache_slots(layouts):
    legacy, mainline = layouts
    assert legacy._n_layers == N_SLOT
    assert mainline._n_layers == N_SLOT
    assert sorted(mainline._sliding_window_pattern) == sorted(legacy._sliding_window_pattern)
    assert len(mainline._n_kv_heads_by_layer) == len(mainline._n_heads_by_layer) == N_SLOT
    # Slot n_phys + k is physical layer start + k % (end - start).
    assert mainline._sliding_window_pattern[N_PHYS:] == PHYS_SWA[START:END] * (COUNT - 1)


@pytest.mark.parametrize("ctx", [32768, 262144, 1048576])
@pytest.mark.parametrize("ct", ["f16", "q8_0"])
def test_kv_and_compute_estimates_match_between_layouts(layouts, ctx, ct):
    legacy, mainline = layouts
    for kw in ({}, {"n_parallel": 4}, {"swa_full": True}):
        assert mainline._estimate_kv_cache_bytes(ctx, ct, **kw) == legacy._estimate_kv_cache_bytes(
            ctx, ct, **kw
        )
    for fa in (True, False):
        assert mainline._compute_buffer_ctx_bytes(ctx, 512, ct, flash_attn = fa) == (
            legacy._compute_buffer_ctx_bytes(ctx, 512, ct, flash_attn = fa)
        )
    assert mainline._inkling2_bias_heads() == legacy._inkling2_bias_heads()


def test_the_loop_layout_takes_the_relative_bias_branch(layouts, tmp_path):
    _, mainline = layouts
    # The same dims under an unrelated architecture take the generic attention rate.
    generic = _read(_write(tmp_path / "plain.gguf", "llama", list(range(N_PHYS)), {}))
    assert mainline._compute_buffer_ctx_bytes(
        800000, 512, "q8_0"
    ) > 10 * generic._compute_buffer_ctx_bytes(800000, 512, "q8_0")


def test_a_non_looped_inkling2_is_unchanged(tmp_path):
    small = _read(_write(tmp_path / "small.gguf", "inkling2", list(range(N_PHYS)), {}))
    assert small._n_layers == N_PHYS
    assert small._sliding_window_pattern == PHYS_SWA
    assert small._loop_count is None


def test_a_loop_layout_without_valid_loop_keys_is_left_alone(tmp_path):
    bad = _read(
        _write(
            tmp_path / "bad.gguf",
            "inkling2_loop",
            list(range(N_PHYS)),
            {"loop.start": 20, "loop.end": 18, "loop.count": 3},
        )
    )
    assert bad._n_layers == N_PHYS


# ---------------------------------------------------------------------------
# The -ngl / -ts / --n-cpu-moe unit. llama.cpp counts it in physical layers
# (llama_model_n_layer_weights): 24 for Inkling-2-Mini on either layout, while
# KV and recurrent state stay per cache slot (48).


def _row_bytes(p):
    """Distinct per physical layer, so the offload order is visible in the layout."""
    return 32 * (p + 1)


def _write_with_weights(
    path,
    arch,
    n_blocks,
    loop,
    *,
    n_phys = N_PHYS,
):
    """A header plus the physical layers' tensors: both real layouts name them blk.0 .. blk.(n_phys - 1)."""
    import numpy as np
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch)
    w.add_context_length(1048576)
    w.add_block_count(n_blocks)
    w.add_embedding_length(2304)
    w.add_head_count([24] * n_blocks)
    w.add_head_count_kv([8] * n_blocks)
    w.add_key_length(128)
    w.add_value_length(128)
    w.add_sliding_window(1024)
    w.add_expert_count(128)
    w.add_expert_used_count(8)
    for key, value in loop.items():
        w.add_uint32(f"{arch}.{key}", value)
    for p in range(n_phys):
        w.add_tensor(f"blk.{p}.attn_q.weight", np.zeros(8, dtype = np.float32))
        w.add_tensor(
            f"blk.{p}.ffn_gate_up_exps.weight", np.zeros(_row_bytes(p) // 4, dtype = np.float32)
        )
    w.add_tensor("output.weight", np.zeros(16, dtype = np.float32))
    w.add_tensor("token_embd.weight", np.zeros(16, dtype = np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


LEGACY_LOOP = {
    "loop.core_start": START,
    "loop.core_end": END,
    "num_loops": COUNT,
    "loop.gate_rank": 64,
}
MAINLINE_LOOP = {"loop.start": START, "loop.end": END, "loop.count": COUNT, "loop.gate_rank": 64}


@pytest.fixture
def weighted(tmp_path):
    return {
        "legacy": _write_with_weights(tmp_path / "legacy.gguf", "inkling2", N_SLOT, LEGACY_LOOP),
        "mainline": _write_with_weights(
            tmp_path / "mainline.gguf", "inkling2_loop", N_PHYS, MAINLINE_LOOP
        ),
        "small": _write_with_weights(tmp_path / "small.gguf", "inkling2", N_PHYS, {}),
    }


@pytest.mark.parametrize("which", ["legacy", "mainline"])
def test_the_offload_unit_is_the_physical_layer_and_kv_keeps_the_slots(weighted, which):
    b = _read(weighted[which])
    assert b._n_layers == N_SLOT
    assert b.n_weight_layers == N_PHYS
    assert b.n_layers == N_PHYS
    assert b.n_moe_layers == N_PHYS
    assert len(b._n_kv_heads_by_layer) == N_SLOT
    # KV: one cache per slot, 48 x 8 heads x (128 + 128) x 2 bytes per token at f16, both layouts.
    assert b._estimate_kv_cache_bytes(4096, "f16", swa_full = True) >= N_SLOT * 8 * 256 * 2 * 4096


@pytest.mark.parametrize("which", ["legacy", "mainline"])
def test_the_ngl_comparisons_use_physical_layers(weighted, which):
    b = _read(weighted[which])
    # -ngl 25 (> 24) offloads every layer plus the output; -ngl 24 leaves the output on the host.
    for full, ngl in ((True, 25), (False, N_PHYS)):
        assert (
            b._offloads_every_layer(gpu_memory_mode = "manual", gpu_layers = ngl, extra_args = None)
            is full
        )
        assert b._partially_offloads_layers(["-ngl", str(ngl)], env = {}) is (not full)


@pytest.mark.parametrize("which", ["legacy", "mainline"])
def test_the_staged_dims_report_physical_layers(weighted, which):
    from utils.models.gguf_metadata import _parse_gguf_staged_dims

    dims = _parse_gguf_staged_dims(weighted[which])
    assert dims["layer_count"] == N_PHYS
    assert dims["moe_layer_count"] == N_PHYS


@pytest.mark.parametrize("which", ["legacy", "mainline"])
def test_the_routes_layer_count_is_physical(weighted, which):
    from routes.inference import _probe_offload_layers
    assert _probe_offload_layers(_read(weighted[which])) == N_PHYS


@pytest.mark.parametrize("which", ["legacy", "mainline"])
def test_the_offload_layout_lists_physical_layers_in_order(weighted, which):
    from core.inference.offload_layout import layout_from_gguf

    layout = layout_from_gguf(weighted[which])
    assert layout.complete
    assert layout.n_layers == N_PHYS
    assert [blk.index for blk in layout.blocks] == list(range(N_PHYS))
    assert [blk.spillable_bytes for blk in layout.blocks] == [_row_bytes(p) for p in range(N_PHYS)]
    assert not layout.has_excluded_blocks
    assert layout.n_cache_slots == layout.n_attention_layers == N_SLOT
    assert sorted(set(layout.cache_slot_rows)) == list(range(N_PHYS))
    assert layout.kv_bytes_per_token_f16 == N_SLOT * 8 * (128 + 128) * 2
    if which == "legacy":
        # Visit order: prelude + first core pass, two repeat passes, coda.
        assert layout.cache_slot_rows[:END] == tuple(range(END))
        assert layout.cache_slot_rows[N_SLOT - (N_PHYS - END) :] == tuple(range(END, N_PHYS))
    else:
        assert layout.cache_slot_rows[:N_PHYS] == tuple(range(N_PHYS))
    assert layout.cache_slot_rows.count(START) == COUNT
    assert layout.cache_slot_rows.count(0) == 1


def test_the_two_layouts_price_the_same_offload(weighted):
    from core.inference.offload_layout import layout_from_gguf

    legacy, mainline = layout_from_gguf(weighted["legacy"]), layout_from_gguf(weighted["mainline"])
    assert legacy.blocks == mainline.blocks
    assert legacy.kv_bytes_per_token_f16 == mainline.kv_bytes_per_token_f16
    assert sorted(legacy.cache_slot_rows) == sorted(mainline.cache_slot_rows)


def test_the_per_device_split_folds_slot_caches_onto_physical_rows(weighted):
    from core.inference.offload_layout import layout_from_gguf
    from core.inference.offload_planner import PlanOptions, _per_device_usage

    layout = layout_from_gguf(weighted["mainline"])
    opts = PlanOptions()
    weights = [1] * N_SLOT
    err, usage, slots = _per_device_usage(
        layout,
        opts,
        4096,
        set(),
        False,
        [1 << 34, 1 << 34],
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = weights,
    )
    assert err is None
    # -ts rows are the 24 physical layers plus the output row.
    assert sorted(r for rows in slots for r in rows) == list(range(N_PHYS + 1))
    # A slot-length vector is accepted (not abstained on) and the whole cache lands on some row.
    assert sum(usage) >= layout.kv_bytes(4096)


def test_a_non_looped_inkling2_keeps_block_count_as_the_unit(weighted):
    from core.inference.offload_layout import layout_from_gguf
    from routes.inference import _probe_offload_layers
    from utils.models.gguf_metadata import _parse_gguf_staged_dims

    b = _read(weighted["small"])
    assert b._n_weight_layers is None
    assert b._n_layers == b.n_layers == b.n_weight_layers == N_PHYS
    assert _probe_offload_layers(b) == N_PHYS
    assert _parse_gguf_staged_dims(weighted["small"])["layer_count"] == N_PHYS
    layout = layout_from_gguf(weighted["small"])
    assert layout.cache_slot_rows == ()
    assert layout.n_cache_slots == layout.n_layers
