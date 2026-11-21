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
