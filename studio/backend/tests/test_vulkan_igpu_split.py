# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Vulkan pin mixing a discrete card with a shared-memory iGPU: the card fills first.
Reported on an RX 7700 XT + Ryzen iGPU (27B: 8.3 GB on the iGPU, 1.05 t/s)."""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import importlib as _importlib  # noqa: E402


def _maybe_stub(name: str, builder):
    try:
        _importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = _types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", lambda: _types.ModuleType("structlog"))

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

MIB = 1024 * 1024

# The reporter's probe: VK0 the RX 7700 XT, VK1 the iGPU after its host reserve.
DGPU, IGPU = 0, 1
GPUS = [(DGPU, 11313), (IGPU, 14352)]
SHARED = {IGPU}


def test_a_model_either_device_holds_lands_on_the_discrete_card():
    picked, use_fit = LlamaCppBackend._select_gpus(
        6000 * MIB, GPUS, usable_fraction = 0.9, shared_gpu_ids = SHARED
    )
    assert (picked, use_fit) == ([DGPU], False)


def test_without_shared_ids_the_larger_shared_pool_still_wins():
    # CUDA / ROCm callers pass no shared set and keep the old ranking.
    picked, _ = LlamaCppBackend._select_gpus(6000 * MIB, GPUS, usable_fraction = 0.9)
    assert picked == [IGPU]


def test_a_model_that_needs_both_still_pins_both():
    picked, use_fit = LlamaCppBackend._select_gpus(
        15000 * MIB, GPUS, usable_fraction = 0.9, shared_gpu_ids = SHARED
    )
    assert (picked, use_fit) == ([DGPU, IGPU], False)


def test_split_aware_passes_the_shared_set_through():
    picked, _ = LlamaCppBackend._select_gpus_split_aware(
        6000 * MIB,
        GPUS,
        usable_fraction = 0.9,
        split_extra_bytes = 256 * MIB,
        shared_gpu_ids = SHARED,
    )
    assert picked == [DGPU]


def test_a_card_too_full_for_its_split_buffers_does_not_outrank_the_igpu():
    picked, use_fit = LlamaCppBackend._select_gpus(
        10000 * MIB,
        [(DGPU, 900), (IGPU, 14352)],
        usable_fraction = 0.9,
        per_device_overhead_bytes = 1024 * MIB,
        shared_gpu_ids = SHARED,
    )
    assert (picked, use_fit) == ([IGPU], False)


def test_every_auto_placement_sort_ranks_discrete_first():
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "key = lambda g: _gpu_usable(" not in src
    assert src.count("key = lambda g: _gpu_rank(") == 3
    assert "_gpu_rank(g, pin_fraction, _rank_floor_mib)" in src


def test_a_busy_card_too_small_alone_does_not_pull_in_the_igpu():
    # Another model runs on the card; the iGPU holds this one alone, so it goes there.
    picked, use_fit = LlamaCppBackend._select_gpus(
        12000 * MIB,
        GPUS,
        usable_fraction = 0.9,
        shared = frozenset({DGPU}),
        shared_gpu_ids = SHARED,
    )
    assert (picked, use_fit) == ([IGPU], False)


def test_the_fit_on_retry_takes_the_generated_split_back_out():
    src = inspect.getsource(LlamaCppBackend.load_model)
    retry = src[src.index("with forced --fit off; the fit estimate was optimistic") :]
    retry = retry[: retry.index("self._fit_load_mode_flags")]
    assert "_without_subsequence(_run, self._mixed_split_flags)" in retry
    assert "self._auto_tensor_split_emitted = None" in retry


split = LlamaCppBackend._discrete_first_split


def test_the_card_takes_the_layers_its_room_holds_and_the_igpu_the_rest():
    # 10 blocks of 1000 MiB + a 500 MiB output layer; the card keeps 300 for itself.
    shares = split(
        [DGPU, IGPU], {DGPU: 10180.0, IGPU: 12917.0}, SHARED, [1000.0] * 10 + [500.0], 300.0
    )
    # 9 layers then 2; each boundary half a layer early against float rounding.
    assert shares == [8.5, 2.5]


def test_the_heaviest_contiguous_run_decides_not_the_average():
    # A full-attention layer among window layers: an even byte share would give 7.
    layers = [100.0] * 4 + [3000.0] + [100.0] * 4
    assert split([DGPU, IGPU], {DGPU: 3500.0, IGPU: 9000.0}, SHARED, layers) == [5.5, 3.5]


def test_nothing_left_over_gives_the_igpu_nothing():
    assert split([DGPU, IGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, [1000.0] * 5 + [100.0]) == [
        5.5,
        0.5,
    ]


def test_the_split_is_positional_over_the_pin_order():
    shares = split([IGPU, DGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, [1000.0] * 12)
    assert shares == [1.5, 10.5]


def test_device_zero_carries_the_one_time_reserve():
    # iGPU first: the flat buffer lands there, so the card keeps its whole budget.
    shares = split(
        [IGPU, DGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, [1000.0] * 12, main_reserve_mib = 1500.0
    )
    assert shares == [1.5, 10.5]
    shares = split(
        [DGPU, IGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, [1000.0] * 12, main_reserve_mib = 1500.0
    )
    assert shares == [7.5, 4.5]


def test_overflow_is_shared_across_igpus_by_their_own_room():
    shares = split(
        [0, 1, 2],
        {0: 4000.0, 1: 3300.0, 2: 1300.0},
        {1, 2},
        [1000.0] * 7,
        per_device_mib = 100.0,
        pipeline_mib = 200.0,
    )
    assert shares == [2.5, 3.0, 1.5]


def test_only_a_mixed_pin_gets_a_split():
    usable = {0: 10000.0, 1: 12000.0}
    assert split([0, 1], usable, set(), [1000.0] * 15) is None
    assert split([0, 1], usable, {0, 1}, [1000.0] * 15) is None
    assert split([0, 1], usable, {1}, []) is None
    # A card that cannot hold even one layer: llama.cpp's split, as before.
    assert split([0, 1], {0: 500.0, 1: 12000.0}, {1}, [1000.0] * 15) is None


def test_the_launch_hands_placement_back_wherever_it_cannot_price_it():
    src = inspect.getsource(LlamaCppBackend._mixed_pin_split)
    for needle in (
        "tensor_parallel",
        "layer_min_gpus > 1",
        'spill_inputs["separate_draft_on_gpu"]',
        '"env_mmproj_unsized"',
        '"host_mmproj_bytes"',
        "_kv_offload_from_args(extra_args, env)",
        "_extra_args_have_tensor_split(extra_args, env)",
        "_extra_args_main_device(extra_args) is not None",
        '"LLAMA_ARG_DEVICE"',
        "_sidecar_adapter_paths(extra_args)",
        '"kv_layer_weights"',
        '"compute_buffer_flat"',
        '"extra_gpu_bytes"',
        '"env_mmproj_bytes"',
    ):
        assert needle in src, needle


def test_the_launch_records_what_it_emitted():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[
        src.index("_mixed_split = self._mixed_pin_split(") : src.index(
            "# Expose Prometheus /metrics"
        )
    ]
    assert "self._mixed_split_flags = [" in arm
    assert "self._auto_tensor_split_emitted = self._auto_split_fingerprint(" in arm


def test_an_igpu_run_that_overflows_its_room_leaves_the_split_off():
    # The pooled fit passes (10 of 10), but the 9 MiB layer would land on the 1 MiB iGPU.
    assert split([DGPU, IGPU], {DGPU: 9.0, IGPU: 1.0}, SHARED, [1.0, 9.0]) is None
