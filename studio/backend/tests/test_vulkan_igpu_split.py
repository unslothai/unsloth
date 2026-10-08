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


def test_the_split_fills_the_discrete_card_first():
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU],
        {DGPU: 10180.0, IGPU: 12917.0},
        SHARED,
        layered_mib = 14200.0,
        per_device_mib = 300.0,
    )
    assert shares == [9880.0, 4320.0]
    assert shares[0] > shares[1]


def test_nothing_left_over_gives_the_igpu_nothing():
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, layered_mib = 8000.0
    )
    assert shares == [8000.0, 0.0]


def test_the_split_is_positional_over_the_pin_order():
    shares = LlamaCppBackend._discrete_first_split(
        [IGPU, DGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, layered_mib = 12000.0
    )
    assert shares == [2000.0, 10000.0]


def test_overflow_is_shared_across_igpus_by_room():
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2], {0: 4000.0, 1: 3000.0, 2: 1000.0}, {1, 2}, layered_mib = 8000.0
    )
    assert shares == [4000.0, 3000.0, 1000.0]


def test_only_a_mixed_pin_gets_a_split():
    usable = {0: 10000.0, 1: 12000.0}
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, set(), 15000.0) is None
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, {0, 1}, 15000.0) is None
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, {1}, 0.0) is None


def test_the_launch_emits_it_only_where_nothing_else_owns_the_split():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    assert "not tensor_parallel" in arm
    assert "_extra_args_have_tensor_split(extra_args, env)" in arm
    assert "_spill_inputs is not None" in arm


def test_non_layer_bytes_stay_off_the_discrete_share():
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU],
        {DGPU: 10180.0, IGPU: 12917.0},
        SHARED,
        layered_mib = 14200.0,
        per_device_mib = 300.0,
        main_reserve_mib = 1500.0,
    )
    assert shares == [8380.0, 5820.0]


def test_the_launch_reserves_the_non_layer_bytes():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    for key in ("compute_buffer_flat", "soft_overhead", "extra_gpu_bytes"):
        assert key in arm


def test_a_card_too_full_for_its_split_buffers_does_not_outrank_the_igpu():
    picked, use_fit = LlamaCppBackend._select_gpus(
        10000 * MIB,
        [(DGPU, 900), (IGPU, 14352)],
        usable_fraction = 0.9,
        per_device_overhead_bytes = 1024 * MIB,
        shared_gpu_ids = SHARED,
    )
    assert (picked, use_fit) == ([IGPU], False)


def test_the_projector_is_reserved_not_split():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    assert '_spill_inputs["model_size"] - mmproj_size' in arm


def test_every_auto_placement_sort_ranks_discrete_first():
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "key = lambda g: _gpu_usable(" not in src
    assert src.count("key = lambda g: _gpu_rank(") == 3
    assert "_gpu_rank(g, pin_fraction, _rank_floor_mib)" in src


def test_device_zero_carries_the_one_time_reserve():
    # iGPU first: the flat buffer lands there, so the card keeps its whole budget.
    shares = LlamaCppBackend._discrete_first_split(
        [IGPU, DGPU],
        {DGPU: 10000.0, IGPU: 12000.0},
        SHARED,
        layered_mib = 12000.0,
        main_reserve_mib = 1500.0,
    )
    assert shares == [2000.0, 10000.0]


def test_igpu_room_keeps_its_own_buffers():
    # Overflow is divided by room AFTER each iGPU's compute buffer and pipeline step.
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2],
        {0: 4000.0, 1: 3300.0, 2: 1300.0},
        {1, 2},
        layered_mib = 7000.0,
        per_device_mib = 100.0,
        pipeline_mib = 200.0,
    )
    assert shares == [3900.0, 2325.0, 775.0]


def test_a_gpu_resident_drafter_keeps_llama_cpp_split_and_status_reports_ours():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("# Expose Prometheus /metrics")]
    assert 'not _spill_inputs["separate_draft_on_gpu"]' in arm
    assert "self._auto_tensor_split_emitted = self._auto_split_fingerprint(" in arm


def test_an_inherited_projector_or_device_list_is_respected():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    for needle in (
        '"env_mmproj_bytes"',
        '"env_mmproj_unsized"',
        '"host_mmproj_bytes"',
        "_kv_offload_from_args(extra_args, env)",
        "_extra_args_main_device(extra_args) is None",
        '"LLAMA_ARG_DEVICE"',
        "_layer_min_gpus <= 1",
    ):
        assert needle in arm


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
