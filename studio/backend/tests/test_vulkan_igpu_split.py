# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Vulkan pin that mixes a discrete card with a shared-memory iGPU.

The iGPU reports its whole shared pool as free, so it outranked the discrete card in
``_select_gpus`` and, with no ``--tensor-split``, took the larger share of llama.cpp's
free-memory layer split. Reported on an RX 7700 XT (12 GB) + Ryzen iGPU, Windows:
Qwen3.8-27B UD-IQ4_XS loaded 4.8 GB on the card and 8.3 GB on the iGPU, 1.05 t/s,
against 4.4 t/s with the iGPU deselected.
"""

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
    # The old ranking, kept for every caller that has no shared set (CUDA, ROCm).
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
    # 13.26 GiB of weights + ~0.6 GiB of KV at 32K on iq4_nl.
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
    assert "_TENSOR_SPLIT_FLAGS" in arm
    assert "_spill_inputs is not None" in arm


def test_non_layer_bytes_stay_off_the_discrete_share():
    # The flat compute buffer, context and projector are not divided by the split,
    # so a card filled to its whole budget would run past it under --fit off.
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU],
        {DGPU: 10180.0, IGPU: 12917.0},
        SHARED,
        layered_mib = 14200.0,
        per_device_mib = 300.0,
        reserve_mib = 1500.0,
    )
    assert shares == [8380.0, 5820.0]


def test_the_launch_reserves_the_non_layer_bytes():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    for key in ("compute_buffer_flat", "soft_overhead", "extra_gpu_bytes"):
        assert key in arm
