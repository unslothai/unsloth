# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A --fit off pin over discrete cards: llama.cpp cuts layers by count, weighted by free memory.
Uneven layers (Unsloth Dynamic quants keep some at higher precision) can push that cut past the
smaller card, so Studio splits by bytes, or lets --fit place the model when no cut fits (#19137)."""

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
split = LlamaCppBackend._byte_balanced_split

# Inkling-2-Mini UD-Q8_K_XL at 32K with 3 slots as Studio prices each row (weights + its cache):
# 24 layers, the last four about 2 GB, then the output. Rooms leave 528 MiB per card, the
# 186 MiB flat compute on card 0 and the 1024 MiB split overhead on card 1.
INKLING_ROWS = (
    [613.0] * 5
    + [728.0]
    + [646.0] * 5
    + [988.0]
    + [646.0] * 5
    + [988.0, 613.0, 613.0]
    + [2011.0, 2011.0, 2011.0, 2125.0, 884.0]
)
RESERVES = (528.0, 186.0, 1024.0)


def _usable(free_by_idx, floor = 1024.0):
    return {i: f - floor for i, f in free_by_idx.items()}


def test_an_even_model_keeps_llama_cpp_own_split():
    free = {0: 16000.0, 1: 12000.0}
    assert split([0, 1], free, _usable(free), [1000.0] * 20, 0.0, 0.0, 0.0) is None


def test_uneven_layers_that_overflow_the_smaller_card_are_cut_by_bytes():
    # #19137's pair with 16384 + 12288 MiB free: by count, layers 15-23 and the output
    # (11.9 GB) land on the 12 GB card.
    free = {0: 16384.0, 1: 12288.0}
    shares = split([0, 1], free, _usable(free), INKLING_ROWS, *RESERVES)
    assert shares == [19.5, 5.5]


def test_the_byte_cut_fills_each_card_within_its_room():
    free = {0: 16384.0, 1: 12288.0}
    usable = _usable(free)
    shares = split([0, 1], free, usable, INKLING_ROWS, *RESERVES)
    first = int(shares[0] + 0.5)
    rooms = [usable[0] - RESERVES[0] - RESERVES[1], usable[1] - RESERVES[0] - RESERVES[2]]
    assert sum(INKLING_ROWS[:first]) <= rooms[0]
    assert sum(INKLING_ROWS[first:]) <= rooms[1]


def test_no_cut_that_fits_asks_for_the_fitter():
    # 15360 + 11264 MiB: the four large layers and the output need 9042 MiB, the small card's
    # room is 8688. A count split put 13.4 GB there on capped B200s.
    free = {0: 15360.0, 1: 11264.0}
    assert split([0, 1], free, _usable(free), INKLING_ROWS, *RESERVES) == []


def test_one_card_or_no_rows_is_not_a_split():
    free = {0: 16000.0}
    assert split([0], free, _usable(free), [1000.0] * 4) is None
    assert split([0, 1], {0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}, []) is None


def test_the_cut_is_positional_over_the_pin_order():
    # Three cards, the large layers last: by count the last card holds 11 GB of its 10.7;
    # by bytes the two large layers go to different cards.
    free = {0: 12000.0, 1: 12000.0, 2: 12000.0}
    usable = _usable(free)
    rows = [500.0] * 16 + [4500.0, 4500.0]
    shares = split([0, 1, 2], free, usable, rows, 0.0, 0.0, 0.0)
    assert shares is not None and sum(shares) == len(rows)
    counts = [int(shares[0] + 0.5), int(shares[1]), int(shares[2] - 0.5)]
    assert counts[2] == 1 and rows[sum(counts[:2]) - 1] == 4500.0
    start = 0
    for c, i in zip(counts, (0, 1, 2)):
        assert sum(rows[start : start + c]) <= usable[i]
        start += c


def _priced_backend(rows_mib):
    blocks = [
        _types.SimpleNamespace(spillable_bytes = int(m * MIB), resident_bytes = 0)
        for m in rows_mib[:-1]
    ]
    layout = _types.SimpleNamespace(
        complete = True,
        blocks = blocks,
        has_excluded_blocks = False,
        n_attention_layers = len(blocks),
        lm_head_bytes = int(rows_mib[-1] * MIB),
        other_resident_bytes = 0,
    )
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._tensor_spill_layout = lambda _path: layout
    backend._kv_lora_rank = backend._ssm_inner_size = backend._sliding_window_pattern = None
    backend._sliding_window = None
    backend._nextn_predict_layers = 0
    return backend


def _pin(
    backend,
    free,
    shared = (),
):
    inputs = {
        "separate_draft_on_gpu": False,
        "model_path": "m.gguf",
        "kv_cache_bytes": 0,
        "kv_layer_weights": [],
        "mtp_will_engage": False,
        "compute_buffer_flat": 0,
        "extra_gpu_bytes": 0,
        "ctx_compute_per_device": 0,
        "soft_overhead": 0,
        "gpus": list(free.items()),
        "gpu_usable_mib": _usable(free),
    }
    return backend._mixed_pin_split(
        list(free),
        inputs,
        set(shared),
        [],
        {},
        layer_min_gpus = 1,
        tensor_parallel = False,
        pin_owns_devices = True,
    )


def test_a_discrete_pin_gets_the_byte_cut_from_its_layout():
    rows = [500.0] * 16 + [3000.0, 3000.0, 100.0]
    # By count, 9.5 of 19 rows go to card 1, which then holds both large layers (6.5 GB).
    assert _pin(_priced_backend(rows), {0: 12000.0, 1: 8000.0}) is not None
    assert _pin(_priced_backend([500.0] * 19), {0: 12000.0, 1: 8000.0}) is None


def test_the_launch_lets_the_fitter_place_a_pin_no_cut_fits():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = None") :]
    arm = arm[: arm.index('if gpu_memory_mode == "manual" and gpu_layers >= 0:')]
    assert "and not use_fit" in arm
    assert "use_fit, gpu_indices, _mixed_split = True, None, None" in arm
