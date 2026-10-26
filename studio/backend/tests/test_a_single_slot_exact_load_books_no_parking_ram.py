# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One slot never shares the cache, so nothing is parked and exact concurrency needs no parking
below two slots. Booking the whole pool of parking RAM for it anyway turned Auto off on a host
short of that RAM (an 8 GiB pool on 4 GiB free) and charged a running single-slot load's phantom
parks to every neighbour's exact check."""

import inspect

import core.inference.llama_cpp as llama_cpp

_GIB = 1024**3


def test_one_slot_writes_no_parks():
    assert llama_cpp._exact_parking_writes_mib(8 * _GIB, parallel = 1) == 0
    assert llama_cpp._exact_parking_writes_mib(8 * _GIB, draft_bytes = _GIB, parallel = 1) == 0


def test_more_slots_keep_the_whole_pool_bound():
    for slots in (2, 4, 8):
        assert llama_cpp._exact_parking_writes_mib(
            8 * _GIB, parallel = slots
        ) == llama_cpp._exact_parking_need_mib(8 * _GIB, parallel = slots)


def test_one_slot_is_never_short_of_a_parking_budget():
    shortfall = llama_cpp._exact_parking_shortfall_mib(
        8 * _GIB, args = ["--preempt-ram", "1024"], env = {}, parallel = 1
    )
    assert shortfall is None
    assert llama_cpp._exact_parking_shortfall_mib(
        8 * _GIB, args = ["--preempt-ram", "1024"], env = {}, parallel = 4
    ) is not None


def test_the_named_budget_still_sizes_for_the_whole_pool():
    # The launch's own --preempt-ram keeps its positive figure: one slot parks nothing anyway.
    assert llama_cpp._exact_parking_budget_mib(8 * _GIB, args = [], env = {}, parallel = 1) > 0


def test_every_exact_writes_figure_uses_the_single_slot_aware_size():
    source = inspect.getsource(llama_cpp.LlamaCppBackend)
    assert "_exact_parking_need_mib(" not in source, (
        "an exact-concurrency writes figure still books parking for one slot"
    )
