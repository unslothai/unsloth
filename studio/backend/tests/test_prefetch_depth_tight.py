# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A memory-tight streamed video request prefetches one group ahead; every other request keeps the built depth."""

from types import SimpleNamespace

import pytest
import torch

from core.inference import diffusion_offload_prefetch as P
from core.inference import video as V

MIB = 1 << 20
BLOCK = 483  # MiB: one LTX-2.5 int8 offload group (two blocks)


@pytest.fixture(autouse = True)
def _no_env(monkeypatch):
    monkeypatch.delenv(P.PREFETCH_DEPTH_ENV, raising = False)


def _prefetcher(
    depth = 2,
    groups = 4,
    block_mib = BLOCK,
    resident = (),
):
    gs = [SimpleNamespace(_unsloth_resident = i in resident) for i in range(groups)]
    pf = SimpleNamespace(
        groups = gs,
        nbytes = {id(g): block_mib * MIB for g in gs},
        depth = depth,
        window = (depth + 1) * block_mib * MIB,
        slot_of = {},
    )
    return pf


def _pipe(pf):
    dit = torch.nn.Module()
    setattr(dit, P.PREFETCHER_ATTR, pf)
    return SimpleNamespace(transformer = dit)


# The G4 measurement: a 6 GiB ballast leaves ~5.2 GiB; the admission need with one group ahead is ~5.8 GiB.
def test_tight_budget_picks_one():
    assert (
        P.tight_prefetch_depth(available_mib = 5300, required_mib = 5000, largest_group_mib = BLOCK) == 1
    )


def test_roomy_budget_keeps_two():
    assert (
        P.tight_prefetch_depth(available_mib = 7200, required_mib = 5800, largest_group_mib = BLOCK) == 2
    )
    # exactly one more group fits
    assert (
        P.tight_prefetch_depth(
            available_mib = 5000 + BLOCK, required_mib = 5000, largest_group_mib = BLOCK
        )
        == 2
    )


def test_short_even_at_one_still_picks_one():
    # the admission guard refuses this request; the prefetcher must not ask for more than one ahead meanwhile
    assert (
        P.tight_prefetch_depth(available_mib = 4000, required_mib = 5800, largest_group_mib = BLOCK) == 1
    )


def test_deeper_build_steps_down_one_group_at_a_time():
    got = P.tight_prefetch_depth(
        available_mib = 5000 + 2 * BLOCK, required_mib = 5000, largest_group_mib = BLOCK, planned_depth = 4
    )
    assert got == 3


@pytest.mark.parametrize(
    "value, want", [("2", 2), ("1", 1), ("3", 3), ("99", P.MAX_PREFETCH_DEPTH)]
)
def test_env_override_always_wins(monkeypatch, value, want):
    monkeypatch.setenv(P.PREFETCH_DEPTH_ENV, value)
    assert (
        P.tight_prefetch_depth(available_mib = 100, required_mib = 5800, largest_group_mib = BLOCK)
        == want
    )
    assert (
        P.tight_prefetch_depth(available_mib = 99999, required_mib = 10, largest_group_mib = BLOCK)
        == want
    )


def test_unparsable_env_is_no_override(monkeypatch):
    monkeypatch.setenv(P.PREFETCH_DEPTH_ENV, "deep")
    assert P.prefetch_depth_override() is None
    assert (
        P.tight_prefetch_depth(available_mib = 5300, required_mib = 5000, largest_group_mib = BLOCK) == 1
    )


def test_set_depth_shrinks_window_and_restores():
    pf = _prefetcher()
    built = pf.window
    assert P.set_request_prefetch_depth([pf], 1) == 1
    assert pf.depth == 1 and pf.window == built - BLOCK * MIB
    assert P.set_request_prefetch_depth([pf], 1) == 0  # unchanged
    assert P.set_request_prefetch_depth([pf], 2) == 1
    assert pf.depth == 2 and pf.window == built
    P.set_request_prefetch_depth([pf], 5)  # never deeper than built
    assert pf.depth == 2 and pf.window == built


def test_explicit_window_shrinks_by_the_dropped_group():
    pf = _prefetcher()
    pf.window = (1095 + 2 * BLOCK) * MIB  # H3-style explicit window: top group + two blocks
    P.set_request_prefetch_depth([pf], 1)
    assert pf.window == (1095 + BLOCK) * MIB


def test_slot_ring_keeps_its_depth():
    pf = _prefetcher()
    pf.slot_of = {1: 0}
    assert P.set_request_prefetch_depth([pf], 1) == 0
    assert pf.depth == 2


def test_resident_groups_do_not_size_the_window():
    pf = _prefetcher(groups = 3)
    pf.nbytes[id(pf.groups[0])] = 2000 * MIB
    pf.groups[0]._unsloth_resident = True
    assert P.streamed_group_mib([pf]) == BLOCK


def test_video_request_tight_picks_one_and_roomy_restores_two():
    pf = _prefetcher()
    pipe = _pipe(pf)
    kw = dict(floor_mib = 3300, denoise_mib = 2600)
    assert V.apply_request_prefetch_depth(pipe, available_mib = 5300, **kw) == 1
    assert pf.depth == 1
    assert V.apply_request_prefetch_depth(pipe, available_mib = 12000, **kw) == 2
    assert pf.depth == 2


def test_video_request_env_wins(monkeypatch):
    pf = _prefetcher()
    pipe = _pipe(pf)
    monkeypatch.setenv(P.PREFETCH_DEPTH_ENV, "2")
    assert (
        V.apply_request_prefetch_depth(pipe, available_mib = 1000, floor_mib = 3300, denoise_mib = 2600)
        == 2
    )
    assert pf.depth == 2


def test_video_request_without_a_prefetcher_is_a_no_op():
    assert (
        V.apply_request_prefetch_depth(
            SimpleNamespace(transformer = torch.nn.Module()), available_mib = 1, floor_mib = 1
        )
        is None
    )


def test_measured_peak_can_admit_depth_two():
    # a recorded peak far below the priced headroom leaves room for the second group ahead
    pf = _prefetcher()
    got = V.apply_request_prefetch_depth(
        _pipe(pf), available_mib = 5300, floor_mib = 3300, denoise_mib = 2600, measured_extra_mib = 400
    )
    assert got == 2
