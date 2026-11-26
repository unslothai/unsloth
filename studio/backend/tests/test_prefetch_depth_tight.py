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


def _slot_prefetcher(
    monkeypatch,
    depth = 2,
    groups = 5,
    nbytes = 10,
):
    gs = [
        SimpleNamespace(offload_leader = None, cpu_param_dict = {}, _unsloth_resident = False)
        for _ in range(groups)
    ]
    pf = P.GroupPrefetcher.__new__(P.GroupPrefetcher)
    pf.module, pf.groups, pf.depth, pf.active = object(), gs, depth, False
    pf.nbytes = {id(g): nbytes for g in gs}
    pf.window = (depth + 1) * nbytes
    pf.slot_of, pf.carried = {}, set()
    monkeypatch.setattr(P, "_group_tensors", lambda g: [])
    monkeypatch.setattr(P, "_packed_bytes", lambda sources: nbytes)
    return pf


def test_a_ring_built_after_a_tight_request_is_sized_for_the_built_depth(monkeypatch):
    pf = _slot_prefetcher(monkeypatch)
    built = pf.window
    P.set_request_prefetch_depth([pf], 1)
    assert pf.depth == 1
    pf.enable_slots()
    # the ring runs the built depth: requests never resize a ring, so a depth-1 ring would pin every later one at 1
    assert pf.depth == 2 and pf.window == built
    assert len(set(pf.slot_of.values())) == pf.depth + 1
    P.set_request_prefetch_depth([pf], 2)
    assert pf.depth == 2 and len(set(pf.slot_of.values())) >= pf.depth + 1


class _Group:
    def __init__(self, name):
        self.name = name


def _event_prefetcher(
    names,
    nbytes = 10,
    depth = 2,
):
    """A CPU GroupPrefetcher whose copies record events; every wait (compute stream or release) is logged."""
    groups = [_Group(n) for n in names]
    pf = P.GroupPrefetcher.__new__(P.GroupPrefetcher)
    pf.module, pf.device, pf.depth, pf.groups = None, torch.device("cpu"), depth, groups
    pf.by_id = {id(g): g for g in groups}
    pf.nbytes = {id(g): nbytes for g in groups}
    pf.window = (depth + 1) * nbytes
    pf.ready, pf.inflight_bytes, pf.peak_inflight_bytes = {}, 0, 0
    pf.order, pf.seen, pf.pos, pf.active, pf.on_order, pf.pending = [], [], 0, False, False, False
    pf.stream, pf.fence_first = None, False
    pf.stats = {"forwards": 0, "copies": 0, "prefetched": 0, "missed": 0, "dropped": 0}
    pf.carried, pf.armed, pf.interleaved, pf.graph_hold, pf.slot_of = set(), False, False, False, {}
    issued, waited = [], []

    def issue(group, must = True):
        assert id(group) not in pf.ready, "a group copied twice while on the device"
        event = object()
        issued.append(event)
        pf.ready[id(group)] = event
        pf.inflight_bytes += pf.nbytes[id(group)]
        pf.peak_inflight_bytes = max(pf.peak_inflight_bytes, pf.inflight_bytes)
        return True

    def release(
        group,
        event,
        counted = True,
    ):
        if event is not None:
            waited.append(event)
        if counted:
            pf.inflight_bytes -= pf.nbytes[id(group)]

    class _Compute:
        def wait_event(self, event):
            waited.append(event)

        def wait_stream(self, stream):
            pass

    pf._issue, pf._release, pf._compute = issue, release, lambda: _Compute()
    pf.owns = lambda g: True
    return pf, groups, issued, waited


def _forward(pf, groups):
    pf.begin()
    for g in groups:
        pf.onload(g)
        pf.offload(g)
    pf.end()


def test_shrinking_with_carried_groups_in_flight_waits_every_copy(monkeypatch):
    monkeypatch.setattr(P, "cross_forward_enabled", lambda: True)
    pf, groups, issued, waited = _event_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)  # records the order
    pf.armed = True
    _forward(pf, groups)
    assert len(pf.carried) == 2  # the next forward's first two groups are already copied
    # a memory-tight request lands with both carried copies in flight
    assert P.set_request_prefetch_depth([pf], 1) == 1
    pf.peak_inflight_bytes = pf.inflight_bytes
    for _ in range(2):
        _forward(pf, groups)
        # running group plus one ahead: what the placement floor prices
        assert pf.peak_inflight_bytes <= 2 * max(pf.nbytes.values())
    pf.armed = False  # the render's last forward carries nothing out
    _forward(pf, groups)
    assert pf.inflight_bytes == 0 and not pf.ready and not pf.carried
    assert set(map(id, waited)) >= set(
        map(id, issued)
    )  # every queued copy was waited on before its group ran or left
    assert pf.peak_inflight_bytes <= 2 * max(pf.nbytes.values())


def test_video_status_reports_the_picked_depth(monkeypatch):
    pf = _prefetcher()
    pipe = _pipe(pf)
    assert V.request_prefetch_status(pipe) is None
    V.apply_request_prefetch_depth(pipe, available_mib = 5300, floor_mib = 3300, denoise_mib = 2600)
    got = V.request_prefetch_status(pipe)
    assert got["depth"] == 1 and got["built_depth"] == 2 and got["source"] == "tight"
    assert got["running_depths"] == [1] and got["largest_group_mib"] == BLOCK
    assert got["required_mib"] == 3300 + 2600 and got["available_mib"] == 5300
    V.apply_request_prefetch_depth(pipe, available_mib = 12000, floor_mib = 3300, denoise_mib = 2600)
    assert V.request_prefetch_status(pipe)["source"] == "built"
    assert V.request_prefetch_status(pipe)["running_depths"] == [2]
    monkeypatch.setenv(P.PREFETCH_DEPTH_ENV, "1")
    V.apply_request_prefetch_depth(pipe, available_mib = 12000, floor_mib = 3300, denoise_mib = 2600)
    assert (
        V.request_prefetch_status(pipe)["source"] == "env"
        and V.request_prefetch_status(pipe)["depth"] == 1
    )
