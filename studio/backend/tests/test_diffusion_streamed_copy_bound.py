# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A block-streamed step whose timed eager steps are bound by their copies is never recorded."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_cuda_graph as cg  # noqa: E402
import core.inference.diffusion_offload_prefetch as op  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cg.COPY_CHECK_ENV, raising = False)
    monkeypatch.delenv(cg.SPEED_CHECK_ENV, raising = False)


class _Ev:
    def __init__(self, ms):
        self.ms = ms
        self.synced = 0

    def query(self):
        return True

    def synchronize(self):
        self.synced += 1

    def elapsed_time(self, end):
        return end.ms - self.ms


class _Placement:
    mode = "group"

    def __init__(
        self,
        streams = True,
        prefetcher = None,
    ):
        self._streams = streams
        self.declined = None
        self.pf = prefetcher

    def streams(self):
        return self._streams

    def decline(
        self,
        reason,
        recorded = True,
    ):
        self.declined = reason
        self.recorded = recorded
        return False

    def _hold_carry(self, hold):
        self.hold = hold

    def _prefetcher(self):
        return self.pf


def _step(eager_ms, copies_ms):
    """One timed eager step: its (start, end) and the copies it issued, each ``copies_ms`` long."""
    eager = (_Ev(0.0), _Ev(eager_ms))
    copies = None if copies_ms is None else [(_Ev(0.0), _Ev(ms)) for ms in copies_ms]
    return eager, copies


def _handle(
    placement,
    steps,
    cached = (),
):
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.stats = {"eager_calls": 0, "speed_eager": 0}
    handle.logger = None
    handle.module = torch.nn.Identity()
    handle._dropped, handle._slower, handle._warmed = set(), None, {}
    handle.capture_error = None
    handle.max_graphs = 4
    handle.plan = None
    handle.placement = placement
    handle.cache = {k: object() for k in cached}
    handle._judge = {
        "k": {
            "seen": 1 + len(steps),
            "eager": [e for e, _ in steps],
            "copies": [c for _, c in steps],
            "graph": [],
            "verdict": None,
        }
    }
    return handle


def test_copies_that_fill_the_eager_step_decline_the_placement_before_it_records(monkeypatch):
    retired = []
    monkeypatch.setattr(cg, "_retire_stable_placement", lambda p: retired.append(p))
    placement = _Placement()
    # 4 copies of 58 ms in a 234 ms step: the replay records the same copies, so it cannot be 1% faster
    handle = _handle(placement, [_step(234.0, [58.0] * 4), _step(235.0, [58.5] * 4)])
    state = handle._judge["k"]
    why = handle._copy_bound("k", state)
    assert why and "bound by its host-to-device copies" in why
    assert "232.0 ms of copies in a 234.0 ms step" in why
    assert state["eager"][-1][1].synced == 1
    handle._decline_unrecorded("k", why)
    assert placement.declined == why and state["verdict"] == why
    # nothing recorded: no pool to flush, and the forwards carry copies across again
    assert placement.recorded is False and placement.hold is False
    assert handle._dropped == {"k"} and handle._slower == why
    assert handle.capture_error == {"type": "Refused", "msg": why}
    assert retired == [placement] and handle.stats["copy_declines"] == 1
    # judged once: a second look returns the stored verdict without reading the events again
    assert handle._copy_bound("k", state) == why and state["eager"][-1][1].synced == 1


def test_a_step_with_room_beside_its_copies_still_records():
    # copies fill 90% of the step: a recorded copy schedule can still win (L4 FLUX.1 16 GB, #12707: -10%)
    placement = _Placement()
    handle = _handle(placement, [_step(100.0, [45.0, 45.0]), _step(100.0, [45.0, 45.0])])
    assert handle._copy_bound("k", handle._judge["k"]) is None
    assert placement.declined is None and not handle._dropped


def test_the_bound_compares_the_best_eager_step_with_the_least_copy_time():
    # best eager 100 ms, least copy total 99.5 ms (from the other step): bound
    handle = _handle(_Placement(), [_step(100.0, [50.0, 49.5]), _step(140.0, [70.0, 69.0])])
    assert handle._copy_bound("k", handle._judge["k"])
    # least copy total 98.0 ms in a 100 ms best step: 2% of room, records
    handle = _handle(_Placement(), [_step(100.0, [49.0, 49.0]), _step(140.0, [70.0, 69.0])])
    assert handle._copy_bound("k", handle._judge["k"]) is None


def test_an_untimed_copy_or_a_recorded_key_leaves_the_check_to_the_replay_judge():
    full = [_step(100.0, [99.5]), _step(100.0, [99.6])]
    # one step's copies went untimed (another thread recorded): no verdict
    handle = _handle(_Placement(), [full[0], _step(100.0, None)])
    assert handle._copy_bound("k", handle._judge["k"]) is None
    # no copies at all (nothing streamed in the step)
    handle = _handle(_Placement(), [_step(100.0, []), _step(100.0, [])])
    assert handle._copy_bound("k", handle._judge["k"]) is None
    # a key of this placement already recorded: the judge's per-placement verdict stands
    handle = _handle(_Placement(), full, cached = ("j",))
    assert handle._copy_bound("k", handle._judge["k"]) is None


def test_no_verdict_while_another_graph_records(monkeypatch):
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 1)
    handle = _handle(_Placement(), [_step(100.0, [99.5]), _step(100.0, [99.6])])
    state = handle._judge["k"]
    assert handle._copy_bound("k", state) is None and "copy_verdict" not in state
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 0)
    assert handle._copy_bound("k", state)


def test_the_kill_switch(monkeypatch):
    assert cg.copy_check_enabled()
    monkeypatch.setenv(cg.COPY_CHECK_ENV, "0")
    assert not cg.copy_check_enabled()


def test_only_a_streaming_group_placement_times_its_copies():
    pf = types.SimpleNamespace()
    assert _handle(_Placement(streams = True, prefetcher = pf), [])._streaming_prefetcher() is pf
    # resident / pinned placements: the recording reads weights in place, a replay can beat eager
    assert _handle(_Placement(streams = False, prefetcher = pf), [])._streaming_prefetcher() is None
    model = _Placement(prefetcher = pf)
    model.mode = "model"
    assert _handle(model, [])._streaming_prefetcher() is None
    assert _handle(None, [])._streaming_prefetcher() is None


def test_a_timed_eager_step_collects_its_copies_and_unhooks_the_prefetcher(monkeypatch):
    issued = []

    class _Pf:
        pass

    pf = _Pf()
    handle = _handle(_Placement(prefetcher = pf), [])
    handle._judge = {}
    monkeypatch.setattr(handle, "_statics", lambda live: list(live))
    monkeypatch.setattr(cg, "_timing_events", lambda: (_Ev(0.0), _Ev(10.0)))
    monkeypatch.setattr(_Ev, "record", lambda self, *a: None, raising = False)

    def step(t):
        times = pf.__dict__.get("copy_times")
        if times is not None:
            times.append((_Ev(0.0), _Ev(4.0)))
            issued.append(len(times))
        return t + 1

    x = torch.ones(2)
    for _ in range(3):
        assert torch.equal(handle._timed_eager(step, (x,), {}, "k"), x + 1)
    state = handle._judge["k"]
    # the untimed warm-up issues no timed copies; each timed step keeps its own list
    assert issued == [1, 1] and len(state["eager"]) == 2
    assert [len(c) for c in state["copies"]] == [1, 1]
    assert pf.copy_times is None


def test_a_copy_the_prefetcher_could_not_time_voids_that_steps_total(monkeypatch):
    pf = types.SimpleNamespace()
    handle = _handle(_Placement(prefetcher = pf), [])
    handle._judge = {}
    monkeypatch.setattr(handle, "_statics", lambda live: list(live))
    monkeypatch.setattr(cg, "_timing_events", lambda: (_Ev(0.0), _Ev(10.0)))
    monkeypatch.setattr(_Ev, "record", lambda self, *a: None, raising = False)

    def step(t):
        if pf.__dict__.get("copy_times") is not None:
            pf.copy_times = None  # what _copy_timer_end does when a copy goes untimed
        return t

    x = torch.ones(2)
    for _ in range(2):
        handle._timed_eager(step, (x,), {}, "k")
    assert handle._judge["k"]["copies"] == [None]


def test_the_prefetcher_times_copies_only_for_a_probe_and_never_inside_a_capture():
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    assert pf._copy_timer(None) is None
    pf.copy_times = []
    pf.capturing = True
    assert pf._copy_timer(None) is None


def test_an_unrecorded_decline_drops_the_ring_without_a_flush(monkeypatch):
    flushes = []
    monkeypatch.setattr(cg, "flush_now_locked", lambda **k: flushes.append(k))
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    pf.active = False
    pf.slot_raw = {}
    pf._drop_slots_at_end = False
    dropped = []
    monkeypatch.setattr(pf, "disable_slots", lambda: dropped.append(1), raising = False)
    pf.drop_slots_after_forward(flush = False)
    assert dropped == [1] and flushes == []
    # a recorded placement's drop still hands the pool and the capture workspace back
    pf.drop_slots_after_forward()
    assert dropped == [1, 1] and flushes == [{"clear_cublas": True}]
    # mid-forward: dropped at the forward's end, still unflushed
    pf.active = True
    pf.drop_slots_after_forward(flush = False)
    assert pf._drop_slots_at_end and dropped == [1, 1]
    pf.active = False
    pf._drop_ring()
    assert dropped == [1, 1, 1] and len(flushes) == 1


def test_placement_decline_forwards_the_unflushed_drop():
    placement = cg.OffloadPlacement.__new__(cg.OffloadPlacement)
    placement.mode = "group"
    placement._declined = None
    placement._slots = True
    calls = []
    pf = types.SimpleNamespace(drop_slots_after_forward = lambda **k: calls.append(k))
    placement._prefetcher = lambda: pf
    assert placement.decline("bound", recorded = False) is True
    assert calls == [{"flush": False}] and placement.refusal({}) == "bound"
    placement._slots = True
    assert placement.decline("slower") is True and calls[-1] == {}
