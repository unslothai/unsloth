# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-forward carry of the next forward's first groups."""

from __future__ import annotations

import copy
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_offload_prefetch as op


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        op.CROSS_FORWARD_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_DIFFUSION_PIN_TOP_GROUP",
    ):
        monkeypatch.delenv(name, raising = False)
    op._LAST_BEGUN.pf = None
    yield
    op._LAST_BEGUN.pf = None


class _Group:
    def __init__(self, name: str):
        self.name = name


def _cpu_prefetcher(
    names,
    nbytes = 10,
    depth = 2,
    window = None,
):
    groups = [_Group(n) for n in names]
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    pf.module = None
    pf.device = torch.device("cpu")
    pf.depth = depth
    pf.groups = groups
    pf.by_id = {id(g): g for g in groups}
    pf.nbytes = {id(g): nbytes for g in groups}
    pf.window = window if window is not None else (depth + 1) * nbytes
    pf.ready, pf.inflight_bytes, pf.peak_inflight_bytes = {}, 0, 0
    pf.order, pf.seen, pf.pos, pf.active, pf.on_order, pf.pending = [], [], 0, False, False, False
    pf.stream, pf.fence_first = None, False
    pf.stats = {"forwards": 0, "copies": 0, "prefetched": 0, "missed": 0, "dropped": 0}
    pf.carried, pf.armed, pf.interleaved, pf.graph_hold = set(), False, False, False
    log: list = []

    def issue(group, must = True):
        assert id(group) not in pf.ready, "a group copied twice while on the device"
        log.append(("copy", group.name))
        pf.ready[id(group)] = object()
        pf.inflight_bytes += pf.nbytes[id(group)]
        pf.peak_inflight_bytes = max(pf.peak_inflight_bytes, pf.inflight_bytes)

    def release(
        group,
        event,
        counted = True,
    ):
        log.append(("free", group.name))
        if counted:
            pf.inflight_bytes -= pf.nbytes[id(group)]

    class _Compute:
        def wait_event(self, event):
            log.append(("wait", None))

        def wait_stream(self, stream):
            log.append(("join", None))

    pf._issue = issue
    pf._release = release
    pf._compute = lambda: _Compute()
    pf.owns = lambda g: not getattr(g, "resident", False)
    return pf, groups, log


def _forward(
    pf,
    groups,
    skip = (),
):
    pf.begin()
    for g in groups:
        if g.name in skip:
            continue
        pf.onload(g)
        pf.offload(g)
    pf.end()


def _pipe(*pfs):
    comps = {
        f"transformer_{i}": types.SimpleNamespace(**{op.PREFETCHER_ATTR: pf})
        for i, pf in enumerate(pfs)
    }
    return types.SimpleNamespace(components = comps, _interrupt = False)


def _render(
    pf,
    groups,
    steps,
    forwards_per_step = 1,
    pipe = None,
):
    pipe = pipe or _pipe(pf)
    with op.cross_forward_render(pipe):
        for i in range(steps):
            for _ in range(forwards_per_step):
                _forward(pf, groups)
            op.cross_forward_step(pipe, i, steps)
    return pipe


def _copies(log):
    return [n for k, n in log if k == "copy"]


def test_carry_wraps_onto_the_next_forwards_first_groups():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)  # records the order
    pf.armed = True
    log.clear()
    _forward(pf, groups)
    # the tail queued the next forward's first two groups; they stay on the device past end()
    assert _copies(log)[-2:] == ["a", "b"]
    assert pf.carried == {id(groups[0]), id(groups[1])}
    assert set(pf.ready) == pf.carried and pf.inflight_bytes == 20
    log.clear()
    pf.begin()
    pf.onload(groups[0])
    assert _copies(log) == ["c"]
    assert pf.stats["carried_hits"] == 1
    for g in groups:
        if g is not groups[0]:
            pf.onload(g)
        pf.offload(g)
    pf.end()
    assert pf.stats["missed"] == 0 and pf.stats["carried_hits"] == 2


def test_while_the_last_block_runs_only_one_group_is_carried():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    pf.begin()
    for g in groups[:-1]:
        pf.onload(g)
        pf.offload(g)
        assert not pf.carried  # a block of this forward is still due: nothing carried
    pf.onload(groups[-1])
    # its activations are live: one group beside them, never the full depth
    assert pf.carried == {id(groups[0])} and op.TAIL_CARRY_GROUPS == 1
    pf.offload(groups[-1])
    # the forward's activations are freed: the rest of the depth follows
    assert pf.carried == {id(groups[0]), id(groups[1])}
    pf.end()
    assert set(pf.ready) == pf.carried and pf.peak_inflight_bytes <= pf.window


def test_unarmed_prefetcher_is_unchanged():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    ref, ref_groups, ref_log = _cpu_prefetcher("abcdef", depth = 2)
    for _ in range(4):
        _forward(pf, groups)
        _forward(ref, ref_groups)
        assert not pf.ready and pf.inflight_bytes == 0
    assert log == ref_log and not pf.stats.get("carried")


@pytest.mark.parametrize("depth,window", [(2, None), (4, 25), (3, 35), (1, None)])
def test_carry_never_passes_the_window(depth, window):
    pf, groups, log = _cpu_prefetcher("abcdefgh", nbytes = 10, depth = depth, window = window)
    _forward(pf, groups)
    pf.armed = True
    for _ in range(4):
        _forward(pf, groups)
        assert pf.peak_inflight_bytes <= pf.window
        ahead = len(pf.carried)
        assert ahead <= depth and pf.inflight_bytes == 10 * ahead
    assert pf.stats["carried"] > 0 and pf.stats["missed"] == 0


def test_final_step_carries_nothing_into_the_decode():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pipe = _pipe(pf)
    with op.cross_forward_render(pipe):
        for i in range(4):
            _forward(pf, groups)
            op.cross_forward_step(pipe, i, 4)
            if i == 2:
                # the second-to-last step's carry is kept for the last step, which carries nothing on
                assert not pf.armed and len(pf.carried) == 2
        # decode runs here (inside the pipeline call): no streamed group is on the device
        assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
    assert pf.stats["carried_hits"] == 6 and pf.stats.get("carried_dropped", 0) == 0


def test_skipped_final_forward_drops_the_carry_at_the_last_callback():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pipe = _pipe(pf)
    with op.cross_forward_render(pipe):
        for i in range(3):
            if i < 2:
                _forward(pf, groups)
            op.cross_forward_step(pipe, i, 3)  # step 2's forward skipped (step cache)
        assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
    assert pf.stats["carried_dropped"] == 2


def test_step_count_from_the_pipeline_and_unknown_count_disarms():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pipe = _pipe(pf)
    pipe._num_timesteps = 5
    with op.cross_forward_render(pipe):
        _forward(pf, groups)
        op.cross_forward_step(pipe, 0)
        assert pf.armed and pf.carried
    pf2, groups2, _ = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf2, groups2)
    pipe2 = _pipe(pf2)  # no _num_timesteps: the last step cannot be told apart
    with op.cross_forward_render(pipe2):
        _forward(pf2, groups2)
        op.cross_forward_step(pipe2, 0)
        assert not pf2.armed and not pf2.carried and not pf2.ready


def test_interrupted_render_drops_the_carry():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pipe = _pipe(pf)
    with op.cross_forward_render(pipe):
        _forward(pf, groups)
        pipe._interrupt = True
        op.cross_forward_step(pipe, 0, 10)
        assert not pf.armed and not pf.ready and pf.inflight_bytes == 0


def test_render_exit_drops_the_carry_even_on_error():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    with pytest.raises(ValueError):
        with op.cross_forward_render(_pipe(pf)):
            _forward(pf, groups)
            assert pf.carried
            raise ValueError("boom")
    assert not pf.armed and not pf.ready and not pf.carried and pf.inflight_bytes == 0


@pytest.mark.parametrize("forwards_per_step", [1, 2])
def test_cfg_double_forward_reads_every_carry(forwards_per_step):
    pf, groups, log = _cpu_prefetcher("abcdefg", depth = 2)
    _forward(pf, groups)
    steps = 4
    _render(pf, groups, steps, forwards_per_step)
    forwards = steps * forwards_per_step
    # the last step carries nothing (the callback cannot tell its forwards apart)
    assert pf.stats["carried_hits"] == 2 * (forwards - forwards_per_step)
    assert pf.stats["carried"] == 2 * (forwards - forwards_per_step)
    assert pf.stats.get("carried_dropped", 0) == 0 and pf.stats["missed"] == 0
    assert not pf.ready and pf.inflight_bytes == 0


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(op.CROSS_FORWARD_ENV, "0")
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    ref, ref_groups, ref_log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(ref, ref_groups)
    log.clear()
    ref_log.clear()
    _render(pf, groups, 3, 2)
    for _ in range(6):
        _forward(ref, ref_groups)
    assert log == ref_log
    assert not pf.stats.get("carried") and not pf.armed


def test_graph_hold_and_slot_ring_block_and_drop_the_carry():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    _forward(pf, groups)
    assert pf.carried
    pf.graph_hold = True  # a step graph installed between forwards: a replay would not read them
    pf.begin()
    assert not pf.carried and pf.inflight_bytes == 0 and pf.stats["carried_dropped"] == 2
    for g in groups:
        pf.onload(g)
        pf.offload(g)
    pf.end()
    assert not pf.ready and not pf.stats.get("carried_hits")
    pf.graph_hold = False
    pf.slot_of = {id(groups[0]): 0}
    _forward(pf, groups)
    assert not pf.ready and not pf.carried


def test_capture_never_starts_on_carried_copies(monkeypatch):
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    _forward(pf, groups)
    assert pf.carried
    pf.stream = types.SimpleNamespace(wait_stream = lambda s: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match = "carried"):
        pf.begin()


def test_capturing_forward_carries_nothing(monkeypatch):
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    pf.stream = types.SimpleNamespace(wait_stream = lambda s: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _forward(pf, groups)
    assert not pf.carried and not pf.ready


def test_another_prefetcher_between_forwards_drops_and_stops_the_carry():
    a, ga, _ = _cpu_prefetcher("abcdef", depth = 2)
    b, gb, _ = _cpu_prefetcher("uvwxyz", depth = 2)
    _forward(a, ga)
    _forward(b, gb)
    pipe = _pipe(a, b)
    with op.cross_forward_render(pipe):
        _forward(a, ga)
        assert a.carried
        _forward(b, gb)  # a second denoiser runs next: a's carried copies would sit through it
        assert a.interleaved and not a.carried and not a.ready
        for _ in range(2):
            _forward(b, gb)
        assert b.carried and not b.interleaved
        _forward(a, ga)
        assert not a.carried and b.interleaved and not b.ready
    assert a.inflight_bytes == 0 and b.inflight_bytes == 0


def test_not_armed_beside_a_controlnet():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pipe = _pipe(pf)
    pipe.components["controlnet"] = object()
    with op.cross_forward_render(pipe) as armed:
        assert armed == [] and not pf.armed
        _forward(pf, groups)
        assert not pf.carried


def test_off_order_forward_drops_the_carry():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    _forward(pf, groups)
    assert pf.carried
    _forward(pf, groups, skip = "c")  # off the recorded order: re-recorded, nothing carried on
    assert not pf.carried and not pf.ready and pf.inflight_bytes == 0
    _forward(pf, groups, skip = "c")  # back on the (new) order: carries again
    assert pf.carried


def test_skipped_forward_keeps_the_carry():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    _forward(pf, groups)
    carried = set(pf.carried)
    pf.begin()
    pf.end()  # a cached step: no block ran
    assert pf.carried == carried and pf.inflight_bytes == 20
    _forward(pf, groups)
    assert pf.stats["carried_hits"] == 2


def test_short_order_never_carries_a_group_still_due():
    pf, groups, log = _cpu_prefetcher("ab", depth = 2)
    _forward(pf, groups)
    pf.armed = True
    for _ in range(3):
        _forward(pf, groups)
        assert pf.inflight_bytes == 10 * len(pf.carried) <= pf.window
    assert pf.stats["missed"] == 0


def test_resident_groups_are_not_carried():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    groups[0].resident = True
    pf.armed = True
    _forward(pf, groups)
    assert pf.carried == {id(groups[1]), id(groups[2])}


# CUDA: real diffusers stream group offload


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: diffusers stream group offload")
    pytest.importorskip("diffusers.hooks")


class _Net(torch.nn.Module):
    def __init__(
        self,
        width = 1024,
        blocks = 8,
        sleep_cycles = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, width)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 64)
        self.sleep_cycles = sleep_cycles

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            if self.sleep_cycles:
                torch.cuda._sleep(self.sleep_cycles)
            x = torch.nn.functional.gelu(block(x))
        return self.proj_out(x)


def _streamed(net):
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net}, _interrupt = False)
    assert dm._apply_group_offload(pipe, "cuda", None)
    return pipe


@pytest.mark.parametrize("sleep_cycles", [0, 2_000_000])
@pytest.mark.parametrize("cfg", [False, True])
def test_streamed_render_is_bit_identical_and_leaves_nothing_resident(sleep_cycles, cfg):
    _cuda()
    torch.manual_seed(0)
    net = _Net(sleep_cycles = sleep_cycles)
    ref = copy.deepcopy(net).cuda()
    pipe = _streamed(net)
    pf = op.module_prefetcher(net)
    assert pf is not None
    xs = [torch.randn(4, 64, device = "cuda") for _ in range(2)]
    steps = 4

    def run(model, on_step):
        x = xs[0]
        for i in range(steps):
            v = model(x)
            if cfg:
                v = v + 0.5 * model(xs[1] + x)
            x = x - 0.1 * v
            on_step(i)
        return x

    with torch.no_grad():
        want = run(ref, lambda i: None)
        net(xs[0])  # records the order
        with op.cross_forward_render(pipe):
            got = run(net, lambda i: op.cross_forward_step(pipe, i, steps))
            torch.cuda.synchronize()
            # where the decode runs: no streamed block left on the device
            assert all(b.weight.device.type == "cpu" for b in net.blocks)
            assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
    assert torch.equal(got, want)
    forwards = steps * (2 if cfg else 1)
    assert pf.stats["carried_hits"] == pf.stats["carried"] > 0
    assert pf.stats["carried_hits"] == 2 * (forwards - (2 if cfg else 1))
    assert pf.stats["missed"] == 0 and pf.peak_inflight_bytes <= pf.window
