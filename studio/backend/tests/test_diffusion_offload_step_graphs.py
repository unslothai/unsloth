# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The whole-step graph of an offloaded denoiser on top of the per-block graphs (diffusion_block_graph).

The step graph records a forward whose blocks compile below their offload hooks; the per-block graphs run their compute
while it records, and arm as its fallback where they are the default. A streamed placement records its copies into the
prefetcher's slot ring and keeps a key only when the replay pays, else hands the ring back."""

from __future__ import annotations

import copy
import types
import warnings

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_block_graph as bg  # noqa: E402
import core.inference.diffusion_cuda_graph as cg  # noqa: E402
import core.inference.diffusion_memory as dm  # noqa: E402
import core.inference.diffusion_offload_prefetch as op  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        bg.BLOCK_GRAPHS_ENV,
        bg.COMPILE_BELOW_HOOKS_ENV,
        cg.CUDA_GRAPHS_ENV,
        cg.CUDA_GRAPH_DISABLE_ENV,
        cg.OFFLOAD_CUDA_GRAPH_ENV,
        cg.STEP_SLOTS_ENV,
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_DIFFUSION_PIN_TOP_GROUP",
    ):
        monkeypatch.delenv(name, raising = False)
    # speed check off; tests that need it drive it explicitly
    monkeypatch.setenv(cg.SPEED_CHECK_ENV, "0")
    yield
    cg._POOL_BOX[0] = None


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.nn.functional.gelu(self.lin(x))


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(
        self,
        width = 256,
        blocks = 8,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(16, width)
        self.blocks = torch.nn.ModuleList(Block(width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 16)

    def forward(
        self,
        x,
        t,
        return_dict = True,
    ):
        x = self.proj_in(x) * t
        for block in self.blocks:
            x = block(x)
        out = self.proj_out(x)
        return (out,) if not return_dict else {"sample": out}


def _pipe(net):
    return types.SimpleNamespace(transformer = net, components = {"transformer": net})


def _target():
    return types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda")


def test_a_block_graph_runs_its_compute_while_a_whole_step_records():
    calls = []
    shared = bg._Shared(None, None)
    graph = bg.BlockGraph(Block(4), lambda *a, **k: calls.append(1) or "computed", shared)
    with cg._recording_step():
        assert cg.step_recording()
        assert graph(torch.zeros(2, 4)) == "computed"
    assert not cg.step_recording()
    assert calls == [1]
    # the step graph records the kernels itself, so the block graph stays idle
    assert graph.stats["eager_calls"] == 0 and not graph.seen and not graph.cache


def _step_handle(
    net,
    mode = "group",
    plan = None,
):
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.module = net
    handle.placement = types.SimpleNamespace(mode = mode) if mode else None
    handle.plan = plan
    handle.fallback = None
    handle.fallback_handle = None
    return handle


def _arm(pipe, monkeypatch, *, hooked, pinned):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    applied = {"cuda_graph": True}
    handles = cg.arm_block_graphs(
        pipe,
        applied,
        target = _target(),
        family = types.SimpleNamespace(),
        hooked = hooked,
        pinned = pinned,
    )
    return handles, applied


def test_an_armed_step_graph_stays_primary_with_per_block_graphs_as_its_fallback(monkeypatch):
    net = Net(blocks = 3)
    pipe = _pipe(net)
    step = _step_handle(net)
    pipe._unsloth_cuda_graphs = (step,)
    pipe._unsloth_cuda_graph_reason = (
        "captured per input shape: block-streamed (copies recorded in the graph)"
    )
    handles, applied = _arm(pipe, monkeypatch, hooked = True, pinned = True)
    assert handles == (step,) and applied["cuda_graph"]
    assert pipe._unsloth_cuda_graph_mode == "step" and callable(step.fallback)
    assert cg.status_reason(pipe, True) == (
        "denoiser step captured per input shape: block-streamed (copies recorded in the graph), replayed bit-identically"
    )
    armed = step.fallback()
    assert isinstance(armed, bg.BlockGraphSet) and len(armed.graphs) == 3
    assert cg.status_reason(pipe, True).endswith("; the steps it leaves eager record per block")
    armed.free()


def test_a_streamed_step_graph_gets_a_per_block_fallback_only_when_asked(monkeypatch):
    net = Net(blocks = 3)
    pipe = _pipe(net)
    step = _step_handle(net)
    pipe._unsloth_cuda_graphs = (step,)
    _arm(pipe, monkeypatch, hooked = True, pinned = False)
    assert step.fallback is None
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "1")
    _arm(pipe, monkeypatch, hooked = True, pinned = False)
    assert callable(step.fallback)
    model = _step_handle(net, mode = "model")
    pipe._unsloth_cuda_graphs = (model,)
    monkeypatch.delenv(bg.BLOCK_GRAPHS_ENV)
    _arm(pipe, monkeypatch, hooked = True, pinned = True)
    assert model.fallback is None
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "0")
    pinned = _step_handle(net)
    pipe._unsloth_cuda_graphs = (pinned,)
    _arm(pipe, monkeypatch, hooked = True, pinned = True)
    assert pinned.fallback is None


def test_a_planned_resident_step_keeps_its_graph_and_the_blocks_wait_as_fallback(monkeypatch):
    net = Net(blocks = 2)
    pipe = _pipe(net)
    step = _step_handle(net, mode = None, plan = lambda *a: None)
    pipe._unsloth_cuda_graphs = (step,)
    pipe._unsloth_cuda_graph_reason = "captured per input shape: resident"
    handles, _ = _arm(pipe, monkeypatch, hooked = False, pinned = False)
    assert handles == (step,) and callable(step.fallback)
    assert cg.status_reason(pipe, True) == cg.WHOLE_REASON


class _Ev:
    def __init__(self, ms):
        self.ms = ms

    def query(self):
        return True

    def elapsed_time(self, end):
        return end.ms - self.ms


class _Placement:
    def __init__(
        self,
        streams,
        ring = False,
    ):
        self._streams = streams
        self.ring = ring
        self.declined = None
        self.mode = "group"

    def streams(self):
        return self._streams

    def decline(self, reason):
        self.declined = reason
        return self.ring


def _judged(
    placement,
    eager_ms,
    graph_ms,
    others = (),
):
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.stats, handle.logger, handle._slower, handle.capture_error = {}, None, None, None
    handle.placement = placement
    handle.cache = {"k": object(), **{o: object() for o in others}}
    handle._dropped, handle.released = set(), []
    handle._release = lambda: handle.released.append(1)
    handle._judge = {
        "k": {
            "seen": 3,
            "eager": [(_Ev(0), _Ev(ms)) for ms in eager_ms],
            "graph": [(_Ev(0), _Ev(ms)) for ms in graph_ms],
            "verdict": None,
        },
        **{o: {"seen": 3, "eager": [], "graph": [], "verdict": ""} for o in others},
    }
    handle._judge_keys()
    return handle, handle._judge["k"]["verdict"]


def test_a_streamed_key_is_kept_only_when_its_replay_pays():
    # streaming tie: ring and pool are not worth it, so the graph is dropped
    placement = _Placement(streams = True)
    handle, why = _judged(placement, [100.0, 100.4], [99.6, 99.8, 99.7], others = ("j",))
    assert why and "within 1%" in why and placement.declined == why
    assert handle.cache == {} and handle._dropped == {"k", "j"} and handle.released == [1]
    # with a slot ring the flush waits for the ring drop at forward end
    placement = _Placement(streams = True, ring = True)
    handle, why = _judged(placement, [100.0, 100.4], [99.6, 99.8, 99.7])
    assert placement.declined == why and handle.cache == {} and handle.released == []
    placement = _Placement(streams = True)
    handle, why = _judged(placement, [100.0, 101.0], [97.0, 97.5, 98.0])
    assert why == "" and placement.declined is None and "k" in handle.cache
    placement = _Placement(streams = False)
    handle, why = _judged(placement, [100.0, 100.4], [99.6, 99.8, 99.7])
    assert why == "" and placement.declined is None and "k" in handle.cache


def test_no_key_is_judged_while_another_graph_records(monkeypatch):
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 1)
    placement = _Placement(streams = True)
    handle, why = _judged(placement, [100.0, 100.4], [99.6, 99.8, 99.7])
    assert why is None and placement.declined is None and "k" in handle.cache
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 0)
    handle._judge_keys()
    assert handle._judge["k"]["verdict"]


class _Wrapper:
    def __init__(self, cache):
        self.cache = cache


def test_every_live_graph_pool_stays_out_of_the_reclaimable_memory(monkeypatch):
    # simulate a shared pool id retired by a later failed capture
    monkeypatch.setattr(cg, "_POOL_BOX", [None])
    live = _Wrapper({"k": types.SimpleNamespace(pool_token = (0, 7))})
    monkeypatch.setattr(cg, "_LIVE_WRAPPERS", {live})
    segments = [
        {"device": 0, "segment_pool_id": (0, 7), "total_size": 100, "allocated_size": 40},
        {"device": 0, "segment_pool_id": (0, 0), "total_size": 500, "allocated_size": 0},
        {
            "device": 1,
            "segment_pool_id": (0, 7),
            "total_size": 900,
            "allocated_size": 0,
        },
    ]
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: segments)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert cg.live_pool_free_bytes() == 60


def test_only_the_streamed_groups_slots_are_allocated_before_a_capture(monkeypatch):
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    streamed, resident = types.SimpleNamespace(), types.SimpleNamespace(_unsloth_resident = True)
    others = [types.SimpleNamespace(_unsloth_resident = True) for _ in range(2)]
    pf.groups = [resident, others[0], streamed, others[1]]
    pf.slot_of = {id(g): i % 3 for i, g in enumerate(pf.groups)}
    pf.slot_size, pf.slot_raw = 16, {}
    made = []
    monkeypatch.setattr(pf, "_alloc_slot", lambda where: made.append(where), raising = False)
    pf.materialize_slots()
    assert made == [2]


def test_a_failed_streamed_capture_hands_the_ring_back(monkeypatch):
    _cuda()
    net, handle = _armed_streamed(monkeypatch)
    pf = op.module_prefetcher(net)

    def broken(*args, **kwargs):
        handle.placement.before_capture()
        raise RuntimeError("capture failed")

    monkeypatch.setattr(handle, "_capture", broken)
    ref = _net().cuda()
    _call(net, 0)
    assert torch.equal(_call(net, 1), _call(ref, 1))
    assert handle.poisoned and not pf.slot_raw
    handle.free()


def test_a_declined_placement_refuses_every_later_call():
    placement = cg.OffloadPlacement.__new__(cg.OffloadPlacement)
    placement.mode = "group"
    placement._declined = None
    placement._slots = False
    placement.decline("replay measured within 1% of the eager step")
    assert placement.refusal({}) == "replay measured within 1% of the eager step"


def test_both_graph_pools_are_kept_out_of_the_reclaimable_memory(monkeypatch):
    monkeypatch.setattr(
        dm,
        "snapshot_device_memory",
        lambda target: dm.DeviceMemory(
            "cuda", "cuda", "discrete_vram", free_mib = 1000, total_mib = 8000
        ),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 900 << 20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 100 << 20)
    monkeypatch.setattr(bg, "pool_bytes", lambda device = None: 300 << 20)
    monkeypatch.setattr(cg, "live_pool_free_bytes", lambda: 200 << 20)
    mem = dm.reclaimable_snapshot_device_memory(types.SimpleNamespace(device = "cuda"))
    assert mem.free_mib == 1000 + 800 - 300 - 200


def _cuda():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        pytest.skip("needs NVIDIA CUDA")
    pytest.importorskip("diffusers.hooks")


def _streamed(net, resident_mib = None):
    kwargs = {"resident_transformer_mib": resident_mib} if resident_mib else {}
    assert dm._apply_group_offload(_pipe(net), "cuda", None, **kwargs)
    # let the background pin finish so the counts below are exact
    for group in dm._offload_groups(net):
        pinner = getattr(group, op._BG_PIN_ATTR, None)
        if pinner is not None:
            pinner.wait(group)
    return net


def _inputs(seed, rows = 8):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return torch.randn(rows, 16, generator = g).cuda(), torch.rand(1, generator = g).cuda()


def _call(
    net,
    seed,
    rows = 8,
):
    x, t = _inputs(seed, rows)
    with torch.no_grad():
        out = net(x, t, return_dict = False)[0].clone()
    torch.cuda.synchronize()
    return out


def _syncs(fn):
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            out = fn()
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    return out, sum(1 for w in caught if "synchroniz" in str(w.message).lower())


def _net():
    torch.manual_seed(0)
    return Net()


def _armed_streamed(monkeypatch, slots = True):
    if not slots:
        monkeypatch.setenv(cg.STEP_SLOTS_ENV, "0")
    net = _streamed(_net())
    handles, reason = cg.arm_after_placement(_pipe(net))
    assert len(handles) == 1 and handles[0].placement.mode == "group", reason
    return net, handles[0]


def test_streamed_step_copies_land_in_the_slot_ring_not_the_graph_pool(monkeypatch):
    _cuda()
    ref = _net().cuda()
    want = [_call(ref, s) for s in range(4)]
    real_to_device = op._to_device
    fresh = []

    def to_device(src, device):
        if torch.cuda.is_current_stream_capturing():
            fresh.append(tuple(src.shape))
        return real_to_device(src, device)

    monkeypatch.setattr(op, "_to_device", to_device)
    counts = {}
    for slots in (False, True):
        cg._POOL_BOX[0] = None
        fresh.clear()
        net, handle = _armed_streamed(monkeypatch, slots)
        pf = op.module_prefetcher(net)
        got = [_call(net, 0), _call(net, 1)]
        got += [_call(net, 2), _call(net, 3)]
        for a, b in zip(got, want):
            assert torch.equal(a, b)
        assert handle.stats["captures"] == 1 and handle.stats["replays"] == 4
        counts[slots] = len(fresh)
        if slots:
            assert handle.placement._slots and pf.slot_raw and pf.stats["slot_fills"] > 0
            assert pf.stats.get("slot_fallbacks", 0) == 0
        else:
            assert not pf.slot_raw
        handle.free()
        monkeypatch.delenv(cg.STEP_SLOTS_ENV, raising = False)
    # without the ring each streamed weight/bias is a fresh pool alloc; with it only top-level 4
    assert counts[False] >= 2 * 7 and counts[True] <= 4


def test_a_declined_streamed_placement_hands_the_ring_back_and_runs_eager(monkeypatch):
    _cuda()
    ref = _net().cuda()
    net, handle = _armed_streamed(monkeypatch)
    pf = op.module_prefetcher(net)
    _call(net, 0)
    _call(net, 1)
    assert pf.slot_raw and handle.stats["replays"] == 2
    handle.placement.decline("replay measured within 1% of the eager step")
    handle.cache.clear()
    for seed in (2, 3):
        assert torch.equal(_call(net, seed), _call(ref, seed))
    assert not pf.slot_raw and not pf.slot_of
    assert handle.stats["captures"] == 1 and handle.capture_error["type"] == "Refused"


def test_a_new_slot_ring_is_a_new_placement(monkeypatch):
    _cuda()
    net, handle = _armed_streamed(monkeypatch)
    pf = op.module_prefetcher(net)
    _call(net, 0)
    _call(net, 1)
    assert handle.stats["captures"] == 1
    pf.disable_slots()
    pf.enable_slots()
    ref = _net().cuda()
    assert torch.equal(_call(net, 2), _call(ref, 2))
    assert handle.stats["invalidations"] == 1 and handle.stats["captures"] == 2


def test_a_failed_pinned_step_graph_hands_its_calls_to_per_block_graphs(monkeypatch):
    _cuda()
    ref = _net().cuda()
    net = _streamed(_net(), resident_mib = 64)
    pipe = _pipe(net)
    cg.arm_after_placement(pipe)
    handles = cg.arm_block_graphs(
        pipe,
        {"cuda_graph": True},
        target = _target(),
        family = types.SimpleNamespace(),
        hooked = True,
        pinned = True,
    )
    step = handles[0]
    assert step.placement is not None and callable(step.fallback)
    assert not step.placement.streams() and not step.placement._slots
    _call(net, 0)
    assert step.stats["captures"] == 1 and step.fallback_handle is None
    step.poisoned = True
    for seed in (1, 2, 3):
        assert torch.equal(_call(net, seed), _call(ref, seed))
    blocks = step.fallback_handle
    assert isinstance(blocks, bg.BlockGraphSet) and step.fallback is None
    # per block: call 1 eager, call 2 records and replays, call 3 replays
    assert blocks.stats["captures"] == 8 and blocks.stats["replays"] == 16
    step.free()
    assert step.fallback_handle is None
    assert not any(
        isinstance(r.forward, bg.BlockGraph) for b in net.blocks for r in b._diffusers_hook._fn_refs
    )


def test_per_block_graphs_under_a_recording_step_record_nothing(monkeypatch):
    _cuda()
    ref = _net().cuda()
    net = _streamed(_net(), resident_mib = 64)
    pipe = _pipe(net)
    handles, _ = cg.arm_after_placement(pipe)
    blocks, reason = bg.install_block_graphs(net, device = "cuda")
    assert reason == "armed"
    want = [_call(ref, s) for s in range(3)]
    got = [_call(net, s) for s in range(3)]
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 3
    assert blocks.stats["captures"] == 0 and blocks.stats["eager_calls"] == 0
    blocks.free()
    handles[0].free()


def test_a_compiled_streamed_step_records_with_no_graph_break_and_a_new_shape_adds_none(
    monkeypatch,
):
    _cuda()
    import torch._dynamo.utils as du

    torch._dynamo.reset()
    ref = _net().cuda()
    net = _net()
    kwargs = {"fullgraph": False, "dynamic": None}
    for b in net.blocks:
        b.compile(**kwargs)
    net._unsloth_regional_compile_kwargs = kwargs
    _streamed(net)
    assert bg.compile_below_offload_hooks(net) == 8
    handles, reason = cg.arm_after_placement(_pipe(net))
    assert len(handles) == 1, reason
    breaks = sum(du.counters["graph_break"].values())
    for rows in (8, 8, 8, 24, 24, 24):
        torch.testing.assert_close(
            _call(net, rows, rows), _call(ref, rows, rows), rtol = 1e-4, atol = 1e-4
        )
    assert sum(du.counters["graph_break"].values()) == breaks
    s = handles[0].stats
    assert s["captures"] == 2 and s["replays"] == 6 and s["fallbacks"] == 0
    first = _call(net, 5, 24)
    assert torch.equal(_call(net, 5, 24), first)
    handles[0].free()


def test_a_static_step_skip_under_the_hooks_is_replayed_past_only_while_it_plans_no_skip():
    _cuda()
    from core.inference import diffusion_step_skip as sk

    ref = _net().cuda()
    net = _net()
    pipe = _pipe(net)
    assert sk.install_static_step_skip(pipe, settings = {"min_steps": 12}) == sk.TC_STATIC
    _streamed(net, resident_mib = 64)
    cg.arm_after_placement(pipe)
    step = cg.arm_block_graphs(
        pipe,
        {"cuda_graph": True},
        target = _target(),
        family = types.SimpleNamespace(),
        hooked = True,
        pinned = True,
    )[0]
    skip = step.placement.skip
    assert isinstance(skip, sk.StaticStepSkip)
    skip.reset(8)
    for seed in range(3):
        assert torch.equal(_call(net, seed), _call(ref, seed))
    assert step.stats["captures"] == 1 and step.stats["replays"] == 3
    assert skip.stats["calls"] == 3 and skip.stats["computed"] == 3
    skip.reset(16)
    assert not all(skip.plan)
    replays = step.stats["replays"]
    for i in range(16):
        _call(net, 0)
        skip.step_end()
    assert step.stats["replays"] == replays and step.stats["skip_eager"] == 16
    assert skip.stats["skipped"] > 0
    assert (
        isinstance(step.fallback_handle, bg.BlockGraphSet)
        and step.fallback_handle.stats["replays"] > 0
    )
    step.free()
    sk.uninstall_static_step_skip(pipe)


@pytest.mark.parametrize("timed", [True, False])
def test_a_key_warmed_by_its_timed_eager_steps_records_without_another_warm_up(monkeypatch, timed):
    _cuda()
    if timed:
        monkeypatch.delenv(cg.SPEED_CHECK_ENV)
    ref = _net().cuda()
    net, handle = _armed_streamed(monkeypatch)
    recorded = []
    real = handle.placement.recorder

    def recorder(call):
        inner = real(call)

        def record(*a, **k):
            recorded.append(torch.cuda.current_stream())
            return inner(*a, **k)

        return record

    handle.placement.recorder = recorder
    for seed in range(5):
        assert torch.equal(_call(net, seed), _call(ref, seed))
    assert handle.stats["captures"] == 1
    # timed: eager steps already warmed it; untimed: side-stream warm-ups run first
    assert len(recorded) == (1 if timed else 1 + cg.WARMUP_ITERS)
    assert handle.stats["speed_eager"] == (3 if timed else 0)
    handle.free()


class _SyncsWhileRecorded(Block):
    def forward(self, x):
        y = super().forward(x)
        if torch.cuda.is_current_stream_capturing():
            y = y + float(y.sum().item()) * 0  # a host read invalidates the recording
        return y


def test_a_failed_block_recording_takes_the_allocator_off_its_pool():
    """As for the whole step (test_diffusion_offload_cuda_graph): capture_end raises before endAllocateToPool, so a
    failed per-block recording left the allocator on the block pool and every later empty_cache freed nothing."""
    _cuda()
    torch.manual_seed(0)
    net = Net(blocks = 2).cuda()
    net.blocks[1] = _SyncsWhileRecorded(256).cuda()
    Net._repeated_blocks = ["Block", "_SyncsWhileRecorded"]
    try:
        handle, reason = bg.install_block_graphs(net, device = "cuda")
        assert reason == "armed"
        x, t = _inputs(0)
        with torch.no_grad():
            for _ in range(3):
                net(x, t, return_dict = False)
        torch.cuda.synchronize()
        assert handle.graphs[1].poisoned and handle.stats["fallbacks"] == 1
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_reserved()
        block = torch.empty(256 << 20, dtype = torch.uint8, device = "cuda")
        del block
        torch.cuda.empty_cache()
        # a stuck dead pool would keep the 256 MiB segment reserved
        assert torch.cuda.memory_reserved() < baseline + (128 << 20)
        torch.randn(4, device = "cuda")
        handle.free()
        other = Net(blocks = 2).cuda()
        again, _ = bg.install_block_graphs(other, device = "cuda")
        with torch.no_grad():
            for _ in range(3):
                other(x, t, return_dict = False)
        assert again.stats["captures"] == 2 and again.stats["replays"] == 4
        again.free()
    finally:
        Net._repeated_blocks = ["Block"]


def test_releasing_an_offloaded_step_graph_frees_its_capture_stream_workspace(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append(1), raising = False
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(2))
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.placement = _Placement(streams = True)
    handle._release()
    assert calls == [1, 2]
    handle.placement = None
    calls.clear()
    handle._release()
    assert calls == [2]


def test_freeing_a_streamed_step_graph_hands_the_slot_ring_back(monkeypatch):
    _cuda()
    net, handle = _armed_streamed(monkeypatch)
    pf = op.module_prefetcher(net)
    _call(net, 0)
    _call(net, 1)
    assert pf.slot_raw
    handle.free()
    assert not pf.slot_raw and not pf.slot_of
    ref = _net().cuda()
    assert torch.equal(_call(net, 2), _call(ref, 2))


def test_the_first_capture_keeps_its_pool_while_its_graph_lives(monkeypatch):
    _cuda()
    monkeypatch.setattr(cg, "_POOL_BOX", [None])
    net = _net().cuda()
    handle = cg.GraphedForward(net, logger = None).enable()
    _call(net, 0)
    _call(net, 1)
    assert handle.stats["captures"] == 1 and cg._POOL_BOX[0] is not None
    cg._drop_pool_if_unused()
    assert cg._POOL_BOX[0] is not None
    handle.free()


def test_freeing_a_resident_graph_keeps_offload_hooks_added_after_it():
    _cuda()
    net = _net()
    handle = cg.GraphedForward(net, logger = None).enable()
    _streamed(net)
    hooked = net.__dict__.get("forward")
    assert hooked is not None and hooked is not handle
    handle.free()
    assert net.__dict__.get("forward") is hooked
    ref = _net().cuda()
    assert torch.equal(_call(net, 0), _call(ref, 0))


def _probe_handle():
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.stats = {"eager_calls": 0, "speed_eager": 0}
    handle._dropped, handle._slower, handle._judge, handle._warmed = set(), None, {}, {}
    handle.max_graphs = 4
    handle.plan = None
    return handle


def test_a_speed_probe_without_memory_for_its_copies_runs_the_callers_step(monkeypatch):
    handle = _probe_handle()

    def oom(live):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(handle, "_statics", oom)
    x = torch.ones(2)
    assert torch.equal(handle._timed_eager(lambda t: t * 2, (x,), {}, "k"), x * 2)
    assert "k" in handle._dropped


def test_no_probe_step_is_timed_while_another_graph_records(monkeypatch):
    handle = _probe_handle()
    monkeypatch.setattr(handle, "_statics", lambda live: list(live))
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 1)
    x = torch.ones(2)
    for _ in range(3):
        handle._timed_eager(lambda t: t + 1, (x,), {}, "k")
    assert handle._judge["k"]["seen"] == 3 and handle._judge["k"]["eager"] == []


def test_a_ring_drop_waits_while_another_graph_records(monkeypatch):
    pf = op.GroupPrefetcher.__new__(op.GroupPrefetcher)
    pf._drop_slots_at_end = False
    dropped = []
    monkeypatch.setattr(pf, "disable_slots", lambda: dropped.append(1), raising = False)
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 1)
    pf._drop_ring()
    assert dropped == [] and pf._drop_slots_at_end
    monkeypatch.setattr(cg, "_CAPTURE_DEPTH", 0)
    pf._drop_ring()
    assert dropped == [1] and not pf._drop_slots_at_end


def test_a_capture_collision_leaves_the_cuda_generators_alone(monkeypatch):
    _cuda()
    net = _net().cuda()
    handle = cg.GraphedForward(net, logger = None).enable()
    healed = []
    monkeypatch.setattr(cg, "_heal_generators", lambda: healed.append(1))

    def collided(*args, **kwargs):
        raise RuntimeError("beginAllocateToPool: already recording to mempool_id")

    monkeypatch.setattr(handle, "_capture", collided)
    ref = _net().cuda()
    for seed in (0, 1):
        assert torch.equal(_call(net, seed), _call(ref, seed))
    assert handle.poisoned and healed == []
    handle.free()


def test_an_offload_only_family_gets_no_resident_graph():
    net = _net()
    pipe = _pipe(net)
    pipe._unsloth_cuda_graph_offload_only = True
    handles, reason = cg.arm_after_placement(pipe)
    assert handles == () and "offloaded steps only" in reason
    pipe._unsloth_cuda_graph_offload_only = False
    handles, _ = cg.arm_after_placement(pipe)
    assert len(handles) == 1 and handles[0].placement is None
    cg.uninstall_all(handles)
