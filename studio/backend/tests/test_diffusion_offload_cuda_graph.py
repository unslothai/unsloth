# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CUDA graphs over an offloaded denoiser (``diffusion_cuda_graph.arm_after_placement``).

Block streaming: Studio's ``_apply_group_offload`` with the event-fenced prefetch; one graph records the whole forward
with the copy-stream copies inside it. Partial residency: some groups pinned, the rest streamed; a release / restore
moves the resident weights and the graph records again. Model offload: accelerate's CPU-offload hook moves the module
eagerly, the replay sits under it and records again when the weights land elsewhere. Every case checks the replay is
bit-identical to the eager forward of the same placement and that a replay makes no host synchronization."""

from __future__ import annotations

import copy
import types
import warnings

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_cuda_graph as cg  # noqa: E402
import core.inference.diffusion_memory as dm  # noqa: E402
import core.inference.diffusion_offload_prefetch as op  # noqa: E402
import core.inference.diffusion_speed as ds  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        cg.OFFLOAD_CUDA_GRAPH_ENV,
        cg.CUDA_GRAPH_DISABLE_ENV,
        cg.SPEED_CHECK_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_DIFFUSION_PIN_TOP_GROUP",
    ):
        monkeypatch.delenv(name, raising = False)
    # Timing a tiny forward on a shared card is noise; these tests check engagement and exactness.
    monkeypatch.setenv(cg.SPEED_CHECK_ENV, "0")
    yield
    cg._POOL_BOX[0] = None


class _Net(torch.nn.Module):
    def __init__(
        self,
        width = 512,
        blocks = 8,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, width)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 64)

    def forward(
        self,
        x,
        t,
        return_dict = True,
    ):
        x = self.proj_in(x) * t
        for block in self.blocks:
            x = torch.nn.functional.gelu(block(x))
        out = self.proj_out(x)
        return (out,) if not return_dict else {"sample": out}


def _cuda():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        pytest.skip("needs NVIDIA CUDA")
    pytest.importorskip("diffusers.hooks")


def _pipe(net):
    return types.SimpleNamespace(transformer = net, components = {"transformer": net})


def _count_syncs(fn):
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            out = fn()
        n = sum(1 for w in caught if "synchroniz" in str(w.message).lower())
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    return out, n


def _inputs(seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return torch.randn(16, 64, generator = g).cuda(), torch.rand(1, generator = g).cuda()


def _run(net, seeds):
    outs = []
    with torch.no_grad():
        for seed in seeds:
            x, t = _inputs(seed)
            outs.append(net(x, t, return_dict = False)[0].clone())
    torch.cuda.synchronize()
    return outs


def _streamed(net, resident_mib = None):
    kwargs = {"resident_transformer_mib": resident_mib} if resident_mib else {}
    assert dm._apply_group_offload(_pipe(net), "cuda", None, **kwargs)
    return net


def _eager_reference(resident_mib, seeds):
    torch.manual_seed(0)
    net = _streamed(_Net(), resident_mib)
    return _run(net, seeds)


@pytest.mark.parametrize("resident_mib", [None, 3])
def test_block_streamed_forward_replays_bit_identically_with_no_host_sync(resident_mib):
    _cuda()
    seeds = list(range(6))
    want = _eager_reference(resident_mib, seeds)
    torch.manual_seed(0)
    net = _streamed(_Net(), resident_mib)
    pipe = _pipe(net)
    handles, reason = cg.arm_after_placement(pipe)
    assert len(handles) == 1 and handles[0].placement.mode == "group", reason
    assert reason.startswith("captured")
    got = _run(net, seeds[:2])  # first call records, the second replays
    assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 2
    with torch.no_grad():
        for seed in seeds[2:]:
            x, t = _inputs(seed)
            out, syncs = _count_syncs(lambda: net(x, t, return_dict = False)[0].clone())
            assert syncs == 0
            got.append(out)
    torch.cuda.synchronize()
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert handles[0].stats["captures"] == 1 and handles[0].stats["eager_calls"] == 0
    assert net.blocks[-1].weight.device.type == "cpu"
    pf = op.module_prefetcher(net)
    assert pf.stats["missed"] == 0
    # The pool (activations plus streamed copies) is what the generate-time guard leaves uncredited.
    assert cg.live_pool_bytes() >= handles[0].stats["pool_bytes"] > 0
    assert 0 < cg.live_pool_free_bytes() <= cg.live_pool_bytes()
    cg.uninstall_all(handles)
    assert net.__dict__["forward"] is handles[0].placement.prev


def test_release_and_restore_records_again_and_stays_exact():
    _cuda()
    seeds = [0, 1, 2, 3]
    want = _eager_reference(3, seeds)
    torch.manual_seed(0)
    net = _streamed(_Net(), 3)
    pipe = _pipe(net)
    handles, _ = cg.arm_after_placement(pipe)
    got = _run(net, seeds[:2])
    restore = dm.release_resident_groups(pipe, 10_000, None)
    assert restore is not None
    got += _run(net, seeds[2:3])
    restore()
    got += _run(net, seeds[3:])
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert handles[0].stats["invalidations"] >= 1 and handles[0].stats["captures"] >= 2


def test_unpinned_host_copies_keep_the_forward_eager_with_the_reason(monkeypatch):
    _cuda()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN", "0")
    torch.manual_seed(0)
    net = _Net()
    pipe = _pipe(net)
    assert dm._apply_group_offload(pipe, "cuda", None, stream_text_encoders = False)
    placement, why = cg.offload_placement(net)
    if why is None:
        pytest.skip("this diffusers pins the stream groups regardless")
    assert placement is None and ("pinned" in why or "prefetch" in why)
    handles, reason = cg.arm_after_placement(pipe)
    assert handles == () and "Net" in reason


def test_kill_switch_keeps_the_old_refusal(monkeypatch):
    monkeypatch.setenv(cg.OFFLOAD_CUDA_GRAPH_ENV, "0")
    assert not cg.offload_graphs_enabled()
    pipe = types.SimpleNamespace(_unsloth_cuda_graph_after_placement = True, _unsloth_cuda_graphs = ())
    monkeypatch.setattr(ds, "_denoiser_dits", lambda p: [])
    applied = ds.arm_graphs_after_placement(pipe, {"cuda_graph": False})
    assert applied["cuda_graph"] is False and pipe._unsloth_cuda_graph_reason == "offload active"


def test_sequential_offload_is_refused_with_the_reason():
    net = _Net()
    net.blocks[0]._hf_hook = object()
    placement, why = cg.offload_placement(net)
    assert placement is None and "sequential" in why


def test_model_offload_replays_under_the_hook_and_records_again_when_weights_move():
    _cuda()
    accelerate = pytest.importorskip("accelerate")
    from accelerate import cpu_offload_with_hook

    seeds = [0, 1, 2, 3, 4]
    torch.manual_seed(0)
    ref = _Net().cuda()
    want = _run(ref, seeds)
    torch.manual_seed(0)
    net = _Net()
    other = torch.nn.Linear(4, 4)
    net, hook = cpu_offload_with_hook(net, execution_device = "cuda")
    _other, other_hook = cpu_offload_with_hook(
        other, execution_device = "cuda", prev_module_hook = hook
    )
    pipe = _pipe(net)
    handles, reason = cg.arm_after_placement(pipe)
    assert len(handles) == 1 and handles[0].placement.mode == "model", reason
    got = _run(net, seeds[:3])
    assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 3
    hook.offload()
    assert net.proj_in.weight.device.type == "cpu"
    got += _run(net, seeds[3:])
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert handles[0].stats["captures"] == 1 + handles[0].stats["invalidations"]
    del accelerate, other_hook


def test_model_offload_re_records_a_moved_module_without_a_warm_up():
    """Every onload of a model-offloaded denoiser may land at new addresses: the same key records again with no
    warm-up step (its kernels exist), and replays stay exact across many moves."""
    _cuda()
    pytest.importorskip("accelerate")
    from accelerate import cpu_offload_with_hook

    torch.manual_seed(0)
    ref = _Net().cuda()
    torch.manual_seed(0)
    net, hook = cpu_offload_with_hook(_Net(), execution_device = "cuda")
    handles, reason = cg.arm_after_placement(_pipe(net))
    assert len(handles) == 1, reason
    handle = handles[0]
    calls = []
    stock = handle.placement.eager

    def counting(*a, **k):
        calls.append(1)
        return stock(*a, **k)

    handle.orig = counting
    flushes = []
    handle._release = lambda: flushes.append(1)  # gc + empty_cache: a move must not pay them
    for render in range(4):
        seeds = [10 * render + i for i in range(5)]
        want = _run(ref, seeds)
        got = _run(net, seeds)
        for a, b in zip(got, want):
            assert torch.equal(a, b)
        hook.offload()
        torch.cuda.empty_cache()
        # Hold a block of the freed size so the next onload likely lands elsewhere.
        _hold = torch.empty(4 << 20, device = "cuda")
    moves = handle.stats["invalidations"]
    assert moves >= 1
    assert handle.stats["captures"] == 1 + moves, (handle.stats, handle.capture_error)
    assert handle.placement.refusal(handle.stats) is None
    # First recording: WARMUP_ITERS warm-ups + capture; each re-record: the capture only.
    assert len(calls) == cg.WARMUP_ITERS + 1 + moves
    assert flushes == []
    del _hold


def test_model_offload_stops_recording_only_when_moves_outpace_replays():
    placement = cg.OffloadPlacement.__new__(cg.OffloadPlacement)
    placement.mode = "model"
    placement._checked = object()
    many = cg.OffloadPlacement.MODEL_MAX_INVALIDATIONS

    def judge(stats, where):
        placement._fp = where
        return placement.refusal(stats)

    assert judge({"invalidations": 1}, 1) is None
    assert "new addresses" in judge({"invalidations": many, "captures": many, "replays": many}, 2)
    assert judge({"invalidations": 5, "captures": 5, "replays": 5 * 8}, 3) is None
    # Judged once per placement: the latest recording does not count until the next move.
    assert placement.refusal({"invalidations": 5, "captures": 6, "replays": 5 * 8 + 1}) is None


class _FailsInCapture(_Net):
    """A forward that makes a host-syncing call after its first blocks, but only while a graph records it."""

    def forward(
        self,
        x,
        t,
        return_dict = True,
    ):
        x = self.proj_in(x) * t
        for i, block in enumerate(self.blocks):
            x = torch.nn.functional.gelu(block(x))
            if i == 3 and torch.cuda.is_current_stream_capturing():
                x = (
                    x + float(x.sum().item()) * 0
                )  # host read: invalidates the capture with copies already queued
        out = self.proj_out(x)
        return (out,) if not return_dict else {"sample": out}


def test_a_failed_streamed_capture_falls_back_eager_without_stale_copies():
    _cuda()
    seeds = [0, 1, 2]
    torch.manual_seed(0)
    want = _run(_streamed(_FailsInCapture()), seeds)
    torch.manual_seed(0)
    net = _streamed(_FailsInCapture())
    handles, reason = cg.arm_after_placement(_pipe(net))
    assert len(handles) == 1, reason
    stream = torch.cuda.current_stream()
    got = _run(net, seeds)
    assert torch.cuda.current_stream() == stream
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert handles[0].poisoned and handles[0].stats["captures"] == 0
    assert op.module_prefetcher(net).ready == {} or all(
        e is None for e in op.module_prefetcher(net).ready.values()
    )
    assert net.blocks[-1].weight.device.type == "cpu"
    cg.uninstall_all(handles)


class _SyncsOnStock(_Net):
    """Stock forward reads a value on the host (as HunyuanImage-2.1's text merge does); its rewrite does not."""

    def forward(
        self,
        x,
        t,
        return_dict = True,
    ):
        x = self.proj_in(x) * t
        gate = x.new_tensor(1.0) if bool((x.abs().sum() >= 0).item()) else x.new_tensor(0.0)
        for block in self.blocks:
            x = torch.nn.functional.gelu(block(x))
        out = self.proj_out(x) * gate
        return (out,) if not return_dict else {"sample": out}


def _capture_safe_forward(
    self,
    x,
    t,
    return_dict = True,
):
    x = self.proj_in(x) * t
    total = x.abs().sum()
    gate = torch.where(total >= 0, torch.ones_like(total), torch.zeros_like(total))
    for block in self.blocks:
        x = torch.nn.functional.gelu(block(x))
    out = self.proj_out(x) * gate
    return (out,) if not return_dict else {"sample": out}


def test_a_capture_safe_rewrite_records_under_block_offload(monkeypatch):
    """The block-offload hook chain ends in the bound class forward; a capture-safe rewrite takes that slot while the
    graph is installed, so a family whose stock forward syncs the host still records when streamed."""
    _cuda()
    import core.inference.diffusion_capture_safe as cs

    monkeypatch.setattr(
        cs,
        "resolve",
        lambda cls: (_capture_safe_forward, None) if cls is _SyncsOnStock else (None, None),
    )
    seeds = [0, 1, 2, 3]
    torch.manual_seed(0)
    want = _run(_streamed(_SyncsOnStock()), seeds)
    torch.manual_seed(0)
    net = _streamed(_SyncsOnStock())
    inner = cg._innermost_class_forward(net)
    assert inner is not None and inner.forward.__func__ is _SyncsOnStock.forward
    handles, reason = cg.arm_after_placement(_pipe(net))
    assert len(handles) == 1 and handles[0].placement.mode == "group", reason
    got = _run(net, seeds[:2])
    with torch.no_grad():
        for seed in seeds[2:]:
            x, t = _inputs(seed)
            out, syncs = _count_syncs(lambda: net(x, t, return_dict = False)[0].clone())
            assert syncs == 0
            got.append(out)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert not handles[0].poisoned, handles[0].capture_error
    assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 4
    cg.uninstall_all(handles)
    assert inner.forward.__func__ is _SyncsOnStock.forward


class _Branchy(_Net):
    """Branches on Python state a pre-hook sets per call (HunyuanVideo-1.5's null-mask flag)."""

    flag = False

    def forward(
        self,
        x,
        t,
        return_dict = True,
    ):
        x = self.proj_in(x) * t
        if self.flag:
            x = x * 2
        out = self.proj_out(x)
        return (out,) if not return_dict else {"sample": out}


def test_per_call_python_state_keys_the_graph():
    _cuda()
    torch.manual_seed(0)
    net = _Branchy().cuda()
    net._unsloth_graph_key_extra = lambda: net.flag
    handle = cg.GraphedForward(net).enable()
    try:
        outs = {}
        with torch.no_grad():
            for flag in (False, True, False, True):
                net.flag = flag
                x, t = _inputs(1)
                outs.setdefault(flag, []).append(net(x, t, return_dict = False)[0].clone())
                handle.set_bypass(True)
                want = net(x, t, return_dict = False)[0]
                handle.set_bypass(False)
                assert torch.equal(outs[flag][-1], want)
    finally:
        cg.uninstall_all([handle])
    assert handle.stats["captures"] == 2 and handle.stats["replays"] == 4


def test_a_failed_capture_leaves_the_cuda_rng_usable_and_on_its_sequence():
    """An invalidated capture raises in cudaStreamEndCapture before torch ends the generators' capture; every later
    eager CUDA draw (a pipeline's noise) then failed for the rest of the process."""
    _cuda()
    torch.cuda.manual_seed(123)
    want = torch.randn(8, device = "cuda")
    torch.manual_seed(0)
    net = _FailsInCapture().cuda()
    handle = cg.GraphedForward(net).enable()
    try:
        torch.cuda.manual_seed(123)
        _run(net, [0])
        assert handle.poisoned
        got = torch.randn(8, device = "cuda")
    finally:
        cg.uninstall_all([handle])
    assert torch.equal(got, want)


def test_a_failed_capture_takes_the_allocator_off_its_pool(monkeypatch):
    """capture_end raises in cudaStreamEndCapture before endAllocateToPool, so an invalidated capture left its pool in
    the allocator's captures_underway: on torch 2.6 the next empty_cache tripped INTERNAL ASSERT captures_underway.empty()
    and no later capture recorded (seen on an A100); later torch skips the global release instead, so empty_cache freed
    nothing. Reproduced here on any torch by putting the pool back under capture before the raise."""
    _cuda()
    real_end = torch.cuda.CUDAGraph.capture_end
    begin = torch._C._cuda_beginAllocateCurrentStreamToPool
    stuck = []

    def end_like_torch_2_6(self):
        real_end(self)
        stuck.append(self.pool())
        begin(torch.cuda.current_device(), stuck[-1])
        raise RuntimeError("CUDA error: operation failed due to a previous error during capture")

    torch.manual_seed(0)
    net = _streamed(_Net())
    handles, _ = cg.arm_after_placement(_pipe(net))
    try:
        monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", end_like_torch_2_6)
        got = _run(net, [0])
        monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", real_end)
        assert stuck and handles[0].poisoned and handles[0].stats["fallbacks"] == 1
        # Under capture, empty_cache raised INTERNAL ASSERT (torch 2.6) or silently freed nothing (later torch).
        block = torch.empty(256 << 20, dtype = torch.uint8, device = "cuda")
        held = torch.cuda.memory_reserved()
        del block
        torch.cuda.empty_cache()
        assert torch.cuda.memory_reserved() <= held - (256 << 20)
        assert torch.equal(got[0], _eager_reference(None, [0])[0])
        torch.manual_seed(0)
        other = _streamed(_Net())
        again, _ = cg.arm_after_placement(_pipe(other))
        _run(other, [0, 1])
        assert again[0].stats["captures"] == 1 and again[0].stats["replays"] == 2
        cg.uninstall_all(again)
    finally:
        end = (
            getattr(torch._C, "_cuda_endAllocateToPool", None)
            or torch._C._cuda_endAllocateCurrentStreamToPool
        )
        for pool in stuck:
            try:
                end(torch.cuda.current_device(), pool)
            except Exception:  # noqa: BLE001 - the fix already took it off
                pass
        cg.uninstall_all(handles)


def test_model_offload_fingerprint_follows_replaced_buffers():
    """Module.to() puts NEW tensor objects in _buffers on every move: the fingerprint must read the registry each call,
    not a list taken once (that list keeps describing the obsolete buffers)."""
    torch.manual_seed(0)
    net = _Net()
    net.register_buffer("scale", torch.ones(4))
    net._old_forward = net.forward  # what accelerate's CpuOffload leaves for the replay slot
    placement = cg.OffloadPlacement(net, "model")
    before = placement.token()
    net.scale = torch.full((4,), 9.0)  # a new buffer object, as Module.to() makes on a move
    assert placement.token() != before


class _Ev:
    def __init__(
        self,
        ms,
        done = True,
    ):
        self.ms, self.done = ms, done

    def query(self):
        return self.done

    def elapsed_time(self, end):
        return end.ms - self.ms


def _judged(
    eager_ms,
    graph_ms,
    done = True,
    others = (),
):
    """A handle with key "k" timed (and ``others`` already kept), judged once; returns (handle, k's verdict)."""
    import os

    os.environ.pop(cg.SPEED_CHECK_ENV, None)
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.stats, handle.logger, handle._slower, handle.capture_error = {}, None, None, None
    handle.cache = {"k": object(), **{o: object() for o in others}}
    handle._dropped, handle._release = set(), lambda: None
    handle._judge = {
        "k": {
            "seen": 3,
            "eager": [(_Ev(0), _Ev(ms)) for ms in eager_ms],
            "graph": [(_Ev(0), _Ev(ms, done)) for ms in graph_ms],
            "verdict": None,
        },
        **{o: {"seen": 3, "eager": [], "graph": [], "verdict": ""} for o in others},
    }
    handle._judge_keys()
    return handle, handle._judge["k"]["verdict"]


def test_an_offloaded_replay_slower_than_eager_drops_the_graphs():
    """The copy engine runs a replay's onload copies at a different rate per machine (an A100 VM: 16% slower than the
    stream's own copies); a key's first replays are timed against its own eager steps and its graph goes when slower."""
    handle, why = _judged([550.0, 552.0], [655.0, 657.0, 654.0])
    assert why and "slower" in why and handle.cache == {} and handle._dropped == {"k"}
    assert handle.capture_error == {"type": "Refused", "msg": why}
    handle._judge_keys()
    assert handle._judge["k"]["verdict"] == why
    handle.placement, handle.poisoned = None, False
    assert cg.stats([handle])["eager_ms"] == 550.0 and cg.stats([handle])["replay_ms"] == 654.0
    handle, why = _judged([550.0, 552.0], [560.0, 700.0, 710.0])
    assert why == "" and handle.cache
    handle, why = _judged([550.0, 552.0], [540.0, 541.0, 539.0])
    assert why == "" and handle.cache and not handle._dropped
    handle, why = _judged([550.0], [560.0, 561.0, 559.0])
    assert why == "" and handle.cache
    handle, why = _judged([550.0], [655.0, 657.0, 654.0], done = False)
    assert why is None and handle._slower is None and handle.cache
    handle, why = _judged([550.0], [655.0, 657.0])
    assert why is None and handle._slower is None


def test_a_slower_key_drops_only_its_own_graph():
    """On an A100 VM Qwen-Image-2.1's first prompt length replayed at eager speed and the next one 16% slower: only the
    slower key goes eager, the other keeps replaying, and the status stays on."""
    handle, why = _judged([550.0, 552.0], [655.0, 657.0, 654.0], others = ("first",))
    assert why and set(handle.cache) == {"first"} and handle._dropped == {"k"}
    assert handle.capture_error is None


def test_the_eager_reference_is_the_callers_own_steps_not_the_recorded_warm_ups(monkeypatch):
    """The capture's warm-ups run under the recorder, whose copies follow the captured schedule: on an A100 VM that
    schedule was the slow part (a replay 17% slower than eager went unjudged because its warm-ups were as slow). The
    reference is the caller's eager steps on the stock prefetch path, timed before the first capture."""
    _cuda()
    monkeypatch.delenv(cg.SPEED_CHECK_ENV, raising = False)
    monkeypatch.setattr(cg, "SPEED_MARGIN", 1e9)
    monkeypatch.setattr(cg, "SPEED_GAIN", -1e9)
    seeds = list(range(8))
    want = _eager_reference(None, seeds)
    torch.manual_seed(0)
    net = _streamed(_Net())
    handles, _ = cg.arm_after_placement(_pipe(net))
    handle = handles[0]
    recorded_steps = []
    recorder = handle.placement.recorder
    monkeypatch.setattr(
        handle.placement, "recorder", lambda call: recorded_steps.append(1) or recorder(call)
    )
    got = _run(
        net, seeds[:3]
    )  # one untimed (may compile), then SPEED_EAGER_SAMPLES timed, all eager
    assert (
        handle.stats["captures"] == 0 and handle.stats["eager_calls"] == 1 + cg.SPEED_EAGER_SAMPLES
    )
    (state,) = handle._judge.values()
    assert len(state["eager"]) == cg.SPEED_EAGER_SAMPLES
    eager_events = list(state["eager"])
    got += _run(net, seeds[3:])
    assert handle.stats["captures"] == 1 and handle.stats["replays"] == len(seeds) - 3
    assert state["eager"] == eager_events
    assert handle.stats["eager_ms"] > 0 and handle.stats["replay_ms"] > 0 and state["verdict"] == ""
    assert recorded_steps
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    cg.uninstall_all(handles)


def test_the_timed_eager_steps_run_on_capture_like_copies(monkeypatch):
    """Compiled code guards on strides and storage offsets: eager steps on the caller's views compiled a variant the
    capture then compiled again (a cold Qwen-Image-2.1 render on an RTX PRO 6000 went from 16.8 to 23.1 s)."""
    _cuda()
    monkeypatch.delenv(cg.SPEED_CHECK_ENV, raising = False)
    monkeypatch.setattr(cg, "SPEED_MARGIN", 1e9)
    monkeypatch.setattr(cg, "SPEED_GAIN", -1e9)
    torch.manual_seed(0)
    net = _streamed(_Net())
    handles, _ = cg.arm_after_placement(_pipe(net))
    offsets = []
    hook = net.proj_in.register_forward_pre_hook(
        lambda _m, a: offsets.append(a[0].storage_offset())
    )
    try:
        with torch.no_grad():
            for seed in range(3):
                x, t = _inputs(seed)
                padded = torch.cat([torch.zeros(1, 64, device = "cuda"), x])[1:]
                assert padded.storage_offset() == 64
                want = _eager_reference(None, [seed])[0]
                assert torch.equal(net(padded, t, return_dict = False)[0], want)
        assert handles[0].stats["captures"] == 0 and offsets == [0, 0, 0]
    finally:
        hook.remove()
        cg.uninstall_all(handles)


def test_every_key_times_its_own_eager_reference_before_it_records(monkeypatch):
    """A new input shape (a new prompt length) after the first capture is judged on its own eager steps too."""
    _cuda()
    monkeypatch.delenv(cg.SPEED_CHECK_ENV, raising = False)
    monkeypatch.setattr(cg, "SPEED_MARGIN", 1e9)
    monkeypatch.setattr(cg, "SPEED_GAIN", -1e9)
    torch.manual_seed(0)
    net = _streamed(_Net())
    handles, _ = cg.arm_after_placement(_pipe(net))
    handle = handles[0]
    try:
        with torch.no_grad():
            for rows in (16, 16, 16, 16, 24, 24, 24, 24, 24):
                x = torch.randn(rows, 64, device = "cuda")
                t = torch.rand(1, device = "cuda")
                net(x, t, return_dict = False)
        torch.cuda.synchronize()
        assert handle.stats["captures"] == 2 and handle.stats["speed_eager"] == 2 * (
            1 + cg.SPEED_EAGER_SAMPLES
        )
        assert len(handle._judge) == 2 and all(
            len(s["eager"]) == cg.SPEED_EAGER_SAMPLES for s in handle._judge.values()
        )
    finally:
        cg.uninstall_all(handles)


def test_status_stays_on_while_the_eager_reference_is_timed():
    """The timed eager steps before the first capture are not a refusal: no "ran eager (bypassed)" status meanwhile."""
    handle = cg.GraphedForward.__new__(cg.GraphedForward)
    handle.stats = {"captures": 0, "replays": 0, "eager_calls": 3, "speed_eager": 3}
    handle.cache, handle.capture_error, handle.poisoned, handle.placement = {}, None, False, None
    assert cg.never_engaged([handle]) is None
    handle.stats.update(eager_calls = 4)
    assert "ran eager" in cg.never_engaged([handle])


def test_status_names_graphs_dropped_after_replaying():
    """A graph dropped mid-load (slower than eager) must not leave the status saying the step replays."""
    handle, why = _judged([550.0, 552.0], [655.0, 657.0, 654.0])
    handle.stats.update(captures = 1, replays = 5, eager_calls = 40)
    handle.capture_error, handle.poisoned, handle.placement = (
        {"type": "Refused", "msg": why},
        False,
        None,
    )
    resolved = {
        "cuda_graph": {"value": "on", "status": "applied", "reason": "captured per input shape"}
    }
    live, optims = cg.live_status(resolved, ["cuda_graph", "int8"], [handle])
    assert (
        live["cuda_graph"]["value"] == "off"
        and "dropped" in live["cuda_graph"]["reason"]
        and why in live["cuda_graph"]["reason"]
    )
    assert optims == ["int8"] and resolved["cuda_graph"]["value"] == "on"
    handle.cache = {"k": object()}
    handle.capture_error = None
    assert cg.live_status(resolved, ["cuda_graph"], [handle])[0]["cuda_graph"]["value"] == "on"
