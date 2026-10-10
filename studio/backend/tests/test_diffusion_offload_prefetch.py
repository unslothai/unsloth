# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Event-fenced prefetch for block-streamed denoisers (``diffusion_offload_prefetch.py``).

The order / depth / window logic runs on CPU with the copies stubbed. The CUDA cases build real diffusers block-level
stream group offload through Studio's ``_apply_group_offload`` and check: no host synchronization per group (torch's
sync debug mode), bit-identical outputs against the resident model, device memory bounded by the window while the host
runs ahead of a slow GPU, no copy landing in a block the compute stream still reads, and a dropped prefetch never
landing in a reused block.
"""

from __future__ import annotations

import copy
import types
import warnings

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_offload_prefetch as op


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_DIFFUSION_PIN_TOP_GROUP",
    ):
        monkeypatch.delenv(name, raising = False)


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
    log: list = []

    def issue(group):
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


def test_first_forward_records_the_order_and_copies_each_group_when_it_runs():
    pf, groups, log = _cpu_prefetcher("abcd")
    _forward(pf, groups)
    assert pf.order == [id(g) for g in groups]
    assert [e for e in log if e[0] == "copy"] == [("copy", n) for n in "abcd"]
    assert pf.stats["missed"] == 0 and pf.inflight_bytes == 0


def test_later_forwards_keep_depth_groups_in_flight_ahead():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    log.clear()
    pf.begin()
    # no copy at forward start: the top-level upload must reach the copy engine first
    assert not log
    pf.onload(groups[0])
    assert [n for k, n in log if k == "copy"] == ["a", "b", "c"]
    for g in groups:
        if g is not groups[0]:
            pf.onload(g)
        pf.offload(g)
    pf.end()
    assert pf.stats["missed"] == 0 and pf.stats["prefetched"] == 6
    assert pf.peak_inflight_bytes <= pf.window


def test_window_bounds_bytes_in_flight():
    pf, groups, log = _cpu_prefetcher("abcdef", nbytes = 10, depth = 4, window = 25)
    _forward(pf, groups)
    for _ in range(2):
        _forward(pf, groups)
        assert pf.peak_inflight_bytes <= 25


def test_one_group_larger_than_the_window_still_progresses():
    pf, groups, log = _cpu_prefetcher("abc", nbytes = 10, depth = 2, window = 5)
    _forward(pf, groups)
    _forward(pf, groups)
    assert pf.stats["forwards"] == 2 and pf.inflight_bytes == 0


def test_resident_groups_are_skipped_and_not_counted():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    for g in groups[:3]:
        g.resident = True
    log.clear()
    pf.begin()
    assert not log
    pf.kick()
    assert [n for k, n in log if k == "copy"] == ["d", "e"]
    pf.kick()
    assert [n for k, n in log if k == "copy"] == ["d", "e"]
    pf.end()


def test_off_order_forward_stops_prefetch_and_rerecords():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    _forward(pf, groups, skip = "c")
    assert ("free", "c") in log
    assert pf.order == [id(g) for g in groups if g.name != "c"]
    assert pf.inflight_bytes == 0 and not pf.ready


def test_exception_mid_forward_releases_everything_on_the_device():
    pf, groups, log = _cpu_prefetcher("abcdef", depth = 2)
    _forward(pf, groups)
    pf.begin()
    pf.onload(groups[0])
    pf.end()
    assert not pf.ready and pf.inflight_bytes == 0


def test_kill_switch_and_cpu_leave_diffusers_alone(monkeypatch):
    m = torch.nn.Linear(4, 4)
    assert op.install_group_prefetch(m, "cpu") == 0
    monkeypatch.setenv(op.ASYNC_PREFETCH_ENV, "0")
    assert op.install_group_prefetch(m, "cuda") == 0


def test_depth_env(monkeypatch):
    assert op.prefetch_depth() == op.DEFAULT_PREFETCH_DEPTH
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "1")
    assert op.prefetch_depth() == 1
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "99")
    assert op.prefetch_depth() == op.MAX_PREFETCH_DEPTH
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "x")
    assert op.prefetch_depth() == op.DEFAULT_PREFETCH_DEPTH


def test_resident_onload_skips_the_copy_stream_wait_when_fenced(monkeypatch):
    pytest.importorskip("diffusers.hooks")  # _keep_groups_resident is a no-op without this import

    class Stream:
        waits = 0

        def synchronize(self):
            Stream.waits += 1

    module = types.SimpleNamespace(_unsloth_stream_state = {"streamed": 1, "fenced": True})
    stream = Stream()
    groups = [
        types.SimpleNamespace(
            modules = [],
            parameters = [],
            buffers = [],
            offload_leader = object(),
            stream = stream,
            cpu_param_dict = {},
            offload_to_disk_path = None,
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda m: groups)
    dm._keep_groups_resident(module, 1, "cpu")
    module._unsloth_stream_state["streamed"] = 1
    for g in groups:
        g.onload_()
    assert Stream.waits == 0
    kicks = []
    module._unsloth_stream_state["kick"] = lambda: kicks.append(1)
    for g in groups:
        g.onload_()
    assert Stream.waits == 0 and len(kicks) == 2
    del module._unsloth_stream_state["kick"]
    module._unsloth_stream_state["fenced"] = False
    for g in groups:
        g.onload_()
    assert Stream.waits == 2


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

    def forward(
        self,
        x,
        upto = None,
    ):
        x = self.proj_in(x)
        for i, block in enumerate(self.blocks):
            if upto is not None and i >= upto:
                break
            if self.sleep_cycles:
                # simulate a slow GPU so the host queues later copies before this block reads
                torch.cuda._sleep(self.sleep_cycles)
            x = torch.nn.functional.gelu(block(x))
        return self.proj_out(x)


def _streamed(
    net,
    resident_mib = None,
    **kwargs,
):
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    if resident_mib:
        kwargs["resident_transformer_mib"] = resident_mib
    assert dm._apply_group_offload(pipe, "cuda", None, **kwargs)
    return pipe


def _count_syncs(fn):
    """Host synchronizations torch reports (stream / device synchronize, blocking copies) while ``fn`` runs."""
    n = 0
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


@pytest.mark.parametrize("depth", ["1", "2", "3"])
@pytest.mark.parametrize("resident_mib", [None, 13])
def test_streamed_forward_is_bit_identical_with_no_host_sync(depth, resident_mib, monkeypatch):
    _cuda()
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, depth)
    torch.manual_seed(0)
    net = _Net()
    ref = copy.deepcopy(net).cuda()
    _streamed(net, resident_mib)
    pf = op.module_prefetcher(net)
    assert pf is not None and pf.depth == int(depth)
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        net(x)
        for _ in range(3):
            got, syncs = _count_syncs(lambda: net(x))
            assert syncs == 0
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    assert pf.stats["prefetched"] > 0 and pf.stats["missed"] == 0
    assert pf.peak_inflight_bytes <= pf.window
    assert net.blocks[-1].weight.device.type == "cpu"


def test_dense_top_level_upload_never_waits_behind_a_block_copy(tmp_path):
    """The dense top-level group uploads on the compute stream at the forward start. Block copies sharing the copy
    engine with it would hold it back, and the previous forward's last offload already lets the copy stream run, so
    the first block copies of a forward are fenced behind it: no compute-stream upload overlaps a block copy."""
    _cuda()
    import json

    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    net = _Net(width = 4096, blocks = 6)
    _streamed(net)
    pf = op.module_prefetcher(net)
    top = net._diffusers_hook.get_hook("group_offloading").group
    assert top not in pf.groups and "onload_" in top.__dict__ and pf.fence_first
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        net(x)
        net(x)
        torch.cuda.synchronize()
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                net(x)
            torch.cuda.synchronize()
    trace = tmp_path / "trace.json"
    prof.export_chrome_trace(str(trace))
    events = [e for e in json.loads(trace.read_text())["traceEvents"] if e.get("ph") == "X"]
    compute = {e["args"].get("stream") for e in events if e.get("cat") == "kernel"}
    h2d = [e for e in events if e.get("cat") == "gpu_memcpy" and "HtoD" in e["name"]]
    top_up = [(e["ts"], e["ts"] + e["dur"]) for e in h2d if e["args"].get("stream") in compute]
    blocks = [(e["ts"], e["ts"] + e["dur"]) for e in h2d if e["args"].get("stream") not in compute]
    assert len(top_up) == 4 * 4 and len(blocks) == 4 * 2 * len(net.blocks)
    assert not [(a, b) for a in top_up for b in blocks if a[0] < b[1] and b[0] < a[1]]
    assert pf.stats["missed"] == 0


def test_diffusers_stream_path_synchronizes_per_group(monkeypatch):
    """The base this replaces: the same streamed model through diffusers' own prefetch syncs the host per group."""
    _cuda()
    monkeypatch.setenv(op.ASYNC_PREFETCH_ENV, "0")
    torch.manual_seed(0)
    net = _Net()
    _streamed(net)
    assert op.module_prefetcher(net) is None
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        net(x)
        _out, syncs = _count_syncs(lambda: net(x))
    assert syncs >= len(net.blocks)


def test_slow_gpu_keeps_memory_in_the_window_and_reads_the_right_weights(monkeypatch):
    """Host far ahead of the GPU: every later copy is queued before the block that frees its memory has run. The copy
    stream's wait on the offload event keeps those copies out of blocks still being read (outputs bit-identical) and
    the allocator never grows past the window."""
    _cuda()
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "1")
    torch.manual_seed(0)
    net = _Net(width = 2048, blocks = 12, sleep_cycles = 20_000_000)
    with torch.no_grad():
        for b in net.blocks:
            b.weight.mul_(
                1.0 + torch.rand(())
            )  # distinct blocks: a wrong-weight read changes the output
    ref = copy.deepcopy(net).cuda()
    _streamed(net)
    pf = op.module_prefetcher(net)
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        net(x)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        got = net(x)
        host_ahead = not torch.cuda.current_stream().query()
        torch.cuda.synchronize()
        # reserved, not allocated: record_stream frees drop allocated bytes but stay reserved
        peak = torch.cuda.max_memory_reserved() - base
    assert torch.equal(got, want)
    assert host_ahead
    block_bytes = 2048 * 2048 * 4 + 2048 * 4
    # window (2 groups) + activations + top-level group
    assert peak <= pf.window + 2 * block_bytes, (peak, pf.window)


@pytest.mark.parametrize("drop", ["exception", "prefix"])
def test_dropped_prefetch_never_lands_in_a_reused_block(drop, monkeypatch):
    _cuda()
    monkeypatch.setenv(op.PREFETCH_DEPTH_ENV, "3")
    H = 2048

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(H, H, bias = False) for _ in range(8))

        def forward(
            self,
            x,
            upto = 8,
            boom = False,
        ):
            for i, m in enumerate(self.blocks):
                if i >= upto:
                    break
                if boom and i == 2:
                    raise RuntimeError("boom")
                x = m(x)
            return x

    net = Net().to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1.0)
    _streamed(net)
    x = torch.zeros(1, H, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        net(x)
        for _ in range(6):
            try:
                net(x, upto = 2, boom = drop == "exception")
            except RuntimeError:
                pass
            outs = [torch.empty(H, H, dtype = torch.bfloat16, device = "cuda") for _ in range(8)]
            for o in outs:
                o.fill_(-2.0)
            torch.cuda.synchronize()
            assert all(bool((o == -2.0).all()) for o in outs)
            del outs
            net(x)
            torch.cuda.synchronize()
    pf = op.module_prefetcher(net)
    assert not pf.ready and pf.inflight_bytes == 0


def test_partial_residency_release_and_restore_stay_bit_identical():
    _cuda()
    torch.manual_seed(0)
    net = _Net()
    ref = copy.deepcopy(net).cuda()
    pipe = _streamed(net, resident_mib = 17)
    pf = op.module_prefetcher(net)
    groups = dm._offload_groups(net)
    resident = [g for g in groups if getattr(g, "_unsloth_resident", False)]
    assert resident and len(resident) < len(groups)
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for _ in range(2):
            assert torch.equal(net(x), want)
        restore = dm.release_resident_groups(pipe, 100, None)
        assert restore is not None
        for _ in range(2):
            got, syncs = _count_syncs(lambda: net(x))
            assert torch.equal(got, want)
        assert syncs == 0
        restore()
        for _ in range(2):
            assert torch.equal(net(x), want)
    assert pf.stats["missed"] <= len(groups)


def test_unpinned_host_copies_stream_bit_identical(monkeypatch):
    """low_cpu_mem_usage (host RAM too small to pin): each onload pins through the host allocator."""
    _cuda()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN", "0")
    torch.manual_seed(0)
    net = _Net()
    ref = copy.deepcopy(net).cuda()
    _streamed(net, stream_text_encoders = True)
    assert not any(
        t.is_pinned() for g in dm._offload_groups(net) for t in g.cpu_param_dict.values()
    )
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for _ in range(3):
            assert torch.equal(net(x), want)
    assert op.module_prefetcher(net).stats["prefetched"] > 0


@pytest.mark.parametrize("version", [None, 2])
def test_torchao_int8_weights_stream_bit_identical(version):
    """torchao's default int8 weight (v1 on 0.17) and ``Int8Tensor`` (version 2), whose 0.17 ``to`` drops
    ``non_blocking`` in its common ``_to_copy``."""
    _cuda()
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    try:
        from torchao.quantization import Int8WeightOnlyConfig as Cfg
    except ImportError:
        pytest.skip(f"torchao {torchao.__version__} has no Int8WeightOnlyConfig")
    torch.manual_seed(0)
    net = _Net().to(torch.bfloat16)
    try:
        cfg = Cfg() if version is None else Cfg(version = version)
    except TypeError:
        pytest.skip(f"torchao {torchao.__version__} has no versioned Int8WeightOnlyConfig")
    quantize_(net, cfg)
    ref = copy.deepcopy(net).cuda()
    _streamed(net)
    pf = op.module_prefetcher(net)
    assert pf is not None
    top = net._diffusers_hook.get_hook("group_offloading").group
    assert top.stream is not None and pf.owns(top)
    assert all(c.is_pinned() for c in top.cpu_param_dict.values())
    x = torch.randn(4, 64, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref(x)
        want, ref_syncs = _count_syncs(
            lambda: ref(x)
        )  # syncs the resident int8 op does itself (torchao 0.17: 1)
        net(x)
        for _ in range(3):
            got, syncs = _count_syncs(lambda: net(x))
            torch.cuda.synchronize()
            assert torch.equal(got, want)
            # torchao 0.17 int8 `to` drops non_blocking; offload must add no syncs
            assert syncs == ref_syncs
    assert pf.stats["prefetched"] > 0
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    # the torchao wrapper reports its onload device, so check the inner tensors
    assert dm._placed_on(net.proj_in.weight, "cpu", go._is_torchao_tensor)
