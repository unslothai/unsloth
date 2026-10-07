# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3's block-streamed denoiser runs on the event-fenced prefetch (``diffusion_offload_prefetch``).

H3 streams through ``stream_prequantized_module`` (diffusers block-level stream groups), pins its top-level group onto
the blocks' copy stream and keeps a resident prefix (``H3Residency``), none of which reached the prefetcher, so every
streamed group was fenced on the host. The CUDA cases build that exact layout and check: no host synchronization per
forward, bit-identical output to the resident model across residency re-fits, the first streamed block prefetched
behind a resident prefix, and the kill switch keeping diffusers' path.
"""

from __future__ import annotations

import copy
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.hooks")

import core.inference.video_minimax_h3_residency as res  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        res.H3_STREAM_PREFETCH_ENV,
        "UNSLOTH_DIFFUSION_ASYNC_PREFETCH",
        "UNSLOTH_DIFFUSION_PREFETCH_DEPTH",
        "UNSLOTH_H3_TOP_GROUP_PIN",
    ):
        monkeypatch.delenv(name, raising = False)


def test_depth_keeps_the_prefetch_inside_the_reserved_stream_window():
    gb = int(1e9)
    # ~0.39 GB blocks: running + 2 ahead fits the 1.5 GB window.
    assert res.h3_stream_prefetch_depth([int(0.39 * gb)] * 50, 2) == 2
    assert res.h3_stream_prefetch_depth([int(0.6 * gb)], 2) == 1
    assert res.h3_stream_prefetch_depth([int(2 * gb)], 4) == 1
    assert res.h3_stream_prefetch_depth([int(0.1 * gb)], 2) == 2
    assert res.h3_stream_prefetch_depth([], 3) == 3


def test_window_is_diffusers_footprint_top_plus_two_blocks():
    assert res.h3_stream_prefetch_window(800, [390, 380]) == 800 + 2 * 390
    assert res.h3_stream_prefetch_window(0, [390]) == 780
    assert res.h3_stream_prefetch_window(0, []) == 0


def test_resident_onload_kicks_the_prefetcher_and_stays_a_noop_without_one():
    calls = []

    class _PF:
        def kick(self):
            calls.append("kick")

    class _G:
        pass

    g = _G()
    res._resident_onload(g)()
    assert calls == []
    g._unsloth_prefetcher = _PF()
    res._resident_onload(g)()
    assert calls == ["kick"]


def test_kill_switch_and_unstreamed_module_leave_it_alone(monkeypatch):
    monkeypatch.setenv(res.H3_STREAM_PREFETCH_ENV, "0")
    assert res.install_h3_stream_prefetch(torch.nn.Linear(2, 2), "cuda") == 0
    monkeypatch.delenv(res.H3_STREAM_PREFETCH_ENV)
    assert res.install_h3_stream_prefetch(torch.nn.Linear(2, 2), "cuda") == 0


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: diffusers stream group offload")


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.nn.functional.gelu(self.lin(x))


class _DiT(torch.nn.Module):
    """H3's layout: top-level embedders / output and ``transformer_blocks``; the top-level group is about twice a
    block, like H3's 0.8 GB top-level group against 0.39 GB blocks."""

    def __init__(
        self,
        width = 1024,
        blocks = 8,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, width)
        self.embed = torch.nn.Linear(width, 2 * width)
        self.transformer_blocks = torch.nn.ModuleList(_Block(width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 64)

    def forward(self, x):
        x = self.proj_in(x)
        x = x + self.embed(x)[..., : x.shape[-1]]
        for block in self.transformer_blocks:
            x = block(x)
        return self.proj_out(x)


def _h3_streamed(
    net,
    monkeypatch,
    prefetch = True,
    outside_inference = True,
    pin_top = True,
    unpinned = False,
):
    """What the H3 load does at a capped tier: stream groups with record_stream, pinned top group, prefetch."""
    import inspect

    from diffusers.hooks import apply_group_offloading

    import core.inference.video_minimax_h3_te as te

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    kwargs = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "offload_type": "block_level",
        "num_blocks_per_group": 1,
        "use_stream": True,
    }
    params = inspect.signature(apply_group_offloading).parameters
    if "record_stream" not in params:
        pytest.skip("diffusers without record_stream")
    kwargs["record_stream"] = True
    if "non_blocking" in params:
        kwargs["non_blocking"] = True
    if unpinned:
        kwargs["low_cpu_mem_usage"] = True
    apply_group_offloading(net, **kwargs)
    if outside_inference:
        from core.inference.diffusion_prequant import _move_groups_outside_inference_mode
        _move_groups_outside_inference_mode(net)
    assert res.pin_streamed_top_level_group(net) is pin_top
    n = res.install_h3_stream_prefetch(net, "cuda") if prefetch else 0
    return n


def _count_late_copies(pf):
    """Count block copies issued after their predecessor block was offloaded (serialized behind its compute)."""
    late = {"n": 0, "offloaded": set()}
    issue, offload, begin = pf._issue, pf.offload, pf.begin

    def _begin():
        late["offloaded"] = set()
        return begin()

    def _offload(group):
        late["offloaded"].add(id(group))
        return offload(group)

    def _issue(group):
        gid = id(group)
        if pf.active and pf.on_order and gid in pf.order:
            k = pf.order.index(gid)
            if k > 0 and pf.order[k - 1] in late["offloaded"]:
                late["n"] += 1
        return issue(group)

    pf.begin, pf.offload, pf._issue = _begin, _offload, _issue
    return late


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


@pytest.mark.parametrize("resident", [0, 1, 3])
def test_streamed_h3_forward_has_no_host_sync_and_is_bit_identical(resident, monkeypatch):
    _cuda()
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    block = sum(p.numel() * p.element_size() for p in net.transformer_blocks[0].parameters())
    # The 1.5 GB H3 reserve scaled: 3.85 blocks of H3's 0.39 GB.
    monkeypatch.setattr(res, "H3_STREAM_WINDOW_GB", 3.85 * block / 1e9)
    covered = _h3_streamed(net, monkeypatch)
    assert covered == 1 + len(net.transformer_blocks)
    pf = module_prefetcher(net)
    assert pf is not None
    residency = res.H3Residency(net, "cuda")
    residency.apply(resident > 0, resident)
    late = _count_late_copies(pf)
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        assert torch.equal(net(x), want)
        late["n"] = 0
        for _ in range(3):
            got, syncs = _count_syncs(lambda: net(x))
            assert syncs == 0
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    assert pf.stats["prefetched"] > 0 and pf.stats["missed"] == 0
    # Each copy must queue before the predecessor's offload fences the stream, else no overlap.
    assert late["n"] == 0, late
    assert pf.peak_inflight_bytes <= pf.window
    if resident:
        assert all(res.is_resident(g) for g in residency.blocks[:resident])
    assert net.transformer_blocks[-1].lin.weight.device.type == "cpu"


def test_streamed_h3_forward_under_inference_mode(monkeypatch):
    """The inference_mode wrappers stream_prequantized_module installs are lifted and re-applied around the prefetch."""
    _cuda()
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    assert _h3_streamed(net, monkeypatch) == 1 + len(net.transformer_blocks)
    pf = module_prefetcher(net)
    residency = res.H3Residency(net, "cuda")
    residency.apply(True, 2)
    x = torch.randn(4, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(3):
            got, syncs = _count_syncs(lambda: net(x))
            torch.cuda.synchronize()
            assert torch.equal(got, want)
        assert syncs == 0
    assert pf.stats["prefetched"] > 0
    for group in residency.blocks[2:]:
        assert pf.owns(group)


def test_refit_between_requests_stays_bit_identical(monkeypatch):
    _cuda()
    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    _h3_streamed(net, monkeypatch)
    residency = res.H3Residency(net, "cuda")
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for top, n in ((True, 4), (True, 1), (False, 0), (True, 6), (False, 2)):
            residency.apply(top, n)
            for _ in range(2):
                got, syncs = _count_syncs(lambda: net(x))
                torch.cuda.synchronize()
                assert torch.equal(got, want), (top, n)
            assert syncs == 0, (top, n)
        res.release_all(residency)
        assert torch.equal(net(x), want)


def test_without_the_prefetch_every_streamed_group_waits_on_the_host(monkeypatch):
    """Control: the H3 layout as main builds it synchronizes the host for every streamed group."""
    _cuda()
    torch.manual_seed(0)
    net = _DiT().eval()
    monkeypatch.setenv(res.H3_STREAM_PREFETCH_ENV, "0")
    assert _h3_streamed(net, monkeypatch) == 0
    x = torch.randn(4, 64, device = "cuda")
    with torch.no_grad():
        net(x)
        _, syncs = _count_syncs(lambda: net(x))
    assert syncs >= len(net.transformer_blocks)


def test_top_pin_kill_switch_keeps_diffusers_top_group(monkeypatch):
    """UNSLOTH_H3_TOP_GROUP_PIN=0: the generic top-group adoption must not pin it behind the H3 switch's back."""
    _cuda()
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    import core.inference.diffusion_offload_prefetch as op

    monkeypatch.setenv(res.H3_TOP_GROUP_PIN_ENV, "0")
    seen = []
    adopt = op._adopt_top_group

    def _spy(module, *a, **k):
        top_group, _ = res.h3_offload_groups(module)
        seen.append(top_group.stream is None and "onload_" not in top_group.__dict__)
        return adopt(module, *a, **k)

    monkeypatch.setattr(op, "_adopt_top_group", _spy)
    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    assert _h3_streamed(net, monkeypatch, pin_top = False) == len(net.transformer_blocks)
    top, _ = res.h3_offload_groups(net)
    pf = module_prefetcher(net)
    assert seen == [False]
    assert top.stream is None and not pf.owns(top)
    x = torch.randn(4, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(2):
            got = net(x)
            torch.cuda.synchronize()
            assert torch.equal(got, want)


def test_capped_pinned_host_prefetches_one_ahead(monkeypatch):
    """Windows / WSL2 pin each in-flight group on the fly under a ~1 GiB cap: one ahead, as diffusers holds. Their
    streamed blocks have unpinned host copies, which elsewhere keep diffusers' onload."""
    _cuda()
    import core.inference.diffusion_memory as dm
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: True)
    net = _DiT().eval()
    assert _h3_streamed(net, monkeypatch, unpinned = True) == 1 + len(net.transformer_blocks)
    assert module_prefetcher(net).depth == 1


def test_unpinned_streamed_blocks_keep_diffusers_onload(monkeypatch):
    """A host whose pin budget refused the denoiser streams it from unpinned copies; the prefetcher would pin a fresh
    copy of every streamed tensor per onload, so diffusers' own onload keeps them."""
    _cuda()
    import core.inference.diffusion_memory as dm
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    assert _h3_streamed(net, monkeypatch, unpinned = True) == 0
    assert module_prefetcher(net) is None
    x = torch.randn(16, 64, device = "cuda")
    with torch.no_grad():
        for _ in range(2):
            assert torch.equal(net(x), ref(x))
