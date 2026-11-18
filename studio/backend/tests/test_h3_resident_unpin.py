# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 streamed denoiser: a block resident on the card drops its host copy (the conditioner's resident share
takes that RAM), and a demoted block gets an exact host copy back from the device before it streams again."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")


class _HostGroup:
    """A diffusers-style offload group: ``cpu_param_dict`` holds the host copy, onload gives the weight new storage."""

    def __init__(
        self,
        seed: int,
        n: int = 64,
    ):
        g = torch.Generator().manual_seed(seed)
        lin = torch.nn.Linear(n, n, bias = True)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(n, n, generator = g))
            lin.bias.copy_(torch.randn(n, generator = g))
        self.modules = [lin]
        self.parameters = []
        self.buffers = []
        self.stream = None
        self.cpu_param_dict = {p: p.data for p in lin.parameters()}
        self.where = "cpu"

    def onload_(self):
        for p, host in self.cpu_param_dict.items():
            p.data = host.clone()  # the "device" copy: distinct storage
        self.where = "cuda"

    def offload_(self):
        for p in list(self.modules[0].parameters()):
            p.data = self.cpu_param_dict[p]
        self.where = "cpu"


def _residency(n_blocks: int, logger = None):
    from core.inference.video_minimax_h3_residency import H3Residency, group_payload_bytes

    r = H3Residency.__new__(H3Residency)
    r.top = None
    r.blocks = [_HostGroup(i) for i in range(n_blocks)]
    r.device = "cuda"
    r.logger = logger
    r.top_bytes = 0
    r.block_bytes = [group_payload_bytes(g) for g in r.blocks]
    r.max_blocks = 0
    r.host_yield = None
    r.released_bytes = 0
    return r


@pytest.fixture(autouse = True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("UNSLOTH_H3_RESIDENT_UNPIN", raising = False)


def test_a_resident_block_drops_its_host_copy():
    from core.inference import video_minimax_h3_residency as res

    r = _residency(4)
    r.fit(r.block_bytes[0] * 2, initial = True)
    assert r.resident_blocks() == 2
    for g in r.blocks[:2]:
        assert g.cpu_param_dict == {}
        assert res.released_host_bytes(g) == res.group_payload_bytes(g)
    for g in r.blocks[2:]:
        assert len(g.cpu_param_dict) == 2 and res.released_host_bytes(g) == 0
    assert r.released_bytes == sum(r.block_bytes[:2])


def test_a_demoted_block_streams_from_an_exact_rebuilt_copy(caplog):
    r = _residency(3, logger = logging.getLogger("h3test"))
    before = [[t.clone() for t in g.modules[0].parameters()] for g in r.blocks]
    r.fit(sum(r.block_bytes), initial = True)
    assert all(g.cpu_param_dict == {} for g in r.blocks)
    with caplog.at_level(logging.INFO, logger = "h3test"):
        r.fit(r.block_bytes[0])  # a bigger request: blocks 1 and 2 go back to streaming
    assert r.resident_blocks() == 1
    assert "rebuilt a demoted block's host copy from the device" in caplog.text
    for g, ref in zip(r.blocks[1:], before[1:]):
        assert g.where == "cpu" and len(g.cpu_param_dict) == 2
        for p, want in zip(g.modules[0].parameters(), ref):
            assert torch.equal(p.data, want)
            assert (
                p.data.data_ptr() == g.cpu_param_dict[p].data_ptr()
            )  # offload re-pointed at the rebuilt copy
        g.onload_()  # streams again from the rebuilt copy
        for p, want in zip(g.modules[0].parameters(), ref):
            assert torch.equal(p.data, want)
    r.fit(sum(r.block_bytes))  # promoted again: released again
    assert all(g.cpu_param_dict == {} for g in r.blocks)


def test_the_unpin_kill_switch_keeps_the_host_copy(monkeypatch):
    monkeypatch.setenv("UNSLOTH_H3_RESIDENT_UNPIN", "0")
    r = _residency(2)
    r.fit(sum(r.block_bytes), initial = True)
    assert r.resident_blocks() == 2
    assert all(len(g.cpu_param_dict) == 2 for g in r.blocks) and r.released_bytes == 0


def test_a_block_still_reading_its_host_copy_is_not_released():
    from core.inference import video_minimax_h3_residency as res

    g = _HostGroup(0)
    res.make_resident(g)
    g.offload_ = res._noop
    for p, host in g.cpu_param_dict.items():
        p.data = host  # onload left it on the host
    assert res.release_host_copy(g) == 0 and len(g.cpu_param_dict) == 2


def test_a_rebuild_asks_the_conditioner_share_for_ram_when_the_host_is_tight(monkeypatch):
    from core.inference import diffusion_small_host

    asked = []
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (16384, 100))
    r = _residency(2)
    r.host_yield = lambda nbytes: asked.append(nbytes) or nbytes
    r.fit(sum(r.block_bytes), initial = True)
    r.fit(0)
    assert len(asked) == 2 and all(n >= (2048 - 100) << 20 for n in asked)
    assert all(len(g.cpu_param_dict) == 2 for g in r.blocks)


def test_release_all_rebuilds_every_released_block():
    from core.inference import video_minimax_h3_residency as res

    r = _residency(3)
    ref = [[t.clone() for t in g.modules[0].parameters()] for g in r.blocks]
    r.fit(sum(r.block_bytes), initial = True)
    res.release_all(r)
    assert r.resident_bytes() == 0
    for g, want in zip(r.blocks, ref):
        assert all(torch.equal(p.data, w) for p, w in zip(g.modules[0].parameters(), want))


def test_the_conditioner_share_yields_back_to_its_mapping():
    from core.inference import video_minimax_h3_te as te

    class _TE:
        pass

    lin = torch.nn.Linear(32, 32)
    mapped = {p: p.data for p in lin.parameters()}
    group = type("G", (), {})()
    group.cpu_param_dict = dict(mapped)
    holder = _TE()
    holder.model = torch.nn.Module()
    entries = [(group, p, h) for p, h in mapped.items()]
    copied = te._copy_host_tensors_(entries, undo = te._undo_list(holder.model))
    assert copied == sum(h.numel() * h.element_size() for h in mapped.values())
    assert all(group.cpu_param_dict[p] is not mapped[p] for p in mapped)
    freed = te.yield_h3_te_resident(holder, 1)
    assert freed > 0
    weight_or_bias = [p for p in mapped if group.cpu_param_dict[p] is mapped[p]]
    assert len(weight_or_bias) == 1  # last copied first, only as much as asked
    te.yield_h3_te_resident(holder, 1 << 30)
    assert all(group.cpu_param_dict[p] is mapped[p] for p in mapped)
    assert all(p.data.data_ptr() == mapped[p].data_ptr() for p in mapped)


def test_a_refit_that_releases_more_hands_it_to_the_conditioner_once():
    r = _residency(4)
    r.topped_bytes = 0
    calls = []
    r.host_top_up = lambda: calls.append(r.released_bytes)
    r.fit(r.block_bytes[0], initial = True)  # the load's fit (a larger preset): one block
    r.fit(sum(r.block_bytes))  # the request's fit: every block, more released
    r.fit(sum(r.block_bytes))  # same request again: nothing new
    assert calls == [r.block_bytes[0], sum(r.block_bytes)]
    r.fit(r.block_bytes[0])  # demote: less released, no top-up
    assert len(calls) == 2 and r.topped_bytes == r.block_bytes[0]


def test_the_conditioner_top_up_copies_only_what_is_still_mapped(monkeypatch):
    from core.inference import diffusion_small_host
    from core.inference import video_minimax_h3_te as te

    lin = torch.nn.Linear(32, 32)
    group = type("G", (), {})()
    group.modules = [lin]
    group.parameters = []
    group.buffers = []
    group.cpu_param_dict = {p: p.data for p in lin.parameters()}

    class _TE:
        pass

    holder = _TE()
    holder.model = torch.nn.Module()
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (65536, 65536))
    monkeypatch.setattr(diffusion_small_host, "host_commit_mib", lambda: None)
    monkeypatch.setattr(te, "_H3_TE_RESIDENT_MIN_MIB", 0)
    assert te.top_up_h3_te_resident(holder) == 0  # not managed by a load yet
    te.keep_h3_te_resident(holder, groups = [group])
    undo = getattr(holder.model, te._TE_RESIDENT_UNDO_ATTR)
    assert len(undo) == 2
    te.keep_h3_te_resident(holder, groups = [group])  # a top-up: nothing still mapped
    assert len(undo) == 2


def test_a_mostly_released_slab_is_compacted_so_it_can_be_freed(monkeypatch):
    """First-fit packs a later block's small weights into an earlier slab's tail: once the resident blocks drop their
    copies, those few weights alone would keep the whole slab pinned. They move to their own copies (exact)."""
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(res, "_H3_COMPACT_MIN_SLAB", 0)
    monkeypatch.setattr(res, "_pin_host_tensor", lambda src: src.clone())
    slab = torch.arange(4096, dtype = torch.float32)
    big = _HostGroup(0, n = 60)  # 60*60 + 60 = 3660 floats: most of the slab
    small = _HostGroup(1, n = 4)  # 20 floats in the slab's tail
    off = 0
    for g in (big, small):
        for p in g.modules[0].parameters():
            view = slab[off : off + p.numel()].view(p.shape)
            view.copy_(p.data)
            g.cpu_param_dict[p] = view
            p.data = view
            off += p.numel()
    want = [p.data.clone() for p in small.modules[0].parameters()]
    r = _residency(0)
    r.blocks = [big, small]
    r.block_bytes = [res.group_payload_bytes(g) for g in r.blocks]
    r.fit(r.block_bytes[0], initial = True)
    assert r.resident_blocks() == 1 and big.cpu_param_dict == {}
    for p, w in zip(small.modules[0].parameters(), want):
        host = small.cpu_param_dict[p]
        assert host.untyped_storage().data_ptr() != slab.untyped_storage().data_ptr()
        assert torch.equal(host, w)
        assert p.data.data_ptr() == host.data_ptr()  # the streamed weight reads its new copy


def test_a_well_used_slab_is_left_alone(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(res, "_H3_COMPACT_MIN_SLAB", 0)
    slab = torch.zeros(30)  # 20 of 30 floats still used
    g = _HostGroup(0, n = 4)
    off = 0
    for p in g.modules[0].parameters():
        g.cpu_param_dict[p] = slab[off : off + p.numel()].view(p.shape)
        off += p.numel()
    assert res.compact_host_slabs([g]) == 0
    assert all(
        h.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()
        for h in g.cpu_param_dict.values()
    )


def test_a_rebuild_restores_the_load_reserve_not_just_the_floor(monkeypatch):
    """A demote after the conditioner share was topped up must give back enough for the load's own reserve (an L4
    renderer was OOM-killed when only a 2 GiB floor was restored)."""
    from core.inference import diffusion_small_host

    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (53000, 5000))
    monkeypatch.setattr(diffusion_small_host, "encoder_resident_keep_mib", lambda total: 7982)
    asked = []
    r = _residency(1)
    r.host_yield = lambda nbytes: asked.append(nbytes) or nbytes
    r.fit(sum(r.block_bytes), initial = True)
    r.fit(0)
    need_mib = r.block_bytes[0] >> 20
    assert asked == [(need_mib + 7982 - 5000) << 20]  # the 2 GiB floor alone would ask for nothing
