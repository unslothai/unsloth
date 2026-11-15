# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: MiniMax-H3 conditioner device share takes only leftover budget and yields before any block demotes."""

from __future__ import annotations

import logging
import random

import pytest

torch = pytest.importorskip("torch")


class _Stream:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


class _TeGroup:
    """A diffusers-style streamed leaf group: onload gives the weights new storage, offload re-points at the host copy."""

    def __init__(
        self,
        leaf,
        events = None,
        tag = None,
    ):
        self.modules = [leaf]
        self.parameters = []
        self.buffers = []
        self.stream = _Stream()
        self.onload_leader = leaf
        self.cpu_param_dict = {p: p.data for p in leaf.parameters()}
        self.where = "cpu"
        self.on = 0
        self.off = 0
        self.events = events
        self.tag = tag

    def onload_(self):
        self.on += 1
        for p, host in self.cpu_param_dict.items():
            p.data = host.clone()
        self.where = "cuda"
        if self.events is not None:
            self.events.append(("on", self.tag))

    def offload_(self):
        self.off += 1
        for p in self.modules[0].parameters():
            p.data = self.cpu_param_dict[p]
        self.where = "cpu"
        if self.events is not None:
            self.events.append(("off", self.tag))


class _Block(_TeGroup):
    pass


class _Model(torch.nn.Module):
    def __init__(
        self,
        n_layers = 3,
        width = 16,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.visual = torch.nn.Sequential(torch.nn.Linear(width, width))
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            layer = torch.nn.Module()
            layer.q = torch.nn.Linear(width, width, bias = False)
            layer.mlp = torch.nn.Linear(width, 4 * width, bias = False)
            self.language_model.layers.append(layer)

    def forward(self, x):
        for layer in self.language_model.layers:
            x = x + layer.mlp(layer.q(x)).view(*x.shape[:-1], 4, -1).sum(-2)
        return x


class _TE(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = _Model(**kw)


def _share(
    monkeypatch,
    te_module,
    events = None,
    logger = None,
):
    import core.inference.diffusion_memory as dm
    from core.inference.video_minimax_h3_te import H3TeDeviceShare

    groups = [
        _TeGroup(m, events, name)
        for name, m in te_module.model.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda target: list(groups))
    return H3TeDeviceShare(te_module, "cuda", logger = logger), groups


def _residency(
    block_bytes,
    events = None,
    top_bytes = 0,
):
    from core.inference.video_minimax_h3_residency import H3Residency, group_payload_bytes

    r = H3Residency.__new__(H3Residency)

    def block(nbytes, tag):
        return _Block(torch.nn.Linear(1, nbytes // 4, bias = False), events, tag)

    r.top = block(top_bytes, "top") if top_bytes else None
    r.blocks = [block(b, f"b{i}") for i, b in enumerate(block_bytes)]
    r.device = "cuda"
    r.logger = None
    r.top_bytes = group_payload_bytes(r.top) if r.top is not None else 0
    r.block_bytes = [group_payload_bytes(g) for g in r.blocks]
    r.max_blocks = 0
    r.te_share = None
    return r


@pytest.fixture(autouse = True)
def _cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("UNSLOTH_H3_TE_DEVICE_RESIDENT", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_RESIDENT_UNPIN", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_HOST_PIN_TRIM", raising = False)


def test_share_reads_language_layers_in_order_and_leaves_the_vision_tower_streamed(monkeypatch):
    te_module = _TE()
    share, groups = _share(monkeypatch, te_module)
    assert [g.tag for g in share.groups] == [
        f"language_model.layers.{i}.{leaf}" for i in range(3) for leaf in ("q", "mlp")
    ]
    assert share.layer_of == [0, 0, 1, 1, 2, 2] and share.layers == 3
    assert all(g.tag != "visual.0" for g in share.groups)


def test_budget_fits_every_block_then_a_conditioner_prefix(monkeypatch):
    te_module = _TE()
    share, _ = _share(monkeypatch, te_module)
    r = _residency([4096] * 4)
    r.te_share = share
    room = sum(share.planned[:3])
    r.fit(sum(r.block_bytes) + room, initial = True)
    assert r.resident_blocks() == 4
    assert share.resident_count() == 3
    assert share.resident_bytes() <= room
    assert r.device_bytes() == r.resident_bytes() + share.resident_bytes()
    assert [g.where for g in share.groups] == ["cuda"] * 3 + ["cpu"] * 3


def test_nothing_for_the_conditioner_unless_every_block_is_resident(monkeypatch):
    """12 / 16 GB: the denoiser keeps a block prefix, and the leftover (smaller than a block) is not handed out."""
    te_module = _TE()
    share, _ = _share(monkeypatch, te_module)
    r = _residency([4096] * 4)
    r.te_share = share
    r.fit(3 * r.block_bytes[0] + r.block_bytes[0] - 1, initial = True)
    assert r.resident_blocks() == 3
    assert share.plan(r.block_bytes[0] - 1) > 0  # it would fit the leftover
    assert share.resident_count() == 0 and share.resident_bytes() == 0
    # a streamed top-level group, or one block short: nothing either
    r2 = _residency([4096] * 2, top_bytes = 8192)
    assert r2.te_budget(10**9, False, len(r2.blocks)) == 0
    assert r2.te_budget(10**9, True, len(r2.blocks) - 1) == 0
    assert r2.te_budget(10**9, True, len(r2.blocks)) == 10**9 - r2.top_bytes - sum(r2.block_bytes)


def test_a_bigger_request_takes_the_share_back_before_demoting_a_block(monkeypatch):
    events: list = []
    te_module = _TE()
    share, groups = _share(monkeypatch, te_module, events)
    r = _residency([4096] * 4, events)
    r.te_share = share
    full = sum(r.block_bytes)
    r.fit(full + sum(share.planned), initial = True)
    assert share.resident_count() == len(share.groups) and r.resident_blocks() == 4
    events.clear()
    # exactly the denoiser's bytes: the share goes, every block stays
    r.fit(full)
    assert share.resident_count() == 0 and r.resident_blocks() == 4
    assert all(kind == "off" and not tag.startswith("b") for kind, tag in events), events
    events.clear()
    r.fit(full + sum(share.planned[:2]))
    assert share.resident_count() == 2
    events.clear()
    # one byte short: the share is given back first, then the last block
    r.fit(full - 1)
    offs = [tag for kind, tag in events if kind == "off"]
    assert offs[-1] == "b3" and all(not t.startswith("b") for t in offs[:-1]), offs
    assert share.resident_count() == 0 and r.resident_blocks() == 3


def test_resident_groups_keep_the_side_stream_sync_and_skip_the_moves(monkeypatch):
    te_module = _TE()
    share, groups = _share(monkeypatch, te_module)
    share.grow(2)
    g = share.groups[0]
    assert g.where == "cuda" and g.on == 1
    syncs = g.stream.syncs
    g.onload_()  # the group before it prefetching it: no move, but the stream is synced
    assert g.on == 1 and g.stream.syncs == syncs + 1
    g.offload_()
    assert g.off == 0 and g.where == "cuda"
    share.shrink(0)
    assert "onload_" not in g.__dict__ and "offload_" not in g.__dict__
    assert g.where == "cpu" and g.off == 1
    for p, host in g.cpu_param_dict.items():
        assert p.data.data_ptr() == host.data_ptr()  # back on the host copy it never dropped


def test_output_is_bit_identical_resident_or_streamed(monkeypatch):
    te_module = _TE()
    x = torch.randn(2, 16)
    with torch.no_grad():
        expect = te_module.model(x)
    share, _ = _share(monkeypatch, te_module)
    for n in (6, 3, 0, 5):
        share.shrink(n)
        share.grow(n)
        with torch.no_grad():
            assert torch.equal(te_module.model(x), expect)


def test_the_budget_counts_the_share_as_demotable(monkeypatch):
    """The share lowers free VRAM, so the budget must add it back or every request would shrink it."""
    import core.inference.diffusion_memory as dm
    from core.inference import video as vid

    te_module = _TE()
    share, _ = _share(monkeypatch, te_module)
    r = _residency([4096] * 4)
    r.fit(10**9, initial = True)
    base_free = 30 * 10**9
    monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: 0)
    monkeypatch.setattr(vid, "_h3_signed_free_device_bytes", lambda device: base_free)
    kw = dict(width = 960, height = 544, frames = 124, te_streamed = True, te_scheme = "int8")
    without = vid._h3_residency_budget_bytes(r, "cuda", **kw)
    r.te_share = share
    share.grow(4)
    held = share.resident_bytes()
    assert held > 0
    monkeypatch.setattr(vid, "_h3_signed_free_device_bytes", lambda device: base_free - held)
    assert vid._h3_residency_budget_bytes(r, "cuda", **kw) == without


def test_randomized_requests_keep_the_denoiser_and_the_device_peak_of_the_base(monkeypatch):
    """Simulated card: denoiser residency matches base at every request and never overcommits the card."""
    rng = random.Random(7)
    for trial in range(40):
        blocks = [rng.choice([4096, 8192]) for _ in range(rng.randint(1, 5))]
        te_module = _TE(n_layers = rng.randint(1, 4))
        share, _ = _share(monkeypatch, te_module)
        head, base = _residency(blocks), _residency(blocks)
        head.te_share = share
        card = sum(head.block_bytes) + rng.randint(0, sum(share.planned) * 2)
        for k in range(8):
            need = rng.randint(0, card // 3)
            outs = []
            for r in (base, head):
                used = r.device_bytes()
                free = card - used
                budget = free + r.device_bytes() - need
                r.fit(max(0, budget), initial = k == 0)
                peak = r.device_bytes() + need
                assert peak <= card, (trial, k, peak, card)
                outs.append(r.resident_blocks())
            assert outs[0] == outs[1], (trial, k)


def test_kill_switch_and_modes(monkeypatch):
    from core.inference.video_minimax_h3_te import h3_te_device_share

    te_module = _TE()
    _share(monkeypatch, te_module)
    assert h3_te_device_share(te_module, "cuda", "stream") is not None
    assert h3_te_device_share(te_module, "cuda", "stream_lazy") is not None
    assert h3_te_device_share(te_module, "cuda", "sync") is None  # integrated GPU: one memory
    assert h3_te_device_share(te_module, "cuda", None) is None
    monkeypatch.setenv("UNSLOTH_H3_TE_DEVICE_RESIDENT", "0")
    assert h3_te_device_share(te_module, "cuda", "stream") is None


def test_engagement_log_line(monkeypatch, caplog):
    logger = logging.getLogger("h3_te_device_test")
    te_module = _TE()
    share, _ = _share(monkeypatch, te_module, logger = logger)
    r = _residency([4096] * 2)
    r.te_share = share
    with caplog.at_level(logging.INFO, logger = "h3_te_device_test"):
        r.fit(sum(r.block_bytes) + sum(share.planned[:4]), initial = True)
    lines = [m for m in caplog.messages if m.startswith("video.h3_te_resident_device:")]
    assert len(lines) == 1 and lines[0].startswith(
        "video.h3_te_resident_device: 2 of 3 TE layers resident on cuda"
    ), lines


def test_a_failed_onload_leaves_the_conditioner_streamed(monkeypatch):
    te_module = _TE()
    share, groups = _share(monkeypatch, te_module)
    r = _residency([4096])
    r.te_share = share

    def boom():
        raise RuntimeError("CUDA out of memory")

    share.groups[2].onload_ = boom
    r.fit(10**9, initial = True)
    assert r.resident_blocks() == 1
    assert share.resident_count() == 0 and all(g.where == "cpu" for g in share.groups)


def test_release_all_returns_the_share(monkeypatch):
    from core.inference.video_minimax_h3_residency import release_all

    te_module = _TE()
    share, _ = _share(monkeypatch, te_module)
    r = _residency([4096])
    r.te_share = share
    r.fit(10**9, initial = True)
    assert share.resident_count() == len(share.groups)
    release_all(r)
    assert share.resident_count() == 0 and all(g.where == "cpu" for g in share.groups)
