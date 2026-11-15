# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 streamed denoiser: the slab compaction after a re-fit pins new copies before any old slab is freed. On
an L4 at a 16 GiB budget the first request's re-fit re-pinned 13.2 GB of whole weights next to a conditioner share
sized to leave only the load's reserve, and the renderer was OOM-killed. The compaction now moves only the inner
tensors that live in the compacted slab, checks the move against the reserve (the conditioner share yields first),
defers slabs that do not fit, and re-sizes the share after the pins exist."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

KEEP_MIB = 7982


class _Q(torch.Tensor):
    """A minimal traceable wrapper subclass (int8 data + scale), like the hosted int8 denoiser's weights."""

    @staticmethod
    def __new__(cls, qdata, scale):
        return torch.Tensor._make_wrapper_subclass(
            cls, qdata.shape, dtype = torch.float32, device = qdata.device
        )

    def __init__(self, qdata, scale):
        self.qdata = qdata
        self.scale = scale

    def __tensor_flatten__(self):
        return ["qdata", "scale"], None

    @staticmethod
    def __tensor_unflatten__(inner, ctx, size, stride):
        return _Q(inner["qdata"], inner["scale"])

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        raise NotImplementedError(func)


MIB = 1 << 20


def _layout():
    """``healthy``: a slab fully used by one weight's 8 MiB int8 data. ``tail``: a 64 MiB slab whose only live bytes
    are that weight's scale (the rest belonged to now-resident blocks)."""
    healthy = torch.zeros(8 * MIB, dtype = torch.int8)
    tail = torch.zeros(16 * MIB, dtype = torch.float32)
    healthy.copy_(torch.randint(-128, 127, (8 * MIB,), dtype = torch.int8))
    tail[:256].copy_(torch.randn(256))
    qd = healthy.view(1024, 8192)
    sc = tail[:256]
    param = _Q(qd, sc)
    host = _Q(qd, sc)
    group = types.SimpleNamespace(cpu_param_dict = {param: host})
    return group, param, host, healthy, tail


class _FakeHost:
    """MemAvailable that pinned copies consume and the conditioner share's yield gives back."""

    def __init__(self, available_mib: float):
        self.available = float(available_mib)
        self.floor = self.available
        self.yields: list = []
        self.events: list = []

    def ram(self):
        return 53000, int(self.available)

    def pin(self, src):
        if isinstance(src, _Q):
            return _Q(self.pin(src.qdata), self.pin(src.scale))
        self.available -= src.numel() * src.element_size() / MIB
        self.floor = min(self.floor, self.available)
        self.events.append("pin")
        return src.clone()

    def host_yield(self, nbytes):
        self.yields.append(nbytes)
        self.events.append("yield")
        self.available += nbytes / MIB
        return nbytes


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    from core.inference import diffusion_small_host
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("UNSLOTH_H3_COMPACT_PRICED", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_RESIDENT_UNPIN", raising = False)
    monkeypatch.setattr(res, "_H3_COMPACT_MIN_SLAB", 0)
    monkeypatch.setattr(diffusion_small_host, "encoder_resident_keep_mib", lambda total: KEEP_MIB)


def _install(monkeypatch, host: _FakeHost):
    from core.inference import diffusion_small_host
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", host.ram)
    monkeypatch.setattr(res, "_pin_host_tensor", host.pin)


def test_only_the_inner_tensor_in_the_slab_moves(monkeypatch):
    """The scale in the dying slab moves; the 8 MiB int8 data in its own healthy slab is not copied again."""
    from core.inference import video_minimax_h3_residency as res

    host = _FakeHost(KEEP_MIB + 4096)
    _install(monkeypatch, host)
    group, param, old, healthy, tail = _layout()
    want_q, want_s = old.qdata.clone(), old.scale.clone()
    assert res.compact_host_slabs([group]) > 0
    new = group.cpu_param_dict[param]
    assert new.qdata.data_ptr() == healthy.data_ptr()  # shared, not re-pinned
    assert new.scale.untyped_storage().data_ptr() != tail.untyped_storage().data_ptr()
    assert torch.equal(new.qdata, want_q) and torch.equal(new.scale, want_s)
    assert param.scale.data_ptr() == new.scale.data_ptr()  # the streamed weight reads its new copy
    assert host.available > KEEP_MIB + 4096 - 1  # 1 KiB pinned, not 8 MiB


def test_the_compaction_never_takes_the_host_under_the_load_reserve(monkeypatch):
    """The L4 16 GiB OOM: the conditioner share left exactly the reserve; the compaction's pins must not eat it."""
    from core.inference import video_minimax_h3_residency as res

    host = _FakeHost(KEEP_MIB + 2)
    _install(monkeypatch, host)
    group, *_ = _layout()
    res.compact_host_slabs([group])
    assert host.floor >= KEEP_MIB


def test_the_conditioner_share_yields_before_the_compaction_pins(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    host = _FakeHost(KEEP_MIB)  # share sized to leave exactly the reserve
    _install(monkeypatch, host)
    group, param, old, _healthy, tail = _layout()
    freed = res.compact_host_slabs([group], host_yield = host.host_yield)
    assert host.events[0] == "yield" and "pin" in host.events
    assert host.yields == [
        1 * MIB
    ]  # the move's cost (1 KiB, priced at one page, rounded up to a MiB)
    assert freed == tail.untyped_storage().nbytes()
    assert host.floor >= KEEP_MIB


def test_a_slab_that_does_not_fit_waits_for_a_later_refit(monkeypatch):
    """Nothing to yield and no room: the slab stays, nothing is pinned, and the next re-fit retries it."""
    from core.inference import video_minimax_h3_residency as res

    host = _FakeHost(KEEP_MIB - 100)
    _install(monkeypatch, host)
    group, param, old, _healthy, tail = _layout()
    report: dict = {}
    assert res.compact_host_slabs([group], host_yield = lambda n: 0, report = report) == 0
    assert report["deferred"] == 1 and "pin" not in host.events
    assert group.cpu_param_dict[param] is old

    r = res.H3Residency.__new__(res.H3Residency)
    r.top, r.blocks, r.block_bytes, r.logger = None, [group], [0], None
    r.host_yield, r.host_top_up = None, None
    r.released_bytes = r.topped_bytes = r.compacted_bytes = 0
    r.compact_pending = True
    host.available = KEEP_MIB + 100
    r.apply(False, 0)  # no new release: the pending slab is compacted anyway
    assert not r.compact_pending and r.compacted_bytes == tail.untyped_storage().nbytes()
    assert (
        group.cpu_param_dict[param].scale.untyped_storage().data_ptr()
        != tail.untyped_storage().data_ptr()
    )


def test_the_share_is_resized_after_a_compaction(monkeypatch):
    """A re-fit whose compaction took RAM from the conditioner share tops the share up after the pins exist."""
    from core.inference import video_minimax_h3_residency as res

    calls = []
    r = res.H3Residency.__new__(res.H3Residency)
    r.top, r.blocks, r.block_bytes, r.logger = None, [], [], None
    r.max_blocks = 0
    r.host_yield = None
    r.host_top_up = lambda: calls.append(1)
    r.released_bytes = r.topped_bytes = 0
    r.compact_pending = False

    def apply(top, n):
        r.compacted_bytes = 64 * MIB
        r.released_bytes = 0

    monkeypatch.setattr(r, "apply", apply)
    r.fit(0)
    assert calls == [1]


def test_the_kill_switch_moves_whole_weights_unpriced(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setenv("UNSLOTH_H3_COMPACT_PRICED", "0")
    host = _FakeHost(KEEP_MIB)
    _install(monkeypatch, host)
    group, param, _old, healthy, _tail = _layout()
    assert res.compact_host_slabs([group], host_yield = host.host_yield) > 0
    assert host.yields == []
    assert group.cpu_param_dict[param].qdata.data_ptr() != healthy.data_ptr()
