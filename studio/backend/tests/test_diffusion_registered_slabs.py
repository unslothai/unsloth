# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Up-front pinned slabs come from registered host memory; ``UNSLOTH_DIFFUSION_REGISTERED_SLABS=0`` restores torch's."""

import numpy as np
import pytest
import torch

from core.inference import diffusion_memory as dm
from core.inference import diffusion_pinned_arena as arena_mod


class _FakeHolder:
    """CPU-only stand-in for ``_RegisteredHostBuffer``."""

    made: list = []

    def __init__(self, nbytes):
        self._arr = np.full(int(nbytes), 7, dtype = np.uint8)  # dirty, so the zero fill is observable
        self.__array_interface__ = self._arr.__array_interface__
        self.registered_after_zero = None
        _FakeHolder.made.append(self)

    def register(self):
        self.registered_after_zero = bool((self._arr == 0).all())


@pytest.fixture
def fake_registered(monkeypatch):
    _FakeHolder.made = []
    monkeypatch.delenv(dm.REGISTERED_SLAB_ENV, raising = False)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: True)
    monkeypatch.setattr(dm, "_RegisteredHostBuffer", _FakeHolder)
    return _FakeHolder.made


def test_registered_pinned_empty_zero_fills_before_registering(fake_registered):
    buf = dm.registered_pinned_empty(4096)
    assert buf is not None and buf.dtype == torch.uint8 and buf.numel() == 4096
    assert len(fake_registered) == 1 and fake_registered[0].registered_after_zero is True
    assert int(buf.sum()) == 0


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch_restores_the_pinned_allocator(fake_registered, monkeypatch, value):
    monkeypatch.setenv(dm.REGISTERED_SLAB_ENV, value)
    assert dm.registered_pinned_empty(4096) is None
    assert fake_registered == []


def test_unsupported_host_or_refused_registration_falls_back(fake_registered, monkeypatch):
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: False)
    assert dm.registered_pinned_empty(4096) is None
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: True)

    def refuse(self):
        raise RuntimeError("cudaHostRegister failed")

    monkeypatch.setattr(_FakeHolder, "register", refuse)
    assert dm.registered_pinned_empty(4096) is None
    assert dm.registered_pinned_empty(0) is None


def test_arena_slabs_come_from_registered_memory(fake_registered):
    weights = [
        torch.randn(64, 32),
        torch.arange(100, dtype = torch.int8),
        torch.randn(7, dtype = torch.float16),
    ]
    arena = arena_mod.PinnedArena(slab_bytes = 1 << 14)
    copies = [arena.pin(w) for w in weights]
    assert len(fake_registered) == len(arena._slabs) >= 1
    slab_ptrs = {s.untyped_storage().data_ptr() for s, _ in arena._slabs}
    for w, c in zip(weights, copies):
        assert torch.equal(w, c)
        assert c.untyped_storage().data_ptr() in slab_ptrs


def test_arena_uses_torch_pinned_allocator_when_registration_is_off(fake_registered, monkeypatch):
    monkeypatch.setenv(dm.REGISTERED_SLAB_ENV, "0")
    seen = []

    def fake_empty(
        n,
        dtype = None,
        pin_memory = False,
    ):
        seen.append(pin_memory)
        return torch.zeros(n, dtype = dtype)

    monkeypatch.setattr(torch, "empty", fake_empty)
    arena_mod._alloc_slab(1 << 12)
    assert seen == [True] and fake_registered == []


def test_h3_conditioner_pin_in_place_uses_registered_arenas(fake_registered):
    from core.inference.video_minimax_h3_te import pin_module_in_place

    m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 8))
    m.register_buffer("q", torch.arange(64, dtype = torch.int8).view(8, 8))
    expect = {k: v.detach().clone() for k, v in m.state_dict().items()}
    pinned = pin_module_in_place(m, arena_bytes = 1 << 14)
    assert pinned == sum(t.numel() * t.element_size() for t in expect.values())
    assert len(fake_registered) == 1
    for k, v in m.state_dict().items():
        assert torch.equal(v, expect[k]), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_real_registered_slab_is_pinned_and_uploads_exactly(monkeypatch):
    monkeypatch.delenv(dm.REGISTERED_SLAB_ENV, raising = False)
    if not dm._fast_pin_supported():
        pytest.skip("registered pinning unsupported on this host")
    buf = dm.registered_pinned_empty(1 << 20)
    assert buf is not None and buf.is_pinned()
    src = torch.randint(0, 255, (1 << 20,), dtype = torch.uint8)
    buf.copy_(src)
    dev = buf.to("cuda", non_blocking = True)
    torch.cuda.synchronize()
    assert torch.equal(dev.cpu(), src)
