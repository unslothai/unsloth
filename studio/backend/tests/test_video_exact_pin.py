# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video whole-module offload pins each chunk page-granular, not a power of two (CPU fakes registration)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from core.inference import diffusion_memory as dm  # noqa: E402


class _FakeRegistered:
    """``_RegisteredHostBuffer`` without CUDA: a numpy buffer, ``register`` records or refuses."""

    made: list = []
    refuse_at: int = -1

    def __init__(self, nbytes):
        self.size = int(nbytes)
        self._arr = np.zeros(self.size, dtype = np.uint8)
        self.__array_interface__ = self._arr.__array_interface__
        self.registered = False
        _FakeRegistered.made.append(self)

    def register(self):
        if len(_FakeRegistered.made) - 1 == _FakeRegistered.refuse_at:
            raise RuntimeError("cudaHostRegister failed (2)")
        self.registered = True


@pytest.fixture
def fake(monkeypatch):
    real_empty = torch.empty
    stock: list = []

    def empty(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            stock.append(args[0])
        return real_empty(*args, **kwargs)

    _FakeRegistered.made = []
    _FakeRegistered.refuse_at = -1
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.delenv(dm.EXACT_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 1 << 15)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: True)
    monkeypatch.setattr(dm, "_RegisteredHostBuffer", _FakeRegistered)
    return stock


def _module():
    torch.manual_seed(0)
    # ~13 KB weights, 200 B biases: two layers per 32 KiB chunk, and a tail chunk far from a power of two
    return torch.nn.Sequential(*(torch.nn.Linear(64, 50) for _ in range(7)))


def _host(module):
    return {name: p.data for name, p in module.named_parameters()}


def _payload(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _check_values(module, ref):
    for name, p in module.named_parameters():
        assert torch.equal(p.data, ref[name]), name


def test_exact_chunks_are_sized_to_their_bytes(fake):
    module = _module()
    ref = _payload(module)
    host = _host(module)
    layout: list = []
    need = dm._pin_host_weights(module, host, None, buffer_host = {}, layout = layout, exact = True)
    assert not fake, "the exact path must not touch the pinned allocator"
    assert len(_FakeRegistered.made) > 1
    sizes = [h.size for h in _FakeRegistered.made]
    assert all(h.registered for h in _FakeRegistered.made)
    assert need == sum(sizes)
    assert all(size % dm._EXACT_PIN_PAGE == 0 for size in sizes)
    payload = sum(p.nbytes for p in module.parameters())
    assert payload <= need < payload + len(sizes) * dm._EXACT_PIN_PAGE
    twin = _module()
    stock_need = dm._pin_host_weights(
        twin, _host(twin), None, buffer_host = {}, layout = [], exact = False
    )
    assert need < stock_need
    _check_values(module, ref)
    bufs = [buf for buf, _ in layout]
    for name, p in module.named_parameters():
        assert host[name].data_ptr() == p.data_ptr()
        assert any(
            buf.data_ptr() <= p.data_ptr() < buf.data_ptr() + buf.nbytes for buf in bufs
        ), name
    assert sum(len(entries) for _, entries in layout) == len(list(module.parameters()))


def test_same_slots_as_the_stock_pin(fake):
    """Device uploads copy ``buf[:used]`` per chunk: the exact path keeps the stock slots, so they are unchanged."""
    a, b = _module(), _module()
    la: list = []
    lb: list = []
    dm._pin_host_weights(a, _host(a), None, buffer_host = {}, layout = la, exact = True)
    dm._pin_host_weights(b, _host(b), None, buffer_host = {}, layout = lb, exact = False)
    assert [[(n, s) for n, s, _ in e] for _, e in la] == [[(n, s) for n, s, _ in e] for _, e in lb]
    assert fake, "the stock path pins through the allocator"
    assert all(n == 1 << (n - 1).bit_length() for n in fake)


@pytest.mark.parametrize("value", ["0", "off"])
def test_the_kill_switch_restores_the_power_of_two_chunks(fake, monkeypatch, value):
    monkeypatch.setenv(dm.EXACT_PIN_ENV, value)
    module = _module()
    dm._pin_host_weights(module, _host(module), None, buffer_host = {}, layout = [], exact = True)
    assert not _FakeRegistered.made
    assert fake


def test_unsupported_registration_keeps_the_stock_pin(fake, monkeypatch):
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: False)
    module = _module()
    dm._pin_host_weights(module, _host(module), None, buffer_host = {}, layout = [], exact = True)
    assert not _FakeRegistered.made
    assert fake


def test_a_refused_first_chunk_falls_back_to_the_stock_pin(fake):
    _FakeRegistered.refuse_at = 0
    module = _module()
    ref = _payload(module)
    layout: list = []
    need = dm._pin_host_weights(
        module, _host(module), None, buffer_host = {}, layout = layout, exact = True
    )
    assert need == sum(fake) and fake
    _check_values(module, ref)
    assert sum(len(e) for _, e in layout) == len(list(module.parameters()))


def test_a_refused_later_chunk_keeps_the_pinned_ones(fake):
    _FakeRegistered.refuse_at = 1
    module = _module()
    ref = _payload(module)
    host = _host(module)
    before = {name: p.data_ptr() for name, p in module.named_parameters()}
    layout: list = []
    need = dm._pin_host_weights(module, host, None, buffer_host = {}, layout = layout, exact = True)
    assert not fake
    assert len(layout) == 1 and need == _FakeRegistered.made[0].size
    moved = {name for name, _, _ in layout[0][1]}
    assert moved
    for name, p in module.named_parameters():
        assert (p.data_ptr() != before[name]) == (name in moved), name
    _check_values(module, ref)


def test_only_a_video_pipe_asks_for_exact_chunks(monkeypatch):
    hooks = pytest.importorskip("accelerate.hooks")
    monkeypatch.delenv(dm.OFFLOAD_KEEP_CPU_ENV, raising = False)

    def pipe(video):
        module = _module()
        hook = hooks.CpuOffload(execution_device = "cpu")
        hooks.add_hook_to_module(module, hook)
        p = types.SimpleNamespace(components = {"transformer": module}, transformer = module)
        if video:
            dm.request_fast_pins(p)
        dm._wrap_cpu_offload_hooks(p, hooks.CpuOffload)
        return module.__dict__[dm._KEEP_ATTR].get("exact_pin")

    assert pipe(True) is True
    assert pipe(False) is False


def _gguf_weight():
    utils = pytest.importorskip("diffusers.quantizers.gguf.utils")
    gguf = pytest.importorskip("gguf")
    q4k = gguf.GGMLQuantizationType.Q4_K
    block, type_size = gguf.GGML_QUANT_SIZES[q4k]
    rows, cols = 8, 2 * block
    data = torch.randint(0, 255, (rows, cols // block * type_size), dtype = torch.uint8)
    return utils.GGUFParameter(data, quant_type = q4k)


def test_a_gguf_weight_pins_registered_and_keeps_its_quant_type(monkeypatch):
    src = _gguf_weight()
    made = []

    def fake_copy(t):
        assert type(t) is torch.Tensor
        made.append(t.nbytes)
        return t.clone()

    monkeypatch.setattr(dm, "registered_host_copy", fake_copy)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: True)
    monkeypatch.delenv(dm.EXACT_PIN_ENV, raising = False)
    out = dm._registered_pin(src)
    assert made == [src.nbytes]
    assert type(out) is type(src) and out.quant_type == src.quant_type
    assert out.quant_shape == src.quant_shape and not out.requires_grad
    assert torch.equal(out.as_tensor(), src.as_tensor())


def test_the_kill_switch_pins_a_gguf_weight_through_the_allocator(monkeypatch):
    src = _gguf_weight()
    monkeypatch.setattr(dm, "registered_host_copy", lambda t: pytest.fail("registered"))
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: True)
    monkeypatch.setenv(dm.EXACT_PIN_ENV, "0")
    calls = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: calls.append(1) or self.clone())
    out = dm._registered_pin(src)
    assert calls and type(out) is type(src)


def test_a_plain_tensor_still_takes_the_registered_copy(monkeypatch):
    monkeypatch.setattr(dm, "registered_host_copy", lambda t: ("registered", t))
    src = torch.randn(4)
    assert dm._registered_pin(src)[0] == "registered"
