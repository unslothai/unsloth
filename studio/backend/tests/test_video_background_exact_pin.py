# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video whole-module offload pins on a worker thread: the first onload never waits for it (CPU fakes registration)."""

from __future__ import annotations

import threading
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
hooks = pytest.importorskip("accelerate.hooks")

from core.inference import diffusion_memory as dm  # noqa: E402


class _GatedRegistered:
    """``_RegisteredHostBuffer`` without CUDA: ``register`` blocks until the gate opens."""

    gate = threading.Event()
    made: list = []

    def __init__(self, nbytes):
        self._arr = np.zeros(int(nbytes), dtype = np.uint8)
        self.__array_interface__ = self._arr.__array_interface__
        _GatedRegistered.made.append(self)

    def register(self):
        assert _GatedRegistered.gate.wait(30), "the gate never opened"


@pytest.fixture
def gated(monkeypatch):
    _GatedRegistered.gate = threading.Event()
    _GatedRegistered.made = []
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.delenv(dm.EXACT_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.OFFLOAD_KEEP_CPU_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_VIDEO_BACKGROUND_EXACT_PIN", raising = False)
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 1 << 15)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: True)
    monkeypatch.setattr(dm, "_RegisteredHostBuffer", _GatedRegistered)
    yield _GatedRegistered.gate
    _GatedRegistered.gate.set()


def _module():
    torch.manual_seed(0)
    # 16 KiB weights per layer: several 32 KiB chunks
    return torch.nn.Sequential(*(torch.nn.Linear(64, 64) for _ in range(7)))


def _offloaded():
    module = _module()
    hooks.add_hook_to_module(module, hooks.CpuOffload(execution_device = "cpu"))
    pipe = types.SimpleNamespace(components = {"transformer": module}, transformer = module)
    dm.request_fast_pins(pipe)
    dm._wrap_cpu_offload_hooks(pipe, hooks.CpuOffload)
    return pipe, module


def _state(module):
    return module.__dict__[dm._KEEP_ATTR]


def _forward_returns(module, x, seconds):
    done = threading.Event()
    out: list = []

    def run():
        with torch.no_grad():
            out.append(module(x))
        done.set()

    threading.Thread(target = run, daemon = True).start()
    return done.wait(seconds), out


def _pinned(module):
    bufs = [buf for buf, _ in _state(module).get("pin_layout") or ()]
    return {
        name
        for name, p in module.named_parameters()
        if any(buf.data_ptr() <= p.data_ptr() < buf.data_ptr() + buf.nbytes for buf in bufs)
    }


def test_the_first_onload_does_not_wait_for_the_pin(gated):
    pipe, module = _offloaded()
    x = torch.randn(3, 64)
    with torch.no_grad():
        ref = module._old_forward(x)
    returned, out = _forward_returns(module, x, 5)
    # the merge base copies and registers every chunk inside this onload, so it waits on the gate
    assert returned, "the first onload waited for the pin"
    assert torch.equal(out[0], ref)
    worker = _state(module).get("pin_worker")
    assert worker is not None and not _pinned(module)
    gated.set()
    assert worker.join(10)
    names = {name for name, _ in module.named_parameters()}
    assert _pinned(module) == names
    layout = _state(module)["pin_layout"]
    assert {name for _, entries in layout for name, _, _ in entries} == names
    for name, p in module.named_parameters():
        assert _state(module)["host"][name].data_ptr() == p.data_ptr(), name
    with torch.no_grad():
        assert torch.equal(module(x), ref)


@pytest.mark.parametrize("value", ["0", "off"])
def test_the_kill_switch_pins_inside_the_first_onload(gated, monkeypatch, value):
    monkeypatch.setenv(dm.BACKGROUND_EXACT_PIN_ENV, value)
    gated.set()
    pipe, module = _offloaded()
    returned, _out = _forward_returns(module, torch.randn(3, 64), 10)
    assert returned
    assert _state(module).get("pin_worker") is None
    assert _pinned(module) == {name for name, _ in module.named_parameters()}


@pytest.mark.parametrize("change", ["replaced", "written"])
def test_a_weight_changed_while_pinning_keeps_its_change(gated, change):
    pipe, module = _offloaded()
    returned, _out = _forward_returns(module, torch.randn(3, 64), 5)
    assert returned
    worker = _state(module)["pin_worker"]
    weight = module[0].weight
    with torch.no_grad():
        if change == "replaced":
            weight.data = torch.ones_like(weight)
        else:
            weight.data.fill_(1.0)
    gated.set()
    assert worker.join(10)
    assert torch.equal(weight.data, torch.ones_like(weight))
    assert "0.weight" not in _pinned(module)
    assert "1.weight" in _pinned(module)


def test_an_onloaded_weight_is_pinned_through_its_kept_host_copy(gated):
    """Onloaded: only the kept host copy moves to its chunk, so the offload re-points the weight at it."""
    module = _module()
    ref = {name: p.detach().clone() for name, p in module.named_parameters()}
    host = {name: p.data for name, p in module.named_parameters()}
    state = {"host": host, "owner": dict(module.named_parameters()), "buffer_host": {}}
    plan: list = []
    need = dm._pin_host_weights(module, host, None, buffer_host = {}, exact = True, plan = plan)
    assert need and plan and not _GatedRegistered.made, "a plan pins nothing"
    for p in module.parameters():
        # stands in for the onloaded weight: same Parameter object, off the host
        torch.utils.swap_tensors(p, torch.nn.Parameter(torch.empty_like(p, device = "meta")))
    worker = dm._WholeModulePinner(module, state, dm._HookLock(), plan, None)
    gated.set()
    worker.start()
    assert worker.join(10)
    bufs = [buf for buf, _ in state["pin_layout"]]
    for name, p in module.named_parameters():
        assert p.device.type == "meta", name
        kept = host[name]
        assert any(buf.data_ptr() <= kept.data_ptr() < buf.data_ptr() + buf.nbytes for buf in bufs)
        assert torch.equal(kept, ref[name])
        torch.utils.swap_tensors(p, torch.nn.Parameter(kept))  # the offload's re-point
        assert p.data_ptr() == kept.data_ptr() and torch.equal(p.data, ref[name])


def test_stopping_the_pipe_drops_a_pin_in_flight(gated):
    pipe, module = _offloaded()
    before = {name: p.data_ptr() for name, p in module.named_parameters()}
    returned, _out = _forward_returns(module, torch.randn(3, 64), 5)
    assert returned
    worker = _state(module)["pin_worker"]
    dm.stop_background_pins(pipe, timeout = 0)
    gated.set()
    assert worker.join(10)
    assert not _state(module).get("pin_layout")
    assert {name: p.data_ptr() for name, p in module.named_parameters()} == before
