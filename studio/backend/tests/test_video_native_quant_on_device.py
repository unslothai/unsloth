# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offloaded native int8 builds: host-resident linears are quantised on the card one at a time."""

from __future__ import annotations

import copy
import inspect
import logging
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq  # noqa: E402

CUDA = torch.cuda.is_available()


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.q = torch.nn.Linear(width, width)
        self.ff = torch.nn.Linear(width, 2 * width, bias = False)
        self.out = torch.nn.Linear(2 * width, width)


class _Dit(torch.nn.Module):
    def __init__(
        self,
        width: int = 512,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(width) for _ in range(depth))
        self.head = torch.nn.Linear(width, 64)


def _dit(seed: int = 0) -> _Dit:
    torch.manual_seed(seed)
    return _Dit().to(torch.bfloat16)


def _keep(mod, name) -> bool:
    return isinstance(mod, torch.nn.Linear) and mod.in_features >= 512


def _quant(dit, logger = None) -> int:
    return nq.apply_native_weight_quant(
        dit, nq.NATIVE_INT8, filter_fn = _keep, act_int8 = True, logger = logger
    )


def _tensors(module) -> dict:
    out = {}
    for name, sub in module.named_modules():
        for key, buf in list(sub._buffers.items()) + list(sub._parameters.items()):
            if buf is not None:
                out[f"{name}.{key}"] = buf.detach()
    return out


def _assert_same_bytes(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        x, y = a[key], b[key]
        assert x.device.type == y.device.type == "cpu", key
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert torch.equal(
            x.contiguous().view(-1).view(torch.uint8), y.contiguous().view(-1).view(torch.uint8)
        ), key


@pytest.mark.parametrize(
    "raw,on", [("", True), ("1", True), ("0", False), ("off", False), ("FALSE", False)]
)
def test_kill_switch(monkeypatch, raw, on):
    monkeypatch.setenv(nq.NATIVE_QUANT_ON_DEVICE_ENV, raw)
    assert nq.native_quant_on_device_enabled() is on


def test_without_a_device_the_build_stays_on_the_host():
    a, b = _dit(), _dit()
    assert _quant(a) == 10
    with nq.quantise_host_linears_on(None):
        assert _quant(b) == 10
    _assert_same_bytes(_tensors(a), _tensors(b))


def test_the_device_scope_ends_with_the_block():
    with nq.quantise_host_linears_on("cpu"):
        assert nq._HOST_QUANT_DEVICE.get() is not None
    assert nq._HOST_QUANT_DEVICE.get() is None


def test_mover_restores_the_source_weight_and_keeps_the_bias_on_the_host(caplog):
    a, b = _dit(), _dit()
    _quant(a)
    log = logging.getLogger("nq-test")
    with caplog.at_level(logging.INFO, logger = "nq-test"), nq.quantise_host_linears_on("cpu"):
        _quant(b, logger = log)
    _assert_same_bytes(_tensors(a), _tensors(b))
    assert "built 10 host linear(s) on cpu one at a time" in caplog.text
    assert b.blocks[0].q.bias is not None and b.blocks[0].q.bias.device.type == "cpu"


def test_a_device_failure_builds_that_layer_and_the_rest_on_the_host(monkeypatch, caplog):
    a, b = _dit(), _dit()
    _quant(a)
    calls = {"n": 0}
    real = nq._HostLinearMover._to_device

    def flaky(self, tensor):
        calls["n"] += 1
        if calls["n"] == 3:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return real(self, tensor)

    monkeypatch.setattr(nq._HostLinearMover, "_to_device", flaky)
    log = logging.getLogger("nq-test")
    with caplog.at_level(logging.INFO, logger = "nq-test"), nq.quantise_host_linears_on("cpu"):
        assert _quant(b, logger = log) == 10
    assert calls["n"] == 3
    _assert_same_bytes(_tensors(a), _tensors(b))
    assert (
        "built 2 host linear(s)" in caplog.text
        and "the rest on the host (OutOfMemoryError" in caplog.text
    )


def test_video_load_routes_offloaded_native_builds_through_the_card():
    import core.inference.video as V

    src = inspect.getsource(V.VideoBackend)
    assert "_host_quant_device(target) if video_offload and native_scheme else None" in src
    assert "with quantise_host_linears_on(host_quant_device):" in src


def test_host_quant_device_gates(monkeypatch):
    import core.inference.video as V

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(nq.NATIVE_QUANT_ON_DEVICE_ENV, raising = False)
    assert V._host_quant_device(
        types.SimpleNamespace(device = "cuda", torch_device = "cuda:1")
    ) == torch.device("cuda:1")
    assert V._host_quant_device(types.SimpleNamespace(device = "mps")) is None
    assert V._host_quant_device(types.SimpleNamespace(device = "cpu")) is None
    monkeypatch.setenv(nq.NATIVE_QUANT_ON_DEVICE_ENV, "0")
    assert V._host_quant_device(types.SimpleNamespace(device = "cuda")) is None
    monkeypatch.delenv(nq.NATIVE_QUANT_ON_DEVICE_ENV)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert V._host_quant_device(types.SimpleNamespace(device = "cuda")) is None


@pytest.mark.skipif(not CUDA, reason = "needs a CUDA device")
def test_card_build_of_host_linears_matches_a_fully_staged_build_bit_for_bit():
    staged, host = _dit(), _dit()
    staged.to("cuda")
    _quant(staged)
    staged.to("cpu")
    with nq.quantise_host_linears_on(torch.device("cuda")):
        _quant(host)
    _assert_same_bytes(_tensors(staged), _tensors(host))
    assert all(t.device.type == "cpu" for t in _tensors(host).values())


@pytest.mark.skipif(not CUDA, reason = "needs a CUDA device")
def test_card_build_runs_the_same_forward_as_a_staged_build():
    staged, host = _dit(), _dit()
    staged.to("cuda")
    _quant(staged)
    with nq.quantise_host_linears_on(torch.device("cuda")):
        _quant(host)
    host.to("cuda")
    x = torch.randn(64, 512, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(
            staged.blocks[1].out(staged.blocks[1].ff(x)), host.blocks[1].out(host.blocks[1].ff(x))
        )


@pytest.mark.skipif(not CUDA, reason = "needs a CUDA device")
def test_partially_staged_and_card_built_layers_agree():
    # Base staged what fit and quantised the rest on the host; the staged part must not change.
    base, head = _dit(), _dit()
    base.blocks[0].to("cuda")
    head.blocks[0].to("cuda")
    _quant(base)
    with nq.quantise_host_linears_on(torch.device("cuda")):
        _quant(head)
    base.to("cpu")
    head.to("cpu")
    a, b = _tensors(base.blocks[0]), _tensors(head.blocks[0])
    _assert_same_bytes(a, b)
    ref = copy.deepcopy(_dit()).to("cuda")
    _quant(ref)
    ref.to("cpu")
    _assert_same_bytes(_tensors(ref), _tensors(head))
