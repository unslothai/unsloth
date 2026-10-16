# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""act_int8 runs torch._int_mm on per-row int8 activations; CPU uses an exact integer reference."""

from __future__ import annotations

import types

import pytest
import torch

import core.inference.diffusion_small_host as sh


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sh.INT8_ACT_ENV, raising = False)
    sh._INT8_ACT_DEVICE_OK.clear()
    yield
    sh._INT8_ACT_DEVICE_OK.clear()


def _exact_int_mm(a, b):
    return (a.to(torch.int64) @ b.to(torch.int64)).to(torch.int32)


def _layer(
    k = 256,
    n = 128,
    bias = True,
    act_int8 = True,
    seed = 0,
):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(k, n, bias = bias)
    model = torch.nn.Sequential(lin)
    sh._INT8_MIN_ELEMENTS, old = 1, sh._INT8_MIN_ELEMENTS
    try:
        sh.quantize_int8_weight_(
            model, compute_dtype = torch.float32, work_device = "cpu", act_int8 = act_int8
        )
    finally:
        sh._INT8_MIN_ELEMENTS = old
    return model[0], lin


def test_family_allowlist_and_kill_switch(monkeypatch):
    assert sh.int8_act_family(types.SimpleNamespace(name = "qwen-image"))
    assert sh.int8_act_family(types.SimpleNamespace(name = "Qwen-Image"))
    assert sh.int8_act_family("qwen-image")
    for other in ("qwen-image-edit", "flux.1", "z-image", "hidream-i1", "qwen-image-2.1", None):
        assert not sh.int8_act_family(types.SimpleNamespace(name = other))
    monkeypatch.setenv(sh.INT8_ACT_ENV, "0")
    assert not sh.int8_act_family(types.SimpleNamespace(name = "qwen-image"))


def test_quantize_sets_the_flag_only_when_asked():
    on, _ = _layer(act_int8 = True)
    off, _ = _layer(act_int8 = False)
    assert isinstance(on, sh.int8_linear_class()) and on.act_int8 is True
    assert off.act_int8 is False


def _fake_cuda(
    monkeypatch,
    cap = (7, 5),
    int_mm = _exact_int_mm,
    hip = None,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx = None: cap)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "_int_mm", int_mm, raising = False)
    real_to = torch.Tensor.to

    def to(self, *a, **k):  # the probe moves operands to cuda; keep them on the CPU
        if a and isinstance(a[0], torch.device) and a[0].type == "cuda":
            return self
        return real_to(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "to", to)


def test_device_gate_turing_only(monkeypatch):
    _fake_cuda(monkeypatch, cap = (7, 5))
    assert sh.int8_act_device_ok("cuda")
    for cap in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (7, 0), (6, 1)):
        sh._INT8_ACT_DEVICE_OK.clear()
        _fake_cuda(monkeypatch, cap = cap)
        assert not sh.int8_act_device_ok("cuda"), cap


def test_device_gate_refuses_rocm_cpu_and_a_wrong_or_failing_kernel(monkeypatch):
    _fake_cuda(monkeypatch, hip = "6.2")
    assert not sh.int8_act_device_ok("cuda")
    sh._INT8_ACT_DEVICE_OK.clear()
    assert not sh.int8_act_device_ok("cpu")

    def wrong(a, b):
        return _exact_int_mm(a, b) + 1

    def boom(a, b):
        raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    for bad in (wrong, boom):
        sh._INT8_ACT_DEVICE_OK.clear()
        _fake_cuda(monkeypatch, int_mm = bad)
        assert not sh.int8_act_device_ok("cuda")


def test_int8_act_math_matches_the_reference_and_the_dense_layer(monkeypatch):
    monkeypatch.setattr(torch, "_int_mm", _exact_int_mm, raising = False)
    layer, lin = _layer(k = 256, n = 128)
    x = torch.randn(4, 9, 256)
    y = layer._forward_int8_act(x)
    assert y.shape == (4, 9, 128) and y.dtype == torch.float32
    # reference: per-row symmetric absmax, round half to even, int32 product, fp32 scales
    x2 = x.reshape(-1, 256)
    xs = x2.abs().amax(1, keepdim = True).clamp_min(1e-12) / 127.0
    xq = torch.round(x2 / xs).clamp(-127, 127).to(torch.int8)
    acc = _exact_int_mm(xq, layer.qweight.t())
    want = (acc.float() * xs) * layer.scale.reshape(1, -1).float() + layer.bias.float()
    assert torch.allclose(y.reshape(-1, 128), want, rtol = 1e-6, atol = 1e-6)
    dense = lin(x)
    rel = (y - dense).norm() / dense.norm()
    assert rel < 0.02, rel


def test_forward_routes_by_dtype_rows_alignment_and_device(monkeypatch):
    calls = []
    layer, _ = _layer(k = 256, n = 128)
    monkeypatch.setattr(sh, "int8_act_device_ok", lambda dev: True)
    monkeypatch.setattr(
        type(layer), "_forward_int8_act", lambda self, x: calls.append(x.shape) or torch.zeros(())
    )
    cuda_x = types.SimpleNamespace(
        dtype = torch.float32, is_cuda = True, device = "cuda", numel = lambda: 17 * 256
    )
    assert layer._int8_act_ok(cuda_x)
    assert not layer._int8_act_ok(
        types.SimpleNamespace(**{**cuda_x.__dict__, "dtype": torch.float16})
    )
    assert not layer._int8_act_ok(
        types.SimpleNamespace(**{**cuda_x.__dict__, "numel": lambda: 16 * 256})
    )
    assert not layer._int8_act_ok(types.SimpleNamespace(**{**cuda_x.__dict__, "is_cuda": False}))
    layer.act_int8 = False
    assert not layer._int8_act_ok(cuda_x)
    layer.act_int8 = True
    layer.out_features = 129
    assert not layer._int8_act_ok(cuda_x)
    layer.out_features = 128
    monkeypatch.setattr(sh, "int8_act_device_ok", lambda dev: False)
    assert not layer._int8_act_ok(cuda_x)
    # CPU float32 takes the dequantised path, bit-identical to a layer without the flag
    x = torch.randn(32, 256)
    off, _ = _layer(k = 256, n = 128, act_int8 = False)
    assert torch.equal(layer(x), off(x))
    assert calls == []
