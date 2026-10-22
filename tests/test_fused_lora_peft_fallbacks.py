# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Configurations the fused LoRA kernels cannot run hand the projection back to PEFT's forward.

The fused Functions compute base(X) + S * lora_B(lora_A(dropout(X))) [+ bias] with frozen base weights in
X's dtype. A PEFT LoRA variant (its own forward), a disabled or merged adapter (base layer only: the
fused backward has no LoRA weights to use), a trainable base weight (its gradient) and an autocast dtype
other than X's (the in-place LoRA accumulation is not autocast) used to run fused anyway, training plain
LoRA, dropping gradients or crashing. Each is checked against PEFT run in float64 (CPU) / float32 (CUDA).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import fast_lora

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fused_lora_dropout_bias import H, PATHS, TARGETS, _Block, _peft_block, _Spy  # noqa: E402

CONFIGS = [(0.0, False), (0.3, False), (0.0, True), (0.3, True)]


def _paths(device):
    # The SwiGLU kernels are Triton.
    return ["o", "qkv"] + (["mlp"] if device == "cuda" else [])


CASES = [("cpu", torch.float64, path) for path in _paths("cpu")]
if has_real_cuda():
    CASES += [("cuda", torch.float32, path) for path in _paths("cuda")]


def _run(fn, model, X0, autocast_dtype = None):
    model.zero_grad(set_to_none = True)
    torch.manual_seed(7)
    X = X0.detach().clone().requires_grad_()
    if autocast_dtype is None:
        out = fn(X * 1)
    else:
        with torch.autocast(X0.device.type, dtype = autocast_dtype):
            out = fn(X * 1)
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator(device = "cpu").manual_seed(11)
    loss = sum((o.double() * torch.randn(o.shape, generator = gen, dtype = torch.float64).to(o.device)).sum() for o in outs)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return [o.detach() for o in outs], X.grad.clone(), grads


def _assert_falls_back_to_peft(monkeypatch, model, block, path, X0, autocast_dtype = None, exact = True):
    spy = _Spy(monkeypatch)
    fused = _run(PATHS[path][0](block), model, X0, autocast_dtype)
    assert spy.calls == [], "a fused LoRA Function ran"
    ref = _run(PATHS[path][1](block), model, X0, autocast_dtype)
    tol = dict(rtol = 0, atol = 0) if exact else {}
    for a, b in zip(fused[0], ref[0]):
        torch.testing.assert_close(a, b, **tol)
    torch.testing.assert_close(fused[1], ref[1], **tol)
    assert fused[2].keys() == ref[2].keys()
    for name in ref[2]:
        torch.testing.assert_close(fused[2][name], ref[2][name], **tol)
    return ref


class _DoubledLoRA:
    # Stand-in variant: PEFT dispatches on `active_adapter in self.lora_variant`, then calls forward.
    @staticmethod
    def forward(module, active_adapter, x, result, **kwargs):
        lora_A, lora_B = module.lora_A[active_adapter], module.lora_B[active_adapter]
        dropout, scaling = module.lora_dropout[active_adapter], module.scaling[active_adapter]
        return result + 2.0 * lora_B(lora_A(dropout(x))) * scaling


@pytest.mark.parametrize("device,dtype,path", CASES)
@pytest.mark.parametrize("dropout,lora_bias", CONFIGS)
def test_lora_variant_runs_peft_forward(monkeypatch, device, dtype, path, dropout, lora_bias):
    model, block = _peft_block(device, dtype, dropout, lora_bias)
    for name in TARGETS:
        block.get_submodule(name).lora_variant["default"] = _DoubledLoRA()
    X0 = torch.randn(2, 5, H, device = device, dtype = dtype)
    _assert_falls_back_to_peft(monkeypatch, model, block, path, X0)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_real_peft_variant_runs_peft_forward(monkeypatch, dropout):
    peft_config = pytest.importorskip("peft.tuners.lora.config")
    if not hasattr(peft_config, "KasaConfig"):
        pytest.skip("PEFT without KaSA")
    from peft import LoraConfig, get_peft_model


    torch.manual_seed(3407)
    cfg = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = dropout,
        target_modules = TARGETS,
        kasa_config = peft_config.KasaConfig(),
    )
    model = get_peft_model(_Block(), cfg).double()
    block = model.base_model.model
    assert "default" in block.o_proj.lora_variant
    with torch.no_grad():
        for name in TARGETS:
            block.get_submodule(name).lora_B["default"].weight.normal_()
    model.train()
    X0 = torch.randn(2, 5, H, dtype = torch.float64)
    for path in _paths("cpu"):
        _assert_falls_back_to_peft(monkeypatch, model, block, path, X0)


@pytest.mark.parametrize("device,dtype,path", CASES)
@pytest.mark.parametrize("state", ["disabled", "merged"])
@pytest.mark.parametrize("dropout,lora_bias", CONFIGS)
def test_disabled_or_merged_adapter_backward_matches_peft(monkeypatch, device, dtype, path, state, dropout, lora_bias):
    if state == "merged" and lora_bias:
        pytest.skip("PEFT refuses to merge a lora_B bias into a base layer without one")
    model, block = _peft_block(device, dtype, dropout, lora_bias)
    if state == "disabled":
        model.base_model.disable_adapter_layers()
    else:
        for name in TARGETS:
            block.get_submodule(name).merge()
    X0 = torch.randn(2, 5, H, device = device, dtype = dtype)
    _assert_falls_back_to_peft(monkeypatch, model, block, path, X0)


@pytest.mark.parametrize("device,dtype,path", CASES)
@pytest.mark.parametrize("dropout,lora_bias", CONFIGS)
def test_trainable_base_weight_gets_its_gradient(monkeypatch, device, dtype, path, dropout, lora_bias):
    model, block = _peft_block(device, dtype, dropout, lora_bias)
    for p in model.parameters():
        p.requires_grad_(True)
    X0 = torch.randn(2, 5, H, device = device, dtype = dtype)
    ref = _assert_falls_back_to_peft(monkeypatch, model, block, path, X0)
    assert any("base_layer.weight" in name for name in ref[2])


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA autocast")
@pytest.mark.parametrize("path", _paths("cuda"))
@pytest.mark.parametrize("dropout,lora_bias", CONFIGS)
@pytest.mark.parametrize(
    "x_dtype,autocast_dtype",
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16), (torch.float32, torch.float16)],
)
def test_autocast_dtype_other_than_input_runs_peft(monkeypatch, path, dropout, lora_bias, x_dtype, autocast_dtype):
    model, block = _peft_block("cuda", torch.bfloat16, dropout, lora_bias)
    X0 = torch.randn(2, 5, H, device = "cuda", dtype = x_dtype)
    _assert_falls_back_to_peft(monkeypatch, model, block, path, X0, autocast_dtype)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if has_real_cuda() else []))
@pytest.mark.parametrize("dropout,lora_bias", CONFIGS)
def test_eligible_configs_still_fuse(monkeypatch, device, dropout, lora_bias):
    # Frozen base, one enabled adapter, no variant: fused, including under autocast in X's own dtype.
    dtype = torch.bfloat16
    model, block = _peft_block(device, dtype, dropout, lora_bias)
    X = torch.randn(2, 5, H, device = device, dtype = dtype)
    spy = _Spy(monkeypatch)
    fast_lora.apply_lora_o(block, X)
    fast_lora.apply_lora_qkv(block, X)
    with torch.autocast(device, dtype = dtype):
        fast_lora.apply_lora_o(block, X)
        fast_lora.apply_lora_qkv(block, X)
    assert [name for name, _ in spy.calls] == ["LoRA_W", "LoRA_QKV"] * 2
    if dropout == 0 and not lora_bias:
        # The default config keeps its pre-existing six-argument call.
        assert len(spy.calls[0][1]) == 6


def test_routing_compiles_without_graph_breaks():
    if not fast_lora.TRACE_LORA_FUNCTIONS:
        pytest.skip("fused Functions run Dynamo-disabled before torch 2.11")
    import torch._dynamo

    model, block = _peft_block("cpu", torch.float32, 0.3, True)
    torch._dynamo.reset()
    compiled = torch.compile(lambda X: fast_lora.apply_lora_o(block, X), backend = "eager", fullgraph = True)
    X = torch.randn(2, 5, H)
    compiled(X)
    model.base_model.disable_adapter_layers()
    torch.testing.assert_close(compiled(X), block.o_proj.base_layer(X))
