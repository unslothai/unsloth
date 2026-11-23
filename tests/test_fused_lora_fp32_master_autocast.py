# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A float32 model under bfloat16 autocast runs the fused LoRA kernels, like a bfloat16 model.

A model loaded in float32 trains under bfloat16 autocast when bf16 = True, or with neither flag on a
bf16 GPU (rl.py picks it). Its float32 activations used to send every projection to PEFT's forward,
which adds the LoRA term to a base output autocast already rounded to bfloat16: a LoRA update under half
a bfloat16 ulp of that output is lost. Early in training lora_B is that small (lora_B starts at zero, and
Adam moves it by about lr per step), so the run learned far slower than both float32 and a bfloat16 load
(SFT on Qwen3-0.6B, lr 2e-5: loss 6.273 vs 6.245 after the first step). The fused kernels round base +
LoRA once, so the update survives. These tests pin that the fused path runs, that it is bitwise the
bfloat16 model's, and that a tiny LoRA update reaches the output.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import fast_lora

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fused_lora_dropout_bias import H, PATHS, TARGETS, _peft_block, _Spy  # noqa: E402

FUSED_NAMES = {"o": "LoRA_W", "qkv": "LoRA_QKV", "mlp": "LoRA_MLP"}
CASES = [("cpu", "o"), ("cpu", "qkv")]
if has_real_cuda():
    CASES += [("cuda", "o"), ("cuda", "qkv"), ("cuda", "mlp")]


def _fp32_block(device, b_value = None):
    model, block = _peft_block(device, torch.float32, 0.0)
    # What patch_peft_model marks on the attention / MLP modules of a float32 model.
    block.__dict__[fast_lora._BF16_AUTOCAST_ONLY] = True
    if b_value is not None:
        gen = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name in TARGETS:
                B = block.get_submodule(name).lora_B["default"].weight
                B.copy_(torch.sign(torch.randn(B.shape, generator = gen)) * b_value)
    model.train()
    return model, block


def _bf16_base_copy(model):
    # What a bfloat16 load trains: bfloat16 frozen bases, float32 LoRA weights.
    model = copy.deepcopy(model)
    block = model.base_model.model
    block.__dict__.pop(fast_lora._BF16_AUTOCAST_ONLY, None)
    for name in TARGETS:
        base = block.get_submodule(name).base_layer
        base.weight.data = base.weight.data.to(torch.bfloat16)
    return model, block


def _forward_backward(fn, model, X0):
    model.zero_grad(set_to_none = True)
    X = X0.detach().clone().requires_grad_()
    with torch.autocast(X0.device.type, dtype = torch.bfloat16):
        out = fn(X)
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator().manual_seed(11)
    loss = sum((o.float() * torch.randn(o.shape, generator = gen).to(o.device)).sum() for o in outs)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return [o.detach() for o in outs], X.grad.clone(), grads


@pytest.mark.parametrize("device,path", CASES)
def test_fp32_model_under_bf16_autocast_matches_bf16_model(monkeypatch, device, path):
    model, block = _fp32_block(device)
    bf_model, bf_block = _bf16_base_copy(model)
    X0 = torch.randn(2, 5, H, device = device)
    spy = _Spy(monkeypatch)
    got = _forward_backward(PATHS[path][0](block), model, X0)
    assert [name for name, _ in spy.calls] == [
        FUSED_NAMES[path]
    ], "the fused LoRA Function did not run"
    ref = _forward_backward(PATHS[path][0](bf_block), bf_model, X0.to(torch.bfloat16))
    for a, b in zip(got[0], ref[0]):
        assert a.dtype == torch.bfloat16
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)
    assert got[1].dtype == torch.float32
    torch.testing.assert_close(got[1].to(torch.bfloat16), ref[1], rtol = 0, atol = 0)
    assert got[2].keys() == ref[2].keys() and len(got[2]) > 0
    for name in ref[2]:
        assert got[2][name].dtype == torch.float32
        torch.testing.assert_close(got[2][name], ref[2][name], rtol = 0, atol = 0)
    # The model itself is untouched: float32 bases stay float32 and frozen.
    for name in TARGETS:
        W = block.get_submodule(name).base_layer.weight
        assert W.dtype == torch.float32 and not W.requires_grad


@pytest.mark.parametrize("device,path", CASES)
def test_small_lora_update_reaches_output(device, path):
    # lora_B of 2e-5 is one Adam step at lr 2e-5 from zero. Rounding the base output to bfloat16 before
    # adding the LoRA term keeps about 5% of it; rounding base + LoRA once keeps all of it on average.
    model, block = _fp32_block(device, b_value = 2e-5)
    gen = torch.Generator().manual_seed(1)
    X = torch.randn(64, 8, H, generator = gen).to(device)
    fn = PATHS[path][0](block)
    with torch.no_grad(), torch.autocast(device, dtype = torch.bfloat16):
        on = fn(X)
        with model.disable_adapter():
            off = PATHS[path][1](block)(X)
    on = on if isinstance(on, tuple) else (on,)
    off = off if isinstance(off, tuple) else (off,)
    names = {"o": ["o_proj"], "qkv": ["q_proj", "k_proj", "v_proj"], "mlp": None}[path]
    if names is None:
        # MLP: the LoRA delta is not linear in lora_B through SwiGLU; compare against float64.
        with torch.no_grad():
            m64 = copy.deepcopy(model).double()
            b64 = m64.base_model.model
            exact = PATHS[path][1](b64)(X.double())
            with m64.disable_adapter():
                exact = (exact - PATHS[path][1](b64)(X.double()),)
    else:
        exact = []
        for name in names:
            proj = block.get_submodule(name)
            A = proj.lora_A["default"].weight.double()
            B = proj.lora_B["default"].weight.double()
            exact.append(X.double() @ A.t() @ B.t() * proj.scaling["default"])
    for o, f, e in zip(on, off, exact):
        delta = o.double() - f.double()
        kept = float((delta * e).sum() / (e * e).sum())
        assert 0.9 < kept < 1.1, f"only {kept:.3f} of the LoRA delta reached the output"


def _peft_run(fn, model, X0, autocast_dtype):
    model.zero_grad(set_to_none = True)
    X = X0.detach().clone().requires_grad_()
    with torch.autocast(
        X0.device.type, dtype = autocast_dtype or torch.float32, enabled = autocast_dtype is not None
    ):
        out = fn(X)
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator().manual_seed(11)
    loss = sum(
        (o.double() * torch.randn(o.shape, generator = gen, dtype = torch.float64).to(o.device)).sum()
        for o in outs
    )
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return [o.detach() for o in outs], X.grad.clone(), grads


@pytest.mark.parametrize("device,path", CASES)
@pytest.mark.parametrize("autocast_dtype", [None, torch.float16])
def test_fp32_model_without_bf16_autocast_keeps_peft_forward(
    monkeypatch, device, path, autocast_dtype
):
    # float32 training (no autocast) and float16 autocast run PEFT's forward, bitwise, as before.
    if (
        autocast_dtype is torch.float16
        and device == "cpu"
        and not hasattr(torch, "get_autocast_dtype")
    ):
        pytest.skip("CPU float16 autocast")
    model, block = _fp32_block(device)
    X0 = torch.randn(2, 5, H, device = device)
    spy = _Spy(monkeypatch)
    got = _peft_run(PATHS[path][0](block), model, X0, autocast_dtype)
    assert spy.calls == [], "a fused LoRA Function ran"
    ref = _peft_run(PATHS[path][1](block), model, X0, autocast_dtype)
    for a, b in zip(got[0], ref[0]):
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)
    torch.testing.assert_close(got[1], ref[1], rtol = 0, atol = 0)
    assert got[2].keys() == ref[2].keys()
    for name in ref[2]:
        torch.testing.assert_close(got[2][name], ref[2][name], rtol = 0, atol = 0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if has_real_cuda() else []))
def test_other_fp32_mismatches_still_run_peft(monkeypatch, device):
    # A trainable float32 base keeps PEFT's forward (its gradient), as before.
    model, block = _fp32_block(device)
    block.o_proj.base_layer.weight.requires_grad_(True)
    X = torch.randn(2, 5, H, device = device)
    spy = _Spy(monkeypatch)
    with torch.autocast(device, dtype = torch.bfloat16):
        fast_lora.apply_lora_o(block, X)
    assert spy.calls == []
    assert fast_lora._autocast_master_dtype(block, (block.o_proj,), X) is None
    block.o_proj.base_layer.weight.requires_grad_(False)
    # No autocast: nothing to cast.
    assert fast_lora._autocast_master_dtype(block, (block.o_proj,), X) is None
    with torch.autocast(device, dtype = torch.bfloat16):
        assert fast_lora._autocast_master_dtype(block, (block.o_proj,), X) is torch.bfloat16
        # bfloat16 activations into float32 bases: cast only on a module patch_peft_model marked.
        assert (
            fast_lora._autocast_master_dtype(block, (block.o_proj,), X.bfloat16()) is torch.bfloat16
        )
        del block.__dict__[fast_lora._BF16_AUTOCAST_ONLY]
        assert fast_lora._autocast_master_dtype(block, (block.o_proj,), X.bfloat16()) is None
