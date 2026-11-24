# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tensor parallel LoRA projection rounds as the one-GPU compiled LoRA forward (unsloth_zoo compiler
`lora_forward`): the low-rank product in the base output's dtype (float16 under forced float32), the update
added by one addmm. Output and lora_A / lora_B gradients must be bitwise those of the one-GPU forward.

Under forced float32 (Gemma 3 / 4 on float16 GPUs) the tp product used to run in float32: per-token LoRA
gradients that underflow float16 on one GPU survived on the tp ranks, so Gemma-4 E4B 4-bit tp=2 on 2 x T4
trained on other gradients (25-step loss 4.0 vs 3.05 on one GPU). CPU, one process, no DTensor."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402  (tp_native without the GPU-only unsloth import)


class _LoraProj(nn.Module):
    def __init__(
        self,
        base_dtype,
        in_features = 64,
        out_features = 48,
        r = 8,
        seed = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base_layer = nn.Linear(in_features, out_features, bias = False).to(base_dtype)
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(out_features, in_features, generator = g) * 0.05)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias = False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias = False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.randn(r, in_features, generator = g) * 0.05)
            self.lora_B["default"].weight.copy_(torch.randn(out_features, r, generator = g) * 0.05)
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.scaling = {"default": 2.0}
        self.disable_adapters = False
        self.merged = False
        self.active_adapters = ["default"]
        self._unsloth_tp_style = tpn.COLWISE


def _with_forward(flag):
    """A layer class whose forward is Unsloth's compiled LoRA forward built under `flag`, or PEFT's (None)."""
    if flag is None:
        return _LoraProj

    def unsloth_forward(self, x):
        return _one_gpu_forward(self, x, flag == "1")

    unsloth_forward._unsloth_force_float32 = flag
    return type("CompiledLoraProj", (_LoraProj,), {"forward": unsloth_forward})


def _peft_forward(proj, x):
    """PEFT's forward (no Unsloth compile): x cast to the LoRA weights' dtype, product in float32."""
    A = proj.lora_A["default"].weight
    B = proj.lora_B["default"].weight
    result = torch.nn.functional.linear(x, proj.base_layer.weight)
    return result + (
        torch.nn.functional.linear(torch.nn.functional.linear(x.to(A.dtype), A), B) * 2.0
    ).to(result.dtype)


def _one_gpu_forward(proj, x, forced):
    """unsloth_zoo compiler.py COMPILED_LORA_FORWARD(_forced_float32) on the base output."""
    A = proj.lora_A["default"].weight
    B = proj.lora_B["default"].weight
    result = torch.nn.functional.linear(x, proj.base_layer.weight)
    if forced:
        xA = x.to(torch.float16) @ A.to(torch.float16).t()
        xA = xA.to(result.dtype)
    else:
        xA = x.to(result.dtype) @ A.to(result.dtype).t()
    shape = result.shape
    return torch.addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        B.to(result.dtype).t(),
        alpha = 2.0,
        beta = 1,
    ).view(shape)


def _run(fn, proj, x, g):
    for p in (proj.lora_A["default"].weight, proj.lora_B["default"].weight):
        p.grad = None
    y = fn(proj, x)
    y.backward(g)
    return (
        y.detach(),
        proj.lora_A["default"].weight.grad.clone(),
        proj.lora_B["default"].weight.grad.clone(),
    )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


@pytest.mark.parametrize("flag", ["1", "0", None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tp_lora_matches_one_gpu_forward(flag, dtype):
    if flag == "1" and dtype is not torch.float16:
        pytest.skip("forced float32 keeps float16 base weights")
    proj = _with_forward(flag)(dtype)
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 16, 64, generator = g) * 0.5).to(dtype)
    # Tiny upstream gradients, as a loss-scaled float16 backward sees in Gemma-4's upper layers: the one-GPU
    # float16 product flushes part of the low-rank gradient to zero, a float32 product keeps it.
    grad = (torch.randn(2, 16, 48, generator = g) * 3e-6).to(dtype)
    one_gpu = (lambda p, xx: p.forward(xx)) if flag is not None else _peft_forward
    ref = _run(one_gpu, proj, x, grad)
    got = _run(tpn._tp_projection_compute, proj, x, grad)
    for name, a, b in zip(("output", "lora_A grad", "lora_B grad"), ref, got):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        bad = int((_bits(a) != _bits(b)).sum())
        assert (
            bad == 0
        ), f"{name}: {bad} of {a.numel()} elements differ from the one-GPU LoRA forward"


def test_forced_float32_one_gpu_flushes_what_float32_keeps():
    """The case that moved Gemma-4 tp=2 off one GPU: under forced float32 the one-GPU lora_A gradient loses
    entries a float32 product keeps, so a float32 tp product trains on different gradients."""
    proj = _with_forward("1")(torch.float16)
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 16, 64, generator = g) * 0.5).to(torch.float16)
    grad = (torch.randn(2, 16, 48, generator = g) * 3e-6).to(torch.float16)
    _, a16, _ = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    _, a32, _ = _run(_peft_forward, proj, x, grad)
    rel = float((a16 - a32).norm() / a32.norm())
    assert rel > 1e-2, rel  # a real difference, not rounding noise
