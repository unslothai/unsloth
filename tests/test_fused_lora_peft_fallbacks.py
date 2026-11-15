# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Configurations the fused LoRA kernels cannot run hand the projection back to PEFT's forward.

The fused Functions compute base(X) + S * lora_B(lora_A(dropout(X))) [+ bias] with frozen base weights in
X's dtype. A PEFT LoRA variant (its own forward), a disabled or merged adapter (base layer only: the
fused backward has no LoRA weights to use), a trainable base weight (its gradient) and an autocast dtype
other than X's (the in-place LoRA accumulation is not autocast) used to run fused anyway, training plain
LoRA, dropping gradients or crashing. Each is checked against PEFT run in float64 (CPU) / float32 (CUDA).
A disabled or merged adapter with no gradient needed (the DPO / GRPO reference forward) stays fused.
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


def _run(
    fn,
    model,
    X0,
    autocast_dtype = None,
):
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
    loss = sum(
        (o.double() * torch.randn(o.shape, generator = gen, dtype = torch.float64).to(o.device)).sum()
        for o in outs
    )
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return [o.detach() for o in outs], X.grad.clone(), grads


def _assert_falls_back_to_peft(
    monkeypatch,
    model,
    block,
    path,
    X0,
    autocast_dtype = None,
    exact = True,
):
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
def test_disabled_or_merged_adapter_backward_matches_peft(
    monkeypatch, device, dtype, path, state, dropout, lora_bias
):
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
def test_trainable_base_weight_gets_its_gradient(
    monkeypatch, device, dtype, path, dropout, lora_bias
):
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
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float32, torch.float16),
    ],
)
def test_autocast_dtype_other_than_input_runs_peft(
    monkeypatch, path, dropout, lora_bias, x_dtype, autocast_dtype
):
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


FUSED_NAMES = {"o": ["LoRA_W"], "qkv": ["LoRA_QKV"], "mlp": ["LoRA_MLP"]}


def _disable_or_merge(model, block, state):
    if state == "disabled":
        model.base_model.disable_adapter_layers()
    else:
        for name in TARGETS:
            block.get_submodule(name).merge()


@pytest.mark.parametrize("device,dtype,path", CASES)
@pytest.mark.parametrize("state", ["disabled", "merged"])
@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode", "grad_frozen_input"])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_disabled_or_merged_adapter_without_grad_stays_fused(
    monkeypatch, device, dtype, path, state, grad_mode, dropout
):
    # The reference forward: no gradient flows, so the fused base matmul runs and equals PEFT bitwise.
    model, block = _peft_block(device, dtype, dropout)
    _disable_or_merge(model, block, state)
    X = torch.randn(2, 5, H, device = device, dtype = dtype)
    spy = _Spy(monkeypatch)
    ctx = {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode}.get(
        grad_mode, torch.enable_grad
    )
    with ctx():
        fused = PATHS[path][0](block)(X)
    assert [name for name, _ in spy.calls] == FUSED_NAMES[path]
    args = spy.calls[0][1]
    if path == "o":
        # The default config's six-argument call.
        assert len(args) == 6
    else:
        # No LoRA weights, no dropout masks, no lora_B biases: the base matmul only.
        assert args[3] is None and args[-4] is None and all(b is None for b in args[-3:])
    with ctx():
        ref = PATHS[path][1](block)(X)
    # Each projection is PEFT's base_layer(X) bitwise; the MLP's Triton SwiGLU rounds its float32
    # activation 1 ulp differently from torch's SiLU * up.
    tol = {} if path == "mlp" else dict(rtol = 0, atol = 0)
    for a, b in zip(
        fused if isinstance(fused, tuple) else (fused,), ref if isinstance(ref, tuple) else (ref,)
    ):
        assert not a.requires_grad
        torch.testing.assert_close(a, b, **tol)


# Two bf16 ulps of the output's largest value: the two GEMMs and the MLP's second projection compound.
_BF16_TOL = 2.0**-6


def _dequantize_then_linear(self, x):
    import bitsandbytes as bnb
    W = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)
    return torch.nn.functional.linear(x, W.to(x.dtype))


@pytest.mark.skipif(not has_real_cuda(), reason = "bitsandbytes 4-bit needs CUDA")
@pytest.mark.parametrize("path", _paths("cuda"))
def test_disabled_adapter_without_grad_on_4bit_base_stays_fused(monkeypatch, path):
    bnb = pytest.importorskip("bitsandbytes")
    from peft import LoraConfig, get_peft_model

    class _Block4bit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, (i, o) in {
                "q_proj": (H, H),
                "k_proj": (H, H),
                "v_proj": (H, H),
                "o_proj": (H, H),
                "gate_proj": (H, 96),
                "up_proj": (H, 96),
                "down_proj": (96, H),
            }.items():
                layer = bnb.nn.Linear4bit(
                    i, o, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4"
                )
                layer.weight = bnb.nn.Params4bit(
                    layer.weight.data.to(torch.bfloat16), requires_grad = False, quant_type = "nf4"
                )
                setattr(self, name, layer)
            self.act_fn = torch.nn.SiLU()

    torch.manual_seed(3407)
    base = _Block4bit().to("cuda")
    cfg = LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.1, target_modules = TARGETS, init_lora_weights = False
    )
    model = get_peft_model(base, cfg)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.to(torch.bfloat16)
    block = model.base_model.model
    assert getattr(block.o_proj.base_layer.weight, "quant_state", None) is not None
    model.train()
    model.base_model.disable_adapter_layers()
    X = torch.randn(2, 7, H, device = "cuda", dtype = torch.bfloat16)
    spy = _Spy(monkeypatch)
    with torch.no_grad():
        fused = PATHS[path][0](block)(X)
        ref = PATHS[path][1](block)(X)
        # The fused kernels dequantize and run one matmul in X's dtype. bitsandbytes >= 0.50 answers
        # few-row inference calls with its fused 4-bit GEMM instead, which rounds differently.
        with monkeypatch.context() as m:
            m.setattr(bnb.nn.Linear4bit, "forward", _dequantize_then_linear)
            base_only = PATHS[path][1](block)(X)
    assert [name for name, _ in spy.calls] == FUSED_NAMES[path]
    outs = [t if isinstance(t, tuple) else (t,) for t in (fused, base_only, ref)]
    # On ROCm the MLP's SwiGLU and GEMMs need not round like torch's SiLU * up and F.linear (1 ulp seen on gfx1151).
    exact = path != "mlp" or torch.version.hip is None
    for a, b, c in zip(*outs):
        torch.testing.assert_close(
            a, b, rtol = 0, atol = 0 if exact else _BF16_TOL * b.abs().max().item()
        )
        # PEFT's own forward agrees to bf16 rounding (the LoRA term here is over 90x larger).
        torch.testing.assert_close(a, c, rtol = 0, atol = _BF16_TOL * c.abs().max().item())


def test_patch_summary_names_a_lora_variant():
    from unsloth.models.llama import _lora_variant_note

    model, block = _peft_block("cpu", torch.float32, 0.0)
    assert _lora_variant_note(model, "default") == ""
    block.o_proj.lora_variant["default"] = _DoubledLoRA()
    note = _lora_variant_note(model, "default")
    assert "_DoubledLoRA" in note and "PEFT's own forward" in note


def test_routing_compiles_without_graph_breaks(monkeypatch):
    # The added routing checks are Python attribute reads and grad / autocast mode queries: fullgraph traces
    # the fused, variant, trainable base, autocast and disabled (with and without grad) routes.
    if not fast_lora.TRACE_LORA_FUNCTIONS:
        pytest.skip("fused Functions run Dynamo-disabled before torch 2.11")
    import torch._dynamo

    model, block = _peft_block("cpu", torch.float32, 0.3, True)
    X = torch.randn(2, 5, H)

    def fn(X):
        return fast_lora.apply_lora_o(block, X), fast_lora.apply_lora_qkv(block, X)

    def check(
        X,
        autocast_dtype = None,
        grad = True,
    ):
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend = "eager", fullgraph = True)
        with (
            torch.set_grad_enabled(grad),
            torch.autocast(
                "cpu", dtype = autocast_dtype or torch.bfloat16, enabled = autocast_dtype is not None
            ),
        ):
            got = compiled(X)
            want = fn(X)
        torch.testing.assert_close(got[0], want[0])
        for a, b in zip(got[1], want[1]):
            torch.testing.assert_close(a, b)

    model.eval()  # deterministic dropout, so compiled and eager agree
    check(X)
    check(X, torch.bfloat16)  # autocast in another dtype: PEFT
    check(X.requires_grad_(), grad = True)
    for name in TARGETS:
        block.get_submodule(name).lora_variant["default"] = _DoubledLoRA()
    check(X)
    for name in TARGETS:
        del block.get_submodule(name).lora_variant["default"]
    block.o_proj.base_layer.weight.requires_grad_(True)
    check(X)
    block.o_proj.base_layer.weight.requires_grad_(False)
    model.base_model.disable_adapter_layers()
    check(X.detach(), grad = False)  # fused
    check(X.detach().requires_grad_(), grad = True)  # PEFT


@pytest.mark.parametrize("device,dtype,path", CASES)
@pytest.mark.parametrize("unload", ["merge_and_unload", "unload"])
@pytest.mark.parametrize("grad_mode", ["no_grad", "enable_grad"])
def test_unloaded_plain_linear_projections_run_the_fused_paths(
    monkeypatch, device, dtype, path, unload, grad_mode
):
    # merge_and_unload() / unload() swap the LoRA layers back to plain nn.Linear while the fused
    # apply_qkv / apply_o / MLP forwards stay installed on the attention and MLP modules, so the
    # routing guards must accept a projection with no lora_A / lora_B (forward / generate after merging).
    model, _ = _peft_block(device, dtype, 0.0)
    block = getattr(model, unload)()
    assert type(block.o_proj) is torch.nn.Linear
    assert "lora_A" not in block.o_proj._modules
    X = torch.randn(2, 5, H, device = device, dtype = dtype)
    ctx = torch.no_grad if grad_mode == "no_grad" else torch.enable_grad
    with ctx():
        fused = PATHS[path][0](block)(X)
        ref = PATHS[path][1](block)(X)
    tol = {} if path == "mlp" else dict(rtol = 0, atol = 0)
    for a, b in zip(
        fused if isinstance(fused, tuple) else (fused,), ref if isinstance(ref, tuple) else (ref,)
    ):
        torch.testing.assert_close(a, b, **tol)


@pytest.mark.parametrize("device,dtype,path", [c for c in CASES if c[2] != "o"])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_partially_merged_block_with_frozen_input_matches_peft(
    monkeypatch, device, dtype, path, dropout
):
    # X needs no grad, but the block's other active adapters train, so the fused backward would run with
    # no LoRA weights for the merged projection ('NoneType' object has no attribute 'to').
    model, block = _peft_block(device, dtype, dropout)
    block.k_proj.merge()
    block.gate_proj.merge()
    model.eval()  # dropout off, so the fused and PEFT passes are deterministic
    X = torch.randn(2, 5, H, device = device, dtype = dtype)
    spy = _Spy(monkeypatch)
    results = []
    for fn in (PATHS[path][0](block), PATHS[path][1](block)):
        out = fn(X)
        outs = out if isinstance(out, tuple) else (out,)
        sum(o.double().square().sum() for o in outs).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none = True)
        results.append(([o.detach() for o in outs], grads))
    assert spy.calls == [], "a fused LoRA Function ran"
    (fused, fused_grads), (ref, ref_grads) = results
    for a, b in zip(fused, ref):
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)
    assert fused_grads.keys() == ref_grads.keys() and fused_grads
    for name in ref_grads:
        torch.testing.assert_close(fused_grads[name], ref_grads[name], rtol = 0, atol = 0)
