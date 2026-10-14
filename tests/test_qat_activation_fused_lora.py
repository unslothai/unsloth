# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Activation QAT must reach every fake quantizer: the fused LoRA paths only saw the shared input."""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device"),
]
torchao_qat = pytest.importorskip("torchao.quantization.qat")

H, I, R = 64, 128, 32
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(H, H, bias = False)
        self.k_proj = torch.nn.Linear(H, H, bias = False)
        self.v_proj = torch.nn.Linear(H, H, bias = False)
        self.o_proj = torch.nn.Linear(H, H, bias = False)
        self.gate_proj = torch.nn.Linear(H, I, bias = False)
        self.up_proj = torch.nn.Linear(H, I, bias = False)
        self.down_proj = torch.nn.Linear(I, H, bias = False)
        self.act_fn = torch.nn.SiLU()


def _qat_block(base_config, filter_fn = None):
    from peft import LoraConfig, get_peft_model
    from torchao.quantization import quantize_

    torch.manual_seed(3407)
    cfg = LoraConfig(r = R, lora_alpha = R, target_modules = TARGETS, init_lora_weights = False)
    model = get_peft_model(_Block(), cfg).to("cuda", torch.bfloat16)
    # Prepared after LoRA is attached, as _prepare_model_for_qat does.
    quantize_(model, torchao_qat.QATConfig(base_config, step = "prepare"), filter_fn = filter_fn)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    return model, model.base_model.model


def _int8_act_int4_weight():
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig
    from torchao.quantization.granularity import PerGroup
    return Int8DynamicActivationIntxWeightConfig(
        weight_dtype = torch.int4, weight_granularity = PerGroup(32)
    )


def _run(fn, model, X):
    X = X.clone().requires_grad_(True)
    out = fn(X)
    out = out if isinstance(out, tuple) else (out,)
    sum(
        (o.float() * torch.linspace(-1, 1, o.shape[-1], device = "cuda")).sum() for o in out
    ).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return out, X.grad, grads


def _assert_matches_modules(
    model,
    block,
    blocks = ("qkv", "o", "mlp"),
):
    from unsloth.kernels import apply_lora_mlp_swiglu, apply_lora_o, apply_lora_qkv

    X = torch.randn(2, 5, H, device = "cuda", dtype = torch.bfloat16)
    pairs = {
        "qkv": (
            lambda X: apply_lora_qkv(block, X),
            lambda X: (block.q_proj(X), block.k_proj(X), block.v_proj(X)),
        ),
        "o": (lambda X: apply_lora_o(block, X), block.o_proj),
        "mlp": (
            lambda X: apply_lora_mlp_swiglu(block, X),
            lambda X: block.down_proj(block.act_fn(block.gate_proj(X)) * block.up_proj(X)),
        ),
    }
    for fast, ref in (pairs[b] for b in blocks):
        got, got_dX, got_grads = _run(fast, model, X)
        exp, exp_dX, exp_grads = _run(ref, model, X)
        for g, e in zip(got, exp):
            torch.testing.assert_close(g, e, rtol = 0, atol = 0)
        torch.testing.assert_close(got_dX, exp_dX, rtol = 0, atol = 0)
        assert got_grads.keys() == exp_grads.keys()
        for n in exp_grads:
            torch.testing.assert_close(got_grads[n], exp_grads[n], rtol = 0, atol = 0)


def test_activation_qat_matches_module_forward():
    _assert_matches_modules(*_qat_block(_int8_act_int4_weight()))


@pytest.mark.parametrize(
    "suffix, blocks", [("down_proj.base_layer", ("mlp",)), ("lora_B.default", ("qkv", "o", "mlp"))]
)
def test_selective_activation_qat_matches_module_forward(suffix, blocks):
    # A quantizer on an inner linear only (not the shared input) is still honoured.
    only = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith(suffix)
    _assert_matches_modules(*_qat_block(_int8_act_int4_weight(), only), blocks)


def test_weight_only_qat_keeps_fused_path(monkeypatch):
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    import unsloth.kernels.fast_lora as fast_lora

    _, block = _qat_block(IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)))
    fused = []
    real_apply = fast_lora._apply
    monkeypatch.setattr(fast_lora, "_apply", lambda fn, *a: fused.append(fn) or real_apply(fn, *a))
    X = torch.randn(2, 5, H, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        fast_lora.apply_lora_qkv(block, X.clone())
        fast_lora.apply_lora_o(block, X.clone())
        fast_lora.apply_lora_mlp_swiglu(block, X.clone())
    assert fused == [fast_lora.LoRA_QKV, fast_lora.LoRA_W, fast_lora.LoRA_MLP]
