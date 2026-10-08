# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The fused LoRA paths must sum every active adapter like PEFT, not just active_adapters[0]."""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device"),
]

H, I = 64, 128


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


def _peft_block(adapters, lora_bias = False):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3407)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = lambda r: LoraConfig(
        r = r,
        lora_alpha = 2 * r,
        target_modules = targets,
        init_lora_weights = False,
        lora_bias = lora_bias,
    )
    model = get_peft_model(_Block(), cfg(2), adapter_name = "a")
    if adapters > 1:
        model.add_adapter("b", cfg(3))
        model.base_model.set_adapter(["a", "b"])
    model = model.to("cuda", torch.bfloat16)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    if lora_bias:
        with torch.no_grad():
            for proj in targets:
                model.base_model.model.get_submodule(proj).lora_B["a"].bias.normal_()
    return model, model.base_model.model


def _lora_b_grads(model):
    return {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if "lora_B" in n and p.grad is not None
    }


def _check(fast, ref, model):
    X = torch.randn(2, 5, H, device = "cuda", dtype = torch.bfloat16)
    out = fast(X.clone())  # the fused kernels may write into X
    sum(o.float().sum() for o in (out if isinstance(out, tuple) else (out,))).backward()
    fast_grads = _lora_b_grads(model)
    model.zero_grad(set_to_none = True)
    expected = ref(X)
    sum(
        o.float().sum() for o in (expected if isinstance(expected, tuple) else (expected,))
    ).backward()
    ref_grads = _lora_b_grads(model)
    model.zero_grad(set_to_none = True)
    for o, e in zip(
        out if isinstance(out, tuple) else (out,),
        expected if isinstance(expected, tuple) else (expected,),
    ):
        torch.testing.assert_close(o, e, rtol = 2e-2, atol = 2e-2)
    assert fast_grads.keys() == ref_grads.keys()
    for n in ref_grads:
        torch.testing.assert_close(fast_grads[n], ref_grads[n], rtol = 5e-2, atol = 5e-2)
    return ref_grads


@pytest.mark.parametrize("adapters", [1, 2])
def test_qkv_o_mlp_match_peft(adapters):
    from unsloth.kernels import apply_lora_mlp_swiglu, apply_lora_o, apply_lora_qkv

    model, block = _peft_block(adapters)
    qkv = _check(
        lambda X: apply_lora_qkv(block, X),
        lambda X: (block.q_proj(X), block.k_proj(X), block.v_proj(X)),
        model,
    )
    _check(lambda X: apply_lora_o(block, X), block.o_proj, model)
    mlp = _check(
        lambda X: apply_lora_mlp_swiglu(block, X),
        lambda X: block.down_proj(block.act_fn(block.gate_proj(X)) * block.up_proj(X)),
        model,
    )
    if adapters > 1:
        # The second adapter must be in the graph, not just the first.
        assert any(".b." in n for n in qkv) and any(".b." in n for n in mlp)


def test_qkv_o_mlp_match_peft_with_lora_bias():
    from unsloth.kernels import apply_lora_mlp_swiglu, apply_lora_o, apply_lora_qkv

    model, block = _peft_block(1, lora_bias = True)
    _check(
        lambda X: apply_lora_qkv(block, X),
        lambda X: (block.q_proj(X), block.k_proj(X), block.v_proj(X)),
        model,
    )
    _check(lambda X: apply_lora_o(block, X), block.o_proj, model)
    _check(
        lambda X: apply_lora_mlp_swiglu(block, X),
        lambda X: block.down_proj(block.act_fn(block.gate_proj(X)) * block.up_proj(X)),
        model,
    )


@pytest.mark.parametrize("adapters", [1, 2])
def test_fast_linear_forward_decode_matches_peft(adapters):
    from unsloth.kernels import fast_linear_forward

    _, block = _peft_block(adapters)
    X = torch.randn(1, 1, H, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        out = torch.empty(1, 1, H, device = "cuda", dtype = torch.bfloat16)
        got = fast_linear_forward(block.q_proj, X, out = out)
        torch.testing.assert_close(got, block.q_proj(X), rtol = 2e-2, atol = 2e-2)
        assert got.data_ptr() == out.data_ptr()


@pytest.mark.parametrize("q_len", [1, 5])
def test_fast_linear_forward_applies_dora_magnitude(q_len):
    from peft import LoraConfig, get_peft_model
    from unsloth.kernels import fast_linear_forward

    torch.manual_seed(3407)
    cfg = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False, use_dora = True
    )
    block = get_peft_model(_Block(), cfg).to("cuda", torch.bfloat16).base_model.model
    X = torch.randn(1, q_len, H, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        # A trained magnitude no longer equals the row norms DoRA starts from.
        block.q_proj.lora_magnitude_vector["default"].weight.mul_(1.5)
        torch.testing.assert_close(
            fast_linear_forward(block.q_proj, X), block.q_proj(X), rtol = 2e-2, atol = 2e-2
        )


@pytest.mark.parametrize("q_len", [1, 5])
def test_fast_linear_forward_adds_lora_bias(q_len):
    from peft import LoraConfig, get_peft_model
    from unsloth.kernels import fast_linear_forward

    torch.manual_seed(3407)
    cfg = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False, lora_bias = True
    )
    block = get_peft_model(_Block(), cfg).to("cuda", torch.bfloat16).base_model.model
    X = torch.randn(1, q_len, H, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        # A trained lora_B bias is no longer the zeros it starts from.
        block.q_proj.lora_B["default"].bias.normal_()
        torch.testing.assert_close(
            fast_linear_forward(block.q_proj, X), block.q_proj(X), rtol = 2e-2, atol = 2e-2
        )
