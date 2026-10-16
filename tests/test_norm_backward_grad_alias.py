# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# incoming grads may be shared by nodes or passed through grad_outputs; do not overwrite them

import pytest
from real_accelerator import has_real_cuda
import torch


def _rms_fp64(x, W, gemma):
    x = x.double()
    W = W.double() + (1.0 if gemma else 0.0)
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + 1e-6) * W


def _ln_fp64(x, W, b):
    x, W, b = x.double(), W.double(), b.double()
    return torch.nn.functional.layer_norm(x, x.shape[-1:], W, b, 1e-6)


def _graphs(norm, shape, device, dtype):
    def user_grad(x, go, reference):
        out = torch.autograd.grad(norm(x), x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64), x64, go.double())[0]

    def add_self(x, go, reference):
        out = torch.autograd.grad(norm(x) + x, x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64) + x64, x64, go.double())[0]

    def two_norms_sum(x, go, reference):
        out = torch.autograd.grad(norm(x) + norm(2 * x), x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64) + reference(2 * x64), x64, go.double())[0]

    return {"user_grad": user_grad, "add_self": add_self, "two_norms_sum": two_norms_sum}


_TOLERANCE = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-1}


def _check(norm, reference, dtype):
    torch.manual_seed(0)
    shape = (3, 5, 256)
    for name, graph in _graphs(norm, shape, "cuda", dtype).items():
        x = torch.randn(shape, device = "cuda", dtype = dtype).requires_grad_()
        go = torch.randn(shape, device = "cuda", dtype = dtype)
        go_before = go.clone()
        dX, dX64 = graph(x, go, reference)
        assert torch.equal(go, go_before), f"{name}: incoming grad was overwritten"
        error = (dX.double() - dX64).abs().max().item()
        assert error <= _TOLERANCE[dtype], f"{name}: max error vs fp64 {error}"


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("save_output", [False, True])
@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_layernorm_backward_leaves_shared_grad_alone(gemma, save_output, train_weight, dtype):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    W = (1.0 + 0.1 * torch.randn(256, device = "cuda")).to(dtype).requires_grad_(train_weight)
    norm = lambda x: Fast_RMS_Layernorm.apply(x, W, 1e-6, gemma, save_output)
    _check(norm, lambda x: _rms_fp64(x, W, gemma), dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layernorm_backward_leaves_shared_grad_alone(train_weight, dtype):
    from unsloth.kernels.layernorm import Fast_Layernorm

    W = (1.0 + 0.1 * torch.randn(256, device = "cuda")).to(dtype).requires_grad_(train_weight)
    b = (0.1 * torch.randn(256, device = "cuda")).to(dtype).requires_grad_(train_weight)
    norm = lambda x: Fast_Layernorm.apply(x, W, b, 1e-6)
    _check(norm, lambda x: _ln_fp64(x, W, b), dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("save_output", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_layernorm_reuse_grad_matches_and_saves_a_buffer(
    gemma, save_output, dtype, monkeypatch
):
    # Pre-norm decoder norms feed only layers that return a fresh grad, so they let dX overwrite
    # dY: same bits as the fresh buffer, and about one activation less at the backward peak.
    from unsloth.kernels import rms_layernorm
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    monkeypatch.setattr(rms_layernorm, "_SAVE_OUTPUT", save_output)
    torch.manual_seed(0)
    shape = (2, 512, 1024)
    W = (1.0 + 0.1 * torch.randn(shape[-1], device = "cuda")).to(dtype)
    proj = torch.randn(shape[-1], 64, device = "cuda", dtype = dtype) / 32
    x0 = torch.randn(shape, device = "cuda", dtype = dtype)

    def step(reuse_grad):
        x = x0.clone().requires_grad_()
        y = Fast_RMS_Layernorm.apply(x, W, 1e-6, gemma, save_output, reuse_grad)
        (y @ proj).float().square().mean().backward()
        return x.grad

    results = {}
    for reuse_grad in (False, True):
        step(reuse_grad)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        grad = step(reuse_grad)
        torch.cuda.synchronize()
        results[reuse_grad] = (grad, torch.cuda.max_memory_allocated() - base)
    assert torch.equal(results[False][0], results[True][0])
    saved = results[False][1] - results[True][1]
    activation = x0.numel() * x0.element_size()
    # The peak moves to a neighbouring allocation, so the drop is most of an activation, not all.
    if gemma:
        assert saved == 0, saved
    else:
        assert activation * 7 // 8 <= saved <= activation, (saved, activation)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
def test_final_norm_reuses_grad_only_under_the_causal_lm_head(monkeypatch):
    # A base model hands the final norm's output to the caller, so the caller's grad_outputs must
    # survive; only the CausalLM head, which consumes it itself, lets the norm write over its grad.
    from unsloth import FastLanguageModel
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    model, _ = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    model.train()
    backbone = model.model
    torch.manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (2, 32), device = "cuda")

    embeds = backbone.embed_tokens(ids).detach().requires_grad_()
    hidden = backbone(inputs_embeds = embeds, use_cache = False).last_hidden_state
    go = torch.randn_like(hidden)
    go_before = go.clone()
    torch.autograd.grad(hidden, embeds, go)
    assert torch.equal(go, go_before), "base model overwrote the caller's grad_outputs"

    calls = []
    apply = Fast_RMS_Layernorm.apply

    def recording_apply(*args):
        calls.append(bool(args[5]) if len(args) > 5 else False)
        return apply(*args)

    monkeypatch.setattr(Fast_RMS_Layernorm, "apply", recording_apply)
    model(input_ids = ids, labels = ids, use_cache = False)
    assert calls and calls[-1], "CausalLM head should let the final norm reuse its grad"
    calls.clear()
    model(input_ids = ids, labels = ids, use_cache = False, output_hidden_states = True)
    assert calls and not calls[-1], "returned hidden states need the final norm's own buffer"
    calls.clear()
    backbone(input_ids = ids, use_cache = False)
    assert calls and not calls[-1]
    assert not getattr(backbone, "_unsloth_head_reads_final_norm", False)
