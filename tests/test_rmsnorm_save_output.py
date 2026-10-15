# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_RMSNORM_SAVE_OUTPUT: Fast_RMS_Layernorm saves its output Y instead of X and rebuilds
normed = Y / W in backward, so the norm and the LoRA layer after it share one saved activation."""

import pytest
import torch

from unsloth.kernels import rms_layernorm


def test_guard_rejects_weights_y_cannot_be_divided_by():
    ok = rms_layernorm._save_output_ok
    w = torch.full((8,), 0.5)
    assert ok(w, False, torch.bfloat16)
    w2 = w.clone()
    w2[3] = 0
    assert not ok(w2, False, torch.bfloat16)
    w3 = torch.zeros(8)
    w3[1] = -1
    assert not ok(w3, True, torch.bfloat16)
    # fp16 Y goes subnormal long before bf16 / fp32 Y does.
    w4 = torch.full((8,), 1e-4)
    assert not ok(w4, False, torch.float16)
    assert ok(w4, False, torch.bfloat16)
    w5 = w.clone()
    w5[0] = float("nan")
    assert not ok(w5, False, torch.float32)


def test_guard_cache_follows_in_place_updates():
    w = torch.full((8,), 0.5)
    assert rms_layernorm._save_output_ok(w, False, torch.bfloat16)
    w.zero_()
    assert not rms_layernorm._save_output_ok(w, False, torch.bfloat16)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _run(x, w, gemma, save_output, monkeypatch, enabled = True):
    monkeypatch.setattr(rms_layernorm, "_SAVE_OUTPUT", enabled)
    x = x.detach().clone().requires_grad_(True)
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = rms_layernorm.Fast_RMS_Layernorm.apply(x, w, 1e-6, gemma, save_output)
    g = torch.randn(y.shape, device = y.device, generator = torch.Generator(y.device).manual_seed(1))
    y.backward(g.to(y.dtype))
    return y, x.grad, saved


def _ref_grad(x, w, gemma, eps = 1e-6):
    x64 = x.detach().double().requires_grad_(True)
    w64 = w.double() + 1 if gemma else w.double()
    y = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim = True) + eps) * w64
    g = torch.randn(y.shape, device = y.device, generator = torch.Generator(y.device).manual_seed(1))
    y.backward(g.to(x.dtype).double())
    return x64.grad


@cuda
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [64, 127, 1000, 4097])
def test_save_output_grads_and_storage(gemma, dtype, dim, monkeypatch):
    gen = torch.Generator("cuda").manual_seed(0)
    x = (torch.randn(3, 37, dim, device = "cuda", generator = gen) * 2 + 0.3).to(dtype)
    w = torch.randn(dim, device = "cuda", generator = gen) * 0.1 + (0 if gemma else 1)
    w = w.to(dtype)
    y0, dx0, saved0 = _run(x, w, gemma, False, monkeypatch)
    y1, dx1, saved1 = _run(x, w, gemma, True, monkeypatch)
    assert torch.equal(y0, y1)
    # Old path saves X, new path saves the returned Y itself (shared storage, no extra bytes).
    assert saved0[0].data_ptr() != y0.data_ptr()
    assert saved1[0].data_ptr() == y1.data_ptr()
    ref = _ref_grad(x, w, gemma)
    err0 = (dx0.double() - ref).abs().max().item()
    err1 = (dx1.double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    # Y / W carries one rounding of Y: at most a few output ulps over the X path.
    assert err1 <= 2 * err0 + 4 * torch.finfo(dtype).eps * scale, (err0, err1, scale)
    assert torch.isfinite(dx1).all()


@cuda
@pytest.mark.parametrize("gemma", [False, True])
def test_save_output_off_or_unsafe_is_bitwise_old_path(gemma, monkeypatch):
    gen = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(2, 33, 256, device = "cuda", generator = gen).to(torch.bfloat16)
    w = (torch.randn(256, device = "cuda", generator = gen) * 0.1 + (0 if gemma else 1)).to(torch.bfloat16)
    y0, dx0, _ = _run(x, w, gemma, False, monkeypatch)
    y1, dx1, saved = _run(x, w, gemma, True, monkeypatch, enabled = False)
    assert torch.equal(dx0, dx1) and saved[0].data_ptr() != y1.data_ptr()
    # A weight Y cannot be divided by falls back to saving X.
    w_bad = w.clone()
    w_bad[5] = -1 if gemma else 0
    y2, dx2, _ = _run(x, w_bad, gemma, False, monkeypatch)
    y3, dx3, saved = _run(x, w_bad, gemma, True, monkeypatch)
    assert torch.equal(dx2, dx3) and saved[0].data_ptr() != y3.data_ptr()
    w_train = w.clone().requires_grad_(True)
    y4, _, saved = _run(x, w_train, gemma, True, monkeypatch)
    assert saved[0].data_ptr() != y4.data_ptr()


@cuda
def test_lora_does_not_overwrite_a_saved_norm_output(monkeypatch):
    from unsloth.kernels import fast_lora

    x = torch.randn(2, 8, 128, device = "cuda", dtype = torch.float16)
    w = torch.ones(128, device = "cuda", dtype = torch.float16)
    for enabled in (False, True):
        monkeypatch.setattr(rms_layernorm, "_SAVE_OUTPUT", enabled)
        y = rms_layernorm.Fast_RMS_Layernorm.apply(x.clone().requires_grad_(True), w, 1e-6, False, True)
        # LoRA_QKV / LoRA_MLP write dX into their input X in backward unless the norm saved it.
        assert fast_lora._may_overwrite_input(y) is not enabled
