# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


class _Norm(torch.nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad = False)
        self.variance_epsilon = eps


def _ulps(a, b):
    bits = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}
    a_i = a.view(bits[a.dtype]).to(torch.int64)
    b_i = b.view(bits[b.dtype]).to(torch.int64)
    return (a_i - b_i).abs().max().item()


def _inputs(rows, hidden, dtype, w_dtype, seed):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    X = torch.randn((rows, 1, hidden), device = "cuda", generator = g).mul_(3).to(dtype)
    R = torch.randn((rows, 1, hidden), device = "cuda", generator = g).mul_(20).to(dtype)
    W = (1 + 0.5 * torch.randn(hidden, device = "cuda", generator = g)).to(w_dtype)
    return X, R, W


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [2048, 3072, 4096, 5120])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("add", [True, False])
@torch.inference_mode()
def test_decode_step_matches_unfused(dtype, hidden, rows, add):
    from unsloth.models.llama import fast_rms_layernorm_inference
    from unsloth.kernels.fused_add_rms_layernorm import (
        fast_add_rms_layernorm_inference,
        fast_add_rms_layernorm_inference_ok,
    )

    X, R, W = _inputs(rows, hidden, dtype, dtype, seed = hidden * 10 + rows)
    norm = _Norm(W, 1e-5)
    # The decode loop's residual is a float32 buffer holding values of the model dtype.
    residual_old = R.float()
    residual_new = residual_old.clone()

    X_old = X.clone()
    XX = torch.empty_like(X_old, dtype = torch.float32)
    XX2 = torch.empty_like(XX)
    variance = torch.empty((rows, 1, 1), dtype = torch.float32, device = "cuda")
    if add:
        X_old += residual_old
    residual_old.copy_(X_old)
    out_old = fast_rms_layernorm_inference(norm, X_old, XX = XX, XX2 = XX2, variance = variance)

    X_new = X.clone()
    assert fast_add_rms_layernorm_inference_ok(X_new, residual_new, norm.weight)
    out_new = fast_add_rms_layernorm_inference(X_new, residual_new, norm.weight, 1e-5, add = add)
    assert out_new.data_ptr() == X_new.data_ptr()

    assert torch.equal(residual_new, residual_old)
    # 1 ulp in the normed row (fp32 sums in another order) can become 2 after a weight > 1.
    assert _ulps(out_new, out_old) <= 2
    assert (out_new != out_old).float().mean().item() < 0.01

    ones = torch.ones_like(W)
    unit_old = fast_rms_layernorm_inference(_Norm(ones, 1e-5), residual_old.to(dtype))
    unit_new = fast_add_rms_layernorm_inference(
        residual_old.to(dtype), residual_old.clone(), ones, 1e-5, add = False
    )
    assert _ulps(unit_new, unit_old) <= 1

    # fp64 oracle from the same rounded h: the fused error is never more than one rounding worse, where
    # the normed value's ulp is scaled by |w| before the output's own rounding.
    h = residual_old.double()
    eps = 2.0 ** -(10 if dtype == torch.float16 else 7)
    normed = (h * torch.rsqrt(h.square().mean(-1, keepdim = True) + 1e-5)).to(dtype).double()
    ref = normed * W.double()
    err_old = (out_old.double() - ref).abs()
    err_new = (out_new.double() - ref).abs()
    ulp = normed.abs() * eps * W.double().abs() + ref.abs().to(dtype).double() * eps
    assert torch.all(err_new <= err_old + ulp + 1e-12)
    assert err_new.mean().item() <= err_old.mean().item() * 1.05 + 1e-9


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4])
@torch.inference_mode()
def test_strided_rows_match_unfused(dtype, rows):
    # The decode MLP output for batch > 1 is X[..., :hidden] of an (rows, 1, mlp_size) buffer.
    from unsloth.models.llama import fast_rms_layernorm_inference
    from unsloth.kernels.fused_add_rms_layernorm import (
        fast_add_rms_layernorm_inference,
        fast_add_rms_layernorm_inference_ok,
    )

    X, R, W = _inputs(rows, 2048, dtype, dtype, seed = 7 + rows)
    norm = _Norm(W, 1e-5)
    wide_old = torch.full((rows, 1, 8192), 5.0, dtype = dtype, device = "cuda")
    wide_new = wide_old.clone()
    wide_old[..., :2048] = X
    wide_new[..., :2048] = X
    X_old, X_new = wide_old[..., :2048], wide_new[..., :2048]
    residual_old, residual_new = R.float(), R.float()

    X_old += residual_old
    residual_old.copy_(X_old)
    fast_rms_layernorm_inference(
        norm,
        X_old,
        XX = torch.empty_like(residual_old),
        XX2 = torch.empty_like(residual_old),
        variance = torch.empty((rows, 1, 1), device = "cuda"),
    )
    assert fast_add_rms_layernorm_inference_ok(X_new, residual_new, norm.weight)
    fast_add_rms_layernorm_inference(X_new, residual_new, norm.weight, 1e-5)

    assert torch.equal(residual_new, residual_old)
    assert _ulps(wide_new, wide_old) <= 2
    assert torch.all(wide_new[..., 2048:] == 5.0)


@torch.inference_mode()
def test_ok_rejects_what_it_cannot_take():
    from unsloth.kernels.fused_add_rms_layernorm import fast_add_rms_layernorm_inference_ok

    X, R, W = _inputs(2, 2048, torch.bfloat16, torch.bfloat16, seed = 0)
    assert fast_add_rms_layernorm_inference_ok(X, R, W)
    assert not fast_add_rms_layernorm_inference_ok(X.cpu(), R.cpu(), W.cpu())
    assert not fast_add_rms_layernorm_inference_ok(X, R[:1], W)
    assert not fast_add_rms_layernorm_inference_ok(X.transpose(0, 2), R.transpose(0, 2), W)
    assert not fast_add_rms_layernorm_inference_ok(X[..., ::2], R[..., ::2], W[::2])
    assert not fast_add_rms_layernorm_inference_ok(X, R, W[:-1])
