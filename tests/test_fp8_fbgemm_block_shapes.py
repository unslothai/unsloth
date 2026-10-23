# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Covers fbgemm <=1.3.0 output corruption on some tile grids, which the 1x1 import probe misses."""

import math

import pytest
import torch

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(not (cuda_available or xpu_available), reason = "needs CUDA or XPU")


def skip_without_fbgemm():
    # unsloth's own probe, not an sm_90 check, so future arches enable themselves.
    # Imported here so collection never imports unsloth.
    from unsloth.kernels import fp8
    if fp8.fp8_block_quant_linear is not fp8.fp8_fbgemm_block_linear:
        pytest.skip("needs fbgemm f8f8bf16_blockwise")


def _block_quantize_weight(W, block):
    n, k = W.shape
    p, q = math.ceil(n / block[0]), math.ceil(k / block[1])
    scale = torch.empty(p, q, device = W.device, dtype = torch.float32)
    Wq = torch.empty(n, k, device = W.device, dtype = torch.float8_e4m3fn)
    for i in range(p):
        for j in range(q):
            blk = W[i * block[0] : (i + 1) * block[0], j * block[1] : (j + 1) * block[1]].float()
            s = blk.abs().amax() / 448.0
            s = torch.tensor(1.0, device = W.device) if s == 0 else s
            scale[i, j] = s
            Wq[i * block[0] : (i + 1) * block[0], j * block[1] : (j + 1) * block[1]] = (blk / s).to(
                torch.float8_e4m3fn
            )
    return Wq, scale


def _dequant(Wq, scale, block):
    n, k = Wq.shape
    s = scale.repeat_interleave(block[0], 0)[:n].repeat_interleave(block[1], 1)[:, :k]
    return Wq.to(torch.float32) * s


def _reference(X, Wq, scale, block):
    return (X.float() @ _dequant(Wq, scale, block).T).to(X.dtype)


def _bf16_atol(ref, floor = 5e-2):
    """atol scales with max|ref| in bf16 ULPs: error is K-term cancellation, not element size."""
    return max(floor, torch.finfo(torch.bfloat16).eps * ref.abs().max().item())


def _check_grad(X, out, Wq, scale, block):
    # grad_output is all-ones, so grad_X is the row-sum of the dequantized weight.
    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    grad_ref = torch.ones(out.shape, device = out.device, dtype = torch.float32) @ _dequant(
        Wq, scale, block
    )
    torch.testing.assert_close(X.grad.float(), grad_ref, atol = _bf16_atol(grad_ref), rtol = 5e-2)


def _rel_err(out, ref):
    out, ref = out.detach().float(), ref.detach().float()
    return float((out - ref).abs().mean() / ref.abs().mean())


def test_output_tile_grid_battery_matches_reference():
    skip_without_fbgemm()
    # On fbgemm <= 1.3.0 the bad shapes hit ~0.7 rel error; healthy quant noise is ~0.04.
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    for M, N, K in [
        (256, 512, 384),
        (512, 1024, 4096),
        (640, 128, 256),
        (128, 128, 128),
        (256, 256, 512),
        # ragged tails the kernel does support: any M, N % 8, K % 16
        (100, 136, 272),
        (64, 8, 16),
    ]:
        W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
        Wq, scale = _block_quantize_weight(W, block)
        scale.block_size = block
        X = torch.randn(M, K, device = dev, dtype = torch.bfloat16)

        out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
        ref = _reference(X, Wq, scale, block)
        rel = _rel_err(out, ref)
        assert rel < 0.10, f"({M},{N},{K}) rel_err={rel:.4f}"


def test_odd_k_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    N, K = 320, 130  # K % 16 != 0
    W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(4, K, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    assert torch.isfinite(out).all()

    ref = _reference(X.detach(), Wq, scale, block)
    torch.testing.assert_close(out, ref, atol = _bf16_atol(ref), rtol = 5e-2)

    _check_grad(X, out, Wq, scale, block)


def test_odd_n_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    N, K = 250, 256  # N % 8 != 0
    W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(4, K, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    ref = _reference(X.detach(), Wq, scale, block)
    torch.testing.assert_close(out, ref, atol = _bf16_atol(ref), rtol = 5e-2)

    _check_grad(X, out, Wq, scale, block)


def test_non_square_block_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 64]  # kernel only implements 128x128x128
    N, K = 256, 256
    W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(64, K, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    ref = _reference(X.detach(), Wq, scale, block)
    rel = _rel_err(out, ref)
    assert rel < 0.10, f"rel_err={rel:.4f}"

    _check_grad(X, out, Wq, scale, block)


@pytest.mark.parametrize("kind", ["per_tensor", "per_tensor_2d", "bf16_scale", "strided_3d"])
def test_inputs_the_kernel_rejects_use_dequant_fallback(kind):
    # No block grid (0-dim or (1, 1)), a non-float32 scale, or a strided view must not reach
    # f8f8bf16_blockwise.
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    # strided_3d needs a shape the kernel rejects too, else it stays on the fast path
    N, K = (250, 130) if kind == "strided_3d" else (256, 256)
    W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    X = torch.randn(8, K, device = dev, dtype = torch.bfloat16)

    if kind.startswith("per_tensor"):
        scale = scale.amax().clone()
        if kind == "per_tensor_2d":
            scale = scale.reshape(1, 1)
        ref = (X.float() @ (Wq.to(torch.float32) * scale).T).to(X.dtype)
    else:
        if kind == "bf16_scale":
            scale = scale.to(torch.bfloat16)
        else:
            X = torch.randn(2, 4, K * 2, device = dev, dtype = torch.bfloat16)[..., ::2]
        scale.block_size = block
        ref = _reference(X, Wq, scale, block)

    out = FP8_fbgemm_block_linear.apply(X.requires_grad_(True), Wq, scale)
    assert out.shape == (*X.shape[:-1], N) and out.dtype == X.dtype
    assert _rel_err(out, ref) < 0.10
    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


@pytest.mark.parametrize("block", [[128, 64], [64, 128]])  # square stays on the kernel
@pytest.mark.parametrize("N,K", [(256, 512), (256, 256)])
def test_transposed_weight_swaps_block_axes(block, N, K):
    # fast_lora's backward passes downW.t(), swapping block axes; at N == K both grids validate.
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    W = torch.randn(N, K, device = dev, dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    Wt = Wq.t()
    Wt.block_size = block

    dY = torch.randn(8, N, device = dev, dtype = torch.bfloat16)
    out = FP8_fbgemm_block_linear.apply(dY, Wt, scale)
    ref = (dY.float() @ _dequant(Wq, scale, block)).to(dY.dtype)
    assert out.shape == (8, K)
    assert _rel_err(out, ref) < 0.10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
