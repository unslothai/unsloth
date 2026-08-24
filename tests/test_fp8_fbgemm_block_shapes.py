"""FP8 fbgemm blockwise linear must stay correct across output tile grids and shapes.

Three things this guards:
  * fbgemm <=1.3.0 silently corrupted whole outputs for some output tile grids
    (raster-order bug in its vendored CUTLASS scheduler). The import-time probe
    uses a 128^3 all-ones GEMM whose 1x1 tile grid can never trigger that class
    of bug, so a regression would load fine and corrupt training. The shape
    battery here covers both scheduler cluster buckets' former failure zones.
  * f8f8bf16_blockwise only supports 128x128x128 blocks with
    in_features % 16 == 0 and out_features % 8 == 0 (measured on H800 /
    fbgemm 1.4.0); anything else crashed the kernel ("cutlass cannot
    implement" / "Only 128x128x128 block size is supported") instead of
    falling back to dequant + matmul.
  * activations are (tokens, K): they quantize with block width bs_k, not bs_n.
"""

import math

import pytest
import torch

cuda_available = torch.cuda.is_available()


# Only the kernel battery needs fbgemm; the fallback tests below never reach it.
pytestmark = pytest.mark.skipif(not cuda_available, reason = "needs CUDA")


def skip_without_fbgemm():
    # Ask unsloth's own import-time probe rather than checking for sm_90: that is
    # exactly when the kernel path is reachable, and future arches enable themselves.
    # Called inside the test so collection never imports unsloth.
    from unsloth.kernels import fp8
    if fp8.fp8_block_quant_linear is not fp8.fp8_fbgemm_block_linear:
        pytest.skip("needs fbgemm f8f8bf16_blockwise")


def _block_quantize_weight(W, block):
    # Per (block[0], block[1])-block absmax quantization to float8_e4m3fn.
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


def _check_grad(X, out, Wq, scale, block):
    # out.sum() makes grad_output all-ones, so grad_X is the row-sum of the
    # dequantized weight; the old backward dequantized with a hardcoded
    # 128x128 block and produced finite but mis-scaled gradients here.
    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    grad_ref = torch.ones(out.shape, device = out.device, dtype = torch.float32) @ _dequant(
        Wq, scale, block
    )
    torch.testing.assert_close(X.grad.float(), grad_ref, atol = 5e-2, rtol = 5e-2)


def _rel_err(out, ref):
    out, ref = out.detach().float(), ref.detach().float()
    return float((out - ref).abs().mean() / ref.abs().mean())


def test_output_tile_grid_battery_matches_reference():
    skip_without_fbgemm()
    # Shapes cover both dispatch buckets' former failure zones plus safe ones.
    # On fbgemm <=1.3.0 the failing ones returned whole outputs at ~0.7 relative
    # error; the healthy bound (activation quant noise) sits around 0.04.
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
        W = torch.randn(N, K, device = "cuda", dtype = torch.bfloat16)
        Wq, scale = _block_quantize_weight(W, block)
        scale.block_size = block
        X = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)

        out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
        ref = _reference(X, Wq, scale, block)
        rel = _rel_err(out, ref)
        assert rel < 0.10, f"({M},{N},{K}) rel_err={rel:.4f}"


def test_odd_k_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    N, K = 320, 130  # K % 16 != 0 used to crash inside the CUTLASS kernel
    W = torch.randn(N, K, device = "cuda", dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(4, K, device = "cuda", dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    assert torch.isfinite(out).all()

    ref = _reference(X.detach(), Wq, scale, block)
    torch.testing.assert_close(out, ref, atol = 5e-2, rtol = 5e-2)

    _check_grad(X, out, Wq, scale, block)


def test_odd_n_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    N, K = 250, 256  # N % 8 != 0 used to crash inside the CUTLASS kernel
    W = torch.randn(N, K, device = "cuda", dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(4, K, device = "cuda", dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    ref = _reference(X.detach(), Wq, scale, block)
    torch.testing.assert_close(out, ref, atol = 5e-2, rtol = 5e-2)

    _check_grad(X, out, Wq, scale, block)


def test_non_square_block_uses_dequant_fallback():
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 64]  # kernel only implements 128x128x128, used to crash
    N, K = 256, 256
    W = torch.randn(N, K, device = "cuda", dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    scale.block_size = block
    X = torch.randn(64, K, device = "cuda", dtype = torch.bfloat16, requires_grad = True)

    out = FP8_fbgemm_block_linear.apply(X, Wq, scale)
    ref = _reference(X.detach(), Wq, scale, block)
    rel = _rel_err(out, ref)
    assert rel < 0.10, f"rel_err={rel:.4f}"

    _check_grad(X, out, Wq, scale, block)


@pytest.mark.parametrize("kind", ["per_tensor", "bf16_scale", "strided_3d"])
def test_inputs_the_kernel_rejects_use_dequant_fallback(kind):
    # Each of these used to reach f8f8bf16_blockwise and raise: a per-tensor scale
    # has no block grid to unpack, the kernel demands float32 scales, and a strided
    # view cannot be .view()ed flat.
    from unsloth.kernels.fp8 import FP8_fbgemm_block_linear

    torch.manual_seed(0)
    block = [128, 128]
    # strided_3d needs a shape the kernel rejects too, else it stays on the fast path
    N, K = (250, 130) if kind == "strided_3d" else (256, 256)
    W = torch.randn(N, K, device = "cuda", dtype = torch.bfloat16)
    Wq, scale = _block_quantize_weight(W, block)
    X = torch.randn(8, K, device = "cuda", dtype = torch.bfloat16)

    if kind == "per_tensor":
        scale = scale.amax().clone()
        ref = (X.float() @ (Wq.to(torch.float32) * scale).T).to(X.dtype)
    else:
        if kind == "bf16_scale":
            scale = scale.to(torch.bfloat16)
        else:
            X = torch.randn(2, 4, K * 2, device = "cuda", dtype = torch.bfloat16)[..., ::2]
        scale.block_size = block
        ref = _reference(X, Wq, scale, block)

    out = FP8_fbgemm_block_linear.apply(X.requires_grad_(True), Wq, scale)
    assert out.shape == (*X.shape[:-1], N) and out.dtype == X.dtype
    assert _rel_err(out, ref) < 0.10
    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
