"""FP8 block-quant linear must handle tiny / non-tileable weights and e8m0 scales.

Two things break the triton block path:
  * a hidden dim not divisible by the activation block size (tiny test models),
  * float8_e8m0fnu weight scales, which have no triton dtype mapping.
The forward falls back to a torch-native blockwise dequant + bf16 matmul; this
test checks that fallback runs finite forward + backward and matches a plain
dequant reference.
"""

import pytest
import torch

gpu_available = torch.cuda.is_available() or torch.xpu.is_available()
dev = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"

pytestmark = pytest.mark.skipif(not gpu_available, reason = "needs CUDA or XPU")


def _reference(X, weight, scale, block):
    # Expand the per-block scale to full weight shape and dequantize.
    m, n = weight.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(block[0], 0)[:m].repeat_interleave(block[1], 1)[:, :n]
    W = (weight.to(torch.float32) * s).to(X.dtype)
    return X @ W.T


def test_tiny_non_tileable_forward_backward_matches_reference():
    from unsloth.kernels.fp8 import FP8BlockQuantLinear

    torch.manual_seed(0)
    block = [128, 128]
    m, n = 8, 8  # non-tileable, in-dim % 128 != 0
    weight = torch.randn(m, n, device = dev, dtype = torch.bfloat16)  # (out=m, in=n)
    scale = torch.rand(1, 1, device = dev, dtype = torch.float32) + 0.5
    X = torch.randn(4, n, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8BlockQuantLinear.apply(X, weight, scale)
    assert torch.isfinite(out).all(), "forward produced non-finite values"

    ref = _reference(X.detach(), weight, scale, block)
    torch.testing.assert_close(out, ref, atol = 5e-2, rtol = 5e-2)

    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all(), "backward non-finite"


def test_e8m0_scale_is_upcast_and_runs():
    from unsloth.kernels.fp8 import FP8BlockQuantLinear

    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build lacks float8_e8m0fnu")

    m, n = 8, 8
    weight = torch.randn(m, n, device = dev, dtype = torch.bfloat16)
    scale = (torch.rand(1, 1, device = dev) + 1.0).to(torch.float8_e8m0fnu)
    X = torch.randn(4, n, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8BlockQuantLinear.apply(X, weight, scale)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(X.grad).all()


def test_rectangular_block_dequant_matches_reference():
    # Rectangular blocks (block_size[0] != block_size[1]) that tile evenly used to
    # route through the triton weight_dequant kernel, which uses a single BLOCK_SIZE
    # for both axes and mis-indexes the column scale. Verify the torch expansion path
    # now matches the reference for a 64x256 weight with block [64, 128] (scale 1x2).
    from unsloth.kernels.fp8 import _blockwise_weight_dequant_any_shape

    torch.manual_seed(0)
    block = [64, 128]
    m, n = 64, 256  # evenly tiled: 64 % 64 == 0, 256 % 128 == 0
    weight = torch.randn(m, n, device = dev, dtype = torch.bfloat16)
    # Distinct per-block column scales expose column mis-indexing.
    scale = torch.tensor([[0.5, 3.0]], device = dev, dtype = torch.float32)

    W_deq = _blockwise_weight_dequant_any_shape(weight, scale, block, torch.bfloat16)

    s = scale.repeat_interleave(block[0], 0)[:m].repeat_interleave(block[1], 1)[:, :n]
    ref = (weight.to(torch.float32) * s).to(torch.bfloat16)
    torch.testing.assert_close(W_deq, ref, atol = 5e-3, rtol = 5e-3)


def test_e8m0_scale_preserves_non_default_block_size_attr():
    # An e8m0 scale carrying a non-default block_size attribute must keep it across
    # the float32 upcast in forward; otherwise the lookup falls back to [128, 128]
    # and a compatible layout is wrongly rejected as incompatible.
    from unsloth.kernels.fp8 import FP8BlockQuantLinear

    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build lacks float8_e8m0fnu")

    torch.manual_seed(0)
    block = [64, 64]
    # in-dim 96 is not divisible by block[1]=64 -> forward takes the torch dequant
    # fallback (no fp8 matmul kernel). Scale shape (2, 2) validates for [64, 64] but
    # not [128, 128] (which expects (1, 1)).
    m, n = 128, 96
    weight = torch.randn(m, n, device = dev, dtype = torch.bfloat16)  # no block_size attr
    scale_f = torch.rand(2, 2, device = dev) + 1.0
    scale = scale_f.to(torch.float8_e8m0fnu)
    scale.block_size = block  # attribute lives on the scale, not the weight
    X = torch.randn(4, n, device = dev, dtype = torch.bfloat16, requires_grad = True)

    # With [128, 128] this raises "not compatible with block size"; success proves
    # the [64, 64] attribute survived the e8m0 -> float32 upcast.
    out = FP8BlockQuantLinear.apply(X, weight, scale)
    assert torch.isfinite(out).all()

    ref = _reference(X.detach(), weight, scale.to(torch.float32), block)
    torch.testing.assert_close(out, ref, atol = 5e-2, rtol = 5e-2)

    out.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
