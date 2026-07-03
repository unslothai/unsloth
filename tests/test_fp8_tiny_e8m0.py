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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


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
    dev = "cuda"
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

    dev = "cuda"
    m, n = 8, 8
    weight = torch.randn(m, n, device = dev, dtype = torch.bfloat16)
    scale = (torch.rand(1, 1, device = dev) + 1.0).to(torch.float8_e8m0fnu)
    X = torch.randn(4, n, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = FP8BlockQuantLinear.apply(X, weight, scale)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(X.grad).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
