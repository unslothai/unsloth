# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Patched FP8Linear forward must honour the module's block_size, not assume 128x128."""

import pytest
import torch

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(
    not ((cuda_available and torch.cuda.get_device_capability() >= (8, 9)) or xpu_available),
    reason = "block-FP8 kernels need CUDA sm89+ or XPU",
)


def _make_layer(out_features, in_features, block, scale_fmt):
    from unsloth.kernels import fp8

    if fp8.FP8Linear is None:
        pytest.skip("this transformers has no FP8Linear")
    if scale_fmt == "ue8m0" and not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("this torch has no float8_e8m0fnu")
    # transformers <= 5.5 FP8Linear has block_size but no scale_fmt (float scales only).
    extra = {} if scale_fmt == "float" else {"scale_fmt": scale_fmt}
    try:
        layer = fp8.FP8Linear(in_features, out_features, block_size = block, **extra)
    except TypeError:
        pytest.skip("this transformers FP8Linear takes no scale_fmt")
    layer = layer.to(dev)
    torch.manual_seed(0)
    weight = (torch.randn(out_features, in_features, device = dev) * 0.5).to(torch.float8_e4m3fn)
    # Power-of-two scales so float and ue8m0 formats hold the same values.
    grid = (-(-out_features // block[0]), -(-in_features // block[1]))
    scale = torch.exp2(torch.randint(-9, -4, grid, device = dev).float())
    layer.weight = torch.nn.Parameter(weight, requires_grad = False)
    layer.weight_scale_inv = torch.nn.Parameter(
        scale.to(layer.weight_scale_inv.dtype), requires_grad = False
    )
    assert getattr(layer.weight, "block_size", None) is None
    assert getattr(layer.weight_scale_inv, "block_size", None) is None
    return layer, weight, scale


def _reference(X, weight, scale, block):
    out_features, in_features = weight.shape
    s = scale.repeat_interleave(block[0], 0)[:out_features]
    s = s.repeat_interleave(block[1], 1)[:, :in_features]
    W = weight.float() * s
    return X.float() @ W.T, W


@pytest.mark.parametrize("scale_fmt", ["float", "ue8m0"])
@pytest.mark.parametrize("block", [(32, 32), (128, 128)], ids = ["32x32", "128x128"])
def test_fp8linear_uses_module_block_size(block, scale_fmt):
    out_features, in_features = 512, 1024
    layer, weight, scale = _make_layer(out_features, in_features, block, scale_fmt)
    torch.manual_seed(1)
    X = torch.randn(3, 64, in_features, device = dev, dtype = torch.bfloat16, requires_grad = True)

    out = layer(X)
    assert out.shape == (3, 64, out_features) and out.dtype == torch.bfloat16
    ref, W = _reference(X.detach(), weight, scale, block)
    # Tolerates fp8 activation rounding; a wrong block size errors at order 1.
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.05, rel.item()

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    grad_ref = grad_out.float() @ W
    rel_grad = (X.grad.float() - grad_ref).norm() / grad_ref.norm()
    assert rel_grad < 0.02, rel_grad.item()
