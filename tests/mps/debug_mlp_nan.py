import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np
from unsloth.kernels.mps.fast_lora import mps_apply_lora_mlp_swiglu
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_context, mlx_to_torch


def test_mlp_correctness():
    print("Testing MLP Correctness with Scaled Initialization...")

    # Dimensions
    b, s, h, i = 2, 128, 1024, 4096

    # Proper Initialization to avoid FP16 overflow
    # Inputs: N(0, 1)
    X = torch.randn(b, s, h, device = "mps", dtype = torch.float16)

    # Weights: Scaled to keep variance constant (He/Xavier-ish)
    # std = 1 / sqrt(in_features)
    upW = torch.randn(i, h, device = "mps", dtype = torch.float16) * (1 / (h**0.5))
    gateW = torch.randn(i, h, device = "mps", dtype = torch.float16) * (1 / (h**0.5))
    downW = torch.randn(h, i, device = "mps", dtype = torch.float16) * (1 / (i**0.5))

    # 1. PyTorch Reference
    print("Running PyTorch Reference...")
    up = F.linear(X, upW)
    gate = F.linear(X, gateW)
    act = F.silu(gate) * up
    ref = F.linear(act, downW)

    if torch.isnan(ref).any():
        print("❌ PyTorch Reference produced NaNs!")
        return

    # 2. Unsloth Fused
    print("Running Unsloth Fused...")
    # Manually attach cache to trigger fast path
    X_chained = X.clone()
    with mlx_context():
        X_chained._mlx_cache = torch_to_mlx(X)

    out_unsloth = mps_apply_lora_mlp_swiglu(
        X_chained,
        gateW,
        None,
        None,
        None,
        1.0,
        upW,
        None,
        None,
        None,
        1.0,
        downW,
        None,
        None,
        None,
        1.0,
    )

    # Convert back to check
    # out_unsloth is a torch tensor (returned from mps_apply_lora...)

    if torch.isnan(out_unsloth).any():
        print("❌ Unsloth Fused produced NaNs!")
        # Debug intermediates?
        return

    diff = (out_unsloth - ref).abs().max()
    print(f"Max Difference: {diff.item():.6f}")

    if diff < 1e-2:
        print("✅ Correctness Passed (Diff < 0.01)")
    else:
        print("❌ Correctness Failed (Diff too large)")


if __name__ == "__main__":
    test_mlp_correctness()
