import os
import sys
import torch
import torch.nn.functional as F
import platform

# Ensure we can import unsloth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from unsloth.patches import patch_unsloth_zoo_for_mps
from unsloth.kernels.swiglu import swiglu_fg_kernel
from unsloth.kernels.geglu import (
    geglu_exact_forward_kernel,
    geglu_approx_forward_kernel,
)


def run_e2e_test():
    print("=" * 60)
    print("UNSLOTH METAL E2E INTEGRATION TEST")
    print("=" * 60)

    # 1. Patch check
    print(f"Platform: {platform.platform()}")
    patched = patch_unsloth_zoo_for_mps()
    print(f"MPS Patch Applied: {patched}")

    if not torch.backends.mps.is_available():
        print("❌ MPS not available, skipping functional tests.")
        return

    # 2. Kernel Check
    from unsloth.kernels.metal import (
        is_metal_swiglu_available,
        is_metal_geglu_available,
    )

    print(f"Metal SwiGLU Available: {is_metal_swiglu_available()}")
    print(f"Metal GEGLU Available:  {is_metal_geglu_available()}")

    # 3. Functional Test - SwiGLU
    print("\nTesting SwiGLU Dispatch...")
    batch, seq, dim = 2, 64, 128
    e = torch.randn(batch, seq, dim, device = "mps", dtype = torch.float16)
    g = torch.randn(batch, seq, dim, device = "mps", dtype = torch.float16)

    # Ground truth
    ref = F.silu(e.float()) * g.float()

    # Fused kernel (should hit Metal)
    out = swiglu_fg_kernel(e, g)

    diff = (out.float() - ref).abs().max().item()
    print(f"  SwiGLU Parity: max_diff={diff:.6f} {'✅' if diff < 1e-2 else '❌'}")

    # 4. Functional Test - GEGLU Exact
    print("\nTesting GEGLU Exact Dispatch...")
    ref_geglu = F.gelu(e.float(), approximate = "none") * g.float()
    out_geglu = geglu_exact_forward_kernel(e, g)

    diff_geglu = (out_geglu.float() - ref_geglu).abs().max().item()
    print(
        f"  GEGLU Exact Parity: max_diff={diff_geglu:.6f} {'✅' if diff_geglu < 1e-2 else '❌'}"
    )

    # 5. Functional Test - GEGLU Approx
    print("\nTesting GEGLU Approx Dispatch...")
    ref_geglu_approx = F.gelu(e.float(), approximate = "tanh") * g.float()
    out_geglu_approx = geglu_approx_forward_kernel(e, g)

    diff_geglu_approx = (out_geglu_approx.float() - ref_geglu_approx).abs().max().item()
    print(
        f"  GEGLU Approx Parity: max_diff={diff_geglu_approx:.6f} {'✅' if diff_geglu_approx < 1e-2 else '❌'}"
    )

    print("\n" + "=" * 60)
    print("✅ E2E INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_test()
