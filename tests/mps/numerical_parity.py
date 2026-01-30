import torch
import torch.nn.functional as F
import os
import sys
import importlib.util


# 1. Standalone Loading
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load kernels directly
fast_lora_path = os.path.join(ROOT, "unsloth/kernels/mps/fast_lora.py")
fast_lora_mod = load_module_from_path("unsloth.kernels.mps.fast_lora", fast_lora_path)
mps_matmul_lora = fast_lora_mod.mps_matmul_lora
mps_apply_lora_mlp_swiglu = fast_lora_mod.mps_apply_lora_mlp_swiglu


def test_matmul_lora_parity():
    print("Testing Matmul LoRA Parity...")
    X = torch.randn(2, 16, 32)
    W = torch.randn(64, 32)
    A = torch.randn(8, 32)
    B = torch.randn(64, 8)
    s = 2.0

    # Reference
    Y_ref = torch.matmul(X, W.t()) + torch.matmul(torch.matmul(X, A.t()), B.t()) * s

    # Implementation
    Y_impl = mps_matmul_lora(X, W, None, A, B, s)

    assert torch.allclose(Y_ref, Y_impl, atol = 1e-5)
    print("✅ Matmul LoRA Passed")


def test_lora_mlp_parity():
    print("Testing LoRA MLP SwiGLU Parity...")
    X = torch.randn(2, 4, 8)
    gateW, upW, downW = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    gateA, gateB = torch.randn(4, 8), torch.randn(16, 4)
    upA, upB = torch.randn(4, 8), torch.randn(16, 4)
    downA, downB = torch.randn(4, 16), torch.randn(8, 4)
    s = 1.5

    # Reference (Manual composition)
    e = (
        torch.matmul(X, gateW.t())
        + torch.matmul(torch.matmul(X, gateA.t()), gateB.t()) * s
    )
    g = torch.matmul(X, upW.t()) + torch.matmul(torch.matmul(X, upA.t()), upB.t()) * s
    h = F.silu(e) * g
    Y_ref = (
        torch.matmul(h, downW.t())
        + torch.matmul(torch.matmul(h, downA.t()), downB.t()) * s
    )

    # Implementation
    Y_impl = mps_apply_lora_mlp_swiglu(
        X,
        gateW,
        None,
        gateA,
        gateB,
        s,
        upW,
        None,
        upA,
        upB,
        s,
        downW,
        None,
        downA,
        downB,
        s,
    )

    assert torch.allclose(Y_ref, Y_impl, atol = 1e-5)
    print("✅ LoRA MLP SwiGLU Passed")


if __name__ == "__main__":
    test_matmul_lora_parity()
    test_lora_mlp_parity()
    print("\n🚀 ALL CHUNK 3 KERNELS VERIFIED NUMERICALLY.")
