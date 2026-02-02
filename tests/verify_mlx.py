import torch
import torch.nn as nn
import time

try:
    import mlx.core as mx

    print("MLX is available.")
except ImportError:
    print("MLX is NOT available. This script requires MLX.")
    exit(0)

from unsloth.kernels.mlx.fast_lora import (
    apply_lora_mlp_swiglu,
    apply_lora_qkv,
    apply_lora_o,
)


def test_mlp():
    print("\n=== Testing LoRA MLP (SwiGLU) ===")
    B, S, D = 1, 128, 4096
    HD = 11008  # Llama-2-7b size approx

    X = torch.randn(B, S, D, device = "cpu", dtype = torch.float16)

    gateW = torch.randn(HD, D, device = "cpu", dtype = torch.float16)
    gateA = torch.randn(8, D, device = "cpu", dtype = torch.float16)
    gateB = torch.randn(HD, 8, device = "cpu", dtype = torch.float16)
    gateS = 1.0

    upW = torch.randn(HD, D, device = "cpu", dtype = torch.float16)
    upA = torch.randn(8, D, device = "cpu", dtype = torch.float16)
    upB = torch.randn(HD, 8, device = "cpu", dtype = torch.float16)
    upS = 1.0

    downW = torch.randn(D, HD, device = "cpu", dtype = torch.float16)
    downA = torch.randn(8, HD, device = "cpu", dtype = torch.float16)
    downB = torch.randn(D, 8, device = "cpu", dtype = torch.float16)
    downS = 1.0

    print("Running MLX MLP...")
    start = time.time()
    # First run triggers compilation
    out = apply_lora_mlp_swiglu(
        X,
        gateW,
        None,
        gateA,
        gateB,
        gateS,
        upW,
        None,
        upA,
        upB,
        upS,
        downW,
        None,
        downA,
        downB,
        downS,
    )
    mx.eval(
        out
    )  # Ensure eval if returning mlx array (though wrapper converts to torch)
    print(f"First run (Compile): {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(5):
        out = apply_lora_mlp_swiglu(
            X,
            gateW,
            None,
            gateA,
            gateB,
            gateS,
            upW,
            None,
            upA,
            upB,
            upS,
            downW,
            None,
            downA,
            downB,
            downS,
        )
    print(f"5 loops (Cached): {time.time() - start:.4f}s")
    print(f"Output shape: {out.shape}")
    print("✅ MLP Test Passed")


def test_qkv():
    print("\n=== Testing LoRA QKV ===")
    B, S, D = 1, 128, 4096

    X = torch.randn(B, S, D, device = "cpu", dtype = torch.float16)

    # Q
    QW = torch.randn(D, D, device = "cpu", dtype = torch.float16)
    QA = torch.randn(8, D, device = "cpu", dtype = torch.float16)
    QB = torch.randn(D, 8, device = "cpu", dtype = torch.float16)
    QS = 1.0

    # K
    KW = torch.randn(D, D, device = "cpu", dtype = torch.float16)
    KA = torch.randn(8, D, device = "cpu", dtype = torch.float16)
    KB = torch.randn(D, 8, device = "cpu", dtype = torch.float16)
    KS = 1.0

    # V
    VW = torch.randn(D, D, device = "cpu", dtype = torch.float16)
    VA = torch.randn(8, D, device = "cpu", dtype = torch.float16)
    VB = torch.randn(D, 8, device = "cpu", dtype = torch.float16)
    VS = 1.0

    print("Running MLX QKV...")
    start = time.time()
    q, k, v = apply_lora_qkv(
        X, QW, None, QA, QB, QS, KW, None, KA, KB, KS, VW, None, VA, VB, VS
    )
    print(f"First run (Compile): {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(5):
        q, k, v = apply_lora_qkv(
            X, QW, None, QA, QB, QS, KW, None, KA, KB, KS, VW, None, VA, VB, VS
        )
    print(f"5 loops (Cached): {time.time() - start:.4f}s")
    print(f"Q Shape: {q.shape}")
    print("✅ QKV Test Passed")


if __name__ == "__main__":
    print("Starting MLX Verification...")
    test_mlp()
    test_qkv()
    print("\nAll tests passed successfully.")
