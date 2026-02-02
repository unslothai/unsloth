import torch
import time
import argparse

try:
    import mlx.core as mx

    print("MLX found. Benchmarking MLX vs PyTorch.")
except ImportError:
    print("MLX not found. This benchmark requires MLX.")
    exit(0)

from unsloth.kernels.mlx.fast_lora import apply_lora_mlp_swiglu, apply_lora_qkv
from unsloth.kernels.mlx.bridge import torch_to_mlx


def benchmark_mlp(batch_size=1, seq_len=128, dim=4096, hidden_dim=11008, iters=100):
    print(f"\n--- Benchmarking MLP [B={batch_size}, S={seq_len}, D={dim}] ---")

    # Setup PyTorch Tensors (CPU for now, would be MPS on Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    X = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)

    # Weights
    gateW = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    gateA = torch.randn(8, dim, device=device, dtype=dtype)
    gateB = torch.randn(hidden_dim, 8, device=device, dtype=dtype)
    gateS = 1.0

    upW = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    upA = torch.randn(8, dim, device=device, dtype=dtype)
    upB = torch.randn(hidden_dim, 8, device=device, dtype=dtype)
    upS = 1.0

    downW = torch.randn(dim, hidden_dim, device=device, dtype=dtype)
    downA = torch.randn(8, hidden_dim, device=device, dtype=dtype)
    downB = torch.randn(dim, 8, device=device, dtype=dtype)
    downS = 1.0

    # --- PyTorch Baseline Implementation (Naive) ---
    def torch_mlp(x):
        # Gate
        gate = torch.nn.functional.linear(x, gateW) + (x @ gateA.T) @ gateB.T * gateS
        # Up
        up = torch.nn.functional.linear(x, upW) + (x @ upA.T) @ upB.T * upS
        # SwiGLU
        act = (
            torch.nn.functional.silu(gate) * up
        )  # Standard SwiGLU is silu(gate) * up vs gate * sigmoid(gate) * up?
        # Unsloth uses: e = e * sigmoid(e) aka swish/silu, then h = f * g.
        # Wait, Unsloth code said: f = e * sigmoid(e). g = X @ U. h = f * g.
        # So it IS SwiGLU.

        # Down
        out = torch.nn.functional.linear(act, downW) + (act @ downA.T) @ downB.T * downS
        return out

    # Warmup PyTorch
    for _ in range(10):
        _ = torch_mlp(X)
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = torch_mlp(X)
    if device.type == "mps":
        torch.mps.synchronize()
    torch_time_ms = (time.time() - start) * 1000 / iters
    print(f"PyTorch (Naive): {torch_time_ms:.3f} ms / iter")

    # --- MLX Implementation ---
    # Warmup
    out_mlx = apply_lora_mlp_swiglu(
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
    for _ in range(10):
        apply_lora_mlp_swiglu(
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

    # --- Detailed Profiling ---
    print(f"\n[Profiling Breakdown]")

    # 1. Measure Weight Conversion Cost
    start = time.time()
    for _ in range(iters):
        # Just convert one weight to see simple overhead
        _ = torch_to_mlx(gateW)
    torch.mps.synchronize() if device.type == "mps" else None
    conv_time = (time.time() - start) * 1000 / iters
    print(
        f"Weight Conversion (1 tensor): {conv_time:.3f} ms"
    )  # Multiplied by ~12 for full layer

    # 2. Measure Exec Time with PRE-CONVERTED Weights (Ideal Case)
    X_mlx = torch_to_mlx(X)
    gateW_mlx = torch_to_mlx(gateW)
    gateA_mlx = torch_to_mlx(gateA)
    gateB_mlx = torch_to_mlx(gateB)
    gateS_mlx = mx.array(1.0)
    upW_mlx = torch_to_mlx(upW)
    upA_mlx = torch_to_mlx(upA)
    upB_mlx = torch_to_mlx(upB)
    upS_mlx = mx.array(1.0)
    downW_mlx = torch_to_mlx(downW)
    downA_mlx = torch_to_mlx(downA)
    downB_mlx = torch_to_mlx(downB)
    downS_mlx = mx.array(1.0)

    from unsloth.kernels.mlx.fast_lora import _compiled_mlp_swiglu

    # Warmup compiled kernel
    out_mlx = _compiled_mlp_swiglu(
        X_mlx,
        gateW_mlx,
        gateA_mlx,
        gateB_mlx,
        gateS_mlx,
        upW_mlx,
        upA_mlx,
        upB_mlx,
        upS_mlx,
        downW_mlx,
        downA_mlx,
        downB_mlx,
        downS_mlx,
    )
    mx.eval(out_mlx)

    start = time.time()
    for _ in range(iters):
        out_mlx = _compiled_mlp_swiglu(
            X_mlx,
            gateW_mlx,
            gateA_mlx,
            gateB_mlx,
            gateS_mlx,
            upW_mlx,
            upA_mlx,
            upB_mlx,
            upS_mlx,
            downW_mlx,
            downA_mlx,
            downB_mlx,
            downS_mlx,
        )
        mx.eval(out_mlx)
    ideal_time_ms = (time.time() - start) * 1000 / iters
    print(f"MLX (Cached Weights + Compiled): {ideal_time_ms:.3f} ms / iter")

    if ideal_time_ms > 0:
        print(f"Potential Speedup (Ideal): {torch_time_ms / ideal_time_ms:.2f}x")

    # 3. Full End-to-End (Current Implementation)
    start = time.time()
    for _ in range(iters):
        apply_lora_mlp_swiglu(
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
    # No explicit sync needed as apply_lora does it
    mlx_full_time_ms = (time.time() - start) * 1000 / iters
    print(f"MLX (Full E2E):          {mlx_full_time_ms:.3f} ms / iter")

    # 4. Measure Exec Time with Pre-Transposed Weights (Optimization check)
    gateW_T = gateW_mlx.T
    upW_T = upW_mlx.T
    downW_T = downW_mlx.T
    gateA_T = gateA_mlx.T
    gateB_T = gateB_mlx.T
    upA_T = upA_mlx.T
    upB_T = upB_mlx.T
    downA_T = downA_mlx.T
    downB_T = downB_mlx.T

    # Define a kernel that expects transposed weights
    @mx.compile
    def _compiled_mlp_transposed(
        X,
        gateW_T,
        gateA_T,
        gateB_T,
        gateS,
        upW_T,
        upA_T,
        upB_T,
        upS,
        downW_T,
        downA_T,
        downB_T,
        downS,
    ):
        # gateW_T is (In, Out)
        gate = X @ gateW_T + (X @ gateA_T) @ gateB_T * gateS
        up = X @ upW_T + (X @ upA_T) @ upB_T * upS
        act = gate * mx.sigmoid(gate) * up
        out = act @ downW_T + (act @ downA_T) @ downB_T * downS
        return out

    out_mlx = _compiled_mlp_transposed(
        X_mlx,
        gateW_T,
        gateA_T,
        gateB_T,
        gateS_mlx,
        upW_T,
        upA_T,
        upB_T,
        upS_mlx,
        downW_T,
        downA_T,
        downB_T,
        downS_mlx,
    )
    mx.eval(out_mlx)

    start = time.time()
    for _ in range(iters):
        out_mlx = _compiled_mlp_transposed(
            X_mlx,
            gateW_T,
            gateA_T,
            gateB_T,
            gateS_mlx,
            upW_T,
            upA_T,
            upB_T,
            upS_mlx,
            downW_T,
            downA_T,
            downB_T,
            downS_mlx,
        )
        mx.eval(out_mlx)
    transposed_time_ms = (time.time() - start) * 1000 / iters
    print(f"MLX (Pre-Transposed + Compiled): {transposed_time_ms:.3f} ms / iter")

    # 5. Eager Mode (No Optimization)
    def _eager_mlp(
        X, gateW, gateA, gateB, gateS, upW, upA, upB, upS, downW, downA, downB, downS
    ):
        gate = X @ gateW.T + (X @ gateA.T) @ gateB.T * gateS
        up = X @ upW.T + (X @ upA.T) @ upB.T * upS
        act = gate * mx.sigmoid(gate) * up
        out = act @ downW.T + (act @ downA.T) @ downB.T * downS
        return out

    start = time.time()
    for _ in range(iters):
        out_mlx = _eager_mlp(
            X_mlx,
            gateW_mlx,
            gateA_mlx,
            gateB_mlx,
            gateS_mlx,
            upW_mlx,
            upA_mlx,
            upB_mlx,
            upS_mlx,
            downW_mlx,
            downA_mlx,
            downB_mlx,
            downS_mlx,
        )
        mx.eval(out_mlx)
    eager_time_ms = (time.time() - start) * 1000 / iters
    print(f"MLX (Eager + Cached):            {eager_time_ms:.3f} ms / iter")


if __name__ == "__main__":
    benchmark_mlp(batch_size=1, seq_len=128)  # Original
    benchmark_mlp(batch_size=2, seq_len=1024, iters=50)  # Larger workload
