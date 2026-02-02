# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx
from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps
from .utils import is_mlx_available


def fast_dequantize(W, W_quant):
    # This is a placeholder. Real implementation needs to match Unsloth's quantization scheme.
    # For now, assuming standard MLX quantization or simple dequant logic if applicable.
    # However, Unsloth usually passes dequantized weights or handles it inside Custom Ops.
    # If W_quant is None, W is already dequantized.
    return W


def matmul_lora(X, W, W_quant, A, B, S):
    # W is (Out, In) or (In, Out) depending on transpose.
    # Unsloth usually keeps weights transposed (In, Out) for Linear.
    # Let's assume standard behavior: X @ W.T + X @ A @ B * S

    # In MLX, we might need to be careful with shapes.
    # X: (Batch, Seq, Dim)

    # 1. Base Proj
    # We really want to use QuantizedLinear if W_quant is present.
    # For now, let's assume W is the weight matrix.
    out = X @ W.T

    # 2. LoRA
    # A: (Rank, In), B: (Out, Rank)? Or (In, Rank), (Rank, Out)?
    # Unsloth: A (Rank, In), B (Out, Rank).
    # dC/dW += dC/dY @ X.T
    # Y = X @ W.T

    # Lora: X @ A.T @ B.T * S
    lora_out = (X @ A.T) @ B.T
    return out + lora_out * S


@mx.compile
def _compiled_mlp_swiglu(
    X, gateW, gateA, gateB, gateS, upW, upA, upB, upS, downW, downA, downB, downS
):
    # X: (Batch, Seq, Dim)

    # Gate
    # gateW: (Dim, Hidden) - Wait, Unsloth weights are usually (Hidden, Dim) for Linear?
    # Let's verify shapes from calling code.
    # Typically Unsloth uses Linear(in, out) -> weight (out, in).
    # So X @ W.T

    # We will assume inputs are already MLX arrays and properly transposed if needed?
    # Or strict linear layers.

    # gate_proj = X @ gateW.T + (X @ gateA.T) @ gateB.T * gateS
    gate = X @ gateW.T + (X @ gateA.T) @ gateB.T * gateS

    # up_proj = X @ upW.T + (X @ upA.T) @ upB.T * upS
    up = X @ upW.T + (X @ upA.T) @ upB.T * upS

    # SwiGLU: (gate * sigmoid(gate)) * up
    # MLX has mx.sigmoid
    # Silu is x * sigmoid(x)

    # Unsloth SwiGLU is likely (gate * sigmoid(gate)) * up  OR  (gate * silu(gate)) * up?
    # Standard: SwiGLU(x) = Swish(xW_g) * (xW_u)
    # Swish(x) = x * sigmoid(x)

    act = gate * mx.sigmoid(gate) * up

    # down_proj = act @ downW.T + (act @ downA.T) @ downB.T * downS
    out = act @ downW.T + (act @ downA.T) @ downB.T * downS

    return out


def apply_lora_mlp_swiglu(
    X,
    gateW,
    gateW_quant,
    gateA,
    gateB,
    gateS,
    upW,
    upW_quant,
    upA,
    upB,
    upS,
    downW,
    downW_quant,
    downA,
    downB,
    downS,
):
    synchronize_mps()

    # Convert inputs to MLX
    # X is likely (Batch, Seq, Dim) or (Batch*Seq, Dim)
    # Unsloth flattens to 2D for some kernels. MLX prefers 3D?
    # Let's stick to whatever shape X comes in, assuming matmul handles it.

    X_mlx = torch_to_mlx(X)

    # Weights. Assuming they are float16 or bfloat16 dequantized, OR handled.
    # If Quantized, we need special handling. For this iteration, let's assume dequantized for simplicity
    # or rely on bridge to handle `gateW` if it's a tensor.
    # Note: `gateW_quant` usage is complex to map 1:1 if we use `mx.compile` on raw weight buffers.
    # Ideally, we dequantize BEFORE calling this or inside if supported.
    # Unsloth passes `gateW` as main weight.

    gateW_mlx = torch_to_mlx(gateW)
    gateA_mlx = torch_to_mlx(gateA)
    gateB_mlx = torch_to_mlx(gateB)
    # gateS is scalar? or tensor?
    gateS_mlx = torch_to_mlx(gateS) if hasattr(gateS, "shape") else gateS

    upW_mlx = torch_to_mlx(upW)
    upA_mlx = torch_to_mlx(upA)
    upB_mlx = torch_to_mlx(upB)
    upS_mlx = torch_to_mlx(upS) if hasattr(upS, "shape") else upS

    downW_mlx = torch_to_mlx(downW)
    downA_mlx = torch_to_mlx(downA)
    downB_mlx = torch_to_mlx(downB)
    downS_mlx = torch_to_mlx(downS) if hasattr(downS, "shape") else downS

    # Execute compiled kernel
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

    return mlx_to_torch(out_mlx, device = X.device, dtype = X.dtype)


@mx.compile
def _compiled_qkv(X, QW, QA, QB, QS, KW, KA, KB, KS, VW, VA, VB, VS):
    # Q = X @ QW.T + (X @ QA.T) @ QB.T * QS
    Q = X @ QW.T + (X @ QA.T) @ QB.T * QS

    # K = X @ KW.T + (X @ KA.T) @ KB.T * KS
    K = X @ KW.T + (X @ KA.T) @ KB.T * KS

    # V = X @ VW.T + (X @ VA.T) @ VB.T * VS
    V = X @ VW.T + (X @ VA.T) @ VB.T * VS

    return Q, K, V


def apply_lora_qkv(
    X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS
):
    synchronize_mps()

    X_mlx = torch_to_mlx(X)

    QW_mlx = torch_to_mlx(QW)
    QA_mlx = torch_to_mlx(QA)
    QB_mlx = torch_to_mlx(QB)
    QS_mlx = torch_to_mlx(QS) if hasattr(QS, "shape") else QS

    KW_mlx = torch_to_mlx(KW)
    KA_mlx = torch_to_mlx(KA)
    KB_mlx = torch_to_mlx(KB)
    KS_mlx = torch_to_mlx(KS) if hasattr(KS, "shape") else KS

    VW_mlx = torch_to_mlx(VW)
    VA_mlx = torch_to_mlx(VA)
    VB_mlx = torch_to_mlx(VB)
    VS_mlx = torch_to_mlx(VS) if hasattr(VS, "shape") else VS

    Q_mlx, K_mlx, V_mlx = _compiled_qkv(
        X_mlx,
        QW_mlx,
        QA_mlx,
        QB_mlx,
        QS_mlx,
        KW_mlx,
        KA_mlx,
        KB_mlx,
        KS_mlx,
        VW_mlx,
        VA_mlx,
        VB_mlx,
        VS_mlx,
    )

    mx.eval(Q_mlx, K_mlx, V_mlx)

    Q = mlx_to_torch(Q_mlx, device = X.device, dtype = X.dtype)
    K = mlx_to_torch(K_mlx, device = X.device, dtype = X.dtype)
    V = mlx_to_torch(V_mlx, device = X.device, dtype = X.dtype)

    return Q, K, V


@mx.compile
def _compiled_o(X, OW, OA, OB, OS):
    # O = X @ OW.T + (X @ OA.T) @ OB.T * OS
    return X @ OW.T + (X @ OA.T) @ OB.T * OS


def apply_lora_o(X, OW, OW_quant, OA, OB, OS):
    synchronize_mps()

    X_mlx = torch_to_mlx(X)
    OW_mlx = torch_to_mlx(OW)
    OA_mlx = torch_to_mlx(OA)
    OB_mlx = torch_to_mlx(OB)
    OS_mlx = torch_to_mlx(OS) if hasattr(OS, "shape") else OS

    out_mlx = _compiled_o(X_mlx, OW_mlx, OA_mlx, OB_mlx, OS_mlx)
    mx.eval(out_mlx)

    return mlx_to_torch(out_mlx, device = X.device, dtype = X.dtype)
