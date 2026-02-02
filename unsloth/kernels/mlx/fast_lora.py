# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx
from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps, mlx_context
from .utils import is_mlx_available


from .quantization import MLXQuantizedWeight


def treeify(w):
    """
    Converts MLXQuantizedWeight objects into dictionaries of arrays
    that mx.compile can trace. Standard arrays are returned as-is.
    """
    w_mlx = torch_to_mlx(w) if not isinstance(w, mx.array) and not isinstance(w, MLXQuantizedWeight) else w
    if isinstance(w_mlx, MLXQuantizedWeight):
        return {
            "weight": w_mlx.weight,
            "scales": w_mlx.scales,
            "biases": w_mlx.biases,
            "group_size": w_mlx.group_size,
            "bits": w_mlx.bits
        }
    return w_mlx


def quantized_matmul(X, W):
    """
    Handles both standard and quantized MLX weights.
    W can be an mx.array or MLXQuantizedWeight.
    Auto-dispatches to Metal GEMV for FP16 Batch=1.
    """
    if isinstance(W, MLXQuantizedWeight):
        # MLX Quantized Matmul expects group_size in the packed domain
        # i.e. original_group_size / (32 / bits)
        bits = getattr(W, "bits", 4)
        group_size = getattr(W, "group_size", 64)
        packed_group_size = group_size // (32 // bits)
        
        return mx.quantized_matmul(
            X, 
            W.weight, 
            W.scales, 
            W.biases, 
            group_size = packed_group_size,
            bits = bits
        )
    
    # Batch=1 Optimization for FP16
    if X.size // X.shape[-1] == 1:
        from ..metal.gemv import fast_gemv
        return fast_gemv(X.reshape(1, -1), W)

    return X @ W.T


def fast_dequantize(W, W_quant):
    # If W is already a quantized object from our cache, return it
    if isinstance(W, MLXQuantizedWeight):
        return W
    return W


def matmul_lora(X, W, W_quant, A, B, S):
    # 1. Base Proj
    out = quantized_matmul(X, W)

    # 2. LoRA
    # Unsloth: A (Rank, In), B (Out, Rank).
    lora_out = (X @ A.T) @ B.T
    return out + lora_out * S


def _dequant(W):
    if isinstance(W, dict) and "weight" in W:
        return mx.dequantize(
            W["weight"], W["scales"], W["biases"], 
            group_size = W["group_size"], bits = W["bits"]
        )
    return W


@mx.compile
def _compiled_mlp_swiglu(
    X, gateW, gateA, gateB, gateS, upW, upA, upB, upS, downW, downA, downB, downS
):
    # Dequantize weights if they are custom objects
    # Note: mx.compile will trace this.
    gW = _dequant(gateW)
    uW = _dequant(upW)
    dW = _dequant(downW)

    gate = X @ gW.T + (X @ gateA.T) @ gateB.T * gateS
    up = X @ uW.T + (X @ upA.T) @ upB.T * upS

    act = gate * mx.sigmoid(gate) * up
    out = act @ dW.T + (act @ downA.T) @ downB.T * downS

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
    with mlx_context():
        # Convert inputs to MLX
        # X is likely (Batch, Seq, Dim) or (Batch*Seq, Dim)
        # Unsloth flattens to 2D for some kernels. MLX prefers 3D?
        # Let's stick to whatever shape X comes in, assuming matmul handles it.

        X_mlx = torch_to_mlx(X)

        # Batch=1 Optimization: Use custom GEMV for base projections
        # X shape is usually (Batch, Seq, Dim).
        # For decoding, Batch=1, Seq=1.
        if X_mlx.size // X_mlx.shape[-1] == 1:
            # Need weights in MLX
            gateW_mlx = torch_to_mlx(gateW)
            gateA_mlx = torch_to_mlx(gateA)
            gateB_mlx = torch_to_mlx(gateB)
            gateS_val = gateS.item() if hasattr(gateS, "item") else gateS
            
            upW_mlx = torch_to_mlx(upW)
            upA_mlx = torch_to_mlx(upA)
            upB_mlx = torch_to_mlx(upB)
            upS_val = upS.item() if hasattr(upS, "item") else upS
            
            downW_mlx = torch_to_mlx(downW)
            downA_mlx = torch_to_mlx(downA)
            downB_mlx = torch_to_mlx(downB)
            downS_val = downS.item() if hasattr(downS, "item") else downS
            
            # 1. Gate
            gate = quantized_matmul(X_mlx.reshape(1, -1), gateW_mlx)
            gate += (X_mlx.reshape(1, -1) @ gateA_mlx.T) @ gateB_mlx.T * gateS_val
            
            # 2. Up
            up = quantized_matmul(X_mlx.reshape(1, -1), upW_mlx)
            up += (X_mlx.reshape(1, -1) @ upA_mlx.T) @ upB_mlx.T * upS_val
            
            # 3. SwiGLU: silu(gate) * up
            act = (gate * mx.sigmoid(gate)) * up
            
            # 4. Down
            out = quantized_matmul(act, downW_mlx)
            out += (act @ downA_mlx.T) @ downB_mlx.T * downS_val
            
            return mlx_to_torch(out.reshape(X.shape), device = X.device, dtype = X.dtype)

        # Standard Batch Path (Compiled)
        gateW_mlx = treeify(gateW)
        gateA_mlx = torch_to_mlx(gateA)
        gateB_mlx = torch_to_mlx(gateB)
        gateS_mlx = torch_to_mlx(gateS) if hasattr(gateS, "shape") else gateS

        upW_mlx = treeify(upW)
        upA_mlx = torch_to_mlx(upA)
        upB_mlx = torch_to_mlx(upB)
        upS_mlx = torch_to_mlx(upS) if hasattr(upS, "shape") else upS

        downW_mlx = treeify(downW)
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

        return mlx_to_torch(out_mlx, device = X.device, dtype = X.dtype)


@mx.compile
def _compiled_qkv(X, QW, QA, QB, QS, KW, KA, KB, KS, VW, VA, VB, VS):
    Q = X @ _dequant(QW).T + (X @ QA.T) @ QB.T * QS
    K = X @ _dequant(KW).T + (X @ KA.T) @ KB.T * KS
    V = X @ _dequant(VW).T + (X @ VA.T) @ VB.T * VS
    return Q, K, V


def apply_lora_qkv(
    X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS
):
    with mlx_context():
        X_mlx = torch_to_mlx(X)

        # Batch=1 Optimization
        if X_mlx.size // X_mlx.shape[-1] == 1:
            # Inputs
            QW_mlx = torch_to_mlx(QW)
            QA_mlx = torch_to_mlx(QA)
            QB_mlx = torch_to_mlx(QB)
            KW_mlx = torch_to_mlx(KW)
            KA_mlx = torch_to_mlx(KA)
            KB_mlx = torch_to_mlx(KB)
            VW_mlx = torch_to_mlx(VW)
            VA_mlx = torch_to_mlx(VA)
            VB_mlx = torch_to_mlx(VB)

            QS_val = QS.item() if hasattr(QS, "item") else QS
            KS_val = KS.item() if hasattr(KS, "item") else KS
            VS_val = VS.item() if hasattr(VS, "item") else VS

            X_flat = X_mlx.reshape(1, -1)

            Q = (
                quantized_matmul(X_flat, QW_mlx)
                + (X_flat @ QA_mlx.T) @ QB_mlx.T * QS_val
            )
            K = (
                quantized_matmul(X_flat, KW_mlx)
                + (X_flat @ KA_mlx.T) @ KB_mlx.T * KS_val
            )
            V = (
                quantized_matmul(X_flat, VW_mlx)
                + (X_flat @ VA_mlx.T) @ VB_mlx.T * VS_val
            )

            return (
                mlx_to_torch(
                    Q.reshape(X.shape[:-1] + (QW.shape[0],)),
                    device = X.device,
                    dtype = X.dtype,
                ),
                mlx_to_torch(
                    K.reshape(X.shape[:-1] + (KW.shape[0],)),
                    device = X.device,
                    dtype = X.dtype,
                ),
                mlx_to_torch(
                    V.reshape(X.shape[:-1] + (VW.shape[0],)),
                    device = X.device,
                    dtype = X.dtype,
                ),
            )

        QW_mlx = treeify(QW)
        QA_mlx = torch_to_mlx(QA)
        QB_mlx = torch_to_mlx(QB)
        QS_mlx = torch_to_mlx(QS) if hasattr(QS, "shape") else QS

        KW_mlx = treeify(KW)
        KA_mlx = torch_to_mlx(KA)
        KB_mlx = torch_to_mlx(KB)
        KS_mlx = torch_to_mlx(KS) if hasattr(KS, "shape") else KS

        VW_mlx = treeify(VW)
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

        return (
            mlx_to_torch(Q_mlx, device = X.device, dtype = X.dtype),
            mlx_to_torch(K_mlx, device = X.device, dtype = X.dtype),
            mlx_to_torch(V_mlx, device = X.device, dtype = X.dtype),
        )


@mx.compile
def _compiled_o(X, OW, OA, OB, OS):
    return X @ _dequant(OW).T + (X @ OA.T) @ OB.T * OS


def apply_lora_o(X, OW, OW_quant, OA, OB, OS):
    with mlx_context():
        X_mlx = torch_to_mlx(X)

        # Batch=1 Optimization
        if X_mlx.size // X_mlx.shape[-1] == 1:
            OW_mlx = torch_to_mlx(OW)
            OA_mlx = torch_to_mlx(OA)
            OB_mlx = torch_to_mlx(OB)
            OS_val = OS.item() if hasattr(OS, "item") else OS

            X_flat = X_mlx.reshape(1, -1)
            out = (
                quantized_matmul(X_flat, OW_mlx)
                + (X_flat @ OA_mlx.T) @ OB_mlx.T * OS_val
            )
            return mlx_to_torch(
                out.reshape(X.shape[:-1] + (OW.shape[0],)),
                device = X.device,
                dtype = X.dtype,
            )

        OW_mlx = treeify(OW)
        OA_mlx = torch_to_mlx(OA)
        OB_mlx = torch_to_mlx(OB)
        OS_mlx = torch_to_mlx(OS) if hasattr(OS, "shape") else OS

        out_mlx = _compiled_o(X_mlx, OW_mlx, OA_mlx, OB_mlx, OS_mlx)
        return mlx_to_torch(out_mlx, device = X.device, dtype = X.dtype)
