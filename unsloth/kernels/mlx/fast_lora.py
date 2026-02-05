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
    if isinstance(w, MLXQuantizedWeight):
        return {
            "weight": w.weight,
            "scales": w.scales,
            "biases": w.biases,
            "group_size": w.group_size,
            "bits": w.bits,
        }
    if not isinstance(w, mx.array):
        return torch_to_mlx(w)
    return w


def quantized_matmul(X, W):
    """
    Universal matmul: handles arrays, MLXQuantizedWeight, and treeified dictionaries.
    """
    if isinstance(W, MLXQuantizedWeight):
        return mx.quantized_matmul(
            X,
            W.weight,
            W.scales,
            W.biases,
            group_size=W.group_size,
            bits=W.bits,
        )
    if isinstance(W, dict) and "weight" in W:
        return mx.quantized_matmul(
            X,
            W["weight"],
            W["scales"],
            W["biases"],
            group_size=W["group_size"],
            bits=W["bits"],
        )

    # Batch=1 Optimization for FP16
    if X.size // X.shape[-1] == 1:
        # For decoding, a simple matmul or custom GEMV is often faster than quantized_matmul 
        # but here we are in the FP16 path.
        return X @ W.T

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
    if A is None:
        return out

    # Unsloth: A (Rank, In), B (Out, Rank).
    lora_out = (X @ A.T) @ B.T
    return out + lora_out * S


def _dequant(W):
    if isinstance(W, dict) and "weight" in W:
        return mx.dequantize(
            W["weight"],
            W["scales"],
            W["biases"],
            group_size=W["group_size"],
            bits=W["bits"],
        )
    return W


@mx.compile
def _gelu_approx(x):
    # f = 1/2 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    s = 0.7978845608028654  # sqrt(2 / pi)
    return (0.5 * x * (1 + mx.tanh(s * (x + 0.044715 * x * x * x))))


@mx.compile
def _compiled_mlp_geglu(
    X, gateW, gateA, gateB, gateS, upW, upA, upB, upS, downW, downA, downB, downS
):
    # Use quantized_matmul directly - MLX will optimize this if weights are quantized trees
    gate = quantized_matmul(X, gateW) + (X @ gateA.T) @ gateB.T * gateS
    up = quantized_matmul(X, upW) + (X @ upA.T) @ upB.T * upS

    # GEGLU: gelu(gate) * up
    act = _gelu_approx(gate) * up
    out = quantized_matmul(act, downW) + (act @ downA.T) @ downB.T * downS

    return out


@mx.compile
def _compiled_mlp_swiglu(
    X, gateW, gateA, gateB, gateS, upW, upA, upB, upS, downW, downA, downB, downS
):
    # Use quantized_matmul directly - MLX will optimize this if weights are quantized trees
    gate = quantized_matmul(X, gateW) + (X @ gateA.T) @ gateB.T * gateS
    up = quantized_matmul(X, upW) + (X @ upA.T) @ upB.T * upS

    # SwiGLU: silu(gate) * up = (gate * sigmoid(gate)) * up
    act = (gate * mx.sigmoid(gate)) * up
    out = quantized_matmul(act, downW) + (act @ downA.T) @ downB.T * downS

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
        if isinstance(X, mx.array):
            X_mlx = X
            # If input is MLX, we return MLX array directly
            return_mlx = True
        else:
            X_mlx = torch_to_mlx(X)
            return_mlx = False

        # Check if LoRA is attached (any adapter is not None)
        has_lora = gateA is not None and gateB is not None

        # Batch=1 Optimization: Use custom GEMV for base projections
        # X shape is usually (Batch, Seq, Dim).
        # For decoding, Batch=1, Seq=1.
        if X_mlx.size // X_mlx.shape[-1] == 1:
            # Need weights in MLX
            gateW_mlx = torch_to_mlx(gateW)
            upW_mlx = torch_to_mlx(upW)
            downW_mlx = torch_to_mlx(downW)

            X_flat = X_mlx.reshape(1, -1)

            # 1. Gate
            gate = quantized_matmul(X_flat, gateW_mlx)
            if has_lora:
                gateA_mlx = torch_to_mlx(gateA)
                gateB_mlx = torch_to_mlx(gateB)
                gateS_val = gateS.item() if hasattr(gateS, "item") else gateS
                gate = gate + (X_flat @ gateA_mlx.T) @ gateB_mlx.T * gateS_val

            # 2. Up
            up = quantized_matmul(X_flat, upW_mlx)
            if has_lora:
                upA_mlx = torch_to_mlx(upA)
                upB_mlx = torch_to_mlx(upB)
                upS_val = upS.item() if hasattr(upS, "item") else upS
                up = up + (X_flat @ upA_mlx.T) @ upB_mlx.T * upS_val

            # 3. SwiGLU: silu(gate) * up
            act = (gate * mx.sigmoid(gate)) * up

            # 4. Down
            out = quantized_matmul(act, downW_mlx)
            if has_lora:
                downA_mlx = torch_to_mlx(downA)
                downB_mlx = torch_to_mlx(downB)
                downS_val = downS.item() if hasattr(downS, "item") else downS
                out = out + (act @ downA_mlx.T) @ downB_mlx.T * downS_val

            out_reshaped = out.reshape(X.shape)
            if return_mlx:
                return out_reshaped
            return mlx_to_torch(out_reshaped, device=X.device, dtype=X.dtype)

        # Standard Batch Path
        gateW_mlx = treeify(gateW)
        upW_mlx = treeify(upW)
        downW_mlx = treeify(downW)
        
        if not has_lora:
            # No LoRA: simple base projection path
            gate = quantized_matmul(X_mlx, gateW_mlx)
            up = quantized_matmul(X_mlx, upW_mlx)
            act = (gate * mx.sigmoid(gate)) * up
            out_mlx = quantized_matmul(act, downW_mlx)
        else:
            # With LoRA: use compiled kernel
            gateA_mlx = torch_to_mlx(gateA)
            gateB_mlx = torch_to_mlx(gateB)
            gateS_mlx = torch_to_mlx(gateS) if hasattr(gateS, "shape") else gateS

            upA_mlx = torch_to_mlx(upA)
            upB_mlx = torch_to_mlx(upB)
            upS_mlx = torch_to_mlx(upS) if hasattr(upS, "shape") else upS

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

        if return_mlx:
            return out_mlx
        return mlx_to_torch(out_mlx, device=X.device, dtype=X.dtype)


def apply_lora_mlp_geglu(
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
        if isinstance(X, mx.array):
            X_mlx = X
            return_mlx = True
        else:
            X_mlx = torch_to_mlx(X)
            return_mlx = False

        # Check if LoRA is attached (any adapter is not None)
        has_lora = gateA is not None and gateB is not None
        
        # Batch=1 Optimization
        if X_mlx.size // X_mlx.shape[-1] == 1:
            gateW_mlx = torch_to_mlx(gateW)
            upW_mlx = torch_to_mlx(upW)
            downW_mlx = torch_to_mlx(downW)

            X_flat = X_mlx.reshape(1, -1)
            
            # 1. Gate
            gate = quantized_matmul(X_flat, gateW_mlx)
            if has_lora:
                gateA_mlx = torch_to_mlx(gateA)
                gateB_mlx = torch_to_mlx(gateB)
                gateS_val = gateS.item() if hasattr(gateS, "item") else gateS
                gate = gate + (X_flat @ gateA_mlx.T) @ gateB_mlx.T * gateS_val

            # 2. Up
            up = quantized_matmul(X_flat, upW_mlx)
            if has_lora:
                upA_mlx = torch_to_mlx(upA)
                upB_mlx = torch_to_mlx(upB)
                upS_val = upS.item() if hasattr(upS, "item") else upS
                up = up + (X_flat @ upA_mlx.T) @ upB_mlx.T * upS_val

            # 3. GEGLU: gelu(gate) * up
            act = _gelu_approx(gate) * up

            # 4. Down
            out = quantized_matmul(act, downW_mlx)
            if has_lora:
                downA_mlx = torch_to_mlx(downA)
                downB_mlx = torch_to_mlx(downB)
                downS_val = downS.item() if hasattr(downS, "item") else downS
                out = out + (act @ downA_mlx.T) @ downB_mlx.T * downS_val

            out_reshaped = out.reshape(X.shape)
            if return_mlx:
                return out_reshaped
            return mlx_to_torch(out_reshaped, device=X.device, dtype=X.dtype)

        # Standard Batch Path
        gateW_mlx = treeify(gateW)
        upW_mlx = treeify(upW)
        downW_mlx = treeify(downW)
        
        if not has_lora:
            # No LoRA: simple base projection path
            gate = quantized_matmul(X_mlx, gateW_mlx)
            up = quantized_matmul(X_mlx, upW_mlx)
            act = _gelu_approx(gate) * up
            out_mlx = quantized_matmul(act, downW_mlx)
        else:
            # With LoRA: use compiled kernel
            gateA_mlx = torch_to_mlx(gateA)
            gateB_mlx = torch_to_mlx(gateB)
            gateS_mlx = torch_to_mlx(gateS) if hasattr(gateS, "shape") else gateS

            upA_mlx = torch_to_mlx(upA)
            upB_mlx = torch_to_mlx(upB)
            upS_mlx = torch_to_mlx(upS) if hasattr(upS, "shape") else upS

            downA_mlx = torch_to_mlx(downA)
            downB_mlx = torch_to_mlx(downB)
            downS_mlx = torch_to_mlx(downS) if hasattr(downS, "shape") else downS

            # Execute compiled kernel
            out_mlx = _compiled_mlp_geglu(
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

        if return_mlx:
            return out_mlx
        return mlx_to_torch(out_mlx, device=X.device, dtype=X.dtype)


@mx.compile
def _compiled_qkv(X, QW, QA, QB, QS, KW, KA, KB, KS, VW, VA, VB, VS):
    Q = quantized_matmul(X, QW) + (X @ QA.T) @ QB.T * QS
    K = quantized_matmul(X, KW) + (X @ KA.T) @ KB.T * KS
    V = quantized_matmul(X, VW) + (X @ VA.T) @ VB.T * VS
    return Q, K, V


def apply_lora_qkv(
    X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS
):
    with mlx_context():
        if isinstance(X, mx.array):
            X_mlx = X
            return_mlx = True
        else:
            X_mlx = torch_to_mlx(X)
            return_mlx = False

        has_lora = QA is not None

        # Batch=1 Optimization
        if X_mlx.size // X_mlx.shape[-1] == 1:
            # Inputs
            QW_mlx = torch_to_mlx(QW)
            KW_mlx = torch_to_mlx(KW)
            VW_mlx = torch_to_mlx(VW)

            X_flat = X_mlx.reshape(1, -1)

            if not has_lora:
                Q = quantized_matmul(X_flat, QW_mlx)
                K = quantized_matmul(X_flat, KW_mlx)
                V = quantized_matmul(X_flat, VW_mlx)
            else:
                QA_mlx = torch_to_mlx(QA)
                QB_mlx = torch_to_mlx(QB)
                QS_val = QS.item() if hasattr(QS, "item") else QS
                
                KA_mlx = torch_to_mlx(KA)
                KB_mlx = torch_to_mlx(KB)
                KS_val = KS.item() if hasattr(KS, "item") else KS
                
                VA_mlx = torch_to_mlx(VA)
                VB_mlx = torch_to_mlx(VB)
                VS_val = VS.item() if hasattr(VS, "item") else VS

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

            Q_out = Q.reshape(X.shape[:-1] + (QW.shape[0],))
            K_out = K.reshape(X.shape[:-1] + (KW.shape[0],))
            V_out = V.reshape(X.shape[:-1] + (VW.shape[0],))

            if return_mlx:
                return Q_out, K_out, V_out

            return (
                mlx_to_torch(Q_out, device=X.device, dtype=X.dtype),
                mlx_to_torch(K_out, device=X.device, dtype=X.dtype),
                mlx_to_torch(V_out, device=X.device, dtype=X.dtype),
            )

        QW_mlx = treeify(QW)
        KW_mlx = treeify(KW)
        VW_mlx = treeify(VW)

        if not has_lora:
            Q_mlx = quantized_matmul(X_mlx, QW_mlx)
            K_mlx = quantized_matmul(X_mlx, KW_mlx)
            V_mlx = quantized_matmul(X_mlx, VW_mlx)
        else:
            QA_mlx = torch_to_mlx(QA)
            QB_mlx = torch_to_mlx(QB)
            QS_mlx = torch_to_mlx(QS) if hasattr(QS, "shape") else QS

            KA_mlx = torch_to_mlx(KA)
            KB_mlx = torch_to_mlx(KB)
            KS_mlx = torch_to_mlx(KS) if hasattr(KS, "shape") else KS

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

        if return_mlx:
            return Q_mlx, K_mlx, V_mlx

        return (
            mlx_to_torch(Q_mlx, device=X.device, dtype=X.dtype),
            mlx_to_torch(K_mlx, device=X.device, dtype=X.dtype),
            mlx_to_torch(V_mlx, device=X.device, dtype=X.dtype),
        )


@mx.compile
def _compiled_o(X, OW, OA, OB, OS):
    return quantized_matmul(X, OW) + (X @ OA.T) @ OB.T * OS


def apply_lora_o(X, OW, OW_quant, OA, OB, OS):
    with mlx_context():
        if isinstance(X, mx.array):
            X_mlx = X
            return_mlx = True
        else:
            X_mlx = torch_to_mlx(X)
            return_mlx = False

        has_lora = OA is not None

        # Batch=1 Optimization
        if X_mlx.size // X_mlx.shape[-1] == 1:
            OW_mlx = torch_to_mlx(OW)
            X_flat = X_mlx.reshape(1, -1)

            if not has_lora:
                out = quantized_matmul(X_flat, OW_mlx)
            else:
                OA_mlx = torch_to_mlx(OA)
                OB_mlx = torch_to_mlx(OB)
                OS_val = OS.item() if hasattr(OS, "item") else OS
                out = (
                    quantized_matmul(X_flat, OW_mlx)
                    + (X_flat @ OA_mlx.T) @ OB_mlx.T * OS_val
                )
            
            out_reshaped = out.reshape(X.shape[:-1] + (OW.shape[0],))
            
            if return_mlx:
                return out_reshaped

            return mlx_to_torch(
                out_reshaped,
                device=X.device,
                dtype=X.dtype,
            )

        OW_mlx = treeify(OW)
        
        if not has_lora:
            out_mlx = quantized_matmul(X_mlx, OW_mlx)
        else:
            OA_mlx = torch_to_mlx(OA)
            OB_mlx = torch_to_mlx(OB)
            OS_mlx = torch_to_mlx(OS) if hasattr(OS, "shape") else OS
            out_mlx = _compiled_o(X_mlx, OW_mlx, OA_mlx, OB_mlx, OS_mlx)
        
        if return_mlx:
            return out_mlx
            
        return mlx_to_torch(out_mlx, device=X.device, dtype=X.dtype)
