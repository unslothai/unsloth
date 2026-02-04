# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from unsloth.kernels.mlx.quantization import MLXQuantizedWeight, quantize_4bit


def _get_mlx_cached(tensor):
    """Refactored helper to get or create cached MLX tensor"""
    res = getattr(tensor, "_mlx_cache", None)
    if res is None:
        res = torch_to_mlx(tensor)
        tensor._mlx_cache = res
    return res


def _mlx_matmul(X_mlx, W, A, B, s):
    """
    MLX Matmul with LoRA support.
    X_mlx: MLX array
    W: PyTorch tensor (will be cached)
    A, B: PyTorch tensors (LoRA adapters, cached)
    s: scaling factor
    """
    import mlx.core as mx

    W_mlx: Any = _get_mlx_cached(W)

    # Base projection
    if isinstance(W_mlx, MLXQuantizedWeight):
        # Quantized MatMul: (x, w, scales, biases, transpose=True/False)
        # MLX quantized_matmul(x, w, scales, biases, transpose=True, group_size=64)
        # Our W_mlx.weight is (Out, In_packed). X is (..., In).
        # We need X @ W.T
        # mx.quantized_matmul supports transpose.
        out = mx.quantized_matmul(
            X_mlx,
            W_mlx.weight,
            scales=W_mlx.scales,
            biases=W_mlx.biases,
            transpose=True,
            group_size=W_mlx.group_size,
        )
    else:
        # Standard Linear: X @ W.T
        out = X_mlx @ W_mlx.T

    if A is not None:
        A_mlx = _get_mlx_cached(A)
        B_mlx = _get_mlx_cached(B)

        # LoRA: (X @ A.T) @ B.T * s
        # X: [..., D], A: [R, D], B: [O, R]
        # X @ A.T -> [..., R]
        # result @ B.T -> [..., O]
        XA = X_mlx @ A_mlx.T
        lora_out = (XA @ B_mlx.T) * s
        out = out + lora_out

    return out


def mps_matmul_lora(X, W, W_quant, A, B, s):
    """
    MPS matmul_lora fallback.
    Assumes W is already in a usable format for MPS (16-bit).
    """
    dtype = X.dtype

    # Ensure W matches X's dtype (prevents half vs bfloat16 errors)
    W_ = W.to(dtype) if W.dtype != dtype else W
    
    # Base projection: X @ W.t()
    out = torch.matmul(X, W_.t())

    # LoRA contribution: (X @ A.t()) @ (B.t() * s)
    if A is not None:
        # X: (..., in_dim), A: (rank, in_dim), B: (out_dim, rank)
        # Prepare operands, ensuring they all match X.dtype strictly
        A_ = A.t().to(dtype)
        B_ = B.t().to(dtype)
        
        XA = torch.matmul(X, A_)
        
        # Strict casting before in-place addmm_
        # MPS matmul or intermediate ops might silently produce float32
        # This prevents "RuntimeError: self and mat2 must have the same dtype"
        if out.dtype != dtype:
            out = out.to(dtype=dtype)
        if XA.dtype != dtype:
            XA = XA.to(dtype=dtype)
        if B_.dtype != dtype:
            B_ = B_.to(dtype=dtype)

        out.view(-1, out.shape[-1]).addmm_(
            XA.view(-1, XA.shape[-1]), B_, alpha=s
        )

    return out


class MPSLoRA_MLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        _forward_function,
    ):
        # Forward pass using MPS-compatible operations
        e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.save_for_backward(
            X, gateW, gateA, gateB, upW, upA, upB, downW, downA, downB, e, g, h
        )
        ctx.gateS, ctx.upS, ctx.downS = gateS, upS, downS
        return i

    @staticmethod
    def backward(ctx, dY):
        X, gateW, gateA, gateB, upW, upA, upB, downW, downA, downB, e, g, h = (
            ctx.saved_tensors
        )
        gateS, upS, downS = ctx.gateS, ctx.upS, ctx.downS

        # simplified backward for now - focus on correctness
        # In a real implementation we would match LoRA_MLP.backward exactly
        # but for MPS fallback we can rely on autograd for these complex fused ops
        # unless performance is critical.
        # Actually, for Unsloth we should probably provide the manual backward to save memory.

        # TODO: Implement full manual backward for LoRA_MLP on MPS
        raise NotImplementedError(
            "Manual backward for MPSLoRA_MLP not yet implemented. Use autograd for now."
        )


def mps_apply_lora_mlp_swiglu(
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
    """MPS SwiGLU MLP fallback using PyTorch operations."""

    # CHAINING: If input has MLX cache, runs entirely in MLX (sandwich)
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)

            # 1. Up/Gate Projections (Fused in MLX)
            e_mlx = _mlx_matmul(X_mlx, gateW, gateA, gateB, gateS)
            g_mlx = _mlx_matmul(X_mlx, upW, upA, upB, upS)

            # 2. SwiGLU Activation (Fused in MLX)
            # e and g are already MLX arrays here
            import mlx.core as mx

            # Ensure we use the fast metal kernel wrapper from metal/swiglu.py
            from ..metal.swiglu import mlx_swiglu_forward

            h_mlx = mlx_swiglu_forward(e_mlx, g_mlx)

            # 3. Down Projection (Fused in MLX)
            out_mlx = _mlx_matmul(h_mlx, downW, downA, downB, downS)

            # Return PyTorch tensor with cache for next layer
            out = mlx_to_torch(out_mlx).view_as(
                X
            )  # Assuming shape preservation? No, dim might change
            # Correct shape handling:
            # MLP usually preserves B, S, but changes hidden size.
            # But Up/Down should return to original dim usually? Or intermediate?
            # Llama MLP: Hidden -> Inter -> Hidden. So output shape ~ input shape (usually).
            # Let's rely on mlx_to_torch handling flat buffer and reshape if needed.
            # But safe bet is constructing correct view.
            # X shape: [B, S, H]. downW shape: [H, I]. Out: [B, S, H].
            # Getting explicit shape from MLX output is safer.
            out = mlx_to_torch(out_mlx)

            # Reshape based on X batch/seq dims and downW output dim
            out_dim = downW.shape[0]
            if X.dim() == 2:
                out = out.view(X.shape[0], out_dim)
            else:
                out = out.view(X.shape[0], X.shape[1], out_dim)

            out._mlx_cache = out_mlx
            return out

    # Dispatching to torch-native implementation
    # For now, we use the device-agnostic logic but with MPS-friendly matmul
    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal SwiGLU if available (now handles autograd correctly)
    from ..metal import is_metal_swiglu_available
    if is_metal_swiglu_available():
        from ..metal.swiglu import metal_swiglu_forward
        h = metal_swiglu_forward(e, g)
    else:
        h = F.silu(e) * g
    
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)


def mps_apply_lora_qkv(
    X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS
):
    """MPS QKV projection fallback using PyTorch operations."""
    
    # CHAINING: MLX Fast Path
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)
            
            # Import MLX kernel locally to avoid circular imports
            from ..metal.lora import apply_lora_qkv_mlx
            
            Q_mlx, K_mlx, V_mlx = apply_lora_qkv_mlx(
                X_mlx, 
                _get_mlx_cached(QW), QW_quant, _get_mlx_cached(QA), _get_mlx_cached(QB), QS,
                _get_mlx_cached(KW), KW_quant, _get_mlx_cached(KA), _get_mlx_cached(KB), KS,
                _get_mlx_cached(VW), VW_quant, _get_mlx_cached(VA), _get_mlx_cached(VB), VS
            )
            
            return mlx_to_torch(Q_mlx), mlx_to_torch(K_mlx), mlx_to_torch(V_mlx)

    Q = mps_matmul_lora(X, QW, QW_quant, QA, QB, QS)
    K = mps_matmul_lora(X, KW, KW_quant, KA, KB, KS)
    V = mps_matmul_lora(X, VW, VW_quant, VA, VB, VS)
    return Q, K, V


def mps_apply_lora_o(X, OW, OW_quant, OA, OB, OS):
    """MPS O projection fallback using PyTorch operations."""
    
    # CHAINING: MLX Fast Path
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)
            
            from ..metal.lora import apply_lora_o_mlx
            
            out_mlx = apply_lora_o_mlx(
                X_mlx, _get_mlx_cached(OW), OW_quant, _get_mlx_cached(OA), _get_mlx_cached(OB), OS
            )
            
            return mlx_to_torch(out_mlx)
            
    return mps_matmul_lora(X, OW, OW_quant, OA, OB, OS)


def mps_apply_lora_mlp_geglu_exact(
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
    """MPS GEGLU (Exact) MLP fallback using PyTorch operations."""

    # CHAINING: MLX Fast Path
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)
            e_mlx = _mlx_matmul(X_mlx, gateW, gateA, gateB, gateS)
            g_mlx = _mlx_matmul(X_mlx, upW, upA, upB, upS)

            from ..metal.geglu import mlx_geglu_exact_forward

            h_mlx = mlx_geglu_exact_forward(e_mlx, g_mlx)

            out_mlx = _mlx_matmul(h_mlx, downW, downA, downB, downS)

            out = mlx_to_torch(out_mlx)
            out_dim = downW.shape[0]
            if X.dim() == 2:
                out = out.view(X.shape[0], out_dim)
            else:
                out = out.view(X.shape[0], X.shape[1], out_dim)

            out._mlx_cache = out_mlx
            return out

    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal GEGLU if available (now handles autograd correctly)
    from ..metal import is_metal_geglu_available
    if is_metal_geglu_available():
        from ..metal.geglu import metal_geglu_exact_forward
        h = metal_geglu_exact_forward(e, g)
    else:
        h = F.gelu(e, approximate="none") * g
    
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)


def mps_apply_lora_mlp_geglu_approx(
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
    """MPS GEGLU (Approximate) MLP fallback using PyTorch operations."""

    # CHAINING: MLX Fast Path
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)
            e_mlx = _mlx_matmul(X_mlx, gateW, gateA, gateB, gateS)
            g_mlx = _mlx_matmul(X_mlx, upW, upA, upB, upS)

            from ..metal.geglu import mlx_geglu_approx_forward

            h_mlx = mlx_geglu_approx_forward(e_mlx, g_mlx)

            out_mlx = _mlx_matmul(h_mlx, downW, downA, downB, downS)

            out = mlx_to_torch(out_mlx)
            out_dim = downW.shape[0]
            if X.dim() == 2:
                out = out.view(X.shape[0], out_dim)
            else:
                out = out.view(X.shape[0], X.shape[1], out_dim)

            out._mlx_cache = out_mlx
            return out

    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal GEGLU if available (now handles autograd correctly)
    from ..metal import is_metal_geglu_available
    if is_metal_geglu_available():
        from ..metal.geglu import metal_geglu_approx_forward
        h = metal_geglu_approx_forward(e, g)
    else:
        h = F.gelu(e, approximate="tanh") * g
    
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)
