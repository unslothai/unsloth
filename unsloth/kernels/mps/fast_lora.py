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

        # Use addmm (not addmm_) to avoid in-place ops that break gradient graph on MPS
        out = out.view(-1, out.shape[-1]).addmm(
            XA.view(-1, XA.shape[-1]), B_, alpha=s
        ).view(out.shape)

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
        _backward_function,
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
        ctx._backward_function = _backward_function
        return i

    @staticmethod
    def backward(ctx, dY):
        X, gateW, gateA, gateB, upW, upA, upB, downW, downA, downB, e, g, h = (
            ctx.saved_tensors
        )
        gateS, upS, downS = ctx.gateS, ctx.upS, ctx.downS
        _backward_function = ctx._backward_function

        # Get shapes and flatten for matmul operations
        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype
        dY = dY.to(dtype)

        # Ensure LoRA weights are correct dtype and transposed
        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype).t(),
            gateB.to(dtype).t(),
            upA.to(dtype).t(),
            upB.to(dtype).t(),
            downA.to(dtype).t(),
            downB.to(dtype).t(),
        )

        # Compute DW = dY @ (downW + downS * downB @ downA)^T
        # First get base projection: dY @ downW.T
        # downW is [Out, In], so we need dY @ downW to get [B, In]
        downW_t = downW.to(dtype)
        DW = torch.matmul(dY, downW_t)
        
        # Add LoRA contribution: dY @ (downS * downB @ downA)^T
        # = downS * dY @ downA.T @ downB.T
        # Use addmm (not addmm_) to avoid in-place ops
        if downA is not None and downB is not None:
             # downA [In, R], downB [R, Out] (after transpose local)
             # we want dY [B, Out] @ downB.t() [Out, R] -> [B, R]
             # then @ downA.t() [R, In] -> [B, In]
            DW = DW.addmm(dY @ downB.t(), downA.t(), alpha=downS)

        # Apply backward function to compute h, df, de
        # _backward_function modifies tensors in place, so we pass copies to preserve
        # the original saved_tensors which don't have requires_grad
        DW_clone = DW.clone()
        e_clone = e.clone()
        g_clone = g.clone()
        h_out, df, de = _backward_function(DW_clone, e_clone, g_clone)
        h_out, df, de = h_out.to(dtype), df.to(dtype), de.to(dtype)

        # Initialize gradient buffers for LoRA weights
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)
        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)

        # Down projection LoRA gradients
        # d_downA = h.T @ (dY @ downB.T) * downS
        # d_downB = (downA.T @ h.T) @ dY * downS
        # Use addmm (not addmm_) to avoid in-place ops on saved tensors
        d_downA = d_downA.addmm(h_out.t(), dY @ downB.t(), alpha=downS, beta=0)
        d_downB = d_downB.addmm(downA.t() @ h_out.t(), dY, alpha=downS, beta=0)

        # Up projection LoRA gradients
        # d_upA = X.T @ (df @ upB.T) * upS
        # d_upB = (upA.T @ X.T) @ df * upS
        # Use addmm (not addmm_) to avoid in-place ops on saved tensors from ctx
        d_upA = d_upA.addmm(X.t(), df @ upB.t(), alpha=upS, beta=0)
        d_upB = d_upB.addmm(upA.t() @ X.t(), df, alpha=upS, beta=0)

        # Gate projection LoRA gradients
        # d_gateA = X.T @ (de @ gateB.T) * gateS
        # d_gateB = (gateA.T @ X.T) @ de * gateS
        # Use addmm (not addmm_) to avoid in-place ops on saved tensors from ctx
        d_gateA = d_gateA.addmm(X.t(), de @ gateB.t(), alpha=gateS, beta=0)
        d_gateB = d_gateB.addmm(gateA.t() @ X.t(), de, alpha=gateS, beta=0)

        # Compute dX
        # dX = df @ upW.T + de @ gateW.T + LoRA contributions
        upW_t = upW.t().to(dtype)
        dX = torch.matmul(df, upW_t.t())
        del upW_t
        
        # Add up projection LoRA contribution: df @ upB.T @ upA.T * upS
        # Use addmm (not addmm_) to avoid in-place ops on tensors without requires_grad
        dX = dX.addmm(df @ upB.t(), upA.t(), alpha=upS)

        # Add gate projection contribution
        gateW_t = gateW.t().to(dtype)
        dX = dX.addmm(de, gateW_t.t())
        del gateW_t
        
        # Add gate projection LoRA contribution: de @ gateB.T @ gateA.T * gateS
        dX = dX.addmm(de @ gateB.t(), gateA.t(), alpha=gateS)

        # Reshape dX back to original shape
        dX = dX.view(batch, seq_len, hd)

        # Return gradients (None for weights/quant states/scalars/_forward/_backward)
        return (
            dX,
            None, None,  # gateW, gateW_quant
            d_gateA.t(), d_gateB.t(),  # gateA, gateB
            None,  # gateS
            None, None,  # upW, upW_quant
            d_upA.t(), d_upB.t(),  # upA, upB
            None,  # upS
            None, None,  # downW, downW_quant
            d_downA.t(), d_downB.t(),  # downA, downB
            None,  # downS
            None, None,  # _forward_function, _backward_function
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
    gate_multiplier=None,
    down_multiplier=None,
):
    """MPS SwiGLU MLP fallback using PyTorch operations."""

    # CHAINING: If input has MLX cache, runs entirely in MLX (sandwich)
    if getattr(X, "_mlx_cache", None) is not None and not torch.is_grad_enabled():
        with mlx_context():
            X_mlx = _get_mlx_cached(X)

            # 1. Up/Gate Projections (Fused in MLX)
            e_mlx = _mlx_matmul(X_mlx, gateW, gateA, gateB, gateS)
            if gate_multiplier is not None:
                e_mlx = e_mlx * gate_multiplier
                
            g_mlx = _mlx_matmul(X_mlx, upW, upA, upB, upS)

            # 2. SwiGLU Activation (Fused in MLX)
            # e and g are already MLX arrays here
            import mlx.core as mx

            # Ensure we use the fast metal kernel wrapper from metal/swiglu.py
            from ..metal.swiglu import mlx_swiglu_forward

            h_mlx = mlx_swiglu_forward(e_mlx, g_mlx)

            # 3. Down Projection (Fused in MLX)
            out_mlx = _mlx_matmul(h_mlx, downW, downA, downB, downS)
            if down_multiplier is not None:
                out_mlx = out_mlx * down_multiplier
                
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
    if gate_multiplier is not None:
        e = e * gate_multiplier
        
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal SwiGLU if available (now handles autograd correctly)
    from ..metal import is_metal_swiglu_available
    if is_metal_swiglu_available():
        from ..metal.swiglu import metal_swiglu_forward
        h = metal_swiglu_forward(e, g)
    else:
        h = F.silu(e) * g
    
    out = mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)
    if down_multiplier is not None:
        out = out * down_multiplier
    return out


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
    gate_multiplier=None,
    down_multiplier=None,
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
    if gate_multiplier is not None:
        e = e * gate_multiplier
        
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal GEGLU if available (now handles autograd correctly)
    from ..metal import is_metal_geglu_available
    if is_metal_geglu_available():
        from ..metal.geglu import metal_geglu_exact_forward
        h = metal_geglu_exact_forward(e, g)
    else:
        h = F.gelu(e, approximate="none") * g
    
    out = mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)
    if down_multiplier is not None:
        out = out * down_multiplier
    return out
    


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
    gate_multiplier=None,
    down_multiplier=None,
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
    if gate_multiplier is not None:
        e = e * gate_multiplier
        
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    
    # Use optimized Metal GEGLU if available (now handles autograd correctly)
    from ..metal import is_metal_geglu_available
    if is_metal_geglu_available():
        from ..metal.geglu import metal_geglu_approx_forward
        h = metal_geglu_approx_forward(e, g)
    else:
        h = F.gelu(e, approximate="tanh") * g
    
    out = mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)
    if down_multiplier is not None:
        out = out * down_multiplier
    return out
