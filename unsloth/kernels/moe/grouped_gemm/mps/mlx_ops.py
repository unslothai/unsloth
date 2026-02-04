# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
MLX implementation of Grouped GEMM for Apple Silicon.
Leverages mx.compile to optimize iterative expert execution into a single graph.
"""

import torch
import mlx.core as mx
from typing import Optional, Tuple
from ....mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context, synchronize_mps
from ....mlx.utils import is_mlx_available
from .fallback import GroupedGemmMPS
from ..reference.moe_ops import unpermute

# Internal MLX Kernels
@mx.compile
def _mx_grouped_gemm_forward_kernel(X, W, m_cumsum, num_experts, N):
    # X: (total_tokens, K) in expert order
    # W: (E*N, K)
    Y = mx.zeros((X.shape[0], N), dtype=X.dtype)
    for i in range(num_experts):
        m_start = m_cumsum[i]
        m_end = m_cumsum[i+1]
        
        # Expert weight (N, K)
        W_exp = W[i*N:(i+1)*N]
        # X segment (m_size, K)
        X_exp = X[m_start:m_end]
        
        # matmul (m_size, K) @ (K, N) -> (m_size, N)
        Y_exp = mx.matmul(X_exp, W_exp.T)
        Y[m_start:m_end] = Y_exp
    return Y

@mx.compile
def _mx_grouped_gemm_dX_kernel(dY, W, m_cumsum, num_experts, K, N):
    # dY: (total_tokens, N) in expert order
    # W: (E*N, K)
    dX = mx.zeros((dY.shape[0], K), dtype=dY.dtype)
    for i in range(num_experts):
        m_start = m_cumsum[i]
        m_end = m_cumsum[i+1]
        
        W_exp = W[i*N:(i+1)*N]
        dY_exp = dY[m_start:m_end]
        
        # dX_exp = dY_exp @ W_exp -> (m_size, N) @ (N, K) -> (m_size, K)
        dX[m_start:m_end] = mx.matmul(dY_exp, W_exp)
    return dX

@mx.compile
def _mx_grouped_gemm_dW_kernel(X, dY, m_cumsum, num_experts, K, N):
    # X: (total_tokens, K) in expert order
    # dY: (total_tokens, N) in expert order
    dW = mx.zeros((num_experts, N, K), dtype=X.dtype)
    for i in range(num_experts):
        m_start = m_cumsum[i]
        m_end = m_cumsum[i+1]
        
        X_exp = X[m_start:m_end]
        dY_exp = dY[m_start:m_end]
        
        # dW_exp = dY_exp.T @ X_exp -> (N, m_size) @ (m_size, K) -> (N, K)
        dW[i] = mx.matmul(dY_exp.T, X_exp)
    return dW

def mx_grouped_gemm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    topk: int,
    m_sizes: torch.Tensor,
    gather_indices: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    fuse_mul_post: bool = False,
) -> torch.Tensor:
    with mlx_context():
        # Prepare inputs
        num_experts = m_sizes.shape[0]
        total_tokens = X.shape[0] if not permute_x else gather_indices.shape[0]
        num_tokens = total_tokens // topk
        
        # MLX View of X and W
        # W in Unsloth is (E, N, K) but interface.py might flatten it to (E*N, K)
        # We assume W arrives as (E*N, K) or similar flattened form for consistency
        X_mlx = torch_to_mlx(X)
        W_mlx = torch_to_mlx(W)
        m_sizes_mlx = torch_to_mlx(m_sizes)
        
        # m_cumsum
        m_cumsum = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(m_sizes_mlx)])
        
        N = W_mlx.shape[0] // num_experts
        
        # 1. Permute X if needed
        if permute_x:
            # gather_indices maps [token*topk] to [expert-ordered-index]
            # Actually, per interface.py, it's the reverse? 
            # In grouped_gemm_mps_forward, we used:
            # expert_indices = gather_indices[m_start:m_end]
            # X_expert = X[expert_indices // topk]
            # Let's perform permutation in MLX for speed if permute_x is True
            gather_mlx = torch_to_mlx(gather_indices)
            X_expert_order = X_mlx[gather_mlx // topk]
        else:
            X_expert_order = X_mlx
            
        # 2. Main Grouped Matmul
        Y_expert_order = _mx_grouped_gemm_forward_kernel(
            X_expert_order, W_mlx, m_cumsum, num_experts, N
        )
        
        # 3. Permute Y / Unpermute / Fuse
        if permute_y:
            gather_mlx = torch_to_mlx(gather_indices) if not permute_x else gather_mlx
            # Unpermute
            Y_unperm = mx.zeros_like(Y_expert_order)
            Y_unperm[gather_mlx] = Y_expert_order
            
            if fuse_mul_post:
                weights = torch_to_mlx(topk_weights).reshape(-1, 1)
                Y_unperm = Y_unperm * weights
                # Reduce topk
                Y_final = Y_unperm.reshape(num_tokens, topk, N).sum(axis=1)
                return mlx_to_torch(Y_final, device=X.device, dtype=X.dtype)
            else:
                return mlx_to_torch(Y_unperm, device=X.device, dtype=X.dtype)
        
        return mlx_to_torch(Y_expert_order, device=X.device, dtype=X.dtype)

class GroupedGemmMLX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, m_sizes, topk, gather_indices, permute_x, permute_y, topk_weights, fuse_mul_post):
        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        ctx.topk = topk
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.fuse_mul_post = fuse_mul_post
        
        return mx_grouped_gemm_forward(
            X, W, topk, m_sizes, gather_indices, topk_weights, permute_x, permute_y, fuse_mul_post
        )

    @staticmethod
    def backward(ctx, dY):
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        permute_x = ctx.permute_x
        permute_y = ctx.permute_y
        
        with mlx_context():
            X_mlx = torch_to_mlx(X)
            W_mlx = torch_to_mlx(W)
            dY_mlx = torch_to_mlx(dY)
            m_sizes_mlx = torch_to_mlx(m_sizes)
            gather_mlx = torch_to_mlx(gather_indices)
            m_cumsum = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(m_sizes_mlx)])
            
            num_experts = m_sizes_mlx.shape[0]
            N = dY.shape[-1]
            K = X.shape[-1]
            
            # Preparation for dW and dX
            if permute_x:
                X_expert_order = X_mlx[gather_mlx // topk]
            else:
                X_expert_order = X_mlx
                
            if permute_y:
                dY_expert_order = dY_mlx[gather_mlx]
            else:
                dY_expert_order = dY_mlx
                
            # Compute dW
            dW_mlx = _mx_grouped_gemm_dW_kernel(X_expert_order, dY_expert_order, m_cumsum, num_experts, K, N)
            
            # Compute dX
            dX_expert_order = _mx_grouped_gemm_dX_kernel(dY_expert_order, W_mlx, m_cumsum, num_experts, K, N)
            
            if permute_x:
                # Unpermute and Sum
                dX_unperm = mx.zeros((gather_mlx.shape[0], K), dtype=X_mlx.dtype)
                dX_unperm[gather_mlx] = dX_expert_order
                dX_final = dX_unperm.reshape(-1, topk, K).sum(axis=1)
                dX_torch = mlx_to_torch(dX_final, device=X.device, dtype=X.dtype)
            else:
                dX_torch = mlx_to_torch(dX_expert_order, device=X.device, dtype=X.dtype)
                
            dW_torch = mlx_to_torch(dW_mlx, device=W.device, dtype=W.dtype)
            
        return dX_torch, dW_torch, None, None, None, None, None, None, None

def mlx_grouped_gemm(X, W, m_sizes, topk, gather_indices=None, permute_x=False, permute_y=False, topk_weights=None, fuse_mul_post=False):
    return GroupedGemmMLX.apply(X, W, m_sizes, topk, gather_indices, permute_x, permute_y, topk_weights, fuse_mul_post)
