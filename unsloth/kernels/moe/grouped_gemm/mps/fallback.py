# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
PyTorch-native fallback implementations of Grouped GEMM for MPS (Apple Silicon).

This module implements the core grouped GEMM operations using standard PyTorch
operations that work on MPS. The implementation uses an iterative approach over
experts, which is less efficient than fused Triton kernels but maintains correctness.

Key operations:
- grouped_gemm_mps_forward: Forward pass computing Y = X @ W for each expert
- grouped_gemm_mps_dX: Backward pass computing dX = dY @ W.T
- grouped_gemm_mps_dW: Backward pass computing dW = X.T @ dY
"""

import torch
import warnings
from typing import Optional
from ..reference.moe_ops import permute, unpermute

# Flag to track warning state
_MPS_WARN_SHOWN = False


def grouped_gemm_mps_forward(
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
    """
    MPS fallback for grouped GEMM forward pass.
    
    Performs grouped matrix multiplication for MoE layers by iterating over
    each expert and performing separate matmuls.
    
    Args:
        X: Input tensor of shape (M, K) where M is num_tokens if permute_x else total_tokens
        W: Weight tensor of shape (E, N, K) where E is num_experts
        topk: Number of experts selected per token
        m_sizes: Tensor of shape (E,) containing tokens assigned to each expert
        gather_indices: Tensor of shape (total_tokens,) with token indices for each expert
        topk_weights: Optional weights for fused multiplication
        permute_x: Whether to permute input from token order to expert-grouped order
        permute_y: Whether to permute output from expert-grouped order to token order
        fuse_mul_post: Whether to fuse multiplication with topk_weights
    
    Returns:
        Y: Output tensor of shape (total_tokens, N)
    """
    global _MPS_WARN_SHOWN
    if not _MPS_WARN_SHOWN:
        warnings.warn(
            "Using MPS fallback for grouped_gemm. Performance may be lower than Triton kernels."
        )
        _MPS_WARN_SHOWN = True
    
    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()
    
    # Validate inputs
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    
    X = X.view(-1, X.shape[-1])
    W = W.view(-1, W.shape[-1])
    
    num_experts = m_sizes.shape[0]
    _, K = X.shape
    N = W.shape[0] // num_experts
    
    if permute_x or permute_y:
        assert gather_indices is not None, "gather_indices required when permuting"
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk
    
    # Allocate expert-order output buffer
    Y_expert_order = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)
    
    if total_tokens == 0 or N == 0:
        return Y_expert_order
    
    # Compute cumulative sums for expert boundaries  
    m_cumsum = torch.zeros(num_experts + 1, device=m_sizes.device, dtype=torch.long)
    m_cumsum[1:] = torch.cumsum(m_sizes, dim=0)
    
    # Process each expert
    for expert_idx in range(num_experts):
        m_start = m_cumsum[expert_idx].item()
        m_end = m_cumsum[expert_idx + 1].item()
        m_size = m_end - m_start
        
        if m_size == 0:
            continue
        
        # Get expert weight: W[expert_idx] has shape (N, K)
        # Weight tensor W is reshaped as (E*N, K) in the caller
        W_expert = W[expert_idx * N : (expert_idx + 1) * N, :]  # (N, K)
        
        # Get input tokens for this expert
        if permute_x:
            # Input is in token order, gather by indices
            expert_indices = gather_indices[m_start:m_end]
            token_indices = expert_indices // topk  # Get original token indices
            X_expert = X[token_indices]  # (m_size, K)
        else:
            # Input is already in expert-grouped order
            X_expert = X[m_start:m_end]  # (m_size, K)
        
        # Compute Y = X @ W.T -> (m_size, K) @ (K, N) = (m_size, N)
        Y_expert = X_expert @ W_expert.t()
        
        # Store output in expert order
        Y_expert_order[m_start:m_end] = Y_expert
    
    if permute_y:
        # Unpermute results back to token order
        Y_unperm = unpermute(Y_expert_order, gather_indices)
        
        if fuse_mul_post:
            # For inference: multiply by topk weights and reduce to token count
            if topk_weights is not None:
                Y_unperm = Y_unperm * topk_weights.view(-1, 1)
            # Sum topk entries for each token
            return Y_unperm.view(num_tokens, topk, N).sum(dim=1)
        else:
            # Just return all total_tokens in unpermuted order
            return Y_unperm
    
    return Y_expert_order


def grouped_gemm_mps_dX(
    dY: torch.Tensor,
    W: torch.Tensor,
    gather_indices: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    permute_x: bool = False,
    permute_y: bool = False,
) -> torch.Tensor:
    """
    MPS fallback for grouped GEMM backward pass computing dX.
    
    Computes dX = dY @ W for each expert.
    
    Args:
        dY: Gradient of output, shape (total_tokens, N)
        W: Weight tensor of shape (E, N, K)
        gather_indices: Token indices for each expert
        m_sizes: Tokens assigned to each expert
        topk: Number of experts per token
        permute_x: Whether X was permuted in forward
        permute_y: Whether Y was permuted in forward
    
    Returns:
        dX: Gradient of input, shape (total_tokens, K)
    """
    dY = dY.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()
    
    num_experts = m_sizes.shape[0]
    dY = dY.view(-1, dY.shape[-1])
    W = W.view(-1, W.shape[-1])
    
    M_total, N_grad = dY.shape
    N_total, K = W.shape
    N = N_total // num_experts
    
    num_tokens = M_total // topk
    total_tokens = gather_indices.shape[0]
    
    # Output shape: always (total_tokens, K) - accumulation happens later if needed
    dX = torch.zeros((total_tokens, K), device=dY.device, dtype=dY.dtype)
    
    # Compute cumulative sums for expert boundaries
    m_cumsum = torch.zeros(num_experts + 1, device=m_sizes.device, dtype=torch.long)
    m_cumsum[1:] = torch.cumsum(m_sizes, dim=0)
    
    for expert_idx in range(num_experts):
        m_start = m_cumsum[expert_idx].item()
        m_end = m_cumsum[expert_idx + 1].item()
        m_size = m_end - m_start
        
        if m_size == 0:
            continue
        
        # Get expert weight
        W_expert = W[expert_idx * N : (expert_idx + 1) * N, :]  # (N, K)
        
        # Get gradient for this expert's outputs
        if permute_y:
            # dY is in token order (total_tokens, N), need to gather to expert order
            expert_indices = gather_indices[m_start:m_end]
            dY_expert = dY[expert_indices]  # (m_size, N)
        else:
            dY_expert = dY[m_start:m_end]  # (m_size, N)
        
        # Compute dX = dY @ W -> (m_size, N) @ (N, K) = (m_size, K)
        dX_expert = dY_expert @ W_expert
        
        # Store result
        dX[m_start:m_end] = dX_expert
    
    return dX


def grouped_gemm_mps_dW(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    permute_x: bool = False,
    permute_y: bool = False,
) -> torch.Tensor:
    """
    MPS fallback for grouped GEMM backward pass computing dW.
    
    Computes dW = X.T @ dY for each expert.
    
    Args:
        X: Input tensor from forward pass
        dY: Gradient of output
        m_sizes: Tokens assigned to each expert
        gather_indices: Token indices for each expert
        topk: Number of experts per token
        permute_x: Whether X was permuted in forward
        permute_y: Whether Y was permuted in forward
    
    Returns:
        dW: Gradient of weights, shape (E, N, K)
    """
    X = X.view(-1, X.shape[-1]).contiguous()
    dY = dY.contiguous()
    m_sizes = m_sizes.contiguous()
    
    num_experts = m_sizes.shape[0]
    
    if permute_x or permute_y:
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk
    
    _, K = X.shape
    M_grad, N = dY.shape
    
    # Allocate weight gradients
    dW = torch.zeros((num_experts, N, K), device=X.device, dtype=X.dtype)
    
    # Compute cumulative sums for expert boundaries
    m_cumsum = torch.zeros(num_experts + 1, device=m_sizes.device, dtype=torch.long)
    m_cumsum[1:] = torch.cumsum(m_sizes, dim=0)
    
    for expert_idx in range(num_experts):
        m_start = m_cumsum[expert_idx].item()
        m_end = m_cumsum[expert_idx + 1].item()
        m_size = m_end - m_start
        
        if m_size == 0:
            continue
        
        # Get input for this expert
        if permute_x:
            expert_indices = gather_indices[m_start:m_end]
            token_indices = expert_indices // topk
            X_expert = X[token_indices]  # (m_size, K)
        else:
            X_expert = X[m_start:m_end]  # (m_size, K)
        
        # Get gradient for this expert
        if permute_y:
            # dY is in token order (total_tokens, N), need to gather to expert order
            expert_indices = gather_indices[m_start:m_end]
            dY_expert = dY[expert_indices]  # (m_size, N)
        else:
            dY_expert = dY[m_start:m_end]  # (m_size, N)
        
        # Compute dW = dY.T @ X -> (N, m_size) @ (m_size, K) = (N, K)
        dW[expert_idx] = dY_expert.t() @ X_expert
    
    return dW


class GroupedGemmMPS(torch.autograd.Function):
    """
    Autograd Function wrapper for MPS grouped GEMM operations.
    
    This provides gradient computation for training MoE models on Apple Silicon.
    """
    
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
        m_sizes: torch.Tensor,
        topk: int,
        gather_indices: torch.Tensor,
        permute_x: bool,
        permute_y: bool,
        topk_weights: Optional[torch.Tensor],
        fuse_mul_post: bool,
    ) -> torch.Tensor:
        ctx.topk = topk
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.fuse_mul_post = fuse_mul_post
        
        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        
        return grouped_gemm_mps_forward(
            X=X,
            W=W,
            topk=topk,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            topk_weights=topk_weights,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
        )
    
    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        permute_x = ctx.permute_x
        permute_y = ctx.permute_y
        fuse_mul_post = ctx.fuse_mul_post
        
        assert not fuse_mul_post, "fuse_mul_post should only be used for inference"
        
        # Compute dW
        dW = grouped_gemm_mps_dW(
            X=X,
            dY=dY,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
        )
        
        # Compute dX
        dX = grouped_gemm_mps_dX(
            dY=dY,
            W=W,
            gather_indices=gather_indices,
            m_sizes=m_sizes,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
        )
        
        # If permute_x was used, we need to unpermute dX and THEN sum
        if permute_x:
            # dX currently has size (total_tokens, K) and is in expert order
            dX_unperm = unpermute(dX, gather_indices)
            dX = dX_unperm.view(X.shape[0], topk, -1).sum(dim=1)
        
        return (
            dX,       # X
            dW,       # W
            None,     # m_sizes
            None,     # topk
            None,     # gather_indices
            None,     # permute_x
            None,     # permute_y
            None,     # topk_weights
            None,     # fuse_mul_post
        )


def grouped_gemm_mps(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: Optional[torch.Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
    fuse_mul_post: bool = False,
) -> torch.Tensor:
    """
    Main entry point for MPS grouped GEMM with automatic gradient computation.
    
    This function wraps the forward pass in an autograd Function to enable
    gradient computation during training.
    
    Args:
        X: Input hidden states
        W: Expert weights
        m_sizes: Tokens per expert
        topk: Experts per token
        gather_indices: Token-to-expert mapping
        permute_x: Permute input to expert order
        permute_y: Permute output to token order
        topk_weights: Weights for fused multiplication
        fuse_mul_post: Fuse topk weight multiplication
    
    Returns:
        Output tensor
    """
    if permute_x or permute_y:
        assert gather_indices is not None, "gather_indices required when permuting"
    
    if fuse_mul_post:
        assert topk_weights is not None, "topk_weights required for fuse_mul_post"
    
    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    if gather_indices is not None:
        gather_indices = gather_indices.view(-1)
    
    return GroupedGemmMPS.apply(
        X,
        W,
        m_sizes,
        topk,
        gather_indices,
        permute_x,
        permute_y,
        topk_weights,
        fuse_mul_post,
    )
