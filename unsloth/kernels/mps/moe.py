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

"""
MoE (Mixture of Experts) MPS Kernel Dispatch.

This module provides MPS-compatible implementations for MoE operations,
specifically the grouped GEMM operations used in expert MLPs.

The core implementation lives in:
    unsloth/kernels/moe/grouped_gemm/mps/fallback.py

This module re-exports those functions for convenient access from the
standard MPS kernels location.
"""

from typing import Optional
import torch

# Re-export from the main implementation
from ..moe.grouped_gemm.mps.fallback import (
    grouped_gemm_mps_forward,
    grouped_gemm_mps_dX,
    grouped_gemm_mps_dW,
    grouped_gemm_mps,
    GroupedGemmMPS,
)

# Re-export reference ops that are MPS-compatible
from ..moe.grouped_gemm.reference.moe_ops import (
    permute,
    unpermute,
    calculate_topk,
    get_routing_indices,
    torch_grouped_gemm,
)

__all__ = [
    # Main grouped GEMM functions
    "grouped_gemm_mps",
    "grouped_gemm_mps_forward",
    "grouped_gemm_mps_dX",
    "grouped_gemm_mps_dW",
    "GroupedGemmMPS",
    # Reference ops
    "permute",
    "unpermute",
    "calculate_topk",
    "get_routing_indices",
    "torch_grouped_gemm",
    # Dispatch function
    "dispatch_grouped_gemm",
]


def dispatch_grouped_gemm(
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
    Dispatch grouped GEMM to the appropriate backend.
    
    On MPS, this uses the PyTorch fallback implementation.
    On CUDA, this routes to the Triton kernels via the main interface.
    
    Args:
        X: Input hidden states, shape (num_tokens, hidden_dim) or (total_tokens, hidden_dim)
        W: Expert weights, shape (num_experts, output_dim, hidden_dim)
        m_sizes: Tokens assigned to each expert, shape (num_experts,)
        topk: Number of experts selected per token
        gather_indices: Token-to-expert mapping, shape (total_tokens,)
        permute_x: Whether to permute input from token order to expert-grouped order
        permute_y: Whether to permute output from expert-grouped order to token order
        topk_weights: Weights for fused multiplication (inference only)
        fuse_mul_post: Whether to fuse topk weight multiplication (inference only)
    
    Returns:
        Output tensor after grouped GEMM
    """
    # Use the interface which handles device dispatch automatically
    from ..moe.grouped_gemm.interface import grouped_gemm
    
    return grouped_gemm(
        X=X,
        W=W,
        m_sizes=m_sizes,
        topk=topk,
        gather_indices=gather_indices,
        permute_x=permute_x,
        permute_y=permute_y,
        topk_weights=topk_weights,
        fuse_mul_post=fuse_mul_post,
    )
