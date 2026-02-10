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
MLX-based LoRA merge implementation for Apple Silicon.

This module provides an MLX-based alternative to CUDA's addmm_ operation
for merging LoRA weights on Apple Silicon Macs, enabling GGUF export without
requiring a CUDA machine.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .bridge import torch_to_mlx, mlx_to_torch
from .utils import is_mlx_available, UnslothMLXError


if is_mlx_available():
    import mlx.core as mx


def _validate_mlx():
    """Validate MLX is available."""
    if not is_mlx_available():
        raise UnslothMLXError(
            "MLX is not available. Install with: pip install mlx"
        )


def mlx_merge_lora(
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    s: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Merge LoRA weights into base weight using MLX.
    
    This replicates the PyTorch operation: W.t(); W.addmm_(A.t(), B.t(), alpha=s); W.t()
    which computes: W_merged = W + s * (A @ B.T).T
    
    Args:
        W: Base weight tensor [out_features, in_features] (transposed internally)
        A: LoRA A matrix [rank, in_features]
        B: LoRA B matrix [out_features, rank]
        s: LoRA scaling factor
        dtype: Target dtype for output
    
    Returns:
        Merged weight tensor of shape [out_features, in_features]
    """
    _validate_mlx()
    
    # In _merge_lora, W is transposed before calling this via addmm_
    # W shape: [out_features, in_features] -> after .t() in caller: [in_features, out_features]
    # A shape: [rank, in_features]
    # B shape: [out_features, rank]
    # Operation: W += s * A^T @ B^T
    # Where A^T: [in_features, rank], B^T: [rank, out_features]
    # Result: [in_features, rank] @ [rank, out_features] = [in_features, out_features]
    
    # Ensure inputs are contiguous and convert to MLX
    W_mlx = torch_to_mlx(W.contiguous().to(torch.float32))
    A_mlx = torch_to_mlx(A.contiguous().to(torch.float32))
    B_mlx = torch_to_mlx(B.contiguous().to(torch.float32))
    
    # Transpose A and B as per PyTorch addmm_ operation
    A_T = mx.transpose(A_mlx)  # [rank, in_features] -> [in_features, rank]
    B_T = mx.transpose(B_mlx)  # [out_features, rank] -> [rank, out_features]
    
    # Compute: W += s * A.T @ B.T
    dW = mx.matmul(A_T, B_T)  # [in_features, rank] @ [rank, out_features] = [in_features, out_features]
    W_merged_mlx = W_mlx + s * dW
    
    # Check for infinities using numpy conversion
    W_check = mx.array(W_merged_mlx)
    import numpy as np
    W_numpy = np.array(W_check)
    if not np.all(np.isfinite(W_numpy)):
        raise ValueError("LoRA merge produced infinite values")
    
    # Convert back to PyTorch
    W_merged = mlx_to_torch(W_merged_mlx, device=W.device, dtype=dtype)
    
    return W_merged


def mlx_merge_lora_layer(
    layer: nn.Module,
    quant_state: Optional[object] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Merge LoRA weights from a layer using MLX.
    
    This is the MLX equivalent of _merge_lora() in save.py.
    
    Args:
        layer: The LoRA layer (Peft_Linear4bit, Peft_Linear, etc.)
        quant_state: Quantization state if layer is quantized
        dtype: Target dtype for output weight
    
    Returns:
        Tuple of (merged_weight, bias)
    """
    _validate_mlx()
    
    from unsloth.kernels.utils import get_lora_parameters_bias, fast_dequantize
    
    # Get LoRA parameters
    W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)
    
    # Dequantize if needed
    if quant_state is not None:
        dtype = quant_state.dtype if type(quant_state) is not list else quant_state[2]
        W = fast_dequantize(W, quant_state)
    else:
        dtype = dtype or W.dtype
    
    # Transpose W as done in _merge_lora
    W = W.to(torch.float32).t()
    
    # Merge using MLX
    W_merged = mlx_merge_lora(W, A, B, s, dtype)
    
    # Transpose back
    W_merged = W_merged.t()
    
    return W_merged, bias
