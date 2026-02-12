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
Pure PyTorch LoRA implementations for MPS that work with gradient checkpointing.

Unlike the custom autograd functions (MPSLoRA_MLP, etc.), these implementations
use standard PyTorch operations that correctly handle gradient checkpointing
on MPS. This avoids the "element 0 of tensors does not require grad" error.
"""

import torch
import torch.nn.functional as F


def pytorch_lora_linear(X, W, A, B, scaling):
    """
    Pure PyTorch LoRA linear layer: Y = X @ W^T + scaling * X @ A^T @ B^T
    
    Args:
        X: Input tensor [batch, seq_len, hidden_dim]
        W: Base weight matrix [out_dim, hidden_dim]
        A: LoRA A matrix [rank, hidden_dim]
        B: LoRA B matrix [out_dim, rank]
        scaling: LoRA scaling factor (alpha / rank)
    
    Returns:
        Y: Output tensor [batch, seq_len, out_dim]
    """
    output = F.linear(X, W)
    
    if A is not None and B is not None:
        lora_output = F.linear(F.linear(X, A), B)
        output = output + scaling * lora_output
    
    return output


def pytorch_lora_mlp_swiglu(
    X,
    gateW, gateW_quant, gateA, gateB, gateS,
    upW, upW_quant, upA, upB, upS,
    downW, downW_quant, downA, downB, downS,
):
    """
    Pure PyTorch LoRA MLP with SwiGLU activation for MPS.
    
    This implementation avoids custom autograd functions to work correctly
    with gradient checkpointing on MPS.
    
    SwiGLU: output = swish(gate) * up
    where swish(x) = x * sigmoid(x)
    """
    # Gate projection with LoRA
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS)
    
    # Up projection with LoRA
    up = pytorch_lora_linear(X, upW, upA, upB, upS)
    
    # SwiGLU: swish(gate) * up
    # swish(x) = x * sigmoid(x)
    swish_gate = gate * torch.sigmoid(gate)
    hidden = swish_gate * up
    
    # Down projection with LoRA
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS)
    
    return output


def pytorch_lora_mlp_geglu_exact(
    X,
    gateW, gateW_quant, gateA, gateB, gateS,
    upW, upW_quant, upA, upB, upS,
    downW, downW_quant, downA, downB, downS,
):
    """
    Pure PyTorch LoRA MLP with exact GeGLU activation for MPS.
    
    GeGLU: output = gelu(gate) * up
    """
    # Gate projection with LoRA
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS)
    
    # Up projection with LoRA
    up = pytorch_lora_linear(X, upW, upA, upB, upS)
    
    # GeGLU: gelu(gate) * up
    gelu_gate = F.gelu(gate)
    hidden = gelu_gate * up
    
    # Down projection with LoRA
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS)
    
    return output


def pytorch_lora_mlp_geglu_approx(
    X,
    gateW, gateW_quant, gateA, gateB, gateS,
    upW, upW_quant, upA, upB, upS,
    downW, downW_quant, downA, downB, downS,
):
    """
    Pure PyTorch LoRA MLP with approximate GeGLU activation for MPS.
    
    Uses tanh approximation of GELU for faster computation.
    """
    # Gate projection with LoRA
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS)
    
    # Up projection with LoRA
    up = pytorch_lora_linear(X, upW, upA, upB, upS)
    
    # Approximate GeGLU using tanh
    # gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    cdf = 0.5 * (1.0 + torch.tanh(sqrt_2_over_pi * (gate + 0.044715 * torch.pow(gate, 3))))
    gelu_approx_gate = gate * cdf
    hidden = gelu_approx_gate * up
    
    # Down projection with LoRA
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS)
    
    return output


def pytorch_lora_qkv(
    X,
    QW, QW_quant, QA, QB, QS,
    KW, KW_quant, KA, KB, KS,
    VW, VW_quant, VA, VB, VS,
):
    """
    Pure PyTorch LoRA QKV projection for MPS.
    
    Projects X to Q, K, V with separate LoRA for each.
    """
    Q = pytorch_lora_linear(X, QW, QA, QB, QS)
    K = pytorch_lora_linear(X, KW, KA, KB, KS)
    V = pytorch_lora_linear(X, VW, VA, VB, VS)
    return Q, K, V


def pytorch_lora_o(
    X,
    OW, OW_quant, OA, OB, OS,
):
    """
    Pure PyTorch LoRA output projection for MPS.
    """
    return pytorch_lora_linear(X, OW, OA, OB, OS)
