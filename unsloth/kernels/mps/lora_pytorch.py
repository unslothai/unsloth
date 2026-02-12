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


def pytorch_lora_linear(X, W, A, B, scaling, W_quant=None):
    """
    Pure PyTorch LoRA linear layer: Y = X @ W^T + scaling * X @ A^T @ B^T
    
    Args:
        X: Input tensor [batch, seq_len, hidden_dim]
        W: Base weight matrix [out_dim, hidden_dim]
        A: LoRA A matrix [rank, hidden_dim]
        B: LoRA B matrix [out_dim, rank]
        scaling: LoRA scaling factor (alpha / rank)
        W_quant: Quantized weight info (ignored for non-quantized models)
    
    Returns:
        Y: Output tensor [batch, seq_len, out_dim]
    """
    dtype = X.dtype
    
    # Base linear: X @ W^T
    W_ = W.to(dtype) if W.dtype != dtype else W
    output = F.linear(X, W_)
    
    if A is not None and B is not None:
        # LoRA path: scaling * X @ A^T @ B^T
        # A is [rank, hidden], B is [out_dim, rank]
        # F.linear(X, A) computes X @ A^T -> [batch, seq, rank]
        # F.linear(XA, B) computes XA @ B^T -> [batch, seq, out_dim]
        lora_out = F.linear(F.linear(X, A.to(dtype)), B.to(dtype))
        output = output + scaling * lora_out
    
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
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS, gateW_quant)
    up = pytorch_lora_linear(X, upW, upA, upB, upS, upW_quant)
    swish_gate = gate * torch.sigmoid(gate)
    hidden = swish_gate * up
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS, downW_quant)
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
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS, gateW_quant)
    up = pytorch_lora_linear(X, upW, upA, upB, upS, upW_quant)
    gelu_gate = F.gelu(gate)
    hidden = gelu_gate * up
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS, downW_quant)
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
    gate = pytorch_lora_linear(X, gateW, gateA, gateB, gateS, gateW_quant)
    up = pytorch_lora_linear(X, upW, upA, upB, upS, upW_quant)
    sqrt_2_over_pi = 0.7978845608028654
    cdf = 0.5 * (1.0 + torch.tanh(sqrt_2_over_pi * (gate + 0.044715 * torch.pow(gate, 3))))
    gelu_approx_gate = gate * cdf
    hidden = gelu_approx_gate * up
    output = pytorch_lora_linear(hidden, downW, downA, downB, downS, downW_quant)
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
    Q = pytorch_lora_linear(X, QW, QA, QB, QS, QW_quant)
    K = pytorch_lora_linear(X, KW, KA, KB, KS, KW_quant)
    V = pytorch_lora_linear(X, VW, VA, VB, VS, VW_quant)
    return Q, K, V


def pytorch_lora_o(
    X,
    OW, OW_quant, OA, OB, OS,
):
    """
    Pure PyTorch LoRA output projection for MPS.
    """
    return pytorch_lora_linear(X, OW, OA, OB, OS, OW_quant)


def pytorch_rope_embedding_qk(Q, K, cos, sin):
    """
    Pure PyTorch RoPE embedding for Q and K tensors.
    
    RoPE: rotate_half(x) = [-x[..., d/2:], x[..., :d/2]]
    output = x * cos + rotate_half(x) * sin
    
    Args:
        Q: Query tensor [batch, n_heads, seq_len, head_dim]
        K: Key tensor [batch, n_heads, seq_len, head_dim]  
        cos: Cosine values [seq_len, head_dim] or [1, seq_len, head_dim]
        sin: Sine values [seq_len, head_dim] or [1, seq_len, head_dim]
    
    Returns:
        Q_out, K_out: Rotated tensors
    """
    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)
    
    seq_len = Q.shape[2]
    cos = cos[:seq_len].view(1, 1, seq_len, -1)
    sin = sin[:seq_len].view(1, 1, seq_len, -1)
    
    if cos.shape[-1] * 2 == Q.shape[-1]:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    
    Q_out = (Q * cos) + (rotate_half(Q) * sin)
    K_out = (K * cos) + (rotate_half(K) * sin)
    
    return Q_out, K_out
