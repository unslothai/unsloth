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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    # Matches Triton rotate_half: [q0, q1] -> [-q1, q0]
    shape = x.shape
    half = shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim = -1)


class MPSRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Q: [batch, seq_len, n_heads, head_dim] or [batch, n_heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]

        # Reshape cos/sin to broadcast over Q
        # RoPE usually applies to the last dimension (head_dim)
        # We assume cos/sin are [seq_len, head_dim/2] or [seq_len, head_dim]
        # Standard Unsloth RoPE kernels expect cos/sin [seq_len, head_dim/2]

        # In Unsloth, cos/sin are usually pre-expanded or have shape [seq_len, head_dim//2]
        # Let's handle the broadcast carefully.

        # If Q is [B, S, H, D]
        if Q.dim() == 4:
            # Check if S is the 1st or 2nd dim
            if Q.shape[1] == cos.shape[0]:  # [B, S, H, D]
                cos_final = cos.view(1, cos.shape[0], 1, cos.shape[1])
                sin_final = sin.view(1, sin.shape[0], 1, sin.shape[1])
            else:  # [B, H, S, D]
                cos_final = cos.view(1, 1, cos.shape[0], cos.shape[1])
                sin_final = sin.view(1, 1, sin.shape[0], sin.shape[1])
        else:
            # Fallback for other dims
            cos_final = cos
            sin_final = sin

        # If head_dim in cos/sin is half of Q's head_dim, we repeat it
        if cos_final.shape[-1] * 2 == Q.shape[-1]:
            cos_final = torch.cat((cos_final, cos_final), dim = -1)
            sin_final = torch.cat((sin_final, sin_final), dim = -1)

        Q_rotated = rotate_half(Q)
        Y = (Q * cos_final) + (Q_rotated * sin_final)

        ctx.save_for_backward(cos_final, sin_final)
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        cos, sin = ctx.saved_tensors

        # dQ = dY * cos - rotate_half(dY) * sin
        dY_rotated = rotate_half(dY)
        dQ = (dY * cos) - (dY_rotated * sin)

        return dQ, None, None


class MPSRoPEEmbeddingQK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ):
        # Similar logic to MPSRoPEEmbedding but for both Q and K
        # Typically Q is [B, H_q, S, D] and K is [B, H_k, S, D]

        # Batch, Heads, Seq, Dim
        # Typical Q shapes:
        # [Batch, Heads, Seq, Dim] -> Unsloth usually uses this (transposed)
        # [Batch, Seq, Heads, Dim] -> HF mostly uses this
        
        # Check input dims
        if Q.dim() == 4:
            # Check if S is dim 1 or dim 2
            # cos shape is [Seq, Dim] usually
            seq_len = cos.shape[0]
            
            if Q.shape[1] == seq_len: # [B, S, H, D]
                cos_final = cos.view(1, seq_len, 1, -1)
                sin_final = sin.view(1, seq_len, 1, -1)
            elif Q.shape[2] == seq_len: # [B, H, S, D]
                cos_final = cos.view(1, 1, seq_len, -1)
                sin_final = sin.view(1, 1, seq_len, -1)
            else:
                # Fallback or weird shape?
                # Maybe seq_len is different due to padding?
                # For now assume broadcasting works if we just view 1s
                # Try to align with the dimension that matches cos.shape[0]
                if Q.shape[1] == cos.shape[0]:
                     cos_final = cos.view(1, cos.shape[0], 1, -1)
                     sin_final = sin.view(1, sin.shape[0], 1, -1)
                else: 
                     cos_final = cos.view(1, 1, cos.shape[0], -1)
                     sin_final = sin.view(1, 1, sin.shape[0], -1)
        else:
             cos_final = cos
             sin_final = sin

        if cos_final.shape[-1] * 2 == Q.shape[-1]:
            cos_final = torch.cat((cos_final, cos_final), dim = -1)
            sin_final = torch.cat((sin_final, sin_final), dim = -1)

        Q_rotated = rotate_half(Q)
        Q_out = (Q * cos_final) + (Q_rotated * sin_final)

        K_rotated = rotate_half(K)
        K_out = (K * cos_final) + (K_rotated * sin_final)

        ctx.save_for_backward(cos_final, sin_final)
        return Q_out, K_out

    @staticmethod
    def backward(ctx, dQ: torch.Tensor, dK: torch.Tensor):
        cos, sin = ctx.saved_tensors

        dQ_rotated = rotate_half(dQ)
        dQ_out = (dQ * cos) - (dQ_rotated * sin)

        dK_rotated = rotate_half(dK)
        dK_out = (dK * cos) - (dK_rotated * sin)

        return dQ_out, dK_out, None, None


def mps_rope_embedding(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return MPSRoPEEmbedding.apply(Q, cos, sin)


def mps_rope_embedding_qk(
    Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    return MPSRoPEEmbeddingQK.apply(Q, K, cos, sin)
