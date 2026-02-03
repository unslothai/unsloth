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
    return torch.cat((-x2, x1), dim=-1)


class MPSRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        # Q: [batch, n_heads, seq_len, head_dim] or [batch, seq_len, n_heads, head_dim]
        # cos, sin: [seq_len, head_dim] or [batch, seq_len, head_dim]

        if position_ids is not None:
            # Handle sliced RoPE (Generation / Prefill with specific positions)
            # cos/sin are likely [MaxSeq, HalfDim]
            # position_ids is [Batch, Seq] or [Seq]
            c = cos[position_ids]  # [B, S, D/2] or [S, D/2]
            s = sin[position_ids]  # [B, S, D/2] or [S, D/2]

            if c.dim() == 2:
                c = c.unsqueeze(0)  # [1, S, D/2]
            if s.dim() == 2:
                s = s.unsqueeze(0)  # [1, S, D/2]

            # Align with Q: typically Q is [B, H, S, D] on MPS
            if Q.dim() == 4:
                S = c.shape[1]
                if Q.shape[2] == S:  # [B, H, S, D]
                    cos_final = c.unsqueeze(1)  # [B, 1, S, D/2]
                    sin_final = s.unsqueeze(1)  # [B, 1, S, D/2]
                elif Q.shape[1] == S:  # [B, S, H, D]
                    cos_final = c.unsqueeze(2)  # [B, S, 1, D/2]
                    sin_final = s.unsqueeze(2)  # [B, S, 1, D/2]
                else:
                    # Fallback
                    cos_final = c.unsqueeze(1)
                    sin_final = s.unsqueeze(1)
            else:
                cos_final = c
                sin_final = s
        else:
            # Full sequence RoPE
            if Q.dim() == 4:
                seq_len = cos.shape[0]
                if Q.shape[2] == seq_len:  # [B, H, S, D]
                    cos_final = cos.view(1, 1, seq_len, -1)
                    sin_final = sin.view(1, 1, seq_len, -1)
                elif Q.shape[1] == seq_len:  # [B, S, H, D]
                    cos_final = cos.view(1, seq_len, 1, -1)
                    sin_final = sin.view(1, seq_len, 1, -1)
                else:
                    # Fallback matching the first dimension of cos
                    if Q.shape[1] == cos.shape[0]:
                        cos_final = cos.view(1, cos.shape[0], 1, -1)
                        sin_final = sin.view(1, sin.shape[0], 1, -1)
                    else:
                        cos_final = cos.view(1, 1, cos.shape[0], -1)
                        sin_final = sin.view(1, 1, sin.shape[0], -1)
            else:
                cos_final = cos
                sin_final = sin

        # If head_dim in cos/sin is half of Q's head_dim, we repeat it
        if cos_final.shape[-1] * 2 == Q.shape[-1]:
            cos_final = torch.cat((cos_final, cos_final), dim=-1)
            sin_final = torch.cat((sin_final, sin_final), dim=-1)

        Q_rotated = rotate_half(Q)
        Y = (Q * cos_final) + (Q_rotated * sin_final)

        ctx.save_for_backward(cos_final, sin_final)
        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dY_rotated = rotate_half(dY)
        dQ = (dY * cos) - (dY_rotated * sin)
        return dQ, None, None, None


class MPSRoPEEmbeddingQK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        if position_ids is not None:
            # Handle sliced RoPE
            c = cos[position_ids]  # [B, S, D/2] or [S, D/2]
            s = sin[position_ids]  # [B, S, D/2] or [S, D/2]

            if c.dim() == 2:
                c = c.unsqueeze(0)  # [1, S, D/2]
            if s.dim() == 2:
                s = s.unsqueeze(0)  # [1, S, D/2]

            if Q.dim() == 4:
                S = c.shape[1]
                if Q.shape[2] == S:  # [B, H, S, D]
                    cos_final = c.unsqueeze(1)  # [B, 1, S, D/2]
                    sin_final = s.unsqueeze(1)  # [B, 1, S, D/2]
                elif Q.shape[1] == S:  # [B, S, H, D]
                    cos_final = c.unsqueeze(2)  # [B, S, 1, D/2]
                    sin_final = s.unsqueeze(2)  # [B, S, 1, D/2]
                else:
                    cos_final = c.unsqueeze(1)
                    sin_final = s.unsqueeze(1)
            else:
                cos_final = c
                sin_final = s
        else:
            if Q.dim() == 4:
                seq_len = cos.shape[0]
                if Q.shape[2] == seq_len:  # [B, H, S, D]
                    cos_final = cos.view(1, 1, seq_len, -1)
                    sin_final = sin.view(1, 1, seq_len, -1)
                elif Q.shape[1] == seq_len:  # [B, S, H, D]
                    cos_final = cos.view(1, seq_len, 1, -1)
                    sin_final = sin.view(1, seq_len, 1, -1)
                else:
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
            cos_final = torch.cat((cos_final, cos_final), dim=-1)
            sin_final = torch.cat((sin_final, sin_final), dim=-1)

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
        return dQ_out, dK_out, None, None, None


def mps_rope_embedding(
    Q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
):
    return MPSRoPEEmbedding.apply(Q, cos, sin, position_ids)


def mps_rope_embedding_qk(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
):
    return MPSRoPEEmbeddingQK.apply(Q, K, cos, sin, position_ids)
