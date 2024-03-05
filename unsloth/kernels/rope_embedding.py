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

import triton
import triton.language as tl
import torch
from .utils import calculate_settings


@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q,
    Q_row_stride,
    cos,
    cos_row_stride,
    sin, 
    sin_row_stride,
    seqlen, 
    head_dim,
    scaled_head_dim, # Added argument for the number of dimensions to scale
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly with partial scaling
        RoPE is Q * cos + rotate_half(Q) * sin for the scaled part of Q
        Non-scaled part of Q remains unchanged
        See our blog post for more info
    """
    row_position  = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < scaled_head_dim # Adjust mask to apply RoPE only to scaled dimensions

    # Load Q1, Q2, sin1, cos1 as before, but now the operation is conditional on scaled_head_dim
    Q1   = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + col_offsets, mask = mask, other = 0)
    Q2   = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + (head_dim//2) + col_offsets, mask = mask, other = 0)
    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        sin1 = -sin1

    # Apply RoPE transformation only to the scaled part of Q
    tl.store(Q + row_position*Q_row_stride + head_position*head_dim + col_offsets, Q1*cos1 - Q2*sin1, mask = mask)
    tl.store(Q + row_position*Q_row_stride + head_position*head_dim + (head_dim//2) + col_offsets, Q2*cos1 + Q1*sin1, mask = mask)
pass 



class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, partial_rotary_factor: float = 1.0):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        # Calculate scaled_head_dim based on partial_rotary_factor
        scaled_head_dim = int(head_dim * partial_rotary_factor)
        Q = Q.reshape(batch*seq_len, n_heads*head_dim)

        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)
        # Pass scaled_head_dim to the kernel
        _rope_embedding[(n_rows, n_heads,)](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, scaled_head_dim,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.cos = cos
        ctx.sin = sin
        ctx.scaled_head_dim = scaled_head_dim  # Store scaled_head_dim for backward pass
        ctx.head_dim = head_dim  # Also store the original head_dim for reshaping in backward
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, _ = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*ctx.head_dim)
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin
        scaled_head_dim = ctx.scaled_head_dim  # Retrieve scaled_head_dim for backward pass
        head_dim = ctx.head_dim  # Retrieve original head_dim for reshaping in backward

        _rope_embedding[(n_rows, n_heads,)](
            dY,  dY.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, scaled_head_dim,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, ctx.head_dim)
        return dY, None, None, None


class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids, partial_rope_factor: float = 1.0):
        if position_ids is not None:
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        # Calculate the number of dimensions to apply RoPE based on partial_rope_factor
        full_dim = Q.shape[-1]
        rope_dim = int(full_dim * partial_rope_factor)
        half_rope_dim = rope_dim // 2

        # Apply RoPE to the portion of Q determined by partial_rope_factor
        Q_partial = Q[..., :rope_dim]
        RH_Q_partial = torch.cat((-Q_partial[..., half_rope_dim:], Q_partial[..., :half_rope_dim]), dim=-1)
        Q_partial *= cos[..., :rope_dim]
        Q_partial.addcmul_(RH_Q_partial, sin[..., :rope_dim])

        # Combine the scaled and unscaled parts
        Q = torch.cat((Q_partial, Q[..., rope_dim:]), dim=-1)

        ctx.save_for_backward(cos, sin, torch.tensor([partial_rope_factor]))
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin, partial_rope_factor_tensor = ctx.saved_tensors
        partial_rope_factor = partial_rope_factor_tensor.item()

        # Calculate the number of dimensions to apply RoPE based on partial_rope_factor
        full_dim = dY.shape[-1]
        rope_dim = int(full_dim * partial_rope_factor)
        half_rope_dim = rope_dim // 2

        # Apply RoPE to the portion of dY determined by partial_rope_factor
        dY_partial = dY[..., :rope_dim]
        RH_dY_partial = torch.cat((dY_partial[..., half_rope_dim:], -dY_partial[..., :half_rope_dim]), dim=-1)
        dY_partial *= cos[..., :rope_dim]
        dY_partial.addcmul_(RH_dY_partial, sin[..., :rope_dim])

        # Combine the scaled and unscaled parts
        dY = torch.cat((dY_partial, dY[..., rope_dim:]), dim=-1)

        return dY, None, None, None, None



def inplace_rope_embedding(Q, K, cos, sin, position_ids, partial_rope_factor=1.0):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids, partial_rope_factor)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids, partial_rope_factor)
    return Q, K
pass

def fast_rope_embedding(Q, K, cos, sin, partial_rope_factor=1.0):
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin, partial_rope_factor).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin, partial_rope_factor).transpose(1, 2)
    return Q, K
pass