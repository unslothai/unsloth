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
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    row_position  = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # TODO: Fixup int32 locations to int64
    rot_position = row_position % seqlen

    Q   += row_position*  Q_row_stride + head_position*head_dim
    cos += rot_position*cos_row_stride
    sin += rot_position*sin_row_stride

    Q1   = tl.load(Q   + half_head_dim*0 + col_offsets, mask = mask, other = 0)
    sin1 = tl.load(sin + half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + half_head_dim*0 + col_offsets, mask = mask, other = 0)

    Q2   = tl.load(Q   + half_head_dim*1 + col_offsets, mask = mask, other = 0)
    # RoPE repeats sin and cos so 128 = [64, 64].

    if BACKWARD_PASS:
        """
            Q * cos + rotate_half(Q) * sin
            is equivalent to
            Q * cos + Q @ R * sin
            where R is a rotation matrix [ 0,  I]
                                         [-I,  0]
            dC/dY = dY * cos + dY @ R.T * sin
            where R.T is again the same  [ 0, -I]
            but the minus is transposed. [ I,  0]
        """
        sin1 = -sin1
    
    # RoPE repeats sin and cos so 128 = [64, 64].
    tl.store(Q + half_head_dim*0 + col_offsets, Q1*cos1 - Q2*sin1, mask = mask)
    tl.store(Q + half_head_dim*1 + col_offsets, Q2*cos1 + Q1*sin1, mask = mask)
pass


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim) # (head_dim//2)
        _rope_embedding[(n_rows, n_heads,)](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.cos = cos # Don't need save_for_backward since a view
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)
    pass

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        # Cannot be .view since the problem lies with dK since
        # K.T's strides are incorrect.
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        _rope_embedding[(n_rows, n_heads,)](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None,
    pass
pass


def fast_rope_embedding(Q, K, cos, sin):
    # We need (batch, [seqlen, n_heads], head_dim)
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    # We need (batch, [n_heads, seqlen], head_dim)
    return Q, K
pass


class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1]//2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        RH_Q *= sin
        Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q
    pass

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1]//2
        # We reverse the minus sign for R.T
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY *= cos
        RH_dY *= sin
        dY += RH_dY
        return dY, None, None, None
    pass
pass


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    return Q, K
pass
