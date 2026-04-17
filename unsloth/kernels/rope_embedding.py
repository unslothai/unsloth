# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import triton
import triton.language as tl
import torch
from ..device_type import DEVICE_COUNT
from .utils import calculate_settings, torch_gpu_device, torch_device_stream


def _rope_embedding_QK(
    Q,
    Q_batch_stride,
    Q_head_stride,
    Q_seq_stride,
    K,
    K_batch_stride,
    K_head_stride,
    K_seq_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    rope_embedding_indices,
    seqlen,
    head_dim: tl.constexpr,
    n_heads_K: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    if HAS_ROPE_INDICES:
        rot_position = tl.load(
            rope_embedding_indices + row_position,
            eviction_policy = "evict_first",
        ).to(tl.int32)
    else:
        rot_position = row_position % seqlen

    cos_ptr = cos + rot_position * cos_row_stride
    sin_ptr = sin + rot_position * sin_row_stride
    sin1 = tl.load(
        sin_ptr + col_offsets,
        mask = mask,
        other = 0,
    )
    cos1 = tl.load(
        cos_ptr + col_offsets,
        mask = mask,
        other = 0,
    )
    if BACKWARD_PASS:
        sin1 = -sin1

    batch_id = row_position // seqlen
    seq_index = row_position - batch_id * seqlen

    q_ptr = (
        Q
        + batch_id * Q_batch_stride
        + head_position * Q_head_stride
        + seq_index * Q_seq_stride
    )
    q0 = tl.load(q_ptr + col_offsets, mask = mask, other = 0)
    q1 = tl.load(q_ptr + half_head_dim + col_offsets, mask = mask, other = 0)
    tl.store(q_ptr + col_offsets, q0 * cos1 - q1 * sin1, mask = mask)
    tl.store(q_ptr + half_head_dim + col_offsets, q1 * cos1 + q0 * sin1, mask = mask)

    if head_position < n_heads_K:
        k_ptr = (
            K
            + batch_id * K_batch_stride
            + head_position * K_head_stride
            + seq_index * K_seq_stride
        )
        k0 = tl.load(k_ptr + col_offsets, mask = mask, other = 0)
        k1 = tl.load(k_ptr + half_head_dim + col_offsets, mask = mask, other = 0)
        tl.store(k_ptr + col_offsets, k0 * cos1 - k1 * sin1, mask = mask)
        tl.store(k_ptr + half_head_dim + col_offsets, k1 * cos1 + k0 * sin1, mask = mask)


_rope_embedding_QK = triton.jit(_rope_embedding_QK)
_rope_embedding_QK = triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: bool(args["BACKWARD_PASS"]),
        "HAS_ROPE_INDICES": lambda args: bool(args["HAS_ROPE_INDICES"]),
    }
)(_rope_embedding_QK)


ROPE_GROUP_SIZE: int = 4


def _rope_embedding(
    Q,
    Q_row_stride: tl.constexpr,
    cos,
    cos_row_stride: tl.constexpr,
    sin,
    sin_row_stride: tl.constexpr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calculates the RoPE Embedding quickly
    RoPE is Q * cos + rotate_half(Q) * sin
    See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(
        sin
        + (row_position % seqlen) * sin_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask = mask,
        other = 0,
    )
    cos1 = tl.load(
        cos
        + (row_position % seqlen) * cos_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask = mask,
        other = 0,
    )

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (
            row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        )

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask = mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask = mask)


_rope_embedding = triton.jit(_rope_embedding)
_rope_embedding = triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: bool(args["BACKWARD_PASS"]),
    }
)(_rope_embedding)


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch: int
        seq_len: int
        n_heads: int
        head_dim: int
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = Q.shape
        assert seq_len <= cos.shape[0]

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)  # (head_dim//2)

        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div: int
        mod: int
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups: int = div + (mod != 0)

        with torch_gpu_device(Q.device):
            _rope_embedding[
                (
                    n_rows,
                    n_groups,
                )
            ](
                Q,
                Q.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS = False,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch: int
        seq_len: int
        n_heads: int
        head_dim: int
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch * seq_len, n_heads * head_dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        with torch_gpu_device(dY.device):
            _rope_embedding[
                (
                    n_rows,
                    ctx.n_groups,
                )
            ](
                dY,
                dY.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS = True,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps = ctx.num_warps,
            )
        dY = dY.reshape(batch, seq_len, n_heads, head_dim)
        return (
            dY,
            None,
            None,
        )


# [TODO] Unsure why RoPE Embedding is not torch.compiling properly
@torch.compiler.disable
def fast_rope_embedding(
    Q,
    K,
    cos,
    sin,
    rope_embedding_indices = None,
):
    if rope_embedding_indices is not None:
        Q_out, K_out = Fast_RoPE_Embedding_QK.apply(
            Q, K, cos, sin, rope_embedding_indices
        )
    else:
        Q_out = Fast_RoPE_Embedding.apply(
            Q.transpose(1, 2).contiguous(), cos, sin
        ).transpose(1, 2)
        K_out = Fast_RoPE_Embedding.apply(
            K.transpose(1, 2).contiguous(), cos, sin
        ).transpose(1, 2)
    if DEVICE_COUNT > 1:
        torch_device_stream(Q.device).synchronize()
    return Q_out, K_out


class Fast_RoPE_Embedding_QK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        has_indices = rope_indices is not None
        cos, sin = cos.squeeze(), sin.squeeze()

        batch, n_heads_Q, seq_len, head_dim = Q.shape
        _, n_heads_K, _, _ = K.shape

        # Inplace rotary embedding is generally fine
        Q_out = Q.clone() if not Q.is_contiguous() else Q
        K_out = K.clone() if not K.is_contiguous() else K

        if has_indices:
            # TRL's rotary indices are always in int32, so casting is just for safety
            rope_ptr = rope_indices.reshape(-1).to(dtype = torch.int32, device = Q.device)
        else:
            rope_ptr = cos.new_empty(1, dtype = torch.int32)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        Q_batch_stride, Q_head_stride, Q_seq_stride = (
            Q_out.stride(0),
            Q_out.stride(1),
            Q_out.stride(2),
        )
        K_batch_stride, K_head_stride, K_seq_stride = (
            K_out.stride(0),
            K_out.stride(1),
            K_out.stride(2),
        )

        with torch_gpu_device(Q.device):
            _rope_embedding_QK[(batch * seq_len, n_heads_Q)](
                Q_out,
                Q_batch_stride,
                Q_head_stride,
                Q_seq_stride,
                K_out,
                K_batch_stride,
                K_head_stride,
                K_seq_stride,
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                rope_ptr,
                seq_len,
                head_dim = head_dim,
                n_heads_K = n_heads_K,
                BACKWARD_PASS = False,
                HAS_ROPE_INDICES = has_indices,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )

        ctx.block_size = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.has_indices = has_indices
        ctx.cos = cos
        ctx.sin = sin
        ctx.rope_indices = rope_ptr if has_indices else None
        ctx.seq_len = seq_len
        ctx.n_heads_Q = n_heads_Q
        ctx.n_heads_K = n_heads_K

        return (
            Q_out,
            K_out,
        )

    @staticmethod
    def backward(ctx, dQ, dK):
        batch, _, _, head_dim = dQ.shape

        rope_ptr = (
            ctx.rope_indices
            if ctx.has_indices
            else ctx.cos.new_empty(1, dtype = torch.int32)
        )

        # Inplace rotary embedding is generally fine
        dQ_out = dQ.clone() if not dQ.is_contiguous() else dQ
        dK_out = dK.clone() if not dK.is_contiguous() else dK

        Q_batch_stride, Q_head_stride, Q_seq_stride = (
            dQ_out.stride(0),
            dQ_out.stride(1),
            dQ_out.stride(2),
        )
        K_batch_stride, K_head_stride, K_seq_stride = (
            dK_out.stride(0),
            dK_out.stride(1),
            dK_out.stride(2),
        )

        with torch_gpu_device(dQ.device):
            _rope_embedding_QK[(batch * ctx.seq_len, ctx.n_heads_Q)](
                dQ_out,
                Q_batch_stride,
                Q_head_stride,
                Q_seq_stride,
                dK_out,
                K_batch_stride,
                K_head_stride,
                K_seq_stride,
                ctx.cos,
                ctx.cos.stride(0),
                ctx.sin,
                ctx.sin.stride(0),
                rope_ptr,
                ctx.seq_len,
                head_dim = head_dim,
                n_heads_K = ctx.n_heads_K,
                BACKWARD_PASS = True,
                HAS_ROPE_INDICES = ctx.has_indices,
                BLOCK_SIZE = ctx.block_size,
                num_warps = ctx.num_warps,
            )

        return (dQ_out, dK_out, None, None, None)


class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]

        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        # RH_Q *= sin
        # Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1] // 2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        # RH_dY *= sin
        # dY += RH_dY
        return dY, None, None, None


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    torch_device_stream(Q.device).synchronize()
    return Q, K
