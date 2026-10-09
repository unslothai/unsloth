# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import triton
import triton.language as tl
import torch
from ..device_type import DEVICE_COUNT
from typing import Tuple
from .utils import calculate_settings, torch_gpu_device, torch_device_stream
from .rms_layernorm import (
    _BF16_TRACEABLE,
    _TRACEABLE,
    _eager_kernel,
    _tag_compile_cache,
    _traced_kernel,
)


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
    LONG_INDEXING: tl.constexpr,
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
    if LONG_INDEXING:
        # Widen after the int32 div/mod: 64-bit div/mod is slow at normal sizes.
        batch_id = batch_id.to(tl.int64)
        seq_index = seq_index.to(tl.int64)
        head_position = head_position.to(tl.int64)

    q_ptr = Q + batch_id * Q_batch_stride + head_position * Q_head_stride + seq_index * Q_seq_stride
    q0 = tl.load(q_ptr + col_offsets, mask = mask, other = 0)
    q1 = tl.load(q_ptr + half_head_dim + col_offsets, mask = mask, other = 0)
    tl.store(q_ptr + col_offsets, q0 * cos1 - q1 * sin1, mask = mask)
    tl.store(q_ptr + half_head_dim + col_offsets, q1 * cos1 + q0 * sin1, mask = mask)

    if head_position < n_heads_K:
        k_ptr = (
            K + batch_id * K_batch_stride + head_position * K_head_stride + seq_index * K_seq_stride
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
        "LONG_INDEXING": lambda args: bool(args["LONG_INDEXING"]),
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
        sin + (row_position % seqlen) * sin_row_stride + half_head_dim * 0 + col_offsets,
        mask = mask,
        other = 0,
    )
    cos1 = tl.load(
        cos + (row_position % seqlen) * cos_row_stride + half_head_dim * 0 + col_offsets,
        mask = mask,
        other = 0,
    )

    if BACKWARD_PASS:
        # See the Unsloth blog post for more info.
        sin1 = -sin1

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% faster kernel from HuyNguyen-hust, unslothai/unsloth#238.
    Q += row_position.to(tl.int64) * Q_row_stride
    for k in range(head_start, head_end):
        offs_q1 = k * head_dim + col_offsets
        offs_q2 = k * head_dim + col_offsets + half_head_dim

        # Gemma sometimes needs RoPE done in float32 rather than bfloat16.
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


def _rope_rows(Q, cos, sin, seq_len, n_heads, head_dim, backward, wrap):
    # Rotates Q [batch * seq_len, n_heads * head_dim] in place; BLOCK_SIZE = head_dim//2 was racy.
    BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
    n_groups = (
        n_heads + ROPE_GROUP_SIZE - 1
    ) // ROPE_GROUP_SIZE  # also a SymInt under dynamic=True
    wrap(_rope_embedding)[
        (
            Q.shape[0],
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
        BACKWARD_PASS = backward,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )


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
        assert seq_len <= cos.shape[0]
        with torch_gpu_device(Q.device):
            _rope_rows(Q, cos, sin, seq_len, n_heads, head_dim, False, _eager_kernel)
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
        # The in-place kernel needs a private contiguous buffer, even for already-contiguous grads.
        dY = dY.clone(memory_format = torch.contiguous_format).view(
            batch * seq_len, n_heads * head_dim
        )
        with torch_gpu_device(dY.device):
            _rope_rows(dY, ctx.cos, ctx.sin, seq_len, n_heads, head_dim, True, _eager_kernel)
        dY = dY.reshape(batch, seq_len, n_heads, head_dim)
        return (
            dY,
            None,
            None,
        )


def _max_offset(t):
    return sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))


def _rope_qk(Q, K, cos, sin, rope_ptr, has_indices, backward, wrap):
    # Rotates Q [batch, n_heads_Q, seq_len, head_dim] and K in place, at any strides.
    batch, n_heads_Q, seq_len, head_dim = Q.shape
    n_heads_K = K.shape[1]
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    wrap(_rope_embedding_QK)[(batch * seq_len, n_heads_Q)](
        Q,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        rope_ptr,
        seq_len,
        head_dim = head_dim,
        n_heads_K = n_heads_K,
        BACKWARD_PASS = backward,
        HAS_ROPE_INDICES = has_indices,
        LONG_INDEXING = max(_max_offset(Q), _max_offset(K)) + BLOCK_SIZE >= 2**31,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )


class Fast_RoPE_Embedding_QK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        has_indices = rope_indices is not None
        cos, sin = cos.squeeze(), sin.squeeze()

        # Inplace rotary embedding is generally fine.
        Q_out = Q.clone() if not Q.is_contiguous() else Q
        K_out = K.clone() if not K.is_contiguous() else K

        if has_indices:
            # TRL's rotary indices are always int32, so the cast is only for safety.
            rope_ptr = rope_indices.reshape(-1).to(dtype = torch.int32, device = Q.device)
        else:
            rope_ptr = cos.new_empty(1, dtype = torch.int32)

        with torch_gpu_device(Q.device):
            _rope_qk(Q_out, K_out, cos, sin, rope_ptr, has_indices, False, _eager_kernel)

        ctx.has_indices = has_indices
        ctx.cos = cos
        ctx.sin = sin
        ctx.rope_indices = rope_ptr if has_indices else None

        return (
            Q_out,
            K_out,
        )

    @staticmethod
    def backward(ctx, dQ, dK):
        rope_ptr = ctx.rope_indices if ctx.has_indices else ctx.cos.new_empty(1, dtype = torch.int32)

        # Contiguous gradients can still share storage with each other or another branch.
        dQ_out = dQ.clone(memory_format = torch.contiguous_format)
        dK_out = dK.clone(memory_format = torch.contiguous_format)

        with torch_gpu_device(dQ.device):
            _rope_qk(
                dQ_out, dK_out, ctx.cos, ctx.sin, rope_ptr, ctx.has_indices, True, _eager_kernel
            )

        return (dQ_out, dK_out, None, None, None)


# Compiled graphs take these ops: the same kernels on fresh copies (the Functions rotate in place).
if _TRACEABLE:

    @torch.library.triton_op("unsloth::rope_embedding", mutates_args = ())
    def _rope_embedding_op(
        Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backward: bool
    ) -> torch.Tensor:
        # Q: [batch, seq_len, n_heads, head_dim] at any strides.
        batch, seq_len, n_heads, head_dim = Q.shape
        out = Q.clone(memory_format = torch.contiguous_format)
        rows = out.view(batch * seq_len, n_heads * head_dim)
        _rope_rows(rows, cos, sin, seq_len, n_heads, head_dim, backward, _traced_kernel)
        return out

    def _rope_setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1], inputs[2])

    def _rope_embedding_op_backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        return torch.ops.unsloth.rope_embedding(dY, cos, sin, True), None, None, None

    _rope_embedding_op.register_autograd(
        _rope_embedding_op_backward, setup_context = _rope_setup_context
    )

    @torch.library.triton_op("unsloth::rope_embedding_qk", mutates_args = ())
    def _rope_embedding_qk_op(
        Q: torch.Tensor,
        K: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_indices: torch.Tensor,
        backward: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q_out, K_out = Q.clone(), K.clone()
        _rope_qk(Q_out, K_out, cos, sin, rope_indices, True, backward, _traced_kernel)
        return Q_out, K_out

    def _rope_qk_setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[2], inputs[3], inputs[4])

    def _rope_embedding_qk_op_backward(ctx, dQ, dK):
        cos, sin, rope_indices = ctx.saved_tensors
        dQ, dK = torch.ops.unsloth.rope_embedding_qk(dQ, dK, cos, sin, rope_indices, True)
        return dQ, dK, None, None, None, None

    _rope_embedding_qk_op.register_autograd(
        _rope_embedding_qk_op_backward, setup_context = _rope_qk_setup_context
    )
    _tag_compile_cache(__file__)


def _fast_rope_embedding_eager(Q, K, cos, sin, rope_embedding_indices):
    if rope_embedding_indices is not None:
        Q_out, K_out = Fast_RoPE_Embedding_QK.apply(Q, K, cos, sin, rope_embedding_indices)
    else:
        Q_out = Fast_RoPE_Embedding.apply(Q.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        K_out = Fast_RoPE_Embedding.apply(K.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
    if DEVICE_COUNT > 1:
        torch_device_stream(Q.device).synchronize()
    return Q_out, K_out


_fast_rope_embedding_untraced = torch.compiler.disable(_fast_rope_embedding_eager)


def fast_rope_embedding(
    Q,
    K,
    cos,
    sin,
    rope_embedding_indices = None,
):
    if not torch.compiler.is_compiling():
        return _fast_rope_embedding_eager(Q, K, cos, sin, rope_embedding_indices)
    if (
        not _TRACEABLE
        or Q.device.type != "cuda"
        or (Q.dtype == torch.bfloat16 and not _BF16_TRACEABLE)
    ):
        return _fast_rope_embedding_untraced(Q, K, cos, sin, rope_embedding_indices)
    cos, sin = cos.squeeze(), sin.squeeze()
    if rope_embedding_indices is not None:
        rope_ptr = rope_embedding_indices.reshape(-1).to(dtype = torch.int32, device = Q.device)
        return torch.ops.unsloth.rope_embedding_qk(Q, K, cos, sin, rope_ptr, False)
    Q_out = torch.ops.unsloth.rope_embedding(Q.transpose(1, 2), cos, sin, False).transpose(1, 2)
    K_out = torch.ops.unsloth.rope_embedding(K.transpose(1, 2), cos, sin, False).transpose(1, 2)
    return Q_out, K_out


class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so squeeze them.
            cos = cos.squeeze(1).squeeze(0)
            sin = sin.squeeze(1).squeeze(0)
            cos = cos[position_ids].unsqueeze(2)
            sin = sin[position_ids].unsqueeze(2)

        # Q * cos + rotate_half(Q) * sin, with the transposed rotate_half in the backward.
        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        ctx.save_for_backward(cos, sin)
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1] // 2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY = dY.clone()
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        return dY, None, None, None


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    torch_device_stream(Q.device).synchronize()
    return Q, K
