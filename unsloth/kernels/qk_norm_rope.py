# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Per-head RMSNorm (Qwen3 q_norm / k_norm) followed by RoPE, as one forward and one backward
# launch for Q and K together. Outputs are always fresh tensors.

import os
import triton
import triton.language as tl
import torch
from typing import Tuple
from .utils import torch_gpu_device, DEVICE_TYPE
from .rms_layernorm import (
    _BF16_TRACEABLE,
    _TRACEABLE,
    _eager_kernel,
    _tag_compile_cache,
    _traced_kernel,
    fast_rms_layernorm,
)
from .rope_embedding import fast_rope_embedding, _sync_if_split, DECODE_FUSED_ROPE

__all__ = [
    "fast_qk_norm_rope",
    "QK_NORM_ROPE_ENABLED",
    "decode_qk_norm_rope",
]


def _qk_norm_rope_default():
    # UNSLOTH_FUSED_QKNORM_ROPE=0 keeps the unfused path, =1 forces the fused one. Unset: on for
    # NVIDIA CUDA; ROCm keeps the unfused training path until its backward is measured there.
    flag = os.environ.get("UNSLOTH_FUSED_QKNORM_ROPE", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return DEVICE_TYPE == "cuda"


QK_NORM_ROPE_ENABLED = _qk_norm_rope_default()


@triton.jit
def _norm_rope_fwd_block(
    Y,
    X,
    X_row_stride,
    X_head_stride,
    W,
    r,
    row,
    heads,
    n_heads,
    cos1,
    sin1,
    cols,
    eps,
    HEAD_DIM: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
):
    HALF = HEAD_DIM // 2
    hmask = heads < n_heads
    mask = hmask[:, None]
    x_ptr = X + row * X_row_stride + heads[:, None].to(tl.int64) * X_head_stride + cols[None, :]
    x0 = tl.load(x_ptr, mask = mask, other = 0).to(tl.float32)
    x1 = tl.load(x_ptr + HALF, mask = mask, other = 0).to(tl.float32)
    var = (tl.sum(x0 * x0, axis = 1) + tl.sum(x1 * x1, axis = 1)) / HEAD_DIM
    inv = tl.math.rsqrt(var + tl.full((), eps, tl.float32))
    tl.store(r + row * n_heads + heads, inv, mask = hmask)
    w0 = tl.load(W + cols)
    w1 = tl.load(W + HALF + cols)
    # Same roundings as fast_rms_layernorm then the RoPE kernel: normed in W's dtype, the norm output
    # in Y's dtype. The rotation matches the RoPE kernel it replaces: the indexed one multiplies the
    # norm output by cos / sin with type promotion (fp32 activations stay fp32 for bf16 tables), the
    # plain one casts to cos's dtype first. A no-op when the dtypes already match.
    y0 = ((x0 * inv[:, None]).to(w0.dtype) * w0[None, :]).to(Y.dtype.element_ty)
    y1 = ((x1 * inv[:, None]).to(w1.dtype) * w1[None, :]).to(Y.dtype.element_ty)
    if not HAS_ROPE_INDICES:
        y0 = y0.to(cos1.dtype)
        y1 = y1.to(cos1.dtype)
    y_ptr = Y + (row * n_heads + heads[:, None].to(tl.int64)) * HEAD_DIM + cols[None, :]
    tl.store(y_ptr, y0 * cos1[None, :] - y1 * sin1[None, :], mask = mask)
    tl.store(y_ptr + HALF, y1 * cos1[None, :] + y0 * sin1[None, :], mask = mask)


@triton.jit
def _qk_norm_rope_forward(
    Qo,
    Q,
    Q_row_stride,
    Q_head_stride,
    Wq,
    rq,
    n_heads_Q,
    eps_Q,
    Ko,
    K,
    K_row_stride,
    K_head_stride,
    Wk,
    rk,
    n_heads_K,
    eps_K,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    rope_indices,
    seqlen,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hblock = tl.program_id(1)
    n_blocks_Q = (n_heads_Q + BLOCK_H - 1) // BLOCK_H
    cols = tl.arange(0, HEAD_DIM // 2)
    if HAS_ROPE_INDICES:
        pos = tl.load(rope_indices + row).to(tl.int64)
    else:
        pos = row % seqlen
    cos1 = tl.load(cos + pos * cos_row_stride + cols)
    sin1 = tl.load(sin + pos * sin_row_stride + cols)
    if hblock < n_blocks_Q:
        heads = hblock * BLOCK_H + tl.arange(0, BLOCK_H)
        _norm_rope_fwd_block(
            Qo, Q, Q_row_stride, Q_head_stride, Wq, rq, row, heads, n_heads_Q,
            cos1, sin1, cols, eps_Q, HEAD_DIM, HAS_ROPE_INDICES,
        )  # fmt: skip
    else:
        heads = (hblock - n_blocks_Q) * BLOCK_H + tl.arange(0, BLOCK_H)
        _norm_rope_fwd_block(
            Ko, K, K_row_stride, K_head_stride, Wk, rk, row, heads, n_heads_K,
            cos1, sin1, cols, eps_K, HEAD_DIM, HAS_ROPE_INDICES,
        )  # fmt: skip


@triton.jit
def _norm_rope_bwd_block(
    dX,
    dY,
    dY_batch_stride,
    dY_seq_stride,
    dY_head_stride,
    X,
    X_row_stride,
    X_head_stride,
    W,
    r,
    row,
    batch_id,
    seq_index,
    heads,
    n_heads,
    cos1,
    sin1,
    cols,
    HEAD_DIM: tl.constexpr,
):
    HALF = HEAD_DIM // 2
    hmask = heads < n_heads
    mask = hmask[:, None]
    hoff = heads[:, None].to(tl.int64)
    dy_ptr = (
        dY
        + batch_id * dY_batch_stride
        + seq_index * dY_seq_stride
        + hoff * dY_head_stride
        + cols[None, :]
    )
    dy0 = tl.load(dy_ptr, mask = mask, other = 0)
    dy1 = tl.load(dy_ptr + HALF, mask = mask, other = 0)
    # Inverse rotation (sin negated), rounded to dY's dtype as the RoPE backward stores it.
    dn0 = (dy0 * cos1[None, :] + dy1 * sin1[None, :]).to(dY.dtype.element_ty).to(tl.float32)
    dn1 = (dy1 * cos1[None, :] - dy0 * sin1[None, :]).to(dY.dtype.element_ty).to(tl.float32)

    x_ptr = X + row * X_row_stride + hoff * X_head_stride + cols[None, :]
    x0 = tl.load(x_ptr, mask = mask, other = 0).to(tl.float32)
    x1 = tl.load(x_ptr + HALF, mask = mask, other = 0).to(tl.float32)
    inv = tl.load(r + row * n_heads + heads, mask = hmask, other = 0)
    w0 = tl.load(W + cols).to(tl.float32)
    w1 = tl.load(W + HALF + cols).to(tl.float32)
    n0 = x0 * inv[:, None]
    n1 = x1 * inv[:, None]
    g0 = dn0 * w0[None, :]
    g1 = dn1 * w1[None, :]
    rowsum = tl.sum(g0 * n0, axis = 1) + tl.sum(g1 * n1, axis = 1)
    scale = (inv / HEAD_DIM)[:, None]
    dx_ptr = dX + (row * n_heads + hoff) * HEAD_DIM + cols[None, :]
    # HEAD_DIM * g - n * rowsum with the contraction pinned (HEAD_DIM * g is exact): left to the backend
    # the eager and the Inductor build contract it differently on ROCm, and fp16 dX lands 1 ulp apart.
    tl.store(dx_ptr, scale * tl.fma(-n0, rowsum[:, None], HEAD_DIM * g0), mask = mask)
    tl.store(dx_ptr + HALF, scale * tl.fma(-n1, rowsum[:, None], HEAD_DIM * g1), mask = mask)


@triton.jit
def _qk_norm_rope_backward(
    dQ,
    dQo,
    dQo_batch_stride,
    dQo_seq_stride,
    dQo_head_stride,
    Q,
    Q_row_stride,
    Q_head_stride,
    Wq,
    rq,
    n_heads_Q,
    dK,
    dKo,
    dKo_batch_stride,
    dKo_seq_stride,
    dKo_head_stride,
    K,
    K_row_stride,
    K_head_stride,
    Wk,
    rk,
    n_heads_K,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    rope_indices,
    seqlen,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hblock = tl.program_id(1)
    n_blocks_Q = (n_heads_Q + BLOCK_H - 1) // BLOCK_H
    cols = tl.arange(0, HEAD_DIM // 2)
    batch_id = row // seqlen
    seq_index = row - batch_id * seqlen
    if HAS_ROPE_INDICES:
        pos = tl.load(rope_indices + row).to(tl.int64)
    else:
        pos = seq_index
    cos1 = tl.load(cos + pos * cos_row_stride + cols)
    sin1 = tl.load(sin + pos * sin_row_stride + cols)
    if hblock < n_blocks_Q:
        heads = hblock * BLOCK_H + tl.arange(0, BLOCK_H)
        _norm_rope_bwd_block(
            dQ, dQo, dQo_batch_stride, dQo_seq_stride, dQo_head_stride, Q, Q_row_stride,
            Q_head_stride, Wq, rq, row, batch_id, seq_index, heads, n_heads_Q, cos1, sin1, cols,
            HEAD_DIM,
        )  # fmt: skip
    else:
        heads = (hblock - n_blocks_Q) * BLOCK_H + tl.arange(0, BLOCK_H)
        _norm_rope_bwd_block(
            dK, dKo, dKo_batch_stride, dKo_seq_stride, dKo_head_stride, K, K_row_stride,
            K_head_stride, Wk, rk, row, batch_id, seq_index, heads, n_heads_K, cos1, sin1, cols,
            HEAD_DIM,
        )  # fmt: skip


def _block_h(n_heads_Q, n_heads_K, head_dim):
    block = 1
    while block < min(n_heads_Q, n_heads_K) and block * 2 * head_dim <= 2048:
        block *= 2
    return block


def _num_warps(block_h, head_dim):
    return 8 if block_h * head_dim >= 4096 else 4


def _launch_forward(Q, K, Wq, Wk, cos, sin, rope_indices, eps_Q, eps_K, has_indices, wrap):
    # Q [batch, seq_len, n_heads_Q, head_dim], K [..., n_heads_K, head_dim]; rows = batch * seq_len.
    batch, seq_len, n_heads_Q, head_dim = Q.shape
    n_heads_K = K.shape[2]
    rows = batch * seq_len
    Q = Q.reshape(rows, n_heads_Q, head_dim)
    K = K.reshape(rows, n_heads_K, head_dim)
    Qo = torch.empty((batch, seq_len, n_heads_Q, head_dim), dtype = Q.dtype, device = Q.device)
    Ko = torch.empty((batch, seq_len, n_heads_K, head_dim), dtype = K.dtype, device = K.device)
    rq = torch.empty((rows, n_heads_Q), dtype = torch.float32, device = Q.device)
    rk = torch.empty((rows, n_heads_K), dtype = torch.float32, device = Q.device)
    BLOCK_H = _block_h(n_heads_Q, n_heads_K, head_dim)
    n_blocks = (n_heads_Q + BLOCK_H - 1) // BLOCK_H + (n_heads_K + BLOCK_H - 1) // BLOCK_H
    wrap(_qk_norm_rope_forward)[(rows, n_blocks)](
        Qo, Q, Q.stride(0), Q.stride(1), Wq, rq, n_heads_Q, eps_Q,
        Ko, K, K.stride(0), K.stride(1), Wk, rk, n_heads_K, eps_K,
        cos, cos.stride(0), sin, sin.stride(0), rope_indices, seq_len,
        HEAD_DIM = head_dim, BLOCK_H = BLOCK_H, HAS_ROPE_INDICES = has_indices,
        num_warps = _num_warps(BLOCK_H, head_dim), enable_fp_fusion = True,
    )  # fmt: skip
    return Qo, Ko, rq, rk


def _launch_backward(dQo, dKo, Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices, has_indices, wrap):
    batch, seq_len, n_heads_Q, head_dim = Q.shape
    n_heads_K = K.shape[2]
    rows = batch * seq_len
    Q = Q.reshape(rows, n_heads_Q, head_dim)
    K = K.reshape(rows, n_heads_K, head_dim)
    if dQo.stride(-1) != 1:
        dQo = dQo.contiguous()
    if dKo.stride(-1) != 1:
        dKo = dKo.contiguous()
    dQ = torch.empty((batch, seq_len, n_heads_Q, head_dim), dtype = dQo.dtype, device = Q.device)
    dK = torch.empty((batch, seq_len, n_heads_K, head_dim), dtype = dKo.dtype, device = Q.device)
    BLOCK_H = _block_h(n_heads_Q, n_heads_K, head_dim)
    n_blocks = (n_heads_Q + BLOCK_H - 1) // BLOCK_H + (n_heads_K + BLOCK_H - 1) // BLOCK_H
    wrap(_qk_norm_rope_backward)[(rows, n_blocks)](
        dQ, dQo, dQo.stride(0), dQo.stride(1), dQo.stride(2), Q, Q.stride(0), Q.stride(1), Wq, rq, n_heads_Q,
        dK, dKo, dKo.stride(0), dKo.stride(1), dKo.stride(2), K, K.stride(0), K.stride(1), Wk, rk, n_heads_K,
        cos, cos.stride(0), sin, sin.stride(0), rope_indices, seq_len,
        HEAD_DIM = head_dim, BLOCK_H = BLOCK_H, HAS_ROPE_INDICES = has_indices,
        num_warps = _num_warps(BLOCK_H, head_dim), enable_fp_fusion = True,
    )  # fmt: skip
    return dQ, dK


class Fast_QK_Norm_RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, Wq, Wk, cos, sin, rope_indices, eps_Q, eps_K, has_indices):
        with torch_gpu_device(Q.device):
            Qo, Ko, rq, rk = _launch_forward(
                Q, K, Wq, Wk, cos, sin, rope_indices, eps_Q, eps_K, has_indices, _eager_kernel
            )
        ctx.has_indices = has_indices
        ctx.save_for_backward(Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices)
        return Qo, Ko

    @staticmethod
    def backward(ctx, dQo, dKo):
        Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices = ctx.saved_tensors
        with torch_gpu_device(dQo.device):
            dQ, dK = _launch_backward(
                dQo, dKo, Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices, ctx.has_indices,
                _eager_kernel,
            )  # fmt: skip
        return dQ, dK, None, None, None, None, None, None, None, None


if _TRACEABLE:

    @torch.library.triton_op("unsloth::qk_norm_rope", mutates_args = ())
    def _qk_norm_rope_op(
        Q: torch.Tensor,
        K: torch.Tensor,
        Wq: torch.Tensor,
        Wk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_indices: torch.Tensor,
        eps_Q: float,
        eps_K: float,
        has_indices: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch_gpu_device(Q.device):
            return _launch_forward(
                Q, K, Wq, Wk, cos, sin, rope_indices, eps_Q, eps_K, has_indices, _traced_kernel
            )

    @torch.library.triton_op("unsloth::qk_norm_rope_backward", mutates_args = ())
    def _qk_norm_rope_backward_op(
        dQo: torch.Tensor,
        dKo: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        Wq: torch.Tensor,
        Wk: torch.Tensor,
        rq: torch.Tensor,
        rk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_indices: torch.Tensor,
        has_indices: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch_gpu_device(dQo.device):
            return _launch_backward(
                dQo, dKo, Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices, has_indices,
                _traced_kernel,
            )  # fmt: skip

    def _qk_norm_rope_setup_context(ctx, inputs, output):
        Q, K, Wq, Wk, cos, sin, rope_indices, _, _, has_indices = inputs
        ctx.has_indices = has_indices
        ctx.save_for_backward(Q, K, Wq, Wk, output[2], output[3], cos, sin, rope_indices)

    def _qk_norm_rope_op_backward(ctx, dQo, dKo, drq, drk):
        Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices = ctx.saved_tensors
        if dQo is None:
            dQo = torch.zeros(Q.shape, dtype = Q.dtype, device = Q.device)
        if dKo is None:
            dKo = torch.zeros(K.shape, dtype = K.dtype, device = K.device)
        dQ, dK = torch.ops.unsloth.qk_norm_rope_backward(
            dQo, dKo, Q, K, Wq, Wk, rq, rk, cos, sin, rope_indices, ctx.has_indices
        )
        return dQ, dK, None, None, None, None, None, None, None, None

    _qk_norm_rope_op.register_autograd(
        _qk_norm_rope_op_backward, setup_context = _qk_norm_rope_setup_context
    )
    _tag_compile_cache(__file__)


def _eps(norm):
    return norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps


def _unfused(q_norm, k_norm, Q, K, cos, sin, position_ids):
    # reuse_grad as in the flag-off Qwen3 path: the norm output only feeds the RoPE kernel.
    Q = fast_rms_layernorm(q_norm, Q, reuse_grad = True).transpose(1, 2)
    K = fast_rms_layernorm(k_norm, K, reuse_grad = True).transpose(1, 2)
    return fast_rope_embedding(Q, K, cos, sin, position_ids)


@torch.compiler.disable
def _fused_untraced(*args):
    return Fast_QK_Norm_RoPE.apply(*args)


def fast_qk_norm_rope(
    q_norm,
    k_norm,
    Q,
    K,
    cos,
    sin,
    position_ids = None,
):
    """RMSNorm each head of Q [batch, seq_len, n_heads, head_dim] and K with q_norm / k_norm, then
    rotate both. Returns Q, K as [batch, n_heads, seq_len, head_dim] like fast_rope_embedding.
    Falls back to fast_rms_layernorm + fast_rope_embedding where the fused kernel does not apply
    (trainable norm weights, non-CUDA, odd head_dim, cos / sin not a [positions, head_dim] table)."""
    Wq, Wk = q_norm.weight, k_norm.weight
    batch, seq_len, _, head_dim = Q.shape
    cos2, sin2 = cos.squeeze(), sin.squeeze()
    if (
        Wq.requires_grad
        or Wk.requires_grad
        or Q.device.type != "cuda"
        or head_dim % 2 != 0
        or head_dim & (head_dim - 1) != 0
        or head_dim < 16
        or head_dim > 1024
        or cos2.dim() != 2
        or cos2.shape[-1] != head_dim
        or sin2.shape != cos2.shape
        or cos2.stride(-1) != 1
        or sin2.stride(-1) != 1
        or Q.stride(-1) != 1
        or K.stride(-1) != 1
        or Wq.shape[-1] != head_dim
        or Wk.shape[-1] != head_dim
    ):
        return _unfused(q_norm, k_norm, Q, K, cos, sin, position_ids)
    has_indices = position_ids is not None
    if has_indices:
        if position_ids.numel() == seq_len and batch != 1:
            position_ids = position_ids.reshape(1, seq_len).expand(batch, seq_len)
        if position_ids.numel() != batch * seq_len:
            return _unfused(q_norm, k_norm, Q, K, cos, sin, position_ids)
        rope_indices = position_ids.reshape(-1).to(dtype = torch.int32, device = Q.device).contiguous()
    else:
        if seq_len > cos2.shape[0]:
            return _unfused(q_norm, k_norm, Q, K, cos, sin, position_ids)
        rope_indices = cos2.new_empty(1, dtype = torch.int32)
    args = (Q, K, Wq.contiguous(), Wk.contiguous(), cos2, sin2, rope_indices, float(_eps(q_norm)), float(_eps(k_norm)), has_indices)  # fmt: skip
    if not torch.compiler.is_compiling():
        Qo, Ko = Fast_QK_Norm_RoPE.apply(*args)
        _sync_if_split(Q)
    elif not _TRACEABLE or (Q.dtype == torch.bfloat16 and not _BF16_TRACEABLE):
        Qo, Ko = _fused_untraced(*args)
    else:
        Qo, Ko, _, _ = torch.ops.unsloth.qk_norm_rope(*args)
    return Qo.transpose(1, 2), Ko.transpose(1, 2)


def decode_qk_norm_rope(q_norm, k_norm, Q, K, cos, sin, position_ids):
    """Decode q/k norm + RoPE in one launch; returns new [bsz, n_heads, 1, head_dim] Q, K or None.
    UNSLOTH_DECODE_FUSED_ROPE=0 disables it."""
    if not DECODE_FUSED_ROPE or torch.compiler.is_compiling():
        return None
    Wq, Wk = q_norm.weight, k_norm.weight
    if Q.device.type != "cuda" or Q.dim() != 4 or K.dim() != 4:
        return None
    batch, seq_len, _, head_dim = Q.shape
    if seq_len != 1 or K.shape[0] != batch or K.shape[1] != 1 or K.shape[3] != head_dim:
        return None
    if position_ids is None or position_ids.numel() != batch:
        return None
    if torch.is_grad_enabled() and (
        Q.requires_grad or K.requires_grad or Wq.requires_grad or Wk.requires_grad
    ):
        return None
    cos2, sin2 = cos.squeeze(), sin.squeeze()
    if (
        head_dim % 2 != 0
        or head_dim & (head_dim - 1) != 0
        or head_dim < 16
        or head_dim > 1024
        or cos2.dim() != 2
        or cos2.shape[-1] != head_dim
        or sin2.shape != cos2.shape
        or cos2.stride(-1) != 1
        or sin2.stride(-1) != 1
        or Q.stride(-1) != 1
        or K.stride(-1) != 1
        or Wq.shape != (head_dim,)
        or Wk.shape != (head_dim,)
        # Matches the unfused rounding only when every operand shares Q's dtype.
        or not (Q.dtype == K.dtype == Wq.dtype == Wk.dtype == cos2.dtype == sin2.dtype)
        or not (Q.device == K.device == Wq.device == Wk.device == cos2.device == sin2.device)
    ):
        return None
    # The kernel indexes positions densely; an int32 last-column slice keeps its row stride.
    rope_indices = position_ids.reshape(-1).to(dtype = torch.int32, device = Q.device).contiguous()
    with torch_gpu_device(Q.device):
        Qo, Ko, _, _ = _launch_forward(
            Q, K, Wq.contiguous(), Wk.contiguous(), cos2, sin2, rope_indices,
            float(_eps(q_norm)), float(_eps(k_norm)), True, _eager_kernel,
        )  # fmt: skip
    # No host drain (see decode_rope_inplace).
    return Qo.transpose(1, 2), Ko.transpose(1, 2)
