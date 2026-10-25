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

import functools
import os
import weakref
import triton
import triton.language as tl
import torch
from ..device_type import DEVICE_COUNT, DEVICE_TYPE_TORCH
from typing import Tuple
from .utils import long_indexing, next_power_of_2, torch_gpu_device, torch_device_stream
from .rms_layernorm import (
    _BF16_TRACEABLE,
    _TRACEABLE,
    _eager_kernel,
    _tag_compile_cache,
    _traced_kernel,
)


@triton.jit
def _rope_heads_tile(
    X,
    OUT,
    x_batch_stride,
    x_seq_stride,
    x_head_stride,
    x_dim_stride,
    out_batch_stride,
    out_seq_stride,
    out_head_stride,
    out_dim_stride,
    batch_id,
    seq_index,
    head_start,
    n_heads,
    cos1,
    sin1,
    half_head_dim: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAST_TO_SIN: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    # Rotates HEADS heads of one (batch, seq) row from X into OUT (OUT may be X: each element is
    # read before it is written by the same program).
    heads = head_start + tl.arange(0, HEADS)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = (heads[:, None] < n_heads) & (cols[None, :] < half_head_dim)
    if LONG_INDEXING:
        heads = heads.to(tl.int64)
    x_ptr = (
        X
        + batch_id * x_batch_stride
        + seq_index * x_seq_stride
        + heads[:, None] * x_head_stride
        + cols[None, :] * x_dim_stride
    )
    x0 = tl.load(x_ptr, mask = mask, other = 0)
    x1 = tl.load(x_ptr + half_head_dim * x_dim_stride, mask = mask, other = 0)
    if CAST_TO_SIN:
        # Gemma sometimes needs RoPE done in float32 rather than bfloat16.
        x0 = x0.to(sin1.dtype)
        x1 = x1.to(sin1.dtype)
    out_ptr = (
        OUT
        + batch_id * out_batch_stride
        + seq_index * out_seq_stride
        + heads[:, None] * out_head_stride
        + cols[None, :] * out_dim_stride
    )
    c = cos1[None, :]
    s = sin1[None, :]
    # x0 * c - x1 * s and x1 * c + x0 * s with the contraction pinned: left to the backend it
    # changes with the tile shape. These are the fma forms the one-head-per-program kernels got.
    out0 = tl.fma(x0, c, -(x1 * s))
    if BACKWARD_PASS:
        out1 = tl.fma(x1, c, x0 * s)
    else:
        out1 = tl.fma(x0, s, x1 * c)
    tl.store(out_ptr, out0, mask = mask)
    tl.store(out_ptr + half_head_dim * out_dim_stride, out1, mask = mask)


def _rope_embedding_QK(
    Q,
    Q_out,
    K,
    K_out,
    q_batch_stride,
    q_seq_stride: tl.constexpr,  # 4096 n_heads * head_dim (6144 fused QKV, 128 contiguous)
    q_head_stride,
    q_dim_stride: tl.constexpr,  # 1
    qo_batch_stride,
    qo_seq_stride: tl.constexpr,  # 4096 or 128
    qo_head_stride,
    qo_dim_stride: tl.constexpr,  # 1
    k_batch_stride,
    k_seq_stride: tl.constexpr,  # 1024 n_kv_heads * head_dim
    k_head_stride,
    k_dim_stride: tl.constexpr,  # 1
    ko_batch_stride,
    ko_seq_stride: tl.constexpr,  # 1024 or 128
    ko_head_stride,
    ko_dim_stride: tl.constexpr,  # 1
    cos,
    cos_row_stride,  # runtime: a SymInt in compiled graphs
    sin,
    sin_row_stride,
    rope_embedding_indices,
    seqlen,
    n_heads_Q: tl.constexpr,  # 32
    n_heads_K: tl.constexpr,  # 8 (0 without K)
    n_groups_Q: tl.constexpr,  # 1
    half_head_dim: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    CAST_TO_SIN: tl.constexpr,
    HEADS_Q: tl.constexpr,
    HEADS_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVICT_INDICES: tl.constexpr,
):
    """
    RoPE is Q * cos + rotate_half(Q) * sin; the backward is the same with -sin.
    One program = one (batch, seq) row x a group of HEADS_Q query heads or HEADS_K key heads,
    BLOCK_SIZE lanes over half of head_dim, read from any strides and written out of place
    (or in place when Q_out is Q).
    """
    row_position = tl.program_id(0)
    group = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    if HAS_ROPE_INDICES:
        if EVICT_INDICES:
            rot_position = tl.load(
                rope_embedding_indices + row_position,
                eviction_policy = "evict_first",
            ).to(tl.int32)
        else:
            rot_position = tl.load(rope_embedding_indices + row_position).to(tl.int32)
    else:
        rot_position = row_position % seqlen
    if LONG_INDEXING:
        # cos / sin tables past 2**31 elements need a 64-bit row offset.
        rot_position = rot_position.to(tl.int64)

    sin1 = tl.load(sin + rot_position * sin_row_stride + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + rot_position * cos_row_stride + col_offsets, mask = mask, other = 0)
    if BACKWARD_PASS:
        sin1 = -sin1

    batch_id = row_position // seqlen
    seq_index = row_position - batch_id * seqlen
    if LONG_INDEXING:
        # Widen after the int32 div/mod: 64-bit div/mod is slow at normal sizes.
        batch_id = batch_id.to(tl.int64)
        seq_index = seq_index.to(tl.int64)

    if group < n_groups_Q:
        _rope_heads_tile(
            Q,
            Q_out,
            q_batch_stride,
            q_seq_stride,
            q_head_stride,
            q_dim_stride,
            qo_batch_stride,
            qo_seq_stride,
            qo_head_stride,
            qo_dim_stride,
            batch_id,
            seq_index,
            group * HEADS_Q,
            n_heads_Q,
            cos1,
            sin1,
            half_head_dim,
            HEADS_Q,
            BLOCK_SIZE,
            CAST_TO_SIN,
            LONG_INDEXING,
            BACKWARD_PASS,
        )
    else:
        _rope_heads_tile(
            K,
            K_out,
            k_batch_stride,
            k_seq_stride,
            k_head_stride,
            k_dim_stride,
            ko_batch_stride,
            ko_seq_stride,
            ko_head_stride,
            ko_dim_stride,
            batch_id,
            seq_index,
            (group - n_groups_Q) * HEADS_K,
            n_heads_K,
            cos1,
            sin1,
            half_head_dim,
            HEADS_K,
            BLOCK_SIZE,
            CAST_TO_SIN,
            LONG_INDEXING,
            BACKWARD_PASS,
        )


# No triton.heuristics wrapper: the launcher passes real bools, and the wrapper's per-call cost
# showed up at small (decode / short-sequence) shapes.
_rope_embedding_QK = triton.jit(_rope_embedding_QK)

# Elements of one half-head tile a program rotates: several heads per program instead of one
# program per (row, head) with half its lanes masked off.
ROPE_TILE_ELEMENTS: int = 1024
# None picks per GPU: 8 warps on pre-Ampere CUDA (T4 measured ~5% faster at Llama-8B shapes), else 4
# (A100 / RTX PRO 6000 are flat between 2 and 8, a little slower at 8).
ROPE_NUM_WARPS = None


@functools.lru_cache(maxsize = None)
def _rope_num_warps_at(device_type, index):
    if device_type == "cuda" and torch.version.hip is None:
        if torch.cuda.get_device_capability(index)[0] < 8:
            return 8
    return 4


def _rope_num_warps(device):
    if ROPE_NUM_WARPS is not None:
        return ROPE_NUM_WARPS
    index = device.index
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device()
    return _rope_num_warps_at(device.type, index)


def _heads_per_program(n_heads, block_size):
    return max(1, min(next_power_of_2(n_heads), ROPE_TILE_ELEMENTS // block_size))


def _rope_launch(Q, Q_out, K, K_out, cos, sin, rope_ptr, has_indices, backward, cast_to_sin, wrap):
    # Q, K, Q_out, K_out: [batch, n_heads, seq_len, head_dim] at any strides (K may be None).
    # Q_out may be Q itself (in place).
    batch, n_heads_Q, seq_len, head_dim = Q.shape
    half_head_dim = head_dim // 2
    BLOCK_SIZE = next_power_of_2(half_head_dim)
    HEADS_Q = _heads_per_program(n_heads_Q, BLOCK_SIZE)
    n_groups_Q = (n_heads_Q + HEADS_Q - 1) // HEADS_Q  # also a SymInt under dynamic=True
    if K is None:
        # No key heads: a dummy, never read or written. Not Q / Q_out, since a traced kernel would
        # see Q_out mutated through two arguments.
        dummy = Q.new_empty((1, 1, 1, 1))
        K, K_out, n_heads_K, HEADS_K, n_groups_K = dummy, dummy, 0, 1, 0
    else:
        n_heads_K = K.shape[1]
        HEADS_K = _heads_per_program(n_heads_K, BLOCK_SIZE)
        n_groups_K = (n_heads_K + HEADS_K - 1) // HEADS_K
    if rope_ptr is None:
        rope_ptr = cos  # never read without HAS_ROPE_INDICES
    tensors = (Q, Q_out, K, K_out, cos, sin)
    wrap(_rope_embedding_QK)[(batch * seq_len, n_groups_Q + n_groups_K)](
        Q,
        Q_out,
        K,
        K_out,
        Q.stride(0),
        Q.stride(2),
        Q.stride(1),
        Q.stride(3),
        Q_out.stride(0),
        Q_out.stride(2),
        Q_out.stride(1),
        Q_out.stride(3),
        K.stride(0),
        K.stride(2),
        K.stride(1),
        K.stride(3),
        K_out.stride(0),
        K_out.stride(2),
        K_out.stride(1),
        K_out.stride(3),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        rope_ptr,
        seq_len,
        n_heads_Q,
        n_heads_K,
        n_groups_Q,
        half_head_dim = half_head_dim,
        BACKWARD_PASS = bool(backward),
        HAS_ROPE_INDICES = bool(has_indices),
        LONG_INDEXING = long_indexing(*tensors, block = 2 * BLOCK_SIZE * max(HEADS_Q, HEADS_K)),
        CAST_TO_SIN = bool(cast_to_sin),
        HEADS_Q = HEADS_Q,
        HEADS_K = HEADS_K,
        BLOCK_SIZE = BLOCK_SIZE,
        EVICT_INDICES = _eviction_hints_ok(Q.device) if has_indices else False,
        num_warps = _rope_num_warps(Q.device),
        enable_fp_fusion = True,
    )


def _rope_rows(Q, Q_out, cos, sin, backward, wrap):
    # Q, Q_out: [batch, seq_len, n_heads, head_dim] at any strides; Q_out may be Q (in place).
    # Rows cast Q to sin's dtype (the QK path does not), kept for bitwise parity with before.
    _rope_launch(
        Q.transpose(1, 2),
        Q_out.transpose(1, 2),
        None,
        None,
        cos,
        sin,
        None,
        False,
        backward,
        True,
        wrap,
    )


def _bshd_buffer(X):
    # A fresh [batch, n_heads, seq_len, head_dim] view of a contiguous [batch, seq_len, n_heads,
    # head_dim] buffer: the layout Q.transpose(1, 2).contiguous().transpose(1, 2) produced.
    batch, n_heads, seq_len, head_dim = X.shape
    return torch.empty(
        (batch, seq_len, n_heads, head_dim), dtype = X.dtype, device = X.device
    ).transpose(1, 2)


class Fast_RoPE_Embedding_QK_Positions(torch.autograd.Function):
    """Fast_RoPE_Embedding on Q and K ([batch, n_heads, seq_len, head_dim]) in one launch each way.

    Same bytes, layouts and aliasing as Fast_RoPE_Embedding.apply(X.transpose(1, 2).contiguous(),
    cos, sin).transpose(1, 2) for X in (Q, K): in place where X.transpose(1, 2) is contiguous,
    else into a fresh [batch, seq_len, n_heads, head_dim] buffer; gradients in that layout too."""

    @staticmethod
    def forward(ctx, Q, K, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        assert Q.shape[2] <= cos.shape[0]
        Q_out = Q if Q.transpose(1, 2).is_contiguous() else _bshd_buffer(Q)
        K_out = K if K.transpose(1, 2).is_contiguous() else _bshd_buffer(K)
        with torch_gpu_device(Q.device):
            _rope_launch(Q, Q_out, K, K_out, cos, sin, None, False, False, True, _eager_kernel)
        ctx.cos = cos
        ctx.sin = sin
        return Q_out.view(Q_out.shape), K_out.view(K_out.shape)

    @staticmethod
    def backward(ctx, dQ, dK):
        dQ_out, dK_out = _bshd_buffer(dQ), _bshd_buffer(dK)
        with torch_gpu_device(dQ.device):
            _rope_launch(
                dQ, dQ_out, dK, dK_out, ctx.cos, ctx.sin, None, False, True, True, _eager_kernel
            )
        return dQ_out, dK_out, None, None


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        assert Q.shape[1] <= cos.shape[0]
        # In place on contiguous Q (callers pass Q.transpose(1, 2).contiguous()), as before;
        # a strided Q is written to a fresh contiguous tensor instead of being copied first.
        Q_out = Q if Q.is_contiguous() else torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
        with torch_gpu_device(Q.device):
            _rope_rows(Q, Q_out, cos, sin, False, _eager_kernel)
        ctx.cos = cos
        ctx.sin = sin
        return Q_out.view(Q_out.shape)

    @staticmethod
    def backward(ctx, dY):
        # Out of place into a fresh contiguous buffer: dY may be shared or stride-0 (expanded).
        dQ = torch.empty(dY.shape, dtype = dY.dtype, device = dY.device)
        with torch_gpu_device(dY.device):
            _rope_rows(dY, dQ, ctx.cos, ctx.sin, True, _eager_kernel)
        return (
            dQ,
            None,
            None,
        )


@functools.lru_cache(maxsize = None)
def _eviction_hints_ok_at(device_type, index):
    # ptxas rejects ld eviction hints below sm_70 (Maxwell, Pascal).
    if device_type != "cuda" or torch.version.hip is not None:
        return True
    return torch.cuda.get_device_capability(index)[0] >= 7


def _eviction_hints_ok(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _eviction_hints_ok_at(device.type, index)


def _rope_qk(Q, K, Q_out, K_out, cos, sin, rope_ptr, has_indices, backward, wrap):
    # Rotates Q [batch, n_heads_Q, seq_len, head_dim] and K into Q_out / K_out, at any strides.
    _rope_launch(Q, Q_out, K, K_out, cos, sin, rope_ptr, has_indices, backward, False, wrap)


class Fast_RoPE_Embedding_QK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        has_indices = rope_indices is not None
        cos, sin = cos.squeeze(), sin.squeeze()

        # Contiguous inputs rotate in place; strided ones (transposed views of the projections)
        # are read where they are and written to a fresh tensor of the same layout.
        Q_out = Q if Q.is_contiguous() else torch.empty_like(Q)
        K_out = K if K.is_contiguous() else torch.empty_like(K)

        if has_indices:
            # TRL's rotary indices are always int32, so the cast is only for safety.
            rope_ptr = rope_indices.reshape(-1).to(dtype = torch.int32, device = Q.device).contiguous()
        else:
            rope_ptr = None

        with torch_gpu_device(Q.device):
            _rope_qk(Q, K, Q_out, K_out, cos, sin, rope_ptr, has_indices, False, _eager_kernel)

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
        rope_ptr = ctx.rope_indices

        # Out of place: gradients can share storage with each other or another branch.
        dQ_out = torch.empty(dQ.shape, dtype = dQ.dtype, device = dQ.device)
        dK_out = torch.empty(dK.shape, dtype = dK.dtype, device = dK.device)

        with torch_gpu_device(dQ.device):
            _rope_qk(
                dQ,
                dK,
                dQ_out,
                dK_out,
                ctx.cos,
                ctx.sin,
                rope_ptr,
                ctx.has_indices,
                True,
                _eager_kernel,
            )

        return (dQ_out, dK_out, None, None, None)


# Compiled graphs take these ops: the same kernel, written out of place (the Functions may rotate
# in place).
if _TRACEABLE:

    @torch.library.triton_op("unsloth::rope_embedding", mutates_args = ())
    def _rope_embedding_op(
        Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backward: bool
    ) -> torch.Tensor:
        # Q: [batch, seq_len, n_heads, head_dim] at any strides.
        out = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
        _rope_rows(Q, out, cos, sin, backward, _traced_kernel)
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
        Q_out, K_out = torch.empty_like(Q), torch.empty_like(K)
        _rope_qk(Q, K, Q_out, K_out, cos, sin, rope_indices, True, backward, _traced_kernel)
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


# The RoPE stream drain exists only for models split over devices (#3049, #3061); single-device skips it.
_SPLIT_MODELS = weakref.WeakSet()

# Unregistered or untrackable split seen: drain every call from then on.
_UNREGISTERED_SPLIT_SEEN = False

if DEVICE_COUNT > 1:
    _current_device = getattr(torch, DEVICE_TYPE_TORCH).current_device
else:
    _current_device = None


def _accelerator_indices(device_map):
    # Index-less "cuda" means the current device, not device 0.
    current = _current_device() if _current_device is not None else 0
    indices = set()
    for value in device_map.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            indices.add(value)
            continue
        text = str(value)
        if text.isdigit():
            indices.add(int(text))
            continue
        if text in ("cpu", "disk", "meta"):
            continue
        try:
            device = torch.device(text)
        except Exception:
            continue
        if device.type in ("cpu", "meta"):
            continue
        indices.add(current if device.index is None else device.index)
    return indices


def register_split_model(model, device_map = None):
    """Never raises: a split that cannot be tracked (an unhashable model) drains every call instead."""
    global _UNREGISTERED_SPLIT_SEEN
    if DEVICE_COUNT <= 1:
        return False
    try:
        if device_map is None:
            device_map = getattr(model, "hf_device_map", None)
        if not isinstance(device_map, dict) or len(_accelerator_indices(device_map)) < 2:
            return False
        _SPLIT_MODELS.add(model)
    except Exception:
        _UNREGISTERED_SPLIT_SEEN = True
    return True


def _register_splits_on_dispatch():
    import sys

    try:
        import accelerate.big_modeling as big_modeling
    except Exception:
        return
    original = big_modeling.dispatch_model
    if getattr(original, "_unsloth_registers_split", False):
        return

    @functools.wraps(original)
    def dispatch_model(model, device_map, *args, **kwargs):
        out = original(model, device_map, *args, **kwargs)
        register_split_model(model, device_map)
        return out

    dispatch_model._unsloth_registers_split = True
    # transformers binds it by name at import.
    for name in (
        "accelerate",
        "accelerate.big_modeling",
        "transformers.modeling_utils",
        "transformers.integrations.accelerate",
        "transformers.integrations.peft",
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "dispatch_model", None) is original:
            module.dispatch_model = dispatch_model


if DEVICE_COUNT > 1:
    _register_splits_on_dispatch()


def _sync_if_split(Q):
    global _UNREGISTERED_SPLIT_SEEN
    if DEVICE_COUNT <= 1:
        return
    if not (_SPLIT_MODELS or _UNREGISTERED_SPLIT_SEEN):
        if Q.get_device() == _current_device():
            return
        _UNREGISTERED_SPLIT_SEEN = True
    torch_device_stream(Q.device).synchronize()


def _fast_rope_embedding_eager(Q, K, cos, sin, rope_embedding_indices):
    if rope_embedding_indices is not None:
        Q_out, K_out = Fast_RoPE_Embedding_QK.apply(Q, K, cos, sin, rope_embedding_indices)
    else:
        # Fast_RoPE_Embedding on Q and K, fused into one launch (forward and backward).
        Q_out, K_out = Fast_RoPE_Embedding_QK_Positions.apply(Q, K, cos, sin)
    _sync_if_split(Q)
    return Q_out, K_out


_fast_rope_embedding_untraced = torch.compiler.disable(_fast_rope_embedding_eager)


# Decode RoPE in one launch instead of ~12 per layer per token; UNSLOTH_DECODE_FUSED_ROPE=0 disables it.
DECODE_FUSED_ROPE = os.environ.get("UNSLOTH_DECODE_FUSED_ROPE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def decode_rope_inplace(Q, K, cos, sin, position_ids):
    """Rotate decode Q, K in place; False (nothing written) where the kernel does not apply."""
    if not DECODE_FUSED_ROPE or torch.compiler.is_compiling():
        return False
    if Q.device.type != "cuda" or K.device != Q.device or Q.dim() != 4 or K.dim() != 4:
        return False
    batch, _, seq_len, head_dim = Q.shape
    if (
        seq_len != 1
        or K.shape[0] != batch
        or K.shape[2] != 1
        or K.shape[3] != head_dim
        or head_dim % 2
    ):
        return False
    if position_ids is None or position_ids.numel() != batch:
        return False
    cos2, sin2 = cos.squeeze(), sin.squeeze()
    if (
        cos2.dim() != 2
        or cos2.shape != sin2.shape
        or cos2.shape[-1] != head_dim
        or cos2.dtype != Q.dtype  # the unfused path multiplies in Q's dtype
        or sin2.dtype != Q.dtype
        or K.dtype != Q.dtype
        or cos2.device != Q.device
        or sin2.device != Q.device
        or cos2.stride(-1) != 1
        or sin2.stride(-1) != 1
    ):
        return False
    # The kernel indexes positions densely; an int32 last-column slice keeps its row stride.
    rope_ptr = position_ids.reshape(-1).to(dtype = torch.int32, device = Q.device).contiguous()
    # No host drain: cross-device copies already order after Q's stream (a drain cost split decode ~40%).
    with torch_gpu_device(Q.device):
        _rope_qk(Q, K, Q, K, cos2, sin2, rope_ptr, True, False, _eager_kernel)
    return True


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
        rope_ptr = (
            rope_embedding_indices.reshape(-1).to(dtype = torch.int32, device = Q.device).contiguous()
        )
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
    _sync_if_split(Q)
    return Q, K
