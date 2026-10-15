# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses sequence parallelism: all-to-all from sequence shards to head shards around attention.

Each rank holds a contiguous slice of the sequence. Before attention Q/K/V are exchanged so every rank
sees the whole sequence for H / N heads, attention runs through the normal dispatcher (varlen, FA3/FA4,
SDPA, xformers all work since nothing is sharded along the sequence), then the output goes back to
sequence shards. Packed metadata (cu_seqlens) therefore stays global.
"""

from dataclasses import replace
from typing import NamedTuple

import torch
import torch.distributed as dist

__all__ = [
    "UlyssesHeadPlan",
    "ulysses_head_plan",
    "seq_to_head",
    "head_to_seq",
    "ulysses_attention",
]


def _all_to_all(x, scatter_dim, gather_dim, group):
    # Chunk j of scatter_dim goes to rank j; received chunks are concatenated along gather_dim in rank order.
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if x.shape[scatter_dim] % world != 0:
        raise ValueError(
            f"Unsloth: Ulysses cannot split dim {scatter_dim} of size {x.shape[scatter_dim]} "
            f"across {world} ranks."
        )
    send = torch.stack(x.chunk(world, dim = scatter_dim), dim = 0)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group = group)
    return torch.cat(recv.unbind(0), dim = gather_dim)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return _all_to_all(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad):
        # A permutation across ranks: the inverse exchange, no scaling.
        return _all_to_all(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None, None


def seq_to_head(x, group):
    """[B, H, T/N, D] sequence shard -> [B, H/N, T, D] head shard (rank r gets heads r*H/N ...)."""
    return _AllToAll.apply(x, 1, 2, group)


def head_to_seq(x, group):
    """run_attention output [B, T, H/N, D] -> [B, T/N, H, D] (the local shard's run_attention layout)."""
    return _AllToAll.apply(x, 1, 2, group)


class UlyssesHeadPlan(NamedTuple):
    kv_replication: int
    local_heads: int
    local_kv_heads: int
    local_groups: int


def ulysses_head_plan(n_heads, n_kv_heads, cp_size):
    if n_heads % cp_size != 0:
        raise ValueError(
            f"Unsloth: Ulysses context parallelism needs the number of attention heads ({n_heads}) "
            f"to be divisible by context_parallel_size ({cp_size})."
        )
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"Unsloth: attention heads ({n_heads}) must be a multiple of key/value heads ({n_kv_heads})."
        )
    if n_kv_heads % cp_size == 0:
        replication = 1
    elif cp_size % n_kv_heads == 0:
        replication = cp_size // n_kv_heads
    else:
        raise ValueError(
            f"Unsloth: Ulysses context parallelism needs key/value heads ({n_kv_heads}) and "
            f"context_parallel_size ({cp_size}) to divide one another."
        )
    local_heads = n_heads // cp_size
    local_kv_heads = n_kv_heads * replication // cp_size
    return UlyssesHeadPlan(replication, local_heads, local_kv_heads, local_heads // local_kv_heads)


def ulysses_attention(run_attention_fn, config, context, Q, K, V, group):
    """Run attention for a sequence shard: Q [B, H, T/N, D], K/V [B, Hkv, T/N, D] (post RoPE).

    Returns [B, T/N, H, D], the layout run_attention returns for the local shard.
    """
    if context.attention_mask is not None or isinstance(context.causal_mask, torch.Tensor):
        raise RuntimeError(
            "Unsloth: Ulysses context parallelism does not take a dense attention mask; padding is "
            "handled by ignoring padded labels and packing by global sequence lengths."
        )
    if context.prefix_seg_info is not None:
        raise RuntimeError("Unsloth: Ulysses context parallelism does not support PrefixGrouper.")
    if context.q_len != context.kv_seq_len:
        raise RuntimeError("Unsloth: Ulysses context parallelism does not support a KV cache.")

    size = dist.get_world_size(group)
    plan = ulysses_head_plan(context.n_heads, config.n_kv_heads, size)
    if plan.kv_replication > 1:
        # Consecutive copies keep query head h on kv head h // n_groups after the split; autograd sums
        # the copies' gradients back into the original kv head.
        K = K.repeat_interleave(plan.kv_replication, dim = 1)
        V = V.repeat_interleave(plan.kv_replication, dim = 1)

    Q = seq_to_head(Q, group)
    K = seq_to_head(K, group)
    V = seq_to_head(V, group)

    config = replace(config, n_kv_heads = plan.local_kv_heads, n_groups = plan.local_groups)
    context = replace(
        context,
        q_len = context.q_len * size,
        kv_seq_len = context.kv_seq_len * size,
        n_heads = plan.local_heads,
    )
    A = run_attention_fn(config = config, context = context, Q = Q, K = K, V = V)
    A = A.reshape(context.bsz, context.q_len, plan.local_heads, context.head_dim)
    return head_to_seq(A, group)


def _document_positions(packed_seq_lengths, total, device):
    # Callers check sum(lengths) <= total first.
    lengths = [int(x) for x in torch.as_tensor(packed_seq_lengths).reshape(-1).tolist()]
    pieces = [torch.arange(length, device = device) for length in lengths]
    used = sum(lengths)
    if used < total:
        pieces.append(torch.arange(total - used, device = device))
    return torch.cat(pieces)[None]
