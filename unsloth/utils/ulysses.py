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

"""Ulysses sequence parallelism: all-to-all from sequence shards to head shards around attention.

Each rank holds a contiguous slice of the sequence. Before attention Q/K/V are exchanged so every rank
sees the whole sequence for H / N heads, attention runs through the normal dispatcher (varlen, FA3/FA4,
SDPA, xformers all work since nothing is sharded along the sequence), then the output goes back to
sequence shards. Packed metadata (cu_seqlens) therefore stays global.
"""

from dataclasses import replace
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = [
    "UlyssesHeadPlan",
    "ulysses_head_plan",
    "seq_to_head",
    "head_to_seq",
    "ulysses_attention",
    "ulysses_prepare_batch",
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
    send = torch.stack(x.chunk(world, dim = scatter_dim), dim = 0).contiguous()
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


def _right_padded(attention_mask):
    keep = attention_mask != 0
    # Right padded: once a row hits padding it never returns to real tokens.
    return bool((keep[:, 1:] <= keep[:, :-1]).all())


def _document_positions(packed_seq_lengths, total, device):
    lengths = [int(x) for x in packed_seq_lengths.tolist()]
    pieces = [torch.arange(length, device = device) for length in lengths]
    used = sum(lengths)
    if used > total:
        raise ValueError(
            f"Unsloth: packed_seq_lengths sum to {used} but the packed row has {total} tokens."
        )
    if used < total:
        pieces.append(torch.arange(total - used, device = device))
    return torch.cat(pieces)[None]


def ulysses_prepare_batch(
    input_ids,
    labels = None,
    position_ids = None,
    attention_mask = None,
    packed_seq_lengths = None,
    *,
    rank,
    size,
    pad_token_id = 0,
):
    """Shift labels and build positions on the full sequence, pad to a multiple of `size`, keep this
    rank's contiguous slice. packed_seq_lengths stays global (attention sees the whole sequence)."""
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(
                "Unsloth: Ulysses context parallelism needs a 2D attention_mask, got "
                f"{attention_mask.dim()}D."
            )
        if not _right_padded(attention_mask):
            raise ValueError(
                "Unsloth: Ulysses context parallelism needs right padding (tokenizer.padding_side = 'right')."
            )
    if packed_seq_lengths is not None and bsz != 1:
        raise ValueError(
            f"Unsloth: packed batches must be a single row under Ulysses context parallelism, got {bsz}."
        )

    shift_labels = None
    if labels is not None:
        shift_labels = F.pad(labels, (0, 1), value = -100)[:, 1:]

    if position_ids is None:
        if packed_seq_lengths is not None:
            position_ids = _document_positions(packed_seq_lengths, seq_len, device)
        else:
            position_ids = torch.arange(seq_len, device = device)[None].expand(bsz, -1)
    position_ids = position_ids.to(device).expand(bsz, -1)

    padded_len = -(-seq_len // size) * size
    pad = padded_len - seq_len
    if pad:
        input_ids = F.pad(input_ids, (0, pad), value = pad_token_id)
        if shift_labels is not None:
            shift_labels = F.pad(shift_labels, (0, pad), value = -100)
        tail = position_ids[:, -1:] + torch.arange(1, pad + 1, device = device)
        position_ids = torch.cat([position_ids, tail.to(position_ids.dtype)], dim = 1)

    local_len = padded_len // size
    start = rank * local_len
    stop = start + local_len
    out = {
        "input_ids": input_ids[:, start:stop].contiguous(),
        "position_ids": position_ids[:, start:stop].contiguous(),
        "packed_seq_lengths": packed_seq_lengths,
        "global_seq_len": padded_len,
        "pad_len": pad,
    }
    if shift_labels is not None:
        out["shift_labels"] = shift_labels[:, start:stop].contiguous()
    return out
