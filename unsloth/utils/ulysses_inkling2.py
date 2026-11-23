# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Inkling 2 (inkling2 / inkling2_loop, transformers >= 5.24).

Every Inkling 2 layer mixes tokens in two ways:

- attention (sliding window or full) with a relative position bias read from the query token's hidden states
  (`rel_logits[b, q, h, q - k]` for `0 <= q - k < rel_extent`) and a per-token sink logit (`sink_proj`);
- short causal convolutions: one on the keys and one on the values inside the attention (static taps in the
  sliding layers, static + per-token taps in the full layers), and two on the residual stream of every decoder layer
  (after attention and after the MoE).

Attention runs after an all-to-all from sequence shards to head shards, on the whole sequence: q / k / v, the
query-side relative logits and the sinks are exchanged, and the bias, the causal / sliding window mask and the sink
softmax are formed over the whole sequence, so they match one process exactly. The short convolutions stay on the
sequence shard: each one only needs the `kernel_size - 1` tokens before the shard, which come from the previous
rank (a halo exchange, zeros on rank 0 as the causal padding). Everything else in the layer (block / message
attention residuals over layers, the MoE, norms, message gates) is per token.

Not supported (refused): packed rows (the convolution and the bias would need document boundaries), a KV cache,
attention dropout, and shards shorter than the halo."""

from __future__ import annotations

import functools
import sys

import torch
import torch.distributed as dist

__all__ = [
    "INKLING2_ATTENTION_CLASSES",
    "INKLING2_CONV_CLASSES",
    "INKLING2_MODEL_TYPES",
    "INKLING2_LAYER_TYPES",
    "is_ulysses_inkling2_attention",
    "is_ulysses_inkling2_conv",
    "check_inkling2_heads",
    "wrap_inkling2_attention_forward",
    "wrap_inkling2_conv_forward",
    "ulysses_inkling2_attention_forward",
    "ulysses_inkling2_conv_forward",
    "previous_rank_halo",
]

INKLING2_ATTENTION_CLASSES = (
    "Inkling2SlidingAttention",
    "Inkling2FullAttention",
    "Inkling2LoopSlidingAttention",
    "Inkling2LoopFullAttention",
)
INKLING2_CONV_CLASSES = (
    "Inkling2ShortConvolution",
    "Inkling2DynamicShortConvolution",
    "Inkling2LoopShortConvolution",
    "Inkling2LoopDynamicShortConvolution",
)
# config.model_type (top level and text) -> the attention class that refuses packed rows.
INKLING2_MODEL_TYPES = {
    "inkling2": "Inkling2SlidingAttention",
    "inkling2_text": "Inkling2SlidingAttention",
    "inkling2_loop": "Inkling2LoopSlidingAttention",
    "inkling2_loop_text": "Inkling2LoopSlidingAttention",
}
# Layer types the Inkling 2 Ulysses forward masks itself (sliding window and full causal).
INKLING2_LAYER_TYPES = ("hybrid", "hybrid_sliding")


def is_ulysses_inkling2_attention(module) -> bool:
    if type(module).__name__ not in INKLING2_ATTENTION_CLASSES:
        return False
    bias = getattr(module, "relative_position_bias", None)
    return all(
        hasattr(module, n)
        for n in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "sink_proj",
            "scaling",
            "num_heads",
            "num_key_value_heads",
        )
    ) and all(hasattr(bias, n) for n in ("linear", "norm", "proj", "rel_extent", "num_heads"))


def is_ulysses_inkling2_conv(module) -> bool:
    return type(module).__name__ in INKLING2_CONV_CLASSES and isinstance(
        getattr(module, "conv_kernel_size", None), int
    )


def check_inkling2_heads(model, cp_size) -> None:
    """Every rank raises here, before any collective: each attention layer's heads split over cp."""
    from .ulysses import ulysses_head_plan

    seen = set()
    for module in model.modules():
        if is_ulysses_inkling2_attention(module):
            heads = (module.num_heads, module.num_key_value_heads)
            if heads not in seen:
                seen.add(heads)
                ulysses_head_plan(heads[0], heads[1], cp_size)


# ---------------------------------------------------------------------------------------------------------------------
# Halo exchange for the short convolutions


def _all_gather(x, group):
    out = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, x.contiguous(), group = group)
    return out


class _PreviousRankHalo(torch.autograd.Function):
    """[B, n, C] tail of every rank's shard -> the previous rank's tail (zeros on rank 0). Backward sends each
    halo's gradient back to the rank that owns those tokens."""

    @staticmethod
    def forward(ctx, tail, group):
        ctx.group = group
        rank = dist.get_rank(group)
        tails = _all_gather(tail, group)
        return tails[rank - 1].clone() if rank > 0 else torch.zeros_like(tail)

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        grads = _all_gather(grad, group)
        return (grads[rank + 1].clone() if rank + 1 < world else torch.zeros_like(grad)), None


def previous_rank_halo(tail, group):
    if dist.get_world_size(group) == 1:
        return torch.zeros_like(tail)
    return _PreviousRankHalo.apply(tail, group)


def ulysses_inkling2_conv_forward(
    module,
    forward,
    hidden_states,
    dynamic_input = None,
    cache = None,
    conv_mask = None,
    cache_offset = 0,
    *,
    group,
    **kwargs,
):
    """The module's own forward on [previous rank's last kernel_size - 1 tokens, this shard], keeping this shard's
    rows: each output token then sees the same kernel_size inputs as in one process."""
    name = type(module).__name__
    if cache is not None:
        raise NotImplementedError(
            f"Unsloth: Ulysses context parallelism does not support a KV cache in {name}."
        )
    if kwargs.get("seq_idx") is not None:
        raise NotImplementedError(
            f"Unsloth: Ulysses context parallelism does not support packed rows for {name}."
        )
    halo = module.conv_kernel_size - 1
    if halo <= 0:
        return forward(
            hidden_states,
            dynamic_input,
            cache = None,
            conv_mask = conv_mask,
            cache_offset = cache_offset,
            **kwargs,
        )
    batch, local, _ = hidden_states.shape
    if local < halo:
        raise ValueError(
            f"Unsloth: Ulysses context parallelism needs at least {halo} tokens per shard for the Inkling 2 short "
            f"convolutions (kernel size {module.conv_kernel_size}), got {local}: use a longer sequence or a "
            "smaller context_parallel_size."
        )
    tail = hidden_states[:, local - halo :]
    if conv_mask is not None:
        # The sender masks its padding as the convolution would; the halo rows are then marked real.
        tail = (tail * conv_mask[:, local - halo :, None]).to(tail.dtype)
        conv_mask = torch.cat([conv_mask.new_ones(batch, halo), conv_mask], dim = 1)
    extended = torch.cat([previous_rank_halo(tail, group), hidden_states], dim = 1)
    if dynamic_input is not None and "Dynamic" in name:
        # Per-token taps of the halo rows: only their outputs use them, and those rows are dropped.
        zeros = dynamic_input.new_zeros(batch, halo, dynamic_input.shape[-1])
        dynamic_input = torch.cat([zeros, dynamic_input], dim = 1)
    out = forward(
        extended,
        dynamic_input,
        cache = None,
        conv_mask = conv_mask,
        cache_offset = cache_offset,
        **kwargs,
    )
    return out[:, halo:]


# ---------------------------------------------------------------------------------------------------------------------
# Attention on head shards over the whole sequence


def _whole_sequence_bias(rel_logits, total, rel_extent):
    """[B, H, T, R] query-side relative logits -> [B, H, T, T] bias, as Inkling2RelativePositionBias over T."""
    idx = torch.arange(total, device = rel_logits.device)
    distance = idx[:, None] - idx[None, :]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    bias = rel_logits.gather(-1, gather_index)
    return bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)


def _flex_wanted(module, total, device) -> bool:
    try:
        from ..models import inkling2_flex as flex
    except Exception:
        return False
    if getattr(module.config, "_attn_implementation", None) != flex.ATTN_KEY:
        return False
    return flex._wants_flex(total, total, 0, 0, device) and not flex._attentions_requested()


def _flex_block_mask(batch, total, window, device):
    from ..models import inkling2_flex as flex
    if window is None:

        def mask_function(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx

    else:

        def mask_function(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx - kv_idx < window)

    return flex._build_block_mask(mask_function, batch, total, total, None, device)


def _eager_attention(module):
    eager = getattr(sys.modules.get(type(module).__module__), "eager_attention_forward", None)
    if eager is None:
        raise NotImplementedError(
            f"Unsloth: Ulysses context parallelism found no eager_attention_forward for {type(module).__name__}."
        )
    return (
        getattr(eager, "__wrapped__", eager)
        if getattr(eager, "_unsloth_cp_ulysses", False)
        else eager
    )


def ulysses_inkling2_attention_forward(
    module,
    hidden_states,
    attention_mask = None,
    relative_distance = None,
    conv_mask = None,
    cache = None,
    key_hidden_states = None,
    value_hidden_states = None,
    cache_offset = 0,
    *,
    group,
    **kwargs,
):
    """Inkling2{,Loop}{Sliding,Full}Attention.forward (transformers 5.24) on a sequence shard: projections (and the
    k / v short convolutions, with their halo) per shard, attention per head shard over the whole sequence."""
    from .ulysses import head_to_seq, seq_to_head, ulysses_head_plan
    from .ulysses_hybrid import _HeadShard, _whole_sequence_mask

    name = type(module).__name__
    kwargs.pop("past_key_values", None)
    if cache is not None:
        raise NotImplementedError(
            f"Unsloth: Ulysses context parallelism does not support a KV cache in {name}."
        )
    if module.training and (getattr(module, "attention_dropout", 0) or 0) != 0:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support attention dropout."
        )
    for key in ("packed_seq_lengths", "cu_seq_lens_q", "cu_seq_lens_k", "seq_idx"):
        if kwargs.get(key) is not None:
            raise NotImplementedError(
                f"Unsloth: Ulysses context parallelism does not support packed rows for {name}."
            )
    key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
    value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
    position_ids = kwargs["position_ids"]
    batch, local = hidden_states.shape[:2]
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    total = local * size

    query = module.q_proj(hidden_states, position_ids)
    key = module.k_proj(
        key_hidden_states, cache = None, conv_mask = conv_mask, cache_offset = cache_offset, **kwargs
    )
    value = module.v_proj(
        value_hidden_states, cache = None, conv_mask = conv_mask, cache_offset = cache_offset, **kwargs
    )
    if key.dtype != query.dtype:
        key = key.to(query.dtype)
    if value.dtype != query.dtype:
        value = value.to(query.dtype)

    bias = module.relative_position_bias
    relative_states = bias.linear(hidden_states).view(batch, local, bias.num_heads, -1)
    relative_states = bias.norm(relative_states, position_ids)
    rel_logits = relative_states @ bias.proj  # [B, L / N, H, R]
    sinks = module.sink_proj(hidden_states)  # [B, L / N, H]

    n_heads = query.shape[1]
    plan = ulysses_head_plan(n_heads, key.shape[1], size)
    if plan.kv_replication > 1:
        key = key.repeat_interleave(plan.kv_replication, dim = 1)
        value = value.repeat_interleave(plan.kv_replication, dim = 1)
    # [B, H, L / N, *] -> [B, H / N, L, *]
    query, key, value = (seq_to_head(x, group) for x in (query, key, value))
    rel_logits = seq_to_head(rel_logits.transpose(1, 2).contiguous(), group)
    sinks = seq_to_head(sinks.transpose(1, 2).unsqueeze(-1).contiguous(), group)
    window = getattr(module, "sliding_window", None)
    window = (
        window if isinstance(window, int) and not isinstance(window, bool) and window > 0 else None
    )

    if _flex_wanted(module, total, query.device):
        from ..models import inkling2_flex as flex

        flex._count("flex")
        block_mask = _flex_block_mask(batch, total, window, query.device)
        out = flex.inkling2_flex_attention(
            query,
            key,
            value,
            block_mask,
            rel_logits.transpose(1, 2),  # [B, L, H / N, R]
            sinks.squeeze(-1).transpose(1, 2),  # [B, L, H / N]
            module.scaling,
            bias.rel_extent,
        )
    else:
        shard = _HeadShard(module, num_key_value_groups = plan.local_groups)
        mask = _whole_sequence_mask(total, query.device, query.dtype, window, None)
        out, _ = _eager_attention(module)(
            shard,
            query,
            key,
            value,
            mask,
            dropout = 0.0,
            scaling = module.scaling,
            position_bias = _whole_sequence_bias(rel_logits, total, bias.rel_extent),
            sinks = sinks,
        )
    # [B, L, H / N, D] -> [B, L / N, H, D]
    out = head_to_seq(out.reshape(batch, total, plan.local_heads, -1), group)
    out = out.reshape(batch, local, -1).contiguous()
    return module.o_proj(out, hidden_states), None


# ---------------------------------------------------------------------------------------------------------------------
# Wrapping


@torch.compiler.disable
def _run_eager(fn, *args, **kwargs):
    # Collectives and the halo stay out of any enclosing compiled region.
    return fn(*args, **kwargs)


def _active(get_cp_manager):
    manager = get_cp_manager()
    if manager is None or manager.backend != "ulysses":
        return None
    if getattr(manager, "packed_seq_lengths", None) is not None:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support packed rows for Inkling 2 (its short "
            "convolutions and relative position bias would need document boundaries). Use packing = False."
        )
    return manager


def wrap_inkling2_attention_forward(module, get_cp_manager) -> None:
    if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
        return
    forward = module.forward

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        manager = _active(get_cp_manager)
        if manager is None:
            return forward(*args, **kwargs)
        # Looked up per call so tests can substitute it.
        return _run_eager(
            globals()["ulysses_inkling2_attention_forward"],
            module,
            *args,
            group = manager.group,
            **kwargs,
        )

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped


def wrap_inkling2_conv_forward(module, get_cp_manager) -> None:
    if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
        return
    forward = module.forward

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        manager = _active(get_cp_manager)
        if manager is None:
            return forward(*args, **kwargs)
        if "past_key_values" in kwargs and "cache" not in kwargs:
            kwargs["cache"] = kwargs.pop("past_key_values")
        return _run_eager(
            globals()["ulysses_inkling2_conv_forward"],
            module,
            forward,
            *args,
            group = manager.group,
            **kwargs,
        )

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped
