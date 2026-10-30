# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for DeepSeek attention that bypasses ALL_ATTENTION_FUNCTIONS.

DeepSeek-V3.2 folds a DSA indexer's top-k selection into its mask, and DeepSeek-V4 mixes sliding,
compressed sparse (CSA) and heavily compressed (HCA) sink attention with K == V. Both forwards are
replaced while Ulysses is active: projections run on the local sequence shard, queries move to head
shards, and the indexer / compressor see the full sequence. Attention runs in query chunks so the
score matrix is [H / N, chunk, T] instead of [H / N, T, T] (UNSLOTH_CP_ATTN_CHUNK, default 1024);
with grad each chunk is recomputed in backward by a single autograd op, so this composes with
activation checkpointing of the decoder layer.
"""

import functools
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .ulysses import _AllToAll

__all__ = [
    "DEEPSEEK_ULYSSES_ATTENTION_CLASSES",
    "ulysses_deepseek_v4_forward",
    "ulysses_deepseek_v32_forward",
    "wrap_deepseek_attention_forward",
]

DEEPSEEK_ULYSSES_ATTENTION_CLASSES = ("DeepseekV32Attention", "DeepseekV4Attention")


def _apply_rotary_fn(mod):
    module = sys.modules[type(mod).__module__]
    return module.apply_rotary_pos_emb


def _gather_seq(
    x,
    group,
    dim = 1,
):
    """All-gather sequence shards along ``dim``; backward reduce-scatters the summed gradient.

    torch.distributed.nn.all_gather's gloo backward scatters from group ranks as if they were global
    ranks, so it fails on every CP group but the first under DP x CP.
    """
    from ..context_parallel import _GatherSequence

    out = _GatherSequence.apply(
        x.movedim(dim, 1), group, dist.get_rank(group), dist.get_world_size(group)
    )
    return out.movedim(1, dim)


def _attn_chunk() -> int:
    return max(1, int(os.environ.get("UNSLOTH_CP_ATTN_CHUNK", "1024")))


def _sink_attention_chunk(
    q, kv, block_bias, sinks, q_start, full_seq, window, scaling, dropout_p, training
):
    """Sink-softmax attention for the query rows ``[q_start, q_start + q.shape[2])``.

    q: [B, H, rows, D]; kv: [B, 1, full_seq + n_blocks, D] (K == V); block_bias: [B, 1, rows, n_blocks].
    Returns [B, rows, H, D].
    """
    batch, _, rows, _ = q.shape
    dtype = q.dtype
    q_idx = torch.arange(q_start, q_start + rows, device = q.device)
    k_idx = torch.arange(full_seq, device = q.device)
    diff = q_idx[:, None] - k_idx[None, :]
    keep = diff >= 0
    if window is not None:
        keep = keep & (diff < window)
    mask = torch.zeros((1, 1, rows, full_seq), dtype = dtype, device = q.device)
    mask.masked_fill_(~keep.view(1, 1, rows, full_seq), torch.finfo(dtype).min)
    extra = kv.shape[2] - full_seq
    if extra > 0:
        if block_bias is not None:
            mask = torch.cat([mask.expand(batch, -1, -1, -1), block_bias.to(dtype)], dim = -1)
        else:
            mask = F.pad(mask, (0, extra), value = 0.0)

    logits = torch.matmul(q, kv.transpose(2, 3)) * scaling + mask
    sink = sinks.reshape(1, -1, 1, 1).expand(batch, -1, rows, -1)
    combined = torch.cat([logits, sink], dim = -1)
    combined = combined - combined.max(dim = -1, keepdim = True).values
    probs = F.softmax(combined, dim = -1, dtype = combined.dtype)[..., :-1]
    probs = F.dropout(probs, p = dropout_p, training = training).to(kv.dtype)
    return torch.matmul(probs, kv).transpose(1, 2).contiguous()


def _refuse_dropout(dropout_p, training):
    if training and dropout_p:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support attention dropout."
        )


def _recompute_rows(chunk, full_seq, run, grad_out, row_inputs, full_inputs, needs):
    """Backward of a query-chunked op: recompute each chunk and accumulate its input gradients.

    run(start, end, *rows, *full) -> [B, rows, H, D]; row_inputs are sliced on dim 2, full_inputs are
    not; needs is ctx.needs_input_grad for row_inputs + full_inputs, in that order.
    """
    row_need, full_need = needs[: len(row_inputs)], needs[len(row_inputs) :]
    row_grads = [torch.zeros_like(t) if n else None for t, n in zip(row_inputs, row_need)]
    full_grads = [
        torch.zeros(t.shape, dtype = torch.float32, device = t.device) if n else None
        for t, n in zip(full_inputs, full_need)
    ]
    for start in range(0, full_seq, chunk):
        end = min(full_seq, start + chunk)
        with torch.enable_grad():
            rows = [
                t[:, :, start:end].detach().requires_grad_(n) for t, n in zip(row_inputs, row_need)
            ]
            full = [t.detach().requires_grad_(n) for t, n in zip(full_inputs, full_need)]
            out = run(start, end, *rows, *full)
            wanted = [t for t in rows + full if t.requires_grad]
            grads = iter(torch.autograd.grad(out, wanted, grad_out[:, start:end]))
        for i, t in enumerate(rows):
            if t.requires_grad:
                row_grads[i][:, :, start:end] = next(grads)
        for i, t in enumerate(full):
            if t.requires_grad:
                full_grads[i] += next(grads).float()
    full_grads = [g if g is None else g.to(t.dtype) for g, t in zip(full_grads, full_inputs)]
    return row_grads, full_grads


class _ChunkedSinkAttention(torch.autograd.Function):
    """Sink attention over query chunks; only the inputs are saved and each chunk is recomputed in
    backward. Being one autograd op (no nested torch.utils.checkpoint) it composes with activation
    checkpointing of the whole layer; custom_fwd / custom_bwd replay the forward's autocast state."""

    @staticmethod
    @torch.amp.custom_fwd(device_type = "cuda")
    def forward(ctx, q, kv, sinks, block_bias, window, scaling, chunk):
        ctx.save_for_backward(q, kv, sinks, block_bias)
        ctx.window, ctx.scaling, ctx.chunk = window, scaling, chunk
        full_seq = q.shape[2]
        with torch.no_grad():
            outs = [
                _sink_attention_chunk(
                    q[:, :, s : s + chunk],
                    kv,
                    None if block_bias is None else block_bias[:, :, s : s + chunk],
                    sinks,
                    s,
                    full_seq,
                    window,
                    scaling,
                    0.0,
                    False,
                )
                for s in range(0, full_seq, chunk)
            ]
        return torch.cat(outs, dim = 1)

    @staticmethod
    @torch.amp.custom_bwd(device_type = "cuda")
    def backward(ctx, grad_out):
        q, kv, sinks, block_bias = ctx.saved_tensors
        full_seq = q.shape[2]

        def run(start, end, q_rows, kv_full, sinks_full):
            bias = None if block_bias is None else block_bias[:, :, start:end]
            return _sink_attention_chunk(
                q_rows,
                kv_full,
                bias,
                sinks_full,
                start,
                full_seq,
                ctx.window,
                ctx.scaling,
                0.0,
                False,
            )

        (gq,), (gkv, gsinks) = _recompute_rows(
            ctx.chunk, full_seq, run, grad_out, [q], [kv, sinks], ctx.needs_input_grad[:3]
        )
        return gq, gkv, gsinks, None, None, None, None


def _sink_attention(q, kv, block_bias, sinks, window, scaling, dropout_p, training):
    """Query-chunked sink attention so the score matrix is [H/cp, chunk, S] rather than [H/cp, S, S].

    With grad, chunks are recomputed in backward (_ChunkedSinkAttention), so peak memory is
    O(chunk * S) per layer.
    """
    _refuse_dropout(dropout_p, training)
    full_seq = q.shape[2]
    chunk = _attn_chunk()
    if block_bias is not None and block_bias.requires_grad:
        raise RuntimeError("Unsloth: DeepSeek-V4 block bias is a mask and must not require grad.")
    if torch.is_grad_enabled() and full_seq > chunk:
        return _ChunkedSinkAttention.apply(q, kv, sinks, block_bias, window, scaling, chunk)
    outs = [
        _sink_attention_chunk(
            q[:, :, s : s + chunk],
            kv,
            None if block_bias is None else block_bias[:, :, s : s + chunk],
            sinks,
            s,
            full_seq,
            window,
            scaling,
            0.0,
            False,
        )
        for s in range(0, full_seq, chunk)
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim = 1)


def ulysses_deepseek_v4_forward(
    module,
    hidden_states,
    position_embeddings = None,
    position_ids = None,
    attention_mask = None,
    past_key_values = None,
    *,
    group,
    **kwargs,
):
    """Training forward of a DeepSeek-V4 attention layer on a contiguous sequence shard."""
    cp = dist.get_world_size(group)
    if past_key_values is not None:
        raise RuntimeError("DeepSeek-V4 context parallel is the training path; cache is off.")
    if module.num_heads % cp != 0:
        raise RuntimeError(
            f"DeepSeek-V4 CP needs cp={cp} to divide attention heads {module.num_heads}."
        )

    apply_rotary_pos_emb = _apply_rotary_fn(module)
    rank = dist.get_rank(group)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = position_embeddings[module.rope_layer_type]

    q_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    q = module.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = module.q_b_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin)

    kv = module.kv_norm(module.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)

    # Scatter query heads across cp ranks and gather full sequence [B, H/cp, S_full, D].
    q = _AllToAll.apply(q, 1, 2, group)
    # Gather single-head KV across sequence [B, 1, S_full, D]; backward sums the gradients
    # of all head-partitioned ranks back to each rank's token shard.
    kv = _gather_seq(kv, group, dim = 2)

    local_heads = q.shape[1]
    block_bias = None
    if module.compressor is not None:
        full_hidden = _gather_seq(hidden_states, group)
        full_q_res = _gather_seq(q_residual, group)
        pos_local = position_ids.contiguous()
        pos_list = [torch.empty_like(pos_local) for _ in range(cp)]
        dist.all_gather(pos_list, pos_local, group = group)
        full_pos_ids = torch.cat(pos_list, dim = -1)
        compressed_kv, block_bias = module.compressor(
            full_hidden, full_q_res, full_pos_ids, None, module.layer_idx
        )
        kv = torch.cat([kv, compressed_kv], dim = 2)

    attn_output = _sink_attention(
        q,
        kv,
        block_bias,
        module.sinks[rank * local_heads : (rank + 1) * local_heads],
        module.sliding_window,
        module.scaling,
        0.0 if not module.training else module.attention_dropout,
        module.training,
    )

    # [B, S_full, H/cp, D] -> [B, S_local, H, D]
    attn_output = _AllToAll.apply(attn_output, 1, 2, group)
    attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

    grouped = attn_output.reshape(*input_shape, module.config.o_groups, -1)
    grouped = module.o_a_proj(grouped).flatten(2)
    output = module.o_b_proj(grouped)
    return output, None


@torch.no_grad()
def _v32_global_topk(indexer, hidden_states, q_resid, position_embeddings, group, module):
    """DSA indexer top-k for this rank's queries against the *full* key sequence, then gathered.

    Indexer keys are a single 128-d vector per token, so gathering them is cheap. Scores are
    ``[B, chunk, H_idx, S]`` in fp32, so queries are processed in chunks.
    Returns int32 indices ``[B, S_full, topk]`` identical on every rank.
    """
    cp = dist.get_world_size(group)
    rank = dist.get_rank(group)
    batch, local_seq, _ = hidden_states.shape
    cos, sin = position_embeddings
    rope = indexer.qk_rope_head_dim

    q = indexer.wq_b(q_resid).view(batch, local_seq, indexer.n_heads, indexer.head_dim)
    q_rot, q_pass = torch.split(q, [rope, indexer.head_dim - rope], dim = -1)
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [rope, indexer.head_dim - rope], dim = -1)
    q_rot, k_rot = module.apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim = 2)
    q = torch.cat([q_rot, q_pass], dim = -1)
    k = torch.cat([k_rot, k_pass], dim = -1).squeeze(2).contiguous()

    k_parts = [torch.empty_like(k) for _ in range(cp)]
    dist.all_gather(k_parts, k, group = group)
    k_full = torch.cat(k_parts, dim = 1)  # [B, S, D]
    full_seq = k_full.shape[1]

    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float()
    weights = weights * (indexer.n_heads**-0.5)
    k_t = k_full.transpose(-1, -2).float().unsqueeze(1)  # [B, 1, D, S]
    k_idx = torch.arange(full_seq, device = hidden_states.device)
    topk = min(indexer.index_topk, full_seq)
    chunk = max(1, _attn_chunk() // 4)
    q_start = rank * local_seq

    parts = []
    for s in range(0, local_seq, chunk):
        e = min(local_seq, s + chunk)
        scores = torch.matmul(q[:, s:e].float(), k_t) * indexer.softmax_scale  # [B, c, H, S]
        scores = F.relu(scores)
        index_scores = torch.matmul(weights[:, s:e].unsqueeze(-2), scores).squeeze(-2)  # [B, c, S]
        q_pos = torch.arange(q_start + s, q_start + e, device = hidden_states.device)
        causal = q_pos[:, None] >= k_idx[None, :]
        index_scores = index_scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        parts.append(index_scores.topk(topk, dim = -1).indices.to(torch.int32))
    local_topk = torch.cat(parts, dim = 1).contiguous()

    gathered = [torch.empty_like(local_topk) for _ in range(cp)]
    dist.all_gather(gathered, local_topk, group = group)
    return torch.cat(gathered, dim = 1)


def _sparse_mla_chunk(q, k, v, topk, q_start, scaling, dropout_p, training):
    """Causal attention restricted to each query's indexer-selected keys. Returns [B, rows, H, Dv]."""
    batch, _, rows, _ = q.shape
    full_seq = k.shape[2]
    q_idx = torch.arange(q_start, q_start + rows, device = q.device)
    k_idx = torch.arange(full_seq, device = q.device)
    causal = q_idx[:, None] >= k_idx[None, :]
    selected = torch.zeros((batch, rows, full_seq), dtype = torch.bool, device = q.device)
    selected.scatter_(-1, topk.long(), True)
    keep = (selected & causal.unsqueeze(0)).unsqueeze(1)  # [B, 1, rows, S]

    logits = torch.matmul(q, k.transpose(2, 3)) * scaling
    logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
    probs = F.softmax(logits, dim = -1, dtype = torch.float32).to(q.dtype)
    probs = F.dropout(probs, p = dropout_p, training = training)
    return torch.matmul(probs, v).transpose(1, 2).contiguous()


class _ChunkedSparseMLA(torch.autograd.Function):
    """Indexer-masked MLA attention over query chunks, recomputed per chunk in backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type = "cuda")
    def forward(ctx, q, k, v, topk, scaling, chunk):
        ctx.save_for_backward(q, k, v, topk)
        ctx.scaling, ctx.chunk = scaling, chunk
        full_seq = q.shape[2]
        with torch.no_grad():
            outs = [
                _sparse_mla_chunk(
                    q[:, :, s : s + chunk], k, v, topk[:, s : s + chunk], s, scaling, 0.0, False
                )
                for s in range(0, full_seq, chunk)
            ]
        return torch.cat(outs, dim = 1)

    @staticmethod
    @torch.amp.custom_bwd(device_type = "cuda")
    def backward(ctx, grad_out):
        q, k, v, topk = ctx.saved_tensors

        def run(start, end, q_rows, k_full, v_full):
            return _sparse_mla_chunk(
                q_rows, k_full, v_full, topk[:, start:end], start, ctx.scaling, 0.0, False
            )

        (gq,), (gk, gv) = _recompute_rows(
            ctx.chunk, q.shape[2], run, grad_out, [q], [k, v], ctx.needs_input_grad[:3]
        )
        return gq, gk, gv, None, None, None


def _sparse_mla_attention(q, k, v, topk, scaling, dropout_p, training):
    _refuse_dropout(dropout_p, training)
    full_seq = q.shape[2]
    chunk = _attn_chunk()
    if torch.is_grad_enabled() and full_seq > chunk:
        return _ChunkedSparseMLA.apply(q, k, v, topk, scaling, chunk)
    outs = [
        _sparse_mla_chunk(
            q[:, :, s : s + chunk], k, v, topk[:, s : s + chunk], s, scaling, 0.0, False
        )
        for s in range(0, full_seq, chunk)
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim = 1)


def ulysses_deepseek_v32_forward(
    module,
    hidden_states,
    position_embeddings = None,
    attention_mask = None,
    past_key_values = None,
    position_ids = None,
    *,
    group,
    **kwargs,
):
    """Training forward of DeepSeek-V3.2 MLA + DSA indexer on a contiguous sequence shard."""
    cp = dist.get_world_size(group)
    if past_key_values is not None:
        raise RuntimeError("DeepSeek-V3.2 context parallel is the training path; cache is off.")
    if module.num_heads % cp != 0:
        raise RuntimeError(
            f"DeepSeek-V3.2 CP needs cp={cp} to divide attention heads {module.num_heads}."
        )

    modeling = sys.modules[type(module).__module__]
    batch, local_seq = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    q_resid = module.q_a_layernorm(module.q_a_proj(hidden_states))
    q_states = (
        module.q_b_proj(q_resid).view(batch, local_seq, -1, module.qk_head_dim).transpose(1, 2)
    )
    q_pass, q_rot = torch.split(
        q_states, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim = -1
    )

    kv_pass, k_rot = torch.split(
        module.kv_a_proj_with_mqa(hidden_states),
        [module.kv_lora_rank, module.qk_rope_head_dim],
        dim = -1,
    )
    k_pass = module.kv_a_layernorm(kv_pass).view(batch, 1, local_seq, module.kv_lora_rank)
    k_rot = k_rot.view(batch, 1, local_seq, module.qk_rope_head_dim)
    q_rot, k_rot = modeling.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    query_states = torch.cat((q_pass, q_rot), dim = -1)
    key_states, value_states = module.expand_kv(k_pass, k_rot)

    topk = _v32_global_topk(
        module.indexer, hidden_states, q_resid, position_embeddings, group, modeling
    )

    # [B, H, S_local, D] -> [B, H/cp, S_full, D]
    q = _AllToAll.apply(query_states, 1, 2, group)
    k = _AllToAll.apply(key_states, 1, 2, group)
    v = _AllToAll.apply(value_states, 1, 2, group)

    attn_output = _sparse_mla_attention(
        q,
        k,
        v,
        topk,
        module.scaling,
        0.0 if not module.training else module.attention_dropout,
        module.training,
    )
    # [B, S_full, H/cp, Dv] -> [B, S_local, H, Dv]
    attn_output = _AllToAll.apply(attn_output, 1, 2, group)
    attn_output = attn_output.reshape(batch, local_seq, -1).contiguous()
    return module.o_proj(attn_output), None


@torch.compiler.disable
def _run_eager(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def wrap_deepseek_attention_forward(module, get_cp_manager) -> None:
    if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
        return
    forward = module.forward
    name = type(module).__name__
    v4 = name == "DeepseekV4Attention"

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        manager = get_cp_manager()
        if manager is None or manager.backend != "ulysses":
            return forward(*args, **kwargs)
        packed = kwargs.get("packed_seq_lengths")
        if packed is not None or getattr(manager, "packed_seq_lengths", None) is not None:
            raise NotImplementedError(
                f"Unsloth: Ulysses context parallelism does not support packing for {name}."
            )
        # Looked up per call (tests substitute it). Eager even inside Unsloth's per-layer compiled
        # region: traced, the list all_gather / sequence gather / no_grad indexer top-k crashed the
        # first CP step (SIGSEGV on L4, torch 2.8 and 2.14).
        ulysses_forward = ulysses_deepseek_v4_forward if v4 else ulysses_deepseek_v32_forward
        return _run_eager(ulysses_forward, module, *args, group = manager.group, **kwargs)

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped
