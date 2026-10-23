# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek family context parallel (MLA + DeepSeek-V4 CSA / HCA / Sink MQA)."""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .seq_comm import SeqAllToAll4D


def _apply_rotary_fn(mod):
    module = sys.modules[type(mod).__module__]
    return module.apply_rotary_pos_emb


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


def _sink_attention(q, kv, block_bias, sinks, window, scaling, dropout_p, training):
    """Query-chunked sink attention so the score matrix is [H/cp, chunk, S] rather than [H/cp, S, S].

    Each chunk is recomputed in backward (checkpoint), so peak memory is O(chunk * S) per layer.
    """
    full_seq = q.shape[2]
    chunk = _attn_chunk()
    outs = []
    for start in range(0, full_seq, chunk):
        end = min(full_seq, start + chunk)
        args = (
            q[:, :, start:end],
            kv,
            block_bias[:, :, start:end] if block_bias is not None else None,
            sinks,
            start,
            full_seq,
            window,
            scaling,
            dropout_p,
            training,
        )
        if torch.is_grad_enabled() and full_seq > chunk:
            outs.append(checkpoint(_sink_attention_chunk, *args, use_reentrant = False))
        else:
            outs.append(_sink_attention_chunk(*args))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim = 1)


def deepseek_v4_attention_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values = None,
    **kwargs,
):
    group = self._cp_group
    cp = dist.get_world_size(group)
    if cp <= 1:
        return self._cp_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            position_ids = position_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            **kwargs,
        )
    if past_key_values is not None:
        raise RuntimeError("DeepSeek-V4 context parallel is the training path; cache is off.")
    if self.num_heads % cp != 0:
        raise RuntimeError(
            f"DeepSeek-V4 CP needs cp={cp} to divide attention heads {self.num_heads}."
        )

    apply_rotary_pos_emb = _apply_rotary_fn(self)
    rank = dist.get_rank(group)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]

    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = self.q_b_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin)

    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)

    # Scatter query heads across cp ranks and gather full sequence [B, H/cp, S_full, D].
    q = SeqAllToAll4D.apply(group, q, 1, 2)
    # Gather single-head KV across sequence [B, 1, S_full, D]; _AllGather backward
    # sums gradients across all head-partitioned ranks back to each rank's token shard.
    kv = torch.cat(dist_nn.all_gather(kv.contiguous(), group = group), dim = 2)

    local_heads = q.shape[1]
    block_bias = None
    if self.compressor is not None:
        full_hidden = torch.cat(
            dist_nn.all_gather(hidden_states.contiguous(), group = group), dim = 1
        )
        full_q_res = torch.cat(
            dist_nn.all_gather(q_residual.contiguous(), group = group), dim = 1
        )
        pos_local = position_ids.contiguous()
        pos_list = [torch.empty_like(pos_local) for _ in range(cp)]
        dist.all_gather(pos_list, pos_local, group = group)
        full_pos_ids = torch.cat(pos_list, dim = -1)
        compressed_kv, block_bias = self.compressor(
            full_hidden, full_q_res, full_pos_ids, None, self.layer_idx
        )
        kv = torch.cat([kv, compressed_kv], dim = 2)

    attn_output = _sink_attention(
        q,
        kv,
        block_bias,
        self.sinks[rank * local_heads : (rank + 1) * local_heads],
        self.sliding_window,
        self.scaling,
        0.0 if not self.training else self.attention_dropout,
        self.training,
    )

    # [B, S_full, H/cp, D] -> [B, S_local, H, D]
    attn_output = SeqAllToAll4D.apply(group, attn_output, 1, 2)
    attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
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


def _sparse_mla_attention(q, k, v, topk, scaling, dropout_p, training):
    full_seq = q.shape[2]
    chunk = _attn_chunk()
    outs = []
    for start in range(0, full_seq, chunk):
        end = min(full_seq, start + chunk)
        args = (
            q[:, :, start:end],
            k,
            v,
            topk[:, start:end],
            start,
            scaling,
            dropout_p,
            training,
        )
        if torch.is_grad_enabled() and full_seq > chunk:
            outs.append(checkpoint(_sparse_mla_chunk, *args, use_reentrant = False))
        else:
            outs.append(_sparse_mla_chunk(*args))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim = 1)


def deepseek_v32_attention_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    position_embeddings = None,
    attention_mask = None,
    past_key_values = None,
    position_ids = None,
    **kwargs,
):
    """DeepSeek-V3.2 MLA + DSA indexer under Ulysses CP (training path, no padding mask / cache)."""
    group = self._cp_group
    cp = dist.get_world_size(group)
    if cp <= 1:
        return self._cp_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            position_ids = position_ids,
            **kwargs,
        )
    if past_key_values is not None:
        raise RuntimeError("DeepSeek-V3.2 context parallel is the training path; cache is off.")
    if self.num_heads % cp != 0:
        raise RuntimeError(
            f"DeepSeek-V3.2 CP needs cp={cp} to divide attention heads {self.num_heads}."
        )

    module = sys.modules[type(self).__module__]
    batch, local_seq = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(batch, local_seq, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)

    kv_pass, k_rot = torch.split(
        self.kv_a_proj_with_mqa(hidden_states), [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1
    )
    k_pass = self.kv_a_layernorm(kv_pass).view(batch, 1, local_seq, self.kv_lora_rank)
    k_rot = k_rot.view(batch, 1, local_seq, self.qk_rope_head_dim)
    q_rot, k_rot = module.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    query_states = torch.cat((q_pass, q_rot), dim = -1)
    key_states, value_states = self.expand_kv(k_pass, k_rot)

    topk = _v32_global_topk(
        self.indexer, hidden_states, q_resid, position_embeddings, group, module
    )

    # [B, H, S_local, D] -> [B, H/cp, S_full, D]
    q = SeqAllToAll4D.apply(group, query_states, 1, 2)
    k = SeqAllToAll4D.apply(group, key_states, 1, 2)
    v = SeqAllToAll4D.apply(group, value_states, 1, 2)

    attn_output = _sparse_mla_attention(
        q,
        k,
        v,
        topk,
        self.scaling,
        0.0 if not self.training else self.attention_dropout,
        self.training,
    )
    # [B, S_full, H/cp, Dv] -> [B, S_local, H, Dv]
    attn_output = SeqAllToAll4D.apply(group, attn_output, 1, 2)
    attn_output = attn_output.reshape(batch, local_seq, -1).contiguous()
    return self.o_proj(attn_output), None


def _dsv4_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim = True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)


def apply_deepseek_cp(model, group) -> int:
    n = 0
    for module in model.modules():
        if type(module).__name__ == "DeepseekV4RMSNorm":
            module.forward = _dsv4_rmsnorm_forward.__get__(module, type(module))
            continue
        forward_with_cp = _CP_ATTENTION_FORWARDS.get(type(module).__name__)
        if forward_with_cp is None:
            continue
        if getattr(module, "_cp_original_forward", None) is None:
            module._cp_original_forward = module.forward
        module._cp_group = group
        module.forward = forward_with_cp.__get__(module, type(module))
        n += 1
    return n


_CP_ATTENTION_FORWARDS = {
    "DeepseekV4Attention": deepseek_v4_attention_forward_with_cp,
    "DeepseekV32Attention": deepseek_v32_attention_forward_with_cp,
}
