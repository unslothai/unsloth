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
        if type(module).__name__ != "DeepseekV4Attention":
            continue
        if getattr(module, "_cp_original_forward", None) is None:
            module._cp_original_forward = module.forward
        module._cp_group = group
        module.forward = deepseek_v4_attention_forward_with_cp.__get__(module, type(module))
        n += 1
    return n
