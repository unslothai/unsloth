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

import sys

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

from .seq_comm import SeqAllToAll4D


def _apply_rotary_fn(mod):
    module = sys.modules[type(mod).__module__]
    return module.apply_rotary_pos_emb


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

    batch, local_heads, full_seq, _ = q.shape
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

    idx = torch.arange(full_seq, device = q.device)
    diff = idx[:, None] - idx[None, :]
    keep = diff >= 0
    if self.sliding_window is not None:
        keep = keep & (diff < self.sliding_window)
    full_mask = torch.zeros(
        (batch, 1, full_seq, full_seq), dtype = q.dtype, device = q.device
    )
    full_mask.masked_fill_(~keep.view(1, 1, full_seq, full_seq), torch.finfo(q.dtype).min)

    if kv.shape[2] > full_mask.shape[-1]:
        if block_bias is not None:
            full_mask = torch.cat([full_mask, block_bias.to(full_mask.dtype)], dim = -1)
        else:
            full_mask = F.pad(full_mask, (0, kv.shape[2] - full_mask.shape[-1]), value = 0.0)

    key_states = kv.expand(-1, local_heads, -1, -1)
    value_states = key_states
    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling
    attn_weights = attn_weights + full_mask

    sinks = self.sinks[rank * local_heads : (rank + 1) * local_heads]
    sinks = sinks.reshape(1, -1, 1, 1).expand(batch, -1, full_seq, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim = -1)
    combined_logits = combined_logits - combined_logits.max(dim = -1, keepdim = True).values
    probs = F.softmax(combined_logits, dim = -1, dtype = combined_logits.dtype)
    scores = probs[..., :-1]
    dropout_p = 0.0 if not self.training else self.attention_dropout
    attn_weights = F.dropout(scores, p = dropout_p, training = self.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()

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
