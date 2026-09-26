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

"""Qwen3.5 GatedDeltaNet context parallel.

The all-to-all and the head-sliced conv / A_log / dt_bias follow
ms-swift ``_run_qwen3_5_gated_delta_net_sequence_parallel_forward``.
The conv and chunk kernels are the ones Hugging Face already bound
on this transformers build (FLA hub kernel, else the torch fallback).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .seq_comm import SeqAllToAll4D


def _local_conv(mod, rank: int, cp: int):
    local_k = mod.num_k_heads // cp
    local_v = mod.num_v_heads // cp
    local_key = local_k * mod.head_k_dim
    local_value = local_v * mod.head_v_dim
    weight = mod.conv1d.weight.squeeze(1)
    bias = mod.conv1d.bias
    q0 = rank * local_key
    k0 = mod.key_dim + rank * local_key
    v0 = 2 * mod.key_dim + rank * local_value
    weight = torch.cat(
        [
            weight[q0 : q0 + local_key],
            weight[k0 : k0 + local_key],
            weight[v0 : v0 + local_value],
        ],
        dim=0,
    )
    if bias is not None:
        bias = torch.cat(
            [
                bias[q0 : q0 + local_key],
                bias[k0 : k0 + local_key],
                bias[v0 : v0 + local_value],
            ],
            dim=0,
        )
    return weight, bias


def gdn_forward_with_cp(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        apply_mask_to_padding_states,
        causal_conv1d_fn,
        torch_chunk_gated_delta_rule,
    )

    group = self._cp_group
    cp = dist.get_world_size(group)
    if cp <= 1:
        return self._cp_original_forward(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    if cache_params is not None:
        raise RuntimeError("GDN context parallel is the training path; cache is off.")

    rank = dist.get_rank(group)
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, seq_len, _ = hidden_states.shape
    if self.num_k_heads % cp != 0 or self.num_v_heads % cp != 0:
        raise RuntimeError(
            f"GDN CP needs cp={cp} to divide key heads {self.num_k_heads} "
            f"and value heads {self.num_v_heads}."
        )

    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q_proj = SeqAllToAll4D.apply(group, q_proj.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim), 2, 1)
    k_proj = SeqAllToAll4D.apply(group, k_proj.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim), 2, 1)
    v_proj = SeqAllToAll4D.apply(group, v_proj.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim), 2, 1)
    b = SeqAllToAll4D.apply(group, b.reshape(batch, seq_len, self.num_v_heads, 1), 2, 1).squeeze(-1)
    a = SeqAllToAll4D.apply(group, a.reshape(batch, seq_len, self.num_v_heads, 1), 2, 1).squeeze(-1)

    full_seq = q_proj.shape[1]
    local_k = self.num_k_heads // cp
    local_v = self.num_v_heads // cp
    mixed = torch.cat(
        [
            q_proj.reshape(batch, full_seq, local_k * self.head_k_dim),
            k_proj.reshape(batch, full_seq, local_k * self.head_k_dim),
            v_proj.reshape(batch, full_seq, local_v * self.head_v_dim),
        ],
        dim=-1,
    ).transpose(1, 2).contiguous()
    conv_weight, conv_bias = _local_conv(self, rank, cp)
    mixed = causal_conv1d_fn(mixed, conv_weight, conv_bias, activation=self.activation)
    mixed = mixed.transpose(1, 2)
    local_key = local_k * self.head_k_dim
    local_value = local_v * self.head_v_dim
    query, key, value = torch.split(mixed, [local_key, local_key, local_value], dim=-1)
    query = query.reshape(batch, full_seq, local_k, self.head_k_dim)
    key = key.reshape(batch, full_seq, local_k, self.head_k_dim)
    value = value.reshape(batch, full_seq, local_v, self.head_v_dim)
    if local_v // local_k > 1:
        repeat = local_v // local_k
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    head_slice = slice(rank * local_v, (rank + 1) * local_v)
    g = -self.A_log[head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[head_slice])
    beta = b.sigmoid()
    core, _ = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    core = SeqAllToAll4D.apply(group, core, 1, 2)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core = core.reshape(batch, seq_len, -1)
    return self.out_proj(core)


def apply_gdn_cp(model, group) -> int:
    n = 0
    for module in model.modules():
        if not type(module).__name__.endswith("GatedDeltaNet"):
            continue
        if getattr(module, "_cp_original_forward", None) is None:
            module._cp_original_forward = module.forward
        module._cp_group = group
        module.forward = gdn_forward_with_cp.__get__(module, type(module))
        n += 1
    return n
