# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallel for MiniMax Lightning Attention.

MiniMaxLightningAttention projects hidden states locally via qkv_proj,
transposes from sequence-sharded to head-sharded layout via SeqAllToAll4D,
runs the exact block-wise exponential decay recurrence over the full sequence
on each rank's head slice, and transposes back to sequence-sharded layout
before applying the full-hidden RMSNorm, output_gate, and out_proj.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .seq_comm import SeqAllToAll4D


def _apply_mask(mod, qkv_states, attention_mask):
    module = sys.modules.get(type(mod).__module__)
    fn = getattr(module, "apply_mask_to_padding_states", None) if module is not None else None
    if fn is not None:
        return fn(qkv_states, attention_mask)
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = qkv_states.dtype
        qkv_states = (qkv_states * attention_mask[:, :, None]).to(dtype)
    return qkv_states


def lightning_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values = None,
    **kwargs,
):
    group = self._cp_group
    cp = dist.get_world_size(group)
    if cp <= 1:
        return self._cp_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            **kwargs,
        )
    if past_key_values is not None:
        raise RuntimeError(
            "MiniMaxLightningAttention context parallel is the training path; cache is off."
        )

    rank = dist.get_rank(group)
    batch_size, local_seq_len, _ = hidden_states.shape
    if self.num_attention_heads % cp != 0:
        raise RuntimeError(
            f"MiniMaxLightningAttention CP needs cp={cp} to divide "
            f"num_attention_heads={self.num_attention_heads}."
        )
    local_heads = self.num_attention_heads // cp
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)

    qkv_states = self.act_fn(self.qkv_proj(hidden_states))
    qkv_states = _apply_mask(self, qkv_states, attention_mask)
    qkv_states = qkv_states.reshape(
        batch_size, local_seq_len, self.num_attention_heads, 3 * self.head_dim
    )
    query_states, key_states, value_states = torch.split(qkv_states, self.head_dim, dim = 3)

    # Sequence-sharded [B, S/cp, H, D] -> Head-sharded [B, S, H/cp, D]
    query_states = SeqAllToAll4D.apply(group, query_states.contiguous(), 2, 1).transpose(1, 2)
    key_states = SeqAllToAll4D.apply(group, key_states.contiguous(), 2, 1).transpose(1, 2)
    value_states = SeqAllToAll4D.apply(group, value_states.contiguous(), 2, 1).transpose(1, 2)

    full_seq_len = query_states.shape[2]
    num_blocks = (full_seq_len + self.block_size - 1) // self.block_size
    target_dtype = value_states.dtype

    slope_rate = self.slope_rate[head_slice].to(target_dtype)
    query_decay = self.query_decay[head_slice].to(target_dtype)
    key_decay = self.key_decay[head_slice].to(target_dtype)
    diagonal_decay = self.diagonal_decay[:, head_slice].to(target_dtype)

    attn_weights_inter = torch.zeros(
        batch_size,
        local_heads,
        self.head_dim,
        self.head_dim,
        dtype = target_dtype,
        device = value_states.device,
    )

    attn_output = []
    for i in range(num_blocks):
        start_idx = i * self.block_size
        end_idx = min(start_idx + self.block_size, full_seq_len)
        current_block_size = end_idx - start_idx

        current_query_states = query_states[:, :, start_idx:end_idx]
        current_key_states = key_states[:, :, start_idx:end_idx]
        current_value_states = value_states[:, :, start_idx:end_idx]

        current_query_decay = query_decay[:, :current_block_size]
        current_key_decay = key_decay[:, -current_block_size:]
        current_diagonal_decay = diagonal_decay[:, :, :current_block_size, :current_block_size]
        block_decay = torch.exp(-slope_rate * current_block_size).to(target_dtype)

        attn_weights_intra = torch.matmul(
            current_query_states, current_key_states.transpose(-1, -2)
        )
        attn_output_intra = torch.matmul(
            (attn_weights_intra * current_diagonal_decay).to(target_dtype),
            current_value_states,
        )
        attn_output_inter = torch.matmul(
            (current_query_states * current_query_decay).to(target_dtype),
            attn_weights_inter,
        )
        current_attn_output = attn_output_inter + attn_output_intra
        attn_output.append(current_attn_output)

        next_attn_weights_inter = torch.matmul(
            (current_key_states * current_key_decay).to(target_dtype).transpose(-1, -2),
            current_value_states,
        )
        attn_weights_inter = attn_weights_inter * block_decay + next_attn_weights_inter

    attn_output = torch.cat(attn_output, dim = -2)

    # Head-sharded [B, H/cp, S, D] -> Sequence-sharded [B, S/cp, H, D]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = SeqAllToAll4D.apply(group, attn_output, 1, 2)
    attn_output = attn_output.reshape(
        batch_size, local_seq_len, self.num_attention_heads * self.head_dim
    )
    attn_output = self.norm(attn_output)
    attn_output = F.sigmoid(self.output_gate(hidden_states)) * attn_output
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_inter


def apply_lightning_cp(model, cp_group: dist.ProcessGroup) -> int:
    """Bind the context-parallel forward onto every MiniMaxLightningAttention in model."""
    patched = 0
    for module in model.modules():
        if not type(module).__name__.endswith("MiniMaxLightningAttention"):
            continue
        for buf_name in ("slope_rate", "query_decay", "key_decay", "diagonal_decay"):
            buf = getattr(module, buf_name, None)
            if buf is not None and hasattr(module, "qkv_proj"):
                setattr(module, buf_name, buf.to(module.qkv_proj.weight.dtype))
        module._cp_group = cp_group
        if not hasattr(module, "_cp_original_forward"):
            module._cp_original_forward = module.forward
        module.forward = types.MethodType(lightning_forward_with_cp, module)
        patched += 1
    return patched
