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

"""Wire LLaMA-Factory Ulysses, ms-swift GDN CP, and FSDP2 into one Unsloth model."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def init_process_group():
    if dist.is_initialized():
        return
    dist.init_process_group(backend = "nccl")
    torch.cuda.set_device(local_rank())


def local_rank() -> int:
    return int(dist.get_rank()) if dist.is_initialized() else 0


def world_size() -> int:
    return int(dist.get_world_size()) if dist.is_initialized() else 1


def _batch_positions(position_ids, batch: int, seq_len: int):
    """Text positions as [batch, seq]. mRoPE stores the text channel on axis 0."""
    if position_ids is None or not torch.is_tensor(position_ids):
        return None
    pos = position_ids
    if pos.ndim == 3 and pos.shape[-1] == seq_len and pos.shape[0] in (3, 4):
        pos = pos[0]
    elif pos.ndim == 2 and pos.shape[-1] == seq_len and pos.shape[0] in (3, 4) and batch == 1:
        pos = pos[:1]
    elif pos.ndim == 2 and pos.shape[-1] == seq_len:
        pass
    else:
        return None
    if pos.shape[0] == 1 and batch > 1:
        pos = pos.expand(batch, -1)
    return pos


def _blocked_key_mask(attention_mask, position_ids, batch: int, seq_len: int, device):
    """Bool mask [batch, 1, seq, seq], True where a query may attend.

    Dense monotonic sequences return None so SDPA keeps the is_causal kernel.
    A 0 in a 2-D padding mask blocks that key. A position id that does not
    increase starts a new packed segment, and queries stay inside their segment.
    """
    pos = _batch_positions(position_ids, batch, seq_len)
    if pos is not None:
        pos = pos.to(device)
    packed = pos is not None and seq_len > 1 and bool((pos[:, 1:] <= pos[:, :-1]).any())
    pad = None
    if (
        torch.is_tensor(attention_mask)
        and attention_mask.ndim == 2
        and attention_mask.shape[-1] == seq_len
    ):
        pad = attention_mask
        if pad.shape[0] == 1 and batch > 1:
            pad = pad.expand(batch, -1)
        if int((pad == 0).sum()) == 0:
            pad = None
    if not packed and pad is None:
        return None

    index = torch.arange(seq_len, device = device)
    keep = index[None, :] <= index[:, None]
    keep = keep.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len).clone()
    if packed:
        reset = torch.zeros(batch, seq_len, dtype = torch.bool, device = device)
        reset[:, 1:] = pos[:, 1:] <= pos[:, :-1]
        segment = reset.long().cumsum(dim = -1)
        same = segment[:, :, None] == segment[:, None, :]
        keep &= same[:, None, :, :]
    if pad is not None:
        keep &= pad[:, None, None, :] != 0
    return keep


def sdpa_flash_fn(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length = None,
    is_causal = True,
    dropout = 0.0,
    softmax_scale = None,
    **kwargs,
):
    """Inner attention for Ulysses. Layout in and out is [batch, seq, heads, dim]."""
    del query_length, dropout
    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)
    keep = _blocked_key_mask(
        attention_mask,
        kwargs.get("position_ids"),
        query.shape[0],
        query.shape[2],
        query.device,
    )
    attn_mask = None
    causal = is_causal
    if keep is not None:
        # Float mask: 0 keeps the score, a large negative drops the key.
        # Bool masks disagree across torch versions about which value is blocked.
        attn_mask = torch.zeros(keep.shape, dtype = torch.float32, device = query.device)
        attn_mask.masked_fill_(~keep, torch.finfo(torch.float32).min)
        causal = False
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask = attn_mask,
        is_causal = causal,
        scale = softmax_scale,
        enable_gqa = query.shape[1] != key.shape[1],
    )
    return out.transpose(1, 2).contiguous()


def text_model(model):
    for module in model.modules():
        if type(module).__name__.endswith("TextModel") and hasattr(module, "layers"):
            return module
    raise RuntimeError("no Qwen text model with .layers")


def apply_cp(model, cp_group) -> int:
    from .gdn_cp import apply_gdn_cp
    from .ulysses import apply_ulysses_attention

    cp = dist.get_world_size(cp_group)
    apply_ulysses_attention(model, cp, cp_group, attn_fn = sdpa_flash_fn)
    n_gdn = apply_gdn_cp(model, cp_group)
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "flash_attention_2"
    if hasattr(model, "config"):
        model.config.use_cache = False
    return n_gdn


def _layer_classes(model) -> set[type]:
    found = set()
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            found.add(type(layers[0]))
    return found


def apply_fsdp(model, mesh) -> None:
    """Same wrap order as LLaMA-Factory FSDP2Engine.prepare_model."""
    policy = MixedPrecisionPolicy(
        param_dtype = torch.bfloat16,
        reduce_dtype = torch.float32,
        cast_forward_inputs = True,
    )
    # Shard decoder layers, not each LoRA leaf. A leaf wrap plus the parent
    # wrap feeds aten.mm a mix of Tensor and DTensor.
    classes = _layer_classes(model)
    for module in model.modules():
        if type(module) in classes:
            fully_shard(module, mesh = mesh, reshard_after_forward = True, mp_policy = policy)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def apply_fsdp_checkpoint(model) -> int:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    def is_decoder(module):
        return type(module).__name__.endswith("Qwen3_5MoeDecoderLayer")

    n = sum(1 for module in model.modules() if is_decoder(module))
    if n == 0:
        return 0
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = lambda module: checkpoint_wrapper(
            module, checkpoint_impl = CheckpointImpl.NO_REENTRANT
        ),
        check_fn = is_decoder,
    )
    return n


def build_meshes(use_fsdp: bool, use_cp: bool):
    """Two meshes over the same GPUs, matching LLaMA-Factory on 2 cards.

    FSDP shards weights. CP shards the sequence. Both can be on at once:
    dp stays 1, so the two ranks own one sample together.
    """
    init_process_group()
    world = world_size()
    device = torch.device("cuda", local_rank())
    fsdp_mesh = None
    cp_group = None
    if use_fsdp:
        if world < 2:
            raise RuntimeError("FSDP needs 2 processes")
        fsdp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names = ("fsdp",))
    if use_cp:
        if world < 2:
            raise RuntimeError("CP needs 2 processes")
        cp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names = ("cp",))
        cp_group = cp_mesh.get_group()
    return device, fsdp_mesh, cp_group


def _chunked_nll(
    lm_head,
    hidden,
    labels,
    chunk: int = 128,
):
    """Sum of token NLL. lm_head is applied on short chunks so the
    vocab-sized logit tensor (about 30 GiB at 65536) is never allocated.
    """
    total = hidden.new_zeros((), dtype = torch.float32)
    length = hidden.shape[1]
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        logits = lm_head(hidden[:, start:stop]).float()
        piece = labels[:, start:stop]
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            piece.reshape(-1),
            reduction = "sum",
            ignore_index = -100,
        )
        del logits
    return total


def sft_loss(model, input_ids, labels, cp_group):
    """Shifted token CE. CP keeps a contiguous shard of the sequence."""
    position_ids = None
    shift_labels = F.pad(labels[:, 1:], (0, 1), value = -100)
    if cp_group is not None:
        cp = dist.get_world_size(cp_group)
        rank = dist.get_rank(cp_group)
        local = input_ids.shape[1] // cp
        start = rank * local
        input_ids = input_ids[:, start : start + local].contiguous()
        shift_labels = shift_labels[:, start : start + local].contiguous()
        position_ids = torch.arange(start, start + local, device = input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    outputs = base.model(input_ids = input_ids, position_ids = position_ids, use_cache = False)
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    local_num = _chunked_nll(base.lm_head, hidden, shift_labels)
    denom = (labels[:, 1:] != -100).sum().clamp_min(1).to(local_num.dtype)
    if cp_group is None:
        return local_num / denom
    from torch.distributed.nn.functional import all_gather

    parts = all_gather(local_num, group = cp_group)
    return torch.stack(parts).sum() / denom


def sync_replicated_grads(model, cp_group) -> None:
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op = dist.ReduceOp.SUM, group = cp_group)
