# Copyright 2025 Bytedance Ltd. and/or its affiliates. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's verl library.
# https://github.com/verl-project/verl/blob/77476af84cc074edf5a6437f8d5ea418d7a54916/verl/utils/ulysses.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
import transformers
from torch import Tensor
from torch.distributed import ProcessGroup

from .seq_comm import SeqAllToAll4D


class _Log:
    def info_rank0(self, msg):
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized() and _dist.get_rank() != 0:
            return
        print(msg, flush = True)


logger = _Log()

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    """Set ulysses sequence parallel process group."""
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get ulysses sequence parallel process group."""
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """Get ulysses sequence parallel world size."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """Get ulysses sequence parallel rank."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def _get_text_position_ids(position_ids: Optional[Tensor]) -> Optional[Tensor]:
    # Transformers < 5.4 broadcasts Qwen3.5 text positions over the mRoPE axes.
    if position_ids is not None and position_ids.ndim == 3 and position_ids.stride(0) == 0:
        position_ids = position_ids[0]

    return (
        position_ids.contiguous() if position_ids is not None and position_ids.ndim == 2 else None
    )


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        attn_fn: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_fn = attn_fn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        dropout_p = 0.0,
        softmax_scale = None,
        position_ids: Optional[torch.Tensor] = None,
        causal = True,
        deterministic = False,
        target_dtype = None,
        sliding_window = None,
        softcap = None,
        s_aux = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Forward.

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            attention_mask (Tensor): attention mask for the layer
            query_length (int): the length of the query sequence
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            softmax_scale (float, optional): scale factor for softmax. Defaults to None,
            position_ids (torch.Tensor, optional): position ids for the attention. Defaults to None.
            causal (bool, optional): whether to apply causal mask. Defaults to True.
            deterministic (bool, optional): whether to apply dropout in deterministic way. Defaults to False.
            target_dtype (torch.dtype, optional): target dtype for attention output. Defaults to None.
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        sp_world_size = get_ulysses_sequence_parallel_world_size(self.spg)
        kv_heads = key.shape[self.scatter_idx]
        if kv_heads % sp_world_size != 0:
            repeat_factor = sp_world_size // math.gcd(kv_heads, sp_world_size)
            key = key.repeat_interleave(repeat_factor, dim = self.scatter_idx)
            value = value.repeat_interleave(repeat_factor, dim = self.scatter_idx)

        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        if s_aux is not None and torch.is_tensor(s_aux) and s_aux.ndim == 1:
            rank = get_ulysses_sequence_parallel_rank(self.spg)
            local_q_heads = q.shape[self.scatter_idx]
            if s_aux.shape[0] == local_q_heads * sp_world_size:
                s_aux = s_aux[rank * local_q_heads : (rank + 1) * local_q_heads]

        # HF FlashAttention only uses 2-D position IDs to detect packed sequences.
        position_ids = _get_text_position_ids(position_ids)
        if position_ids is not None:
            global_position_ids = [torch.empty_like(position_ids) for _ in range(sp_world_size)]
            dist.all_gather(global_position_ids, position_ids, group = self.spg)
            position_ids = torch.cat(global_position_ids, dim = -1).contiguous()

        # HF may turn an all-ones local attention_mask into None before this
        # function. Under CP, different ranks can then disagree: some local
        # shards still contain padding and keep a mask, while others see None.
        # Synchronize that boolean first so every rank takes the same collective
        # path below.
        has_attention_mask = torch.tensor(
            [attention_mask is not None], dtype = torch.int64, device = query.device
        )
        global_has_attention_mask = [
            torch.empty_like(has_attention_mask) for _ in range(sp_world_size)
        ]
        dist.all_gather(global_has_attention_mask, has_attention_mask, group = self.spg)

        # Padded path: at least one shard has real padding, so rebuild the full
        # sequence mask for all ranks. Ranks whose local mask was optimized away
        # contribute an all-ones shard.
        if torch.any(torch.stack(global_has_attention_mask)):
            if attention_mask is None:
                attention_mask = torch.ones(
                    query.shape[0], query.shape[1], dtype = torch.int64, device = query.device
                )
            else:
                attention_mask = attention_mask.to(torch.int64)

            attention_mask = attention_mask.contiguous()
            global_attention_mask = [torch.empty_like(attention_mask) for _ in range(sp_world_size)]
            dist.all_gather(global_attention_mask, attention_mask, group = self.spg)
            attention_mask = torch.cat(global_attention_mask, dim = 1).contiguous()

        # Packed/dense path: no rank has a mask, so leave attention_mask as None.
        # HF can then use position_ids for padding-free packed varlen attention,
        # or dense flash attention when position_ids are monotonic.
        context_layer = self.attn_fn(
            q,
            k,
            v,
            attention_mask,
            query_length = query_length,
            is_causal = causal,
            dropout = dropout_p,
            position_ids = position_ids,
            softmax_scale = softmax_scale,
            deterministic = deterministic,
            target_dtype = target_dtype,
            sliding_window = sliding_window,
            softcap = softcap,
            s_aux = s_aux,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size = 1,
    dropout = 0,
    deterministic = False,
    is_causal = True,
    group = None,
    mode = "ulysses",
    attn_fn = None,
    target_dtype = None,
    softmax_scale = None,
    sliding_window = None,
    softcap = None,
    s_aux = None,
    **kwargs,
):
    """Route causal language attention through Ulysses and leave replicated encoders native."""
    if mode == "ulysses":
        if not is_causal:
            return attn_fn(
                query_states,
                key_states,
                value_states,
                attention_mask,
                is_causal = False,
                dropout = dropout,
                deterministic = deterministic,
                target_dtype = target_dtype,
                softmax_scale = softmax_scale,
                sliding_window = sliding_window,
                softcap = softcap,
                s_aux = s_aux,
                **kwargs,
            )

        dist_attn = UlyssesAttention(sequence_process_group = group, attn_fn = attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length = query_states.shape[1] * sequence_parallel_size,
            deterministic = deterministic,
            dropout_p = dropout,
            softmax_scale = softmax_scale,
            causal = is_causal,
            position_ids = kwargs.get("position_ids", None),
            target_dtype = target_dtype,
            sliding_window = sliding_window,
            softcap = softcap,
            s_aux = s_aux,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def apply_ulysses_attention(
    model,
    cp_size: int,
    group: dist.ProcessGroup,
    attn_fn = None,
) -> None:
    """Validate and install the Ulysses FlashAttention bridge for one process group."""
    # Replace _flash_attention_forward with new_flash_attn_forward
    set_ulysses_sequence_parallel_group(group)

    try:
        num_attention_heads = model.config.num_attention_heads
    except AttributeError:
        num_attention_heads = model.config.text_config.num_attention_heads

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"

    origin_attn = attn_fn or transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group = get_ulysses_sequence_parallel_group(),
        mode = "ulysses",
        attn_fn = origin_attn,
        sequence_parallel_size = cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "")
                == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue

    if attn_fn is not None:
        # When routing through PyTorch SDPA (`sdpa_flash_fn`), avoid HF's
        # `flash_attention_forward` zero-padding MLA `v_head_dim` up to `qk_head_dim`:
        # keeping native `v_head_dim` cuts all-to-all volume and matches SDPA numerics.
        try:
            from transformers.integrations import flash_attention as _fa_mod
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            def _ulysses_fa_interface(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                dropout: float = 0.0,
                scaling: Optional[float] = None,
                sliding_window: Optional[int] = None,
                softcap: Optional[float] = None,
                is_causal: Optional[bool] = None,
                s_aux: Optional[torch.Tensor] = None,
                **kwargs,
            ):
                seq_len = query.shape[2]
                v_head_dim = getattr(module, "v_head_dim", None)
                if v_head_dim is not None and value.shape[-1] > v_head_dim:
                    value = value[..., :v_head_dim]
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                target_dtype = _fa_mod.get_target_dtype(query, module)
                is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
                attn_output = _fa_mod._flash_attention_forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    query_length = seq_len,
                    is_causal = is_causal,
                    dropout = dropout,
                    softmax_scale = scaling,
                    sliding_window = sliding_window,
                    softcap = softcap,
                    use_top_left_mask = _fa_mod._use_top_left_mask,
                    target_dtype = target_dtype,
                    attn_implementation = getattr(module.config, "_attn_implementation", None),
                    layer_idx = getattr(module, "layer_idx", None),
                    s_aux = s_aux.to(query.dtype) if s_aux is not None else None,
                    **kwargs,
                )
                return attn_output, None

            _fa_mod.flash_attention_forward = _ulysses_fa_interface
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _ulysses_fa_interface
        except Exception:
            pass
