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

"""Utilities for enabling packed (padding-free) batches across Unsloth."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch

try:
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask as _XFormersBlockMask,
    )
except Exception:
    try:
        from xformers.attn_bias import BlockDiagonalCausalMask as _XFormersBlockMask
    except Exception:
        _XFormersBlockMask = None

_XFORMERS_MASK_CACHE_MAXSIZE = 32
_XFORMERS_MASK_CACHE: OrderedDict[Tuple[Tuple[int, ...], int], Any] = OrderedDict()


def _window_cache_key(sliding_window: Optional[int]) -> int:
    if sliding_window is None or sliding_window <= 0:
        return 0
    return int(sliding_window)


def _get_cached_block_mask(
    lengths: Tuple[int, ...],
    sliding_window: Optional[int],
):
    if _XFormersBlockMask is None:
        return None

    window_key = _window_cache_key(sliding_window)
    cache_key = (lengths, window_key)
    cached = _XFORMERS_MASK_CACHE.get(cache_key)
    if cached is not None:
        _XFORMERS_MASK_CACHE.move_to_end(cache_key)
        return cached

    mask = _XFormersBlockMask.from_seqlens(list(lengths))
    if window_key and mask is not None and hasattr(mask, "make_local_attention"):
        mask = mask.make_local_attention(window_size = window_key)

    _XFORMERS_MASK_CACHE[cache_key] = mask
    if len(_XFORMERS_MASK_CACHE) > _XFORMERS_MASK_CACHE_MAXSIZE:
        _XFORMERS_MASK_CACHE.popitem(last = False)
    return mask


class _TrlPackingWarningFilter(logging.Filter):
    to_filter = (
        "attention implementation is not",
        "kernels-community",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(substring in message for substring in self.to_filter)


_TRL_FILTER_INSTALLED = False


def _ensure_trl_warning_filter():
    global _TRL_FILTER_INSTALLED
    if _TRL_FILTER_INSTALLED:
        return
    logging.getLogger("trl.trainer.sft_trainer").addFilter(_TrlPackingWarningFilter())
    _TRL_FILTER_INSTALLED = True


def mark_allow_overlength(module):
    """Mark a module hierarchy so padding-free batches can exceed max_seq_length."""
    if module is None:
        return
    if hasattr(module, "max_seq_length"):
        setattr(module, "_unsloth_allow_packed_overlength", True)
    children = getattr(module, "children", None)
    if children is None:
        return
    for child in children():
        mark_allow_overlength(child)


def configure_sample_packing(config):
    """Mutate an ``SFTConfig`` so TRL prepares packed batches."""
    _ensure_trl_warning_filter()
    setattr(config, "packing", True)
    setattr(config, "padding_free", True)


def configure_padding_free(config):
    """Mutate an ``SFTConfig`` so TRL enables padding-free batching without packing."""
    _ensure_trl_warning_filter()
    setattr(config, "padding_free", True)


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for packed batches on an existing trainer."""
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    mark_allow_overlength(model)

    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_packing_wrapped", False):
        return

    if hasattr(collator, "padding_free"):
        collator.padding_free = True
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_lengths(examples: Sequence[dict]):
        batch = original_torch_call(examples)
        if examples and isinstance(examples[0], dict):
            seq_lengths: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    seq_lengths.extend(int(length) for length in lengths)
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(
                    seq_lengths, dtype = torch.int32
                )
                if "attention_mask" in batch:
                    batch.pop("attention_mask")
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True


def enable_padding_free_metadata(model, trainer):
    """Inject seq-length metadata when padding-free batching is enabled without packing."""
    collator = getattr(trainer, "data_collator", None)
    if (
        collator is None
        or getattr(collator, "_unsloth_padding_free_lengths_wrapped", False)
        or not getattr(collator, "padding_free", False)
    ):
        return

    mark_allow_overlength(model)
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_padding_free_metadata(examples: Sequence[dict]):
        seq_lengths: list[int] = []
        if examples and isinstance(examples[0], dict):
            for example in examples:
                lengths = example.get("seq_lengths")
                if lengths is None:
                    ids = example.get("input_ids")
                    if ids is None:
                        continue
                    lengths = [len(ids)]
                    example["seq_lengths"] = lengths
                seq_lengths.extend(lengths)

        batch = original_torch_call(examples)
        if seq_lengths:
            batch["packed_seq_lengths"] = torch.tensor(
                seq_lengths,
                dtype = torch.int32,
            )
        return batch

    collator.torch_call = torch_call_with_padding_free_metadata
    collator._unsloth_padding_free_lengths_wrapped = True


def get_packed_info_from_kwargs(
    kwargs: dict,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Return packed sequence metadata expected by the attention kernels."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    lengths = seq_lengths.to(device = device, dtype = torch.int32, non_blocking = True)
    cu_seqlens = torch.empty(lengths.numel() + 1, dtype = torch.int32, device = device)
    cu_seqlens[0] = 0
    torch.cumsum(lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])

    max_seqlen = int(lengths.max().item())
    return lengths, cu_seqlens, max_seqlen


def build_xformers_block_causal_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
    base_mask: Optional[Any] = None,
):
    if _XFormersBlockMask is None:
        return None
    if seq_info is not None:
        seq_lengths, _, _ = seq_info
        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        mask = _get_cached_block_mask(lengths, sliding_window)
    else:
        mask = base_mask

        if (
            sliding_window is not None
            and sliding_window > 0
            and mask is not None
            and hasattr(mask, "make_local_attention")
        ):
            mask = mask.make_local_attention(window_size = sliding_window)
    return mask


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info
    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype = dtype,
        device = device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype = dtype, device = device)
        upper = torch.triu(
            torch.ones((length, length), device = device), diagonal = 1
        ).bool()
        block = block.masked_fill(upper, float("-inf"))
        if (
            sliding_window is not None
            and sliding_window > 0
            and length > sliding_window
        ):
            idx = torch.arange(length, device = device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            window_mask = dist >= sliding_window
            block = block.masked_fill(window_mask, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length
    return mask.unsqueeze(0).unsqueeze(0)


def _normalize_packed_lengths(
    seq_lengths: Any,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if seq_lengths is None:
        return None
    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device = device, dtype = torch.int64)
    else:
        lengths = torch.tensor(seq_lengths, device = device, dtype = torch.int64)
    if lengths.ndim != 1:
        lengths = lengths.reshape(-1)
    if lengths.numel() == 0:
        return None
    return lengths


def mask_packed_sequence_boundaries(
    shift_labels: torch.Tensor,
    seq_lengths: Any,
    *,
    ignore_index: int = -100,
) -> bool:
    """Mark final token of every packed sample so CE ignores boundary predictions."""
    lengths = _normalize_packed_lengths(seq_lengths, device = shift_labels.device)
    if lengths is None:
        return False

    flat = shift_labels.reshape(-1)
    total_tokens = flat.shape[0]
    boundary_positions = torch.cumsum(lengths, dim = 0) - 1
    valid = boundary_positions < total_tokens
    if not torch.all(valid):
        boundary_positions = boundary_positions[valid]
    if boundary_positions.numel() == 0:
        return False
    flat[boundary_positions] = ignore_index
    return True


__all__ = [
    "configure_sample_packing",
    "configure_padding_free",
    "enable_sample_packing",
    "enable_padding_free_metadata",
    "mark_allow_overlength",
    "get_packed_info_from_kwargs",
    "build_xformers_block_causal_mask",
    "build_sdpa_packed_attention_mask",
    "mask_packed_sequence_boundaries",
]
