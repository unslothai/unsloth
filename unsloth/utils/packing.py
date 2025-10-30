"""Utilities for enabling packed (padding-free) batches across Unsloth."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple

import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except Exception as exc:  # pragma: no cover
    flash_attn_varlen_func = None

try:
    from xformers.attn_bias import BlockDiagonalCausalMask as _XFormersBlockMask
except Exception:  # pragma: no cover
    _XFormersBlockMask = None


class _TrlPackingWarningFilter(logging.Filter):
    _NEEDLES = (
        "Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'",
        "You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        message = record.getMessage()
        return not any(needle in message for needle in self._NEEDLES)


_TRL_FILTER_INSTALLED = False


def _ensure_trl_warning_filter() -> None:
    global _TRL_FILTER_INSTALLED
    if _TRL_FILTER_INSTALLED:
        return
    logging.getLogger("trl.trainer.sft_trainer").addFilter(_TrlPackingWarningFilter())
    _TRL_FILTER_INSTALLED = True


def configure_sample_packing(config) -> None:
    """Mutate an ``SFTConfig`` so TRL prepares packed batches."""
    _ensure_trl_warning_filter()
    setattr(config, "packing", True)
    setattr(config, "padding_free", True)
    setattr(config, "remove_unused_columns", False)


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for packed batches on an existing trainer."""
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    def _mark_allow_overlength(module):
        if hasattr(module, "max_seq_length"):
            setattr(module, "_unsloth_allow_packed_overlength", True)
        for child in module.children():
            _mark_allow_overlength(child)

    _mark_allow_overlength(model)

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
            per_example_counts: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    numeric_lengths = [int(length) for length in lengths]
                    seq_lengths.extend(numeric_lengths)
                    per_example_counts.append(len(numeric_lengths))
                else:
                    per_example_counts.append(0)
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(seq_lengths, dtype=torch.int32)

                position_ids = batch.get("position_ids")
                input_ids = batch.get("input_ids")
                if position_ids is None and input_ids is not None:
                    position_ids = torch.zeros_like(
                        input_ids, dtype=torch.long, device=input_ids.device
                    )

                if position_ids is not None and input_ids is not None:
                    seq_index = 0
                    for row_idx, count in enumerate(per_example_counts):
                        cursor = 0
                        for _ in range(count):
                            length = seq_lengths[seq_index]
                            if length > 0:
                                position_ids[row_idx, cursor : cursor + length] = torch.arange(
                                    length, dtype=torch.long, device=position_ids.device
                                )
                                cursor += length
                            seq_index += 1
                    batch["position_ids"] = position_ids

                if "attention_mask" in batch and getattr(collator, "return_position_ids", False):
                    batch.pop("attention_mask")
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True


def get_packed_info_from_kwargs(
    kwargs: dict,
    total_tokens: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Extract packed sequence information from attention kwargs."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device=device, dtype=torch.int32)
    else:
        lengths = torch.tensor(seq_lengths, device=device, dtype=torch.int32)

    if lengths.ndim > 1:
        lengths = lengths.reshape(-1)

    if lengths.numel() == 0:
        return None

    if int(lengths.sum().item()) != total_tokens:
        return None

    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
        ]
    )
    max_seqlen = int(lengths.max().item())
    return lengths, cu_seqlens, max_seqlen


def build_xformers_block_causal_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int]
):
    if _XFormersBlockMask is None:
        return None
    seq_lengths, _, _ = seq_info
    lengths = seq_lengths.to("cpu", torch.int32).tolist()
    if not lengths:
        return None
    return _XFormersBlockMask.from_seqlens(lengths)


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info
    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype=dtype, device=device)
        upper = torch.triu(torch.ones((length, length), device=device), diagonal=1).bool()
        block = block.masked_fill(upper, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length
    return mask.unsqueeze(0).unsqueeze(0)

__all__ = [
    "configure_sample_packing",
    "enable_sample_packing",
]
