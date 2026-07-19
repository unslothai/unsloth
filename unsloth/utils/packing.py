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

import inspect
import logging
import os
from collections import OrderedDict
from functools import wraps
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

# Cache per device for get_packed_info_from_kwargs to avoid repeated D2H sync across layers
_PACKED_INFO_CACHE: dict = {}

# Cache per device for build_sdpa_packed_attention_mask to avoid repeated D2H sync across layers
_SDPA_MASK_CACHE: dict = {}

# Cache per device for build_xformers_block_causal_mask to avoid repeated D2H sync across layers
_XFORMERS_BLOCK_MASK_CACHE: dict = {}


def _window_cache_key(sliding_window: Optional[int]) -> int:
    if sliding_window is None or sliding_window <= 0:
        return 0
    return int(sliding_window)


def _get_cached_block_mask(lengths: Tuple[int, ...], sliding_window: Optional[int]):
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
    setattr(config, "remove_unused_columns", False)


def configure_padding_free(config):
    """Mutate an ``SFTConfig`` so TRL enables padding-free batching without packing."""
    _ensure_trl_warning_filter()
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
            # Fallback: infer lengths from tokenized inputs when metadata is absent
            if not seq_lengths:
                for example in examples:
                    ids = example.get("input_ids")
                    if isinstance(ids, Iterable):
                        seq_lengths.append(len(ids))
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(seq_lengths, dtype = torch.int32)
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
    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

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


# --- Experimental: correct packing / padding-free for hybrid linear-attention ---
# Qwen3.5 / Qwen3-Next mix a gated-delta recurrence with a causal conv1d. Packing
# flattens the batch, and both ops leak state across sequence boundaries unless we
# pass seq_idx (conv) and cu_seqlens (scan). Both accelerated kernels support this;
# the pure-torch fallbacks do not, so we fail closed. Gated behind an env flag.
#
# The shim overrides only the training/prefill kernels self.causal_conv1d_fn /
# self.chunk_gated_delta_rule per module; the decode kernels (causal_conv1d_update /
# recurrent_gated_delta_rule) are left untouched so generation is unaffected. It
# targets the standard Trainer forward-then-backward loop (recompute-safe under
# gradient checkpointing) and never fires for cached forwards. Following the
# import_fixes.py house style: feature-detect (never version-detect), fail closed,
# idempotent, and emit one deduped diagnostic when it declines to activate.
_HYBRID_PACKING_ENV_VAR = "UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"
_HYBRID_LOGGER = logging.getLogger("unsloth.hybrid_packing")
_HYBRID_WARNED: set = set()


def _hybrid_packing_enabled() -> bool:
    # Read at call time so setting the flag after `import unsloth` still takes effect.
    return os.environ.get(_HYBRID_PACKING_ENV_VAR, "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _hybrid_reject(reason: str) -> bool:
    # One deduped diagnostic explaining why hybrid packing stayed on the padded path.
    if reason not in _HYBRID_WARNED:
        _HYBRID_WARNED.add(reason)
        _HYBRID_LOGGER.warning(
            "Unsloth: hybrid linear-attention packing disabled (padded path): %s.", reason,
        )
    return False


def _iter_gated_delta_modules(model):
    modules, seen = [], set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if type(module).__name__.endswith("GatedDeltaNet") and hasattr(module, "conv1d"):
            modules.append(module)
    return modules


def _hybrid_varlen_kernels_available(gated_delta_modules) -> Optional[str]:
    """None if every module can use the accelerated varlen path, else a short
    reason string. All modules are validated before any are mutated. Signatures
    are read off the captured originals when the module is already wrapped.

    Dispatch (the mixer actually calling self.causal_conv1d_fn /
    self.chunk_gated_delta_rule) is verified at RUNTIME by the handshake in the
    forward wrapper, not statically: Unsloth wraps each module forward with a
    compile-disable shim, so inspect.getsource sees only that wrapper, and every
    supported transformers release dispatches through the instance attribute."""
    if not gated_delta_modules:
        return "no gated-delta modules found"
    for module in gated_delta_modules:
        conv = getattr(module, "_unsloth_varlen_orig_conv", None) or getattr(
            module, "causal_conv1d_fn", None,
        )
        scan = getattr(module, "_unsloth_varlen_orig_scan", None) or getattr(
            module, "chunk_gated_delta_rule", None,
        )
        if conv is None or scan is None:
            return "accelerated kernels missing (install causal_conv1d and fla)"
        if getattr(scan, "__name__", "").startswith("torch_") or getattr(
            conv, "__name__", "",
        ).startswith("torch_"):
            return "pure-torch kernel fallback in use"
        try:
            if "seq_idx" not in inspect.signature(conv).parameters:
                return "conv kernel does not accept seq_idx"
            if "cu_seqlens" not in inspect.signature(scan).parameters:
                return "scan kernel does not accept cu_seqlens"
        except (TypeError, ValueError):
            return "kernel signature not introspectable"
    return None


def _varlen_from_position_ids(position_ids):
    """(cu_seqlens int32[n+1], seq_idx int32[1,T]) for a flattened padding-free
    batch, else None. Padding-free position_ids reset to 0 at each sequence start.
    Only a validated single-row pack is accepted; a normal batch or single
    sequence returns None (a no-op)."""
    if position_ids is None:
        return None
    pos = position_ids
    if pos.dim() == 3:  # MRoPE [n_planes, 1, T] -> text plane is index 0
        pos = pos[0]
    if pos.dim() != 2 or pos.shape[0] != 1:
        return None
    row = pos[0]
    total = row.shape[0]
    starts = (row == 0).nonzero(as_tuple = False).flatten()
    if starts.numel() <= 1 or int(starts[0].item()) != 0:
        return None
    cu_seqlens = torch.cat([
        starts.to(torch.int32),
        torch.tensor([total], dtype = torch.int32, device = row.device),
    ])
    return _seq_idx_from_cu_seqlens(cu_seqlens, total)


def _seq_idx_from_cu_seqlens(cu_seqlens, total):
    """(cu_seqlens int32[n+1], seq_idx int32[1,total]) partitioning [0, total),
    else None. Appends a trailing segment for pad_to_multiple_of zero tokens so the
    boundaries always cover the full flattened length the kernels see."""
    if cu_seqlens is None or cu_seqlens.numel() < 2 or int(cu_seqlens[0].item()) != 0:
        return None
    boundaries = cu_seqlens.to(torch.int32)
    last = int(boundaries[-1].item())
    if last > total:
        return None
    if last < total:  # trailing pad tokens -> one final segment
        boundaries = torch.cat([
            boundaries,
            torch.tensor([total], dtype = torch.int32, device = boundaries.device),
        ])
    lengths = boundaries[1:] - boundaries[:-1]
    if not bool((lengths > 0).all()):
        return None
    seq_idx = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype = torch.int32, device = boundaries.device),
        lengths.to(torch.int64),
    ).unsqueeze(0)
    return boundaries, seq_idx


def _hybrid_varlen_metadata(kwargs):
    """Boundary metadata (cu_seqlens, seq_idx) for one flattened packed forward,
    else None. Prefers the authoritative packed_seq_lengths, falls back to
    reset-style position_ids. Returns None for cached forwards and non-packed
    batches so decode / eval / normal batches are a strict no-op."""
    if kwargs.get("use_cache"):
        return None
    if kwargs.get("past_key_values") is not None or kwargs.get("cache_params") is not None:
        return None
    total, device = None, None
    for key in ("input_ids", "inputs_embeds", "position_ids"):
        tensor = kwargs.get(key)
        if tensor is not None and hasattr(tensor, "shape"):
            total = tensor.shape[1] if key == "inputs_embeds" else tensor.shape[-1]
            device = tensor.device
            break
    if total is None:
        return None
    info = get_packed_info_from_kwargs(kwargs, device)  # authoritative packed_seq_lengths
    if info is not None:
        _, cu_seqlens, _ = info
        built = _seq_idx_from_cu_seqlens(cu_seqlens, total)
        if built is not None:
            return built
    return _varlen_from_position_ids(kwargs.get("position_ids"))


def patch_hybrid_linear_attention_varlen(model) -> bool:
    """Feed seq_idx / cu_seqlens to the gated-delta conv + scan so packing and
    padding-free reset state at sequence boundaries. Gated by
    UNSLOTH_EXPERIMENTAL_HYBRID_PACKING and fail-closed. Returns True when the
    varlen path is active, so the caller may allow packing for the model.
    Idempotent: repeat calls on an already-patched model return True."""
    if not _hybrid_packing_enabled():
        return False
    gated_delta_modules = _iter_gated_delta_modules(model)

    # Idempotency: an already fully-patched model stays active without re-validation.
    if getattr(model, "_unsloth_varlen_forward_wrapped", False) and gated_delta_modules and all(
        getattr(m, "_unsloth_varlen_wrapped", False) for m in gated_delta_modules
    ):
        return True

    reason = _hybrid_varlen_kernels_available(gated_delta_modules)
    if reason is not None:
        return _hybrid_reject(reason)

    # Transactional: every module validated above, now wrap each and stash originals.
    for module in gated_delta_modules:
        if getattr(module, "_unsloth_varlen_wrapped", False):
            continue
        conv_orig, scan_orig = module.causal_conv1d_fn, module.chunk_gated_delta_rule
        module._unsloth_varlen_orig_conv = conv_orig
        module._unsloth_varlen_orig_scan = scan_orig

        @wraps(conv_orig)
        def conv_fn(*args, _orig = conv_orig, _module = module, **kwargs):
            varlen = getattr(_module, "_unsloth_varlen", None)
            if varlen is not None:
                _module._unsloth_varlen_hit = True  # runtime dispatch handshake
                if kwargs.get("seq_idx") is None:
                    kwargs["seq_idx"] = varlen[1]
            return _orig(*args, **kwargs)

        @wraps(scan_orig)
        def scan_fn(*args, _orig = scan_orig, _module = module, **kwargs):
            varlen = getattr(_module, "_unsloth_varlen", None)
            if varlen is not None:
                _module._unsloth_varlen_hit = True
                if kwargs.get("cu_seqlens") is None:
                    kwargs["cu_seqlens"] = varlen[0]
            return _orig(*args, **kwargs)

        module.causal_conv1d_fn = conv_fn
        module.chunk_gated_delta_rule = scan_fn
        module._unsloth_varlen = None
        module._unsloth_varlen_wrapped = True

    # Refresh the boundary stash on the outermost forward (runs once per step,
    # outside gradient-checkpoint recompute, so it stays valid for recomputed inner
    # forwards of the same batch). position_ids / use_cache are read from both
    # positional and keyword args via the bound signature.
    if not getattr(model, "_unsloth_varlen_forward_wrapped", False):
        forward_orig = model.forward
        try:
            forward_sig = inspect.signature(forward_orig)
        except (TypeError, ValueError):
            forward_sig = None

        @wraps(forward_orig)
        def forward_with_varlen(*args, **kwargs):
            try:
                bound = dict(kwargs)
                if forward_sig is not None and args:
                    bound.update(forward_sig.bind_partial(*args).arguments)
                varlen = _hybrid_varlen_metadata(bound)
            except Exception:
                varlen = None
            for module in gated_delta_modules:
                module._unsloth_varlen = varlen
                if varlen is not None:
                    module._unsloth_varlen_hit = False
            out = forward_orig(*args, **kwargs)
            # Runtime dispatch handshake: on the first real packed forward, confirm the
            # shim was actually reached. If not (a future version stopped dispatching
            # through self.<kernel>), warn once so a silent regression is visible.
            if varlen is not None and not getattr(model, "_unsloth_varlen_handshake_done", False):
                model._unsloth_varlen_handshake_done = True
                if not any(getattr(m, "_unsloth_varlen_hit", False) for m in gated_delta_modules):
                    _hybrid_reject("varlen shim never invoked on a packed batch (dispatch changed?)")
            return out

        model.forward = forward_with_varlen
        model._unsloth_varlen_forward_wrapped = True
    return True


def get_packed_info_from_kwargs(
    kwargs: dict, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Return packed sequence metadata expected by the attention kernels."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    entry = _PACKED_INFO_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths:
        return entry["result"]

    lengths = seq_lengths.to(device = device, dtype = torch.int32, non_blocking = True)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype = torch.int32, device = device)
    torch.cumsum(lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])

    max_seqlen = int(lengths.max().item())
    result = (lengths, cu_seqlens, max_seqlen)
    _PACKED_INFO_CACHE[device] = {"seq_lengths": seq_lengths, "result": result}
    return result


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
        # Cache the mask to avoid repeated D2H sync across layers
        device = seq_lengths.device
        params = (sliding_window,)
        entry = _XFORMERS_BLOCK_MASK_CACHE.get(device)
        if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
            return entry["mask"]

        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        mask = _get_cached_block_mask(lengths, sliding_window)

        _XFORMERS_BLOCK_MASK_CACHE[device] = {
            "seq_lengths": seq_lengths,
            "params": params,
            "mask": mask,
        }
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

    params = (dtype, sliding_window)
    entry = _SDPA_MASK_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
        return entry["mask"]

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
        upper = torch.triu(torch.ones((length, length), device = device), diagonal = 1).bool()
        block = block.masked_fill(upper, float("-inf"))
        if sliding_window is not None and sliding_window > 0 and length > sliding_window:
            idx = torch.arange(length, device = device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            window_mask = dist >= sliding_window
            block = block.masked_fill(window_mask, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length

    result = mask.unsqueeze(0).unsqueeze(0)
    _SDPA_MASK_CACHE[device] = {
        "seq_lengths": seq_lengths,
        "params": params,
        "mask": result,
    }
    return result


def _normalize_packed_lengths(seq_lengths: Any, *, device: torch.device) -> Optional[torch.Tensor]:
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


def clear_packed_caches():
    """Release cached masks/metadata to free device memory."""
    _PACKED_INFO_CACHE.clear()
    _SDPA_MASK_CACHE.clear()
    _XFORMERS_BLOCK_MASK_CACHE.clear()


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
    "clear_packed_caches",
]
