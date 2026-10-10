# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bound Qwen-Image-2.1 math attention in both segmented prefill and cached decode."""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

QUERY_CHUNK_SIZE = 512
# MiB one score tensor may take before MPS attention is split; unset = 1/8 of the GPU working set.
SCORE_BUDGET_ENV = "UNSLOTH_DIFFUSION_ATTN_SCORE_BUDGET_MB"


def mps_score_budget() -> int:
    """Bytes one ``batch x heads x queries x keys`` score tensor may take on MPS before attention is split.

    Sized so 1024x1024 in bf16 on a 16 GB Mac (about 1.1 GB of scores) stays one call, exactly as before; only sizes
    that would otherwise swap (2048x2048 is about 17 GB per tensor) are split."""
    raw = (os.environ.get(SCORE_BUDGET_ENV) or "").strip()
    if raw:
        try:
            return max(0, int(float(raw) * 2**20))
        except ValueError:
            pass
    try:
        import torch

        total = int(torch.mps.recommended_max_memory())
    except Exception:  # noqa: BLE001 - no MPS runtime: the fixed floor below
        total = 0
    return total // 8 if total > 0 else 2**30


def _query_rows(query, key, budget: Optional[int]) -> int:
    """Query rows per call: ``QUERY_CHUNK_SIZE`` without a budget, else as many as the budget holds (at least that)."""
    if budget is None:
        return QUERY_CHUNK_SIZE
    per_row = query.shape[0] * query.shape[2] * key.shape[1] * query.element_size()
    return max(QUERY_CHUNK_SIZE, budget // max(per_row, 1))


def _bounded_attention(
    query,
    key,
    value,
    dispatch,
    *,
    mask = None,
    causal_offset = None,
    budget = None,
    **kwargs,
):
    import torch

    rows = _query_rows(query, key, budget)
    outputs = []
    for start in range(0, query.shape[1], rows):
        part = query[:, start : start + rows]
        part_mask = mask
        if mask is not None and mask.shape[-2] != 1:
            part_mask = mask[..., start : start + rows, :]
        if causal_offset is not None:
            # The segment can see all preceding segments, then a triangle over its own keys.
            causal = torch.arange(key.shape[1], device = query.device)[None, :] <= (
                causal_offset + start + torch.arange(part.shape[1], device = query.device)[:, None]
            )
            causal = causal[None, None]
            part_mask = causal if part_mask is None else causal & part_mask
        outputs.append(dispatch(part, key, value, attn_mask = part_mask, dropout_p = 0.0, **kwargs))
    if not outputs:
        return query.new_empty((*query.shape[:-1], value.shape[-1]))
    return torch.cat(outputs, dim = 1)


def _eligible(processor, hidden_states, attention_mask, segments):
    import torch
    from .diffusion_qwenimage21_rocm import _native_backend

    if (
        processor._parallel_config is not None
        or torch.is_grad_enabled()
        or not _native_backend(processor)
    ):
        return False
    # Segmented prefill ignores the flex BlockMask, exactly as the upstream stock processor does.
    return (
        segments is not None
        or attention_mask is None
        or (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim >= 2
            and attention_mask.shape[-2] in (1, hidden_states.shape[1])
        )
    )


@functools.lru_cache(maxsize = 1)
def _processor_class():
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor,
        _qwenimage21_prepare_qkv,
        dispatch_attention_fn,
    )

    class BoundedMathQwenImage21AttnProcessor(QwenImage21AttnProcessor):
        # None: fixed QUERY_CHUNK_SIZE rows (math-only ROCm). Bytes: split only past that score size (MPS).
        _unsloth_score_budget = None

        def __call__(
            self,
            attn,
            hidden_states,
            attention_mask = None,
            rotary_emb = None,
            layer_cache = None,
            kv_cache_mode = None,
            cache_write_slice = None,
            segments = None,
            key_valid = None,
        ):
            if not _eligible(self, hidden_states, attention_mask, segments):
                return super().__call__(
                    attn,
                    hidden_states,
                    attention_mask,
                    rotary_emb,
                    layer_cache,
                    kv_cache_mode,
                    cache_write_slice,
                    segments,
                    key_valid,
                )
            query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
                attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
            )
            if segments is None:
                output = _bounded_attention(
                    query,
                    key,
                    value,
                    dispatch_attention_fn,
                    mask = attention_mask,
                    budget = self._unsloth_score_budget,
                    backend = self._attention_backend,
                    parallel_config = self._parallel_config,
                )
            else:
                outputs = []
                for start, end, is_text in segments:
                    outputs.append(
                        _bounded_attention(
                            query[:, start:end],
                            key[:, :end],
                            value[:, :end],
                            dispatch_attention_fn,
                            mask = None if key_valid is None else key_valid[:, None, None, :end],
                            causal_offset = start if is_text else None,
                            budget = self._unsloth_score_budget,
                            backend = None,
                            parallel_config = self._parallel_config,
                        )
                    )
                prefix_len = segments[-1][1] if segments else 0
                outputs.append(
                    _bounded_attention(
                        query[:, prefix_len:],
                        key,
                        value,
                        dispatch_attention_fn,
                        mask = None if key_valid is None else key_valid[:, None, None, :],
                        budget = self._unsloth_score_budget,
                        backend = None,
                        parallel_config = self._parallel_config,
                    )
                )
                output = torch.cat(outputs, dim = 1)
            output = output[:, :seq_len_q].flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](output))

    return BoundedMathQwenImage21AttnProcessor


def needs_bounded_attention(target: Any) -> bool:
    """ROCm with only SDPA math, or MPS.

    MPS SDPA ignores ``sdpa_kernel`` (so the probe reads every backend as available) and, in the torch Studio installs
    on macOS, builds the full score matrix plus its softmax copy for any query over 8 tokens: about 34 GB per call at
    2048x2048 in bf16, which the uncapped MPS allocator serves from swap."""
    if getattr(target, "device", None) == "mps":
        return True
    if getattr(target, "backend", None) != "rocm":
        return False
    from .diffusion_attention import sdpa_math_only

    return sdpa_math_only(target)


def install(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> bool:
    """A correctness fallback for every ROCm architecture and Apple Silicon, independent of speed mode."""
    from .diffusion_qwenimage21_rocm import _processor_class as speed_processor_class
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

    if not needs_bounded_attention(target):
        return False
    transformer = getattr(pipe, "transformer", None)
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        return False
    cls = _processor_class()
    budget = mps_score_budget() if getattr(target, "device", None) == "mps" else None
    changed = False
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if type(processor) in (QwenImage21AttnProcessor, speed_processor_class()):
            replacement = cls()
            replacement.__dict__.update(processor.__dict__)
            replacement._unsloth_score_budget = budget
            module.set_processor(replacement)
            changed = True
    if changed and logger is not None:
        logger.info(
            "diffusion.qwenimage21: bounding math attention to %s, including prefill",
            "512 query rows" if budget is None else f"{budget / 2**30:.2f} GiB of scores per call",
        )
    return bounded_math_attention(pipe)


def bounded_math_attention(pipe: Any) -> bool:
    """Credit bounded memory only when every attention processor on this transformer qualifies."""
    from .diffusion_qwenimage21_rocm import _native_backend

    transformer = getattr(pipe, "transformer", None)
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        return False
    processors = [m.processor for m in transformer.modules() if hasattr(m, "processor")]
    return bool(processors) and all(
        type(p) is _processor_class() and p._parallel_config is None and _native_backend(p)
        for p in processors
    )
