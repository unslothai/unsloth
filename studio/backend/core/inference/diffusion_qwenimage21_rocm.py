# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed up large Qwen-Image-2.1 ROCm attention with query chunks.

Unmasked, noncausal queries are independent; each chunk keeps all keys and values.
"""

from __future__ import annotations

import functools
import os
from typing import Any

QUERY_CHUNK_ENV = "UNSLOTH_QWEN_IMAGE_ROCM_QUERY_CHUNKING"
QUERY_CHUNK_SIZE = 512
MIN_QUERY_TOKENS = 8192
# gfx1151 and gfx1201 are measured; gfx1200 runs gfx1201's AOTriton kernels and tuning.
CHUNKED_ARCHES = frozenset({"gfx1151", "gfx1200", "gfx1201"})


def _device_supported(target: Any) -> bool:
    if getattr(target, "backend", None) != "rocm" or getattr(target, "device", None) != "cuda":
        return False
    setting = (os.environ.get(QUERY_CHUNK_ENV) or "auto").strip().lower()
    if setting in ("0", "false", "off", "no"):
        return False
    import torch

    if not getattr(torch.version, "hip", None):
        return False
    if setting in ("1", "true", "on", "yes"):
        return True
    ordinal = getattr(target, "ordinal", None)
    props = torch.cuda.get_device_properties(
        torch.cuda.current_device() if ordinal is None else ordinal
    )
    from utils.hardware.hardware import _props_gfx_arch

    return _props_gfx_arch(props) in CHUNKED_ARCHES


def _native_backend(processor: Any) -> bool:
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    backend = processor._attention_backend
    if backend is None:
        backend, _ = _AttentionBackendRegistry.get_active_backend()
    return backend == "native"


def _can_chunk(processor, attn, hidden_states, attention_mask, segments, key_valid) -> bool:
    import torch
    return not (
        segments is not None
        or attention_mask is not None
        or key_valid is not None
        or processor._parallel_config is not None
        or hidden_states.device.type != "cuda"
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] < MIN_QUERY_TOKENS
        or attn.heads != 32
        or attn.inner_dim != 4096
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not _native_backend(processor)
    )


def _chunk_attention(query: Any, key: Any, value: Any, dispatch: Any, **kwargs: Any) -> Any:
    import torch
    return torch.cat(
        [
            dispatch(query[:, start : start + QUERY_CHUNK_SIZE], key, value, **kwargs)
            for start in range(0, query.shape[1], QUERY_CHUNK_SIZE)
        ],
        dim = 1,
    )


@functools.lru_cache(maxsize = 1)
def _processor_class() -> type:
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor,
        _qwenimage21_prepare_qkv,
        dispatch_attention_fn,
    )

    class ChunkedQwenImage21AttnProcessor(QwenImage21AttnProcessor):
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
            if not _can_chunk(self, attn, hidden_states, attention_mask, segments, key_valid):
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
            output = _chunk_attention(
                query,
                key,
                value,
                dispatch_attention_fn,
                attn_mask = None,
                dropout_p = 0.0,
                backend = self._attention_backend,
                parallel_config = self._parallel_config,
            )
            output = output[:, :seq_len_q].flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](output))

    return ChunkedQwenImage21AttnProcessor


def install(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> bool:
    """Replace only this pipeline's stock Qwen-Image-2.1 processors, before placement."""
    if not _device_supported(target):
        return False
    transformer = getattr(pipe, "transformer", None)
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        return False
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

    cls = _processor_class()
    installed = False
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if type(processor) is cls:
            installed = True
        elif type(processor) is QwenImage21AttnProcessor:
            replacement = cls()
            replacement.__dict__.update(processor.__dict__)
            module.set_processor(replacement)
            installed = True
    if installed and logger is not None:
        logger.info(
            "diffusion.qwenimage21: large ROCm decode queries are eligible for 512-row chunks"
        )
    return installed
