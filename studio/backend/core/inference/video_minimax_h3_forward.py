# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 denoiser outer forward, op for op the diffusers 0.42.1 one (installed only over the mirrored hash), with:

* the text stream run once per render inside ``h3_render_scope``, keyed on the conditioning tensor and LoRA scale; an
  inference tensor has no version counter, so an in-place edit inside a render is not seen
  (``UNSLOTH_H3_REFINER_CACHE=0`` disables).
* the packed sequence filled in place with ``index_copy_`` (``UNSLOTH_H3_INPLACE_PACK=0`` disables).

Under a whole-forward ``torch.compile`` the stock forward runs (regional compile keeps this forward eager).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import threading
import types
from typing import Any, Iterator, Optional

REFINER_CACHE_ENV = "UNSLOTH_H3_REFINER_CACHE"
INPLACE_PACK_ENV = "UNSLOTH_H3_INPLACE_PACK"

_CLASS_NAME = "MiniMaxH3Transformer3DModel"
# sha256[:16] of inspect.getsource(MiniMaxH3Transformer3DModel.forward) this module mirrors (diffusers 0.42.1).
_KNOWN_FORWARD_SHA = frozenset({"ae200f2868ad388f"})

_PATCHED_ATTR = "_unsloth_h3_forward"
_SCOPE_ATTR = "_unsloth_h3_text_cache"
STATS_ATTR = "_unsloth_h3_forward_stats"
DENOISER_COMPONENTS = ("transformer", "transformer_ref")

_LOCK = threading.Lock()
_SHA_CACHE: dict = {}


def _switch(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() not in ("0", "off", "false", "no")


def refiner_cache_enabled() -> bool:
    return _switch(REFINER_CACHE_ENV)


def inplace_pack_enabled() -> bool:
    return _switch(INPLACE_PACK_ENV)


def _forward_sha(cls: type) -> Optional[str]:
    if cls in _SHA_CACHE:
        return _SHA_CACHE[cls]
    try:
        sha = hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        sha = None
    _SHA_CACHE[cls] = sha
    return sha


def _compiling() -> bool:
    import torch
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(is_compiling is not None and is_compiling())


def _stats(module: Any) -> dict:
    stats = module.__dict__.get(STATS_ATTR)
    if stats is None:
        stats = {"text_hits": 0, "text_misses": 0, "inplace_packs": 0, "stock": 0}
        module.__dict__[STATS_ATTR] = stats
    return stats


class _TextCache:
    """One render's text-stream rows: the conditioning tensor they came from (held, so its storage cannot be
    reused by another tensor while the entry lives) and everything else that decides them."""

    __slots__ = ("source", "key", "value")

    def __init__(self) -> None:
        self.source = None
        self.key = None
        self.value = None


def _source_key(t: Any, dtype: Any) -> tuple:
    version = None if t.is_inference() else t._version  # inference tensors carry no version counter
    return (t.data_ptr(), tuple(t.shape), t.dtype, str(t.device), version, dtype)


def _text_rows(
    self: Any,
    encoder_hidden_states: Any,
    lora_scale: Any = None,
) -> Any:
    from diffusers.models.modeling_utils import get_parameter_dtype

    dtype = get_parameter_dtype(self.context_embedder)
    cache = self.__dict__.get(_SCOPE_ATTR)
    if cache is None or not refiner_cache_enabled():
        text_embeds = self.context_embedder(encoder_hidden_states.to(dtype))
        return self.token_refiner(text_embeds)
    # the LoRA scale apply_lora_scale applied to this call scales the refiner's adapters too
    key = (_source_key(encoder_hidden_states, dtype), lora_scale)
    if cache.value is not None and cache.source is encoder_hidden_states and cache.key == key:
        _stats(self)["text_hits"] += 1
        return cache.value
    text_embeds = self.context_embedder(encoder_hidden_states.to(dtype))
    text_embeds = self.token_refiner(text_embeds)
    cache.source, cache.key, cache.value = encoder_hidden_states, key, text_embeds
    _stats(self)["text_misses"] += 1
    return text_embeds


def _body(
    self: Any,
    hidden_states: Any,
    audio_hidden_states: Any,
    encoder_hidden_states: Any,
    timestep: Any,
    timestep_indices: Any,
    token_tags: Any,
    position_ids: Any,
    video_indices: Any,
    audio_indices: Any,
    text_indices: Any,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,
    _unsloth_lora_scale: Any = None,
) -> Any:
    """``MiniMaxH3Transformer3DModel.forward`` (diffusers 0.42.1) op for op, except the text stream and the pack."""
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3TransformerOutput,
    )

    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(
            f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}."
        )
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(
            "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
            f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
        )

    rotary_emb = self.rope(position_ids)

    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(
        audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in))
    )
    text_embeds = _text_rows(self, encoder_hidden_states, _unsloth_lora_scale)

    hidden_states = text_embeds.new_zeros(
        (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
    )
    if inplace_pack_enabled():
        hidden_states.index_copy_(1, text_indices, text_embeds)
        hidden_states.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        _stats(self)["inplace_packs"] += 1
    else:
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(
            1, video_indices, video_embeds.to(text_embeds.dtype)
        )
        hidden_states = hidden_states.index_copy(
            1, audio_indices, audio_embeds.to(text_embeds.dtype)
        )
    del video_embeds, audio_embeds

    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))

    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)

    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(
        get_parameter_dtype(self.proj_out)
    )
    video_output = self.proj_out(hidden_states).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample = video_output, audio_sample = audio_output)


_SCALED_BODY = None


def _scaled_body() -> Any:
    """``_body`` under the same ``apply_lora_scale`` the stock forward carries (built once, lazily)."""
    global _SCALED_BODY
    if _SCALED_BODY is None:
        from diffusers.utils import apply_lora_scale
        _SCALED_BODY = apply_lora_scale("attention_kwargs")(_body)
    return _SCALED_BODY


def _forward(
    self: Any,
    hidden_states: Any,
    audio_hidden_states: Any,
    encoder_hidden_states: Any,
    timestep: Any,
    timestep_indices: Any,
    token_tags: Any,
    position_ids: Any,
    video_indices: Any,
    audio_indices: Any,
    text_indices: Any,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,
) -> Any:
    import torch

    extra = {}
    if _compiling():
        # cache key (is_inference, data_ptr) and stats counters cause graph breaks / recompiles when traced
        fn = type(self).forward
    elif torch.is_grad_enabled() or not (
        (self.__dict__.get(_SCOPE_ATTR) is not None and refiner_cache_enabled())
        or inplace_pack_enabled()
    ):
        _stats(self)["stock"] += 1
        fn = type(self).forward
    else:
        fn = _scaled_body()
        # apply_lora_scale pops the scale before the body runs; the text cache keys on it
        extra["_unsloth_lora_scale"] = (
            attention_kwargs.get("scale") if isinstance(attention_kwargs, dict) else None
        )
    return fn(
        self,
        hidden_states = hidden_states,
        audio_hidden_states = audio_hidden_states,
        encoder_hidden_states = encoder_hidden_states,
        timestep = timestep,
        timestep_indices = timestep_indices,
        token_tags = token_tags,
        position_ids = position_ids,
        video_indices = video_indices,
        audio_indices = audio_indices,
        text_indices = text_indices,
        attention_kwargs = attention_kwargs,
        return_dict = return_dict,
        **extra,
    )


def _hooked(module: Any) -> bool:
    return getattr(module, "_hf_hook", None) is not None or bool(
        getattr(getattr(module, "_diffusers_hook", None), "hooks", None)
    )


def install_h3_forward(transformer: Any, logger: Any = None) -> bool:
    """Bind the mirrored forward before any offload hook wraps ``forward``; idempotent, never raises, False = stock."""
    try:
        if transformer is None or type(transformer).__name__ != _CLASS_NAME:
            return False
        if transformer.__dict__.get(_PATCHED_ATTR):
            return True
        if not (refiner_cache_enabled() or inplace_pack_enabled()):
            return False
        with _LOCK:
            if _forward_sha(type(transformer)) not in _KNOWN_FORWARD_SHA:
                if logger is not None:
                    logger.info("video.h3_forward: unknown denoiser forward source, keeping stock")
                return False
            if "forward" in transformer.__dict__ or _hooked(transformer):
                if logger is not None:
                    logger.info("video.h3_forward: denoiser forward already wrapped, keeping stock")
                return False
            transformer.forward = types.MethodType(_forward, transformer)
            transformer.__dict__[_PATCHED_ATTR] = True
            _stats(transformer)
    except Exception as exc:  # noqa: BLE001 -- an optimisation, never fail a load
        if logger is not None:
            logger.warning("video.h3_forward: install skipped: %s", exc)
        return False
    if logger is not None:
        logger.info(
            "video.h3_forward: text stream once per render (%s), packed sequence in place (%s)",
            "on" if refiner_cache_enabled() else "off",
            "on" if inplace_pack_enabled() else "off",
        )
    return True


def install_h3_forward_on_pipe(pipe: Any, logger: Any = None) -> int:
    return sum(
        1
        for name in DENOISER_COMPONENTS
        if install_h3_forward(getattr(pipe, name, None), logger = logger)
    )


@contextlib.contextmanager
def h3_render_scope(pipe: Any, logger: Any = None) -> Iterator[None]:
    """One render: the text-stream cache exists only inside, and is dropped on exit or error."""
    modules = [
        m
        for m in (getattr(pipe, name, None) for name in DENOISER_COMPONENTS)
        if m is not None and m.__dict__.get(_PATCHED_ATTR)
    ]
    before = []
    for m in modules:
        m.__dict__[_SCOPE_ATTR] = _TextCache()
        before.append(dict(_stats(m)))
    try:
        yield
    finally:
        for m, start in zip(modules, before):
            m.__dict__.pop(_SCOPE_ATTR, None)
            if logger is not None:
                now = _stats(m)
                delta = {k: now[k] - start.get(k, 0) for k in now}
                logger.info(
                    "video.h3_forward: render ran the text stream %d time(s), reused it %d time(s), "
                    "%d in-place packs, %d stock calls",
                    delta["text_misses"],
                    delta["text_hits"],
                    delta["inplace_packs"],
                    delta["stock"],
                )
