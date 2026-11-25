# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit prompt encoding with ``mm_token_type_ids`` (UNSLOTH_QWEN_VL_MM_TOKEN_TYPES=0 disables).

transformers 5.x Qwen2.5-VL builds 3D M-RoPE positions only from ``mm_token_type_ids``, which the diffusers
edit pipelines drop (diffusers #14189); 4.57 takes no such argument, so engage only where forward accepts it."""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any

_ENV = "UNSLOTH_QWEN_VL_MM_TOKEN_TYPES"
_MARK = "_unsloth_qwen_vl_mm_types"


def _enabled() -> bool:
    return (os.environ.get(_ENV) or "1").strip().lower() not in ("0", "false", "off", "no")


def _accepts_mm_token_type_ids(encoder: Any) -> bool:
    try:
        return "mm_token_type_ids" in inspect.signature(encoder.forward).parameters
    except (TypeError, ValueError):
        return False


def mm_token_type_ids(input_ids: Any, config: Any) -> Any:
    """0 for text, 1 for image tokens, 2 for video tokens: the processor's ``create_mm_token_type_ids``."""
    types = input_ids.new_zeros(input_ids.shape)
    image_id = getattr(config, "image_token_id", None)
    video_id = getattr(config, "video_token_id", None)
    if image_id is not None:
        types = types.masked_fill(input_ids == int(image_id), 1)
    if video_id is not None:
        types = types.masked_fill(input_ids == int(video_id), 2)
    return types


def _add_mm_token_type_ids(module: Any, args: tuple, kwargs: dict) -> tuple:
    input_ids = kwargs.get("input_ids")
    if (
        kwargs.get("mm_token_type_ids") is None
        and input_ids is not None
        and (kwargs.get("image_grid_thw") is not None or kwargs.get("video_grid_thw") is not None)
    ):
        kwargs["mm_token_type_ids"] = mm_token_type_ids(input_ids, module.config)
    return args, kwargs


def forward_mm_token_type_ids(pipe: Any, logger: Any = None) -> bool:
    """Pass ``mm_token_type_ids`` to the encoder during prompt encode. Idempotent; False when not needed."""
    if not _enabled():
        return False
    encoder = getattr(pipe, "text_encoder", None)
    original = getattr(pipe, "_get_qwen_prompt_embeds", None)
    if (
        encoder is None
        or not callable(original)
        or not type(encoder).__name__.startswith("Qwen2_5_VL")
        or not hasattr(encoder, "register_forward_pre_hook")
    ):
        return False
    if getattr(original, _MARK, False):
        return True
    if not _accepts_mm_token_type_ids(encoder):
        return False

    @functools.wraps(original)
    def _get_qwen_prompt_embeds(*args: Any, **kwargs: Any) -> Any:
        # read at call time: a later load step may swap in a quantised copy
        current = getattr(pipe, "text_encoder", None)
        if not hasattr(current, "register_forward_pre_hook") or not _accepts_mm_token_type_ids(
            current
        ):
            return original(*args, **kwargs)
        handle = current.register_forward_pre_hook(_add_mm_token_type_ids, with_kwargs = True)
        try:
            return original(*args, **kwargs)
        finally:
            handle.remove()

    setattr(_get_qwen_prompt_embeds, _MARK, True)
    from .diffusion_prompt_cache import is_transparent_wrapper, mark_transparent_wrapper

    if is_transparent_wrapper(original) or (
        inspect.ismethod(original) and original.__self__ is pipe
    ):
        mark_transparent_wrapper(_get_qwen_prompt_embeds)
    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    if logger is not None:
        logger.info(
            "diffusion.qwen_image: the Qwen2.5-VL prompt encoder gets mm_token_type_ids for its 3D positions"
        )
    return True
