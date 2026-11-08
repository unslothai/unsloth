# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 prompt encoding without the text encoder's K/V cache (UNSLOTH_QWEN21_TE_NO_KV=0 disables): the
pipeline's one hidden-state forward never passes ``use_cache``, so the config default built a cache nobody reads."""

from __future__ import annotations

import functools
import os
from typing import Any

_ENV = "UNSLOTH_QWEN21_TE_NO_KV"
_MARK = "_unsloth_te_no_kv"


def _enabled() -> bool:
    return (os.environ.get(_ENV) or "1").strip().lower() not in ("0", "false", "off", "no")


def _no_cache(module: Any, args: tuple, kwargs: dict) -> tuple:
    kwargs["use_cache"] = False
    return args, kwargs


def skip_text_encoder_kv_cache(pipe: Any, logger: Any = None) -> bool:
    """Wrap ``pipe._get_qwen_prompt_embeds`` so its encoder forward runs with ``use_cache=False``. Idempotent."""
    if not _enabled():
        return False
    encoder = getattr(pipe, "text_encoder", None)
    original = getattr(pipe, "_get_qwen_prompt_embeds", None)
    if (
        encoder is None
        or not callable(original)
        or not type(encoder).__name__.startswith("Qwen3VL")
    ):
        return False
    if getattr(original, _MARK, False):
        return True
    if not hasattr(encoder, "register_forward_pre_hook"):
        return False

    @functools.wraps(original)
    def _get_qwen_prompt_embeds(*args: Any, **kwargs: Any) -> Any:
        # read at call time: a later load step may swap in a quantised copy
        current = getattr(pipe, "text_encoder", None)
        if not hasattr(current, "register_forward_pre_hook"):
            return original(*args, **kwargs)
        handle = current.register_forward_pre_hook(_no_cache, with_kwargs = True)
        try:
            return original(*args, **kwargs)
        finally:
            handle.remove()

    setattr(_get_qwen_prompt_embeds, _MARK, True)
    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    if logger is not None:
        logger.info(
            "diffusion.qwen_image_21: prompt encoding skips the text encoder's unused K/V cache"
        )
    return True
