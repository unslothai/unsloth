# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 prompt encoding without the text encoder's K/V cache (UNSLOTH_QWEN21_TE_NO_KV=0 disables): the
pipeline's one hidden-state forward never passes ``use_cache``, so the config default built a cache nobody reads.
It also keeps only the one hidden state it reads (``hidden_states[-1]``) instead of every decoder layer's output
(diffusion_te_select; UNSLOTH_DIFFUSION_TE_SELECT_HIDDEN=0 disables)."""

from __future__ import annotations

import functools
import os
from typing import Any

_ENV = "UNSLOTH_QWEN21_TE_NO_KV"
_MARK = "_unsloth_te_no_kv"
_SELECT_MARK = "_unsloth_te_select_final"
# The pipeline's own norm-neutralizing hook makes its ``hidden_states[-1]`` the final norm's input on every
# transformers version; the selective encode records that input, so it is used only when the source still has it.
_NEUTRALIZED_NORM = ("norm.register_forward_hook", "outputs.hidden_states[-1]")


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
    from .diffusion_prompt_cache import mark_transparent_wrapper

    mark_transparent_wrapper(_get_qwen_prompt_embeds)
    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    if logger is not None:
        logger.info(
            "diffusion.qwen_image_21: prompt encoding skips the text encoder's unused K/V cache"
        )
    return True


def select_final_hidden_state(pipe: Any, logger: Any = None) -> bool:
    """Wrap ``pipe._get_qwen_prompt_embeds`` so its encoder forward records only the final norm's input, the one
    entry the pipeline reads. Idempotent; False when the pipeline is not the expected one."""
    from .diffusion_te_select import enabled as select_enabled
    from .diffusion_te_select import run_selected

    if not select_enabled():
        return False
    encoder = getattr(pipe, "text_encoder", None)
    original = getattr(pipe, "_get_qwen_prompt_embeds", None)
    if (
        encoder is None
        or not callable(original)
        or not type(encoder).__name__.startswith("Qwen3VL")
    ):
        return False
    if getattr(original, _SELECT_MARK, False):
        return True
    try:
        import inspect
        source = inspect.getsource(getattr(type(pipe), "_get_qwen_prompt_embeds"))
    except Exception:  # noqa: BLE001
        return False
    if not all(needle in source for needle in _NEUTRALIZED_NORM):
        return False

    @functools.wraps(original)
    def _get_qwen_prompt_embeds(*args: Any, **kwargs: Any) -> Any:
        current = getattr(pipe, "text_encoder", None)
        return run_selected(current, (-1,), lambda: original(*args, **kwargs), last_pre_norm = True)

    setattr(_get_qwen_prompt_embeds, _SELECT_MARK, True)
    setattr(_get_qwen_prompt_embeds, _MARK, getattr(original, _MARK, False))
    from .diffusion_prompt_cache import is_transparent_wrapper, mark_transparent_wrapper

    if is_transparent_wrapper(original):
        mark_transparent_wrapper(_get_qwen_prompt_embeds)
    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    if logger is not None:
        logger.info(
            "diffusion.qwen_image_21: prompt encoding keeps only the final hidden state it reads"
        )
    return True
