# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Z-Image and Krea 2 prompt encodes without the text encoder's K/V cache, keeping only the hidden states they read.

Both pipelines run one ``output_hidden_states=True`` forward and never pass ``use_cache``, so the config default
builds a cache nobody reads, and every decoder layer's output is kept although Z-Image reads ``hidden_states[-2]``
and Krea 2 its ``text_encoder_select_layers``. The wrapper forces ``use_cache=False`` for that one forward
(UNSLOTH_DIFFUSION_TE_NO_KV=0 disables) and records only the read entries (diffusion_te_select;
UNSLOTH_DIFFUSION_TE_SELECT_HIDDEN=0 disables). The embeddings are bit-identical; a pipeline whose source no
longer has the expected call is left alone.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_TE_NO_KV"
_MARK = "_unsloth_te_lean"

# (method, source needles, indices read off the pipe at call time)
_TARGETS: tuple[tuple[str, tuple[str, ...], Callable[[Any], Any]], ...] = (
    (
        "_encode_prompt",
        ("output_hidden_states=True", ").hidden_states[-2]"),
        lambda pipe: (-2,),
    ),
    (
        "get_text_hidden_states",
        (
            "output_hidden_states=True",
            "outputs.hidden_states[i] for i in self.text_encoder_select_layers",
        ),
        lambda pipe: tuple(getattr(pipe, "text_encoder_select_layers", None) or ()),
    ),
)


def _enabled() -> bool:
    return (os.environ.get(_ENV) or "1").strip().lower() not in ("0", "false", "off", "no")


def _no_cache(module: Any, args: tuple, kwargs: dict) -> tuple:
    kwargs["use_cache"] = False
    return args, kwargs


def _target(pipe: Any) -> Optional[tuple[str, Callable[[Any], Any]]]:
    for name, needles, indices in _TARGETS:
        method = inspect.getattr_static(type(pipe), name, None)
        if not callable(method):
            continue
        try:
            source = inspect.getsource(method)
        except Exception:  # noqa: BLE001
            continue
        if all(needle in source for needle in needles):
            return name, indices
    return None


def lean_text_encode(pipe: Any, logger: Any = None) -> bool:
    """Wrap the Z-Image / Krea 2 prompt helper so its encoder forward builds no K/V cache and keeps only the hidden
    states the helper reads. Idempotent; False when the pipeline is not one of these or the kill switch is set."""
    if not _enabled() or pipe is None:
        return False
    found = _target(pipe)
    if found is None:
        return False
    name, indices_of = found
    original = getattr(pipe, name, None)
    if not callable(original) or getattr(pipe, "text_encoder", None) is None:
        return False
    if getattr(original, _MARK, False):
        return True

    from .diffusion_prompt_cache import is_transparent_wrapper, mark_transparent_wrapper
    from .diffusion_te_select import run_selected

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # read at call time: a later load step may swap in a quantised copy
        current = getattr(pipe, "text_encoder", None)
        if not hasattr(current, "register_forward_pre_hook"):
            return original(*args, **kwargs)
        handle = current.register_forward_pre_hook(_no_cache, with_kwargs = True)
        try:
            indices = indices_of(pipe)
            if not indices:
                return original(*args, **kwargs)
            return run_selected(current, indices, lambda: original(*args, **kwargs))
        finally:
            handle.remove()

    setattr(wrapper, _MARK, True)
    stock = inspect.getattr_static(type(pipe), name, None)
    if is_transparent_wrapper(original) or (
        getattr(original, "__self__", None) is pipe and getattr(original, "__func__", None) is stock
    ):
        mark_transparent_wrapper(wrapper)
    setattr(pipe, name, wrapper)
    if logger is not None:
        logger.info(
            "diffusion.te_lean: %s.%s skips the text encoder's unused K/V cache and keeps only the hidden "
            "states it reads",
            type(pipe).__name__,
            name,
        )
    return True


__all__ = ["lean_text_encode"]
