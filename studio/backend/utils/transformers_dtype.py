# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Version-safe fp-dtype kwarg for transformers/sentence-transformers loads.

transformers renamed the ``torch_dtype`` kwarg to ``dtype`` in the 5.x line, and
emits ``torch_dtype is deprecated! Use dtype instead!`` when the old name is
passed. But our floor (``transformers>=4.51.3``) predates ``dtype`` and only
accepts ``torch_dtype``, so a bare rename would ``TypeError`` on the floor. Pick
the name the installed version accepts instead.

Mirrors ``unsloth_zoo.hf_utils.HAS_TORCH_DTYPE`` rather than importing it: this
runs in the main backend process (the RAG embedder warms here at startup), which
does not import unsloth, and ``unsloth_zoo`` refuses to import until ``unsloth``
has run first.
"""

from functools import lru_cache


@lru_cache(maxsize = 1)
def _has_torch_dtype_kwarg() -> bool:
    """True if the installed transformers still expects the legacy ``torch_dtype``
    name (i.e. predates the ``dtype`` rename). False when ``dtype`` is the accepted
    name, or when transformers is missing/broken (prefer the modern name)."""
    try:
        try:
            from transformers import PreTrainedConfig as _Config
        except ImportError:
            from transformers import PretrainedConfig as _Config
        return "torch_dtype" in (_Config.__doc__ or "")
    except Exception:
        return False


def dtype_kwargs(value) -> dict:
    """``{"torch_dtype": value}`` on old transformers, ``{"dtype": value}`` on new.

    Splat into a load call (``pipeline(..., **dtype_kwargs(torch.float16))``) or use
    directly as ``model_kwargs`` (``model_kwargs = dtype_kwargs("float16")``).
    """
    return {"torch_dtype" if _has_torch_dtype_kwarg() else "dtype": value}
