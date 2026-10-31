# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Picks the fp dtype kwarg the installed transformers accepts: torch_dtype before 4.56.0 (the
4.51.3 floor needs it), dtype after. Avoids unsloth_zoo's HAS_TORCH_DTYPE, which breaks under
python -OO and whose import runs a torch-importing __init__."""

from functools import lru_cache


@lru_cache(maxsize = 1)
def _has_torch_dtype_kwarg() -> bool:
    """True if the installed transformers still expects the legacy ``torch_dtype``
    name (i.e. predates the ``dtype`` rename). False when ``dtype`` is the accepted
    name, or when transformers is missing/broken (prefer the modern name)."""
    try:
        import transformers
        from packaging.version import Version

        # Compare release tuples: 4.56.0.dev0/rc1 sort below 4.56.0 but already accept ``dtype``.
        return Version(transformers.__version__).release < (4, 56, 0)
    except Exception:
        return False


def dtype_kwargs(value) -> dict:
    """``{"torch_dtype": value}`` on old transformers, ``{"dtype": value}`` on new.

    Splat into a load call (``pipeline(..., **dtype_kwargs(torch.float16))``) or use
    directly as ``model_kwargs`` (``model_kwargs = dtype_kwargs("float16")``).
    """
    return {"torch_dtype" if _has_torch_dtype_kwarg() else "dtype": value}
