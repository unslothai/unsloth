# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Picks the fp dtype kwarg transformers accepts: torch_dtype before 4.56.0 (4.51.3 floor needs it),
dtype after; avoids unsloth_zoo HAS_TORCH_DTYPE (breaks under python -OO; imports torch)."""

from functools import lru_cache


@lru_cache(maxsize = 1)
def _has_torch_dtype_kwarg() -> bool:
    """True if transformers wants legacy torch_dtype; False for dtype or if missing/broken."""
    try:
        import transformers
        from packaging.version import Version

        # Compare release tuples: 4.56.0.dev0/rc1 sort below 4.56.0 but already accept ``dtype``.
        return Version(transformers.__version__).release < (4, 56, 0)
    except Exception:
        return False


def dtype_kwargs(value) -> dict:
    return {"torch_dtype" if _has_torch_dtype_kwarg() else "dtype": value}
