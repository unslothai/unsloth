# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Host check for the LONG_INDEXING constexpr: int32 Triton offsets unless a tensor needs int64."""

__all__ = ["long_indexing"]

try:
    from math import sumprod as _sumprod
except ImportError:  # Python < 3.12

    def _sumprod(a, b):
        return sum(x * y for x, y in zip(a, b))


def long_indexing(*tensors, block = 0):
    # numel() is not enough for strided views (transposed Q / K): their offsets reach past it.
    for t in tensors:
        if t.is_contiguous():
            extent = t.numel()
        else:
            stride = t.stride()
            extent = _sumprod(t.shape, stride) - sum(stride) + 1
        if extent + block > 2**31:
            return True
    return False
