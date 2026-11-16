# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: attention with a head wider than 256 runs on the SDPA math backend.

AMD's Windows torch 2.11.0+rocm7.14.0 on gfx1151 accepts head dim 512 in its flash and efficient SDPA kernels and
computes it wrong: at Gemma 4's global-layer shapes (causal, GQA 8:1 and 16:2, bf16, 128 to 8192 tokens) the forward
is off fp64 math by relative error 1 to 1e20 or non-finite, and so are dq / dk / dv; math is right (0.003). Linux ROCm
refuses those kernels at head 512 and already runs math there, so this makes Windows do what Linux does. Heads up to
256 keep their fused kernels (measured correct). A no-op off Windows ROCm."""

from __future__ import annotations

import functools
import logging
from typing import Any

WIDEST_FUSED_HEAD_DIM = 256
_GUARD_ATTR = "_unsloth_windows_rocm_wide_head_guard"

logger = logging.getLogger(__name__)


def _wide_cuda_head(query: Any) -> bool:
    shape = getattr(query, "shape", None) or ()
    return (
        len(shape) > 0
        and int(shape[-1]) > WIDEST_FUSED_HEAD_DIM
        and getattr(getattr(query, "device", None), "type", None) == "cuda"
    )


def guard_sdpa(sdpa: Any) -> Any:
    """``sdpa`` with every call whose head is wider than 256 pinned to the math backend."""
    if getattr(sdpa, _GUARD_ATTR, False):
        return sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel

    @functools.wraps(sdpa)
    def guarded(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if _wide_cuda_head(query):
            with sdpa_kernel([SDPBackend.MATH]):
                return sdpa(query, key, value, *args, **kwargs)
        return sdpa(query, key, value, *args, **kwargs)

    setattr(guarded, _GUARD_ATTR, True)
    return guarded


def install_windows_rocm_wide_head_sdpa_guard(log: Any = None) -> bool:
    """Wrap ``torch.nn.functional.scaled_dot_product_attention`` on Windows ROCm; idempotent; never raises.
    transformers' sdpa attention looks the function up per call, so models loaded later are covered."""
    try:
        from core._torchao_stub import _is_windows_rocm

        if not _is_windows_rocm():
            return False
        import torch.nn.functional as F

        F.scaled_dot_product_attention = guard_sdpa(F.scaled_dot_product_attention)
        (log or logger).info(
            "Windows ROCm: attention heads wider than %d run on the SDPA math backend",
            WIDEST_FUSED_HEAD_DIM,
        )
        return True
    except Exception:  # noqa: BLE001 -- the stock kernels stay on rather than failing the worker
        logger.debug("Windows ROCm wide-head SDPA guard not installed", exc_info = True)
        return False
