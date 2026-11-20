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
import threading
from typing import Any

WIDEST_FUSED_HEAD_DIM = 256
_GUARD_ATTR = "_unsloth_windows_rocm_wide_head_guard"

logger = logging.getLogger(__name__)

# Backend flags are process-wide: the first wide-head call in flight saves them and pins math, the last restores them,
# so an overlapping call (another thread) never sees the fused kernels back early nor saves the math-only state.
_HELD = [0, None]
_LOCK = threading.Lock()


def _math_on() -> None:
    from torch.nn.attention import SDPBackend, _cur_sdpa_kernel_backends, _sdpa_kernel
    with _LOCK:
        if _HELD[0] == 0:
            _HELD[1] = _cur_sdpa_kernel_backends()
        # Re-pinned on every entry: another thread's sdpa_kernel may have changed the flags since the first.
        _sdpa_kernel([SDPBackend.MATH])
        _HELD[0] += 1


def _math_off() -> None:
    from torch.nn.attention import _sdpa_kernel
    with _LOCK:
        _HELD[0] -= 1
        if _HELD[0] == 0:
            _sdpa_kernel(_HELD[1])
            _HELD[1] = None


def _wide_cuda_head(query: Any) -> bool:
    shape = getattr(query, "shape", None) or ()
    return (
        len(shape) > 0
        and int(shape[-1]) > WIDEST_FUSED_HEAD_DIM
        and getattr(getattr(query, "device", None), "type", None) == "cuda"
    )


_ORIGINALS: dict = {}  # str token -> wrapped SDPA (a graph constant, unlike a function)


def _math_only(token: str, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    """allow_in_graph: Dynamo can't enter the lock; restore run-time, not trace-time, backends."""
    _math_on()
    try:
        return _ORIGINALS[token](query, key, value, *args, **kwargs)
    finally:
        _math_off()


def guard_sdpa(sdpa: Any) -> Any:
    """``sdpa`` with every call whose head is wider than 256 pinned to the math backend."""
    if getattr(sdpa, _GUARD_ATTR, False):
        return sdpa
    try:
        import torch._dynamo
        torch._dynamo.allow_in_graph(_math_only)
    except Exception:  # noqa: BLE001 -- eager calls still take the math backend
        pass
    token = f"sdpa_{len(_ORIGINALS)}"  # a str: Dynamo may make an int symbolic
    _ORIGINALS[token] = sdpa

    @functools.wraps(sdpa)
    def guarded(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if _wide_cuda_head(query):
            return _math_only(token, query, key, value, *args, **kwargs)
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
