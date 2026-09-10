# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bias add for the NVFP4 layer's M x N output, on the EAGER path only.

``mm_fp4`` has no bias epilogue in FlashInfer 0.6.6, so the add is a separate pass that must stay
bit-identical to ``Tensor.add_``; the fused CUTLASS FP4 epilogue is WRONG for that reason, adding
into the fp32 accumulator before the single rounding. Under ``torch.compile`` this falls back to
``add_``: an opaque launch would block inductor's own fusion of the bias into the next op.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 - any triton import failure means the torch add, not a crash
    _HAVE_TRITON = False

NVFP4_FAST_BIAS_ENV = "UNSLOTH_NVFP4_FAST_BIAS"

_TRUE_TOKENS = ("1", "true", "yes", "on")
_FALSE_TOKENS = ("0", "false", "no", "off")

_BLOCK = 4096


if _HAVE_TRITON:

    @triton.jit
    def _bias_add_kernel(ptr, bias_ptr, n_elements, n_cols: "tl.constexpr", BLOCK: "tl.constexpr"):
        """``row-major[m, n] += bias[n]``, flattened to a 1-D pass."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(ptr + offsets, mask = mask)
        b = tl.load(bias_ptr + (offsets % n_cols), mask = mask)
        # fp32 accumulate then a single round on store, matching torch's bf16 add exactly.
        tl.store(ptr + offsets, (x.to(tl.float32) + b.to(tl.float32)).to(x.dtype), mask = mask)


def fast_bias_enabled() -> bool:
    """Whether the Triton pass is allowed here at all. ``UNSLOTH_NVFP4_FAST_BIAS=auto|0|1``."""
    raw = os.environ.get(NVFP4_FAST_BIAS_ENV, "").strip().lower()
    if raw in _FALSE_TOKENS:
        return False
    return True


def _eligible(out: Any, bias: Any) -> bool:
    """Whether this pair is covered; each clause is a case the flat 1-D indexing gets WRONG."""
    import torch
    return (
        _HAVE_TRITON
        and out.is_cuda
        and out.is_contiguous()
        and bias.is_contiguous()
        and bias.dim() == 1
        and out.shape[-1] == bias.shape[0]
        and out.dtype == bias.dtype
        and out.dtype in (torch.bfloat16, torch.float16)
    )


def fused_bias_add_(out: Any, bias: Any):
    """``out += bias`` in place, bit-identical to ``out.add_(bias)``, falling back to ``add_``. The
    device guard is load-bearing: Triton takes its device from the CURRENT context."""
    import torch

    if torch.compiler.is_compiling() or not fast_bias_enabled() or not _eligible(out, bias):
        return out.add_(bias)

    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, _BLOCK),)
    with torch.cuda.device(out.device):
        _bias_add_kernel[grid](out, bias, n_elements, out.shape[-1], _BLOCK, num_warps = 8)
    return out
