# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bias add for the NVFP4 layer's M x N output, on the EAGER path only.

``mm_fp4`` has no bias epilogue in FlashInfer 0.6.6 -- no ``bias`` argument, no beta accumulate --
so the add is a separate pass over the output. Profiling the assembled layer put that pass at 48
percent of layer time at M=16384/N=12288, larger than the GEMM itself (0.4578 ms against 0.3409),
and that looked like an unavoidable memory-bound cost. It was not:

    output 16384x12288 bf16, 402.7 MB, so an in-place add moves 805 MB

    torch  out.add_(bias)   0.4560 ms   1.77 TB/s
    torch  out.copy_(other) 0.1282 ms   6.28 TB/s   <- same traffic, same dtype
    this kernel             0.1210 ms   6.65 TB/s

A plain copy moves the identical bytes 3.6x faster, so the eager broadcast add was never near the
memory bound; it is the kernel, not the traffic. The output is contiguous, so the 2-D add is
written here as a 1-D pass with the column index recovered as ``offset % n_cols``; a 2-D launch
over a short, wide matrix leaves most of the grid idle, which is the likely reason the eager kernel
sits at 1.77 TB/s on hardware that copies at 6.28.

**Bit-identical to ``Tensor.add_``**, verified on 16384x12288, 4096x3072, 1024x12288, 333x4096,
1x512, 9304x3072 and 7x1024: ``torch.equal`` True, max abs difference 0.0. It accumulates in fp32
and rounds once on store, which is what PyTorch's own bf16 add does. That matters because the
accuracy gates compare renders against stored references, so an arithmetic change here would
silently invalidate every number behind them.

**Why this and not the fused CUTLASS epilogue.** A bias epilogue built into the FP4 GEMM was
written and measured (``outputs/nvfp4_fused_bias_epilogue.json``) and is DROPPED. It folds the bias
into the GEMM for free -- 0.0643 ms against 0.0628 for the GEMM alone -- but it is not bit-identical
to the unfused path: max abs difference 2.0 to 4.0 against outputs of order 10, because the
epilogue adds the bias to the fp32 accumulator before the single rounding while the unfused path
rounds the GEMM to bf16 first and then adds. That breaks the zero-max-abs identity the precision
contract and the CUDA graph tests are written against. It is also sm_100a only and needs a 104 s
nvcc build at runtime, so it would ship an arch-specific JIT into a path whose whole point is that
it has none. Reopen only when FlashInfer exposes ``bias=`` on ``mm_fp4``, or when it ships the GDC
fix and the epilogue can be validated against a rounding-equivalent reference. The CUTLASS sources
stay in ``scripts/`` as research artifacts.

**Deliberately not used under torch.compile.** Inductor already fuses the bias into the neighbouring
GELU or residual: differencing a ``bias=True`` module against ``bias=False`` inside a compiled layer
gives 0.0156 ms at M=4096 against eager ``add_`` at 0.1138, so a custom op here would PREVENT the
fusion that already removed 3.7x to 7.3x of it. This kernel only brings eager level with what
compilation was doing anyway, which is why ``fused_bias_add_`` falls straight back to ``add_`` under
``torch.compiler.is_compiling()``. Eager is a live configuration, not a warm-up detail: flux under
Studio's default ``dynamic=True`` fails quantized lowering (``CantSplit: 3072*s31 + 3072*s87 not
divisible by s31 + s87``) and every quantized scheme falls back to eager, and the step cache, the
offload and the float-timestep paths refuse capture too.
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

# ``auto`` (on wherever the kernel is eligible), ``0`` (never), ``1`` (force). ``auto`` and ``1``
# coincide today because no shape was found where ``add_`` wins; the knob exists so an operator who
# suspects the kernel can take it out of the path without reinstalling anything.
NVFP4_FAST_BIAS_ENV = "UNSLOTH_NVFP4_FAST_BIAS"

_TRUE_TOKENS = ("1", "true", "yes", "on")
_FALSE_TOKENS = ("0", "false", "no", "off")

# Measured at 4096: 0.1210 ms against 0.1250 at 2048 and 0.1230 at 8192. Flat enough that the
# choice barely matters, which is what bandwidth-bound should look like.
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
    """Whether this exact pair is one the kernel covers.

    Every clause is a case where the flat 1-D indexing above would be wrong rather than slow: a
    strided output, a bias that is not one row, a dtype the fp32-accumulate store does not match.
    """
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
    """``out += bias`` in place, bit-identical to ``out.add_(bias)`` and up to 3.8x faster.

    Falls back to ``add_`` whenever the fast path does not apply: tracing under ``torch.compile``
    (see the module docstring -- inductor's fusion is better than this kernel and an opaque launch
    would block it), the env switch off, no triton, CPU tensors, a non-contiguous output, or a
    dtype or shape the kernel does not cover. The fallback is the previous behaviour, so this is
    safe to call unconditionally.

    The launch sits inside a device guard. A Triton kernel takes its device and stream from the
    CURRENT context, not from the tensors it is handed, so an unguarded launch against an output
    that lives on another card is exactly the fault the FlashInfer guards exist for.
    """
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
