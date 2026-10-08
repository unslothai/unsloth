# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 RoPE as one Triton kernel on fp16 GPUs (T4 and other pre-Ampere cards).

Same fp32 arithmetic and rounding order as stock ``apply_rotary_emb`` (FMA contraction off), so the output is
bit-identical. Patches only ``transformer_flux2``'s global for fp16 CUDA loads; any other call falls through to stock.
Kill switch: ``UNSLOTH_DIFFUSION_FLUX2_FUSED_ROPE=0``.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable, Optional

FLUX2_FUSED_ROPE_ENV = "UNSLOTH_DIFFUSION_FLUX2_FUSED_ROPE"
_MODULE = "diffusers.models.transformers.transformer_flux2"
_ATTR = "apply_rotary_emb"
_LOCK = threading.Lock()
_STOCK: dict = {}
# Survives uninstall so a forward that looked up the patch just before teardown still reaches stock.
_ORIGINAL: list = [None]


def disabled() -> bool:
    return (os.environ.get(FLUX2_FUSED_ROPE_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def wanted(dtype: Any, device: Any = "cuda") -> bool:
    if disabled():
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if dtype is not torch.float16 or not str(device).startswith("cuda"):
        return False
    if getattr(torch.version, "hip", None):
        return False
    return _kernel() is not None


@functools.lru_cache(maxsize = 1)
def _kernel() -> Optional[Callable]:
    """None when Triton is unavailable."""
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        return None

    @triton.jit
    def _rope_fwd(
        X,
        COS,
        SIN,
        OUT,
        S,
        H,
        sxb,
        sxs,
        sxh,
        scs,
        sss,
        sob,
        sos,
        soh,
        HALF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // S
        s = row - b * S
        idx = tl.arange(0, BLOCK)
        mask = idx < H * HALF
        h = idx // HALF
        i = idx - h * HALF
        xo = X + b * sxb + s * sxs + h * sxh + 2 * i
        xe = tl.load(xo, mask = mask, other = 0.0).to(tl.float32)
        xd = tl.load(xo + 1, mask = mask, other = 0.0).to(tl.float32)
        ce = tl.load(COS + s * scs + 2 * i, mask = mask, other = 0.0).to(tl.float32)
        cd = tl.load(COS + s * scs + 2 * i + 1, mask = mask, other = 0.0).to(tl.float32)
        se = tl.load(SIN + s * sss + 2 * i, mask = mask, other = 0.0).to(tl.float32)
        sd = tl.load(SIN + s * sss + 2 * i + 1, mask = mask, other = 0.0).to(tl.float32)
        # stock: x.float() * cos + stack([-x_odd, x_even]).float() * sin, each product rounded on its own
        oe = xe * ce + (-xd) * se
        od = xd * cd + xe * sd
        oo = OUT + b * sob + s * sos + h * soh + 2 * i
        tl.store(oo, oe.to(OUT.dtype.element_ty), mask = mask)
        tl.store(oo + 1, od.to(OUT.dtype.element_ty), mask = mask)

    def launch(x: Any, cos: Any, sin: Any) -> Any:
        import torch

        B, S, H, D = x.shape
        # Same layout stock returns: a dense input keeps its strides, anything else comes back contiguous.
        out = torch.empty_like(x)
        if out.stride(3) != 1:
            out = torch.empty((B, S, H, D), dtype = x.dtype, device = x.device)
        half = D // 2
        block = triton.next_power_of_2(H * half)
        with torch.cuda.device(x.device):
            _rope_fwd[(B * S,)](
                x,
                cos,
                sin,
                out,
                S,
                H,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                cos.stride(0),
                sin.stride(0),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                HALF = half,
                BLOCK = block,
                num_warps = 4 if block <= 2048 else 8,
                enable_fp_fusion = False,
            )
        return out

    return launch


def _eligible(
    x: Any, freqs_cis: Any, use_real: bool, use_real_unbind_dim: int, sequence_dim: int
) -> bool:
    import torch

    if not use_real or use_real_unbind_dim != -1 or sequence_dim != 1:
        return False
    if not isinstance(freqs_cis, (tuple, list)) or len(freqs_cis) != 2:
        return False
    if not torch.is_tensor(x) or x.dtype is not torch.float16 or not x.is_cuda or x.dim() != 4:
        return False
    if x.requires_grad and torch.is_grad_enabled():
        return False
    B, S, H, D = x.shape
    if D % 2 or x.stride(3) != 1 or x.numel() == 0:
        return False
    # Kernel offsets are int32.
    if x.storage_offset() + sum((n - 1) * st for n, st in zip(x.shape, x.stride())) >= 2**31 - 1:
        return False
    cos, sin = freqs_cis
    for f in (cos, sin):
        if not torch.is_tensor(f) or f.dim() != 2 or tuple(f.shape) != (S, D):
            return False
        if f.device != x.device or f.stride(1) != 1:
            return False
        if f.dtype not in (torch.float32, torch.float16):
            return False
    return True


def _fused_apply_rotary_emb(
    x,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
):
    import torch

    stock = _STOCK.get(_ATTR) or _ORIGINAL[0]
    # Checked before _kernel(): Dynamo graph-breaks on Triton's import path, so a compiled block stays whole on stock.
    if torch.compiler.is_compiling():
        return stock(
            x,
            freqs_cis,
            use_real = use_real,
            use_real_unbind_dim = use_real_unbind_dim,
            sequence_dim = sequence_dim,
        )
    launch = _kernel()
    if launch is not None and _eligible(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim):
        try:
            return launch(x, freqs_cis[0], freqs_cis[1])
        except Exception:  # noqa: BLE001 -- never fail a render over the fast path
            pass
    return stock(
        x,
        freqs_cis,
        use_real = use_real,
        use_real_unbind_dim = use_real_unbind_dim,
        sequence_dim = sequence_dim,
    )


def install(
    dtype: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> bool:
    """Idempotent; False (nothing patched) when not wanted or diffusers' symbol is not the expected one."""
    if not wanted(dtype, device):
        uninstall()
        return False
    import importlib

    with _LOCK:
        if _ATTR in _STOCK:
            return True
        try:
            mod = importlib.import_module(_MODULE)
        except Exception:  # noqa: BLE001
            return False
        stock = getattr(mod, _ATTR, None)
        if stock is None or getattr(stock, "__name__", "") != "apply_rotary_emb":
            return False
        _STOCK[_ATTR] = stock
        _ORIGINAL[0] = stock
        setattr(mod, _ATTR, _fused_apply_rotary_emb)
    if logger is not None:
        try:
            logger.info("diffusion.flux2_rope: fused fp16 RoPE kernel installed")
        except Exception:  # noqa: BLE001
            pass
    return True


def install_for_pipe(
    pipe: Any,
    dtype: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> bool:
    """Install for an fp16 FLUX.2 denoiser, otherwise restore stock."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or type(transformer).__module__ != _MODULE:
        uninstall()
        return False
    return install(dtype, device, logger)


def uninstall() -> None:
    import sys
    with _LOCK:
        stock = _STOCK.pop(_ATTR, None)
        if stock is None:
            return
        mod = sys.modules.get(_MODULE)
        if mod is not None and getattr(mod, _ATTR, None) is _fused_apply_rotary_emb:
            setattr(mod, _ATTR, stock)


def is_installed() -> bool:
    return _ATTR in _STOCK
