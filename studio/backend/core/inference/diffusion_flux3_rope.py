# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bit-identical one-kernel RoPE for FLUX.3's eager attention; off via UNSLOTH_DIFFUSION_FLUX3_FUSED_ROPE=0.

Reuses the FLUX.2 kernel (fp32 products and sum, no FMA contraction) on the fp32 Q / K norm outputs. Compiled blocks
keep stock, so only eager renders change: before a deferred compile, with LoRA, and speed ``off``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .rocm_bf16 import is_rocm_torch

FLUX3_FUSED_ROPE_ENV = "UNSLOTH_DIFFUSION_FLUX3_FUSED_ROPE"
_MODULE = "diffusers.models.transformers.transformer_flux3"
_ATTR = "apply_rotary_emb"
_LOCK = threading.Lock()
_STOCK: dict = {}
# Survives uninstall so a forward that looked up the patch just before teardown still reaches stock.
_ORIGINAL: list = [None]
# Set by the first failed launch (JIT / driver / PTX): the rest of the load runs stock, not one failing JIT per call.
_FAILED = [False]
_CALLS = [0]


def disabled() -> bool:
    return (os.environ.get(FLUX3_FUSED_ROPE_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _kernel() -> Any:
    from .diffusion_flux2_rope import _kernel as flux2_kernel
    return flux2_kernel()


def wanted(device: Any = "cuda") -> bool:
    """Any load dtype: the processor rotates fp32 Q / K norm outputs either way."""
    if disabled():
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not str(device).startswith("cuda") or is_rocm_torch(torch):
        return False
    return _kernel() is not None


def _eligible(
    x: Any, freqs_cis: Any, use_real: bool, use_real_unbind_dim: int, sequence_dim: int
) -> bool:
    import torch

    if not use_real or use_real_unbind_dim != -1 or sequence_dim != 1:
        return False
    if not isinstance(freqs_cis, (tuple, list)) or len(freqs_cis) != 2:
        return False
    if not torch.is_tensor(x) or not x.is_cuda or x.dim() != 4:
        return False
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if x.requires_grad and torch.is_grad_enabled():
        return False
    B, S, H, D = x.shape
    if D % 2 or x.stride(3) != 1 or x.numel() == 0:
        return False
    # int32 kernel offsets
    if x.storage_offset() + sum((n - 1) * st for n, st in zip(x.shape, x.stride())) >= 2**31 - 1:
        return False
    cos, sin = freqs_cis
    for f in (cos, sin):
        if not torch.is_tensor(f) or f.dim() != 2 or tuple(f.shape) != (S, D):
            return False
        if f.device != x.device or f.stride(1) != 1:
            return False
        if f.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
    return True


def _log_failed(exc: BaseException) -> None:
    try:
        logging.getLogger(__name__).warning(
            "diffusion.flux3_rope: fused RoPE launch failed (%s: %s); stock RoPE for the rest of this load",
            type(exc).__name__,
            exc,
        )
    except Exception:  # noqa: BLE001
        pass


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
    launch = None if _FAILED[0] else _kernel()
    if launch is not None and _eligible(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim):
        try:
            out = launch(x, freqs_cis[0], freqs_cis[1])
            _CALLS[0] += 1
            return out
        except Exception as exc:  # noqa: BLE001 -- never fail a render over the fast path
            _FAILED[0] = True
            _log_failed(exc)
    return stock(
        x,
        freqs_cis,
        use_real = use_real,
        use_real_unbind_dim = use_real_unbind_dim,
        sequence_dim = sequence_dim,
    )


def install(device: Any = "cuda", logger: Any = None) -> bool:
    """Idempotent; False (nothing patched) when not wanted or diffusers' symbol is not the expected one."""
    if not wanted(device):
        uninstall()
        return False
    import sys

    with _LOCK:
        _FAILED[0] = False
        if _ATTR in _STOCK:
            return True
        # Only a module already loaded (its transformer is in use) is patched.
        mod = sys.modules.get(_MODULE)
        if mod is None:
            return False
        stock = getattr(mod, _ATTR, None)
        if stock is None or getattr(stock, "__name__", "") != "apply_rotary_emb":
            return False
        _STOCK[_ATTR] = stock
        _ORIGINAL[0] = stock
        setattr(mod, _ATTR, _fused_apply_rotary_emb)
    if logger is not None:
        try:
            logger.info("diffusion.flux3_rope: fused RoPE kernel installed")
        except Exception:  # noqa: BLE001
            pass
    return True


def install_for_pipe(
    pipe: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> bool:
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or type(transformer).__module__ != _MODULE:
        uninstall()
        return False
    return install(device, logger)


def uninstall() -> None:
    import sys
    with _LOCK:
        _FAILED[0] = False
        stock = _STOCK.pop(_ATTR, None)
        if stock is None:
            return
        mod = sys.modules.get(_MODULE)
        if mod is not None and getattr(mod, _ATTR, None) is _fused_apply_rotary_emb:
            setattr(mod, _ATTR, stock)


def is_installed() -> bool:
    return _ATTR in _STOCK


def call_count() -> int:
    return _CALLS[0]
