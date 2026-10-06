# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Query-chunked VAE attention on ROCm, where no fused SDPA kernel takes head dim 384 / 512.

AOTriton caps gfx11 at head dim 256, so SDPA falls back to the math kernel and materialises the full fp32 score
matrix (16 GiB for a 2048x2048 FLUX.1 decode). Unmasked query rows are independent, so chunking them is exact.
``UNSLOTH_VAE_ATTN_CHUNKED=0`` disables it; ``UNSLOTH_VAE_ATTN_CHUNK_MB`` sets the per-chunk fp32 score budget.
"""

from __future__ import annotations

import os
from typing import Any, Optional

VAE_ATTN_CHUNK_ENV = "UNSLOTH_VAE_ATTN_CHUNKED"
VAE_ATTN_CHUNK_MB_ENV = "UNSLOTH_VAE_ATTN_CHUNK_MB"
_FALSE = ("0", "false", "no", "off")
# fp32 score bytes per chunk; a full matrix at or under this runs unchunked (FLUX.1 1024x1024 is exactly 1 GiB).
DEFAULT_SCORE_BUDGET_MB = 1024
_PATCHED = "_unsloth_vae_attn_chunked"


def chunking_disabled() -> bool:
    return os.environ.get(VAE_ATTN_CHUNK_ENV, "").strip().lower() in _FALSE


def score_budget_bytes() -> int:
    raw = os.environ.get(VAE_ATTN_CHUNK_MB_ENV, "").strip()
    try:
        mb = int(raw) if raw else DEFAULT_SCORE_BUDGET_MB
    except ValueError:
        mb = DEFAULT_SCORE_BUDGET_MB
    return max(1, mb) * 2**20


def fused_sdpa_available(q: Any, k: Any, v: Any) -> bool:
    """Whether flash, memory-efficient or cuDNN SDPA takes this call; non-CUDA tensors or query errors count as yes."""
    import torch

    if getattr(q, "device", None) is None or q.device.type != "cuda":
        return True
    cb = torch.backends.cuda
    try:
        try:
            params = cb.SDPAParams(q, k, v, None, 0.0, False, False)
        except TypeError:  # torch without the enable_gqa field
            params = cb.SDPAParams(q, k, v, None, 0.0, False)
        for name in (
            "can_use_flash_attention",
            "can_use_efficient_attention",
            "can_use_cudnn_attention",
        ):
            fn = getattr(cb, name, None)
            if fn is not None and fn(params, False):
                return True
    except Exception:  # noqa: BLE001 - unknown: keep the stock kernel
        return True
    return False


def _score_bytes(q: Any, k: Any) -> int:
    rows = 1
    for s in q.shape[:-1]:
        rows *= int(s)
    return rows * int(k.shape[-2]) * 4


def chunked_sdpa(
    sdpa: Any,
    q: Any,
    k: Any,
    v: Any,
    budget: int,
    scale: Optional[float] = None,
) -> Any:
    """``sdpa(q, k, v)`` over query-row chunks whose fp32 score tile stays within ``budget`` bytes."""
    import torch

    length = int(q.shape[-2])
    per_row = max(1, _score_bytes(q, k) // max(1, length))
    rows = max(1, min(length, budget // per_row))
    kw = {} if scale is None else {"scale": scale}
    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype = q.dtype, device = q.device)
    for i in range(0, length, rows):
        out[..., i : i + rows, :] = sdpa(q[..., i : i + rows, :], k, v, **kw)
    return out


def _plain_call(args: tuple, kwargs: dict) -> Optional[tuple]:
    """(q, k, v, scale) for an unmasked, dropout-free, non-causal, non-GQA call; else None."""
    if len(args) > 4:
        return None
    extra = dict(kwargs)
    # diffusers' dispatch_attention_fn passes query= / key= / value= by keyword
    qkv = list(args[:3])
    for name in ("query", "key", "value")[len(qkv) :]:
        if name not in extra:
            return None
        qkv.append(extra.pop(name))
    q, k, v = qkv
    if len(args) == 4:
        extra["attn_mask"] = args[3]
    scale = extra.pop("scale", None)
    if extra.pop("attn_mask", None) is not None:
        return None
    if extra.pop("dropout_p", 0.0) not in (0, 0.0):
        return None
    if extra.pop("is_causal", False):
        return None
    if extra.pop("enable_gqa", False) or extra:
        return None
    if q.ndim < 3 or q.ndim != k.ndim or k.ndim != v.ndim or q.shape[:-2] != k.shape[:-2]:
        return None
    if q.requires_grad or k.requires_grad or v.requires_grad:
        return None
    return q, k, v, scale


def _mode_class() -> Any:
    import torch.nn.functional as F
    from torch.overrides import TorchFunctionMode

    class ChunkedVaeAttentionMode(TorchFunctionMode):
        def __init__(self, budget: int):
            super().__init__()
            self.budget = budget
            self.chunked = 0

        def __torch_function__(
            self,
            func: Any,
            types: Any,
            args: tuple = (),
            kwargs: Any = None,
        ) -> Any:
            kwargs = kwargs or {}
            if func is F.scaled_dot_product_attention:
                call = _plain_call(args, kwargs)
                if call is not None:
                    q, k, v, scale = call
                    if _score_bytes(q, k) > self.budget and not fused_sdpa_available(q, k, v):
                        self.chunked += 1
                        return chunked_sdpa(func, q, k, v, self.budget, scale)
            return func(*args, **kwargs)

    return ChunkedVaeAttentionMode


_MODE_CLS: Any = None


def chunked_attention_mode(budget: Optional[int] = None) -> Any:
    global _MODE_CLS
    if _MODE_CLS is None:
        _MODE_CLS = _mode_class()
    return _MODE_CLS(score_budget_bytes() if budget is None else budget)


def _is_attention_module(module: Any) -> bool:
    name = type(module).__name__
    # diffusers Attention (AutoencoderKL, FLUX.2) and the single-head blocks of Wan / Qwen-Image / HunyuanImage VAEs
    return name == "Attention" or name.endswith(("AttentionBlock", "AttnBlock"))


def _patch(module: Any) -> None:
    import torch

    inner = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        if chunking_disabled() or torch.compiler.is_compiling():
            return inner(*args, **kwargs)
        with chunked_attention_mode():
            return inner(*args, **kwargs)

    forward._unsloth_vae_attn_chunked = True
    forward._unsloth_vae_attn_inner = inner
    module.forward = forward
    setattr(module, _PATCHED, True)


def wants_install(vae: Any, target: Any = None) -> bool:
    if vae is None or chunking_disabled() or not callable(getattr(vae, "modules", None)):
        return False
    if target is not None and getattr(target, "backend", None) != "rocm":
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.version, "hip", None))


def patch_attention_modules(vae: Any) -> int:
    """Patch every attention module of ``vae`` (outermost only, idempotent). Returns the number newly patched."""
    n = 0
    covered: list = []
    for name, module in vae.named_modules():
        if not _is_attention_module(module):
            continue
        if any(name.startswith(prefix + ".") for prefix in covered):
            continue
        covered.append(name)
        if getattr(module, _PATCHED, False):
            continue
        _patch(module)
        n += 1
    return n


def install(
    vae: Any,
    target: Any = None,
    logger: Any = None,
) -> int:
    """Chunk the VAE attention on ROCm, where no fused SDPA kernel takes head dim > 256. Never raises."""
    try:
        if not wants_install(vae, target):
            return 0
        n = patch_attention_modules(vae)
        if n and logger is not None:
            logger.info(
                "diffusion.vae_attn_chunked: %d %s attention modules chunk SDPA calls no fused kernel takes",
                n,
                type(vae).__name__,
            )
        return n
    except Exception as exc:  # noqa: BLE001 - stock attention on any failure
        if logger is not None:
            logger.warning("diffusion.vae_attn_chunked: not installed (%s)", exc)
        return 0
