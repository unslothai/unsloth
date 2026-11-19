# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 cuDNN attention on the processor's strided [B, S, H, D] views, skipping diffusers' contiguous
Q / K / V copies and the output copy into ``to_out``. Only where Studio pins cuDNN (sm86 / sm90 Auto, a cudnn
request), on cards where a probe finds it bitwise equal to the contiguous call; masked, causal, scaled, dropout, GQA
and context-parallel calls keep diffusers' path. UNSLOTH_DIFFUSION_Q21_STRIDED_ATTN=0 disables."""

from __future__ import annotations

import os
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

STRIDED_ATTN_ENV = "UNSLOTH_DIFFUSION_Q21_STRIDED_ATTN"
_CUDNN = "_native_cudnn"
_DITS = frozenset({"QwenImage21Transformer2DModel"})
_PROCESSORS = frozenset({"QwenImage21AttnProcessor"})
# (device index, dtype, heads, head_dim) -> strided cuDNN output == contiguous cuDNN output, bitwise
_PROBED: dict = {}
# plain calls by outcome, eager only
STATS: dict = {}


def enabled() -> bool:
    return (os.environ.get(STRIDED_ATTN_ENV) or "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


class StridedCudnn(str):
    """A processor's cuDNN pin whose plain calls run on the strided views. Equal to ``"_native_cudnn"`` for every
    reader (and diffusers' dispatch, which the other calls still reach); ``cards`` holds the probed CUDA indices,
    ``dtype`` the probed dtype (other dtypes keep diffusers' call)."""

    def __new__(
        cls,
        cards: Any = (),
        dtype: Any = None,
    ) -> "StridedCudnn":
        obj = str.__new__(cls, _CUDNN)
        obj.cards = frozenset(int(c) for c in cards)
        obj.dtype = dtype
        return obj

    @property
    def value(self) -> str:
        return str.__str__(self)

    @property
    def name(self) -> str:
        return "_NATIVE_CUDNN"

    def __reduce__(self) -> tuple:
        return (StridedCudnn, (tuple(sorted(self.cards)), self.dtype))


def _arg(
    args: tuple,
    kwargs: dict,
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    return args[index] if len(args) > index else kwargs.get(name, default)


_KNOWN = frozenset(
    {
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "enable_gqa",
        "attention_kwargs",
        "backend",
        "parallel_config",
    }
)


def _on_card(x: Any, backend: StridedCudnn) -> bool:
    return bool(x.is_cuda) and x.device.index in backend.cards


def plain_qkv(backend: StridedCudnn, args: tuple, kwargs: dict) -> Optional[tuple]:
    """(query, key, value) of a dispatch call the strided cuDNN call reproduces bit for bit, else None."""
    import torch

    if len(args) > 9 or any(name not in _KNOWN for name in kwargs):
        return None
    q, k, v = (_arg(args, kwargs, i, n) for i, n in enumerate(("query", "key", "value")))
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return None
    if (
        _arg(args, kwargs, 3, "attn_mask") is not None
        or _arg(args, kwargs, 4, "dropout_p", 0.0)
        or _arg(args, kwargs, 5, "is_causal", False)
        or _arg(args, kwargs, 6, "scale") is not None
        or _arg(args, kwargs, 7, "enable_gqa", False)
        or _arg(args, kwargs, 8, "attention_kwargs")
        or kwargs.get("parallel_config") is not None
    ):
        return None
    if q.dim() != 4 or k.dim() != 4 or tuple(k.shape) != tuple(v.shape):
        return None
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        return None
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.bfloat16, torch.float16):
        return None
    if backend.dtype is not None and q.dtype != backend.dtype:
        return None
    if q.device != k.device or q.device != v.device or not _on_card(q, backend):
        return None
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None
    return q, k, v


def strided_cudnn_attention(q: Any, k: Any, v: Any) -> Any:
    """diffusers' ``_native_cudnn_attention`` (no mask, default scale) without its ``.contiguous()`` copies: cuDNN
    reads the [B, S, H, D] strides itself and writes its output in the query's layout, so the processor's
    ``flatten(2, 3)`` is a view."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False,
            scale = None,
            enable_gqa = False,
        )
    return out.permute(0, 2, 1, 3)


def maybe_strided(backend: Any, args: tuple, kwargs: dict) -> Optional[Any]:
    """The strided call's output for a plain call of a ``StridedCudnn`` backend, else None (diffusers' call runs)."""
    if not isinstance(backend, StridedCudnn):
        return None
    qkv = plain_qkv(backend, args, kwargs)
    import torch

    if not torch.compiler.is_compiling():
        STATS["strided" if qkv is not None else "stock"] = (
            STATS.get("strided" if qkv is not None else "stock", 0) + 1
        )
    if qkv is None:
        return None
    return strided_cudnn_attention(*qkv)


def strided_matches(device: Any, dtype: Any, heads: int, head_dim: int) -> bool:
    """Cached per (card, dtype, heads, head_dim); False on any error. Two sequence lengths, one of them with a
    longer key (the cached-prefix decode), both at the processor's layouts."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    key = (device.index, str(dtype), int(heads), int(head_dim))
    if key in _PROBED:
        return _PROBED[key]
    ok = False
    try:
        gen = torch.Generator(device = device).manual_seed(0)
        ok = True
        for sq, sk in ((333, 333), (256, 279)):
            # query as the projection's unflattened [B, S, H * D] row, key / value as the cached concatenation
            q = torch.randn((2, sq, heads * head_dim), generator = gen, device = device).to(dtype)
            q = q.unflatten(-1, (heads, head_dim))
            k, v = (
                torch.randn((2, sk, heads, head_dim), generator = gen, device = device).to(dtype)
                for _ in range(2)
            )
            with torch.no_grad():
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    ref = F.scaled_dot_product_attention(
                        *(x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
                    ).permute(0, 2, 1, 3)
                got = strided_cudnn_attention(q, k, v)
            ok = ok and bool(torch.equal(ref, got)) and bool(torch.isfinite(ref.float()).all())
            ok = ok and bool(torch.equal(ref.flatten(2, 3), got.flatten(2, 3)))
    except Exception:  # noqa: BLE001 - an unprobeable card keeps diffusers' contiguous call
        ok = False
    _PROBED[key] = ok
    return ok


def _attention_modules(transformer: Any) -> list:
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return []
    return [
        m
        for m in modules()
        if type(getattr(m, "processor", None)).__name__ in _PROCESSORS and hasattr(m, "heads")
    ]


def _backend_value(backend: Any) -> Optional[str]:
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


def _cards(target: Any, transformer: Any) -> list:
    """The load's card first, then every other card the DiT's weights sit on (a layer split)."""
    import torch

    raw = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "cuda")
    device = torch.device(raw)
    if device.type != "cuda":
        return []
    first = device.index if device.index is not None else torch.cuda.current_device()
    cards = [int(first)]
    try:
        from .diffusion_int8_fused import resident_cuda_cards
        cards += [int(c) for c in (resident_cuda_cards(transformer) or ()) if int(c) not in cards]
    except Exception:  # noqa: BLE001 - the load's card alone
        pass
    return cards


def _head_dim(attn: Any) -> Optional[int]:
    heads = int(getattr(attn, "heads", 0) or 0)
    inner = int(getattr(attn, "inner_dim", 0) or 0)
    if heads <= 0 or inner <= 0 or inner % heads:
        return None
    return inner // heads


def install(
    pipe: Any,
    backend: Any,
    target: Any,
    logger: Any = None,
) -> int:
    """Swap the cuDNN pin of each Qwen-Image-2.1 processor for ``StridedCudnn``. Returns the count; never raises."""
    try:
        return _install(pipe, backend, target, logger)
    except Exception as exc:  # noqa: BLE001 - the stock dispatch stays
        if logger is not None:
            logger.info("diffusion.q21_attn: strided cuDNN attention skipped (%s)", exc)
        return 0


def _install(pipe: Any, backend: Any, target: Any, logger: Any) -> int:
    if not enabled() or _backend_value(backend) != _CUDNN:
        return 0
    import torch

    if is_rocm_torch(torch) or not torch.cuda.is_available():
        return 0
    from .diffusion_attention import _attention_dits

    dits = [d for d in _attention_dits(pipe) if type(d).__name__ in _DITS]
    if not dits:
        return 0
    from .diffusion_int8_attention_dispatch import install_dispatch_route

    dtype = getattr(target, "dtype", None)
    if dtype not in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16  # a quantized DiT computes in bf16
    swapped = 0
    marks: dict = {}
    for dit in dits:
        cards = _cards(target, dit)
        for attn in _attention_modules(dit):
            proc = attn.processor
            if (
                isinstance(proc._attention_backend, StridedCudnn)
                and proc._attention_backend.dtype == dtype
            ):
                swapped += 1
                continue
            if _backend_value(getattr(proc, "_attention_backend", None)) != _CUDNN:
                continue
            if getattr(proc, "_parallel_config", None) is not None:
                continue
            head_dim = _head_dim(attn)
            if head_dim is None:
                continue
            ok = tuple(
                c
                for c in cards
                if strided_matches(torch.device("cuda", c), dtype, int(attn.heads), head_dim)
            )
            if not ok:
                continue
            if not marks and install_dispatch_route() == 0:
                return 0
            mark = marks.get(ok)
            if mark is None:
                mark = marks[ok] = StridedCudnn(ok, dtype)
            proc._attention_backend = mark
            swapped += 1
    if swapped and logger is not None:
        logger.info(
            "diffusion.q21_attn: %d Qwen-Image-2.1 attention module(s) pass cuDNN their strided views "
            "(no Q/K/V or output copies; cards %s)",
            swapped,
            ",".join(sorted({str(c) for m in marks.values() for c in m.cards})),
        )
    return swapped


__all__ = [
    "STRIDED_ATTN_ENV",
    "StridedCudnn",
    "enabled",
    "install",
    "maybe_strided",
    "plain_qkv",
    "strided_cudnn_attention",
    "strided_matches",
]
