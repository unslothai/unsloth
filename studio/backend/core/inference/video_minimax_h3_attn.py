# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 attention fast path, one kill switch (``=0``) per lever:

1. ``UNSLOTH_H3_STRIDED_ATTN``: torch cuDNN / flash SDPA take the permuted (B, S, H, D) views, so the q / k / v and
   output copies disappear; bit-identical (attention kernels are layout-invariant).
2. ``UNSLOTH_H3_FUSED_QKV``: one row-concatenated int8 ``to_qkv`` (one rotation, act quant and GEMM instead of three);
   bit-identical because int accumulation is exact and the dequant epilogue is per row.
3. ``UNSLOTH_H3_ATTN_ARCH``: flash instead of cuDNN on the archs in ``H3_FLASH_FASTER_ARCHS`` only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

STRIDED_ATTN_ENV = "UNSLOTH_H3_STRIDED_ATTN"
FUSED_QKV_ENV = "UNSLOTH_H3_FUSED_QKV"
ATTN_ARCH_ENV = "UNSLOTH_H3_ATTN_ARCH"

# Measured at H3's shape (torch 2.11): flash beats cuDNN on sm80 / sm89, ties or loses on sm120; unmeasured keep cuDNN.
H3_FLASH_FASTER_ARCHS = frozenset({(8, 0), (8, 9)})

_CUDNN = "_native_cudnn"
_FLASH = "_native_flash"


def _enabled(env: str) -> bool:
    return (os.environ.get(env) or "1").strip().lower() not in ("0", "off", "false", "no")


def _capability(device: Any = None) -> Optional[tuple[int, int]]:
    try:
        import torch
        if not torch.cuda.is_available() or torch.version.hip is not None:
            return None
        return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def h3_attention_backend(
    selected: Optional[str],
    capability: Optional[tuple[int, int]] = None,
    requested: Optional[str] = None,
) -> Optional[str]:
    """Moves only the automatic cuDNN pick (``requested`` auto / unset), and only to flash on a measured-faster arch;
    anything else unchanged."""
    if selected != _CUDNN or not _enabled(ATTN_ARCH_ENV):
        return selected
    try:
        from .diffusion_attention import ATTN_AUTO, normalize_attention_backend
        if normalize_attention_backend(requested) != ATTN_AUTO:
            return selected
    except Exception:  # noqa: BLE001 -- unparseable request: leave it alone
        return selected
    cap = capability if capability is not None else _capability()
    if cap is None or tuple(cap) not in H3_FLASH_FASTER_ARCHS:
        return selected
    return _FLASH


def _backend_value(backend: Any) -> Optional[str]:
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


@lru_cache(maxsize = None)
def strided_processor_class() -> Any:
    """Strided ``MiniMaxH3AttnProcessor`` subclass. Built once: dynamo guards on the class id, so a class per call
    would retrace every block. Masks, context parallelism and other backends take the stock ``__call__``."""
    import torch
    import torch.nn.functional as F
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
        _apply_rotary_emb,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _fusable_norm(norm: Any) -> bool:
        return (
            isinstance(norm, torch.nn.RMSNorm)
            and norm.weight is not None
            and norm.eps is not None
            and tuple(norm.normalized_shape) == (attn_head_dim(norm),)
        )

    def attn_head_dim(norm: Any) -> int:
        return int(norm.weight.shape[-1])

    kernels = {
        _CUDNN: SDPBackend.CUDNN_ATTENTION,
        _FLASH: SDPBackend.FLASH_ATTENTION,
        "_native_math": SDPBackend.MATH,
        "_native_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }

    class UnslothH3StridedAttnProcessor(MiniMaxH3AttnProcessor):
        _unsloth_qk_rope = False

        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            rotary_emb: Any = None,
            attention_mask: Any = None,
        ) -> Any:
            kernel = kernels.get(_backend_value(self._attention_backend))
            if kernel is None or attention_mask is not None or self._parallel_config is not None:
                return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
            if attn.fused_projections:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim = -1)
            else:
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            if (
                rotary_emb is not None
                and self._unsloth_qk_rope
                and _fusable_norm(attn.norm_q)
                and _fusable_norm(attn.norm_k)
            ):
                # Through torch.ops, not the Python wrapper: dynamo graph-breaks inside the wrapper's lru_cache.
                cos, sin = rotary_emb
                op = torch.ops.unsloth_h3.qk_norm_rope
                query = op(query, attn.norm_q.weight, cos, sin, float(attn.norm_q.eps))
                key = op(key, attn.norm_k.weight, cos, sin, float(attn.norm_k.eps))
            else:
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                if rotary_emb is not None:
                    query = _apply_rotary_emb(query, *rotary_emb)
                    key = _apply_rotary_emb(key, *rotary_emb)

            with sdpa_kernel(kernel):
                out = F.scaled_dot_product_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    dropout_p = 0.0,
                    is_causal = False,
                )
            hidden_states = out.permute(0, 2, 1, 3).flatten(2, 3).type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    return UnslothH3StridedAttnProcessor


def _h3_attention_modules(transformer: Any) -> list:
    try:
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
    except Exception:  # noqa: BLE001
        return []
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return []
    return [m for m in modules() if isinstance(m, MiniMaxH3Attention)]


def install_strided_attention(transformer: Any, logger: Any = None) -> int:
    """Swap every H3 attention processor for the strided one, keeping its backend. Returns the count."""
    if not _enabled(STRIDED_ATTN_ENV):
        return 0
    try:
        cls = strided_processor_class()
    except Exception as exc:  # noqa: BLE001 -- older diffusers without the H3 processor
        if logger is not None:
            logger.info("video.h3_attn: strided processor unavailable, keeping stock: %s", exc)
        return 0
    from .video_minimax_h3_qknorm import qk_norm_rope_op, qk_rope_enabled

    qk_rope_on = qk_rope_enabled()
    if qk_rope_on:
        try:
            qk_norm_rope_op()
        except Exception as exc:  # noqa: BLE001 -- keep the stock norm + rope
            qk_rope_on = False
            if logger is not None:
                logger.info("video.h3_attn: fused q/k norm+rope unavailable: %s", exc)
    swapped = 0
    for module in _h3_attention_modules(transformer):
        old = getattr(module, "processor", None)
        if old is None or isinstance(old, cls):
            continue
        new = cls()
        new._unsloth_qk_rope = qk_rope_on
        new._attention_backend = getattr(old, "_attention_backend", None)
        new._parallel_config = getattr(old, "_parallel_config", None)
        module.set_processor(new)
        swapped += 1
    if logger is not None and swapped:
        logger.info("video.h3_attn: strided SDPA layout on %d attention modules", swapped)
    return swapped


def strided_attention_count(transformer: Any) -> int:
    """Engagement census."""
    try:
        cls = strided_processor_class()
    except Exception:  # noqa: BLE001
        return 0
    return sum(
        1
        for m in _h3_attention_modules(transformer)
        if isinstance(getattr(m, "processor", None), cls)
    )


def _per_token_int8(weight: Any) -> bool:
    """Per-token activation quant (depends on the input alone, so appending weight rows cannot change it)."""
    name = type(weight).__name__
    if name == "LinearActivationQuantizedTensor":
        fn = getattr(weight, "input_quant_func", None)
        inner = getattr(weight, "original_weight_tensor", None)
        return (
            getattr(fn, "__name__", "").startswith("_int8_symm_per_token")
            and not getattr(weight, "quant_kwargs", None)
            and type(inner).__name__ == "AffineQuantizedTensor"
            and tuple(getattr(inner, "block_size", ())) == (1, int(weight.shape[-1]))
        )
    if name == "Int8Tensor":
        from .diffusion_int8_fused import _plain_int8_weight
        return _plain_int8_weight(weight)
    return False


def _same_attr(a: Any, b: Any) -> bool:
    try:
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(_same_attr(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_same_attr(x, y) for x, y in zip(a, b))
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- uncomparable: treat as different
        return False


def _cat_rows(parts: list) -> Any:
    """Row-concatenate tensors or identically structured torchao subclasses. Every inner tensor must be per row (a
    per-tensor scale is refused); non-tensor attributes must match, except a recorded outer shape, which is rewritten."""
    import torch

    t0 = parts[0]
    rows = [int(p.shape[0]) for p in parts]
    if any(type(p) is not type(t0) for p in parts):
        raise ValueError("projection weights differ in type")
    if any(
        p.dim() != t0.dim() or tuple(p.shape[1:]) != tuple(t0.shape[1:]) or p.dtype != t0.dtype
        for p in parts
    ):
        raise ValueError("projection weights differ in shape or dtype")
    if type(t0) is torch.Tensor:
        if t0.dim() == 0:
            raise ValueError("scalar leaf")
        return torch.cat([p.detach() for p in parts], dim = 0)
    flatten = getattr(t0, "__tensor_flatten__", None)
    unflatten = getattr(type(t0), "__tensor_unflatten__", None)
    if flatten is None or unflatten is None:
        raise ValueError(f"{type(t0).__name__} is not a flattenable tensor subclass")
    names, ctx = flatten()
    own = tuple(t0.shape)

    def is_shape(x: Any, shape: tuple) -> bool:
        return isinstance(x, (tuple, list, torch.Size)) and tuple(x) == shape

    def normalised(c: Any, shape: tuple) -> Any:
        if isinstance(c, dict):
            return {k: ("<outer-shape>" if is_shape(v, shape) else v) for k, v in c.items()}
        return ["<outer-shape>" if is_shape(x, shape) else x for x in c]

    for p in parts[1:]:
        p_names, p_ctx = p.__tensor_flatten__()
        if list(p_names) != list(names) or not _same_attr(
            normalised(p_ctx, tuple(p.shape)), normalised(ctx, own)
        ):
            raise ValueError("projection weights differ in quantization attributes")
    inner = {}
    for n in names:
        leaves = [getattr(p, n) for p in parts]
        if any(leaf.dim() == 0 or int(leaf.shape[0]) != r for leaf, r in zip(leaves, rows)):
            raise ValueError(f"{type(t0).__name__}.{n} is not per-row")
        inner[n] = _cat_rows(leaves)
    size = torch.Size((sum(rows), *own[1:]))
    if isinstance(ctx, dict):
        new_ctx: Any = {k: (size if is_shape(v, own) else v) for k, v in ctx.items()}
    else:
        new_ctx = [size if is_shape(x, own) else x for x in ctx]
        if isinstance(ctx, tuple):
            new_ctx = tuple(new_ctx)
    return unflatten(inner, new_ctx, size, None)


def _fusable_parts(module: Any) -> Optional[tuple]:
    """(to_q, to_k, to_v, rotation group or None), or None when not fusable."""
    from .diffusion_convrot import is_rotated_linear

    if getattr(module, "fused_projections", False):
        return None
    parts = tuple(getattr(module, name, None) for name in ("to_q", "to_k", "to_v"))
    if any(p is None for p in parts):
        return None
    from torch import nn

    # never a wrapper: an unpadded token refiner (PadToMinM) hits _int_mm's M > 16 assert on a short compiled prompt
    if not all(isinstance(p, nn.Linear) for p in parts):
        return None
    if not all(_per_token_int8(getattr(p, "weight", None)) for p in parts):
        return None
    if any(getattr(p, "bias", None) is not None for p in parts):
        return None
    # Hooks (offload, prefetch, LoRA) are keyed to the three modules; a fused module would silently drop them.
    if any(
        getattr(p, "_forward_pre_hooks", None) or getattr(p, "_forward_hooks", None) for p in parts
    ):
        return None
    if any(hasattr(p, "_hf_hook") or hasattr(p, "_diffusers_hook") for p in parts):
        return None
    if len({type(p) for p in parts}) != 1 or len({p.weight.device for p in parts}) != 1:
        return None
    rotated = [is_rotated_linear(p) for p in parts]
    if any(rotated) and not all(rotated):
        return None
    group = None
    if rotated[0]:
        groups = {int(p.convrot_groupsize) for p in parts}
        if len(groups) != 1:
            return None
        group = groups.pop()
    return parts[0], parts[1], parts[2], group


def _self_check(parts: tuple, fused: Any) -> bool:
    """Bit-exact against the three projections on this device."""
    import torch

    w = parts[0].weight
    device = w.device
    dtype = getattr(w, "dtype", torch.bfloat16)
    g = torch.Generator(device = device).manual_seed(0)
    x = torch.randn(1, 33, int(w.shape[-1]), device = device, dtype = dtype, generator = g)
    with torch.no_grad():
        ref = torch.cat([p(x) for p in parts], dim = -1)
        out = fused(x)
    return out.shape == ref.shape and torch.equal(out, ref)


def fuse_h3_qkv_(transformer: Any, logger: Any = None) -> int:
    """Fuse ``to_q``/``to_k``/``to_v`` into ``to_qkv`` in place where ``_fusable_parts`` allows and ``_self_check``
    passes; one module at a time, so the transient extra memory is one block's QKV."""
    if not _enabled(FUSED_QKV_ENV):
        return 0
    from torch import nn

    from .diffusion_convrot import _install_rotation

    fused = 0
    for module in _h3_attention_modules(transformer):
        try:
            spec = _fusable_parts(module)
        except Exception:  # noqa: BLE001 -- not this layout
            spec = None
        if spec is None:
            continue
        to_q, to_k, to_v, group = spec
        try:
            weight = _cat_rows([to_q.weight, to_k.weight, to_v.weight])
            linear = nn.Linear(
                int(weight.shape[1]), int(weight.shape[0]), bias = False, device = "meta"
            )
            linear.weight = nn.Parameter(weight, requires_grad = False)
            if group is not None:
                _install_rotation(linear, group)
            if not _self_check((to_q, to_k, to_v), linear):
                raise ValueError("fused output differs from the three projections")
        except Exception as exc:  # noqa: BLE001 -- leave this module unfused
            if logger is not None:
                logger.info("video.h3_attn: QKV left unfused: %s", exc)
            continue
        module.to_qkv = linear
        module.fused_projections = True
        for name in ("to_q", "to_k", "to_v"):
            delattr(module, name)
        fused += 1
    if fused:
        # The freed per-projection blocks are fragments the larger fused weights cannot reuse (~5.8 GB on G4).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if logger is not None:
            logger.info("video.h3_attn: fused int8 QKV on %d attention modules", fused)
    return fused


def fused_qkv_count(transformer: Any) -> int:
    """Engagement census."""
    return sum(
        1 for m in _h3_attention_modules(transformer) if getattr(m, "fused_projections", False)
    )
