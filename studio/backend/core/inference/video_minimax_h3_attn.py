# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 attention fast path: strided SDPA layout, fused int8 QKV, and the per-arch kernel pick.

H3 runs ONE full self-attention per block over the whole packed sequence (B=1, 56 heads of 128, ~19.3k rows at
960x544x124), so attention is about half of every denoising step and its prologue is the bulk of the per-block
memory traffic outside the GEMMs. Three levers, each with its own kill switch, all lossless where measured:

1. Strided layout (``UNSLOTH_H3_STRIDED_ATTN=0`` turns it off). diffusers' pinned cuDNN backend copies q, k and v
   into ``(B, H, S, D)`` with ``.contiguous()`` and its output is then permuted back and copied again by the
   processor's ``flatten``. Torch's cuDNN and flash SDPA both accept the permuted ``(B, S, H, D)`` views directly and
   return their output in that same physical layout, so this processor hands them the views: under Inductor the v
   copy and the output permute kernels disappear and the rope / norm prologue writes coalesced rows instead of a
   transposed scatter. The attention kernels are layout-invariant (eager outputs bit-identical across layouts), and
   with Studio's ``emulate_precision_casts`` the compiled block is bit-identical too.

2. Fused int8 QKV (``UNSLOTH_H3_FUSED_QKV=0``). The three projections read the same input, so the stock path
   block-Hadamard-rotates it three times, dynamically quantizes it three times and launches three int8 GEMMs. A
   resident pre-quantized denoiser gets one ``to_qkv`` whose torchao ``Int8Tensor`` is the row-concatenation of the
   three (per-row weight scales concatenate the same way): one rotation, one activation quant, one GEMM. Integer
   accumulation is exact and the dequant epilogue is elementwise, so the result is bit-identical. The diffusers H3
   processor already has a ``fused_projections`` branch, so the module layout is the one diffusers expects.

3. Per-arch kernel (``UNSLOTH_H3_ATTN_ARCH=0``). Studio pins cuDNN attention under a speed profile. At H3's shape
   torch's FlashAttention-2 SDPA is faster on the archs listed in ``H3_FLASH_FASTER_ARCHS`` (measured; see the
   table there), while cuDNN wins or ties on sm120. Only measured archs move; every other arch keeps cuDNN.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

STRIDED_ATTN_ENV = "UNSLOTH_H3_STRIDED_ATTN"
FUSED_QKV_ENV = "UNSLOTH_H3_FUSED_QKV"
ATTN_ARCH_ENV = "UNSLOTH_H3_ATTN_ARCH"

# torch SDPA FlashAttention-2 vs cuDNN at B=1 H=56 D=128 bf16, no mask, (B, S, H, D) views; median ms per layer,
# torch 2.11 / cuDNN 9.19 (Colab, this change's microbench):
#   sm80  A100-40GB  S=19303: flash 53.4  cuDNN 57.2 (cuDNN contiguous, today's path: 59.5);  S=37525: 202.6 / 216.4
#   sm89  L4         S=19303: flash 170.6 cuDNN 173.1 (180.6);                                 S=37525: 655.9 / 678.4
#   sm120 RTX PRO 6000 S=19303: flash 29.8 cuDNN 28.8 (30.0);                                  S=37525: 113.2 / 109.2
# sm90 / sm100 and anything unmeasured keep cuDNN.
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
    selected: Optional[str], capability: Optional[tuple[int, int]] = None
) -> Optional[str]:
    """The dispatcher backend H3 should run given what Studio's generic selection picked.

    Only the automatic cuDNN pick moves, and only to torch's flash SDPA on an arch where flash was measured faster
    at H3's shape. An explicit request (anything but cuDNN), a CPU / ROCm target or an unmeasured arch is returned
    unchanged."""
    if selected != _CUDNN or not _enabled(ATTN_ARCH_ENV):
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
    """``MiniMaxH3AttnProcessor`` subclass that feeds torch SDPA the permuted views, built once.

    One class object for every attention module (dynamo guards on the type id of each module it closes over; a
    class per call would retrace every block). Anything this path does not cover (a mask, context parallelism, a
    backend other than torch's cuDNN / flash SDPA) goes through the stock ``__call__`` unchanged. The kill switch is
    read at install, not per call."""
    import torch
    import torch.nn.functional as F
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
        _apply_rotary_emb,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # The torch-native dispatcher backends that already hand SDPA strided views (math / efficient) are listed too,
    # so the processor stays exercisable off CUDA; Studio only ever pins cuDNN or flash here.

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
        # set per instance at install from UNSLOTH_H3_QK_ROPE; the class default keeps the stock norm + rope
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
                # one read + one write per row for norm and rope together (video_minimax_h3_qknorm). Called through
                # torch.ops (registered at install): dynamo traces into a Python wrapper around the op's builder and
                # graph-breaks there, which split every compiled block in two (measured: 2x slower steps).
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

            # (B, S, H, D) -> (B, H, S, D) views, no copy: both kernels take the strides and write their output in
            # the query's physical layout, so the permute back below is a view as well.
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
    """Swap every H3 attention processor for the strided one, keeping the backend it was set to. Returns the count.

    Call after ``set_attention_backend`` (or before: the backend attribute is carried over either way, and a later
    ``set_attention_backend`` reaches the new processor through the same ``_attention_backend`` attribute)."""
    if not _enabled(STRIDED_ATTN_ENV):
        return 0
    try:
        cls = strided_processor_class()
    except Exception as exc:  # noqa: BLE001 -- an older diffusers without the H3 processor: stock path
        if logger is not None:
            logger.info("video.h3_attn: strided processor unavailable, keeping stock: %s", exc)
        return 0
    from .video_minimax_h3_qknorm import qk_norm_rope_op, qk_rope_enabled

    qk_rope_on = qk_rope_enabled()
    if qk_rope_on:
        try:
            # Register the custom op now, outside any traced region.
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
    """How many H3 attention modules run the strided processor (engagement census)."""
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
    """A torchao int8 weight whose linear quantizes the ACTIVATION per row (per token), so the activation quant
    depends on the input alone and stays the same when weight rows are appended: v1
    ``LinearActivationQuantizedTensor`` with the per-token int8 quant, or a plain per-row ``Int8Tensor``."""
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
    """Row-concatenate tensors or torchao tensor subclasses of identical structure.

    Plain tensors concatenate on dim 0. A subclass is flattened, every inner tensor is concatenated the same way
    (each must carry one row per OUTER row: per-row int data, scales, zero points; a per-tensor scale is refused),
    and the subclass is rebuilt with the outer size updated. Non-tensor attributes must match exactly, apart from
    an attribute that IS the outer shape (v1 ``AffineQuantizedTensor`` records it), which is rewritten."""
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
        # TorchAOBaseTensor subclasses (Int8Tensor) flatten their attributes into a dict, v1 ones into a list.
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
    """(to_q, to_k, to_v, rotation group or None) when the three projections can become one int8 GEMM, else None."""
    from .diffusion_convrot import is_rotated_linear

    if getattr(module, "fused_projections", False):
        return None
    parts = tuple(getattr(module, name, None) for name in ("to_q", "to_k", "to_v"))
    if any(p is None for p in parts):
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
    """The fused projection reproduces the three it replaces, bit for bit, on this device (a few rows)."""
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
    """Fuse ``to_q``/``to_k``/``to_v`` into ``to_qkv`` on every eligible H3 attention module, in place.

    Eligible = all three are bias-free ``nn.Linear`` (or all ConvRot-rotated at one group) carrying torchao int8
    weights with per-token activation quant and per-row weight scales, identical quantization attributes, one
    device, no hooks. Each fused module must reproduce the three projections bit for bit on its device before it is
    swapped in; anything else is left exactly as it was. One module at a time, so the transient extra memory is one
    block's QKV."""
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
        # The three per-module projections are freed piecemeal while each fused weight is a fresh, larger block, so
        # the caching allocator is left holding ~5.8 GB of unreusable fragments (measured: cold-render reserved
        # 58.7 -> 64.5 GB on an RTX PRO 6000). Hand them back.
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
    """How many H3 attention modules run one fused QKV projection (engagement census)."""
    return sum(
        1 for m in _h3_attention_modules(transformer) if getattr(m, "fused_projections", False)
    )
