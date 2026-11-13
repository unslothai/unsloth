# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 cuDNN attention on strided [B, S, H, D] views, skipping diffusers' contiguous Q/K/V copies. Installed only
when a load-time probe finds it bitwise equal to the contiguous call (UNSLOTH_DIFFUSION_FLUX3_STRIDED_ATTN=0 disables)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

STRIDED_ATTN_ENV = "UNSLOTH_DIFFUSION_FLUX3_STRIDED_ATTN"
_CUDNN = "_native_cudnn"
_MODULE = "diffusers.models.transformers.transformer_flux3"
# diffusion_autoopt.patches.source_digest of Flux3AttnProcessor.__call__ the subclass mirrors (d42ebf9c and 0.42.1).
_PROCESSOR_DIGESTS = frozenset({"e3282b2e03be05d8"})
# (device index, dtype, heads, head_dim) -> strided cuDNN output == contiguous cuDNN output, bitwise.
_PROBED: dict[tuple, bool] = {}


def enabled() -> bool:
    return (os.environ.get(STRIDED_ATTN_ENV) or "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _backend_value(backend: Any) -> Optional[str]:
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


@lru_cache(maxsize = None)
def strided_processor_class() -> Any:
    """Built once: dynamo guards on the class. Masks, context parallelism and non-cuDNN backends use the stock call."""
    import torch.nn.functional as F
    from diffusers.models.transformers import transformer_flux3 as tf3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from .diffusion_flux3_qk_rope import qk_norm_rope

    class UnslothFlux3StridedAttnProcessor(tf3.Flux3AttnProcessor):
        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            attention_mask: Any = None,
            image_rotary_emb: Any = None,
        ) -> Any:
            if (
                attention_mask is not None
                or self._parallel_config is not None
                or _backend_value(self._attention_backend) != _CUDNN
            ):
                return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
            query, key, value = tf3._get_qkv_projections(attn, hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            # module attribute at call time: the fused RoPE install patches it there
            query, key = qk_norm_rope(
                attn, query, key, value.dtype, image_rotary_emb, tf3.apply_rotary_emb
            )

            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    attn_mask = None,
                    dropout_p = 0.0,
                    is_causal = False,
                    scale = None,
                    enable_gqa = False,
                )
            hidden_states = out.permute(0, 2, 1, 3).flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    return UnslothFlux3StridedAttnProcessor


def _attention_modules(transformer: Any) -> list:
    try:
        from diffusers.models.transformers.transformer_flux3 import Flux3Attention
    except Exception:  # noqa: BLE001
        return []
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return []
    return [m for m in modules() if isinstance(m, Flux3Attention)]


def _device(target: Any) -> Any:
    import torch

    raw = str(getattr(target, "torch_device", None) or getattr(target, "device", None) or "cuda")
    device = torch.device(raw)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def strided_matches(device: Any, dtype: Any, heads: int, head_dim: int) -> bool:
    """Cached per (card, dtype, heads, head_dim); False on any error."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    key = (device.index, dtype, int(heads), int(head_dim))
    if key in _PROBED:
        return _PROBED[key]
    ok = False
    try:
        gen = torch.Generator(device = device).manual_seed(0)
        q, k, v = (
            torch.randn(
                (2, 333, heads, head_dim), generator = gen, device = device, dtype = torch.float32
            ).to(dtype)
            for _ in range(3)
        )
        with torch.no_grad(), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            ref = F.scaled_dot_product_attention(
                *(x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
            ).permute(0, 2, 1, 3)
            got = F.scaled_dot_product_attention(
                *(x.permute(0, 2, 1, 3) for x in (q, k, v))
            ).permute(0, 2, 1, 3)
        ok = bool(torch.equal(ref, got)) and bool(torch.isfinite(ref.float()).all())
    except Exception:  # noqa: BLE001 - an unprobeable card keeps diffusers' contiguous call
        ok = False
    _PROBED[key] = ok
    return ok


def install(
    pipe: Any,
    backend: Any,
    target: Any,
    logger: Any = None,
) -> int:
    """Returns the number of processors swapped. Idempotent; never raises."""
    try:
        return _install(pipe, backend, target, logger)
    except Exception as exc:  # noqa: BLE001 - the stock processors stay
        if logger is not None:
            logger.info("diffusion.flux3_attn: strided cuDNN attention skipped (%s)", exc)
        return 0


def _install(pipe: Any, backend: Any, target: Any, logger: Any) -> int:
    if not enabled() or _backend_value(backend) != _CUDNN:
        return 0
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or type(transformer).__module__ != _MODULE:
        return 0
    import torch

    if is_rocm_torch(torch) or not torch.cuda.is_available():
        return 0
    from .diffusion_autoopt.patches import source_digest
    from .diffusion_flux3_qk_rope import is_qk_processor
    from diffusers.models.transformers import transformer_flux3 as tf3

    if source_digest(tf3.Flux3AttnProcessor.__call__) not in _PROCESSOR_DIGESTS or not callable(
        getattr(tf3, "_get_qkv_projections", None)
    ):
        if logger is not None:
            logger.info(
                "diffusion.flux3_attn: diffusers' FLUX.3 attention changed; stock processor kept"
            )
        return 0
    device = _device(target)
    if device.type != "cuda":
        return 0
    cls = strided_processor_class()
    dtype = getattr(target, "dtype", None)
    if dtype not in (torch.bfloat16, torch.float16):
        # cuDNN attention runs bf16 / fp16 only; a quantized DiT computes in bf16
        dtype = torch.bfloat16
    swapped = 0
    for attn in _attention_modules(transformer):
        proc = getattr(attn, "processor", None)
        if type(proc) is cls:
            continue
        if type(proc) is not tf3.Flux3AttnProcessor and not is_qk_processor(proc):
            continue
        if _backend_value(getattr(proc, "_attention_backend", None)) != _CUDNN:
            continue
        if not strided_matches(device, dtype, int(attn.heads), int(attn.head_dim)):
            continue
        new = cls()
        new._attention_backend = proc._attention_backend
        new._parallel_config = proc._parallel_config
        new._unsloth_probe_dtype = dtype
        attn.set_processor(new)
        swapped += 1
    if swapped and logger is not None:
        logger.info(
            "diffusion.flux3_attn: %d FLUX.3 attention calls pass cuDNN their strided views (no Q/K/V copies)",
            swapped,
        )
    return swapped


def verify_cards(
    pipe: Any,
    cards: Any,
    logger: Any = None,
) -> int:
    """Probe each split card (install probed the primary only); any failure restores stock processors. Never raises."""
    try:
        import torch
        from diffusers.models.transformers import transformer_flux3 as tf3

        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            return 0
        cls = strided_processor_class()
        swapped = [
            m for m in _attention_modules(transformer) if type(getattr(m, "processor", None)) is cls
        ]
        if not swapped:
            return 0
        bad = sorted(
            {
                int(o)
                for o in cards
                for m in swapped
                if not strided_matches(
                    torch.device("cuda", int(o)),
                    getattr(m.processor, "_unsloth_probe_dtype", torch.bfloat16),
                    int(m.heads),
                    int(m.head_dim),
                )
            }
        )
        if not bad:
            return 0
        for m in swapped:
            stock = tf3.Flux3AttnProcessor()
            stock._attention_backend = m.processor._attention_backend
            stock._parallel_config = m.processor._parallel_config
            m.set_processor(stock)
        if logger is not None:
            logger.info(
                "diffusion.flux3_attn: strided cuDNN attention failed its probe on cuda:%s; stock processor kept",
                ",".join(str(o) for o in bad),
            )
        return len(swapped)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.info("diffusion.flux3_attn: per-card strided attention check skipped (%s)", exc)
        return 0


__all__ = [
    "STRIDED_ATTN_ENV",
    "enabled",
    "install",
    "strided_matches",
    "strided_processor_class",
    "verify_cards",
]
