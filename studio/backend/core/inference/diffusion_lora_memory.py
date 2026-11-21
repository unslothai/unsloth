# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extra working memory of a LoRA set attached at load, so placement keeps the base build's measured peak.
Adapter weights are already in the loaded sizes; only the unfused PEFT forward is added: the widest layer's
``B(A(x))`` plus its scaled copy (2 x out_features + rank per token), per guidance branch and image. Merged, DoRA,
LoKR and non-LoRA tuners are unpriced and keep the flat placement; ``UNSLOTH_DIFFUSION_LORA_MEASURED=0`` disables."""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

LORA_MEASURED_ENV = "UNSLOTH_DIFFUSION_LORA_MEASURED"
PRICED_ATTR = "_unsloth_lora_priced"
_DENOISERS = ("transformer", "transformer_2", "unconditional_transformer", "unet")
# DiT: one image token per 16x16 pixels plus joint text tokens (longer pipeline max_sequence_length wins)
_DIT_PIXELS_PER_TOKEN = 16 * 16
_DIT_TEXT_TOKENS = 1024
_UNET_PIXELS_PER_TOKEN = 8 * 8
_ROUND_MIB = 64
_DEFAULT_SIDE = 1024


@dataclass(frozen = True)
class LoraPricing:
    """``mib`` is the reserve at the default 1024x1024 single image."""

    signature: tuple
    bytes_per_token: int
    unet: bool
    branches: int
    mib: int
    text_tokens: int = _DIT_TEXT_TOKENS


def enabled() -> bool:
    return (os.environ.get(LORA_MEASURED_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def lora_signature(pipe: Any) -> tuple:
    return tuple(
        sorted(
            (str(entry[0]), str(entry[1])) for entry in getattr(pipe, "_unsloth_loras", ()) or ()
        )
    )


def _element_bytes(*tensors: Any) -> int:
    best = 2
    for t in tensors:
        size = getattr(t, "element_size", None)
        if callable(size):
            try:
                best = max(best, int(size()))
            except Exception:  # noqa: BLE001
                continue
    return best


def _widest_layer_bytes_per_token(
    module: Any,
    compute_bytes: int,
    compute_dtype: Any = None,
) -> Optional[int]:
    """0 without LoRA layers; None when any adapted layer is unpriceable."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:  # noqa: BLE001 - no PEFT: no adapter layers
        BaseTunerLayer = None
    widest = 0
    for layer in module.modules():
        lora_a = getattr(layer, "lora_A", None)
        # A PEFT Embedding keeps its factors in lora_embedding_A/B (Parameters), not lora_A/B
        embed_a = getattr(layer, "lora_embedding_A", None) or {}
        is_tuner = BaseTunerLayer is not None and isinstance(layer, BaseTunerLayer)
        if lora_a is None:
            if is_tuner:
                return None  # another PEFT tuner (LoKr, LoHa, ...): unpriced
            continue
        if getattr(layer, "merged", False):
            return None
        active = [
            a
            for a in (getattr(layer, "active_adapters", None) or ())
            if a in lora_a or a in embed_a
        ]
        if not active:
            continue
        if any(bool((getattr(layer, "use_dora", None) or {}).get(a, False)) for a in active):
            return None  # DoRA rebuilds the weight norm per forward
        base = layer.get_base_layer() if callable(getattr(layer, "get_base_layer", None)) else None
        out = (
            getattr(base, "out_features", None)
            or getattr(base, "out_channels", None)
            or getattr(base, "embedding_dim", None)
        )
        if not out:
            return None
        rank = 0
        elem = compute_bytes
        adapter_elem = 0
        mismatch = False
        for a in active:
            if a in embed_a:
                # embedding(x, A.T) is [tokens, rank], then @ B.T is [tokens, out]: same shape as a Linear
                down_w = embed_a[a]
                up_w = getattr(layer, "lora_embedding_B", {})[a]
                rank += int(down_w.shape[0])
            else:
                down = lora_a[a]
                up = getattr(layer, "lora_B", {})[a]
                down_w, up_w = getattr(down, "weight", None), getattr(up, "weight", None)
                rank += int(
                    getattr(down, "out_features", None) or getattr(down, "out_channels", None) or 0
                )
            elem = max(elem, _element_bytes(down_w, up_w))
            adapter_elem = max(adapter_elem, _element_bytes(down_w, up_w))
            for w in (down_w, up_w):
                dt = getattr(w, "dtype", None)
                if dt is None:
                    continue
                if compute_dtype is not None:
                    mismatch = mismatch or dt != compute_dtype
                else:
                    mismatch = mismatch or _element_bytes(w) != compute_bytes
        per = (2 * int(out) + rank) * elem
        if mismatch:
            # dtype mismatch: input cast copy + delta + promoted (fp32) sum stay alive together
            inf = getattr(base, "in_features", None) or getattr(base, "in_channels", None) or 0
            per = max(per, (int(inf) + int(out) + rank) * adapter_elem + int(out) * 4)
        widest = max(widest, per)
    return widest


def _text_window(pipe: Any) -> int:
    try:
        param = inspect.signature(type(pipe).__call__).parameters.get("max_sequence_length")
        default = getattr(param, "default", None)
        if (
            isinstance(default, int)
            and not isinstance(default, bool)
            and default > _DIT_TEXT_TOKENS
        ):
            return int(default)
    except (TypeError, ValueError):
        pass
    return _DIT_TEXT_TOKENS


def _tokens(
    unet: bool,
    width: Optional[int],
    height: Optional[int],
    text_tokens: int = _DIT_TEXT_TOKENS,
    extra_pixels: int = 0,
) -> int:
    """``extra_pixels``: condition / reference images that join the denoiser's sequence (edit references, layers)."""
    w = max(64, int(width or _DEFAULT_SIDE))
    h = max(64, int(height or _DEFAULT_SIDE))
    pixels = w * h + max(0, int(extra_pixels or 0))
    if unet:
        return math.ceil(pixels / _UNET_PIXELS_PER_TOKEN)
    return math.ceil(pixels / _DIT_PIXELS_PER_TOKEN) + max(_DIT_TEXT_TOKENS, int(text_tokens))


def _mib(
    pricing_bytes: int,
    unet: bool,
    branches: int,
    width: Optional[int],
    height: Optional[int],
    batch_size: int,
    text_tokens: int = _DIT_TEXT_TOKENS,
    extra_pixels: int = 0,
) -> int:
    nbytes = (
        pricing_bytes
        * _tokens(unet, width, height, text_tokens, extra_pixels)
        * max(1, int(branches))
        * max(1, int(batch_size or 1))
    )
    mib = -(-nbytes // (1 << 20))
    return int(-(-mib // _ROUND_MIB) * _ROUND_MIB) if mib > 0 else 0


def price_attached_loras(
    pipe: Any,
    *,
    cfg: bool,
    compute_bytes: Optional[int] = None,
    compute_dtype: Any = None,
) -> Optional[LoraPricing]:
    """Price and record the attached LoRA set; None (nothing recorded) when it cannot be priced."""
    if not enabled() or not getattr(pipe, "_unsloth_loras", None):
        return None
    try:
        from .diffusion_lora_files import kron_names
        if kron_names(pipe):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        widest = 0
        unet = False
        found = False
        for name in _DENOISERS:
            module = getattr(pipe, name, None)
            if module is None or not callable(getattr(module, "modules", None)):
                continue
            found = True
            per = _widest_layer_bytes_per_token(
                module, max(2, int(compute_bytes or 2)), compute_dtype
            )
            if per is None:
                return None
            if per > widest:
                widest, unet = per, name == "unet"
        if not found or widest <= 0:
            return None
        branches = 2 if cfg else 1
        text_tokens = _text_window(pipe)
        pricing = LoraPricing(
            signature = lora_signature(pipe),
            bytes_per_token = int(widest),
            unet = unet,
            branches = branches,
            mib = _mib(widest, unet, branches, None, None, 1, text_tokens),
            text_tokens = text_tokens,
        )
        setattr(pipe, PRICED_ATTR, pricing)
        return pricing
    except Exception:  # noqa: BLE001 - unpriced keeps the flat placement
        return None


def priced(pipe: Any) -> Optional[LoraPricing]:
    pricing = getattr(pipe, PRICED_ATTR, None)
    if not isinstance(pricing, LoraPricing):
        return None
    if lora_signature(pipe) != pricing.signature:
        return None
    return pricing


def load_reserve_mib(pipe: Any) -> int:
    pricing = priced(pipe)
    return int(pricing.mib) if pricing is not None else 0


def request_extra_mib(
    pipe: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    extra_pixels: int = 0,
) -> int:
    pricing = priced(pipe)
    if pricing is None:
        return 0
    need = _mib(
        pricing.bytes_per_token,
        pricing.unet,
        pricing.branches,
        width,
        height,
        batch_size,
        pricing.text_tokens,
        extra_pixels,
    )
    return max(0, int(need) - int(pricing.mib))
