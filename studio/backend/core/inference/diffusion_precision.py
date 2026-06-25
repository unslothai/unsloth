# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in FP8 layerwise casting for the diffusion pipeline's text encoder(s).

The transformer arrives quantised in the GGUF, but the companion text encoder loads
dense (bf16) from the base repo and is often the largest resident component (a Qwen3
/ T5-XXL / Mistral encoder runs to many GB). Layerwise casting stores its linear
weights in 8-bit (e4m3) and upcasts each layer to the compute dtype on the fly,
roughly halving the encoder's footprint while normalisations and embeddings stay
full precision. This pairs especially well with streamed (group) offload, where the
text encoder stays resident: on Z-Image it dropped generation peak VRAM ~37%
(10.8 -> 6.8 GB), taking the balanced tier below the lowest-VRAM offload while
keeping its near-resident speed.

It is a memory-vs-quality tradeoff, NOT free: fp8's ~3-bit mantissa perturbs the
text embeddings enough to shift fine detail (on Z-Image, ~20 dB PSNR vs the bf16
encoder -- a larger change than one transformer quant step). Hence off by default;
use the quality harness (scripts/diffusion_quality.py) to confirm it stays within
budget for a given model. Gated to CUDA with a bf16 compute dtype and fp8 dtype
support, and best-effort: any failure leaves the encoder dense. torch / diffusers
are imported lazily so the module stays importable in a no-torch runtime.
"""

from __future__ import annotations

from typing import Any

# Pipeline attributes that hold a text encoder, in order.
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")


def fp8_text_encoder_supported(target: Any) -> bool:
    """Whether fp8 layerwise casting is usable for ``target``: a CUDA device, a bf16
    compute dtype (fp16 + fp8 is too lossy to pair), and torch fp8 dtype support."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16 and hasattr(torch, "float8_e4m3fn")
    except Exception:
        return False


def apply_fp8_text_encoder(
    pipe: Any,
    target: Any,
    *,
    enable: bool,
    logger: Any = None,
) -> list[str]:
    """Cast each text encoder's linear weights to fp8 storage (compute stays the
    target dtype). Returns the names of the encoders actually cast (empty when
    disabled, unsupported, or none present)."""
    if not enable or not fp8_text_encoder_supported(target):
        return []
    try:
        import torch
        from diffusers.hooks import apply_layerwise_casting
        from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "import", exc)
        return []

    cast: list[str] = []
    for attr in _TEXT_ENCODER_ATTRS:
        encoder = getattr(pipe, attr, None)
        if encoder is None:
            continue
        try:
            apply_layerwise_casting(
                encoder,
                storage_dtype = torch.float8_e4m3fn,
                compute_dtype = target.dtype,
                skip_modules_pattern = DEFAULT_SKIP_MODULES_PATTERN,
            )
            cast.append(attr)
        except Exception as exc:  # noqa: BLE001 — leave this encoder dense
            _warn(logger, attr, exc)
    return cast


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: fp8 text-encoder (%s) failed: %s", what, exc)
