# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision casting of the diffusion pipeline's text encoder(s).

The transformer arrives quantised in the GGUF, but the companion text encoder loads
dense (bf16) from the base repo and is often the largest resident component (a Qwen3
/ T5-XXL / Mistral encoder runs to many GB). This shrinks it in place, with two
backends:

  fp8   - diffusers layerwise casting: 8-bit (e4m3) storage, upcast per layer to the
          compute dtype. ~2x smaller. Works on any fp8-capable CUDA card (cc >= 8.9).
  nvfp4 - torchao NVFP4 weight-only: 4-bit float with two-level microscaling, run on
          Blackwell's (sm_100+) FP4 tensor cores. ~4x smaller and the lowest-VRAM
          option, but a steeper quality cost than fp8.

Both keep normalisations / embeddings full precision and are a memory-vs-quality
tradeoff, not free, so both are off by default. They pair especially well with
streamed (group) offload, where the text encoder stays resident -- this is where the
companion footprint dominates. Quantify the quality cost per model with the quality
harness (scripts/diffusion_quality.py). torch / diffusers / torchao are imported
lazily so the module stays importable in a no-torch runtime.
"""

from __future__ import annotations

from typing import Any, Optional

TE_QUANT_FP8 = "fp8"
TE_QUANT_NVFP4 = "nvfp4"
TE_QUANT_MODES = (TE_QUANT_FP8, TE_QUANT_NVFP4)

# Pipeline attributes that hold a text encoder, in order.
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")


def normalize_te_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested text-encoder quant; None / "" / "none" -> None.

    Raises ValueError for an unsupported value. The route rejects bad values at the
    Pydantic Literal boundary; this guard covers direct / script callers."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized == "none":
        return None
    if normalized not in TE_QUANT_MODES:
        raise ValueError(
            f"Unsupported text_encoder_quant '{value}'. Use one of: {', '.join(TE_QUANT_MODES)}."
        )
    return normalized


def te_quant_supported(target: Any, mode: str) -> bool:
    """Whether ``mode`` is usable for ``target``: a CUDA device with a bf16 compute
    dtype, plus fp8 dtype support (fp8) or Blackwell sm_100+ tensor cores (nvfp4)."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch

        if getattr(target, "dtype", None) is not torch.bfloat16:
            return False
        if mode == TE_QUANT_FP8:
            return hasattr(torch, "float8_e4m3fn")
        if mode == TE_QUANT_NVFP4:
            # NVFP4 tensor cores need Blackwell (compute capability major >= 10).
            return torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False
    return False


def quantize_text_encoders(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    logger: Any = None,
) -> Optional[str]:
    """Quantise each present text encoder in place with ``mode`` (fp8 / nvfp4).
    Returns the mode actually applied, or None when disabled, unsupported, or no
    encoder was cast. Best-effort: any failure leaves the encoder dense."""
    mode = normalize_te_quant(mode)
    if mode is None or not te_quant_supported(target, mode):
        return None
    caster = _cast_fp8 if mode == TE_QUANT_FP8 else _cast_nvfp4
    cast: list[str] = []
    for attr in _TEXT_ENCODER_ATTRS:
        encoder = getattr(pipe, attr, None)
        if encoder is None:
            continue
        try:
            caster(encoder, target)
            cast.append(attr)
        except Exception as exc:  # noqa: BLE001 — leave this encoder dense
            _warn(logger, f"{mode}:{attr}", exc)
    return mode if cast else None


def _cast_fp8(encoder: Any, target: Any) -> None:
    import torch
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    apply_layerwise_casting(
        encoder,
        storage_dtype = torch.float8_e4m3fn,
        compute_dtype = target.dtype,
        skip_modules_pattern = DEFAULT_SKIP_MODULES_PATTERN,
    )


def _cast_nvfp4(encoder: Any, target: Any) -> None:
    # Weight-only NVFP4: linear weights become 4-bit (packed) NVFP4 tensors and run
    # on Blackwell FP4 tensor cores; norms / embeddings (not nn.Linear) are untouched.
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    quantize_(encoder, NVFP4WeightOnlyConfig())


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.precision: text-encoder quant (%s) failed: %s", what, exc)
