# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Qwen-Image-2.1 reference-image encoding finite on measured ROCm GPUs.

Defaults to gfx1151. UNSLOTH_QWEN_IMAGE_ROCM_VISION_EAGER=0 opts out;
=1 allows testing on other ROCm GPUs. Only the vision SDPA backend is changed.
Query chunks bound temporary attention memory without dropping any keys or values.
"""

from __future__ import annotations

import os


_ENV = "UNSLOTH_QWEN_IMAGE_ROCM_VISION_EAGER"
_BACKEND = "unsloth_qwen3_vl_eager"
_QUERY_ROWS = 512


def _eager_vision_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    # Keep the stock dropout RNG sequence for training; Studio uses this in eval.
    if query.shape[-2] <= _QUERY_ROWS or kwargs.get("dropout", 0.0):
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
    import torch

    outputs = []
    for start in range(0, query.shape[-2], _QUERY_ROWS):
        end = start + _QUERY_ROWS
        mask = attention_mask
        if mask is not None and mask.shape[-2] != 1:
            mask = mask[..., start:end, :]
        output, weights = eager_attention_forward(
            module, query[..., start:end, :], key, value, mask, **kwargs
        )
        del weights
        outputs.append(output)
    # Transformers' attention interface returns [batch, query, heads, head_dim].
    return torch.cat(outputs, dim = 1), None


def configure_vision_attention(pipe, *, family, device, logger) -> bool:
    """Change only this encoder's vision backend, leaving text and denoising alone."""
    if family != "qwen-image-2.1":
        return False
    import torch

    if not (getattr(torch.version, "hip", None) or "rocm" in torch.__version__.lower()):
        return False
    target = torch.device(device)
    if target.type != "cuda":
        return False
    override = (os.environ.get(_ENV) or "auto").strip().lower()
    if override in ("0", "false", "off", "no"):
        return False
    if override not in ("auto", "1", "true", "on", "yes"):
        raise ValueError(f"Invalid {_ENV}={override!r}; use auto, 1/true/on/yes or 0/false/off/no")
    if override == "auto":
        try:
            from utils.hardware.hardware import _props_gfx_arch
            if _props_gfx_arch(torch.cuda.get_device_properties(target)) != "gfx1151":
                return False
        except Exception:
            return False
    encoder = getattr(pipe, "text_encoder", None)
    visual = getattr(getattr(encoder, "model", None), "visual", None)
    if visual is None or type(visual).__name__ != "Qwen3VLVisionModel":
        return False
    # Respect explicitly selected non-SDPA backends. This fixes the native SDPA
    # path, including Speed Off; it is a correctness fix, not a speed option.
    if visual.config._attn_implementation != "sdpa":
        return False
    # Unlike a speed optimization, silently skipping this can produce blank images.
    try:
        from transformers import AttentionInterface
        AttentionInterface.register(_BACKEND, _eager_vision_attention)
        visual.set_attn_implementation(_BACKEND)
    except Exception as exc:
        raise RuntimeError(
            "Could not enable Qwen-Image-2.1 ROCm vision attention. "
            "Check the installed Transformers version. "
            f"Set {_ENV}=0 to opt out; reference images may produce NaNs."
        ) from exc
    logger.info(
        "diffusion.qwenimage21: using query-chunked eager vision attention on ROCm to avoid NaN image conditioning"
    )
    return True
