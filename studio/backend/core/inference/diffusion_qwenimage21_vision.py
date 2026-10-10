# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 vision SDPA gives NaN conditioning on gfx1151: use query-chunked eager there.

UNSLOTH_QWEN_IMAGE_ROCM_VISION_EAGER: auto (gfx1151 only), 1 (any ROCm GPU), 0 (off).
"""

from __future__ import annotations

import os


_ENV = "UNSLOTH_QWEN_IMAGE_ROCM_VISION_EAGER"
_BACKEND = "unsloth_qwen3_vl_eager"
_QUERY_ROWS = 512


def _eager_vision_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

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
    return torch.cat(outputs, dim = 1), None


def configure_vision_attention(pipe, *, family, target, logger) -> bool:
    if family != "qwen-image-2.1":
        return False
    if getattr(target, "backend", None) != "rocm" or getattr(target, "device", None) != "cuda":
        return False
    override = (os.environ.get(_ENV) or "auto").strip().lower()
    if override in ("0", "false", "off", "no"):
        return False
    if override not in ("auto", "1", "true", "on", "yes"):
        raise ValueError(f"Invalid {_ENV}={override!r}; use auto, 1/true/on/yes or 0/false/off/no")
    if override == "auto":
        import torch
        try:
            from utils.hardware.hardware import _props_gfx_arch
            props = torch.cuda.get_device_properties(torch.device(target.torch_device))
            if _props_gfx_arch(props) != "gfx1151":
                return False
        except Exception:
            return False
    encoder = getattr(pipe, "text_encoder", None)
    visual = getattr(getattr(encoder, "model", None), "visual", None)
    if visual is None or type(visual).__name__ != "Qwen3VLVisionModel":
        return False
    if visual.config._attn_implementation != "sdpa":
        return False
    # Never skip silently: that renders blank images.
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
