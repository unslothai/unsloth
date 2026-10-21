# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inpaint mask settings: AUTOMATIC1111's "Mask blur" and "Inpaint area: Only masked".

Mask blur is a Gaussian blur of the mask before the inpaint pipeline sees it (diffusers
``VaeImageProcessor.blur`` is exactly ``PIL.ImageFilter.GaussianBlur(radius)``). "Only masked"
is diffusers ``padding_mask_crop``: the pipeline crops the masked region plus padding, renders
it at full resolution and pastes it back over the source. Unset / 0 / False leave the request
exactly as before."""

from __future__ import annotations

import inspect
from typing import Any, Optional

# A1111's "Only masked padding, pixels" default (modules/processing.py, infotext "Masked area padding").
DEFAULT_INPAINT_PADDING = 32

ONLY_MASKED_UNSUPPORTED_MSG = (
    "Inpaint area 'Only masked' is not supported by this model's inpaint pipeline."
)


def blur_mask(mask: Any, radius: Optional[int]) -> Any:
    """``mask`` blurred by ``radius`` px; the same object when the radius is unset or 0."""
    if not radius or int(radius) <= 0:
        return mask
    from PIL import ImageFilter

    return mask.filter(ImageFilter.GaussianBlur(int(radius)))


def resolve_padding(only_masked: Optional[bool], padding: Optional[int]) -> Optional[int]:
    """The ``padding_mask_crop`` value, or None when "Only masked" is off."""
    if not only_masked:
        return None
    return DEFAULT_INPAINT_PADDING if padding is None else int(padding)


def pipeline_accepts_padding_mask_crop(pipeline_cls: Any) -> bool:
    call = getattr(pipeline_cls, "__call__", None)
    if call is None:
        return False
    try:
        return "padding_mask_crop" in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


def inpaint_class_accepts_padding_mask_crop(class_name: Optional[str]) -> bool:
    """Whether the diffusers inpaint pipeline named ``class_name`` takes ``padding_mask_crop``."""
    if not class_name:
        return False
    try:
        import diffusers
        cls = getattr(diffusers, class_name, None)
    except Exception:  # noqa: BLE001 - a status read must not fail on an import
        return False
    return cls is not None and pipeline_accepts_padding_mask_crop(cls)
