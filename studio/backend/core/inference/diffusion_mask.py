# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inpaint mask blur and only-masked crop (padding_mask_crop); unset, 0 or False is a no-op."""

from __future__ import annotations

import inspect
from typing import Any, Optional

# A1111's "Only masked padding, pixels" default.
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


def mask_is_empty(mask: Any) -> bool:
    """No nonzero pixel: diffusers' get_crop_region would find a zero-area box."""
    return mask.convert("L").getbbox() is None


def overlay_with_mask(image: Any, source: Any, mask: Any) -> Any:
    """``image`` pasted over ``source`` through the (blurred) ``mask``, as A1111's apply_overlay does.

    Inpaint pipelines binarize the mask they are handed, so without this a blur only moves the hard edge.
    A non-PIL output (a latent or tensor return) is returned unchanged."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        return image
    size = image.size
    if source.size != size:
        source = source.resize(size, Image.LANCZOS)
    if mask.size != size:
        mask = mask.resize(size, Image.BILINEAR)
    mode = image.mode
    out = Image.composite(image.convert("RGB"), source.convert("RGB"), mask.convert("L"))
    return out if mode == "RGB" else out.convert(mode)


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
