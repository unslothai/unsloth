# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How far an image is scaled down before chat inference.

Defaults (chosen with measured token/latency costs, see the PR):
- fixed-size / ``longest_edge`` processors: the LOADED MODEL's own size — the same rule training's
  "Image Size" default follows. These models bound their token count themselves, and a smaller cap throws
  away detail they were trained to read (OCR models keep a page's layout but lose its characters).
  Clamped to MAX_SAFE_SIDE (2048 px) so an unusually large declared size cannot explode the cost.
- pixel-budget processors (``max_pixels``, the Qwen-VL family): 1024 px on the longest side. Their own
  budget (Qwen2.5-VL: ~12.8 MP) turns a phone photo into ~15k image tokens — far too slow for chat.
- no declared size: 1024 px.

``UNSLOTH_STUDIO_INFERENCE_IMAGE_MAX_SIDE`` overrides everything: a pixel value caps the longest side,
``0`` disables resizing.
"""

import math
import os

INFERENCE_IMAGE_MAX_SIDE_ENV = "UNSLOTH_STUDIO_INFERENCE_IMAGE_MAX_SIDE"
DEFAULT_INFERENCE_IMAGE_MAX_SIDE = 1024  # no declared size
PIXEL_BUDGET_MAX_SIDE = 1024  # chat default for pixel-budget (max_pixels) processors
MAX_SAFE_SIDE = 2048  # hard ceiling for a model-declared side


def inference_image_max_side() -> int | None:
    """The env override in pixels (0 = no resize), or None when unset. Raises ValueError on a malformed value."""
    raw = os.environ.get(INFERENCE_IMAGE_MAX_SIDE_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{INFERENCE_IMAGE_MAX_SIDE_ENV} must be a whole number of pixels (0 = no resize), got {raw!r}"
        ) from None
    if value < 0:
        raise ValueError(f"{INFERENCE_IMAGE_MAX_SIDE_ENV} must be >= 0, got {value}")
    return value


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def native_image_limit(processor = None, model = None) -> dict | None:
    """The model's image limit: {"max_side": px[, "max_pixels": px]}, else None.

    Order: a pixel budget (``max_pixels`` — Qwen-VL style processors, which ALSO spell it as
    ``size["longest_edge"]``, so it is checked first) → ``size["longest_edge"]`` → the larger of
    ``size["height"]`` / ``size["width"]`` → ``model.config.vision_config.image_size`` (what training's
    default reads). A pixel budget maps to PIXEL_BUDGET_MAX_SIDE (plus the budget itself); a side is
    clamped to MAX_SAFE_SIDE."""
    image_processor = getattr(processor, "image_processor", None) if processor is not None else None
    if image_processor is not None:
        size = getattr(image_processor, "size", None)
        size = size if isinstance(size, dict) else {}
        budget = _positive_int(getattr(image_processor, "max_pixels", None)) or _positive_int(
            size.get("max_pixels")
        )
        if budget is not None:
            # the budget still applies when it is the tighter bound (e.g. a small max_pixels)
            return {"max_side": PIXEL_BUDGET_MAX_SIDE, "max_pixels": budget}
        side = _positive_int(size.get("longest_edge"))
        if side is None:
            sides = [
                s
                for s in (_positive_int(size.get("height")), _positive_int(size.get("width")))
                if s
            ]
            side = max(sides) if sides else None
        if side is not None:
            return {"max_side": min(side, MAX_SAFE_SIDE)}
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    image_size = getattr(vision_config, "image_size", None)
    if isinstance(image_size, (tuple, list)):
        sides = [s for s in (_positive_int(x) for x in image_size) if s]
        image_size = max(sides) if sides else None
    side = _positive_int(image_size)
    if side is not None:
        return {"max_side": min(side, MAX_SAFE_SIDE)}
    return None


def effective_image_limit(model_limit: dict | None = None) -> dict | None:
    """The limit actually applied: the env override if set (0 → None = no resize), else the model's own
    limit, else the 1024 px fallback."""
    override = inference_image_max_side()
    if override is not None:
        return {"max_side": override} if override > 0 else None
    if model_limit:
        return model_limit
    return {"max_side": DEFAULT_INFERENCE_IMAGE_MAX_SIDE}


def resize_for_inference(
    img,
    max_side: int | None = None,
    *,
    model_limit: dict | None = None,
):
    """Downscale ``img`` to the effective limit, keeping the aspect ratio. Never upscales. An explicit
    ``max_side`` wins over everything (0 = no resize)."""
    if img is None:
        return None
    limit = (
        ({"max_side": max_side} if max_side else None)
        if max_side is not None
        else effective_image_limit(model_limit)
    )
    if not limit:
        return img
    width, height = img.size
    ratio = 1.0
    if limit.get("max_side"):
        ratio = min(ratio, limit["max_side"] / width, limit["max_side"] / height)
    if limit.get("max_pixels") and width * height > limit["max_pixels"]:
        ratio = min(ratio, math.sqrt(limit["max_pixels"] / (width * height)))
    if ratio >= 1.0:
        return img
    from PIL import Image

    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return img.resize(new_size, Image.Resampling.LANCZOS)
