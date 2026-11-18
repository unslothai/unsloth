# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Device-side uint8 postprocess for output_type pil, bit-identical to stock numpy; other cases take
the stock path; NaN raises instead of saving blank"""

from __future__ import annotations

import os
from typing import Any, Optional

DEVICE_POSTPROCESS_ENV = "UNSLOTH_DIFFUSION_DEVICE_POSTPROCESS"
STRIP_UINT8_ENV = "UNSLOTH_DIFFUSION_UINT8_STRIPS"
_STRIP_BYTES = 32 << 20
_MARK = "_unsloth_device_postprocess"
NAN_IMAGE_MESSAGE = (
    "The decoded image contains NaN values and would be saved as a blank image. The model or VAE overflowed at this "
    "resolution and precision; try a smaller resolution or another quant of this model."
)


def disabled() -> bool:
    return (os.environ.get(DEVICE_POSTPROCESS_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _stock_postprocess_class(processor: Any) -> bool:
    """Subclasses overriding ``postprocess`` (LDM3D, PixArt) keep theirs."""
    try:
        from diffusers.image_processor import VaeImageProcessor
    except Exception:  # noqa: BLE001
        return False
    return isinstance(processor, VaeImageProcessor) and (
        getattr(type(processor), "postprocess", None) is VaeImageProcessor.postprocess
    )


def _all_denormalized(processor: Any, image: Any, do_denormalize: Optional[list]) -> bool:
    if do_denormalize is None:
        return bool(getattr(getattr(processor, "config", None), "do_normalize", False))
    return len(do_denormalize) == int(image.shape[0]) and all(bool(d) for d in do_denormalize)


def to_pil_on_device(
    processor: Any,
    image: Any,
    do_denormalize: Optional[list] = None,
) -> Optional[list]:
    """The PIL list stock ``postprocess(image, "pil", do_denormalize)`` returns, or None when the stock path must run."""
    import torch

    if not isinstance(image, torch.Tensor) or image.device.type == "cpu":
        return None
    # RGBA: Qwen-Image-2.1's VAE decodes 4 channels.
    if image.ndim != 4 or int(image.shape[1]) not in (3, 4) or not image.is_floating_point():
        return None
    if not _all_denormalized(processor, image, do_denormalize):
        return None
    from PIL import Image

    image = processor._denormalize_conditionally(image, do_denormalize)
    if bool(torch.isnan(image).any()):
        return None
    pixels = uint8_hwc(image).cpu().numpy()
    return [Image.fromarray(pixels[i]) for i in range(pixels.shape[0])]


def has_nan(image: Any) -> bool:
    import torch
    return (
        isinstance(image, torch.Tensor)
        and image.is_floating_point()
        and image.device.type != "meta"
        and bool(torch.isnan(image).any())
    )


def _strips_disabled() -> bool:
    return (os.environ.get(STRIP_UINT8_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def uint8_hwc(image: Any) -> Any:
    """Values must lie in [0, 1]: the uint8 cast of anything else is undefined."""
    import torch

    if _strips_disabled() or image.ndim != 4:
        return (image.float() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    batch, channels, height, width = image.shape
    out = torch.empty((batch, height, width, channels), dtype = torch.uint8, device = image.device)
    rows = max(1, _STRIP_BYTES // max(1, batch * channels * width * 4))
    for start in range(0, height, rows):
        piece = image[:, :, start : start + rows].to(torch.float32, copy = True)
        piece.mul_(255).round_()
        out[:, start : start + rows].copy_(piece.permute(0, 2, 3, 1))
        del piece
    return out


def install(pipe: Any, logger: Any = None) -> bool:
    """Idempotent; an instance attribute, so ``uninstall`` restores stock."""
    if disabled():
        return False
    processor = getattr(pipe, "image_processor", None)
    if (
        processor is None
        or getattr(processor, _MARK, False)
        or not _stock_postprocess_class(processor)
    ):
        return False
    stock = processor.postprocess

    def postprocess(
        image: Any,
        output_type: str = "pil",
        do_denormalize: Optional[list] = None,
    ) -> Any:
        if output_type != "latent" and has_nan(image):
            # Stock casts NaN to 0, a blank image that looks like a finished render.
            raise RuntimeError(NAN_IMAGE_MESSAGE)
        if output_type == "pil":
            try:
                out = to_pil_on_device(processor, image, do_denormalize)
            except Exception as exc:  # noqa: BLE001 - the stock path is always correct
                out = None
                if logger is not None:
                    logger.debug("diffusion.postprocess: stock path (%s)", exc)
            if out is not None:
                return out
        return stock(image, output_type = output_type, do_denormalize = do_denormalize)

    processor.postprocess = postprocess
    setattr(processor, _MARK, True)
    return True


def uninstall(pipe: Any) -> None:
    processor = getattr(pipe, "image_processor", None)
    if processor is None or not getattr(processor, _MARK, False):
        return
    try:
        del processor.postprocess
    except AttributeError:
        pass
    try:
        delattr(processor, _MARK)
    except AttributeError:
        pass
