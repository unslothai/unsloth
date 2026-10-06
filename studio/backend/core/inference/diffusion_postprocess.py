# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The decoded image to uint8 on the device, not in numpy on the host.

diffusers' ``VaeImageProcessor.postprocess(output_type="pil")`` copies the decoded float image to the host and runs
``(x * 255).round().astype("uint8")`` in numpy over a planar (channel-major) float32 view: three full-size float32
temporaries per 1024 px image. Alone that costs ~25-30 ms on a B200 host, but inside a loaded Studio server it was the
whole post-decode window (0.02-0.54 s on warm renders, 1.6 s on one, with every other step of the window under 2 ms),
because host work there competes with everything else on the box. The same arithmetic on the device is one small
kernel and a 3 MB copy (2 ms).

Bit-identical: the float32 multiply by 255, round-half-to-even and the uint8 cast are the same IEEE operations on
either side, and the cast only runs on values the denormalize already clamped to [0, 1]. Anything else (a CPU tensor,
a batch that skips the denormalize, a grayscale image, NaNs, a processor that overrides ``postprocess``) takes the stock
path. Kill switch: ``UNSLOTH_DIFFUSION_DEVICE_POSTPROCESS=0``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

DEVICE_POSTPROCESS_ENV = "UNSLOTH_DIFFUSION_DEVICE_POSTPROCESS"
_MARK = "_unsloth_device_postprocess"


def disabled() -> bool:
    return (os.environ.get(DEVICE_POSTPROCESS_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _stock_postprocess_class(processor: Any) -> bool:
    """Only a processor whose ``postprocess`` is diffusers' own VaeImageProcessor one (subclasses that override it,
    such as the LDM3D or PixArt processors, keep theirs)."""
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


def to_pil_on_device(processor: Any, image: Any, do_denormalize: Optional[list] = None) -> Optional[list]:
    """The PIL list stock ``postprocess(image, "pil", do_denormalize)`` returns, or None when the stock path must run."""
    import torch

    if not isinstance(image, torch.Tensor) or image.device.type == "cpu":
        return None
    # RGB, or RGBA (Qwen-Image-2.1's VAE decodes 4 channels); stock fromarray picks the mode from the channel count.
    if image.ndim != 4 or int(image.shape[1]) not in (3, 4) or not image.is_floating_point():
        return None
    if not _all_denormalized(processor, image, do_denormalize):
        return None
    from PIL import Image

    # The same call stock makes, so the values (and their dtype) entering the uint8 conversion are the same.
    image = processor._denormalize_conditionally(image, do_denormalize)
    if bool(torch.isnan(image).any()):
        return None
    pixels = uint8_hwc(image).cpu().numpy()
    return [Image.fromarray(pixels[i]) for i in range(pixels.shape[0])]


def uint8_hwc(image: Any) -> Any:
    """Stock's ``.permute(0, 2, 3, 1).float()`` then numpy ``(x * 255).round().astype("uint8")``, as torch ops on
    ``image``'s own device. Values must already lie in [0, 1] (the uint8 cast of anything else is undefined)."""
    import torch

    return (image.float() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def install(pipe: Any, logger: Any = None) -> bool:
    """Route ``pipe.image_processor.postprocess(..., output_type="pil")`` through ``to_pil_on_device``. Idempotent;
    an instance attribute, so ``uninstall`` (or dropping the pipe) restores stock."""
    if disabled():
        return False
    processor = getattr(pipe, "image_processor", None)
    if processor is None or getattr(processor, _MARK, False) or not _stock_postprocess_class(processor):
        return False
    stock = processor.postprocess

    def postprocess(image: Any, output_type: str = "pil", do_denormalize: Optional[list] = None) -> Any:
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
