# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Condition images and output geometry for the reference and unified-edit workflows.

Shared by the diffusers and sd.cpp engines, so both read the same request the same way: one
ordered image list (primary first, never sorted), one alpha policy per family, one size rule,
and one localized-edit contract.

Localized editing follows the conventions of the official Qwen-Image-2.1 demo cases, which are
generative guides rather than inpainting masks: nothing here keeps pixels outside the region.

- ``annotate``: coloured marks (outlines, arrows) drawn ON the source. The instruction names the
  colours and asks for the marks to be left out of the result.
- ``paint``: an opaque white region painted ON the source. The instruction refers to the white
  area.
- ``mask``: a separate white-on-black mask at the source's geometry, sent as Image 2 right after
  the source. The instruction refers to the marked area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

LOCALIZED_EDIT_MODES = ("annotate", "paint", "mask")

# Decoded source pixels across ALL condition images of one call. Each image already has its own
# 4096 px side bound; ten of those would still decode to over 600 MB of RGBA, so the call as a whole
# is bounded too, before any of them is loaded.
MAX_CONDITION_SOURCE_PIXELS = 64_000_000

# The shortest output side any family accepts (DiffusionGenerateRequest's lower bound).
MIN_OUTPUT_SIDE = 256

# Upstream QwenImage21Pipeline's own ``output_resolution`` default.
DEFAULT_REFERENCE_RESOLUTION = 1024


@dataclass(frozen = True)
class LocalizedEdit:
    mode: str
    image: str


def check_output_size(fam: Any, width: int, height: int) -> None:
    """Refuse an output size the loaded family cannot render as asked. A size the pipeline would
    silently floor to its own grid is refused rather than changed behind the caller's back."""
    multiple = int(getattr(fam, "dimension_multiple", 16) or 16)
    max_side = int(getattr(fam, "max_output_side", 2048) or 2048)
    max_pixels = int(getattr(fam, "max_output_pixels", 2048 * 2048) or 2048 * 2048)
    name = getattr(fam, "name", "this")
    if min(width, height) < MIN_OUTPUT_SIDE:
        raise ValueError(
            f"Width and height must be at least {MIN_OUTPUT_SIDE}px (got {width}x{height})."
        )
    if width % multiple or height % multiple:
        raise ValueError(
            f"The {name} model needs width and height in multiples of {multiple} "
            f"(got {width}x{height})."
        )
    if max(width, height) > max_side:
        raise ValueError(
            f"The {name} model generates at most {max_side}px per side (got {width}x{height})."
        )
    if width * height > max_pixels:
        raise ValueError(
            f"The {name} model generates at most {max_pixels:,} pixels "
            f"(got {width}x{height} = {width * height:,})."
        )


def match_source_size(fam: Any, source_size: tuple[int, int], resolution: int) -> tuple[int, int]:
    """Output (width, height) with the source's aspect ratio at a ``resolution`` x ``resolution``
    area, on the family grid and inside its bounds. Same rounding as upstream's
    ``calculate_dimensions``, so an omitted size matches what the pipeline would pick from Image 1,
    never from the LAST reference as upstream does. A source too elongated to keep its ratio with
    both sides inside [MIN_OUTPUT_SIDE, max side] keeps the short side at the minimum and caps the
    long side, so the size is always one the request would accept."""
    multiple = int(getattr(fam, "dimension_multiple", 16) or 16)
    max_side = int(getattr(fam, "max_output_side", 2048) or 2048)
    max_pixels = int(getattr(fam, "max_output_pixels", 2048 * 2048) or 2048 * 2048)
    sw, sh = source_size
    ratio = max(1e-6, float(sw) / float(max(1, sh)))
    area = float(resolution) * float(resolution)
    # Shrink the target area until the rounded result fits both bounds.
    for _ in range(64):
        w = max(multiple, int(round(math.sqrt(area * ratio) / multiple)) * multiple)
        h = max(multiple, int(round(math.sqrt(area / ratio) / multiple)) * multiple)
        if max(w, h) <= max_side and w * h <= max_pixels:
            break
        area *= 0.9
    short_min = -(-MIN_OUTPUT_SIDE // multiple) * multiple
    long_max = max_side // multiple * multiple
    if min(w, h) < short_min:
        long_side = int(round(short_min * max(ratio, 1.0 / ratio) / multiple)) * multiple
        long_side = min(max(long_side, short_min), long_max)
        while long_side > short_min and long_side * short_min > max_pixels:
            long_side -= multiple
        w, h = (long_side, short_min) if ratio >= 1.0 else (short_min, long_side)
    return w, h


def effective_reference_resolution(fam: Any, requested: Optional[int]) -> Optional[int]:
    """The condition-image preprocessing resolution this call uses, or None for a family without
    the control. Refuses a value the family does not list rather than rounding it."""
    allowed = tuple(getattr(fam, "reference_resolutions", ()) or ())
    if not allowed:
        if requested is not None:
            raise ValueError(
                f"reference_resolution is not supported for the '{getattr(fam, 'name', '')}' "
                "model family."
            )
        return None
    if requested is None:
        return (
            DEFAULT_REFERENCE_RESOLUTION if DEFAULT_REFERENCE_RESOLUTION in allowed else allowed[0]
        )
    if requested not in allowed:
        raise ValueError(
            "reference_resolution must be one of "
            + ", ".join(str(v) for v in allowed)
            + f" for this model (got {requested})."
        )
    return int(requested)


def _decode_bounded(data: str, mode: str, budget: list[int], what: str) -> Any:
    from core.inference.diffusion import decode_b64_image

    try:
        img = decode_b64_image(data, mode = mode, max_pixels = max(1, budget[0]))
    except ValueError as exc:
        if "source pixels" in str(exc):
            raise ValueError(
                f"The input images are too large together: {what} exceeds the "
                f"{MAX_CONDITION_SOURCE_PIXELS:,} pixel total for one request. Use smaller images."
            ) from exc
        raise ValueError(f"{what}: {exc}") from exc
    budget[0] -= img.width * img.height
    return img


def _to_source_geometry(img: Any, source: Any, what: str, resample: Any) -> Any:
    """``img`` at the source's pixel size. The localized-edit layers are drawn against the source
    as displayed, so any size difference must be a pure scale; a different aspect ratio means the
    layer belongs to another image."""
    if img.size == source.size:
        return img
    sw, sh = source.size
    iw, ih = img.size
    if abs(iw / float(ih) - sw / float(sh)) > 0.02 * (sw / float(sh)):
        raise ValueError(
            f"The {what} is {iw}x{ih} but the source image is {sw}x{sh}; it must be drawn over "
            "the same image."
        )
    return img.resize(source.size, resample)


def apply_localized_edit(source: Any, localized: LocalizedEdit, budget: list[int]) -> list[Any]:
    """The images the localized edit contributes, source first: ``[marked_source]`` for annotate
    and paint, ``[source, mask]`` for a separate mask. Alpha outside the marks is preserved."""
    from PIL import Image

    if localized.mode == "annotate":
        overlay = _decode_bounded(localized.image, "RGBA", budget, "The annotation layer")
        overlay = _to_source_geometry(overlay, source, "annotation layer", Image.LANCZOS)
        base = source.convert("RGBA")
        marked = Image.alpha_composite(base, overlay)
        return [marked if source.mode == "RGBA" else marked.convert(source.mode)]
    if localized.mode == "paint":
        mask = _decode_bounded(localized.image, "L", budget, "The painted region")
        mask = _to_source_geometry(mask, source, "painted region", Image.LANCZOS)
        base = source.convert("RGBA")
        white = Image.new("RGBA", base.size, (255, 255, 255, 255))
        painted = Image.composite(white, base, mask)
        return [painted if source.mode == "RGBA" else painted.convert(source.mode)]
    if localized.mode == "mask":
        mask = _decode_bounded(localized.image, "L", budget, "The mask")
        # NEAREST keeps it binary; the threshold then fixes the polarity white = region.
        mask = _to_source_geometry(mask, source, "mask", Image.NEAREST)
        mask = mask.point(lambda v: 255 if v >= 128 else 0)
        if not mask.getbbox():
            raise ValueError("The mask is empty: paint the region to edit in white.")
        return [source, mask.convert("RGB")]
    raise ValueError("localized_edit mode must be one of " + ", ".join(LOCALIZED_EDIT_MODES) + ".")


def decode_condition_images(
    fam: Any,
    init_image: str,
    reference_images: Optional[Sequence[str]],
    localized: Optional[LocalizedEdit] = None,
) -> list[Any]:
    """Decode the ordered condition list for one reference or unified-edit call: the (possibly
    marked) source first, then a separate mask when there is one, then every reference in request
    order. Refuses more images than the family takes instead of dropping the tail."""
    limit = int(getattr(fam, "max_condition_images", 4) or 4)
    mode = getattr(fam, "condition_image_mode", "RGB") or "RGB"
    refs = [r for r in (reference_images or []) if r]
    extra_from_mask = 1 if localized is not None and localized.mode == "mask" else 0
    total = 1 + extra_from_mask + len(refs)
    if total > limit:
        raise ValueError(
            f"The {getattr(fam, 'name', 'loaded')} model takes at most {limit} input images in "
            f"total (got {total}" + (", counting the mask" if extra_from_mask else "") + ")."
        )
    budget = [MAX_CONDITION_SOURCE_PIXELS]
    source = _decode_bounded(init_image, mode, budget, "The source image")
    images = [source]
    if localized is not None:
        images = apply_localized_edit(source, localized, budget)
    for ref in refs:
        images.append(_decode_bounded(ref, mode, budget, f"Image {len(images) + 1}"))
    return images


def conditioning_capabilities(fam: Any, workflows: Sequence[str]) -> dict[str, Any]:
    """What the UI may offer for ``fam`` given the ``workflows`` the active engine runs for it.
    Localized editing is advertised only where the unified edit workflow itself is."""
    unified = bool(getattr(fam, "unified_edit", False)) and "edit" in workflows
    return {
        "max_condition_images": int(getattr(fam, "max_condition_images", 4) or 4),
        "alpha": (getattr(fam, "condition_image_mode", "RGB") or "RGB") == "RGBA",
        "dimension_multiple": int(getattr(fam, "dimension_multiple", 16) or 16),
        "max_output_side": int(getattr(fam, "max_output_side", 2048) or 2048),
        "max_output_pixels": int(getattr(fam, "max_output_pixels", 2048 * 2048) or 2048 * 2048),
        "reference_resolutions": list(getattr(fam, "reference_resolutions", ()) or ()),
        "unified_edit": unified,
        "localized_edit_modes": list(LOCALIZED_EDIT_MODES) if unified else [],
    }
