# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any


def exif_upright(image: Any) -> Any:
    """*image* turned per its EXIF orientation, or *image* itself when nothing turns it."""
    from PIL import Image

    # Not getexif(): it also recovers orientation from XMP/ImageMagick profiles Chromium ignores.
    exif = Image.Exif()
    try:
        if image.info.get("exif"):
            exif.load(image.info["exif"])
    except Exception:  # noqa: BLE001 - a malformed block leaves it unrotated, as the preview
        return image
    # Chromium ignores WebP orientation but WebKit applies it: known macOS desktop divergence.
    # TIFF needs no branch: its plugin applies orientation during load().
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(None if image.format == "WEBP" else exif.get(0x0112))
    if method is None:
        return image
    upright = image.transpose(method)
    # Drop every source so a re-save cannot re-apply it.
    for consumed in ("exif", "XML:com.adobe.xmp", "xmp", "Raw profile type exif"):
        upright.info.pop(consumed, None)
    return upright
