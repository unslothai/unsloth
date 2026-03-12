# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any


def _pil_to_preview_payload(image: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format = "JPEG", quality = 85)
    return {
        "type": "image",
        "mime": "image/jpeg",
        "width": image.width,
        "height": image.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _open_pil_image_from_bytes(raw_bytes: bytes):
    from PIL import Image  # type: ignore

    with Image.open(io.BytesIO(raw_bytes)) as image:
        return image.copy()


def _to_pil_from_hf_image_dict(value: Any) -> Any | None:
    if not isinstance(value, dict):
        return None

    raw_bytes = value.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) > 0:
        try:
            return _open_pil_image_from_bytes(bytes(raw_bytes))
        except (OSError, ValueError):
            pass
    if (
        isinstance(raw_bytes, list)
        and len(raw_bytes) > 0
        and all(isinstance(item, int) and 0 <= item <= 255 for item in raw_bytes)
    ):
        try:
            return _open_pil_image_from_bytes(bytes(raw_bytes))
        except (OSError, ValueError):
            pass

    path_value = value.get("path")
    if isinstance(path_value, str) and path_value.strip():
        try:
            from PIL import Image  # type: ignore

            with Image.open(Path(path_value)) as image:
                return image.copy()
        except (OSError, ValueError, TypeError):
            return None

    return None


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas-ish values into plain JSON-safe values."""
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            return value

    return value


def _to_preview_image_payload(value: Any) -> dict[str, Any] | None:
    try:
        from PIL.Image import Image as PILImage  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    if not isinstance(value, PILImage):
        hf_image = _to_pil_from_hf_image_dict(value)
        if hf_image is None:
            return None
        value = hf_image

    return _pil_to_preview_payload(value)


def to_preview_jsonable(value: Any) -> Any:
    """Convert values into JSON-safe preview values, including PIL images."""
    image_payload = _to_preview_image_payload(value)
    if image_payload is not None:
        return image_payload

    converted = to_jsonable(value)
    if converted is None or isinstance(converted, (str, int, float, bool)):
        return converted
    if isinstance(converted, dict):
        return {str(k): to_preview_jsonable(v) for k, v in converted.items()}
    if isinstance(converted, (list, tuple, set)):
        return [to_preview_jsonable(v) for v in converted]
    if isinstance(converted, (bytes, bytearray)):
        return base64.b64encode(bytes(converted)).decode("ascii")
    return str(converted)
