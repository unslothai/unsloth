from __future__ import annotations

import base64
import io
from typing import Any


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
        return None

    buffer = io.BytesIO()
    value.convert("RGB").save(buffer, format="JPEG", quality=85)
    return {
        "type": "image",
        "mime": "image/jpeg",
        "width": value.width,
        "height": value.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


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
