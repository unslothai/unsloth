# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
from decimal import Decimal
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


# Resolved once: to_jsonable runs per value, and importing pandas per value cost 30% of it. The
# placeholder is a private object rather than None, which a real value can be.
_NO_SENTINEL = object()
_PANDAS_NA: Any = _NO_SENTINEL
_PANDAS_NAT: Any = _NO_SENTINEL
_PANDAS_SENTINELS_READY = False


def _is_pandas_missing(value: Any) -> bool:
    """pandas' own missing sentinels. Identity rather than ``pd.isna``, which answers element-wise
    for a list or an array; these two are singletons."""
    global _PANDAS_NA, _PANDAS_NAT, _PANDAS_SENTINELS_READY
    if not _PANDAS_SENTINELS_READY:
        try:
            import pandas as pd  # type: ignore
            _PANDAS_NA, _PANDAS_NAT = pd.NA, pd.NaT
        except ImportError:  # pragma: no cover
            _PANDAS_NA = _PANDAS_NAT = _NO_SENTINEL
        _PANDAS_SENTINELS_READY = True
    return value is _PANDAS_NA or value is _PANDAS_NAT


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas-ish values into plain JSON-safe values."""
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover
        np = None  # type: ignore

    # Ahead of everything below: NaT isoformat()s to "NaT" and NA hits the str() fallback.
    if _is_pandas_missing(value):
        return None

    # DuckDB hands a DECIMAL back as a float and pyarrow as a Decimal: 1.2 against "1.20".
    if isinstance(value, Decimal):
        return float(value)

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


def to_preview_jsonable_row(row: Any) -> Any:
    """A dataset row, converted a column at a time.

    ``to_preview_jsonable`` answers about a VALUE, and its Hugging Face image detection matches any
    mapping carrying ``bytes`` or ``path`` -- which a row can be. Handing it a whole row replaced
    every column, labels included, with one JPEG preview payload."""
    if isinstance(row, list):
        return [to_preview_jsonable_row(item) for item in row]
    if not isinstance(row, dict):
        return to_preview_jsonable(row)
    return {str(key): to_preview_jsonable(value) for key, value in row.items()}
