# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
import io
import math
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
        # Row cells are user data: a managed account may only preview files it could read itself.
        from fastapi import HTTPException
        from core.training.account_jobs import account_path

        try:
            account_path(path_value)
        except (HTTPException, OSError, RuntimeError, ValueError, TypeError):
            return None
        try:
            from PIL import Image  # type: ignore
            with Image.open(Path(path_value)) as image:
                return image.copy()
        except (OSError, ValueError, TypeError):
            return None

    return None


# Resolved once: importing pandas per value cost 30%. Sentinel, since None is a real value.
_NO_SENTINEL = object()
_PANDAS_NA: Any = _NO_SENTINEL
_PANDAS_NAT: Any = _NO_SENTINEL
_PANDAS_SENTINELS_READY = False


def _is_pandas_missing(value: Any) -> bool:
    """Identity check, not pd.isna (element-wise on arrays); pandas' sentinels are singletons."""
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
            value = value.item()
            if not isinstance(value, float):
                return value

    # pandas' missing number is NaN; Starlette refuses NaN/inf, so one blank cell 500'd the page.
    if isinstance(value, float):
        return value if math.isfinite(value) else None

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
    """Per column: a row with bytes or path keys looks like one image and replaces every column."""
    if isinstance(row, list):
        return [to_preview_jsonable_row(item) for item in row]
    if not isinstance(row, dict):
        return to_preview_jsonable(row)
    return {str(key): to_preview_jsonable(value) for key, value in row.items()}


_IMAGE_THUMBNAIL_SIZE = (384, 384)
_GENERATED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def image_column_names(recipe: Any) -> list[str]:
    """Only these columns' cells are read as generated images."""
    columns = recipe.get("columns") if isinstance(recipe, dict) else None
    return [
        column["name"]
        for column in (columns if isinstance(columns, list) else [])
        if isinstance(column, dict)
        and column.get("column_type") == "image"
        and isinstance(column.get("name"), str)
    ]


def _open_generated_image(item: str, dataset_dir: Path | None) -> Any | None:
    relative = Path(item)
    if (
        dataset_dir is not None
        and relative.suffix.lower() in _GENERATED_IMAGE_SUFFIXES
        and not relative.is_absolute()
        and not relative.drive
    ):
        # A full run's images/<column>/<file>; nothing outside images/ is read.
        root = dataset_dir.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root / "images"):
            return None
        from fastapi import HTTPException
        from core.training.account_jobs import account_path

        try:
            account_path(str(target))
            raw_bytes = target.read_bytes()
        except (HTTPException, OSError, RuntimeError, ValueError, TypeError):
            return None
    else:
        try:
            raw_bytes = base64.b64decode(item, validate = True)
        except (binascii.Error, ValueError):
            return None
    try:
        return _open_pil_image_from_bytes(raw_bytes)
    except Exception:  # A bomb, truncated file or non-image stays as text.
        return None


def image_cell_to_preview(value: Any, *, dataset_dir: Path | None = None) -> Any:
    """A generated image cell (base64 strings or images/ paths) as one thumbnail; unreadable cells pass through."""
    items = value if isinstance(value, list) else [value]
    items = [item for item in items if isinstance(item, str) and item]
    if not items:
        return value
    image = _open_generated_image(items[0], dataset_dir)
    if image is None:
        return value
    image.thumbnail(_IMAGE_THUMBNAIL_SIZE)
    payload = _pil_to_preview_payload(image)
    if len(items) > 1:
        payload["count"] = len(items)
    return payload


def render_image_columns(
    rows: list[Any],
    columns: list[str],
    *,
    dataset_dir: Path | None = None,
) -> list[Any]:
    """In place."""
    if not columns:
        return rows
    for row in rows:
        if not isinstance(row, dict):
            continue
        for column in columns:
            if column in row:
                row[column] = image_cell_to_preview(row[column], dataset_dir = dataset_dir)
    return rows
