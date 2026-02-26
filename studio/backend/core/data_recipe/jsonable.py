from __future__ import annotations

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
