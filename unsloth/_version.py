from __future__ import annotations

from ._build_support import compute_version_string

_BASE_VERSION = "2025.11.6"

try:
    __version__ = compute_version_string(_BASE_VERSION)
except Exception as exc:  # pragma: no cover - fallback if detection fails
    print(f"Unsloth: Falling back to base version {_BASE_VERSION}: {exc}")
    __version__ = _BASE_VERSION

__all__ = ["__version__"]
