# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Path utilities for model and dataset handling
"""

import os
import sys
from pathlib import Path
from typing import Optional
import structlog
from loggers import get_logger

logger = get_logger(__name__)


def _is_wsl() -> bool:
    """Detect if we are running inside WSL (Windows Subsystem for Linux)."""
    if sys.platform == "win32":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


_IS_WSL: bool = _is_wsl()


def normalize_path(path: str) -> str:
    """
    Normalize filesystem paths for cross-platform use.

    On WSL, converts Windows drive-letter paths to ``/mnt/<drive>/...``.
    On native Windows, keeps the drive letter and normalizes separators.
    On Linux/macOS (non-WSL), paths are returned with forward slashes.

    Examples (WSL):
        C:\\Users\\... -> /mnt/c/Users/...
    Examples (native Windows):
        C:\\Users\\... -> C:/Users/...
    Examples (Linux/macOS):
        /home/user/... -> /home/user/... (unchanged)
    """
    if not path:
        return path

    # Handle Windows drive letters (C:\\ or c:\\)
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        # Only map to /mnt/<drive>/ when running under WSL;
        # on native Windows the drive letter must be preserved.
        if _IS_WSL:
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return path.replace("\\", "/")

    # Already Unix-style or relative
    return path.replace("\\", "/")


def is_local_path(path: str) -> bool:
    """
    Check if path is a local filesystem path vs HuggingFace model identifier.

    Examples:
        True: /home/user/model, C:\\models, ./model, ~/model
        False: unsloth/llama-3.1-8b, microsoft/phi-2
    """
    if not path:
        return False

    # If it exists on disk, treat as local (covers relative paths like "outputs/foo").
    try:
        if Path(normalize_path(path)).expanduser().exists():
            return True
    except Exception:
        pass

    # Obvious HF patterns
    if path.count("/") == 1 and not path.startswith(("/", ".", "~")):
        return False  # Looks like org/model format

    # Filesystem indicators
    return (
        path.startswith(("/", ".", "~"))  # Unix absolute/relative
        or ":" in path  # Windows drive or URL
        or "\\" in path  # Windows separator
        or os.path.isabs(path)  # System-absolute
    )


def get_cache_path(model_name: str) -> Optional[Path]:
    """Get HuggingFace cache path for a model if it exists."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = model_name.replace("/", "--")
    model_cache_path = cache_dir / f"models--{model_cache_name}"

    return model_cache_path if model_cache_path.exists() else None


def is_model_cached(model_name: str) -> bool:
    """Check if model is downloaded in HuggingFace cache."""
    cache_path = get_cache_path(model_name)
    if not cache_path:
        return False

    # Check for actual model files
    for suffix in [".safetensors", ".bin", ".json"]:
        if list(cache_path.rglob(f"*{suffix}")):
            return True

    return False
