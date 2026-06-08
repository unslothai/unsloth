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

# Per-process cache to avoid repeated cache-dir scans for the same identifier.
_CACHE_CASE_RESOLUTION_MEMO: dict[str, str] = {}

# Lightweight instrumentation counters for operational visibility.
_CACHE_CASE_RESOLUTION_STATS: dict[str, int] = {
    "calls": 0,
    "memo_hits": 0,
    "exact_hits": 0,
    "variant_hits": 0,
    "tie_breaks": 0,
    "fallbacks": 0,
    "errors": 0,
}


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
    cache_dir = _hf_hub_cache_dir()
    resolved_name = resolve_cached_repo_id_case(model_name)
    model_cache_name = resolved_name.replace("/", "--")
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


def _hf_hub_cache_dir() -> Path:
    """Return HF cache root honoring HF_HUB_CACHE when available."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception as exc:
        logger.debug(
            "Could not read huggingface_hub HF_HUB_CACHE, using default hub path: %s",
            exc,
        )
        return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_repo_id_case(model_name: str, use_memo: bool = True) -> str:
    """Resolve repo_id to the exact casing already present in local HF cache.

    Policy: prefer the requested/canonical repo_id, but if a case-variant already
    exists in local HF cache, reuse that exact cached spelling. This avoids
    duplicate downloads while preserving user intent whenever possible.
    """
    _CACHE_CASE_RESOLUTION_STATS["calls"] += 1

    if not model_name or "/" not in model_name:
        _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
        return model_name

    cache_dir = _hf_hub_cache_dir()
    if not cache_dir.exists():
        _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
        return model_name

    expected_dir = f"models--{model_name.replace('/', '--')}"

    # Always check the exact-case path first so a newly-appeared exact match
    # wins over any previously memoized variant.
    exact_path = cache_dir / expected_dir
    if exact_path.is_dir():
        if use_memo:
            _CACHE_CASE_RESOLUTION_MEMO[model_name] = model_name
        _CACHE_CASE_RESOLUTION_STATS["exact_hits"] += 1
        return model_name

    # Validate memoized entries still exist on disk before returning them.
    # This prevents stale results when cache dirs are deleted/recreated.
    if use_memo:
        cached = _CACHE_CASE_RESOLUTION_MEMO.get(model_name)
        if cached is not None:
            cached_path = cache_dir / f"models--{cached.replace('/', '--')}"
            if cached_path.is_dir():
                _CACHE_CASE_RESOLUTION_STATS["memo_hits"] += 1
                return cached
            # Stale entry -- drop it and re-scan below.
            _CACHE_CASE_RESOLUTION_MEMO.pop(model_name, None)

    expected_lower = expected_dir.lower()
    try:
        candidates: list[str] = []
        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.lower() != expected_lower:
                continue
            if not entry.name.startswith("models--"):
                continue
            repo_part = entry.name[len("models--") :]
            if not repo_part:
                continue
            candidates.append(repo_part.replace("--", "/"))

        if candidates:
            # Deterministic tie-break if multiple case variants coexist.
            resolved = sorted(candidates)[0]
            if len(candidates) > 1:
                _CACHE_CASE_RESOLUTION_STATS["tie_breaks"] += 1
            _CACHE_CASE_RESOLUTION_STATS["variant_hits"] += 1
            if use_memo:
                _CACHE_CASE_RESOLUTION_MEMO[model_name] = resolved
            return resolved
    except Exception as exc:
        _CACHE_CASE_RESOLUTION_STATS["errors"] += 1
        logger.debug(f"Could not resolve cached repo_id case for '{model_name}': {exc}")

    _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
    return model_name


def get_cache_case_resolution_stats() -> dict[str, int]:
    """Return a copy of case-resolution instrumentation counters."""
    return dict(_CACHE_CASE_RESOLUTION_STATS)


def reset_cache_case_resolution_state() -> None:
    """Clear resolver memo and counters (primarily for tests)."""
    _CACHE_CASE_RESOLUTION_MEMO.clear()
    for key in _CACHE_CASE_RESOLUTION_STATS:
        _CACHE_CASE_RESOLUTION_STATS[key] = 0
