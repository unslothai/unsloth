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

# Instrumentation counters for operational visibility.
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
        with open("/proc/version", "r", encoding = "utf-8") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


_IS_WSL: bool = _is_wsl()


def normalize_path(path: str) -> str:
    """Normalize filesystem paths for cross-platform use.

    WSL maps drive-letter paths to ``/mnt/<drive>/...``; native Windows keeps
    the drive and normalizes separators; elsewhere slashes are forward-only.
    """
    if not path:
        return path

    # Handle Windows drive letters (C:\\ or c:\\)
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        # Map to /mnt/<drive>/ only under WSL; native Windows keeps the drive letter.
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

    # Exists on disk → local (covers relative paths like "outputs/foo").
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

    # Check for model files
    for suffix in [".safetensors", ".bin", ".json"]:
        if list(cache_path.rglob(f"*{suffix}")):
            return True

    return False


def _hf_hub_cache_dir() -> Path:
    """Return HF cache root honoring HF_HUB_CACHE when available."""
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


def resolve_cached_repo_id_case(model_name: str, use_memo: bool = True) -> str:
    """Resolve repo_id to the exact casing already present in local HF cache.

    Policy: prefer the requested/canonical repo_id, but reuse a case-variant's
    exact cached spelling if one already exists in local HF cache. Avoids
    duplicate downloads while preserving user intent where possible.
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

    # Exact-case path first so a new exact match beats a memoized variant.
    exact_path = cache_dir / expected_dir
    if exact_path.is_dir():
        if use_memo:
            _CACHE_CASE_RESOLUTION_MEMO[model_name] = model_name
        _CACHE_CASE_RESOLUTION_STATS["exact_hits"] += 1
        return model_name

    # Revalidate memoized entries on disk to avoid stale results.
    if use_memo:
        cached = _CACHE_CASE_RESOLUTION_MEMO.get(model_name)
        if cached is not None:
            cached_path = cache_dir / f"models--{cached.replace('/', '--')}"
            if cached_path.is_dir():
                _CACHE_CASE_RESOLUTION_STATS["memo_hits"] += 1
                return cached
            # Stale entry -- drop it and re-scan below
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
            # Deterministic tie-break if multiple case variants coexist
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


def _wsl_reveal_in_explorer(path: Path, is_file: bool) -> bool:
    import subprocess
    if not _IS_WSL:
        return False
    try:
        windows_path = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            check = True,
            timeout = 10,
        ).stdout.strip()
        if not windows_path:
            return False
        argument = f"/select,{windows_path}" if is_file else windows_path
        subprocess.Popen(["explorer.exe", argument])
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def reveal_in_file_manager(path: Path, expect_dir: bool = False) -> None:
    """Open the OS file manager with *path* selected (best effort per platform).

    Raises ``FileNotFoundError`` when the target is already gone: the Linux
    branch falls back to the parent directory, so a path deleted between the
    caller's check and this call would open the directory holding it, which for
    a sandbox is the root holding every other chat's.

    ``expect_dir`` says the caller only ever reveals a directory, and refuses
    everything else: a target swapped for a regular file takes the file branch
    on every platform, and that branch names the enclosing directory, which for
    a sandbox is the root holding every other chat's. A symlink is refused too,
    and that is the whole reason this is one ``lstat`` rather than a couple of
    ``Path`` predicates. ``is_dir()`` follows links, so a sandbox swapped for a
    link to somewhere else passes it and the launcher opens the target; asking
    for the link's OWN type answers both questions in a single syscall, leaving
    no window between them. (``is_dir(follow_symlinks = False)`` would say the
    same thing but only on 3.13+, and this runs on 3.10.)

    Off by default: the cached-model reveal legitimately points at a file, and
    at a symlinked one, since a Hugging Face cache snapshot is a link farm.
    """
    import stat as stat_module
    import subprocess

    if expect_dir:
        try:
            # No-follow, and the only stat this branch takes.
            entry = os.lstat(path)
        except OSError as exc:
            raise FileNotFoundError(str(path)) from exc
        if not stat_module.S_ISDIR(entry.st_mode):
            raise FileNotFoundError(str(path))
        is_dir, is_file = True, False
    else:
        if not path.exists():
            raise FileNotFoundError(str(path))
        # Decided ONCE and then only read. Each branch used to re-stat, so a
        # swap landing between two of them could still take a path the guard
        # above had already ruled out.
        is_dir = path.is_dir()
        is_file = not is_dir and path.is_file()
        if not is_dir and not is_file:
            # Neither, so nothing to show and no parent worth widening to.
            raise FileNotFoundError(str(path))

    target = str(path)
    if sys.platform == "darwin":
        cmd = ["open", "-R", target] if is_file else ["open", target]
        subprocess.Popen(cmd)
    elif os.name == "nt":
        if is_file:
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            os.startfile(target)  # noqa: S606 - local user's own file manager
    elif not _wsl_reveal_in_explorer(path, is_file):
        # No cross-desktop "select file" standard on Linux; open the directory.
        subprocess.Popen(["xdg-open", str(path.parent) if is_file else target])
