# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clean up the Unsloth compiled cache directory.

unsloth_compiled_cache (created by unsloth_zoo/compiler.py during
FastModel.from_pretrained) holds model-type-specific compiled files. Clear it
selectively between model loads, preserving model-agnostic components (Trainers)
that spawned subprocesses need.
"""

import shutil
import structlog
from loggers import get_logger
from pathlib import Path
from typing import List, Optional

logger = get_logger(__name__)

# Possible locations where unsloth_compiled_cache may appear
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_PROJECT_ROOT = _BACKEND_DIR.parent.parent  # repo root

_CACHE_DIRS = [
    _BACKEND_DIR / "unsloth_compiled_cache",
    _PROJECT_ROOT / "unsloth_compiled_cache",
    _PROJECT_ROOT / "studio" / "tmp" / "unsloth_compiled_cache",
]


def _configured_cache_dirs() -> List[Path]:
    """Cache dirs outside the source tree: the configured one, and the CWD.

    The candidates above are all source-tree relative, so a cache created in
    the launcher's CWD (the user profile on Windows) was invisible to cleanup.
    The CWD is still checked for installs that predate the pinned location.
    """
    import os

    dirs: List[Path] = []
    configured = (os.environ.get("UNSLOTH_COMPILE_LOCATION") or "").strip()
    if configured:
        dirs.append(Path(configured).expanduser())
    try:
        dirs.append(Path.cwd() / "unsloth_compiled_cache")
    except OSError:
        pass
    return dirs


def get_existing_cache_dirs() -> List[Path]:
    """Return known compiled-cache directories that currently exist on disk."""
    seen: set = set()
    found: List[Path] = []
    for candidate in [*_CACHE_DIRS, *_configured_cache_dirs()]:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            found.append(candidate)
    return found


# What a compiled cache is made of. Anything else means the configured path is
# a directory the user also keeps other things in.
_CACHE_ENTRY_SUFFIXES = (".py", ".pyc", ".pyi")
_CACHE_ENTRY_NAMES = ("__pycache__", ".locks", "unsloth_compiled_cache")


def _is_cache_shaped(path: Path) -> bool:
    """True when every entry looks like something the compiler wrote."""
    try:
        for item in path.iterdir():
            if item.name in _CACHE_ENTRY_NAMES:
                continue
            if item.is_file() and item.suffix in _CACHE_ENTRY_SUFFIXES:
                continue
            return False
    except OSError:
        return False
    return True


def _cleanable_cache_dirs() -> List[Path]:
    """Cache dirs safe to delete from.

    UNSLOTH_COMPILE_LOCATION is a user-set variable, so it can name a directory
    that holds other things (`$HOME/.cache`). The built-in paths are ours by
    construction; a configured one has to look like a cache before anything here
    deletes from it.
    """
    builtin = {str(p) for p in _CACHE_DIRS}
    cleanable: List[Path] = []
    for cache_dir in get_existing_cache_dirs():
        if str(cache_dir) in builtin or _is_cache_shaped(cache_dir):
            cleanable.append(cache_dir)
        else:
            logger.warning(
                "Not clearing %s: it holds files the compiler did not write. Point "
                "UNSLOTH_COMPILE_LOCATION at a directory used only for the compiled cache.",
                cache_dir,
            )
    return cleanable


def register_compiled_cache_on_path() -> None:
    """Add all existing compiled-cache directories to sys.path and PYTHONPATH.

    Ensures spawned workers (on 'spawn'-start platforms, i.e. Windows and macOS)
    can import dynamically compiled modules such as UnslothSFTTrainer.
    """
    import os
    import sys

    pypath = os.environ.get("PYTHONPATH", "")
    pypath_entries = [p for p in pypath.split(os.pathsep) if p]

    # Iterate in reverse so earlier _CACHE_DIRS entries (higher priority) are
    # inserted last and thus end up first in sys.path / PYTHONPATH.
    for cache_dir in reversed(get_existing_cache_dirs()):
        resolved = str(cache_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        if resolved not in pypath_entries:
            pypath_entries.insert(0, resolved)

    os.environ["PYTHONPATH"] = os.pathsep.join(pypath_entries)


def clear_unsloth_compiled_cache(preserve_patterns: Optional[List[str]] = None) -> None:
    """
    Remove compiled files from the cache directory (idempotent).

    Args:
        preserve_patterns: glob patterns for files to keep
                           (e.g., ["Unsloth*Trainer.py"]). If None or empty,
                           the entire cache directory is deleted (legacy behavior).
    """
    for cache_dir in _cleanable_cache_dirs():
        if preserve_patterns:
            logger.info(
                f"Cleaning unsloth compiled cache (preserving {preserve_patterns}): " f"{cache_dir}"
            )

            for item in cache_dir.iterdir():
                if item.is_file():
                    preserve = any(item.match(pattern) for pattern in preserve_patterns)
                    if not preserve:
                        try:
                            item.unlink()
                        except OSError as e:
                            logger.debug(f"Could not delete {item}: {e}")

                elif item.is_dir():
                    # Always clear __pycache__ and other subdirectories
                    shutil.rmtree(item, ignore_errors = True)
        else:
            # Legacy: remove the entire directory
            logger.info(f"Removing unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors = True)
