# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Utility for cleaning up the Unsloth compiled cache directory.

The unsloth_compiled_cache is created by unsloth_zoo/compiler.py during
FastModel.from_pretrained() and contains model-type-specific compiled Python
files. It should be selectively cleared between model loads to avoid stale
artefacts, while preserving model-agnostic components (like Trainers) needed
by spawned subprocesses.
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


def get_existing_cache_dirs() -> List[Path]:
    """Return known compiled-cache directories that currently exist on disk."""
    return [d for d in _CACHE_DIRS if d.exists()]


def register_compiled_cache_on_path() -> None:
    """Add all existing compiled-cache directories to sys.path and PYTHONPATH.

    This ensures spawned workers (on platforms using the 'spawn' start method,
    i.e. Windows and macOS) can import dynamically compiled modules such as
    UnslothSFTTrainer.
    """
    import os
    import sys

    pypath = os.environ.get("PYTHONPATH", "")
    pypath_entries = [p for p in pypath.split(os.pathsep) if p]

    # Iterate in reverse so that earlier _CACHE_DIRS entries (higher priority)
    # are inserted last and therefore end up first in sys.path / PYTHONPATH.
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
        preserve_patterns: A list of glob patterns for files to keep
                           (e.g., ["Unsloth*Trainer.py"]). If None or empty,
                           the entire cache directory is deleted (legacy behavior).
    """
    for cache_dir in _CACHE_DIRS:
        if not cache_dir.exists():
            continue

        if preserve_patterns:
            logger.info(
                f"Cleaning unsloth compiled cache (preserving {preserve_patterns}): "
                f"{cache_dir}"
            )

            for item in cache_dir.iterdir():
                if item.is_file():
                    # Check if the file matches any of the patterns we want to keep
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
            # Legacy behavior: nuke the entire directory
            logger.info(f"Removing unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors = True)
