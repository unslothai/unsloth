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
                    shutil.rmtree(item, ignore_errors=True)
        else:
            # Legacy behavior: nuke the entire directory
            logger.info(f"Removing unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)