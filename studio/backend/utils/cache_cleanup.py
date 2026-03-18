# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Utility for cleaning up the Unsloth compiled cache directory.

The unsloth_compiled_cache is created by unsloth_zoo/compiler.py during
FastModel.from_pretrained() and contains model-type-specific compiled Python
files. It should be cleared between model loads to avoid stale artefacts.
"""

import shutil
from loggers import get_logger
from pathlib import Path

logger = get_logger(__name__)

# Possible locations where unsloth_compiled_cache may appear
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_PROJECT_ROOT = _BACKEND_DIR.parent.parent  # repo root

_CACHE_DIRS = [
    _BACKEND_DIR / "unsloth_compiled_cache",
    _PROJECT_ROOT / "unsloth_compiled_cache",
    _PROJECT_ROOT / "studio" / "tmp" / "unsloth_compiled_cache",
]


def clear_unsloth_compiled_cache() -> None:
    """Remove every known unsloth_compiled_cache directory (idempotent)."""
    for cache_dir in _CACHE_DIRS:
        if cache_dir.exists():
            logger.info(f"Removing unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors = True)
