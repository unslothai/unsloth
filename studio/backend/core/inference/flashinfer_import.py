# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import flashinfer, undoing its process-wide TRITON_PTXAS_BLACKWELL_PATH override (CUDA >= 13 ptxas on PATH)
unless newer than Triton's bundled ptxas-blackwell (fixed upstream in flashinfer #10079); a preset value is kept."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

PTXAS_BLACKWELL_ENV = "TRITON_PTXAS_BLACKWELL_PATH"
_LOGGED = False


def parse_ptxas_release(text: str) -> Optional[tuple]:
    match = re.search(r"release (\d+)\.(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def ptxas_release(path: str) -> Optional[tuple]:
    try:
        out = subprocess.run(
            [path, "--version"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            timeout = 30,
            check = True,
        ).stdout.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - unreadable ptxas: no verdict
        return None
    return parse_ptxas_release(out)


def bundled_blackwell_release() -> Optional[tuple]:
    saved = os.environ.pop(PTXAS_BLACKWELL_ENV, None)
    try:
        from triton.backends.nvidia.compiler import get_ptxas
        version = str(get_ptxas(100).version)
    except Exception:  # noqa: BLE001 - old Triton (no per-arch ptxas) or none: no verdict
        return None
    finally:
        if saved is not None:
            os.environ[PTXAS_BLACKWELL_ENV] = saved
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def restore_triton_ptxas(before: Optional[str]) -> bool:
    """Undo the variable an import just set unless newer than Triton's bundled ptxas. True when restored."""
    global _LOGGED
    if before:
        return False
    after = os.environ.get(PTXAS_BLACKWELL_ENV)
    if not after:
        return False
    system = ptxas_release(after)
    bundled = bundled_blackwell_release()
    if system is None or bundled is None or system > bundled:
        return False
    if before is None:
        os.environ.pop(PTXAS_BLACKWELL_ENV, None)
    else:
        os.environ[PTXAS_BLACKWELL_ENV] = before
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "flashinfer pointed %s at %s (CUDA %d.%d), older than or equal to Triton's bundled ptxas-blackwell "
            "(CUDA %d.%d); keeping Triton's own. Set %s yourself to choose a ptxas.",
            PTXAS_BLACKWELL_ENV,
            after,
            *system,
            *bundled,
            PTXAS_BLACKWELL_ENV,
        )
    return True


def import_flashinfer() -> Any:
    module = sys.modules.get("flashinfer")
    if module is not None:
        return module
    before = os.environ.get(PTXAS_BLACKWELL_ENV)
    try:
        import flashinfer as module
    finally:
        # Also on a failed import: flashinfer.triton may already have run.
        try:
            restore_triton_ptxas(before)
        except Exception:  # noqa: BLE001 - never turn the guard into an import failure
            pass
    return module
