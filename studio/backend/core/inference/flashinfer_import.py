# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import flashinfer without letting it downgrade Triton's Blackwell ptxas.

``flashinfer/triton/__init__.py`` (reached by ``import flashinfer`` from 0.7.x through ``flashinfer.mamba``) sets
``TRITON_PTXAS_BLACKWELL_PATH`` to the ``ptxas`` on PATH whenever that one is CUDA >= 13, even when the
``ptxas-blackwell`` Triton bundles is newer (Triton 3.7 bundles 13.1, 3.8 bundles 13.3). The variable is process
wide, so every later Triton compile on sm_100 / sm_120 would assemble with the older toolkit. flashinfer PR #10079
makes it only take a strictly newer ptxas; until a release with it is the one installed, ``import_flashinfer`` undoes
the override when it is not an upgrade. A value set before the import (by the user) is never touched."""

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
    """``(major, minor)`` from ``ptxas --version`` output (``Cuda compilation tools, release 13.0, V13.0.88``)."""
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
    """The release of Triton's own ``ptxas-blackwell`` (what it uses with the variable unset), or None."""
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
    """Undo a ``TRITON_PTXAS_BLACKWELL_PATH`` an import just set if it is not newer than Triton's bundled ptxas.
    ``before``: the value before the import. True when it restored."""
    global _LOGGED
    if before:
        return (
            False  # set before the import (the user's, or flashinfer's own guard skipped): leave it
        )
    after = os.environ.get(PTXAS_BLACKWELL_ENV)
    if not after:
        return False
    system = ptxas_release(after)
    bundled = bundled_blackwell_release()
    if system is None or bundled is None or system > bundled:
        return False  # an upgrade, or nothing to compare against: flashinfer's choice stands
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
    """``import flashinfer`` (raises what it raises), then ``restore_triton_ptxas``. Already imported: returned."""
    module = sys.modules.get("flashinfer")
    if module is not None:
        return module
    before = os.environ.get(PTXAS_BLACKWELL_ENV)
    try:
        import flashinfer as module
    finally:
        # Also on a failed import: flashinfer.triton may have run before the part that raised.
        try:
            restore_triton_ptxas(before)
        except Exception:  # noqa: BLE001 - never turn the guard into an import failure
            pass
    return module
