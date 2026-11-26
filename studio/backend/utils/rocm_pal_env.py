# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm (PAL) runtime defaults that must be in the environment before HIP initialises.

On Windows the HIP runtime runs on PAL, which issues device-to-host and host-to-device copies on the SDMA engine.
A copy enqueued after ``hipEventRecord`` on the same stream can then run before a kernel enqueued earlier on that
stream when the kernel reaches its buffer through a by-value struct argument, which is how PyTorch's elementwise and
reduction kernels receive their pointers (ROCm/rocm-systems#12571, fix pending in #12952). The result is silent: a
blocking ``tensor.cpu()`` returns the pre-kernel bytes, and an upload into a freed block is overwritten by the
earlier kernel. On a gfx1151 runner with torch 2.11.0+rocm7.14.0 (HIP 7.14.60850) the sequence was wrong in 100 of
100 reads and 100 of 100 uploads, and clean in 0 of 200 with ``PAL_DISABLE_SDMA=1``, which routes the copies onto
the compute queue behind a barrier. The same setting measured inside the A/A spread on a Qwen-Image-2.1 render.

The value is a default: an explicit ``PAL_DISABLE_SDMA`` (including ``0``) is left alone.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, MutableMapping, Optional

PAL_DISABLE_SDMA_ENV = "PAL_DISABLE_SDMA"


def _installed_torch_version() -> Optional[str]:
    """The installed torch distribution's version string, read from metadata (torch is not imported)."""
    try:
        from importlib.metadata import version
        return version("torch")
    except Exception:  # noqa: BLE001 - no torch, or unreadable metadata
        return None


def _is_rocm_torch(torch_version: Optional[str]) -> bool:
    return bool(torch_version) and "rocm" in str(torch_version).lower()


def apply_windows_rocm_pal_defaults(
    environ: Optional[MutableMapping[str, str]] = None,
    *,
    platform: Optional[str] = None,
    torch_version: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[str]:
    """Set ``PAL_DISABLE_SDMA=1`` for a Windows process that runs a ROCm build of torch, unless already set.

    Returns the name of the variable it set, else None. Must run before torch (or anything else) initialises HIP;
    spawned children inherit it through the environment. Never raises.
    """
    env = os.environ if environ is None else environ
    try:
        if (platform or sys.platform) != "win32":
            return None
        if PAL_DISABLE_SDMA_ENV in env:
            return None
        if not _is_rocm_torch((torch_version or _installed_torch_version)()):
            return None
        env[PAL_DISABLE_SDMA_ENV] = "1"
        return PAL_DISABLE_SDMA_ENV
    except Exception:  # noqa: BLE001 - a startup default must never stop the server
        return None
