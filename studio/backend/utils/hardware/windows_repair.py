# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows: whether an existing install should be repaired onto the managed PyTorch.

An existing home keeps the torch it was installed with; only a repair (UNSLOTH_TORCH_UPGRADE=1) moves it to
the managed build. On RTX 50-series (sm_120) cards xFormers trains only with torch 2.13 and xFormers 0.0.35,
so an older home there runs attention on SDPA instead, or fails in the backward of an old xFormers build.
Torch-free: package metadata and the cached nvidia-smi capability query, so the polled /api/system may call it."""

from __future__ import annotations

import re
import sys
from typing import Callable, Optional

# Lower bound of install.ps1's managed CUDA torch for new installs ($script:Cu130NewInstallTorchSpec).
WINDOWS_MANAGED_TORCH_FLOOR = (2, 13)
# The first xFormers whose sm_120 build ships the backward kernels, paired with that torch.
WINDOWS_MANAGED_XFORMERS_FLOOR = (0, 0, 35)
_SM120 = 12


def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001 -- not installed or unreadable
        return None


def _release(version: Optional[str]) -> Optional[tuple[int, ...]]:
    match = re.match(r"(\d+(?:\.\d+)*)", version or "")
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def _capabilities() -> dict:
    try:
        from utils.hardware.nvidia import get_compute_capabilities
        return get_compute_capabilities()
    except Exception:  # noqa: BLE001 -- unreadable counts as no card
        return {}


def windows_repair_recommendation(
    platform: str = sys.platform,
    dist_version: Callable[[str], Optional[str]] = _dist_version,
    capabilities: Callable[[], dict] = _capabilities,
) -> Optional[dict]:
    """``{"reason", "torch", "xformers", "managed_torch", "managed_xformers"}`` when this Windows CUDA install
    has an sm_120+ card and a torch (or an installed xFormers) older than the managed pair; else None."""
    if platform != "win32":
        return None
    torch_version = dist_version("torch")
    # Only CUDA builds: a CPU wheel is a different repair (the GPU verdict already says so).
    if not torch_version or "+cu" not in torch_version:
        return None
    if not any(int(cap[0]) >= _SM120 for cap in capabilities().values()):
        return None
    torch_release = _release(torch_version)
    xformers_version = dist_version("xformers")
    xformers_release = _release(xformers_version)
    if torch_release is not None and torch_release[:2] < WINDOWS_MANAGED_TORCH_FLOOR:
        reason = "torch_older_than_managed"
    elif xformers_release is not None and xformers_release < WINDOWS_MANAGED_XFORMERS_FLOOR:
        reason = "xformers_older_than_managed"
    else:
        return None
    return {
        "reason": reason,
        "torch": torch_version,
        "xformers": xformers_version,
        "managed_torch": ".".join(map(str, WINDOWS_MANAGED_TORCH_FLOOR)),
        "managed_xformers": ".".join(map(str, WINDOWS_MANAGED_XFORMERS_FLOOR)),
    }
