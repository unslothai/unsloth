# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows: whether an existing install should be repaired onto the managed PyTorch.

An existing home keeps the torch it was installed with; only a repair (UNSLOTH_TORCH_UPGRADE=1) moves it to
the managed build. On RTX 50-series (sm_120) cards xFormers trains only with torch 2.13 and xFormers 0.0.35,
so an older home there runs attention on SDPA instead, or fails in the backward of an old xFormers build.
Torch-free: package metadata and the cached nvidia-smi capability query, so the polled /api/system may call it."""

from __future__ import annotations

import functools
import os
import platform as _platform
import re
import sys
from typing import Callable, Optional

# Lower bound of install.ps1's managed CUDA torch for new installs ($script:Cu130NewInstallTorchSpec).
WINDOWS_MANAGED_TORCH_FLOOR = (2, 13)
# The first xFormers whose sm_120 build ships the backward kernels, paired with that torch.
WINDOWS_MANAGED_XFORMERS_FLOOR = (0, 0, 35)
_SM120 = 12
# The managed torch comes from the cu130 index, which install.ps1 picks only for a driver supporting CUDA 13.0.
_MANAGED_ROUTE_DRIVER_CUDA = (13, 0)


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


@functools.lru_cache(maxsize = 1)
def _driver_cuda_version() -> Optional[tuple[int, int]]:
    """The CUDA version the driver supports, from nvidia-smi's header as install.ps1 reads it; None when
    unreadable. Cached: the driver does not change under a running server."""
    try:
        from utils.hardware import gpu_query
        from utils.hardware.nvidia import _nvidia_smi_executable

        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable()],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout or "")
        return (int(match.group(1)), int(match.group(2))) if match else None
    except Exception:  # noqa: BLE001 -- unknown driver
        return None


def _repair_python_minor(environ = os.environ) -> str:
    """The Python minor a Repair builds its venv with: UNSLOTH_PYTHON when set, else install.ps1's 3.13
    (UNSLOTH_TORCH_UPGRADE=1 does not keep the existing venv's minor)."""
    match = re.match(r"\s*(3\.\d+)", environ.get("UNSLOTH_PYTHON") or "")
    return match.group(1) if match else "3.13"


def _managed_route_host() -> bool:
    """install.ps1's 2.13 route (Test-Cu130Torch213Platform): x64 Windows, and the venv Repair builds is 3.13."""
    return _platform.machine().lower() in ("amd64", "x86_64") and _repair_python_minor() == "3.13"


def windows_repair_recommendation(
    platform: str = sys.platform,
    dist_version: Callable[[str], Optional[str]] = _dist_version,
    capabilities: Callable[[], dict] = _capabilities,
    managed_route_host: Callable[[], bool] = _managed_route_host,
    driver_cuda_version: Callable[[], Optional[tuple[int, int]]] = _driver_cuda_version,
) -> Optional[dict]:
    """``{"reason", "torch", "xformers", "managed_torch", "managed_xformers"}`` when this Windows CUDA install
    has an sm_120+ card and a torch (or an installed xFormers) older than the managed pair, and Repair can
    reach that pair; else None. A driver too old for the cu130 route gets ``driver_older_than_managed``
    instead, since Repair alone would keep the old torch."""
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
    # Only x64 with the Python 3.13 venv takes the managed torch; elsewhere Repair would not change it.
    if not managed_route_host():
        return None
    driver = driver_cuda_version()
    if driver is not None and driver < _MANAGED_ROUTE_DRIVER_CUDA:
        reason = "driver_older_than_managed"
    return {
        "reason": reason,
        "torch": torch_version,
        "xformers": xformers_version,
        "managed_torch": ".".join(map(str, WINDOWS_MANAGED_TORCH_FLOOR)),
        "managed_xformers": ".".join(map(str, WINDOWS_MANAGED_XFORMERS_FLOOR)),
    }
