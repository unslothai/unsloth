# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows: whether an existing install should be repaired onto the managed PyTorch.

An existing home keeps the torch it was installed with; only a repair (UNSLOTH_TORCH_UPGRADE=1) moves it to
the managed build (torch 2.13 on the cu130 route), which is what RTX 50-series (sm_120) support is tested on.
Torch only: the xFormers version is never a reason. The managed 0.0.35 cu130 wheel builds no FlashAttention 2
of its own and its cutlass kernels stop at sm_90, so on sm_120 it is not an upgrade over an installed 0.0.34.
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
    """The CUDA version the driver supports, from nvidia-smi's header as install.ps1 reads it ("CUDA Version" or,
    on newer drivers, "CUDA UMD Version"); None when
    unreadable. Cached: the driver does not change under a running server."""
    try:
        from utils.hardware import gpu_query
        from utils.hardware.nvidia import _nvidia_smi_executable
        from utils.native_path_leases import child_env_without_native_path_secret
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        # Same launch as nvidia.py's queries: no console window on Windows, no native-path secret.
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable()],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **windows_hidden_subprocess_kwargs(),
        )
        match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout or "")
        return (int(match.group(1)), int(match.group(2))) if match else None
    except Exception:  # noqa: BLE001 -- unknown driver
        return None


def _repair_python_minor(environ = os.environ) -> str:
    """The Python minor a Repair builds its venv with: UNSLOTH_PYTHON when set, else install.ps1's 3.13
    (UNSLOTH_TORCH_UPGRADE=1 does not keep the existing venv's minor)."""
    match = re.match(r"\s*(3\.\d+)", environ.get("UNSLOTH_PYTHON") or "")
    return match.group(1) if match else "3.13"


def _index_override_leaf(environ = os.environ) -> Optional[str]:
    """The torch index leaf a user override pins ("cu128"), as install.ps1's Get-TorchIndexUrl honors it
    before the driver; None when nothing is overridden."""
    url = (environ.get("UNSLOTH_TORCH_INDEX_URL") or "").strip()
    if url:
        return re.split(r"[?#]", url, maxsplit = 1)[0].rstrip("/").split("/")[-1].lower()
    family = (environ.get("UNSLOTH_TORCH_INDEX_FAMILY") or "").strip().strip("/")
    return family.split("/")[-1].lower() if family else None


def _managed_route_host(environ = os.environ) -> bool:
    """install.ps1's 2.13 route (Test-Cu130Torch213Platform): x64 Windows, the venv Repair builds is 3.13, and
    no index override pins a leaf other than cu130."""
    leaf = _index_override_leaf(environ)
    return (
        _platform.machine().lower() in ("amd64", "x86_64")
        and _repair_python_minor(environ) == "3.13"
        and leaf in (None, "cu130")
    )


def windows_repair_recommendation(
    platform: str = sys.platform,
    dist_version: Callable[[str], Optional[str]] = _dist_version,
    capabilities: Callable[[], dict] = _capabilities,
    managed_route_host: Callable[[], bool] = _managed_route_host,
    driver_cuda_version: Callable[[], Optional[tuple[int, int]]] = _driver_cuda_version,
) -> Optional[dict]:
    """``{"reason", "torch", "managed_torch"}`` when this Windows CUDA install has an sm_120+ card and a torch
    older than the managed one, and Repair can reach it; else None. xFormers is never a reason (see above). A driver too old for the cu130 route gets ``driver_older_than_managed``
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
    if torch_release is None or torch_release[:2] >= WINDOWS_MANAGED_TORCH_FLOOR:
        return None
    reason = "torch_older_than_managed"
    # Only x64 with the Python 3.13 venv takes the managed torch; elsewhere Repair would not change it.
    if not managed_route_host():
        return None
    driver = driver_cuda_version()
    if driver is not None and driver < _MANAGED_ROUTE_DRIVER_CUDA:
        reason = "driver_older_than_managed"
    return {
        "reason": reason,
        "torch": torch_version,
        "managed_torch": ".".join(map(str, WINDOWS_MANAGED_TORCH_FLOOR)),
    }
