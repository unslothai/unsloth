# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How much memory the NPU can map, shared by the catalog's fit warning and Auto context."""

from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Windows caps NPU RAM at about half (lemonade-sdk/lemonade#839, FastFlowLM#904); not measured.
_WINDOWS_NPU_RAM_SHARE = 0.5
_NPU_OS_RESERVE_GB = 2.0
# Flat 1 GB covers context buffers and runtime at default 8192 context; borderline models pass.
_NPU_LOAD_MARGIN_GB = 1.0


def system_ram_gb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception as exc:  # noqa: BLE001 -- no reading means no window, never a failed caller
        logger.debug("Reading system RAM for the NPU memory window failed: %s", exc)
        return None


def npu_memory_window_gb(
    total_ram_gb: Optional[float], platform: Optional[str] = None
) -> Optional[float]:
    if not total_ram_gb or total_ram_gb <= 0:
        return None
    if (platform or sys.platform) == "win32":
        return round(total_ram_gb * _WINDOWS_NPU_RAM_SHARE, 1)
    return round(max(total_ram_gb - _NPU_OS_RESERVE_GB, 0.0), 1)


def npu_memory_needed_gb(size_gb: Optional[float]) -> Optional[float]:
    if size_gb is None or size_gb <= 0:
        return None
    return round(size_gb + _NPU_LOAD_MARGIN_GB, 1)


def fits_npu_memory(size_gb: Optional[float], window_gb: Optional[float]) -> Optional[bool]:
    needed = npu_memory_needed_gb(size_gb)
    if needed is None or window_gb is None:
        return None
    return needed <= window_gb
