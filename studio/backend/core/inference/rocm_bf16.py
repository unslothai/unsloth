# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native bf16 on a ROCm card, by gfx target.

``torch.cuda.is_bf16_supported()`` returns True on every HIP build without looking at the device.
bf16 math needs MFMA (gfx908+ CDNA), WMMA or bf16 dot (gfx11+) per llvm/lib/Target/AMDGPU/AMDGPU.td;
GCN, Vega and RDNA1/2 emulate it through fp32, so they get float16, as pre-Ampere NVIDIA does.
``UNSLOTH_STUDIO_ROCM_BF16=1`` restores torch's answer everywhere; ``=0`` forces float16.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

ROCM_BF16_ENV = "UNSLOTH_STUDIO_ROCM_BF16"

# GCN gfx6xx-8xx, Vega gfx90x except gfx908 / gfx90a (MFMA), RDNA1/2 gfx101x-103x. Unknown targets keep torch's answer.
_NO_NATIVE_BF16_ARCH = re.compile(r"^gfx(?:[6-8][0-9a-f]{2}|90[0-79c]|10[0-3][0-9a-f])$")


def is_rocm_torch(torch: Any) -> bool:
    """ROCm build, incl. AMD SDK / Radeon wheels that leave ``torch.version.hip`` unset (tag in ``__version__``)."""
    return bool(
        getattr(getattr(torch, "version", None), "hip", None)
        or "rocm" in str(getattr(torch, "__version__", "") or "").lower()
    )


def rocm_bf16_forced_off() -> bool:
    return os.environ.get(ROCM_BF16_ENV, "").strip().lower() in ("0", "false", "no", "off")


def normalize_gfx_arch(arch: Any) -> str:
    """``gfx906:sramecc+:xnack-`` -> ``gfx906``; anything unreadable -> ""."""
    try:
        return str(arch or "").split(":")[0].strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def gfx_arch_lacks_native_bf16(arch: Any) -> bool:
    """True for a gfx target known to have no bf16 MFMA / WMMA / dot instructions."""
    return bool(_NO_NATIVE_BF16_ARCH.match(normalize_gfx_arch(arch)))


def _device_gfx_arch(torch: Any, ordinal: Optional[int]) -> str:
    try:
        index = torch.cuda.current_device() if ordinal is None else ordinal
        props = torch.cuda.get_device_properties(index)
    except Exception:  # noqa: BLE001 -- unreadable properties: no arch, torch's answer stands
        return ""
    # gcnArchName alone is empty on some AMD SDK / Radeon wheels (same spellings utils.hardware reads).
    for attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        arch = normalize_gfx_arch(getattr(props, attr, ""))
        if arch:
            return arch
    return ""


def rocm_bf16_supported(torch: Any, ordinal: Optional[int] = None) -> bool:
    """Native bf16 on the ROCm card ``ordinal`` (current device when None); HIP builds only. Raises what torch raises."""
    if rocm_bf16_forced_off():
        return False
    override = os.environ.get(ROCM_BF16_ENV, "").strip().lower()
    if override not in ("1", "true", "yes", "on") and gfx_arch_lacks_native_bf16(
        _device_gfx_arch(torch, ordinal)
    ):
        return False
    return bool(torch.cuda.is_bf16_supported())
