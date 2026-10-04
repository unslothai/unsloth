# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native bf16 on a ROCm card, by gfx target.

``torch.cuda.is_bf16_supported()`` returns True for every HIP build before it looks at the device
(``if torch.version.hip: return True`` in torch/cuda/__init__.py, torch 2.6 through 2.11), so it
says yes on targets with no bf16 matrix path at all. Per LLVM's AMDGPU target definitions
(llvm/lib/Target/AMDGPU/AMDGPU.td), bf16 math needs MFMA (``mai-insts``: gfx908 and later CDNA),
WMMA (gfx11 / gfx12) or the bf16 dot instructions (``dot9-insts`` v_dot2_bf16_bf16, ``dot12-insts``
v_dot2_f32_bf16: gfx11 and later). GCN (gfx6xx-gfx8xx), Vega (gfx900 / gfx902 / gfx904 / gfx906 /
gfx909 / gfx90c) and RDNA1 / RDNA2 (gfx101x / gfx103x) have none of them: bf16 there is emulated
through fp32, so float16 is the compute dtype, as on pre-Ampere NVIDIA.

torch is imported lazily (callers pass their module) so this stays importable without torch.
``UNSLOTH_STUDIO_ROCM_BF16=1`` restores torch's answer on every target; ``=0`` forces float16.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

ROCM_BF16_ENV = "UNSLOTH_STUDIO_ROCM_BF16"

# gfx6xx-gfx8xx (GCN1-GCN4), gfx900-gfx90c except gfx908 / gfx90a (Vega, incl. gfx906 MI50 / Radeon VII),
# gfx1010-gfx103f (RDNA1 / RDNA2). gfx908, gfx90a, gfx94x, gfx950, gfx11xx, gfx12xx and anything unknown keep torch's answer.
_NO_NATIVE_BF16_ARCH = re.compile(r"^gfx(?:[6-8][0-9a-f]{2}|90[0-79c]|10[0-3][0-9a-f])$")


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
    # Same attribute spellings utils.hardware reads: gcnArchName alone is empty on some AMD SDK / Radeon wheels.
    for attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        arch = normalize_gfx_arch(getattr(props, attr, ""))
        if arch:
            return arch
    return ""


def rocm_bf16_supported(torch: Any, ordinal: Optional[int] = None) -> bool:
    """Native bf16 on the ROCm card ``ordinal`` (current device when None). Only call on a HIP build.

    ``torch.cuda.is_bf16_supported()`` takes no device argument, so a caller probing a selected card
    scopes it current first; the arch is read from ``ordinal`` directly. Raises what torch raises."""
    override = os.environ.get(ROCM_BF16_ENV, "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override not in ("1", "true", "yes", "on") and gfx_arch_lacks_native_bf16(
        _device_gfx_arch(torch, ordinal)
    ):
        return False
    return bool(torch.cuda.is_bf16_supported())
