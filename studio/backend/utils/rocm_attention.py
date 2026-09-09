# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Open the gate on the ROCm AOTriton attention kernels torch ships but hides.

Torch gates its AOTriton flash / mem-efficient SDPA kernels behind
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL on the arches AOTriton still calls experimental.
Shut, SDPA falls through to MATH, whose peak VRAM grows with the square of the token count
(#8225: a 16 GB card asked for 66 GiB). Open, the kernels match MATH to one fp16 ulp.

Torch latches the value at its first SDPA capability check, not at import, so the deadline
is the first attention call. Stdlib only, so the launcher can run it before importing torch.
"""

# studio/ still ships on the 3.9 floor, where `dict | None` in a signature raises at def time.
from __future__ import annotations

import os

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def enable_rocm_aotriton_attention(env: dict | None = None) -> bool:
    """Set the gate to "1" unless a value is already present; returns whether it was set.

    Any pre-existing value wins, including "0", the opt-out for an AOTriton bug. Set
    unconditionally: knowing the build means importing torch first, and non-ROCm torch never
    reads a TORCH_ROCM_* variable.
    """
    target = os.environ if env is None else env
    if AOTRITON_ENV in target:
        return False
    target[AOTRITON_ENV] = "1"
    return True
