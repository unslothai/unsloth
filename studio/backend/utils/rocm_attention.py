# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unlock the ROCm AOTriton attention kernels torch ships but hides.

On a ROCm build torch gates its AOTriton flash / mem-efficient SDPA kernels behind
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL. With the gate closed, every sub-quadratic
backend declines and SDPA falls through to MATH, which materialises the full
batch x heads x tokens x tokens score matrix. Peak VRAM then grows with the SQUARE of
the token count, so a small model can ask for many times the card at video resolutions
(#8225: a 3.4 GB Q4_K_M asking a 16 GB card for one 66.54 GiB allocation).

Measured on an RX 9060 XT (gfx1200, torch 2.11.0+rocm7.13.0) at B=1 H=16 N=9408 D=64:

    math            387.2 ms   peak 12.14 GiB
    flash            17.3 ms   peak  0.18 GiB   max|d| vs math = 1.221e-04
    mem_efficient    24.6 ms   peak  0.20 GiB   max|d| vs math = 1.221e-04

1.221e-04 is one fp16 ulp for outputs in [0.125, 0.25), which is where these peak; it is
not fp16 epsilon, which is 2**-10 at 1.0. Re-measured on an R9700 (gfx1201): flash and
mem-efficient differ from MATH by exactly one ulp at the output magnitude, and both sit
as far from an fp64 reference as MATH does. The gate is not buying accuracy. It is only
costing the kernel.

Stdlib only. Torch latches the value into a function-local static at its first SDPA
capability check (aten/.../sdp_utils.cpp), not at import, so the deadline is the first
attention call. Running before torch is imported is the earliest point, and the one that
needs no reasoning about who probes first.
"""

# studio/ still ships on the 3.9 floor, where `dict | None` in a signature raises at
# def time. tests/test_python39_compatibility.py ratchets on that.
from __future__ import annotations

import os

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def enable_rocm_aotriton_attention(env: dict | None = None) -> bool:
    """Open the AOTriton gate unless the operator already had an opinion.

    Returns whether this call set it. Any pre-existing value wins, including "0": that
    is the opt-out for someone who hits an AOTriton bug and wants the math fallback back.

    Set unconditionally rather than only on ROCm, because knowing the build means importing
    torch first, which the launcher deliberately defers. Non-ROCm torch never reads a
    TORCH_ROCM_* var, so the cost of being wrong is one unused entry in the environment.
    """
    target = os.environ if env is None else env
    if AOTRITON_ENV in target:
        return False
    target[AOTRITON_ENV] = "1"
    return True
