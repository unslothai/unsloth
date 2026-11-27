# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: import paths only.
# The vendored portions remain under the MIT License below:
#
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton
import triton.language as tl

from ._kernel_repr import make_kernel_repr
from ._config import get_conv_config, has_conv_config


def _get_config(
    shape_key = None,
    M = None,
    variants = (),
):
    return get_conv_config("CONV-PREPACK", shape_key = shape_key, M = M, variants = variants)


def _get_config_3d(shape_key = None, M = None):
    # Untuned archs reuse the safe Conv2D fallback tile for this transpose.
    name = "CONV3D-PREPACK" if has_conv_config("CONV3D-PREPACK") else "CONV-PREPACK"
    return get_conv_config(name, shape_key = shape_key, M = M)


_ncx_to_cblocked_kernel_repr = make_kernel_repr(
    "_ncx_to_cblocked_kernel",
    ["BLOCK_C", "BLOCK_M", "CB"],
)


@triton.jit(repr = _ncx_to_cblocked_kernel_repr)
def _ncx_to_cblocked_kernel(
    X, Y, C, M, C_PAD: tl.constexpr, CB: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr
):
    """Transpose a contiguous [C, spatial] plane to [C/CB, spatial, CB]."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    x_ptrs = X + pid_n * C * M + offs_c[:, None] * M + offs_m[None, :]
    tile = tl.load(
        x_ptrs,
        mask = (offs_c[:, None] < C) & (offs_m[None, :] < M),
        other = 0.0,
    )

    c_block = offs_c // CB
    c_local = offs_c % CB
    y_ptrs = (
        Y + pid_n * C_PAD * M + c_block[None, :] * M * CB + offs_m[:, None] * CB + c_local[None, :]
    )
    tl.store(
        y_ptrs,
        tl.trans(tile),
        mask = (offs_m[:, None] < M) & (offs_c[None, :] < C_PAD),
    )
