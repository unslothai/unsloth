# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/_triton_kernels/conv/conv3d_1x1x1.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: import paths; _conv3d_1x1x1_kernel: N / D / H / W_in / OD / P / Q / M_total are runtime ints (do_not_specialize)
# instead of tl.constexpr, so a new resolution reuses the compiled kernel (upstream recompiles per spatial size).
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

from ._activation import (
    _apply_activation_from_str,
)
from ._kernel_repr import make_kernel_repr
from ._config import get_conv_config, has_conv_config

# Spatial sizes stay runtime ints: one compile per channel / tile config, not per resolution.
_SIZE_ARGS = ["N", "D", "H", "W_in", "OD", "P", "Q", "M_total"]


def _get_config(
    shape_key = None,
    M = None,
    variants = (),
):
    name = "CONV3D-1X1X1" if has_conv_config("CONV3D-1X1X1") else "CONV-1X1"
    return get_conv_config(name, shape_key = shape_key, M = M, variants = variants)


_conv3d_1x1x1_kernel_repr = make_kernel_repr(
    "_conv3d_1x1x1_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
        "LAYOUT",
    ],
)


@triton.jit(repr = _conv3d_1x1x1_kernel_repr, do_not_specialize = _SIZE_ARGS)
def _conv3d_1x1x1_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    D,
    H,
    W_in,
    K_out: tl.constexpr,
    OD,
    P,
    Q,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """Specialized 1x1x1 conv3d kernel.

    A 1x1x1 conv is a pure GEMM over channels, with no (t, r, s) taps to decode
    and no spatial-tap masking:

        y[n,k,o,p,q] = Σ_c  x[n,c,o,p,q] · w[k,c]   (stride/padding still map
                                                     the output cell back to its
                                                     input cell)

    ``LAYOUT`` ("ncdhw" or "ndhwc") selects input/output strides; weight is
    always ``[K_out, C]`` contiguous. Loads are masked on ``c < C`` so no weight
    padding/prepack is needed.
    """
    # W: [K_out, C] contiguous
    stride_w_kout: tl.constexpr = C
    stride_w_c: tl.constexpr = 1
    if LAYOUT == "ncdhw":
        stride_x_w: tl.constexpr = 1
        stride_x_h = W_in
        stride_x_d = H * W_in
        stride_x_c = D * H * W_in
        stride_x_n = C * D * H * W_in
        stride_y_q: tl.constexpr = 1
        stride_y_p = Q
        stride_y_o = P * Q
        stride_y_k = OD * P * Q
        stride_y_n = K_out * OD * P * Q
    else:
        stride_x_c: tl.constexpr = 1
        stride_x_w: tl.constexpr = C
        stride_x_h = W_in * C
        stride_x_d = H * W_in * C
        stride_x_n = D * H * W_in * C
        stride_y_k: tl.constexpr = 1
        stride_y_q: tl.constexpr = K_out
        stride_y_p = Q * K_out
        stride_y_o = P * Q * K_out
        stride_y_n = OD * P * Q * K_out

    pid = tl.program_id(axis = 0)

    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_total
    kout_mask = offs_n < K_out

    opq = OD * P * Q
    n_idx = offs_m[:, None] // opq
    rem = offs_m[:, None] % opq
    o_idx = rem // (P * Q)
    pq = rem % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q

    id_ = o_idx * stride_d - pad_d
    ih = p_idx * stride_h - pad_h
    iw = q_idx * stride_w - pad_w

    spatial_valid = (
        (id_ >= 0) & (id_ < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W_in) & (n_idx < N)
    )

    x_base = (
        X + n_idx * stride_x_n + id_ * stride_x_d + ih * stride_x_h + iw * stride_x_w
    )  # [BLOCK_M, 1]
    w_base = W + offs_n[None, :] * stride_w_kout  # [1, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k0 in range(0, C, BLOCK_K):
        k_offs = k0 + offs_k
        k_mask = k_offs < C

        x_ptrs = x_base + k_offs[None, :] * stride_x_c
        x_tile = tl.load(x_ptrs, mask = spatial_valid & k_mask[None, :], other = 0.0)

        w_ptrs = w_base + k_offs[:, None] * stride_w_c
        w_tile = tl.load(w_ptrs, mask = k_mask[:, None] & kout_mask[None, :], other = 0.0)

        acc = tl.dot(x_tile, w_tile, acc = acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask = kout_mask, other = 0.0).to(tl.float32)
        acc += b[None, :]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + o_idx * stride_y_o
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask = (m_mask[:, None] & kout_mask[None, :]))
