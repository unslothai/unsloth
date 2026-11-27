# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
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

"""2.5D Winograd for 3x3x3 conv3d: F(4x4, 3x3) on the (H, W) plane with the
depth axis handled as a direct 3-tap reduction.

Pipeline (NCDHW, stride=1, pad, dilation=1):
  1. input transform  : V[36, N*D*T_hw, C_pad] = Bᵀ d B on every input (H,W)
                         plane of every input depth slice
  2. batched GEMM      : M[36, N*OD*T_hw, K] = Σ_t Σ_c U[t] · V(depth-shifted),
                         reducing over the 3 depth taps (id = od - pad_d + t,
                         masked to [0, D)) and channels
  3. output transform  : Y[N, K, OD, P, Q] = Aᵀ M A + bias/act per (n, od, tile)

Winograd runs only on H,W (large dims); depth stays a direct fp32 sum, so the
numerical amplification is the 2D F(4,3) level (~361x), not cubed. The 2D
input and output transforms are shared with Conv2D.
"""

import triton
import triton.language as tl

from .winograd_f4x3_common import (
    _winograd_f4x3_input_transform,
    _winograd_f4x3_output_transform,
)
from ._kernel_repr import make_kernel_repr
from ._config import get_conv_config, has_conv_config


def _config_name(name_3d, fallback):
    return name_3d if has_conv_config(name_3d) else fallback


def _get_config_input(
    shape_key = None,
    M = None,
    variants = (),
):
    name = _config_name("CONV3D-WINO-HW-F4X3-INPUT", "CONV-WINO-F4X3-INPUT")
    return get_conv_config(name, shape_key = shape_key, M = M, variants = variants)


def _get_config_gemm(shape_key = None, M = None):
    name = _config_name("CONV3D-WINO-HW-F4X3-GEMM", "CONV-WINO-F4X3-GEMM")
    return get_conv_config(name, shape_key = shape_key, M = M)


def _get_config_output(shape_key = None, M = None):
    name = _config_name("CONV3D-WINO-HW-F4X3-OUTPUT", "CONV-WINO-F4X3-OUTPUT")
    return get_conv_config(name, shape_key = shape_key, M = M)


_winograd_hw_f4x3_input_transform_kernel_repr = make_kernel_repr(
    "_winograd_hw_f4x3_input_transform_kernel", ["BLOCK_C", "CBLOCKED"]
)
_winograd_hw_f4x3_batched_gemm_kernel_repr = make_kernel_repr(
    "_winograd_hw_f4x3_batched_gemm_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M"],
)
_winograd_hw_f4x3_output_transform_kernel_repr = make_kernel_repr(
    "_winograd_hw_f4x3_output_transform_kernel",
    ["BLOCK_K", "HAS_BIAS", "ACTIVATION"],
)


@triton.jit(repr = _winograd_hw_f4x3_input_transform_kernel_repr)
def _winograd_hw_f4x3_input_transform_kernel(
    X,
    V,
    N,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T_v: tl.constexpr,
    pad_h,
    pad_w,
    BLOCK_C: tl.constexpr,
    Cb: tl.constexpr,
    CBLOCKED: tl.constexpr,
):
    """Apply BᵀdB to every (n, d) input plane."""
    if CBLOCKED:
        # X: NCDHWc [N, C_blocks, D, H, W_in, Cb]
        stride_x_w: tl.constexpr = Cb
        stride_x_h: tl.constexpr = W_in * Cb
        stride_x_d: tl.constexpr = H * W_in * Cb
        stride_x_cblock: tl.constexpr = D * H * W_in * Cb
        stride_x_n: tl.constexpr = (C_pad // Cb) * D * H * W_in * Cb
    else:
        # X: NCDHW [N, C, D, H, W_in]
        stride_x_w: tl.constexpr = 1
        stride_x_h: tl.constexpr = W_in
        stride_x_d: tl.constexpr = H * W_in
        stride_x_c: tl.constexpr = D * H * W_in
        stride_x_n: tl.constexpr = C * D * H * W_in

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    dtw = D * tile_H * tile_W
    n = tile_idx // dtw
    rem = tile_idx % dtw
    d = rem // (tile_H * tile_W)
    rem2 = rem % (tile_H * tile_W)
    th = rem2 // tile_W
    tw = rem2 % tile_W

    h_start = th * 4 - pad_h
    w_start = tw * 4 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    if CBLOCKED:
        cblock_idx = offs_c // Cb
        c_local = offs_c % Cb
        base = X + n * stride_x_n + d * stride_x_d + cblock_idx * stride_x_cblock + c_local
    else:
        base = X + n * stride_x_n + d * stride_x_d + offs_c * stride_x_c

    _winograd_f4x3_input_transform(
        X,
        V,
        base,
        tile_idx,
        offs_c,
        c_mask,
        n < N,
        h_start,
        w_start,
        H,
        W_in,
        C_pad,
        T_v,
        stride_x_h,
        stride_x_w,
        BLOCK_C,
    )


@triton.jit(repr = _winograd_hw_f4x3_batched_gemm_kernel_repr)
def _winograd_hw_f4x3_batched_gemm_kernel(
    V,
    U,
    M_out,
    N,
    D: tl.constexpr,
    OD: tl.constexpr,
    T_hw: tl.constexpr,
    T_v: tl.constexpr,
    T_out: tl.constexpr,
    K_out: tl.constexpr,
    C_pad: tl.constexpr,
    pad_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Reduce the three depth taps with a batched GEMM for each transform."""
    # V: [36, T_v, C_pad]; U: [3, 36, K_out, C_pad]; M: [36, T_out, K_out]
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T_v * C_pad
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    stride_u_tap: tl.constexpr = 36 * K_out * C_pad
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T_out * K_out

    pid = tl.program_id(0)
    alpha = tl.program_id(1)

    num_pid_m = tl.cdiv(T_out, BLOCK_M)
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

    tile = offs_m % T_hw
    od = (offs_m // T_hw) % OD
    n = offs_m // (OD * T_hw)

    v_alpha = V + alpha * stride_v_alpha
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for t in tl.static_range(3):
        id_ = od - pad_d + t
        row_valid = (id_ >= 0) & (id_ < D) & (offs_m < T_out)
        v_row = n * (D * T_hw) + id_ * T_hw + tile
        u_base = U + t * stride_u_tap + alpha * stride_u_alpha

        for k0 in range(0, C_pad, BLOCK_K):
            k_offs = k0 + offs_k
            k_in = k_offs[None, :] < C_pad

            v_ptrs = v_alpha + v_row[:, None] * stride_v_tile + k_offs[None, :] * stride_v_c
            v_tile = tl.load(v_ptrs, mask = row_valid[:, None] & k_in, other = 0.0)

            u_ptrs = u_base + offs_n[:, None] * stride_u_k + k_offs[None, :] * stride_u_c
            u_tile = tl.load(u_ptrs, mask = (offs_n[:, None] < K_out) & k_in, other = 0.0)

            acc = tl.dot(v_tile, tl.trans(u_tile), acc = acc)

    m_ptrs = (
        M_out
        + alpha * stride_m_alpha
        + offs_m[:, None] * stride_m_tile
        + offs_n[None, :] * stride_m_k
    )
    m_mask = (offs_m[:, None] < T_out) & (offs_n[None, :] < K_out)
    tl.store(m_ptrs, acc, mask = m_mask)


@triton.jit(repr = _winograd_hw_f4x3_output_transform_kernel_repr)
def _winograd_hw_f4x3_output_transform_kernel(
    M_in,
    BIAS,
    Y,
    N,
    K_out: tl.constexpr,
    OD: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T_out: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Apply AᵀMA with bias and activation to every output tile."""
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T_out * K_out
    # Y: NCDHW [N, K_out, OD, P, Q]
    stride_y_q: tl.constexpr = 1
    stride_y_p: tl.constexpr = Q
    stride_y_o: tl.constexpr = P * Q
    stride_y_k: tl.constexpr = OD * P * Q
    stride_y_n: tl.constexpr = K_out * OD * P * Q

    tile_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    otw = OD * tile_H * tile_W
    n = tile_idx // otw
    rem = tile_idx % otw
    od = rem // (tile_H * tile_W)
    rem2 = rem % (tile_H * tile_W)
    th = rem2 // tile_W
    tw = rem2 % tile_W

    p_start = th * 4
    q_start = tw * 4

    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K_out
    m_base = M_in + tile_idx * stride_m_tile + offs_k
    y_base = Y + n * stride_y_n + od * stride_y_o + offs_k * stride_y_k

    _winograd_f4x3_output_transform(
        BIAS,
        m_base,
        y_base,
        offs_k,
        k_mask,
        n < N,
        p_start,
        q_start,
        P,
        Q,
        stride_m_alpha,
        stride_y_p,
        stride_y_q,
        HAS_BIAS,
        ACTIVATION,
    )
