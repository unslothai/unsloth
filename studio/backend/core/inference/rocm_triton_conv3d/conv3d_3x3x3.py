# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: import paths; _conv3d_3x3x3_cblocked_kernel: N / D / H / W_in / OD / P / Q / M_total are runtime ints (do_not_specialize)
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


def _get_config_ndhwc(shape_key = None, M = None):
    name = "CONV3D-3X3X3-NDHWC" if has_conv_config("CONV3D-3X3X3-NDHWC") else "CONV-3X3-NHWC"
    return get_conv_config(name, shape_key = shape_key, M = M)


def _get_config_cblocked(shape_key = None, M = None):
    name = (
        "CONV3D-3X3X3-CBLOCKED" if has_conv_config("CONV3D-3X3X3-CBLOCKED") else "CONV-3X3-CBLOCKED"
    )
    return get_conv_config(name, shape_key = shape_key, M = M)


_conv3d_3x3x3_ndhwc_kernel_repr = make_kernel_repr(
    "_conv3d_3x3x3_ndhwc_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "HAS_BIAS", "ACTIVATION"],
)


_conv3d_3x3x3_cblocked_kernel_repr = make_kernel_repr(
    "_conv3d_3x3x3_cblocked_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "HAS_BIAS", "ACTIVATION"],
)


@triton.jit(repr = _conv3d_3x3x3_ndhwc_kernel_repr)
def _conv3d_3x3x3_ndhwc_kernel(
    X,
    W,
    BIAS,
    Y,
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    OD: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """NDHWC-native 3x3x3 conv3d. Weight is K-major ``[K_out, 27, C_pad]``; the
    27 taps (``trs = t*9 + r*3 + s``) are walked in static_range with spatial
    validity hoisted out of the channel loop, and the reduction runs over
    channels (contiguous in NDHWC, so ``stride_x_c=1`` is hardcoded for
    coalesced loads)."""
    # X: [N, D, H, W_in, C] contiguous NDHWC (stride_x_c=1 hardcoded below)
    stride_x_w: tl.constexpr = C
    stride_x_h: tl.constexpr = W_in * C
    stride_x_d: tl.constexpr = H * W_in * C
    stride_x_n: tl.constexpr = D * H * W_in * C
    # W: [K_out, 27, C_pad] contiguous
    stride_w_c: tl.constexpr = 1
    stride_w_trs: tl.constexpr = C_pad
    stride_w_kout: tl.constexpr = 27 * C_pad
    # Y: [N, OD, P, Q, K_out] contiguous NDHWC (stride_y_k=1 hardcoded below)
    stride_y_q: tl.constexpr = K_out
    stride_y_p: tl.constexpr = Q * K_out
    stride_y_o: tl.constexpr = P * Q * K_out
    stride_y_n: tl.constexpr = OD * P * Q * K_out

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
    od_idx = rem // (P * Q)
    pq = rem % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q
    n_valid = n_idx < N

    base_id = od_idx * stride_d - pad_d
    base_ih = p_idx * stride_h - pad_h
    base_iw = q_idx * stride_w - pad_w
    stride_x_dd = dil_d * stride_x_d
    stride_x_dh = dil_h * stride_x_h
    stride_x_dw = dil_w * stride_x_w

    x_base = (
        X + n_idx * stride_x_n + base_id * stride_x_d + base_ih * stride_x_h + base_iw * stride_x_w
    )
    w_base = W + offs_n[None, :] * stride_w_kout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for t in tl.static_range(3):
        id_ = base_id + t * dil_d
        valid_id = n_valid & (id_ >= 0) & (id_ < D)
        for r in tl.static_range(3):
            ih = base_ih + r * dil_h
            valid_ir = valid_id & (ih >= 0) & (ih < H)
            for s in tl.static_range(3):
                trs_idx = t * 9 + r * 3 + s
                iw = base_iw + s * dil_w
                spatial_valid = valid_ir & (iw >= 0) & (iw < W_in)

                for k0 in range(0, C_pad, BLOCK_K):
                    k_offs = k0 + offs_k
                    k_mask = k_offs < C

                    x_ptrs = (
                        x_base
                        + k_offs[None, :]
                        + t * stride_x_dd
                        + r * stride_x_dh
                        + s * stride_x_dw
                    )
                    w_ptrs = w_base + trs_idx * stride_w_trs + k_offs[:, None] * stride_w_c

                    x_tile = tl.load(x_ptrs, mask = spatial_valid & k_mask[None, :], other = 0.0)
                    w_tile = tl.load(w_ptrs, mask = k_mask[:, None] & kout_mask[None, :], other = 0.0)
                    acc = tl.dot(x_tile, w_tile, acc = acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask = offs_n < K_out, other = 0.0).to(tl.float32)
        acc += b[None, :]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :]
        + od_idx * stride_y_o
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask = (m_mask[:, None] & kout_mask[None, :]))


@triton.jit(repr = _conv3d_3x3x3_cblocked_kernel_repr, do_not_specialize = _SIZE_ARGS)
def _conv3d_3x3x3_cblocked_kernel(
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
    C_pad: tl.constexpr,
    Cb: tl.constexpr,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """3x3x3 conv3d over channel-blocked ``[N, C_blocks, D, H, W_in, Cb]`` input,
    NCDHW output. Same 27-tap / channel-reduction structure as the NDHWC kernel,
    but reads the repacked NCDHWc layout so an NCDHW source gets coalesced
    channel loads (``stride_x_cb=1`` hardcoded). Stays coalesced only when
    ``BLOCK_K <= Cb``."""
    # X: [N, C_blocks, D, H, W_in, Cb] where C_blocks = C_pad // Cb
    stride_x_cb: tl.constexpr = 1
    stride_x_w: tl.constexpr = Cb
    stride_x_h = W_in * Cb
    stride_x_d = H * W_in * Cb
    stride_x_cblock = D * H * W_in * Cb
    stride_x_n = (C_pad // Cb) * D * H * W_in * Cb
    # W: [K_out, 27, C_pad] contiguous
    stride_w_c: tl.constexpr = 1
    stride_w_trs: tl.constexpr = C_pad
    stride_w_kout: tl.constexpr = 27 * C_pad
    # Y: [N, K_out, OD, P, Q] contiguous NCDHW
    stride_y_q: tl.constexpr = 1
    stride_y_p = Q
    stride_y_o = P * Q
    stride_y_k = OD * P * Q
    stride_y_n = K_out * OD * P * Q

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
    od_idx = rem // (P * Q)
    pq = rem % (P * Q)
    p_idx = pq // Q
    q_idx = pq % Q
    n_valid = n_idx < N

    base_id = od_idx * stride_d - pad_d
    base_ih = p_idx * stride_h - pad_h
    base_iw = q_idx * stride_w - pad_w
    stride_x_dd = dil_d * stride_x_d
    stride_x_dh = dil_h * stride_x_h
    stride_x_dw = dil_w * stride_x_w

    x_base = (
        X + n_idx * stride_x_n + base_id * stride_x_d + base_ih * stride_x_h + base_iw * stride_x_w
    )
    w_base = W + offs_n[None, :] * stride_w_kout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for t in tl.static_range(3):
        id_ = base_id + t * dil_d
        valid_id = n_valid & (id_ >= 0) & (id_ < D)
        for r in tl.static_range(3):
            ih = base_ih + r * dil_h
            valid_ir = valid_id & (ih >= 0) & (ih < H)
            for s in tl.static_range(3):
                trs_idx = t * 9 + r * 3 + s
                iw = base_iw + s * dil_w
                spatial_valid = valid_ir & (iw >= 0) & (iw < W_in)

                for k0 in range(0, C_pad, BLOCK_K):
                    k_offs = k0 + offs_k
                    k_mask = k_offs < C

                    cblock_idx = k_offs // Cb
                    k_local = k_offs % Cb

                    x_ptrs = (
                        x_base
                        + cblock_idx[None, :] * stride_x_cblock
                        + k_local[None, :] * stride_x_cb
                        + t * stride_x_dd
                        + r * stride_x_dh
                        + s * stride_x_dw
                    )
                    w_ptrs = w_base + trs_idx * stride_w_trs + k_offs[:, None] * stride_w_c

                    x_tile = tl.load(x_ptrs, mask = spatial_valid & k_mask[None, :], other = 0.0)
                    w_tile = tl.load(w_ptrs, mask = k_mask[:, None] & kout_mask[None, :], other = 0.0)
                    acc = tl.dot(x_tile, w_tile, acc = acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask = offs_n < K_out, other = 0.0).to(tl.float32)
        acc += b[None, :]

    acc = _apply_activation_from_str(acc, ACTIVATION)

    y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + od_idx * stride_y_o
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )
    tl.store(y_ptrs, acc, mask = (m_mask[:, None] & kout_mask[None, :]))
