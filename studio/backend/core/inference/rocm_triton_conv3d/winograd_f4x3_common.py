# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/_triton_kernels/conv/winograd_f4x3_common.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
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

"""Shared F(4x4, 3x3) Winograd input and output transforms."""

import triton
import triton.language as tl

from ._activation import (
    _apply_activation_from_str,
)


@triton.jit
def _winograd_f4x3_input_transform(
    X,
    V,
    base,
    tile_idx,
    offs_c,
    c_mask,
    n_valid,
    h_start,
    w_start,
    H: tl.constexpr,
    W_in: tl.constexpr,
    C_pad: tl.constexpr,
    T: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Load one 6x6 input tile, apply B-transforms, and store V."""
    INPUT_DTYPE: tl.constexpr = X.type.element_ty
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad

    d00 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d01 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d02 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d03 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d04 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d05 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d10 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d11 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d12 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d13 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d14 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d15 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d20 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d21 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d22 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d23 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d24 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d25 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d30 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d31 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d32 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d33 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d34 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d35 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d40 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d41 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d42 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d43 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d44 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d45 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d50 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d51 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d52 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d53 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d54 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)
    d55 = tl.zeros((BLOCK_C,), dtype = INPUT_DTYPE)

    for r in tl.static_range(6):
        h = h_start + r
        h_valid = n_valid & (h >= 0) & (h < H)
        for s in tl.static_range(6):
            w = w_start + s
            valid = h_valid & (w >= 0) & (w < W_in)
            ptr = base + h * stride_x_h + w * stride_x_w
            val = tl.load(ptr, mask = valid & c_mask, other = 0.0)
            if r == 0:
                if s == 0:
                    d00 = val
                elif s == 1:
                    d01 = val
                elif s == 2:
                    d02 = val
                elif s == 3:
                    d03 = val
                elif s == 4:
                    d04 = val
                else:
                    d05 = val
            elif r == 1:
                if s == 0:
                    d10 = val
                elif s == 1:
                    d11 = val
                elif s == 2:
                    d12 = val
                elif s == 3:
                    d13 = val
                elif s == 4:
                    d14 = val
                else:
                    d15 = val
            elif r == 2:
                if s == 0:
                    d20 = val
                elif s == 1:
                    d21 = val
                elif s == 2:
                    d22 = val
                elif s == 3:
                    d23 = val
                elif s == 4:
                    d24 = val
                else:
                    d25 = val
            elif r == 3:
                if s == 0:
                    d30 = val
                elif s == 1:
                    d31 = val
                elif s == 2:
                    d32 = val
                elif s == 3:
                    d33 = val
                elif s == 4:
                    d34 = val
                else:
                    d35 = val
            elif r == 4:
                if s == 0:
                    d40 = val
                elif s == 1:
                    d41 = val
                elif s == 2:
                    d42 = val
                elif s == 3:
                    d43 = val
                elif s == 4:
                    d44 = val
                else:
                    d45 = val
            else:
                if s == 0:
                    d50 = val
                elif s == 1:
                    d51 = val
                elif s == 2:
                    d52 = val
                elif s == 3:
                    d53 = val
                elif s == 4:
                    d54 = val
                else:
                    d55 = val

    # Use fp32 transform arithmetic to avoid fp16 overflow from the 4x/5x terms.
    d00f = d00.to(tl.float32)
    d01f = d01.to(tl.float32)
    d02f = d02.to(tl.float32)
    d03f = d03.to(tl.float32)
    d04f = d04.to(tl.float32)
    d05f = d05.to(tl.float32)
    d10f = d10.to(tl.float32)
    d11f = d11.to(tl.float32)
    d12f = d12.to(tl.float32)
    d13f = d13.to(tl.float32)
    d14f = d14.to(tl.float32)
    d15f = d15.to(tl.float32)
    d20f = d20.to(tl.float32)
    d21f = d21.to(tl.float32)
    d22f = d22.to(tl.float32)
    d23f = d23.to(tl.float32)
    d24f = d24.to(tl.float32)
    d25f = d25.to(tl.float32)
    d30f = d30.to(tl.float32)
    d31f = d31.to(tl.float32)
    d32f = d32.to(tl.float32)
    d33f = d33.to(tl.float32)
    d34f = d34.to(tl.float32)
    d35f = d35.to(tl.float32)
    d40f = d40.to(tl.float32)
    d41f = d41.to(tl.float32)
    d42f = d42.to(tl.float32)
    d43f = d43.to(tl.float32)
    d44f = d44.to(tl.float32)
    d45f = d45.to(tl.float32)
    d50f = d50.to(tl.float32)
    d51f = d51.to(tl.float32)
    d52f = d52.to(tl.float32)
    d53f = d53.to(tl.float32)
    d54f = d54.to(tl.float32)
    d55f = d55.to(tl.float32)

    t00 = 4 * d00f - 5 * d20f + d40f
    t01 = 4 * d01f - 5 * d21f + d41f
    t02 = 4 * d02f - 5 * d22f + d42f
    t03 = 4 * d03f - 5 * d23f + d43f
    t04 = 4 * d04f - 5 * d24f + d44f
    t05 = 4 * d05f - 5 * d25f + d45f

    t10 = -4 * d10f - 4 * d20f + d30f + d40f
    t11 = -4 * d11f - 4 * d21f + d31f + d41f
    t12 = -4 * d12f - 4 * d22f + d32f + d42f
    t13 = -4 * d13f - 4 * d23f + d33f + d43f
    t14 = -4 * d14f - 4 * d24f + d34f + d44f
    t15 = -4 * d15f - 4 * d25f + d35f + d45f

    t20 = 4 * d10f - 4 * d20f - d30f + d40f
    t21 = 4 * d11f - 4 * d21f - d31f + d41f
    t22 = 4 * d12f - 4 * d22f - d32f + d42f
    t23 = 4 * d13f - 4 * d23f - d33f + d43f
    t24 = 4 * d14f - 4 * d24f - d34f + d44f
    t25 = 4 * d15f - 4 * d25f - d35f + d45f

    t30 = -2 * d10f - d20f + 2 * d30f + d40f
    t31 = -2 * d11f - d21f + 2 * d31f + d41f
    t32 = -2 * d12f - d22f + 2 * d32f + d42f
    t33 = -2 * d13f - d23f + 2 * d33f + d43f
    t34 = -2 * d14f - d24f + 2 * d34f + d44f
    t35 = -2 * d15f - d25f + 2 * d35f + d45f

    t40 = 2 * d10f - d20f - 2 * d30f + d40f
    t41 = 2 * d11f - d21f - 2 * d31f + d41f
    t42 = 2 * d12f - d22f - 2 * d32f + d42f
    t43 = 2 * d13f - d23f - 2 * d33f + d43f
    t44 = 2 * d14f - d24f - 2 * d34f + d44f
    t45 = 2 * d15f - d25f - 2 * d35f + d45f

    t50 = 4 * d10f - 5 * d30f + d50f
    t51 = 4 * d11f - 5 * d31f + d51f
    t52 = 4 * d12f - 5 * d32f + d52f
    t53 = 4 * d13f - 5 * d33f + d53f
    t54 = 4 * d14f - 5 * d34f + d54f
    t55 = 4 * d15f - 5 * d35f + d55f

    v00 = 4 * t00 - 5 * t02 + t04
    v01 = -4 * t01 - 4 * t02 + t03 + t04
    v02 = 4 * t01 - 4 * t02 - t03 + t04
    v03 = -2 * t01 - t02 + 2 * t03 + t04
    v04 = 2 * t01 - t02 - 2 * t03 + t04
    v05 = 4 * t01 - 5 * t03 + t05

    v10 = 4 * t10 - 5 * t12 + t14
    v11 = -4 * t11 - 4 * t12 + t13 + t14
    v12 = 4 * t11 - 4 * t12 - t13 + t14
    v13 = -2 * t11 - t12 + 2 * t13 + t14
    v14 = 2 * t11 - t12 - 2 * t13 + t14
    v15 = 4 * t11 - 5 * t13 + t15

    v20 = 4 * t20 - 5 * t22 + t24
    v21 = -4 * t21 - 4 * t22 + t23 + t24
    v22 = 4 * t21 - 4 * t22 - t23 + t24
    v23 = -2 * t21 - t22 + 2 * t23 + t24
    v24 = 2 * t21 - t22 - 2 * t23 + t24
    v25 = 4 * t21 - 5 * t23 + t25

    v30 = 4 * t30 - 5 * t32 + t34
    v31 = -4 * t31 - 4 * t32 + t33 + t34
    v32 = 4 * t31 - 4 * t32 - t33 + t34
    v33 = -2 * t31 - t32 + 2 * t33 + t34
    v34 = 2 * t31 - t32 - 2 * t33 + t34
    v35 = 4 * t31 - 5 * t33 + t35

    v40 = 4 * t40 - 5 * t42 + t44
    v41 = -4 * t41 - 4 * t42 + t43 + t44
    v42 = 4 * t41 - 4 * t42 - t43 + t44
    v43 = -2 * t41 - t42 + 2 * t43 + t44
    v44 = 2 * t41 - t42 - 2 * t43 + t44
    v45 = 4 * t41 - 5 * t43 + t45

    v50 = 4 * t50 - 5 * t52 + t54
    v51 = -4 * t51 - 4 * t52 + t53 + t54
    v52 = 4 * t51 - 4 * t52 - t53 + t54
    v53 = -2 * t51 - t52 + 2 * t53 + t54
    v54 = 2 * t51 - t52 - 2 * t53 + t54
    v55 = 4 * t51 - 5 * t53 + t55

    v_base = V + tile_idx * stride_v_tile + offs_c
    c_store_mask = offs_c < C_pad

    tl.store(v_base + 0 * stride_v_alpha, v00, mask = c_store_mask)
    tl.store(v_base + 1 * stride_v_alpha, v01, mask = c_store_mask)
    tl.store(v_base + 2 * stride_v_alpha, v02, mask = c_store_mask)
    tl.store(v_base + 3 * stride_v_alpha, v03, mask = c_store_mask)
    tl.store(v_base + 4 * stride_v_alpha, v04, mask = c_store_mask)
    tl.store(v_base + 5 * stride_v_alpha, v05, mask = c_store_mask)
    tl.store(v_base + 6 * stride_v_alpha, v10, mask = c_store_mask)
    tl.store(v_base + 7 * stride_v_alpha, v11, mask = c_store_mask)
    tl.store(v_base + 8 * stride_v_alpha, v12, mask = c_store_mask)
    tl.store(v_base + 9 * stride_v_alpha, v13, mask = c_store_mask)
    tl.store(v_base + 10 * stride_v_alpha, v14, mask = c_store_mask)
    tl.store(v_base + 11 * stride_v_alpha, v15, mask = c_store_mask)
    tl.store(v_base + 12 * stride_v_alpha, v20, mask = c_store_mask)
    tl.store(v_base + 13 * stride_v_alpha, v21, mask = c_store_mask)
    tl.store(v_base + 14 * stride_v_alpha, v22, mask = c_store_mask)
    tl.store(v_base + 15 * stride_v_alpha, v23, mask = c_store_mask)
    tl.store(v_base + 16 * stride_v_alpha, v24, mask = c_store_mask)
    tl.store(v_base + 17 * stride_v_alpha, v25, mask = c_store_mask)
    tl.store(v_base + 18 * stride_v_alpha, v30, mask = c_store_mask)
    tl.store(v_base + 19 * stride_v_alpha, v31, mask = c_store_mask)
    tl.store(v_base + 20 * stride_v_alpha, v32, mask = c_store_mask)
    tl.store(v_base + 21 * stride_v_alpha, v33, mask = c_store_mask)
    tl.store(v_base + 22 * stride_v_alpha, v34, mask = c_store_mask)
    tl.store(v_base + 23 * stride_v_alpha, v35, mask = c_store_mask)
    tl.store(v_base + 24 * stride_v_alpha, v40, mask = c_store_mask)
    tl.store(v_base + 25 * stride_v_alpha, v41, mask = c_store_mask)
    tl.store(v_base + 26 * stride_v_alpha, v42, mask = c_store_mask)
    tl.store(v_base + 27 * stride_v_alpha, v43, mask = c_store_mask)
    tl.store(v_base + 28 * stride_v_alpha, v44, mask = c_store_mask)
    tl.store(v_base + 29 * stride_v_alpha, v45, mask = c_store_mask)
    tl.store(v_base + 30 * stride_v_alpha, v50, mask = c_store_mask)
    tl.store(v_base + 31 * stride_v_alpha, v51, mask = c_store_mask)
    tl.store(v_base + 32 * stride_v_alpha, v52, mask = c_store_mask)
    tl.store(v_base + 33 * stride_v_alpha, v53, mask = c_store_mask)
    tl.store(v_base + 34 * stride_v_alpha, v54, mask = c_store_mask)
    tl.store(v_base + 35 * stride_v_alpha, v55, mask = c_store_mask)


@triton.jit
def _winograd_f4x3_output_transform(
    BIAS,
    m_base,
    y_base,
    offs_k,
    k_mask,
    n_valid,
    p_start,
    q_start,
    P: tl.constexpr,
    Q: tl.constexpr,
    stride_m_alpha: tl.constexpr,
    stride_y_p: tl.constexpr,
    stride_y_q: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Load M, apply A-transforms with epilogue, and store one 4x4 tile."""
    m00 = tl.load(m_base + 0 * stride_m_alpha, mask = k_mask, other = 0.0)
    m01 = tl.load(m_base + 1 * stride_m_alpha, mask = k_mask, other = 0.0)
    m02 = tl.load(m_base + 2 * stride_m_alpha, mask = k_mask, other = 0.0)
    m03 = tl.load(m_base + 3 * stride_m_alpha, mask = k_mask, other = 0.0)
    m04 = tl.load(m_base + 4 * stride_m_alpha, mask = k_mask, other = 0.0)
    m05 = tl.load(m_base + 5 * stride_m_alpha, mask = k_mask, other = 0.0)
    m10 = tl.load(m_base + 6 * stride_m_alpha, mask = k_mask, other = 0.0)
    m11 = tl.load(m_base + 7 * stride_m_alpha, mask = k_mask, other = 0.0)
    m12 = tl.load(m_base + 8 * stride_m_alpha, mask = k_mask, other = 0.0)
    m13 = tl.load(m_base + 9 * stride_m_alpha, mask = k_mask, other = 0.0)
    m14 = tl.load(m_base + 10 * stride_m_alpha, mask = k_mask, other = 0.0)
    m15 = tl.load(m_base + 11 * stride_m_alpha, mask = k_mask, other = 0.0)
    m20 = tl.load(m_base + 12 * stride_m_alpha, mask = k_mask, other = 0.0)
    m21 = tl.load(m_base + 13 * stride_m_alpha, mask = k_mask, other = 0.0)
    m22 = tl.load(m_base + 14 * stride_m_alpha, mask = k_mask, other = 0.0)
    m23 = tl.load(m_base + 15 * stride_m_alpha, mask = k_mask, other = 0.0)
    m24 = tl.load(m_base + 16 * stride_m_alpha, mask = k_mask, other = 0.0)
    m25 = tl.load(m_base + 17 * stride_m_alpha, mask = k_mask, other = 0.0)
    m30 = tl.load(m_base + 18 * stride_m_alpha, mask = k_mask, other = 0.0)
    m31 = tl.load(m_base + 19 * stride_m_alpha, mask = k_mask, other = 0.0)
    m32 = tl.load(m_base + 20 * stride_m_alpha, mask = k_mask, other = 0.0)
    m33 = tl.load(m_base + 21 * stride_m_alpha, mask = k_mask, other = 0.0)
    m34 = tl.load(m_base + 22 * stride_m_alpha, mask = k_mask, other = 0.0)
    m35 = tl.load(m_base + 23 * stride_m_alpha, mask = k_mask, other = 0.0)
    m40 = tl.load(m_base + 24 * stride_m_alpha, mask = k_mask, other = 0.0)
    m41 = tl.load(m_base + 25 * stride_m_alpha, mask = k_mask, other = 0.0)
    m42 = tl.load(m_base + 26 * stride_m_alpha, mask = k_mask, other = 0.0)
    m43 = tl.load(m_base + 27 * stride_m_alpha, mask = k_mask, other = 0.0)
    m44 = tl.load(m_base + 28 * stride_m_alpha, mask = k_mask, other = 0.0)
    m45 = tl.load(m_base + 29 * stride_m_alpha, mask = k_mask, other = 0.0)
    m50 = tl.load(m_base + 30 * stride_m_alpha, mask = k_mask, other = 0.0)
    m51 = tl.load(m_base + 31 * stride_m_alpha, mask = k_mask, other = 0.0)
    m52 = tl.load(m_base + 32 * stride_m_alpha, mask = k_mask, other = 0.0)
    m53 = tl.load(m_base + 33 * stride_m_alpha, mask = k_mask, other = 0.0)
    m54 = tl.load(m_base + 34 * stride_m_alpha, mask = k_mask, other = 0.0)
    m55 = tl.load(m_base + 35 * stride_m_alpha, mask = k_mask, other = 0.0)

    s00 = m00 + m10 + m20 + m30 + m40
    s10 = m10 - m20 + 2 * m30 - 2 * m40
    s20 = m10 + m20 + 4 * m30 + 4 * m40
    s30 = m10 - m20 + 8 * m30 - 8 * m40 + m50
    s01 = m01 + m11 + m21 + m31 + m41
    s11 = m11 - m21 + 2 * m31 - 2 * m41
    s21 = m11 + m21 + 4 * m31 + 4 * m41
    s31 = m11 - m21 + 8 * m31 - 8 * m41 + m51
    s02 = m02 + m12 + m22 + m32 + m42
    s12 = m12 - m22 + 2 * m32 - 2 * m42
    s22 = m12 + m22 + 4 * m32 + 4 * m42
    s32 = m12 - m22 + 8 * m32 - 8 * m42 + m52
    s03 = m03 + m13 + m23 + m33 + m43
    s13 = m13 - m23 + 2 * m33 - 2 * m43
    s23 = m13 + m23 + 4 * m33 + 4 * m43
    s33 = m13 - m23 + 8 * m33 - 8 * m43 + m53
    s04 = m04 + m14 + m24 + m34 + m44
    s14 = m14 - m24 + 2 * m34 - 2 * m44
    s24 = m14 + m24 + 4 * m34 + 4 * m44
    s34 = m14 - m24 + 8 * m34 - 8 * m44 + m54
    s05 = m05 + m15 + m25 + m35 + m45
    s15 = m15 - m25 + 2 * m35 - 2 * m45
    s25 = m15 + m25 + 4 * m35 + 4 * m45
    s35 = m15 - m25 + 8 * m35 - 8 * m45 + m55

    y00 = s00 + s01 + s02 + s03 + s04
    y01 = s01 - s02 + 2 * s03 - 2 * s04
    y02 = s01 + s02 + 4 * s03 + 4 * s04
    y03 = s01 - s02 + 8 * s03 - 8 * s04 + s05
    y10 = s10 + s11 + s12 + s13 + s14
    y11 = s11 - s12 + 2 * s13 - 2 * s14
    y12 = s11 + s12 + 4 * s13 + 4 * s14
    y13 = s11 - s12 + 8 * s13 - 8 * s14 + s15
    y20 = s20 + s21 + s22 + s23 + s24
    y21 = s21 - s22 + 2 * s23 - 2 * s24
    y22 = s21 + s22 + 4 * s23 + 4 * s24
    y23 = s21 - s22 + 8 * s23 - 8 * s24 + s25
    y30 = s30 + s31 + s32 + s33 + s34
    y31 = s31 - s32 + 2 * s33 - 2 * s34
    y32 = s31 + s32 + 4 * s33 + 4 * s34
    y33 = s31 - s32 + 8 * s33 - 8 * s34 + s35

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_k, mask = k_mask, other = 0.0).to(tl.float32)
        y00 += bias
        y01 += bias
        y02 += bias
        y03 += bias
        y10 += bias
        y11 += bias
        y12 += bias
        y13 += bias
        y20 += bias
        y21 += bias
        y22 += bias
        y23 += bias
        y30 += bias
        y31 += bias
        y32 += bias
        y33 += bias

    y00 = _apply_activation_from_str(y00, ACTIVATION)
    y01 = _apply_activation_from_str(y01, ACTIVATION)
    y02 = _apply_activation_from_str(y02, ACTIVATION)
    y03 = _apply_activation_from_str(y03, ACTIVATION)
    y10 = _apply_activation_from_str(y10, ACTIVATION)
    y11 = _apply_activation_from_str(y11, ACTIVATION)
    y12 = _apply_activation_from_str(y12, ACTIVATION)
    y13 = _apply_activation_from_str(y13, ACTIVATION)
    y20 = _apply_activation_from_str(y20, ACTIVATION)
    y21 = _apply_activation_from_str(y21, ACTIVATION)
    y22 = _apply_activation_from_str(y22, ACTIVATION)
    y23 = _apply_activation_from_str(y23, ACTIVATION)
    y30 = _apply_activation_from_str(y30, ACTIVATION)
    y31 = _apply_activation_from_str(y31, ACTIVATION)
    y32 = _apply_activation_from_str(y32, ACTIVATION)
    y33 = _apply_activation_from_str(y33, ACTIVATION)

    if n_valid:
        for r in tl.static_range(4):
            p = p_start + r
            if p < P:
                for s in tl.static_range(4):
                    q = q_start + s
                    if q < Q:
                        if r == 0:
                            if s == 0:
                                val = y00
                            elif s == 1:
                                val = y01
                            elif s == 2:
                                val = y02
                            else:
                                val = y03
                        elif r == 1:
                            if s == 0:
                                val = y10
                            elif s == 1:
                                val = y11
                            elif s == 2:
                                val = y12
                            else:
                                val = y13
                        elif r == 2:
                            if s == 0:
                                val = y20
                            elif s == 1:
                                val = y21
                            elif s == 2:
                                val = y22
                            else:
                                val = y23
                        else:
                            if s == 0:
                                val = y30
                            elif s == 1:
                                val = y31
                            elif s == 2:
                                val = y32
                            else:
                                val = y33
                        tl.store(
                            y_base + p * stride_y_p + q * stride_y_q,
                            val,
                            mask = k_mask,
                        )
