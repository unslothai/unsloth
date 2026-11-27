# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/utils/conv_config_utils.py and aiter/ops/triton/configs/gfx1151/triton/conv/conv3d_*/DEFAULT.json (PR #6973); gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: the gfx1151 conv3d JSON tables inlined (NDHWC table unused, dropped); the same four-tier walk as get_conv_config().
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

"""Conv3D kernel configs: the gfx1151 tables of ROCm/aiter PR #6973 and the four-tier lookup of aiter's
``get_conv_config()`` (variant exact shape, exact shape, ``M_LEQ_<n>`` bucket, ``any``). Other gfx11 archs (explicit
opt-in only) borrow the gfx1151 tables."""

import functools

import triton

CONV_STANDARD_M_BOUNDS: tuple[int, ...] = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)

TABLES = {
    "gfx1151": {
        "CONV3D-1X1X1": {
            "any": {
                "BLOCK_K": 64,
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "GROUP_SIZE_M": 4,
                "num_stages": 1,
                "num_warps": 4,
            },
            "shapes_ncdhw": {
                "N=1,C=1024,D=1,H=176,W=320,K=512,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=1024,D=4,H=176,W=320,K=512,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=160,D=1,H=176,W=320,K=320,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=160,D=4,H=176,W=320,K=320,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=320,D=2,H=88,W=160,K=640,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=512,D=1,H=352,W=640,K=256,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=512,D=4,H=352,W=640,K=256,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
            },
            "shapes_ndhwc": {
                "N=1,C=160,D=1,H=176,W=320,K=320,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 2,
                },
                "N=1,C=160,D=4,H=176,W=320,K=320,T=1,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 2,
                },
            },
        },
        "CONV3D-3X3X3-CBLOCKED": {
            "any": {
                "BLOCK_K": 32,
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 2,
                "num_warps": 4,
            },
            "shapes": {
                "N=1,C=1024,D=34,H=34,W=60,K=128,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=1,pw=1,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=128,D=123,H=136,W=240,K=48,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=1,pw=1,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=160,D=3,H=354,W=642,K=160,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=256,D=3,H=354,W=642,K=12,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=256,D=6,H=354,W=642,K=12,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
            },
        },
        "CONV3D-GENERAL": {
            "any": {
                "BLOCK_K": 64,
                "BLOCK_M": 32,
                "BLOCK_N": 128,
                "GROUP_SIZE_M": 8,
                "num_stages": 1,
                "num_warps": 4,
            },
            "shapes_ncdhw": {
                "N=1,C=1024,D=3,H=44,W=80,K=2048,T=3,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 256,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 8,
                },
                "N=1,C=1024,D=4,H=88,W=160,K=2048,T=3,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 8,
                },
                "N=1,C=12,D=3,H=354,W=642,K=160,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=12,D=6,H=354,W=642,K=160,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=3,D=4,H=224,W=224,K=1408,T=1,R=14,S=14,sd=1,sh=14,sw=14,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=320,D=5,H=88,W=160,K=320,T=3,R=1,S=1,sd=2,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=1,C=640,D=3,H=44,W=80,K=640,T=3,R=1,S=1,sd=2,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=2304,C=3,D=2,H=16,W=16,K=1024,T=2,R=16,S=16,sd=2,sh=16,sw=16,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 2,
                },
                "N=2304,C=3,D=2,H=16,W=16,K=1152,T=2,R=16,S=16,sd=2,sh=16,sw=16,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=256,C=3,D=2,H=14,W=14,K=1280,T=2,R=14,S=14,sd=2,sh=14,sw=14,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
            },
            "shapes_ndhwc": {
                "N=1,C=1024,D=3,H=44,W=80,K=2048,T=3,R=1,S=1,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 1,
                    "num_warps": 8,
                },
                "N=1,C=3,D=4,H=224,W=224,K=1408,T=1,R=14,S=14,sd=1,sh=14,sw=14,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
                "N=2304,C=3,D=2,H=16,W=16,K=1024,T=2,R=16,S=16,sd=2,sh=16,sw=16,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=2304,C=3,D=2,H=16,W=16,K=1152,T=2,R=16,S=16,sd=2,sh=16,sw=16,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=256,C=3,D=2,H=14,W=14,K=1280,T=2,R=14,S=14,sd=2,sh=14,sw=14,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                },
            },
        },
        "CONV3D-PREPACK": {
            "any": {"BLOCK_C": 64, "BLOCK_M": 16, "num_stages": 1, "num_warps": 4},
            "shapes": {
                "N=1,C=128,D=123,H=136,W=240,CB=64": {
                    "BLOCK_C": 32,
                    "BLOCK_M": 128,
                    "num_stages": 1,
                    "num_warps": 8,
                    "waves_per_eu": 2,
                },
                "N=1,C=160,D=3,H=354,W=642,CB=64": {
                    "BLOCK_C": 32,
                    "BLOCK_M": 128,
                    "num_stages": 1,
                    "num_warps": 8,
                    "waves_per_eu": 2,
                },
            },
        },
        "CONV3D-WINO-HW-F4X3-GEMM": {
            "any": {
                "BLOCK_K": 64,
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 1,
                "num_warps": 2,
            },
            "shapes": {
                "N=1,C=320,D=3,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 32,
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=384,D=4,H=90,W=100,K=384,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=640,D=3,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 64,
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "GROUP_SIZE_M": 4,
                    "num_stages": 1,
                    "num_warps": 4,
                },
            },
        },
        "CONV3D-WINO-HW-F4X3-INPUT": {
            "any": {"BLOCK_C": 256, "num_stages": 1, "num_warps": 4},
            "shapes_ncdhw": {
                "N=1,C=160,D=3,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=160,D=6,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=160,D=6,H=354,W=642,K=160,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 64,
                    "num_stages": 1,
                    "num_warps": 2,
                    "waves_per_eu": 2,
                },
                "N=1,C=192,D=6,H=178,W=198,K=192,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 64,
                    "num_stages": 1,
                    "num_warps": 2,
                    "waves_per_eu": 2,
                },
                "N=1,C=256,D=6,H=354,W=642,K=256,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=320,D=3,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=320,D=3,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 64,
                    "num_stages": 1,
                    "num_warps": 2,
                    "waves_per_eu": 2,
                },
                "N=1,C=320,D=4,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=320,D=6,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
                "N=1,C=384,D=4,H=90,W=100,K=384,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 64,
                    "num_stages": 1,
                    "num_warps": 2,
                    "waves_per_eu": 2,
                },
                "N=1,C=512,D=6,H=354,W=642,K=256,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 64,
                    "num_stages": 1,
                    "num_warps": 8,
                    "waves_per_eu": 2,
                },
                "N=1,C=640,D=3,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_C": 128,
                    "num_stages": 1,
                    "num_warps": 4,
                },
            },
        },
        "CONV3D-WINO-HW-F4X3-OUTPUT": {
            "any": {"BLOCK_K": 256, "num_stages": 1, "num_warps": 8},
            "shapes": {
                "N=1,C=160,D=6,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 128,
                    "num_stages": 1,
                    "num_warps": 8,
                },
                "N=1,C=320,D=3,H=90,W=162,K=640,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 128,
                    "num_stages": 1,
                    "num_warps": 8,
                    "waves_per_eu": 2,
                },
                "N=1,C=320,D=6,H=178,W=322,K=320,T=3,R=3,S=3,sd=1,sh=1,sw=1,pd=0,ph=0,pw=0,dd=1,dh=1,dw=1": {
                    "BLOCK_K": 128,
                    "num_stages": 1,
                    "num_warps": 8,
                },
            },
        },
    }
}


def format_shape_key_3d(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    K: int,
    T: int,
    R: int,
    S: int,
    sd: int,
    sh: int,
    sw: int,
    pd: int,
    ph: int,
    pw: int,
    dd: int,
    dh: int,
    dw: int,
) -> str:
    """Canonical string key for a user-visible Conv3D call."""
    return (
        f"N={N},C={C},D={D},H={H},W={W},K={K},T={T},R={R},S={S},"
        f"sd={sd},sh={sh},sw={sw},pd={pd},ph={ph},pw={pw},"
        f"dd={dd},dh={dh},dw={dw}"
    )


def format_prepack_shape_key_3d(N: int, C: int, D: int, H: int, W: int, CB: int) -> str:
    """Canonical key for an NCDHW-to-NCDHWc activation pack."""
    return f"N={N},C={C},D={D},H={H},W={W},CB={CB}"


@functools.lru_cache(maxsize = None)
def get_arch() -> str:
    return str(triton.runtime.driver.active.get_current_target().arch).split(":")[0]


def tables_for(arch: str):
    if arch in TABLES:
        return TABLES[arch]
    if arch.startswith("gfx11"):
        return TABLES["gfx1151"]
    return None


def _table(config_name: str):
    tables = tables_for(get_arch())
    return None if tables is None else tables.get(config_name)


def has_conv_config(config_name: str) -> bool:
    """Return whether the running architecture ships this table."""
    return _table(config_name) is not None


@functools.lru_cache(maxsize = 512)
def _get_conv_config_cached(
    config_name: str, shape_key: str | None, M: int | None, variants: tuple[str, ...]
) -> dict:
    """Config walk: variant shape entries, generic shape, M bucket, any."""
    config_dict = _table(config_name)
    if config_dict is None:
        raise KeyError(f"No conv config '{config_name}' on arch {get_arch()}")

    if shape_key is not None:
        for variant in variants:
            shapes = config_dict.get(f"shapes_{variant}", {})
            if shape_key in shapes:
                return shapes[shape_key]

    shapes = config_dict.get("shapes", {})
    if shape_key is not None and shape_key in shapes:
        return shapes[shape_key]

    if M is not None and M >= 0:
        for bound in CONV_STANDARD_M_BOUNDS:
            key = f"M_LEQ_{bound}"
            if M <= bound and key in config_dict:
                return config_dict[key]

    if "any" in config_dict:
        return config_dict["any"]

    raise KeyError(
        f"No matching config in '{config_name}' for shape_key={shape_key!r}, "
        f"M={M} on arch {get_arch()} (no literal shape, no bucket, no 'any' fallback)."
    )


def get_conv_config(
    config_name: str,
    shape_key: str | None = None,
    M: int | None = None,
    variants: tuple[str, ...] = (),
) -> dict:
    """Load a conv kernel config for the running GPU arch (a fresh shallow copy)."""
    config = _get_conv_config_cached(config_name, shape_key, M, variants)
    return dict(config)
