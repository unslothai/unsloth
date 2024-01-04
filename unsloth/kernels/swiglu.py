# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings


@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0).to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row / (1 + tl.exp(-e_row))
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = "cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024,)
    return h
pass


@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0).to(tl.float32)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0).to(tl.float32)

    # f = e * sigmoid(e)
    se_row = 1 / (1 + tl.exp(-e_row))
    # f = e * se
    f_row = e_row * se_row
    # h = f * g
    h_row = f_row * g_row
    # DW_f = DW * f
    DWf_row = DW_row * f_row
    # DW_dfg = DW * (se*(g - h) + h)
    DW_dfg_row = DW_row * (se_row*(g_row - h_row) + h_row)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row,      mask = mask)
    tl.store(e  + offsets, DWf_row,    mask = mask)
    tl.store(g  + offsets, DW_dfg_row, mask = mask)
pass


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return DW, e, g
pass
