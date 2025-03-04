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
from .utils import (
    calculate_settings,
    triton_tanh,
    torch_cuda_device,
)


@triton.jit
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    # h = f * up
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def geglu_exact_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    device = gate.device
    out = torch.empty((batch, seq_len, hd), dtype = gate.dtype, device = device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(device):
        _exact_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE = 1024,)
    return out
pass


@triton.jit
def _exact_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    h = f * up

    df/de (with help of Wolfram :)
    df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)

    Reuse via
    f =        1/2 * (1 + erf(1/sqrt(2) * e)) * e
    """
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)#.to(tl.float32)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)#.to(tl.float32)

    # Break e_row away for re-use
    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row  =  f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row

    # df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)
    t = 0.3989422804014327 # 1/sqrt(2*pi)
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row,  mask = mask) # h  = f * g
    tl.store(e  + offsets, df_row, mask = mask) # df = DW * f
    tl.store(g  + offsets, de_row, mask = mask) # de
pass


def geglu_exact_backward_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(e.device):
        _exact_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return DW, e, g
pass


@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))
    # h = f * up
    s = 0.7978845608028654 # math.sqrt(2 / math.pi)
    
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    f_row = 0.5 * e_row * (
        triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) \
        + 1.0
    )
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def geglu_approx_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    device = gate.device
    out = torch.empty((batch, seq_len, hd), dtype = gate.dtype, device = device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(device):
        _approx_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE = 1024,)
    return out
pass


@triton.jit
def _approx_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    """
    f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))
    h = f * up

    df/de (with help from https://arxiv.org/pdf/2305.12073.pdf :))
    df/de = 1/2 * [1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )] +
            1/2 * sech^2 [   sqrt(2/pi) * x * (1 + 0.044715 * x^2 )  ] * \
                           ( sqrt(2/pi) * x * (1 + 0.044715 * x^2 * 3 ) )

    Notice sech^2(x) = 1 - tanh^2(x)
    So reuse tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )

    See https://www.desmos.com/calculator/nqprfoni6x
    """
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)#.to(tl.float32)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)#.to(tl.float32)

    # See https://www.desmos.com/calculator/nqprfoni6x
    s = 0.7978845608028654 # math.sqrt(2 / math.pi)
    a = s * e_row # a = sqrt(2 / pi) * x
    b = a * 0.044715 * e_row * e_row # b = a * 0.044715 * x^2
    T = 1.0 + triton_tanh(a + b)
    T2 = 0.5 * T
    # Q = 0.5 * -T * (T - 2.0) * (a + 3.0 * b)
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b) 
    df_de = T2 + Q2 # 1/2 * (T + Q)

    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    f_row = T2 * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row  =  f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row,  mask = mask) # h  = f * g
    tl.store(e  + offsets, df_row, mask = mask) # df = DW * f
    tl.store(g  + offsets, de_row, mask = mask) # de
pass


def geglu_approx_backward_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(e.device):
        _approx_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return DW, e, g
pass
