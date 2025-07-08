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
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _fg_kernel(e: tl.tensor, g: tl.tensor, h: tl.tensor, n_elements: int, BLOCK_SIZE : tl.constexpr,) -> None:
    """
    Triton kernel to compute the SwiGLU activation function.
    
    The SwiGLU (Swish-Gated Linear Unit) operation computes:
        f = e * sigmoid(e)  # Swish activation
        h = f * g           # Gated output
    
    This is commonly used in transformer MLP blocks where e and g are 
    projections of the input, and the gating mechanism helps with gradient flow.
    
    Args:
        e (`tl.tensor`):
            Input tensor e (gate input).
        g (`tl.tensor`):
            Input tensor g (up projection).
        h (`tl.tensor`):
            Output tensor h where results are stored.
        n_elements (`int`):
            Number of elements in the tensors.
        BLOCK_SIZE (`tl.constexpr`):
            Block size for parallel processing.
    
    Returns:
        None: Results are written directly to tensor h.
    """
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def swiglu_fg_kernel(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Compute the SwiGLU activation function using Triton.
    
    Args:
        e (`torch.Tensor`):
            Input tensor e.
        g (`torch.Tensor`):
            Input tensor g.
    
    Returns:
        `torch.Tensor`: Output tensor h.
    """
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_gpu_device(e.device):
        _fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024,)
    return h
pass


@triton.jit
def _DWf_DW_dfg_kernel(DW: tl.tensor, e: tl.tensor, g: tl.tensor, n_elements: int, BLOCK_SIZE : tl.constexpr,) -> None:
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)#.to(tl.float32)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)#.to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row) # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row  =  f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row,  mask = mask) # h  = f * g
    tl.store(e  + offsets, df_row, mask = mask) # df = DW * f
    tl.store(g  + offsets, de_row, mask = mask) # de
pass


def swiglu_DWf_DW_dfg_kernel(DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SwiGLU activation function and its derivatives using Triton.
    
    Args:
        DW (`torch.Tensor`):
            Input tensor DW.
        e (`torch.Tensor`):
            Input tensor e.
        g (`torch.Tensor`):
            Input tensor g.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors DW, e, g.
    """
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_gpu_device(e.device):
        _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE = 1024,)
    return DW, e, g
pass
