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

import triton
import triton.language as tl
import torch 


BLOCK_SIZE = 1024

@triton.jit
def _seeded_dropout(
    x_ptr: tl.intptr,  # Pointer to the input tensor
    output_ptr: tl.intptr,  # Pointer to the output tensor
    n_elements: int,  # Number of elements in the input tensor
    p: float,  # Dropout probability
    seed: int,  # Seed for random number generation
    BLOCK_SIZE: tl.constexpr,  # Block size, a compile-time constant
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4

    off0 = block_start + BLOCK_SIZE * 0 + tl.arange(0, BLOCK_SIZE)
    off1 = block_start + BLOCK_SIZE * 1 + tl.arange(0, BLOCK_SIZE)
    off2 = block_start + BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE)
    off3 = block_start + BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE)

    mask0 = off0 < n_elements
    mask1 = off1 < n_elements
    mask2 = off2 < n_elements
    mask3 = off3 < n_elements

    x0 = tl.load(x_ptr + off0, mask = mask0)
    x1 = tl.load(x_ptr + off1, mask = mask1)
    x2 = tl.load(x_ptr + off2, mask = mask2)
    x3 = tl.load(x_ptr + off3, mask = mask3)

    r0, r1, r2, r3 = tl.random.rand4x(seed, off0)
    keep0, keep1, keep2, keep3 = r0 > p, r1 > p, r2 > p, r3 > p

    o0 = tl.where(keep0, x0 / (1 - p), 0.0)
    o1 = tl.where(keep1, x1 / (1 - p), 0.0)
    o2 = tl.where(keep2, x2 / (1 - p), 0.0)
    o3 = tl.where(keep3, x3 / (1 - p), 0.0)

    tl.store(output_ptr + off0, o0, mask = mask0)
    tl.store(output_ptr + off1, o1, mask = mask1)
    tl.store(output_ptr + off2, o2, mask = mask2)
    tl.store(output_ptr + off3, o3, mask = mask3)


def seeded_dropout(
    x: torch.Tensor, 
    p: float, 
    seed: int, 
    BLOCK_SIZE: int = 1024
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous(), "Tensor must be on GPU and stored in contiguous block!"
    output = torch.empty_like(x)
    n_elements = x.numel()
    # Define the grid size based on the number of elements and BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE * 4),)
    # Launch the kernel with the grid configuration
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=BLOCK_SIZE)  # Pass BLOCK_SIZE as a named argument
    return output
