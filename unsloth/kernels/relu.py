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
def _relu_kernel(output_ptr, input_ptr, n_elements, n_cols, BLOCK_SIZE : tl.constexpr,): 
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_elements

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    #x = max(0,x)
    row = tl.load(input_ptrs, mask=mask, other=0)
    row_relu = row * (row>0)

    output_row_start_ptr = output_ptr + row_idx * n_elements
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, row_relu, mask=mask)
pass 

def relu_kernel(x: torch.Tensor):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    # Define the grid of blocks
    # Here, we divide the number of rows by the block size to determine the number of blocks needed
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_rows, BLOCK_SIZE)

    # Launch the kernel with the grid configuration
    _relu_kernel[(num_blocks,)](
        output_ptr=y.data_ptr(),
        input_ptr=x.data_ptr(),
        n_elements=x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
