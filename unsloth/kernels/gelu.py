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

def gelu_forward_triton(x: torch.Tensor):
    # Assuming x.shape is [batch_size, seq_len, feature_dim]
    batch_size, seq_len, feature_dim = x.shape
    # Flatten the first two dimensions
    x_flattened = x.view(-1, feature_dim)
    y_flattened = torch.empty_like(x_flattened)

    # Compute the number of blocks needed for the flattened 2D tensor
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(x_flattened.shape[0], BLOCK_SIZE)

    # Launch the kernel with the adjusted grid configuration
    _gelu_forward_kenel[(num_blocks,)](
        output_ptr=y_flattened.data_ptr(),
        input_ptr=x_flattened.data_ptr(),
        n_elements=x_flattened.stride(0),
        n_cols=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape y_flattened back to the original 3D shape
    y = y_flattened.view(batch_size, seq_len, feature_dim)
    return y



def gelu_backward_triton(x: torch.Tensor, grad_output: torch.Tensor):
    n_rows, n_cols = x.shape
    grad_input = torch.empty_like(x)

    # Define the grid of blocks
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_rows, BLOCK_SIZE)

    # Launch the kernel with the grid configuration
    _gelu_backward_kernel[(num_blocks,)](
        grad_input_ptr=grad_input.data_ptr(),
        input_ptr=x.data_ptr(),
        grad_output_ptr=grad_output.data_ptr(),
        n_elements=x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_input


@triton.jit 
def _gelu_forward_kenel(output_ptr, input_ptr, n_elements, n_cols: tl.int32, BLOCK_SIZE : tl.constexpr,): 
    '''
    Triton kernel for the forward pass of a GeLU function based off the equation 
    https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

    output_ptr : the pointer for the first memory adress of the output tensor
    input_ptr : the pointer for the input of the first element of the first row of the input tensor.
    n_elements : 
    '''
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_elements

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0)

    output_values = gelu_operation(x=row)

    output_row_start_ptr = output_ptr + row_idx * n_elements
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output_values, mask=mask)
pass 


@triton.jit
def _gelu_backward_kernel(grad_input_ptr, input_ptr, grad_output_ptr, n_elements: tl.int32, n_cols: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)

    # Compute pointers to the start of the row for input, gradient output, and gradient input
    row_start_input_ptr = input_ptr + row_idx * n_elements
    row_start_grad_output_ptr = grad_output_ptr + row_idx * n_elements
    row_start_grad_input_ptr = grad_input_ptr + row_idx * n_elements

    # Iterate over the columns of the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_input_ptr + col_offsets
    grad_output_ptrs = row_start_grad_output_ptr + col_offsets
    grad_input_ptrs = row_start_grad_input_ptr + col_offsets

    # Mask to avoid out-of-bounds memory access
    mask = col_offsets < n_cols

    # Load input and gradient output values
    x = tl.load(input_ptrs, mask=mask, other=0)
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0)

    # Compute the GELU backward operation using gelu_bwd_pass
    dgelu_dx = gelu_bwd_pass(x)
    grad_input = dgelu_dx * grad_output

    # Store the computed gradient input
    tl.store(grad_input_ptrs, grad_input, mask=mask)
pass 



@triton.jit
def gelu_operation(x): 
    PI = 3.141592653589793  # Define Pi
    coefficient = tl.sqrt(2/PI) * (x + 0.044715 * tl.math.pow(x, 3))
    # Compute the tanh term
    tanh_term = tl.math.tanh(coefficient)
    # Compute the final GELU response
    response = 0.5 * x * (1 + tanh_term)
    return response


@triton.jit 
def gelu_bwd_pass(x):
    """
    https://github.com/unslothai/unsloth/pull/97
    """
    PI = 3.141592653589793
    sqrt_2_over_pi = tl.sqrt(2 / PI)
    x_cubed = 0.044715 * tl.math.pow(x, 3)
    

    tanh_term = tl.math.tanh(sqrt_2_over_pi * (x + x_cubed))
    sech_term = sech_operation(sqrt_2_over_pi * (x + x_cubed))
    sech_squared = tl.math.pow(sech_term, 2)
    
    first_part = 0.5 * (1 + tanh_term) * sqrt_2_over_pi * (1 + 3 * 0.044715 * tl.math.pow(x, 2))
    
    # Compute the second part of the derivative
    second_part = 0.5 * x * sech_squared
    
    # Combine both parts to get the full derivative
    dgelu_dx = first_part + second_part
    return dgelu_dx


@triton.jit
def sech_operation(x):
    # Calculate sech(x) = 2 / (exp(x) + exp(-x))
    tanh = tl.math.tanh(x)
    tanh_squared = tl.math.pow(tanh, 2)
    sech = tl.math.sqrt(1-tanh_squared)
    return sech