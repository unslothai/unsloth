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

@triton.jit 
def gelu_forward_kenel(output_ptr: tl.pointer, input_ptr: tl.pointer, n_elements: tl.int32, n_cols: tl.int32, BLOCK_SIZE : tl.constexpr,): 
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
def gelu_backward_kernel(output_ptr: tl.pointer, input_ptr: tl.pointer, n_elements: tl.int32, n_cols: tl.int32, BLOCK_SIZE : tl.constexpr,): 
    '''
    Triton kernel for the backward pass of a GeLU function based off eq [13] 
    https://arxiv.org/pdf/2305.12073.pdf

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

    output_values = gelu_bwd_pass(x=row)

    output_row_start_ptr = output_ptr + row_idx * n_elements
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output_values, mask=mask)
pass 


@triton.jit
def gelu_operation(x): 
    PI = 3.141592653589793  # Define Pi
    coefficient = tl.sqrt(2/PI) * (x + 0.044715 * tl.pow(x, 3))
    # Compute the tanh term
    tanh_term = tanh_operation(coefficient)
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
    x_cubed = 0.044715 * tl.pow(x, 3)
    

    tanh_term = tanh_operation(sqrt_2_over_pi * (x + x_cubed))
    sech_term = sech_operation(sqrt_2_over_pi * (x + x_cubed))
    sech_squared = tl.pow(sech_term, 2)
    
    first_part = 0.5 * (1 + tanh_term) * sqrt_2_over_pi * (1 + 3 * 0.044715 * tl.pow(x, 2))
    
    # Compute the second part of the derivative
    second_part = 0.5 * x * sech_squared
    
    # Combine both parts to get the full derivative
    dgelu_dx = first_part + second_part
    return dgelu_dx


@triton.jit
def sech_operation(x):
    # Calculate sech(x) = 2 / (exp(x) + exp(-x))
    exp_x = tl.exp(x)
    exp_minus_x = tl.exp(-x)
    sech_x = 2 / (exp_x + exp_minus_x)

    # Handle potential numerical instabilities
    # For large positive x, exp(x) dominates, and sech(x) approaches 0.
    # For large negative x, exp(-x) dominates, and sech(x) again approaches 0.
    # These cases are handled naturally by the above formula.
    return sech_x


@triton.jit
def tanh_operation(x):
    # Handle large positive values
    pos_mask = x > 0
    exp2x = tl.exp(x)
    tanh_pos = (exp2x - 1) / (exp2x + 1)

    # Handle large negative values
    neg_mask = ~pos_mask
    exp_minus_2x = tl.exp(-2 * x)
    tanh_neg = (1 - exp_minus_2x) / (1 + exp_minus_2x)

    # Combine results using masks
    tanh_x = tl.where(pos_mask, tanh_pos, tanh_neg)
    return tanh_x

