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


import pytest
import torch

from unsloth.kernels.gelu import gelu_forward_kenel, gelu_backward_kernel
from tests.kernels.conftest import set_seed, assert_all_close

@set_seed
@pytest.fixture(params=[(100, 100), (1024, 1024), (5000, 1024), (12345, 5678)])
def test_matrix(request):
    shape = request.param
    x = torch.randn(shape, device='cuda')
    return x

# Test function
def test_relu_kernel_fwd(test_matrix):
    # Apply your Triton-based ReLU kernel
    triton_output = gelu_forward_kenel(test_matrix)
    
    # Apply PyTorch's ReLU for comparison
    torch_gelu = torch.nn.GELU()
    torch_output = torch_gelu(test_matrix)

    # Check if the outputs are close enough using assert_all_close
    assert_all_close(triton_output, torch_output, rtol=1e-05, atol=1e-08)


# Test function for GeLU backward kernel
def test_gelu_backward_kernel(test_matrix):
    # Create a tensor representing gradients (e.g., random gradients)
    grad_input = torch.randn_like(test_matrix)

    # Apply your Triton-based GeLU backward kernel
    triton_output = gelu_backward_kernel(test_matrix, grad_input)

    # Compute PyTorch's GeLU gradient for comparison
    torch_gelu = torch.nn.GELU()
    torch_output = torch.autograd.grad(torch_output.sum(), test_matrix, grad_outputs=grad_input)[0]

    # Check if the outputs are close enough using assert_all_close
    assert_all_close(triton_output, torch_output, rtol=1e-05, atol=1e-08)