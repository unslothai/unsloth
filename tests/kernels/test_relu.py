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

from unsloth.kernels.relu import relu_kernel
from tests.kernels.conftest import set_seed, assert_all_close

@set_seed
@pytest.fixture(params=[(100, 100), (1024, 1024), (5000, 1024), (12345, 5678)])
def test_matrix(request):
    shape = request.param
    x = torch.randn(shape, device='cuda')
    return x

# Test function
def test_relu_kernel(test_matrix):
    # Apply your Triton-based ReLU kernel
    triton_output = relu_kernel(test_matrix)

    # Apply PyTorch's ReLU for comparison
    torch_relu = torch.nn.ReLU()
    torch_output = torch_relu(test_matrix)

    # Check if the outputs are close enough using assert_all_close
    assert_all_close(triton_output, torch_output, rtol=1e-05, atol=1e-08)
