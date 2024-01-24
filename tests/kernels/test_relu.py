import pytest
import torch
import triton

from unsloth.kernels.relu import relu_kernel
from tests.conftest import set_seed, assert_all_close

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
