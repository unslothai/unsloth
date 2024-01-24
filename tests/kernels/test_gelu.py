import pytest
import torch
import triton

from unsloth.kernels.gelu import gelu_forward_kenel, gelu_backward_kenel
from tests.conftest import set_seed, assert_all_close

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
    triton_output = gelu_backward_kenel(test_matrix, grad_input)

    # Compute PyTorch's GeLU gradient for comparison
    torch_gelu = torch.nn.GELU()
    torch_output = torch.autograd.grad(torch_output.sum(), test_matrix, grad_outputs=grad_input)[0]

    # Check if the outputs are close enough using assert_all_close
    assert_all_close(triton_output, torch_output, rtol=1e-05, atol=1e-08)