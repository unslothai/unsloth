import pytest
import torch
import triton

from unsloth.kernels.layernorm import LayerNorm
from tests.conftest import set_seed, assert_all_close

# Fixture for test matrices and associated parameters
@set_seed()
@pytest.fixture(params=[(64, 64), (1024, 512), (2048, 1024)])
def test_data(request):
    torch.manual_seed(0)  # For reproducibility
    batch_size, num_features = request.param
    x = torch.randn(batch_size, num_features, device='cuda')
    weight = torch.randn(num_features, device='cuda')
    bias = torch.randn(num_features, device='cuda')
    eps = 1e-5
    return x, weight, bias, eps

# Test forward pass
def test_layer_norm_forward(test_data):
    x, weight, bias, eps = test_data

    # Triton layer norm forward
    triton_output = LayerNorm.apply(x, x.size(), weight, bias, eps)

    # PyTorch layer norm forward
    pytorch_layer_norm = torch.nn.LayerNorm(x.size()[1:], eps=eps, elementwise_affine=True)
    pytorch_layer_norm.weight = torch.nn.Parameter(weight)
    pytorch_layer_norm.bias = torch.nn.Parameter(bias)
    pytorch_output = pytorch_layer_norm(x)

    # Check if outputs are close using assert_all_close
    assert_all_close(triton_output, pytorch_output, rtol=1e-05, atol=1e-08)


def test_layer_norm_backward(test_data):
    x, weight, bias, eps = test_data
    x.requires_grad = True

    # Triton layer norm backward
    triton_output = LayerNorm.apply(x, x.size(), weight, bias, eps)
    triton_grad = torch.autograd.grad(triton_output.sum(), x)[0]

    # PyTorch layer norm backward
    pytorch_layer_norm = torch.nn.LayerNorm(x.size()[1:], eps=eps, elementwise_affine=True)
    pytorch_layer_norm.weight = torch.nn.Parameter(weight)
    pytorch_layer_norm.bias = torch.nn.Parameter(bias)
    pytorch_output = pytorch_layer_norm(x)
    pytorch_output.sum().backward()
    pytorch_grad = x.grad

    # Check if gradients are close using assert_all_close
    assert_all_close(triton_grad, pytorch_grad, rtol=1e-05, atol=1e-08)
