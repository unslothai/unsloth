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
import torch.nn as nn

from unsloth.kernels.layernorm import fast_layernorm_inference
from tests.kernels.conftest import set_seed, assert_all_close


def assert_all_close(a: torch.Tensor, b: torch.Tensor, rtol=0, atol=1e-1) -> None:
    """
    Check that all elements of tensors a and b are within provided thresholds.
    """
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} != {b.dtype}"
    assert a.device == b.device, f"Devices don't match: {a.device} != {b.device}"
    max_abs_diff = torch.max(torch.abs(a - b))
    rel_diff = torch.abs(a / b)
    max_rel_diff = torch.max(rel_diff)
    mismatch_elements = torch.sum(torch.abs(a - b) > atol + rtol * torch.abs(b))
    nb_elements = torch.numel(a)
    msg = (
        f"Differences: "
        f"{max_abs_diff:.3f} (max abs), "
        f"{max_rel_diff:.3f} (max rel), "
        f"{mismatch_elements}/{nb_elements} (mismatch elements)"
    )
    assert torch.allclose(a, b, rtol=rtol, atol=atol), msg

# Fixture for test matrices and associated parameters
@set_seed()
@pytest.fixture(params=[(64, 64), (1024, 512), (2048, 1024)])
def test_data(request):
    batch_size, num_features = request.param
    x = torch.randn(batch_size, num_features, device='cuda')
    weight = torch.randn(num_features, device='cuda')
    bias = torch.randn(num_features, device='cuda')
    eps = 1e-5
    return x, weight, bias, eps

# Test forward pass
@set_seed()
def test_fast_layernorm_inference_forward(test_data):
    x, weight, bias, eps = test_data

    # Using fast_layernorm_inference for the forward pass
    normalized_shape = x.size()[1:]
    triton_output = fast_layernorm_inference(x, normalized_shape, eps=eps, use_triton=True)

    # PyTorch layer norm forward pass for comparison
    pytorch_layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    pytorch_layer_norm.weight = nn.Parameter(weight)
    pytorch_layer_norm.bias = nn.Parameter(bias)
    pytorch_output = pytorch_layer_norm(x)

    # Assert that the outputs are close
    assert_all_close(triton_output, pytorch_output)



# Test function for backward pass
@set_seed()
def test_layer_norm_backward(test_data):
    x, weight, bias, eps = test_data
    x.requires_grad = True

    # Custom Triton layer norm backward pass
    normalized_shape = x.size()[1:]

    triton_output = fast_layernorm_inference(x, normalized_shape, eps=eps, use_triton=True)

    triton_grad = torch.autograd.grad(triton_output.sum(), x, retain_graph=True)[0]

    # Reset gradients
    x.grad = None

    # PyTorch layer norm backward pass
    pytorch_layer_norm = nn.LayerNorm(x.size()[1:], eps=eps, elementwise_affine=True)
    pytorch_layer_norm.weight = nn.Parameter(weight)
    pytorch_layer_norm.bias = nn.Parameter(bias)
    pytorch_output = pytorch_layer_norm(x)
    pytorch_output.sum().backward()
    pytorch_grad = x.grad

    # Assert that the gradients are close
    assert_all_close(triton_grad, pytorch_grad)


if __name__ == "__main__":
    import torch
    from unsloth.kernels.layernorm import fast_layernorm_inference

    def manual_test():
        torch.manual_seed(0)
        batch_size, num_features = 64, 128
        x = torch.randn(batch_size, num_features, device='cuda')
        bias = torch.randn(num_features, device='cuda')

        gamma = torch.randn(num_features, device='cuda')
        eps = 1e-5
        
        try:
            output = fast_layernorm_inference(x, beta = bias, gamma=gamma, eps=1e-5, use_triton=True)
            print("Output shape:", output.shape)
        except Exception as e:
            print("Error during manual test:", str(e))

    if __name__ == "__main__":
        manual_test()