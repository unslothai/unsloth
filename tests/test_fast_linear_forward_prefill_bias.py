"""fast_linear_forward with q_len > 1 (generate's prefill through fast_swiglu_inference) must add the
base layer's bias, as the q_len == 1 decode path does (Llama-arch configs with mlp_bias = True)."""

from pathlib import Path

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.kernels import utils as kernel_utils


def test_tree_under_test():
    assert Path(kernel_utils.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("q_len", [1, 5])
def test_prefill_adds_base_bias(q_len):
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 24, bias = True)
    with torch.no_grad():
        layer.bias.uniform_(1.0, 2.0)
    X = torch.randn(2, q_len, 16)
    with torch.inference_mode():
        got = kernel_utils.fast_linear_forward(layer, X)
        want = torch.nn.functional.linear(X, layer.weight, layer.bias)
    torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-4)
