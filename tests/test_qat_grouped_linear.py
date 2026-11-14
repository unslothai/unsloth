# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT must leave block-diagonal grouped Linears (DeepSeek-V4 / V4.1 `o_a_proj`) in full precision.

torchao swaps a matched Linear for a dense fake-quantized Linear: the grouped layer lost `n_groups` and its
grouped matmul, and GroupedLinearLoRA failed on the first forward."""

import importlib.util
import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

from unsloth.models._utils import _prepare_model_for_qat  # noqa: E402
from unsloth.models.grouped_linear_lora import is_grouped_linear  # noqa: E402


class _GroupedLinear(torch.nn.Linear):
    """Block-diagonal: group g maps in_features // n_groups inputs to out_features // n_groups outputs."""

    def __init__(self, in_features, out_features, n_groups):
        super().__init__(in_features // n_groups, out_features, bias = False)
        self.n_groups = n_groups

    def forward(self, x):
        g = self.n_groups
        w = self.weight.view(g, self.out_features // g, -1)
        return torch.einsum("...gd,grd->...gr", x.unflatten(-1, (g, -1)), w).flatten(-2)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig()
        torch.manual_seed(0)
        self.o_a_proj = _GroupedLinear(256, 128, n_groups = 2)
        self.o_b_proj = torch.nn.Linear(128, 128, bias = False)

    def forward(self, x):
        return self.o_b_proj(self.o_a_proj(x))


def _fake_quantized(module):
    return "FakeQuantized" in type(module).__name__ or hasattr(module, "weight_fake_quantizer")


@pytest.mark.parametrize("scheme", ["int4"])
def test_qat_skips_the_grouped_linear_and_keeps_its_math(scheme):
    model = _Model()
    assert is_grouped_linear(model.o_a_proj)
    x = torch.randn(3, 256)
    expected = model.o_a_proj(x).detach()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, scheme)
    assert type(model.o_a_proj) is _GroupedLinear and model.o_a_proj.n_groups == 2
    assert not _fake_quantized(model.o_a_proj)
    assert torch.equal(model.o_a_proj(x), expected)
    # Every other Linear is still quantized: the guard only removes the grouped layer.
    assert _fake_quantized(model.o_b_proj)
    assert any("o_a_proj" in str(w.message) and "full precision" in str(w.message) for w in caught)
    model(x).sum().backward()
