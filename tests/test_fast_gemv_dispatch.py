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

"""`get_lora_parameters` must not treat a `weight_scale` as a quant state for a weight that is
already dequantized to bf16 (e.g. a compressed-tensors layer at forward time). Otherwise the
bnb fast_gemv / fast_dequantize path reads a missing `absmax` and crashes.
"""

from types import SimpleNamespace

import pytest
import torch

# unsloth.kernels.utils imports bitsandbytes unconditionally, so skip the whole module up
# front on runners without it (e.g. CPU-only) before importing unsloth, otherwise collection
# errors instead of producing a skip. Any other import error still surfaces as a failure.
pytest.importorskip("bitsandbytes")

import unsloth  # noqa: F401  (sets UNSLOTH_IS_PRESENT before transformers)
from unsloth.kernels.utils import get_lora_parameters_bias, _FP8_WEIGHT_DTYPES

_FP8 = _FP8_WEIGHT_DTYPES[0] if _FP8_WEIGHT_DTYPES else None


def _proj(weight, weight_scale = None):
    proj = SimpleNamespace(weight = weight, bias = None, merged = False)
    if weight_scale is not None:
        proj.weight_scale = weight_scale
    return proj


def test_bf16_weight_scale_not_used_as_quant_state():
    """A bf16 weight carrying a weight_scale (compressed-tensors) -> quant state must be None."""
    proj = _proj(torch.randn(4, 4, dtype = torch.bfloat16), torch.rand(2, 2))
    W, W_quant = get_lora_parameters_bias(proj)[:2]
    assert W_quant is None


def test_fp8_weight_keeps_scale():
    """An actual fp8 weight still resolves its weight_scale as the quant state."""
    if _FP8 is None:
        pytest.skip("no float8 dtype in this torch build")
    scale = torch.rand(2, 2)
    proj = _proj(torch.randn(4, 4).to(_FP8), scale)
    W, W_quant = get_lora_parameters_bias(proj)[:2]
    assert W_quant is scale


def test_plain_bf16_has_no_quant_state():
    proj = _proj(torch.randn(4, 4, dtype = torch.bfloat16))
    W, W_quant = get_lora_parameters_bias(proj)[:2]
    assert W_quant is None
