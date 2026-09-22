# SPDX-License-Identifier: AGPL-3.0-only
"""A remote-code from_pretrained that ends with model.to(dtype) must not break a 4-bit load.

microsoft/Phi-4-reasoning-vision-15B overrides from_pretrained and finishes with
`model.to(dtype)` where dtype is the model's own dtype. transformers refuses any
dtype cast on a bitsandbytes model, so the 4-bit load died with "You cannot cast a
bitsandbytes model in a new dtype" while the 16-bit load was fine. Inside Unsloth's
load, that call now casts only the floating tensors that are not quantized weights.

Built on a small PreTrainedModel with a fake packed weight, no downloads; each test
states which arm it measures.
"""

import pytest
import torch
import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel

from unsloth.models.vision import (
    _cast_unquantized_floats,
    _tolerate_dtype_cast_on_quantized_model,
)


class _Params4bit(nn.Parameter):
    """The shape of a bitsandbytes packed weight: uint8 storage plus a quant_state."""

    def __new__(cls, data):
        p = super().__new__(cls, data, requires_grad = False)
        p.quant_state = object()
        return p


class _Cfg(PretrainedConfig):
    model_type = "unsloth-test-quantized"


class _Model(PreTrainedModel):
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.norm = nn.LayerNorm(4)  # plain float parameters
        self.packed = _Params4bit(torch.zeros(8, dtype = torch.uint8))
        self.register_buffer("scale", torch.ones(4, dtype = torch.float32))

    def forward(self, x):
        return self.norm(x)


def _quantized_model():
    m = _Model(_Cfg())
    from transformers.utils.quantization_config import QuantizationMethod

    m.quantization_method = QuantizationMethod.BITS_AND_BYTES
    m.is_quantized = True
    return m


def test_transformers_still_refuses_outside_the_context():
    """The behaviour we are scoping around, on this transformers version."""
    m = _quantized_model()
    with pytest.raises(ValueError, match = "cannot cast a bitsandbytes model"):
        m.to(torch.bfloat16)


def test_cast_inside_the_context_moves_only_unquantized_floats():
    m = _quantized_model()
    with _tolerate_dtype_cast_on_quantized_model(True):
        out = m.to(torch.bfloat16)
    assert out is m
    assert m.norm.weight.dtype == torch.bfloat16
    assert m.scale.dtype == torch.bfloat16
    assert m.packed.dtype == torch.uint8, "the packed weight must never be cast"
    # and the shim is gone again
    with pytest.raises(ValueError):
        m.to(torch.float16)


def test_disabled_context_changes_nothing():
    m = _quantized_model()
    with _tolerate_dtype_cast_on_quantized_model(False):
        with pytest.raises(ValueError):
            m.to(torch.bfloat16)


def test_unquantized_model_and_device_moves_pass_through():
    m = _Model(_Cfg())
    with _tolerate_dtype_cast_on_quantized_model(True):
        m.to(torch.bfloat16)  # plain model: transformers' own path
        assert m.norm.weight.dtype == torch.bfloat16
        q = _quantized_model()
        q.to("cpu")  # device only: untouched, no error
        q.to(device = "cpu", dtype = torch.bfloat16)  # both: cast floats, then move
        assert q.norm.weight.dtype == torch.bfloat16
        assert q.packed.dtype == torch.uint8


def test_cast_helper_skips_every_quantized_flavour():
    m = _Model(_Cfg())
    m.int8 = nn.Parameter(torch.zeros(2, dtype = torch.int8), requires_grad = False)
    m.int8.SCB = torch.ones(1)
    _cast_unquantized_floats(m, torch.bfloat16)
    assert m.packed.dtype == torch.uint8 and m.int8.dtype == torch.int8
    assert m.norm.bias.dtype == torch.bfloat16
