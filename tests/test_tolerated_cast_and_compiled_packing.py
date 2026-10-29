# SPDX-License-Identifier: AGPL-3.0-only
"""Unquantized casts inside the tolerated load, and padding-free on compiled models."""

import torch
import torch.nn as nn
import pytest

from transformers import LlamaConfig, LlamaForCausalLM
from unsloth.models.vision import _tolerate_dtype_cast_on_quantized_model
from unsloth.trainer import _forward_accepts_packing_kwargs


def _tiny():
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            vocab_size = 99,
        )
    ).to(torch.float32)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.to(dtype = torch.bfloat16),
        lambda m: m.to(torch.bfloat16),
        lambda m: m.to(device = "cpu", dtype = torch.bfloat16),
    ],
)
def test_an_unquantized_cast_inside_the_tolerated_load_is_not_dropped(call):
    """The context must not swallow a cast on an unquantized model."""
    model = _tiny()
    with _tolerate_dtype_cast_on_quantized_model(True):
        call(model)
    assert next(model.parameters()).dtype is torch.bfloat16


def test_a_compiled_model_keeps_padding_free():
    """OptimizedModule sets `forward` on the instance, not the class."""

    class Kwargs(nn.Module):
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            return input_ids

    assert _forward_accepts_packing_kwargs(Kwargs()) is True
    assert _forward_accepts_packing_kwargs(torch.compile(Kwargs())) is True


def test_a_fixed_signature_is_still_refused():
    """The negative control: the Phi-4 shape must still block padding-free."""

    class Fixed(nn.Module):
        def forward(
            self,
            input_ids = None,
            attention_mask = None,
        ):
            return input_ids

    assert _forward_accepts_packing_kwargs(Fixed()) is False
