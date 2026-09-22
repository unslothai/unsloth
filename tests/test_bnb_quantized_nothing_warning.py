# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A load handed a bitsandbytes config that still built a 16bit model must say so.

Every branch that turns 4bit/8bit off prints why, but a load can keep the request and
still quantize nothing (every Linear in the skip list, or weights in module types
bitsandbytes does not replace). Without a message that surfaces as an out of memory
error, or as a 16bit run reported as 4bit.
"""

import ast
from pathlib import Path

import pytest
import torch

from unsloth.models.loader_utils import (
    _bnb_bits_requested,
    warn_if_bitsandbytes_quantized_nothing,
)

ROOT = Path(__file__).resolve().parents[1]


class Params4bit(torch.nn.Parameter):
    """Stands in for bitsandbytes.nn.Params4bit: only the type name is read."""


class Linear4bit(torch.nn.Linear):
    pass


def _bnb4():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit = True)


def test_bits_requested_reads_objects_and_dicts():
    from transformers import BitsAndBytesConfig

    assert _bnb_bits_requested(None) is None
    assert _bnb_bits_requested(_bnb4()) == 4
    assert _bnb_bits_requested(BitsAndBytesConfig(load_in_8bit = True)) == 8
    assert _bnb_bits_requested({"quant_method": "bitsandbytes", "load_in_4bit": True}) == 4
    assert _bnb_bits_requested({"quant_method": "fp8", "load_in_4bit": True}) is None
    assert _bnb_bits_requested({"quant_method": "gptq", "bits": 4}) is None


def test_warns_when_a_4bit_load_quantized_nothing(capsys):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert warn_if_bitsandbytes_quantized_nothing(model, _bnb4(), "org/model") is True
    out = capsys.readouterr().out
    assert "4bit loading was on" in out and "org/model" in out and "16bit" in out


def test_silent_when_a_linear_or_a_packed_parameter_is_quantized(capsys):
    assert (
        warn_if_bitsandbytes_quantized_nothing(torch.nn.Sequential(Linear4bit(4, 4)), _bnb4(), "m")
        is False
    )
    experts = torch.nn.Module()  # a 3-D expert stack packed in place, no Linear4bit anywhere
    experts.gate_up_proj = Params4bit(torch.zeros(8, 1), requires_grad = False)
    assert (
        warn_if_bitsandbytes_quantized_nothing(
            torch.nn.Sequential(torch.nn.Linear(4, 4), experts), _bnb4(), "m"
        )
        is False
    )
    assert capsys.readouterr().out == ""


def test_silent_without_a_bitsandbytes_request(capsys):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert warn_if_bitsandbytes_quantized_nothing(model, None, "m") is False
    assert warn_if_bitsandbytes_quantized_nothing(model, {"quant_method": "fp8"}, "m") is False
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "relpath, expected",
    [("unsloth/models/vision.py", 1), ("unsloth/models/llama.py", 2)],
)
def test_every_in_process_load_is_checked(relpath, expected):
    """Each transformers from_pretrained that can take a bitsandbytes config is followed
    by the check, fed the same kwargs the load received."""
    src = (ROOT / relpath).read_text(encoding = "utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "warn_if_bitsandbytes_quantized_nothing"
    ]
    assert len(calls) == expected, relpath
    for call in calls:
        assert "quantization_config" in ast.unparse(call.args[1]), ast.unparse(call)
