# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""extract_quant_model_param_count (the printed parameter count of a quantized model) counts a
bitsandbytes 4-bit weight as two values per packed byte. FSDP2 QLoRA stores those bytes in a float
dtype (bnb_4bit_quant_storage = bfloat16 / float16 / float32), where one element is 2 or 4 bytes,
so counting two values per element printed about half (or a quarter) of the real count.

The function is read from unsloth/models/_utils.py without importing unsloth (which needs a GPU)."""

import ast
from pathlib import Path

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

_UTILS = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "extract_quant_model_param_count"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(_UTILS), "exec"), namespace)
    return namespace["extract_quant_model_param_count"]


def _model(storage):
    lin = bnb.nn.Linear4bit(64, 32, bias = True, quant_type = "nf4")
    lin.weight = bnb.nn.Params4bit(
        torch.randn(32, 64), requires_grad = False, quant_type = "nf4", blocksize = 64, module = lin
    )
    lin = lin.to("cpu")  # quantizes
    w = lin.weight
    if storage is not torch.uint8:
        # What unsloth's fsdp2_qlora.params4bit_to_float_storage does: same bytes, float elements.
        w.data = w.data.reshape(-1).view(storage).reshape(-1, 1)
        w.quant_storage = storage
    return torch.nn.Sequential(lin)


@pytest.mark.parametrize("storage", [torch.uint8, torch.bfloat16, torch.float16, torch.float32])
def test_float_packed_4bit_weights_count_every_value(storage):
    count = _load()(_model(storage))
    assert count == 32 * 64 + 32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_unquantized_params4bit_counts_its_dense_values(dtype):
    # Before bitsandbytes' first quantizing move a Params4bit holds dense values (bnb_quantized
    # False): one element is one weight, whatever its dtype.
    lin = bnb.nn.Linear4bit(64, 32, bias = False, quant_type = "nf4")
    lin.weight = bnb.nn.Params4bit(
        torch.randn(32, 64, dtype = dtype), requires_grad = False, quant_type = "nf4", module = lin
    )
    assert lin.weight.bnb_quantized is False
    assert _load()(torch.nn.Sequential(lin)) == 32 * 64


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32])
def test_fsdp2_sharded_4bit_weight_counts_every_value(storage):
    # FSDP2 replaces the Params4bit with a DTensor-like parameter that keeps quant_state and the
    # other quant attributes (fsdp2_qlora), not the Params4bit class.
    packed = 32 * 64 // 2 // torch.empty((), dtype = storage).element_size()
    p = torch.nn.Parameter(torch.zeros(packed, 1, dtype = storage), requires_grad = False)
    p.quant_state = object()
    p.bnb_quantized = True
    module = torch.nn.Module()
    module.weight = p
    assert _load()(module) == 32 * 64
