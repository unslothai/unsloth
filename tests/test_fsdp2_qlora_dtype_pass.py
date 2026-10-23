# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 packs 4-bit weights in a float dtype. The vision loader's dtype pass (unsloth_zoo's
patch_model_and_tokenizer: `module.to(module._pre_set_compute_dtype)`, set on gpt-oss's 4-bit
down_projs when loading in float16) would then cast the packed bytes as if they were weights.
`packed_4bit_as_uint8` shows them as uint8 for that pass, as without FSDP2."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = _ROOT / "unsloth" / "models" / "fsdp2_qlora.py"
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_qlora_dtype_pass", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _float_packed_linear4bit(storage):
    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(
        128, 64, bias = True, compute_dtype = torch.float16, quant_type = "nf4", quant_storage = storage
    )
    lin.weight = bnb.nn.Params4bit(
        torch.randn(64, 128) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        quant_storage = storage,
        blocksize = 64,
        module = lin,
    )
    lin = lin.to("cpu")
    lin.bias = torch.nn.Parameter(torch.ones(64, dtype = torch.float16))
    assert lin.weight.dtype == storage and lin.weight.bnb_quantized
    return lin


def _dtype_pass(model):
    # unsloth_zoo.patching_utils.patch_model_and_tokenizer, "Upcast or downcast if explicitly set".
    for module in model.modules():
        if hasattr(module, "_pre_set_compute_dtype"):
            module.to(module._pre_set_compute_dtype)


@pytest.mark.parametrize("storage", [torch.float16, torch.bfloat16])
def test_packed_bytes_survive_the_pre_set_compute_dtype_pass(storage):
    fsdp2_qlora = _load_module()
    lin = _float_packed_linear4bit(storage)
    lin._pre_set_compute_dtype = torch.float32
    model = torch.nn.Sequential(lin)
    weight = lin.weight
    before = weight.data.clone()
    with fsdp2_qlora.packed_4bit_as_uint8(model) as n:
        assert n == 1
        _dtype_pass(model)
    assert lin.weight is weight
    assert weight.dtype == storage and weight.quant_storage == storage
    assert torch.equal(weight.data.view(torch.uint8), before.view(torch.uint8))
    # The rest of the module is still cast, as without FSDP2.
    assert lin.bias.dtype == torch.float32


def test_disabled_and_uint8_are_untouched():
    fsdp2_qlora = _load_module()
    lin = _float_packed_linear4bit(torch.bfloat16)
    with fsdp2_qlora.packed_4bit_as_uint8(lin, enabled = False) as n:
        assert n == 0 and lin.weight.dtype == torch.bfloat16
    lin8 = _float_packed_linear4bit(torch.uint8)
    with fsdp2_qlora.packed_4bit_as_uint8(lin8) as n:
        assert n == 0 and lin8.weight.dtype == torch.uint8


def test_without_it_the_pass_casts_packed_bytes():
    # The failure the helper prevents: the packed float16 words are converted to float32 values.
    lin = _float_packed_linear4bit(torch.float16)
    lin._pre_set_compute_dtype = torch.float32
    _dtype_pass(torch.nn.Sequential(lin))
    assert lin.weight.dtype == torch.float32


def test_vision_loader_wraps_patch_model_and_tokenizer():
    tree = ast.parse((_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8"))
    wrapped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        ctx = [
            item.context_expr.func.id
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
        ]
        if "packed_4bit_as_uint8" not in ctx:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "patch_model_and_tokenizer"
            ):
                wrapped.append(inner)
    assert len(wrapped) == 1
