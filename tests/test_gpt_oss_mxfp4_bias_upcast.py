# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""gpt-oss MXFP4 expert biases go to fp32, except experts transformers 4.x dequantized (an fp32 bias broke its bmm)."""

import ast
from pathlib import Path

import pytest
import torch

LOADER = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"


def _registered():
    values = []
    for node in ast.walk(ast.parse(LOADER.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_custom_dtype":
            value = ast.literal_eval(node.args[0])
            if "gate_up_proj_bias" in value:
                values.append(value)
    guarded = [v for v in values if "dense" in v]
    plain = [v for v in values if "dense" not in v]
    assert len(guarded) == 1 and len(plain) == 1, values
    return {True: guarded[0], False: plain[0]}


class _PackedParam(torch.nn.Parameter):
    """Stands in for Mxfp4ExpertParam."""


def _experts(weight):
    m = torch.nn.Module()
    m.gate_up_proj = weight
    m.gate_up_proj_bias = torch.nn.Parameter(torch.zeros(2, 4, dtype = torch.bfloat16))
    m.down_proj_bias = torch.nn.Parameter(torch.zeros(2, 4, dtype = torch.bfloat16))
    return m


def _run(code, module):
    fields = code.split(";")
    assert fields[:3] == ["all", "None", "None"] and fields[4] == ""
    exec(fields[3], {"torch": torch, "module": module, "name": "model.layers.0.mlp.experts"})
    return module.gate_up_proj_bias.dtype, module.down_proj_bias.dtype


@pytest.mark.parametrize("dense_guard", [True, False])
def test_quantized_experts_get_fp32_biases(dense_guard):
    code = _registered()[dense_guard]
    packed = _experts(_PackedParam(torch.zeros(2, 4, dtype = torch.uint8), requires_grad = False))
    assert _run(code, packed) == (torch.float32, torch.float32)
    # Mxfp4GptOssExperts holds a triton_kernels tensor, not a Parameter.
    native = _experts(None)
    native.gate_up_proj = object()
    assert _run(code, native) == (torch.float32, torch.float32)


def test_dense_experts_keep_their_bias_dtype_on_transformers_4():
    dense = _experts(torch.nn.Parameter(torch.zeros(2, 4, dtype = torch.bfloat16)))
    assert _run(_registered()[True], dense) == (torch.bfloat16, torch.bfloat16)
    # transformers 5.x is unchanged: Unsloth's own MXFP4 forward reads fp32 biases.
    dense = _experts(torch.nn.Parameter(torch.zeros(2, 4, dtype = torch.bfloat16)))
    assert _run(_registered()[False], dense) == (torch.float32, torch.float32)


def test_modules_without_expert_biases_are_untouched():
    m = torch.nn.Linear(4, 4, dtype = torch.bfloat16)
    exec(_registered()[True].split(";")[3], {"torch": torch, "module": m, "name": "lm_head"})
    assert m.bias.dtype == torch.bfloat16
