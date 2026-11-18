# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""original_apply_o on a float32 model outside autocast: run_attention downcasts fp32 Q / K / V to bf16 / fp16 for
kernels without an fp32 path (FlashAttention, xformers on sm_100+) and returns that dtype, which an fp32 o_proj
cannot take. Extracted with ast so the test runs on CPU without importing the GPU stack."""

import ast
import os

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = os.path.join(HERE, "unsloth", "models", "llama.py")


def _apply_o():
    src = open(LLAMA, encoding = "utf-8").read()
    node = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "original_apply_o"
    )
    ns = {"torch": torch}
    exec(ast.get_source_segment(src, node), ns)
    return ns["original_apply_o"]


class _Attn(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        torch.manual_seed(0)
        self.o_proj = torch.nn.Linear(16, 8, bias = False).to(dtype)


@pytest.mark.parametrize("half", [torch.bfloat16, torch.float16])
def test_fp32_o_proj_takes_a_downcast_attention_output(half):
    apply_o = _apply_o()
    attn = _Attn(torch.float32)
    X = torch.randn(2, 3, 16).to(half)
    out = apply_o(attn, X)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, attn.o_proj(X.float()))


def test_half_o_proj_and_autocast_are_untouched():
    apply_o = _apply_o()
    attn = _Attn(torch.bfloat16)
    X = torch.randn(2, 3, 16, dtype = torch.bfloat16)
    assert apply_o(attn, X).dtype == torch.bfloat16
    # DoRA on a bf16 model hands an fp32 X to a bf16 o_proj under autocast: the cast is autocast's job.
    attn32 = _Attn(torch.float32)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        assert apply_o(attn32, X).dtype == torch.bfloat16
