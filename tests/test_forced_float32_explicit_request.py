# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel's FORCE_FLOAT32 scan (Qwen3-MoE, Qwen3.5, Gemma 3/4, gpt-oss, GLM-4 MoE) on a GPU without bfloat16
(T4 / V100): dtype = torch.float32 asked explicitly for a 16-bit load used to take the bf16-load / fp16-weights path,
so a T4 trained these models at float16 precision. Source-level: from_pretrained cannot run without a GPU."""

import ast
from pathlib import Path

import pytest
import torch

LOADER = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"


def _scan_condition():
    """The dtype part of the `for disable_name in FORCE_FLOAT32:` test (everything after the name match)."""
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "FORCE_FLOAT32"
        ):
            test = next(s for s in node.body if isinstance(s, ast.If)).test
            assert isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            body = (
                test.values[1]
                if len(test.values) == 2
                else ast.BoolOp(op = ast.And(), values = test.values[1:])
            )
            rest = ast.Expression(body)
            return compile(ast.fix_missing_locations(rest), str(LOADER), "eval")
    raise AssertionError("FORCE_FLOAT32 scan not found")


def _forced(
    dtype,
    bf16,
    user_float32 = None,
    load_in_4bit = False,
    load_in_8bit = False,
    quantization_config = None,
):
    if user_float32 is None:
        user_float32 = dtype is torch.float32
    ns = dict(
        torch = torch,
        dtype = dtype,
        SUPPORTS_BFLOAT16 = bf16,
        user_float32 = user_float32,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
        quantization_config = quantization_config,
    )
    return bool(eval(_scan_condition(), ns))


def test_explicit_float32_stays_float32_without_bf16():
    assert _forced(torch.float32, bf16 = False) is False


@pytest.mark.parametrize(
    "dtype, bf16, kw, expected",
    [
        (None, False, {}, True),  # auto dtype on a T4: the fp16-safe forced path, as before
        (torch.float16, False, {}, True),
        (torch.float16, True, {}, True),
        (torch.bfloat16, False, {}, False),  # explicit bf16 (XPU) stays bf16
        (torch.float32, True, {}, False),
        (None, True, {}, False),
        (
            torch.float32,
            False,
            {"load_in_4bit": True},
            True,
        ),  # quantized loads keep the forced path
        (torch.float32, False, {"load_in_8bit": True}, True),
        (torch.float32, False, {"quantization_config": object()}, True),
        (
            torch.float32,
            False,
            {"user_float32": False},
            True,
        ),  # float32 derived from a 4bit compute dtype
    ],
)
def test_forced_path_unchanged_elsewhere(dtype, bf16, kw, expected):
    assert _forced(dtype, bf16, **kw) is expected
