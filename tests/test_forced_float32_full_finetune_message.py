# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A FORCE_FLOAT32 family (Qwen3-MoE, Gemma 3, gpt-oss) asked for float16 on a GPU without bfloat16 loads in bfloat16
and casts, and the trainer then runs float32 without autocast. The full-finetuning notice used to read the bf16 load
dtype and print "Using bfloat16 full finetuning" on a T4. Source-level: from_pretrained cannot run without a GPU."""

import ast
from pathlib import Path

VISION = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


def _full_finetuning_branch():
    tree = ast.parse(VISION.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "full_finetuning"
        ):
            if any("UNSLOTH_ENABLE_FULL_FINETUNING" in ast.unparse(s) for s in node.body):
                return node
    raise AssertionError("full_finetuning notice block not found")


def test_forced_float32_without_bf16_is_announced_as_float32_first():
    block = _full_finetuning_branch()
    chain = [s for s in block.body if isinstance(s, ast.If)][0]
    first = ast.unparse(chain.test)
    assert "do_forced_float32" in first and "SUPPORTS_BFLOAT16" in first, first
    assert "float32" in ast.unparse(chain.body)
    assert "Using bfloat16 full finetuning" not in ast.unparse(chain.body)
