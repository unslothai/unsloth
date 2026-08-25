# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The '-bf16' notice must not claim 16bit when a quantization_config survives.

A `-bf16` repo name drops the plain `load_in_4bit` / `load_in_8bit` / `load_in_fp8`
flags, and the notice added for that is accurate. It is NOT accurate when the caller
passed `quantization_config = BitsAndBytesConfig(load_in_4bit = True)`: that object
is only read into the local flags, it stays in `**kwargs`, and both loaders forward
it to Transformers, which quantizes anyway. Saying "will load in 16bit" there tells
the user the opposite of what happens.

Source-level, because reaching the branch needs a real checkpoint download.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "unsloth" / "models" / "loader.py"
SRC = LOADER.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


def _bf16_notice_guards():
    """Every `if` whose body is only the '-bf16' notice `print`."""
    guards = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        stmt = node.body[0]
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        func = stmt.value.func
        if not (isinstance(func, ast.Name) and func.id == "print"):
            continue
        text = ast.get_source_segment(SRC, stmt) or ""
        if "load in 16bit" in text and "-bf16" in text:
            guards.append(node)
    return guards


def test_all_four_bf16_branches_carry_the_notice():
    """Two in FastLanguageModel.from_pretrained, two in FastModel.from_pretrained."""
    assert len(_bf16_notice_guards()) == 4


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_the_notice_is_gated_on_no_user_quantization_config(index):
    guards = _bf16_notice_guards()
    assert len(guards) == 4, "the notice moved; update this test"
    test_src = ast.get_source_segment(SRC, guards[index].test) or ""
    assert "quantization_config" in test_src, (
        "the '-bf16' notice claims a 16bit load, but a user-supplied "
        "quantization_config is still forwarded to Transformers and still "
        "quantizes, so the notice must be suppressed when one is present"
    )
    # It must read the value that is actually forwarded, not the stale local.
    assert "kwargs" in test_src, (
        "read kwargs['quantization_config']: the local `quantization_config` is "
        "not cleared when the bitsandbytes-unavailable branch pops it from kwargs"
    )


def test_the_flags_are_still_cleared():
    """The notice is a message; it must not change what the branch does."""
    assert SRC.count("load_in_16bit = True") >= 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
