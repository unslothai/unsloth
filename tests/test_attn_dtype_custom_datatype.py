# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The dtype handed to `resolve_attention_implementation` is the *attention* dtype.

The resolver turns Flash Attention off for float32 because the kernels only accept fp16/bf16,
so it must be told the dtype the attention projections end up in, not the checkpoint load
dtype. Two loader paths load wide and narrow again afterwards:

  UNSLOTH_FORCE_FLOAT32   loads bfloat16 (loader.py sets dtype = torch.bfloat16) with a
                          float16 bnb compute dtype.

  UNSLOTH_FORCE_CUSTOM_DTYPE  csm, falcon_h1 and nemotron_h load float32 so the Mamba /
                          Triton kernels keep ieee precision, then cast every projection to
                          `correct_dtype` (float16) after the load. Attention never sees
                          float32, so flash is usable and must not be turned off.

Falcon-H1 at dtype = torch.float16 takes that second path, and reporting the transient
float32 downgraded a working flash_attention_2 to sdpa.

This checks the selection expression itself rather than a whole model load: the statements
between `model_class = resolve_model_class(...)` and the resolver call are lifted out of
vision.py and evaluated against each combination.
"""

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VISION = REPO_ROOT / "unsloth" / "models" / "vision.py"
SRC = VISION.read_text(encoding = "utf-8")


def _attention_dtype_expression():
    """The statements building the resolver's `dtype`, plus that keyword's expression.

    Found structurally, so inserting code above it cannot retarget this at another call.
    """
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.FunctionDef) or node.name != "from_pretrained":
            continue
        body = node.body
        for index, statement in enumerate(body):
            call = getattr(statement, "value", None)
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", None) != "resolve_attention_implementation":
                continue
            keyword = next((k for k in call.keywords if k.arg == "dtype"), None)
            if keyword is None:
                pytest.fail(
                    "vision.py no longer passes dtype = to resolve_attention_implementation"
                )
            # Everything between `model_class = resolve_model_class(...)` and the call.
            for start in range(index - 1, -1, -1):
                previous = body[start]
                if (
                    isinstance(previous, ast.Assign)
                    and getattr(previous.targets[0], "id", None) == "model_class"
                ):
                    return body[start + 1 : index], keyword.value
            pytest.fail("no model_class assignment ahead of the resolver call in vision.py")
    pytest.fail("could not find the resolve_attention_implementation call in vision.py")


PREAMBLE, DTYPE_EXPR = _attention_dtype_expression()


def _selected(dtype, do_forced_float32, correct_dtype):
    namespace = {
        "torch": torch,
        "dtype": dtype,
        "do_forced_float32": do_forced_float32,
        "correct_dtype": correct_dtype,
        "auto_config": None,
        "auto_model": None,
        "model_name": "",
        "resolve_model_class": lambda *args, **kwargs: None,
    }
    module = ast.Module(body = list(PREAMBLE), type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(VISION), "exec"), namespace)
    expression = ast.Expression(body = DTYPE_EXPR)
    ast.fix_missing_locations(expression)
    return eval(compile(expression, str(VISION), "eval"), namespace)


@pytest.mark.parametrize(
    "dtype, do_forced_float32, correct_dtype, expected",
    [
        # Plain loads: the load dtype is the attention dtype.
        (torch.float32, False, None, torch.float32),
        (torch.bfloat16, False, None, torch.bfloat16),
        (torch.float16, False, None, torch.float16),
        # UNSLOTH_FORCE_FLOAT32: loaded bfloat16 despite the name, so flash stays on.
        (torch.bfloat16, True, None, torch.bfloat16),
        (torch.float32, True, None, torch.bfloat16),
        # UNSLOTH_FORCE_CUSTOM_DTYPE: csm / falcon_h1 / nemotron_h load float32 and cast the projections back to
        # correct_dtype, so attention runs in float16.
        (torch.float32, False, torch.float16, torch.float16),
    ],
)
def test_attention_dtype_is_the_post_cast_dtype(dtype, do_forced_float32, correct_dtype, expected):
    assert _selected(dtype, do_forced_float32, correct_dtype) is expected


def test_custom_datatype_load_does_not_disable_flash_attention():
    """The falcon_h1 / nemotron_h shape end to end through the resolver."""
    import unsloth  # noqa: F401
    from unsloth.models import _utils

    class SupportsFlashAndSdpa:
        _supports_flash_attn_2 = True
        _supports_flex_attn = False
        _supports_sdpa = True

    from types import SimpleNamespace

    original = _utils.HAS_FLASH_ATTENTION
    _utils.HAS_FLASH_ATTENTION = True
    try:
        config = SimpleNamespace(model_type = "falcon_h1", attention_dropout = 0)
        impl = _utils.resolve_attention_implementation(
            SupportsFlashAndSdpa,
            config,
            supports_sdpa = True,
            dtype = _selected(torch.float32, False, torch.float16),
        )
    finally:
        _utils.HAS_FLASH_ATTENTION = original

    assert impl == "flash_attention_2"
