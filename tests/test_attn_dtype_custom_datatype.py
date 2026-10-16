# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The resolver needs the attention dtype: a transient float32 load must not disable flash."""

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
    """Finds the resolver's dtype keyword by AST, so code inserted above cannot retarget the check."""
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
        "model_class": None,
        "attention_class_for_load": lambda *args, **kwargs: (None, True),
        "_builds_remote_class": False,
        "_remote_class": None,
        "supports_sdpa": True,
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
        (torch.float32, False, None, torch.float32),
        (torch.bfloat16, False, None, torch.bfloat16),
        (torch.float16, False, None, torch.float16),
        # UNSLOTH_FORCE_FLOAT32: loaded bfloat16 despite the name, so flash stays on.
        (torch.bfloat16, True, None, torch.bfloat16),
        (torch.float32, True, None, torch.bfloat16),
        # UNSLOTH_FORCE_CUSTOM_DTYPE models load float32 but cast projections back, so attention is fp16.
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
        _supports_flash_attn = True  # the flag transformers >= 4.53 dispatches on
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
