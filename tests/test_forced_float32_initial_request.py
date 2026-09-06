# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FORCE_FLOAT32=1 set before import must survive into the load decision.

`FastModel.from_pretrained` resets UNSLOTH_FORCE_FLOAT32 to "0" and re-derives
it per load: forced only for an fp16 request or hardware without bf16. That
threw away a request the user made before importing unsloth. unsloth_zoo gates
its gemma4_float32 patches on that same variable at IMPORT time, so the user's
patches were installed (fp32 residual stream, fp16 sub-layers) while the loader
took the ordinary bf16 path, and the first matmul died on a 4-bit gemma-4:

    RuntimeError: expected mat1 and mat2 to have the same dtype,
                  but got: float != c10::BFloat16

The fix reads the variable once at import and honours it in the decision. The
decision is a loop inside `from_pretrained`, so these tests lift that loop out
structurally and execute it against fake inputs. No GPU, no model load.
"""

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOADER = REPO_ROOT / "unsloth" / "models" / "loader.py"
SRC = LOADER.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


def _forced_float32_decision():
    """The statements from `os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"` through
    the `for disable_name in FORCE_FLOAT32:` loop, inside FastModel.from_pretrained."""
    for cls in ast.walk(TREE):
        if not (isinstance(cls, ast.ClassDef) and cls.name == "FastModel"):
            continue
        fn = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
        )
        body = fn.body
        for index, statement in enumerate(body):
            if (
                isinstance(statement, ast.For)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "disable_name"
                and getattr(statement.iter, "id", None) == "FORCE_FLOAT32"
            ):
                # Walk back to the reset that opens the block.
                start = index
                while start > 0:
                    prev = body[start - 1]
                    if (
                        isinstance(prev, ast.Assign)
                        and isinstance(prev.targets[0], ast.Subscript)
                        and getattr(prev.targets[0].value, "attr", None) == "environ"
                    ):
                        start -= 1
                        break
                    start -= 1
                return body[start : index + 1]
    pytest.fail("the forced-float32 decision loop is gone from FastModel.from_pretrained")


def _decide(
    *,
    dtype,
    supports_bf16,
    requested,
    arch = "gemma4",
):
    module = ast.Module(body = _forced_float32_decision(), type_ignores = [])
    ast.fix_missing_locations(module)
    env = {}

    class _Environ(dict):
        pass

    fake_os = type("os", (), {"environ": env})
    namespace = {
        "os": fake_os,
        "torch": torch,
        "FORCE_FLOAT32": ["gemma4", "qwen3_5"],
        "model_types": ["siglip", arch],
        "model_type_arch": arch,
        "model_types_all": f"{arch},",
        "dtype": dtype,
        "SUPPORTS_BFLOAT16": supports_bf16,
        "_UNSLOTH_REQUESTED_FORCE_FLOAT32": requested,
    }
    exec(compile(module, str(LOADER), "exec"), namespace)
    return {
        "forced": namespace["do_forced_float32"],
        "env": env.get("UNSLOTH_FORCE_FLOAT32"),
        "dtype": namespace["dtype"],
    }


def test_the_request_is_captured_once_at_module_level():
    """Inside from_pretrained the reset above it would read "0" every time."""
    assigns = [
        n
        for n in TREE.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_UNSLOTH_REQUESTED_FORCE_FLOAT32" for t in n.targets)
    ]
    assert len(assigns) == 1, "the capture must be a single module-level assignment"
    text = ast.get_source_segment(SRC, assigns[0].value)
    assert "UNSLOTH_FORCE_FLOAT32" in text and "environ" in text


def test_a_bf16_load_without_a_request_is_not_forced():
    out = _decide(dtype = torch.bfloat16, supports_bf16 = True, requested = False)
    assert out == {"forced": False, "env": "0", "dtype": torch.bfloat16}


def test_a_bf16_load_with_the_request_set_before_import_is_forced():
    out = _decide(dtype = torch.bfloat16, supports_bf16 = True, requested = True)
    assert out["forced"] is True
    assert out["env"] == "1"
    assert out["dtype"] == torch.bfloat16


def test_an_fp16_load_is_forced_with_or_without_the_request():
    for requested in (False, True):
        out = _decide(dtype = torch.float16, supports_bf16 = True, requested = requested)
        assert out["forced"] is True and out["env"] == "1", requested


def test_hardware_without_bf16_is_forced_as_before():
    out = _decide(dtype = torch.bfloat16, supports_bf16 = False, requested = False)
    assert out["forced"] is True and out["env"] == "1"


def test_the_request_does_not_force_a_family_that_never_forces():
    """The zoo patches only exist for the FORCE_FLOAT32 families; a request on
    a llama load has nothing installed to reconcile with and must stay inert."""
    out = _decide(dtype = torch.bfloat16, supports_bf16 = True, requested = True, arch = "llama")
    assert out == {"forced": False, "env": "0", "dtype": torch.bfloat16}
