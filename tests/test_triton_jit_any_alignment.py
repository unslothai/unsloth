# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""jit_any_alignment keeps Triton's == 1 specialization of a size argument but drops the % 16 one, so a
size that is 1 in decode and anything in prefill builds two binaries, not a third for multiple-of-16
prompts. RoPE and the fused QK norm + RoPE kernels take seqlen this way."""

import ast
import importlib.util
import pathlib

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "kernels"


def _load_triton_launch():
    # By path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location("_unsloth_triton_launch", ROOT / "triton_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _param(kernel, name):
    return next(p for p in kernel.params if p.name == name)


def test_alignment_dropped_one_kept():
    jit_any_alignment = _load_triton_launch().jit_any_alignment

    @jit_any_alignment("n")
    def kernel(x_ptr, n, m, BLOCK: tl.constexpr):  # BLOCK: elements per program
        pass

    n, m = _param(kernel, "n"), _param(kernel, "m")
    if not hasattr(n, "do_not_specialize_on_alignment"):
        pytest.skip("this Triton has no do_not_specialize_on_alignment")
    assert n.do_not_specialize_on_alignment and not n.do_not_specialize
    assert not m.do_not_specialize_on_alignment and not m.do_not_specialize


@pytest.mark.parametrize(
    "path, kernels",
    [
        ("rope_embedding.py", {"_rope_embedding_QK"}),
        ("qk_norm_rope.py", {"_qk_norm_rope_forward", "_qk_norm_rope_backward"}),
    ],
)
def test_seqlen_kernels_use_it(path, kernels):
    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    wrapped = set()
    for node in ast.walk(tree):
        calls = []
        if isinstance(node, ast.FunctionDef):
            calls = [(d, node.name) for d in node.decorator_list if isinstance(d, ast.Call)]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and node.args:
            calls = [(node.func, getattr(node.args[0], "id", None))]
        for call, name in calls:
            if getattr(call.func, "id", None) == "jit_any_alignment":
                assert [a.value for a in call.args] == ["seqlen"]
                wrapped.add(name)
    assert kernels <= wrapped
