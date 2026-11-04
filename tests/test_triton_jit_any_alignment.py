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
    spec = importlib.util.spec_from_file_location(
        "_unsloth_triton_launch", ROOT / "triton_launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _param(kernel, name):
    if not hasattr(kernel, "params"):  # TRITON_INTERPRET=1 returns an InterpretedFunction
        pytest.skip("interpreted Triton function has no params")
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


def test_nf4_gemv_rows_alignment_unspecialized():
    """M (rows) of the NF4 GEMV kernels keeps the == 1 specialization (batch-1 speed) but not the % 16
    one, and the fast launch keys it the same way: every M > 1 shares one binary and one cache entry,
    so bsz 3 / 8 / 16 after bsz 2 compile nothing."""
    tree = ast.parse((ROOT / "nf4_gemv.py").read_text(encoding = "utf-8"))
    kernels = {}
    launch_call = args_tuple = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_gemv_nf4_kernel",
            "_gemv_nf4_words_kernel",
        ):
            kernels[node.name] = node
        if isinstance(node, ast.FunctionDef) and node.name == "_launch":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and getattr(sub.targets[0], "id", None) == "args":
                    args_tuple = sub.value
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "launch":
                    launch_call = sub
    assert set(kernels) == {"_gemv_nf4_kernel", "_gemv_nf4_words_kernel"}
    for fn in kernels.values():
        (deco,) = fn.decorator_list
        assert deco.func.id == "jit_any_alignment" and [a.value for a in deco.args] == ["M"]
        assert fn.args.args[10].arg == "M"
    unaligned = next(kw.value for kw in launch_call.keywords if kw.arg == "unaligned")
    assert unaligned.id == "_UNALIGNED_M"
    assert args_tuple.elts[10].id == "M"
    source = "".join((ROOT / "nf4_gemv.py").read_text(encoding = "utf-8").split())
    # Only keyed without % 16 when Triton really dropped it.
    assert (
        '(10,)ifall(any(p.name=="M"andgetattr(p,"do_not_specialize_on_alignment",False)' in source
    )


class _Compiled:
    packed_metadata = function = None

    def __init__(self, log):
        self.log = log

    def run(self, *args):
        self.log.append("cached")


class _Kernel:
    arg_names = ["out", "n", "BLOCK"]

    def __init__(self):
        self.log = []

    def __getitem__(self, grid):
        def slow(*args, **kwargs):
            self.log.append("slow")
            return _Compiled(self.log)

        return slow


def _fast_launch(monkeypatch):
    mod = _load_triton_launch()
    monkeypatch.setattr(mod, "_ENABLED", True)
    monkeypatch.setattr(mod, "_hooks_set", lambda: False)
    monkeypatch.setattr(mod, "_raw_stream", lambda device: 0)
    return mod


@pytest.mark.parametrize(
    "mode, same, different",
    [
        # unspecialized: one entry for every int32 value; int32 vs int64 still split.
        ("unspecialized", [2, 3, 16, 1, 17], [2**31]),
        # unaligned: == 1 split, % 16 not.
        ("unaligned", [2, 3, 16, 17], [1, 2**31]),
        # default: == 1 and the low 4 bits split.
        ("default", [2, 18, 34], [1, 16, 2**31]),
    ],
)
def test_launch_cache_keys(monkeypatch, mode, same, different):
    torch = pytest.importorskip("torch")
    mod = _fast_launch(monkeypatch)
    kernel, out = _Kernel(), torch.empty(4)
    extra = {mode: (1,)} if mode != "default" else {}

    def go(n):
        mod.launch(kernel, (1,), (out, n), 1, {"BLOCK": 4}, 0, **extra)

    for n in same:
        go(n)
    assert kernel.log == ["slow"] + ["cached"] * (len(same) - 1)
    for n in different:
        kernel.log.clear()
        go(n)
        assert kernel.log == ["slow"], n
