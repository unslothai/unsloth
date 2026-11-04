# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call sizes (token rows, split counts) stay runtime Triton arguments with do_not_specialize, so a
kernel does not compile a new variant per prompt length, batch size or split count."""

import ast
import pathlib

import pytest

KERNELS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "kernels"

CASES = {
    "layernorm.py": {
        "layernorm_backward_dw": ("n_rows", "rows_per_program"),
        "_layernorm_sum_partials": ("n_programs",),
    },
    "fp8.py": {"_w8a8_block_fp8_matmul": ("M",)},
}


def _defs(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _do_not_specialize(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("jit"):
            for kw in dec.keywords:
                if kw.arg == "do_not_specialize":
                    return {e.value for e in kw.value.elts}
    return set()


@pytest.mark.parametrize("rel", sorted(CASES))
def test_runtime_sizes_are_not_specialized(rel):
    defs = _defs(KERNELS / rel)
    for name, args in CASES[rel].items():
        fn = defs[name]
        params = {a.arg: a for a in fn.args.args}
        dns = _do_not_specialize(fn)
        for arg in args:
            assert arg in params, (name, arg)
            ann = params[arg].annotation
            assert ann is None or "constexpr" not in ast.unparse(ann), (name, arg)
            assert arg in dns, (name, arg, sorted(dns))


def test_nvfp4_split_sum_keeps_split_constexpr():
    # The unrolled static_range over SPLIT (2, 4, 8 or 16) is 13-34% faster than a runtime loop.
    fn = _defs(KERNELS / "nvfp4.py")["_split_sum_kernel"]
    ann = {a.arg: a.annotation for a in fn.args.args}["SPLIT"]
    assert ann is not None and "constexpr" in ast.unparse(ann)
