# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-call sizes stay do_not_specialize (no recompile per shape); every tl.constexpr says why it is static."""

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
    "rope_embedding.py": {"_rope_embedding_QK": ("seqlen",)},
    "qk_norm_rope.py": {
        "_qk_norm_rope_forward": ("seqlen",),
        "_qk_norm_rope_backward": ("seqlen",),
    },
}

# Must stay specialized: % 16 hint on masked/vector loads; NF4 GEMV M == 1 folds the row split.
KEEP_SPECIALIZED = {
    "swiglu.py": {"_fg_kernel": ("n_elements",), "_DWf_DW_dfg_kernel": ("n_elements",)},
    "geglu.py": {
        "_exact_forward_kernel": ("n_elements",),
        "_approx_forward_kernel": ("n_elements",),
    },
    "nf4.py": {"_nf4_dequant_kernel": ("n_elements", "n_bytes")},
    "nf4_gemv.py": {
        "_gemv_nf4_kernel": ("OUT_STRIDE", "K", "M"),
        "_gemv_nf4_words_kernel": ("OUT_STRIDE", "K", "M"),
    },
    "fp8_lora.py": {"_fp8_tensorwise_cast_kernel": ("numel",)},
}

COMMENTED = ("nf4.py", "nf4_gemv.py", "fp8_lora.py", "multi_lora.py")


def _tree(rel):
    path = KERNELS / rel
    return path.read_text(encoding = "utf-8"), ast.parse(path.read_text(encoding = "utf-8"))


def _defs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _names(node):
    return {e.value for e in node.elts} if isinstance(node, (ast.Tuple, ast.List)) else set()


def _do_not_specialize(tree, fn):
    calls = [d for d in fn.decorator_list if isinstance(d, ast.Call)]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == fn.name
        ):
            calls.append(node.value)
    out = set()
    for call in calls:
        if ast.unparse(call.func).endswith("jit"):
            for kw in call.keywords:
                if kw.arg == "do_not_specialize":
                    out |= _names(kw.value)
    return out


def _is_constexpr(arg):
    return arg.annotation is not None and "constexpr" in ast.unparse(arg.annotation)


@pytest.mark.parametrize("rel", sorted(CASES))
def test_runtime_sizes_are_not_specialized(rel):
    _, tree = _tree(rel)
    defs = _defs(tree)
    for name, args in CASES[rel].items():
        fn = defs[name]
        params = {a.arg: a for a in fn.args.args}
        dns = _do_not_specialize(tree, fn)
        for arg in args:
            assert arg in params, (name, arg)
            assert not _is_constexpr(params[arg]), (name, arg)
            assert arg in dns, f"{rel}:{name}: {arg} varies per call, needs do_not_specialize"


@pytest.mark.parametrize("rel", sorted(KEEP_SPECIALIZED))
def test_vector_sizes_keep_specialization(rel):
    _, tree = _tree(rel)
    defs = _defs(tree)
    for name, args in KEEP_SPECIALIZED[rel].items():
        fn = defs[name]
        params = {a.arg for a in fn.args.args}
        dns = _do_not_specialize(tree, fn)
        for arg in args:
            assert arg in params, (name, arg)
            assert arg not in dns, f"{rel}:{name}: {arg} needs its % 16 hint"


@pytest.mark.parametrize("rel", COMMENTED)
def test_every_constexpr_is_commented(rel):
    src, tree = _tree(rel)
    lines = src.splitlines()
    missing = []
    for fn in _defs(tree).values():
        cx = [a for a in fn.args.args if _is_constexpr(a)]
        if not cx:
            continue
        end = fn.body[0].lineno - 1
        comments = " ".join(l.split("#", 1)[1] for l in lines[fn.lineno - 1 : end] if "#" in l)
        for a in cx:
            own = lines[a.lineno - 1]
            if "#" not in own and a.arg not in comments:
                missing.append(f"{fn.name}.{a.arg}")
    assert not missing, f"{rel}: constexpr without a why-static comment: {missing}"


def test_nvfp4_split_sum_keeps_split_constexpr():
    # The unrolled static_range over SPLIT (2, 4, 8 or 16) is 13-34% faster than a runtime loop.
    fn = _defs(_tree("nvfp4.py")[1])["_split_sum_kernel"]
    ann = {a.arg: a.annotation for a in fn.args.args}["SPLIT"]
    assert ann is not None and "constexpr" in ast.unparse(ann)
