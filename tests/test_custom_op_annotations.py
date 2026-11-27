# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Op schemas are inferred from annotations. Under `from __future__ import annotations` they are strings that
torch must resolve itself, and older torch resolves only a few names (a failure is swallowed or breaks import).
Functions registered as ops in such modules may only use annotations proven to infer on the torch floor (2.6)."""

import ast
import pathlib

UNSLOTH = pathlib.Path(__file__).resolve().parents[1] / "unsloth"

# Every annotation string inferred cleanly on torch 2.6.0 and 2.16 nightly.
_SAFE = {"torch.Tensor", "Optional[torch.Tensor]", "int", "bool", "float", "None"}
# Calls / decorators that infer a schema from the function they are given.
_REGISTRARS = {"custom_op", "triton_op", "infer_schema", "_opaque_under_compile"}


def _name(node):
    target = node.func if isinstance(node, ast.Call) else node
    return getattr(target, "attr", getattr(target, "id", None))


def _postponed(tree):
    return any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )


def _registered(tree):
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    out = {}
    for fn in defs.values():
        if any(_name(d) in _REGISTRARS for d in fn.decorator_list):
            out[fn.name] = fn
    for node in ast.walk(tree):
        # torch.library.triton_op(name, ...)(fn)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Call)
            and _name(node.func) in _REGISTRARS
        ):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in defs:
                    out[arg.id] = defs[arg.id]
    return out


def _scan():
    for path in sorted(UNSLOTH.rglob("*.py")):
        src = path.read_text(encoding = "utf-8")
        if not any(r in src for r in _REGISTRARS):
            continue
        tree = ast.parse(src)
        if _postponed(tree):
            yield path.relative_to(UNSLOTH).as_posix(), _registered(tree)


def _annotations(fn):
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    out = [(a.arg, a.annotation) for a in args if a.annotation is not None]
    if fn.returns is not None:
        out.append(("return", fn.returns))
    return out


def test_the_scan_sees_postponed_op_modules():
    found = {rel: set(fns) for rel, fns in _scan()}
    assert {"_launch_seg_mm", "_launch_seg_wgrad"} <= found.get(
        "kernels/multi_lora.py", set()
    ), found
    assert "fp8_lora_fwd" in found.get("kernels/fp8_lora.py", set()), found


def test_registered_ops_use_floor_safe_annotations():
    bad = []
    for rel, fns in _scan():
        for name, fn in fns.items():
            for arg, ann in _annotations(fn):
                if ast.unparse(ann) not in _SAFE:
                    bad.append(f"{rel}:{fn.lineno}:{name}({arg}: {ast.unparse(ann)})")
    assert not bad, (
        "op registered from a `from __future__ import annotations` module with an annotation torch 2.6 may "
        "not resolve: " + ", ".join(bad)
    )


def test_the_annotation_check_can_fail():
    tree = ast.parse(
        "from __future__ import annotations\n"
        "def f(x: torch.Tensor, n: Tuple[int, int]) -> torch.Tensor: ...\n"
        "op = torch.library.triton_op('u::f', mutates_args = ())(f)\n"
    )
    fns = _registered(tree)
    assert set(fns) == {"f"}
    assert [ast.unparse(a) for _, a in _annotations(fns["f"]) if ast.unparse(a) not in _SAFE] == [
        "Tuple[int, int]"
    ]
