# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Guards `_attn_mask_compat.py` against drifting from the upstream it vendors.

`unsloth/models/_attn_mask_compat.py` is a copy of Transformers'
`modeling_attn_mask_utils.py`, kept because that module is deprecated and will be
deleted. A hand-copied file drifts silently: two defects during review of #6880
were both invisible in the diff and only surfaced under differential testing.

So compare the two ASTs directly, after erasing differences that are stylistic
rather than semantic:

  * docstrings, type-annotation text, and the deprecation `warnings.warn` calls
    the vendored copy exists to drop;
  * `is_tracing`, which 4.x computes inline as a 3-way `or` and 5.x exposes as a
    helper the vendored copy imports (with a fallback);
  * single-use temporaries and a dead `else` after a `return`.

Two further differences are deliberate: the vendored copy carries forward-ports
that older Transformers lacks. Those are relaxed **only** below the version that
introduced them, so on newer installs they are still compared exactly:

  * the `xpu` device gate, added in 5.x;
  * the 0-dim inversion tensor, added in 4.53.0 (huggingface/transformers#38637).

Skips when the upstream module is gone, which is the end state this file is for.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPAT_PATH = _REPO_ROOT / "unsloth" / "models" / "_attn_mask_compat.py"


def _upstream_source():
    try:
        legacy = importlib.import_module("transformers.modeling_attn_mask_utils")
    except ImportError:
        pytest.skip("transformers.modeling_attn_mask_utils removed upstream")
    try:
        return inspect.getsource(legacy)
    except (OSError, TypeError):
        pytest.skip("upstream source unavailable (zipimport or compiled install)")


def _transformers_version():
    import transformers

    parts = []
    for chunk in transformers.__version__.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def _is_deprecation_warning(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and "warn" in ast.dump(stmt.value.func)
        and "DEPRECATION_MESSAGE" in ast.dump(stmt.value)
    )


def _inline_tracing_expr(node):
    """Recognise 4.x's inline `torch.jit.is_tracing() or isinstance(...) or ...`."""
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return None
    dumped = ast.dump(node)
    if "is_tracing" not in dumped or "is_torchdynamo_compiling" not in dumped:
        return None
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "isinstance"
            and sub.args
        ):
            return sub.args[0]
    return ast.Constant(value = None)


class _Canonicalise(ast.NodeTransformer):
    def __init__(self, relax_device, relax_inversion):
        self.relax_device = relax_device
        self.relax_inversion = relax_inversion

    def _clean(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        node.body = [s for s in node.body if not _is_deprecation_warning(s)] or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node = self._clean(node)
        node.returns = None
        for arg in list(node.args.args) + list(node.args.kwonlyargs) + list(node.args.posonlyargs):
            arg.annotation = None
        node.body = _hoist_dead_else(_inline_single_use(node.body))
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._clean(node)

    def visit_AnnAssign(self, node):
        # Dataclass field: the explicit __init__ makes any default unreachable.
        return ast.AnnAssign(
            target = node.target,
            annotation = ast.Name(id = "_", ctx = ast.Load()),
            value = None,
            simple = node.simple,
        )

    def visit_Name(self, node):
        # `is_tracing_` only avoids shadowing the imported helper.
        if node.id == "is_tracing_":
            return ast.Name(id = "is_tracing", ctx = node.ctx)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        arg = _inline_tracing_expr(node)
        if arg is not None:
            return ast.Call(func = ast.Name(id = "_TRACING", ctx = ast.Load()), args = [arg], keywords = [])
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "is_tracing":
            return ast.Call(
                func = ast.Name(id = "_TRACING", ctx = ast.Load()),
                args = list(node.args),
                keywords = [],
            )
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.relax_device and "'cuda'" in ast.dump(node):
            return ast.Name(id = "_DEVICE_GATE", ctx = ast.Load())
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (
            self.relax_inversion
            and isinstance(node.op, ast.Sub)
            and "value=1.0" in ast.dump(node.left)
        ):
            return ast.BinOp(
                left = ast.Name(id = "_ONE", ctx = ast.Load()), op = ast.Sub(), right = node.right
            )
        return node


def _inline_single_use(body):
    """`t = expr` with exactly one later read of `t` becomes that expr inline."""
    out = list(body)
    changed = True
    while changed:
        changed = False
        for i, stmt in enumerate(out[:-1]):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                continue
            name = stmt.targets[0].id
            rest = out[i + 1 :]
            reads = sum(
                1
                for s in rest
                for n in ast.walk(s)
                if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
            )
            writes = sum(
                1
                for s in rest
                for n in ast.walk(s)
                if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
            )
            if reads != 1 or writes:
                continue

            class _Sub(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id == name and isinstance(n.ctx, ast.Load):
                        return stmt.value
                    return n

            out = out[:i] + [_Sub().visit(s) for s in rest]
            changed = True
            break
    return out


def _hoist_dead_else(body):
    """`if c: return A` + `else: B` is the same as the `if` followed by `B`."""
    out = []
    for stmt in body:
        if (
            isinstance(stmt, ast.If)
            and stmt.orelse
            and stmt.body
            and isinstance(stmt.body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))
        ):
            tail = stmt.orelse
            stmt.orelse = []
            out.append(stmt)
            out.extend(_hoist_dead_else(tail))
        else:
            out.append(stmt)
    return out


def _symbols(source, relax_device, relax_inversion):
    tree = _Canonicalise(relax_device, relax_inversion).visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef):
                        found[f"{node.name}.{member.name}"] = member
    return found


def test_vendored_module_has_not_drifted_from_upstream():
    upstream_src = _upstream_source()
    version = _transformers_version()
    relax_device = version < (5, 0)  # xpu gate forward-ported from 5.x
    relax_inversion = version < (4, 53)  # 0-dim inversion forward-ported from 4.53.0

    upstream = _symbols(upstream_src, relax_device, relax_inversion)
    vendored = _symbols(_COMPAT_PATH.read_text(encoding = "utf-8"), relax_device, relax_inversion)

    shared = sorted(set(upstream) & set(vendored))
    assert shared, "no shared symbols found; the comparison is not doing anything"

    drifted = []
    for name in shared:
        if ast.dump(upstream[name]) != ast.dump(vendored[name]):
            drifted.append(
                f"\n--- {name} ---\nupstream:\n{ast.unparse(upstream[name])}\n"
                f"vendored:\n{ast.unparse(vendored[name])}"
            )

    assert not drifted, (
        f"{len(drifted)} symbol(s) drifted from transformers "
        f"{'.'.join(str(v) for v in version)}. Re-sync `_attn_mask_compat.py`, or if the "
        f"divergence is deliberate, relax it here behind a version gate and say why."
        + "".join(drifted)
    )


def test_vendored_module_exports_everything_unsloth_imports():
    """The copy may be a subset of upstream, but not of what Unsloth uses."""
    upstream_src = _upstream_source()
    upstream = _symbols(upstream_src, False, False)
    vendored = _symbols(_COMPAT_PATH.read_text(encoding = "utf-8"), False, False)

    # Anything vendored must actually exist upstream; inventing symbols under an upstream module's name would be a
    # silent behavioural fork.
    invented = sorted(set(vendored) - set(upstream))
    assert invented == [], f"vendored symbols with no upstream counterpart: {invented}"
