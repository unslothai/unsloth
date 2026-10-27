# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""autograd.Function forwards must not ATen-mutate a fresh tensor they return: torch 2.11 compiles a wrong
gradient for it (#14712). Kernel writes are fine; finish out of place (`out = out + x`)."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth"
ALLOC = {
    "empty",
    "empty_like",
    "empty_strided",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "new_empty",
    "new_zeros",
    "new_ones",
    "new_full",
}
NOT_MUTATING = {"requires_grad_", "share_memory_", "record_stream"}


def _call_name(node):
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else None


def _base_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _returned_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
            names.update(filter(None, map(_base_name, values)))
    return names


def _violations(source, path = "<src>"):
    found = []
    for cls in ast.walk(ast.parse(source)):
        if not isinstance(cls, ast.ClassDef):
            continue
        if not any("Function" in ast.unparse(base) for base in cls.bases):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "forward"):
                continue
            allocated = {
                node.targets[0].id
                for node in ast.walk(fn)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) in ALLOC
            }
            hot = allocated & _returned_names(fn)
            for node in ast.walk(fn):
                if isinstance(node, ast.AugAssign) and _base_name(node.target) in hot:
                    found.append(f"{path}:{node.lineno} {cls.name}: {ast.unparse(node)}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    if (
                        name.endswith("_")
                        and not name.startswith("_")
                        and name not in NOT_MUTATING
                        and _base_name(node.func.value) in hot
                    ):
                        found.append(f"{path}:{node.lineno} {cls.name}: {ast.unparse(node)}")
    return found


def test_no_inplace_mutation_of_returned_fresh_tensors():
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        found += _violations(path.read_text(encoding = "utf-8"), path.relative_to(ROOT.parent))
    assert not found, "\n".join(found)


def test_detector_flags_the_cross_entropy_and_gather_shapes():
    cross_entropy = """
class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        losses = torch.empty(logits.shape[0], device = logits.device)
        kernel(logits, losses)
        losses += lse
        losses.masked_fill_(labels == -100, 0)
        return losses
"""
    gather = """
class G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = x.new_empty(x.shape)
        return out.copy_(x)
"""
    fixed = """
class H(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q):
        out = torch.empty_like(x)
        kernel(x, out)
        out = (out + 1).masked_fill(x == 0, 0)
        q *= 2
        return out, q
"""
    assert len(_violations(cross_entropy)) == 2
    assert len(_violations(gather)) == 1
    assert _violations(fixed) == []
