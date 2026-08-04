"""The package must stay importable on the Python floor ``pyproject.toml`` declares.

It was not. ``requires-python`` says ``>=3.9`` while ``registry/registry.py`` annotated a
dataclass field ``list[QuantType] | dict[str, list[QuantType]]``, which evaluates at class
creation, so importing ``unsloth.registry.registry`` died with ``TypeError: unsupported
operand type(s) for |``. Nothing caught it: every CI job here pins 3.12.

The floor is read from ``pyproject.toml`` rather than hardcoded, so raising
``requires-python`` relaxes these checks instead of turning them into a false alarm.

This is a static AST check. It imports nothing from the package, so it needs no torch, no
GPU and no network, and it sees files that are never imported at test time.
"""

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "unsloth"


def declared_floor():
    """The ``>=X.Y`` in requires-python, as a tuple for ast.parse(feature_version=)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def package_files():
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(files) > 50, f"only found {len(files)} files, the glob is wrong"
    return files


def signature_annotations(node):
    """Parameter and return annotations, evaluated when the ``def`` executes."""
    args = node.args
    out = [
        a.annotation
        for a in [*args.args, *args.kwonlyargs, *args.posonlyargs, args.vararg, args.kwarg]
        if a is not None and a.annotation is not None
    ]
    if node.returns:
        out.append(node.returns)
    return out


def evaluated_annotations(tree):
    """Annotations Python evaluates, so the future import is what defers them.

    Variable annotations are evaluated at module and class scope only; inside a function
    body they are never evaluated, so flagging those is a false positive. Signature
    annotations are evaluated wherever their ``def`` is.
    """
    out = []

    def walk(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AnnAssign) and child.annotation and not in_function:
                out.append(child.annotation)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(signature_annotations(child))
            nested = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            walk(child, in_function or nested)

    walk(tree, False)
    return out


def looks_like_a_type(node):
    """Conservative: enough for ``str | Path``, not enough for ``re.A | re.M``.

    Flag constants are ALL_CAPS by convention, so requiring a non-caps name keeps bitwise
    arithmetic (``re.DOTALL | re.MULTILINE``, ``os.W_OK | os.X_OK``) out.
    """
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Subscript):
        return looks_like_a_type(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return looks_like_a_type(node.left) and looks_like_a_type(node.right)
    name = getattr(node, "id", None) or getattr(node, "attr", None)
    return bool(name) and not name.isupper()


def evaluated_values(tree):
    """Assigned values at module and class scope, which run on import.

    Type aliases are the case that matters: ``PathLike = str | Path`` raises below 3.10
    and, unlike an annotation, the future import does not defer it.
    """
    out = []

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                out.append(child.value)
            if isinstance(child, ast.ClassDef):
                walk(child)

    walk(tree)
    return out


def has_future_annotations(tree):
    return any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(alias.name == "annotations" for alias in n.names)
        for n in tree.body
    )


def test_every_module_parses_on_the_declared_floor():
    floor = declared_floor()
    broken = []
    for path in package_files():
        try:
            ast.parse(
                path.read_text(encoding = "utf-8"),
                filename = str(path),
                feature_version = floor,
            )
        except SyntaxError as error:
            broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, (
        "syntax newer than the declared floor "
        f"{floor[0]}.{floor[1]}; either rewrite it or raise requires-python:\n  "
        + "\n  ".join(broken)
    )


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """``X | Y`` is a TypeError below 3.10 unless the module defers it."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    for path in package_files():
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        where = path.relative_to(REPO_ROOT)
        # The future import defers annotations only; an assigned value still runs.
        deferred = has_future_annotations(tree)
        for expression in [] if deferred else evaluated_annotations(tree):
            for inner in ast.walk(expression):
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                    offenders.append(f"{where}:{inner.lineno}: {ast.unparse(inner)} (annotation)")
        for expression in evaluated_values(tree):
            for inner in ast.walk(expression):
                if (
                    isinstance(inner, ast.BinOp)
                    and isinstance(inner.op, ast.BitOr)
                    and looks_like_a_type(inner)
                ):
                    offenders.append(f"{where}:{inner.lineno}: {ast.unparse(inner)} (type alias)")
    assert not offenders, (
        "PEP 604 unions evaluated below 3.10. Annotations are deferred by `from __future__ "
        "import annotations`; type aliases need typing.Union, which that import does NOT "
        "defer:\n  " + "\n  ".join(sorted(set(offenders)))
    )
