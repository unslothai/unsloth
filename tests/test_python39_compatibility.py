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
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def package_files():
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(files) > 50, f"only found {len(files)} files, the glob is wrong"
    return files


def annotations_of(node):
    """Annotation expressions that this node evaluates at def/exec time."""
    if isinstance(node, ast.AnnAssign):
        return [node.annotation] if node.annotation else []
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    out = [a.annotation for a in
           [*args.args, *args.kwonlyargs, *args.posonlyargs, args.vararg, args.kwarg]
           if a is not None and a.annotation is not None]
    if node.returns:
        out.append(node.returns)
    return out


def has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(alias.name == "annotations" for alias in n.names)
               for n in tree.body)


def test_every_module_parses_on_the_declared_floor():
    floor = declared_floor()
    broken = []
    for path in package_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
                      feature_version=floor)
        except SyntaxError as error:
            broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, (
        "syntax newer than the declared floor "
        f"{floor[0]}.{floor[1]}; either rewrite it or raise requires-python:\n  "
        + "\n  ".join(broken)
    )


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """`X | Y` is a TypeError below 3.10 unless the module defers annotations."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    for path in package_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if has_future_annotations(tree):
            continue
        for node in ast.walk(tree):
            for annotation in annotations_of(node):
                for inner in ast.walk(annotation):
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{inner.lineno}: "
                            f"{ast.unparse(inner)}"
                        )
    assert not offenders, (
        "PEP 604 unions evaluated at def time; add `from __future__ import "
        "annotations` to these modules:\n  " + "\n  ".join(sorted(set(offenders)))
    )
