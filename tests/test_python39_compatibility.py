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

# studio/ ships under the same requires-python, so it is held to the syntax check. Its
# evaluated-union debt is ratcheted rather than fixed here: the 35 files involved include
# FastAPI routers and pydantic models, where `from __future__ import annotations` is
# supported but has real failure modes around class dependencies, so converting them needs
# Studio actually booted and its routes exercised. The ratchet stops the debt growing.
STUDIO_UNION_DEBT = 35


def declared_floor():
    """The ``>=X.Y`` in requires-python, as a tuple for ast.parse(feature_version=)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def package_files(root = PACKAGE_ROOT, minimum = 50):
    files = sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "node_modules" not in p.parts
    )
    assert len(files) >= minimum, f"only found {len(files)} files under {root}, glob is wrong"
    return files


def packaged_roots():
    """Top-level directories setuptools ships, from the `include` list in pyproject.

    Read rather than hardcoded so a newly packaged directory cannot silently escape the
    floor guarantee.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    block = re.search(r"^include\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert block, "no packages.find include list in pyproject.toml"
    roots = []
    for pattern in re.findall(r"[\"']([^\"']+)[\"']", block.group(1)):
        top = pattern.split(".")[0].rstrip("*")
        candidate = REPO_ROOT / top
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    assert roots, "no packaged directories resolved from pyproject"
    return roots


def evaluated_union_files(root):
    """Files under `root` that would raise on the floor, i.e. need the future import."""
    offenders = set()
    for path in package_files(root, minimum = 1):
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        if has_future_annotations(tree):
            continue
        for expression in evaluated_annotations(tree):
            for inner in ast.walk(expression):
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                    offenders.add(path)
    return offenders


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
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, True)
            elif isinstance(child, ast.ClassDef):
                walk(child, False)   # a class body is class scope wherever it sits
            else:
                walk(child, in_function)

    walk(tree, False)
    return out


# A `|` between these is a union, not arithmetic: builtin types, `None`, and whatever the
# module pulled in from typing.
TYPE_ANCHORS = frozenset({
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object", "type",
    "list", "dict", "tuple", "set", "frozenset",
})
TYPING_MODULES = frozenset({"typing", "typing_extensions", "collections.abc"})


def typing_names(tree):
    """Names this module bound with ``from typing import X`` and friends."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in TYPING_MODULES
        for alias in node.names
    }


def union_operands(node):
    """Operands of an ``A | B | C`` chain, or None if any part is not name-shaped."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = union_operands(node.left), union_operands(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Subscript):
        return union_operands(node.value)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return [node]
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return [node]
    return None


def looks_like_a_type_alias(node, known_typing_names):
    """``PathLike = str | Path`` yes; ``defaults | extra`` and ``re.A | re.M`` no.

    Every operand must be name-shaped and at least one must be a recognisable type.
    Without that anchor a ``|`` between plain names is far more likely to be a dict merge
    (PEP 584, valid on 3.9), a set union or flag arithmetic, and failing the gate on those
    would block code that runs perfectly well on the floor.
    """
    operands = union_operands(node)
    if not operands:
        return False
    for operand in operands:
        if isinstance(operand, ast.Constant):
            if operand.value is None:
                return True
            continue  # a bare string is a forward reference, never an anchor alone
        name = operand.id if isinstance(operand, ast.Name) else operand.attr
        if name in TYPE_ANCHORS or name in known_typing_names:
            return True
    return False


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


def test_every_packaged_module_parses_on_the_declared_floor():
    """Every directory pyproject ships, not just unsloth/ - studio/ is packaged too."""
    floor = declared_floor()
    broken = []
    for root in packaged_roots():
        for path in package_files(root, minimum = 1):
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
        f"{floor[0]}.{floor[1]} in a packaged directory; either rewrite it or raise "
        "requires-python:\n  " + "\n  ".join(broken)
    )


def test_studio_evaluated_unions_do_not_grow():
    """studio/ is shipped on the same floor but still carries unions that raise there.

    A ratchet, not a pass: converting those files needs Studio booted and its routes
    exercised, because FastAPI resolves annotations when it builds each endpoint. This
    keeps the debt from growing in the meantime.
    """
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    studio = REPO_ROOT / "studio"
    if not studio.is_dir():
        pytest.skip("no studio/ directory in this checkout")
    offenders = evaluated_union_files(studio)
    assert len(offenders) <= STUDIO_UNION_DEBT, (
        f"{len(offenders)} studio files now evaluate PEP 604 unions on the floor, up from "
        f"{STUDIO_UNION_DEBT}. Add `from __future__ import annotations` to the new ones:\n  "
        + "\n  ".join(sorted(str(p.relative_to(REPO_ROOT)) for p in offenders))[:2000]
    )


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """``X | Y`` is a TypeError below 3.10 unless the module defers it."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    for path in package_files():
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        where = path.relative_to(REPO_ROOT)
        known_typing_names = typing_names(tree)
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
                    and looks_like_a_type_alias(inner, known_typing_names)
                ):
                    offenders.append(f"{where}:{inner.lineno}: {ast.unparse(inner)} (type alias)")
    assert not offenders, (
        "PEP 604 unions evaluated below 3.10. Annotations are deferred by `from __future__ "
        "import annotations`; type aliases need typing.Union, which that import does NOT "
        "defer:\n  " + "\n  ".join(sorted(set(offenders)))
    )
