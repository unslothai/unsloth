# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""`os.geteuid` does not exist on Windows, and a `@pytest.mark.skipif` decorator is
evaluated at COLLECTION. An unguarded one therefore does not skip the test on Windows,
it raises AttributeError while the module is being imported and takes the whole file's
collection down with it, which pytest reports as an error for every test in it.

That is invisible to the org CI, which runs the Python suites on Linux only, so it
surfaces on a Windows runner as a file that suddenly has no tests at all. Found exactly
that way: tests/python/test_docker_rocm.py errored on a windows-latest staging run while
Linux and macOS both reported 70 passed.

Two guards are accepted, both already used in the tree, and their POLARITY is checked
rather than their presence:

    os.name != "posix" or os.geteuid() == 0        short-circuits on Windows
    os.geteuid() == 0 if hasattr(os, "geteuid")    the conditional form

`os.name == "posix" or os.geteuid() == 0` mentions the same attribute and spares nothing,
so it is flagged.

A third form in the tree, `getattr(os, "geteuid", lambda: 1)()`, is a guard in itself: the
fallback is what makes it safe, so the two-argument `getattr(os, "geteuid")()` is treated
as the plain lookup it is. And nothing else counts as a guard merely for sitting to the
left of the call, since `is_ci() or os.geteuid() == 0` still raises on Windows every time
is_ci() is false.

Runtime uses inside a function body are not covered here: they only run on a platform
the test already reached, and a POSIX-only test that gets that far has a skip of its own.
A default argument is not a runtime use, because it is evaluated where the `def` is, and
neither is an annotation in a module without `from __future__ import annotations`. A `def`
nested inside another function is the other way round: nothing of it is evaluated until
the outer one runs, so none of it can break collection. Nor is a lambda's body, though
its defaults are.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS = REPO_ROOT / "tests"


def _is_os_geteuid(node: ast.AST) -> bool:
    """A lookup of os.geteuid that raises on Windows. Both spellings: the attribute, and
    the two-argument getattr, which has no fallback and so raises exactly the same way.
    Three-argument getattr does not, and is the form already used in the tree."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "geteuid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "os"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "geteuid"
    )


def _geteuid_sites(expr: ast.AST):
    """Every os.geteuid lookup performed when THIS expression is evaluated. A lambda's
    body is not: it runs when the lambda is called, so `helper = lambda: os.geteuid()`
    is as safe as the same line inside a def. Its defaults are evaluated here and stay."""
    stack = [expr]
    while stack:
        node = stack.pop()
        if _is_os_geteuid(node):
            yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.Lambda) and child is node.body:
                continue
            stack.append(child)


def _is_os_name(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_hasattr_geteuid(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "os"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "geteuid"
    )


def _windows_value(node: ast.AST) -> bool | None:
    """What this expression is worth ON WINDOWS, or None when that is not decidable
    from the source. Polarity is the whole point: `os.name != "posix"` spares the call
    to its right and `os.name == "posix"` does not, and a scan that only looked for the
    words `os.name` would accept both and pass the failure it exists to catch."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _windows_value(node.operand)
        return None if inner is None else not inner
    if _is_hasattr_geteuid(node):
        return False
    if isinstance(node, ast.BoolOp):
        values = [_windows_value(value) for value in node.values]
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            return False if all(value is False for value in values) else None
        if any(value is False for value in values):
            return False
        return True if all(value is True for value in values) else None
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, op, right = node.left, node.ops[0], node.comparators[0]
        if _is_os_name(right) and isinstance(left, ast.Constant):
            left, right = right, left
        if _is_os_name(left) and isinstance(right, ast.Constant) and isinstance(right.value, str):
            equal = right.value == "nt"  # os.name on every Windows CPython
            if isinstance(op, ast.Eq):
                return equal
            if isinstance(op, ast.NotEq):
                return not equal
    return None


def _spared_on_windows(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Walk out from the `os.geteuid` itself and ask each enclosing operator whether it
    can be reached on a platform without the attribute."""
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.BoolOp) and child in parent.values:
            for earlier in parent.values[: parent.values.index(child)]:
                value = _windows_value(earlier)
                if isinstance(parent.op, ast.Or) and value is True:
                    return True  # `os.name != "posix" or geteuid()` never gets there
                if isinstance(parent.op, ast.And) and value is False:
                    return True  # `hasattr(...) and geteuid()` never gets there
        elif isinstance(parent, ast.IfExp):
            test = _windows_value(parent.test)
            if child is parent.body and test is False:
                return True
            if child is parent.orelse and test is True:
                return True
        child, parent = parent, parents.get(parent)
    return False


def _is_guarded(expr: ast.AST) -> bool:
    """Whether every `os.geteuid` in this expression is unreachable on Windows.

    Only `os.name` comparisons and `hasattr(os, "geteuid")` decide anything. Nothing
    else counts for merely sitting to the left of the call: `is_ci() or os.geteuid() == 0`
    still raises every time is_ci() is false.

    The three-argument `getattr(os, "geteuid", lambda: 1)()` needs no case here: the
    fallback means no lookup can fail, so it is not a site at all."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(expr):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return all(_spared_on_windows(node, parents) for node in ast.walk(expr) if _is_os_geteuid(node))


def _definition_expressions(node: ast.AST, eager_annotations: bool):
    """The parts of a `def` or `class` evaluated where it is written, not where it runs."""
    yield from node.decorator_list
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    args = node.args
    yield from args.defaults
    yield from (default for default in args.kw_defaults if default)
    if eager_annotations:
        # `from __future__ import annotations` turns these into strings; without it they
        # are evaluated at the `def`, so `def helper(uid: os.geteuid()): ...` breaks
        # collection exactly like a decorator does
        if node.returns is not None:
            yield node.returns
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        yield from (arg.annotation for arg in every if arg is not None and arg.annotation)


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _import_time_expressions(tree: ast.Module):
    """Yield the expressions evaluated when the module is imported: the module body,
    class bodies (which run on import too), and the decorators, defaults and eager
    annotations of every definition reached at import."""
    eager_annotations = not _has_future_annotations(tree)

    def walk(node: ast.AST, import_time: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if import_time:
                    # A `def` nested in another function is not evaluated until the outer
                    # one runs, so its decorators and defaults cannot break collection and
                    # are not flagged. Only a definition reached at import counts.
                    yield from _definition_expressions(child, eager_annotations)
                yield from walk(child, isinstance(child, ast.ClassDef) and import_time)
            else:
                if import_time:
                    # the WHOLE statement, never its subexpressions as well: a guard
                    # lives in the enclosing `or`, and yielding the bare call too would
                    # report every guarded site as unguarded
                    yield child
                yield from walk(child, False)

    yield from walk(tree, True)


def test_no_test_module_calls_os_geteuid_unguarded_at_import():
    offenders = []
    checked = 0
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue  # not ours to parse; the lint job owns syntax
        checked += 1
        for expr in _import_time_expressions(tree):
            for node in _geteuid_sites(expr):
                if not _is_guarded(expr):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                    break
    assert checked > 100, f"only {checked} test modules parsed; the scan lost its tree"
    assert not offenders, (
        "os.geteuid is evaluated at import with nothing to spare it on Windows, so the "
        "whole module fails to collect there rather than skipping:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_the_scan_still_recognises_an_unguarded_call():
    """The scan is only worth anything if it would still fail. Both shapes, since the
    decorator on a method inside a class is how the live one was written."""
    bad_module = ast.parse(
        "import os, pytest\n"
        '@pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "def test_a(): pass\n"
    )
    bad_method = ast.parse(
        "import os, pytest\n"
        "class TestB:\n"
        '    @pytest.mark.skipif(os.geteuid() != 0, reason = "x")\n'
        "    def test_b(self): pass\n"
    )
    for tree in (bad_module, bad_method):
        found = [
            expr
            for expr in _import_time_expressions(tree)
            if any(True for _ in _geteuid_sites(expr)) and not _is_guarded(expr)
        ]
        assert found, "an unguarded os.geteuid decorator no longer trips the scan"

    good = ast.parse(
        "import os, pytest\n"
        '@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason = "x")\n'
        "def test_c(): pass\n"
        "@pytest.mark.skipif(\n"
        '    os.geteuid() == 0 if hasattr(os, "geteuid") else True, reason = "x"\n'
        ")\n"
        "def test_d(): pass\n"
        '_ROOT = getattr(os, "geteuid", lambda: 1)() == 0\n'
    )
    for expr in _import_time_expressions(good):
        if any(True for _ in _geteuid_sites(expr)):
            assert _is_guarded(expr), ast.dump(expr)


def test_an_unrelated_call_to_the_left_is_not_a_guard():
    """`is_ci() or os.geteuid() == 0` short-circuits only when is_ci() is true, so on
    Windows it still raises the rest of the time. Treating any call as protective would
    wave through exactly what this scan exists to catch."""
    tree = ast.parse(
        "import os, pytest\n"
        '@pytest.mark.skipif(is_ci() or os.geteuid() == 0, reason = "x")\n'
        "def test_a(): pass\n"
    )
    flagged = [
        expr
        for expr in _import_time_expressions(tree)
        if any(True for _ in _geteuid_sites(expr)) and not _is_guarded(expr)
    ]
    assert flagged, "an unrelated call is being accepted as a Windows guard"


def test_a_default_argument_is_import_time():
    """A default is evaluated where the `def` is, so it breaks collection like a
    decorator does, even though it reads like it belongs to the call."""
    tree = ast.parse("import os\ndef helper(uid = os.geteuid()):\n    return uid\n")
    flagged = [
        expr
        for expr in _import_time_expressions(tree)
        if any(True for _ in _geteuid_sites(expr)) and not _is_guarded(expr)
    ]
    assert flagged, "a default argument is evaluated at import and must be scanned"


def test_a_runtime_call_inside_a_function_is_not_flagged():
    """Only import-time evaluation breaks collection; a call in a body is the test's
    own business and is left alone, or every POSIX helper in the tree would be noise."""
    tree = ast.parse("import os\ndef test_x():\n    if os.geteuid() == 0:\n        return\n")
    assert not [
        expr
        for expr in _import_time_expressions(tree)
        if any(_is_os_geteuid(n) for n in ast.walk(expr))
    ]


def _flagged(source: str) -> list[int]:
    tree = ast.parse(source)
    return [
        node.lineno
        for expr in _import_time_expressions(tree)
        for node in _geteuid_sites(expr)
        if not _is_guarded(expr)
    ]


def test_an_os_name_guard_is_read_for_polarity_not_presence():
    """`os.name == "posix" or os.geteuid() == 0` names the same attribute as the real
    guard and spares nothing: on Windows the left side is false, so the call is reached.
    A scan that matched the words alone would pass exactly this."""
    assert _flagged(
        "import os, pytest\n"
        '@pytest.mark.skipif(os.name == "posix" or os.geteuid() == 0, reason = "x")\n'
        "def test_a(): pass\n"
    )
    assert _flagged(
        "import os, pytest\n"
        '@pytest.mark.skipif(os.name != "posix" and os.geteuid() == 0, reason = "x")\n'
        "def test_b(): pass\n"
    )
    # the two that do spare it, and the mirrored and negated spellings of each
    for guard in (
        'os.name != "posix" or os.geteuid() == 0',
        '"posix" != os.name or os.geteuid() == 0',
        'os.name == "nt" or os.geteuid() == 0',
        'not (os.name == "posix") or os.geteuid() == 0',
        'os.name == "posix" and os.geteuid() == 0',
        'hasattr(os, "geteuid") and os.geteuid() == 0',
        'os.geteuid() == 0 if hasattr(os, "geteuid") else True',
    ):
        assert not _flagged(
            f'import os, pytest\n@pytest.mark.skipif({guard}, reason = "x")\ndef test_c(): pass\n'
        ), guard


def test_an_annotation_is_import_time_only_without_the_future_import():
    """Without `from __future__ import annotations` an annotation is evaluated at the
    `def`; with it, it is a string and cannot raise."""
    body = "import os\ndef helper(uid: os.geteuid() = 1) -> os.geteuid():\n    return uid\n"
    assert _flagged(body)
    assert not _flagged("from __future__ import annotations\n" + body)


def test_a_nested_definition_is_not_import_time():
    """A `def` inside a function is not evaluated until the outer one runs, so its
    decorator cannot break collection and must not be reported."""
    assert not _flagged(
        "import os, pytest\n"
        "def test_outer():\n"
        '    @pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "    def inner(uid = os.geteuid()): pass\n"
    )
    # but a method of a class defined at module level still is
    assert _flagged(
        "import os, pytest\n"
        "class TestA:\n"
        '    @pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "    def test_b(self): pass\n"
    )


def test_a_two_argument_getattr_is_the_same_lookup():
    """`getattr(os, "geteuid")()` raises exactly as `os.geteuid()` does; only the
    three-argument form has a fallback, and that is the one already in the tree."""
    assert _flagged(
        "import os, pytest\n"
        '@pytest.mark.skipif(getattr(os, "geteuid")() == 0, reason = "x")\n'
        "def test_a(): pass\n"
    )
    assert not _flagged('import os\n_ROOT = getattr(os, "geteuid", lambda: 1)() == 0\n')


def test_a_lambda_body_is_not_import_time():
    """`helper = lambda: os.geteuid()` does not look anything up until it is called, so
    it is as safe as the same line inside a def. A lambda's defaults are evaluated at the
    lambda, so those still count."""
    assert not _flagged("import os\nhelper = lambda: os.geteuid()\n")
    assert _flagged("import os\nhelper = lambda uid = os.geteuid(): uid\n")
