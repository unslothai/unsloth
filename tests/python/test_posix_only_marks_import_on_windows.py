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

Two guards are accepted, both already used in the tree:

    os.name != "posix" or os.geteuid() == 0        short-circuits on Windows
    os.geteuid() == 0 if hasattr(os, "geteuid")    the conditional form

A third form in the tree, `getattr(os, "geteuid", lambda: 1)()`, needs no case: it names
the function with a string, so it never reaches this scan at all. And nothing else counts
as a guard merely for sitting to the left of the call, since `is_ci() or os.geteuid() == 0`
still raises on Windows every time is_ci() is false.

Runtime uses inside a function body are not covered here: they only run on a platform
the test already reached, and a POSIX-only test that gets that far has a skip of its own.
A default argument is not a runtime use, because it is evaluated where the `def` is.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS = REPO_ROOT / "tests"


def _is_os_geteuid(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "geteuid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _mentions_os_name(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "name" and isinstance(n.value, ast.Name)
        and n.value.id == "os"
        for n in ast.walk(node)
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


def _guards(expr: ast.AST) -> list[ast.AST]:
    """Every subexpression that would spare `os.geteuid` on a platform without it."""
    out: list[ast.AST] = []
    for node in ast.walk(expr):
        if isinstance(node, ast.BoolOp):
            # `A or geteuid()` and `A and geteuid()` both spare it when A settles the
            # answer first, so only the values BEFORE the call count as its guard.
            for value in node.values:
                if any(_is_os_geteuid(n) for n in ast.walk(value)):
                    break
                out.append(value)
        elif isinstance(node, ast.IfExp):
            out.append(node.test)
    return out


def _is_guarded(expr: ast.AST) -> bool:
    """Only `os.name` and `hasattr(os, "geteuid")` count. Anything else that merely sits
    to the left of the call is not a guard: `is_ci() or os.geteuid() == 0` still raises
    on Windows every time is_ci() is false, so accepting any call there would wave through
    the exact regression this scan exists to catch.

    `getattr(os, "geteuid", lambda: 1)()` needs no case: it names the function with a
    string, so it has no `os.geteuid` attribute node and never reaches the scan at all."""
    return any(
        _mentions_os_name(guard) or _is_hasattr_geteuid(guard) for guard in _guards(expr)
    )


def _import_time_expressions(tree: ast.AST):
    """Yield the expressions evaluated when the module is imported: the module body,
    class bodies (which run on import too), and every decorator anywhere."""

    def walk(node: ast.AST, import_time: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in child.decorator_list:
                    yield decorator
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # a default is evaluated where the `def` is, not where the call is,
                    # so `def helper(uid = os.geteuid()): ...` breaks collection too
                    args = child.args
                    for default in (*args.defaults, *(d for d in args.kw_defaults if d)):
                        yield default
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
            for node in ast.walk(expr):
                if _is_os_geteuid(node) and not _is_guarded(expr):
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
        'import os, pytest\n'
        '@pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        'def test_a(): pass\n'
    )
    bad_method = ast.parse(
        'import os, pytest\n'
        'class TestB:\n'
        '    @pytest.mark.skipif(os.geteuid() != 0, reason = "x")\n'
        '    def test_b(self): pass\n'
    )
    for tree in (bad_module, bad_method):
        found = [
            expr
            for expr in _import_time_expressions(tree)
            if any(_is_os_geteuid(n) for n in ast.walk(expr)) and not _is_guarded(expr)
        ]
        assert found, "an unguarded os.geteuid decorator no longer trips the scan"

    good = ast.parse(
        'import os, pytest\n'
        '@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason = "x")\n'
        'def test_c(): pass\n'
        '@pytest.mark.skipif(\n'
        '    os.geteuid() == 0 if hasattr(os, "geteuid") else True, reason = "x"\n'
        ')\n'
        'def test_d(): pass\n'
        '_ROOT = getattr(os, "geteuid", lambda: 1)() == 0\n'
    )
    for expr in _import_time_expressions(good):
        if any(_is_os_geteuid(n) for n in ast.walk(expr)):
            assert _is_guarded(expr), ast.dump(expr)


def test_an_unrelated_call_to_the_left_is_not_a_guard():
    """`is_ci() or os.geteuid() == 0` short-circuits only when is_ci() is true, so on
    Windows it still raises the rest of the time. Treating any call as protective would
    wave through exactly what this scan exists to catch."""
    tree = ast.parse(
        'import os, pytest\n'
        '@pytest.mark.skipif(is_ci() or os.geteuid() == 0, reason = "x")\n'
        'def test_a(): pass\n'
    )
    flagged = [
        expr
        for expr in _import_time_expressions(tree)
        if any(_is_os_geteuid(n) for n in ast.walk(expr)) and not _is_guarded(expr)
    ]
    assert flagged, "an unrelated call is being accepted as a Windows guard"


def test_a_default_argument_is_import_time():
    """A default is evaluated where the `def` is, so it breaks collection like a
    decorator does, even though it reads like it belongs to the call."""
    tree = ast.parse("import os\ndef helper(uid = os.geteuid()):\n    return uid\n")
    flagged = [
        expr
        for expr in _import_time_expressions(tree)
        if any(_is_os_geteuid(n) for n in ast.walk(expr)) and not _is_guarded(expr)
    ]
    assert flagged, "a default argument is evaluated at import and must be scanned"


def test_a_runtime_call_inside_a_function_is_not_flagged():
    """Only import-time evaluation breaks collection; a call in a body is the test's
    own business and is left alone, or every POSIX helper in the tree would be noise."""
    tree = ast.parse("import os\ndef test_x():\n    if os.geteuid() == 0:\n        return\n")
    assert not [
        expr for expr in _import_time_expressions(tree) if any(_is_os_geteuid(n) for n in ast.walk(expr))
    ]
