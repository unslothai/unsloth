# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""An unguarded `os.geteuid` in a decorator or import-time expression raises at collection on Windows."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS = REPO_ROOT / "tests"


def _is_os_geteuid(node: ast.AST) -> bool:
    """Matches `os.geteuid` lookups that fail on Windows: the attribute and the two-argument `getattr`."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "geteuid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        # An augmented target is a Store but reads first; _geteuid_sites admits it.
        and not isinstance(node.ctx, ast.Store)
    ):
        return True
    call = _getattr_geteuid(node)
    return call is not None and len(call.args) == 2


def _is_augmented_geteuid(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "geteuid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _getattr_geteuid(node: ast.AST) -> ast.Call | None:
    """The `getattr(os, "geteuid", ...)` call this node is, if it is one."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and not node.keywords
        and len(node.args) in (2, 3)
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "os"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "geteuid"
    ):
        return node
    return None


def _lambda_accepts(lam: ast.Lambda, call: ast.Call) -> bool:
    """Whether this lambda could take this call's arguments. True when the source does
    not say (a `*args` at the call site, say), since a false alarm is worse than a miss
    in a scan everybody has to keep green."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return True
    given = len(call.args)
    by_name = {keyword.arg for keyword in call.keywords}
    positional = [*lam.args.posonlyargs, *lam.args.args]
    required = positional[: len(positional) - len(lam.args.defaults)]
    if given > len(positional) and lam.args.vararg is None:
        return False
    if any(arg.arg not in by_name for arg in required[given:]):
        return False
    missing_keywords = [
        arg.arg
        for arg, default in zip(lam.args.kwonlyargs, lam.args.kw_defaults)
        if default is None and arg.arg not in by_name
    ]
    # **kwargs cannot satisfy a required keyword-only parameter.
    return not missing_keywords


def _fallback_takes(fallback: ast.AST, call: ast.Call) -> bool:
    """Whether getattr's third argument survives being called this way on Windows. A
    literal never does. A lambda is checked against the call. Anything else is taken at
    its word, since whether a name is callable is not decidable from this file."""
    if isinstance(fallback, ast.Lambda):
        return _lambda_accepts(fallback, call)
    return not isinstance(
        fallback,
        (
            ast.Constant,
            ast.List,
            ast.Dict,
            ast.Set,
            ast.Tuple,
            ast.JoinedStr,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            # -1 is a UnaryOp around a Constant, not a Constant
            ast.UnaryOp,
        ),
    ) and not _is_literal_arithmetic(fallback)


def _is_literal_arithmetic(node: ast.AST) -> bool:
    """`1 + 2` is a BinOp, and still a number nobody can call. Only when every leaf is a
    literal: `a + b` could be anything."""
    if not isinstance(node, ast.BinOp):
        return False
    return all(
        isinstance(leaf, (ast.BinOp, ast.UnaryOp, ast.Constant, ast.operator, ast.unaryop))
        for leaf in ast.walk(node)
    )


def _geteuid_sites(expr: ast.AST):
    """Every os.geteuid lookup performed when THIS expression is evaluated. A lambda's
    body is not: it runs when the lambda is called, so `helper = lambda: os.geteuid()`
    is as safe as the same line inside a def. Its defaults are evaluated here and stay."""
    # An immediately called lambda is not deferred; a star-unpacked generator is consumed here.
    unpacked = {
        node.value
        for node in ast.walk(expr)
        if isinstance(node, ast.Starred) and isinstance(node.value, ast.GeneratorExp)
    }
    # A comprehension runs eagerly, so a generator it iterates is advanced.
    unpacked.update(
        generator.iter
        for node in ast.walk(expr)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp))
        for generator in node.generators
        if isinstance(generator.iter, ast.GeneratorExp)
    )
    unpacked.update(
        node.value
        for node in ast.walk(expr)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.GeneratorExp)
        and any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets)
    )
    invoked = set()
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Lambda):
            invoked.add(node.func)
            continue
        inner = _getattr_geteuid(node.func)
        if inner is not None and len(inner.args) == 3 and isinstance(inner.args[2], ast.Lambda):
            invoked.add(inner.args[2])
    stack = [expr]
    while stack:
        node = stack.pop()
        if _is_os_geteuid(node):
            yield node
        elif isinstance(node, ast.AugAssign) and _is_augmented_geteuid(node.target):
            yield node.target
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and node.level == 0  # `from .os import geteuid` is a local module, not stdlib
            and any(alias.name == "geteuid" for alias in node.names)
        ):
            # `from os import geteuid` raises ImportError on Windows at the import itself
            yield node
        elif isinstance(node, ast.Call):
            inner = _getattr_geteuid(node.func)
            # A three-arg getattr only matters when its result is called; binding None is fine.
            if (
                inner is not None
                and len(inner.args) == 3
                and not _fallback_takes(inner.args[2], node)
            ):
                yield inner
        if isinstance(node, ast.GeneratorExp) and node not in unpacked:
            # A generator expression evaluates only its first iterable until iterated.
            stack.append(node.generators[0].iter)
            continue
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.Lambda) and child is node.body and node not in invoked:
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
    """Windows truth value of an expression, or None; the polarity of `os.name` comparisons matters."""
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
            equal = right.value == "nt"
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
                    return True
                if isinstance(parent.op, ast.And) and value is False:
                    return True
        elif isinstance(parent, ast.IfExp):
            test = _windows_value(parent.test)
            if child is parent.body and test is False:
                return True
            if child is parent.orelse and test is True:
                return True
        child, parent = parent, parents.get(parent)
    return False


def _is_guarded(expr: ast.AST) -> bool:
    """Only `os.name` comparisons and `hasattr(os, 'geteuid')` guard a call; `is_ci() or ...` does not."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(expr):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return all(_spared_on_windows(node, parents) for node in _geteuid_sites(expr))


def _definition_expressions(node: ast.AST, eager_annotations: bool):
    """The parts of a `def` or `class` evaluated where it is written, not where it runs."""
    yield from node.decorator_list
    if isinstance(node, ast.ClassDef):
        yield from node.bases
        yield from (keyword.value for keyword in node.keywords)
        return
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    args = node.args
    yield from args.defaults
    yield from (default for default in args.kw_defaults if default)
    if eager_annotations:
        # Without future annotations these are evaluated at the def, like a decorator.
        if node.returns is not None:
            yield node.returns
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        yield from (arg.annotation for arg in every if arg is not None and arg.annotation)


# ast.match_case is not an ast.stmt; 3.9 has no Match at all.
MATCH_CASE = getattr(ast, "match_case", ())
TYPE_ALIAS = getattr(ast, "TypeAlias", None)
LAZY_ANNOTATIONS = sys.version_info >= (3, 14)
MATCH_SYNTAX = sys.version_info >= (3, 10)


def _normalise_os_aliases(tree: ast.Module) -> ast.Module:
    """Rewrite `import os as _os` so every later `_os.geteuid()` reads as `os.geteuid()`.
    Cheaper and less error-prone than threading an alias set through every predicate, and
    the alias is only ever rebound by shadowing, which no test module here does."""
    aliases = {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "os" and alias.asname
    }
    if aliases:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in aliases:
                node.id = "os"
    return tree


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _offending_sites(tree: ast.Module):
    """Every unguarded import-time lookup, in source order."""
    for expr in _import_time_expressions(tree):
        if not _is_guarded(expr):
            yield from _geteuid_sites(expr)


def _import_time_expressions(tree: ast.Module):
    """Yields import-time expressions; `try`, loops, `with` and `match` are skipped whole, not
    guessed at."""
    _normalise_os_aliases(tree)
    # PEP 649 makes annotations lazy from 3.14, and requires-python spans both sides.
    eager_annotations = not _has_future_annotations(tree) and not LAZY_ANNOTATIONS

    def block(statements):
        for statement in statements:
            yield from walk(statement)

    def walk(statement):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _definition_expressions(statement, eager_annotations)
            return
        if isinstance(statement, ast.ClassDef):
            yield from _definition_expressions(statement, eager_annotations)
            yield from block(statement.body)
            return
        if isinstance(statement, ast.If):
            yield statement.test
            reached = _windows_value(statement.test)
            if reached is not False:
                yield from block(statement.body)
            if reached is not True:
                yield from block(statement.orelse)
            return
        if TYPE_ALIAS is not None and isinstance(statement, TYPE_ALIAS):
            # `type X = ...` (3.12+) is lazy: the value is not evaluated on import.
            return
        if isinstance(statement, ast.AnnAssign) and not eager_annotations:
            # The value and a non-name target are evaluated even when the annotation is a string.
            if statement.value is not None:
                yield statement.value
            if not isinstance(statement.target, ast.Name):
                yield statement.target
            return
        if any(
            isinstance(child, (ast.stmt, ast.ExceptHandler, MATCH_CASE))
            for child in ast.iter_child_nodes(statement)
        ):
            return
        # Yield the whole statement so a guard in an enclosing `or` is still seen.
        yield statement

    yield from block(tree.body)


def test_no_test_module_calls_os_geteuid_unguarded_at_import():
    offenders = []
    checked = 0
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        checked += 1
        for node in _offending_sites(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
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
    return [node.lineno for node in _offending_sites(ast.parse(source))]


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
    annotation_only = "import os\ndef helper(uid: os.geteuid()):\n    return uid\n"
    assert bool(_flagged(annotation_only)) is not LAZY_ANNOTATIONS


def test_a_nested_definition_is_not_import_time():
    """A `def` inside a function is not evaluated until the outer one runs, so its
    decorator cannot break collection and must not be reported."""
    assert not _flagged(
        "import os, pytest\n"
        "def test_outer():\n"
        '    @pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "    def inner(uid = os.geteuid()): pass\n"
    )
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


def test_a_statement_level_platform_guard_is_honoured():
    """`if os.name == "posix":` decides what runs, so a lookup in its body is not reached
    on Windows. Yielding the whole `if` and walking it would report the body regardless."""
    assert not _flagged('import os\nif os.name == "posix":\n    ROOT = os.geteuid() == 0\n')
    assert not _flagged(
        'import os\nif os.name == "nt":\n    pass\nelse:\n    ROOT = os.geteuid() == 0\n'
    )
    assert _flagged('import os\nif os.name == "nt":\n    ROOT = os.geteuid() == 0\n')
    assert _flagged("import os\nif is_ci():\n    ROOT = os.geteuid() == 0\n")


def test_a_getattr_fallback_has_to_be_callable():
    """getattr(os, "geteuid", None)() picks None on Windows and raises TypeError there,
    which Linux never shows because the real function is picked. A literal fallback is
    no fallback."""
    assert _flagged('import os\nROOT = getattr(os, "geteuid", None)() == 0\n')
    assert _flagged('import os\nROOT = getattr(os, "geteuid", 1)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", lambda: 1)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", _fallback)() == 0\n')


def test_a_direct_import_fails_before_any_decorator_can_skip():
    """`from os import geteuid` raises ImportError on Windows at the import line, so no
    skipif downstream can help. The statement guard still applies to it."""
    assert _flagged("from os import geteuid\n")
    assert _flagged("from os import getpid, geteuid\n")
    assert not _flagged('import os\nif os.name == "posix":\n    from os import geteuid\n')
    assert not _flagged("from os import getpid\n")


def test_a_class_header_is_evaluated_where_the_class_is_written():
    """Bases and keywords run at the `class`, exactly as a decorator does, and the body
    was already traversed while they were not."""
    assert _flagged("import os\nclass C(Base if os.geteuid() else Other):\n    pass\n")
    assert _flagged("import os\nclass D(metaclass = meta(os.geteuid())):\n    pass\n")
    assert not _flagged(
        'import os\nclass E(Base if os.name != "posix" or os.geteuid() else Other):\n    pass\n'
    )


def test_a_lambda_fallback_has_to_take_the_call():
    """`getattr(os, "geteuid", lambda required: 1)()` picks the lambda on Windows and
    then calls it wrongly. Linux never shows it, because the real function is picked."""
    assert _flagged('import os\nROOT = getattr(os, "geteuid", lambda required: 1)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", lambda: 1)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", lambda *a: 1)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", lambda x = 1: x)() == 0\n')


def test_an_immediately_invoked_lambda_runs_its_body_here():
    """A lambda body is deferred only because it is not called yet. `(lambda: ...)()`
    calls it on the spot, so the deferral must not apply to that one."""
    assert _flagged("import os\nROOT = (lambda: os.geteuid())() == 0\n")
    assert not _flagged("import os\nhelper = lambda: os.geteuid()\n")


def test_the_os_module_is_found_under_an_alias():
    """`import os as _os` then `_os.geteuid()` raises the same AttributeError, and the
    guard spelling moves with it."""
    assert _flagged("import os as _os\nROOT = _os.geteuid() == 0\n")
    assert not _flagged('import os as _os\nROOT = _os.name != "posix" or _os.geteuid() == 0\n')
    assert not _flagged("import os\nROOT = shutil.geteuid() == 0\n")


def test_an_uninvoked_lookup_with_a_literal_fallback_is_fine():
    """`GETEUID = getattr(os, "geteuid", None)` binds None on Windows and collection
    succeeds; it is only calling the result that fails. Feature detection is not a bug."""
    assert not _flagged('import os\nGETEUID = getattr(os, "geteuid", None)\n')
    assert _flagged('import os\nROOT = getattr(os, "geteuid", None)() == 0\n')


def test_a_non_callable_container_fallback_is_no_fallback():
    """`getattr(os, "geteuid", [])()` picks the list on Windows and raises TypeError."""
    for fallback in ("[]", "{}", "()", "{1}", '"text"', "0"):
        assert _flagged(f'import os\nROOT = getattr(os, "geteuid", {fallback})() == 0\n'), fallback


def test_an_invoked_fallback_lambda_runs_its_body_on_windows():
    """`getattr(os, "geteuid", lambda: os.geteuid())()` picks the fallback there and runs
    it, so its body is the one place Windows definitely reaches."""
    assert _flagged('import os\nROOT = getattr(os, "geteuid", lambda: os.geteuid())() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", lambda: 1)() == 0\n')


def test_unmodelled_control_flow_is_skipped_whole():
    """Code under `try`, loops, `with` or `match` is skipped whole rather than guessed at."""
    for source in (
        "try:\n    from os import geteuid\nexcept ImportError:\n    geteuid = None\n",
        'import os\nwhile os.name == "posix":\n    ROOT = os.geteuid() == 0\n',
        "import os\nfor _ in range(1):\n    ROOT = os.geteuid() == 0\n",
        "import os\nwith open('x') as fh:\n    ROOT = os.geteuid() == 0\n",
    ):
        assert not _flagged(source), source
    if MATCH_SYNTAX:  # `match` is a SyntaxError below 3.10
        assert not _flagged(
            'import os\nmatch os.name:\n    case "posix":\n        ROOT = os.geteuid() == 0\n'
        )
    assert not _flagged(
        "import os, pytest\n"
        "for _ in range(1):\n"
        '    @pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "    def test_a(): pass\n"
    )


def test_a_generator_expression_does_not_look_anything_up_yet():
    """`(os.geteuid() for _ in xs)` builds a generator and evaluates nothing but the first
    iterable. Whether the call around it consumes the generator is not readable from the
    source, `iter(...)` does not, so the body stays deferred either way."""
    assert not _flagged("import os\nGEN = (os.geteuid() for _ in range(1))\n")
    assert not _flagged("import os\nROOT = any(os.geteuid() == 0 for _ in range(1))\n")
    assert _flagged("import os\nGEN = (x for x in [os.geteuid()])\n")


def test_an_annotation_is_a_string_under_the_future_import():
    """`x: os.geteuid() = 1` looks nothing up when the module carries
    `from __future__ import annotations`, though the value beside it is still evaluated."""
    assert bool(_flagged("import os\nx: os.geteuid() = 1\n")) is not LAZY_ANNOTATIONS
    assert not _flagged("from __future__ import annotations\nimport os\nx: os.geteuid() = 1\n")
    assert _flagged("from __future__ import annotations\nimport os\nx: int = os.geteuid()\n")


def test_a_lambda_fallback_short_of_a_keyword_only_argument():
    """**kwargs does not satisfy a required keyword-only parameter: Python still wants it
    by name, so the fallback fails on Windows whatever it collects."""
    assert _flagged(
        'import os\nROOT = getattr(os, "geteuid", lambda *, required, **kw: 1)() == 0\n'
    )
    assert not _flagged(
        'import os\nROOT = getattr(os, "geteuid", lambda *, required = 1, **kw: 1)() == 0\n'
    )
    assert not _flagged(
        'import os\nROOT = getattr(os, "geteuid", lambda *, required: 1)(required = 1) == 0\n'
    )


def test_a_negative_number_is_still_a_literal():
    """-1 is a UnaryOp around a Constant, so the literal check missed it."""
    assert _flagged('import os\nROOT = getattr(os, "geteuid", -1)() == 0\n')
    assert _flagged('import os\nROOT = getattr(os, "geteuid", not True)() == 0\n')


def test_starred_unpacking_exhausts_the_generator_here():
    """A star says outright that the generator is consumed on the spot, which a call
    around it does not."""
    assert _flagged("import os\nROOT = [*(os.geteuid() for _ in range(1))]\n")
    assert _flagged("import os\nROOT = f(*(os.geteuid() for _ in range(1)))\n")
    assert not _flagged("import os\nGEN = (os.geteuid() for _ in range(1))\n")


def test_a_store_creates_the_attribute_rather_than_reading_it():
    """`os.geteuid = lambda: 0` works on Windows: it creates the attribute that is not
    there. tests/test_allow_cpu_import_driverless.py does exactly this."""
    assert not _flagged("import os\nos.geteuid = lambda: 0\n")
    assert _flagged("import os\nROOT = os.geteuid() == 0\n")


def test_a_lazy_type_alias_evaluates_nothing():
    """`type UID = os.geteuid()` on 3.12+ is lazy: the value waits for UID.__value__,
    which importing the module never reads."""
    if TYPE_ALIAS is None:
        pytest.skip("PEP 695 type aliases need Python 3.12")
    assert not _flagged("import os\ntype UID = os.geteuid()\n")


def test_a_deferred_annotation_does_not_defer_its_target():
    """`slots[os.geteuid()]: int` evaluates the subscript at import whatever PEP 563 or
    649 does with the annotation beside it."""
    assert _flagged(
        "from __future__ import annotations\nimport os\nslots = {}\nslots[os.geteuid()]: int\n"
    )
    assert not _flagged("from __future__ import annotations\nimport os\nx: os.geteuid()\n")


def test_an_augmented_assignment_reads_before_it_writes():
    """`os.geteuid += 1` is a Store in the AST but fetches the attribute first, unlike a
    plain assignment, which creates it."""
    assert _flagged("import os\nos.geteuid += 1\n")
    assert not _flagged("import os\nos.geteuid = lambda: 0\n")


def test_a_destructured_generator_is_advanced_here():
    """`ROOT, = (os.geteuid() for _ in xs)` has to advance the generator to unpack it."""
    assert _flagged("import os\nROOT, = (os.geteuid() for _ in range(1))\n")
    assert not _flagged("import os\nGEN = (os.geteuid() for _ in range(1))\n")


def test_literal_arithmetic_is_still_a_literal():
    """`1 + 2` is a BinOp and still a number nobody can call; `a + b` could be anything."""
    assert _flagged('import os\nROOT = getattr(os, "geteuid", 1 + 2)() == 0\n')
    assert not _flagged('import os\nROOT = getattr(os, "geteuid", a + b)() == 0\n')


def test_a_relative_import_is_not_the_stdlib_os():
    """A package with its own os.py makes `from .os import geteuid` a local import that
    collects fine on Windows."""
    assert not _flagged("from .os import geteuid\n")
    assert not _flagged("from ..os import geteuid\n")
    assert _flagged("from os import geteuid\n")


def test_an_eager_comprehension_advances_the_generator_it_iterates():
    """A list, set or dict comprehension runs on the spot, so a generator it iterates is
    advanced there too."""
    assert _flagged("import os\nROOT = [x for x in (os.geteuid() for _ in range(1))]\n")
    assert _flagged("import os\nROOT = {x for x in (os.geteuid() for _ in range(1))}\n")
    assert not _flagged("import os\nGEN = (x for x in (os.geteuid() for _ in range(1)))\n")


def test_a_later_read_of_a_module_written_attribute_is_still_reported():
    """The scan does not track module reassignment of `os.geteuid`, so such reads are reported."""
    assert _flagged("import os\nos.geteuid = lambda: 0\nROOT = os.geteuid() == 0\n")
    assert not _flagged("import os\nos.geteuid = lambda: 0\n")
