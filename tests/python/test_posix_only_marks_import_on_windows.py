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

A third form in the tree, `getattr(os, "geteuid", lambda: 1)()`, is a guard in itself, but
only because the fallback is there and can be called: `getattr(os, "geteuid")()` and
`getattr(os, "geteuid", None)()` both fail on Windows and are treated as the plain lookup.
`from os import geteuid` fails earlier still, at the import, and counts too. A module
imported as `import os as _os` is normalised back to `os` before any of this. And nothing else counts as a guard merely for sitting to the
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
    """A lookup of os.geteuid that fails on Windows wherever it appears: the attribute,
    and the two-argument getattr, which has no fallback and raises the same AttributeError.
    The three-argument form depends on how it is used, so it is decided in _geteuid_sites,
    which can see the call around it."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "geteuid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ):
        return True
    call = _getattr_geteuid(node)
    return call is not None and len(call.args) == 2


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
    return not (missing_keywords and lam.args.kwarg is None)


def _fallback_takes(fallback: ast.AST, call: ast.Call) -> bool:
    """Whether getattr's third argument survives being called this way on Windows. A
    literal never does. A lambda is checked against the call. Anything else is taken at
    its word, since whether a name is callable is not decidable from this file."""
    if isinstance(fallback, ast.Lambda):
        return _lambda_accepts(fallback, call)
    # a literal of any shape: a number, a string, None, a list, a dict, an f-string
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
        ),
    )


def _geteuid_sites(expr: ast.AST):
    """Every os.geteuid lookup performed when THIS expression is evaluated. A lambda's
    body is not: it runs when the lambda is called, so `helper = lambda: os.geteuid()`
    is as safe as the same line inside a def. Its defaults are evaluated here and stay."""
    # `(lambda: os.geteuid())()` runs its body right there, so only a lambda that is not
    # the callee of a call in this expression gets the deferral
    invoked = set()
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Lambda):
            invoked.add(node.func)
            continue
        # `getattr(os, "geteuid", lambda: os.geteuid())()` picks the fallback on Windows
        # and runs its body there, which is the only platform that ever sees it
        inner = _getattr_geteuid(node.func)
        if inner is not None and len(inner.args) == 3 and isinstance(inner.args[2], ast.Lambda):
            invoked.add(inner.args[2])
    stack = [expr]
    while stack:
        node = stack.pop()
        if _is_os_geteuid(node):
            yield node
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name == "geteuid" for alias in node.names)
        ):
            # `from os import geteuid` raises ImportError on Windows at the import
            # itself, before any decorator gets a chance to skip anything
            yield node
        elif isinstance(node, ast.Call):
            inner = _getattr_geteuid(node.func)
            # A three-argument lookup is only a problem when its RESULT is called:
            # `GETEUID = getattr(os, "geteuid", None)` just binds None on Windows, which
            # is ordinary feature detection. Called, it has to pick something callable.
            if (
                inner is not None
                and len(inner.args) == 3
                and not _fallback_takes(inner.args[2], node)
            ):
                yield inner
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

    The three-argument `getattr(os, "geteuid", lambda: 1)()` needs no case here: a
    fallback that can be called means no lookup can fail, so it is not a site at all."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(expr):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return all(_spared_on_windows(node, parents) for node in _geteuid_sites(expr))


def _definition_expressions(node: ast.AST, eager_annotations: bool):
    """The parts of a `def` or `class` evaluated where it is written, not where it runs."""
    yield from node.decorator_list
    if isinstance(node, ast.ClassDef):
        # `class C(Base if os.geteuid() else Other, metaclass = M())` is evaluated where
        # the class is written, exactly as a decorator is
        yield from node.bases
        yield from (keyword.value for keyword in node.keywords)
        return
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


TRY_STATEMENTS = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)

# A handler for any of these turns a missing os.geteuid from a collection failure into a
# branch the module handles itself. Listed generously on purpose: a scan everybody has to
# keep green should err towards silence, not towards rejecting portable code.
CAUGHT = frozenset(
    {"ImportError", "AttributeError", "TypeError", "OSError", "Exception", "BaseException"}
)


def _catches_a_missing_geteuid(statement) -> bool:
    for handler in statement.handlers:
        if handler.type is None:  # bare except
            return True
        named = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(isinstance(node, ast.Name) and node.id in CAUGHT for node in named):
            return True
    return False


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


def _import_time_expressions(tree: ast.Module):
    """Yield the expressions evaluated when the module is imported.

    Walked as statements rather than as one flat tree, because control flow decides what
    runs: a lookup under `if os.name == "posix":` is never reached on Windows, and a
    function body is never reached at import however deeply it is nested. Yielding whole
    statements and walking those would report both."""
    _normalise_os_aliases(tree)
    eager_annotations = not _has_future_annotations(tree)

    def block(statements):
        for statement in statements:
            yield from walk(statement)

    def walk(statement):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # the body is runtime, whatever encloses the def
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
        if isinstance(statement, TRY_STATEMENTS):
            # `try: from os import geteuid / except ImportError: geteuid = None` is the
            # portable spelling, and Windows lands in the handler. The handler, the else
            # and the finally all still run, so they are walked.
            if not _catches_a_missing_geteuid(statement):
                yield from block(statement.body)
            yield from block(statement.orelse)
            for handler in statement.handlers:
                yield from walk(handler)
            yield from block(statement.finalbody)
            return
        if isinstance(statement, ast.ExceptHandler):
            if statement.type is not None:
                yield statement.type
            yield from block(statement.body)
            return
        nested = [
            child
            for child in ast.iter_child_nodes(statement)
            if isinstance(child, (ast.stmt, ast.ExceptHandler))
        ]
        if not nested:
            # a plain statement holds only expressions, and the WHOLE statement is
            # yielded rather than its parts: a guard lives in the enclosing `or`, and
            # yielding the bare call too would report every guarded site as unguarded
            yield statement
            return
        # a compound statement whose header runs at import (for/while/with/try/match):
        # the header expressions here, the blocks through walk
        for _, value in ast.iter_fields(statement):
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, (ast.stmt, ast.ExceptHandler)):
                    yield from walk(item)
                elif isinstance(item, ast.AST):
                    yield item

    yield from block(tree.body)


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


def test_a_statement_level_platform_guard_is_honoured():
    """`if os.name == "posix":` decides what runs, so a lookup in its body is not reached
    on Windows. Yielding the whole `if` and walking it would report the body regardless."""
    assert not _flagged('import os\nif os.name == "posix":\n    ROOT = os.geteuid() == 0\n')
    assert not _flagged(
        'import os\nif os.name == "nt":\n    pass\nelse:\n    ROOT = os.geteuid() == 0\n'
    )
    # the wrong polarity is still reached, and so is an undecidable test
    assert _flagged('import os\nif os.name == "nt":\n    ROOT = os.geteuid() == 0\n')
    assert _flagged("import os\nif is_ci():\n    ROOT = os.geteuid() == 0\n")


def test_a_function_body_under_a_compound_statement_is_still_runtime():
    """A def nested in a `for` or a `with` is reached at import, but its BODY is not, so a
    lookup there cannot break collection however deep the statement nesting goes."""
    assert not _flagged(
        "import os\n"
        "with open('x') as fh:\n"
        "    for _ in range(1):\n"
        "        def helper():\n"
        "            return os.geteuid()\n"
    )
    # the def's own decorator under the same nesting IS reached
    assert _flagged(
        "import os, pytest\n"
        "with open('x') as fh:\n"
        "    for _ in range(1):\n"
        '        @pytest.mark.skipif(os.geteuid() == 0, reason = "x")\n'
        "        def test_a(): pass\n"
    )


def test_a_try_that_catches_the_failure_is_the_portable_spelling():
    """`try: from os import geteuid / except ImportError: geteuid = None` is how portable
    code is written, and Windows lands in the handler. Rejecting it would be the scan
    telling people to stop doing the right thing."""
    assert not _flagged(
        "try:\n    from os import geteuid\nexcept ImportError:\n    geteuid = None\n"
    )
    assert not _flagged(
        "import os\ntry:\n    ROOT = os.geteuid() == 0\nexcept AttributeError:\n    ROOT = False\n"
    )
    # a handler that cannot catch it leaves the body reported
    assert _flagged(
        "import os\ntry:\n    ROOT = os.geteuid() == 0\nexcept KeyError:\n    ROOT = False\n"
    )
    # and the handler's own body is still walked, since that is what Windows runs
    assert _flagged(
        "import os\ntry:\n    pass\nexcept ImportError:\n    ROOT = os.geteuid() == 0\n"
    )


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
    # a name that is not the alias is left alone
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
