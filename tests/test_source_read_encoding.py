# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: tests that read checked-in files must name their encoding.

`Path.read_text()` and `open()` with no encoding use `locale.getencoding()`:
UTF-8 on the Linux and macOS runners, cp1252 on a stock Windows install. A test
that reads a repo file that way passes in CI and raises UnicodeDecodeError for a
Windows contributor as soon as that file gains a non-ASCII byte, which the
source-scanning tests do constantly:

    studio/backend/routes/inference.py carries the DeepSeek tool-call token
    regexes, so it holds U+FF5C and U+2581. Reading it as cp1252 dies on
    "byte 0x81", taking test_cancel_atomicity.py and test_cancel_id_wiring.py
    out at collection time.

A call is an offence when it does un-pinned text I/O and any of three things
holds. It runs at import, where nothing can see a tmp_path fixture yet. Its
path is a module-level constant, which a fixture parameter cannot be. Or its
path is built from `__file__`, which is checked in by construction. The last
two reach test bodies, where the same failure lands one step later:

    test_gemma4_chat_template.py opens unsloth/chat_templates.py through a
    helper its tests call, and cp1252 cannot decode that file ("byte 0x90").
    test_gguf_load_cache_reuse.py reads routes/inference.py the same way, via
    `Path(__file__).parent.parent`, and dies on the same 0x81 as the two cancel
    modules do at collection.

Each rule is a property of the path expression alone, so together they stay
mechanical enough to enforce with no allowlist while staying quiet about
temp-dir I/O, where the platform default is harmless and the test wrote the
bytes itself.

One shape is deliberately out of reach: text read from a checked-in file and
then written back out to a tmp_path, as at test_studio_install_workspace_guard
.py:851, is only unsafe because of where the string came from, and tracing that
needs dataflow rather than a look at the call.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Both trees ship to Windows contributors, and separate CI jobs collect them
# (repo-cpu-tests and the studio-backend matrix), so the rule covers both.
ROOTS = (TESTS, REPO / "studio" / "backend" / "tests")
GUARDED_METHODS = {"read_text", "write_text"}
NOT_PATH_RECEIVERS = {
    "bz2",
    "codecs",
    "dbm",
    "fitz",
    "gzip",
    "lzma",
    "os",
    "pymupdf",
    "shelve",
    "sqlite3",
    "tarfile",
    "wave",
    "webbrowser",
    "zipfile",
}
# Callables that drain a generator argument immediately.
EAGER_CONSUMERS = {
    "all",
    "any",
    "dict",
    "frozenset",
    "list",
    "max",
    "min",
    "set",
    "sorted",
    "sum",
    "tuple",
}
# Values that re-select the platform default when passed as the encoding.
PLATFORM_DEFAULT_ENCODINGS = (None, "locale")
# `Path.read_text(p)` is the unbound spelling of `p.read_text()`: same API, same
# platform default, but the instance takes the first slot so every argument
# shifts one place right.
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
# Where each API takes its encoding positionally, for the bound call.
ENCODING_POSITION = {"read_text": 0, "write_text": 1, "Path.open": 2, "open": 3}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _is_main_guard(node: ast.AST) -> bool:
    """True for `if __name__ == "__main__":`, whose body never runs at import.

    The operator has to be `==`: `if __name__ != "__main__":` runs its body at
    import, so treating it as script-only would invert the rule.
    """
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if not all(isinstance(op, ast.Eq) for op in node.test.ops):
        return False
    operands = [node.test.left, *node.test.comparators]
    # Either spelling: `__name__ == "__main__"` or `"__main__" == __name__`.
    has_name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
    has_main = any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)
    return has_name and has_main


def _is_eager_consumer(func: ast.expr) -> bool:
    """True for a callee that drains a generator argument on the spot.

    iter/zip/map/filter/enumerate/reversed hand back another lazy object, so a
    genexp passed to those still has not run.
    """
    if isinstance(func, ast.Attribute):
        return func.attr in {"join", "extend", "update", "writelines"}
    return isinstance(func, ast.Name) and func.id in EAGER_CONSUMERS


def _import_time_calls(tree: ast.Module):
    """Yield Call nodes that run at import time.

    That is module scope, class bodies, and the bodies of module-level helpers
    invoked from either. A helper is the same hazard as an inline read:
    `CODE = _extract_mixed_precision_code()` runs its `read_text()` during
    collection, so skipping every def would let the Windows failure back in.

    A def's body waits for a call, but its decorators and argument defaults run
    when the def executes, so those are followed. Lambda bodies are skipped for
    the same reason, as is everything but the outermost iterable of a generator
    expression. List, set and dict comprehensions are walked in full: unlike a
    genexp they run their element, filters and nested iterators immediately.

    A body is only ever entered through an executed statement, never by walking
    into a def, so the "this definitely runs" property that makes the rule
    allowlist-free holds. Not followed: the body of
    `if __name__ == "__main__":`, which pytest never runs (its `else` arm does,
    so that is walked), and non-name calls, which are left unresolved rather
    than guessed at.
    """
    # Defs reachable from a scope that executes at import: module body, any
    # class body, and (added when the helper is entered) any def nested inside
    # a helper we follow. `class F: def _load(): ...; DATA = _load()` runs
    # _load while the class is constructed.
    helpers: dict = {}

    def _collect(body):
        scopes = [body]
        while scopes:
            for node in scopes.pop():
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    helpers.setdefault(node.name, node)
                elif isinstance(node, ast.ClassDef):
                    scopes.append(node.body)

    _collect(tree.body)
    consumed = _consumed_genexps(tree)
    entered = set()
    frontier = [list(tree.body)]
    while frontier:
        stack = frontier.pop()
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # The body waits for a call; these two run right now.
                stack.extend(node.decorator_list)
                stack.extend(d for d in node.args.defaults if d is not None)
                stack.extend(d for d in node.args.kw_defaults if d is not None)
                continue
            if isinstance(node, ast.Lambda):
                stack.extend(d for d in node.args.defaults if d is not None)
                stack.extend(d for d in node.args.kw_defaults if d is not None)
                continue
            if isinstance(node, ast.GeneratorExp) and id(node) not in consumed:
                # Lazy: only the outermost iterable is evaluated where written.
                if node.generators:
                    stack.append(node.generators[0].iter)
                continue
            if _is_main_guard(node):
                stack.extend(node.orelse)  # the else arm runs at import
                continue
            if isinstance(node, ast.Call):
                yield node
                func = node.func
                if isinstance(func, ast.Name) and func.id in helpers and func.id not in entered:
                    entered.add(func.id)
                    body = list(helpers[func.id].body)
                    _collect(body)  # a def nested in this helper is now callable
                    frontier.append(body)
            stack.extend(ast.iter_child_nodes(node))


def _consumed_genexps(tree: ast.Module) -> set:
    """Generator expressions handed straight to something that drains them.

    An unconsumed one has not run where it is written, so neither rule should
    read anything into the calls inside it.
    """
    return {
        id(arg)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and _is_eager_consumer(call.func)
        for arg in [*call.args, *(k.value for k in call.keywords)]
        if isinstance(arg, ast.GeneratorExp)
    }


def _module_level_names(tree: ast.Module) -> set:
    """Names assigned at module scope."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _local_names(func) -> set:
    """Every name the function binds, so a module constant it shadows is skipped.

    Walking nested defs too over-approximates, which only ever drops a call from
    the scan.
    """
    args = func.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _path_expr(call: ast.Call):
    """The expression the call reads from."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.value
    if isinstance(func, ast.Name) and func.id == "open" and call.args:
        return call.args[0]
    return None


def _rooted_at_file(node: ast.AST) -> bool:
    """True for a path built from `__file__`, which is checked in by definition.

    `(Path(__file__).parent.parent / "routes" / "inference.py").read_text()` is
    the same read as through a constant, just spelled inline, and no tmp_path
    can be written that way.
    """
    return any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))


def _checked_in_path_calls(tree: ast.Module):
    """Yield calls, at any depth, whose path is provably a checked-in file.

    The import-time walk alone leaves test bodies unguarded, and a bare read
    there is the same Windows failure one step later: `_extract_template()` in
    test_gemma4_chat_template.py opens unsloth/chat_templates.py, which cp1252
    cannot decode ("byte 0x90"), so the test errors rather than the collection.

    Two spellings qualify. A tmp_path arrives as a fixture parameter and a
    tempfile is built in the body, so neither can be bound at module scope nor
    derived from `__file__`. That keeps temp-dir I/O out of scope without an
    allowlist, since there the platform default is harmless and the test wrote
    the bytes itself.
    """
    module_names = _module_level_names(tree)
    consumed = _consumed_genexps(tree)

    def visit(node, shadowed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            shadowed = shadowed | _local_names(node)
        elif _is_main_guard(node):
            # Never runs under pytest, so rule 1 skips it for the same reason.
            for child in node.orelse:
                yield from visit(child, shadowed)
            return
        elif isinstance(node, ast.GeneratorExp) and id(node) not in consumed:
            if node.generators:
                yield from visit(node.generators[0].iter, shadowed)
            return
        elif isinstance(node, ast.Call):
            expr = _path_expr(node)
            name = expr.id if isinstance(expr, ast.Name) else None
            if expr is not None and (
                _rooted_at_file(expr) or (name in module_names and name not in shadowed)
            ):
                yield node
        for child in ast.iter_child_nodes(node):
            yield from visit(child, shadowed)

    yield from visit(tree, frozenset())


def _open_mode(call: ast.Call, mode_index: int):
    """The literal mode of an open() call, or UNKNOWN_MODE.

    A splat or a non-literal hides the mode. Defaulting those to "r" would
    demand an encoding on a call that may resolve to "rb", where passing one is
    a ValueError, so the contributor would have no compliant edit.
    """
    if any(isinstance(a, ast.Starred) for a in call.args):
        return UNKNOWN_MODE
    if any(kw.arg is None for kw in call.keywords):
        return UNKNOWN_MODE
    if len(call.args) > mode_index:
        node = call.args[mode_index]
        return node.value if isinstance(node, ast.Constant) else UNKNOWN_MODE
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else UNKNOWN_MODE
    return "r"


def _is_text(call: ast.Call, mode_index: int) -> bool:
    mode = _open_mode(call, mode_index)
    return mode is not UNKNOWN_MODE and "b" not in str(mode)


def _names_encoding(call: ast.Call) -> bool:
    """True only for an encoding that actually pins one.

    `encoding = None` and `encoding = "locale"` both re-select the platform
    default, so the keyword being present is not enough. A `**kwargs` may carry
    one we cannot see, so it counts as named rather than risking a false alarm.
    """
    for kw in call.keywords:
        if kw.arg is None:
            return True
        if kw.arg != "encoding":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value in PLATFORM_DEFAULT_ENCODINGS:
            return False
        return True
    return False


def _pins_encoding(call: ast.Call, position: int) -> bool:
    """True when the call names an encoding, positionally or by keyword.

    A splat makes the positions meaningless, so it counts as named rather than
    demanding an edit the contributor cannot make correctly.
    """
    if any(isinstance(a, ast.Starred) for a in call.args):
        return True
    if len(call.args) > position:
        node = call.args[position]
        if isinstance(node, ast.Constant):
            return node.value not in PLATFORM_DEFAULT_ENCODINGS
        return True
    return _names_encoding(call)


def _offender(call: ast.Call) -> str | None:
    """The call's name if it reads text without an encoding, else None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        # An unbound `Path.read_text(p)` puts the instance in slot 0.
        shift = 1 if receiver in PATH_CLASSES else 0
        if func.attr in GUARDED_METHODS:
            if func.attr == "read_text" and not shift and call.args:
                first = call.args[0]
                # Bound read_text takes encoding first, so None or "locale"
                # there is a platform-default read. Any other positional means
                # the receiver is importlib.metadata's Distribution, whose
                # argument is a filename and which takes no encoding at all.
                if isinstance(first, ast.Constant) and first.value in PLATFORM_DEFAULT_ENCODINGS:
                    return "read_text()"
                return None
            position = ENCODING_POSITION[func.attr] + shift
            return None if _pins_encoding(call, position) else f"{func.attr}()"
        if func.attr == "open":
            # io.open IS the builtin, so it takes the builtin's argument
            # positions and carries the same platform default.
            if receiver == "io":
                if not _is_text(call, 1) or _pins_encoding(call, ENCODING_POSITION["open"]):
                    return None
                return "io.open()"
            # Any other `<module>.open(...)` is somebody else's opener:
            # tarfile.open takes a compression mode, fitz.open takes filetype=.
            if receiver in NOT_PATH_RECEIVERS:
                return None
            if not _is_text(call, shift):
                return None
            return (
                None
                if _pins_encoding(call, ENCODING_POSITION["Path.open"] + shift)
                else "Path.open()"
            )
        return None
    # Binary handles have no encoding to name.
    if isinstance(func, ast.Name) and func.id == "open" and _is_text(call, 1):
        return None if _pins_encoding(call, ENCODING_POSITION["open"]) else "open()"
    return None


def _scan(tree: ast.Module, rel: str):
    """Offenders from both rules, reported once each and in source order."""
    calls = {id(c): c for c in _import_time_calls(tree)}
    calls.update({id(c): c for c in _checked_in_path_calls(tree)})
    for call in sorted(calls.values(), key = lambda c: (c.lineno, c.col_offset)):
        name = _offender(call)
        if name is not None:
            yield f"{rel}:{call.lineno}: {name}"


def test_checked_in_file_reads_name_an_encoding():
    offenders = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
            offenders.extend(_scan(tree, path.relative_to(REPO).as_posix()))
    assert offenders == [], (
        f"{len(offenders)} file reads in the test trees touch a checked-in file "
        "with the platform default encoding, so they break on Windows as soon "
        'as that file gains a non-ASCII byte. Pass encoding = "utf-8": '
        f"{offenders[:10]}"
    )
