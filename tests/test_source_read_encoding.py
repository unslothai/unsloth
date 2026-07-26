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

A call is an offence when it does un-pinned text I/O and either of two things
holds. It runs at import, where nothing can see a tmp_path fixture yet. Or the
path it reads anchors on something checked in: a module-level constant or
import, which a fixture parameter can never be, `__file__`, or a relative
literal that names a file actually present in the tree. Anchoring is what
decides the second one, followed through `/` joins, path methods, the locals
and loop variables of the enclosing function, and the parameters of helpers
every caller hands a checked-in path. So `for p in (_B / "routes").rglob("*.py")`
is in scope, `_source(LOADER_PATH)` puts the bare read inside `_source` in
scope, and anything growing out of a tmp_path stays out. That reaches test
bodies, where the same failure lands one step later:

    test_gemma4_chat_template.py opens unsloth/chat_templates.py through a
    helper its tests call, and cp1252 cannot decode that file ("byte 0x90").
    test_consent_gate.py reads routes/inference.py as `(_BACKEND / rel)` and
    test_gguf_load_cache_reuse.py as `Path(__file__).parent.parent / ...`, both
    dying on the same 0x81 the two cancel modules hit at collection.

Every question the rules ask is answered by the call, its path expression, or
the call sites of the helper it sits in, which keeps them mechanical enough to
enforce with no allowlist and quiet about temp-dir I/O, where the platform
default is harmless and the test wrote the bytes itself.

Two shapes are consequently out of reach, both fixed by hand and neither
decidable without tracking values through returns. A path a helper hands back
rather than takes in, as `for path in _iter_caller_files()` does in
test_security_gate_consistency.py, says nothing about itself at the read. And
text read from a checked-in file then written back to a tmp_path, as at
test_studio_install_workspace_guard.py:851 and test_scan_packages.py:40, is
unsafe only because of where the string came from. Reviewers have to catch
those two.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
import os
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Both trees ship to Windows contributors, and separate CI jobs collect them
# (repo-cpu-tests and the studio-backend matrix), so the rule covers both.
# Not a hand-written list: studio/backend/hub/tests and unsloth/kernels/moe/tests
# are already here, and the next one has to be covered the day it lands.
SKIP_DIRS = {".git", ".venv", "build", "dist", "frontend", "node_modules", "site-packages"}


def _test_roots(repo: Path) -> tuple:
    """Every checked-in pytest tree, discovered rather than enumerated."""
    roots = []
    for dirpath, dirnames, _ in os.walk(repo):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if os.path.basename(dirpath) == "tests":
            roots.append(Path(dirpath))
            dirnames[:] = []  # rglob below already covers everything under it
    return tuple(roots)


ROOTS = _test_roots(REPO)
GUARDED_METHODS = {"read_text", "write_text"}
NOT_PATH_RECEIVERS = {
    "codecs",
    "dbm",
    "fitz",
    "os",
    "pymupdf",
    "shelve",
    "sqlite3",
    "tarfile",
    "wave",
    "webbrowser",
    "zipfile",
}
# These wrap their stream in a TextIOWrapper for a "t" mode, which takes the
# platform default exactly like builtin open. Unlike open they default to "rb",
# so only an explicit text mode is in scope. lzma takes encoding keyword-only.
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
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
# Path methods that hand back another path, so the receiver is still the anchor.
PATH_METHODS = {
    "absolute",
    "as_posix",
    "expanduser",
    "glob",
    "iterdir",
    "joinpath",
    "resolve",
    "rglob",
    "with_name",
    "with_suffix",
}
# Receivers that are not themselves a path, so the path is the first argument.
MODULE_RECEIVERS = set(NOT_PATH_RECEIVERS) | set(COMPRESSED_OPENERS) | PATH_CLASSES | {"io"}
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
    consumed = _eagerly_consumed(tree)
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
                    helper = helpers[func.id]
                    # `READS = _load(paths)` on a generator function only builds
                    # the generator, so its body waits for a consumer just as a
                    # genexp does.
                    if not _is_generator(helper) or id(node) in consumed:
                        entered.add(func.id)
                        body = list(helper.body)
                        _collect(body)  # a def nested here is now callable
                        frontier.append(body)
            stack.extend(ast.iter_child_nodes(node))


def _eagerly_consumed(tree: ast.Module) -> set:
    """Nodes whose lazy value is drained right where it is written.

    Covers both things that defer: a generator expression, and a call to a
    generator function. Neither runs its body until something pulls from it, so
    an unconsumed one has not happened yet.
    """
    consumed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_eager_consumer(node.func):
            consumed.update(id(a) for a in node.args)
            consumed.update(id(k.value) for k in node.keywords)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            consumed.add(id(node.iter))  # the loop pulls every item
    return consumed


def _is_generator(func) -> bool:
    """True when calling this only builds a generator, leaving the body unrun.

    Yields inside a nested def belong to that def, so those do not count.
    """
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _module_level_names(tree: ast.Module) -> set:
    """Names assigned at module scope."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # `start._CODEX_FALLBACK_PROMPT` is a path another module defines at
            # its own module scope, so the import is an anchor like any constant.
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
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
    """The expression naming the file the call reads.

    Usually the receiver, but a module or the Path class in that slot means the
    path is the first argument instead: `Path.read_text(REPO / "x.py")` and
    `gzip.open(path, "rt")` both read their argument, not `Path` or `gzip`.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id in MODULE_RECEIVERS:
            return call.args[0] if call.args else None
        return func.value
    if isinstance(func, ast.Name) and func.id == "open":
        if call.args:
            return call.args[0]
        # open(file = CHECKED_IN) is the same call spelled out.
        for kw in call.keywords:
            if kw.arg == "file":
                return kw.value
    return None


def _path_root(node: ast.AST) -> ast.AST:
    """Follow a path expression back to whatever it is anchored on.

    `(_BACKEND / rel).read_text()` anchors on _BACKEND and
    `Path(__file__).parent / "routes"` on __file__, so joining a relative name
    onto a checked-in root stays in scope. Anchoring is what decides it, not the
    names further down: `tmp_path / SUBDIR` anchors on the fixture, so a
    constant used as a leaf cannot drag temp-dir I/O in.
    """
    while True:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node = node.left
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.Call):
            func = node.func
            # `p.rglob("*.py")` anchors on p, not on the pattern, while
            # Path(x), str(x) and os.path.join(x, ...) anchor on the argument.
            if isinstance(func, ast.Attribute) and func.attr in PATH_METHODS:
                node = func.value
            elif node.args:
                node = node.args[0]
            else:
                # An unrecognised no-argument method says nothing about where
                # its result points, so tempfile.mkdtemp() stops here.
                return node
        else:
            return node


def _is_checked_in_root(
    node: ast.AST,
    module_names: set,
    shadowed,
    derived = (),
) -> bool:
    """True when a path expression anchors on something that ships in the repo."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        # `for path in (MODEL_SELECTOR, APP_SIDEBAR)` is checked in when every
        # element is, which is what makes the loop variable one too.
        return bool(node.elts) and all(
            _is_checked_in_root(e, module_names, shadowed, derived) for e in node.elts
        )
    root = _path_root(node)
    if isinstance(root, ast.Constant) and isinstance(root.value, str):
        # A relative literal naming something that exists here is checked in; a
        # path the test creates at runtime is not in the tree to be found.
        value = root.value
        if not value or "\n" in value or "\0" in value or os.path.isabs(value):
            return False
        try:
            return (REPO / value).exists()
        except OSError:
            return False  # too long to be a name, so not one
    if not isinstance(root, ast.Name):
        return False
    if root.id in derived:
        return True
    return root.id == "__file__" or (root.id in module_names and root.id not in shadowed)


def _reads_itself(name: str, value: ast.AST) -> bool:
    """`source = source.read_text()` reads the path before replacing it.

    The name holds a checked-in path right up to that call, so the assignment
    is not evidence against it; it is the very read we are looking for.
    """
    if not isinstance(value, ast.Call):
        return False
    expr = _path_expr(value)
    return isinstance(expr, ast.Name) and expr.id == name


def _checked_in_locals(
    func,
    module_names: set,
    shadowed,
    seed = (),
) -> set:
    """Locals that only ever hold a checked-in path.

    `route = Path(_BACKEND_DIR) / "routes" / "inference.py"` followed by
    `route.read_text()` is the same read one line apart. A name bound any other
    way, or assigned anything else anywhere in the scope, is not tracked, and
    the pass repeats so that a path built up over several locals still counts.
    """
    assignments = []
    targets = set()
    bad = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            # `for p in SRC_DIR.rglob("*.py")` binds p to a checked-in path too.
            target, value = node.target, node.iter
        else:
            continue
        if isinstance(target, ast.Name):
            targets.add(id(target))
            if not _reads_itself(target.id, value):
                assignments.append((target.id, value))
    for node in ast.walk(func):
        # A with-as or an augassign says nothing about the value it binds.
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if id(node) not in targets:
                bad.add(node.id)
    args = func.args
    bad.update(a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs])
    # A parameter every caller hands a checked-in path is the exception.
    bad -= set(seed)
    good: set = set(seed)
    while True:
        grown = set(good) | {
            name
            for name, value in assignments
            if name not in bad and _is_checked_in_root(value, module_names, shadowed, good)
        }
        # A name assigned a checked-in path somewhere and something else
        # elsewhere stays out, since only one of the two is provable.
        grown -= {
            name
            for name, value in assignments
            if name in grown and not _is_checked_in_root(value, module_names, shadowed, good)
        }
        if grown == good:
            return good
        good = grown


def _checked_in_params(tree: ast.Module, module_names: set) -> set:
    """(function, parameter) pairs that only ever receive a checked-in path.

    `_source(LOADER_PATH)` is what tells us that the `path` parameter of
    `_source` is reading a file that ships in the repo; the bare
    `path.read_text()` inside it cannot say so on its own. One hop only, and a
    parameter any call leaves out, or passes anything else, is not tracked.
    """
    funcs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Which function each call sits in, so a parameter already known to hold a
    # checked-in path can be passed on to the next helper.
    owner: dict = {}

    def _mark(node, owning):
        if isinstance(node, ast.Call):
            owner[id(node)] = owning
        for child in ast.iter_child_nodes(node):
            nested = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            _mark(child, child if nested else owning)

    _mark(tree, None)
    good: set = set()
    while True:
        grown, bad = set(), set()
        # What the calling function itself can prove, recomputed each pass so a
        # parameter resolved last time can feed a local this time.
        scope: dict = {}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            func = funcs.get(call.func.id)
            if func is None or any(isinstance(a, ast.Starred) for a in call.args):
                continue
            caller = owner.get(id(call))
            if caller is None:
                here = set()
            elif id(caller) in scope:
                here = scope[id(caller)]
            else:
                params = {p for f, p in good if f == caller.name}
                here = _checked_in_locals(caller, module_names, _local_names(caller), params)
                scope[id(caller)] = here
            params = [a.arg for a in [*func.args.posonlyargs, *func.args.args]]
            supplied = dict(zip(params, call.args))
            supplied.update({k.arg: k.value for k in call.keywords if k.arg in params})
            for param in params:
                value = supplied.get(param)
                ok = value is not None and _is_checked_in_root(value, module_names, (), here)
                (grown if ok else bad).add((func.name, param))
        grown -= bad
        if grown == good:
            return good
        good = grown


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
    consumed = _eagerly_consumed(tree)
    params = _checked_in_params(tree, module_names)

    def visit(
        node,
        shadowed,
        derived = frozenset(),
    ):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            shadowed = shadowed | _local_names(node)
            derived = _checked_in_locals(node, module_names, shadowed)
            name = getattr(node, "name", None)
            derived = derived | {p for f, p in params if f == name}
        elif _is_main_guard(node):
            # Never runs under pytest, so rule 1 skips it for the same reason.
            for child in node.orelse:
                yield from visit(child, shadowed, derived)
            return
        elif isinstance(node, ast.GeneratorExp) and id(node) not in consumed:
            if node.generators:
                yield from visit(node.generators[0].iter, shadowed, derived)
            return
        elif isinstance(node, ast.Call):
            expr = _path_expr(node)
            if expr is not None and _is_checked_in_root(expr, module_names, shadowed, derived):
                yield node
        for child in ast.iter_child_nodes(node):
            yield from visit(child, shadowed, derived)

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


def _pins_encoding(call: ast.Call, position: int | None) -> bool:
    """True when the call names an encoding, positionally or by keyword.

    `position` is None where the API takes it keyword-only. A splat makes the
    positions meaningless, so it counts as named rather than demanding an edit
    the contributor cannot make correctly.
    """
    if any(isinstance(a, ast.Starred) for a in call.args):
        return True
    if position is not None and len(call.args) > position:
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
            if receiver in COMPRESSED_OPENERS:
                mode = _open_mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None  # "rb" default, so binary unless asked otherwise
                position = COMPRESSED_OPENERS[receiver]
                return None if _pins_encoding(call, position) else f"{receiver}.open()"
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
