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

Three shapes are consequently out of reach, all fixed by hand and none decidable
from the call. A path a helper hands back rather than takes in, as
`for path in _iter_caller_files()` does in test_security_gate_consistency.py,
says nothing about itself at the read. Text read from a checked-in file and
then written back to a tmp_path, at test_studio_install_workspace_guard.py:851
and test_scan_packages.py:40, is unsafe only because of where the string came
from. And a read inside a `python -c` snippet, as test_studio_import_no_torch.py
and test_e2e_no_torch_sandbox.py build for their subprocess tests, runs in a
child interpreter this scan never parses: the snippet is an f-string whose paths
are replacement fields, so recovering it would mean evaluating the
interpolation. Reviewers have to catch those three; running the suite under
LC_ALL=C is the cheapest way to find them, since ASCII rejects every byte cp1252
does and more.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and pyproject declares requires-python =
# ">=3.9,<3.15".
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Both trees ship to Windows contributors, and separate CI jobs collect them (repo-cpu-tests and the studio-backend
SKIP_DIRS = {".git", ".venv", "build", "dist", "frontend", "node_modules", "site-packages"}


def _walked_test_files(repo: Path):
    """Every *.py under a tests directory, found by walking."""
    found = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if "tests" not in Path(dirpath).relative_to(repo).parts:
            continue
        found.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    return found


def _tracked_test_files(repo: Path):
    """The same, but only what git is actually tracking.

    A walk picks up whatever happens to be lying in the checkout: a scratch
    directory, a nested worktree, a vendored dependency. None of those are ours
    to police, and a single syntax error in one would fail this test for
    everybody who has one. Asking git keeps the promise the docstring makes.
    """
    try:
        listed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--", "*.py"],
            capture_output = True,
            timeout = 60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    names = listed.stdout.decode("utf-8", errors = "replace").split("\0")
    return [
        repo / name
        for name in names
        if name and "tests" in Path(name).parts and not SKIP_DIRS.intersection(Path(name).parts)
    ]


SOURCES = _tracked_test_files(REPO)
if SOURCES is None:
    SOURCES = _walked_test_files(REPO)
GUARDED_METHODS = {"read_text", "write_text"}
# Openers that are somebody else's are recognised by the file's own imports rather than a fixed list, so `import
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
# Wrappers that stay lazy, so draining one drains what it was given.
LAZY_ADAPTERS = {"enumerate", "filter", "islice", "map", "reversed", "zip"}
# Callables that drain a generator argument immediately.
EAGER_CONSUMERS = {
    "all",
    "any",
    "dict",
    "frozenset",
    "list",
    "max",
    "min",
    "next",
    "set",
    "sorted",
    "sum",
    "tuple",
}
# Values that re-select the platform default when passed as the encoding.
PLATFORM_DEFAULT_ENCODINGS = (None, "locale")
# `Path.read_text(p)` is the unbound spelling of `p.read_text()`:
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
# Modules whose `open` IS the builtin:
BUILTIN_OPEN_MODULES = {"builtins", "io"}
# Receivers `self.SOURCE` and `cls.SOURCE` reach a class attribute through.
SELF_NAMES = {"cls", "self"}
# A module-level name is normally an anchor, since a fixture cannot reach one.
TEMP_FACTORIES = {
    "NamedTemporaryFile",
    "TemporaryDirectory",
    "gettempdir",
    "mkdtemp",
    "mkstemp",
}
# Functions that hand back a path still pointing at their first argument.
PATH_FUNCTIONS = {
    "abspath",
    "dirname",
    "expanduser",
    "fspath",
    "join",
    "normpath",
    "realpath",
    "relpath",
    "str",
}
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
    "with_stem",
    "with_suffix",
}
# Where each API takes its encoding positionally, for the bound call.
ENCODING_POSITION = {"read_text": 0, "write_text": 1, "Path.open": 2, "open": 3}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()
# Stand-in for a file whose imports are not to hand, so every helper can be called on its own without pretending it
NO_MODULES: dict = {}


def _static_truth(node: ast.AST):
    """Whether a condition is a literal true or false, else None for "depends"."""
    return bool(node.value) if isinstance(node, ast.Constant) else None


def _live_branches(node: ast.AST):
    """The children of a branch that can actually run, or None if it is not one.

    `if False:` and the right of `False and ...` never execute, so reporting a
    read there is a CI failure with no reachable cause and no correct edit.
    """
    if isinstance(node, ast.If):
        taken = _static_truth(node.test)
        if taken is None:
            return None
        return [node.test, *(node.body if taken else node.orelse)]
    if isinstance(node, ast.IfExp):
        taken = _static_truth(node.test)
        if taken is None:
            return None
        return [node.test, node.body if taken else node.orelse]
    if isinstance(node, ast.BoolOp) and node.values:
        # `and` stops at the first false operand, `or` at the first true one.
        stops = isinstance(node.op, ast.Or)
        live = []
        for value in node.values:
            live.append(value)
            if _static_truth(value) is stops:
                break
        return live if len(live) < len(node.values) else None
    return None


def _callee_name(func: ast.AST):
    """The bare name a callee ends in, whether or not it is qualified."""
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


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
    # Defs reachable from a scope that executes at import: module body, any class body, and (added when the helper is
    # entered) any def nested inside a helper we follow.
    # DATA = _load()` runs _load while the class is constructed.
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
                # The body waits for a call;
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
                stack.extend(node.orelse)
                continue
            live = _live_branches(node)
            if live is not None:
                stack.extend(live)  # the dead arm never runs, so nothing in it does
                continue
            if isinstance(node, ast.Call):
                yield node
                func = node.func
                if isinstance(func, ast.Name) and func.id in helpers and func.id not in entered:
                    helper = helpers[func.id]
                    # `READS = _load(paths)` on a generator function only builds the generator, so its body waits for a
                    if not _is_generator(helper) or id(node) in consumed:
                        entered.add(func.id)
                        body = list(helper.body)
                        _collect(body)
                        frontier.append(body)
            stack.extend(ast.iter_child_nodes(node))


def _eagerly_consumed(tree: ast.Module) -> set:
    """Nodes whose lazy value is drained right where it is written.

    Covers both things that defer: a generator expression, and a call to a
    generator function. Neither runs its body until something pulls from it, so
    an unconsumed one has not happened yet.
    """
    # `texts = (p.read_text() for p in ...)` then `list(texts)` consumes the generator through a name, so the name has
    named: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.GeneratorExp):
                named.setdefault(target.id, node.value)

    def _resolve(node):
        if isinstance(node, ast.Name) and node.id in named:
            return named[node.id]
        return node

    consumed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_eager_consumer(node.func):
            consumed.update(id(_resolve(a)) for a in node.args)
            consumed.update(id(_resolve(k.value)) for k in node.keywords)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            consumed.add(id(_resolve(node.iter)))  # the loop pulls every item `list(enumerate(_paths()))` drains
    by_id = {id(n): n for n in ast.walk(tree)}
    queue = [by_id[i] for i in list(consumed) if i in by_id]
    while queue:
        node = queue.pop()
        if isinstance(node, ast.Call) and _callee_name(node.func) in LAZY_ADAPTERS:
            for arg in node.args:
                target = _resolve(arg)
                if id(target) not in consumed:
                    consumed.add(id(target))
                    queue.append(target)
    return consumed


def _temp_rooted_names(tree: ast.Module) -> set:
    """Module-level names anchored on a directory the run itself created."""
    names = set()
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if value is None:
            continue
        if any(
            isinstance(n, ast.Call) and _callee_name(n.func) in TEMP_FACTORIES
            for n in ast.walk(value)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _non_path_names(tree: ast.Module) -> set:
    """Module-level names bound to a call that plainly does not make a path.

    `response = requests.get(...)` then `response.read_text()` at import is not
    pathlib I/O, and demanding an encoding there leaves no compliant edit.
    """
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if _is_path_preserving(func) or _callee_name(func) in PATH_METHODS:
            continue
        names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


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

    def _bound(target):
        # `SOURCE, CONFIG = Path(...), Path(...)` binds both.
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from _bound(element)
        elif isinstance(target, ast.Starred):
            yield from _bound(target.value)

    def _is_temp(value) -> bool:
        return value is not None and any(
            isinstance(n, ast.Call) and _callee_name(n.func) in TEMP_FACTORIES
            for n in ast.walk(value)
        )

    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if _is_temp(node.value):
                continue
            for target in node.targets:
                names.update(_bound(target))
        elif isinstance(node, ast.AnnAssign):
            if _is_temp(node.value):
                continue  # TMP = Path(tempfile.mkdtemp()) is not checked in
            names.update(_bound(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
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
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # A comprehension target binds in its own scope, so it shadows nothing out here;
            for gen in node.generators:
                stack.append(gen.iter)
                stack.extend(gen.ifs)
            stack.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # `start._CODEX_FALLBACK_PROMPT` is a path another module defines at its own module scope, so the import is
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _imported_names(node) -> dict:
    """Names this scope's own imports bind, mapped to where they came from.

    The name alone is not enough in either direction. `import gzip as gz` binds
    a name nobody would recognise to an opener that does take an encoding, and
    `from PIL.Image import open` binds a name everybody recognises to one that
    does not. Keeping the origin settles both.

    Nested function bodies are left out: an import inside one is that
    function's business, and treating it as the module's would let a single
    local `from PIL.Image import open` turn off the builtin check everywhere.
    """
    bound = {}
    stack = list(ast.iter_child_nodes(node))
    while stack:
        item = stack.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound.update(_import_bindings(item))
        else:
            stack.extend(ast.iter_child_nodes(item))
    return bound


def _import_bindings(node) -> dict:
    """What one import statement binds, mapped to where each name came from."""
    if isinstance(node, ast.Import):
        return {(a.asname or a.name).split(".")[0]: a.name for a in node.names}
    return {
        a.asname or a.name: (f"{node.module}.{a.name}" if node.module else a.name)
        for a in node.names
    }


def _imports_at_each_call(tree: ast.Module) -> dict:
    """The imports visible at every call, keyed by node id.

    A function's own imports are added on the way in and go out of view again
    on the way out, which is what keeps a local alias local. Within a scope they
    accumulate in statement order, so `DATA = open(p)` above a later
    `from gzip import open` still resolves to the builtin it actually called.
    """
    visible_at = {}

    def walk(node, visible):
        if isinstance(node, ast.Call):
            visible_at[id(node)] = dict(visible)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            visible.update(_import_bindings(node))
            return
        if isinstance(node, ast.If):
            # Only a branch that certainly runs may bind a name for the code after it;
            taken = _static_truth(node.test)
            walk(node.test, visible)
            for arm, runs in ((node.body, taken is not False), (node.orelse, taken is not True)):
                if not runs:
                    continue
                inner = visible if taken is not None else dict(visible)
                for child in arm:
                    walk(child, inner)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, dict(visible))  # its own scope, so its own copy
            else:
                walk(child, visible)

    walk(tree, {})
    return visible_at


def _open_alias(name, modules):
    """What a bare callable resolves to: "builtin", a COMPRESSED_OPENERS key, or None.

    `from io import open as io_open` is the builtin under another name and
    `from gzip import open as gzopen` is gzip's, while `from PIL.Image import
    open` is neither and takes no encoding at all.
    """
    origin = modules.get(name)
    if origin is None:
        return "builtin" if name == "open" else None
    parts = origin.split(".")
    if parts[-1] != "open":
        return None
    if parts[0] in BUILTIN_OPEN_MODULES or origin == "open":
        return "builtin"
    return parts[0] if parts[0] in COMPRESSED_OPENERS else None


def _origin_root(name, modules) -> str:
    """The top-level module a bound name came from, or the name itself."""
    return modules.get(name, name).split(".")[0]


def _compressed_key(name, modules):
    """The COMPRESSED_OPENERS entry this receiver resolves to, if any."""
    for candidate in (name, _origin_root(name, modules)):
        if candidate in COMPRESSED_OPENERS:
            return candidate
    return None


def _is_path_class(name, modules) -> bool:
    """True for a pathlib class, including under an alias.

    `from pathlib import Path as P` still puts the instance in slot 0 of an
    unbound `P.read_text(SOURCE)`, so matching the bare name is not enough.
    """
    if name is None:
        return False
    return (modules.get(name) or name).split(".")[-1] in PATH_CLASSES


def _is_path_attr(node: ast.AST) -> bool:
    """True for a qualified path class, as in `pathlib.Path` or `pl.Path`."""
    return isinstance(node, ast.Attribute) and node.attr in PATH_CLASSES


def _is_path_preserving(func) -> bool:
    """True for a call whose result still points at its first argument.

    Qualified spellings count: `pathlib.Path(p)` and `os.path.join(p, x)` are
    the same constructors as the bare names.
    """
    name = _callee_name(func)
    return name in PATH_CLASSES or name in PATH_FUNCTIONS


def _is_module_receiver(name, modules) -> bool:
    """True for a receiver that is not itself a path."""
    return (
        name in modules
        or _is_path_class(name, modules)
        or _compressed_key(name, modules) is not None
        or _origin_root(name, modules) in BUILTIN_OPEN_MODULES
    )


def _path_expr(call: ast.Call, modules = NO_MODULES):
    """The expression naming the file the call reads.

    Usually the receiver, but a module or the Path class in that slot means the
    path is the first argument instead: `Path.read_text(REPO / "x.py")` and
    `gzip.open(path, "rt")` both read their argument, not `Path` or `gzip`.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        if _is_path_attr(func.value) or (
            isinstance(func.value, ast.Name) and _is_module_receiver(func.value.id, modules)
        ):
            return call.args[0] if call.args else _path_keyword(call)
        return func.value
    if isinstance(func, ast.Name) and _open_alias(func.id, modules) is not None:
        return call.args[0] if call.args else _path_keyword(call)
    return None


def _path_keyword(call: ast.Call):
    """The path passed by keyword: `file` for open, `filename` for gzip and kin."""
    for kw in call.keywords:
        if kw.arg in ("file", "filename"):
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
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in SELF_NAMES
            ):
                # An unrecognised call says nothing about where its result points, so tempfile.mkdtemp() and a helper
                return node
            node = node.value
        elif isinstance(node, ast.Call):
            func = node.func
            # `p.rglob("*.py")` anchors on p, not on the pattern, while Path(x), str(x) and os.path.join(x, ...) anchor
            if isinstance(func, ast.Attribute) and func.attr in PATH_METHODS:
                node = func.value
            elif _is_path_preserving(func) and node.args:
                node = node.args[0]
            else:
                return node  # self.SOURCE names the class attribute, not self
        else:
            return node


def _is_checked_in_root(
    node: ast.AST,
    module_names: set,
    shadowed,
    derived = (),
    attrs = (),
) -> bool:
    """True when a path expression anchors on something that ships in the repo."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        # `for path in (MODEL_SELECTOR, APP_SIDEBAR)` is checked in when every element is, which is what makes the loop
        return bool(node.elts) and all(
            _is_checked_in_root(
                e.value if isinstance(e, ast.Starred) else e,
                module_names,
                shadowed,
                derived,
                attrs,
            )
            for e in node.elts
        )
    root = _path_root(node)
    if isinstance(root, ast.Constant) and isinstance(root.value, str):
        # A relative literal naming something that exists here is checked in;
        value = root.value
        if not value or "\n" in value or "\0" in value or os.path.isabs(value):
            return False
        try:
            return (REPO / value).exists()
        except OSError:
            return False
    if isinstance(root, ast.Attribute):
        # `self.SOURCE`, where the class body bound SOURCE to a checked-in path.
        return root.attr in attrs
    if not isinstance(root, ast.Name):
        return False
    if root.id in derived:
        return True
    return root.id == "__file__" or (root.id in module_names and root.id not in shadowed)


def _class_path_attrs(tree: ast.Module, module_names: set) -> set:
    """Class-body names bound to a checked-in path, read back as `self.NAME`.

    `class T: _SETUP_SH = ROOT / "setup.sh"` then `self._SETUP_SH.read_text()`
    is as statically provable as the module-level spelling, and the repository
    reads seven real source files exactly that way.
    """
    attrs, mixed = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
            else:
                continue
            bound = {t.id for t in targets if isinstance(t, ast.Name)}
            # One attribute name, two classes, two meanings:
            found = attrs if _is_checked_in_root(stmt.value, module_names, ()) else mixed
            found.update(bound)
    return attrs - mixed


def _reads_itself(name: str, value: ast.AST) -> bool:
    """`source = source.read_text()` reads the path before replacing it.

    The name holds a checked-in path right up to that call, so the assignment
    is not evidence against it; it is the very read we are looking for.
    """
    if not isinstance(value, ast.Call):
        return False
    expr = _path_expr(value)
    return isinstance(expr, ast.Name) and expr.id == name


def _unpack(target, value, paired: bool):
    """Yield (name node, the value it is bound to) for one binding.

    A destructured target contributes every name inside it. Where the two sides
    line up, as in `A, B = P1, P2`, each name takes its own element; where they
    do not, as in `for name, path in CASES`, they all take the iterable, which
    is the thing whose provenance is known.
    """
    if isinstance(target, ast.Name):
        yield target, value
        return
    if not isinstance(target, (ast.Tuple, ast.List)):
        return
    elements = None
    if paired and isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
        elements = value.elts
    for index, element in enumerate(target.elts):
        if isinstance(element, ast.Starred):
            element = element.value
        yield from _unpack(element, elements[index] if elements else value, paired)


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
        paired = False
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            paired = True  # `A, B = P1, P2` lines its sides up element by element
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            # `for p in SRC_DIR.rglob("*.py")` binds p to a checked-in path too, and `for name, path in CASES` binds
            target, value = node.target, node.iter
        else:
            continue
        for name_node, bound in _unpack(target, value, paired):
            targets.add(id(name_node))
            if not _reads_itself(name_node.id, bound):
                assignments.append((name_node.id, bound))
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
        # A name assigned a checked-in path somewhere and something else elsewhere stays out, since only one of the two
        grown -= {
            name
            for name, value in assignments
            if name in grown and not _is_checked_in_root(value, module_names, shadowed, good)
        }
        if grown == good:
            return good
        good = grown


def _unwrap_param(node: ast.AST) -> ast.AST:
    """`pytest.param(SOURCE, id = "x")` is a wrapper around the real value."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "param"
        and node.args
    ):
        return node.args[0]
    return node


def _parametrized_values(func) -> dict:
    """Parameter values supplied by @pytest.mark.parametrize.

    pytest calls a parametrized test itself, so the decorator is the only call
    site there is; without reading it every such parameter looks unprovable.
    """
    supplied: dict = {}
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
            continue
        if not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "parametrize":
            continue
        names, values = decorator.args[0], decorator.args[1]
        if not isinstance(names, ast.Constant) or not isinstance(names.value, str):
            continue
        if not isinstance(values, (ast.List, ast.Tuple, ast.Set)):
            continue
        argnames = [n.strip() for n in names.value.split(",") if n.strip()]
        for element in values.elts:
            paired = len(argnames) > 1 and isinstance(element, (ast.Tuple, ast.List))
            row = element.elts if paired else [element]
            for argname, value in zip(argnames, row):
                supplied.setdefault(argname, []).append(_unwrap_param(value))
    return supplied


def _checked_in_params(tree: ast.Module, module_names: set) -> set:
    """(function, parameter) pairs that only ever receive a checked-in path.

    `_source(LOADER_PATH)` is what tells us that the `path` parameter of
    `_source` is reading a file that ships in the repo; the bare
    `path.read_text()` inside it cannot say so on its own. One hop only, and a
    parameter any call leaves out, or passes anything else, is not tracked.

    Definitions are held by identity, not by name. Two tests that each nest a
    `_read` helper are two different functions, and merging them would let the
    one handed a tmp_path rule out what the other proves.
    """
    # Every definition, plus which scope it was written in, so a call resolves to the nearest enclosing `def` of that
    scope_of: dict = {}
    defs_in: dict = {}

    def _index(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs_in.setdefault(id(scope), {}).setdefault(child.name, child)
                scope_of[id(child)] = scope
                _index(child, child)
            elif isinstance(child, ast.ClassDef):
                _index(child, scope)
            else:
                _index(child, scope)  # a class body is not a name lookup scope

    _index(tree, tree)

    def _lookup(name, scope):
        while scope is not None:
            found = defs_in.get(id(scope), {}).get(name)
            if found is not None:
                return found
            scope = scope_of.get(id(scope))
        return None

    # Which function each call sits in, so a parameter already known to hold a checked-in path can be passed on to the
    owner: dict = {}

    def _mark(node, owning):
        if isinstance(node, ast.Call):
            owner[id(node)] = owning
        for child in ast.iter_child_nodes(node):
            nested = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            _mark(child, child if nested else owning)

    _mark(tree, None)
    # Which class body each call sits in, so `self._read(...)` resolves to that class's method and not a same-named one
    in_class: dict = {}

    def _mark_class(node, cls):
        if isinstance(node, ast.Call):
            in_class[id(node)] = cls
        for child in ast.iter_child_nodes(node):
            _mark_class(child, child if isinstance(child, ast.ClassDef) else cls)

    _mark_class(tree, None)

    def _method(cls, name):
        if cls is None:
            return None
        for stmt in cls.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                return stmt
        return None

    good: set = set()
    while True:
        grown, bad = set(), set()
        for fnode in [d for scope in defs_in.values() for d in scope.values()]:
            for argname, values in _parametrized_values(fnode).items():
                ok = all(_is_checked_in_root(v, module_names, ()) for v in values)
                (grown if ok else bad).add((id(fnode), argname))
        # What the calling function itself can prove, recomputed each pass so a parameter resolved last time can feed a
        scope: dict = {}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            caller = owner.get(id(call))
            callee, bound = call.func, False
            if isinstance(callee, ast.Name):
                func = _lookup(callee.id, caller if caller is not None else tree)
            elif (
                isinstance(callee, ast.Attribute)
                and isinstance(callee.value, ast.Name)
                and callee.value.id in SELF_NAMES
            ):
                # `self._read(ROOT / "x.py")` seeds `_read`'s path parameter too.
                func, bound = _method(in_class.get(id(call)), callee.attr), True
            else:
                continue
            if func is None or any(isinstance(a, ast.Starred) for a in call.args):
                continue
            if caller is None:
                here = set()
            elif id(caller) in scope:
                here = scope[id(caller)]
            else:
                params = {p for f, p in good if f == id(caller)}
                here = _checked_in_locals(caller, module_names, _local_names(caller), params)
                scope[id(caller)] = here
            positional = [a.arg for a in [*func.args.posonlyargs, *func.args.args]]
            if bound:
                positional = positional[1:]  # the receiver already fills `self` A keyword-only parameter never takes a
            # A keyword-only parameter never takes a positional slot, so it is
            params = positional + [a.arg for a in func.args.kwonlyargs]
            supplied = dict(zip(positional, call.args))
            supplied.update({k.arg: k.value for k in call.keywords if k.arg in params})
            for param in params:
                value = supplied.get(param)
                ok = value is not None and _is_checked_in_root(value, module_names, (), here)
                (grown if ok else bad).add((id(func), param))
        grown -= bad
        if grown == good:
            return good
        good = grown


def _checked_in_path_calls(
    tree: ast.Module,
    modules = NO_MODULES,
    visible_at = None,
):
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
    visible_at = _imports_at_each_call(tree) if visible_at is None else visible_at
    params = _checked_in_params(tree, module_names)
    attrs = _class_path_attrs(tree, module_names)

    def visit(
        node,
        shadowed,
        derived = frozenset(),
    ):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            shadowed = shadowed | _local_names(node)
            # Seed with the parameters first:
            seeded = {p for f, p in params if f == id(node)}
            derived = _checked_in_locals(node, module_names, shadowed, seeded)
        elif _is_main_guard(node):
            # Never runs under pytest, so rule 1 skips it for the same reason.
            for child in node.orelse:
                yield from visit(child, shadowed, derived)
            return
        elif isinstance(node, ast.GeneratorExp) and id(node) not in consumed:
            if node.generators:
                yield from visit(node.generators[0].iter, shadowed, derived)
            return
        elif (live := _live_branches(node)) is not None:
            for child in live:
                yield from visit(child, shadowed, derived)
            return
        elif isinstance(node, ast.Call):
            expr = _path_expr(node, visible_at.get(id(node), modules))
            if expr is not None and _is_checked_in_root(
                expr, module_names, shadowed, derived, attrs
            ):
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


def _offender(call: ast.Call, modules = NO_MODULES) -> str | None:
    """The call's name if it reads text without an encoding, else None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        # An unbound `Path.read_text(p)` puts the instance in slot 0, and `pathlib.Path.read_text(p)` is the same call
        shift = 1 if _is_path_class(receiver, modules) or _is_path_attr(func.value) else 0
        if func.attr in GUARDED_METHODS:
            if func.attr == "read_text" and not shift and call.args:
                first = call.args[0]
                # Bound read_text takes encoding first, so None or "locale" there is a platform-default read.
                # Any other positional means the receiver is importlib.metadata's Distribution, whose argument is a
                # filename and which takes no encoding at all.
                if isinstance(first, ast.Constant) and first.value in PLATFORM_DEFAULT_ENCODINGS:
                    return "read_text()"
                return None
            position = ENCODING_POSITION[func.attr] + shift
            return None if _pins_encoding(call, position) else f"{func.attr}()"
        if func.attr == "open":
            # io.open and builtins.open ARE the builtin, so they take the builtin's argument positions and the same
            if receiver is not None and _origin_root(receiver, modules) in BUILTIN_OPEN_MODULES:
                if not _is_text(call, 1) or _pins_encoding(call, ENCODING_POSITION["open"]):
                    return None
                return f"{receiver}.open()"
            compressed = _compressed_key(receiver, modules) if receiver else None
            if compressed is not None:
                mode = _open_mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None  # "rb" default, so binary unless asked otherwise
                return (
                    None
                    if _pins_encoding(call, COMPRESSED_OPENERS[compressed])
                    else f"{compressed}.open()"
                )
            # Any other module receiver is somebody else's opener: tarfile.open takes a compression mode, Image.open
            # takes a binary file.
            if (
                receiver is not None
                and receiver in modules
                and not _is_path_class(receiver, modules)
            ):
                return None
            if not _is_text(call, shift):
                return None
            return (
                None
                if _pins_encoding(call, ENCODING_POSITION["Path.open"] + shift)
                else "Path.open()"
            )
        return None
    if isinstance(func, ast.Name):
        alias = _open_alias(func.id, modules)
        # Binary handles have no encoding to name.
        if alias == "builtin" and _is_text(call, 1):
            return None if _pins_encoding(call, ENCODING_POSITION["open"]) else "open()"
        if alias is not None and alias != "builtin":
            mode = _open_mode(call, 1)
            if mode is UNKNOWN_MODE or "t" not in str(mode):
                return None  # "rb" default, so binary unless asked otherwise
            position = COMPRESSED_OPENERS[alias]
            return None if _pins_encoding(call, position) else f"{alias}.open()"
    return None


def _scan(tree: ast.Module, rel: str):
    """Offenders from both rules, reported once each and in source order."""
    modules = _imported_names(tree)
    visible_at = _imports_at_each_call(tree)
    calls = {id(c): c for c in _import_time_calls(tree)}
    calls.update({id(c): c for c in _checked_in_path_calls(tree, modules, visible_at)})
    not_paths = _non_path_names(tree)
    temp_roots = _temp_rooted_names(tree)
    for call in sorted(calls.values(), key = lambda c: (c.lineno, c.col_offset)):
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and (func.attr in GUARDED_METHODS or func.attr == "open")
            and isinstance(func.value, ast.Name)
            and func.value.id in not_paths
        ):
            continue  # the run made this file, so the platform default is safe
        expr = _path_expr(call, visible_at.get(id(call), modules))
        root = _path_root(expr) if expr is not None else None
        if isinstance(root, ast.Name) and root.id in temp_roots:
            continue  # ZipFile.open and friends have no encoding to name
        name = _offender(call, visible_at.get(id(call), modules))
        if name is not None:
            yield f"{rel}:{call.lineno}: {name}"


def test_checked_in_file_reads_name_an_encoding():
    offenders = []
    for path in sorted(SOURCES):
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        offenders.extend(_scan(tree, path.relative_to(REPO).as_posix()))
    assert offenders == [], (
        f"{len(offenders)} file reads in the test trees touch a checked-in file "
        "with the platform default encoding, so they break on Windows as soon "
        'as that file gains a non-ASCII byte. Pass encoding = "utf-8": '
        f"{offenders[:10]}"
    )
