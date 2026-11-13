# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests that read checked-in files must name their encoding; the locale default is cp1252 on Windows."""

# `str | None` below is evaluated at import on Python 3.9 without this.
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Walk every tree, not a hand list, so new test dirs are covered the day they land.
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
    """Lists tracked *.py files only, so scratch dirs, worktrees and vendored code are not scanned."""
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

# Walks are cached: fixed-point passes re-walk the same unmutated AST ~170M times.
# Keyed on id(), so entries hold the node and _scan clears the cache per file.
_WALKS: dict = {}
_KIDS: dict = {}
_CONSUMED: dict = {}


def _walk(node):
    cached = _WALKS.get(id(node))
    if cached is None:
        cached = _WALKS[id(node)] = (node, tuple(ast.walk(node)))
    return cached[1]


def _kids(node):
    cached = _KIDS.get(id(node))
    if cached is None:
        cached = _KIDS[id(node)] = (node, tuple(ast.iter_child_nodes(node)))
    return cached[1]


GUARDED_METHODS = {"read_text", "write_text"}
# Foreign openers are found via the file's imports. They default to "rb"; lzma takes
# encoding keyword-only.
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
LAZY_ADAPTERS = {"enumerate", "filter", "islice", "map", "reversed", "zip"}
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
PLATFORM_DEFAULT_ENCODINGS = (None, "locale")
# `Path.read_text(p)` is the unbound `p.read_text()`: arguments shift one place right.
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
BUILTIN_OPEN_MODULES = {"builtins", "io"}
SELF_NAMES = {"cls", "self"}
# Module-level names are normally anchors, but names rooted in these are temp I/O.
TEMP_FACTORIES = {
    "NamedTemporaryFile",
    "TemporaryDirectory",
    "gettempdir",
    "mkdtemp",
    "mkstemp",
}
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
# Positional encoding slot for each API's bound call.
ENCODING_POSITION = {"read_text": 0, "write_text": 1, "Path.open": 2, "open": 3}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()
NO_MODULES: dict = {}


def _static_truth(node: ast.AST):
    """Whether a condition is a literal true or false, else None for "depends"."""
    return bool(node.value) if isinstance(node, ast.Constant) else None


def _live_branches(node: ast.AST):
    """Skips branches that can never run, such as if False: or the right side of False and ...."""
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
    """Matches if __name__ == "__main__" only, since a != guard does run at import."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if not all(isinstance(op, ast.Eq) for op in node.test.ops):
        return False
    operands = [node.test.left, *node.test.comparators]
    has_name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
    has_main = any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)
    return has_name and has_main


def _is_eager_consumer(func: ast.expr) -> bool:
    """Lazy builtins like zip or map return another lazy object, so a genexp passed to them has not run."""
    if isinstance(func, ast.Attribute):
        return func.attr in {"join", "extend", "update", "writelines"}
    return isinstance(func, ast.Name) and func.id in EAGER_CONSUMERS


def _import_time_calls(tree: ast.Module):
    """Yields Call nodes that run at import: module and class bodies, and helpers they call."""
    # Defs reachable from import-time scopes: module body, class bodies, and nested helpers.
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
                # The body waits for a call; decorators and defaults run right now.
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
                stack.extend(live)
                continue
            if isinstance(node, ast.Call):
                yield node
                func = node.func
                if isinstance(func, ast.Name) and func.id in helpers and func.id not in entered:
                    helper = helpers[func.id]
                    # Calling a generator function only builds the generator; its body waits for a consumer.
                    if not _is_generator(helper) or id(node) in consumed:
                        entered.add(func.id)
                        body = list(helper.body)
                        _collect(body)
                        frontier.append(body)
            stack.extend(_kids(node))


def _eagerly_consumed(tree: ast.Module) -> set:
    """Both rules ask for this, so compute it once per file. See _walk."""
    cached = _CONSUMED.get(id(tree))
    if cached is None:
        cached = _CONSUMED[id(tree)] = (tree, _eagerly_consumed_uncached(tree))
    return cached[1]


def _eagerly_consumed_uncached(tree: ast.Module) -> set:
    """Generator expressions and generator calls drained where written; an unconsumed one has not run."""
    # A generator consumed through a name must lead back to its definition.
    named: dict = {}
    for node in _walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.GeneratorExp):
                named.setdefault(target.id, node.value)

    def _resolve(node):
        if isinstance(node, ast.Name) and node.id in named:
            return named[node.id]
        return node

    consumed = set()
    for node in _walk(tree):
        if isinstance(node, ast.Call) and _is_eager_consumer(node.func):
            consumed.update(id(_resolve(a)) for a in node.args)
            consumed.update(id(_resolve(k.value)) for k in node.keywords)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            consumed.add(id(_resolve(node.iter)))
    # `list(enumerate(_paths()))` drains _paths() as well, one wrapper down.
    by_id = {id(n): n for n in _walk(tree)}
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
            isinstance(n, ast.Call) and _callee_name(n.func) in TEMP_FACTORIES for n in _walk(value)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _non_path_names(tree: ast.Module) -> set:
    """Module names bound to calls that do not build a path, such as requests.get, are not pathlib I/O."""
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
    """Only yields in this function's own body count; yields inside a nested def belong to that def."""
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(_kids(node))
    return False


def _module_level_names(tree: ast.Module) -> set:
    """Names assigned at module scope."""

    def _bound(target):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from _bound(element)
        elif isinstance(target, ast.Starred):
            yield from _bound(target.value)

    def _is_temp(value) -> bool:
        return value is not None and any(
            isinstance(n, ast.Call) and _callee_name(n.func) in TEMP_FACTORIES for n in _walk(value)
        )

    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if _is_temp(node.value):
                continue  # TMP = Path(tempfile.mkdtemp()) is not checked in
            for target in node.targets:
                names.update(_bound(target))
        elif isinstance(node, ast.AnnAssign):
            if _is_temp(node.value):
                continue
            names.update(_bound(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # An imported module-scope path from another module is an anchor like any constant.
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _local_names(func) -> set:
    """Every name the function binds, so a shadowed module constant is not treated as a path."""
    args = func.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    stack = list(_kids(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # A comprehension target binds in its own scope, so it shadows nothing out here.
            for gen in node.generators:
                stack.append(gen.iter)
                stack.extend(gen.ifs)
            stack.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        stack.extend(_kids(node))
    return names


def _imported_names(node) -> dict:
    """Maps import-bound names to their origin; a nested function's imports stay with that function."""
    bound = {}
    stack = list(_kids(node))
    while stack:
        item = stack.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound.update(_import_bindings(item))
        else:
            stack.extend(_kids(item))
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
    """A function's imports leave view on exit, and within a scope they apply in statement order."""
    visible_at = {}

    def walk(node, visible):
        if isinstance(node, ast.Call):
            visible_at[id(node)] = dict(visible)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            visible.update(_import_bindings(node))
            return
        if isinstance(node, ast.If):
            # Only a branch that certainly runs may bind a name for later code; others use a copy.
            taken = _static_truth(node.test)
            walk(node.test, visible)
            for arm, runs in ((node.body, taken is not False), (node.orelse, taken is not True)):
                if not runs:
                    continue
                inner = visible if taken is not None else dict(visible)
                for child in arm:
                    walk(child, inner)
            return
        for child in _kids(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, dict(visible))
            else:
                walk(child, visible)

    walk(tree, {})
    return visible_at


def _open_alias(name, modules):
    """Maps a bare callable to builtin, a COMPRESSED_OPENERS key or None, so PIL's open gives None."""
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
    """Matches pathlib classes under any alias, since P.read_text(SOURCE) passes the path in slot 0."""
    if name is None:
        return False
    return (modules.get(name) or name).split(".")[-1] in PATH_CLASSES


def _is_path_attr(node: ast.AST) -> bool:
    """True for a qualified path class, as in `pathlib.Path` or `pl.Path`."""
    return isinstance(node, ast.Attribute) and node.attr in PATH_CLASSES


def _is_path_preserving(func) -> bool:
    """Qualified spellings like pathlib.Path(p) or os.path.join(p, x) count the same as bare names."""
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
    """The path a read targets: the receiver, or the first argument when the receiver is Path or a
    module."""
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
    """Anchor decides scope: a checked-in root keeps a path in scope, while tmp_path / SUBDIR does not."""
    while True:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node = node.left
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in SELF_NAMES
            ):
                return node  # self.SOURCE names the class attribute, not self
            node = node.value
        elif isinstance(node, ast.Call):
            func = node.func
            # `p.rglob(...)` anchors on p; Path(x), str(x), os.path.join(x, ...) anchor on the argument.
            if isinstance(func, ast.Attribute) and func.attr in PATH_METHODS:
                node = func.value
            elif _is_path_preserving(func) and node.args:
                node = node.args[0]
            else:
                # An unrecognised call says nothing about where its result points.
                return node
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
        # A runtime-created path is not in the tree, so only existing relative literals count.
        value = root.value
        if not value or "\n" in value or "\0" in value or os.path.isabs(value):
            return False
        try:
            return (REPO / value).exists()
        except OSError:
            return False
    if isinstance(root, ast.Attribute):
        return root.attr in attrs
    if not isinstance(root, ast.Name):
        return False
    if root.id in derived:
        return True
    return root.id == "__file__" or (root.id in module_names and root.id not in shadowed)


def _class_path_attrs(tree: ast.Module, module_names: set) -> set:
    """Class attributes holding a checked-in path, read as self.NAME, are as provable as module names."""
    attrs, mixed = set(), set()
    for node in _walk(tree):
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
            # One attribute name with two meanings in two classes: neither is claimed.
            found = attrs if _is_checked_in_root(stmt.value, module_names, ()) else mixed
            found.update(bound)
    return attrs - mixed


def _reads_itself(name: str, value: ast.AST) -> bool:
    """A name rebound from its own read_text() still held the checked-in path at that read, so it counts."""
    if not isinstance(value, ast.Call):
        return False
    expr = _path_expr(value)
    return isinstance(expr, ast.Name) and expr.id == name


def _unpack(target, value, paired: bool):
    """Destructured names take their own element when the sides line up, else the whole iterable."""
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
    """Locals that only ever hold a checked-in path, iterated to a fixpoint so chained locals still
    count."""
    assignments = []
    targets = set()
    bad = set()
    for node in _walk(func):
        paired = False
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            paired = True
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            target, value = node.target, node.iter
        else:
            continue
        for name_node, bound in _unpack(target, value, paired):
            targets.add(id(name_node))
            if not _reads_itself(name_node.id, bound):
                assignments.append((name_node.id, bound))
    for node in _walk(func):
        # A with-as or an augassign says nothing about the value it binds.
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if id(node) not in targets:
                bad.add(node.id)
    args = func.args
    bad.update(a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs])
    bad -= set(seed)
    good: set = set(seed)
    while True:
        grown = set(good) | {
            name
            for name, value in assignments
            if name not in bad and _is_checked_in_root(value, module_names, shadowed, good)
        }
        # A name assigned both a checked-in path and something else stays out.
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
    """Reads parameter values from @pytest.mark.parametrize, the only call site a parametrized test has."""
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
    """Parameters that only receive a checked-in path, one call hop out; defs are matched by identity."""
    # Record each def's scope so calls resolve to the nearest enclosing def, as Python does.
    scope_of: dict = {}
    defs_in: dict = {}

    def _index(node, scope):
        for child in _kids(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs_in.setdefault(id(scope), {}).setdefault(child.name, child)
                scope_of[id(child)] = scope
                _index(child, child)
            elif isinstance(child, ast.ClassDef):
                _index(child, scope)  # a class body is not a name lookup scope
            else:
                _index(child, scope)

    _index(tree, tree)

    def _lookup(name, scope):
        while scope is not None:
            found = defs_in.get(id(scope), {}).get(name)
            if found is not None:
                return found
            scope = scope_of.get(id(scope))
        return None

    owner: dict = {}

    def _mark(node, owning):
        if isinstance(node, ast.Call):
            owner[id(node)] = owning
        for child in _kids(node):
            nested = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            _mark(child, child if nested else owning)

    _mark(tree, None)
    # So `self._read(...)` resolves to this class's method, not a sibling class's.
    in_class: dict = {}

    def _mark_class(node, cls):
        if isinstance(node, ast.Call):
            in_class[id(node)] = cls
        for child in _kids(node):
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
        # Recomputed each pass so a parameter resolved last time can feed a local this time.
        scope: dict = {}
        for call in _walk(tree):
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
                positional = positional[1:]  # the receiver already fills `self`
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
    """Calls at any depth whose path is provably checked in; tmp_path and tempfile I/O stay out of scope."""
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
            # Seed parameters first: `p = root / "x.py"` needs `root` known as checked-in.
            seeded = {p for f, p in params if f == id(node)}
            derived = _checked_in_locals(node, module_names, shadowed, seeded)
        elif _is_main_guard(node):
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
        for child in _kids(node):
            yield from visit(child, shadowed, derived)

    yield from visit(tree, frozenset())


def _open_mode(call: ast.Call, mode_index: int):
    """A splat or non-literal mode is UNKNOWN_MODE, not "r", since "rb" would reject an encoding."""
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
    """Only a real encoding pins one: encoding = None or "locale" re-selects the platform default."""
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
    """A splat makes positions meaningless, so it counts as naming an encoding."""
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
        # Unbound `Path.read_text(p)` (or `pathlib.Path...`) puts the instance in slot 0.
        shift = 1 if _is_path_class(receiver, modules) or _is_path_attr(func.value) else 0
        if func.attr in GUARDED_METHODS:
            if func.attr == "read_text" and not shift and call.args:
                first = call.args[0]
                # Bound read_text takes encoding first; any other positional is importlib.metadata's
                # Distribution, which takes a filename and no encoding.
                if isinstance(first, ast.Constant) and first.value in PLATFORM_DEFAULT_ENCODINGS:
                    return "read_text()"
                return None
            position = ENCODING_POSITION[func.attr] + shift
            return None if _pins_encoding(call, position) else f"{func.attr}()"
        if func.attr == "open":
            if receiver is not None and _origin_root(receiver, modules) in BUILTIN_OPEN_MODULES:
                if not _is_text(call, 1) or _pins_encoding(call, ENCODING_POSITION["open"]):
                    return None
                return f"{receiver}.open()"
            compressed = _compressed_key(receiver, modules) if receiver else None
            if compressed is not None:
                mode = _open_mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None
                return (
                    None
                    if _pins_encoding(call, COMPRESSED_OPENERS[compressed])
                    else f"{compressed}.open()"
                )
            # Other module receivers are foreign openers: tarfile takes a compression mode, Image binary.
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
        if alias == "builtin" and _is_text(call, 1):
            return None if _pins_encoding(call, ENCODING_POSITION["open"]) else "open()"
        if alias is not None and alias != "builtin":
            mode = _open_mode(call, 1)
            if mode is UNKNOWN_MODE or "t" not in str(mode):
                return None
            position = COMPRESSED_OPENERS[alias]
            return None if _pins_encoding(call, position) else f"{alias}.open()"
    return None


def _scan(tree: ast.Module, rel: str):
    """Offenders from both rules, reported once each and in source order."""
    try:
        yield from _scan_one(tree, rel)
    finally:
        # Freed nodes' ids can be reused by a later file's nodes.
        _WALKS.clear()
        _KIDS.clear()
        _CONSUMED.clear()


def _scan_one(tree: ast.Module, rel: str):
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
            continue  # ZipFile.open and friends have no encoding to name
        expr = _path_expr(call, visible_at.get(id(call), modules))
        root = _path_root(expr) if expr is not None else None
        if isinstance(root, ast.Name) and root.id in temp_roots:
            continue  # the run made this file, so the platform default is safe
        name = _offender(call, visible_at.get(id(call), modules))
        if name is not None:
            yield f"{rel}:{call.lineno}: {name}"


# Batched so the scan spreads across xdist workers; each file is still scanned once.
_BATCHES = 16


def _batches():
    ordered = sorted(SOURCES)
    return [ordered[i::_BATCHES] for i in range(_BATCHES)]


@pytest.mark.parametrize("batch", range(_BATCHES))
def test_checked_in_file_reads_name_an_encoding(batch: int):
    offenders = []
    for path in _batches()[batch]:
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        offenders.extend(_scan(tree, path.relative_to(REPO).as_posix()))
    assert offenders == [], (
        f"{len(offenders)} file reads in the test trees touch a checked-in file "
        "with the platform default encoding, so they break on Windows as soon "
        'as that file gains a non-ASCII byte. Pass encoding = "utf-8": '
        f"{offenders[:10]}"
    )
