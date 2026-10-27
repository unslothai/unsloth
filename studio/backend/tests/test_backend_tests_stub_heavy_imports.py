# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No test module may import a heavy backend module at module scope without stubbing it first."""

from __future__ import annotations

import ast
import warnings
from functools import lru_cache
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND = _TESTS_DIR.parent

# Packages the backend pytest job does not install.
_HEAVY_PACKAGES = ("unsloth", "unsloth_zoo", "trl")

_SKIP_TOP_LEVEL = frozenset({"tests", "vendor", "unsloth_compiled_cache"})

_REQUIRED_STUB = "unsloth"


def _parse(source: str) -> ast.Module:
    """Parse with SyntaxWarning silenced, so invalid escapes in other test files do not warn here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def _module_name(path: Path) -> str:
    rel = path.relative_to(_BACKEND)
    parts = list(rel.parts)
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_scope_imports(tree: ast.Module, package: str) -> set[str]:
    """Collect module-scope imports only; in-function or try imports are lazy or guarded."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".") if package else []
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base]) if base else ".".join(anchor)
            if not base:
                continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _needs_heavy(names: set[str]) -> bool:
    return any(n == h or n.startswith(f"{h}.") for n in names for h in _HEAVY_PACKAGES)


@lru_cache(maxsize = 1)
def _heavy_backend_modules() -> frozenset[str]:
    """Backend modules needing unsloth/trl: direct importers, closed over backend module-scope imports."""
    imports: dict[str, set[str]] = {}
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND)
        if rel.parts[0] in _SKIP_TOP_LEVEL:
            continue
        try:
            tree = _parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        name = _module_name(path)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        imports[name] = _module_scope_imports(tree, package)

    tainted = {name for name, names in imports.items() if _needs_heavy(names)}
    changed = True
    while changed:
        changed = False
        for name, names in imports.items():
            if name not in tainted and names & tainted:
                tainted.add(name)
                changed = True
    return frozenset(tainted)


# ModuleNotFoundError catches strictly less than ImportError, so it is excluded.
_CATCHES_IMPORT_ERROR = frozenset({"ImportError", "Exception", "BaseException"})


def _catches_import_error(node: ast.AST) -> bool:
    """Whether an except type is ImportError by whole dotted name; MyImportError does not count."""
    if isinstance(node, ast.Tuple):
        return any(_catches_import_error(element) for element in node.elts)
    if isinstance(node, ast.Name):
        return node.id in _CATCHES_IMPORT_ERROR
    if isinstance(node, ast.Attribute):
        return node.attr in _CATCHES_IMPORT_ERROR
    return False


def _reraises(handler: ast.ExceptHandler) -> bool:
    """A reachable raise in the handler lets ImportError escape; a pytest.skip call does not count."""
    return any(isinstance(node, ast.Raise) for node in _reachable_import_time_nodes(handler))


def _absorbs_import_error(node: ast.Try) -> bool:
    """Only the first matching handler decides; an earlier re-raising handler lets ImportError escape."""
    for handler in node.handlers:
        if handler.type is None or _catches_import_error(handler.type):
            return not _reraises(handler)
    return False


def _guarded_by_import_error(tree: ast.Module) -> set[int]:
    """Nodes in a try whose handler catches ImportError by whole name; substring matches do not count."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if _absorbs_import_error(node):
            for statement in node.body:
                guarded.update(id(child) for child in ast.walk(statement))
    return guarded


def _first_heavy_import_line(tree: ast.Module, heavy: frozenset[str]) -> int | None:
    """First import-time heavy import line, walking with/if blocks but skipping unreachable branches."""
    lines = _heavy_import_lines(tree, heavy)
    return lines[0] if lines else None


def _heavy_import_lines(tree: ast.Module, heavy: frozenset[str]) -> list[int]:
    """Every import-time heavy import, since each if/else branch can import one that needs stubbing."""
    guarded = _guarded_by_import_error(tree)
    lines: list[int] = []
    for statement in tree.body:
        for node in _reachable_import_time_nodes(statement):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in guarded:
                continue
            if _module_scope_imports(ast.Module(body = [node], type_ignores = []), "") & heavy:
                lines.append(node.lineno)
    return sorted(lines)


def _runtime_nodes(node: ast.AST):
    """Import-time nodes; def bodies and provably dead branches are skipped, since they install nothing."""
    if isinstance(node, ast.If) and _constant_test(node) is not None:
        for child in node.body if _constant_test(node) else node.orelse:
            yield from _runtime_nodes(child)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _runtime_nodes(child)


def _names_required_stub(nodes: list[ast.AST]) -> bool:
    """Whether any node is the string ``unsloth`` (or a submodule of it)."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value == _REQUIRED_STUB or node.value.startswith(f"{_REQUIRED_STUB}."))
        for node in nodes
    )


def _reads_a_named_stub(nodes: list[ast.AST], named: frozenset[str]) -> bool:
    """Whether any node reads a module-level name that was bound to the required stub."""
    return any(isinstance(node, ast.Name) and node.id in named for node in nodes)


def _bound_names(statement: ast.AST) -> set[str]:
    """Module-level names this statement assigns to."""
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    return {
        node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name)
    }


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _writes_sys_modules(nodes: list[ast.AST]) -> bool:
    """True when nodes write sys.modules; reads such as .get() or a membership test are checks only."""
    for node in nodes:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("setdefault", "update", "__setitem__")
        ):
            targets = [node.func.value]
        if any(
            _is_sys_modules(target)
            or (isinstance(target, ast.Subscript) and _is_sys_modules(target.value))
            for target in targets
        ):
            return True
    return False


def _installs_stub(
    nodes: list[ast.AST],
    helpers: dict[str, ast.AST] | None = None,
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Calls count by what the helper does, not its name; the name is a fallback for unresolved calls."""
    if _writes_sys_modules(nodes):
        return True
    helpers = helpers or {}
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name in helpers:
            if name not in seen and _installs_stub(
                list(_helper_nodes_entered(helpers[name])), helpers, seen | {name}
            ):
                return True
        elif "stub" in name.lower():
            return True
    return False


def _stub_helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level sync defs by name; async defs are excluded because calling one runs no body."""
    return {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
    }


def _own_yields(node: ast.AST):
    """``yield`` expressions belonging to this function, not to one nested inside it."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            yield child
        yield from _own_yields(child)


def _probes_availability(node: ast.Try) -> bool:
    """A try that imports the module and returns when it succeeds; the stub below it is skipped on
    purpose."""
    if any(_reraises(handler) for handler in node.handlers):
        return False
    return any(
        isinstance(child, ast.Call) and _callee_name(child) in ("import_module", "__import__")
        for statement in node.body
        for child in ast.walk(statement)
    )


def _can_exit_early(node: ast.AST, flags: dict[str, bool] | None = None) -> bool:
    """An early exit before a stub is a gap, except the already-imported and importable-probe returns."""
    flags = flags or {}
    if isinstance(node, ast.If) and _constant_test(node) is None:
        if _true_when_imported(node.test, flags, allow_variable_key = True) is True:
            return any(_can_exit_early(child, flags) for child in node.orelse)
        return any(_can_exit_early(child, flags) for child in [*node.body, *node.orelse])
    if isinstance(node, ast.Try) and _probes_availability(node):
        return False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        return False
    if isinstance(node, (ast.Return, ast.Raise)):
        return True
    return any(_can_exit_early(child, flags) for child in ast.iter_child_nodes(node))


def _helper_nodes_entered(helper: ast.AST):
    """Stops at a generator's first yield, since code after it runs on exit, after the import is tried."""
    stop = min((node.lineno for node in _own_yields(helper)), default = None)
    for statement in getattr(helper, "body", []):
        if stop is not None and statement.lineno >= stop:
            break
        for node in _certain_nodes(statement):
            if stop is None or getattr(node, "lineno", 0) < stop:
                yield node
        if _can_exit_early(statement):
            return


def _helper_installs_stub(
    name: str,
    helpers: dict[str, ast.AST],
    named: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Whether calling a helper installs the stub, following nested calls; only the entered part counts."""
    helper = helpers.get(name)
    if helper is None or name in seen:
        return False
    nodes = list(_helper_nodes_entered(helper))
    if not (_names_required_stub(nodes) or _reads_a_named_stub(nodes, named)):
        return False
    if _installs_stub(nodes, helpers, seen | {name}):
        return True
    return any(
        _helper_installs_stub(_callee_name(node), helpers, named, seen | {name})
        for node in nodes
        if isinstance(node, ast.Call)
    )


_BLOCK_FIELDS = ("body", "orelse", "finalbody")


def _end_line(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or node.lineno


def _true_when_imported(
    node: ast.AST,
    flags: dict[str, bool],
    *,
    allow_variable_key: bool = False,
) -> bool | None:
    """Whether the expression is true once the module is in sys.modules; None if it does not ask."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _true_when_imported(node.operand, flags, allow_variable_key = allow_variable_key)
        return None if inner is None else not inner
    if isinstance(node, ast.Name):
        return flags.get(node.id)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        if any(_is_sys_modules(comparator) for comparator in node.comparators):
            if not _key_decides_the_import(node.left, allow_variable_key):
                return None
            if isinstance(node.ops[0], ast.In):
                return True
            if isinstance(node.ops[0], ast.NotIn):
                return False
    return None


def _key_decides_the_import(node: ast.AST, allow_variable: bool) -> bool:
    """The key must be the required stub or a heavy backend module, not an unrelated cached package."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (
            node.value == _REQUIRED_STUB
            or node.value.startswith(f"{_REQUIRED_STUB}.")
            or node.value in _heavy_backend_modules()
        )
    return allow_variable


def _sys_modules_flags(tree: ast.Module) -> dict[str, bool]:
    """Module-level names bound to an "is it already imported" test, with polarity."""
    flags: dict[str, bool] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or statement.value is None:
            continue
        answer = _true_when_imported(statement.value, flags)
        if answer is not None:
            for name in _bound_names(statement):
                flags[name] = answer
    return flags


def _certain_nodes(node: ast.AST, flags: dict[str, bool] | None = None):
    """Like _runtime_nodes, but drops while, except and undecided if bodies, which might not run."""
    flags = flags or {}
    if isinstance(node, ast.If) and _constant_test(node) is None:
        imported = _true_when_imported(node.test, flags)
        if imported is None:
            return
        for child in node.orelse if imported else node.body:
            yield from _certain_nodes(child, flags)
        return
    if isinstance(node, ast.While):
        return
    if isinstance(node, ast.Try):
        # Only the finally: a handled exception can stop the try body part way.
        for child in node.finalbody:
            yield from _certain_nodes(child, flags)
        return
    yield from _runtime_nodes(node)


def _running_before(
    body: list[ast.stmt],
    line: int,
    flags: dict[str, bool] | None = None,
):
    """Statements that ran before line, following only the branch that contains it, not sibling branches."""
    for statement in body:
        if statement.lineno >= line:
            return
        if _end_line(statement) < line:
            yield statement, list(_certain_nodes(statement, flags))
            continue
        header = [
            node
            for field, value in ast.iter_fields(statement)
            if field not in _BLOCK_FIELDS and field != "handlers"
            for child in (value if isinstance(value, list) else [value])
            if isinstance(child, ast.AST)
            for node in _certain_nodes(child, flags)
        ]
        yield statement, header
        blocks = [getattr(statement, field, None) or [] for field in _BLOCK_FIELDS]
        blocks += [handler.body for handler in getattr(statement, "handlers", None) or []]
        for block in blocks:
            if block and block[0].lineno <= line <= _end_line(block[-1]):
                yield from _running_before(block, line, flags)


def _stubs_before(tree: ast.Module, line: int | None) -> bool:
    """Module-scope code before line must name unsloth and install a stub in one statement."""
    if line is None:
        return True
    named: frozenset[str] = frozenset()
    helpers = _stub_helpers(tree)
    for statement, nodes in _running_before(tree.body, line, _sys_modules_flags(tree)):
        if any(
            isinstance(node, ast.Call) and _helper_installs_stub(_callee_name(node), helpers, named)
            for node in nodes
        ):
            return True
        names_it = _names_required_stub(nodes) or _reads_a_named_stub(nodes, named)
        if not names_it:
            continue
        if _installs_stub(nodes, helpers):
            return True
        named |= _bound_names(statement)
    return False


def _is_offender(source: str, heavy: frozenset[str]) -> bool:
    """Parses every candidate; a textual prefilter misses forms like from core.training import trainer."""
    try:
        tree = _parse(source)
    except SyntaxError:
        return False
    lines = _heavy_import_lines(tree, heavy)
    return any(not _stubs_before(tree, line) for line in lines)


def _offenders() -> list[str]:
    heavy = _heavy_backend_modules()
    return [
        path.name
        for path in sorted(_TESTS_DIR.glob("test_*.py"))
        if _is_offender(path.read_text(encoding = "utf-8"), heavy)
    ]


def _import_time_nodes(node: ast.AST):
    """Import-time nodes: class bodies, decorators and defaults run at import; function bodies defer."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        body = node.body if isinstance(node.body, list) else [node.body]
        deferred = {id(statement) for statement in body}
        for child in ast.iter_child_nodes(node):
            if id(child) not in deferred:
                yield from _import_time_nodes(child)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _import_time_nodes(child)


def _constant_test(node: ast.If) -> bool | None:
    """Constant truth of TYPE_CHECKING, True/False literals; None otherwise, so both branches are walked."""
    test = node.test
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return False
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return False
    return None


def _reachable_import_time_nodes(node: ast.AST):
    """Import-time nodes minus dead branches; an eager import in a class body still caches the module."""
    if isinstance(node, ast.If) and _constant_test(node) is not None:
        taken = node.body if _constant_test(node) else node.orelse
        for child in taken:
            yield from _reachable_import_time_nodes(child)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        body = node.body if isinstance(node.body, list) else [node.body]
        deferred = {id(statement) for statement in body}
        for child in ast.iter_child_nodes(node):
            if id(child) not in deferred:
                yield from _reachable_import_time_nodes(child)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _reachable_import_time_nodes(child)


def _eagerly_imports(tree: ast.Module, target: str, line: int) -> bool:
    """Module-scope import of target under live stubs before line; a probe try/except does not count."""
    for statement in tree.body:
        if statement.lineno >= line:
            break
        if not _stubs_before(tree, statement.lineno):
            continue
        for node in _reachable_import_time_nodes(statement):
            if isinstance(node, ast.Import):
                if any(alias.name == target for alias in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == target or any(
                    f"{module}.{alias.name}" == target for alias in node.names
                ):
                    return True
    return False


def _importorskip_target(node: ast.Call) -> ast.AST | None:
    """Module name from positional arg or the modname keyword, since importorskip accepts either."""
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "modname":
            return keyword.value
    return None


def _skips_on_plain_import_error(node: ast.Call) -> bool:
    """Only an explicit exc_type=ImportError lets importorskip skip a plain ImportError."""
    for keyword in node.keywords:
        if keyword.arg != "exc_type":
            continue
        value = keyword.value
        if isinstance(value, ast.Name):
            return value.id == "ImportError"
        if isinstance(value, ast.Attribute):
            return value.attr == "ImportError"
    return False


def _module_functions(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """All defs by name, nested included, so a nested helper called at import keeps the call's line."""
    functions: dict[str, list[ast.AST]] = {}

    def _collect(body):
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(statement.name, []).append(statement)
                _collect(statement.body)
            elif hasattr(statement, "body"):
                _collect(statement.body)
                _collect(getattr(statement, "orelse", []) or [])
                _collect(getattr(statement, "finalbody", []) or [])

    _collect(tree.body)
    return functions


def _functions_called_at_import(tree: ast.Module) -> dict[str, int]:
    """Earliest import-time line reaching each def, followed transitively through helper calls."""
    functions = _module_functions(tree)
    first: dict[str, int] = {}
    for statement in tree.body:
        for node in _reachable_import_time_nodes(statement):
            if isinstance(node, ast.Call):
                name = _callee_name(node)
                if name in functions and node.lineno < first.get(name, node.lineno + 1):
                    first[name] = node.lineno
    changed = True
    while changed:
        changed = False
        for caller, line in list(first.items()):
            # Only calls that actually run; every same-named helper takes the boundary.
            for definition in functions[caller]:
                for statement in definition.body:
                    for node in _reachable_import_time_nodes(statement):
                        if not isinstance(node, ast.Call):
                            continue
                        callee = _callee_name(node)
                        if callee in functions and line < first.get(callee, line + 1):
                            first[callee] = line
                            changed = True
                    # Calls after an unconditional exit are unreachable at import.
                    if isinstance(statement, (ast.Return, ast.Raise)):
                        break
    return first


def _reachable_nodes(node: ast.AST):
    """Like the import-time walk but enters def bodies, since a call there runs when its test does."""
    if isinstance(node, ast.If) and _constant_test(node) is not None:
        yield from _reachable_body(node.body if _constant_test(node) else node.orelse)
        return
    yield node
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            yield from _reachable_body(value)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, ast.AST):
                    yield from _reachable_nodes(child)
        elif isinstance(value, ast.AST):
            yield from _reachable_nodes(value)


def _reachable_body(body: list[ast.stmt]):
    """Statements up to the first return, raise, break or continue; code after those is dead."""
    for statement in body:
        yield from _reachable_nodes(statement)
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return


def _importorskip_bare_names(tree: ast.Module) -> frozenset[str]:
    """Bare names bound to importorskip by from pytest import, so an aliased call is still recognised."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "importorskip":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "importorskip":
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(names)


def _importorskip_calls(tree: ast.Module, heavy: frozenset[str]) -> list[tuple[str, int]]:
    """Heavy importorskip targets; boundary is the call's line at module scope, else end of module."""
    bare_names = _importorskip_bare_names(tree)
    module_scope = {
        id(node) for statement in tree.body for node in _reachable_import_time_nodes(statement)
    }
    reachable = {id(node) for node in _reachable_nodes(tree)}
    end = max((statement.lineno for statement in tree.body), default = 0) + 1
    called_at_import = _functions_called_at_import(tree)
    # Calls belong to the innermost def; nested bodies are deferred.
    in_function = {
        id(node): name
        for name, definitions in _module_functions(tree).items()
        for definition in definitions
        for statement in definition.body
        for node in _reachable_import_time_nodes(statement)
    }
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) not in reachable:
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr != "importorskip":
                continue
        elif isinstance(func, ast.Name):
            if func.id not in bare_names:
                continue
        else:
            continue
        if _skips_on_plain_import_error(node):
            continue
        target = _importorskip_target(node)
        if (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value in heavy
        ):
            if id(node) in module_scope:
                boundary = node.lineno
            else:
                boundary = called_at_import.get(in_function.get(id(node)), end)
            calls.append((target.value, boundary))
    return calls


def _stub_record_names(tree: ast.Module) -> frozenset[str]:
    """Module-level list the stub helper appends to, found via the call naming the required stub."""
    helpers = {
        _callee_name(node)
        for statement in tree.body
        for node in _runtime_nodes(statement)
        if isinstance(node, ast.Call) and _names_required_stub(list(ast.walk(node)))
    }
    bound = {name for statement in tree.body for name in _bound_names(statement)}
    recorded: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name not in helpers:
            continue
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "add", "insert")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in bound
            ):
                recorded.add(node.func.value.id)
    return frozenset(recorded)


def _drops_stubs(tree: ast.Module, line: int) -> bool:
    """Whether sys.modules.pop removes our recorded stubs at module scope before line, not any pop."""
    recorded = _stub_record_names(tree)
    for statement in tree.body:
        if statement.lineno >= line:
            break
        nodes = list(_runtime_nodes(statement))
        pops_sys_modules = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and _is_sys_modules(node.func.value)
            for node in nodes
        )
        if pops_sys_modules and (
            _names_required_stub(nodes) or _reads_a_named_stub(nodes, recorded)
        ):
            return True
    return False


def _importorskip_offence(tree: ast.Module, heavy: frozenset[str]) -> bool:
    """Whether a heavy importorskip target lacks stubs before its boundary; guard and test share it."""
    for target, boundary in _importorskip_calls(tree, heavy):
        if not _stubs_before(tree, boundary):
            return True
        # Judge both install and pop against the call's own boundary.
        if _drops_stubs(tree, boundary) and not _eagerly_imports(tree, target, boundary):
            return True
    return False


def _importorskip_offenders() -> list[str]:
    heavy = _heavy_backend_modules()
    offenders: list[str] = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        try:
            tree = _parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if _importorskip_offence(tree, heavy):
            offenders.append(path.name)
    return offenders


def test_no_test_module_reaches_a_heavy_module_through_importorskip_unstubbed():
    """importorskip skips only ModuleNotFoundError by default; a plain ImportError from unsloth fails."""
    offenders = _importorskip_offenders()
    assert not offenders, (
        f"{len(offenders)} test module(s) reach a backend module that needs unsloth through "
        f"pytest.importorskip without installing the stub, so they fail (not skip) on the "
        f"backend pytest matrix whenever they run before whichever file stubs it: {offenders}. "
        f"Copy the _stub_if_missing block from test_audio_type_inconclusive.py."
    )


def test_the_importorskip_guard_would_catch_an_unstubbed_module():
    """Pin every answer, so the check above cannot pass by matching nothing."""
    heavy = _heavy_backend_modules()
    assert "core.inference.inference" in heavy

    def calls(source):
        return _importorskip_calls(_parse(source), heavy)

    def _offender_free(source):
        return not _importorskip_offence(_parse(source), heavy)

    def safe(source):
        return not _importorskip_offence(_parse(source), heavy)

    lazy_unstubbed = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert calls(lazy_unstubbed) and not safe(lazy_unstubbed)

    lazy_stubbed = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert safe(lazy_stubbed)

    bare_unstubbed = (
        "from pytest import importorskip\n"
        "def test_x():\n"
        "    inf = importorskip('core.inference.inference')\n"
    )
    assert calls(bare_unstubbed) == [("core.inference.inference", 3)]
    assert not safe(bare_unstubbed)

    stub_too_late = (
        "import pytest, sys\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(stub_too_late) == [("core.inference.inference", 2)]
    assert not safe(stub_too_late)

    stub_in_time = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert safe(stub_in_time)

    nested_called_at_import = (
        "import pytest, sys\n"
        "def _outer():\n"
        "    def _inner():\n"
        "        return pytest.importorskip('core.inference.inference')\n"
        "    return _inner()\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(nested_called_at_import) == [("core.inference.inference", 6)]
    assert not safe(nested_called_at_import)

    nested_stub_in_time = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "def _outer():\n"
        "    def _inner():\n"
        "        return pytest.importorskip('core.inference.inference')\n"
        "    return _inner()\n"
        "_outer()\n"
    )
    assert safe(nested_stub_in_time)

    after_return = (
        "import pytest, sys\n"
        "def _outer():\n"
        "    return None\n"
        "    return _inner()\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(after_return), after_return

    same_name = (
        "import pytest, sys\n"
        "def _unused():\n"
        "    def _probe():\n        return None\n"
        "    return _probe()\n"
        "def _outer():\n"
        "    def _probe():\n"
        "        return pytest.importorskip('core.inference.inference')\n"
        "    return _probe()\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(same_name) == [("core.inference.inference", 10)], calls(same_name)
    assert not safe(same_name), same_name

    for dead in (
        "import pytest\n"
        "def test_x():\n"
        "    return None\n"
        "    pytest.importorskip('core.inference.inference')\n",
        "import pytest\n"
        "raise SystemExit(0)\n"
        "inf = pytest.importorskip('core.inference.inference')\n",
    ):
        assert safe(dead), dead

    nested_deferred = (
        "import pytest, sys\n"
        "def _outer():\n"
        "    def _inner():\n"
        "        return pytest.importorskip('core.inference.inference')\n"
        "    return _inner()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(nested_deferred)

    class_body_late = (
        "import pytest, sys\n"
        "class T:\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(class_body_late) == [("core.inference.inference", 3)]
    assert not safe(class_body_late)

    default_arg_late = (
        "import pytest, sys\n"
        "def f(x = pytest.importorskip('core.inference.inference')):\n"
        "    pass\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(default_arg_late) == [("core.inference.inference", 2)]
    assert not safe(default_arg_late)

    stub_import_drop = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(stub_import_drop)

    stub_drop_no_import = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(stub_drop_no_import)

    drop_then_import_time_call = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert calls(drop_then_import_time_call) == [("core.inference.inference", 5)]
    assert not _offender_free(drop_then_import_time_call)

    import_time_call_then_drop = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
    )
    assert _offender_free(import_time_call_then_drop)

    eager_then_drop_then_call = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(eager_then_drop_then_call)

    failed_probe_then_drop = (
        "import pytest, sys\n"
        "try:\n"
        "    import core.inference.inference\n"
        "except ImportError:\n"
        "    pass\n"
        "_stub_if_missing('unsloth', ())\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(failed_probe_then_drop)

    unrelated_pop = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "sys.modules.pop('routes.foo', None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(unrelated_pop)
    foreign_modules = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "importlib.modules.pop('unsloth', None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(foreign_modules)
    _HELPER = (
        "def _stub_if_missing(name, attrs):\n"
        "    _STUBBED.append(name)\n"
        "    sys.modules[name] = object()\n"
    )
    drop_through_a_list = (
        "import pytest, sys\n"
        "_STUBBED = []\n" + _HELPER + "_stub_if_missing('unsloth', ())\n"
        "for _n in reversed(_STUBBED):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(drop_through_a_list)

    unrelated_cleanup_list = (
        "import pytest, sys\n"
        "_STUBBED = []\n"
        "_JUNK = ['routes.foo']\n" + _HELPER + "_stub_if_missing('unsloth', ())\n"
        "for _n in _JUNK:\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(unrelated_cleanup_list)

    keyword_target = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(modname = 'core.inference.inference')\n"
    )
    assert calls(keyword_target) == [("core.inference.inference", 3)]
    assert not safe(keyword_target)

    explicit_import_error = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(\n"
        "        'core.inference.inference', exc_type = ImportError\n"
        "    )\n"
    )
    assert calls(explicit_import_error) == []
    assert safe(explicit_import_error)
    explicit_default = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(\n"
        "        'core.inference.inference', exc_type = ModuleNotFoundError\n"
        "    )\n"
    )
    assert calls(explicit_default) and not safe(explicit_default)

    type_checking_import = (
        "import pytest, sys\n"
        "from typing import TYPE_CHECKING\n"
        "_stub_if_missing('unsloth', ())\n"
        "if TYPE_CHECKING:\n"
        "    import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(type_checking_import)

    class_body_import = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "class _Eager:\n"
        "    import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(class_body_import)

    helper_called_at_import = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "_probe()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(helper_called_at_import) == [("core.inference.inference", 4)]
    assert not safe(helper_called_at_import)

    nested_helper_at_import = (
        "import pytest, sys\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "def _outer():\n"
        "    return _inner()\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(nested_helper_at_import) == [("core.inference.inference", 6)]
    assert not safe(nested_helper_at_import)

    unreachable_inner_call = (
        "import pytest, sys\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "def _outer():\n"
        "    if False:\n"
        "        _inner()\n"
        "    return None\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(unreachable_inner_call)
    deferred_inner_call = (
        "import pytest, sys\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "def _outer():\n"
        "    def _later():\n"
        "        return _inner()\n"
        "    return _later\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(deferred_inner_call)

    nested_helper_never_called = (
        "import pytest, sys\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "def _outer():\n"
        "    return _inner()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(nested_helper_never_called)

    helper_never_called = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(helper_never_called)

    aliased = (
        "from pytest import importorskip as ios\n"
        "def test_x():\n"
        "    inf = ios('core.inference.inference')\n"
    )
    assert calls(aliased) == [("core.inference.inference", 3)]
    assert not safe(aliased)
    rebound = (
        "import pytest\n"
        "_ios = pytest.importorskip\n"
        "def test_x():\n"
        "    inf = _ios('core.inference.inference')\n"
    )
    assert calls(rebound) and not safe(rebound)
    unrelated_bare = (
        "from mymod import importorskip\n"
        "def test_x():\n"
        "    inf = importorskip('core.inference.inference')\n"
    )
    assert calls(unrelated_bare) == []

    else_of_true = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "if True:\n"
        "    pass\n"
        "else:\n"
        "    import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(else_of_true)
    else_of_false = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "if False:\n"
        "    pass\n"
        "else:\n"
        "    import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(else_of_false)

    nested_def_holds_the_call = (
        "import pytest, sys\n"
        "def _outer():\n"
        "    def _later():\n"
        "        return pytest.importorskip('core.inference.inference')\n"
        "    return _later\n"
        "_outer()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(nested_def_holds_the_call)

    type_checking_call = (
        "import pytest\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert calls(type_checking_call) == []
    assert safe(type_checking_call)
    disabled_call = (
        "import pytest\n"
        "if False:\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert calls(disabled_call) == []
    assert safe(disabled_call)
    helper_called_only_when_disabled = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "if False:\n"
        "    _probe()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(helper_called_only_when_disabled)

    assert calls("import pytest\ndef test_x():\n    pytest.importorskip('numpy')\n") == []


def test_the_heavy_module_set_is_derived_from_the_backend_sources():
    """A derivation that quietly returned nothing would make the guard below pass on anything."""
    heavy = _heavy_backend_modules()
    assert "core.training.trainer" in heavy
    assert "core.inference.inference" in heavy
    assert not any(name.startswith("tests.") for name in heavy)


def test_no_test_module_imports_an_unsloth_backed_module_unstubbed():
    offenders = _offenders()
    assert not offenders, (
        f"{len(offenders)} test module(s) import a backend module that needs unsloth at module "
        f"scope without stubbing its heavy deps first, so they fail COLLECTION on the backend "
        f"pytest matrix (which installs neither unsloth nor trl) and take the whole job down: "
        f"{offenders}. Copy the _stub_if_missing block from test_trainer_stdout_quiet.py, above "
        f"the import."
    )


def test_the_guard_would_catch_an_unstubbed_module():
    """The guard above passes trivially if its matching is wrong, so pin both answers here."""
    heavy = _heavy_backend_modules()

    for source in (
        "from core.training import trainer as t\n",
        "from core.training.trainer import UnslothTrainer\n",
        "import core.training.trainer\n",
        "from core.inference.inference import InferenceEngine\n",
        "from core.inference import inference\n",
    ):
        assert _first_heavy_import_line(ast.parse(source), heavy) == 1, source
        assert not _stubs_before(ast.parse(source), 1), source
        assert _is_offender(source, heavy), source

    for stubbed in (
        '_stub_if_missing("unsloth", ())\nfrom core.training import trainer as t\n',
        'for _n, _a in (("unsloth", ()),):\n'
        "    _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
        'import sys\nsys.modules["unsloth"] = object()\n'
        "from core.training import trainer as t\n",
        'import sys\n_STUBS = {"unsloth": ()}\n'
        "for _n, _a in _STUBS.items():\n"
        "    _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
    ):
        assert _stubs_before(
            _parse(stubbed), _first_heavy_import_line(_parse(stubbed), heavy)
        ), stubbed
        assert not _is_offender(stubbed, heavy), stubbed

    for nested, at in (
        ("with open('x') as fh:\n    from core.training import trainer as t\n", 2),
        ("import os\nif os.environ.get('X'):\n    from core.inference import inference\n", 3),
    ):
        assert _first_heavy_import_line(_parse(nested), heavy) == at, nested
        assert _is_offender(nested, heavy), nested

    held = (
        "import contextlib, sys\n"
        '_STUBS = (("unsloth", ()),)\n'
        "@contextlib.contextmanager\n"
        "def _stubbed():\n"
        "    for _n, _a in _STUBS:\n"
        "        sys.modules[_n] = object()\n"
        "    yield\n"
        "with _stubbed():\n"
        "    from core.training import trainer as t\n"
    )
    assert _first_heavy_import_line(_parse(held), heavy) == 9, held
    assert _stubs_before(_parse(held), 9), held
    assert not _is_offender(held, heavy), held

    assert _is_offender(held.replace("with _stubbed():\n    from", "from"), heavy)

    after_yield = held.replace(
        "    for _n, _a in _STUBS:\n        sys.modules[_n] = object()\n    yield\n",
        "    yield\n    for _n, _a in _STUBS:\n        sys.modules[_n] = object()\n",
    )
    assert after_yield != held
    assert _is_offender(after_yield, heavy), after_yield
    never_runs = held.replace(
        "    for _n, _a in _STUBS:\n        sys.modules[_n] = object()\n",
        "    if False:\n        for _n, _a in _STUBS:\n            sys.modules[_n] = object()\n",
    )
    assert never_runs != held
    assert _is_offender(never_runs, heavy), never_runs
    nested_yield = held.replace(
        "def _stubbed():\n",
        "def _stubbed():\n    def _inner():\n        yield 1\n",
    )
    assert nested_yield != held
    assert not _is_offender(nested_yield, heavy), nested_yield

    inside_after = (
        "if True:\n"
        "    from core.inference import inference\n"
        '    _stub_if_missing("unsloth", ())\n'
    )
    assert _first_heavy_import_line(_parse(inside_after), heavy) == 2, inside_after
    assert _is_offender(inside_after, heavy), inside_after
    inside_before = (
        "if True:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
    )
    assert not _is_offender(inside_before, heavy), inside_before

    for unreachable in (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from core.training.trainer import UnslothTrainer\n",
        "if False:\n    import core.inference.inference\n",
    ):
        assert _first_heavy_import_line(_parse(unreachable), heavy) is None, unreachable
        assert not _is_offender(unreachable, heavy), unreachable
    stub_that_never_runs = (
        'if False:\n    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(stub_that_never_runs, heavy), stub_that_never_runs

    swallowed = (
        "try:\n"
        "    import optional_thing\n"
        '    _stub_if_missing("unsloth", ())\n'
        "except ImportError:\n"
        "    pass\n"
        "from core.training import trainer as t\n"
    )
    assert _is_offender(swallowed, heavy), swallowed
    in_finally = (
        "try:\n"
        "    import optional_thing\n"
        "except ImportError:\n"
        "    pass\n"
        "finally:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert not _is_offender(in_finally, heavy), in_finally

    one_branch_stubbed = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
        "else:\n"
        "    from core.training import trainer as t\n"
    )
    assert _heavy_import_lines(_parse(one_branch_stubbed), heavy) == [3, 5], one_branch_stubbed
    assert _is_offender(one_branch_stubbed, heavy), one_branch_stubbed
    both_stubbed = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
        "else:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.training import trainer as t\n"
    )
    assert not _is_offender(both_stubbed, heavy), both_stubbed

    other_branch = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "else:\n"
        "    from core.inference import inference\n"
    )
    assert _is_offender(other_branch, heavy), other_branch
    same_branch = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
    )
    assert not _is_offender(same_branch, heavy), same_branch

    optional = (
        'if enabled():\n    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(optional, heavy), optional

    already_imported = (
        "import sys\n"
        '_PRE = "core.training.trainer" in sys.modules\n'
        "if not _PRE:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert not _is_offender(already_imported, heavy), already_imported
    assert not _is_offender(
        "import sys\n"
        'if "core.training.trainer" not in sys.modules:\n'
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n",
        heavy,
    )
    wrong_way_up = (
        "import sys\n"
        '_PRE = "core.training.trainer" in sys.modules\n'
        "if _PRE:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(wrong_way_up, heavy), wrong_way_up

    for reraised in (
        "try:\n    from core.training import trainer as t\nexcept ImportError:\n    raise\n",
        "try:\n    from core.training import trainer as t\n"
        "except ImportError as exc:\n    raise RuntimeError('no trainer') from exc\n",
    ):
        assert _is_offender(reraised, heavy), reraised
    broad_first = (
        "try:\n    from core.training import trainer as t\n"
        "except Exception:\n    raise\n"
        "except ImportError:\n    t = None\n"
    )
    assert _is_offender(broad_first, heavy), broad_first
    absorbing_first = (
        "try:\n    from core.training import trainer as t\n"
        "except ImportError:\n    t = None\n"
        "except Exception:\n    raise\n"
    )
    assert not _is_offender(absorbing_first, heavy), absorbing_first
    defines_a_raiser = (
        "try:\n    from core.training import trainer as t\n"
        "except ImportError:\n"
        "    def _later():\n        raise RuntimeError('no trainer')\n"
        "    t = _later\n"
    )
    assert not _is_offender(defines_a_raiser, heavy), defines_a_raiser

    skipped = (
        "import pytest\n"
        "try:\n    from core.training import trainer as t\n"
        "except ImportError:\n    pytest.skip('no trainer', allow_module_level = True)\n"
    )
    assert not _is_offender(skipped, heavy), skipped

    early_return = (
        "import sys\n"
        "def _setup():\n"
        "    if disabled:\n        return\n"
        '    sys.modules["unsloth"] = object()\n'
        "_setup()\n"
        "from core.training import trainer as t\n"
    )
    assert _is_offender(early_return, heavy), early_return
    available_return = (
        "import sys\n"
        "def _setup():\n"
        '    if "unsloth" in sys.modules:\n        return\n'
        '    sys.modules["unsloth"] = object()\n'
        "_setup()\n"
        "from core.training import trainer as t\n"
    )
    assert not _is_offender(available_return, heavy), available_return

    for not_installing in (
        "import sys\n"
        "def _remove_stub(name):\n    sys.modules.pop(name, None)\n"
        '_remove_stub("unsloth")\n'
        "from core.training import trainer as t\n",
        "import sys\n"
        "def _validate_stub(name):\n    assert name\n"
        '_validate_stub("unsloth")\n'
        "from core.training import trainer as t\n",
    ):
        assert _is_offender(not_installing, heavy), not_installing

    while_else = (
        "while enabled():\n"
        "    break\n"
        "else:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(while_else, heavy), while_else

    unrelated_key = (
        "import sys\n"
        'if "pytest" not in sys.modules:\n'
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(unrelated_key, heavy), unrelated_key
    for key in ('"unsloth"', '"core.training.trainer"'):
        source = (
            "import sys\n"
            f"if {key} not in sys.modules:\n"
            '    _stub_if_missing("unsloth", ())\n'
            "from core.training import trainer as t\n"
        )
        assert not _is_offender(source, heavy), source

    variable_key = (
        "import sys\n"
        '_KEY = "pytest"\n'
        "if _KEY not in sys.modules:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(variable_key, heavy), variable_key

    unawaited = (
        "import sys\n"
        "async def _setup():\n"
        '    sys.modules["unsloth"] = object()\n'
        "_setup()\n"
        "from core.training import trainer as t\n"
    )
    assert _is_offender(unawaited, heavy), unawaited
    awaited_equivalent = unawaited.replace("async def _setup", "def _setup")
    assert not _is_offender(awaited_equivalent, heavy), awaited_equivalent

    for not_guarded in (
        "try:\n    from core.training import trainer as t\nexcept MyImportError:\n    t = None\n",
        "try:\n    from core.training import trainer as t\nexcept ExceptionGroup:\n    t = None\n",
    ):
        assert _is_offender(not_guarded, heavy), not_guarded
    for guarded in (
        "try:\n    from core.training import trainer as t\nexcept ImportError:\n    t = None\n",
        "try:\n    from core.training import trainer as t\n"
        "except (ValueError, ImportError):\n    t = None\n",
        "import builtins\ntry:\n    from core.training import trainer as t\n"
        "except builtins.ImportError:\n    t = None\n",
    ):
        assert not _is_offender(guarded, heavy), guarded

    too_late = 'from core.training import trainer as t\n_stub_if_missing("unsloth", ())\n'
    assert not _stubs_before(_parse(too_late), _first_heavy_import_line(_parse(too_late), heavy))

    lazy = "def test_x():\n    from core.training.trainer import UnslothTrainer\n"
    assert _first_heavy_import_line(ast.parse(lazy), heavy) is None


def test_only_an_installed_stub_counts_as_stubbing():
    """Text that mentions unsloth and stubs without installing one must still count as unstubbed."""
    heavy = _heavy_backend_modules()

    for source in (
        '"""Stubs unsloth before importing, or it would need sys.modules surgery."""\n'
        "from core.training import trainer as t\n",
        "import sys\n"
        '_STUBS = {"unsloth": ()}\n'
        "def _stub_if_missing(name, attrs):\n"
        "    sys.modules[name] = attrs\n"
        "from core.training import trainer as t\n",
        "import sys\n"
        '_STUBS = {"unsloth": ()}\n'
        "def fixture():\n"
        "    for _n, _a in _STUBS.items():\n"
        "        _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
        'import sys\nassert "unsloth" not in sys.modules\n'
        "from core.training import trainer as t\n",
        "import sys\n"
        "def _require_absent():\n"
        '    assert "unsloth" not in sys.modules\n'
        "_require_absent()\n"
        "from core.training import trainer as t\n",
        'import sys\n_HEAVY = "unsloth"\nsys.modules["fake_backend"] = object()\n'
        "from core.training import trainer as t\n",
        '_stub_if_missing("trl", ())\n_HEAVY = ("unsloth",)\n'
        "from core.training import trainer as t\n",
    ):
        assert not _stubs_before(_parse(source), 1), source
        assert _is_offender(source, heavy), source
