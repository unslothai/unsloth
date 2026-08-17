# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No test module in this tree may import a backend module that needs unsloth without stubbing first.

``core/training/trainer.py`` and ``core/inference/inference.py`` import ``unsloth`` (and through
it ``unsloth_zoo``) and ``trl`` at module scope. The ``pytest`` matrix in
``.github/workflows/studio-backend-ci.yml`` installs studio.txt plus torch and transformers and
deliberately stops there: the ``repo-cpu-tests`` job beside it is the one that installs
``unsloth_zoo``, and it runs the REPO-ROOT ``tests/`` tree, not this one.

An unstubbed module fails COLLECTION, which takes the entire job down on all four Python
versions, as ``test_trainer_stdout_quiet.py`` and then ``test_audio_type_inconclusive.py`` did.

The earlier version of this guard hardcoded ``core.training.trainer``, so a test reaching the
same ``import unsloth`` through any other backend module was invisible to it. The set is now
derived from the backend sources: every module importing a heavy package at module scope, closed
transitively over the backend's own module-scope imports. Source check rather than runtime,
because where the real packages ARE installed the import succeeds and proves nothing.
"""

from __future__ import annotations

import ast
import warnings
from functools import lru_cache
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND = _TESTS_DIR.parent

# Top-level packages the backend pytest job does not install. `unsloth` is the one that raises
# (its _gpu_init insists on unsloth_zoo); the other two are unimportable there for the same reason.
_HEAVY_PACKAGES = ("unsloth", "unsloth_zoo", "trl")

# Outside the backend's own import graph (unsloth_compiled_cache is a gitignored artifact dir).
_SKIP_TOP_LEVEL = frozenset({"tests", "vendor", "unsloth_compiled_cache"})

# Naming `unsloth` is enough to prove intent: a module that stubs it and forgets `trl` fails
# loudly at collection, whereas one that stubs nothing is the silent case this guard catches.
_REQUIRED_STUB = "unsloth"


def _parse(source: str) -> ast.Module:
    """``ast.parse`` without re-reporting warnings the file's own import already emits.

    Every test module is parsed here, and a few carry an invalid escape sequence in a docstring,
    which would otherwise add a SyntaxWarning per run that belongs to those files, not to this
    guard.
    """
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
    """Absolute dotted names imported at module scope only.

    An import inside a function or a ``try`` is already lazy or guarded and cannot break
    collection, so only ``tree.body`` is walked. Both the module and the module.attr form of a
    ``from X import Y`` are recorded, since either can be the one naming the module.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative: resolve against the containing package
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
    """Backend modules unimportable unless unsloth/unsloth_zoo/trl are installed.

    Seeded with the direct module-scope importers, then closed over the backend's own
    module-scope imports so a module that merely re-exports one is caught too.
    """
    imports: dict[str, set[str]] = {}
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND)
        if rel.parts[0] in _SKIP_TOP_LEVEL:
            continue
        try:
            tree = _parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # not this guard's job to report
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


# What an `except` clause has to name for an unstubbed import under it to be a
# deliberate guard rather than an omission. ModuleNotFoundError is a SUBCLASS of
# ImportError, so it does not belong here: it catches strictly less, and a stub whose
# absence raises plain ImportError would go through it.
_CATCHES_IMPORT_ERROR = frozenset({"ImportError", "Exception", "BaseException"})


def _catches_import_error(node: ast.AST) -> bool:
    """Whether this ``except`` clause's type catches a plain ``ImportError``.

    Compared as whole dotted names, and only the last component of one, so
    ``builtins.ImportError`` counts and ``MyImportError`` does not.
    """
    if isinstance(node, ast.Tuple):
        return any(_catches_import_error(element) for element in node.elts)
    if isinstance(node, ast.Name):
        return node.id in _CATCHES_IMPORT_ERROR
    if isinstance(node, ast.Attribute):
        return node.attr in _CATCHES_IMPORT_ERROR
    return False


def _guarded_by_import_error(tree: ast.Module) -> set[int]:
    """Ids of nodes inside a ``try`` whose handler catches ``ImportError``.

    That is a deliberate guard, not an omission: the file either skips at module level
    or falls back, so an unstubbed import there cannot take collection down.
    ``test_chat_eos_template_refresh.py`` and ``test_generation_timing.py`` are both
    written that way.

    The exception names are compared whole rather than searched for as substrings.
    ``except MyImportError:`` and ``except ExceptionGroup:`` both contain one of the
    names and catch neither a plain ``ImportError`` nor anything above it, so a
    substring test exempted a ``try`` that does not in fact guard the import, and the
    collection it kills is the one this guard exists to report. Reported on this PR.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches = any(
            handler.type is None or _catches_import_error(handler.type) for handler in node.handlers
        )
        if catches:
            for statement in node.body:
                guarded.update(id(child) for child in ast.walk(statement))
    return guarded


def _first_heavy_import_line(tree: ast.Module, heavy: frozenset[str]) -> int | None:
    """Line of the first import-time import of a heavy backend module, or None.

    Not just the direct children of the module body. An import inside a module-scope
    ``with`` or ``if`` runs at import time exactly like a top-level one, and reading
    only ``tree.body`` made it invisible to this guard -- a file that wrote
    ``with something(): from core.training.trainer import X`` without stubbing would
    take collection down unseen.

    Reachability-aware, so ``if TYPE_CHECKING:`` and ``if False:`` are not read as
    import-time dependencies: an import there never executes, and reporting it would
    make a file with a legitimate type-only import fail this guard until someone added
    a stub it does not need. Reported on this PR, against the first version of this
    function, which used the plain import-time traversal.

    A ``try`` that catches ``ImportError`` is exempt, since that is a deliberate guard
    rather than an omission.
    """
    guarded = _guarded_by_import_error(tree)
    for statement in tree.body:
        for node in _reachable_import_time_nodes(statement):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in guarded:
                continue
            if _module_scope_imports(ast.Module(body = [node], type_ignores = []), "") & heavy:
                return node.lineno
    return None


def _runtime_nodes(node: ast.AST):
    """``node`` and every descendant that runs when the module is imported.

    Bodies of ``def``/``class`` are not walked into: a stub call in a helper that nothing
    calls before the import installs nothing, and the ``def _stub_if_missing`` block itself
    would otherwise read as its own proof.

    Branches that provably do not run are not walked into either, for the same reason from
    the other direction: a stub installed under ``if False:`` installs nothing, so counting
    it would report a file safe that still raises.
    """
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
    """``sys.modules[...] = ...`` or a call that adds to it, rather than a read of it.

    ``sys.modules.get(...)`` and ``"unsloth" in sys.modules`` are how a file CHECKS for the
    real package, which is the opposite of installing a stub.
    """
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


def _installs_stub(nodes: list[ast.AST]) -> bool:
    """Whether the nodes call a stub helper or write ``sys.modules`` themselves."""
    calls_stub = any(
        isinstance(node, ast.Call) and "stub" in _callee_name(node).lower() for node in nodes
    )
    return calls_stub or _writes_sys_modules(nodes)


def _stub_helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level ``def``s by name, so a call to one can be read through to its body."""
    return {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _helper_installs_stub(
    name: str,
    helpers: dict[str, ast.AST],
    named: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Whether calling ``name`` installs a stub for the required module.

    ``_runtime_nodes`` deliberately stops at every ``def``, which is right for "does this
    statement install a stub" and wrong for "does calling this helper install one". Two
    files hold their stubs with ``with _stubbed():`` and import the heavy module inside
    that block, so the installing code is one call away from module scope and the
    statement itself spells neither ``unsloth`` nor ``sys.modules``. Read through the
    call, the same way the stub-record check already reads through the helper that
    appends to the record.

    Recursive with a ``seen`` set, since the helper that names the module and the helper
    that writes ``sys.modules`` are usually not the same one, and a cycle would otherwise
    not terminate.
    """
    helper = helpers.get(name)
    if helper is None or name in seen:
        return False
    nodes = list(ast.walk(helper))
    if not (_names_required_stub(nodes) or _reads_a_named_stub(nodes, named)):
        return False
    if _installs_stub(nodes):
        return True
    return any(
        _helper_installs_stub(_callee_name(node), helpers, named, seen | {name})
        for node in nodes
        if isinstance(node, ast.Call)
    )


_BLOCK_FIELDS = ("body", "orelse", "finalbody")


def _end_line(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or node.lineno


def _tests_sys_modules(node: ast.AST, flags: frozenset[str]) -> bool:
    """Whether this ``if`` test asks whether something is already in ``sys.modules``.

    ``test_training_progress_callback.py`` installs its stubs under
    ``if not _TRAINER_PRE_IMPORTED:``, where the flag is
    ``"core.training.trainer" in sys.modules``. Skipping the stubs on the other branch
    is not an omission: the import resolves out of ``sys.modules`` there and never
    reaches the real dependency, which is the whole reason the file is written that
    way. Read through a module-level flag as well as the direct form, since that is
    how it is spelled.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
        ):
            if any(_is_sys_modules(comparator) for comparator in child.comparators):
                return True
        if isinstance(child, ast.Name) and child.id in flags:
            return True
    return False


def _sys_modules_flags(tree: ast.Module) -> frozenset[str]:
    """Module-level names bound to an "is it already imported" test."""
    return frozenset(
        name
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and statement.value is not None
        and _tests_sys_modules(statement.value, frozenset())
        for name in _bound_names(statement)
    )


def _certain_nodes(node: ast.AST, flags: frozenset[str] = frozenset()):
    """``_runtime_nodes``, minus what only MIGHT run.

    A stub inside ``if something():`` above the import is not a stub the import can
    rely on, and reading it as one meant an optional stub counted as a guaranteed
    one. ``while`` and ``except`` bodies go the same way. ``for`` bodies stay: the
    loop over a table is the idiom these files use, and an empty table would not
    name ``unsloth`` in the first place, so it cannot pass this check anyway. The
    "already imported" guard stays too, for the reason in ``_tests_sys_modules``.
    """
    if (
        isinstance(node, ast.If)
        and _constant_test(node) is None
        and not _tests_sys_modules(node.test, flags)
    ):
        return
    if isinstance(node, ast.While):
        for child in node.orelse:
            yield from _certain_nodes(child, flags)
        return
    if isinstance(node, ast.Try):
        for field in ("body", "finalbody"):
            for child in getattr(node, field):
                yield from _certain_nodes(child, flags)
        return
    yield from _runtime_nodes(node)


def _running_before(
    body: list[ast.stmt],
    line: int,
    flags: frozenset[str] = frozenset(),
):
    """``(statement, nodes)`` for everything that has run once ``line`` is reached.

    Line order alone merges branches that exclude each other:

        if enabled:
            _stub_if_missing("unsloth", ())
        else:
            import core.inference.inference

    puts the stub above the import while the two can never both run, and the guard
    called that file stubbed. Reported on this PR. What is walked instead is the
    import's own chain: at each level, only the block that CONTAINS the line is
    descended into, and only the statements above the line inside it. Everything
    there did run, because the import running means its branch was taken.
    """
    for statement in body:
        if statement.lineno >= line:
            return
        if _end_line(statement) < line:
            yield statement, list(_certain_nodes(statement, flags))
            continue
        # This statement encloses the line. Its header ran; its blocks did not,
        # except for the one holding the line.
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
    """Whether a stub naming ``unsloth`` is INSTALLED at module scope before ``line``.

    Structural, not textual. The text form of this check ("the word stub or sys.modules
    appears above the import, and so does the word unsloth") is satisfied by a docstring that
    merely discusses stubbing, and by a helper that is defined but never called, so a module
    could lose its stubs and stay green. What is read here is the code that actually RUNS
    before the import: the module-scope statements above it, minus the ``def``/``class`` bodies
    that only run when something calls them. It counts when those name ``unsloth`` as a string
    and either call a stub helper or write ``sys.modules``.

    The name and the operation have to meet in ONE statement, or the guard goes green on a
    file that stubs something else while ``unsloth`` merely appears above
    (``sys.modules["fake"] = ...`` next to an unrelated ``"unsloth"`` string). The name is
    allowed to arrive through a module-level table the statement reads, because that is how
    the real files are written (``test_training_progress_callback.py`` keeps ``_STUBS`` above
    the loop that feeds it to the helper), so a table naming ``unsloth`` marks the names it
    binds and a later statement reading one of them counts as naming it.

    Order is the whole point: a stub registered afterwards lands after the real import has
    already been attempted and raised.
    """
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
        if _installs_stub(nodes):
            return True
        named |= _bound_names(statement)
    return False


def _is_offender(source: str, heavy: frozenset[str]) -> bool:
    """Whether ``source`` imports a heavy backend module at module scope unstubbed.

    Every candidate is parsed. A textual prefilter on the dotted module names looks like a
    cheap skip but is wrong: ``from core.training import trainer`` never spells the contiguous
    string ``core.training.trainer``, so the file it was meant to skip is the collection-killing
    one, and the guard reported no offender while the job died.
    """
    try:
        tree = _parse(source)
    except SyntaxError:  # not this guard's job to report
        return False
    return not _stubs_before(tree, _first_heavy_import_line(tree, heavy))


def _offenders() -> list[str]:
    heavy = _heavy_backend_modules()
    return [
        path.name
        for path in sorted(_TESTS_DIR.glob("test_*.py"))
        if _is_offender(path.read_text(encoding = "utf-8"), heavy)
    ]


def _import_time_nodes(node: ast.AST):
    """``node`` and every descendant evaluated while the module is being imported.

    Wider than ``_runtime_nodes``, which stops at every ``def``/``class``. That is
    right for "did a stub get installed", but wrong for "when does this call run":
    a class BODY executes at import time, and so do decorators, default values and
    annotations on a ``def``. Only a function body is deferred.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        # Lambda.body is one expression; a def's is a list. Compared by identity so a
        # node is never tested with `in` against another AST node.
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
    """Which branch of this ``if`` the interpreter always takes, or None if it depends.

    ``if TYPE_CHECKING:`` and ``if False:`` are the two that matter most: an import
    under either never executes, so it cannot be what left the target in
    ``sys.modules``. ``if True:`` matters the same way from the other side, since its
    ``else:`` never executes. Nothing else is guessed -- a test this cannot evaluate
    returns None and both branches are walked.
    """
    test = node.test
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return False
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return False
    return None


def _reachable_import_time_nodes(node: ast.AST):
    """``_import_time_nodes``, minus the branches that provably do not run.

    Two corrections, both reported on this PR. Import time rather than runtime,
    because a class body, a decorator, a default and an annotation all execute while
    the module is being imported, so an eager import written in one of them DOES cache
    the target -- ``_runtime_nodes`` stopped at every def and class and reported no
    eager import for a file that was in fact safe. And reachable, because
    ``_runtime_nodes`` walked into ``if TYPE_CHECKING:`` and ``if False:``, where an
    import never runs, and reported a file safe that still raises.
    """
    if isinstance(node, ast.If) and _constant_test(node) is not None:
        # Only the branch the interpreter takes. Pruning the body of `if False:` but
        # descending into the `else:` of `if True:` left the second half of the same
        # hole open: an import there never runs either. Reported on this PR.
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
    """Whether the module imports ``target`` under live stubs, at module scope, before ``line``.

    A file may install the stubs, import the heavy module under them, then drop the
    stubs -- which is the shape this PR's fix uses and the shape the sibling files
    use. A later ``importorskip`` then resolves out of ``sys.modules`` and never
    touches the real dependency. Without this, a copy of that file that forgot the
    eager import would read as stubbed and still raise at test time.

    Bounded by ``line`` for the same reason ``_stubs_before`` is: an eager import
    BELOW an import-time ``importorskip`` has not run when that call is evaluated.

    Bounded BELOW by the stub install as well, because what makes the eager import
    leave the target in ``sys.modules`` is the stubs being live for it. An import
    attempted before them is the ``try: import X except ImportError: pass`` probe,
    which on the dependency-light matrix fails and leaves nothing cached -- and
    Python removes the half-initialised module on the way out, so the later call
    still reaches the real dependency. Counting that probe reported such a file
    safe. Reported on this PR.
    """
    for statement in tree.body:
        if statement.lineno >= line:
            break
        # Asked through _stubs_before rather than restated here, so "are the stubs
        # live" has one answer in this file and cannot drift between the two callers.
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
    """The module name argument, positional or as the ``modname`` keyword.

    ``importorskip(modname, minversion=None, reason=None, *, exc_type=None)``, so
    ``pytest.importorskip(modname = "core.inference.inference")`` is a valid call
    with an empty ``node.args``. Matching only the positional form let a file
    written that way walk past this guard. Reported on this PR.
    """
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "modname":
            return keyword.value
    return None


def _skips_on_plain_import_error(node: ast.Call) -> bool:
    """Whether the call asked pytest to treat a plain ``ImportError`` as a skip.

    ``exc_type`` arrived in pytest 8.2 and is documented as "the exception that
    should be captured in order to skip modules. Must be ImportError or a
    subclass", defaulting to ``ModuleNotFoundError``. That default is the whole
    reason this guard exists: an unstubbed heavy module raises a plain
    ``ImportError``, which the default does not catch, so the call raises instead
    of skipping. A call that passes ``ImportError`` explicitly has opted into the
    broad behaviour and is safe unstubbed, so flagging it would be a false report.
    ``ModuleNotFoundError`` passed explicitly is the default and stays flagged.
    Reported on this PR.
    """
    for keyword in node.keywords:
        if keyword.arg != "exc_type":
            continue
        value = keyword.value
        if isinstance(value, ast.Name):
            return value.id == "ImportError"
        if isinstance(value, ast.Attribute):
            return value.attr == "ImportError"
    return False


def _module_functions(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level ``def``s by name."""
    return {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _functions_called_at_import(tree: ast.Module) -> dict[str, int]:
    """Module-level ``def`` name -> the earliest line import time can reach it from.

    A ``def`` is only deferred while nothing runs it. A helper the module body calls
    executes during collection, so an ``importorskip`` inside it runs then too, and
    giving it the end-of-module boundary let a stub installed BELOW the call site read
    as being in place.

    Followed transitively, to a fixed point. An earlier version stopped after one level
    and called that conservative; it is not. Stopping hands the inner helper the
    end-of-module boundary, which is the LENIENT answer -- a stub installed anywhere in
    the file then counts as in time, while collection has already run the inner import
    and failed. So a helper reached only through another helper inherits the outer call
    site's line, which is when it actually runs. Reported on this PR.
    """
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
            # Only the calls that actually run when the outer helper runs. ast.walk
            # would also visit one under `if False:` and one inside a nested def that
            # nothing calls, and handing those the outer boundary fails a file Python
            # never executes that way. Reported on this PR. Same reachability rule as
            # the module body uses, so the two cannot answer differently.
            for statement in functions[caller].body:
                for node in _reachable_import_time_nodes(statement):
                    if not isinstance(node, ast.Call):
                        continue
                    callee = _callee_name(node)
                    # The inner helper runs when the OUTER one is called, so it
                    # inherits that line rather than its own, which is only where it
                    # is written.
                    if callee in functions and line < first.get(callee, line + 1):
                        first[callee] = line
                        changed = True
    return first


def _reachable_nodes(node: ast.AST):
    """``node`` and every descendant on a path the interpreter can take.

    Unlike ``_reachable_import_time_nodes`` this does NOT stop at a ``def``: a call in
    a function body runs when the test runs. What it does share is the pruning, so a
    call under ``if False:`` or ``if TYPE_CHECKING:`` is not reported at all, wherever
    it is written.
    """
    if isinstance(node, ast.If) and _constant_test(node) is not None:
        for child in node.body if _constant_test(node) else node.orelse:
            yield from _reachable_nodes(child)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _reachable_nodes(child)


def _importorskip_bare_names(tree: ast.Module) -> frozenset[str]:
    """Bare names bound to ``pytest.importorskip``, including aliases.

    ``from pytest import importorskip as ios`` then ``ios(...)`` is a valid call, and
    matching the callee against the literal string missed it, so an unstubbed module
    written that way walked past the guard. Reported on this PR. The attribute form is
    still matched on the attribute name, so ``pytest.importorskip`` needs no binding.
    """
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
    """``(target, line the stub must be installed before)`` per heavy ``importorskip``.

    Two forms count. ``pytest.importorskip(...)`` is the attribute one, and
    ``from pytest import importorskip`` then a bare ``importorskip(...)`` is equally
    supported by pytest, so matching only the attribute form leaves the bare one
    invisible to this guard.

    The boundary differs by scope, and that is the point:

    - **inside a def**: the call runs at test time, so a stub installed anywhere in
      the module body is in place by then. Boundary is the end of the module.
    - **at module scope**: the call runs during collection, so a stub installed
      BELOW it lands too late and the import has already raised. Boundary is the
      call's own line. Scanning to the end of the module here would see that later
      stub and call the file safe while it still breaks collection.
    """
    bare_names = _importorskip_bare_names(tree)
    # Reachability-aware on both counts. A call under `if False:` or `if TYPE_CHECKING:`
    # never runs, so it is neither an import-time call nor a call at all, and reporting
    # it failed a file that type-checks or deliberately disables an import. Reported on
    # this PR.
    module_scope = {
        id(node) for statement in tree.body for node in _reachable_import_time_nodes(statement)
    }
    reachable = {id(node) for node in _reachable_nodes(tree)}
    end = max((statement.lineno for statement in tree.body), default = 0) + 1
    # A call inside a def is deferred only if nothing runs that def during import. Where
    # the module body calls it, the body runs at collection like any other import-time
    # statement, and the end-of-module boundary would let a stub installed BELOW the call
    # count. Those calls take the line the helper is invoked from. Reported on this PR.
    called_at_import = _functions_called_at_import(tree)
    # The function a call belongs to is the INNERMOST one. ast.walk descends into a
    # nested def, which attributed a call there to the enclosing module-level function
    # and handed it that function's import-time boundary, while a nested body cannot run
    # until something calls it -- by which time a stub installed below is in place. So
    # this defers nested bodies, and a call inside one falls through to the end-of-module
    # boundary, which is what deferred means. Reported on this PR.
    in_function = {
        id(node): name
        for name, function in _module_functions(tree).items()
        for statement in function.body
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
    """Module-level names the stub helper records installed stubs into.

    The real files pop through one (``for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)``), so a drop has to be recognisable through it.
    The link is followed rather than guessed: a module-scope call that names the
    required stub identifies the helper, and whatever module-level name that
    helper appends to is the record.

    An earlier version instead accumulated EVERY module-level assignment target,
    which made an unrelated cleanup list read as the stub record and got properly
    stubbed files reported as offenders. Reported on this PR.
    """
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
    """Whether the module removes its stubs again, at module scope, before ``line``.

    The convention is ``sys.modules.pop(name, None)`` over the recorded list, so that
    the stubs do not outlive the module. A module that never drops them can reach a
    heavy module and still resolve, because the stub is still installed when the call
    runs; one that drops them first cannot, unless it imported the target first.

    Bounded by ``line`` because a pop below an import-time call has not happened yet
    when that call runs, and a pop above one has.

    It has to be OUR stubs. The receiver is checked to be ``sys.modules``, not any
    object with a ``.modules``, and the statement has to name the required stub the
    same way an install does -- as the string, or through a module-level list it
    reads, which is how the real files spell it (``for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)``). Before that, an unrelated
    ``sys.modules.pop("routes.foo", None)`` in a properly stubbed file read as a
    drop and got the file reported as an offender. Reported on this PR.
    """
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
    """Whether this module reaches a heavy module through importorskip unsafely.

    One entry point, used by the guard and by the test that pins it, so the pinned
    answers cannot drift away from the answers the guard actually gives.
    """
    for target, boundary in _importorskip_calls(tree, heavy):
        if not _stubs_before(tree, boundary):
            return True
        # Installing the stub is not enough if the module has already dropped it again
        # by the time the call runs. What makes the call resolve then is the module
        # having imported the target itself while the stubs were live. Both questions
        # are asked against the same boundary as the install, so an import-time call is
        # judged on the pops above IT (a call between the install and the pop is fine,
        # one below the pop is not) and a lazy call on the whole module body. Judging
        # only lazy calls here let a module install, pop, then call at import time and
        # still read as safe, while it raises during collection.
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
    """The lazy-import exemption above has a hole, and this closes it.

    The guard beside this one only looks at module-scope imports, because those fail
    COLLECTION and take the whole job down. An import inside a test body is deliberately
    exempt: it is lazy, and a lazy import of an uninstallable module is normally written to
    degrade into a skip.

    ``pytest.importorskip`` is that idiom, and since pytest 8.2 it no longer covers this
    case: it skips on ``ModuleNotFoundError`` only, and ``unsloth/_gpu_init.py`` raises a
    plain ``ImportError`` when unsloth_zoo is absent, which is exactly the backend matrix's
    situation. So the call raises instead of skipping and the test fails.

    test_safetensors_reasoning_stream.py sat in that hole. It never stubbed anything, and
    passed only when an earlier file in the session had installed the stub and left the
    imported module in ``sys.modules`` for it, which makes it a pass that depends on
    collection order.
    """
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

    # `from pytest import importorskip` then a bare call: the same offence, and the
    # attribute-only match this guard shipped with could not see it at all.
    bare_unstubbed = (
        "from pytest import importorskip\n"
        "def test_x():\n"
        "    inf = importorskip('core.inference.inference')\n"
    )
    assert calls(bare_unstubbed) == [("core.inference.inference", 3)]  # end-of-module boundary
    assert not safe(bare_unstubbed)

    # A module-scope call runs during collection, so a stub BELOW it is too late.
    # Scanning to the end of the module would call this safe.
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

    # A class body runs at import time, so a stub below it is too late, exactly like a
    # module-scope call. Only a function BODY is deferred.
    class_body_late = (
        "import pytest, sys\n"
        "class T:\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(class_body_late) == [("core.inference.inference", 3)]
    assert not safe(class_body_late)

    # So do default expressions on a def.
    default_arg_late = (
        "import pytest, sys\n"
        "def f(x = pytest.importorskip('core.inference.inference')):\n"
        "    pass\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(default_arg_late) == [("core.inference.inference", 2)]
    assert not safe(default_arg_late)

    # A lazy call in a module that stubs, imports the target, then DROPS the stubs is
    # fine, because the module is in sys.modules by then.
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

    # The same file without the eager import is NOT fine: the stubs are gone by the
    # time the lazy call runs, so it raises. This is the shape a copy-paste drops.
    stub_drop_no_import = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "def test_x():\n"
        "    inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert not _offender_free(stub_drop_no_import)

    # The same trap at IMPORT time, which the drop check used to skip entirely: stub,
    # pop, then call at module scope. The stub is installed above the call, so the
    # install check is satisfied, but it is gone again by the time the call runs and
    # collection dies. Reported on this PR.
    drop_then_import_time_call = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert calls(drop_then_import_time_call) == [("core.inference.inference", 5)]
    assert not _offender_free(drop_then_import_time_call)

    # And its safe sibling: the pop comes AFTER the call, so the stub is live for it.
    # Bounding the drop check by the call's line is what separates these two.
    import_time_call_then_drop = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
    )
    assert _offender_free(import_time_call_then_drop)

    # Eager import above the pop, import-time call below it: also safe, because the
    # target is in sys.modules by then. The eager-import check has to be bounded by
    # the call's line too, or the mirror image of this file would read as safe.
    eager_then_drop_then_call = (
        "import pytest, sys\n"
        "_stub_if_missing('unsloth', ())\n"
        "import core.inference.inference\n"
        "for _n in ('unsloth',):\n"
        "    sys.modules.pop(_n, None)\n"
        "inf = pytest.importorskip('core.inference.inference')\n"
    )
    assert _offender_free(eager_then_drop_then_call)

    # A try/except probe BEFORE the stubs is not an eager import: on the matrix it
    # fails, and Python drops the half-initialised module, so the later call still
    # reaches the real dependency. Counting it read this file as safe. Reported here.
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

    # Unrelated module-cache cleanup in a properly stubbed file is not a stub drop,
    # and neither is a pop on something that merely has a `.modules`. Both used to
    # report the file as an offender. Reported here.
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
    # The real files pop through a module-level list the stub helper records into,
    # so that still reads as a drop. The link is followed through the helper body,
    # which is why the helper is spelled out here as the real files spell it.
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

    # But an UNRELATED module-level list is not the stub record, even when it is
    # popped from sys.modules. Accumulating every assignment target made this read
    # as a stub drop and reported a properly stubbed file. Reported on this PR.
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

    # importorskip(modname = ...) is the documented signature, so matching only the
    # positional form let a file written that way past the guard. Reported here.
    keyword_target = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(modname = 'core.inference.inference')\n"
    )
    assert calls(keyword_target) == [("core.inference.inference", 3)]  # end-of-module
    assert not safe(keyword_target)

    # exc_type=ImportError is pytest's own opt-in to skipping on a plain ImportError,
    # which is the exact failure this guard is about, so such a call is safe unstubbed
    # and flagging it was a false report. Reported here.
    explicit_import_error = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(\n"
        "        'core.inference.inference', exc_type = ImportError\n"
        "    )\n"
    )
    assert calls(explicit_import_error) == []
    assert safe(explicit_import_error)
    # ModuleNotFoundError is the default, so spelling it out changes nothing.
    explicit_default = (
        "import pytest\n"
        "def test_x():\n"
        "    inf = pytest.importorskip(\n"
        "        'core.inference.inference', exc_type = ModuleNotFoundError\n"
        "    )\n"
    )
    assert calls(explicit_default) and not safe(explicit_default)

    # An import under `if TYPE_CHECKING:` never runs, so it is not what cached the
    # target, and the file still raises after the stubs are dropped. Reported here.
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

    # But a class body DOES run at import time, so an eager import written there
    # caches the target and the file is safe. Stopping at every class reported it as
    # an offender. Reported here.
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

    # An importorskip inside a helper the MODULE BODY calls runs at collection, so a
    # stub installed below that call site is too late. The end-of-module boundary said
    # it was in time. Reported here.
    helper_called_at_import = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "_probe()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert calls(helper_called_at_import) == [("core.inference.inference", 4)]
    assert not safe(helper_called_at_import)

    # And through a second helper: the module calls the outer one, the inner one holds
    # the importorskip, and the stub arrives after. Stopping at one level handed the
    # inner call the end-of-module boundary and read the later stub as in time, while
    # collection has already run the inner import. Reported on this PR.
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

    # A call the outer helper never makes does not propagate: neither one under a
    # constant-false test, nor one inside a nested def that nothing invokes. Handing
    # those the outer boundary failed a file Python never executes that way. Reported
    # on this PR.
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

    # A chain nothing calls at import time still keeps the deferred boundary.
    nested_helper_never_called = (
        "import pytest, sys\n"
        "def _inner():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "def _outer():\n"
        "    return _inner()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(nested_helper_never_called)

    # The same helper NOT called at import time keeps the deferred boundary.
    helper_never_called = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(helper_never_called)

    # `from pytest import importorskip as ios` is a valid call the raw-name match could
    # not see, so an unstubbed module written that way walked past the guard. Reported here.
    aliased = (
        "from pytest import importorskip as ios\n"
        "def test_x():\n"
        "    inf = ios('core.inference.inference')\n"
    )
    assert calls(aliased) == [("core.inference.inference", 3)]
    assert not safe(aliased)
    # A name bound to the attribute form counts too.
    rebound = (
        "import pytest\n"
        "_ios = pytest.importorskip\n"
        "def test_x():\n"
        "    inf = _ios('core.inference.inference')\n"
    )
    assert calls(rebound) and not safe(rebound)
    # A bare call to something that is NOT pytest's importorskip is not one.
    unrelated_bare = (
        "from mymod import importorskip\n"
        "def test_x():\n"
        "    inf = importorskip('core.inference.inference')\n"
    )
    assert calls(unrelated_bare) == []

    # The other half of the unreachable-branch hole: an import in the `else:` of a
    # constant-true test never runs either. Reported here.
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
    # And the branch a constant-false test DOES take still counts.
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

    # An importorskip inside a NESTED def does not run when the outer helper is called,
    # so it keeps the deferred boundary and a stub installed after the outer call is in
    # time. Attributing it to the outer function rejected a file that works. Reported on
    # this PR.
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

    # A call under a branch the interpreter never takes is not a call. Reporting it
    # failed a file that only type-checks the import, or deliberately disables it.
    # Reported on this PR.
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
    # And a helper invoked only from such a branch is not called at import time either,
    # so a stub below it is still in time for the lazy call that does run.
    helper_called_only_when_disabled = (
        "import pytest, sys\n"
        "def _probe():\n"
        "    return pytest.importorskip('core.inference.inference')\n"
        "if False:\n"
        "    _probe()\n"
        "_stub_if_missing('unsloth', ())\n"
    )
    assert safe(helper_called_only_when_disabled)

    # importorskip of something harmless is not an offence.
    assert calls("import pytest\ndef test_x():\n    pytest.importorskip('numpy')\n") == []


def test_the_heavy_module_set_is_derived_from_the_backend_sources():
    """A derivation that quietly returned nothing would make the guard below pass on anything."""
    heavy = _heavy_backend_modules()
    assert "core.training.trainer" in heavy
    # The one the hardcoded version of this guard could never have seen.
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
        # Split form: the source never spells "core.training.trainer". Asserted through
        # _is_offender, the same entry point _offenders uses, so no textual prefilter can be
        # reintroduced in front of it.
        "from core.training import trainer as t\n",
        "from core.training.trainer import UnslothTrainer\n",
        "import core.training.trainer\n",
        # The shape the hardcoded guard was blind to.
        "from core.inference.inference import InferenceEngine\n",
        "from core.inference import inference\n",
    ):
        assert _first_heavy_import_line(ast.parse(source), heavy) == 1, source
        assert not _stubs_before(ast.parse(source), 1), source
        assert _is_offender(source, heavy), source

    for stubbed in (
        '_stub_if_missing("unsloth", ())\nfrom core.training import trainer as t\n',
        # The loop form the preflight tests use: the package names sit in the iterable, not in
        # the call, so the whole module-scope statement has to be read, not just the call node.
        'for _n, _a in (("unsloth", ()),):\n'
        "    _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
        # No helper at all, just the assignment the helper would have made.
        'import sys\nsys.modules["unsloth"] = object()\n'
        "from core.training import trainer as t\n",
        # The shape of the real files: the table sits above the loop that feeds it to the
        # helper, so the name reaches the call through ``_STUBS`` rather than in it.
        'import sys\n_STUBS = {"unsloth": ()}\n'
        "for _n, _a in _STUBS.items():\n"
        "    _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
    ):
        assert _stubs_before(
            _parse(stubbed), _first_heavy_import_line(_parse(stubbed), heavy)
        ), stubbed
        assert not _is_offender(stubbed, heavy), stubbed

    # A module-scope `with` or `if` runs at import exactly like a top-level line, so a heavy
    # import inside one is an offence. Reading only the direct children of the module body
    # made this shape invisible to the guard entirely.
    for nested, at in (
        ("with open('x') as fh:\n    from core.training import trainer as t\n", 2),
        ("import os\nif os.environ.get('X'):\n    from core.inference import inference\n", 3),
    ):
        assert _first_heavy_import_line(_parse(nested), heavy) == at, nested
        assert _is_offender(nested, heavy), nested

    # ...and the context manager that HOLDS the stubs is how two of the real files are
    # written, so the installing code sits one call away from module scope and the
    # statement itself spells neither the module name nor sys.modules.
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

    # Same helper, never called: still not stubbed, or reading through the call would have
    # made a defined-and-unused helper into its own proof.
    assert _is_offender(held.replace("with _stubbed():\n    from", "from"), heavy)

    # Order still decides it INSIDE the block. Widening the search into compound
    # statements without narrowing the stub side to match read the whole enclosing
    # statement as preceding the import, so a stub written below it counted.
    inside_after = (
        "if True:\n"
        "    from core.inference import inference\n"
        '    _stub_if_missing("unsloth", ())\n'
    )
    assert _first_heavy_import_line(_parse(inside_after), heavy) == 2, inside_after
    assert _is_offender(inside_after, heavy), inside_after
    # ...and the same block with the two lines the other way round is fine.
    inside_before = (
        "if True:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
    )
    assert not _is_offender(inside_before, heavy), inside_before

    # A branch that never runs is not an import, and not a stub either.
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

    # A stub on a branch the import cannot be on. Line order alone put it above the
    # import while the two exclude each other.
    other_branch = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "else:\n"
        "    from core.inference import inference\n"
    )
    assert _is_offender(other_branch, heavy), other_branch
    # The same shape with the import on the stub's own branch is fine.
    same_branch = (
        "if enabled():\n"
        '    _stub_if_missing("unsloth", ())\n'
        "    from core.inference import inference\n"
    )
    assert not _is_offender(same_branch, heavy), same_branch

    # A stub that only MIGHT have run is not one the import can rely on.
    optional = (
        'if enabled():\n    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _is_offender(optional, heavy), optional

    # ...except the one condition that makes skipping the stubs safe, which is how
    # test_training_progress_callback.py is written: on the branch that skips them the
    # import resolves out of sys.modules and never reaches the real dependency.
    already_imported = (
        "import sys\n"
        '_PRE = "core.training.trainer" in sys.modules\n'
        "if not _PRE:\n"
        '    _stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert not _is_offender(already_imported, heavy), already_imported

    # An except clause is matched by what it CATCHES, not by what its name contains.
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

    # And a stub that lands too late does not count.
    too_late = 'from core.training import trainer as t\n_stub_if_missing("unsloth", ())\n'
    assert not _stubs_before(_parse(too_late), _first_heavy_import_line(_parse(too_late), heavy))

    # An import inside a function is lazy already, so it is not an offence.
    lazy = "def test_x():\n    from core.training.trainer import UnslothTrainer\n"
    assert _first_heavy_import_line(ast.parse(lazy), heavy) is None


def test_only_an_installed_stub_counts_as_stubbing():
    """What the textual form of this check accepted and the structural one does not.

    Each source below reads as stubbed to a substring match over the lines above the import
    (the words ``unsloth`` and ``stub``/``sys.modules`` are all present) while installing
    nothing, so a module could lose its stubs and the guard would stay green.
    """
    heavy = _heavy_backend_modules()

    for source in (
        # Prose about stubbing unsloth, in the module docstring.
        '"""Stubs unsloth before importing, or it would need sys.modules surgery."""\n'
        "from core.training import trainer as t\n",
        # The names in a module-level table and a stub helper that is never called, which is
        # what a file looks like the moment its one call site is dropped.
        "import sys\n"
        '_STUBS = {"unsloth": ()}\n'
        "def _stub_if_missing(name, attrs):\n"
        "    sys.modules[name] = attrs\n"
        "from core.training import trainer as t\n",
        # The same, with the call parked inside a fixture that runs long after collection.
        "import sys\n"
        '_STUBS = {"unsloth": ()}\n'
        "def fixture():\n"
        "    for _n, _a in _STUBS.items():\n"
        "        _stub_if_missing(_n, _a)\n"
        "from core.training import trainer as t\n",
        # Reading sys.modules is not writing it.
        'import sys\nassert "unsloth" not in sys.modules\n'
        "from core.training import trainer as t\n",
        # The same read, one call away: a helper that is called at module scope and names
        # the module without installing anything. Reading through a call has to keep
        # asking what the helper DOES, or every helper mentioning the name would count.
        "import sys\n"
        "def _require_absent():\n"
        '    assert "unsloth" not in sys.modules\n'
        "_require_absent()\n"
        "from core.training import trainer as t\n",
        # A stub of something ELSE, with the required name loose in the file rather than in
        # the operation. Both halves are present, so a prefix-wide pairing reads this as
        # stubbed while unsloth is not stubbed at all.
        'import sys\n_HEAVY = "unsloth"\nsys.modules["fake_backend"] = object()\n'
        "from core.training import trainer as t\n",
        '_stub_if_missing("trl", ())\n_HEAVY = ("unsloth",)\n'
        "from core.training import trainer as t\n",
    ):
        assert not _stubs_before(_parse(source), 1), source
        assert _is_offender(source, heavy), source
