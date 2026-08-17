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


def _first_heavy_import_line(tree: ast.Module, heavy: frozenset[str]) -> int | None:
    """Line of the first module-scope import of a heavy backend module, or None."""
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _module_scope_imports(ast.Module(body = [node], type_ignores = []), "") & heavy:
            return node.lineno
    return None


def _runtime_nodes(node: ast.AST):
    """``node`` and every descendant that runs when the module is imported.

    Bodies of ``def``/``class`` are not walked into: a stub call in a helper that nothing
    calls before the import installs nothing, and the ``def _stub_if_missing`` block itself
    would otherwise read as its own proof.
    """
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
    for statement in tree.body:
        if statement.lineno >= line:
            break
        nodes = list(_runtime_nodes(statement))
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
        for node in _runtime_nodes(statement):
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
    module_scope = {id(node) for statement in tree.body for node in _import_time_nodes(statement)}
    end = max((statement.lineno for statement in tree.body), default = 0) + 1
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            called = func.attr
        elif isinstance(func, ast.Name):
            called = func.id
        else:
            continue
        if called != "importorskip":
            continue
        if _skips_on_plain_import_error(node):
            continue
        target = _importorskip_target(node)
        if (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value in heavy
        ):
            calls.append((target.value, node.lineno if id(node) in module_scope else end))
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
    bound = {
        name
        for statement in tree.body
        for name in _bound_names(statement)
    }
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
        "_STUBBED = []\n"
        + _HELPER
        + "_stub_if_missing('unsloth', ())\n"
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
        "_JUNK = ['routes.foo']\n"
        + _HELPER
        + "_stub_if_missing('unsloth', ())\n"
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
