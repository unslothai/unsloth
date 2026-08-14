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

    The two halves are looked for across the whole prefix rather than within one statement,
    because the names routinely sit in a module-level table that a later loop feeds to the
    helper (``test_training_progress_callback.py``).

    Order is the whole point: a stub registered afterwards lands after the real import has
    already been attempted and raised.
    """
    if line is None:
        return True
    nodes: list[ast.AST] = []
    for statement in tree.body:
        if statement.lineno >= line:
            break
        nodes.extend(_runtime_nodes(statement))
    return _names_required_stub(nodes) and _installs_stub(nodes)


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
    ):
        assert not _stubs_before(_parse(source), 1), source
        assert _is_offender(source, heavy), source
