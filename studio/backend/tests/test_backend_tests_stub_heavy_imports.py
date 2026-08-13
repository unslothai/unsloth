# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No test module in this tree may import a backend module that needs unsloth without stubbing first.

``core/training/trainer.py`` imports ``unsloth`` (and through it ``unsloth_zoo``) and ``trl`` at
module scope, and it is not the only one: ``core/inference/inference.py`` does the same. The
``pytest`` matrix in ``.github/workflows/studio-backend-ci.yml`` installs studio.txt plus torch
and transformers and deliberately stops there. The heavier ``repo-cpu-tests`` job beside it does
install ``unsloth_zoo``, but it runs the REPO-ROOT ``tests/`` tree, not this one, so nothing here
can lean on that.

The consequence is worse than one skipped test: an unstubbed module fails COLLECTION, and a
collection error takes down the entire job on all four Python versions. That happened when
``test_trainer_stdout_quiet.py`` landed, and again with ``test_audio_type_inconclusive.py``,
whose module-scope ``from core.training.trainer import ...`` turned every job in the matrix into
``ImportError: Unsloth: Please install unsloth_zoo`` at collection time.

The earlier version of this guard hardcoded ``core.training.trainer`` as the one import to watch,
so a test that reached the same ``import unsloth`` through any other backend module was invisible
to it. The set is now derived from the backend sources: every module that imports a heavy package
at module scope, plus everything that imports one of those at module scope, transitively. It is a
source check rather than a runtime one because on a box where the real packages ARE installed the
import succeeds and proves nothing.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND = _TESTS_DIR.parent

# Top-level packages the backend pytest job does not install. `unsloth` is the one that raises
# (its _gpu_init insists on unsloth_zoo); the other two are listed because a module that pulls
# them in at module scope is unimportable there for the same reason.
_HEAVY_PACKAGES = ("unsloth", "unsloth_zoo", "trl")

# Not part of the backend's own import graph: the test tree itself, vendored third-party code,
# and unsloth_compiled_cache, which is a gitignored runtime artifact directory.
_SKIP_TOP_LEVEL = frozenset({"tests", "vendor", "unsloth_compiled_cache"})

# What a module must stub before such an import. Naming `unsloth` is enough to prove intent: a
# module that stubs it and forgets `trl` fails loudly at collection on CI, whereas a module that
# stubs nothing is the silent case this guard exists to catch.
_REQUIRED_STUB = "unsloth"


def _module_name(path: Path) -> str:
    rel = path.relative_to(_BACKEND)
    parts = list(rel.parts)
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_scope_imports(tree: ast.Module, package: str) -> set[str]:
    """Absolute dotted names imported at module scope only.

    An import inside a function or a ``try`` is already lazy or already guarded, and neither can
    break collection, so only ``tree.body`` is walked. Both the module and the module.attr form
    of a ``from X import Y`` are recorded, because ``from core.training import trainer`` names a
    module while ``from core.training.trainer import UnslothTrainer`` names an attribute of one.
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
    """Backend modules that cannot be imported unless unsloth/unsloth_zoo/trl are installed.

    Seeded with the modules importing one of those directly at module scope, then closed over
    the backend's own module-scope imports, so a module that merely re-exports one of them is
    caught too.
    """
    imports: dict[str, set[str]] = {}
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND)
        if rel.parts[0] in _SKIP_TOP_LEVEL:
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
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


def _stubs_before(source: str, line: int | None) -> bool:
    """Whether a stub call naming ``unsloth`` appears at module scope BEFORE ``line``.

    Order is the whole point. A stub registered afterwards is registered after the real import
    has already been attempted and raised, so it changes nothing."""
    if line is None:
        return True
    head = "\n".join(source.splitlines()[: line - 1])
    return _REQUIRED_STUB in head and ("stub" in head or "sys.modules" in head)


def _offenders() -> list[str]:
    heavy = _heavy_backend_modules()
    offenders = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        source = path.read_text(encoding = "utf-8")
        if not any(module in source for module in heavy):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # not this guard's job to report
            continue
        if not _stubs_before(source, _first_heavy_import_line(tree, heavy)):
            offenders.append(path.name)
    return offenders


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
        "from core.training import trainer as t\n",
        "from core.training.trainer import UnslothTrainer\n",
        "import core.training.trainer\n",
        # The shape the hardcoded guard was blind to.
        "from core.inference.inference import InferenceEngine\n",
    ):
        assert _first_heavy_import_line(ast.parse(source), heavy) == 1, source
        assert not _stubs_before(source, 1), source

    stubbed = '_stub_if_missing("unsloth", ())\nfrom core.training import trainer as t\n'
    assert _stubs_before(stubbed, _first_heavy_import_line(ast.parse(stubbed), heavy))

    # And a stub that lands too late does not count.
    too_late = 'from core.training import trainer as t\n_stub_if_missing("unsloth", ())\n'
    assert not _stubs_before(too_late, _first_heavy_import_line(ast.parse(too_late), heavy))

    # An import inside a function is lazy already, so it is not an offence.
    lazy = "def test_x():\n    from core.training.trainer import UnslothTrainer\n"
    assert _first_heavy_import_line(ast.parse(lazy), heavy) is None
