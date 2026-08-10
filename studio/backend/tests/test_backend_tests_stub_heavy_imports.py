# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No test module in this tree may import the LLM trainer without stubbing its heavy deps first.

``core/training/trainer.py`` imports ``unsloth`` (and through it ``unsloth_zoo``) and ``trl`` at
module scope. The ``pytest`` matrix in ``.github/workflows/studio-backend-ci.yml`` installs
studio.txt plus torch and transformers and deliberately stops there. The heavier
``repo-cpu-tests`` job beside it does install ``unsloth_zoo``, but it runs the REPO-ROOT
``tests/`` tree, not this one, so nothing here can lean on that.

The consequence is worse than one skipped test: an unstubbed module fails COLLECTION, and a
collection error takes down the entire job on all four Python versions. That is what happened
when ``test_trainer_stdout_quiet.py`` landed, and every open PR went red until it was fixed.

Three modules import the trainer today and each stubs first. This asserts the rule so a fourth
cannot arrive without it, and it is a source check rather than a runtime one because on a box
where the real packages ARE installed the import succeeds and proves nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

# The import that pulls the heavy chain in. Matched on the module path, so
# `from core.training import trainer` and `import core.training.trainer` both count.
_TRAINER_MODULE = "core.training.trainer"
# What a module must stub before that import. Naming `unsloth` is enough to prove intent: a
# module that stubs it and forgets `trl` fails loudly at collection on CI, whereas a module that
# stubs nothing is the silent case this guard exists to catch.
_REQUIRED_STUB = "unsloth"


def _imports_trainer_at_module_scope(tree: ast.Module) -> bool:
    for node in tree.body:  # module scope only: an import inside a test function is already lazy
        if isinstance(node, ast.Import):
            if any(a.name == _TRAINER_MODULE for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == _TRAINER_MODULE:
                return True
            if mod == "core.training" and any(a.name == "trainer" for a in node.names):
                return True
    return False


def _stubs_before_that_import(source: str, tree: ast.Module) -> bool:
    """Whether a stub call naming ``unsloth`` appears at module scope BEFORE the trainer import.

    Order is the whole point. A stub registered afterwards is registered after the real import
    has already been attempted and raised, so it changes nothing."""
    trainer_line = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and _imports_trainer_at_module_scope(
            ast.Module(body = [node], type_ignores = [])
        ):
            trainer_line = node.lineno
            break
    if trainer_line is None:
        return True
    head = "\n".join(source.splitlines()[: trainer_line - 1])
    return _REQUIRED_STUB in head and ("stub" in head or "sys.modules" in head)


def test_no_test_module_imports_the_trainer_unstubbed():
    offenders = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        source = path.read_text(encoding = "utf-8")
        if _TRAINER_MODULE not in source and "core.training import trainer" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # not this guard's job to report
            continue
        if not _imports_trainer_at_module_scope(tree):
            continue
        if not _stubs_before_that_import(source, tree):
            offenders.append(path.name)

    assert not offenders, (
        f"{len(offenders)} test module(s) import {_TRAINER_MODULE} at module scope without "
        f"stubbing its heavy deps first, so they fail COLLECTION on the backend pytest matrix "
        f"(which installs neither unsloth nor trl) and take the whole job down: {offenders}. "
        f"Copy the _stub_if_missing block from test_trainer_stdout_quiet.py, above the import."
    )


def test_the_guard_would_catch_an_unstubbed_module(tmp_path):
    """The guard above passes trivially if its matching is wrong, so pin both answers here."""
    unstubbed = tmp_path / "test_unstubbed.py"
    unstubbed.write_text("from core.training import trainer as t\n", encoding = "utf-8")
    tree = ast.parse(unstubbed.read_text(encoding = "utf-8"))
    assert _imports_trainer_at_module_scope(tree)
    assert not _stubs_before_that_import(unstubbed.read_text(encoding = "utf-8"), tree)

    stubbed_source = (
        '_stub_if_missing("unsloth", ())\n'
        "from core.training import trainer as t\n"
    )
    assert _stubs_before_that_import(stubbed_source, ast.parse(stubbed_source))

    # And a stub that lands too late does not count.
    too_late = (
        "from core.training import trainer as t\n"
        '_stub_if_missing("unsloth", ())\n'
    )
    assert not _stubs_before_that_import(too_late, ast.parse(too_late))
