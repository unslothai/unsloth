# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account roots must be resolved when work runs, never while importing it."""

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_PATH_MODULES = ("utils.paths", "hub.utils.paths")
# Shared by every account, so a value captured at import cannot leak between them.
_INSTALL_WIDE = ("studio_root", "auth_db_path", "bin_root", "cache_root", "logs_root")


def import_time_path_calls(source: str, module: str = "") -> list[tuple[int, str]]:
    tree = ast.parse(source)
    names = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            origin = node.module or ""
            if node.level:
                origin = ".".join(module.split(".")[:-node.level] + [origin])
            for alias in node.names:
                names[alias.asname or alias.name] = f"{origin}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

    def target(node):
        if isinstance(node, ast.Name):
            return names.get(node.id, "")
        if isinstance(node, ast.Attribute):
            return f"{target(node.value)}.{node.attr}"
        return ""

    found = []

    class AtImport(ast.NodeVisitor):
        def visit_Call(self, node):
            name = target(node.func)
            if (
                name.startswith(tuple(f"{prefix}." for prefix in _PATH_MODULES))
                and name.endswith(("_root", "_path"))
                and name.rsplit(".", 1)[-1] not in _INSTALL_WIDE
            ):
                found.append((node.lineno, name))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Function bodies run later; their defaults and decorators run now.
            for value in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
                if value is not None:
                    self.visit(value)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            for value in [*node.args.defaults, *node.args.kw_defaults]:
                if value is not None:
                    self.visit(value)

    AtImport().visit(tree)
    return found


@pytest.mark.parametrize("source", [
    "from utils.paths import outputs_root\nROOT = outputs_root()",
    "from utils.paths import outputs_root as root\ndef f(p = str(root())): pass",
    "import utils.paths as p\nclass C: root = p.outputs_root()",
    "from utils import paths as p\nf = lambda root = p.outputs_root(): root",
    "from hub.utils.paths import datasets_root\nROOT = datasets_root()",
])
def test_guard_detects_captured_roots(source):
    assert import_time_path_calls(source)


def test_guard_allows_call_time_accessors():
    source = "from utils.paths import outputs_root\ndef f(): return outputs_root()\nROOT = lambda: outputs_root()"
    assert import_time_path_calls(source) == []


def test_backend_does_not_capture_account_paths_at_import():
    failures = []
    for path in sorted(_BACKEND.rglob("*.py")):
        relative = path.relative_to(_BACKEND)
        if "tests" in relative.parts:
            continue
        module = ".".join(relative.with_suffix("").parts)
        for line, target in import_time_path_calls(path.read_text(encoding = "utf-8"), module):
            failures.append(f"{relative}:{line}: {target}")
    assert failures == [], "Resolve paths at call time:\n" + "\n".join(failures)
