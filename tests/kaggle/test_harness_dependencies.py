# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Modules the test files import must be in the CI pip line, or their tests vanish from the summary."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests" / "kaggle"
WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"

# Import name -> distribution name, only where they differ.
DISTRIBUTION = {
    "PIL": "pillow",
    "yaml": "pyyaml",
}

# Directories the suite puts on sys.path, so their modules are not third-party.
LOCAL_DIRS = (
    TESTS,
    TESTS / "t4_smoke",
    TESTS / "studio_gpu",
    ROOT / ".github" / "scripts",
    ROOT / ".github" / "scripts" / "kaggle_t4_ci",
    ROOT / ".github" / "scripts" / "kaggle_studio_ci",
)


def _local_names() -> set[str]:
    names = set()
    for directory in LOCAL_DIRS:
        names.add(directory.name)
        for path in directory.glob("*.py"):
            names.add(path.stem)
    return names


def _imported_third_party() -> dict[str, set[str]]:
    """Includes imports inside functions, not just module level; those fail only when their test runs."""
    local = _local_names()
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        # importorskip marks a module optional, so it is not an install requirement.
        guarded = {
            node.args[0].value.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "importorskip"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in sys.stdlib_module_names or name in local or name in guarded:
                    continue
                found.setdefault(name, set()).add(path.name)
    return found


def _install_line() -> str:
    text = WORKFLOW.read_text(encoding = "utf-8")
    step = text.index("- name: Test the harness")
    body = text[step : text.index("\n      - name:", step + 1)]
    matches = re.findall(r"pip install[^\n]*", body)
    assert matches, "the Test the harness step no longer installs anything"
    return " ".join(matches)


def test_every_unguarded_import_is_installed_by_the_workflow():
    """The whole point. A module missing here costs tests silently."""
    line = _install_line()
    missing = {
        module: sorted(files)
        for module, files in _imported_third_party().items()
        if DISTRIBUTION.get(module, module) not in line
    }
    assert not missing, (
        "these modules are imported by tests/kaggle/test_*.py without a "
        "pytest.importorskip guard and are not installed by the Test the "
        f"harness step, so those tests will error rather than run: {missing}"
    )


def test_the_import_scan_finds_something_at_all():
    """A scan that silently matched nothing would satisfy the rule above
    forever. Two modules the suite genuinely imports are named here, so a
    refactor that breaks the walk fails rather than passes."""
    found = _imported_third_party()
    assert "datasets" in found, "the walk no longer sees the vision dataset imports"
    assert "torch" in found
    assert found["datasets"], "no file recorded for datasets"


def test_the_scan_looks_inside_functions_and_not_only_at_module_level():
    """A module-level scan would miss datasets, which test_vision_run.py imports only inside test bodies."""
    source = (TESTS / "test_vision_run.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    module_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "datasets" not in module_level, (
        "datasets is now imported at module level in test_vision_run.py, which "
        "makes this test tautological; point it at another function-level "
        "import instead of deleting it"
    )
    assert "datasets" in _imported_third_party()


def test_a_guarded_import_is_not_treated_as_a_requirement():
    """`pytest.importorskip` is how this suite says a module is optional, and
    demanding those on the runner would put a CUDA stack on an ubuntu box."""
    line = _install_line()
    assert "vllm" not in line
    assert "unsloth" not in line
