# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A collection error hides the rest of the suite: pytest reports Interrupted and runs nothing else."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SAVING_DIR = TESTS_DIR / "saving"


def _test_modules() -> list[Path]:
    return sorted(TESTS_DIR.rglob("test_*.py"))


def _has_test_items(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return True
    return False


def test_no_test_module_filename_contains_a_dot():
    """A dot in a test module's stem is read as a package separator, breaking collection."""
    dotted = [str(p.relative_to(REPO_ROOT)) for p in _test_modules() if "." in p.stem]
    assert not dotted, f"test module filenames must not contain '.': {dotted}"


def test_saving_scripts_opt_in_before_running_at_import():
    """Standalone tests/saving scripts run at import; each must call require_opt_in() to skip visibly."""
    ungated = []
    for path in sorted(SAVING_DIR.rglob("test_*.py")):
        source = path.read_text(encoding = "utf-8")
        if _has_test_items(ast.parse(source)):
            continue
        if "require_opt_in(" not in source:
            ungated.append(str(path.relative_to(REPO_ROOT)))
    assert not ungated, f"tests/saving scripts missing the require_opt_in gate: {ungated}"


def test_raw_text_does_not_leave_its_datasets_mock_in_sys_modules():
    """A datasets stub left in sys.modules makes later datasets imports raise ImportError."""
    pytest.importorskip("datasets")
    child = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"runpy.run_path({str(TESTS_DIR / 'test_raw_text.py')!r}, run_name='test_raw_text')\n"
        "from datasets import Dataset, IterableDataset\n"
        "print('DATASETS_OK', Dataset.__module__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        capture_output = True,
        text = True,
        check = False,
    )
    assert (
        "DATASETS_OK" in result.stdout
    ), f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "DATASETS_OK datasets." in result.stdout, result.stdout
