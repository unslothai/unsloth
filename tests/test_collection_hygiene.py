# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`pytest tests` must collect cleanly.

A collection ERROR is not a failing test: pytest reports "Interrupted" and
never runs the rest of the suite, so one bad module hides thousands of good
ones. These guard the three ways modules under tests/ used to break collection.
"""

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
    """A dot in the stem is a package separator to pytest's importer.

    tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py was
    imported as ...test_merged_model_perplexity_qwen_2.5 and died with
    "No module named 'test_merged_model_perplexity_qwen_2'".
    """
    dotted = [str(p.relative_to(REPO_ROOT)) for p in _test_modules() if "." in p.stem]
    assert not dotted, f"test module filenames must not contain '.': {dotted}"


def test_saving_scripts_opt_in_before_running_at_import():
    """Files under tests/saving with no test items are standalone GPU scripts.

    Their whole body runs at import, so bare collection downloads checkpoints,
    trains and pushes to the Hub. Each must call require_opt_in() so it is a
    visible SKIP instead.
    """
    ungated = []
    for path in sorted(SAVING_DIR.rglob("test_*.py")):
        source = path.read_text(encoding = "utf-8")
        if _has_test_items(ast.parse(source)):
            continue
        if "require_opt_in(" not in source:
            ungated.append(str(path.relative_to(REPO_ROOT)))
    assert not ungated, f"tests/saving scripts missing the require_opt_in gate: {ungated}"


def test_raw_text_does_not_leave_its_datasets_mock_in_sys_modules():
    """tests/test_raw_text.py stubs `datasets` to import unsloth.dataprep.raw_text.

    It used to leave the stub in sys.modules for the rest of the session, so
    every later `from datasets import IterableDataset` raised ImportError and
    tests/utils/test_packing.py failed to collect.
    """
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
