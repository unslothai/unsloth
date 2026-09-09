# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pytest config for studio/install tests: add studio/ to sys.path so `backend` imports work from the repo root."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

# <repo-root>/studio  →  makes `backend` importable as a package
_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))


_STACK_FILE = _STUDIO_DIR / "install_python_stack.py"

# What install_python_stack() itself resets at the top of every dependency pass. Kept in
# step with it: the module is loaded once per test FILE, so a test that drives
# _ensure_rocm_torch or a gated step leaves the pass half-finished for every test after
# it, and the next one measures the previous one.
_PASS_STATE_DEFAULTS = {
    "_INSTALL_ACTIONS": 0,
    "_PASS_EVIDENCE": None,
    "_CONSTRAINTS_CACHE": None,
    "_CLOSURE_INDEX_CACHE": None,
    "_BNB_ROCM_PASS_PROVENANCE": None,
    "_BNB_ROCM_PASS_ASSET": None,
}


def _loaded_stacks(test_module):
    """Every live copy of install_python_stack, sys.modules or not.

    Three test files load it into sys.modules under the same name
    ("studio_install_python_stack"), so the last one imported owns that key and the
    others keep a module object nothing in sys.modules points at any more. Resetting
    only sys.modules leaves those copies carrying the previous test's pass state, which
    is invisible when the file runs alone and decides the answer when the directory
    runs together. So the module under test is looked up through the test file that
    holds it as well.
    """
    target = os.path.realpath(_STACK_FILE)
    found = {}
    candidates = list(sys.modules.values())
    if test_module is not None:
        candidates.extend(vars(test_module).values())
    for module in candidates:
        if not isinstance(module, ModuleType):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            if os.path.realpath(path) != target:
                continue
        except OSError:
            continue
        found[id(module)] = module
    return found.values()


def _reset_pass_state(test_module) -> None:
    for module in _loaded_stacks(test_module):
        for name, value in _PASS_STATE_DEFAULTS.items():
            if hasattr(module, name):
                setattr(module, name, value)
        results = getattr(module, "_STEP_RESULTS", None)
        if isinstance(results, dict):
            results.clear()


@pytest.fixture(autouse = True)
def reset_install_pass_state(request):
    """Start and end every test with the state a fresh dependency pass would have.

    Here rather than in each file because the state belongs to install_python_stack, not
    to any one suite, and a file that forgets it fails in a way that only shows up when
    the whole directory runs in one process.
    """
    _reset_pass_state(request.module)
    yield
    _reset_pass_state(request.module)
