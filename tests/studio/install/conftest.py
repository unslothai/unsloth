# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pytest config for studio/install tests: add studio/ to sys.path so `backend` imports work from the repo root."""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))


_STACK_FILE = _STUDIO_DIR / "install_python_stack.py"

# The module is loaded once per test file, so per-pass state must be reset.
_PASS_STATE_DEFAULTS = {
    "_INSTALL_ACTIONS": 0,
    "_PASS_EVIDENCE": None,
    "_CONSTRAINTS_CACHE": None,
    "_CLOSURE_INDEX_CACHE": None,
    "_BNB_ROCM_PASS_PROVENANCE": None,
    "_BNB_ROCM_PASS_ASSET": None,
}


@functools.lru_cache(maxsize = None)
def _realpath(path: str) -> str:
    """A module __file__ never changes once imported, so the scan below can resolve each
    distinct path once instead of syscalling over all of sys.modules twice per test."""
    return os.path.realpath(path)


def _loaded_stacks(test_module):
    """Finds every loaded install_python_stack copy, not just the one sys.modules still points at."""
    target = _realpath(str(_STACK_FILE))
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
            if _realpath(path) != target:
                continue
        except OSError:
            continue
        except TypeError:
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
    """Resets install_python_stack pass state around each test, so it cannot leak between files."""
    _reset_pass_state(request.module)
    yield
    _reset_pass_state(request.module)


@pytest.fixture(autouse = True)
def pin_installer_torch_vendor(request, monkeypatch):
    """Pin the installer's torch-vendor probe so a ROCm-torch dev box answers like CI."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    for module in [*sys.modules.values(), *vars(request.module).values()]:
        # __dict__: hasattr would trip a lazy __getattr__.
        if "_rocm_torch_preferred" in (getattr(module, "__dict__", None) or {}):
            monkeypatch.setattr(module, "_installed_torch_is_rocm", lambda: None)
