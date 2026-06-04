# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Unit tests for ``maybe_set_windows_rocm_bnb_version`` and its
``_detect_installed_bnb_rocm_version`` helper in ``unsloth/import_fixes.py``.

These pin ``BNB_ROCM_VERSION`` from the installed bitsandbytes ROCm wheel on
Windows + ROCm torch so the right native backend loads (AMD's Windows wheel
ships e.g. ``libbitsandbytes_rocm72.dll`` while ``torch.version.hip`` reports
7.13). The module is loaded in isolation -- its top-level imports are stdlib +
packaging only, so no torch / GPU is required and unsloth's GPU init never runs.
"""

from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import pytest

_IMPORT_FIXES_PATH = (
    Path(__file__).resolve().parent.parent / "unsloth" / "import_fixes.py"
)


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", _IMPORT_FIXES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def import_fixes():
    return _load_import_fixes()


@pytest.fixture()
def clean_env(monkeypatch):
    """Start with both env vars unset; guarantee they are removed afterwards
    (the function writes ``os.environ`` directly, which monkeypatch will not
    auto-revert)."""
    for var in ("BNB_ROCM_VERSION", "UNSLOTH_SKIP_BNB_ROCM_VERSION"):
        monkeypatch.delenv(var, raising = False)
    yield monkeypatch
    for var in ("BNB_ROCM_VERSION", "UNSLOTH_SKIP_BNB_ROCM_VERSION"):
        os.environ.pop(var, None)


def _force(import_fixes, monkeypatch, *, win, rocm, detected):
    monkeypatch.setattr(import_fixes.sys, "platform", "win32" if win else "linux")
    monkeypatch.setattr(import_fixes, "_is_rocm_torch_build", lambda: rocm)
    monkeypatch.setattr(
        import_fixes, "_detect_installed_bnb_rocm_version", lambda: detected
    )


# ---------------------------------------------------------------------------
# _detect_installed_bnb_rocm_version
# ---------------------------------------------------------------------------


def test_detect_picks_highest_rocm_suffix(import_fixes, tmp_path, monkeypatch):
    pkg = tmp_path / "bitsandbytes"
    pkg.mkdir()
    for name in (
        "libbitsandbytes_rocm72.dll",
        "libbitsandbytes_rocm713.dll",  # numerically highest -> should win
        "libbitsandbytes_cpu.dll",
        "__init__.py",
    ):
        (pkg / name).write_text("")
    fake_spec = types.SimpleNamespace(submodule_search_locations = [str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)
    assert import_fixes._detect_installed_bnb_rocm_version() == "713"


def test_detect_none_when_only_non_rocm_dlls(import_fixes, tmp_path, monkeypatch):
    pkg = tmp_path / "bitsandbytes"
    pkg.mkdir()
    (pkg / "libbitsandbytes_cpu.dll").write_text("")
    (pkg / "libbitsandbytes_cuda124.dll").write_text("")
    fake_spec = types.SimpleNamespace(submodule_search_locations = [str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)
    assert import_fixes._detect_installed_bnb_rocm_version() is None


def test_detect_none_when_bnb_absent(import_fixes, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert import_fixes._detect_installed_bnb_rocm_version() is None


# ---------------------------------------------------------------------------
# maybe_set_windows_rocm_bnb_version
# ---------------------------------------------------------------------------


def test_sets_bnb_version_on_windows_rocm(import_fixes, clean_env):
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() == "72"
    assert os.environ["BNB_ROCM_VERSION"] == "72"


def test_noop_off_windows(import_fixes, clean_env):
    # Linux ROCm resolves its backend correctly from torch.version.hip.
    _force(import_fixes, clean_env, win = False, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_noop_when_not_rocm_torch(import_fixes, clean_env):
    _force(import_fixes, clean_env, win = True, rocm = False, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_noop_when_no_rocm_dll_installed(import_fixes, clean_env):
    # Never force a ROCm backend name when no ROCm DLL ships (avoid breaking a
    # non-ROCm bitsandbytes that happens to sit next to a ROCm torch build).
    _force(import_fixes, clean_env, win = True, rocm = True, detected = None)
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_respects_user_provided_value(import_fixes, clean_env):
    clean_env.setenv("BNB_ROCM_VERSION", "999")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == "999"


def test_explicit_opt_out(import_fixes, clean_env):
    clean_env.setenv("UNSLOTH_SKIP_BNB_ROCM_VERSION", "1")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ
