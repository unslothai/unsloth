# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _select_torchao_spec in install_python_stack.py.

torchao's C++ extensions are built against one exact torch release, so the
installer must pick the torchao version matching the torch installed in the
venv (otherwise the cpp kernels are skipped). This pins that mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# install_python_stack.py lives at repo_root/studio/install_python_stack.py
_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"


def _load_module(monkeypatch):
    """(Re-)import install_python_stack and return it (mirrors test_pytorch_mirror)."""
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


@pytest.mark.parametrize(
    "torch_version, expected",
    [
        # torch 2.10 (the reported bug: cu130 resolves 2.10.0) -> 0.16.0,
        # independent of the local +cuXXX/+rocm/+cpu suffix or patch level.
        ("2.10.0+cu130", "torchao==0.16.0"),
        ("2.10.0+rocm6.4", "torchao==0.16.0"),
        ("2.10.0+cpu", "torchao==0.16.0"),
        ("2.10.1", "torchao==0.16.0"),
        ("2.10.0", "torchao==0.16.0"),
        # Pre-release / dev / rc builds: the minor is cleaned of non-digits.
        ("2.10.0rc1", "torchao==0.16.0"),
        ("2.10.0.dev20250804+cu130", "torchao==0.16.0"),
        ("2.10rc1", "torchao==0.16.0"),
        # torch 2.11 (reachable via ROCm rocm7.2) and forward -> 0.17.0.
        ("2.11.0+cu130", "torchao==0.17.0"),
        ("2.11.0", "torchao==0.17.0"),
        ("2.12.0", "torchao==0.17.0"),
        # torch <=2.9 keeps today's pin (already a correct match for 2.9.0).
        ("2.9.0+cu128", "torchao==0.14.0"),
        ("2.9.1", "torchao==0.14.0"),
        ("2.8.0", "torchao==0.14.0"),
        ("2.4.0", "torchao==0.14.0"),
        # Unparseable / missing / non-2.x major -> conservative default.
        (None, "torchao==0.14.0"),
        ("", "torchao==0.14.0"),
        ("garbage", "torchao==0.14.0"),
        ("2", "torchao==0.14.0"),
        ("3.0.0", "torchao==0.14.0"),
    ],
)
def test_select_torchao_spec(monkeypatch, torch_version, expected):
    mod = _load_module(monkeypatch)
    assert mod._select_torchao_spec(torch_version) == expected


def test_default_spec_matches_table(monkeypatch):
    """The default/floor stays the historical pin so older torch is unchanged."""
    mod = _load_module(monkeypatch)
    assert mod._TORCHAO_DEFAULT_SPEC == "torchao==0.14.0"
    assert mod._select_torchao_spec("2.9.0") == mod._TORCHAO_DEFAULT_SPEC


def test_skips_torchao_on_windows_rocm():
    """The overrides step must skip torchao on Windows ROCm: no working build exists
    there (it imports an absent c10d backend and crashes transformers.quantizers),
    so the installer skips it and relies on the runtime stub instead."""
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    # Branches on the Windows-ROCm marker set by _ensure_rocm_torch ...
    assert "elif _rocm_windows_torch_installed:" in source
    # ... and reports the skip in the progress label.
    assert "dependency overrides (skipped, Windows ROCm)" in source
