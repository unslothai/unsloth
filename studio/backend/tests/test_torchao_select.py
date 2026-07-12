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
from unittest.mock import MagicMock

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
        # torch 2.10 on CUDA <= 12 -> 0.16.0 (its cpp is built for torch 2.10.0 and
        # loads against the CUDA-12 PyPI wheel). Independent of patch level.
        ("2.10.0+cu128", "torchao==0.16.0"),
        ("2.10.0+cu126", "torchao==0.16.0"),
        ("2.10.0+rocm6.4", "torchao==0.16.0"),
        ("2.10.0+cpu", "torchao==0.16.0"),
        ("2.10.1", "torchao==0.16.0"),
        ("2.10.0", "torchao==0.16.0"),
        # torch 2.10 on CUDA >= 13 (Blackwell / cu130): 0.16.0's CUDA-12 cpp can't
        # load against a CUDA-13 torch (libcudart.so.12 error), so use 0.17.0.
        ("2.10.0+cu130", "torchao==0.17.0"),
        ("2.10.0+cu140", "torchao==0.17.0"),
        # Pre-release / dev / rc builds: the minor is cleaned of non-digits; the
        # CUDA tag still decides 0.16.0 vs 0.17.0.
        ("2.10.0rc1", "torchao==0.16.0"),
        ("2.10.0.dev20250804+cu130", "torchao==0.17.0"),
        ("2.10.0.dev20250804+cu128", "torchao==0.16.0"),
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


@pytest.mark.parametrize(
    ("rocm_windows_torch_installed", "installed_torch_is_windows_rocm"),
    [
        (True, False),
        (False, True),
    ],
)
def test_skips_torchao_on_windows_rocm(
    monkeypatch, tmp_path, rocm_windows_torch_installed, installed_torch_is_windows_rocm
):
    """The overrides step must skip torchao on Windows ROCm: no working build exists
    there (it imports an absent c10d backend and crashes transformers.quantizers),
    so the installer skips it and relies on the runtime stub instead."""
    mod = _load_module(monkeypatch)
    installed_specs: list[str] = []
    progress_labels: list[str] = []

    def _record_pip_install(*args, **kwargs):
        installed_specs.extend(str(arg) for arg in args)
        return 0

    unstructured_plugin = tmp_path / "unstructured"
    github_plugin = tmp_path / "github"
    unstructured_plugin.mkdir()
    github_plugin.mkdir()

    subprocess_result = MagicMock()
    subprocess_result.returncode = 0
    subprocess_result.stdout = ""

    monkeypatch.setenv("SKIP_STUDIO_BASE", "1")
    monkeypatch.setattr(mod, "IS_WINDOWS", True)
    monkeypatch.setattr(mod, "IS_MACOS", False)
    monkeypatch.setattr(mod, "IS_MAC_ARM", False)
    monkeypatch.setattr(mod, "NO_TORCH", False)
    monkeypatch.setattr(mod, "_rocm_windows_torch_installed", rocm_windows_torch_installed)
    monkeypatch.setattr(
        mod, "_installed_torch_is_windows_rocm", lambda: installed_torch_is_windows_rocm
    )
    monkeypatch.setattr(mod, "_bootstrap_uv", lambda: False)
    monkeypatch.setattr(mod, "_repair_bad_anyio", lambda: None)
    monkeypatch.setattr(mod, "_ensure_rocm_torch", lambda: None)
    monkeypatch.setattr(mod, "_ensure_cuda_torch", lambda: None)
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "pip_install", _record_pip_install)
    monkeypatch.setattr(mod, "_progress", lambda label: progress_labels.append(label))
    monkeypatch.setattr(mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin)
    monkeypatch.setattr(mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: subprocess_result)

    assert mod.install_python_stack() == 0

    assert not any(spec.startswith("torchao") for spec in installed_specs)
    assert "dependency overrides (skipped, Windows ROCm)" in progress_labels
