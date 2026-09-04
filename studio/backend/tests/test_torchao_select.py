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
_EXTRAS_REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements" / "extras.txt"


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


def test_matching_torchao_pin_does_not_need_force_reinstall(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_installed_distribution_version", lambda _name: "0.17.0")
    assert mod._exact_distribution_spec_is_installed("torchao==0.17.0")
    assert not mod._exact_distribution_spec_is_installed("torchao==0.16.0")


def test_windows_first_hop_uses_einx_wheel_without_shared_test_tree():
    requirements = _EXTRAS_REQUIREMENTS.read_text(encoding = "utf-8")
    assert 'einx<0.4.3; sys_platform == "win32"' in requirements
    # einx dropped 3.9 in 0.4.0, so the non-Windows side is split by interpreter.
    assert 'einx==0.4.3; sys_platform != "win32" and python_version >= "3.10"' in requirements
    assert 'einx==0.3.0; sys_platform != "win32" and python_version < "3.10"' in requirements


@pytest.mark.parametrize(
    ("rocm_windows_torch_installed", "installed_torch_is_windows_rocm"),
    [
        (True, False),
        (False, True),
        # Both signals agree: the ordinary Windows ROCm host, and the case a
        # two-mixed-only parametrization never covered.
        (True, True),
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
    # #10053 added a require_present gate to install_python_stack: after the core phase
    # it refuses when a managed distribution is not installed at all, which SKIP_STUDIO_BASE
    # guarantees here. Unstubbed, this test asks whether unsloth happens to be installed in
    # whatever environment runs it -- it passes on a developer machine that has it and fails
    # in CI, which is not what the test is about. Stubbed like every other installer side
    # effect below.
    monkeypatch.setattr(mod, "_repair_damaged_core_payload", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_bootstrap_uv", lambda: False)
    monkeypatch.setattr(mod, "_repair_bad_anyio", lambda: None)
    monkeypatch.setattr(mod, "_ensure_rocm_torch", lambda: None)
    monkeypatch.setattr(mod, "_ensure_cuda_torch", lambda: None)
    # A Windows ROCm box has no usable NVIDIA GPU. Claiming one here described a
    # machine that cannot exist, and _expected_torch_flavor_tag reads exactly this
    # flag to decide whether a CUDA expectation exists at all: with it True, the
    # Windows flavor invariant demanded a cu* build, found the runner's CPU torch,
    # and failed the whole install long after the torchao branch under test.
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: False)
    # The installed torch is ambient, so leaving it unpatched made the verdict depend
    # on the developer's machine: a CUDA workstation passed and a CPU-only CI runner
    # failed, on identical code.
    monkeypatch.setattr(mod, "_RECORDED_TORCH_TAG", "")
    monkeypatch.setattr(
        mod, "_probe_torch_runtime", lambda *args, **kwargs: (True, True, "2.9.1+cpu", "", "")
    )
    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "pip_install", _record_pip_install)
    monkeypatch.setattr(mod, "_progress", lambda label: progress_labels.append(label))
    monkeypatch.setattr(mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin)
    monkeypatch.setattr(mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: subprocess_result)

    # Checked BEFORE the install so a regression names its cause here rather than as
    # an opaque `assert 1 == 0` on the line below, which is how this surfaced: the run
    # returned 1 on a CPU-only runner and 0 on a CUDA workstation, on identical code.
    assert mod._expected_torch_flavor_tag() == "", (
        "no CUDA expectation may exist on a Windows ROCm host: a non-empty tag means "
        "the Windows flavor invariant will demand a cu* build, not find one, and fail "
        "the install long after the torchao branch this test is about"
    )

    assert mod.install_python_stack() == 0

    assert not any(spec.startswith("torchao") for spec in installed_specs)
    assert "dependency overrides (skipped, Windows ROCm)" in progress_labels
