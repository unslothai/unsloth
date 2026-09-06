# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _torch_step_label in install_python_stack.py.

The label is the only place a standalone `unsloth studio update` states which
torch backend it is working on. On Windows the ROCm probe reads rocminfo and
amd-smi, which ship with the HIP SDK and not with AMD's bundled-runtime wheels,
so a working ROCm host printed "torch check (cpu)" on the same line-block where
the next step correctly reported Windows ROCm.
"""

from __future__ import annotations

import sys
from pathlib import Path

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"


def _load_module(monkeypatch):
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


def _label(
    monkeypatch,
    *,
    nvidia,
    rocm_probe,
    windows_rocm_torch,
    known_backend = "",
):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_TORCH_BACKEND", known_backend)
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: nvidia)
    monkeypatch.setattr(mod, "_has_rocm_gpu", lambda: rocm_probe)
    monkeypatch.setattr(mod, "_installed_torch_is_windows_rocm", lambda: windows_rocm_torch)
    return mod._torch_step_label("check")


def test_an_explicit_backend_wins_over_every_probe(monkeypatch):
    label = _label(
        monkeypatch,
        nvidia = True,
        rocm_probe = True,
        windows_rocm_torch = True,
        known_backend = "xpu",
    )
    assert label == "torch check (xpu)"


def test_nvidia_still_takes_priority(monkeypatch):
    label = _label(monkeypatch, nvidia = True, rocm_probe = False, windows_rocm_torch = True)
    assert label == "torch check (cuda)"


def test_the_rocm_probe_still_answers(monkeypatch):
    label = _label(monkeypatch, nvidia = False, rocm_probe = True, windows_rocm_torch = False)
    assert label == "torch check (rocm)"


def test_a_windows_rocm_torch_is_rocm_even_with_no_rocm_tooling(monkeypatch):
    """The regression: rocminfo and amd-smi are absent, torch.version.hip is not."""
    label = _label(monkeypatch, nvidia = False, rocm_probe = False, windows_rocm_torch = True)
    assert label == "torch check (rocm)"


def test_a_host_with_neither_is_still_cpu(monkeypatch):
    label = _label(monkeypatch, nvidia = False, rocm_probe = False, windows_rocm_torch = False)
    assert label == "torch check (cpu)"
