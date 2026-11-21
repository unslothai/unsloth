# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "install Triton" hint names the triton-windows range setup.ps1 would install for this torch."""

import importlib.metadata
import platform

import pytest

from core import _msvc_env


@pytest.mark.parametrize(
    "torch_version, spec",
    [
        ("2.13.0+cu130", "triton-windows>=3.7,<3.8"),
        ("2.14.0", "triton-windows>=3.8,<3.9"),
        ("2.11.0+cu128", "triton-windows<3.7"),
        ("2.12.1+cu130", "triton-windows>=3.7,<3.8"),
        ("2.12.0", "triton-windows>=3.7,<3.7.1"),
        ("2.12.0+cu130", "triton-windows>=3.7,<3.7.1"),
        ("2.13.0a0+gitabc", "triton-windows>=3.7,<3.8"),
        ("2.9.1", "triton-windows<3.7"),
    ],
)
def test_spec_follows_the_resident_torch(monkeypatch, torch_version, spec):
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    assert _msvc_env._triton_windows_spec() == spec


@pytest.mark.parametrize(
    "torch_version, spec",
    [
        ("2.11.0+rocm7.14.0", "triton-windows>=3.8,<3.9"),
        ("2.12.0+rocm7.14.0", "triton-windows>=3.8,<3.9"),
        ("2.12.1+rocm7.14.0", "triton-windows>=3.8,<3.9"),
        ("2.13.0+rocm7.14.0", "triton-windows>=3.7,<3.8"),
        ("2.10.0+rocm7.2", "triton-windows<3.7"),
    ],
)
def test_rocm_torch_gets_the_setup_pairing(monkeypatch, torch_version, spec):
    # setup.ps1 pairs ROCm torch 2.11 / 2.12 with triton-windows 3.8; the hint must not name the NVIDIA range.
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: True)
    assert _msvc_env._triton_windows_spec() == spec


def test_unreadable_torch_keeps_the_old_range(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert _msvc_env._triton_windows_spec() == "triton-windows<3.7"


def test_windows_on_arm_gets_the_first_win_arm64_release(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.11.0")
    assert _msvc_env._triton_windows_spec() == "triton-windows>=3.8.0.post28"
