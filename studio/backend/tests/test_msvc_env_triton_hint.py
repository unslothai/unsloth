# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "install Triton" hint names the triton-windows range setup.ps1 would install for this torch."""

import importlib.metadata
import platform
import re
from pathlib import Path

import pytest

from core import _msvc_env


@pytest.mark.parametrize(
    "torch_version, spec",
    [
        ("2.13.0+cu130", "triton-windows>=3.7,<3.8"),
        ("2.14.0", "triton-windows>=3.8,<3.9"),
        ("2.15.0", "triton-windows>=3.9,<3.10"),
        ("2.15.0+cu130", "triton-windows>=3.9,<3.10"),
        ("2.15.1", "triton-windows>=3.9,<3.10"),
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
        ("2.15.0+rocm7.14.0", "triton-windows>=3.9,<3.10"),
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


@pytest.mark.parametrize("torch_version", ["2.11.0", "2.15.0"])
def test_windows_on_arm_gets_the_first_win_arm64_release(monkeypatch, torch_version):
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    assert _msvc_env._triton_windows_spec() == "triton-windows>=3.8.0.post28"


_SETUP_PS1 = Path(__file__).resolve().parents[2] / "setup.ps1"


def _setup_ps1_triton_spec(torch_version: str) -> str:
    """Evaluate setup.ps1's Get-TritonWindowsSpec (non-ARM) for a torch version, from its source."""
    body = re.search(
        r"^function Get-TritonWindowsSpec \{\n(.*?)^\}\n",
        _SETUP_PS1.read_text(encoding = "utf-8"),
        re.M | re.S,
    ).group(1)
    arms = {}
    for minor, arm in re.findall(r"^\s+(\d+) \{ (.*) \}$", body, re.M):
        exact = re.match(
            r'if \(\[int\]\$Matches\[2\] -eq (\d+)\) \{ return "([^"]+)" \} return "([^"]+)"$',
            arm,
        )
        if exact:
            arms[int(minor)] = (int(exact.group(1)), exact.group(2), exact.group(3))
        else:
            plain = re.fullmatch(r'return "([^"]+)"', arm)
            assert plain, f"unparsed setup.ps1 switch arm: {arm!r}"
            arms[int(minor)] = (None, plain.group(1), plain.group(1))
    fallback = re.findall(r'^\s+return "([^"]+)"\s*$', body, re.M)[-1]
    assert arms, "no switch arms found in Get-TritonWindowsSpec"
    major, minor, patch = (int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", torch_version).groups())
    if major != 2 or minor not in arms:
        return fallback
    exact_patch, exact_spec, spec = arms[minor]
    return exact_spec if exact_patch == patch else spec


@pytest.mark.parametrize(
    "torch_version",
    ["2.9.1", "2.11.0", "2.12.0", "2.12.1", "2.13.0", "2.14.0", "2.15.0", "2.15.1", "2.16.0"],
)
def test_hint_matches_setup_ps1(monkeypatch, torch_version):
    # The runtime hint must name exactly what setup.ps1 installs for the same (non-ROCm, x64) torch.
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    assert _msvc_env._triton_windows_spec() == _setup_ps1_triton_spec(torch_version)


def test_setup_ps1_pairs_torch_215_with_triton_39():
    assert _setup_ps1_triton_spec("2.15.0") == "triton-windows>=3.9,<3.10"
