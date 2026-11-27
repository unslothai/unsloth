# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "install Triton" hint names the triton-windows range setup.ps1 would install for this torch."""

import importlib.metadata
import platform
import re
import shutil
from pathlib import Path

import pytest

from core import _msvc_env

# All pwsh spawns go through run_pwsh: parallel startups race on the shared cache and SIGABRT.
from unsloth_pwsh_runner import run_pwsh


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


@pytest.mark.parametrize(
    "torch_version, spec",
    [
        # Torch below 2.14 pairs with no win_arm64 triton-windows: the open floor.
        ("2.11.0", "triton-windows>=3.8.0.post28"),
        ("2.13.0+cu130", "triton-windows>=3.8.0.post28"),
        # The x64 per-minor upper bound, never below the first win_arm64 wheel.
        ("2.14.0+cu134", "triton-windows>=3.8.0.post28,<3.9"),
        ("2.15.0", "triton-windows>=3.9,<3.10"),
        ("2.16.0", "triton-windows>=3.8.0.post28"),
        # The nightly index's torch: a minor the table does not name keeps the open floor.
        ("2.16.0.dev20261126+cu135", "triton-windows>=3.8.0.post28"),
    ],
)
def test_windows_on_arm_pairs_by_torch_minor(monkeypatch, torch_version, spec):
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    assert _msvc_env._triton_windows_spec() == spec


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh unavailable")
@pytest.mark.parametrize(
    "torch_version",
    [
        "",
        "2.11.0+cu130",
        "2.13.0+cu130",
        "2.14.0+cu134",
        "2.15.0+cu134",
        "2.16.0",
        "2.16.0.dev20261126+cu135",
    ],
)
def test_windows_on_arm_hint_matches_setup_ps1(monkeypatch, torch_version):
    out = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            _ps1_function("Get-TritonWindowsSpec")
            + f"\n\"RESULT:$(Get-TritonWindowsSpec -TorchVersion '{torch_version}' -WinArm64 $true)\"",
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    ps1 = out.stdout.strip().splitlines()[-1].removeprefix("RESULT:")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    if torch_version:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    else:

        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", missing)
    assert _msvc_env._triton_windows_spec() == ps1


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
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch_version)
    monkeypatch.setattr(_msvc_env, "_torch_is_rocm_build", lambda: False)
    assert _msvc_env._triton_windows_spec() == _setup_ps1_triton_spec(torch_version)


def test_setup_ps1_pairs_torch_215_with_triton_39():
    assert _setup_ps1_triton_spec("2.15.0") == "triton-windows>=3.9,<3.10"


def _ps1_function(name: str) -> str:
    match = re.search(
        rf"^function {re.escape(name)} \{{\n.*?^\}}\n",
        _SETUP_PS1.read_text(encoding = "utf-8"),
        re.M | re.S,
    )
    assert match, f"function {name} not found in setup.ps1"
    return match.group(0)


def _fast_path_needs_pass(
    torch_version: str,
    triton_versions: list,
    *,
    arm64 = False,
    offline = False,
):
    """Run setup.ps1's Test-TritonWindowsPairingNeedsPass in pwsh against a stubbed metadata probe."""
    triton = triton_versions[0] if len(triton_versions) == 1 else ""
    stubs = (
        "function Invoke-BoundedPythonProbe { param([string]$PythonExe, [string]$Code)\n"
        f'  [pscustomobject]@{{ Ok = $true; Output = "TORCH={torch_version}`nTRITONWIN={triton}`n" }} }}\n'
        f"function Test-WinArm64Venv {{ return ${str(arm64).lower()} }}\n"
        f"function Test-UvOfflineRequested {{ return ${str(offline).lower()} }}\n"
        'function substep { param($m, $c) Write-Host "SUBSTEP: $m" }\n'
        "$script:OfflineFastPath = $false\n"
    )
    body = "\n".join(
        _ps1_function(n)
        for n in (
            "Get-TritonWindowsSpec",
            "Get-TritonWindowsVersionKey",
            "Test-TritonWindowsVersionInSpec",
            "Get-TritonWindowsPairing",
            "Test-TritonWindowsPairingNeedsPass",
        )
    )
    out = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            f'{stubs}\n{body}\n"RESULT:$(Test-TritonWindowsPairingNeedsPass)"',
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    return out.stdout.strip().splitlines()[-1] == "RESULT:True"


_SKIP_NO_PWSH = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh unavailable")


@_SKIP_NO_PWSH
@pytest.mark.parametrize(
    "torch_version, triton_versions, needs_pass",
    [
        ("2.15.0+cu130", ["3.6.0.post25"], True),
        ("2.15.0+cu130", ["3.8.0.post29"], True),
        ("2.15.0+cu130", ["3.9.0.post30"], False),
        ("2.14.0+cu130", ["3.8.0.post29"], False),
        ("2.14.0+cu130", ["3.9.0.post30"], True),
        ("2.12.0+cu130", ["3.7.0.post21"], False),
        ("2.12.0+cu130", ["3.7.1.post22"], True),
        ("2.12.1+cu130", ["3.7.1.post22"], False),
        ("2.10.0+cu128", ["3.6.0.post25"], False),
        ("2.10.0+cu128", ["3.7.0.post21"], True),
        # Nothing the CUDA branch of the pass would repair: no or several records, non-CUDA torch, unparseable version.
        ("2.15.0+cu130", [], False),
        ("2.15.0+cu130", ["3.6.0.post25", "3.9.0.post30"], False),
        ("2.15.0+cpu", ["3.6.0.post25"], False),
        ("2.15.0+xpu", ["3.6.0.post25"], False),
        ("2.12.0+rocm7.14.0", ["3.6.0.post25"], False),
        ("2.15.0+cu130", ["3.10.0rc1"], False),
    ],
)
def test_fast_path_forces_pass_for_an_unpaired_cuda_triton(
    torch_version, triton_versions, needs_pass
):
    assert _fast_path_needs_pass(torch_version, triton_versions) is needs_pass


@_SKIP_NO_PWSH
def test_fast_path_pairing_uses_the_windows_on_arm_floor_and_bound():
    assert _fast_path_needs_pass("2.14.0+cu134", ["3.8.0.post28"], arm64 = True) is False
    assert _fast_path_needs_pass("2.14.0+cu134", ["3.8.0.post27"], arm64 = True) is True
    assert _fast_path_needs_pass("2.14.0+cu134", ["3.9.0.post30"], arm64 = True) is True
    assert _fast_path_needs_pass("2.15.0+cu134", ["3.9.0.post30"], arm64 = True) is False
    assert _fast_path_needs_pass("2.15.0+cu134", ["3.8.0.post29"], arm64 = True) is True
    assert _fast_path_needs_pass("2.15.0+cu130", ["3.7.0.post21"], arm64 = True) is True
    # The nightly index's torch 2.16 dev with the 3.9 wheel the open floor resolves to: kept.
    assert _fast_path_needs_pass("2.16.0.dev20261126+cu135", ["3.9.0.post30"], arm64 = True) is False


@_SKIP_NO_PWSH
def test_fast_path_pairing_waits_when_offline():
    assert _fast_path_needs_pass("2.15.0+cu130", ["3.6.0.post25"], offline = True) is False


def test_fast_path_runs_the_pairing_escape_after_its_definitions():
    # PowerShell defines a function only when execution reaches it: the escape must follow every helper it calls.
    text = _SETUP_PS1.read_text(encoding = "utf-8")
    call = text.index("(Test-TritonWindowsPairingNeedsPass)) {")
    assert (
        "$SkipPythonDeps -and -not $NoTorchMode -and (Test-TritonWindowsPairingNeedsPass)" in text
    )
    for name in (
        "Get-TritonWindowsSpec",
        "Get-TritonWindowsVersionKey",
        "Test-TritonWindowsVersionInSpec",
        "Get-TritonWindowsPairing",
        "Test-TritonWindowsPairingNeedsPass",
    ):
        assert text.index(f"\nfunction {name} {{") < call, name
