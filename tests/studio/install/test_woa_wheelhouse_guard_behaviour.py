# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The PyPI-first wheelhouse guard, executed rather than read.

test_woa_wheelhouse_prefers_pypi.py pins the SHAPE of the guard against install.ps1's
source. That is worth having, but it cannot catch a helper that is wired correctly and
answers wrongly, and two of the three bugs covered here were exactly that.

The bugs, all found by reviewing the first version of the guard:

1. A wheel the guard skipped stopped counting as available. $WoaWheelNames is rebuilt by
   scanning the staging directory, and a $WoaDropCandidates name missing from it is
   emitted as `name ; platform_machine == "AMD64"`, which EXCLUDES the package on ARM64.
   So the day PyPI published a win_arm64 hf_transfer or brotli, the guard would have
   turned "installed from our wheelhouse" into "not installed at all" -- the opposite of
   preferring upstream, and firing on precisely the event the guard exists for.

2. abi3 was treated as universally compatible. It is forward compatible from the version
   it was built against, so a cp314-abi3 wheel does not import on cp313, and calling ours
   redundant against one would leave the package uninstallable.

3. PyPI publishing a wheel is only availability if the resolve will look at PyPI. Offline,
   or pointed at an exclusive mirror, dropping our copy leaves it obtainable from nowhere.

Hermetic: no network. The live-PyPI leg of the guard is exercised on hardware, not here.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "PowerShell is unavailable"
)

HELPERS = (
    "Test-WoaVersionAtLeast",
    "Test-WoaWheelTags",
    "Test-WoaWheelTagsUsable",
    "Test-WoaResolveReachesPyPI",
)


def _source() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8-sig")


def _function(source: str, name: str) -> str:
    match = re.search(r"(?ms)^    function " + re.escape(name) + r"\b.*?^    \}\r?$", source)
    assert match, f"install.ps1 no longer defines {name}"
    # Lifted out of Install-UnslothStudio, so the four-space body indent comes with it.
    return textwrap.dedent(match.group(0))


def _drop_list_block(source: str) -> str:
    """The real block that builds $WoaWheelNames and emits the override drop lines."""
    start = re.search(r"(?m)^        \$WoaWheelNames = @\{\}\s*$", source)
    end = re.search(r"(?m)^        \$WoaOverrideLines \+= 'torch>=2\.4'\s*$", source)
    assert start and end, "the drop-list block moved"
    return textwrap.dedent(source[start.start():end.start()].replace("\r\n", "\n"))


def _run(script: str) -> str:
    source = _source()
    prelude = "\n".join(_function(source, name) for name in HELPERS)
    proc = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", prelude + "\n" + script],
        capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 120,
    )
    assert proc.returncode == 0, f"pwsh failed: {proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


def _dropped_names(provided: str) -> set[str]:
    """Which $WoaDropCandidates get excluded on ARM64, given what PyPI supplies.

    The staging directory is empty on purpose: the only thing standing between a name and
    the AMD64 drop line is the $WoaPyPIProvided bookkeeping this exercises.
    """
    source = _source()
    out = _run(f"""
$dir = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$WoaWheelDir = $dir
$WoaVenvMinor = "3.13"
$WoaWheelTag = "cp313"; $WoaWheelAbi = "cp313"; $WoaWheelStable = $true; $WoaWheelMinor = 13
$script:WoaTorchAudio = $false
$script:WoaPyarrowWheelName = $null
$script:WoaPyPIProvided = {provided}
function substep {{ param($a, $b) }}
{_drop_list_block(source)}
Remove-Item -LiteralPath $dir -Recurse -Force
($WoaOverrideLines | Where-Object {{ $_ -like "*platform_machine*" }} |
    ForEach-Object {{ ($_ -split ' ')[0] }} | Sort-Object -Unique) -join ","
""")
    return {name for name in out.split(",") if name}


@requires_pwsh
@pytest.mark.parametrize(
    ("wheel", "py_tag", "abi_tag", "usable"),
    [
        # The two interpreter-agnostic shapes the wheelhouse actually ships.
        ("hf_transfer-0.1.9-cp38-abi3-win_arm64.whl", "cp313", "cp313", True),
        ("sqlite_vec-0.1.9-py3-none-win_arm64.whl", "cp313", "cp313", True),
        # Bug 2: abi3 does not reach BACKWARDS to an older interpreter.
        ("hf_transfer-0.2.0-cp314-abi3-win_arm64.whl", "cp313", "cp313", False),
        ("foo-1.0-py314-none-win_arm64.whl", "cp313", "cp313", False),
        # Free-threaded CPython has no stable ABI (CPython #111506).
        ("hf_transfer-0.1.9-cp38-abi3-win_arm64.whl", "cp313", "cp313t", False),
        # An exact match needs no special case.
        ("regex-2026.9.3-cp313-cp313-win_arm64.whl", "cp313", "cp313", True),
        # A wheel built for another interpreter is not this venv's business either way.
        ("regex-2026.9.3-cp312-cp312-win_arm64.whl", "cp313", "cp313", False),
    ],
)
def test_a_wheel_is_usable_only_where_it_actually_imports(wheel, py_tag, abi_tag, usable):
    out = _run(f'Test-WoaWheelTagsUsable -Name "{wheel}" -PyTag "{py_tag}" -AbiTag "{abi_tag}"')
    assert out == str(usable), f"{wheel} on {py_tag}/{abi_tag}: expected {usable}, got {out}"


@requires_pwsh
@pytest.mark.parametrize(
    ("env", "reaches"),
    [
        ({}, True),
        # Bug 3: these REPLACE the default index, so PyPI is not consulted at all.
        ({"UV_DEFAULT_INDEX": "https://corp.example.com/simple"}, False),
        ({"UV_INDEX_URL": "https://corp.example.com/simple"}, False),
        ({"PIP_INDEX_URL": "https://corp.example.com/simple"}, False),
        # Pointed at PyPI explicitly is still PyPI.
        ({"UV_DEFAULT_INDEX": "https://pypi.org/simple"}, True),
        ({"UV_OFFLINE": "1"}, False),
        ({"PIP_NO_INDEX": "1"}, False),
        # An unset-looking value must not read as "offline".
        ({"UV_OFFLINE": "0"}, True),
        ({"PIP_NO_INDEX": "false"}, True),
    ],
)
def test_pypi_counts_only_when_the_resolve_would_reach_it(env, reaches):
    sets = "".join(f'$env:{key} = "{value}"; ' for key, value in env.items())
    assert _run(sets + "Test-WoaResolveReachesPyPI") == str(reaches)


@requires_pwsh
def test_a_wheel_taken_from_pypi_is_not_then_excluded_on_arm64():
    """Bug 1, the one that matters: skipping our copy must not read as "no wheel exists"."""
    nothing_anywhere = _dropped_names("@{}")
    assert {"brotli", "hf-transfer", "hf_transfer"} <= nothing_anywhere, (
        f"a name with no wheel anywhere must be dropped on ARM64, got {nothing_anywhere}"
    )

    from_pypi = _dropped_names('@{ "brotli" = @("1.3.0"); "hf-transfer" = @("0.2.0") }')
    assert not ({"brotli", "hf-transfer", "hf_transfer"} & from_pypi), (
        f"PyPI supplies these, so they must not be excluded on ARM64; dropped {from_pypi}"
    )
    # What is genuinely unavailable is still dropped, so the fix did not blanket-allow.
    assert {"xformers", "torchcodec", "brotlicffi"} <= from_pypi


@requires_pwsh
@pytest.mark.parametrize(
    ("version", "still_dropped"),
    [("0.0.20", True), ("0.0.30", False)],
)
def test_a_pypi_version_below_a_floor_keeps_the_drop(version, still_dropped):
    """xformers carries a floor because the released metadata does, so a PyPI version below
    it is not a usable answer and the drop has to stay."""
    dropped = _dropped_names(f'@{{ "xformers" = @("{version}") }}')
    assert ("xformers" in dropped) is still_dropped
