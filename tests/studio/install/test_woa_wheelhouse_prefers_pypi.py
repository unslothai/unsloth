# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The wheelhouse is for wheels PyPI does not build, and only for those.

The Windows on ARM wheelhouse exists because PyPI publishes no win_arm64 build of
pyarrow, tiktoken, grpcio, brotli, hf_transfer or sqlite-vec. regex is a different
case: PyPI has shipped win_arm64 regex since 2025.7.29, so a copy in the wheelhouse
is a second source for an artifact upstream already ships. It would win, too, because
the staging directory is first in UV_FIND_LINKS, so a user asking for regex would get
our binary rather than the one the project released.

That is the shape these tests pin. Staging a wheel from the wheelhouse asks PyPI
first, and skips the copy when PyPI publishes the same project at or above the
version that would have been staged. The version half matters as much as the check:
an upstream that is BEHIND the wheelhouse has to leave ours in place, or the guard
turns into a downgrade.

Text-level tests, like the rest of the installer suite: install.ps1 is PowerShell,
so the shape is asserted against its source rather than by running it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


@pytest.fixture(scope = "module")
def source() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8-sig")


def _function_body(source: str, name: str) -> str:
    """One nested `    function Name {` block, to its closing `    }`."""
    match = re.search(
        r"(?ms)^    function " + re.escape(name) + r"\b.*?^    \}$", source
    )
    assert match, f"install.ps1 no longer defines {name}"
    return match.group(0)


def _staging_loop(source: str) -> str:
    """The generic wheelhouse mirror: the block that copies everything but pyarrow."""
    match = re.search(
        r"(?ms)^        \$WoaExtraStaged = 0$.*?^        if \(\$WoaExtraStaged -gt 0\)", source
    )
    assert match, "the wheelhouse staging loop moved"
    return match.group(0)


def test_the_pypi_probe_is_pypi_only(source):
    """Test-WoaWheelAvailable falls back to the wheelhouse, so staging cannot use it: a
    wheel is always in the wheelhouse it is being staged from, and the guard would never
    fire. The probe staging uses has to reach PyPI and nothing else."""
    body = _function_body(source, "Test-WoaPyPIWheel")
    assert "pypi.org/simple" in body
    for other in ("WoaWheelhouse", "index.txt", "Get-ChildItem"):
        assert other not in body, f"Test-WoaPyPIWheel consults {other}; it must ask PyPI alone"


def test_both_staging_branches_skip_what_pypi_publishes(source):
    """The local-directory branch and the URL branch both mirror the wheelhouse, so a
    guard on one of them leaves the other shipping our copy."""
    loop = _staging_loop(source)
    calls = re.findall(r"Test-WoaWheelhouseWheelIsRedundant", loop)
    assert len(calls) == 2, (
        f"expected the redundancy guard in both staging branches, found {len(calls)}"
    )


def test_the_guard_is_version_aware(source):
    """Published is not enough. If upstream is behind the wheelhouse, dropping ours
    downgrades the install, so the floor is the staged wheel's own version."""
    body = _function_body(source, "Test-WoaWheelhouseWheelIsRedundant")
    assert "-Floor $fields[1]" in body, "the guard must floor PyPI at the staged version"
    probe = _function_body(source, "Test-WoaPyPIWheel")
    assert "Test-WoaVersionAtLeast" in probe, "the probe must compare versions, not just presence"


def test_the_guard_only_judges_wheels_this_venv_could_use(source):
    """A cp312 wheel in the wheelhouse is not made redundant by a cp313 wheel on PyPI."""
    body = _function_body(source, "Test-WoaWheelhouseWheelIsRedundant")
    assert "Test-WoaWheelTagsUsable" in body


def test_interpreter_agnostic_wheels_still_count(source):
    """hf_transfer ships cp38-abi3 and sqlite_vec ships py3-none, both usable on cp313.
    An exact-tag test would call them foreign and never consider them redundant, so the
    day upstream publishes one we would go on shipping ours."""
    body = _function_body(source, "Test-WoaWheelTagsUsable")
    assert '"abi3"' in body and '"none"' in body
    # Free-threaded venvs are the exception the exact-tag helper exists for.
    assert '$AbiTag -like "*t"' in body


def test_pyarrow_keeps_its_own_pypi_first_path(source):
    """Get-WoaPyarrowSource already asks PyPI before the wheelhouse and returns which one
    it used; the generic loop skips pyarrow for exactly that reason. Both halves have to
    stay, or pyarrow is either staged twice or not probed upstream at all."""
    body = _function_body(source, "Get-WoaPyarrowSource")
    assert 'return "pypi"' in body
    loop = _staging_loop(source)
    assert re.search(r'-like "pyarrow-\*"\) \{ continue \}', loop), (
        "the generic loop must leave pyarrow to Get-WoaPyarrowSource"
    )
