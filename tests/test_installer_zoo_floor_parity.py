# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The installer's `--no-deps` unsloth-zoo specs are a floor in their own right, so they must
match pyproject.toml's.

`uv pip install --no-deps` never reads unsloth's metadata, so on those paths the
`unsloth-zoo>=...` written on the command line is the ONLY thing deciding which zoo the user
ends up with. When pyproject.toml moves and the installers do not, a no-torch install happily
resolves a zoo that pyproject.toml already rejects, and the defect the floor was raised for is
still shipped. That is not hypothetical: main carried `unsloth-zoo>=2026.9.3` in the
installers while pyproject.toml said `>=2026.9.4`.

The with-deps sites are deliberately NOT checked. There the resolver intersects the
command-line spec with unsloth's own metadata, so pyproject.toml is already the binding floor
and a lower spec on the command line changes nothing.

Reads files only, so this runs on the Windows and macOS runners too.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version


REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
INSTALL_SH = REPO / "install.sh"
INSTALL_PS1 = REPO / "install.ps1"

# Every `--no-deps` site that names a zoo floor, one per installer branch:
#   install.sh   migrated + SKIP_TORCH, fresh + SKIP_TORCH
#   install.ps1  migrated + SkipTorch, fresh + SkipTorch
# Pinned so that adding a fifth without thinking about the floor fails here rather than
# shipping a path nothing checks.
EXPECTED_NO_DEPS_SITES = 4

_ZOO_SPEC = re.compile(r"""unsloth[-_]zoo\s*(?:>=|==)\s*[0-9][^"'\s]*""")


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    """Join shell backslash continuations so one command is one string."""
    out: list[tuple[int, str]] = []
    buffer = ""
    start = 0
    for number, raw in enumerate(path.read_text(encoding = "utf-8").splitlines(), start = 1):
        if not buffer:
            start = number
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += stripped
        out.append((start, buffer))
        buffer = ""
    if buffer:
        out.append((start, buffer))
    return out


def _pyproject_floor() -> Version:
    data = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))
    project = data.get("project") or {}
    raws: list[str] = list(project.get("dependencies") or [])
    for extra in (project.get("optional-dependencies") or {}).values():
        raws.extend(extra)
    floors: dict[str, Version] = {}
    for raw in raws:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower().replace("_", "-") != "unsloth-zoo":
            continue
        lower = [
            Version(str(spec.version))
            for spec in req.specifier
            if spec.operator in (">=", "==", "~=")
        ]
        if lower:
            floors[raw] = max(lower)
    assert floors, "pyproject.toml declares no versioned unsloth_zoo requirement at all"
    assert len(set(floors.values())) == 1, (
        f"pyproject.toml declares more than one unsloth_zoo floor: "
        f"{ {raw: str(v) for raw, v in floors.items()} }. One of them is the one users "
        f"actually resolve, and which one depends on the extra they picked."
    )
    return next(iter(floors.values()))


def _installer_sites() -> list[tuple[str, int, str, bool]]:
    """(file, line, spec, is_no_deps) for every zoo version spec on an install command line."""
    sites: list[tuple[str, int, str, bool]] = []
    for path in (INSTALL_SH, INSTALL_PS1):
        lines = (
            _logical_lines(path)
            if path.suffix == ".sh"
            else list(enumerate(path.read_text(encoding = "utf-8").splitlines(), start = 1))
        )
        for number, text in lines:
            if "pip install" not in text:
                continue
            no_deps = "--no-deps" in text
            for spec in _ZOO_SPEC.findall(text):
                sites.append((path.name, number, spec, no_deps))
    return sites


def test_the_no_deps_installer_sites_carry_the_pyproject_floor() -> None:
    floor = _pyproject_floor()
    sites = [site for site in _installer_sites() if site[3]]
    assert sites, (
        "found no --no-deps unsloth-zoo spec in either installer. Either the no-torch paths "
        "were reworked, in which case retune this test, or the floor they carried was "
        "dropped, in which case those installs resolve any zoo at all."
    )
    assert len(sites) == EXPECTED_NO_DEPS_SITES, (
        f"expected {EXPECTED_NO_DEPS_SITES} --no-deps zoo floors, found {len(sites)}: "
        f"{[(f, n, s) for f, n, s, _ in sites]}. A new one needs the floor too, since "
        f"--no-deps means the command line is all the resolver sees."
    )
    stale = [
        (name, number, spec)
        for name, number, spec, _ in sites
        if Version(spec.split(">=")[-1].split("==")[-1]) < floor
    ]
    assert not stale, (
        f"these --no-deps installer sites still accept an unsloth_zoo older than the "
        f"pyproject.toml floor {floor}: {stale}. --no-deps ignores package metadata, so such "
        f"an install resolves a zoo pyproject.toml already rejects."
    )


def test_every_installer_zoo_spec_is_a_real_version() -> None:
    """A typo'd spec would silently become a name-only requirement and pin nothing."""
    sites = _installer_sites()
    assert sites, "no unsloth-zoo version spec found in the installers at all"
    for name, number, spec, _ in sites:
        try:
            Requirement(spec)
        except InvalidRequirement as error:  # pragma: no cover - only on a malformed spec
            pytest.fail(f"{name}:{number} has an unparseable zoo spec {spec!r}: {error}")
