# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""pymupdf ships wheels for a few platforms per release (2.0 has no win_arm64 one), so a lower bound alone can
resolve to a release that only has an sdist and build MuPDF from source. Every platform gets one exact pin,
the same in every file that installs it."""

from __future__ import annotations

import pathlib
import tomllib

import pytest
from packaging.requirements import Requirement

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"

PLATFORMS = {
    "win_arm64": {"sys_platform": "win32", "platform_machine": "ARM64"},
    "win_amd64": {"sys_platform": "win32", "platform_machine": "AMD64"},
    "linux_x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "linux_aarch64": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "macos_arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
}


def _requirement_files():
    for name in ("studio.txt", "no-torch-runtime.txt"):
        path = REQ_ROOT / name
        lines = path.read_text(encoding = "utf-8").splitlines()
        yield (
            name,
            [
                line.split(" #")[0].strip()
                for line in lines
                if line.strip() and not line.startswith("#")
            ],
        )
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))["project"]
    yield "pyproject.toml [studio]", list(project["optional-dependencies"]["studio"])


def _pymupdf_for(lines, environment):
    env = {"python_version": "3.12", "python_full_version": "3.12.0", **environment}
    found = []
    for line in lines:
        requirement = Requirement(line)
        if requirement.name.lower() != "pymupdf":
            continue
        if requirement.marker is None or requirement.marker.evaluate(env):
            found.append(requirement)
    return found


@pytest.mark.parametrize("platform", sorted(PLATFORMS))
def test_every_file_pins_one_exact_pymupdf_per_platform(platform):
    pins = {}
    for source, lines in _requirement_files():
        found = _pymupdf_for(lines, PLATFORMS[platform])
        assert len(found) == 1, (source, platform, [str(r) for r in found])
        specifiers = list(found[0].specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "==", (
            f"{source} lets {platform} resolve pymupdf {found[0].specifier}: a newer release without a "
            f"{platform} wheel would build MuPDF from source. Pin it with ==."
        )
        pins[source] = specifiers[0].version
    assert len(set(pins.values())) == 1, (platform, pins)


def test_win_arm64_stays_on_the_release_that_has_its_wheel():
    # 1.28.2 is the only pymupdf release with a win_arm64 wheel; 1.27.x and 2.0 have none.
    for source, lines in _requirement_files():
        (requirement,) = _pymupdf_for(lines, PLATFORMS["win_arm64"])
        assert str(requirement.specifier) == "==1.28.2", (source, str(requirement))
