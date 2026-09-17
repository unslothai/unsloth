# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The installer's --no-deps unsloth-zoo floor has to keep step with pyproject.toml.

pyproject's `unsloth_zoo>=X` is metadata, and `uv pip install --no-deps` ignores metadata
by definition. On those paths the literal `unsloth-zoo>=X` the installer passes on the
command line is the ONLY floor there is, so an index or a constraint that offers an older
zoo installs it happily and the pair goes out of step at runtime.

The two floors have now drifted twice, so this test ties them together: every --no-deps
installer command that names unsloth-zoo must ask for at least what pyproject asks for.

Sites WITHOUT --no-deps are deliberately not covered. There the resolver reads unsloth's
own metadata and intersects it with the command-line spec, so the pyproject floor is
already enforced and the literal cannot select a zoo below it.

Reads files only, so it runs on the Windows and macOS runners too.
"""

from __future__ import annotations

import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_ZOO_FLOOR = re.compile(r"unsloth[-_]zoo\s*>=\s*([0-9][0-9.]*)")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _logical_lines(path: pathlib.Path) -> list[str]:
    """Backslash continuations joined, so a multi-line uv invocation is one string."""
    joined: list[str] = []
    buffer = ""
    for line in path.read_text(encoding = "utf-8").splitlines():
        stripped = line.strip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _pip_install_commands(path: pathlib.Path) -> list[str]:
    """Every pip install invocation naming an unsloth-zoo floor."""
    return [
        line for line in _logical_lines(path) if "pip install" in line and _ZOO_FLOOR.search(line)
    ]


def _pyproject_floor() -> tuple[int, ...]:
    floors = [
        _version(match.group(1))
        for line in PYPROJECT.read_text(encoding = "utf-8").splitlines()
        for match in [_ZOO_FLOOR.search(line)]
        if match
    ]
    assert floors, "pyproject.toml names no unsloth_zoo floor"
    return max(floors)


@pytest.mark.parametrize("installer", [INSTALL_SH, INSTALL_PS1], ids = ["install.sh", "install.ps1"])
def test_no_deps_zoo_floor_is_not_below_pyproject(installer: pathlib.Path) -> None:
    floor = _pyproject_floor()
    no_deps = [cmd for cmd in _pip_install_commands(installer) if "--no-deps" in cmd]
    assert no_deps, f"{installer.name}: no --no-deps command names unsloth-zoo"
    for command in no_deps:
        asked = _ZOO_FLOOR.search(command).group(1)
        assert _version(asked) >= floor, (
            f"{installer.name}: --no-deps asks for unsloth-zoo>={asked}, below pyproject's "
            f"{'.'.join(str(part) for part in floor)}. --no-deps ignores metadata, so this "
            f"literal is the only floor on that path."
        )


@pytest.mark.parametrize("installer", [INSTALL_SH, INSTALL_PS1], ids = ["install.sh", "install.ps1"])
def test_every_installer_zoo_floor_is_a_recognised_pin(installer: pathlib.Path) -> None:
    """Guards the parser above: both installers still carry the sites it reads."""
    assert (
        len(_pip_install_commands(installer)) >= 4
    ), f"{installer.name}: fewer unsloth-zoo pins than expected, the parser may be stale"
