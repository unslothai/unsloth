# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_python_stack.py step 11 installs every seed plugin unconditionally, so each plugin's
requires-python must admit the oldest Python Studio supports, or setup aborts and rolls back there."""

import pathlib
import re

import pytest
from packaging.specifiers import SpecifierSet

ROOT = pathlib.Path(__file__).resolve().parents[3]
PLUGINS = ROOT / "studio" / "backend" / "plugins"
BACKEND_CI = ROOT / ".github" / "workflows" / "studio-backend-ci.yml"


def _studio_python_floor() -> str:
    text = BACKEND_CI.read_text(encoding = "utf-8")
    match = re.search(r"^\s*PYTHON_FLOOR:\s*['\"]?(\d+\.\d+)", text, re.M)
    assert match, "studio-backend-ci.yml declares PYTHON_FLOOR"
    return match.group(1)


def _plugin_pyprojects():
    found = sorted(PLUGINS.glob("*/pyproject.toml"))
    assert found, "no seed plugins found"
    return found


@pytest.mark.parametrize("pyproject", _plugin_pyprojects(), ids = lambda p: p.parent.name)
def test_seed_plugin_admits_the_studio_python_floor(pyproject):
    text = pyproject.read_text(encoding = "utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    assert match, f"{pyproject.parent.name} declares requires-python"
    floor = _studio_python_floor()
    assert SpecifierSet(match.group(1)).contains(f"{floor}.0"), (
        f"{pyproject.parent.name} requires-python {match.group(1)!r} rejects Python {floor}, "
        "which install.sh accepts; the setup step installing it would fail and roll back"
    )
