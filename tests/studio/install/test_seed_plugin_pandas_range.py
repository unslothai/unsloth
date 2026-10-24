# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installed with --no-deps, so its pandas range must admit every pandas the constraints can install."""

import pathlib
import re

from packaging.specifiers import SpecifierSet

ROOT = pathlib.Path(__file__).resolve().parents[3]
PLUGIN = (
    ROOT / "studio" / "backend" / "plugins" / "data-designer-unstructured-seed" / "pyproject.toml"
)
CONSTRAINTS = ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"


def _plugin_pandas_range():
    text = PLUGIN.read_text(encoding = "utf-8")
    match = re.search(r'"pandas([^"]*)"', text)
    assert match, "the plugin declares pandas"
    return SpecifierSet(match.group(1))


def _constraint_pandas_pins():
    pins = []
    for line in CONSTRAINTS.read_text(encoding = "utf-8").splitlines():
        if line.startswith("pandas"):
            spec = line.split(";", 1)[0][len("pandas") :].strip()
            pins.append(SpecifierSet(spec))
    assert pins, "constraints.txt pins pandas"
    return pins


def test_every_constrained_pandas_is_inside_the_plugins_range():
    plugin = _plugin_pandas_range()
    for pin in _constraint_pandas_pins():
        # The lowest version each constraint admits is the one that gets installed.
        floor = next(s.version for s in pin if s.operator in ("==", ">="))
        assert plugin.contains(
            floor
        ), f"constraints install pandas {floor}, outside the plugin's {plugin}"


def test_the_arm64_row_installs_pandas_3():
    """The row this test exists for: without it the range check would pass on pandas 2 alone."""
    rows = [
        l for l in CONSTRAINTS.read_text(encoding = "utf-8").splitlines() if l.startswith("pandas>=3")
    ]
    assert len(rows) == 1 and 'platform_machine == "ARM64"' in rows[0]
