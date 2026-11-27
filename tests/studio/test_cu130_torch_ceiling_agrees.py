# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""install.ps1 keeps a resident cu130 torch below $Cu130TorchCeiling (exclusive) and studio/setup.ps1
keeps one inside its inclusive cu130 minor range. Both must name the same top release, and install.sh
and the Pester copy of the ceiling must match install.ps1, so one bump cannot leave the others behind."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding = "utf-8")


def _ceiling_minor(text: str, pattern: str) -> int:
    match = re.search(pattern, text, re.M)
    assert match, pattern
    major, minor, patch = (int(x) for x in match.group(1).split("."))
    assert (major, patch) == (2, 0), match.group(1)
    return minor


def test_setup_ps1_cu130_kept_range_tops_out_below_install_ps1_ceiling():
    ceiling = _ceiling_minor(
        _read("install.ps1"), r'^\s*\$script:Cu130TorchCeiling\s*=\s*"([\d.]+)"'
    )
    kept = re.search(r'"cu130"\s*=\s*@\((\d+),\s*(\d+)\)', _read("studio/setup.ps1"))
    assert kept, "cu130 kept range not found in studio/setup.ps1"
    low, high = int(kept.group(1)), int(kept.group(2))
    assert low <= high
    assert (
        high == ceiling - 1
    ), f"studio/setup.ps1 keeps cu130 up to 2.{high}, install.ps1 below 2.{ceiling}.0"


def test_cu130_ceiling_copies_match_install_ps1():
    ceiling = _ceiling_minor(
        _read("install.ps1"), r'^\s*\$script:Cu130TorchCeiling\s*=\s*"([\d.]+)"'
    )
    assert ceiling == _ceiling_minor(_read("install.sh"), r'^_CU130_TORCH_CEILING="([\d.]+)"')
    assert ceiling == _ceiling_minor(
        _read("tests/studio/test_cu130_torch213_route.ps1"),
        r'^\s*\$script:Cu130TorchCeiling\s*=\s*"([\d.]+)"',
    )
