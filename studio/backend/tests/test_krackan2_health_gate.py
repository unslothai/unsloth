# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A lone Linux Krackan Point 2 (gfx1153) gets CPU torch on purpose, so the health gate must not
read it as a gfx1152 the installers would have served and prompt for a repair."""

from __future__ import annotations

import pytest

import utils.hardware.hardware as hw


@pytest.mark.parametrize(
    "system,name,expected",
    [
        ("Linux", "AMD Radeon 820M", False),
        ("Linux", "AMD Ryzen AI 5 330 w/ Radeon Graphics", False),
        ("Linux", "AMD Radeon 860M", True),
        ("Windows", "AMD Radeon 820M", True),
    ],
)
def test_a_linux_krackan_point_2_cannot_establish_a_mismatch(monkeypatch, system, name, expected):
    # Windows routing is unchanged: the multi-arch route serves gfx1153 there.
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    assert hw._amd_device_can_establish_a_mismatch({"name": name}) is expected
