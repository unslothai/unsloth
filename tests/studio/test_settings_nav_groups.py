# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Settings rail renders only tabs listed in a nav group: a tab left out of every group is
defined, routable and searchable, but never shown."""

from __future__ import annotations

import re
from pathlib import Path

SETTINGS = (
    Path(__file__).resolve().parents[2] / "studio" / "frontend" / "src" / "features" / "settings"
)


def _tabs() -> list[str]:
    text = (SETTINGS / "stores" / "settings-dialog-store.ts").read_text(encoding = "utf-8")
    block = text[text.index("export const SETTINGS_TABS = [") :]
    block = block[: block.index("] as const")]
    return re.findall(r'"([a-z0-9-]+)"', block)


def _grouped() -> list[str]:
    text = (SETTINGS / "settings-dialog.tsx").read_text(encoding = "utf-8")
    block = text[text.index("export const SETTINGS_NAV_GROUPS") :]
    block = block[block.index("= [") : block.index("\n];")]
    out: list[str] = []
    for ids in re.findall(r"ids:\s*\[(.*?)\]", block, re.S):
        out += re.findall(r'"([a-z0-9-]+)"', ids)
    return out


def test_every_settings_tab_is_in_exactly_one_nav_group() -> None:
    tabs, grouped = _tabs(), _grouped()
    assert "machines" in tabs
    missing = sorted(set(tabs) - set(grouped))
    assert not missing, f"never shown in the Settings rail: {missing}"
    assert sorted(set(grouped) - set(tabs)) == []
    assert len(grouped) == len(set(grouped)), "a tab listed in two groups"
