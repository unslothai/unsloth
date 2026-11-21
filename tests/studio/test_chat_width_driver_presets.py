# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""playwright_chat_width.py picks the chat width options by position, so the component's value order is its contract."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = (
    ROOT
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "settings"
    / "components"
    / "appearance-custom-controls.tsx"
)
DRIVER = Path(__file__).resolve().parent / "playwright_chat_width.py"


def _component_order():
    source = COMPONENT.read_text(encoding = "utf-8")
    found = re.findall(r"\(\[([^\]]*)\] as const\)\.map\(\(width\)", source)
    assert (
        len(found) == 1
    ), "the chat width options are no longer one `[...] as const` map over `width`"
    return tuple(part.strip().strip("\"'") for part in found[0].split(","))


def _driver_presets():
    tree = ast.parse(DRIVER.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CHAT_WIDTH_PRESETS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("CHAT_WIDTH_PRESETS is gone from playwright_chat_width.py")


def test_the_driver_presets_follow_the_component_order():
    assert _component_order() == ("standard", "wide", "full")
    assert len(_driver_presets()) == len(_component_order())
