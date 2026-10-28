# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Z-order alone decides whether a monitor under the notification stack has a clickable Close button."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jsx_tags import opening_tag, without_comments  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
Z_LAYERS = FRONTEND / "lib/z-layers.ts"
MONITOR = FRONTEND / "components/floating-monitor.tsx"
API_PANEL = FRONTEND / "features/api-monitor/api-monitor-overlay.tsx"
PROVIDER = FRONTEND / "app/provider.tsx"
STARTUP = FRONTEND / "components/tauri/startup-screen.tsx"
TOOLTIP = FRONTEND / "components/ui/tooltip.tsx"

_Z = re.compile(r"z-\[(\d+)\]")
_LAYER = re.compile(r"^\s{2}([A-Z_]+): (\d+),$", re.MULTILINE)


def _layers() -> dict[str, int]:
    """The named scale, straight out of z-layers.ts."""
    text = Z_LAYERS.read_text(encoding = "utf-8")
    found = {name: int(value) for name, value in _LAYER.findall(text)}
    assert found, "no named layers were found in z-layers.ts"
    return found


def _z_indexes(path: Path) -> list[int]:
    return [int(m) for m in _Z.findall(path.read_text(encoding = "utf-8"))]


def _container(path: Path, ref: str) -> str:
    """The JSX attributes of the fixed container that stacks a floating panel."""
    src = path.read_text(encoding = "utf-8")
    found = re.search(rf"ref=\{{{ref}\}}\s*\n(?:\s*[^\n]*\n)*?\s*>", src)
    assert found, f"{path.name}: the container with ref={{{ref}}} was not found"
    return found.group(0)


def test_the_floating_panels_paint_over_the_notification_stack():
    """The stack holds its corner, so a monitor parked there is under it. The stack
    is passive status; the panels are windows being dragged, resized and closed."""
    layers = _layers()
    assert layers["FLOATING_PANEL"] > layers["OVERLAY_STACK"], (
        "the notification stack paints over the floating panels, so their Close "
        "buttons cannot be clicked once a panel fills the viewport"
    )


def test_the_front_panel_does_not_climb_past_the_layer_above_it():
    """Only one panel is ever raised, by one step, so the pair cannot straddle
    the startup screen."""
    layers = _layers()
    assert layers["FLOATING_PANEL_TOP"] == layers["FLOATING_PANEL"] + 1
    assert layers["FLOATING_PANEL_TOP"] < layers["STARTUP_SCREEN"]


@pytest.mark.parametrize("path", [MONITOR, API_PANEL])
def test_both_floating_panels_stack_on_the_shared_layer(path: Path):
    """Both containers take their z-index from the order store, not from a class.
    Sharing the layer is what lets the one the user touched last come forward,
    which is the only way out of a monitor resized over the whole viewport."""
    container = _container(path, "setConstraintsElement")
    style_applies_shared_layer = re.search(r"style=\{\{\s*zIndex\s*[,}]", container) or re.search(
        r"style=\{floatingMonitorConstraintStyle\(\{\s*zIndex,", container
    )
    assert style_applies_shared_layer, (
        f"{path.name}: the panel container no longer applies the floating panel "
        f"order store's z-index: {container!r}"
    )
    assert not _Z.search(container), (
        f"{path.name}: the panel container still carries a hard-coded z-index, "
        f"which would win over the shared layer: {container!r}"
    )
    src = path.read_text(encoding = "utf-8")
    assert (
        "useFloatingPanelZIndex" in src
    ), f"{path.name}: the panel no longer reads the shared floating panel layer"


_RAIL_TESTID = 'data-testid="overlay-rail"'


def test_the_notification_stack_uses_the_named_layer():
    """Locates both stack copies by data-testid, since class-anchored matching broke when the rail moved."""
    src = without_comments(PROVIDER.read_text(encoding = "utf-8"))
    stacks = []
    at = src.find(_RAIL_TESTID)
    while at != -1:
        # opening_tag, not rfind/find: `<` or `=>` inside attributes truncates a hand-rolled scan.
        start, end = opening_tag(src, at)
        stacks.append(src[start:end])
        at = src.find(_RAIL_TESTID, end)
    assert len(stacks) == 2, f"expected the two bottom-right stacks, found {len(stacks)}"
    for stack in stacks:
        assert not _Z.search(stack), f"the stack still carries a hard-coded z-index: {stack!r}"
    assert (
        src.count("zIndex: Z_LAYER.OVERLAY_STACK") == 2
    ), "the two bottom-right stacks disagree on their layer"


@pytest.mark.parametrize(
    ("path", "layer"),
    [(STARTUP, "STARTUP_SCREEN"), (TOOLTIP, "TOOLTIP")],
)
def test_the_layers_that_outrank_the_panels_still_do(path: Path, layer: str):
    """Also checks the Tailwind class literal against the scale, so the two cannot disagree."""
    layers = _layers()
    assert (
        layers[layer] > layers["FLOATING_PANEL_TOP"]
    ), f"{layer} no longer outranks the floating panels"
    assert layers[layer] in _z_indexes(
        path
    ), f"{path.name} has drifted from Z_LAYER.{layer} ({layers[layer]})"
