# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who paints over whom in the bottom-right corner.

The Live resource monitor, the API monitor panel and the notification stack all
live there. The first two keep out of each other's way geometrically
(panel-placement and panel-placement.test.ts); the stack does not move for
anyone -- it is anchored to the corner in CSS, because placing it from the
boxes the others publish is what moved it to the middle and the top of the
window. So the corner is shared, and z-order is the only thing deciding whether
a monitor sitting under the stack still has a clickable Close button.

The Windows UI smoke does exactly that drag-and-resize and then clicks Close, so
it catches a regression here for real. It takes about twenty minutes and needs a
Windows runner. These read the numbers straight out of the source instead.

The numbers themselves live in one place now, studio/frontend/src/lib/z-layers.ts.
These tests compare them rather than pinning any one of them, so renumbering a
layer does not break them -- what breaks them is a surface leaving the named scale
or two layers swapping places.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    assert "style={{ zIndex }}" in container, (
        f"{path.name}: the panel container no longer takes its z-index from the "
        f"floating panel order store: {container!r}"
    )
    assert not _Z.search(container), (
        f"{path.name}: the panel container still carries a hard-coded z-index, "
        f"which would win over the shared layer: {container!r}"
    )
    src = path.read_text(encoding = "utf-8")
    assert (
        "useFloatingPanelZIndex" in src
    ), f"{path.name}: the panel no longer reads the shared floating panel layer"


def test_the_notification_stack_uses_the_named_layer():
    """Both copies, browser and desktop. They drifted apart once already."""
    src = PROVIDER.read_text(encoding = "utf-8")
    stacks = re.findall(r'"pointer-events-none fixed bottom-0 right-4 ([^"]*)"', src)
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
    """The startup screen blocks the app while the backend comes up, and tooltips
    are transient and have to be readable above whatever spawned them. Both are
    still Tailwind classes, so check the literal against the scale as well as the
    ordering -- otherwise the scale can say one thing and the class another."""
    layers = _layers()
    assert (
        layers[layer] > layers["FLOATING_PANEL_TOP"]
    ), f"{layer} no longer outranks the floating panels"
    assert layers[layer] in _z_indexes(
        path
    ), f"{path.name} has drifted from Z_LAYER.{layer} ({layers[layer]})"
