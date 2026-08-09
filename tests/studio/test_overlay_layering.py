# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who paints over whom in the bottom-right corner.

The floating API monitor and the notification stack both live there. The stack
tries to dodge the monitor (see stackGeometry and monitor-stack-inset.test.ts),
but the dodge has a floor: a monitor dragged to the corner and resized to fill
the viewport leaves nowhere to dodge to, and the stack is parked at the top of
the screen, on top of the monitor's own title bar. At that point z-order is the
only thing deciding whether the monitor's Close button can be clicked.

The Windows UI smoke does exactly that drag-and-resize and then clicks Close, so
it catches a regression here for real. It takes about twenty minutes and needs a
Windows runner. These read the numbers straight out of the source instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
MONITOR = FRONTEND / "components/floating-monitor.tsx"
PROVIDER = FRONTEND / "app/provider.tsx"
STARTUP = FRONTEND / "components/tauri/startup-screen.tsx"
TOOLTIP = FRONTEND / "components/ui/tooltip.tsx"

_Z = re.compile(r"z-\[(\d+)\]")


def _z_indexes(path: Path) -> list[int]:
    return [int(m) for m in _Z.findall(path.read_text(encoding = "utf-8"))]


def _monitor_root_z() -> int:
    """The z on the monitor's constraints container, which is what stacks it."""
    src = MONITOR.read_text(encoding = "utf-8")
    root = re.search(r"ref=\{setConstraintsElement\}\s*\n\s*className=\"([^\"]+)\"", src)
    assert root, "the monitor's constraints container was not found"
    found = _Z.search(root.group(1))
    assert found, f"the monitor's container carries no z-\\[n\\]: {root.group(1)!r}"
    return int(found.group(1))


def _stack_z() -> int:
    """The bottom-right overlay stack. Both copies, browser and desktop."""
    src = PROVIDER.read_text(encoding = "utf-8")
    stacks = re.findall(r'className="pointer-events-none fixed right-4 z-\[(\d+)\]', src)
    assert stacks, "the bottom-right overlay stack was not found"
    assert len(set(stacks)) == 1, f"the two stacks disagree on z-index: {stacks}"
    return int(stacks[0])


def test_the_monitor_paints_over_the_notification_stack():
    """A full-viewport monitor parks the stack over its own title bar. The stack is
    passive status; the monitor is a window being dragged, resized and closed."""
    assert _monitor_root_z() > _stack_z(), (
        "the notification stack paints over the floating monitor, so its Close "
        "button cannot be clicked once the monitor fills the viewport"
    )


@pytest.mark.parametrize("path", [STARTUP, TOOLTIP])
def test_the_monitor_stays_under_the_layers_that_outrank_it(path: Path):
    """Raising the monitor must not put it over the startup screen, which blocks
    the app while the backend comes up, or over tooltips, which are transient and
    have to be readable above whatever spawned them."""
    above = [z for z in _z_indexes(path) if z >= _monitor_root_z()]
    assert above, f"{path.name} no longer outranks the floating monitor"
