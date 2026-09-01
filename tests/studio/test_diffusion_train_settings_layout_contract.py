# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layout contract for the Images -> Train "Training settings" grid.

The panel is a fraction of the window, so a viewport breakpoint put three columns
in a ~280px pane; and each cell was a bare `grid`, whose implicit column is
auto-sized and froze at its widest child's min-content, so the cell painted over
its neighbour instead of shrinking. Both fixes have to stay: the container query
keeps cells wide enough, grid-cols-1 keeps a cell inside its column if it ever
is not.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PANEL_TSX = REPO / "studio/frontend/src/features/images/train/diffusion-train-panel.tsx"


def _source() -> str:
    return PANEL_TSX.read_text(encoding = "utf-8")


def test_settings_cell_cannot_outgrow_its_grid_column():
    assert 'const fieldClass = "grid grid-cols-1 min-w-0 gap-2";' in _source()


def test_settings_columns_key_off_the_pane_width_not_the_viewport():
    source = _source()

    # The run area declares itself the query container...
    assert '<div className="@container flex flex-col gap-6">' in source
    # ...and all three settings grids step up on ITS width.
    # 324px fits two 150px cells plus the 24px gutter, 498px fits three.
    assert source.count("@min-[324px]:grid-cols-2 @min-[498px]:grid-cols-3") == 3
    # No viewport breakpoint left behind: that is what put 3 columns in a 280px pane.
    assert "lg:grid-cols-3" not in source


def test_field_label_ellipses_instead_of_cutting_mid_glyph():
    # Label's own display is flex and text-overflow does nothing on a flex container, so truncate only ellipses when
    assert '<Label className="block min-w-0 truncate text-xs">{children}</Label>' in _source()
