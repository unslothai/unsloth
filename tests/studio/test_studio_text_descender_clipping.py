"""Regression guard: Studio text spans must not pair `leading-none` with
`truncate`, which clips glyph descenders (g, p, q, y, j) in visible labels.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[2]
MODEL_SELECTOR = (
    WORKDIR / "studio" / "frontend" / "src" / "components" / "assistant-ui" / "model-selector.tsx"
)
APP_SIDEBAR = WORKDIR / "studio" / "frontend" / "src" / "components" / "app-sidebar.tsx"


def _read(path: Path) -> str:
    assert path.exists(), f"missing source file: {path}"
    return path.read_text()


def test_model_selector_trigger_label_uses_leading_tight():
    src = _read(MODEL_SELECTOR)
    pattern = re.compile(
        r'<span\s+className="[^"]*\bmin-w-0\b[^"]*\bflex-1\b[^"]*\btruncate\b[^"]*\bfont-heading\b[^"]*\btext-\[16px\][^"]*"',
    )
    matches = pattern.findall(src)
    assert matches, "could not find ModelSelectorTrigger model-name span"
    for cls in matches:
        assert "leading-tight" in cls, f"expected leading-tight, got: {cls}"
        assert "leading-none" not in cls, f"leading-none must not coexist with truncate here: {cls}"


def test_sidebar_account_block_uses_leading_tight():
    src = _read(APP_SIDEBAR)
    # Identify each account-block parent div by its class SET (order-independent),
    # not a positional regex: a flex-col column hidden when the sidebar collapses
    # to icons. EVERY such block must carry leading-tight and never leading-none,
    # checked per block -- a filter-then-"any" check would let one block drop the
    # class and hide behind another, defeating the per-block descender guard.
    div_classes = re.findall(r'<div\s+[^>]*className="([^"]*)"', src)
    markers = {"flex", "flex-col", "group-data-[collapsible=icon]:hidden"}
    blocks = [c.split() for c in div_classes if markers <= set(c.split())]
    assert blocks, "could not find sidebar account-block parent div"
    for classes in blocks:
        assert "leading-tight" in classes, f"sidebar account-block must use leading-tight, got: {classes}"
        assert "leading-none" not in classes, f"leading-none must not coexist with truncate here: {classes}"


def test_no_truncate_plus_leading_none_in_changed_files():
    for path in (MODEL_SELECTOR, APP_SIDEBAR):
        src = _read(path)
        for line in src.splitlines():
            if "truncate" in line and "leading-none" in line:
                raise AssertionError(
                    f"{path.name}: same line uses truncate + leading-none, descenders will clip: {line.strip()}"
                )
