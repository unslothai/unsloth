"""Regression guard: Unsloth text spans must not pair `leading-none` with
`truncate`, which clips glyph descenders (g, p, q, y, j) in visible labels.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[2]
MODEL_SELECTOR = (
    WORKDIR
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "model-picker"
    / "components"
    / "model-selector.tsx"
)
APP_SIDEBAR = WORKDIR / "studio" / "frontend" / "src" / "components" / "app-sidebar.tsx"


def _read(path: Path) -> str:
    assert path.exists(), f"missing source file: {path}"
    return path.read_text()


def test_model_selector_trigger_label_uses_leading_tight():
    src = _read(MODEL_SELECTOR)
    pattern = re.compile(
        r'<span\s+className="[^"]*\bmin-w-0\b[^"]*\bflex-1\b[^"]*\btruncate\b[^"]*\bfont-heading\b[^"]*\btext-ui-16[^"]*"',
    )
    matches = pattern.findall(src)
    assert matches, "could not find ModelSelectorTrigger model-name span"
    for cls in matches:
        assert "leading-tight" in cls, f"expected leading-tight, got: {cls}"
        assert "leading-none" not in cls, f"leading-none must not coexist with truncate here: {cls}"


def test_sidebar_account_block_uses_leading_tight():
    src = _read(APP_SIDEBAR)
    class_names = re.findall(r'<div\s+className="([^"]+)"', src)
    required = {
        "flex",
        "flex-1",
        "flex-col",
        "group-data-[collapsible=icon]:hidden",
    }
    matches = [classes for classes in class_names if required <= set(classes.split())]
    assert matches, "could not find sidebar account-block parent div"
    for classes in matches:
        leading_classes = [cls for cls in classes.split() if cls.startswith("leading-")]
        assert leading_classes, f"no leading-* class on sidebar account-block parent: {classes}"
        for cls in leading_classes:
            assert (
                cls == "leading-tight"
            ), f"sidebar account-block must use leading-tight, got: {cls}"


def test_no_truncate_plus_leading_none_in_changed_files():
    for path in (MODEL_SELECTOR, APP_SIDEBAR):
        src = _read(path)
        for line in src.splitlines():
            if "truncate" in line and "leading-none" in line:
                raise AssertionError(
                    f"{path.name}: same line uses truncate + leading-none, descenders will clip: {line.strip()}"
                )
