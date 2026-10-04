# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Colab-style arrow navigation must not swallow wrapped-line movement.

`getCursorPosition().line` and `lineCount` are both LOGICAL, while JupyterLab wraps
markdown and raw editors by default, so a one-line markdown header is
line 0 == lineCount - 1 from every visual row and the wrapped rows are unreachable.

CodeMirror's own answer is `EditorView.moveVertically(range, forward)`, which returns
an unchanged head only at offset 0 / doc.length.

A static source guard: the labextension is only built inside Dockerfile.studio.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_NAV = REPO_ROOT / "docker" / "jupyter" / "unsloth_labext" / "src" / "cellNav.ts"


@pytest.fixture(scope = "module")
def source() -> str:
    assert CELL_NAV.is_file(), f"missing {CELL_NAV}"
    return CELL_NAV.read_text(encoding = "utf-8")


def test_the_edit_mode_boundary_test_asks_codemirror_for_a_visual_line(source: str):
    assert "moveVertically" in source, (
        "the edit-mode boundary check must ask CodeMirror whether it can still "
        "move one VISUAL line (EditorView.moveVertically); a logical lineCount "
        "test makes the wrapped rows of a markdown cell unreachable"
    )


def test_the_visual_check_compares_screen_rows(source: str):
    assert "coordsAtPos" in source, (
        "moveVertically clamps to the document edge instead of returning the "
        "same head, so the two positions have to be compared by visual row"
    )


def test_the_logical_line_test_is_only_a_fallback(source: str):
    body = source[source.index("const editing = notebook.mode === 'edit'") :]
    logical = re.search(r"editor\.lineCount - 1", body)
    assert logical, "the non-CodeMirror fallback should still exist"
    visual = re.search(r"moveVertically", body)
    assert visual and visual.start() < logical.start(), (
        "the visual-line test has to run first; the logical one is only for an "
        "editor that is not a CodeMirrorEditor"
    )
