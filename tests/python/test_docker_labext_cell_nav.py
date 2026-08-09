# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Colab-style arrow navigation must not swallow wrapped-line movement.

`cellNav.ts` owns ArrowUp/ArrowDown in the capture phase and jumps to the
previous/next cell when the cursor sits on the first/last line of the editor.
That test used `editor.getCursorPosition().line` against `editor.lineCount`,
both of which are LOGICAL (JupyterLab's CodeMirrorEditor: `get lineCount() {
return this.doc.lines }`), while JupyterLab wraps markdown and raw cell editors
by default (`StaticNotebook.defaultEditorConfig` -> `markdown: { lineWrap: true
}`, `raw: { lineWrap: true }`; the image's `docker/jupyter/overrides.json` only
sets `autoClosingBrackets`).

So for a one-line markdown header -- what every Unsloth notebook opens with --
`lineCount === 1`, the cursor is on line 0 == lineCount - 1 from every visual
row, and BOTH arrows leave the cell: the wrapped rows in between cannot be
reached at all. Measured in Chromium with CodeMirror 6 + EditorView.lineWrapping
at the notebook's editor width: 1 logical line renders as 7 visual rows and the
logical test hijacks the arrows on 7 of 7 rows, in both directions. The same
measurement on an unwrapped code cell shows the visual test agreeing with the
logical one on every row, so the Colab-style jump is unchanged there.

CodeMirror's own answer is `EditorView.moveVertically(range, forward)`, which
moves "to the next line (including wrapped lines)"; it returns the unchanged
head only at offset 0 / doc.length, so a move that stays on the same visual row
(same `coordsAtPos().top`) is the real editor edge.

Static source guard: the labextension is only built inside Dockerfile.studio
(`jlpm install && jlpm build:prod`), so there is no TS test runner in-repo.
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
