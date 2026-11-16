# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat UI driver loads a model before it drives the model picker search, and a loaded model opens
the picker collapsed to its own row. The driver must expand that row by the aria-label the picker
really renders, or the search step reads "search input not found" on every run."""

from __future__ import annotations

import ast
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRIVER = (HERE / "playwright_chat_ui.py").read_text(encoding = "utf-8")
PICKER = (
    REPO
    / "studio/frontend/src/components/assistant-ui/composer-model-picker/composer-model-picker.tsx"
).read_text(encoding = "utf-8")
STEP = "model picker: open + drive search bar"


def _suffix() -> str:
    for node in ast.parse(DRIVER).body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "PICKER_SHOW_ALL_SUFFIX" for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    raise AssertionError("PICKER_SHOW_ALL_SUFFIX is no longer a module-level string in the driver")


def _step_body() -> str:
    start = DRIVER.index(f'step("{STEP}")')
    return DRIVER[start : DRIVER.index("\n    step(", start + 1)]


def test_the_suffix_matches_the_collapsed_row_label():
    suffix = _suffix()
    assert re.search(r"aria-label=\{`\$\{\w+\}" + re.escape(suffix) + r"`\}", PICKER), (
        f"composer-model-picker.tsx no longer renders an aria-label ending in {suffix!r} on the "
        "collapsed row; update PICKER_SHOW_ALL_SUFFIX in playwright_chat_ui.py to the new label"
    )


def test_a_loaded_model_still_opens_the_picker_collapsed():
    # Why the driver needs the expand click at all: a settled load sets expanded to false.
    assert re.search(
        r"setExpanded\(\s*\([^;]*\)\s*\?\?\s*false,?\s*\)", PICKER
    ), "the load effect no longer collapses the picker; the driver's expand click may be dead code"
    assert "const isExpanded = expanded || !chosenName;" in PICKER


def test_the_search_step_expands_the_collapsed_row_before_waiting_for_the_search():
    body = _step_body()
    show_all = body.index("PICKER_SHOW_ALL_SUFFIX")
    click = body.index("show_all.first.click()", show_all)
    wait = body.index("search = wait_for_first(search_box", click)
    assert show_all < click < wait
