# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Playwright drivers load a model before they drive the model picker, and a loaded model opens
the picker collapsed to its own row (no tabs, search or rows). Each driver must expand that row, by
the labels the picker really renders, or its picker steps read "search input not found" / "picker
shows no rows" on every run. The run settings context field is found by its rendered name too."""

from __future__ import annotations

import re
from pathlib import Path

from tests.studio import _playwright_robust as robust


HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parents[1] / "studio/frontend/src"
PICKER = (
    FRONTEND / "components/assistant-ui/composer-model-picker/composer-model-picker.tsx"
).read_text(encoding = "utf-8")
CONFIG_PAGE = (FRONTEND / "features/model-picker/components/model-config-page.tsx").read_text(
    encoding = "utf-8"
)
PICKER_DRIVERS = (
    "playwright_chat_ui.py",
    "playwright_model_config.py",
    "playwright_memory_estimate.py",
)


def _driver(name: str) -> str:
    return (HERE / name).read_text(encoding = "utf-8")


def _nested_def(src: str, header: str) -> str:
    start = src.index(header)
    return src[start : src.index("\n    def ", start + 1)]


def test_the_collapsed_row_label_is_the_one_the_picker_renders():
    suffix = robust.PICKER_SHOW_ALL_SUFFIX
    assert re.search(r"aria-label=\{`\$\{\w+\}" + re.escape(suffix) + r"`\}", PICKER), (
        f"composer-model-picker.tsx no longer renders an aria-label ending in {suffix!r} on the "
        "collapsed row; update PICKER_SHOW_ALL_SUFFIX in _playwright_robust.py"
    )


def test_the_expanded_picker_marker_is_the_one_the_picker_renders():
    assert re.search(r'role="tablist"\s+aria-label="Model sources"', PICKER)
    assert robust.PICKER_SOURCES_TABLIST == '[role="tablist"][aria-label="Model sources"]'


def test_a_loaded_model_still_opens_the_picker_collapsed():
    # Why the drivers need the expand click at all: a settled load sets expanded to false.
    assert re.search(
        r"setExpanded\(\s*\([^;]*\)\s*\?\?\s*false,?\s*\)", PICKER
    ), "the load effect no longer collapses the picker; the drivers' expand click may be dead code"
    # Collapsed unless the user expanded it or nothing is chosen; more reasons to open expanded (a model with no
    # thinking toggle, #18128) are fine, since expand_collapsed_model_picker clicks nothing on an expanded picker.
    assert re.search(r"const isExpanded = expanded \|\| !chosenName(?: \|\| \w+)*;", PICKER)


def test_every_picker_driver_imports_the_expand_helper():
    for name in PICKER_DRIVERS:
        assert "    expand_collapsed_model_picker,\n" in _driver(name), name


def test_the_chat_search_step_expands_before_it_waits_for_the_search():
    src = _driver("playwright_chat_ui.py")
    start = src.index('step("model picker: open + drive search bar")')
    body = src[start : src.index("\n    step(", start + 1)]
    click = body.index("picker_btn.click()")
    # Opened before the selector names the loaded model, the picker collapses under the typing.
    settled = body.index('"select model" not in')
    assert settled < click
    expand = body.index("expand_collapsed_model_picker(page)", click)
    wait = body.index("search = wait_for_first(", expand)
    assert click < expand < wait
    # An attached but hidden search field would pass the wait and time out in fill().
    assert '.locator("visible=true")' in body[wait : body.index("timeout_ms", wait)]


def test_the_expand_helper_only_trusts_controls_on_screen():
    src = (HERE / "_playwright_robust.py").read_text(encoding = "utf-8")
    start = src.index("def expand_collapsed_model_picker(")
    body = src[start : src.index("\ndef ", start + 1)]
    assert body.count(">> visible=true") == 2


def test_open_picker_expands_after_clicking_the_trigger():
    for name in ("playwright_model_config.py", "playwright_memory_estimate.py"):
        body = _nested_def(_driver(name), "    def open_picker():")
        click = body.index("TRIGGER).first.click()")
        assert body.index("expand_collapsed_model_picker(popover)") > click, name


def test_the_context_field_name_is_the_one_the_run_settings_render():
    label = robust.RUN_SETTINGS_CONTEXT_LABEL
    assert f'ariaLabel="{label}"' in CONFIG_PAGE, (
        f"model-config-page.tsx no longer names its context field {label!r}; update "
        "RUN_SETTINGS_CONTEXT_LABEL in _playwright_robust.py"
    )
    for name in ("playwright_model_config.py", "playwright_memory_estimate.py"):
        body = _nested_def(_driver(name), "    def context_input(popover):")
        assert "name = RUN_SETTINGS_CONTEXT_LABEL, exact = True" in body, name
        assert '"Context Length"' not in body, name
