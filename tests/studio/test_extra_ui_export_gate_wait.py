# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The chat-only /export check in playwright_extra_ui.py must wait for the gate, not count it.

export-page.tsx renders "Export unavailable" only once the hardware query has answered
(hardware.loaded && exportSupported === false). The form and its export CTA render first and
satisfy the step's opening wait, so a count() taken then misses a gate that is on its way."""

from pathlib import Path

DRIVER = Path(__file__).resolve().parent / "playwright_extra_ui.py"
EXPORT_PAGE = (
    Path(__file__).resolve().parents[2] / "studio/frontend/src/features/export/export-page.tsx"
)


def _chat_only_branch() -> str:
    source = DRIVER.read_text(encoding = "utf-8")
    start = source.index('    if chat_only:\n        if "/export" not in page.url:')
    return source[start : source.index("\n    else:", start)]


def test_the_gate_is_waited_for_not_counted():
    branch = _chat_only_branch()
    assert "unavailable.wait_for(" in branch and 'state = "visible"' in branch, branch
    assert (
        "unavailable.count()" not in branch
    ), "a count() reads the page before the gate can render"


def test_the_gate_still_waits_on_the_hardware_answer():
    # If the page stopped gating on the hardware load, the wait above would be unnecessary, and
    # this test is where that shows.
    page = EXPORT_PAGE.read_text(encoding = "utf-8")
    assert "hardware.loaded && hardware.exportSupported === false" in page
    assert "<AlertTitle>Export unavailable</AlertTitle>" in page
