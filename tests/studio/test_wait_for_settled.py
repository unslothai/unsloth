# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""`wait_for_settled` replaces "sleep N ms, then measure" in the Playwright smokes.

The fixed sleep is wrong in both directions: on a fast box it waits long after the
transition ended, and on a loaded runner it measures a box mid-flight. The helper has to
return once the element is still, keep waiting while it moves, and time out on an
element that never stops. Run against a real headless Chromium; skipped where there is
no Playwright or no browser build.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _playwright_robust import chromium_launch_args, wait_for_settled  # noqa: E402

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope = "module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless = True, args = chromium_launch_args())
        except Exception as exc:  # no browser build on this machine
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page()
        yield page
        browser.close()


def test_a_still_element_settles_at_once(page):
    page.set_content('<div id="box" style="width:100px;height:40px">x</div>')
    started = time.monotonic()
    wait_for_settled(page.locator("#box"), timeout_ms = 5_000)
    assert time.monotonic() - started < 2.0


def test_it_waits_for_a_transition_to_finish(page):
    page.set_content(
        '<div id="box" style="width:100px;height:40px;transition:width 900ms linear">x</div>'
    )
    page.eval_on_selector("#box", "el => { el.getBoundingClientRect(); el.style.width = '400px'; }")
    wait_for_settled(page.locator("#box"), timeout_ms = 10_000)
    assert page.eval_on_selector("#box", "el => el.getBoundingClientRect().width") == 400


def test_an_element_that_never_stops_times_out(page):
    page.set_content(
        "<style>@keyframes slide { from { margin-left: 0 } to { margin-left: 300px } }</style>"
        '<div id="box" style="width:50px;height:40px;animation:slide 400ms linear infinite">x</div>'
    )
    with pytest.raises(sync_api.TimeoutError):
        wait_for_settled(page.locator("#box"), timeout_ms = 1_500)


def test_a_node_replaced_mid_wait_is_looked_up_again(page):
    # A re-render that swaps the node must not strand the wait on the detached one.
    page.set_content('<div id="wrap"><div id="box" style="width:100px;height:40px">x</div></div>')
    page.evaluate(
        """() => setTimeout(() => {
            document.getElementById("wrap").innerHTML =
                '<div id="box" style="width:220px;height:40px">y</div>';
        }, 30)"""
    )
    wait_for_settled(page.locator("#box"), frames = 20, timeout_ms = 5_000)
    assert page.eval_on_selector("#box", "el => el.textContent") == "y"
