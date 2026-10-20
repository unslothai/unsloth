# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""New chat is hover-revealed, pointer-events none until hovered; the click must hover first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402

#: Unsloth's real 20x20 action in a 279x49 header row; a bigger target could be hit by chance.
FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; font: 12px sans-serif; }
  .sidebar { width: 280px; height: 900px; }
  .sidebar-header {
    display: flex; align-items: center; gap: 4px;
    width: 279px; height: 49px; padding: 0 4px; margin-top: 295px;
  }
  .sidebar-header-action {
    display: inline-flex; width: 20px; height: 20px;
    opacity: 0; pointer-events: none; transition: opacity 150ms;
  }
  .sidebar-header:hover .sidebar-header-action {
    opacity: 1; pointer-events: auto;
  }
</style>
<div class="sidebar">
  <div class="sidebar-header"><span>RECENTS</span>
    <button aria-label="New chat" class="sidebar-header-action">+</button>
  </div>
</div>
<script>
  window.__clicks = 0;
  document.querySelector('button[aria-label="New chat"]')
    .addEventListener("click", () => { window.__clicks += 1; });
</script>
"""

SELECTOR = 'button[aria-label="New chat"]'

#: A URL that loads, so the navigation branch (not the nav-failed branch) is exercised.
FALLBACK_URL = "about:blank"


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    # One page per test: mouse position is page state and a prior hover would pre-reveal the control.
    pg = browser.new_page(viewport = {"width": 1280, "height": 900})
    pg.set_content(FIXTURE)
    yield pg
    pg.close()


def _ctx(page, log = None) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.base.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1", "base_url": "http://127.0.0.1:1"},
        budget_ms = 30_000,
        dom = None,
        log = log or (lambda _m: None),
    )


def test_the_control_really_is_unreachable_until_it_is_hovered(page):
    """WITHOUT THIS THE REST PROVES NOTHING. If the fixture's button were hit-testable at rest,
    every assertion below would pass with or without the fix."""
    box = page.eval_on_selector(SELECTOR, "(el) => el.getBoundingClientRect().toJSON()")
    assert round(box["width"]) == 20 and round(box["height"]) == 20, "the fixture lost its geometry"
    style = page.eval_on_selector(
        SELECTOR,
        "(el) => { const s = getComputedStyle(el);"
        " return { pe: s.pointerEvents, op: s.opacity, vis: s.visibility }; }",
    )
    assert style == {"pe": "none", "op": "0", "vis": "visible"}, style
    assert (
        A._reachable_point(_ctx(page), SELECTOR) is None
    ), "the un-hovered control hit-tests to itself, so this fixture does not reproduce the failure"


def test_hovering_reveals_the_control_and_returns_a_point_on_it(page):
    point = A._reveal_by_hover(_ctx(page), SELECTOR)
    assert point is not None, "hovering did not make the control hit-testable"
    x, y = point
    # Compare against the control's own box, not hard-coded app coordinates.
    box = page.eval_on_selector(SELECTOR, "(el) => el.getBoundingClientRect().toJSON()")
    assert box["left"] <= x <= box["right"], (point, box)
    assert box["top"] <= y <= box["bottom"], (point, box)


def test_the_action_clicks_the_control_instead_of_substituting_a_navigation(page):
    """The action clicks the control rather than substituting page.goto, which thread_reopen would
    refuse."""
    got = A._click_or_navigate(_ctx(page), SELECTOR, FALLBACK_URL)
    assert got.path == "click", got.reason
    assert got.navigated is False
    assert page.evaluate("window.__clicks") == 1, "the app's own handler never fired"


def test_the_hover_is_reported_so_the_gesture_is_not_silent(page):
    """A harness that quietly does something extra to make a control work is how the last defect
    got in. The reveal is logged."""
    said: list[str] = []
    A._click_or_navigate(_ctx(page, said.append), SELECTOR, FALLBACK_URL)
    assert any("hover-revealed" in m for m in said), said


def test_an_ordinary_control_is_not_hovered_first(page):
    """The reveal is a FALLBACK. A control that hit-tests at rest is clicked without any mouse
    movement being introduced into a measured window, which is checked here by its absence from
    the log rather than by asking `_reveal_by_hover` what it would have done."""
    page.eval_on_selector(
        SELECTOR,
        # `opacity` transitions over 150ms, so getComputedStyle would still read 0 without this.
        "(el) => { el.style.transition = 'none'; el.style.opacity = '1';"
        " el.style.pointerEvents = 'auto'; }",
    )
    assert A._reachable_point(_ctx(page), SELECTOR) is not None
    said: list[str] = []
    got = A._click_or_navigate(_ctx(page, said.append), SELECTOR, FALLBACK_URL)
    assert got.path == "click"
    assert not any("hover-revealed" in m for m in said), said


def test_a_genuinely_unreachable_control_is_still_reported_unreachable(page):
    """The fix must not become a way of clicking things a user cannot click. A control covered by
    a real overlay stays unreachable after hovering, and the navigation fallback is taken with the
    reason saying so."""
    page.evaluate(
        "() => { const d = document.createElement('div');"
        " d.style.cssText = 'position:fixed;inset:0;z-index:9;background:#000';"
        " document.body.appendChild(d); }"
    )
    said: list[str] = []
    got = A._click_or_navigate(_ctx(page, said.append), SELECTOR, FALLBACK_URL)
    assert got.path == "navigate"
    assert "even after hovering it" in (got.reason or ""), got.reason
