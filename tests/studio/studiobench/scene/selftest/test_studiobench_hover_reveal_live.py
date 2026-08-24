# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE "NEW CHAT" CONTROL WAS NEVER COVERED. It was never HOVERED.

`thread_reopen` reported NOT RUN on every run, with two diagnostics that were both accurate and
both pointed the wrong way:

    button[aria-label="New chat"] was not clickable: TimeoutError;
    no point on the control hit-tests to it

That reads like an overlay, or a collapsed sidebar, or a zero-size button, and it was none of
them. Asking the live app directly returned a control that is 20x20 at (243, 319), `visibility:
visible`, `display: flex`, one instance, nothing drawn over it -- and `pointer-events: none` with
`opacity: 0`. Studio styles it as a hover-revealed action:

    .sidebar-header-action { opacity-0 pointer-events-none }
    .group\\/sidebar-header:hover .sidebar-header-action { opacity-100 pointer-events-auto }

With no mouse over the header the button is laid out, passes every actionability check Playwright
makes, and is transparent to every hit test. So `click()` waited out its timeout, the hit-test
spread found no reachable point, and the harness substituted `page.goto` -- timing a full document
navigation as though it were the client-side subtree rebuild the action exists to measure.

The fixture below is the app's rule, not an approximation of it. Both halves are asserted: that
the un-hovered control really is unreachable (otherwise the test proves nothing and would pass
without the fix), and that hovering makes it clickable through the production
`_click_or_navigate` path.

    python -m pytest tests/studio/studiobench/scene/selftest/test_studiobench_hover_reveal_live.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402

#: Studio's own geometry: a 20x20 action inside a 279x49 sticky header row, at the same offset the
#: live probe measured. The numbers matter -- a control large enough to be hit by chance would let
#: the test pass for the wrong reason.
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

#: A URL that actually loads. Pointing the fallback at a dead port exercises the "navigation
#: failed too" branch instead of the navigation branch under test.
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
    # A PAGE PER TEST, because the mouse position is page state and a previous test's hover would
    # carry the control into the next one already revealed.
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


# ── the condition, reproduced ───────────────────────────────────────


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


# ── the fix ─────────────────────────────────────────────────────────


def test_hovering_reveals_the_control_and_returns_a_point_on_it(page):
    point = A._reveal_by_hover(_ctx(page), SELECTOR)
    assert point is not None, "hovering did not make the control hit-testable"
    x, y = point
    # Against the control's OWN box, read from the page. Hard-coding the live app's (243, 319)
    # asserted the fixture's layout rather than the behaviour, and failed on a fixture that was
    # working correctly.
    box = page.eval_on_selector(SELECTOR, "(el) => el.getBoundingClientRect().toJSON()")
    assert box["left"] <= x <= box["right"], (point, box)
    assert box["top"] <= y <= box["bottom"], (point, box)


def test_the_action_clicks_the_control_instead_of_substituting_a_navigation(page):
    """THE BEHAVIOUR THAT CHANGED, end to end through the production path.

    Before the fix this returned `path == "navigate"`, and `thread_reopen` correctly refused to
    score it -- so the action reported NOT RUN on every single run.
    """
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


# ── and what it must NOT do ─────────────────────────────────────────


def test_an_ordinary_control_is_not_hovered_first(page):
    """The reveal is a FALLBACK. A control that hit-tests at rest is clicked without any mouse
    movement being introduced into a measured window, which is checked here by its absence from
    the log rather than by asking `_reveal_by_hover` what it would have done."""
    page.eval_on_selector(
        SELECTOR,
        # `transition: none` as well: `opacity` is transitioned over 150ms, so `getComputedStyle`
        # sampled on the next tick still reports "0" and the control looks hidden to any check
        # that reads it. That is a real hazard for the production code too, and the reason the
        # reveal waits out `_REVEAL_SETTLE_MS` rather than sampling immediately.
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
