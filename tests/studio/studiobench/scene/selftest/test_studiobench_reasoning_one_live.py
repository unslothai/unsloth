# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Times one pane's expansion, unlike reasoning_toggle; its cost must not scale with thread length."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"

#: Radix keeps collapsed content mounted, so only root `data-state` tells open from closed.
BODY_TEMPLATE = """<!doctype html><meta charset=utf-8><title>t</title>
<div class="aui-thread-root"><div class="aui-thread-viewport" id="vp"></div></div>
<script>
  const vp = document.getElementById("vp");
  const PANES = __PANES__;
  const SPANS = __SPANS__;
  for (let i = 0; i < PANES; i++) {
    const root = document.createElement("div");
    root.setAttribute("data-slot", "reasoning-root");
    root.setAttribute("data-state", "closed");
    const trigger = document.createElement("button");
    trigger.setAttribute("data-slot", "reasoning-trigger");
    trigger.textContent = "pane " + i;
    const content = document.createElement("pre");
    trigger.addEventListener("click", () => {
      const open = root.getAttribute("data-state") === "open";
      root.setAttribute("data-state", open ? "closed" : "open");
      // The CONTENT element stays mounted either way, as Radix leaves it for the animation, and
      // the highlight spans inside it do not: that is where the cost is, and it is why the action
      // counts `pre span` rather than trusting the pane count. `hidden` would not reproduce this,
      // because a hidden element still matches a selector.
      content.replaceChildren();
      if (!open) {
        for (let s = 0; s < SPANS; s++) {
          const span = document.createElement("span");
          span.textContent = "tok";
          content.appendChild(span);
        }
      }
    });
    root.appendChild(trigger);
    root.appendChild(content);
    vp.appendChild(root);
  }
  window.__sbNextPaint = () => new Promise(
    (r) => requestAnimationFrame(() => requestAnimationFrame(r)));
</script>
"""


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
def make_page(browser):
    opened = []

    def _make(panes: int, spans_per_pane: int = 4):
        page = browser.new_page()
        body = BODY_TEMPLATE.replace("__PANES__", str(panes)).replace(
            "__SPANS__", str(spans_per_pane)
        )
        page.set_content(body)
        # `set_content` does not reliably run init scripts, so attach the module after.
        page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        opened.append(page)
        return page

    yield _make
    for page in opened:
        page.close()


def _ctx(page) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.base.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1"},
        budget_ms = 30_000,
        dom = None,
        log = lambda _m: None,
    )


def test_it_opens_exactly_one_pane_and_leaves_the_rest_shut(make_page):
    got = A.reasoning_toggle_one(_ctx(make_page(panes = 8)))
    assert got.ran is True, got.reason
    assert got.expect_ok is True, got.reason
    assert got.expect["open_after_expand"] == 1
    assert got.expect["open_after_collapse"] == 0
    assert got.expect["panes"] == 8
    assert got.expect["panes_opened"] == 1


def test_the_cost_does_not_scale_with_thread_length(make_page):
    """Cost must not grow with thread length: one pane's spans are materialised, not the whole thread's."""
    small = A.reasoning_toggle_one(_ctx(make_page(panes = 2, spans_per_pane = 4)))
    large = A.reasoning_toggle_one(_ctx(make_page(panes = 40, spans_per_pane = 4)))
    assert small.ran and large.ran
    assert small.expect["highlight_spans_added"] == large.expect["highlight_spans_added"] == 4


def test_a_thread_with_no_reasoning_pane_is_NOT_RUN(make_page):
    got = A.reasoning_toggle_one(_ctx(make_page(panes = 0)))
    assert got.ran is False
    assert "no reasoning pane" in (got.reason or "")


def test_it_refuses_a_thread_that_is_already_open(make_page):
    """A pane left open by an earlier action would make this measure a CLOSE and call it an open.
    Refusing is right: the number is the only reason the action exists."""
    page = make_page(panes = 3)
    page.evaluate("() => window.__sb.dom.reasoningTriggers()[0].click()")
    got = A.reasoning_toggle_one(_ctx(page))
    assert got.ran is False
    assert "already open" in (got.reason or "")


def test_it_is_not_in_the_standard_film(make_page):
    """Not in the standard film: a new slot shifts later windows and breaks comparison with old payloads."""
    schedule = (Path(_STUDIO_TESTS) / "studiobench" / "scene" / "schedule.py").read_text(
        encoding = "utf-8"
    )
    assert '"reasoning_toggle_one"' not in schedule
