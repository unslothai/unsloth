# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A KEYSTROKE THAT WENT NOWHERE WAS BEING REPORTED AS A MEASUREMENT.

`select_all_copy` pressed Control+C, slept 250ms for the copy to land, and reported the elapsed
time as `copy_ms`. On any engine that does not perform the copy, that number is the sleep.

Measured across the eleven WebKit payloads on this machine, 43 rows:

    r1K      258.5 ms
    r10K     258.8 ms
    r100K    263.9 ms

A hundredfold change in the quantity under study moved the "measurement" by two percent, because
it was measuring `wait_for_timeout(250)`. Chromium reads about 1,538 ms at the 100K rung for the
same action. Playwright's WebKit never performs a clipboard copy on Control+C, so every one of
those rows was a sleep wearing a timing's name -- and unlike a missing reading, it is perfectly
stable, so it looks like data.

The guard is on the CLIPBOARD and not on the engine name. A sentinel is written before the
keystroke; if it survives, nothing was copied. That admits an engine that starts working and
refuses one that stops, with no list to maintain.
"""

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

#: `navigator.clipboard` does not exist outside a secure context, and `set_content` leaves the page
#: on about:blank, which is not one. Served from a route so no server is needed.
ORIGIN = "https://studiobench.localhost"

BODY = """<!doctype html><meta charset=utf-8><title>t</title>
<style>.aui-thread-viewport { height: 300px; overflow-y: auto; }</style>
<div class="aui-thread-root"><div class="aui-thread-viewport" id="vp"></div></div>
<script>
  const vp = document.getElementById("vp");
  for (let i = 1; i <= 6; i++) {
    const m = document.createElement("div");
    m.setAttribute("data-role", i % 2 ? "user" : "assistant");
    m.textContent = "message " + i + " " + "x".repeat(400);
    vp.appendChild(m);
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
def context():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        ctx = b.new_context(permissions = ["clipboard-read", "clipboard-write"])
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin = ORIGIN)
        yield ctx
        ctx.close()
        b.close()


@pytest.fixture()
def page(context):
    pg = context.new_page()
    pg.route(
        "**/*",
        lambda route: route.fulfill(status = 200, content_type = "text/html; charset=utf-8", body = BODY),
    )
    pg.goto(ORIGIN + "/chat")
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    yield pg
    pg.close()


def _ctx(page, log = None) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.base.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1", "base_url": ORIGIN},
        budget_ms = 30_000,
        dom = None,
        log = log or (lambda _m: None),
    )


def _swallow_the_copy(page) -> None:
    """An engine that does not copy, reproduced faithfully: the keystroke is delivered, a `copy`
    event fires, and nothing reaches the clipboard."""
    page.evaluate("() => document.addEventListener('copy', (e) => { e.preventDefault(); }, true)")


def test_a_real_copy_is_still_measured(page):
    """THE CONTROL. Without it the guard could refuse everything and the tests below would pass on
    an action that never reports a timing at all."""
    got = A.select_all_copy(_ctx(page))
    assert got.ran is True, got.reason
    assert got.timings.get("copy_ms") is not None
    assert (got.expect or {}).get("clipboard_chars", 0) > 0


def test_an_engine_that_does_not_copy_is_NOT_RUN_rather_than_a_timing(page):
    """THE DEFECT. The keystroke is sent, 250ms passes, and before this the action returned that
    250ms as `copy_ms` -- stable, plausible, and about nothing."""
    _swallow_the_copy(page)
    got = A.select_all_copy(_ctx(page))
    assert got.ran is False, got
    assert "did not put anything on the clipboard" in (got.reason or "")
    assert not got.timings.get("copy_ms")


def test_the_refusal_says_the_number_would_have_been_the_harness_own_settle(page):
    """The reason has to name the failure, or the next person reads NOT RUN as a flaky run and adds
    the action to an allow-list."""
    _swallow_the_copy(page)
    said: list[str] = []
    got = A.select_all_copy(_ctx(page, said.append))
    assert "settle" in (got.reason or ""), got.reason


def test_the_guard_is_on_the_clipboard_and_not_on_the_engine_name():
    """No engine list to maintain. An engine that starts performing the copy is admitted with no
    change here, and one that stops is refused with no change here."""
    import inspect

    src = inspect.getsource(A.select_all_copy)
    assert "sentinel" in src
    # EXECUTABLE LINES ONLY. The engine names belong in the comment that explains which engine was
    # observed failing and with what numbers; what must not exist is a BRANCH on one.
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    # The word "engine" is allowed: it appears in the refusal MESSAGE, which is where it belongs.
    # A specific engine's NAME is not, because that is what a branch would need.
    for name in ("webkit", "WebKit", "firefox", "Firefox", "browser_name", "browser_type"):
        assert name not in code, f"the guard branches on {name!r} rather than on the clipboard"
