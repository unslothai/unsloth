# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A windowed thread's select-all copy must yield the whole conversation, not only the mounted rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

TURNS = 9
MESSAGES = TURNS * 2
WINDOW = 6

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_FIXTURE_JS = _HERE.parent / "thread_fixture.js"


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
        # navigator.clipboard needs a secure context and about:blank is not one, so serve from a route.
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin = ORIGIN)
        yield ctx
        ctx.close()
        b.close()


ORIGIN = "https://studiobench.localhost"


def _page(context, mode: str, copy_from_store: bool):
    page = context.new_page()
    page.route(
        "**/*",
        lambda route: route.fulfill(
            status = 200,
            content_type = "text/html; charset=utf-8",
            body = "<!doctype html><meta charset=utf-8><title>t</title><body></body>",
        ),
    )
    page.goto(ORIGIN + "/chat")
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _FIXTURE_JS.read_text(encoding = "utf-8"))
    page.evaluate(
        "(o) => window.__fixture.build(o)",
        {
            "mode": mode,
            "turns": TURNS,
            "windowSize": WINDOW,
            "copyFromStore": copy_from_store,
        },
    )
    return page


SELECT_JS = """
async () => {
  const v = window.__sb.dom.viewport();
  v.focus({ preventScroll: true });
  const sel = window.getSelection();
  sel.removeAllRanges();
  const range = document.createRange();
  range.selectNodeContents(v);
  sel.addRange(range);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return sel.toString().length;
}
"""


def _select_all_copy(page) -> dict:
    selected = page.evaluate(SELECT_JS)
    page.keyboard.press("Control+C")
    page.wait_for_timeout(250)
    clip = page.evaluate("async () => await navigator.clipboard.readText()")
    return {
        "selected_chars": selected,
        "clipboard_chars": len(clip) if isinstance(clip, str) else None,
        "clipboard": clip,
        "mounted": page.evaluate("() => window.__sb.dom.messageCount()"),
        "total": page.evaluate("() => window.__sb.dom.threadTotal()"),
    }


def _markers_present(clip: str) -> int:
    return sum(1 for i in range(TURNS) if f"studiobench turn {i}:" in clip)


def test_a_full_mount_copies_the_whole_thread(context):
    page = _page(context, "full", copy_from_store = False)
    try:
        got = _select_all_copy(page)
    finally:
        page.close()
    assert got["mounted"] == got["total"] == MESSAGES
    assert _markers_present(got["clipboard"]) == TURNS


def test_a_windowed_thread_without_the_handler_loses_most_of_the_conversation(context):
    """THE REGRESSION, REPRODUCED. Not described, not inferred from the mounted count: the
    clipboard is read back and most of the conversation is not in it."""
    page = _page(context, "windowed", copy_from_store = False)
    try:
        got = _select_all_copy(page)
    finally:
        page.close()
    assert got["mounted"] == WINDOW
    assert got["total"] == MESSAGES
    present = _markers_present(got["clipboard"])
    assert present < TURNS, "the windowed thread copied every turn, so there is nothing to fix"
    assert present <= WINDOW // 2 + 1


def test_the_handler_puts_the_whole_conversation_on_the_clipboard(context):
    """Asserts the clipboard is whole while the DOM stays windowed, so un-virtualising cannot pass it."""
    page = _page(context, "windowed", copy_from_store = True)
    try:
        got = _select_all_copy(page)
    finally:
        page.close()
    assert got["mounted"] == WINDOW < got["total"] == MESSAGES
    assert _markers_present(got["clipboard"]) == TURNS, got["clipboard"][:400]
    assert got["selected_chars"] < got["clipboard_chars"]


def test_a_partial_selection_is_not_replaced_by_the_whole_conversation(context):
    """A partial selection must not silently copy the whole conversation; a sentinel is written first."""
    page = _page(context, "windowed", copy_from_store = True)
    try:
        page.evaluate("async () => await navigator.clipboard.writeText('SENTINEL')")
        selected = page.evaluate("""
          async () => {
            const rows = document.querySelectorAll("[aria-posinset]");
            const target = rows[rows.length - 1];
            target.scrollIntoView();
            const sel = window.getSelection();
            sel.removeAllRanges();
            const range = document.createRange();
            range.selectNodeContents(target);
            sel.addRange(range);
            await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
            return sel.toString().length;
          }
        """)
        page.keyboard.press("Control+C")
        page.wait_for_timeout(250)
        clip = page.evaluate("async () => await navigator.clipboard.readText()")
    finally:
        page.close()
    assert _markers_present(clip) <= 1, clip[:300]
    if selected > 0:
        assert clip != "SENTINEL"
        assert len(clip) < 1000, clip[:300]
    else:
        # Chromium gave no selection for an off-screen row, so the guard was not exercised.
        assert clip == "SENTINEL"
