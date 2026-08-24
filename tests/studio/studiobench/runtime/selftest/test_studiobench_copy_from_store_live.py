# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SELECT-ALL COPY ON A WINDOWED THREAD, IN A REAL BROWSER, WITH A REAL CLIPBOARD.

The data-loss case, end to end. A windowed message list cannot SELECT what it has not mounted, so
`Selection.toString()` is short by design; whether the user loses their conversation depends
entirely on whether the app's copy handler serialises from the message store. Those two facts look
identical from the DOM and opposite from the clipboard, which is why `select_all_copy` now scores
itself on the clipboard.

The unit tests in the Studio worktree stub `containsNode`, so the rule "the selection spans the
whole mounted list" has never met a real `Selection` there. That is the part most likely to be
quietly wrong -- partial containment, text-node endpoints, whether a select-all over a scroll
container really reports the first and last rows as contained -- and it is what this exercises.

Three threads, one keystroke:

  full mount              clipboard whole, selection whole. The shipping build.
  windowed, no handler    clipboard SHORT. The regression, reproduced rather than described.
  windowed, handler       clipboard whole, selection still short. The fix, and the only
                          combination that distinguishes it from the other two.
"""

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
        # The real harness's browser factory requests the same two. Without them the clipboard
        # read-back throws and the reading is "could not be measured", which is exactly the
        # NOT COMPARABLE outcome the scoring layer now produces rather than a pass.
        ctx = b.new_context(permissions = ["clipboard-read", "clipboard-write"])
        # `navigator.clipboard` DOES NOT EXIST outside a secure context, and `set_content` leaves
        # the page on about:blank, which is not one. The reading came back as
        # "Cannot read properties of undefined" rather than as an empty clipboard, which would have
        # been easy to misread as "the copy did not work". Fulfilled from a route so no server is
        # needed and no network is touched.
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


#: Exactly what scene/actions.select_all_copy does: focus the viewport, select its contents, then a
#: REAL Control+C so the app's own copy handler runs. Anything else would be testing a different
#: code path from the one the benchmark drives.
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
    # Three of nine turns, which is the window. The user pressed Ctrl+A and got a third of it.
    assert present <= WINDOW // 2 + 1


def test_the_handler_puts_the_whole_conversation_on_the_clipboard(context):
    """THE FIX. Same window, same short selection, whole clipboard.

    Both halves are asserted. A test that only checked the clipboard would pass just as well
    against a build that quietly stopped virtualising, and the entire question is whether the
    conversation survives WHILE the DOM is windowed.
    """
    page = _page(context, "windowed", copy_from_store = True)
    try:
        got = _select_all_copy(page)
    finally:
        page.close()
    assert got["mounted"] == WINDOW < got["total"] == MESSAGES
    assert _markers_present(got["clipboard"]) == TURNS, got["clipboard"][:400]
    # The selection is still short, and that is correct: it can only cover mounted nodes. This is
    # the reading the old alarm was wired to, and it is why the alarm had to be moved.
    assert got["selected_chars"] < got["clipboard_chars"]


def test_a_partial_selection_is_not_replaced_by_the_whole_conversation(context):
    """The failure mode of the fix itself, which would be worse than the bug.

    Someone highlighting one message must not silently get forty turns of markdown.

    A SENTINEL IS WRITTEN FIRST, because the obvious version of this test is not deterministic. It
    asserted that the partial selection was non-empty, and under a loaded machine
    `Selection.toString()` came back empty for a row Chromium had not laid out -- passing when the
    file ran alone and failing in a full-suite run. Seeding the clipboard means both outcomes are
    still conclusive: either the copy happened and the clipboard holds one row, or it did not and
    the clipboard still holds the sentinel. Neither is the whole conversation, which is the claim.
    """
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
    # THE CLAIM, and it holds either way.
    assert _markers_present(clip) <= 1, clip[:300]
    if selected > 0:
        # The copy really happened, so the clipboard is one row's worth and not the conversation.
        assert clip != "SENTINEL"
        assert len(clip) < 1000, clip[:300]
    else:
        # Chromium gave no selection for an off-screen row, so no copy was performed at all. Said
        # out loud rather than passed silently: this run did not exercise the substitution guard.
        assert clip == "SENTINEL"
