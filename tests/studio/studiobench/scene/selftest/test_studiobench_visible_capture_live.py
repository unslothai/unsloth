# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE VISIBLE-REGION OBSERVER, in a real browser.

`compare_visible` is tested offline against hand-written captures, which proves the verdict logic
and proves nothing about whether the captures describe reality. Whether IntersectionObserver sees
what a user sees -- across a scroll, across mounting and unmounting, at the edge of the viewport --
is not a question a fixture dictionary can answer, so it is asked of Chromium here.

The three things that would make this instrument quietly wrong, each pinned:

  IT MUST SEE A PARTLY VISIBLE MESSAGE. One pixel into the viewport is visible to a user, and a
  threshold that rounded it away would exempt real differences at the top and bottom of the screen
  -- the two places a scroll spends most of its time.

  IT MUST ACCUMULATE ACROSS THE ACTION. An action that scrolls shows messages and hides them again.
  A single sample at the close of the window compares wherever the scroll happened to stop and
  silently ignores everything the user saw on the way. The compared set is the UNION.

  IT MUST NOT READ GEOMETRY. `getBoundingClientRect()` / `getClientRects()` on content inside a
  `content-visibility` locked subtree makes Chromium render that subtree to answer, so a
  geometry-based visibility probe unlocks exactly what it came to observe. One session reported 0
  off-screen unrendered roots while the event counter recorded 22 in the skipped state. This test
  installs a counting trap on both methods and fails if the capture touches either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_PARITY_JS = _STUDIO_TESTS / "studiobench" / "scene" / "parity.js"

#: A thread of tall messages in a short viewport, so only a few are ever on screen at once. Each
#: message publishes `aria-posinset`, which is how a windowed arm states thread position; the
#: capture keys on it so a window and a full mount are comparable.
FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  /* The OBSERVED element is the tall one, as in the app: `[data-role]` is the message and the
     virtualizer's row wrapper around it carries aria-posinset. An earlier version of this fixture
     made the row tall and left `[data-role]` an 18px line of text inside it, so a scroll step
     jumped clean over the observed target and the union test failed for a reason that was entirely
     the fixture's. */
  [data-role] { height: 500px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp"></div>
</div>
<script>
  window.__build = (count) => {
    const vp = document.getElementById("vp");
    vp.innerHTML = "";
    for (let i = 1; i <= count; i++) {
      const row = document.createElement("div");
      row.className = "row";
      row.setAttribute("aria-posinset", String(i));
      row.setAttribute("aria-setsize", String(count));
      const msg = document.createElement("div");
      msg.setAttribute("data-role", i % 2 ? "user" : "assistant");
      msg.textContent = "message " + i;
      row.appendChild(msg);
      vp.appendChild(row);
    }
  };
  window.__build(20);
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
def page(browser):
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(FIXTURE)
    # `add_script_tag` after the content, not `add_init_script` before it: Playwright's
    # `set_content` does not always run init scripts, and the symptom is `window.__sb` simply not
    # existing, which reads like a broken instrument rather than a mis-ordered fixture.
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    yield pg
    pg.close()


def _watch(page) -> None:
    got = page.evaluate("() => window.__sb.parityVisible.watch()")
    assert got.get("visible_attempted") is True, got
    # IntersectionObserver's first delivery is asynchronous, so give it a frame before anything
    # scrolls. Without this the initial viewport contents are attributed to whatever came next.
    page.wait_for_timeout(120)


def _capture(page) -> dict:
    page.wait_for_timeout(120)
    return page.evaluate("async () => await window.__sb.parityVisible.capture()")


# ── what the viewport actually showed ───────────────────────────────


def test_it_reports_only_what_the_viewport_showed(page):
    _watch(page)
    got = _capture(page)
    assert got["visible_attempted"] is True
    # A 400px viewport over 500px rows: message 1 fills it, message 2 is not reached.
    assert got["ever_visible"] == [1], got["ever_visible"]
    assert set(got["messages"]) == {"1"}


def test_a_partly_visible_message_counts_as_visible(page):
    """One pixel of message 2 inside the viewport. A threshold that rounded this away would exempt
    real differences at the top and bottom of the screen."""
    _watch(page)
    page.evaluate("() => { document.getElementById('vp').scrollTop = 101; }")
    got = _capture(page)
    assert got["ever_visible"] == [1, 2], got["ever_visible"]


def test_the_compared_set_is_the_union_across_the_whole_action(page):
    """The scroll passes over messages 1 to 8 and lands showing 7 and 8. A single sample at the
    close would report two; the user saw eight."""
    _watch(page)
    for top in (0, 700, 1400, 2100, 2800, 3200):
        page.evaluate(f"() => {{ document.getElementById('vp').scrollTop = {top}; }}")
        page.wait_for_timeout(60)
    got = _capture(page)
    assert got["ever_visible"][0] == 1
    assert len(got["ever_visible"]) >= 7, got["ever_visible"]
    assert 7 in got["ever_visible"] and 8 in got["ever_visible"]


def test_a_message_mounted_mid_action_is_observed_too(page):
    """A windowed list mounts rows as it scrolls. Rows that appear after the observer was installed
    have to be picked up, or a windowed arm reports only what it happened to have mounted at the
    start and the comparison silently shrinks to nothing."""
    page.evaluate("() => window.__build(2)")
    _watch(page)
    page.evaluate("() => window.__build(20)")
    page.wait_for_timeout(120)
    page.evaluate("() => { document.getElementById('vp').scrollTop = 1400; }")
    got = _capture(page)
    assert 3 in got["ever_visible"] or 4 in got["ever_visible"], got["ever_visible"]


def test_an_unmounted_message_is_still_reported_as_having_been_visible(page):
    """The honest residue. It was on screen, so it belongs in `ever_visible`; it is gone, so it
    cannot be digested, and the gap is reported rather than quietly closed."""
    _watch(page)
    page.evaluate("() => { document.getElementById('vp').scrollTop = 2100; }")
    page.wait_for_timeout(120)
    page.evaluate("() => window.__build(0)")
    got = _capture(page)
    assert got["ever_visible_count"] > 0
    assert got["mounted_ever_visible"] == 0
    assert got["unmounted_at_capture"] == got["ever_visible_count"]


# ── the trap ────────────────────────────────────────────────────────


def test_the_capture_never_reads_geometry(page):
    """THE CONTENT-VISIBILITY TRAP, held closed by construction rather than by review.

    Reading a rect inside a `content-visibility` locked subtree makes Chromium render it, so a
    geometry-based visibility probe destroys the very state it is measuring and then reports that
    nothing was skipped. If anyone reaches for `getBoundingClientRect` in this path later, this
    fails.
    """
    page.evaluate(
        """() => {
             window.__geom = 0;
             for (const name of ["getBoundingClientRect", "getClientRects"]) {
               const original = Element.prototype[name];
               Element.prototype[name] = function () {
                 window.__geom += 1;
                 return original.apply(this, arguments);
               };
             }
           }"""
    )
    _watch(page)
    page.evaluate("() => { document.getElementById('vp').scrollTop = 1400; }")
    got = _capture(page)
    assert got["ever_visible"], "the capture returned nothing, so the trap proves nothing"
    assert page.evaluate("() => window.__geom") == 0, (
        "the visible-region capture read element geometry, which forces a content-visibility "
        "locked subtree to render and makes the probe change what it observes"
    )


# ── refusals ────────────────────────────────────────────────────────


def test_capturing_without_watching_is_refused_not_reported_empty(page):
    got = page.evaluate("async () => await window.__sb.parityVisible.capture()")
    assert got["visible_attempted"] is False
    assert "never installed" in got["reason"]


def test_a_page_with_no_thread_viewport_is_refused(page):
    page.evaluate("() => { document.querySelector('.aui-thread-viewport').remove(); }")
    got = page.evaluate("() => window.__sb.parityVisible.watch()")
    assert got["visible_attempted"] is False, got
    assert "viewport" in got["reason"]


# ── the instrument must not charge its own cost to the action ───────


def test_the_top_up_is_proportional_to_the_mutation_not_to_the_document(page):
    """WORKSPACE TASK #102, WHICH THIS NEARLY REPEATED.

    The visibility observer's MutationObserver is the one part of this instrument that runs INSIDE
    the measured action window. The obvious implementation re-runs the full `querySelectorAll` scan
    on every mutation batch, which charges an O(document) walk to the action, once per batch, on a
    DOM whose size is the quantity under investigation. That is exactly the defect that reported
    `delete_message` at 14.3 fps when the action costs 49.0.

    So the top-up walks only `addedNodes`. This is asserted by counting `querySelectorAll` calls
    against the DOCUMENT while a stream-like mutation storm runs: text churning inside already
    mounted rows must produce none at all, because childList records do not fire for it.
    """
    page.evaluate(
        """() => {
             window.__docQsa = 0;
             const original = Document.prototype.querySelectorAll;
             Document.prototype.querySelectorAll = function () {
               window.__docQsa += 1;
               return original.apply(this, arguments);
             };
           }"""
    )
    _watch(page)
    # The handle is taken BEFORE the baseline, because looking the element up is itself a
    # document-wide query and would otherwise be counted against the instrument.
    page.evaluate(
        """() => {
             const rows = document.querySelectorAll("[data-role]");
             window.__last = rows[rows.length - 1];
           }"""
    )
    baseline = page.evaluate("() => window.__docQsa")
    # A stream: the last message's text changes many times, and nothing is added or removed.
    page.evaluate(
        "() => { for (let i = 0; i < 200; i++) window.__last.textContent = 'streaming ' + i; }"
    )
    page.wait_for_timeout(150)
    assert page.evaluate("() => window.__docQsa") == baseline, (
        "a text mutation inside a mounted row triggered a document-wide scan, so the instrument "
        "charges an O(document) walk to whatever action happens to be streaming"
    )


def test_a_row_mounted_during_the_action_is_still_picked_up_cheaply(page):
    """The top-up has to actually work, or the previous test passes by doing nothing."""
    page.evaluate("() => window.__build(2)")
    _watch(page)
    page.evaluate(
        """() => {
             const vp = document.getElementById("vp");
             const row = document.createElement("div");
             row.setAttribute("aria-posinset", "3");
             const msg = document.createElement("div");
             msg.setAttribute("data-role", "assistant");
             msg.textContent = "late arrival";
             row.appendChild(msg);
             vp.appendChild(row);
           }"""
    )
    page.wait_for_timeout(120)
    page.evaluate("() => { document.getElementById('vp').scrollTop = 900; }")
    got = _capture(page)
    assert 3 in got["ever_visible"], got["ever_visible"]
