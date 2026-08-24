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


# ── where the virtualizer publishes its ordinals is not a UI change ──
#
# `runtime/readiness.py` accepts `aria-posinset` / `aria-setsize` on the `[data-role]` message OR on
# an ancestor row wrapper -- it walks with `closest()`, because the ordinal belongs on whichever
# element is the member of the set, and refusing the first option would refuse a correctly
# implemented arm for putting the attribute in a place the gate itself calls right.
#
# The visible-region digest then read every attribute on the message, so an arm that took that
# option differed from the fully mounted arm on EVERY message -- which publishes neither attribute
# anywhere -- while the rendered content was identical. Auto-mode parity was unusable for a DOM
# shape the gate explicitly permits, and a wall of differences that are all the same non-finding
# buries anything real underneath it.


def _arm_html(ordinals: str, suffix: str = "") -> str:
    """One thread of twenty messages, with the virtualization ordinals published `on_the_message`,
    `on_the_row` wrapper, or `nowhere` -- which is what the shipped build does."""
    return """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  [data-role] { height: 500px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp"></div>
</div>
<script>
  const WHERE = "__WHERE__";
  const vp = document.getElementById("vp");
  for (let i = 1; i <= 20; i++) {
    const row = document.createElement("div");
    row.className = "row";
    const msg = document.createElement("div");
    msg.setAttribute("data-role", i % 2 ? "user" : "assistant");
    msg.textContent = "message " + i + "__SUFFIX__";
    row.appendChild(msg);
    if (WHERE !== "nowhere") {
      const owner = WHERE === "on_the_message" ? msg : row;
      owner.setAttribute("aria-posinset", String(i));
      owner.setAttribute("aria-setsize", "20");
    }
    vp.appendChild(row);
  }
</script>
""".replace("__WHERE__", ordinals).replace("__SUFFIX__", suffix)


def _capture_arm(
    browser,
    ordinals: str,
    suffix: str = "",
) -> dict:
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(_arm_html(ordinals, suffix))
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    try:
        _watch(pg)
        return _capture(pg)
    finally:
        pg.close()


def _thread_digests(browser, ordinals: str) -> dict:
    """The WHOLE-DOCUMENT structural digest of the same page, per message."""
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(_arm_html(ordinals))
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    try:
        got = pg.evaluate("() => window.__sb.parity.capture()")
        return {row["i"]: row["digest"] for row in got["messages"]}
    finally:
        pg.close()


def test_ordinals_on_the_message_do_not_make_every_message_differ(browser):
    """THE DEFECT. Same twenty messages, same text, same everything a user can see -- one arm
    publishing the ordinals the gate requires of it, the other publishing none."""
    from studiobench.analysis import parity as P

    windowed = _capture_arm(browser, "on_the_message")
    full = _capture_arm(browser, "nowhere")
    shared = sorted(set(windowed["messages"]) & set(full["messages"]))
    assert shared, (windowed["messages"], full["messages"])
    for key in shared:
        assert (
            windowed["messages"][key]["digest"] == full["messages"][key]["digest"]
        ), f"ordinal {key} differed on the virtualization bookkeeping alone"
    verdict = P.compare_visible(full, windowed)
    assert verdict["verdict"] == P.MATCH, verdict


def test_ordinals_on_the_row_wrapper_are_unaffected_as_they_always_were(browser):
    """The other permitted placement, which was never inside the message's subtree and so was never
    part of the defect. It must stay comparable."""
    from studiobench.analysis import parity as P
    assert (
        P.compare_visible(_capture_arm(browser, "nowhere"), _capture_arm(browser, "on_the_row"))[
            "verdict"
        ]
        == P.MATCH
    )


def test_a_real_rendering_difference_is_still_caught(browser):
    """THE POSITIVE CONTROL, without which the test above passes on a digest that stopped looking
    at anything. One arm renders different text; that is a visible difference and must still be."""
    from studiobench.analysis import parity as P

    verdict = P.compare_visible(
        _capture_arm(browser, "nowhere"), _capture_arm(browser, "on_the_message", suffix = " (v2)")
    )
    assert verdict["verdict"] == P.DIFFER, verdict


# ── a rebuilt row is at its own position, not at the end of a lifetime count ──
#
# `thread_reopen` leaves the thread and comes back, and a FULLY MOUNTED arm answers that by
# removing every message row and creating a new one for every message inside the same document.
# Those rebuilt rows publish no `aria-posinset` -- only a windowed arm publishes one -- so their
# ordinal comes from the fallback, and the fallback used to be a LIFETIME counter of observed
# nodes. It already stood at N, so the rebuilt rows were stamped N+1..2N while the windowed arm on
# the other side of the A/B stamped its real 1..N. `compare_visible` compares the two sets of
# ordinals first, found them disjoint, and reported "the two arms put DIFFERENT MESSAGES on
# screen" -- a hard visible-difference verdict -- for a rebuild that was identical.


#: The shipped shape: every message mounted, and NOTHING publishing a virtualization ordinal.
#: `__rebuild()` is the thread_reopen rebuild, in one commit, in the same document.
REBUILD_FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  [data-role] { height: 500px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp"></div>
</div>
<script>
  window.__rebuild = () => {
    const vp = document.getElementById("vp");
    vp.innerHTML = "";
    for (let i = 1; i <= 20; i++) {
      const row = document.createElement("div");
      row.className = "row";
      const msg = document.createElement("div");
      msg.setAttribute("data-role", i % 2 ? "user" : "assistant");
      msg.textContent = "message " + i;
      row.appendChild(msg);
      vp.appendChild(row);
    }
  };
  window.__rebuild();
</script>
"""


def _rebuild_page(browser):
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(REBUILD_FIXTURE)
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    return pg


def _capture_after_rebuild(browser) -> dict:
    pg = _rebuild_page(browser)
    try:
        _watch(pg)
        pg.evaluate("() => window.__rebuild()")
        pg.wait_for_timeout(150)
        return _capture(pg)
    finally:
        pg.close()


def test_a_rebuilt_row_carries_its_thread_position_not_a_lifetime_count(browser):
    """THE DEFECT. Twenty messages are observed, then all twenty rows are replaced. The viewport
    still shows message 1 and nothing else, so that is the only ordinal the capture may report."""
    got = _capture_after_rebuild(browser)
    assert got["ever_visible"] == [1], got["ever_visible"]
    assert set(got["messages"]) == {"1"}, got["messages"]
    # Every rebuilt row was placed, so nothing was dropped to buy the assertion above.
    assert got["unplaced_rows"] == 0, got


def test_a_rebuilt_full_mount_still_matches_a_windowed_arm(browser):
    """THE SYMPTOM, end to end: the verdict `thread_reopen` was getting on every pair.

    The two arms render the same twenty messages and show the same one. One rebuilds its rows the
    way a fully mounted arm does on reopen; the other publishes the ordinals the readiness gate
    requires of a windowed arm. Nothing a user could see differs, and the pair must say so.
    """
    from studiobench.analysis import parity as P

    verdict = P.compare_visible(
        _capture_after_rebuild(browser), _capture_arm(browser, "on_the_row")
    )
    assert verdict["verdict"] == P.MATCH, verdict


def test_a_real_difference_after_a_rebuild_is_still_caught(browser):
    """THE POSITIVE CONTROL for the two above, without which they pass on an instrument that
    stopped distinguishing anything. Same rebuild, different rendered text on the other arm."""
    from studiobench.analysis import parity as P

    verdict = P.compare_visible(
        _capture_after_rebuild(browser), _capture_arm(browser, "on_the_row", suffix = " (v2)")
    )
    assert verdict["verdict"] == P.DIFFER, verdict


def _count_document_queries(pg) -> None:
    pg.evaluate(
        """() => {
             window.__docQsa = 0;
             const original = Document.prototype.querySelectorAll;
             Document.prototype.querySelectorAll = function () {
               window.__docQsa += 1;
               return original.apply(this, arguments);
             };
           }"""
    )


def test_placing_a_batch_of_rebuilt_rows_costs_ONE_document_read_for_the_batch(browser):
    """THE PRICE OF GETTING THE ORDINAL RIGHT, and the reason a counter was tempting.

    Resolving a position needs the thread's current message list, and `observeAdded` runs inside
    the MEASURED action window. Read per row, a twenty-row rebuild would charge twenty O(document)
    walks to `thread_reopen` on a DOM whose size is the quantity under investigation -- workspace
    task #102 all over again. The index is therefore built once per mutation batch and shared by
    every row in it.

    Exactly one, in both directions: more than one means the lookup is back on the per-row path,
    and NONE means no position was resolved from the DOM at all, which is the lifetime counter
    this replaced.
    """
    pg = _rebuild_page(browser)
    try:
        _count_document_queries(pg)
        _watch(pg)
        baseline = pg.evaluate("() => window.__docQsa")
        pg.evaluate("() => window.__rebuild()")
        pg.wait_for_timeout(150)
        spent = pg.evaluate("() => window.__docQsa") - baseline
    finally:
        pg.close()
    assert spent == 1, f"a twenty-row rebuild in one batch cost {spent} document-wide queries"


def test_a_row_that_publishes_its_ordinal_costs_no_document_read_at_all(browser):
    """A windowed arm mounts rows continuously as it scrolls, and it is the arm that would pay
    most for a document read per batch. It publishes `aria-posinset`, which is read first and
    answers the question outright, so it never reaches the position index."""
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    pg.set_content(FIXTURE)
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    try:
        _count_document_queries(pg)
        _watch(pg)
        baseline = pg.evaluate("() => window.__docQsa")
        # Ten more rows, each publishing its own position, in one batch.
        pg.evaluate(
            """() => {
                 const vp = document.getElementById("vp");
                 for (let i = 21; i <= 30; i++) {
                   const row = document.createElement("div");
                   row.setAttribute("aria-posinset", String(i));
                   row.setAttribute("aria-setsize", "30");
                   const msg = document.createElement("div");
                   msg.setAttribute("data-role", "assistant");
                   msg.textContent = "message " + i;
                   row.appendChild(msg);
                   vp.appendChild(row);
                 }
               }"""
        )
        pg.wait_for_timeout(150)
        spent = pg.evaluate("() => window.__docQsa") - baseline
    finally:
        pg.close()
    assert spent == 0, f"a windowed arm's mounted rows cost {spent} document-wide queries"


def test_the_structural_digest_still_sees_the_ordinals(browser):
    """THE SCOPING DECISION, in the browser. The exclusion is passed in by the visible-region
    caller and is NOT baked into the shared `signature`: the structural digest only ever scores
    pairs where neither arm is windowing, and there an ordinal appearing on every message is a real
    change that somebody should be shown."""
    numbered = _thread_digests(browser, "on_the_message")
    plain = _thread_digests(browser, "nowhere")
    assert set(numbered) == set(plain)
    assert all(numbered[i] != plain[i] for i in numbered), (numbered, plain)


# ── a recycled node, and the ordinal it used to be denied ──────────────────


#: An EMPTY viewport, so a row appended to it is on screen immediately. `__churn` mounts a row and
#: unmounts it inside ONE task, which is how a row comes to be observed while detached: the
#: MutationObserver batch runs after both operations, the node is in `addedNodes` and is no longer
#: among the document's messages, so it has no position the instrument can honestly claim.
#: `__remount` then hands THE SAME NODE back, which is what a virtualizer that recycles its rows
#: does on the next scroll step.
RECYCLE_FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  [data-role] { height: 100px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp"></div>
</div>
<script>
  window.__row = null;
  window.__churn = () => {
    const vp = document.getElementById("vp");
    const msg = document.createElement("div");
    msg.setAttribute("data-role", "assistant");
    msg.textContent = "recycled row";
    window.__row = msg;
    vp.appendChild(msg);
    vp.removeChild(msg);
  };
  window.__remount = () => {
    document.getElementById("vp").appendChild(window.__row);
  };
</script>
"""


def test_a_recycled_row_is_placed_when_it_finally_mounts(browser):
    """REGRESSION. An unplaceable row must not be blacklisted from ever being placed.

    `observeOne` marked every node it looked at as `seen` BEFORE trying to place it, so a row
    observed while detached -- unplaceable, and correctly so -- could never be stamped afterwards.
    A virtualizer that recycles DOM nodes hands that same node back mounted and about to be shown;
    the early return fired, no ordinal was stamped, and every intersection it reported was dropped
    for want of one. The row was on screen and absent from `ever_visible`, which is exactly the
    silence `compare_visible` cannot see: with the other rows placed and matching, it returns
    MATCH while a visible row went uncompared.
    """

    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    try:
        pg.set_content(RECYCLE_FIXTURE)
        pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        _watch(pg)
        pg.evaluate("() => window.__churn()")
        pg.wait_for_timeout(150)
        # It was refused a position, and that refusal is counted rather than silent.
        assert pg.evaluate("async () => (await window.__sb.parityVisible.capture()).unplaced_rows")
        pg.evaluate("() => window.__remount()")
        got = _capture(pg)
    finally:
        pg.close()

    # Mounted into an empty viewport, so it is message 1 and it is on screen.
    assert got["ever_visible"] == [1], got


#: A row that is RENUMBERED IN PLACE. It stays connected, stays intersecting, and is handed to
#: another message -- which is what a virtualizer that recycles its row nodes does, and it is not a
#: childList mutation, so nothing about it reaches a childList-only observer.
RENUMBER_FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  [data-role] { height: 100px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp">
    <div aria-posinset="1"><div data-role="user" id="row">message 1</div></div>
  </div>
</div>
<script>
  window.__renumber = () => {
    const holder = document.getElementById("row").parentElement;
    holder.setAttribute("aria-posinset", "42");
    document.getElementById("row").textContent = "message 42";
  };
</script>
"""


def test_a_row_renumbered_in_place_is_restamped_and_reported(browser):
    """REGRESSION. A placed row used to be marked `seen` and never looked at again, which assumed
    a row's position in the thread cannot change while the node lives. A recycling virtualizer
    breaks that on purpose.

    The node kept its original `__sbOrdinal`, so the message it now showed was never reported
    visible, and its content was digested under the position it no longer held: `ever_visible` said
    [1] and `messages["1"]` carried message 42's digest. Because the row never stopped intersecting
    the IntersectionObserver had no change to report, and because the mutation observer took
    childList records only, the renumbering was invisible to every path at once.
    """

    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    try:
        pg.set_content(RENUMBER_FIXTURE)
        pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        _watch(pg)
        pg.evaluate("() => window.__renumber()")
        got = _capture(pg)
    finally:
        pg.close()

    # The row is still on screen, so BOTH the position it held and the one it holds now were shown.
    assert 42 in got["ever_visible"], got["ever_visible"]
    # And its content is filed under the position it actually holds.
    assert "42" in got["messages"], got["messages"]
    assert got["unplaced_rows"] == 0, got
    # A LEGITIMATE RENUMBER IS NOT A COLLISION. One node holds one position at a time here, so the
    # collision counter added for the two-rows-one-ordinal case must not fire on this, or a correct
    # recycling virtualizer would be refused on every action it recycles a row in.
    assert got["ordinal_collisions"] == 0, got


#: TWO MOUNTED ROWS PUBLISHING ONE POSITION. A virtualizer that renumbers a recycled row wrongly
#: leaves an extra message on screen wearing a position another row already holds.
COLLISION_FIXTURE = """
<!doctype html><meta charset="utf-8">
<style>
  body { margin: 0; }
  .aui-thread-viewport { height: 400px; overflow-y: auto; }
  [data-role] { height: 100px; }
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport" id="vp">
    <div aria-posinset="1"><div data-role="user">message 1</div></div>
    <div aria-posinset="2"><div data-role="assistant">message 2</div></div>
  </div>
</div>
<script>
  window.__ghost = (before) => {
    const vp = document.getElementById("vp");
    const holder = document.createElement("div");
    holder.setAttribute("aria-posinset", "1");
    const row = document.createElement("div");
    row.setAttribute("data-role", "user");
    row.textContent = "a ghost the user can see";
    holder.appendChild(row);
    if (before) vp.insertBefore(holder, vp.firstChild);
    else vp.appendChild(holder);
  };
</script>
"""


def _capture_with_ghost(browser, *, before: bool) -> dict:
    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    try:
        pg.set_content(COLLISION_FIXTURE)
        pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        _watch(pg)
        pg.evaluate("(b) => window.__ghost(b)", before)
        pg.wait_for_timeout(150)
        return _capture(pg)
    finally:
        pg.close()


def test_two_rows_sharing_a_thread_position_are_counted_rather_than_overwritten(browser):
    """THE DEFECT. The digest map is keyed by ordinal and the assignment was unconditional, so the
    second row in DOM order silently REPLACED the first one's entry; `VIS.ever` is a Set of
    numbers, so it collapsed the pair as well. Three rows on screen came out of the capture looking
    exactly like a capture of two.

    Whether that was ever caught was DOM order and nothing else. Reproduced in this browser, the
    same ghost inserted BEFORE the row it shadows leaves the surviving digest agreeing with the
    other arm and the pair returns MATCH; inserted after, it returns DIFFER. Last writer wins, so
    the collision passes exactly when the survivor happens to be the one that agrees.
    """
    for before in (True, False):
        got = _capture_with_ghost(browser, before = before)
        assert got["ordinal_collisions"] == 1, (before, got)
        assert got["collided_ordinals"] == [1], (before, got)
        # The counter is not derived from the arithmetic, and this is why: two of the three rows
        # on screen are filed under one key, so the map still holds two entries.
        assert set(got["messages"]) == {"1", "2"}, got["messages"]


def test_a_collision_refuses_the_pair_instead_of_reporting_agreement(browser):
    """THE CONSEQUENCE, through the comparison. Three rows are on screen on one arm and two on the
    other, and the capture cannot say so: the digest map holds two entries either way and
    `ever_visible` holds two ordinals either way. So `compare_visible` reached a verdict out of a
    set it did not know was short, and WHICH verdict depended only on which of the two rows sharing
    the position happened to be written last -- MATCH when the survivor agrees with the other arm,
    DIFFER when it does not, neither of them a statement about the row that was dropped.

    Refused rather than answered. A pair whose inputs are known to be incomplete carries no verdict
    in either direction, which is the rule this file applies to every other unreadable capture."""
    from studiobench.analysis import parity as P

    clean = _capture_with_ghost(browser, before = False)
    clean["ordinal_collisions"] = 0
    clean["collided_ordinals"] = []
    ghosted = _capture_with_ghost(browser, before = True)
    verdict = P.compare_visible(clean, ghosted)
    assert verdict["verdict"] == P.NOT_COMPARABLE, verdict
    assert "SAME thread position" in verdict["reason"], verdict


def test_losing_the_thread_outranks_the_collision_refusal(browser):
    """THE ONE FINDING A COLLISION CANNOT HAVE MANUFACTURED, so the refusal must not swallow it.

    "One arm's viewport ended EMPTY and the other's did not" is raised with `severe: True` and is
    documented at that site as not suppressible, because losing the conversation is a different kind
    of statement from a capture that could not be read. A blanket collision refusal placed ahead of
    it downgraded it to NOT COMPARABLE.

    A collision provably cannot cause it: a collision needs TWO mounted rows sharing one position,
    so the map it corrupts still holds an entry. It can merge two entries and never empty a map.
    """
    from studiobench.analysis import parity as P

    ghosted = _capture_with_ghost(browser, before = True)
    assert ghosted["ordinal_collisions"] == 1, ghosted
    # The other arm ended with nothing on screen at all, which is the 100K `model_change` shape:
    # 12 mounted messages to 0, never recovered.
    empty = dict(ghosted)
    empty["messages"] = {}
    empty["ordinal_collisions"] = 0
    empty["collided_ordinals"] = []

    verdict = P.compare_visible(empty, ghosted)
    assert verdict["verdict"] == P.DIFFER, verdict
    assert verdict.get("severe") is True, verdict
    assert "lost the thread" in verdict["reason"], verdict


def test_a_collision_with_both_viewports_alive_is_still_refused(browser):
    """THE CONTROL for the one above. The severe finding is the only thing that outranks the
    refusal, so a collision on an arm whose viewport is perfectly healthy must still refuse."""
    from studiobench.analysis import parity as P

    clean = _capture_with_ghost(browser, before = False)
    clean["ordinal_collisions"] = 0
    clean["collided_ordinals"] = []
    ghosted = _capture_with_ghost(browser, before = True)
    assert clean["messages"] and ghosted["messages"], (clean, ghosted)
    assert P.compare_visible(clean, ghosted)["verdict"] == P.NOT_COMPARABLE
