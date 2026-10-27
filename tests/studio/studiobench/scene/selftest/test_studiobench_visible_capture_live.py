# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Visible capture sees partly visible rows, accumulates over the action, and never reads geometry."""

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

#: Tall messages in a short viewport; capture keys on `aria-posinset` so windows compare.
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
    # After the content: Playwright's `set_content` does not always run init scripts.
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
    yield pg
    pg.close()


def _watch(page) -> None:
    got = page.evaluate("() => window.__sb.parityVisible.watch()")
    assert got.get("visible_attempted") is True, got
    # IntersectionObserver delivers asynchronously; wait a frame before scrolling.
    page.wait_for_timeout(120)


def _capture(page) -> dict:
    page.wait_for_timeout(120)
    return page.evaluate("async () => await window.__sb.parityVisible.capture()")


def test_it_reports_only_what_the_viewport_showed(page):
    _watch(page)
    got = _capture(page)
    assert got["visible_attempted"] is True
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


def test_the_capture_never_reads_geometry(page):
    """Reading rects in a content-visibility subtree renders it, so the capture must never read geometry."""
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


def test_capturing_without_watching_is_refused_not_reported_empty(page):
    got = page.evaluate("async () => await window.__sb.parityVisible.capture()")
    assert got["visible_attempted"] is False
    assert "never installed" in got["reason"]


def test_a_page_with_no_thread_viewport_is_refused(page):
    page.evaluate("() => { document.querySelector('.aui-thread-viewport').remove(); }")
    got = page.evaluate("() => window.__sb.parityVisible.watch()")
    assert got["visible_attempted"] is False, got
    assert "viewport" in got["reason"]


def test_the_top_up_is_proportional_to_the_mutation_not_to_the_document(page):
    """The top-up walks only addedNodes, so the measured action is not charged for a whole-document scan."""
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
    # Take the handle before the baseline: the lookup is itself a document-wide query.
    page.evaluate(
        """() => {
             const rows = document.querySelectorAll("[data-role]");
             window.__last = rows[rows.length - 1];
           }"""
    )
    baseline = page.evaluate("() => window.__docQsa")
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


# Readiness accepts ordinals on the message or an ancestor row, so the digest must ignore where
# they are published or such arms differ on every message.


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


# Rebuilt rows without `aria-posinset` must get their own position, not a lifetime counter value.


#: Shipped shape: all messages mounted, no virtualization ordinal; `__rebuild()` mimics thread_reopen.
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
    assert got["unplaced_rows"] == 0, got


def test_a_rebuilt_full_mount_still_matches_a_windowed_arm(browser):
    """A rebuilt full mount and a windowed arm showing the same messages must still compare as a match."""
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
    """Placing a batch of rebuilt rows reads the document once per batch, not once per row."""
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
    """Ordinal exclusion is passed in by the visible caller, so structural digests still see ordinals."""
    numbered = _thread_digests(browser, "on_the_message")
    plain = _thread_digests(browser, "nowhere")
    assert set(numbered) == set(plain)
    assert all(numbered[i] != plain[i] for i in numbered), (numbered, plain)


#: `__churn` mounts and unmounts a row in one task so it is observed detached; `__remount` reuses
#: the same node, as a recycling virtualizer does.

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
    """A row unplaceable on first sight must still be stamped when it later mounts, not blacklisted."""

    pg = browser.new_page(viewport = {"width": 800, "height": 600})
    try:
        pg.set_content(RECYCLE_FIXTURE)
        pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        pg.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        _watch(pg)
        pg.evaluate("() => window.__churn()")
        pg.wait_for_timeout(150)
        assert pg.evaluate("async () => (await window.__sb.parityVisible.capture()).unplaced_rows")
        pg.evaluate("() => window.__remount()")
        got = _capture(pg)
    finally:
        pg.close()

    assert got["ever_visible"] == [1], got


#: Renumbered in place: not a childList mutation, so a childList-only observer misses it.
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
    """A recycled row renumbered in place must be restamped and reported at its new thread position."""

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

    assert 42 in got["ever_visible"], got["ever_visible"]
    assert "42" in got["messages"], got["messages"]
    assert got["unplaced_rows"] == 0, got
    # A legitimate renumber is not a collision.
    assert got["ordinal_collisions"] == 0, got


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
    """Rows sharing a thread position are counted as a collision, not overwritten by the last one."""
    for before in (True, False):
        got = _capture_with_ghost(browser, before = before)
        assert got["ordinal_collisions"] == 1, (before, got)
        assert got["collided_ordinals"] == [1], (before, got)
        assert set(got["messages"]) == {"1", "2"}, got["messages"]


def test_a_collision_refuses_the_pair_instead_of_reporting_agreement(browser):
    """A position collision drops a row invisibly, so the pair is refused rather than compared."""
    from studiobench.analysis import parity as P

    clean = _capture_with_ghost(browser, before = False)
    clean["ordinal_collisions"] = 0
    clean["collided_ordinals"] = []
    ghosted = _capture_with_ghost(browser, before = True)
    verdict = P.compare_visible(clean, ghosted)
    assert verdict["verdict"] == P.NOT_COMPARABLE, verdict
    assert "SAME thread position" in verdict["reason"], verdict


def test_losing_the_thread_outranks_the_collision_refusal(browser):
    """A collision cannot empty a map, so the severe empty-viewport finding must not be refused with it."""
    from studiobench.analysis import parity as P

    ghosted = _capture_with_ghost(browser, before = True)
    assert ghosted["ordinal_collisions"] == 1, ghosted
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
